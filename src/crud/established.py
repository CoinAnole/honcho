"""Write-time established pass after working-layer ``create_documents`` commit.

Three phases, no DB session held across LLM confirm:

1. Short ``tracked_db`` — ``find_neighbours`` per accepted explicit
   (``NeighbourScope.established()``, ``max_distance=CANDIDATE_MAX``).
2. No DB — confirmer loop for ``NeedsConfirm`` (nearest-only; merge
   confirmations until a terminal verdict or UNDECIDED; prefer Leave).
3. Mode ``on`` only — one ``tracked_db("established_apply")`` window:
   mint via ``_document_model_from_create`` + flush, ledger + clock at
   ``times_derived=1``, loser soft-delete, then healer.

Durability choices (B8):

- Mint does not call ``create_documents`` and does not commit. Winner
  insert, ledger, clock, loser soft-delete, and both metadata edges share
  the apply commit. Crash before that commit leaves nothing.
- Evidence idempotency uses the ``established_evidence`` uniqueness table
  ``(established_id, evidence_digest)`` with ON CONFLICT DO NOTHING.
  Metadata ``evidence`` is mirrored for digest.
- Supersession edges use set-once ``internal_metadata.superseded_by`` on the
  loser and ``supersedes`` list union on the winner (same locked txn).
- Healer pairs live established rows at ``CANDIDATE_MAX`` and cannot key on
  ``supersedes`` / ``superseded_by``. Promotion does not call it and does
  not emit ``Supersede``.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import func

from src import models, schemas
from src.config import settings
from src.crud.document import (
    EstablishedPassResult,
    Neighbour,
    NeighbourScope,
    _DocumentRowOp,
    _apply_document_row_updates,
    _document_model_from_create,
    _insert_established_evidence,
    _normalize_content,
    find_neighbours,
)
from src.dependencies import tracked_db
from src.memory.bands import CANDIDATE_MAX, Band, classify_distance
from src.memory.confirm import ConfirmAnswer, Confirmations, Confirmed, Confirmer
from src.memory.evidence import EvidenceKey
from src.memory.verdict import (
    Leave,
    NeedsConfirm,
    Reinforce,
    Supersede,
    Verdict,
    decide_established,
)
from src.utils.types import DocumentLevel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _AcceptedExplicit:
    """Snapshot of one explicit row after the working-layer commit.

    Plain data: the ORM object is detached after the caller's commit.
    """

    id: str
    content: str
    embedding: list[float]
    session_name: str
    message_ids: tuple[int, ...]
    message_created_at: str
    evidence: EvidenceKey


@dataclass(frozen=True, slots=True)
class _LiveEstablished:
    """Read-side snapshot of one live derived row in a heal scan."""

    id: str
    created_at: datetime.datetime
    embedding: list[float]
    content: str
    level: DocumentLevel


@dataclass(frozen=True, slots=True)
class _EstablishedPair:
    """One live established pair. kept_id is newer; drop_id is older.

    kind is the distance band, not a caller policy flag.
    SAME_CLAIM → collapse with no confirm.
    CANDIDATE → confirm; collapse only on SAME_CLAIM or SAME_SUBJECT_NEW_VALUE.
    """

    kept_id: str
    drop_id: str
    distance: float
    kind: Band

    def __post_init__(self) -> None:
        if self.kept_id == self.drop_id:
            raise ValueError("_EstablishedPair cannot pair a row with itself")
        if self.kind is Band.UNRELATED:
            raise ValueError("_EstablishedPair rejects UNRELATED")


@dataclass(frozen=True, slots=True)
class _SameClaimHeal:
    """Distance-proven duplicate; newest row is the winner."""

    winner_id: str
    loser_id: str

    def __post_init__(self) -> None:
        if self.winner_id == self.loser_id:
            raise ValueError("_SameClaimHeal cannot target a row as its own loser")


@dataclass(frozen=True, slots=True)
class _ConfirmedReplacementHeal:
    """Candidate-band deletion is unrepresentable without this proof."""

    winner_id: str
    loser_id: str
    proof: Confirmed

    def __post_init__(self) -> None:
        if self.winner_id == self.loser_id:
            raise ValueError(
                "_ConfirmedReplacementHeal cannot target a row as its own loser"
            )
        if self.proof.answer not in (
            ConfirmAnswer.SAME_CLAIM,
            ConfirmAnswer.SAME_SUBJECT_NEW_VALUE,
        ):
            raise ValueError(
                "_ConfirmedReplacementHeal requires SAME_CLAIM or "
                "SAME_SUBJECT_NEW_VALUE proof"
            )


_HealDirective: TypeAlias = _SameClaimHeal | _ConfirmedReplacementHeal


@dataclass(frozen=True, slots=True)
class HealResult:
    collapsed: list[tuple[str, str]]  # (kept_id, drop_id)
    left_unconfirmed: int = 0
    left_unrelated: int = 0


def _as_document_create(accepted: _AcceptedExplicit) -> schemas.DocumentCreate:
    return schemas.DocumentCreate(
        content=accepted.content,
        embedding=accepted.embedding,
        session_name=accepted.session_name,
        level="explicit",
        metadata=schemas.DocumentMetadata(
            message_ids=list(accepted.message_ids),
            message_created_at=accepted.message_created_at,
        ),
    )


def _merge_confirmations(base: Confirmations, extra: Confirmations) -> Confirmations:
    merged = dict(base.by_neighbour)
    merged.update(extra.by_neighbour)
    return Confirmations(by_neighbour=merged)


def _log_established_pass(mode: str, result: EstablishedPassResult) -> None:
    logger.info(
        "Established pass complete: mode=%s reinforced=%s superseded=%s "
        "left_working=%s undecided=%s shadow=%s",
        mode,
        len(result.reinforced),
        len(result.superseded),
        result.left_working,
        result.awaiting_confirm_undecided,
        ",".join(result.shadow_verdicts) or "-",
    )


async def run_established_pass(
    accepted: Sequence[_AcceptedExplicit],
    *,
    workspace_name: str,
    observer: str,
    observed: str,
    confirmer: Confirmer,
    mode: Literal["shadow", "on"],
) -> EstablishedPassResult:
    """Run the established pass. Never raises into ``create_documents``.

    Failures log and leave rows working. Retries converge via evidence keys.
    MODE=on calls the healer after apply, including an empty apply. Promotion
    does not call the healer.
    """
    result = EstablishedPassResult()
    if not accepted:
        return result
    try:
        neighbourhoods = await _phase_neighbours(
            accepted,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        verdicts, confirmations = await _phase_decide_and_confirm(
            accepted,
            neighbourhoods,
            confirmer=confirmer,
        )
        names = [type(verdicts[acc.id]).__name__ for acc in accepted]
        if mode == "shadow":
            result.shadow_verdicts.extend(names)
            for acc in accepted:
                v = verdicts[acc.id]
                if isinstance(v, Leave):
                    result.left_working += 1
                    if v.reason == "undecided":
                        result.awaiting_confirm_undecided += 1
                elif isinstance(v, NeedsConfirm):
                    # Should be rare after the confirm loop; count as leave.
                    result.left_working += 1
                    result.awaiting_confirm_undecided += 1
            _log_established_pass(mode, result)
            return result

        result = await _phase_apply(
            accepted,
            verdicts,
            confirmations,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        result.shadow_verdicts.extend(names)
        try:
            await heal_live_established_pairs(
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                confirmer=confirmer,
                anchors=[acc.embedding for acc in accepted],
            )
        except Exception:
            logger.exception(
                "Established healer failed for %s/%s/%s; write-time result kept",
                workspace_name,
                observer,
                observed,
            )
        _log_established_pass(mode, result)
        return result
    except Exception:
        logger.exception(
            "Established pass failed for %s/%s/%s",
            workspace_name,
            observer,
            observed,
        )
        return result


async def _phase_neighbours(
    accepted: Sequence[_AcceptedExplicit],
    *,
    workspace_name: str,
    observer: str,
    observed: str,
) -> dict[str, list[Neighbour]]:
    top_k = settings.ESTABLISHED.CANDIDATE_TOP_K
    out: dict[str, list[Neighbour]] = {}
    async with tracked_db("established_neighbours") as db:
        for acc in accepted:
            out[acc.id] = await find_neighbours(
                db,
                workspace_name,
                observer=observer,
                observed=observed,
                embedding=acc.embedding,
                scope=NeighbourScope.established(),
                max_distance=CANDIDATE_MAX,
                top_k=top_k,
            )
    return out


async def _decide_one(
    accepted: _AcceptedExplicit,
    neighbours: list[Neighbour],
    confirmer: Confirmer,
    *,
    sem: asyncio.Semaphore,
) -> tuple[Verdict, Confirmations]:
    new = _as_document_create(accepted)
    confirmations = Confirmations.empty()
    top_k = settings.ESTABLISHED.CANDIDATE_TOP_K
    for _ in range(top_k):
        verdict = decide_established(
            new=new,
            neighbours=neighbours,
            confirmations=confirmations,
        )
        if not isinstance(verdict, NeedsConfirm):
            return verdict, confirmations
        async with sem:
            batch = await confirmer.confirm(accepted.content, list(verdict.candidates))
        confirmations = _merge_confirmations(confirmations, batch)
        # Prefer under-grouping: UNDECIDED on the asked neighbour → Leave.
        for candidate in verdict.candidates:
            proof = batch.for_neighbour(candidate.id)
            if proof is not None and proof.answer is ConfirmAnswer.UNDECIDED:
                return Leave(reason="undecided"), confirmations
    return Leave(reason="undecided"), confirmations


async def _phase_decide_and_confirm(
    accepted: Sequence[_AcceptedExplicit],
    neighbourhoods: dict[str, list[Neighbour]],
    *,
    confirmer: Confirmer,
) -> tuple[dict[str, Verdict], dict[str, Confirmations]]:
    sem = asyncio.Semaphore(settings.ESTABLISHED.CONFIRM_CONCURRENCY)
    tasks = [
        _decide_one(
            acc,
            neighbourhoods.get(acc.id, []),
            confirmer,
            sem=sem,
        )
        for acc in accepted
    ]
    decided = await asyncio.gather(*tasks)
    verdicts = {acc.id: v for acc, (v, _) in zip(accepted, decided, strict=True)}
    confirmations = {acc.id: c for acc, (_, c) in zip(accepted, decided, strict=True)}
    return verdicts, confirmations


async def mint_established_row(
    db: AsyncSession,
    *,
    workspace_name: str,
    observer: str,
    observed: str,
    content: str,
    embedding: list[float],
    level: DocumentLevel,
    source_ids: Sequence[str],
    evidence: Sequence[EvidenceKey],
    message_ids: Sequence[int],
    message_created_at: str,
) -> models.Document:
    """Flush-only established mint. Ledger and clock at times_derived=1.

    Exact-dedup of an already-minted winner returns that live row without
    a second insert and without counting the digest again. Does not call
    ``create_documents`` and does not commit. Promotion shares this contract.
    """
    existing = await _find_exact_established(
        db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        content=content,
        level=level,
    )
    if existing is not None:
        return existing

    sources = [sid for sid in source_ids if isinstance(sid, str) and sid]
    if not sources:
        raise ValueError("mint_established_row requires at least one source_id")

    doc = schemas.DocumentCreate(
        content=content,
        embedding=embedding,
        session_name=None,
        level=level,
        times_derived=1,
        source_ids=sources,
        metadata=schemas.DocumentMetadata(
            message_ids=list(message_ids),
            message_created_at=message_created_at,
            source_ids=sources,
            evidence=[str(e) for e in evidence],
        ),
    )
    winner = _document_model_from_create(
        doc,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
    )
    # Clock before the single flush: populate_existing=True on the later
    # apply lock would erase a post-flush attribute change.
    winner.last_reinforced_at = datetime.datetime.now(datetime.UTC)
    db.add(winner)
    await db.flush()
    for digest in evidence:
        await _insert_established_evidence(
            db, established_id=winner.id, evidence_digest=str(digest)
        )
    return winner


async def _find_exact_established(
    db: AsyncSession,
    *,
    workspace_name: str,
    observer: str,
    observed: str,
    content: str,
    level: DocumentLevel,
) -> models.Document | None:
    """Live exact content match. Newest wins so remint agrees with healer kept_id."""
    normalized = _normalize_content(content)
    normalized_content_sql = func.lower(
        func.regexp_replace(models.Document.content, r"^\s+|\s+$", "", "g")
    )
    result = await db.execute(
        select(models.Document)
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
            models.Document.level == level,
            models.Document.deleted_at.is_(None),
            normalized_content_sql == normalized,
        )
        .order_by(models.Document.created_at.desc(), models.Document.id.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _refresh_supersede(
    db: AsyncSession,
    accepted: _AcceptedExplicit,
    verdict: Supersede,
    confirmations: Confirmations,
    *,
    workspace_name: str,
    observer: str,
    observed: str,
) -> Verdict:
    """Re-run neighbours + decide on the apply session with held confirmations.

    A same-batch winner flushed earlier in this window is visible here
    (pgvector). The second ``Supersede`` against one loser becomes
    ``Reinforce`` or ``Leave``. ``NeedsConfirm`` is treated as Leave so the
    apply window does not open a second LLM round.
    """
    del verdict
    neighbours = await find_neighbours(
        db,
        workspace_name,
        observer=observer,
        observed=observed,
        embedding=accepted.embedding,
        scope=NeighbourScope.established(),
        max_distance=CANDIDATE_MAX,
        top_k=settings.ESTABLISHED.CANDIDATE_TOP_K,
    )
    refreshed = decide_established(
        new=_as_document_create(accepted),
        neighbours=neighbours,
        confirmations=confirmations,
    )
    if isinstance(refreshed, NeedsConfirm):
        return Leave(reason="undecided")
    return refreshed


async def _phase_apply(
    accepted: Sequence[_AcceptedExplicit],
    verdicts: dict[str, Verdict],
    confirmations: dict[str, Confirmations],
    *,
    workspace_name: str,
    observer: str,
    observed: str,
) -> EstablishedPassResult:
    """Mint + loser soft-delete in one apply window. No post-mint reinforce."""
    result = EstablishedPassResult()
    needs_window = any(
        isinstance(verdicts[acc.id], (Reinforce, Supersede)) for acc in accepted
    )
    if not needs_window:
        for acc in accepted:
            verdict = verdicts[acc.id]
            if isinstance(verdict, Leave):
                result.left_working += 1
                if verdict.reason == "undecided":
                    result.awaiting_confirm_undecided += 1
            elif isinstance(verdict, NeedsConfirm):
                result.left_working += 1
                result.awaiting_confirm_undecided += 1
        return result

    ops: list[_DocumentRowOp] = []

    async with tracked_db("established_apply") as db:
        for acc in accepted:
            verdict = verdicts[acc.id]
            if isinstance(verdict, Supersede):
                verdict = await _refresh_supersede(
                    db,
                    acc,
                    verdict,
                    confirmations.get(acc.id, Confirmations.empty()),
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                )
            if isinstance(verdict, Leave):
                result.left_working += 1
                if verdict.reason == "undecided":
                    result.awaiting_confirm_undecided += 1
                continue
            if isinstance(verdict, NeedsConfirm):
                result.left_working += 1
                result.awaiting_confirm_undecided += 1
                continue
            if isinstance(verdict, Reinforce):
                ops.append(
                    _DocumentRowOp(
                        "reinforce",
                        verdict.target_id,
                        evidence_key=str(verdict.evidence),
                    )
                )
                result.reinforced.append(verdict.target_id)
                continue
            if isinstance(verdict, Supersede):
                winner = await mint_established_row(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    content=acc.content,
                    embedding=acc.embedding,
                    level=verdict.winner_level,
                    source_ids=[acc.id],
                    evidence=[verdict.evidence],
                    message_ids=acc.message_ids,
                    message_created_at=acc.message_created_at,
                )
                ops.append(
                    _DocumentRowOp(
                        "supersede",
                        verdict.loser_id,
                        winner_id=winner.id,
                    )
                )
                result.superseded.append((winner.id, verdict.loser_id))

        if ops:
            await _apply_document_row_updates(
                db,
                ops,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
            )
            await db.commit()
    return result


def _pair_from_live_rows(
    left: _LiveEstablished,
    right: _LiveEstablished,
    distance: float,
) -> _EstablishedPair | None:
    """Classify distance; drop UNRELATED; encode older as drop_id."""
    kind = classify_distance(distance)
    if kind is Band.UNRELATED:
        return None
    if (left.created_at, left.id) >= (right.created_at, right.id):
        newer, older = left, right
    else:
        newer, older = right, left
    return _EstablishedPair(
        kept_id=newer.id,
        drop_id=older.id,
        distance=distance,
        kind=kind,
    )


async def _load_live_established(
    db: AsyncSession,
    *,
    workspace_name: str,
    observer: str,
    observed: str,
    anchors: Sequence[list[float]] | None,
) -> list[_LiveEstablished]:
    """Live inductive/deductive rows, optionally restricted to an anchor neighbourhood."""
    if anchors:
        seed_ids: set[str] = set()
        top_k = settings.ESTABLISHED.CANDIDATE_TOP_K
        for embedding in anchors:
            hits = await find_neighbours(
                db,
                workspace_name,
                observer=observer,
                observed=observed,
                embedding=embedding,
                scope=NeighbourScope.established(),
                max_distance=CANDIDATE_MAX,
                top_k=top_k,
            )
            seed_ids.update(hit.id for hit in hits)
        if not seed_ids:
            return []
        result = await db.execute(
            select(models.Document).where(
                models.Document.id.in_(seed_ids),
                models.Document.workspace_name == workspace_name,
                models.Document.observer == observer,
                models.Document.observed == observed,
                models.Document.deleted_at.is_(None),
                models.Document.embedding.is_not(None),
            )
        )
    else:
        result = await db.execute(
            select(models.Document).where(
                models.Document.workspace_name == workspace_name,
                models.Document.observer == observer,
                models.Document.observed == observed,
                models.Document.level.in_(tuple(NeighbourScope.established().levels)),
                models.Document.deleted_at.is_(None),
                models.Document.embedding.is_not(None),
            )
        )
    rows: list[_LiveEstablished] = []
    for doc in result.scalars():
        if doc.embedding is None:
            continue
        rows.append(
            _LiveEstablished(
                id=doc.id,
                created_at=doc.created_at,
                embedding=list(doc.embedding),
                content=doc.content,
                level=doc.level,
            )
        )
    return rows


async def _scan_live_established_pairs(
    db: AsyncSession,
    rows: Sequence[_LiveEstablished],
    *,
    workspace_name: str,
    observer: str,
    observed: str,
) -> list[_EstablishedPair]:
    """Pair loaded rows at CANDIDATE_MAX. Canonical key so A-vs-B is one pair."""
    by_id = {row.id: row for row in rows}
    seen: set[tuple[str, str]] = set()
    pairs: list[_EstablishedPair] = []
    top_k = settings.ESTABLISHED.CANDIDATE_TOP_K
    for row in rows:
        hits = await find_neighbours(
            db,
            workspace_name,
            observer=observer,
            observed=observed,
            embedding=row.embedding,
            scope=NeighbourScope.established(exclude_id=row.id),
            max_distance=CANDIDATE_MAX,
            top_k=top_k,
        )
        for hit in hits:
            if hit.id not in by_id:
                continue
            key = (min(row.id, hit.id), max(row.id, hit.id))
            if key in seen:
                continue
            seen.add(key)
            pair = _pair_from_live_rows(row, by_id[hit.id], hit.distance)
            if pair is not None:
                pairs.append(pair)
    return pairs


async def _confirm_candidate_pairs(
    pairs: Sequence[_EstablishedPair],
    *,
    rows: dict[str, _LiveEstablished],
    confirmer: Confirmer,
) -> tuple[list[_HealDirective], int, int]:
    """No DB. SAME_CLAIM collapses without confirm. Candidate needs proof."""
    directives: list[_HealDirective] = []
    left_unconfirmed = 0
    left_unrelated = 0
    candidates: list[_EstablishedPair] = []
    for pair in pairs:
        if pair.kind is Band.SAME_CLAIM:
            directives.append(
                _SameClaimHeal(winner_id=pair.kept_id, loser_id=pair.drop_id)
            )
        elif pair.kind is Band.CANDIDATE:
            candidates.append(pair)
    if not candidates:
        return directives, left_unconfirmed, left_unrelated

    sem = asyncio.Semaphore(settings.ESTABLISHED.CONFIRM_CONCURRENCY)

    async def _confirm_one(
        pair: _EstablishedPair,
    ) -> tuple[_EstablishedPair, Confirmed | None]:
        drop = rows[pair.drop_id]
        async with sem:
            batch = await confirmer.confirm(
                rows[pair.kept_id].content,
                [
                    Neighbour(
                        id=drop.id,
                        distance=pair.distance,
                        level=drop.level,
                        session_name=None,
                        content=drop.content,
                    )
                ],
            )
        return pair, batch.for_neighbour(pair.drop_id)

    confirmed = await asyncio.gather(*(_confirm_one(pair) for pair in candidates))
    for pair, proof in confirmed:
        if proof is None:
            left_unconfirmed += 1
            continue
        if proof.answer in (
            ConfirmAnswer.SAME_CLAIM,
            ConfirmAnswer.SAME_SUBJECT_NEW_VALUE,
        ):
            directives.append(
                _ConfirmedReplacementHeal(
                    winner_id=pair.kept_id,
                    loser_id=pair.drop_id,
                    proof=proof,
                )
            )
        elif proof.answer is ConfirmAnswer.UNRELATED:
            left_unrelated += 1
        else:
            left_unconfirmed += 1
    return directives, left_unconfirmed, left_unrelated


async def heal_live_established_pairs(
    *,
    workspace_name: str,
    observer: str,
    observed: str,
    confirmer: Confirmer,
    anchors: Sequence[list[float]] | None = None,
) -> HealResult:
    """Close leftover live established pairs. Owns scan / confirm / apply sessions.

    Pairing key is CANDIDATE_MAX. SAME_CLAIM collapses without confirm.
    Candidate-band pairs collapse only with SAME_CLAIM or
    SAME_SUBJECT_NEW_VALUE proof. Apply reuses ``_DocumentRowOp("supersede")``
    so edges backfill. Does not key on ``supersedes`` / ``superseded_by``.
    ``anchors`` restricts the scan to established rows within CANDIDATE_MAX of
    those embeddings. Unit tests and operator one-shots call it unscoped.
    Does not run from promotion.
    """
    async with tracked_db("established_heal_scan") as db:
        rows = await _load_live_established(
            db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            anchors=anchors,
        )
        pairs = await _scan_live_established_pairs(
            db,
            rows,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        by_id = {row.id: row for row in rows}
    directives, left_unconfirmed, left_unrelated = await _confirm_candidate_pairs(
        pairs,
        rows=by_id,
        confirmer=confirmer,
    )
    if not directives:
        return HealResult(
            collapsed=[],
            left_unconfirmed=left_unconfirmed,
            left_unrelated=left_unrelated,
        )
    async with tracked_db("established_heal_apply") as db:
        ops = [
            _DocumentRowOp("supersede", directive.loser_id, winner_id=directive.winner_id)
            for directive in directives
        ]
        await _apply_document_row_updates(
            db,
            ops,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        await db.commit()
    return HealResult(
        collapsed=[(directive.winner_id, directive.loser_id) for directive in directives],
        left_unconfirmed=left_unconfirmed,
        left_unrelated=left_unrelated,
    )
