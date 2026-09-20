"""Write-time established pass after working-layer ``create_documents`` commit.

Three phases, no DB session held across LLM confirm:

1. Short ``tracked_db`` — ``find_neighbours`` per accepted explicit
   (``NeighbourScope.established()``, ``max_distance=CANDIDATE_MAX``).
2. No DB — confirmer loop for ``NeedsConfirm`` (nearest-only; merge
   confirmations until a terminal verdict or UNDECIDED; prefer Leave).
3. Mode ``on`` only — short txn of reinforce / supersede row ops.

Durability choices (synthesis: base A + B grafts):

- Evidence idempotency uses the ``established_evidence`` uniqueness table
  ``(established_id, evidence_digest)`` with ON CONFLICT DO NOTHING, not an
  unbounded JSON list alone. Metadata ``evidence`` is mirrored for digest.
- Supersession edges use set-once ``internal_metadata.superseded_by`` on the
  loser and ``supersedes`` list union on the winner (same locked txn). No
  separate supersession table in PR3.

Winner minting always routes through ``create_documents(...,
deduplicate=False, _established_pass=False)`` so that path remains the only
Document insert choke point. Promotion reconciler (PR4) is not wired here.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

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
    _normalize_content,
    create_documents,
    find_neighbours,
)
from src.dependencies import tracked_db
from src.memory.bands import CANDIDATE_MAX
from src.memory.confirm import ConfirmAnswer, Confirmations, Confirmer
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
        verdicts = await _phase_decide_and_confirm(
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
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        result.shadow_verdicts.extend(names)
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
) -> Verdict:
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
            return verdict
        async with sem:
            batch = await confirmer.confirm(accepted.content, list(verdict.candidates))
        confirmations = _merge_confirmations(confirmations, batch)
        # Prefer under-grouping: UNDECIDED on the asked neighbour → Leave.
        for candidate in verdict.candidates:
            proof = batch.for_neighbour(candidate.id)
            if proof is not None and proof.answer is ConfirmAnswer.UNDECIDED:
                return Leave(reason="undecided")
    return Leave(reason="undecided")


async def _phase_decide_and_confirm(
    accepted: Sequence[_AcceptedExplicit],
    neighbourhoods: dict[str, list[Neighbour]],
    *,
    confirmer: Confirmer,
) -> dict[str, Verdict]:
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
    verdicts = await asyncio.gather(*tasks)
    return {acc.id: v for acc, v in zip(accepted, verdicts, strict=True)}


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
    """Mint an established (derived) row through ``create_documents``.

    On exact-dedup of an already-minted winner, returns that live row without
    a second insert. Exact match is resolved before insert so retries do not
    bump ``times_derived`` via the working-layer reinforce path.
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
    await create_documents(
        db,
        [doc],
        workspace_name,
        observer=observer,
        observed=observed,
        deduplicate=False,
        _established_pass=False,
    )
    found = await _find_exact_established(
        db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        content=content,
        level=level,
    )
    if found is None:
        raise RuntimeError("mint_established_row did not persist a document")
    return found


async def _find_exact_established(
    db: AsyncSession,
    *,
    workspace_name: str,
    observer: str,
    observed: str,
    content: str,
    level: DocumentLevel,
) -> models.Document | None:
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
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _phase_apply(
    accepted: Sequence[_AcceptedExplicit],
    verdicts: dict[str, Verdict],
    *,
    workspace_name: str,
    observer: str,
    observed: str,
) -> EstablishedPassResult:
    result = EstablishedPassResult()
    ops: list[_DocumentRowOp] = []

    for acc in accepted:
        verdict = verdicts[acc.id]
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
            async with tracked_db("established_mint") as mint_db:
                winner = await mint_established_row(
                    mint_db,
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
                    evidence_key=str(verdict.evidence),
                )
            )
            ops.append(
                _DocumentRowOp(
                    "reinforce",
                    winner.id,
                    evidence_key=str(verdict.evidence),
                )
            )
            result.superseded.append((winner.id, verdict.loser_id))
            result.reinforced.append(winner.id)

    if not ops:
        return result

    async with tracked_db("established_apply") as db:
        await _apply_document_row_updates(
            db,
            ops,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        await db.commit()
    return result
