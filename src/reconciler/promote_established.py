"""Promotion reconciler: cross-session same-claim → established derived row.

Owns Track B §5 outcome three: the same claim in enough independent
conversations, with no established neighbour yet, mints via
``mint_established_row``. Existing same-claim established rows are reinforced
instead. Dreamer stays a backup consolidator.

Stateless between runs: an explicit is examined when
``internal_metadata.promotion_examined_at`` is set. The stamp lands in the
same txn as mint/reinforce so a crash before commit re-examines (idempotent
via exact-dedup mint + evidence keys).

MODE gating (PR4 choice):

- ``off``: cycle is a no-op; scheduler does not enqueue.
- ``shadow``: phase 1–2 run for metrics only; no mint, reinforce, or examined
  stamp so a later ``on`` run still sees the work.
- ``on``: full write path.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from src import models
from src.config import settings
from src.crud.document import (
    Neighbour,
    NeighbourScope,
    _DocumentRowOp,
    _apply_document_row_updates,
    fetch_documents_by_ids,
    find_neighbours,
)
from src.crud.established import mint_established_row
from src.dependencies import tracked_db
from src.dreamer.dream_scheduler import check_and_schedule_dream
from src.memory.bands import CANDIDATE_MAX
from src.memory.confirm import ConfirmAnswer, Confirmations, Confirmer
from src.memory.evidence import Conversation, EvidenceKey, evidence_key
from src.memory.verdict import (
    Leave,
    NeedsConfirm,
    Promote,
    PromotionSeed,
    PromotionVerdict,
    Reinforce,
    decide_promotion,
)
from src.utils.formatting import parse_datetime_iso

logger = logging.getLogger(__name__)

_PROMOTION_EXAMINED_AT = "promotion_examined_at"


@dataclass
class PromotionCycleMetrics:
    examined: int = 0
    promoted: int = 0
    reinforced_established: int = 0
    undecided: int = 0
    left_working: int = 0
    dream_hints: int = 0
    shadow_verdicts: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _PromotionWorkItem:
    """Phase-1 snapshot: seed + neighbourhoods + peer evidence maps."""

    seed: PromotionSeed
    content: str
    embedding: list[float]
    message_ids: tuple[int, ...]
    message_created_at: str
    workspace_name: str
    observer: str
    observed: str
    established: tuple[Neighbour, ...]
    peers: tuple[Neighbour, ...]
    peer_evidence: Mapping[str, EvidenceKey]
    peer_conversations: Mapping[str, Conversation]


def _unexamined_predicate():
    """Live explicits lacking a non-null ``promotion_examined_at`` stamp."""
    meta = models.Document.internal_metadata
    return ~meta.has_key(_PROMOTION_EXAMINED_AT) | meta[
        _PROMOTION_EXAMINED_AT
    ].astext.is_(None)


async def has_pending_promotion_work(db: AsyncSession) -> bool:
    """EXISTS(live explicit rows lacking promotion_examined_at)."""
    stmt = (
        select(models.Document.id)
        .where(
            models.Document.level == "explicit",
            models.Document.deleted_at.is_(None),
            _unexamined_predicate(),
        )
        .limit(1)
    )
    return (await db.execute(stmt)).first() is not None


def _message_ids_from_meta(meta: dict[str, Any]) -> list[int] | None:
    raw = meta.get("message_ids")
    if not isinstance(raw, list) or not raw:
        return None
    ids: list[int] = []
    for item in raw:
        if isinstance(item, int):
            ids.append(item)
        elif isinstance(item, str) and item.isdigit():
            ids.append(int(item))
        else:
            return None
    return ids or None


def _seed_from_document(doc: models.Document) -> _PromotionWorkItem | None:
    """Build a work item shell from a claimed row. Neighbours filled later."""
    if doc.embedding is None or doc.session_name is None:
        return None
    meta = dict(doc.internal_metadata or {})
    message_ids = _message_ids_from_meta(meta)
    raw_created = meta.get("message_created_at")
    if message_ids is None or not isinstance(raw_created, str):
        return None
    try:
        created_at = parse_datetime_iso(raw_created)
    except (ValueError, TypeError):
        return None
    embedding = list(doc.embedding)
    return _PromotionWorkItem(
        seed=PromotionSeed(
            id=doc.id,
            content=doc.content,
            session_name=doc.session_name,
            evidence=evidence_key(doc.session_name, message_ids),
            conversation=Conversation(
                session_name=doc.session_name,
                message_created_at=created_at,
            ),
        ),
        content=doc.content,
        embedding=embedding,
        message_ids=tuple(message_ids),
        message_created_at=raw_created,
        workspace_name=doc.workspace_name,
        observer=doc.observer,
        observed=doc.observed,
        established=(),
        peers=(),
        peer_evidence={},
        peer_conversations={},
    )


async def _peer_maps(
    db: AsyncSession,
    *,
    workspace_name: str,
    observer: str,
    observed: str,
    peers: Sequence[Neighbour],
) -> tuple[dict[str, EvidenceKey], dict[str, Conversation]]:
    if not peers:
        return {}, {}
    docs = await fetch_documents_by_ids(
        db=db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        document_ids=[p.id for p in peers],
        filters=None,
    )
    evidence: dict[str, EvidenceKey] = {}
    conversations: dict[str, Conversation] = {}
    for doc in docs:
        if doc.session_name is None:
            continue
        meta = dict(doc.internal_metadata or {})
        message_ids = _message_ids_from_meta(meta)
        raw_created = meta.get("message_created_at")
        if message_ids is None or not isinstance(raw_created, str):
            continue
        try:
            created_at = parse_datetime_iso(raw_created)
        except (ValueError, TypeError):
            continue
        evidence[doc.id] = evidence_key(doc.session_name, message_ids)
        conversations[doc.id] = Conversation(
            session_name=doc.session_name,
            message_created_at=created_at,
        )
    return evidence, conversations


async def _claim_and_snapshot(
    db: AsyncSession, batch_size: int
) -> list[_PromotionWorkItem]:
    """Claim unexamined explicits (SKIP LOCKED), snapshot neighbourhoods."""
    stmt = (
        select(models.Document)
        .where(
            models.Document.level == "explicit",
            models.Document.deleted_at.is_(None),
            models.Document.embedding.isnot(None),
            _unexamined_predicate(),
        )
        .order_by(models.Document.created_at.asc(), models.Document.id.asc())
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    )
    result = await db.execute(stmt)
    rows = list(result.scalars().all())
    top_k = settings.ESTABLISHED.CANDIDATE_TOP_K
    items: list[_PromotionWorkItem] = []
    for doc in rows:
        shell = _seed_from_document(doc)
        if shell is None:
            # Unpromotable shape: still examined in phase 3 via a Leave stub.
            items.append(
                _PromotionWorkItem(
                    seed=PromotionSeed(
                        id=doc.id,
                        content=doc.content,
                        session_name=doc.session_name,
                        evidence=EvidenceKey(f"_invalid:{doc.id}"),
                        conversation=Conversation(
                            session_name=doc.session_name,
                            message_created_at=datetime.now(UTC),
                        ),
                    ),
                    content=doc.content,
                    embedding=list(doc.embedding) if doc.embedding is not None else [],
                    message_ids=(),
                    message_created_at="",
                    workspace_name=doc.workspace_name,
                    observer=doc.observer,
                    observed=doc.observed,
                    established=(),
                    peers=(),
                    peer_evidence={},
                    peer_conversations={},
                )
            )
            continue

        session_name = shell.seed.session_name
        assert session_name is not None
        established = await find_neighbours(
            db,
            shell.workspace_name,
            observer=shell.observer,
            observed=shell.observed,
            embedding=shell.embedding,
            scope=NeighbourScope.established(),
            max_distance=CANDIDATE_MAX,
            top_k=top_k,
        )
        peers = await find_neighbours(
            db,
            shell.workspace_name,
            observer=shell.observer,
            observed=shell.observed,
            embedding=shell.embedding,
            scope=NeighbourScope.working_cross_session(exclude_session=session_name),
            max_distance=CANDIDATE_MAX,
            top_k=top_k,
        )
        peer_evidence, peer_conversations = await _peer_maps(
            db,
            workspace_name=shell.workspace_name,
            observer=shell.observer,
            observed=shell.observed,
            peers=peers,
        )
        items.append(
            _PromotionWorkItem(
                seed=shell.seed,
                content=shell.content,
                embedding=shell.embedding,
                message_ids=shell.message_ids,
                message_created_at=shell.message_created_at,
                workspace_name=shell.workspace_name,
                observer=shell.observer,
                observed=shell.observed,
                established=tuple(established),
                peers=tuple(peers),
                peer_evidence=peer_evidence,
                peer_conversations=peer_conversations,
            )
        )
    return items


def _merge_confirmations(base: Confirmations, extra: Confirmations) -> Confirmations:
    merged = dict(base.by_neighbour)
    merged.update(extra.by_neighbour)
    return Confirmations(by_neighbour=merged)


async def _decide_one(
    item: _PromotionWorkItem,
    confirmer: Confirmer,
) -> tuple[PromotionVerdict, Confirmations]:
    if not item.message_ids or not item.message_created_at:
        return Leave(reason="no_evidence_key"), Confirmations.empty()
    if item.seed.session_name is None:
        return Leave(reason="not_explicit"), Confirmations.empty()

    confirmations = Confirmations.empty()
    top_k = settings.ESTABLISHED.CANDIDATE_TOP_K
    for _ in range(top_k):
        verdict = decide_promotion(
            seed=item.seed,
            established=item.established,
            peers=item.peers,
            confirmations=confirmations,
            peer_evidence=item.peer_evidence,
            peer_conversations=item.peer_conversations,
        )
        if not isinstance(verdict, NeedsConfirm):
            return verdict, confirmations
        batch = await confirmer.confirm(item.content, list(verdict.candidates))
        confirmations = _merge_confirmations(confirmations, batch)
        for candidate in verdict.candidates:
            proof = batch.for_neighbour(candidate.id)
            if proof is not None and proof.answer is ConfirmAnswer.UNDECIDED:
                return Leave(reason="undecided"), confirmations
    return Leave(reason="undecided"), confirmations


def _stamp_examined(row: models.Document, when: datetime) -> None:
    meta = dict(row.internal_metadata or {})
    meta[_PROMOTION_EXAMINED_AT] = when.isoformat()
    row.internal_metadata = meta
    flag_modified(row, "internal_metadata")


async def _refresh_verdict(
    db: AsyncSession,
    item: _PromotionWorkItem,
    verdict: PromotionVerdict,
    confirmations: Confirmations,
) -> PromotionVerdict:
    """Re-decide against live established neighbours before write.

    Earlier seeds in the same cycle may have minted; convert Promote→Reinforce.
    """
    if not isinstance(verdict, (Promote, Reinforce)) or not item.embedding:
        return verdict
    top_k = settings.ESTABLISHED.CANDIDATE_TOP_K
    established = await find_neighbours(
        db,
        item.workspace_name,
        observer=item.observer,
        observed=item.observed,
        embedding=item.embedding,
        scope=NeighbourScope.established(),
        max_distance=CANDIDATE_MAX,
        top_k=top_k,
    )
    return decide_promotion(
        seed=item.seed,
        established=established,
        peers=item.peers,
        confirmations=confirmations,
        peer_evidence=item.peer_evidence,
        peer_conversations=item.peer_conversations,
    )


async def _apply_one(
    db: AsyncSession,
    item: _PromotionWorkItem,
    verdict: PromotionVerdict,
    confirmations: Confirmations,
    metrics: PromotionCycleMetrics,
) -> None:
    """Mint/reinforce + stamp examined in this session. Caller commits."""
    now = datetime.now(UTC)
    seed_row = await db.get(models.Document, item.seed.id)
    if seed_row is None or seed_row.deleted_at is not None:
        return

    verdict = await _refresh_verdict(db, item, verdict, confirmations)

    if isinstance(verdict, Promote):
        await mint_established_row(
            db,
            workspace_name=item.workspace_name,
            observer=item.observer,
            observed=item.observed,
            content=item.content,
            embedding=item.embedding,
            level=verdict.level,
            source_ids=verdict.source_ids,
            evidence=verdict.evidence,
            message_ids=item.message_ids,
            message_created_at=item.message_created_at,
        )
        _stamp_examined(seed_row, now)
        metrics.promoted += 1
        metrics.examined += 1
        collection = await db.scalar(
            select(models.Collection).where(
                models.Collection.workspace_name == item.workspace_name,
                models.Collection.observer == item.observer,
                models.Collection.observed == item.observed,
            )
        )
        if collection is not None:
            try:
                scheduled = await check_and_schedule_dream(db, collection)
                if scheduled:
                    metrics.dream_hints += 1
            except Exception:
                logger.exception(
                    "Dream scheduling hint failed after promote for %s",
                    item.seed.id,
                )
        return

    if isinstance(verdict, Reinforce):
        await _apply_document_row_updates(
            db,
            [
                _DocumentRowOp(
                    "reinforce",
                    verdict.target_id,
                    evidence_key=str(verdict.evidence),
                )
            ],
            workspace_name=item.workspace_name,
            observer=item.observer,
            observed=item.observed,
        )
        # Re-load seed: reinforce locks a different row; stamp the seed after.
        seed_row = await db.get(models.Document, item.seed.id)
        if seed_row is not None and seed_row.deleted_at is None:
            _stamp_examined(seed_row, now)
        metrics.reinforced_established += 1
        metrics.examined += 1
        return

    # Leave / residual NeedsConfirm → stamp anyway (stateless progress).
    if isinstance(verdict, Leave) and verdict.reason == "undecided":
        metrics.undecided += 1
    else:
        metrics.left_working += 1
    _stamp_examined(seed_row, now)
    metrics.examined += 1


async def run_promotion_cycle(
    *,
    batch_size: int,
    confirmer: Confirmer,
) -> PromotionCycleMetrics:
    """One promotion batch. Never raises per-seed failures into the consumer."""
    metrics = PromotionCycleMetrics()
    mode = settings.ESTABLISHED.MODE
    if mode == "off":
        return metrics

    try:
        async with tracked_db("promotion_claim") as db:
            items = await _claim_and_snapshot(db, batch_size)
            await db.commit()
    except Exception:
        logger.exception("Promotion cycle phase 1 failed")
        return metrics

    if not items:
        return metrics

    verdicts: dict[str, PromotionVerdict] = {}
    confirmations_by_seed: dict[str, Confirmations] = {}
    for item in items:
        try:
            verdict, confirmations = await _decide_one(item, confirmer)
            verdicts[item.seed.id] = verdict
            confirmations_by_seed[item.seed.id] = confirmations
        except Exception:
            logger.exception(
                "Promotion confirm/decide failed for seed %s", item.seed.id
            )
            verdicts[item.seed.id] = Leave(reason="undecided")
            confirmations_by_seed[item.seed.id] = Confirmations.empty()

    if mode == "shadow":
        for item in items:
            v = verdicts[item.seed.id]
            metrics.shadow_verdicts.append(type(v).__name__)
            if isinstance(v, Promote):
                metrics.promoted += 1
            elif isinstance(v, Reinforce):
                metrics.reinforced_established += 1
            elif isinstance(v, Leave) and v.reason == "undecided":
                metrics.undecided += 1
            else:
                metrics.left_working += 1
        # No examined stamp in shadow so on→real still sees the work.
        return metrics

    for item in items:
        verdict = verdicts[item.seed.id]
        confirmations = confirmations_by_seed[item.seed.id]
        try:
            async with tracked_db("promotion_apply") as db:
                await _apply_one(db, item, verdict, confirmations, metrics)
                await db.commit()
        except Exception:
            logger.exception(
                "Promotion apply failed for seed %s; leaving unexamined",
                item.seed.id,
            )
            # Prefer re-examine on crash: do not stamp when apply failed.

    return metrics
