"""Promotion reconciler: pending gate, promote, reinforce, MODE, idempotency."""

from __future__ import annotations

import logging
import math

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from src import crud, models, schemas
from src.config import settings
from src.deriver.consumer import process_reconciler
from src.memory.confirm import NeverConfirmer
from src.reconciler.promote_established import (
    PromotionCycleMetrics,
    has_pending_promotion_work,
    run_promotion_cycle,
)
from src.schemas import ReconcilerType
from src.utils.queue_payload import ReconcilerPayload

_DIM = 1536


def _axis_embedding() -> list[float]:
    vector = [0.0] * _DIM
    vector[0] = 1.0
    return vector


def _embedding_at_distance(distance: float) -> list[float]:
    cosine = 1.0 - distance
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    vector = [0.0] * _DIM
    vector[0] = cosine
    vector[1] = sine
    return vector


@pytest.fixture
def established_mode(monkeypatch: pytest.MonkeyPatch):
    def _set(mode: str):
        monkeypatch.setattr(settings.ESTABLISHED, "MODE", mode)
        return mode

    return _set


def _explicit(
    content: str,
    *,
    embedding: list[float],
    session_name: str,
    message_id: int,
    message_created_at: str = "2026-01-01T00:00:00Z",
) -> schemas.DocumentCreate:
    return schemas.DocumentCreate(
        content=content,
        embedding=embedding,
        session_name=session_name,
        level="explicit",
        metadata=schemas.DocumentMetadata(
            message_ids=[message_id],
            message_created_at=message_created_at,
        ),
    )


def _derived(
    content: str,
    *,
    embedding: list[float],
    source_ids: list[str],
    level: str = "inductive",
) -> schemas.DocumentCreate:
    return schemas.DocumentCreate(
        content=content,
        embedding=embedding,
        session_name=None,
        level=level,  # type: ignore[arg-type]
        source_ids=source_ids,
        metadata=schemas.DocumentMetadata(
            message_ids=[1],
            message_created_at="2026-01-01T00:00:00Z",
            source_ids=source_ids,
        ),
    )


class TestPromoteEstablished:
    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session, models.Session]:
        observed = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session_a = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session_b = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([observed, session_a, session_b])
        await db_session.flush()
        db_session.add(
            models.Collection(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed.name,
            )
        )
        await db_session.commit()
        return observed, session_a, session_b

    async def _explicits(
        self, db_session: AsyncSession, workspace: str, observer: str, observed: str
    ) -> list[models.Document]:
        result = await db_session.execute(
            select(models.Document).where(
                models.Document.workspace_name == workspace,
                models.Document.observer == observer,
                models.Document.observed == observed,
                models.Document.level == "explicit",
                models.Document.deleted_at.is_(None),
            )
        )
        return list(result.scalars().all())

    async def _derived_rows(
        self, db_session: AsyncSession, workspace: str, observer: str, observed: str
    ) -> list[models.Document]:
        result = await db_session.execute(
            select(models.Document).where(
                models.Document.workspace_name == workspace,
                models.Document.observer == observer,
                models.Document.observed == observed,
                models.Document.level.in_(("inductive", "deductive")),
                models.Document.deleted_at.is_(None),
            )
        )
        return list(result.scalars().all())

    @pytest.mark.asyncio
    async def test_has_pending_true_then_false_after_examined(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
    ):
        test_workspace, test_peer = sample_data
        established_mode("off")
        observed, session_a, session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        assert await has_pending_promotion_work(db_session) is False

        await crud.create_documents(
            db_session,
            [
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_a.name,
                    message_id=1,
                ),
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_b.name,
                    message_id=2,
                    message_created_at="2026-01-02T00:00:00Z",
                ),
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        await db_session.commit()

        assert await has_pending_promotion_work(db_session) is True

        established_mode("on")
        metrics = await run_promotion_cycle(batch_size=50, confirmer=NeverConfirmer())
        assert metrics.examined == 2
        assert await has_pending_promotion_work(db_session) is False

    @pytest.mark.asyncio
    async def test_promote_two_independent_same_claim(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
    ):
        test_workspace, test_peer = sample_data
        established_mode("off")
        observed, session_a, session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_a.name,
                    message_id=10,
                ),
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_b.name,
                    message_id=20,
                    message_created_at="2026-06-01T00:00:00Z",
                ),
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        await db_session.commit()

        established_mode("on")
        metrics = await run_promotion_cycle(batch_size=50, confirmer=NeverConfirmer())

        assert metrics.promoted == 1
        assert metrics.examined == 2
        # Second seed in the same batch reinforces the mint.
        assert metrics.reinforced_established == 1

        derived = await self._derived_rows(
            db_session, test_workspace.name, test_peer.name, observed.name
        )
        assert len(derived) == 1
        assert derived[0].content == "Alice likes tea"

        explicits = await self._explicits(
            db_session, test_workspace.name, test_peer.name, observed.name
        )
        assert all(
            (doc.internal_metadata or {}).get("promotion_examined_at")
            for doc in explicits
        )

    @pytest.mark.asyncio
    async def test_same_session_peers_promote_across_independence_gap(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
    ):
        test_workspace, test_peer = sample_data
        established_mode("off")
        observed, session_a, _session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                _explicit(
                    "Alice likes tea in the morning",
                    embedding=_axis_embedding(),
                    session_name=session_a.name,
                    message_id=10,
                    message_created_at="2026-01-01T00:00:00Z",
                ),
                _explicit(
                    "Alice drinks tea before work",
                    embedding=_embedding_at_distance(0.04),
                    session_name=session_a.name,
                    message_id=20,
                    message_created_at="2026-01-01T07:00:00Z",
                ),
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        await db_session.commit()

        established_mode("on")
        metrics = await run_promotion_cycle(batch_size=50, confirmer=NeverConfirmer())

        assert metrics.promoted == 1
        assert metrics.reinforced_established == 1
        assert metrics.examined == 2

        derived = await self._derived_rows(
            db_session, test_workspace.name, test_peer.name, observed.name
        )
        assert len(derived) == 1

        explicits = await self._explicits(
            db_session, test_workspace.name, test_peer.name, observed.name
        )
        assert len(explicits) == 2
        assert all(
            (doc.internal_metadata or {}).get("promotion_examined_at")
            for doc in explicits
        )

    @pytest.mark.asyncio
    async def test_same_session_peers_within_gap_leave_working(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
    ):
        test_workspace, test_peer = sample_data
        established_mode("off")
        observed, session_a, _session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                _explicit(
                    "Alice likes tea in the morning",
                    embedding=_axis_embedding(),
                    session_name=session_a.name,
                    message_id=10,
                    message_created_at="2026-01-01T00:00:00Z",
                ),
                _explicit(
                    "Alice drinks tea before work",
                    embedding=_embedding_at_distance(0.04),
                    session_name=session_a.name,
                    message_id=20,
                    message_created_at="2026-01-01T01:00:00Z",
                ),
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        await db_session.commit()

        established_mode("on")
        metrics = await run_promotion_cycle(batch_size=50, confirmer=NeverConfirmer())

        assert metrics.promoted == 0
        assert metrics.reinforced_established == 0
        assert metrics.examined == 2
        assert (
            len(
                await self._derived_rows(
                    db_session, test_workspace.name, test_peer.name, observed.name
                )
            )
            == 0
        )
        explicits = await self._explicits(
            db_session, test_workspace.name, test_peer.name, observed.name
        )
        assert len(explicits) == 2
        assert all(doc.deleted_at is None for doc in explicits)
        assert all(
            (doc.internal_metadata or {}).get("promotion_examined_at")
            for doc in explicits
        )

    @pytest.mark.asyncio
    async def test_existing_established_reinforces_not_second_mint(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
    ):
        test_workspace, test_peer = sample_data
        established_mode("off")
        observed, session_a, session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        seed_result = await crud.create_documents(
            db_session,
            [
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_a.name,
                    message_id=1,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        seed_id = (
            await self._explicits(
                db_session, test_workspace.name, test_peer.name, observed.name
            )
        )[0].id
        del seed_result

        await crud.create_documents(
            db_session,
            [
                _derived(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    source_ids=[seed_id],
                ),
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_b.name,
                    message_id=2,
                    message_created_at="2026-06-01T00:00:00Z",
                ),
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        await db_session.commit()

        before = await self._derived_rows(
            db_session, test_workspace.name, test_peer.name, observed.name
        )
        assert len(before) == 1
        times_before = before[0].times_derived

        # Mark session_a seed examined so only session_b is claimed.
        a_docs = [
            d
            for d in await self._explicits(
                db_session, test_workspace.name, test_peer.name, observed.name
            )
            if d.session_name == session_a.name
        ]
        for doc in a_docs:
            meta = dict(doc.internal_metadata or {})
            meta["promotion_examined_at"] = "2026-01-01T00:00:00+00:00"
            doc.internal_metadata = meta
            flag_modified(doc, "internal_metadata")
        await db_session.commit()

        established_mode("on")
        metrics = await run_promotion_cycle(batch_size=50, confirmer=NeverConfirmer())

        assert metrics.promoted == 0
        assert metrics.reinforced_established == 1
        assert metrics.examined == 1

        workspace_name = test_workspace.name
        observer_name = test_peer.name
        observed_name = observed.name
        db_session.expire_all()
        after = await self._derived_rows(
            db_session, workspace_name, observer_name, observed_name
        )
        assert len(after) == 1
        assert after[0].times_derived == times_before + 1

    @pytest.mark.asyncio
    async def test_mode_off_no_mint(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
    ):
        test_workspace, test_peer = sample_data
        established_mode("off")
        observed, session_a, session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_a.name,
                    message_id=1,
                ),
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_b.name,
                    message_id=2,
                    message_created_at="2026-06-01T00:00:00Z",
                ),
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        await db_session.commit()

        metrics = await run_promotion_cycle(batch_size=50, confirmer=NeverConfirmer())
        assert metrics.examined == 0
        assert metrics.promoted == 0
        assert (
            len(
                await self._derived_rows(
                    db_session, test_workspace.name, test_peer.name, observed.name
                )
            )
            == 0
        )
        assert await has_pending_promotion_work(db_session) is True

    @pytest.mark.asyncio
    async def test_mode_shadow_metrics_no_write_no_stamp(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
    ):
        test_workspace, test_peer = sample_data
        established_mode("off")
        observed, session_a, session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_a.name,
                    message_id=1,
                ),
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_b.name,
                    message_id=2,
                    message_created_at="2026-06-01T00:00:00Z",
                ),
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        await db_session.commit()

        established_mode("shadow")
        metrics = await run_promotion_cycle(batch_size=50, confirmer=NeverConfirmer())

        assert "Promote" in metrics.shadow_verdicts
        assert metrics.promoted >= 1
        assert (
            len(
                await self._derived_rows(
                    db_session, test_workspace.name, test_peer.name, observed.name
                )
            )
            == 0
        )
        explicits = await self._explicits(
            db_session, test_workspace.name, test_peer.name, observed.name
        )
        assert all(
            not (doc.internal_metadata or {}).get("promotion_examined_at")
            for doc in explicits
        )
        assert await has_pending_promotion_work(db_session) is True

    @pytest.mark.asyncio
    async def test_promotion_consumer_logs_shadow_verdict_names(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        async def _cycle(*, batch_size, confirmer):
            del batch_size, confirmer
            return PromotionCycleMetrics(
                examined=1,
                promoted=1,
                shadow_verdicts=["Promote"],
            )

        monkeypatch.setattr(
            "src.deriver.consumer.run_promotion_cycle",
            _cycle,
        )
        caplog.set_level(logging.INFO, logger="src.deriver.consumer")
        await process_reconciler(
            ReconcilerPayload(reconciler_type=ReconcilerType.PROMOTE_ESTABLISHED)
        )
        line = next(
            rec.getMessage()
            for rec in caplog.records
            if "Promotion cycle complete:" in rec.getMessage()
        )
        assert "shadow=Promote" in line

    @pytest.mark.asyncio
    async def test_idempotent_rerun_no_double_mint(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
    ):
        test_workspace, test_peer = sample_data
        established_mode("off")
        observed, session_a, session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_a.name,
                    message_id=1,
                ),
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_b.name,
                    message_id=2,
                    message_created_at="2026-06-01T00:00:00Z",
                ),
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        await db_session.commit()

        established_mode("on")
        first = await run_promotion_cycle(batch_size=50, confirmer=NeverConfirmer())
        assert first.promoted == 1
        assert await has_pending_promotion_work(db_session) is False

        second = await run_promotion_cycle(batch_size=50, confirmer=NeverConfirmer())
        assert second.examined == 0
        assert second.promoted == 0

        derived = await self._derived_rows(
            db_session, test_workspace.name, test_peer.name, observed.name
        )
        assert len(derived) == 1

    @pytest.mark.asyncio
    async def test_leave_undecided_does_not_stamp(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
    ):
        test_workspace, test_peer = sample_data
        established_mode("off")
        observed, session_a, session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                _explicit(
                    "Alice likes tea in the morning",
                    embedding=_axis_embedding(),
                    session_name=session_a.name,
                    message_id=10,
                ),
                _explicit(
                    "Alice prefers coffee after lunch",
                    embedding=_embedding_at_distance(0.09),
                    session_name=session_b.name,
                    message_id=20,
                    message_created_at="2026-06-01T00:00:00Z",
                ),
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        await db_session.commit()

        established_mode("on")
        metrics = await run_promotion_cycle(batch_size=50, confirmer=NeverConfirmer())

        assert metrics.promoted == 0
        assert metrics.reinforced_established == 0
        assert metrics.undecided == 2
        assert metrics.examined == 0
        explicits = await self._explicits(
            db_session, test_workspace.name, test_peer.name, observed.name
        )
        assert len(explicits) == 2
        assert all(
            not (doc.internal_metadata or {}).get("promotion_examined_at")
            for doc in explicits
        )
        assert await has_pending_promotion_work(db_session) is True

    @pytest.mark.asyncio
    async def test_leave_unrelated_still_stamps(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
    ):
        test_workspace, test_peer = sample_data
        established_mode("off")
        observed, session_a, _session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                _explicit(
                    "Alice likes tea",
                    embedding=_axis_embedding(),
                    session_name=session_a.name,
                    message_id=10,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        await db_session.commit()

        established_mode("on")
        metrics = await run_promotion_cycle(batch_size=50, confirmer=NeverConfirmer())

        assert metrics.promoted == 0
        assert metrics.left_working == 1
        assert metrics.examined == 1
        explicits = await self._explicits(
            db_session, test_workspace.name, test_peer.name, observed.name
        )
        assert len(explicits) == 1
        assert (explicits[0].internal_metadata or {}).get("promotion_examined_at")
        assert await has_pending_promotion_work(db_session) is False
