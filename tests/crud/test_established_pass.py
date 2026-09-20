"""Established pass behaviour: MODE off/shadow/on, reinforce, supersede, evidence."""

from __future__ import annotations

import logging
import math

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.crud.document import find_neighbours, NeighbourScope
from src.crud.established import (
    _AcceptedExplicit,
    mint_established_row,
    run_established_pass,
)
from src.memory.bands import CANDIDATE_MAX
from src.memory.confirm import ClaimKind, ConfirmAnswer, Confirmations, NeverConfirmer
from src.memory.evidence import EvidenceKey

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


class FakeConfirmer:
    def __init__(self, answer: ConfirmAnswer, claim_kind: ClaimKind | None = None):
        self.answer = answer
        self.claim_kind = claim_kind

    async def confirm(self, new_content: str, candidates):
        del new_content
        return Confirmations.from_answers(
            [
                (n.id, self.answer, self.claim_kind, n.distance)
                for n in candidates
            ]
        )


@pytest.fixture
def established_mode(monkeypatch: pytest.MonkeyPatch):
    def _set(mode: str):
        monkeypatch.setattr(settings.ESTABLISHED, "MODE", mode)
        return mode

    return _set


class TestEstablishedPass:
    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session]:
        observed = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([observed, session])
        await db_session.flush()
        db_session.add(
            models.Collection(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed.name,
            )
        )
        await db_session.flush()
        return observed, session

    def _explicit(
        self,
        content: str,
        *,
        embedding: list[float],
        session_name: str,
        message_id: int = 10,
    ) -> schemas.DocumentCreate:
        return schemas.DocumentCreate(
            content=content,
            embedding=embedding,
            session_name=session_name,
            level="explicit",
            metadata=schemas.DocumentMetadata(
                message_ids=[message_id],
                message_created_at="2026-01-01T00:00:00Z",
            ),
        )

    def _derived(
        self,
        content: str,
        *,
        embedding: list[float],
        source_ids: list[str],
        level: str = "deductive",
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

    @pytest.mark.asyncio
    async def test_mode_off_no_side_effects(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
    ):
        test_workspace, test_peer = sample_data
        established_mode("off")
        observed, session = await self._setup(db_session, test_workspace, test_peer)
        seed_explicit = await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "seed",
                    embedding=_axis_embedding(),
                    session_name=session.name,
                    message_id=1,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        source_id = (
            await db_session.execute(
                select(models.Document.id).where(
                    models.Document.content == "seed",
                    models.Document.deleted_at.is_(None),
                )
            )
        ).scalar_one()
        await crud.create_documents(
            db_session,
            [
                self._derived(
                    "Alice works at Blue",
                    embedding=_axis_embedding(),
                    source_ids=[source_id],
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        result = await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "Alice works at Blue Facility",
                    embedding=_axis_embedding(),
                    session_name=session.name,
                    message_id=20,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        assert result.established.reinforced == []
        assert result.established.superseded == []
        assert result.established.shadow_verdicts == []
        established = (
            await db_session.execute(
                select(models.Document).where(
                    models.Document.level == "deductive",
                    models.Document.deleted_at.is_(None),
                )
            )
        ).scalar_one()
        assert established.last_reinforced_at is None
        assert established.times_derived == 1
        del seed_explicit

    @pytest.mark.asyncio
    async def test_mode_shadow_counts_without_writes(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        test_workspace, test_peer = sample_data
        established_mode("shadow")
        caplog.set_level(logging.INFO, logger="src.crud.established")
        monkeypatch.setattr(
            "src.memory.confirm.confirmer_from_settings",
            lambda _s: NeverConfirmer(),
        )
        observed, session = await self._setup(db_session, test_workspace, test_peer)
        await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "seed",
                    embedding=_axis_embedding(),
                    session_name=session.name,
                    message_id=1,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        source_id = (
            await db_session.execute(
                select(models.Document.id).where(models.Document.content == "seed")
            )
        ).scalar_one()
        await crud.create_documents(
            db_session,
            [
                self._derived(
                    "Alice works at Blue",
                    embedding=_axis_embedding(),
                    source_ids=[source_id],
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        # Candidate-band neighbour (NeverConfirmer → Leave undecided)
        result = await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "Alice works at Blue Facility",
                    embedding=_embedding_at_distance(0.08),
                    session_name=session.name,
                    message_id=30,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        assert result.established.shadow_verdicts
        assert result.established.left_working >= 1
        line = next(
            rec.getMessage()
            for rec in caplog.records
            if "Established pass complete:" in rec.getMessage()
        )
        assert "shadow=Leave" in line
        established = (
            await db_session.execute(
                select(models.Document).where(
                    models.Document.level == "deductive",
                    models.Document.deleted_at.is_(None),
                )
            )
        ).scalar_one()
        assert established.last_reinforced_at is None
        assert established.deleted_at is None

    @pytest.mark.asyncio
    async def test_mode_on_never_confirmer_same_claim_reinforces(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
        monkeypatch: pytest.MonkeyPatch,
    ):
        test_workspace, test_peer = sample_data
        established_mode("on")
        monkeypatch.setattr(
            "src.memory.confirm.confirmer_from_settings",
            lambda _s: NeverConfirmer(),
        )
        observed, session = await self._setup(db_session, test_workspace, test_peer)
        await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "seed",
                    embedding=_axis_embedding(),
                    session_name=session.name,
                    message_id=1,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        source_id = (
            await db_session.execute(
                select(models.Document.id).where(models.Document.content == "seed")
            )
        ).scalar_one()
        await crud.create_documents(
            db_session,
            [
                self._derived(
                    "Alice works at Blue",
                    embedding=_axis_embedding(),
                    source_ids=[source_id],
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        established_before = (
            await db_session.execute(
                select(models.Document).where(models.Document.level == "deductive")
            )
        ).scalar_one()
        assert established_before.times_derived == 1

        result = await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "Alice works at Blue Facility",
                    embedding=_embedding_at_distance(0.02),
                    session_name=session.name,
                    message_id=40,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        assert established_before.id in result.established.reinforced
        await db_session.refresh(established_before)
        assert established_before.times_derived == 2
        assert established_before.last_reinforced_at is not None
        ledger = (
            await db_session.execute(
                select(models.EstablishedEvidence).where(
                    models.EstablishedEvidence.established_id == established_before.id
                )
            )
        ).scalars().all()
        assert len(ledger) == 1
        assert ledger[0].evidence_digest == f"{session.name}:40-40"

    @pytest.mark.asyncio
    async def test_mode_on_never_confirmer_candidate_leaves(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
        monkeypatch: pytest.MonkeyPatch,
    ):
        test_workspace, test_peer = sample_data
        established_mode("on")
        monkeypatch.setattr(
            "src.memory.confirm.confirmer_from_settings",
            lambda _s: NeverConfirmer(),
        )
        observed, session = await self._setup(db_session, test_workspace, test_peer)
        await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "seed",
                    embedding=_axis_embedding(),
                    session_name=session.name,
                    message_id=1,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        source_id = (
            await db_session.execute(
                select(models.Document.id).where(models.Document.content == "seed")
            )
        ).scalar_one()
        await crud.create_documents(
            db_session,
            [
                self._derived(
                    "Alice works at Blue",
                    embedding=_axis_embedding(),
                    source_ids=[source_id],
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        result = await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "Alice works at Blue Facility",
                    embedding=_embedding_at_distance(0.08),
                    session_name=session.name,
                    message_id=50,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        assert result.established.reinforced == []
        assert result.established.superseded == []
        assert result.established.left_working >= 1
        assert result.established.awaiting_confirm_undecided >= 1
        established = (
            await db_session.execute(
                select(models.Document).where(models.Document.level == "deductive")
            )
        ).scalar_one()
        assert established.deleted_at is None
        assert established.last_reinforced_at is None

    @pytest.mark.asyncio
    async def test_mode_on_supersede_mints_and_soft_deletes(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
        monkeypatch: pytest.MonkeyPatch,
    ):
        test_workspace, test_peer = sample_data
        established_mode("on")
        fake = FakeConfirmer(
            ConfirmAnswer.SAME_SUBJECT_NEW_VALUE, claim_kind=ClaimKind.STATE
        )
        monkeypatch.setattr(
            "src.memory.confirm.confirmer_from_settings",
            lambda _s: fake,
        )
        observed, session = await self._setup(db_session, test_workspace, test_peer)
        await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "seed",
                    embedding=_axis_embedding(),
                    session_name=session.name,
                    message_id=1,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        source_id = (
            await db_session.execute(
                select(models.Document.id).where(models.Document.content == "seed")
            )
        ).scalar_one()
        await crud.create_documents(
            db_session,
            [
                self._derived(
                    "Alice works at Blue",
                    embedding=_axis_embedding(),
                    source_ids=[source_id],
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        loser = (
            await db_session.execute(
                select(models.Document).where(models.Document.level == "deductive")
            )
        ).scalar_one()

        result = await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "Alice works at Red Facility",
                    embedding=_embedding_at_distance(0.09),
                    session_name=session.name,
                    message_id=60,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        assert result.established.superseded
        winner_id, loser_id = result.established.superseded[0]
        assert loser_id == loser.id
        await db_session.refresh(loser)
        assert loser.deleted_at is not None
        assert loser.internal_metadata.get("superseded_by") == winner_id
        winner = (
            await db_session.execute(
                select(models.Document).where(models.Document.id == winner_id)
            )
        ).scalar_one()
        assert winner.deleted_at is None
        assert winner.level == "deductive"
        assert loser.id in (winner.internal_metadata.get("supersedes") or [])

        # Evidence key replay: second reinforce same key is no-op on times_derived.
        td_before = winner.times_derived
        snap = _AcceptedExplicit(
            id=str(generate_nanoid()),
            content="Alice works at Red Facility",
            embedding=_embedding_at_distance(0.02),
            session_name=session.name,
            message_ids=(60,),
            message_created_at="2026-01-01T00:00:00Z",
            evidence=EvidenceKey(f"{session.name}:60-60"),
        )
        # Neighbour search will hit winner at same-claim; reinforce same evidence.
        again = await run_established_pass(
            [snap],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            confirmer=NeverConfirmer(),
            mode="on",
        )
        assert winner.id in again.reinforced
        await db_session.refresh(winner)
        assert winner.times_derived == td_before

    @pytest.mark.asyncio
    async def test_evidence_replay_noop_on_times_derived(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        established_mode,
        monkeypatch: pytest.MonkeyPatch,
    ):
        test_workspace, test_peer = sample_data
        established_mode("on")
        monkeypatch.setattr(
            "src.memory.confirm.confirmer_from_settings",
            lambda _s: NeverConfirmer(),
        )
        observed, session = await self._setup(db_session, test_workspace, test_peer)
        await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "seed",
                    embedding=_axis_embedding(),
                    session_name=session.name,
                    message_id=1,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        source_id = (
            await db_session.execute(
                select(models.Document.id).where(models.Document.content == "seed")
            )
        ).scalar_one()
        await crud.create_documents(
            db_session,
            [
                self._derived(
                    "habit: tea",
                    embedding=_axis_embedding(),
                    source_ids=[source_id],
                    level="inductive",
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        first = await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "Alice drinks tea",
                    embedding=_embedding_at_distance(0.01),
                    session_name=session.name,
                    message_id=70,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
        established = (
            await db_session.execute(
                select(models.Document).where(models.Document.level == "inductive")
            )
        ).scalar_one()
        assert established.times_derived == 2
        assert first.established.reinforced

        # Same evidence key again via direct pass.
        snap = _AcceptedExplicit(
            id=str(generate_nanoid()),
            content="Alice drinks tea again",
            embedding=_embedding_at_distance(0.01),
            session_name=session.name,
            message_ids=(70,),
            message_created_at="2026-01-01T00:00:00Z",
            evidence=EvidenceKey(f"{session.name}:70-70"),
        )
        await run_established_pass(
            [snap],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            confirmer=NeverConfirmer(),
            mode="on",
        )
        await db_session.refresh(established)
        assert established.times_derived == 2

    @pytest.mark.asyncio
    async def test_neighbour_content_populated(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        test_workspace, test_peer = sample_data
        observed, session = await self._setup(db_session, test_workspace, test_peer)
        await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "seed",
                    embedding=_axis_embedding(),
                    session_name=session.name,
                    message_id=1,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        source_id = (
            await db_session.execute(
                select(models.Document.id).where(models.Document.content == "seed")
            )
        ).scalar_one()
        await crud.create_documents(
            db_session,
            [
                self._derived(
                    "content for llm",
                    embedding=_axis_embedding(),
                    source_ids=[source_id],
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        nbrs = await find_neighbours(
            db_session,
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            embedding=_axis_embedding(),
            scope=NeighbourScope.established(),
            max_distance=CANDIDATE_MAX,
            top_k=4,
        )
        assert nbrs
        assert nbrs[0].content == "content for llm"

    @pytest.mark.asyncio
    async def test_mint_routes_through_create_documents(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        test_workspace, test_peer = sample_data
        observed, session = await self._setup(db_session, test_workspace, test_peer)
        await crud.create_documents(
            db_session,
            [
                self._explicit(
                    "seed",
                    embedding=_axis_embedding(),
                    session_name=session.name,
                    message_id=1,
                )
            ],
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            _established_pass=False,
        )
        source_id = (
            await db_session.execute(
                select(models.Document.id).where(models.Document.content == "seed")
            )
        ).scalar_one()
        winner = await mint_established_row(
            db_session,
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            content="minted winner",
            embedding=_axis_embedding(),
            level="deductive",
            source_ids=[source_id],
            evidence=[EvidenceKey(f"{session.name}:1-1")],
            message_ids=[1],
            message_created_at="2026-01-01T00:00:00Z",
        )
        assert winner.level == "deductive"
        assert winner.session_name is None
        again = await mint_established_row(
            db_session,
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            content="minted winner",
            embedding=_axis_embedding(),
            level="deductive",
            source_ids=[source_id],
            evidence=[EvidenceKey(f"{session.name}:1-1")],
            message_ids=[1],
            message_created_at="2026-01-01T00:00:00Z",
        )
        assert again.id == winner.id
        count = (
            await db_session.execute(
                select(models.Document).where(models.Document.content == "minted winner")
            )
        ).scalars().all()
        assert len(count) == 1
