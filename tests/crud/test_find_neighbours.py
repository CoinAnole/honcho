import datetime
import math

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.crud.document import NeighbourScope, find_neighbours
from src.memory.bands import (
    CANDIDATE_MAX,
    SAME_CLAIM_MAX,
    Band,
    classify_distance,
)

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


class TestClassifyDistance:
    def test_same_claim_includes_cutoff(self):
        assert classify_distance(0.0) is Band.SAME_CLAIM
        assert classify_distance(SAME_CLAIM_MAX) is Band.SAME_CLAIM

    def test_candidate_band(self):
        assert classify_distance(0.0500001) is Band.CANDIDATE
        assert classify_distance(CANDIDATE_MAX) is Band.CANDIDATE

    def test_unrelated_and_nan(self):
        assert classify_distance(0.1500001) is Band.UNRELATED
        assert classify_distance(float("nan")) is Band.UNRELATED


class TestFindNeighbours:
    async def _setup_test_data(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session, models.Session]:
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session_a = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session_b = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([test_peer2, session_a, session_b])
        await db_session.flush()
        db_session.add(
            models.Collection(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
            )
        )
        await db_session.flush()
        return test_peer2, session_a, session_b

    def _doc(
        self,
        content: str,
        *,
        embedding: list[float],
        session_name: str | None,
        level: str = "explicit",
        message_id: int = 1,
    ) -> schemas.DocumentCreate:
        return schemas.DocumentCreate(
            content=content,
            embedding=embedding,
            session_name=session_name,
            level=level,  # pyright: ignore[reportArgumentType]
            metadata=schemas.DocumentMetadata(
                message_ids=[message_id],
                message_created_at="2026-01-01T00:00:00Z",
            ),
        )

    async def _create(
        self,
        db_session: AsyncSession,
        docs: list[schemas.DocumentCreate],
        workspace_name: str,
        observer: str,
        observed: str,
    ) -> None:
        await crud.create_documents(
            db_session,
            docs,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

    @pytest.mark.asyncio
    async def test_ascending_distances_across_established_levels(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        test_workspace, test_peer = sample_data
        observed, session_a, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        await self._create(
            db_session,
            [
                self._doc(
                    "explicit near",
                    embedding=_embedding_at_distance(0.01),
                    session_name=session_a.name,
                    level="explicit",
                ),
                self._doc(
                    "inductive nearer",
                    embedding=_embedding_at_distance(0.02),
                    session_name=None,
                    level="inductive",
                    message_id=2,
                ),
                self._doc(
                    "deductive farther",
                    embedding=_embedding_at_distance(0.10),
                    session_name=None,
                    level="deductive",
                    message_id=3,
                ),
                self._doc(
                    "contradiction near",
                    embedding=_embedding_at_distance(0.03),
                    session_name=None,
                    level="contradiction",
                    message_id=4,
                ),
            ],
            test_workspace.name,
            test_peer.name,
            observed.name,
        )

        neighbours = await find_neighbours(
            db_session,
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            embedding=_axis_embedding(),
            scope=NeighbourScope.established(),
            max_distance=CANDIDATE_MAX,
            top_k=4,
        )

        contents = await self._contents_by_id(db_session, [n.id for n in neighbours])
        assert contents == ["inductive nearer", "deductive farther"]
        assert neighbours[0].distance <= neighbours[1].distance
        assert neighbours[0].distance == pytest.approx(0.02, abs=1e-5)
        assert neighbours[1].distance == pytest.approx(0.10, abs=1e-5)
        assert {n.level for n in neighbours} == {"inductive", "deductive"}

    @pytest.mark.asyncio
    async def test_max_distance_filters_far_neighbours(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        test_workspace, test_peer = sample_data
        observed, session_a, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        await self._create(
            db_session,
            [
                self._doc(
                    "near",
                    embedding=_embedding_at_distance(0.02),
                    session_name=session_a.name,
                ),
                self._doc(
                    "far",
                    embedding=_embedding_at_distance(0.20),
                    session_name=session_a.name,
                    message_id=2,
                ),
            ],
            test_workspace.name,
            test_peer.name,
            observed.name,
        )

        neighbours = await find_neighbours(
            db_session,
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            embedding=_axis_embedding(),
            scope=NeighbourScope.working(
                self._doc(
                    "query",
                    embedding=_axis_embedding(),
                    session_name=session_a.name,
                )
            ),
            max_distance=SAME_CLAIM_MAX,
            top_k=5,
        )

        contents = await self._contents_by_id(db_session, [n.id for n in neighbours])
        assert contents == ["near"]
        assert neighbours[0].distance <= SAME_CLAIM_MAX

    @pytest.mark.asyncio
    async def test_working_scope_respects_session_for_explicit(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        test_workspace, test_peer = sample_data
        observed, session_a, session_b = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        await self._create(
            db_session,
            [
                self._doc(
                    "session a",
                    embedding=_embedding_at_distance(0.04),
                    session_name=session_a.name,
                ),
                self._doc(
                    "session b closer",
                    embedding=_embedding_at_distance(0.01),
                    session_name=session_b.name,
                    message_id=2,
                ),
            ],
            test_workspace.name,
            test_peer.name,
            observed.name,
        )

        query_doc = self._doc(
            "probe",
            embedding=_axis_embedding(),
            session_name=session_a.name,
        )
        neighbours = await find_neighbours(
            db_session,
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            embedding=_axis_embedding(),
            scope=NeighbourScope.working(query_doc),
            max_distance=CANDIDATE_MAX,
            top_k=5,
        )

        contents = await self._contents_by_id(db_session, [n.id for n in neighbours])
        assert contents == ["session a"]
        assert neighbours[0].session_name == session_a.name

    def test_working_scope_none_for_sessionless_explicit(self):
        doc = self._doc(
            "global explicit",
            embedding=_axis_embedding(),
            session_name=None,
        )
        assert NeighbourScope.working(doc) is None

    @pytest.mark.asyncio
    async def test_soft_deleted_excluded(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        test_workspace, test_peer = sample_data
        observed, session_a, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        await self._create(
            db_session,
            [
                self._doc(
                    "live far",
                    embedding=_embedding_at_distance(0.04),
                    session_name=session_a.name,
                ),
                self._doc(
                    "deleted nearer",
                    embedding=_embedding_at_distance(0.01),
                    session_name=session_a.name,
                    message_id=2,
                ),
            ],
            test_workspace.name,
            test_peer.name,
            observed.name,
        )

        result = await db_session.execute(
            select(models.Document).where(
                models.Document.workspace_name == test_workspace.name,
                models.Document.observer == test_peer.name,
                models.Document.observed == observed.name,
            )
        )
        docs = {doc.content: doc for doc in result.scalars().all()}
        docs["deleted nearer"].deleted_at = datetime.datetime.now(datetime.UTC)
        await db_session.commit()

        query_doc = self._doc(
            "probe",
            embedding=_axis_embedding(),
            session_name=session_a.name,
        )
        neighbours = await find_neighbours(
            db_session,
            test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
            embedding=_axis_embedding(),
            scope=NeighbourScope.working(query_doc),
            max_distance=CANDIDATE_MAX,
            top_k=5,
        )

        contents = await self._contents_by_id(db_session, [n.id for n in neighbours])
        assert contents == ["live far"]

    async def _contents_by_id(
        self, db_session: AsyncSession, document_ids: list[str]
    ) -> list[str]:
        if not document_ids:
            return []
        result = await db_session.execute(
            select(models.Document).where(models.Document.id.in_(document_ids))
        )
        by_id = {doc.id: doc.content for doc in result.scalars().all()}
        return [by_id[doc_id] for doc_id in document_ids if doc_id in by_id]
