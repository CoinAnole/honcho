"""decide_established / decide_promotion behaviour matrix."""

from datetime import datetime, timedelta

from src.crud.document import Neighbour
from src.memory.confirm import ConfirmAnswer, Confirmations
from src.memory.evidence import Conversation, EvidenceKey
from src.memory.verdict import (
    Leave,
    NeedsConfirm,
    Promote,
    PromotionSeed,
    Reinforce,
    Supersede,
    decide_established,
    decide_promotion,
)
from src.schemas.internal import DocumentCreate, DocumentMetadata

_T0 = datetime(2024, 1, 1, 12, 0, 0)


def _doc(
    *,
    session_name: str | None = "sess-a",
    level: str = "explicit",
    message_ids: list[int] | None = None,
) -> DocumentCreate:
    return DocumentCreate(
        content="Alice works at Blue Facility",
        session_name=session_name,
        level=level,  # type: ignore[arg-type]
        metadata=DocumentMetadata(
            message_ids=message_ids if message_ids is not None else [1, 3],
            message_created_at="2024-01-01T12:00:00Z",
        ),
        embedding=[0.1, 0.2, 0.3],
    )


def _n(
    nid: str,
    distance: float,
    *,
    level: str = "deductive",
    session_name: str | None = None,
) -> Neighbour:
    return Neighbour(
        id=nid,
        distance=distance,
        level=level,  # type: ignore[arg-type]
        session_name=session_name,
    )


def _seed(*, session: str = "sess-a") -> PromotionSeed:
    return PromotionSeed(
        id="seed-1",
        content="Alice prefers tea",
        session_name=session,
        evidence=EvidenceKey(f"{session}:1-1"),
        conversation=Conversation(session, _T0),
    )


class TestDecideEstablished:
    def test_same_claim_band_auto_reinforce(self):
        v = decide_established(
            new=_doc(),
            neighbours=[_n("est-1", 0.04), _n("est-2", 0.09)],
            confirmations=Confirmations.empty(),
        )
        assert isinstance(v, Reinforce)
        assert v.target_id == "est-1"
        assert v.evidence == "sess-a:1-3"

    def test_candidate_needs_confirm_nearest_only(self):
        v = decide_established(
            new=_doc(),
            neighbours=[_n("near", 0.07), _n("far", 0.12)],
            confirmations=Confirmations.empty(),
        )
        assert isinstance(v, NeedsConfirm)
        assert len(v.candidates) == 1
        assert v.candidates[0].id == "near"

    def test_same_claim_confirmation_reinforces(self):
        conf = Confirmations.from_answers(
            [
                ("near", ConfirmAnswer.SAME_CLAIM, None, 0.07),
                ("far", ConfirmAnswer.SAME_SUBJECT_NEW_VALUE, None, 0.12),
            ]
        )
        v = decide_established(
            new=_doc(),
            neighbours=[_n("near", 0.07), _n("far", 0.12)],
            confirmations=conf,
        )
        assert isinstance(v, Reinforce)
        assert v.target_id == "near"

    def test_same_claim_beats_new_value(self):
        conf = Confirmations.from_answers(
            [
                ("a", ConfirmAnswer.SAME_SUBJECT_NEW_VALUE, None, 0.08),
                ("b", ConfirmAnswer.SAME_CLAIM, None, 0.10),
            ]
        )
        v = decide_established(
            new=_doc(),
            neighbours=[_n("a", 0.08), _n("b", 0.10)],
            confirmations=conf,
        )
        assert isinstance(v, Reinforce)
        assert v.target_id == "b"

    def test_new_value_supersedes(self):
        conf = Confirmations.from_answers(
            [("loser", ConfirmAnswer.SAME_SUBJECT_NEW_VALUE, None, 0.09)]
        )
        v = decide_established(
            new=_doc(),
            neighbours=[_n("loser", 0.09, level="inductive")],
            confirmations=conf,
        )
        assert isinstance(v, Supersede)
        assert v.loser_id == "loser"
        assert v.winner_level == "inductive"
        assert v.proof.answer is ConfirmAnswer.SAME_SUBJECT_NEW_VALUE

    def test_undecided_leaves(self):
        conf = Confirmations.from_answers([("n1", ConfirmAnswer.UNDECIDED, None, 0.09)])
        v = decide_established(
            new=_doc(),
            neighbours=[_n("n1", 0.09)],
            confirmations=conf,
        )
        assert isinstance(v, Leave)
        assert v.reason == "undecided"

    def test_unrelated_leaves(self):
        conf = Confirmations.from_answers([("n1", ConfirmAnswer.UNRELATED, None, 0.09)])
        v = decide_established(
            new=_doc(),
            neighbours=[_n("n1", 0.09)],
            confirmations=conf,
        )
        assert isinstance(v, Leave)
        assert v.reason == "unrelated"

    def test_not_explicit(self):
        v = decide_established(
            new=_doc(level="deductive", session_name=None),
            neighbours=[_n("n1", 0.04)],
            confirmations=Confirmations.empty(),
        )
        assert isinstance(v, Leave)
        assert v.reason == "not_explicit"

    def test_session_less_explicit(self):
        v = decide_established(
            new=_doc(session_name=None),
            neighbours=[_n("n1", 0.04)],
            confirmations=Confirmations.empty(),
        )
        assert isinstance(v, Leave)
        assert v.reason == "not_explicit"

    def test_empty_neighbours(self):
        v = decide_established(
            new=_doc(),
            neighbours=[],
            confirmations=Confirmations.empty(),
        )
        assert isinstance(v, Leave)
        assert v.reason == "no_neighbours"

    def test_no_evidence_key(self):
        v = decide_established(
            new=_doc(message_ids=[]),
            neighbours=[_n("n1", 0.04)],
            confirmations=Confirmations.empty(),
        )
        assert isinstance(v, Leave)
        assert v.reason == "no_evidence_key"


class TestDecidePromotion:
    def test_reinforce_existing_established_same_claim_band(self):
        v = decide_promotion(
            seed=_seed(),
            established=[_n("est", 0.03)],
            peers=[_n("peer", 0.04, session_name="sess-b", level="explicit")],
            confirmations=Confirmations.empty(),
            peer_evidence={},
            peer_conversations={},
        )
        assert isinstance(v, Reinforce)
        assert v.target_id == "est"

    def test_promote_with_independent_peer(self):
        peer = _n("peer", 0.04, session_name="sess-b", level="explicit")
        v = decide_promotion(
            seed=_seed(),
            established=[],
            peers=[peer],
            confirmations=Confirmations.empty(),
            peer_evidence={"peer": EvidenceKey("sess-b:2-2")},
            peer_conversations={
                "peer": Conversation("sess-b", _T0 + timedelta(minutes=5))
            },
        )
        assert isinstance(v, Promote)
        assert v.level == "inductive"
        assert "seed-1" in v.source_ids
        assert "peer" in v.source_ids

    def test_new_value_peer_never_counts(self):
        peer = _n("peer", 0.08, session_name="sess-b", level="explicit")
        conf = Confirmations.from_answers(
            [("peer", ConfirmAnswer.SAME_SUBJECT_NEW_VALUE, None, 0.08)]
        )
        v = decide_promotion(
            seed=_seed(),
            established=[],
            peers=[peer],
            confirmations=conf,
            peer_evidence={"peer": EvidenceKey("sess-b:2-2")},
            peer_conversations={
                "peer": Conversation("sess-b", _T0 + timedelta(hours=1))
            },
        )
        assert isinstance(v, Leave)

    def test_established_new_value_leaves_despite_same_claim_peer(self):
        peer = _n("peer", 0.04, session_name="sess-b", level="explicit")
        conf = Confirmations.from_answers(
            [("est", ConfirmAnswer.SAME_SUBJECT_NEW_VALUE, None, 0.09)]
        )
        v = decide_promotion(
            seed=_seed(),
            established=[_n("est", 0.09)],
            peers=[peer],
            confirmations=conf,
            peer_evidence={"peer": EvidenceKey("sess-b:2-2")},
            peer_conversations={
                "peer": Conversation("sess-b", _T0 + timedelta(hours=7))
            },
        )
        assert isinstance(v, Leave)
        assert v.reason == "unrelated"

    def test_same_session_within_gap_does_not_promote(self):
        peer = _n("peer", 0.04, session_name="sess-a", level="explicit")
        v = decide_promotion(
            seed=_seed(),
            established=[],
            peers=[peer],
            confirmations=Confirmations.empty(),
            peer_evidence={"peer": EvidenceKey("sess-a:9-9")},
            peer_conversations={
                "peer": Conversation("sess-a", _T0 + timedelta(hours=1))
            },
        )
        assert isinstance(v, Leave)
        assert v.reason == "unrelated"
