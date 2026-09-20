"""Evidence key format and independent-conversation counting."""

from datetime import datetime, timedelta

from src.memory.evidence import (
    INDEPENDENCE_GAP,
    Conversation,
    count_independent_conversations,
    evidence_key,
    evidence_key_from_document,
)
from src.schemas.internal import DocumentCreate, DocumentMetadata

_T0 = datetime(2024, 1, 1, 12, 0, 0)


def _doc(
    *,
    session_name: str | None = "sess-a",
    message_ids: list[int] | None = None,
    level: str = "explicit",
) -> DocumentCreate:
    return DocumentCreate(
        content="observation",
        session_name=session_name,
        level=level,  # type: ignore[arg-type]
        metadata=DocumentMetadata(
            message_ids=message_ids if message_ids is not None else [10, 12],
            message_created_at="2024-01-01T12:00:00Z",
        ),
        embedding=[0.0, 0.0, 0.0],
    )


class TestEvidenceKey:
    def test_format_session_min_max(self):
        assert evidence_key("alpha", [5, 1, 9]) == "alpha:1-9"
        assert evidence_key("alpha", [7]) == "alpha:7-7"

    def test_from_document(self):
        assert evidence_key_from_document(_doc()) == "sess-a:10-12"

    def test_from_document_none_without_session(self):
        assert evidence_key_from_document(_doc(session_name=None)) is None

    def test_from_document_none_with_empty_message_ids(self):
        assert evidence_key_from_document(_doc(message_ids=[])) is None


class TestIndependentConversations:
    def test_same_session_within_gap_is_one(self):
        obs = [
            Conversation("s1", _T0),
            Conversation("s1", _T0 + INDEPENDENCE_GAP - timedelta(seconds=1)),
        ]
        assert count_independent_conversations(obs) == 1

    def test_different_session_is_two(self):
        obs = [
            Conversation("s1", _T0),
            Conversation("s2", _T0 + timedelta(minutes=1)),
        ]
        assert count_independent_conversations(obs) == 2

    def test_gap_at_least_six_hours_is_two(self):
        obs = [
            Conversation("s1", _T0),
            Conversation("s1", _T0 + INDEPENDENCE_GAP),
        ]
        assert count_independent_conversations(obs) == 2

    def test_empty_is_zero(self):
        assert count_independent_conversations([]) == 0
