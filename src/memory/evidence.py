"""Evidence keys and independent-conversation counting for established memory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, NewType

from src.schemas.internal import DocumentCreate

EvidenceKey = NewType("EvidenceKey", str)

INDEPENDENCE_GAP: Final[timedelta] = timedelta(hours=6)
PROMOTION_MIN_CONVERSATIONS: Final[int] = 2


def evidence_key(session_name: str, message_ids: Sequence[int]) -> EvidenceKey:
    """Format ``session:min-max`` from a non-empty message id set."""
    if not message_ids:
        raise ValueError("message_ids must be non-empty")
    lo = min(message_ids)
    hi = max(message_ids)
    return EvidenceKey(f"{session_name}:{lo}-{hi}")


def evidence_key_from_document(doc: DocumentCreate) -> EvidenceKey | None:
    """Derive an evidence key from a document create payload.

    Returns None when the document has no session or empty message_ids.
    """
    if doc.session_name is None:
        return None
    message_ids = doc.metadata.message_ids
    if not message_ids:
        return None
    return evidence_key(doc.session_name, message_ids)


@dataclass(frozen=True, slots=True)
class Conversation:
    """One observation used for promotion independence counting."""

    session_name: str | None
    message_created_at: datetime


def count_independent_conversations(observations: Sequence[Conversation]) -> int:
    """Count conversations that are independent for promotion.

    Two observations are independent when they have different session_name or
    their message_created_at times differ by at least INDEPENDENCE_GAP.
    Same-session observations within the gap merge greedily after sorting by time.
    """
    if not observations:
        return 0

    ordered = sorted(observations, key=lambda o: o.message_created_at)
    groups = 0
    last_session: str | None = None
    last_time: datetime | None = None

    for obs in ordered:
        if last_time is None:
            groups = 1
            last_session = obs.session_name
            last_time = obs.message_created_at
            continue

        different_session = obs.session_name != last_session
        gap_ok = (obs.message_created_at - last_time) >= INDEPENDENCE_GAP
        if different_session or gap_ok:
            groups += 1
            last_session = obs.session_name
            last_time = obs.message_created_at

    return groups
