"""Public memory package surface.

Eager exports stay limited to ``bands`` so ``src.crud.document`` can import
cutoffs without a circular import through confirm/verdict.
"""

from __future__ import annotations

from typing import Any

from src.memory.bands import (
    CANDIDATE_MAX,
    SAME_CLAIM_MAX,
    Band,
    classify_distance,
)

__all__ = [
    "SAME_CLAIM_MAX",
    "CANDIDATE_MAX",
    "Band",
    "classify_distance",
    "EvidenceKey",
    "Conversation",
    "INDEPENDENCE_GAP",
    "PROMOTION_MIN_CONVERSATIONS",
    "evidence_key",
    "evidence_key_from_document",
    "count_independent_conversations",
    "ConfirmAnswer",
    "ClaimKind",
    "Confirmed",
    "Confirmations",
    "Confirmer",
    "ConfirmNeighbour",
    "NeverConfirmer",
    "LLMConfirmer",
    "confirmer_from_settings",
    "Reinforce",
    "Supersede",
    "Promote",
    "NeedsConfirm",
    "Leave",
    "Verdict",
    "PromotionSeed",
    "decide_established",
    "decide_promotion",
]


def __getattr__(name: str) -> Any:
    if name in {
        "EvidenceKey",
        "Conversation",
        "INDEPENDENCE_GAP",
        "PROMOTION_MIN_CONVERSATIONS",
        "evidence_key",
        "evidence_key_from_document",
        "count_independent_conversations",
    }:
        from src.memory import evidence as _evidence

        return getattr(_evidence, name)
    if name in {
        "ConfirmAnswer",
        "ClaimKind",
        "Confirmed",
        "Confirmations",
        "Confirmer",
        "ConfirmNeighbour",
        "NeverConfirmer",
        "LLMConfirmer",
        "confirmer_from_settings",
    }:
        from src.memory import confirm as _confirm

        return getattr(_confirm, name)
    if name in {
        "Reinforce",
        "Supersede",
        "Promote",
        "NeedsConfirm",
        "Leave",
        "Verdict",
        "PromotionSeed",
        "decide_established",
        "decide_promotion",
    }:
        from src.memory import verdict as _verdict

        return getattr(_verdict, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
