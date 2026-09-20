"""Pure verdict decisions for established writes and promotion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.memory.bands import Band, classify_distance
from src.memory.confirm import ClaimKind, ConfirmAnswer, Confirmations, Confirmed
from src.memory.evidence import (
    PROMOTION_MIN_CONVERSATIONS,
    Conversation,
    EvidenceKey,
    count_independent_conversations,
    evidence_key_from_document,
)
from src.utils.types import DocumentLevel

if TYPE_CHECKING:
    from src.crud.document import Neighbour
    from src.schemas.internal import DocumentCreate


@dataclass(frozen=True, slots=True)
class Reinforce:
    target_id: str
    evidence: EvidenceKey


@dataclass(frozen=True, slots=True)
class Supersede:
    loser_id: str
    proof: Confirmed
    evidence: EvidenceKey
    winner_level: DocumentLevel

    def __post_init__(self) -> None:
        if self.proof.answer != ConfirmAnswer.SAME_SUBJECT_NEW_VALUE:
            raise ValueError(
                "Supersede requires proof.answer == SAME_SUBJECT_NEW_VALUE"
            )


@dataclass(frozen=True, slots=True)
class Promote:
    level: DocumentLevel
    proofs: tuple[Confirmed, ...]
    evidence: tuple[EvidenceKey, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NeedsConfirm:
    candidates: tuple[Neighbour, ...]


@dataclass(frozen=True, slots=True)
class Leave:
    reason: Literal[
        "no_neighbours",
        "unrelated",
        "undecided",
        "no_evidence_key",
        "not_explicit",
    ]


Verdict = Reinforce | Supersede | NeedsConfirm | Leave
PromotionVerdict = Promote | Reinforce | NeedsConfirm | Leave


@dataclass(frozen=True, slots=True)
class PromotionSeed:
    """Seed explicit row for promotion decisions (no DB / Neighbour expansion)."""

    id: str
    content: str
    session_name: str | None
    evidence: EvidenceKey
    conversation: Conversation


def decide_established(
    *,
    new: DocumentCreate,
    neighbours: Sequence[Neighbour],
    confirmations: Confirmations,
) -> Verdict:
    """Pure established-write verdict. First matching rule wins."""
    if new.level != "explicit" or new.session_name is None:
        return Leave(reason="not_explicit")

    evidence = evidence_key_from_document(new)
    if evidence is None:
        return Leave(reason="no_evidence_key")

    if not neighbours:
        return Leave(reason="no_neighbours")

    nearest = neighbours[0]
    if classify_distance(nearest.distance) is Band.SAME_CLAIM:
        return Reinforce(target_id=nearest.id, evidence=evidence)

    candidates = [
        n for n in neighbours if classify_distance(n.distance) is Band.CANDIDATE
    ]

    same_claim = _nearest_confirmed(
        candidates, confirmations, ConfirmAnswer.SAME_CLAIM
    )
    if same_claim is not None:
        _proof, neighbour = same_claim
        return Reinforce(target_id=neighbour.id, evidence=evidence)

    unconfirmed = [
        n for n in candidates if confirmations.for_neighbour(n.id) is None
    ]
    if unconfirmed:
        return NeedsConfirm(candidates=(unconfirmed[0],))

    new_value = _nearest_confirmed(
        candidates, confirmations, ConfirmAnswer.SAME_SUBJECT_NEW_VALUE
    )
    if new_value is not None:
        proof, neighbour = new_value
        return Supersede(
            loser_id=neighbour.id,
            proof=proof,
            evidence=evidence,
            winner_level=neighbour.level,
        )

    if _any_answer(candidates, confirmations, ConfirmAnswer.UNDECIDED):
        return Leave(reason="undecided")
    return Leave(reason="unrelated")


def decide_promotion(
    *,
    seed: PromotionSeed,
    established: Sequence[Neighbour],
    peers: Sequence[Neighbour],
    confirmations: Confirmations,
    peer_evidence: Mapping[str, EvidenceKey],
    peer_conversations: Mapping[str, Conversation],
) -> PromotionVerdict:
    """Pure promotion verdict. Peers with SAME_SUBJECT_NEW_VALUE never count."""

    # 1. Reinforce an existing established same-claim neighbour.
    for n in established:
        if classify_distance(n.distance) is Band.SAME_CLAIM:
            return Reinforce(target_id=n.id, evidence=seed.evidence)
        confirmed = confirmations.for_neighbour(n.id)
        if confirmed is not None and confirmed.answer is ConfirmAnswer.SAME_CLAIM:
            return Reinforce(target_id=n.id, evidence=seed.evidence)

    # 2. NeedsConfirm for nearest unconfirmed candidate-band neighbour.
    pool = [
        n
        for n in (*established, *peers)
        if classify_distance(n.distance) in (Band.SAME_CLAIM, Band.CANDIDATE)
    ]
    needs = [
        n
        for n in pool
        if classify_distance(n.distance) is Band.CANDIDATE
        and confirmations.for_neighbour(n.id) is None
    ]
    if needs:
        return NeedsConfirm(candidates=(needs[0],))

    # 3. Same-claim peers + seed → independent conversations ≥ min → Promote.
    same_claim_peers: list[Neighbour] = []
    proofs: list[Confirmed] = []
    for n in peers:
        band = classify_distance(n.distance)
        confirmed = confirmations.for_neighbour(n.id)
        if (
            confirmed is not None
            and confirmed.answer is ConfirmAnswer.SAME_SUBJECT_NEW_VALUE
        ):
            continue
        if band is Band.SAME_CLAIM:
            same_claim_peers.append(n)
            if confirmed is not None and confirmed.answer is ConfirmAnswer.SAME_CLAIM:
                proofs.append(confirmed)
            continue
        if (
            band is Band.CANDIDATE
            and confirmed is not None
            and confirmed.answer is ConfirmAnswer.SAME_CLAIM
        ):
            same_claim_peers.append(n)
            proofs.append(confirmed)

    if not same_claim_peers:
        if _any_answer(pool, confirmations, ConfirmAnswer.UNDECIDED):
            return Leave(reason="undecided")
        return Leave(reason="unrelated")

    conversations: list[Conversation] = [seed.conversation]
    evidence_keys: list[EvidenceKey] = [seed.evidence]
    source_ids: list[str] = [seed.id]
    for n in same_claim_peers:
        conv = peer_conversations.get(n.id)
        if conv is not None:
            conversations.append(conv)
        key = peer_evidence.get(n.id)
        if key is not None:
            evidence_keys.append(key)
        source_ids.append(n.id)

    if count_independent_conversations(conversations) < PROMOTION_MIN_CONVERSATIONS:
        return Leave(reason="unrelated")

    return Promote(
        level=_promotion_level(proofs),
        proofs=tuple(proofs),
        evidence=tuple(evidence_keys),
        source_ids=tuple(source_ids),
    )


def _nearest_confirmed(
    candidates: Sequence[Neighbour],
    confirmations: Confirmations,
    answer: ConfirmAnswer,
) -> tuple[Confirmed, Neighbour] | None:
    for n in candidates:
        confirmed = confirmations.for_neighbour(n.id)
        if confirmed is not None and confirmed.answer is answer:
            return confirmed, n
    return None


def _any_answer(
    candidates: Sequence[Neighbour],
    confirmations: Confirmations,
    answer: ConfirmAnswer,
) -> bool:
    for n in candidates:
        confirmed = confirmations.for_neighbour(n.id)
        if confirmed is not None and confirmed.answer is answer:
            return True
    return False


def _promotion_level(proofs: Sequence[Confirmed]) -> DocumentLevel:
    """Majority claim_kind → inductive (HABIT) or deductive (STATE); default inductive."""
    habit = 0
    state = 0
    for p in proofs:
        if p.claim_kind is ClaimKind.HABIT:
            habit += 1
        elif p.claim_kind is ClaimKind.STATE:
            state += 1
    if state > habit:
        return "deductive"
    return "inductive"
