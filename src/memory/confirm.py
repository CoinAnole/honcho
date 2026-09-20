"""Confirmer protocol and proof tokens for established-memory verdicts."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final, Protocol, runtime_checkable, final

from pydantic import BaseModel, Field

from src.config import ConfiguredModelSettings, ModelConfig

logger = logging.getLogger(__name__)

_MINT: Final[object] = object()


class ConfirmAnswer(Enum):
    SAME_CLAIM = "same_claim"
    SAME_SUBJECT_NEW_VALUE = "same_subject_new_value"
    UNRELATED = "unrelated"
    UNDECIDED = "undecided"


class ClaimKind(Enum):
    HABIT = "habit"
    STATE = "state"


@runtime_checkable
class ConfirmNeighbour(Protocol):
    """Minimal neighbour shape for confirmation (avoids crud ↔ memory import cycle)."""

    id: str
    distance: float
    level: str


@final
class Confirmed:
    """Proof token. Mintable only inside this module via ``_MINT``."""

    __slots__ = ("neighbour_id", "answer", "claim_kind", "distance")

    def __init__(
        self,
        _token: object,
        *,
        neighbour_id: str,
        answer: ConfirmAnswer,
        claim_kind: ClaimKind | None,
        distance: float,
    ) -> None:
        if _token is not _MINT:
            raise TypeError("Confirmed can only be issued by a Confirmer")
        self.neighbour_id = neighbour_id
        self.answer = answer
        self.claim_kind = claim_kind
        self.distance = distance

    def __repr__(self) -> str:
        return (
            f"Confirmed(neighbour_id={self.neighbour_id!r}, answer={self.answer!r}, "
            f"claim_kind={self.claim_kind!r}, distance={self.distance!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Confirmed):
            return NotImplemented
        return (
            self.neighbour_id == other.neighbour_id
            and self.answer == other.answer
            and self.claim_kind == other.claim_kind
            and self.distance == other.distance
        )


def _mint(
    *,
    neighbour_id: str,
    answer: ConfirmAnswer,
    claim_kind: ClaimKind | None,
    distance: float,
) -> Confirmed:
    return Confirmed(
        _MINT,
        neighbour_id=neighbour_id,
        answer=answer,
        claim_kind=claim_kind,
        distance=distance,
    )


@dataclass(frozen=True, slots=True)
class Confirmations:
    """Answers keyed by neighbour id. Missing key means no confirmation yet."""

    by_neighbour: Mapping[str, Confirmed]

    @staticmethod
    def empty() -> Confirmations:
        return Confirmations(by_neighbour={})

    def for_neighbour(self, neighbour_id: str) -> Confirmed | None:
        return self.by_neighbour.get(neighbour_id)

    @staticmethod
    def from_answers(
        items: Sequence[tuple[str, ConfirmAnswer, ClaimKind | None, float]],
    ) -> Confirmations:
        """Factory used by confirmers to mint proof tokens."""
        return Confirmations(
            by_neighbour={
                neighbour_id: _mint(
                    neighbour_id=neighbour_id,
                    answer=answer,
                    claim_kind=claim_kind,
                    distance=distance,
                )
                for neighbour_id, answer, claim_kind, distance in items
            }
        )


class Confirmer(Protocol):
    async def confirm(
        self, new_content: str, candidates: Sequence[ConfirmNeighbour]
    ) -> Confirmations:
        """Confirm new content against candidate neighbours.

        Must not raise: failures map to UNDECIDED for every candidate.
        Must not be invoked with a DB session open.
        """
        ...


class NeverConfirmer:
    """Returns UNDECIDED for every candidate. Safe default / shadow fallback."""

    async def confirm(
        self, new_content: str, candidates: Sequence[ConfirmNeighbour]
    ) -> Confirmations:
        del new_content
        return Confirmations.from_answers(
            [
                (n.id, ConfirmAnswer.UNDECIDED, None, n.distance)
                for n in candidates
            ]
        )


class _ConfirmAnswerItem(BaseModel):
    neighbour_index: int = Field(ge=0)
    answer: ConfirmAnswer
    claim_kind: ClaimKind | None = None


class _ConfirmResponse(BaseModel):
    answers: list[_ConfirmAnswerItem] = Field(default_factory=list)


_CONFIRM_SYSTEM = """\
You classify whether a new observation relates to each candidate memory.

For each candidate choose exactly one answer:
- same_claim: paraphrase of the same claim (reinforce)
- same_subject_new_value: same subject, updated value (supersede)
- unrelated: different subject or different facility at the same station

Prefer under-grouping. When unsure, choose unrelated.
A facility swap at the same station is unrelated, not a new value.
Optionally set claim_kind to habit or state when the answer is same_claim
or same_subject_new_value.
"""


def build_confirm_prompt(
    new_content: str, candidates: Sequence[ConfirmNeighbour]
) -> str:
    """Build the user prompt for LLM confirmation (testable without network)."""
    lines = [
        _CONFIRM_SYSTEM,
        "",
        f"New observation:\n{new_content}",
        "",
        "Candidates (nearest first):",
    ]
    for i, n in enumerate(candidates):
        content = getattr(n, "content", None)
        if isinstance(content, str) and content:
            body = content
        else:
            body = f"(id={n.id}, level={n.level})"
        lines.append(f"[{i}] distance={n.distance:.4f}\n{body}")
    lines.append("")
    lines.append(
        "Respond with answers listing neighbour_index, answer, and optional claim_kind."
    )
    return "\n".join(lines)


def parse_confirm_response(
    response: _ConfirmResponse,
    candidates: Sequence[ConfirmNeighbour],
) -> Confirmations:
    """Map a parsed model response onto candidates; gaps become UNDECIDED."""
    by_index: dict[int, _ConfirmAnswerItem] = {}
    for item in response.answers:
        if 0 <= item.neighbour_index < len(candidates):
            by_index[item.neighbour_index] = item

    minted: list[tuple[str, ConfirmAnswer, ClaimKind | None, float]] = []
    for i, n in enumerate(candidates):
        item = by_index.get(i)
        if item is None:
            minted.append((n.id, ConfirmAnswer.UNDECIDED, None, n.distance))
        else:
            minted.append((n.id, item.answer, item.claim_kind, n.distance))
    return Confirmations.from_answers(minted)


class LLMConfirmer:
    """LLM-backed confirmer. Any failure or timeout yields UNDECIDED for all."""

    def __init__(
        self,
        model_config: ModelConfig | ConfiguredModelSettings,
        *,
        timeout_s: float,
    ) -> None:
        self._model_config = model_config
        self._timeout_s = timeout_s

    async def confirm(
        self, new_content: str, candidates: Sequence[ConfirmNeighbour]
    ) -> Confirmations:
        if not candidates:
            return Confirmations.empty()
        try:
            return await asyncio.wait_for(
                self._confirm_inner(new_content, candidates),
                timeout=self._timeout_s,
            )
        except Exception:
            logger.exception("LLMConfirmer failed; mapping all candidates to UNDECIDED")
            return Confirmations.from_answers(
                [
                    (n.id, ConfirmAnswer.UNDECIDED, None, n.distance)
                    for n in candidates
                ]
            )

    async def _confirm_inner(
        self, new_content: str, candidates: Sequence[ConfirmNeighbour]
    ) -> Confirmations:
        from src.llm import honcho_llm_call

        prompt = build_confirm_prompt(new_content, candidates)
        max_tokens = 1024
        if isinstance(self._model_config, ConfiguredModelSettings):
            configured_max = self._model_config.max_output_tokens
            if configured_max and configured_max > 0:
                max_tokens = configured_max
        elif self._model_config.max_output_tokens:
            max_tokens = self._model_config.max_output_tokens

        response = await honcho_llm_call(
            model_config=self._model_config,
            prompt=prompt,
            max_tokens=max_tokens,
            response_model=_ConfirmResponse,
            json_mode=True,
            enable_retry=False,
            trace_name="established_confirm",
        )
        return parse_confirm_response(response.content, candidates)


def confirmer_from_settings(settings: object) -> Confirmer:
    """Build a confirmer from ``settings.ESTABLISHED`` (or an EstablishedSettings)."""
    established = getattr(settings, "ESTABLISHED", settings)
    model = getattr(established, "CONFIRM_MODEL", None)
    if model is None:
        return NeverConfirmer()
    timeout_s = float(getattr(established, "CONFIRM_TIMEOUT_SECONDS", 8.0))
    return LLMConfirmer(model, timeout_s=timeout_s)
