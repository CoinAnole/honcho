"""Proof-token discipline and NeverConfirmer behaviour."""

import pytest

from src.crud.document import Neighbour
from src.memory.confirm import (
    ConfirmAnswer,
    Confirmations,
    Confirmed,
    NeverConfirmer,
    build_confirm_prompt,
    confirmer_from_settings,
    parse_confirm_response,
)
from src.memory.confirm import _ConfirmAnswerItem, _ConfirmResponse
from src.memory.evidence import EvidenceKey
from src.memory.verdict import Supersede


def _neighbour(nid: str = "n1", distance: float = 0.08) -> Neighbour:
    return Neighbour(
        id=nid, distance=distance, level="deductive", session_name=None
    )


class TestConfirmedProof:
    def test_confirmed_not_constructible_outside_module(self):
        with pytest.raises(TypeError, match="Confirmer"):
            Confirmed(
                object(),
                neighbour_id="n1",
                answer=ConfirmAnswer.SAME_SUBJECT_NEW_VALUE,
                claim_kind=None,
                distance=0.08,
            )

    def test_from_answers_mints_tokens(self):
        conf = Confirmations.from_answers(
            [("n1", ConfirmAnswer.SAME_CLAIM, None, 0.09)]
        )
        proof = conf.for_neighbour("n1")
        assert proof is not None
        assert proof.answer is ConfirmAnswer.SAME_CLAIM
        assert conf.for_neighbour("missing") is None

    def test_empty(self):
        assert Confirmations.empty().by_neighbour == {}


class TestNeverConfirmer:
    @pytest.mark.asyncio
    async def test_all_undecided(self):
        result = await NeverConfirmer().confirm(
            "new", [_neighbour("a", 0.07), _neighbour("b", 0.11)]
        )
        assert result.for_neighbour("a").answer is ConfirmAnswer.UNDECIDED  # type: ignore[union-attr]
        assert result.for_neighbour("b").answer is ConfirmAnswer.UNDECIDED  # type: ignore[union-attr]

    def test_factory_without_model_is_never(self):
        class _Est:
            CONFIRM_MODEL = None
            CONFIRM_TIMEOUT_SECONDS = 8.0

        class _Settings:
            ESTABLISHED = _Est()

        assert isinstance(confirmer_from_settings(_Settings()), NeverConfirmer)


class TestConfirmPromptParse:
    def test_prompt_prefers_under_grouping(self):
        prompt = build_confirm_prompt("new", [_neighbour()])
        assert "facility swap" in prompt.lower() or "under-grouping" in prompt.lower()
        assert "unrelated" in prompt.lower()

    def test_parse_maps_answers_and_fills_gaps(self):
        candidates = [_neighbour("a", 0.07), _neighbour("b", 0.11)]
        parsed = parse_confirm_response(
            _ConfirmResponse(
                answers=[
                    _ConfirmAnswerItem(
                        neighbour_index=0,
                        answer=ConfirmAnswer.SAME_CLAIM,
                    )
                ]
            ),
            candidates,
        )
        assert parsed.for_neighbour("a").answer is ConfirmAnswer.SAME_CLAIM  # type: ignore[union-attr]
        assert parsed.for_neighbour("b").answer is ConfirmAnswer.UNDECIDED  # type: ignore[union-attr]


class TestSupersedeProofGate:
    def test_rejects_wrong_proof_answer(self):
        proof = Confirmations.from_answers(
            [("n1", ConfirmAnswer.SAME_CLAIM, None, 0.08)]
        ).for_neighbour("n1")
        assert proof is not None
        with pytest.raises(ValueError, match="SAME_SUBJECT_NEW_VALUE"):
            Supersede(
                loser_id="n1",
                proof=proof,
                evidence=EvidenceKey("s:1-1"),
                winner_level="deductive",
            )

    def test_accepts_new_value_proof(self):
        proof = Confirmations.from_answers(
            [("n1", ConfirmAnswer.SAME_SUBJECT_NEW_VALUE, None, 0.08)]
        ).for_neighbour("n1")
        assert proof is not None
        v = Supersede(
            loser_id="n1",
            proof=proof,
            evidence=EvidenceKey("s:1-1"),
            winner_level="inductive",
        )
        assert v.loser_id == "n1"
        assert v.winner_level == "inductive"
