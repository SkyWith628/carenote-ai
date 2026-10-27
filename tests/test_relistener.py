"""Re-listener 텍스트 폴백 테스트. 가짜 클라이언트만 쓰며 API를 호출하지 않는다."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import anthropic
import httpx2
import pytest

from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import FieldKey, Flag, Source
from carenote.extraction.llm_extractor import ParseClient, ParseResponse
from carenote.extraction.pricing import BudgetExceeded, CostGuard, price_for
from carenote.extraction.relistener import (
    ANSWER_FORMAT,
    LLMReadback,
    TextRelistener,
    build_user_message,
    compare_readback,
)
from carenote.extraction.verifier import LLMVerdict, Verifier, VerifyingExtractor

TEXT = "방문했어요. 혈압은 130에 80이고요. 맥박은 칠십이에요. 약은 드셨어요. 넘어지신 적은 없어요."
BP = PredictedField(key=FieldKey.BLOOD_PRESSURE, value="130/80", source=Source.STATED)
PULSE = PredictedField(key=FieldKey.PULSE, value="72", source=Source.STATED)
MED = PredictedField(key=FieldKey.MEDICATION_TAKEN, value="복용", source=Source.STATED)


def rb(value: str, quote: str, conflict: bool = False) -> LLMReadback:
    return LLMReadback(value=value, quote=quote, conflict=conflict)


# -- 코드 비교 -------------------------------------------------------------------


def test_matching_numeric_readback_is_supported() -> None:
    v = compare_readback(TEXT, BP, rb("130/80", "혈압은 130에 80이고요"))
    assert v.status == "supported"


def test_hangul_numeral_readback_is_supported() -> None:
    assert compare_readback(TEXT, PULSE, rb("72", "맥박은 칠십이에요")).status == "supported"


def test_readback_digits_missing_from_quote_is_ambiguous() -> None:
    v = compare_readback(TEXT, BP, rb("130/90", "혈압은 130에 80이고요"))
    # 읽은 값 130/90의 90이 인용에서 읽히지 않으므로 신뢰하지 않고 ambiguous
    assert v.status == "ambiguous" and v.reason.startswith("R2")


def test_disagreeing_but_grounded_readback_is_unsupported() -> None:
    text = "혈압은 140에 90이고요. 다른 말."
    v = compare_readback(text, BP, rb("140/90", "혈압은 140에 90이고요"))
    assert v.status == "unsupported"


def test_conflict_or_empty_value_is_ambiguous() -> None:
    assert compare_readback(TEXT, BP, rb("", "", conflict=True)).status == "ambiguous"
    assert compare_readback(TEXT, BP, rb("", "")).status == "ambiguous"


def test_fabricated_quote_is_ambiguous_even_if_value_matches() -> None:
    v = compare_readback(TEXT, BP, rb("130/80", "없는 문장 130에 80"))
    assert v.status == "ambiguous" and v.reason.startswith("R1")


def test_malformed_numeric_value_is_ambiguous() -> None:
    v = compare_readback(TEXT, BP, rb("백삼십 팔십", "혈압은 130에 80이고요"))
    assert v.status == "ambiguous"


def test_categorical_match_and_mismatch() -> None:
    quote = "약은 드셨어요"
    assert compare_readback(TEXT, MED, rb("복용", quote)).status == "supported"
    assert compare_readback(TEXT, MED, rb("미복용", quote)).status == "unsupported"


# -- 호출 ------------------------------------------------------------------------


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 500
    output_tokens: int = 40
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class Client:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []
        self.messages = self

    def parse(self, **kw: object) -> ParseResponse:
        self.calls.append(kw)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return cast(ParseResponse, self.outcome)


def make(outcome: object, tmp: Path | None = None, **kw: object) -> tuple[TextRelistener, Client]:
    client = Client(outcome)
    return (
        TextRelistener(
            cast(ParseClient, client),
            model="claude-haiku-5-5",
            effort="low",
            cache_dir=tmp,
            **kw,  # type: ignore[arg-type]
        ),
        client,
    )


def ok(value: str = "130/80", quote: str = "혈압은 130에 80이고요") -> _Response:
    return _Response(rb(value, quote))


def test_prompt_never_contains_the_extracted_value() -> None:
    message = build_user_message(TEXT, FieldKey.BLOOD_PRESSURE)
    assert "<item>" in message and ANSWER_FORMAT[FieldKey.BLOOD_PRESSURE] in message
    relistener, client = make(ok())
    relistener.verify(TEXT, BP.model_copy(update={"value": "999/99"}))
    sent = cast(list[dict[str, str]], client.calls[0]["messages"])[0]["content"]
    assert "999/99" not in sent  # 읽기는 주장을 모른 채 이뤄진다(확인 편향 방지)


def test_cache_is_independent_of_claimed_value(tmp_path: Path) -> None:
    relistener, client = make(ok(), tmp_path)
    relistener.verify(TEXT, BP)
    other = relistener.verify(TEXT, BP.model_copy(update={"value": "140/90"}))
    assert len(client.calls) == 1 and relistener.stats.cache_hits == 1
    assert other.status in ("unsupported", "ambiguous")  # 같은 읽기를 다른 주장과 비교한다


def test_api_error_and_unusable_responses_are_unavailable() -> None:
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))
    assert make(err)[0].verify(TEXT, BP).status == "unavailable"
    assert make(_Response(None))[0].verify(TEXT, BP).status == "unavailable"
    assert make(_Response(rb("1", "x"), stop_reason="max_tokens"))[0].verify(TEXT, BP).status == (
        "unavailable"
    )


def test_budget_guard_stops_before_call() -> None:
    price = price_for("claude-haiku-5-5")
    assert price is not None
    relistener, client = make(ok(), guard=CostGuard(0.0, price))
    with pytest.raises(BudgetExceeded):
        relistener.verify(TEXT, BP)
    assert client.calls == []


# -- Verifier와 함께 쓰기 ----------------------------------------------------------


class Inner:
    def extract(self, transcript: str) -> list[PredictedField]:
        return [BP]


def test_relistener_plugs_into_verifying_extractor_and_stacks_with_verifier() -> None:
    relistener, _ = make(ok("140/90", "혈압은 130에 80이고요"))  # 값 140/90은 인용에서 안 읽힘
    verifier_client = Client(
        _Response(LLMVerdict(verdict="supported", quote="혈압은 130에 80이고요"))
    )
    verifier = Verifier(cast(ParseClient, verifier_client), model="claude-haiku-5-5", effort="low")
    stack = VerifyingExtractor(VerifyingExtractor(Inner(), verifier), relistener, name="relistener")
    out = stack.extract(TEXT)
    assert out[0].flag is Flag.NEEDS_CONFIRM  # Verifier는 supported, Re-listener가 올린다
    usage = stack.usage_summary()
    assert "relistener" in usage and "verifier" in usage
    assert stack.verdicts_for(TEXT)[FieldKey.BLOOD_PRESSURE].status == "ambiguous"
