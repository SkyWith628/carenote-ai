"""Verifier 테스트. 가짜 클라이언트만 쓰며 API를 호출하지 않는다."""

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
from carenote.extraction.verifier import (
    FieldVerdict,
    LLMVerdict,
    Verifier,
    VerifyingExtractor,
    apply_verdicts,
    build_user_message,
    check_verdict,
    needs_verification,
)

TEXT = "방문했어요. 혈압은 130에 80이고요. 점심은 드셨어요. 약은 드셨어요."
BP = PredictedField(key=FieldKey.BLOOD_PRESSURE, value="130/80", source=Source.STATED)
QUOTE = "혈압은 130에 80이고요"


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 1000
    output_tokens: int = 50
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class _FakeClient:
    """outcome이 예외면 던지고, 아니면 응답으로 돌려준다. 호출 인자를 기록한다."""

    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []
        self.messages = self

    def parse(self, **kw: object) -> ParseResponse:
        self.calls.append(kw)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return cast(ParseResponse, self.outcome)


def answer(verdict: str, quote: str = QUOTE, **kw: object) -> _Response:
    return _Response(LLMVerdict.model_validate({"verdict": verdict, "quote": quote}), **kw)  # type: ignore[arg-type]


def make(outcome: object, tmp: Path | None = None, **kw: object) -> tuple[Verifier, _FakeClient]:
    client = _FakeClient(outcome)
    verifier = Verifier(
        cast(ParseClient, client),
        model="claude-haiku-5-5",
        effort="low",
        cache_dir=tmp,
        **kw,  # type: ignore[arg-type]
    )
    return verifier, client


def api_error() -> anthropic.APIError:
    return anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))


# -- 코드 검사 -------------------------------------------------------------------


def test_supported_with_real_quote_and_matching_numbers_is_kept() -> None:
    verdict, downgraded = check_verdict(TEXT, BP, LLMVerdict(verdict="supported", quote=QUOTE))
    assert verdict.status == "supported" and not downgraded


def test_supported_with_fabricated_quote_is_downgraded() -> None:
    out = LLMVerdict(verdict="supported", quote="혈압은 130에 90이고요")
    verdict, downgraded = check_verdict(TEXT, BP, out)
    assert verdict.status == "ambiguous" and downgraded
    assert verdict.reason.startswith("V1")


def test_supported_with_empty_quote_is_downgraded() -> None:
    verdict, downgraded = check_verdict(TEXT, BP, LLMVerdict(verdict="supported", quote=""))
    assert verdict.status == "ambiguous" and downgraded


def test_supported_numeric_value_not_in_quote_is_downgraded() -> None:
    wrong = BP.model_copy(update={"value": "140/90"})
    verdict, downgraded = check_verdict(TEXT, wrong, LLMVerdict(verdict="supported", quote=QUOTE))
    assert verdict.status == "ambiguous" and downgraded
    assert verdict.reason.startswith("V2")


def test_non_numeric_supported_only_needs_real_quote() -> None:
    med = PredictedField(key=FieldKey.MEDICATION_TAKEN, value="복용", source=Source.STATED)
    verdict, downgraded = check_verdict(
        TEXT, med, LLMVerdict(verdict="supported", quote="약은 드셨어요")
    )
    assert verdict.status == "supported" and not downgraded


@pytest.mark.parametrize("status", ["unsupported", "ambiguous"])
def test_unsupported_and_ambiguous_pass_through(status: str) -> None:
    verdict, downgraded = check_verdict(
        TEXT, BP, LLMVerdict.model_validate({"verdict": status, "quote": ""})
    )
    assert verdict.status == status and not downgraded


# -- 대상 선택 -------------------------------------------------------------------


def test_needs_verification_only_for_valued_risk_fields() -> None:
    assert needs_verification(BP)
    assert not needs_verification(
        PredictedField(key=FieldKey.MOOD, value="좋음", source=Source.STATED)
    )
    assert not needs_verification(PredictedField(key=FieldKey.BLOOD_PRESSURE))  # missing
    ambiguous = PredictedField(
        key=FieldKey.BLOOD_PRESSURE, candidates=("130/80", "130/90"), source=Source.STATED
    )
    assert not needs_verification(ambiguous)


# -- Consensus -------------------------------------------------------------------


def verdict(status: str) -> FieldVerdict:
    return FieldVerdict(FieldKey.BLOOD_PRESSURE, status)  # type: ignore[arg-type]


@pytest.mark.parametrize("status", ["unsupported", "ambiguous"])
def test_ok_is_raised_to_needs_confirm(status: str) -> None:
    out = apply_verdicts([BP], {FieldKey.BLOOD_PRESSURE: verdict(status)})
    assert out[0].flag is Flag.NEEDS_CONFIRM


@pytest.mark.parametrize("status", ["supported", "unavailable"])
def test_supported_and_unavailable_do_not_change_flag(status: str) -> None:
    out = apply_verdicts([BP], {FieldKey.BLOOD_PRESSURE: verdict(status)})
    assert out[0].flag is Flag.OK


@pytest.mark.parametrize("flag", [Flag.RISK, Flag.NEEDS_CONFIRM])
def test_flag_is_never_lowered_or_replaced(flag: Flag) -> None:
    field_ = BP.model_copy(update={"flag": flag})
    out = apply_verdicts([field_], {FieldKey.BLOOD_PRESSURE: verdict("unsupported")})
    assert out[0].flag is flag


# -- 호출 ------------------------------------------------------------------------


def test_verify_calls_with_claim_only_not_extractor_evidence() -> None:
    verifier, client = make(answer("supported"))
    with_evidence = BP.model_copy(update={"evidence": ()})
    assert verifier.verify(TEXT, with_evidence).status == "supported"
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    content = cast(list[dict[str, str]], call["messages"])[0]["content"]
    assert content == build_user_message(TEXT, FieldKey.BLOOD_PRESSURE, "130/80")
    assert "<claim>" in content and "130/80" in content
    assert verifier.stats.calls == 1 and verifier.stats.input_tokens == 1000


def test_api_error_is_unavailable_not_supported() -> None:
    verifier, _ = make(api_error())
    result = verifier.verify(TEXT, BP)
    assert result.status == "unavailable" and verifier.stats.failures == 1


@pytest.mark.parametrize("reason", ["refusal", "max_tokens"])
def test_unusable_stop_reason_is_unavailable(reason: str) -> None:
    verifier, _ = make(answer("supported", stop_reason=reason))
    assert verifier.verify(TEXT, BP).status == "unavailable"


def test_unparsed_output_is_unavailable() -> None:
    verifier, _ = make(_Response(None))
    assert verifier.verify(TEXT, BP).status == "unavailable"


def test_cache_avoids_second_call(tmp_path: Path) -> None:
    verifier, client = make(answer("supported"), tmp_path)
    verifier.verify(TEXT, BP)
    assert verifier.is_cached(TEXT, BP)
    verifier.verify(TEXT, BP)
    assert len(client.calls) == 1 and verifier.stats.cache_hits == 1


def test_cache_key_depends_on_claimed_value(tmp_path: Path) -> None:
    verifier, client = make(answer("supported"), tmp_path)
    verifier.verify(TEXT, BP)
    verifier.verify(TEXT, BP.model_copy(update={"value": "140/90"}))
    assert len(client.calls) == 2


def test_cache_stores_raw_output_so_check_changes_need_no_recall(tmp_path: Path) -> None:
    verifier, client = make(answer("supported", quote="없는 문장"), tmp_path)
    first = verifier.verify(TEXT, BP)
    second = verifier.verify(TEXT, BP)
    assert first.status == second.status == "ambiguous"
    assert len(client.calls) == 1


def test_budget_guard_stops_before_call() -> None:
    price = price_for("claude-haiku-5-5")
    assert price is not None
    guard = CostGuard(0.0, price)
    verifier, client = make(answer("supported"), guard=guard)
    with pytest.raises(BudgetExceeded):
        verifier.verify(TEXT, BP)
    assert client.calls == []


def test_usage_summary_has_cost() -> None:
    verifier, _ = make(answer("supported"))
    verifier.verify(TEXT, BP)
    summary = verifier.usage_summary()
    assert summary["calls"] == 1
    assert isinstance(summary["estimated_cost_usd"], float)


# -- VerifyingExtractor ------------------------------------------------------------


class _Inner:
    def __init__(self, fields: list[PredictedField]) -> None:
        self._fields = fields

    def extract(self, transcript: str) -> list[PredictedField]:
        return list(self._fields)

    def usage_summary(self) -> dict[str, object]:
        return {"calls": 1}


def test_verifying_extractor_checks_only_risk_fields_and_raises_flag() -> None:
    mood = PredictedField(key=FieldKey.MOOD, value="좋음", source=Source.STATED)
    verifier, client = make(answer("unsupported", quote=""))
    ext = VerifyingExtractor(_Inner([BP, mood]), verifier)
    out = ext.extract(TEXT)
    assert len(client.calls) == 1  # mood는 검증하지 않는다
    assert {f.key: f.flag for f in out} == {
        FieldKey.BLOOD_PRESSURE: Flag.NEEDS_CONFIRM,
        FieldKey.MOOD: Flag.OK,
    }
    assert ext.verdicts_for(TEXT)[FieldKey.BLOOD_PRESSURE].status == "unsupported"
    assert ext.usage_summary()["calls"] == 1
    assert "verifier" in ext.usage_summary()


def test_verifying_extractor_failure_keeps_flag_and_records_unavailable() -> None:
    verifier, _ = make(api_error())
    ext = VerifyingExtractor(_Inner([BP]), verifier)
    out = ext.extract(TEXT)
    assert out[0].flag is Flag.OK
    assert ext.verdicts_for(TEXT)[FieldKey.BLOOD_PRESSURE].status == "unavailable"
