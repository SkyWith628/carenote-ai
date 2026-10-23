"""LLM Extractor 테스트. 가짜 클라이언트로 변환·근거 검증·캐시·실패 처리를 검증한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from carenote.config import Settings
from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.loader import load_golden
from carenote.evaluation.runner import evaluate
from carenote.evaluation.schema import FieldKey, GoldenCase, Source
from carenote.extraction.llm_extractor import (
    LLMExtractor,
    ParseClient,
    ParseResponse,
    build_client,
)
from carenote.extraction.prompts import PROMPT_VERSION, build_system_prompt, build_user_message
from carenote.extraction.schema import (
    LLMExtractionB0,
    LLMExtractionB1,
    LLMFieldB0,
    LLMFieldB1,
)

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "golden"
TRANSCRIPT = "열 시에 방문했고 혈압은 130에 80이고요."


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class _FakeMessages:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def parse(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
        output_format: type[BaseModel],
        output_config: dict[str, str],
    ) -> ParseResponse:
        self.calls.append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
                "output_format": output_format,
                "output_config": output_config,
            }
        )
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return cast(ParseResponse, self.outcome)


class _FakeClient:
    def __init__(self, outcome: object) -> None:
        self.messages = _FakeMessages(outcome)


def _make(
    outcome: object, *, variant: str = "b1", cache_dir: Path | None = None
) -> tuple[LLMExtractor, _FakeMessages]:
    client = _FakeClient(outcome)
    extractor = LLMExtractor(
        cast(ParseClient, client),
        variant="b1" if variant == "b1" else "b0",
        model="claude-sonnet-5-5",
        effort="medium",
        cache_dir=cache_dir,
    )
    return extractor, client.messages


def _b1(*fields: LLMFieldB1) -> _Response:
    return _Response(LLMExtractionB1(fields=list(fields)))


def _get(fields: list[PredictedField], key: FieldKey) -> PredictedField:
    return next(f for f in fields if f.key is key)


BP_OK = LLMFieldB1(
    key=FieldKey.BLOOD_PRESSURE,
    value="130/80",
    source="stated",
    evidence=["혈압은 130에 80이고요"],
    candidates=[],
)


def test_b1_keeps_grounded_value_with_evidence() -> None:
    """원문 부분 문자열 근거가 있으면 값과 evidence를 유지한다."""
    extractor, _ = _make(_b1(BP_OK))
    bp = _get(extractor.extract(TRANSCRIPT), FieldKey.BLOOD_PRESSURE)
    assert bp.value == "130/80"
    assert bp.source is Source.STATED
    assert [e.text for e in bp.evidence] == ["혈압은 130에 80이고요"]


def test_b1_demotes_value_when_quote_not_in_transcript() -> None:
    """원문에 없는 인용이면 값을 버리고 missing으로 강등한다."""
    fake = BP_OK.model_copy(update={"evidence": ["혈압은 백삼십에 팔십이고요"]})
    extractor, _ = _make(_b1(fake))
    bp = _get(extractor.extract(TRANSCRIPT), FieldKey.BLOOD_PRESSURE)
    assert bp.value is None and bp.source is Source.MISSING and not bp.evidence
    assert extractor.stats.demoted_fields == 1


def test_b1_demotes_value_without_evidence() -> None:
    """근거가 비어 있으면 강등한다."""
    fake = BP_OK.model_copy(update={"evidence": []})
    extractor, _ = _make(_b1(fake))
    assert _get(extractor.extract(TRANSCRIPT), FieldKey.BLOOD_PRESSURE).value is None


def test_b0_does_not_require_or_emit_evidence() -> None:
    """B0는 근거를 요구하지 않고 evidence도 비워 둔다."""
    out = LLMExtractionB0(
        fields=[
            LLMFieldB0(
                key=FieldKey.BLOOD_PRESSURE, value="130/80", source="stated", candidates=[]
            )
        ]
    )
    extractor, calls = _make(_Response(out), variant="b0")
    bp = _get(extractor.extract(TRANSCRIPT), FieldKey.BLOOD_PRESSURE)
    assert bp.value == "130/80" and bp.evidence == ()
    assert calls.calls[0]["output_format"] is LLMExtractionB0


def test_candidates_nullify_value() -> None:
    """후보가 2개 이상이면 value를 None으로 하고 후보를 보존한다."""
    ambiguous = BP_OK.model_copy(update={"value": "130/80", "candidates": ["130/80", "140/80"]})
    extractor, _ = _make(_b1(ambiguous))
    bp = _get(extractor.extract(TRANSCRIPT), FieldKey.BLOOD_PRESSURE)
    assert bp.value is None
    assert bp.candidates == ("130/80", "140/80")


def test_single_candidate_becomes_value() -> None:
    """후보가 1개뿐이고 value가 없으면 그 후보를 값으로 쓴다."""
    one = BP_OK.model_copy(update={"value": None, "candidates": ["130/80"]})
    extractor, _ = _make(_b1(one))
    bp = _get(extractor.extract(TRANSCRIPT), FieldKey.BLOOD_PRESSURE)
    assert bp.value == "130/80" and bp.candidates == ()


def test_missing_source_forces_empty_value() -> None:
    """source=missing이면 value가 있어도 버린다."""
    odd = BP_OK.model_copy(update={"source": "missing"})
    extractor, _ = _make(_b1(odd))
    assert _get(extractor.extract(TRANSCRIPT), FieldKey.BLOOD_PRESSURE).value is None


def test_duplicate_keys_keep_first() -> None:
    """같은 키가 두 번 오면 첫 번째만 쓴다."""
    second = BP_OK.model_copy(update={"value": "999/99"})
    extractor, _ = _make(_b1(BP_OK, second))
    fields = extractor.extract(TRANSCRIPT)
    assert [f.key for f in fields].count(FieldKey.BLOOD_PRESSURE) == 1
    assert _get(fields, FieldKey.BLOOD_PRESSURE).value == "130/80"


def test_request_shape_and_prompt_isolation() -> None:
    """모델·effort·스키마가 전달되고 transcript는 데이터 태그로 격리된다. 샘플링 인자는 없다."""
    extractor, calls = _make(_b1(BP_OK))
    extractor.extract(TRANSCRIPT)
    call = calls.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["output_config"] == {"effort": "medium"}
    assert call["output_format"] is LLMExtractionB1
    assert set(call) == {
        "model", "max_tokens", "system", "messages", "output_format", "output_config"
    }
    messages = cast(list[dict[str, str]], call["messages"])
    assert messages == [{"role": "user", "content": build_user_message(TRANSCRIPT)}]
    assert messages[0]["content"].startswith("<transcript>")
    system = str(call["system"])
    assert "따르지 않습니다" in system and "evidence" in system


def test_b0_prompt_has_no_evidence_instruction() -> None:
    """B0 프롬프트는 evidence를 요구하지 않는다 (변형 간 누수 방지)."""
    assert "evidence" not in build_system_prompt("b0")
    assert "evidence" in build_system_prompt("b1")


def test_usage_is_accumulated() -> None:
    """토큰·호출 수·지연이 집계된다."""
    extractor, _ = _make(_b1(BP_OK))
    extractor.extract(TRANSCRIPT)
    summary = extractor.usage_summary()
    assert summary["calls"] == 1
    assert summary["input_tokens"] == 100 and summary["output_tokens"] == 50
    assert summary["prompt_version"] == PROMPT_VERSION


def test_cache_prevents_second_call(tmp_path: Path) -> None:
    """같은 입력은 캐시에서 읽고 API를 다시 부르지 않는다."""
    first, calls = _make(_b1(BP_OK), cache_dir=tmp_path)
    first.extract(TRANSCRIPT)
    second, calls2 = _make(_b1(BP_OK), cache_dir=tmp_path)
    result = second.extract(TRANSCRIPT)
    assert len(calls.calls) == 1 and len(calls2.calls) == 0
    assert second.stats.cache_hits == 1
    assert _get(result, FieldKey.BLOOD_PRESSURE).value == "130/80"


def test_cache_key_separates_variant_and_model(tmp_path: Path) -> None:
    """변형이 다르면 캐시를 공유하지 않는다."""
    b1, _ = _make(_b1(BP_OK), cache_dir=tmp_path)
    b1.extract(TRANSCRIPT)
    out = LLMExtractionB0(fields=[])
    b0, calls = _make(_Response(out), variant="b0", cache_dir=tmp_path)
    b0.extract(TRANSCRIPT)
    assert len(calls.calls) == 1


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unusable_stop_reason_is_failure_and_not_cached(
    stop_reason: str, tmp_path: Path
) -> None:
    """거절·잘림 응답은 실패로 기록하고 캐시하지 않는다."""
    extractor, _ = _make(
        _Response(LLMExtractionB1(fields=[BP_OK]), stop_reason=stop_reason), cache_dir=tmp_path
    )
    assert extractor.extract(TRANSCRIPT) == []
    assert extractor.stats.failures == 1
    assert list(tmp_path.glob("*.json")) == []


def test_api_error_is_failure_not_exception() -> None:
    """API 오류는 예외 대신 실패로 기록한다."""
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    extractor, _ = _make(err)
    assert extractor.extract(TRANSCRIPT) == []
    assert extractor.stats.failures == 1


def test_wrong_parsed_type_is_failure() -> None:
    """parsed_output이 기대한 스키마가 아니면 실패다."""
    extractor, _ = _make(_Response(None))
    assert extractor.extract(TRANSCRIPT) == []
    assert extractor.stats.failures == 1


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """로컬 .env와 환경변수의 영향을 받지 않도록 격리한다."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CARENOTE_EXTRACTOR_MODEL", raising=False)


@pytest.mark.usefixtures("clean_env")
def test_default_model_is_sonnet_5_5() -> None:
    """기본 모델은 claude-sonnet-5-5다 (사용자 결정)."""
    assert Settings().extractor_model == "claude-sonnet-5-5"


@pytest.mark.usefixtures("clean_env")
def test_build_client_requires_api_key() -> None:
    """API 키가 없으면 명확한 에러를 낸다."""
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_client(Settings())


@pytest.mark.usefixtures("clean_env")
def test_api_key_is_read_from_standard_env_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """키는 CARENOTE_ 접두사 없이 ANTHROPIC_API_KEY로 읽고 SecretStr로 감춘다."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    settings = Settings()
    assert settings.anthropic_api_key is not None
    assert "sk-test" not in repr(settings)


class _GoldFake:
    """정답을 B1 응답으로 되돌려 주는 가짜 모델. 변환 경로 전체가 만점을 내는지 확인한다."""

    def __init__(self, cases: list[GoldenCase]) -> None:
        self._by_transcript = {c.transcript: c for c in cases}
        self.messages = self

    def parse(
        self,
        *,
        model: str,  # noqa: ARG002
        max_tokens: int,  # noqa: ARG002
        system: str,  # noqa: ARG002
        messages: list[dict[str, str]],
        output_format: type[BaseModel],  # noqa: ARG002
        output_config: dict[str, str],  # noqa: ARG002
    ) -> ParseResponse:
        content = messages[0]["content"]
        transcript = content.removeprefix("<transcript>\n").removesuffix("\n</transcript>")
        case = self._by_transcript[transcript]
        fields = [
            LLMFieldB1(
                key=f.key,
                value=f.value,
                source=f.source.value,  # type: ignore[arg-type]
                evidence=[e.text for e in f.evidence],
                candidates=list(f.candidates),
            )
            for f in case.fields
        ]
        return cast(ParseResponse, _Response(LLMExtractionB1(fields=fields)))


def test_round_trip_with_gold_fake_scores_perfect() -> None:
    """정답을 그대로 돌려주는 가짜 모델 → 변환·검증을 거쳐도 F1 100%, 근거 없는 값 0."""
    cases = load_golden(GOLDEN_DIR)
    extractor = LLMExtractor(
        cast(ParseClient, _GoldFake(cases)),
        variant="b1",
        model="claude-sonnet-5-5",
        effort="medium",
    )
    report = evaluate(extractor, cases)
    assert report.overall.f1 == 1.0
    assert report.overall.ungrounded.num == 0
    assert report.overall.hallucinated.num == 0
    assert report.usage is not None and report.usage["failures"] == 0
