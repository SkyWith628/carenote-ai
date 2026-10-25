"""생성기와 생성 CLI 테스트. 가짜 클라이언트만 쓰며 실제 API는 호출하지 않는다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import anthropic
import httpx2
import pytest
from conftest import GOOD_CASES, GOOD_D1, SPEC_DIR
from pydantic import BaseModel

from carenote.config import Settings
from carenote.extraction.llm_extractor import ParseClient, ParseResponse
from carenote.extraction.pricing import BudgetExceeded, CostGuard, Price
from carenote.generation.__main__ import main
from carenote.generation.candidates import (
    apply_batch_checks,
    pick_labeling_sample,
    render_review_sheet,
)
from carenote.generation.estimate import estimate_generation
from carenote.generation.generator import Generator
from carenote.generation.prompts import (
    GENERATION_PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_message,
)
from carenote.generation.schema import GeneratedCase
from carenote.generation.spec import Spec

GOLDEN_DIR = SPEC_DIR.parent


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 3000
    output_tokens: int = 1500
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class _FakeMessages:
    """scene으로 스펙을 찾아 미리 정한 후보를 돌려준다. 호출 내용을 기록한다."""

    def __init__(
        self,
        by_scene: dict[str, list[GeneratedCase]],
        usage: _Usage | None = None,
        error: Exception | None = None,
    ) -> None:
        self._by_scene = by_scene
        self._usage = usage or _Usage()
        self._error = error
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
        content = messages[0]["content"]
        self.calls.append(
            {
                "model": model,
                "system": system,
                "content": content,
                "output_format": output_format,
                "output_config": output_config,
                "max_tokens": max_tokens,
            }
        )
        if self._error is not None:
            raise self._error
        payload = json.loads(content.split("\n\n이전 시도")[0])
        queue = self._by_scene[payload["scene"]]
        case = queue.pop(0) if len(queue) > 1 else queue[0]
        return cast(ParseResponse, _Response(case, usage=self._usage))


class _FakeClient:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages


def _client_for(
    spec_by_id: dict[str, Spec], plan: dict[str, list[GeneratedCase]], **kw: object
) -> tuple[ParseClient, _FakeMessages]:
    by_scene = {spec_by_id[sid].scene: cases for sid, cases in plan.items()}
    messages = _FakeMessages(by_scene, **kw)  # type: ignore[arg-type]
    return cast(ParseClient, _FakeClient(messages)), messages


def _generator(client: ParseClient, **kw: object) -> Generator:
    return Generator(client, model="claude-opus-5-5", effort="medium", **kw)  # type: ignore[arg-type]


def _good_plan() -> dict[str, list[GeneratedCase]]:
    return {sid: [case] for sid, case in GOOD_CASES.items()}


def test_generate_passes_on_first_attempt(spec_by_id: dict[str, Spec]) -> None:
    """점검을 통과하면 1회 시도로 끝난다. 샘플링 인자 없이 모델·effort만 전달한다."""
    client, messages = _client_for(spec_by_id, _good_plan())
    result = _generator(client).generate(spec_by_id["D1"])
    assert result.status == "passed" and result.attempts == 1
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "medium"}
    assert call["output_format"] is GeneratedCase
    assert call["system"] == SYSTEM_PROMPT


def test_retry_adds_failure_feedback_and_then_passes(spec_by_id: dict[str, Spec]) -> None:
    """첫 시도가 점검에 실패하면 사유를 프롬프트에 덧붙여 다시 시도한다."""
    bad = GOOD_D1.model_copy(
        update={"transcript": GOOD_D1.transcript.replace("128에 80", "120에 80")}
    )
    client, messages = _client_for(spec_by_id, {"D1": [bad, GOOD_D1]})
    result = _generator(client).generate(spec_by_id["D1"])
    assert result.status == "passed" and result.attempts == 2
    first, second = (str(c["content"]) for c in messages.calls)
    assert "이전 시도" not in first
    assert "이전 시도" in second and "[E]" in second


def test_gives_up_after_max_attempts_and_keeps_last_candidate(
    spec_by_id: dict[str, Spec],
) -> None:
    """계속 실패하면 needs_human으로 마지막 후보와 점검 결과를 남긴다."""
    bad = GOOD_D1.model_copy(update={"transcript": "가" * 20})
    client, messages = _client_for(spec_by_id, {"D1": [bad]})
    result = _generator(client, max_attempts=3).generate(spec_by_id["D1"])
    assert result.status == "needs_human" and result.attempts == 3
    assert result.case is not None and any(c.level == "fail" for c in result.checks)
    assert len(messages.calls) == 3


def test_human_authored_spec_is_refused_unless_comparison(spec_by_id: dict[str, Spec]) -> None:
    """직접 작성 스펙은 생성하지 않고, 비교용 플래그가 있을 때만 생성한다."""
    client, _ = _client_for(spec_by_id, _good_plan())
    generator = _generator(client)
    with pytest.raises(ValueError, match="직접 작성"):
        generator.generate(spec_by_id["L1"])
    assert generator.generate(spec_by_id["L1"], comparison_only=True).status == "passed"


def test_api_error_becomes_api_failure(spec_by_id: dict[str, Spec]) -> None:
    """API 오류는 예외 대신 api_failure로 기록한다."""
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    client, _ = _client_for(spec_by_id, _good_plan(), error=err)
    generator = _generator(client)
    assert generator.generate(spec_by_id["D1"]).status == "api_failure"
    assert generator.stats.api_failures == 1


def test_cache_replays_without_api_calls(spec_by_id: dict[str, Spec], tmp_path: Path) -> None:
    """같은 조건 재실행은 캐시에서 읽고 API를 부르지 않는다. 재시도 경로도 그대로 재현된다."""
    bad = GOOD_D1.model_copy(
        update={"transcript": GOOD_D1.transcript.replace("128에 80", "120에 80")}
    )
    first_client, first = _client_for(spec_by_id, {"D1": [bad, GOOD_D1]})
    _generator(first_client, cache_dir=tmp_path).generate(spec_by_id["D1"])
    second_client, second = _client_for(spec_by_id, {"D1": [bad, GOOD_D1]})
    generator = _generator(second_client, cache_dir=tmp_path)
    result = generator.generate(spec_by_id["D1"])
    assert len(first.calls) == 2 and len(second.calls) == 0
    assert result.status == "passed" and result.attempts == 2
    assert generator.stats.cache_hits == 2 and generator.is_cached(spec_by_id["D1"])


def test_cost_guard_stops_generation(spec_by_id: dict[str, Spec]) -> None:
    """누적 비용이 한도에 도달하면 다음 호출 전에 BudgetExceeded."""
    client, _ = _client_for(spec_by_id, _good_plan(), usage=_Usage(0, 100_000))
    guard = CostGuard(0.5, Price(4.0, 20.0))  # 첫 호출이 $2
    generator = _generator(client, guard=guard)
    generator.generate(spec_by_id["D1"])
    with pytest.raises(BudgetExceeded):
        generator.generate(spec_by_id["H3"])


def test_prompt_hides_hypothesis_and_describes_facts(spec_by_id: dict[str, Spec]) -> None:
    """생성기에게 사전 가설을 보여주지 않고, 정답은 facts로 전달한다. 피드백은 덧붙는다."""
    spec = spec_by_id["D1"]
    message = build_user_message(spec)
    assert spec.hypothesis not in message and "hypothesis" not in message
    payload = json.loads(message)
    assert payload["facts"]["blood_pressure"]["value"] == "128/80"
    assert payload["length_chars"] == [150, 300]
    assert "stt_errors" not in payload
    assert "stt_errors" in json.loads(build_user_message(spec_by_id["S1"]))
    assert "- [E] 숫자 없음" in build_user_message(spec, "- [E] 숫자 없음")
    assert GENERATION_PROMPT_VERSION == "g9"
    assert "칠십이" in SYSTEM_PROMPT and "따르지 않습니다" in SYSTEM_PROMPT
    # g2~g3 규칙: 종결·간투사·지정 외 오인식·해설·첫 문장·새 임상 사실
    for keyword in (
        "이상입니다",
        "간투사",
        "열거된 오인식 외에는",
        "해설 금지",
        "방문 시각으로 시작하지 않습니다",
        "새 임상 사실 금지",
        "chatter_topics",
        "field_order",
        "phrasing_hints",
        "opening_scene",
        "명시적인 문장",
        "감정·기분 표현 금지",
        "웃으시면서",
        "말하는 방식만 설명합니다",
        "거절하거나 평가하거나 훈계하지도",
        "오전 방문에서 '점심'이라는 말을 쓰지 않습니다",
    ):
        assert keyword in SYSTEM_PROMPT


def test_estimate_excludes_cached_specs(spec_by_id: dict[str, Spec], tmp_path: Path) -> None:
    """캐시에 있는 스펙은 호출 예정에서 빠지고 비용 범위가 낮아진다."""
    client, _ = _client_for(spec_by_id, _good_plan())
    generator = _generator(client, cache_dir=tmp_path)
    specs = [spec_by_id[i] for i in ("D1", "H3", "S1")]
    before = estimate_generation(generator, specs, Price(4.0, 20.0))
    generator.generate(spec_by_id["D1"])
    after = estimate_generation(generator, specs, Price(4.0, 20.0))
    assert (before.to_call, after.to_call) == (3, 2) and after.cost_high < before.cost_high
    assert before.cost_low < before.cost_high


# -- CLI ---------------------------------------------------------------------


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """.env와 환경변수에서 격리된 설정 (키는 가짜 값)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CARENOTE_GENERATOR_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    return Settings()


@pytest.fixture
def no_key_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """API 키가 없는 설정."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return Settings()


def _must_not_build(_: Settings) -> ParseClient:
    raise AssertionError("클라이언트를 만들면 안 되는 경로다")


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--spec-dir", str(SPEC_DIR),
        "--out-dir", str(tmp_path / "cand"),
        "--cache-dir", str(tmp_path / "cache"),
        "--golden", str(GOLDEN_DIR),
        *extra,
    ]  # fmt: skip


def test_cli_dry_run_needs_no_key_or_client(
    no_key_settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """dry-run은 키 없이 동작하고 클라이언트를 만들지 않으며 파일도 만들지 않는다."""
    code = main(
        _args(tmp_path, "--pilot", "--dry-run"),
        settings=no_key_settings,
        client_factory=_must_not_build,
    )
    out = capsys.readouterr().out
    assert code == 0 and "claude-opus-5-5" in out and "API 호출 예정 3건" in out
    assert "직접 작성 스펙 ['L1']" in out and "예상 비용" in out
    assert not (tmp_path / "cand").exists()


def test_cli_default_selection_is_llm_authored_only(
    no_key_settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """기본 선택은 직접 작성 스펙 10건을 뺀 30건이다. split으로 좁힐 수 있다."""
    main(_args(tmp_path, "--dry-run"), settings=no_key_settings, client_factory=_must_not_build)
    assert "스펙 30건" in capsys.readouterr().out
    main(
        _args(tmp_path, "--dry-run", "--split", "dev"),
        settings=no_key_settings,
        client_factory=_must_not_build,
    )
    assert "스펙 7건" in capsys.readouterr().out  # dev 10건 중 직접 작성 3건(L1, V1, G2) 제외


def test_cli_refuses_without_approval_over_budget_or_key(
    settings: Settings,
    no_key_settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """승인 없음, 추정 초과, 키 없음, 알 수 없는 모델은 모두 거부(2)한다."""
    base = _args(tmp_path, "--pilot")
    assert main(base, settings=settings, client_factory=_must_not_build) == 2
    assert "비용 승인이 없다" in capsys.readouterr().err
    assert main([*base, "--approve-cost", "0.001"], settings=settings,
                client_factory=_must_not_build) == 2  # fmt: skip
    assert "승인액" in capsys.readouterr().err
    assert main([*base, "--approve-cost", "5"], settings=no_key_settings,
                client_factory=_must_not_build) == 2  # fmt: skip
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert main([*base, "--model", "claude-mystery-9", "--dry-run"], settings=settings,
                client_factory=_must_not_build) == 2  # fmt: skip
    assert "단가가 등록되지 않은" in capsys.readouterr().err


def test_cli_pilot_run_writes_candidates_and_review_sheet(
    settings: Settings,
    spec_by_id: dict[str, Spec],
    all_specs: list[Spec],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """파일럿 3건을 생성하고 후보 JSON, pilot 폴더(L1), 배치 점검, 검수 시트를 만든다."""
    client, messages = _client_for(spec_by_id, _good_plan())
    code = main(
        _args(tmp_path, "--pilot", "--approve-cost", "5"),
        settings=settings,
        client_factory=lambda _: client,
    )
    out = capsys.readouterr().out
    cand = tmp_path / "cand"
    assert code == 0 and len(messages.calls) == 3
    assert {p.name for p in cand.glob("*.json")} == {"H3.json", "S1.json"}
    assert (cand / "pilot" / "L1.json").exists()  # 직접 작성 스펙은 비교용으로만
    doc = json.loads((cand / "H3.json").read_text(encoding="utf-8"))
    assert doc["status"] == "passed" and doc["review"]["status"] == "pending"
    assert doc["generator"]["prompt_version"] == "g9"
    diversity = doc["generator"]["diversity"]
    assert diversity["opening_type"] and len(diversity["chatter_topics"]) == 2
    assert doc["batch_checks"]
    assert "usage: api_calls=3" in out

    sheet = tmp_path / "sheet.md"
    assert main(["--spec-dir", str(SPEC_DIR), "--out-dir", str(cand),
                 "--render-review", str(sheet)], settings=settings) == 0  # fmt: skip
    text = sheet.read_text(encoding="utf-8")
    assert "## H3" in text and "## S1" in text and "검수: ☐ 승인" in text
    assert render_review_sheet(cand, all_specs).count("검수: ☐ 승인") == 2


def test_cli_budget_stop_returns_3_and_keeps_cache(
    settings: Settings, spec_by_id: dict[str, Spec], tmp_path: Path
) -> None:
    """호출 한 번이 승인액을 넘기면 다음 스펙 전에 멈추고(3) 끝낸 결과는 캐시에 남는다."""
    client, _ = _client_for(spec_by_id, _good_plan(), usage=_Usage(0, 100_000))
    code = main(
        _args(tmp_path, "--spec", "D1", "--spec", "H3", "--approve-cost", "1.5"),
        settings=settings,
        client_factory=lambda _: client,
    )
    assert code == 3
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_batch_checks_flag_duplicate_candidates(
    spec_by_id: dict[str, Spec], tmp_path: Path
) -> None:
    """저장된 후보 사이의 중복(H)이 파일에 기록된다."""
    out = tmp_path / "cand"
    out.mkdir()
    for sid in ("D1", "H3"):
        doc: dict[str, Any] = {
            "spec_id": sid,
            "transcript": GOOD_D1.transcript,
            "checks": [],
            "batch_checks": [],
        }
        (out / f"{sid}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    summary = apply_batch_checks(out, golden_dir=None)
    assert "H3" in summary and "[H] fail" in summary["H3"][0] and "D1" not in summary


def test_batch_checks_tolerate_missing_golden_dir(tmp_path: Path) -> None:
    """기준 골든셋 폴더가 없어도 중복 점검은 후보끼리만 수행하고 죽지 않는다."""
    out = tmp_path / "cand"
    out.mkdir()
    doc: dict[str, Any] = {
        "spec_id": "D1",
        "transcript": GOOD_D1.transcript,
        "checks": [],
        "batch_checks": [],
    }
    (out / "D1.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    assert apply_batch_checks(out, golden_dir=tmp_path / "nope") == {}


def test_labeling_sample_includes_all_risky_llm_specs(all_specs: list[Spec]) -> None:
    """독립 라벨링 표본은 위험도 medium/high인 생성 스펙을 모두 포함하고 최소 10건이며 재현된다."""
    sample = pick_labeling_sample(all_specs)
    risky = {s.spec_id for s in all_specs if s.author == "llm" and s.label_risk != "low"}
    assert risky <= set(sample) and len(sample) >= 10
    assert sample == pick_labeling_sample(all_specs)
    assert not {s.spec_id for s in all_specs if s.author == "human"} & set(sample)
