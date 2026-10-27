"""비용·지연 실측 도구 테스트. 가짜 클라이언트만 쓰며 실제 API는 호출하지 않는다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from carenote.config import Settings
from carenote.evaluation.costlatency import (
    CallRecord,
    Measurement,
    VisitTiming,
    main,
    summarize,
)
from carenote.evaluation.loader import load_golden
from carenote.evaluation.schema import GoldenCase
from carenote.extraction.llm_extractor import ParseClient, ParseResponse
from carenote.extraction.relistener import LLMReadback
from carenote.extraction.schema import LLMExtractionB1, LLMFieldB1
from carenote.extraction.sentinel_llm import LLMSentinelOutput
from carenote.extraction.verifier import LLMVerdict

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "golden" / "v1.jsonl"


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 1000
    output_tokens: int = 100
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class _Client:
    def __init__(self, cases: list[GoldenCase]) -> None:
        self._by = {c.transcript: c for c in cases}
        self.messages = self

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
        if output_format is LLMVerdict:
            return cast(ParseResponse, _Response(LLMVerdict(verdict="unsupported", quote="")))
        if output_format is LLMReadback:
            return cast(ParseResponse, _Response(LLMReadback(value="", quote="", conflict=True)))
        if output_format is LLMSentinelOutput:
            return cast(ParseResponse, _Response(LLMSentinelOutput(signals=[])))
        content = messages[0]["content"]
        transcript = content.removeprefix("<transcript>\n").removesuffix("\n</transcript>")
        fields = [
            LLMFieldB1(
                key=f.key,
                value=f.value,
                source=f.source.value,  # type: ignore[arg-type]
                evidence=[e.text for e in f.evidence],
                candidates=list(f.candidates),
            )
            for f in self._by[transcript].fields
        ]
        return cast(ParseResponse, _Response(LLMExtractionB1(fields=fields)))


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    return Settings()


def _args(*extra: str) -> list[str]:
    return ["--golden", str(GOLDEN_DIR), *extra]


def _no_client(_: Settings) -> ParseClient:
    raise AssertionError("클라이언트를 만들면 안 되는 경로다")


def test_dry_run_shows_every_component_without_client(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(_args("--dry-run", "--limit", "3"), settings=settings, client_factory=_no_client)
    out = capsys.readouterr().out
    assert code == 0
    for word in ("Extractor", "Sentinel", "Verifier", "Re-listener", "합계 최대"):
        assert word in out


def test_refuses_without_approval_or_when_estimate_exceeds(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_args("--limit", "3"), settings=settings, client_factory=_no_client) == 2
    assert "비용 승인이 없다" in capsys.readouterr().err
    code = main(
        _args("--limit", "3", "--approve-cost", "0.0001"),
        settings=settings,
        client_factory=_no_client,
    )
    assert code == 2 and "승인액" in capsys.readouterr().err


def test_run_records_every_call_and_reports_all_configs(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _Client(load_golden(GOLDEN_DIR))
    out_path = tmp_path / "cl.json"
    code = main(
        _args("--limit", "3", "--approve-cost", "5", "--out", str(out_path)),
        settings=settings,
        client_factory=lambda _: cast(ParseClient, client),
    )
    assert code == 0
    summary = json.loads(out_path.read_text(encoding="utf-8"))["summary"]
    assert summary["visits"] == 3
    assert [r["config"] for r in summary["rows"]] == [
        "Extractor(B1 v4)",
        "+ Sentinel LLM 보조",
        "+ Verifier",
        "+ Re-listener 폴백",
    ]
    calls = [r["calls_per_visit"] for r in summary["rows"]]
    assert calls == sorted(calls) and calls[0] == 1.0 and calls[1] == 2.0  # 누적이라 단조 증가
    costs = [r["cost_per_visit_usd"] for r in summary["rows"]]
    assert costs == sorted(costs) and costs[0] > 0
    assert set(summary["per_call"]) == {"extractor", "sentinel", "verifier", "relistener"}
    assert "Re-listener 폴백" in capsys.readouterr().out


def test_budget_stop_returns_3_with_partial_summary(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _Client(load_golden(GOLDEN_DIR))
    # 한도를 낮춰 호출 몇 번 만에 닿게 하려고 추정 상한 검사를 건너뛴다.
    import carenote.evaluation.costlatency as cl

    original = cl._estimate_high  # pyright: ignore[reportPrivateUsage]
    cl._estimate_high = lambda s, c: (0.0, [])  # type: ignore[assignment]  # noqa: SLF001
    try:
        code = main(
            _args("--limit", "3", "--approve-cost", "0.0005"),
            settings=settings,
            client_factory=lambda _: cast(ParseClient, client),
        )
    finally:
        cl._estimate_high = original  # type: ignore[assignment]  # noqa: SLF001
    assert code == 3 and "중단" in capsys.readouterr().err


def test_summarize_math_with_known_numbers() -> None:
    def rec(case: str, comp: str, seconds: float, cost: float) -> CallRecord:
        return CallRecord(case, comp, "m", seconds, 10, 10, cost)

    m = Measurement(
        visits=[VisitTiming("a", 2.0, 1.0, 3.0, 2), VisitTiming("b", 4.0, 3.0, 7.0, 1)],
        records=[
            rec("a", "extractor", 2.0, 0.01),
            rec("a", "sentinel", 1.0, 0.001),
            rec("a", "verifier", 1.0, 0.0002),
            rec("a", "verifier", 0.5, 0.0002),
            rec("a", "relistener", 0.8, 0.0002),
            rec("a", "relistener", 0.9, 0.0002),
            rec("b", "extractor", 4.0, 0.02),
            rec("b", "sentinel", 2.0, 0.001),
            rec("b", "verifier", 3.0, 0.0002),
            rec("b", "relistener", 2.5, 0.0002),
        ],
    )
    s = summarize(m)
    first, last = s["rows"][0], s["rows"][-1]
    assert first["calls_per_visit"] == 1.0 and last["calls_per_visit"] == 5.0  # (2+8... )/2 방문
    assert first["cost_per_visit_usd"] == pytest.approx(0.015)
    assert first["latency_parallel_s"] == pytest.approx(3.0)  # 추출 호출 평균 (2+4)/2
    assert last["latency_parallel_s"] == pytest.approx(5.0)  # 실측 총 벽시계 평균 (3+7)/2
    assert last["latency_sequential_s"] > last["latency_parallel_s"] - 1  # 순차가 병렬 이상
    assert s["targets_per_visit"] == pytest.approx(1.5)


def test_partial_config_parallel_latency_is_stage1_plus_slowest_call() -> None:
    """Verifier만 켠 구성은 실측하지 않고 1단계 + 가장 느린 Verifier 호출로 추정한다."""

    def rec(comp: str, seconds: float) -> CallRecord:
        return CallRecord("a", comp, "m", seconds, 1, 1, 0.0)

    m = Measurement(
        visits=[VisitTiming("a", 3.0, 2.0, 5.0, 2)],
        records=[
            rec("extractor", 3.0),
            rec("sentinel", 1.0),
            rec("verifier", 1.0),
            rec("verifier", 1.5),
            rec("relistener", 2.0),
        ],
    )
    rows = summarize(m)["rows"]
    assert rows[2]["latency_parallel_s"] == pytest.approx(4.5)  # 3.0 + max(1.0, 1.5)
    assert rows[2]["latency_parallel_measured"] is False
    assert (
        rows[3]["latency_parallel_s"] == pytest.approx(5.0) and rows[3]["latency_parallel_measured"]
    )


def test_production_mode_times_the_real_extractor_end_to_end(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _Client(load_golden(GOLDEN_DIR))
    out_path = tmp_path / "prod.json"
    code = main(
        _args("--limit", "3", "--approve-cost", "5", "--production", "--out", str(out_path)),
        settings=settings,
        client_factory=lambda _: cast(ParseClient, client),
    )
    assert code == 0 and "운영 경로 3건" in capsys.readouterr().out
    summary = json.loads(out_path.read_text(encoding="utf-8"))["summary"]
    assert summary["mode"] == "production" and summary["visits"] == 3
    assert len(summary["walls_s"]) == 3 and summary["calls_per_visit"] >= 2.0
    assert summary["wall_mean_s"] <= summary["sequential_sum_s"] + 0.5  # 병렬이면 합 이하
