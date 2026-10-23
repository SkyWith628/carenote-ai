"""실측 CLI 테스트. 가짜 클라이언트만 쓰며 실제 API는 호출하지 않는다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from carenote.config import Settings
from carenote.evaluation.__main__ import main
from carenote.evaluation.loader import load_golden
from carenote.evaluation.schema import GoldenCase
from carenote.extraction.estimate import estimate_run
from carenote.extraction.llm_extractor import LLMExtractor, ParseClient, ParseResponse
from carenote.extraction.pricing import PRICES, CostGuard, Price, cost_usd, price_for
from carenote.extraction.schema import LLMExtractionB1, LLMFieldB1

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "golden"


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 1500
    output_tokens: int = 1000
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class _GoldClient:
    """정답을 B1 응답으로 돌려주고 호출 수를 센다."""

    def __init__(self, cases: list[GoldenCase], usage: _Usage | None = None) -> None:
        self._by_transcript = {c.transcript: c for c in cases}
        self._usage = usage or _Usage()
        self.calls = 0
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
        self.calls += 1
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
        return cast(ParseResponse, _Response(LLMExtractionB1(fields=fields), usage=self._usage))


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """.env와 환경변수에서 격리된 설정 (키는 가짜 값)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CARENOTE_EXTRACTOR_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    return Settings()


@pytest.fixture
def no_key_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """API 키가 없는 설정."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return Settings()


def _factory_that_must_not_run(_: Settings) -> ParseClient:
    raise AssertionError("클라이언트를 만들면 안 되는 경로다")


def _args(tmp_path: Path, *extra: str, extractor: str = "llm-b1") -> list[str]:
    return [
        "--extractor", extractor,
        "--golden", str(GOLDEN_DIR),
        "--cache-dir", str(tmp_path / "cache"),
        *extra,
    ]  # fmt: skip


def test_dry_run_prints_estimate_without_client_or_key(
    no_key_settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """dry-run은 키 없이도 동작하고 클라이언트를 만들지 않는다."""
    code = main(
        _args(tmp_path, "--dry-run"),
        settings=no_key_settings,
        client_factory=_factory_that_must_not_run,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "예상 비용" in out and "API 호출 예정 20건" in out and "claude-sonnet-5-5" in out
    assert "API를 호출하지 않았다" in out


def test_dry_run_counts_cache_hits(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """캐시에 있는 케이스는 호출 예정에서 빠진다."""
    cases = load_golden(GOLDEN_DIR)
    client = _GoldClient(cases)
    assert main(_args(tmp_path, "--limit", "5", "--approve-cost", "5"),
                settings=settings, client_factory=lambda _: cast(ParseClient, client)) == 0  # fmt: skip
    capsys.readouterr()
    main(
        _args(tmp_path, "--dry-run"),
        settings=settings,
        client_factory=_factory_that_must_not_run,
    )
    out = capsys.readouterr().out
    assert "캐시 적중 5건" in out and "API 호출 예정 15건" in out


def test_refuses_without_cost_approval(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """승인 금액이 없으면 추정만 보여주고 실행을 거부한다."""
    code = main(_args(tmp_path), settings=settings, client_factory=_factory_that_must_not_run)
    captured = capsys.readouterr()
    assert code == 2
    assert "예상 비용" in captured.out and "비용 승인이 없다" in captured.err


def test_refuses_when_estimate_exceeds_approval(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """추정 최대 비용이 승인액보다 크면 거부한다."""
    code = main(
        _args(tmp_path, "--approve-cost", "0.001"),
        settings=settings,
        client_factory=_factory_that_must_not_run,
    )
    assert code == 2
    assert "승인액" in capsys.readouterr().err


def test_refuses_without_api_key(
    no_key_settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """승인이 있어도 키가 없으면 거부한다."""
    code = main(
        _args(tmp_path, "--approve-cost", "5"),
        settings=no_key_settings,
        client_factory=_factory_that_must_not_run,
    )
    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_refuses_unknown_model_price(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """단가를 모르는 모델은 비용을 추정할 수 없어 거부한다 (dry-run도 마찬가지)."""
    code = main(
        _args(tmp_path, "--model", "claude-mystery-9", "--dry-run"),
        settings=settings,
        client_factory=_factory_that_must_not_run,
    )
    assert code == 2
    assert "단가가 등록되지 않은 모델" in capsys.readouterr().err


def test_full_run_with_approval_writes_report(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """승인·키가 있으면 실행하고 보고서에 사용량과 비용이 들어간다. 위험 필드 flag도 계산된다."""
    cases = load_golden(GOLDEN_DIR)
    client = _GoldClient(cases)
    out_path = tmp_path / "report.json"
    code = main(
        _args(tmp_path, "--approve-cost", "5", "--out", str(out_path)),
        settings=settings,
        client_factory=lambda _: cast(ParseClient, client),
    )
    printed = capsys.readouterr().out
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert code == 0 and client.calls == 20
    assert "usage: calls=20" in printed
    assert report["usage"]["calls"] == 20 and report["usage"]["estimated_cost_usd"] > 0
    assert report["overall"]["risk_recall"]["den"] == 10
    assert report["overall"]["risk_recall"]["num"] == 6  # Flagger 상한 (design 11장)


def test_rerun_uses_cache_and_makes_no_calls(settings: Settings, tmp_path: Path) -> None:
    """같은 조건 재실행은 API를 부르지 않는다."""
    cases = load_golden(GOLDEN_DIR)
    first = _GoldClient(cases)
    main(_args(tmp_path, "--approve-cost", "5"), settings=settings,
         client_factory=lambda _: cast(ParseClient, first))  # fmt: skip
    second = _GoldClient(cases)
    main(_args(tmp_path, "--approve-cost", "5"), settings=settings,
         client_factory=lambda _: cast(ParseClient, second))  # fmt: skip
    assert first.calls == 20 and second.calls == 0


def test_budget_guard_stops_run_and_keeps_completed_results(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """실제 사용량이 추정보다 커서 승인액에 도달하면 중단(3)하고, 끝낸 결과는 캐시에 남는다."""
    cases = load_golden(GOLDEN_DIR)
    # 호출 1회가 약 $0.5가 되도록 부풀린 사용량 (sonnet: 입력 $2 / 출력 $10 per MTok)
    client = _GoldClient(cases, _Usage(input_tokens=0, output_tokens=50_000))
    code = main(
        _args(tmp_path, "--approve-cost", "0.5", "--limit", "1"),
        settings=settings,
        client_factory=lambda _: cast(ParseClient, client),
    )
    assert code == 0  # 1건이라 중단 전에 끝남
    capsys.readouterr()

    code = main(
        _args(tmp_path, "--approve-cost", "1.0", "--limit", "4"),
        settings=settings,
        client_factory=lambda _: cast(ParseClient, client),
    )
    err = capsys.readouterr().err
    assert code == 3 and "승인 한도" in err
    assert len(list((tmp_path / "cache").glob("*.json"))) >= 2  # 끝낸 호출은 저장됨


def test_oracle_and_null_still_work(capsys: pytest.CaptureFixture[str]) -> None:
    """기준선 extractor는 키·비용 없이 동작한다."""
    assert main(["--extractor", "oracle", "--golden", str(GOLDEN_DIR)]) == 0
    assert "overall" in capsys.readouterr().out
    assert main(["--extractor", "null", "--golden", str(GOLDEN_DIR), "--limit", "3"]) == 0
    assert "cases: 3" in capsys.readouterr().out


def test_price_table_and_cost_math() -> None:
    """단가표와 비용 계산."""
    assert set(PRICES) >= {"claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"}
    price = price_for("claude-sonnet-5-5")
    assert price == Price(2.0, 10.0)
    assert price is not None
    assert cost_usd(price, 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert price_for("nope") is None


def test_cost_guard_blocks_after_limit() -> None:
    """누적 비용이 한도에 도달하면 다음 check가 실패한다."""
    from carenote.extraction.pricing import BudgetExceeded

    guard = CostGuard(0.01, Price(2.0, 10.0))
    guard.check()
    guard.add(0, 1_000)  # $0.01
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_estimate_is_zero_when_everything_cached(tmp_path: Path) -> None:
    """전부 캐시에 있으면 호출 예정 0건, 비용 0."""
    cases = load_golden(GOLDEN_DIR)[:3]
    client = _GoldClient(cases)
    llm = LLMExtractor(
        cast(ParseClient, client),
        variant="b1",
        model="claude-sonnet-5-5",
        effort="medium",
        cache_dir=tmp_path,
    )
    for c in cases:
        llm.extract(c.transcript)
    price = price_for("claude-sonnet-5-5")
    assert price is not None
    est = estimate_run(llm, cases, price)
    assert est.to_call == 0 and est.cached == 3 and est.cost_high == 0
