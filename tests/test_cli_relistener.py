"""--verifier CLI 테스트. 가짜 클라이언트만 쓰며 실제 API는 호출하지 않는다."""

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
from carenote.extraction.llm_extractor import ParseClient, ParseResponse
from carenote.extraction.relistener import LLMReadback
from carenote.extraction.schema import LLMExtractionB1, LLMFieldB1
from carenote.extraction.verifier import LLMVerdict

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "golden" / "v1.jsonl"


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 1500
    output_tokens: int = 100
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class _Client:
    """추출에는 정답을, 검증에는 unsupported를, 재읽기에는 conflict를 돌려준다."""

    def __init__(self, cases: list[GoldenCase]) -> None:
        self._by_transcript = {c.transcript: c for c in cases}
        self.extract_calls = 0
        self.verify_calls = 0
        self.read_calls = 0
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
        if output_format is LLMReadback:
            self.read_calls += 1
            return cast(ParseResponse, _Response(LLMReadback(value="", quote="", conflict=True)))
        if output_format is LLMVerdict:
            self.verify_calls += 1
            return cast(ParseResponse, _Response(LLMVerdict(verdict="unsupported", quote="")))
        self.extract_calls += 1
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


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CARENOTE_EXTRACTOR_MODEL", raising=False)
    monkeypatch.delenv("CARENOTE_VERIFIER_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    return Settings()


def _no_client(_: Settings) -> ParseClient:
    raise AssertionError("클라이언트를 만들면 안 되는 경로다")


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--extractor", "llm-b1",
        "--golden", str(GOLDEN_DIR),
        "--cache-dir", str(tmp_path / "cache"),
        "--relistener",
        *extra,
    ]  # fmt: skip


def test_dry_run_shows_relistener_estimate_without_client(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(_args(tmp_path, "--dry-run"), settings=settings, client_factory=_no_client)
    out = capsys.readouterr().out
    assert code == 0
    assert "Re-listener(텍스트 폴백) 모델 claude-haiku-5-5" in out and "합계 최대" in out


def test_refuses_on_test_split(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        _args(tmp_path, "--split", "test", "--final-report", "--approve-cost", "5"),
        settings=settings,
        client_factory=_no_client,
    )
    assert code == 2 and "dev 분할" in capsys.readouterr().err


def test_refuses_when_estimate_exceeds_approval(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        _args(tmp_path, "--approve-cost", "0.001"), settings=settings, client_factory=_no_client
    )
    assert code == 2 and "승인액" in capsys.readouterr().err


def test_run_rereads_risk_fields_and_reports_usage(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _Client(load_golden(GOLDEN_DIR))
    out_path = tmp_path / "report.json"
    code = main(
        _args(tmp_path, "--limit", "3", "--approve-cost", "5", "--out", str(out_path)),
        settings=settings,
        client_factory=lambda _: cast(ParseClient, client),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert client.extract_calls == 3 and client.read_calls > 0 and client.verify_calls == 0
    assert "relistener: calls=" in out
    usage = json.loads(out_path.read_text(encoding="utf-8"))["usage"]
    assert usage["relistener"]["calls"] == client.read_calls


def test_stacks_with_verifier(settings: Settings, tmp_path: Path) -> None:
    client = _Client(load_golden(GOLDEN_DIR))
    out_path = tmp_path / "report.json"
    code = main(
        _args(
            tmp_path, "--verifier", "--limit", "3", "--approve-cost", "5", "--out", str(out_path)
        ),
        settings=settings,
        client_factory=lambda _: cast(ParseClient, client),
    )
    assert code == 0 and client.verify_calls > 0 and client.read_calls > 0
    usage = json.loads(out_path.read_text(encoding="utf-8"))["usage"]
    assert usage["verifier"]["calls"] == client.verify_calls
    assert usage["relistener"]["calls"] == client.read_calls
