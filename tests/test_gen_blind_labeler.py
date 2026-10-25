"""블라인드 모델 라벨러 테스트. 가짜 클라이언트만 쓰며 실제 API는 호출하지 않는다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import anthropic
import httpx2
import pytest
from conftest import GOOD_D1, GOOD_H3, SPEC_DIR
from pydantic import BaseModel

from carenote.config import Settings
from carenote.extraction.llm_extractor import ParseClient, ParseResponse
from carenote.generation.__main__ import main
from carenote.generation.blind_labeler import (
    BLIND_LABELER_VERSION,
    BlindLabeler,
    LabeledCase,
    LabeledField,
    build_system_prompt,
    build_user_message,
    to_label_file,
)
from carenote.generation.candidates import blind_ids, compare_blind, format_blind_report
from carenote.generation.generator import ApiUnavailable
from carenote.generation.spec import Spec

POLICY = Path(__file__).parent.parent / "docs" / "label-policy.md"


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 4000
    output_tokens: int = 1500
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


def _labeled(
    spec: Spec, *, uncertain: set[str] | None = None, wrong: set[str] | None = None
) -> LabeledCase:
    """스펙 정답을 그대로 옮긴 라벨 (일부 필드는 틀리게 하거나 애매하다고 표시)."""
    fields: list[LabeledField] = []
    for key, gold in spec.gold.items():
        value = gold.value or ""
        if wrong and key.value in wrong:
            value = "틀린값"
        fields.append(
            LabeledField(
                key=key,
                value=value,
                source=gold.source.value,  # type: ignore[arg-type]
                flag=gold.flag.value,  # type: ignore[arg-type]
                candidates=list(gold.candidates),
                uncertain=bool(uncertain and key.value in uncertain),
                note="두 가지로 읽힘" if uncertain and key.value in uncertain else "",
            )
        )
    return LabeledCase(fields=fields, overall_note="")


class _FakeMessages:
    def __init__(self, by_transcript: dict[str, LabeledCase], error: Exception | None = None):
        self._by = by_transcript
        self._error = error
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"model": model, "system": system, "content": content})
        if self._error is not None:
            raise self._error
        transcript = content.removeprefix("<transcript>\n").removesuffix("\n</transcript>")
        assert output_format is LabeledCase and output_config == {"effort": "medium"}
        return cast(ParseResponse, _Response(self._by[transcript]))


class _FakeClient:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages


def _labeler(client: _FakeClient, **kw: Any) -> BlindLabeler:
    return BlindLabeler(
        cast(ParseClient, client),
        model="claude-opus-5-5",
        effort="medium",
        policy_path=POLICY,
        **kw,
    )


def test_system_prompt_contains_policy_but_no_spec_or_gold(all_specs: list[Spec]) -> None:
    """라벨러에게는 라벨 안내와 정책 문서만 간다. 스펙 제목·정답·가설·프로브 정보는 없다."""
    system = build_system_prompt(POLICY)
    assert "정답을 모릅니다" in system and "라벨 정책" in system
    assert "사흘 전에 넘어지셨다" in system  # 정책 #7 예시가 문서째로 들어간다
    for spec in all_specs:
        assert spec.title not in system, spec.spec_id
        assert spec.hypothesis not in system and spec.traps not in system
    for hidden in ("label_risk", "gray_fields", "hypothesis", "field_review_required"):
        assert hidden not in system
    assert build_user_message("안녕") == "<transcript>\n안녕\n</transcript>"


def test_labeler_returns_cached_result_and_records_usage(
    spec_by_id: dict[str, Spec], tmp_path: Path
) -> None:
    """같은 transcript는 캐시에서 읽고 API를 다시 부르지 않는다. 호출에는 transcript만 간다."""
    messages = _FakeMessages({GOOD_D1.transcript: _labeled(spec_by_id["D1"])})
    first = _labeler(_FakeClient(messages), cache_dir=tmp_path)
    assert first.label(GOOD_D1.transcript) is not None
    assert len(messages.calls) == 1 and messages.calls[0]["content"].startswith("<transcript>")
    assert "D1" not in messages.calls[0]["content"]
    second_messages = _FakeMessages({})
    second = _labeler(_FakeClient(second_messages), cache_dir=tmp_path)
    assert second.label(GOOD_D1.transcript) is not None
    assert second_messages.calls == [] and second.stats.cache_hits == 1
    assert first.usage_summary()["labeler_version"] == BLIND_LABELER_VERSION


def test_labeler_stops_after_three_consecutive_failures() -> None:
    """연속 3회 실패하면 마지막 오류와 함께 멈춘다."""
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    labeler = _labeler(_FakeClient(_FakeMessages({}, error=err)))
    assert labeler.label("a") is None and labeler.label("b") is None
    with pytest.raises(ApiUnavailable, match="APIConnectionError"):
        labeler.label("c")


def test_label_file_feeds_compare_and_surfaces_disagreements_and_uncertainty(
    all_specs: list[Spec],
) -> None:
    """라벨 파일을 비교하면 불일치와 '애매하다고 표시한 필드'가 따로 나온다."""
    key = blind_ids(["D1", "H3"])
    labeled = {
        key["D1"]: _labeled(next(s for s in all_specs if s.spec_id == "D1"), wrong={"pulse"}),
        key["H3"]: _labeled(
            next(s for s in all_specs if s.spec_id == "H3"), uncertain={"fall_event"}
        ),
    }
    file = to_label_file(labeled, "테스트")
    report = compare_blind(file, key, all_specs)
    assert report.total == 16 and report.value_agree == 15
    text = format_blind_report(report)
    assert "pulse" in text and "애매하다고 표시한 필드 1개" in text and "두 가지로 읽힘" in text
    assert file["cases"][key["H3"]]["fall_event"]["uncertain"] is True


@pytest.fixture
def blind_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Settings, Path, Path]:
    """D1·H3 후보, 블라인드 시트(키 포함), 가짜 키가 있는 설정."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    settings = Settings()
    out = tmp_path / "candidates" / "v2"
    out.mkdir(parents=True)
    for sid, case in (("D1", GOOD_D1), ("H3", GOOD_H3)):
        doc: dict[str, Any] = {
            "spec_id": sid,
            "transcript": case.transcript,
            "checks": [],
            "batch_checks": [],
        }
        (out / f"{sid}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    sheet = tmp_path / "round.md"
    assert main(["--spec-dir", str(SPEC_DIR), "--out-dir", str(out),
                 "--render-review", str(sheet), "--blind"]) == 0  # fmt: skip
    return settings, out, sheet


def test_cli_model_label_dry_run_refusal_and_full_run(
    blind_setup: tuple[Settings, Path, Path],
    all_specs: list[Spec],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """dry-run은 API 없이, 승인 없으면 거부, 승인 후에는 라벨 파일을 쓰고 비교할 수 있다."""
    settings, out, sheet = blind_setup
    by_id = {s.spec_id: s for s in all_specs}
    messages = _FakeMessages(
        {
            GOOD_D1.transcript: _labeled(by_id["D1"]),
            GOOD_H3.transcript: _labeled(by_id["H3"], uncertain={"fall_event"}),
        }
    )
    base = ["--spec-dir", str(SPEC_DIR), "--out-dir", str(out), "--cache-dir", str(tmp_path / "c")]
    ops = ["--model-label", str(tmp_path / "labels.json"), "--blind-sheet", str(sheet)]

    def must_not_build(_: Settings) -> ParseClient:
        raise AssertionError("dry-run/거부 경로에서는 클라이언트를 만들면 안 된다")

    assert main([*base, *ops, "--dry-run"], settings=settings, client_factory=must_not_build) == 0
    assert "API 호출 예정 2건" in capsys.readouterr().out
    assert main([*base, *ops], settings=settings, client_factory=must_not_build) == 2
    assert "--approve-cost" in capsys.readouterr().err

    code = main(
        [*base, *ops, "--approve-cost", "2"],
        settings=settings,
        client_factory=lambda _: cast(ParseClient, _FakeClient(messages)),
    )
    assert code == 0 and len(messages.calls) == 2
    assert "라벨 2/2건 저장" in capsys.readouterr().out

    # 비교: 라벨 파일 이름(round.model-labels.json)으로 라운드 키를 찾는다
    model_labels = tmp_path / "round.model-labels.json"
    model_labels.write_text(
        (tmp_path / "labels.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert main([*base, "--compare-blind", str(model_labels)]) == 0
    printed = capsys.readouterr().out
    assert "필드 16개" in printed and "애매하다고 표시한 필드 1개" in printed
