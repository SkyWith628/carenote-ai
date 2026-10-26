"""v4 ablation 도구: --prompt-edits, --case-ids, evaluation.ablate."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from carenote.config import Settings
from carenote.evaluation.__main__ import main as eval_main
from carenote.evaluation.ablate import (
    fisher_one_sided,
    format_table,
    main,
    parse_targets,
    parse_variant,
)
from carenote.extraction.llm_extractor import LLMExtractor, ParseClient, ParseResponse
from carenote.extraction.prompts import ALL_V4_EDITS, build_system_prompt, normalize_edits
from carenote.extraction.schema import LLMExtractionB1, LLMFieldB1

GOLDEN = Path(__file__).parent.parent / "data" / "golden"
CULPRIT = "일삼오 → 135"  # v4 편집 2의 문구 (가짜 모델이 이 문구가 있으면 번복을 확정해 버린다)


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CARENOTE_EXTRACTOR_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    return Settings()


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 1000
    output_tokens: int = 300
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class _Client:
    """프롬프트에 CULPRIT 문구가 있으면 혈압을 확정 값(오답)으로, 없으면 후보 두 개(정답)로 낸다."""

    def __init__(self, usage: _Usage | None = None) -> None:
        self.calls = 0
        self.messages = self
        self._usage = usage or _Usage()

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
        quote = "혈압은 백삼십 팔십이었나"
        if CULPRIT in system:
            field_ = LLMFieldB1(
                key="blood_pressure",  # type: ignore[arg-type]
                value="140/82",
                source="stated",
                evidence=[quote],
                candidates=[],
            )
        else:
            field_ = LLMFieldB1(
                key="blood_pressure",  # type: ignore[arg-type]
                value=None,
                source="stated",
                evidence=[quote],
                candidates=["130/80", "140/80"],
            )
        return cast(ParseResponse, _Response(LLMExtractionB1(fields=[field_]), usage=self._usage))


def _boom(_: Settings) -> ParseClient:
    raise AssertionError("클라이언트를 만들면 안 되는 경로다")


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--extractor", "llm-b1", "--model", "claude-haiku-5-5",
        "--golden", str(GOLDEN), "--cache-dir", str(tmp_path / "cache"),
        "--targets", "gs-003:blood_pressure", *extra,
    ]  # fmt: skip


# -- 변형 표기와 통계 --------------------------------------------------------------


def test_parse_variant_forms() -> None:
    assert parse_variant("v3").version == "v3" and parse_variant("v4").edits is None
    assert parse_variant("v4-drop:2").edits == (1, 3, 4, 5)
    assert parse_variant("v4-only:2").edits == (2,)
    assert parse_variant("v4-edits:1,3").edits == (1, 3)
    assert parse_variant("v4-drop:2").label == "v4-drop:2"
    for bad in ("v9", "v4-drop", "v4-drop:1,2", "v4-only:x", "v4-edits:"):
        with pytest.raises(ValueError):
            parse_variant(bad)


def test_fisher_matches_known_values() -> None:
    assert fisher_one_sided(3, 5, 0, 5, b_higher=False) == pytest.approx(10 / 120)
    assert fisher_one_sided(0, 5, 5, 5, b_higher=True) == pytest.approx(1 / 252)
    assert fisher_one_sided(5, 5, 5, 5, b_higher=True) == pytest.approx(1.0)
    assert parse_targets(["gs-003:blood_pressure"]) == [("gs-003", "blood_pressure")]
    with pytest.raises(ValueError):
        parse_targets(["gs-003"])


# -- --prompt-edits --------------------------------------------------------------


def test_full_edit_set_is_exactly_v4_and_subsets_differ() -> None:
    full = build_system_prompt("b1", "v4")
    assert build_system_prompt("b1", "v4", ALL_V4_EDITS) == full
    assert normalize_edits(ALL_V4_EDITS) is None and normalize_edits(None) is None
    without2 = build_system_prompt("b1", "v4", (1, 3, 4, 5))
    assert CULPRIT in full and CULPRIT not in without2
    only2 = build_system_prompt("b1", "v4", (2,))
    v3 = build_system_prompt("b1", "v3")
    assert CULPRIT in only2 and only2 != v3 and len(only2) < len(full)
    assert "evidence" in only2 and "evidence" not in build_system_prompt("b0", "v4", (2,))
    for bad in ((0,), (6,), ()):
        with pytest.raises(ValueError):
            normalize_edits(bad)
    with pytest.raises(ValueError, match="v4에서만"):
        build_system_prompt("b1", "v3", (2,))


def test_prompt_edits_change_the_cache_key_but_full_set_keeps_the_v4_key(tmp_path: Path) -> None:
    def make(edits: tuple[int, ...] | None) -> LLMExtractor:
        return LLMExtractor(
            cast(ParseClient, _Client()),
            variant="b1",
            model="claude-haiku-5-5",
            effort="medium",
            cache_dir=tmp_path,
            prompt_version="v4",
            prompt_edits=edits,
        )

    def key(e: LLMExtractor) -> str:
        return e._cache_key("가")  # pyright: ignore[reportPrivateUsage]

    assert key(make(None)) == key(make(ALL_V4_EDITS))
    assert len({key(make(None)), key(make((1, 3, 4, 5))), key(make((2,)))}) == 3
    assert make((2,)).usage_summary()["prompt_edits"] == [2]
    assert make(None).usage_summary()["prompt_edits"] is None
    with pytest.raises(ValueError):
        LLMExtractor(
            cast(ParseClient, _Client()),
            variant="b1",
            model="claude-haiku-5-5",
            effort="medium",
            prompt_version="v3",
            prompt_edits=(2,),
        )


def test_evaluation_cli_prompt_edits_and_case_ids(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = [
        "--extractor", "llm-b1", "--golden", str(GOLDEN), "--cache-dir", str(tmp_path / "c"),
        "--dry-run",
    ]  # fmt: skip
    assert eval_main([*base, "--prompt", "v4", "--prompt-edits", "2,4", "--case-ids", "gs-003"],
                     settings=settings, client_factory=_boom) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "API 호출 예정 1건" in out and "편집 2,4" in out
    assert eval_main([*base, "--prompt-edits", "2"], settings=settings, client_factory=_boom) == 2
    assert "v4에서만" in capsys.readouterr().err
    assert eval_main([*base, "--case-ids", "gs-999"], settings=settings, client_factory=_boom) == 2
    # test 케이스는 기본(dev) 분할에서 지정할 수 없다
    assert eval_main([*base, "--case-ids", "gs-060"], settings=settings, client_factory=_boom) == 2


# -- ablate 드라이버 --------------------------------------------------------------


def test_dry_run_estimates_without_a_client(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        _args(tmp_path, "--variants", "v3", "v4", "--runs", "20", "--dry-run"),
        settings=settings,
        client_factory=_boom,
    )
    out = capsys.readouterr().out
    assert code == 0 and "호출 최대 40건" in out and "API를 호출하지 않았다" in out


def test_refuses_without_approval_or_with_test_cases_and_bad_input(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    v = ["--variants", "v3", "v4", "--runs", "2"]
    assert main(_args(tmp_path, *v), settings=settings, client_factory=_boom) == 2
    assert "--approve-cost" in capsys.readouterr().err
    assert main(_args(tmp_path, *v, "--approve-cost", "0.0001"), settings=settings,
                client_factory=_boom) == 2  # fmt: skip
    assert "추정 최대" in capsys.readouterr().err
    test_case = _args(tmp_path, *v, "--approve-cost", "1")
    test_case[test_case.index("gs-003:blood_pressure")] = "gs-060:blood_pressure"
    assert main(test_case, settings=settings, client_factory=_boom) == 2
    assert "test는 쓸 수 없다" in capsys.readouterr().err
    assert main(_args(tmp_path, "--variants", "v9", "--approve-cost", "1"), settings=settings,
                client_factory=_boom) == 2  # fmt: skip
    bad_field = _args(tmp_path, *v, "--approve-cost", "1")
    bad_field[bad_field.index("gs-003:blood_pressure")] = "gs-003:nope"
    assert main(bad_field, settings=settings, client_factory=_boom) == 2


def test_ablation_finds_the_culprit_edit_and_writes_json(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _Client()
    out_json = tmp_path / "ab.json"
    code = main(
        _args(
            tmp_path,
            "--variants",
            "v3",
            "v4",
            "v4-drop:1",
            "v4-drop:2",
            "--runs",
            "6",
            "--approve-cost",
            "0.5",
            "--out",
            str(out_json),
        ),  # fmt: skip
        settings=settings,
        client_factory=lambda _: cast(ParseClient, client),
    )
    text = capsys.readouterr().out
    assert code == 0 and client.calls == 24
    table = text.split("[gs-003:blood_pressure]")[1]
    assert "v3             6/6" in table and "v4             0/6" in table
    assert "v4-drop:1      0/6" in table and "v4-drop:2      6/6" in table
    assert table.count("원인 후보") == 1
    assert table.index("원인 후보") > table.index("v4-drop:2")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["variants"]["v4-drop:2"]["edits"] == [1, 3, 4, 5]
    assert data["variants"]["v4"]["targets"]["gs-003:blood_pressure"]["correct"] == 0
    assert data["cost_usd"] > 0


def test_second_run_is_served_from_cache(settings: Settings, tmp_path: Path) -> None:
    argv = _args(tmp_path, "--variants", "v3", "v4", "--runs", "3", "--approve-cost", "0.5")
    first, second = _Client(), _Client()
    assert main(argv, settings=settings, client_factory=lambda _: cast(ParseClient, first)) == 0
    assert main(argv, settings=settings, client_factory=lambda _: cast(ParseClient, second)) == 0
    assert first.calls == 6 and second.calls == 0


def test_budget_limit_stops_with_partial_results_balanced_across_variants(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 호출 1건 = (100000×0.10 + 20000×0.50)/1e6 = $0.02. 한도 $0.05는 추정 최대 이상이라 시작하고,
    # 3건(=$0.06)에서 멈춘다. 변형을 번갈아 돌리므로 v3 2건, v4 1건이 남는다.
    client = _Client(_Usage(100_000, 20_000))
    out_json = tmp_path / "partial.json"
    code = main(
        _args(
            tmp_path,
            "--variants",
            "v3",
            "v4",
            "--runs",
            "10",
            "--approve-cost",
            "0.05",
            "--out",
            str(out_json),
        ),  # fmt: skip
        settings=settings,
        client_factory=lambda _: cast(ParseClient, client),
    )
    captured = capsys.readouterr()
    assert code == 3 and client.calls == 3
    assert "승인 한도 $0.05에 도달" in captured.err
    data = json.loads(out_json.read_text(encoding="utf-8"))
    n = {k: v["targets"]["gs-003:blood_pressure"]["n"] for k, v in data["variants"].items()}
    assert n == {"v3": 2, "v4": 1}
    assert "v3             2/2" in captured.out


def test_format_table_needs_no_anchor_variants() -> None:
    from carenote.evaluation.ablate import Tally

    t = Tally(outcomes=["tn", "fp", "tn"], failures=1)
    spec = parse_variant("v4-only:2")
    table = format_table({spec.label: {("gs-003", "blood_pressure"): t}}, [spec])
    assert "2/3" in table and "실패 1" in table and "원인 후보" not in table
