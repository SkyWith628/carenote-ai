"""test 최종 보고 준비: 헤드라인 집계, 비교 도구의 케이스 제외 옵션, final_table, 실행 배너."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from carenote.config import Settings
from carenote.evaluation.__main__ import final_run_banner, format_report
from carenote.evaluation.__main__ import main as eval_main
from carenote.evaluation.compare import load_per_case, select_per_case
from carenote.evaluation.compare import main as compare_main
from carenote.evaluation.extractor import NullExtractor, OracleExtractor, PredictedField
from carenote.evaluation.final_table import (
    all_configs,
    compare_configs,
    comparison_pairs,
    format_kr1,
    format_table,
    judge,
    kr1_before_after,
    report_path,
)
from carenote.evaluation.final_table import main as table_main
from carenote.evaluation.loader import load_golden
from carenote.evaluation.runner import evaluate
from carenote.evaluation.schema import Evidence, FieldKey, Flag, GoldenCase, Source
from carenote.evaluation.stats import Interval
from carenote.evaluation.variance import compare_runs
from carenote.extraction.llm_extractor import ParseClient, ParseResponse
from carenote.extraction.schema import LLMExtractionB1, LLMFieldB1

GOLDEN = Path(__file__).parent.parent / "data" / "golden"
EXCLUDED = ("gs-052", "gs-060")  # G1, R3 (현장 검토 대기)


@pytest.fixture(scope="module")
def cases() -> list[GoldenCase]:
    return load_golden(GOLDEN)


@pytest.fixture(scope="module")
def test_cases() -> list[GoldenCase]:
    return load_golden(GOLDEN, "test")


# -- 헤드라인 집계 ---------------------------------------------------------------------


def test_headline_excludes_exactly_the_review_required_cases(cases: list[GoldenCase]) -> None:
    report = evaluate(OracleExtractor(cases), cases, "all")
    assert report.review_required_ids == EXCLUDED
    assert report.headline is not None and report.headline.f1 == 1.0
    assert report.headline_all_fields is not None and report.headline_all_fields.f1 == 1.0
    assert report.headline_strict is not None and report.headline_strict.f1 == 1.0
    assert report.headline_ci is not None and report.headline_ci.point == 1.0
    assert len(report.per_case) == 60 and len(report.per_case_recoverable) == 60


def test_headline_denominators_are_smaller_than_overall(cases: list[GoldenCase]) -> None:
    report = evaluate(NullExtractor(), cases, "all")
    assert report.headline_all_fields is not None
    assert report.headline_all_fields.hallucinated.den < report.overall.hallucinated.den
    # 두 케이스를 빼도 어느 한쪽이 아닌 정확히 그 두 케이스만 빠졌는지: 합이 맞는다
    gap = report.overall.hallucinated.den - report.headline_all_fields.hallucinated.den
    removed = sum(1 for c in cases if c.id in EXCLUDED for f in c.fields if f.value is None)
    assert gap == removed


def test_recoverable_counts_drop_the_unrecoverable_field(cases: list[GoldenCase]) -> None:
    report = evaluate(NullExtractor(), cases, "all")
    full, rec = report.per_case["gs-046"], report.per_case_recoverable["gs-046"]  # S2 복약
    assert full.fn - rec.fn == 1


def test_test_split_headline_has_28_cases_and_report_json_and_text(
    test_cases: list[GoldenCase],
) -> None:
    report = evaluate(OracleExtractor(test_cases), test_cases, "test")
    assert report.case_count == 30 and report.review_required_ids == EXCLUDED
    data = cast(dict[str, Any], report.to_dict())
    assert data["review_required_ids"] == list(EXCLUDED)
    assert data["headline"]["f1"] == 1.0 and data["headline_ci"]["point"] == 1.0
    assert "gs-046" in data["per_case_recoverable"]
    text = format_report(report)
    assert "헤드라인: 현장 검토 대기 gs-052, gs-060 제외 28건" in text
    assert "헤드라인(복원가능)" in text and "헤드라인 F1 95% 구간" in text


def test_dev_report_has_no_headline_block_but_keeps_the_field(cases: list[GoldenCase]) -> None:
    dev = [c for c in cases if c.split == "dev"]
    report = evaluate(OracleExtractor(dev), dev, "dev")
    assert report.review_required_ids == () and report.headline is not None
    assert "-- 헤드라인" not in format_report(report)


# -- 비교 도구의 케이스 제외 옵션 -------------------------------------------------------------


def _doc(per_case: dict[str, tuple[int, int, int]], rec: dict[str, tuple[int, int, int]],
         excluded: list[str]) -> dict[str, Any]:  # fmt: skip
    def block(src: dict[str, tuple[int, int, int]]) -> dict[str, dict[str, int]]:
        return {k: {"tp": v[0], "fp": v[1], "fn": v[2]} for k, v in src.items()}

    return {
        "overall": {"f1": 0.9},
        "per_case": block(per_case),
        "per_case_recoverable": block(rec),
        "review_required_ids": excluded,
    }


def test_select_per_case_options() -> None:
    doc = _doc(
        {"a": (4, 0, 0), "b": (0, 4, 4)}, {"a": (4, 0, 0), "b": (1, 3, 3)}, ["b"]
    )  # fmt: skip
    assert set(select_per_case(doc)) == {"a", "b"}
    assert set(select_per_case(doc, exclude_review_required=True)) == {"a"}
    assert select_per_case(doc, recoverable=True)["b"].tp == 1
    assert select_per_case(doc)["b"].tp == 0
    with pytest.raises(ValueError, match="review_required_ids"):
        select_per_case({"per_case": doc["per_case"]}, exclude_review_required=True)
    with pytest.raises(ValueError, match="per_case_recoverable"):
        select_per_case({"per_case": doc["per_case"]}, recoverable=True)


def test_compare_cli_headline_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ids = [f"c{i}" for i in range(12)]
    good = {i: (4, 0, 0) for i in ids}
    bad_excluded = {**good, "c0": (0, 4, 4)}  # 제외 대상 케이스에서만 나쁘다
    a = _doc(bad_excluded, bad_excluded, ["c0"])
    b = _doc(good, good, ["c0"])
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(b), encoding="utf-8")
    assert compare_main([str(pa), str(pb)]) == 0
    assert "+" in capsys.readouterr().out  # 전체에서는 b가 낫다
    assert compare_main([str(pa), str(pb), "--exclude-review-required", "--recoverable"]) == 0
    assert "+0.0%p" in capsys.readouterr().out  # 제외하면 같다
    assert len(load_per_case(pa, exclude_review_required=True)) == 11


def test_variance_compare_runs_honors_the_headline_options() -> None:
    ids = [f"c{i}" for i in range(15)]

    def run(f1_bias: int, excluded_case: tuple[int, int, int]) -> dict[str, Any]:
        per = {i: (4 + f1_bias, 0, 0) for i in ids}
        per["c0"] = excluded_case
        doc = _doc(per, per, ["c0"])
        doc["per_field"] = {"c0": {"x": "tp"}}
        doc["overall"] = {"f1": 0.9}
        return doc

    xs = [run(0, (0, 4, 4)) for _ in range(3)]
    ys = [run(0, (4, 0, 0)) for _ in range(3)]  # 차이는 제외 대상 케이스에만 있다
    assert compare_runs(xs, ys).mean_diff > 0
    assert compare_runs(xs, ys, exclude_review_required=True).mean_diff == pytest.approx(0.0)


# -- final_table -----------------------------------------------------------------------


def _write_reports(
    directory: Path, test_cases: list[GoldenCase], pairs: dict[tuple[str, str, str], str]
) -> None:
    for config, kind in pairs.items():
        extractor = OracleExtractor(test_cases) if kind == "oracle" else NullExtractor()
        report = evaluate(extractor, test_cases, "test").to_dict()
        report["usage"] = {"estimated_cost_usd": 0.0123}
        report_path(directory, config).write_text(json.dumps(report), encoding="utf-8")


def test_pre_registered_comparisons_cover_twelve_pairs() -> None:
    pairs = comparison_pairs()
    assert len(pairs) == 12 and len(all_configs()) == 8
    names = [p[0] for p in pairs]
    assert names.count("B1−B0") == 4 and names.count("Sonnet−Haiku") == 4
    assert names.count("v4−v1") == 4
    assert ("B1−B0", ("v4", "haiku", "b0"), ("v4", "haiku", "b1")) in pairs


@pytest.mark.parametrize(
    ("low", "high", "point", "sonnet", "label"),
    [
        (0.01, 0.03, 0.02, False, "차이 있음"),  # 구간이 0 밖이고 |Δ| > 1.4%p
        (0.001, 0.012, 0.007, False, "구분 불가"),  # 구간은 0 밖이지만 |Δ|가 변동 기준 미달
        (-0.01, 0.03, 0.02, False, "구분 불가"),  # 구간이 0을 포함
        (0.001, 0.012, 0.007, True, "방향만(Sonnet 변동 미측정)"),  # Sonnet은 구간만
        (-0.01, 0.03, 0.02, True, "구분 불가"),
        (-0.03, -0.01, -0.02, False, "차이 있음"),  # 음의 방향도 동일
    ],
)
def test_judge_applies_the_pre_registered_rules(
    low: float, high: float, point: float, sonnet: bool, label: str
) -> None:
    verdict, threshold = judge(Interval(point, low, high), sonnet)
    assert verdict == label and threshold == pytest.approx(0.0141, abs=1e-4)
    assert judge(Interval(None, None, None), sonnet)[0] == "비교 불가"


def test_table_and_comparisons_from_reports(
    tmp_path: Path, test_cases: list[GoldenCase], capsys: pytest.CaptureFixture[str]
) -> None:
    _write_reports(
        tmp_path,
        test_cases,
        {
            ("v4", "haiku", "b0"): "null",
            ("v4", "haiku", "b1"): "oracle",
            ("v4", "sonnet", "b0"): "null",
            ("v4", "sonnet", "b1"): "oracle",
        },
    )
    assert table_main(["--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "현장 검토 대기 케이스(gs-052, gs-060)" in out
    assert "| v4 Haiku B1 | **100.0%**" in out and "| v4 Haiku B0 | **0.0%**" in out
    assert "| v1 Sonnet B0 | 없음 |" in out  # 아직 없는 구성
    assert "$0.012" in out
    haiku = next(line for line in out.splitlines() if "| v4 Haiku B0 | v4 Haiku B1 |" in line)
    assert "+100.0%p" in haiku and "**차이 있음**" in haiku
    sonnet = next(line for line in out.splitlines() if "| v4 Sonnet B0 | v4 Sonnet B1 |" in line)
    assert "**방향만(Sonnet 변동 미측정)**" in sonnet
    cross = next(line for line in out.splitlines() if "| v4 Haiku B1 | v4 Sonnet B1 |" in line)
    assert "+0.0%p" in cross and "구분 불가" in cross
    assert len(compare_configs({})) == 0
    assert table_main(["--dir", str(tmp_path / "missing")]) == 2


def test_format_table_notes_missing_headline() -> None:
    text = format_table({("v4", "haiku", "b0"): {"overall": {"f1": 0.9}}})
    assert "헤드라인 없음" in text


class _WrongBp:
    """모든 케이스의 혈압을 transcript에 없는 값으로 내고 flag는 ok로 둔다."""

    def extract(self, transcript: str) -> list[PredictedField]:
        return [
            PredictedField(
                key=FieldKey.BLOOD_PRESSURE,
                value="140/82",
                source=Source.STATED,
                evidence=(Evidence(text=transcript[:2]),),
                flag=Flag.OK,
            )
        ]


def test_kr1_before_after_on_a_hedged_case(cases: list[GoldenCase]) -> None:
    hedged = [c for c in cases if c.id == "gs-003"]  # "팔십이었나, 아니 백사십 팔십이었나"
    row = kr1_before_after(hedged, _WrongBp())
    assert row.silent_before.startswith("1/") and row.silent_after.startswith("0/")
    assert row.f1_same
    text = format_kr1({("v4", "haiku", "b0"): row, ("v4", "haiku", "b1"): None})
    assert "| v4 Haiku B0 | 1/" in text and "| v4 Haiku B1 | 캐시 없음 |" in text


# -- 실행 배너 ---------------------------------------------------------------------------


def test_banner_has_everything_the_decision_log_needs() -> None:
    banner = final_run_banner(
        split="test",
        model="claude-haiku-5-5",
        effort="medium",
        variant="b1",
        prompt_version="v4",
        system_prompt="프롬프트 본문",
        run_tag=None,
        case_count=30,
        commit="abc1234",
        now=datetime(2026, 11, 1, 9, 30, tzinfo=UTC),
    )
    assert banner.startswith("[test-run] 2026-11-01T09:30:00+00:00 split=test")
    assert "프롬프트=v4(" in banner and "모델=claude-haiku-5-5" in banner
    assert "변형=b1" in banner and "케이스=30" in banner and "커밋=abc1234" in banner
    assert "태그" not in banner
    assert "태그=r1" in final_run_banner(
        split="test", model="m", effort="e", variant="b0", prompt_version="v1",
        system_prompt="x", run_tag="r1", case_count=1, commit=None,
    )  # fmt: skip
    assert "커밋=알 수 없음" in final_run_banner(
        split="test", model="m", effort="e", variant="b0", prompt_version="v1",
        system_prompt="x", run_tag=None, case_count=1, commit=None,
    )  # fmt: skip


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


class _GoldClient:
    def __init__(self, cases: list[GoldenCase]) -> None:
        self._by = {c.transcript: c for c in cases}
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
        case = self._by[content.removeprefix("<transcript>\n").removesuffix("\n</transcript>")]
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    return Settings()


def test_test_run_prints_the_banner_and_dev_run_does_not(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    all_cases = load_golden(GOLDEN)
    client = _GoldClient(all_cases)
    base = [
        "--extractor", "llm-b1", "--model", "claude-haiku-5-5", "--golden", str(GOLDEN),
        "--cache-dir", str(tmp_path / "c"), "--approve-cost", "5",
    ]  # fmt: skip

    def factory(_: Settings) -> ParseClient:
        return cast(ParseClient, client)

    out_json = tmp_path / "t.json"
    assert eval_main([*base, "--split", "test", "--final-report", "--out", str(out_json)],
                     settings=settings, client_factory=factory) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert out.count("[test-run]") == 1 and "split=test" in out and "케이스=30" in out
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["review_required_ids"] == list(EXCLUDED) and data["headline"] is not None
    assert eval_main([*base, "--split", "dev"], settings=settings, client_factory=factory) == 0
    assert "[test-run]" not in capsys.readouterr().out
