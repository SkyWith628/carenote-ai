"""per_field 보고서, --run-tag, 실행 간 변동 분석 도구."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from carenote.config import Settings
from carenote.evaluation.__main__ import main
from carenote.evaluation.extractor import NullExtractor, OracleExtractor, PredictedField
from carenote.evaluation.loader import load_golden
from carenote.evaluation.metrics import CORRECT_OUTCOMES, FIELD_OUTCOMES, field_outcomes
from carenote.evaluation.runner import evaluate
from carenote.evaluation.schema import Evidence, FieldKey, Flag, GoldenCase, Source
from carenote.evaluation.variance import (
    analyze,
    compare_runs,
    format_summary,
    format_verdict,
    load_reports,
)
from carenote.evaluation.variance import main as variance_main
from carenote.extraction.llm_extractor import LLMExtractor, ParseClient
from carenote.extraction.schema import LLMExtractionB1

GOLDEN = Path(__file__).parent.parent / "data" / "golden"


@pytest.fixture(scope="module")
def cases() -> list[GoldenCase]:
    return load_golden(GOLDEN)


# -- per_field -------------------------------------------------------------------


def test_oracle_per_field_is_all_correct(cases: list[GoldenCase]) -> None:
    report = evaluate(OracleExtractor(cases), cases)
    assert len(report.per_field) == 60
    flat = [o for fields in report.per_field.values() for o in fields.values()]
    assert len(flat) == 480 and set(flat) <= {"tp", "tn"}


def test_null_extractor_per_field_is_fn_or_tn(cases: list[GoldenCase]) -> None:
    report = evaluate(NullExtractor(), cases)
    flat = {o for fields in report.per_field.values() for o in fields.values()}
    assert flat == {"fn", "tn"}


class _WrongBloodPressure:
    def extract(self, transcript: str) -> list[PredictedField]:
        return [
            PredictedField(
                key=FieldKey.BLOOD_PRESSURE,
                value="1/1",
                source=Source.STATED,
                evidence=(Evidence(text=transcript[:2]),),
                flag=Flag.OK,
            ),
            PredictedField(key=FieldKey.MOOD, value="양호", source=Source.INFERRED),
        ]


def test_per_field_outcomes_match_the_case_counts(cases: list[GoldenCase]) -> None:
    """필드별 결과의 tp/fp/fn 합이 케이스 집계(Counts)와 정확히 같다."""
    report = evaluate(_WrongBloodPressure(), cases)
    assert set(report.per_field[cases[0].id].values()) <= set(FIELD_OUTCOMES)
    table = {"tp": (1, 0, 0), "fn": (0, 0, 1), "fpfn": (0, 1, 1), "fp": (0, 1, 0), "tn": (0, 0, 0)}
    for case in cases:
        sums = [0, 0, 0]
        for outcome in report.per_field[case.id].values():
            for i, n in enumerate(table[outcome]):
                sums[i] += n
        c = report.per_case[case.id]
        assert sums == [c.tp, c.fp, c.fn], case.id
    assert "fpfn" in {o for f in report.per_field.values() for o in f.values()}


def test_special_notes_is_scored_by_presence(cases: list[GoldenCase]) -> None:
    case = next(c for c in cases if c.field(FieldKey.SPECIAL_NOTES).value)
    pred = [
        PredictedField(key=FieldKey.SPECIAL_NOTES, value="전혀 다른 문장", source=Source.STATED)
    ]
    assert field_outcomes(case, pred)["special_notes"] == "tp"


def test_report_json_carries_per_field(cases: list[GoldenCase]) -> None:
    data = cast(dict[str, Any], evaluate(OracleExtractor(cases), cases).to_dict())
    assert data["per_field"]["gs-001"]["blood_pressure"] == "tp"


# -- --run-tag -------------------------------------------------------------------


class _Boom:
    @property
    def messages(self) -> object:
        raise AssertionError("호출되면 안 된다")


def _extractor(cache: Path, tag: str | None) -> LLMExtractor:
    return LLMExtractor(
        _Boom_client(),
        variant="b1",
        model="claude-haiku-5-5",
        effort="medium",
        cache_dir=cache,
        run_tag=tag,
    )


def _Boom_client() -> ParseClient:  # noqa: N802
    return _Boom()  # type: ignore[return-value]


def test_run_tag_separates_cache_but_none_keeps_the_old_key(tmp_path: Path) -> None:
    plain, tagged_1, tagged_2 = (_extractor(tmp_path, t) for t in (None, "r1", "r2"))
    keys = {e._cache_key("가") for e in (plain, tagged_1, tagged_2)}  # pyright: ignore[reportPrivateUsage]
    assert len(keys) == 3
    legacy = json.dumps(["claude-haiku-5-5", "medium", "b1", "v1", "가"], ensure_ascii=False)
    import hashlib

    assert plain._cache_key("가") == hashlib.sha256(legacy.encode()).hexdigest()  # pyright: ignore[reportPrivateUsage]
    assert plain.usage_summary()["run_tag"] is None
    assert tagged_1.usage_summary()["run_tag"] == "r1"


def test_cli_run_tag_bypasses_cached_result_in_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = load_golden(GOLDEN, "dev")[0]
    seeded = LLMExtractor(
        _Boom_client(),
        variant="b1",
        model="claude-sonnet-5-5",
        effort="medium",
        cache_dir=tmp_path,
    )
    seeded._write_cache(  # pyright: ignore[reportPrivateUsage]
        seeded._cache_key(case.transcript),  # pyright: ignore[reportPrivateUsage]
        LLMExtractionB1(fields=[]),
    )
    base = ["--extractor", "llm-b1", "--split", "dev", "--dry-run", "--cache-dir", str(tmp_path)]
    assert main(base, settings=Settings()) == 0
    assert "캐시 적중 1건" in capsys.readouterr().out
    assert main([*base, "--run-tag", "r1"], settings=Settings()) == 0
    out = capsys.readouterr().out
    assert "캐시 적중 0건" in out and "실행 태그 r1" in out


# -- variance --------------------------------------------------------------------


def _report(
    f1: float, outcomes: dict[str, dict[str, str]], per_case: dict[str, tuple[int, int, int]]
) -> dict[str, Any]:
    return {
        "overall": {"f1": f1},
        "per_field": outcomes,
        "per_case": {k: {"tp": v[0], "fp": v[1], "fn": v[2]} for k, v in per_case.items()},
    }


CASES = {"c1": (4, 0, 0), "c2": (3, 1, 1), "c3": (4, 0, 0)}


def _fields(a: str, b: str) -> dict[str, dict[str, str]]:
    return {"c1": {"x": "tp", "y": a}, "c2": {"x": "tp", "y": b}, "c3": {"x": "tp", "y": "tn"}}


def test_analyze_computes_sigma_instability_and_majority() -> None:
    reports = [
        _report(0.90, _fields("tp", "fn"), CASES),
        _report(0.92, _fields("tp", "tp"), CASES),
        _report(0.94, _fields("fn", "fn"), CASES),
    ]
    s = analyze(reports)
    assert s.runs == 3 and s.mean_f1 == pytest.approx(0.92)
    assert s.sd_f1 == pytest.approx(0.02)
    assert s.mean_pairwise_abs_diff == pytest.approx((0.02 + 0.04 + 0.02) / 3)
    assert set(s.unstable) == {"c1 y", "c2 y"} and s.stable_errors == ()
    assert s.instability_rate == pytest.approx(2 / 6)
    assert s.majority_f1 == pytest.approx(1.0 * 0 + _majority_expected())
    assert "불안정 필드 2/6" in format_summary(s)


def _majority_expected() -> float:
    # c1 y: tp·tp·fn → tp / c2 y: fn·tp·fn → fn / 나머지 tp 3개 → tp 4, fn 1 → P=1, R=0.8
    return 2 * 1.0 * 0.8 / 1.8


def test_stable_errors_are_wrong_in_every_run() -> None:
    reports = [_report(0.9, _fields("fn", "tp"), CASES), _report(0.9, _fields("fpfn", "tp"), CASES)]
    s = analyze(reports)
    assert s.stable_errors == ("c1 y",) and s.unstable == () and s.sd_f1 == pytest.approx(0.0)


def test_analyze_rejects_bad_inputs() -> None:
    one = _report(0.9, _fields("tp", "tp"), CASES)
    with pytest.raises(ValueError, match="2번 이상"):
        analyze([one])
    no_field = {"overall": {"f1": 0.9}, "per_case": {}}
    with pytest.raises(ValueError, match="per_field"):
        analyze([one, no_field])
    other = _report(0.9, {"c9": {"x": "tp"}}, {"c9": (1, 0, 0)})
    with pytest.raises(ValueError, match="필드 집합"):
        analyze([one, other])


def _group(base_tp: int, f1s: list[float]) -> list[dict[str, Any]]:
    per_case = {f"c{i}": (base_tp, 1, 1) for i in range(20)}
    outcomes = {f"c{i}": {"x": "tp"} for i in range(20)}
    return [_report(f1, outcomes, per_case) for f1 in f1s]


def test_compare_runs_applies_both_pre_registered_rules() -> None:
    x = _group(8, [0.90, 0.91, 0.89])
    # 평균 차이 +10%p, σ 작음, 케이스 결과도 일관 → 차이 있음
    y = _group(9, [1.00, 0.99, 1.01])
    v = compare_runs(x, y)
    assert v.passes_sigma_rule and v.passes_interval_rule and v.different
    assert "차이 있음" in format_verdict(v)


def test_compare_runs_says_indistinguishable_when_noise_is_large() -> None:
    x = _group(8, [0.90, 0.96, 0.84])
    y = _group(8, [0.91, 0.97, 0.85])  # 평균 차이 +1%p, σ가 큼
    v = compare_runs(x, y)
    assert not v.passes_sigma_rule and not v.different
    assert "구분 불가" in format_verdict(v)


def test_variance_cli_reads_report_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths: list[str] = []
    for i, f1 in enumerate([0.90, 0.92]):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(_report(f1, _fields("tp", "tp"), CASES)), encoding="utf-8")
        paths.append(str(p))
    assert variance_main(paths) == 0
    assert "실행 2회" in capsys.readouterr().out
    assert variance_main([paths[0]]) == 2
    assert len(load_reports([Path(paths[0])])) == 1
    assert CORRECT_OUTCOMES == {"tp", "tn"}
