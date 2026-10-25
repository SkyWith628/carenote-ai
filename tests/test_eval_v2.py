"""러너 확장: 분할, 프로브별 지표, 복원가능 제외, 기권/오답, 부트스트랩, 짝지은 비교."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carenote.config import Settings
from carenote.evaluation.__main__ import format_report, main
from carenote.evaluation.compare import format_difference, load_per_case
from carenote.evaluation.compare import main as compare_main
from carenote.evaluation.extractor import NullExtractor, OracleExtractor, PredictedField
from carenote.evaluation.loader import load_golden
from carenote.evaluation.metrics import Counts, score_case
from carenote.evaluation.runner import evaluate, probe_of
from carenote.evaluation.schema import Evidence, FieldKey, Flag, GoldenCase, Source
from carenote.evaluation.stats import bootstrap_f1, paired_f1_difference
from carenote.extraction.llm_extractor import ParseClient

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "golden"


@pytest.fixture(scope="module")
def all_cases() -> list[GoldenCase]:
    return load_golden(GOLDEN_DIR)


def _case(spec_id: str, cases: list[GoldenCase]) -> GoldenCase:
    return next(c for c in cases if c.spec_id == spec_id)


def test_oracle_is_perfect_on_every_subset(all_cases: list[GoldenCase]) -> None:
    report = evaluate(OracleExtractor(all_cases), all_cases)
    for m in (report.overall, report.recoverable, report.strict):
        assert m is not None and m.f1 == 1.0
        assert m.abstained.num == 0 and m.wrong_value.num == 0
    assert report.f1_ci is not None and report.f1_ci.low == 1.0 == report.f1_ci.high


def test_null_extractor_only_abstains(all_cases: list[GoldenCase]) -> None:
    m = evaluate(NullExtractor(), all_cases).overall
    assert m.abstained.value == 1.0
    assert m.wrong_value.num == 0


class _Wrong:
    """모든 케이스의 맥박에 틀린 값을 낸다."""

    def extract(self, transcript: str) -> list[PredictedField]:
        return [
            PredictedField(
                key=FieldKey.PULSE,
                value="1",
                source=Source.STATED,
                evidence=(Evidence(text=transcript[:3]),),
                flag=Flag.OK,
            )
        ]


def test_wrong_value_is_separated_from_abstention(all_cases: list[GoldenCase]) -> None:
    m = evaluate(_Wrong(), all_cases).overall
    pulse_gold = sum(c.field(FieldKey.PULSE).value is not None for c in all_cases)
    assert m.wrong_value.num == pulse_gold
    assert m.abstained.num == m.abstained.den - pulse_gold


def test_unrecoverable_field_is_excluded_from_recoverable_subset(
    all_cases: list[GoldenCase],
) -> None:
    s2 = _case("S2", all_cases)
    full = score_case(s2, [], "all")
    rec = score_case(s2, [], "recoverable")
    assert full.fn - rec.fn == 1  # 복약 한 필드


def test_gray_fields_are_excluded_only_from_strict(all_cases: list[GoldenCase]) -> None:
    v2 = _case("V3", all_cases)  # gray: mood
    assert FieldKey.MOOD in v2.gray_fields
    rec = score_case(v2, [], "recoverable")
    strict = score_case(v2, [], "strict")
    assert rec.hallucinated.den - strict.hallucinated.den == 1  # mood 정답이 비어 있는 필드


def test_report_has_probe_breakdown_and_per_case(all_cases: list[GoldenCase]) -> None:
    report = evaluate(OracleExtractor(all_cases), all_cases, "all")
    assert set(report.by_probe) == {"v1", *"HCDNLSVGIR"}
    assert probe_of(_case("H3", all_cases)) == "H"
    assert len(report.per_case) == 60
    d = report.to_dict()
    assert d["split"] == "all" and d["f1_ci"] is not None
    json.dumps(d)  # 직렬화 가능
    text = format_report(report)
    assert "프로브별" in text and "복원가능·gray 제외" in text and "기권" in text


def test_bootstrap_is_deterministic_and_brackets_point() -> None:
    per = [Counts(tp=t, fp=f, fn=n) for t, f, n in [(6, 0, 0), (4, 1, 1), (3, 2, 2), (5, 0, 1)]]
    a, b = bootstrap_f1(per), bootstrap_f1(per)
    assert a == b
    assert a.low is not None and a.high is not None and a.point is not None
    assert a.low <= a.point <= a.high


def test_paired_difference_detects_consistent_gap_and_rejects_mismatch() -> None:
    a = {f"c{i}": Counts(tp=3, fp=1, fn=1) for i in range(20)}
    b = {f"c{i}": Counts(tp=4, fp=0, fn=0) for i in range(20)}
    diff = paired_f1_difference(a, b)
    assert diff.low is not None and diff.low > 0
    same = paired_f1_difference(a, a)
    assert same.point == 0 and same.low == 0 == same.high
    with pytest.raises(ValueError, match="케이스 집합"):
        paired_f1_difference(a, {"x": Counts(tp=1)})


def test_compare_cli_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name, extractor in (("null", "null"), ("oracle", "oracle")):
        assert (
            main(
                [
                    "--extractor",
                    extractor,
                    "--split",
                    "all",
                    "--out",
                    str(tmp_path / f"{name}.json"),
                ]
            )
            == 0
        )
    capsys.readouterr()
    assert compare_main([str(tmp_path / "null.json"), str(tmp_path / "oracle.json")]) == 0
    out = capsys.readouterr().out
    assert "+100.0%p" in out and "구분 가능" in out
    assert "구분 불가" in format_difference(
        "a",
        "b",
        paired_f1_difference(
            load_per_case(tmp_path / "null.json"), load_per_case(tmp_path / "null.json")
        ),
    )


def test_compare_rejects_report_without_per_case(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"overall": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="per_case"):
        load_per_case(old)
    assert compare_main([str(old), str(old)]) == 2


def test_default_split_is_dev_and_excludes_test(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--extractor", "null"]) == 0
    assert "cases: 30 (split=dev)" in capsys.readouterr().out
    assert main(["--extractor", "null", "--split", "test"]) == 0
    assert "cases: 30 (split=test)" in capsys.readouterr().out


def _factory(_: Settings) -> ParseClient:
    raise AssertionError("클라이언트를 만들면 안 된다")


@pytest.mark.parametrize("split", ["test", "all"])
def test_llm_run_on_test_requires_final_report_flag(
    split: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["--extractor", "llm-b1", "--split", split, "--dry-run", "--cache-dir", str(tmp_path)],
        settings=Settings(),
        client_factory=_factory,
    )
    assert code == 2
    assert "--final-report" in capsys.readouterr().err


def test_llm_dry_run_on_test_allowed_with_final_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["--extractor", "llm-b1", "--split", "test", "--final-report", "--dry-run",
         "--cache-dir", str(tmp_path)],
        settings=Settings(),
        client_factory=_factory,
    )  # fmt: skip
    assert code == 0
    assert "API 호출 예정 30건" in capsys.readouterr().out
