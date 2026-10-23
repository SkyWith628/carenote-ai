"""Eval Runner 테스트: 기준 extractor와 일부러 틀리는 extractor로 지표를 검증한다."""

from pathlib import Path

import pytest

from carenote.evaluation.__main__ import format_report, main
from carenote.evaluation.extractor import NullExtractor, OracleExtractor, PredictedField
from carenote.evaluation.loader import load_golden
from carenote.evaluation.metrics import Counts, Metrics, Ratio, score_case
from carenote.evaluation.runner import evaluate
from carenote.evaluation.schema import Evidence, FieldKey, Flag, GoldenCase, Source

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "golden"


@pytest.fixture(scope="module")
def cases() -> list[GoldenCase]:
    """골든셋 전체."""
    return load_golden(GOLDEN_DIR)


def test_oracle_scores_perfect(cases: list[GoldenCase]) -> None:
    """정답을 그대로 내는 extractor는 모든 지표가 만점/0이다."""
    m = evaluate(OracleExtractor(cases), cases).overall
    assert m.f1 == 1.0
    assert m.ungrounded.num == 0 and m.ungrounded.den > 0
    assert m.hallucinated.num == 0
    assert m.numeric_error.num == 0
    assert m.risk_recall.value == 1.0
    assert m.confirm_recall.value == 1.0


def test_null_extractor_finds_nothing(cases: list[GoldenCase]) -> None:
    """아무것도 못 내면 F1=0, 위험 recall=0, 근거 없는 값은 분모 0(None)."""
    m = evaluate(NullExtractor(), cases).overall
    assert m.f1 == 0.0
    assert m.risk_recall.value == 0.0
    assert m.confirm_recall.value == 0.0
    assert m.ungrounded.value is None
    assert m.hallucinated.num == 0 and m.hallucinated.den > 0


class _FabricatingExtractor:
    """모든 케이스의 혈압을 지어내고 근거도 지어낸다."""

    def extract(self, transcript: str) -> list[PredictedField]:
        return [
            PredictedField(
                key=FieldKey.BLOOD_PRESSURE,
                value="200/100",
                source=Source.STATED,
                evidence=(Evidence(text="이 문장은 transcript에 없다"),),
                flag=Flag.OK,
            )
        ]


def test_fabricated_values_are_caught(cases: list[GoldenCase]) -> None:
    """지어낸 값은 근거 없음, 환각, 수치 오류로 각각 잡힌다."""
    m = evaluate(_FabricatingExtractor(), cases).overall
    assert m.ungrounded.value == 1.0
    assert m.hallucinated.num > 0
    assert m.numeric_error.num > 0
    assert m.risk_recall.value == 0.0


def test_wrong_value_counts_as_both_fp_and_fn(cases: list[GoldenCase]) -> None:
    """틀린 값은 FP와 FN에 모두 센다."""
    case = cases[0]
    gold = case.field(FieldKey.BLOOD_PRESSURE)
    assert gold.value == "130/80"
    wrong = PredictedField(
        key=FieldKey.BLOOD_PRESSURE,
        value="120/80",
        source=Source.STATED,
        evidence=gold.evidence,
    )
    c = score_case(case, [wrong])
    assert (c.tp, c.fp, c.fn) == (0, 1, 7)
    assert c.numeric_error == Ratio(1, 1)
    assert c.ungrounded == Ratio(0, 1)


def test_value_whitespace_is_ignored(cases: list[GoldenCase]) -> None:
    """공백 차이는 같은 값으로 본다."""
    case = cases[0]
    gold = case.field(FieldKey.BLOOD_PRESSURE)
    pred = PredictedField(
        key=FieldKey.BLOOD_PRESSURE,
        value=" 130 / 80 ",
        source=Source.STATED,
        evidence=gold.evidence,
    )
    assert score_case(case, [pred]).tp == 1


def test_ratio_with_zero_denominator_is_none() -> None:
    """분모가 0이면 비율은 None이고 F1도 None이다."""
    assert Ratio().value is None
    assert Metrics.from_counts(Counts()).f1 is None


def test_category_counts_sum_to_overall(cases: list[GoldenCase]) -> None:
    """유형별 케이스 합이 전체와 일치한다."""
    report = evaluate(OracleExtractor(cases), cases)
    assert report.case_count == len(cases)
    assert set(report.by_category) == {c.category.value for c in cases}
    den = sum(m.risk_recall.den for m in report.by_category.values())
    assert den == report.overall.risk_recall.den


def test_cli_writes_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI가 표를 출력하고 JSON 보고서를 저장한다."""
    out = tmp_path / "report.json"
    assert main(["--extractor", "oracle", "--golden", str(GOLDEN_DIR), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "overall" in printed and "risk_combo" in printed
    assert out.read_text(encoding="utf-8").startswith("{")


def test_format_report_handles_none_metrics() -> None:
    """None 지표도 표로 출력된다."""
    report = evaluate(NullExtractor(), [])
    assert "n/a" in format_report(report)
