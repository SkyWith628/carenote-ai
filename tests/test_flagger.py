"""Flagger 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from carenote.evaluation.extractor import NullExtractor, PredictedField
from carenote.evaluation.loader import load_golden
from carenote.evaluation.runner import evaluate
from carenote.evaluation.schema import FieldKey, Flag, Source
from carenote.extraction.flagger import FlaggingExtractor, flag_field

# 20건 기준선(v1). 분할 파일(v2_*)은 tests/test_golden_v2.py에서 따로 검증한다.
GOLDEN_DIR = Path(__file__).parent.parent / "data" / "golden" / "v1.jsonl"


def _f(key: FieldKey, value: str | None, **kw: object) -> PredictedField:
    source = Source.MISSING if value is None else Source.STATED
    return PredictedField(key=key, value=value, source=source, **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        (FieldKey.BLOOD_PRESSURE, "130/80", Flag.OK),
        (FieldKey.BLOOD_PRESSURE, "13/80", Flag.NEEDS_CONFIRM),  # STT가 130을 13으로
        (FieldKey.BLOOD_PRESSURE, "40/90", Flag.NEEDS_CONFIRM),  # '백' 탈락
        (FieldKey.BLOOD_PRESSURE, "80/120", Flag.NEEDS_CONFIRM),  # 수축기 < 이완기
        (FieldKey.BLOOD_PRESSURE, "130 80", Flag.NEEDS_CONFIRM),  # 형식 불량
        (FieldKey.BLOOD_PRESSURE, "300/100", Flag.NEEDS_CONFIRM),
        (FieldKey.BLOOD_PRESSURE, "70/30", Flag.OK),  # 경계값
        (FieldKey.BLOOD_PRESSURE, "260/160", Flag.OK),  # 경계값
        (FieldKey.BLOOD_PRESSURE, "69/40", Flag.NEEDS_CONFIRM),
        (FieldKey.PULSE, "72", Flag.OK),
        (FieldKey.PULSE, "30", Flag.OK),
        (FieldKey.PULSE, "29", Flag.NEEDS_CONFIRM),
        (FieldKey.PULSE, "220", Flag.OK),
        (FieldKey.PULSE, "221", Flag.NEEDS_CONFIRM),
        (FieldKey.PULSE, "칠십이", Flag.NEEDS_CONFIRM),
        (FieldKey.MEDICATION_TAKEN, "복용", Flag.OK),
        (FieldKey.MEDICATION_TAKEN, "미복용", Flag.RISK),
        (FieldKey.MEDICATION_TAKEN, "중복 복용 의심", Flag.RISK),
        (FieldKey.FALL_EVENT, "없음", Flag.OK),
        (FieldKey.FALL_EVENT, "있음", Flag.RISK),
        (FieldKey.MEAL_INTAKE, "미섭취", Flag.OK),
        (FieldKey.MOOD, "저조", Flag.OK),
        (FieldKey.SPECIAL_NOTES, "무릎 통증 호소", Flag.OK),
    ],
)
def test_value_rules(key: FieldKey, value: str, expected: Flag) -> None:
    """값 기반 규칙."""
    assert flag_field(_f(key, value)) is expected


@pytest.mark.parametrize("key", list(FieldKey))
def test_missing_field_flag_depends_on_risk_membership(key: FieldKey) -> None:
    """위험 필드가 missing이면 확인 필요, 그 외 필드는 ok."""
    risky = {
        FieldKey.BLOOD_PRESSURE,
        FieldKey.PULSE,
        FieldKey.MEDICATION_TAKEN,
        FieldKey.FALL_EVENT,
    }
    expected = Flag.NEEDS_CONFIRM if key in risky else Flag.OK
    assert flag_field(PredictedField(key=key)) is expected


def test_candidates_need_confirm_for_any_field() -> None:
    """후보가 2개 이상이면 필드 종류와 상관없이 확인 필요."""
    field = _f(FieldKey.MOOD, None, candidates=("양호", "저조"))
    assert flag_field(field) is Flag.NEEDS_CONFIRM


def test_single_candidate_is_not_ambiguity() -> None:
    """후보가 1개뿐이면 모호하지 않다."""
    assert flag_field(_f(FieldKey.MOOD, "양호", candidates=("양호",))) is Flag.OK


def test_wrapper_fills_omitted_keys_and_flags_risk_fields() -> None:
    """extractor가 키를 빠뜨려도 위험 필드는 확인 필요가 된다 (조용한 ok 방지)."""
    fields = FlaggingExtractor(NullExtractor()).extract("아무 말")
    assert [f.key for f in fields] == list(FieldKey)
    by_key = {f.key: f.flag for f in fields}
    assert by_key[FieldKey.BLOOD_PRESSURE] is Flag.NEEDS_CONFIRM
    assert by_key[FieldKey.MOOD] is Flag.OK


def test_wrapper_overrides_inner_flag() -> None:
    """내부 extractor가 낸 flag는 무시하고 규칙으로 다시 계산한다."""

    class _Liar:
        def extract(self, transcript: str) -> list[PredictedField]:
            return [_f(FieldKey.FALL_EVENT, "있음", flag=Flag.OK)]

    fields = FlaggingExtractor(_Liar()).extract("x")
    assert next(f for f in fields if f.key is FieldKey.FALL_EVENT).flag is Flag.RISK


def test_wrapper_forwards_usage_summary() -> None:
    """사용량 요약은 내부 extractor 것을 전달하고, 러너 보고서에 실린다."""

    class _WithUsage:
        def extract(self, transcript: str) -> list[PredictedField]:
            return []

        def usage_summary(self) -> dict[str, object]:
            return {"calls": 3}

    cases = load_golden(GOLDEN_DIR)[:1]
    report = evaluate(FlaggingExtractor(_WithUsage()), cases)
    assert report.usage == {"calls": 3}


# 정답 값을 Flagger에 넣었을 때 정답 flag와 다른 (케이스, 필드). 규칙의 한계를 문서화하는 목록이다.
# 이 목록이 줄어드는 것이 Rule Engine·Sentinel·Verifier의 효과이고, 늘어나면 규칙이 퇴행한 것이다.
KNOWN_GAPS: dict[tuple[str, FieldKey], str] = {
    ("gs-013", FieldKey.FALL_EVENT): "미끄러질 뻔함: 의미 판단이 필요 (Verifier/Sentinel)",
    ("gs-014", FieldKey.BLOOD_PRESSURE): "정답은 reference 값(130/80). 오인식 값(13/80)에서 잡힌다",
    ("gs-015", FieldKey.BLOOD_PRESSURE): "정답은 reference 값(140/90). 오인식 값(40/90)에서 잡힌다",
    ("gs-017", FieldKey.BLOOD_PRESSURE): "고혈압 risk 임계값은 Rule Engine에서 확정",
    ("gs-017", FieldKey.SPECIAL_NOTES): "통증 호소의 위험 판정은 Sentinel",
    ("gs-018", FieldKey.BLOOD_PRESSURE): "고혈압 risk 임계값은 Rule Engine에서 확정",
    ("gs-018", FieldKey.SPECIAL_NOTES): "상태 변화 의심 신호는 Sentinel",
}


def test_agreement_with_gold_flags_on_gold_values() -> None:
    """정답 값으로 계산한 flag가 정답과 다른 곳은 KNOWN_GAPS와 정확히 일치한다."""
    mismatches: dict[tuple[str, FieldKey], tuple[Flag, Flag]] = {}
    for case in load_golden(GOLDEN_DIR):
        for gold in case.fields:
            predicted = PredictedField(
                key=gold.key,
                value=gold.value,
                source=gold.source,
                candidates=gold.candidates,
            )
            got = flag_field(predicted)
            if got is not gold.flag:
                mismatches[(case.id, gold.key)] = (got, gold.flag)
    assert set(mismatches) == set(KNOWN_GAPS), mismatches


def test_stt_error_cases_are_caught_on_transcript_values() -> None:
    """STT 오류 케이스는 transcript에 보이는 오인식 값(13/80, 40/90)에서 확인 필요가 된다."""
    assert flag_field(_f(FieldKey.BLOOD_PRESSURE, "13/80")) is Flag.NEEDS_CONFIRM
    assert flag_field(_f(FieldKey.BLOOD_PRESSURE, "40/90")) is Flag.NEEDS_CONFIRM
