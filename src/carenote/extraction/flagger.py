"""최소 Flagger: 추출된 필드에 flag를 붙이는 결정적 규칙.

3주차 Rule Engine(YAML)이 대체하기 전까지의 임시 규칙이다. 수치 범위는 임상 기준이 아니라
STT 오인식을 걸러내기 위한 "생리적으로 불가능한 범위"다. 고혈압 등 임상 risk 기준은 포함하지 않는다.
"""

from __future__ import annotations

import re

from carenote.evaluation.extractor import Extractor, PredictedField, SupportsUsage
from carenote.evaluation.schema import RISK_FIELDS, FieldKey, Flag, Source

RISK_MEDICATION = frozenset({"미복용", "중복 복용 의심"})
RISK_FALL = frozenset({"있음"})

# (최솟값, 최댓값) 이 범위를 벗어나면 측정값이 아니라 오인식일 가능성이 크다.
SYSTOLIC_RANGE = (70, 260)
DIASTOLIC_RANGE = (30, 160)
PULSE_RANGE = (30, 220)

_BP_PATTERN = re.compile(r"^(\d{2,3})/(\d{2,3})$")
_PULSE_PATTERN = re.compile(r"^\d{2,3}$")


def _within(value: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _blood_pressure_plausible(value: str) -> bool:
    match = _BP_PATTERN.match(value.strip())
    if match is None:
        return False
    systolic, diastolic = int(match.group(1)), int(match.group(2))
    return (
        _within(systolic, SYSTOLIC_RANGE)
        and _within(diastolic, DIASTOLIC_RANGE)
        and systolic > diastolic
    )


def _pulse_plausible(value: str) -> bool:
    text = value.strip()
    return _PULSE_PATTERN.match(text) is not None and _within(int(text), PULSE_RANGE)


def flag_field(field: PredictedField) -> Flag:
    """필드 하나의 flag를 정한다. 우선순위: 확인 필요(누락·모호·불가능 수치) → risk → ok."""
    if len(field.candidates) >= 2:
        return Flag.NEEDS_CONFIRM
    if field.source is Source.MISSING or field.value is None:
        return Flag.NEEDS_CONFIRM if field.key in RISK_FIELDS else Flag.OK

    value = field.value
    if field.key is FieldKey.BLOOD_PRESSURE and not _blood_pressure_plausible(value):
        return Flag.NEEDS_CONFIRM
    if field.key is FieldKey.PULSE and not _pulse_plausible(value):
        return Flag.NEEDS_CONFIRM
    if field.key is FieldKey.MEDICATION_TAKEN and value in RISK_MEDICATION:
        return Flag.RISK
    if field.key is FieldKey.FALL_EVENT and value in RISK_FALL:
        return Flag.RISK
    return Flag.OK


class FlaggingExtractor:
    """내부 extractor의 결과에 flag를 붙인다. 빠진 키는 missing으로 채워 flag를 계산한다.

    키를 빠뜨린 위험 필드가 조용히 ok가 되는 일을 막는 것이 이 클래스의 핵심 역할이다.
    """

    def __init__(self, inner: Extractor) -> None:
        self._inner = inner

    def extract(self, transcript: str) -> list[PredictedField]:
        """추출 후 모든 키에 대해 flag를 계산해 반환한다."""
        by_key = {f.key: f for f in self._inner.extract(transcript)}
        result: list[PredictedField] = []
        for key in FieldKey:
            field = by_key.get(key, PredictedField(key=key))
            result.append(field.model_copy(update={"flag": flag_field(field)}))
        return result

    def usage_summary(self) -> dict[str, object]:
        """내부 extractor의 사용량을 그대로 전달한다 (없으면 빈 요약)."""
        if isinstance(self._inner, SupportsUsage):
            return self._inner.usage_summary()
        return {}
