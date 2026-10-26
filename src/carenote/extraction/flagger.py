"""최소 Flagger: 추출된 필드에 flag를 붙이는 결정적 규칙.

3주차 Rule Engine(YAML)이 대체하기 전까지의 임시 규칙이다. 수치 범위는 임상 기준이 아니라
STT 오인식을 걸러내기 위한 "생리적으로 불가능한 범위"다. 고혈압 등 임상 risk 기준은 포함하지 않는다.
"""

from __future__ import annotations

import re

from carenote.evaluation.extractor import Extractor, PredictedField, SupportsUsage
from carenote.evaluation.schema import RISK_FIELDS, FieldKey, Flag, Source
from carenote.generation.numerals import extract_numbers, parse_sino

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


# KR-1(docs/known-risks.md, docs/flagger-kr1-design.md): 모델이 번복이 풀리지 않은 수치를 확정하거나
# transcript에 없는 숫자를 만들어도 flag가 ok로 나가는 조용한 오답을 막는 결정적 규칙.
# 규칙은 flag를 올리기만 한다. 값은 바꾸지 않는다.
NUMERIC_KEYS: frozenset[FieldKey] = frozenset({FieldKey.BLOOD_PRESSURE, FieldKey.PULSE})
_KEYWORDS = {FieldKey.BLOOD_PRESSURE: "혈압", FieldKey.PULSE: "맥박"}
# 수사(아라비아 숫자 또는 한자어 수사 연속) + 선택적 '이' + 의문형 어미
_HEDGE_RE = re.compile(
    r"(\d+|[영공일이삼사오육칠팔구십백천]+)\s?이?(?:었나|였나|인가|던가|일까|인지)"
)
_SENTENCE_BREAK = re.compile(r"[.!?]")
_KEYWORD_WINDOW = 40
_PLAUSIBLE = (30, 260)  # 혈압·맥박 값으로 볼 수 있는 수(일상어 '일이었나'의 12 같은 읽기를 거른다)

REASON_HEDGE = "hedge_in_transcript"
REASON_FABRICATED = "number_not_in_transcript"


def _numeral_value(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return parse_sino(token)


def hedged_fields(transcript: str) -> frozenset[FieldKey]:
    """transcript에서 번복·불확실 의문형 표지가 붙은 수치의 필드(혈압/맥박).

    표지가 속한 문장에서 앞쪽의 가장 가까운 키워드 필드에 붙이고, 키워드가 없으면 둘 다에 붙인다.
    수사가 혈압·맥박으로 볼 수 없는 수(예: '일이었나'의 12)이면 건너뛴다.
    """
    found: set[FieldKey] = set()
    for match in _HEDGE_RE.finditer(transcript):
        number = _numeral_value(match.group(1))
        if number is None or not _within(number, _PLAUSIBLE):
            continue
        before = transcript[max(0, match.start() - _KEYWORD_WINDOW) : match.start()]
        before = _SENTENCE_BREAK.split(before)[-1]
        positions = {key: before.rfind(word) for key, word in _KEYWORDS.items()}
        near = [key for key, pos in positions.items() if pos >= 0]
        if near:
            found.add(max(near, key=lambda key: positions[key]))
        else:
            found |= set(NUMERIC_KEYS)
    return frozenset(found)


def _value_numbers(field: PredictedField) -> list[int] | None:
    """수치 필드 값의 숫자 부분(혈압은 둘, 맥박은 하나). 형식이 다르면 None."""
    value = (field.value or "").strip()
    if field.key is FieldKey.BLOOD_PRESSURE:
        match = _BP_PATTERN.match(value)
        return [int(match.group(1)), int(match.group(2))] if match else None
    return [int(value)] if _PULSE_PATTERN.match(value) else None


def flag_reasons(field: PredictedField, transcript: str | None) -> list[str]:
    """transcript 기반 규칙(R1·R2)의 발동 이유 코드. 단일 값을 낸 수치 필드에만 적용한다."""
    if transcript is None or field.key not in NUMERIC_KEYS:
        return []
    if field.value is None or field.source is Source.MISSING or len(field.candidates) >= 2:
        return []
    reasons: list[str] = []
    if field.key in hedged_fields(transcript):
        reasons.append(REASON_HEDGE)
    parts = _value_numbers(field)
    if parts is not None:
        readable = extract_numbers(transcript)
        if any(part not in readable for part in parts):
            reasons.append(REASON_FABRICATED)
    return reasons


def flag_field(field: PredictedField, transcript: str | None = None) -> Flag:
    """필드 하나의 flag를 정한다. 우선순위: 확인 필요(누락·모호·불가능 수치) → risk → ok.

    `transcript`를 주면 KR-1 규칙(R1 의문형 표지, R2 만들어진 숫자)이 확인 필요를 더한다.
    주지 않으면 이전과 똑같이 동작한다.
    """
    if len(field.candidates) >= 2:
        return Flag.NEEDS_CONFIRM
    if field.source is Source.MISSING or field.value is None:
        return Flag.NEEDS_CONFIRM if field.key in RISK_FIELDS else Flag.OK
    if flag_reasons(field, transcript):
        return Flag.NEEDS_CONFIRM

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
            result.append(field.model_copy(update={"flag": flag_field(field, transcript)}))
        return result

    def usage_summary(self) -> dict[str, object]:
        """내부 extractor의 사용량을 그대로 전달한다 (없으면 빈 요약)."""
        if isinstance(self._inner, SupportsUsage):
            return self._inner.usage_summary()
        return {}
