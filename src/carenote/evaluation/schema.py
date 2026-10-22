"""골든셋 스키마 (방문기록 1종).

한 케이스는 합성 transcript와 그에 대한 정답 필드 전체를 담는다. 모든 필드를 항상
나열하므로 "빠진 필드"와 "정답이 missing인 필드"가 구분된다.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FieldKey(StrEnum):
    """방문기록 필드."""

    VISIT_START = "visit_start"
    BLOOD_PRESSURE = "blood_pressure"
    PULSE = "pulse"
    MEAL_INTAKE = "meal_intake"
    MEDICATION_TAKEN = "medication_taken"
    FALL_EVENT = "fall_event"
    MOOD = "mood"
    SPECIAL_NOTES = "special_notes"


# 위험 필드: 수치·복약·낙상. 비싼 검증(Verifier, Re-listener)을 거는 대상이다.
RISK_FIELDS: frozenset[FieldKey] = frozenset(
    {FieldKey.BLOOD_PRESSURE, FieldKey.PULSE, FieldKey.MEDICATION_TAKEN, FieldKey.FALL_EVENT}
)


class Category(StrEnum):
    """케이스 유형. 평가 결과를 유형별로 나눠 본다."""

    NORMAL = "normal"
    MISSING = "missing"
    AMBIGUOUS_NUMBER = "ambiguous_number"
    DIALECT = "dialect"
    STT_ERROR = "stt_error"
    RISK_COMBO = "risk_combo"
    INJECTION = "injection"


class Source(StrEnum):
    """값의 출처. 필드 계약(blueprint 4.2)과 같다."""

    STATED = "stated"
    INFERRED = "inferred"
    MISSING = "missing"


class Flag(StrEnum):
    """검수자에게 보이는 최종 플래그."""

    OK = "ok"
    NEEDS_CONFIRM = "needs_confirm"
    RISK = "risk"


# 개인정보처럼 보이는 패턴. 골든셋은 합성만 허용하므로 걸리면 거부한다.
_PII_PATTERNS = (
    re.compile(r"\d{6}-?[1-4]\d{6}"),  # 주민등록번호
    re.compile(r"01[016789]-?\d{3,4}-?\d{4}"),  # 휴대전화
)


class Evidence(BaseModel):
    """transcript 안의 근거 구간. 원문 그대로의 부분 문자열이어야 한다."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)


class ExpectedField(BaseModel):
    """한 필드의 정답."""

    model_config = ConfigDict(frozen=True)

    key: FieldKey
    value: str | None
    source: Source
    evidence: tuple[Evidence, ...] = ()
    flag: Flag
    # 값이 모호할 때 허용되는 해석들. 이때 value는 None이고 flag는 needs_confirm이다.
    candidates: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_consistency(self) -> ExpectedField:
        if self.source is Source.MISSING:
            if self.value is not None or self.evidence or self.candidates:
                raise ValueError(f"{self.key}: missing은 값·근거·후보가 없어야 한다")
            if self.key in RISK_FIELDS and self.flag is not Flag.NEEDS_CONFIRM:
                raise ValueError(f"{self.key}: 위험 필드가 missing이면 needs_confirm이어야 한다")
            return self
        if not self.evidence:
            raise ValueError(f"{self.key}: 값이 있으면 evidence가 필요하다")
        if self.candidates:
            if self.value is not None or self.flag is not Flag.NEEDS_CONFIRM:
                raise ValueError(f"{self.key}: 모호한 값은 value=None, needs_confirm이어야 한다")
            if len(self.candidates) < 2:
                raise ValueError(f"{self.key}: candidates는 2개 이상이어야 한다")
        elif self.value is None:
            raise ValueError(f"{self.key}: value가 None이면 missing이거나 candidates가 있어야 한다")
        return self


class GoldenCase(BaseModel):
    """골든셋 케이스 1건."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^gs-\d{3}$")
    category: Category
    synthetic: Literal[True]
    transcript: str = Field(min_length=1)
    # stt_error 케이스에서 실제로 발화된 문장. 그 외에는 None.
    reference_transcript: str | None = None
    injection_attempt: bool = False
    fields: tuple[ExpectedField, ...]
    notes: str = ""

    @model_validator(mode="after")
    def _check_case(self) -> GoldenCase:
        for pattern in _PII_PATTERNS:
            if pattern.search(self.transcript):
                raise ValueError(f"{self.id}: 개인정보 패턴이 transcript에 있다")
        keys = [f.key for f in self.fields]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{self.id}: 필드 키가 중복된다")
        if set(keys) != set(FieldKey):
            missing = sorted(set(FieldKey) - set(keys))
            raise ValueError(f"{self.id}: 모든 필드를 나열해야 한다 (빠짐: {missing})")
        for f in self.fields:
            for ev in f.evidence:
                if ev.text not in self.transcript:
                    raise ValueError(f"{self.id}/{f.key}: 근거가 transcript에 없다: {ev.text!r}")
        if (self.category is Category.STT_ERROR) != (self.reference_transcript is not None):
            raise ValueError(f"{self.id}: reference_transcript는 stt_error 케이스에만 둔다")
        if (self.category is Category.INJECTION) != self.injection_attempt:
            raise ValueError(f"{self.id}: injection_attempt는 injection 케이스와 일치해야 한다")
        return self

    def field(self, key: FieldKey) -> ExpectedField:
        """키로 정답 필드를 찾는다."""
        return next(f for f in self.fields if f.key is key)
