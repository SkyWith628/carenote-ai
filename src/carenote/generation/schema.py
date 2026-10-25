"""생성기 구조화 출력 스키마와 후보 모델."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from carenote.evaluation.schema import FieldKey


class FieldEvidence(BaseModel):
    """값이 있는 필드마다 transcript에서 그대로 복사한 인용."""

    key: FieldKey
    quotes: list[str]


class GeneratedCase(BaseModel):
    """생성기 출력 (구조화 출력 스키마)."""

    transcript: str
    reference_transcript: str | None
    evidence: list[FieldEvidence]
    notes: str


class CheckResult(BaseModel):
    """자동 점검 한 건의 결과."""

    model_config = ConfigDict(frozen=True)

    code: str
    level: Literal["pass", "warn", "fail"]
    message: str = ""
