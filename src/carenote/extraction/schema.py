"""LLM 구조화 출력 스키마.

flag는 LLM이 내지 않는다. 판정은 코드(Flagger)가 한다.
구조화 출력 스키마는 단순하게 유지한다 (길이·범위 제약 없음).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from carenote.evaluation.schema import FieldKey

SourceLiteral = Literal["stated", "inferred", "missing"]


class LLMFieldB0(BaseModel):
    """B0: 근거를 요구하지 않는 필드."""

    key: FieldKey
    value: str | None
    source: SourceLiteral
    candidates: list[str]


class LLMExtractionB0(BaseModel):
    """B0 출력."""

    fields: list[LLMFieldB0]


class LLMFieldB1(BaseModel):
    """B1: 원문 인용 근거를 요구하는 필드."""

    key: FieldKey
    value: str | None
    source: SourceLiteral
    evidence: list[str]
    candidates: list[str]


class LLMExtractionB1(BaseModel):
    """B1 출력."""

    fields: list[LLMFieldB1]
