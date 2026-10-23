"""Extractor 인터페이스. 평가기는 이 경계만 안다.

extractor는 transcript만 받는다. 정답(GoldenCase)을 볼 수 없게 해서 평가 누수를 막는다.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from carenote.evaluation.schema import Evidence, FieldKey, Flag, GoldenCase, Source


class PredictedField(BaseModel):
    """extractor가 낸 필드. 모델 출력은 일관성을 보장할 수 없으므로 검증을 느슨하게 둔다."""

    model_config = ConfigDict(frozen=True)

    key: FieldKey
    value: str | None = None
    source: Source = Source.MISSING
    evidence: tuple[Evidence, ...] = ()
    flag: Flag = Flag.OK


class Extractor(Protocol):
    """transcript에서 필드를 추출한다. 빠뜨린 키는 missing으로 취급된다."""

    def extract(self, transcript: str) -> list[PredictedField]:
        """필드 목록을 반환한다."""
        ...


class NullExtractor:
    """아무것도 추출하지 않는다. recall 하한(0)을 확인하는 기준선."""

    def extract(self, transcript: str) -> list[PredictedField]:
        """빈 목록을 반환한다."""
        return []


class OracleExtractor:
    """정답을 그대로 돌려준다. 평가기가 만점을 내는지 확인하는 기준선.

    transcript로 정답을 찾으므로 골든셋 안에서 transcript가 유일해야 한다.
    """

    def __init__(self, cases: list[GoldenCase]) -> None:
        self._by_transcript = {c.transcript: c for c in cases}

    def extract(self, transcript: str) -> list[PredictedField]:
        """해당 케이스의 정답 필드를 반환한다."""
        case = self._by_transcript[transcript]
        return [
            PredictedField(
                key=f.key, value=f.value, source=f.source, evidence=f.evidence, flag=f.flag
            )
            for f in case.fields
        ]
