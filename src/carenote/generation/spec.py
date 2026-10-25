"""케이스 스펙 모델과 로더 (`data/golden/specs/probe_*.json`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from carenote.evaluation.schema import FieldKey, Flag, Source

Probe = Literal["H", "C", "D", "N", "L", "S", "V", "G", "I", "R"]


class GoldField(BaseModel):
    """스펙의 필드 정답. evidence는 transcript 생성 후에 채워진다."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: str | None
    source: Source
    flag: Flag
    candidates: tuple[str, ...] = ()


class SttError(BaseModel):
    """S 프로브의 오인식 한 건."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: FieldKey
    reference: str
    heard: str
    kind: str
    recoverable_from_text: bool


class Stt(BaseModel):
    """S 프로브 메타데이터."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    errors: tuple[SttError, ...]


class Spec(BaseModel):
    """케이스 스펙 1건."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec_id: str
    probe: Probe
    split: Literal["dev", "test"]
    title: str
    scene: str
    style: dict[str, str]
    length_chars: tuple[int, int]
    must_include: tuple[str, ...]
    must_not: tuple[str, ...]
    traps: str
    hypothesis: str
    policy: str
    label_risk: Literal["low", "medium", "high"]
    gray_fields: tuple[FieldKey, ...] = ()
    author: Literal["human", "llm"]
    field_review_required: bool = False
    stt: Stt | None = None
    gold: dict[FieldKey, GoldField]

    def required_evidence_keys(self) -> list[FieldKey]:
        """evidence 인용이 필요한 필드 (값이나 후보가 있는 필드)."""
        return [k for k, g in self.gold.items() if g.value is not None or g.candidates]


def load_specs(spec_dir: Path) -> list[Spec]:
    """폴더의 `probe_*.json`을 읽어 검증한다. spec_id가 중복되면 ValueError."""
    specs: list[Spec] = []
    for path in sorted(spec_dir.glob("probe_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for raw in doc["specs"]:
            try:
                specs.append(Spec.model_validate(raw))
            except ValueError as exc:
                raise ValueError(f"{path.name}/{raw.get('spec_id')}: {exc}") from exc
    ids = [s.spec_id for s in specs]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"중복된 spec_id: {duplicated}")
    return specs
