"""골든셋 v2 스펙 검증: 정답이 스키마·정책 규칙을 어기지 않는지, 분할이 유효한지 확인한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from carenote.evaluation.schema import RISK_FIELDS, Evidence, ExpectedField, FieldKey, Flag, Source

SPEC_DIR = Path(__file__).parent.parent / "data" / "golden" / "specs"
SPEC_FILES = sorted(SPEC_DIR.glob("probe_*.json"))


def _specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for path in SPEC_FILES:
        specs.extend(json.loads(path.read_text(encoding="utf-8"))["specs"])
    return specs


def test_spec_files_exist_and_ids_unique() -> None:
    """스펙이 있고 spec_id가 전역에서 유일하다."""
    ids = [str(s["spec_id"]) for s in _specs()]
    assert ids and len(ids) == len(set(ids))


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: str(s["spec_id"]))
def test_spec_gold_follows_schema_rules(spec: dict[str, Any]) -> None:
    """정답 필드가 8개 모두 있고 ExpectedField 일관성 규칙(누락·후보·근거)을 통과한다."""
    gold = cast(dict[str, dict[str, Any]], spec["gold"])
    assert set(gold) == {k.value for k in FieldKey}
    for key, g in gold.items():
        has_content = g["value"] is not None or bool(g["candidates"])
        # evidence는 생성 후 채워지므로 값이 있으면 자리표시 근거로 규칙만 검사한다
        ExpectedField(
            key=FieldKey(key),
            value=g["value"],
            source=Source(g["source"]),
            evidence=(Evidence(text="x"),) if has_content else (),
            flag=Flag(g["flag"]),
            candidates=tuple(g["candidates"]),
        )


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: str(s["spec_id"]))
def test_spec_metadata_is_valid(spec: dict[str, Any]) -> None:
    """분할·라벨 위험도·길이 범위가 유효하다."""
    assert spec["split"] in ("dev", "test")
    assert spec["label_risk"] in ("low", "medium", "high")
    low, high = cast(list[int], spec["length_chars"])
    assert 0 < low < high
    assert spec["traps"] and spec["hypothesis"] and spec["policy"]


def test_missing_risk_fields_in_specs_need_confirm() -> None:
    """정책 #1·#2: 위험 필드가 missing인 정답은 모두 needs_confirm이다."""
    for spec in _specs():
        gold = cast(dict[str, dict[str, Any]], spec["gold"])
        for key in RISK_FIELDS:
            g = gold[key.value]
            if g["source"] == "missing":
                assert g["flag"] == "needs_confirm", (spec["spec_id"], key)
