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
    assert set(spec["gray_fields"]) <= {k.value for k in FieldKey}


def test_stt_specs_carry_error_metadata_and_only_they_do() -> None:
    """S 프로브만 오인식 메타데이터(대상 필드, 원문, 복원 가능 여부)를 가진다."""
    for spec in _specs():
        errors = cast(list[dict[str, Any]], spec["stt"]["errors"]) if "stt" in spec else []
        if spec["probe"] == "S":
            assert errors, spec["spec_id"]
            for e in errors:
                assert e["field"] in {k.value for k in FieldKey}
                assert e["reference"] != e["heard"]
                assert isinstance(e["recoverable_from_text"], bool)
        else:
            assert "stt" not in spec, spec["spec_id"]


def test_each_probe_has_exactly_one_dev_spec_so_far() -> None:
    """프로브당 dev는 1건이다 (설계 6장: 신규 40건 중 dev 10건)."""
    dev_per_probe: dict[str, int] = {}
    for spec in _specs():
        if spec["split"] == "dev":
            dev_per_probe[spec["probe"]] = dev_per_probe.get(spec["probe"], 0) + 1
    assert dev_per_probe and all(n == 1 for n in dev_per_probe.values()), dev_per_probe


def test_missing_risk_fields_in_specs_need_confirm() -> None:
    """정책 #1·#2: 위험 필드가 missing인 정답은 모두 needs_confirm이다."""
    for spec in _specs():
        gold = cast(dict[str, dict[str, Any]], spec["gold"])
        for key in RISK_FIELDS:
            g = gold[key.value]
            if g["source"] == "missing":
                assert g["flag"] == "needs_confirm", (spec["spec_id"], key)


def test_spec_set_is_complete_forty_with_ten_dev_and_thirty_test() -> None:
    """설계 3·6장: 신규 40건(H6 C5 D5 N4 L4 S3 V4 G3 I3 R3), dev 10 / test 30."""
    specs = _specs()
    by_probe: dict[str, int] = {}
    for spec in specs:
        by_probe[spec["probe"]] = by_probe.get(spec["probe"], 0) + 1
    assert by_probe == {
        "H": 6, "C": 5, "D": 5, "N": 4, "L": 4, "S": 3, "V": 4, "G": 3, "I": 3, "R": 3,
    }  # fmt: skip
    assert len(specs) == 40
    assert sum(s["split"] == "dev" for s in specs) == 10
    assert sum(s["split"] == "test" for s in specs) == 30


def test_risk_signal_specs_expect_risk_notes_beyond_flagger() -> None:
    """R 프로브는 특이사항 risk를 정답으로 둔다 (현 Flagger가 못 잡는 Sentinel 대상)."""
    risk_specs = [s for s in _specs() if s["probe"] == "R"]
    assert risk_specs
    for spec in risk_specs:
        gold = cast(dict[str, dict[str, Any]], spec["gold"])
        assert gold["special_notes"]["flag"] == "risk", spec["spec_id"]


def test_authorship_matches_decision_ten_human_written() -> None:
    """생성 단계 설계 11장 결정: 직접 작성 10건은 지정 목록과 같다."""
    human = {str(s["spec_id"]) for s in _specs() if s["author"] == "human"}
    assert human == {"H1", "H2", "C1", "D3", "N4", "V1", "G2", "I2", "R2", "L1"}
    assert all(s["author"] in ("human", "llm") for s in _specs())


def test_high_label_risk_specs_require_field_review() -> None:
    """라벨 위험도 high(R3)와 방언 표기(G1)는 현장 확인 전에는 동결 대상에서 보류한다."""
    needs_review = {str(s["spec_id"]) for s in _specs() if s.get("field_review_required")}
    assert needs_review == {"R3", "G1"}
    for spec in _specs():
        if spec["label_risk"] == "high":
            assert spec.get("field_review_required") is True, spec["spec_id"]


def test_c2_special_notes_is_a_gray_field() -> None:
    """C2의 '첫 혈압 측정이 높았음'은 transcript에 '높다'는 표현이 없을 수 있어 회색 필드다."""
    c2 = next(s for s in _specs() if s["spec_id"] == "C2")
    assert c2["gray_fields"] == ["special_notes"]


def test_label_policy_document_keeps_the_2026_10_26_clarifications() -> None:
    """블라인드 라벨링(83.9%) 후 보완한 정책 문구가 문서에서 사라지지 않는다."""
    text = (Path(__file__).parent.parent / "docs" / "label-policy.md").read_text(encoding="utf-8")
    for fragment in (
        '사흘 전에 넘어지셨다"가 함께 나오면 `있음`',  # #7 예시
        "추측한 값(130)도 value에 넣지 않고",  # #1
        "숟가락 두세 번·서너 번",  # #9
        "복약을 거부했다는 보고는",  # #13
        "특이사항 risk 기준",
    ):
        assert fragment in text, fragment
