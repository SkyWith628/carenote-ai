"""골든셋 스키마와 로더 테스트."""

from pathlib import Path
from typing import cast

import pytest

from carenote.evaluation.loader import load_golden
from carenote.evaluation.schema import (
    Category,
    FieldKey,
    Flag,
    GoldenCase,
)

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "golden"


def _case_dict(**overrides: object) -> dict[str, object]:
    """유효한 최소 케이스를 만들고 일부를 덮어쓴다."""
    base: dict[str, object] = load_golden(GOLDEN_DIR)[0].model_dump(mode="json")
    base.update(overrides)
    return base


def test_seed_cases_are_valid() -> None:
    """시드 케이스는 모두 스키마를 통과하고 id가 유일하다."""
    cases = load_golden(GOLDEN_DIR)
    assert len(cases) >= 4
    assert {c.category for c in cases} >= {
        Category.NORMAL,
        Category.MISSING,
        Category.AMBIGUOUS_NUMBER,
        Category.INJECTION,
    }


def test_ambiguous_case_asks_instead_of_guessing() -> None:
    """모호한 수치는 value=None, 후보 2개, needs_confirm이다."""
    case = next(c for c in load_golden(GOLDEN_DIR) if c.category is Category.AMBIGUOUS_NUMBER)
    bp = case.field(FieldKey.BLOOD_PRESSURE)
    assert bp.value is None
    assert bp.flag is Flag.NEEDS_CONFIRM
    assert bp.candidates == ("130/80", "140/80")


def test_evidence_must_be_substring_of_transcript() -> None:
    """transcript에 없는 근거는 거부한다."""
    data = _case_dict(transcript="전혀 다른 문장이다.")
    with pytest.raises(ValueError, match="근거가 transcript에 없다"):
        GoldenCase.model_validate(data)


def test_all_fields_must_be_listed() -> None:
    """필드를 하나라도 빼면 거부한다."""
    data = _case_dict()
    fields = data["fields"]
    assert isinstance(fields, list)
    data["fields"] = fields[:-1]
    with pytest.raises(ValueError, match="모든 필드를 나열"):
        GoldenCase.model_validate(data)


def test_pii_pattern_is_rejected() -> None:
    """전화번호 패턴이 있으면 거부한다."""
    data = _case_dict(transcript="연락처는 010-1234-5678 이에요.")
    with pytest.raises(ValueError, match="개인정보 패턴"):
        GoldenCase.model_validate(data)


def test_missing_risk_field_must_need_confirm() -> None:
    """위험 필드가 missing인데 flag가 ok면 거부한다."""
    data = _case_dict()
    fields = cast(list[dict[str, object]], data["fields"])
    for f in fields:
        if f["key"] == "blood_pressure":
            f.update(value=None, source="missing", evidence=[], candidates=[], flag="ok")
    with pytest.raises(ValueError, match="needs_confirm"):
        GoldenCase.model_validate(data)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    """같은 id가 두 번 나오면 로더가 거부한다."""
    line = (GOLDEN_DIR / "v1.jsonl").read_text(encoding="utf-8").splitlines()[0]
    (tmp_path / "dup.jsonl").write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="중복된 케이스 id"):
        load_golden(tmp_path)


def test_golden_set_has_twenty_cases_covering_all_categories() -> None:
    """20건이고 모든 유형이 최소 2건씩이다 (injection 제외 유형 포함)."""
    cases = load_golden(GOLDEN_DIR)
    assert len(cases) == 20
    counts = {cat: sum(c.category is cat for c in cases) for cat in Category}
    assert all(n >= 2 for n in counts.values()), counts


def test_stt_error_cases_keep_reference_transcript() -> None:
    """stt_error 케이스는 실제 발화(reference)와 오인식 transcript가 다르다."""
    cases = [c for c in load_golden(GOLDEN_DIR) if c.category is Category.STT_ERROR]
    assert cases
    assert all(c.reference_transcript != c.transcript for c in cases)


def test_every_risk_field_appears_as_risk_or_confirm_somewhere() -> None:
    """위험·확인 필요 정답이 골든셋에 충분히 있어 recall을 잴 수 있다."""
    flags = [f.flag for c in load_golden(GOLDEN_DIR) for f in c.fields]
    assert flags.count(Flag.RISK) >= 8
    assert flags.count(Flag.NEEDS_CONFIRM) >= 12
