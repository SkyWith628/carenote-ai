"""골든셋 v2 (승격된 40건)의 구성과 승격·분할 규칙."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from carenote.evaluation.loader import load_golden
from carenote.evaluation.schema import Category, FieldKey, GoldenCase
from carenote.generation.promote import PromotionError, assign_ids, build_case, promote
from carenote.generation.spec import Spec, load_specs

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "golden"
SPEC_DIR = GOLDEN_DIR / "specs"
HUMAN_IDS = {"H1", "H2", "C1", "D3", "N4", "V1", "G2", "I2", "R2", "L1"}


@pytest.fixture(scope="module")
def v2() -> list[GoldenCase]:
    return [c for c in load_golden(GOLDEN_DIR) if c.spec_id is not None]


def test_v2_has_forty_cases_split_ten_thirty(v2: list[GoldenCase]) -> None:
    assert len(v2) == 40
    assert Counter(c.split for c in v2) == {"dev": 10, "test": 30}


def test_total_is_sixty_with_dev_thirty_test_thirty() -> None:
    assert len(load_golden(GOLDEN_DIR)) == 60
    assert len(load_golden(GOLDEN_DIR, "dev")) == 30
    assert len(load_golden(GOLDEN_DIR, "test")) == 30


def test_v1_cases_are_all_dev() -> None:
    v1 = load_golden(GOLDEN_DIR / "v1.jsonl")
    assert all(c.split == "dev" for c in v1)


def test_probe_distribution_matches_design(v2: list[GoldenCase]) -> None:
    probes = Counter(c.tags[0] for c in v2)
    assert probes == {
        "H": 6, "C": 5, "D": 5, "N": 4, "L": 4, "S": 3, "V": 4, "G": 3, "I": 3, "R": 3,
    }  # fmt: skip


def test_ids_are_contiguous_and_stable(v2: list[GoldenCase]) -> None:
    ids = sorted(c.id for c in v2)
    assert ids == [f"gs-{n:03d}" for n in range(21, 61)]
    specs = load_specs(SPEC_DIR)
    assert {c.spec_id: c.id for c in v2} == assign_ids(specs)


def test_ten_human_authored_cases(v2: list[GoldenCase]) -> None:
    assert {c.spec_id for c in v2 if c.author == "human"} == HUMAN_IDS


def test_gold_comes_from_specs(v2: list[GoldenCase]) -> None:
    specs = {s.spec_id: s for s in load_specs(SPEC_DIR)}
    for case in v2:
        spec = specs[case.spec_id or ""]
        for key in FieldKey:
            got, want = case.field(key), spec.gold[key]
            assert (got.value, got.source, got.flag, got.candidates) == (
                want.value, want.source, want.flag, want.candidates,
            ), (case.id, key)  # fmt: skip


def test_review_required_cases_are_r3_and_g1(v2: list[GoldenCase]) -> None:
    assert {c.spec_id for c in v2 if c.review_required} == {"R3", "G1"}


def test_unrecoverable_fields_come_only_from_stt_metadata(v2: list[GoldenCase]) -> None:
    flagged = {(c.spec_id, f.key.value) for c in v2 for f in c.fields if not f.recoverable}
    assert flagged == {("S2", "medication_taken")}


def test_stt_and_injection_categories_are_consistent(v2: list[GoldenCase]) -> None:
    for c in v2:
        assert (c.category is Category.STT_ERROR) == (c.tags[0] == "S")
        assert c.injection_attempt == (c.tags[0] == "I")
    assert next(c for c in v2 if c.spec_id == "H2").category is Category.BLANKET


def test_promote_is_deterministic(tmp_path: Path) -> None:
    specs = load_specs(SPEC_DIR)
    promote(specs, GOLDEN_DIR / "candidates" / "v2", GOLDEN_DIR / "human", tmp_path)
    for split in ("dev", "test"):
        assert (tmp_path / f"v2_{split}.jsonl").read_text(encoding="utf-8") == (
            GOLDEN_DIR / f"v2_{split}.jsonl"
        ).read_text(encoding="utf-8")


def test_promote_rejects_candidate_with_failed_check(tmp_path: Path) -> None:
    cand = tmp_path / "cand"
    cand.mkdir()
    for p in (GOLDEN_DIR / "candidates" / "v2").glob("*.json"):
        (cand / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    doc = json.loads((cand / "H3.json").read_text(encoding="utf-8"))
    doc["checks"][0]["level"] = "fail"
    (cand / "H3.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(PromotionError, match="H3"):
        promote(load_specs(SPEC_DIR), cand, GOLDEN_DIR / "human", out)
    assert list(out.iterdir()) == []  # 실패하면 아무것도 쓰지 않는다


def test_build_case_rejects_quote_not_in_transcript() -> None:
    from carenote.generation.promote import Source

    spec: Spec = next(s for s in load_specs(SPEC_DIR) if s.spec_id == "V1")
    evidence = {k: ["없는 문장"] for k in spec.required_evidence_keys()}
    with pytest.raises(PromotionError, match="V1"):
        build_case(
            spec,
            Source("오늘 오전 11시에 방문했어요", None, {k.value: v for k, v in evidence.items()}),
            "gs-099",
        )


def test_human_transcripts_pass_length_after_h1_change() -> None:
    specs = {s.spec_id: s for s in load_specs(SPEC_DIR)}
    cases = {c.spec_id: c for c in load_golden(GOLDEN_DIR, "all") if c.spec_id}
    for sid in HUMAN_IDS:
        low, high = specs[sid].length_chars
        assert low <= len(cases[sid].transcript) <= high, sid


def test_batch_checks_ignore_the_promoted_copy_of_each_candidate(tmp_path: Path) -> None:
    """승격본이 있어도 후보가 자기 자신과 중복(유사도 1.00)으로 실패하지 않는다."""
    from carenote.generation.candidates import apply_batch_checks

    cand = tmp_path / "cand"
    cand.mkdir()
    for p in (GOLDEN_DIR / "candidates" / "v2").glob("*.json"):
        (cand / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    summary = apply_batch_checks(cand, GOLDEN_DIR)
    assert not [m for msgs in summary.values() for m in msgs if "[H] fail" in m]


def test_replaced_s3_and_h6_follow_the_replacement_design(v2: list[GoldenCase]) -> None:
    by = {c.spec_id: c for c in v2}
    s3, h6 = by["S3"], by["H6"]
    assert s3.reference_transcript is not None and s3.reference_transcript != s3.transcript
    for text in ("너머진 이른", "드셔써요", "약또"):
        assert text in s3.transcript and text not in s3.reference_transcript
    assert all(f.recoverable for f in s3.fields)
    assert s3.field(FieldKey.MEDICATION_TAKEN).value == "복용"
    assert h6.field(FieldKey.MEDICATION_TAKEN).value is None
    assert h6.field(FieldKey.FALL_EVENT).value is None
    assert "약" not in h6.transcript and "넘어" not in h6.transcript
    retired = GOLDEN_DIR / "retired" / "specs" / "S3_H6_retired_2026-10-26.json"
    assert retired.exists()
