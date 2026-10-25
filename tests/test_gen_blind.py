"""블라인드 검수 시트와 라벨 비교 도구 테스트."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import GOOD_D1, GOOD_H3, SPEC_DIR

from carenote.generation.__main__ import main
from carenote.generation.candidates import (
    BLIND_AGREEMENT_TARGET,
    blind_ids,
    blind_label_template,
    compare_blind,
    format_blind_report,
    render_blind_sheet,
)
from carenote.generation.spec import Spec


@pytest.fixture
def cand_dir(tmp_path: Path) -> Path:
    """D1·H3 후보가 든 폴더 (키 파일은 상위 폴더에 놓인다)."""
    out = tmp_path / "candidates" / "v2"
    out.mkdir(parents=True)
    for sid, case in (("D1", GOOD_D1), ("H3", GOOD_H3)):
        doc: dict[str, Any] = {
            "spec_id": sid,
            "transcript": case.transcript,
            "checks": [],
            "batch_checks": [],
        }
        (out / f"{sid}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return out


def _perfect_labels(key: dict[str, str], specs: list[Spec]) -> dict[str, Any]:
    """정답을 그대로 옮긴 라벨 (일치율 100% 확인용)."""
    by_id = {s.spec_id: s for s in specs}
    labels = blind_label_template(key)
    for sid, blind in key.items():
        for fld, gold in by_id[sid].gold.items():
            labels["cases"][blind][fld.value] = {
                "value": gold.value or "",
                "source": gold.source.value,
                "flag": gold.flag.value,
                "candidates": list(gold.candidates),
                "memo": "",
            }
    return labels


def test_blind_ids_are_deterministic_anonymous_and_shuffled() -> None:
    """B01.. 형태로 가려지고 입력 순서와 상관없이 같은 매핑이며, 시드가 바뀌면 순서도 바뀐다."""
    ids = ["C2", "D1", "H3", "I1", "N1", "R1", "S1"]
    key = blind_ids(ids)
    assert key == blind_ids(list(reversed(ids)))
    assert sorted(key.values()) == [f"B0{i}" for i in range(1, 8)]
    assert len({tuple(blind_ids(ids, seed=s).items()) for s in range(5)}) > 1


def test_blind_sheet_hides_gold_title_and_probe(cand_dir: Path, all_specs: list[Spec]) -> None:
    """시트에는 transcript와 안내만 있고 스펙 제목·프로브·위험도·정답 표가 없다."""
    key = blind_ids(["D1", "H3"])
    sheet = render_blind_sheet(cand_dir, key)
    by_id = {s.spec_id: s for s in all_specs}
    assert GOOD_D1.transcript in sheet and GOOD_H3.transcript in sheet
    headings = [line for line in sheet.splitlines() if line.startswith("## B")]
    assert sorted(headings) == ["## B01", "## B02"]
    for sid in ("D1", "H3"):
        assert by_id[sid].title not in sheet
    for hidden in ("label_risk", "프로브", "gray_fields", "| 정답 |", "(없음)"):
        assert hidden not in sheet
    assert "docs/label-policy.md" in sheet


def test_template_has_all_cases_and_eight_empty_fields() -> None:
    """빈 라벨 파일은 케이스마다 8개 필드를 비워 둔다."""
    key = blind_ids(["D1", "H3"])
    template = blind_label_template(key)
    assert set(template["cases"]) == {"B01", "B02"}
    for fields in template["cases"].values():
        assert len(fields) == 8 and all(
            f["value"] == "" and f["flag"] == "" for f in fields.values()
        )


def test_compare_perfect_labels_scores_100_percent(all_specs: list[Spec]) -> None:
    """정답과 같은 라벨은 값·flag 일치율 100%이고 불일치가 없다."""
    key = blind_ids(["D1", "H3"])
    report = compare_blind(_perfect_labels(key, all_specs), key, all_specs)
    assert report.total == 16 and report.value_rate == 1.0 and report.flag_rate == 1.0
    assert report.disagreements == [] and report.unlabeled == 0
    assert "달성" in format_blind_report(report)


def test_compare_counts_disagreements_with_target_and_unlabeled(all_specs: list[Spec]) -> None:
    """틀린 값·flag를 세고, 95% 미만이면 미달, 비워 둔 필드는 미작성으로 센다."""
    key = blind_ids(["D1", "H3"])
    labels = _perfect_labels(key, all_specs)
    d1, h3 = key["D1"], key["H3"]
    labels["cases"][d1]["blood_pressure"]["value"] = "128/82"  # 값 불일치
    labels["cases"][h3]["medication_taken"]["flag"] = "ok"  # flag 불일치(정답은 needs_confirm)
    labels["cases"][h3]["fall_event"] = {
        "value": "", "source": "", "flag": "", "candidates": [], "memo": ""
    }  # fmt: skip
    report = compare_blind(labels, key, all_specs)
    assert report.total == 16 and report.unlabeled == 1
    assert report.value_agree == 14 and report.flag_agree == 14
    assert (report.value_rate or 0) < BLIND_AGREEMENT_TARGET
    text = format_blind_report(report)
    assert "미달" in text and "blood_pressure" in text and "medication_taken" in text


def test_compare_handles_candidates_and_special_notes_presence(all_specs: list[Spec]) -> None:
    """후보가 있는 정답은 후보 집합이 같아야 하고, 특이사항은 값 유무만 비교한다."""
    key = blind_ids(["C3", "L1"])
    labels = _perfect_labels(key, all_specs)
    c3, l1 = key["C3"], key["L1"]
    labels["cases"][c3]["blood_pressure"]["candidates"] = ["130/80", "120/80"]  # 순서 다름: 일치
    labels["cases"][l1]["special_notes"]["value"] = "무릎이 아프다고 함"  # 표현 다름: 일치
    report = compare_blind(labels, key, all_specs)
    assert report.value_rate == 1.0
    labels["cases"][c3]["blood_pressure"]["candidates"] = ["130/80"]  # 후보 하나 빠짐: 불일치
    assert compare_blind(labels, key, all_specs).value_agree == 15


def test_gray_fields_are_reported_separately(all_specs: list[Spec]) -> None:
    """회색 필드(C2 special_notes)의 불일치는 따로 집계되고 제외 일치율에 영향이 없다."""
    key = blind_ids(["C2"])
    labels = _perfect_labels(key, all_specs)
    labels["cases"][key["C2"]]["special_notes"]["value"] = ""  # 라벨러는 특이사항을 안 적음
    report = compare_blind(labels, key, all_specs)
    assert report.value_rate == pytest.approx(7 / 8)
    assert report.value_rate_excluding_gray == 1.0
    assert "[gray]" in "\n".join(report.disagreements)


def test_cli_renders_blind_sheet_and_compares(
    cand_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--blind가 시트·라벨 파일·키를 만들고 --compare-blind가 비교한다(API 없음)."""
    sheet = cand_dir.parent / "blind.md"
    argv = ["--spec-dir", str(SPEC_DIR), "--out-dir", str(cand_dir)]
    assert main([*argv, "--render-review", str(sheet), "--blind"]) == 0
    template = sheet.with_suffix(".labels.json")
    key_path = cand_dir.parent / "blind_key.json"
    assert sheet.exists() and template.exists() and key_path.exists()
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert set(key) == {"D1", "H3"}
    assert "D1" not in sheet.read_text(encoding="utf-8").replace("GOOD", "")
    capsys.readouterr()
    assert main([*argv, "--compare-blind", str(template)]) == 0
    out = capsys.readouterr().out
    assert "미작성 16개" in out and "미달" in out  # 비어 있는 템플릿은 전부 미작성
