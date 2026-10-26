"""동결 도구와 (동결된 경우) 실제 골든셋의 무결성."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from carenote.evaluation import freeze
from carenote.evaluation.freeze import (
    MANIFEST_PATH,
    build_manifest,
    check,
    collect_hashes,
    load_manifest,
    write_manifest,
)
from carenote.evaluation.freeze import main as freeze_main

REPO = Path(__file__).parent.parent


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """동결 대상 파일만 복사한 임시 프로젝트 트리."""
    for pattern in freeze.FROZEN_GLOBS:
        for src in REPO.glob(pattern):
            if src.is_file():
                dst = tmp_path / src.relative_to(REPO)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    return tmp_path


def _freeze(tree: Path) -> dict[str, Any]:
    manifest = build_manifest(tree, expert_review="pending", today=date(2026, 11, 1))
    write_manifest(tree, manifest)
    return manifest


def test_manifest_records_hashes_counts_and_limitations(tree: Path) -> None:
    manifest = _freeze(tree)
    assert manifest["version"] == "2.0" and manifest["frozen_at"] == "2026-11-01"
    assert manifest["counts"] == {"all": 60, "dev": 30, "test": 30}
    assert manifest["test_runs"] == 0
    assert manifest["status"] == {"F2_human_labeling": "unmet", "expert_review_R3_G1": "pending"}
    files: dict[str, str] = manifest["files"]
    assert {"data/golden/v1.jsonl", "data/golden/v2_test.jsonl", "docs/label-policy.md"} <= set(
        files
    )
    assert sum(p.startswith("data/golden/candidates/v2/") for p in files) == 30
    assert str(MANIFEST_PATH) not in files
    text = " ".join(manifest["limitations"])
    assert "F2" in text and "약 7%" in text and "1인" in text


def test_check_passes_when_unchanged_and_detects_edits(tree: Path) -> None:
    _freeze(tree)
    assert check(tree).ok
    target = tree / "data/golden/v2_test.jsonl"
    target.write_text(
        target.read_text(encoding="utf-8").replace("절반", "소량", 1), encoding="utf-8"
    )
    report = check(tree)
    assert report.changed == ("data/golden/v2_test.jsonl",) and not report.ok
    assert "변경: data/golden/v2_test.jsonl" in report.format()


def test_check_detects_missing_and_added_files(tree: Path) -> None:
    _freeze(tree)
    (tree / "data/golden/candidates/v2/H3.json").unlink()
    (tree / "data/golden/candidates/v2/NEW.json").write_text("{}", encoding="utf-8")
    report = check(tree)
    assert report.missing == ("data/golden/candidates/v2/H3.json",)
    assert report.added == ("data/golden/candidates/v2/NEW.json",)


def test_check_without_manifest_raises(tree: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check(tree)
    assert freeze_main(["--check"], root=tree) == 2


def test_refuses_to_freeze_an_invalid_golden_set(tree: Path) -> None:
    path = tree / "data/golden/v2_dev.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    broken = json.loads(lines[0])
    broken["fields"] = broken["fields"][:3]
    path.write_text(
        "\n".join([json.dumps(broken, ensure_ascii=False), *lines[1:]]), encoding="utf-8"
    )
    assert freeze_main(["--write", "--expert-review", "done", "--allow-dirty"], root=tree) == 2
    assert load_manifest(tree) is None


def test_write_requires_expert_review_flag(tree: Path) -> None:
    assert freeze_main(["--write", "--allow-dirty"], root=tree) == 2


def test_rewrite_needs_a_reason_and_revision_bumps_the_version(tree: Path) -> None:
    assert freeze_main(["--write", "--expert-review", "pending", "--allow-dirty"], root=tree) == 0
    assert freeze_main(["--write", "--expert-review", "pending", "--allow-dirty"], root=tree) == 2
    target = tree / "data/golden/v1.jsonl"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert freeze_main(["--check"], root=tree) == 1
    code = freeze_main(
        ["--write", "--expert-review", "done", "--allow-dirty", "--revise", "공백 줄 추가"],
        root=tree,
    )
    assert code == 0
    manifest = load_manifest(tree)
    assert manifest is not None and manifest["version"] == "2.1"
    revision = manifest["revisions"][0]
    assert revision["from"] == "2.0" and revision["reason"] == "공백 줄 추가"
    assert revision["changed"] == ["data/golden/v1.jsonl"]
    assert manifest["status"]["expert_review_R3_G1"] == "done"
    assert freeze_main(["--check"], root=tree) == 0


def test_dirty_tree_blocks_freeze(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def dirty(_root: Path) -> bool:
        return False

    monkeypatch.setattr(freeze, "is_clean", dirty)
    assert freeze_main(["--write", "--expert-review", "pending"], root=tree) == 2
    assert load_manifest(tree) is None


def test_hashes_are_stable_and_content_sensitive(tree: Path) -> None:
    a = collect_hashes(tree)
    assert a == collect_hashes(tree)
    (tree / "docs/label-policy.md").write_text("changed", encoding="utf-8")
    b = collect_hashes(tree)
    assert a["docs/label-policy.md"] != b["docs/label-policy.md"]
    assert {k: v for k, v in a.items() if k != "docs/label-policy.md"} == {
        k: v for k, v in b.items() if k != "docs/label-policy.md"
    }


def test_real_golden_set_matches_its_manifest_when_frozen() -> None:
    """동결된 뒤에는 이 테스트가 라벨의 조용한 변경을 막는다. 동결 전에는 건너뛴다."""
    if load_manifest(REPO) is None:
        pytest.skip("아직 동결하지 않았다 (data/golden/FROZEN.json 없음)")
    report = check(REPO)
    assert report.ok, report.format()
