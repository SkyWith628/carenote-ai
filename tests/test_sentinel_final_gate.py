"""test R-set 최종 채점 가드: 동결 태그·파일 일치·사전 등록 설정·1회 실행."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from carenote.evaluation import sentinel_llm_run as run
from carenote.evaluation.sentinel_llm_run import (
    EXIT_REFUSED,
    FREEZE_TAG,
    FROZEN_FILES,
    TEST_MARKER,
    final_gate_error,
    main,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    for rel in FROZEN_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("v1\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "freeze")
    _git(tmp_path, "tag", FREEZE_TAG)
    return tmp_path


def gate(root: Path, **kw: object) -> str | None:
    args: dict[str, object] = {
        "model_key": "haiku",
        "prompt": "s2",
        "run_tag": None,
        "rset_dir": root / "rset",
        "root": root,
    }
    args.update(kw)
    (root / "rset").mkdir(exist_ok=True)
    return final_gate_error(**args)  # type: ignore[arg-type]


def test_gate_passes_when_frozen_files_match_tag(repo: Path) -> None:
    assert gate(repo) is None


def test_gate_refuses_after_frozen_file_edit(repo: Path) -> None:
    (repo / FROZEN_FILES[0]).write_text("v2\n", encoding="utf-8")
    assert "바뀌었다" in (gate(repo) or "")


def test_gate_refuses_staged_edit_too(repo: Path) -> None:
    (repo / FROZEN_FILES[1]).write_text("v2\n", encoding="utf-8")
    _git(repo, "add", FROZEN_FILES[1])
    assert "바뀌었다" in (gate(repo) or "")


def test_gate_refuses_without_tag(repo: Path) -> None:
    _git(repo, "tag", "-d", FREEZE_TAG)
    assert "태그" in (gate(repo) or "")


def test_gate_refuses_non_registered_settings(repo: Path) -> None:
    assert gate(repo, model_key="sonnet") is not None
    assert gate(repo, prompt="s1") is not None
    assert gate(repo, run_tag="x") is not None


def test_gate_refuses_second_scoring(repo: Path) -> None:
    (repo / "rset").mkdir()
    (repo / "rset" / TEST_MARKER).write_text("{}", encoding="utf-8")
    assert "이미 채점" in (gate(repo) or "")


def test_cli_requires_source_and_flag_together(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--sentinel-llm", "haiku", "--source", "rset-test"]) == EXIT_REFUSED
    assert main(["--sentinel-llm", "haiku", "--final-report"]) == EXIT_REFUSED
    assert "함께만" in capsys.readouterr().err


def test_cli_final_report_requires_out(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--sentinel-llm", "haiku", "--source", "rset-test", "--final-report"]
    assert main(argv) == EXIT_REFUSED
    assert "--out" in capsys.readouterr().err


def test_dev_sources_never_load_test_candidates(tmp_path: Path) -> None:
    items = run.load_items(Path("data/golden"), Path("data/golden/rset"), "rset-dev")
    assert items and all(i.item_id.startswith("K") for i in items)
    assert len(items) == 20
