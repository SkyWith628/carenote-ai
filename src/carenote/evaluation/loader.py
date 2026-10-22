"""골든셋 JSONL 로더."""

from __future__ import annotations

from pathlib import Path

from carenote.evaluation.schema import GoldenCase


def load_golden(path: Path) -> list[GoldenCase]:
    """JSONL 파일 하나 또는 디렉터리의 *.jsonl을 읽어 검증한다.

    id가 중복되거나 한 줄이라도 스키마를 어기면 ValueError를 던진다.
    """
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    cases: list[GoldenCase] = []
    for file in files:
        for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                cases.append(GoldenCase.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{file.name}:{lineno}: {exc}") from exc
    ids = [c.id for c in cases]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"중복된 케이스 id: {duplicated}")
    return cases
