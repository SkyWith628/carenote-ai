"""골든셋 JSONL 로더."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from carenote.evaluation.schema import GoldenCase

SplitChoice = Literal["dev", "test", "all"]


def load_golden(path: Path, split: SplitChoice = "all") -> list[GoldenCase]:
    """JSONL 파일 하나 또는 디렉터리의 *.jsonl을 읽어 검증한다.

    id가 중복되거나 한 줄이라도 스키마를 어기면 ValueError를 던진다. 검증은 분할과 무관하게
    전체에 적용하고, 마지막에 `split`으로 거른다. 평가 CLI의 기본값은 test 누수를 막기 위해
    dev다(이 함수의 기본값 all은 데이터 점검용).
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
    return cases if split == "all" else [c for c in cases if c.split == split]
