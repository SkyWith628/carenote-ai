"""골든셋 동결 매니페스트 (sha256).

    python -m carenote.evaluation.freeze --check
    python -m carenote.evaluation.freeze --write --expert-review pending|done
    python -m carenote.evaluation.freeze --write --revise "사유"      # 동결 후 개정(v2.1 ...)

동결 = 라벨과 그 근거가 조용히 바뀌지 못하게 하는 장치다. 매니페스트에 적힌 파일의 해시가
달라지면 `--check`와 `tests/test_frozen.py`가 실패한다. 의도적으로 고칠 때는 `--revise`로
사유를 남기고 버전을 올린다. API를 쓰지 않는다.

한계: 매니페스트를 쓰는 시점의 git 커밋은 매니페스트를 담는 커밋의 **부모**다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from carenote.evaluation.loader import load_golden

MANIFEST_PATH = Path("data/golden/FROZEN.json")
# 동결 대상 (프로젝트 루트 기준 glob). 매니페스트 자신은 제외한다.
FROZEN_GLOBS: tuple[str, ...] = (
    "data/golden/v1.jsonl",
    "data/golden/v2_dev.jsonl",
    "data/golden/v2_test.jsonl",
    "data/golden/specs/probe_*.json",
    "data/golden/human/*.json",
    "data/golden/retired/**/*.json",
    "data/golden/candidates/v2/*.json",
    "docs/label-policy.md",
)
INITIAL_VERSION = "2.0"
DEFAULT_LIMITATIONS: tuple[str, ...] = (
    "F2(사람 독립 라벨링 값 일치율 95%) 미충족: 사람 라운드 83.9%(dev 7), 93.8%, 90.0%, 87.5%, "
    "교체 후 합산 93.2%. 정답과 사람의 불일치 하한은 약 7%이다.",
    "라벨러는 정책 작성자 1인이다. 독립 라벨러가 없다.",
    "생성기(Claude Opus)와 평가 대상(Claude Sonnet/Haiku)이 같은 계열이다. 모델 라벨러 일치율은 "
    "사람보다 높게 나와(98~100%) 사람 일치율의 대체가 아니다.",
    "dev 라벨은 모델 출력을 본 뒤 정책이 확정되며 여러 번 고쳐졌다(docs/decisions.md). "
    "test는 어떤 모델로도 실행하지 않은 상태로 동결한다.",
    "낙상 7일 기준과 특이사항 risk 기준은 현장 확인 전 임시 관례이다.",
    "test 30건이면 F1 95% 구간이 ±2~3%p라 1~2%p 차이는 구분되지 않을 수 있다.",
)


@dataclass(frozen=True)
class CheckReport:
    """매니페스트와 현재 파일의 비교 결과."""

    changed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    added: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """차이가 없는가."""
        return not (self.changed or self.missing or self.added)

    def format(self) -> str:
        """사람이 읽는 요약."""
        if self.ok:
            return "동결 매니페스트와 일치한다."
        lines = ["동결 매니페스트와 다르다:"]
        lines += [f"  변경: {p}" for p in self.changed]
        lines += [f"  삭제: {p}" for p in self.missing]
        lines += [f"  추가: {p}" for p in self.added]
        lines.append(
            '의도한 수정이면 --write --revise "사유"로 개정(버전 상승)하고 결정 로그에 기록한다.'
        )
        return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_hashes(root: Path) -> dict[str, str]:
    """동결 대상 파일의 {상대 경로: sha256}."""
    hashes: dict[str, str] = {}
    for pattern in FROZEN_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                hashes[path.relative_to(root).as_posix()] = _sha256(path)
    return hashes


def load_manifest(root: Path) -> dict[str, Any] | None:
    """저장된 매니페스트. 없으면 None."""
    path = root / MANIFEST_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check(root: Path) -> CheckReport:
    """현재 파일이 매니페스트와 같은지 비교한다. 매니페스트가 없으면 FileNotFoundError."""
    manifest = load_manifest(root)
    if manifest is None:
        raise FileNotFoundError(f"동결 매니페스트가 없다: {root / MANIFEST_PATH}")
    frozen: dict[str, str] = manifest["files"]
    now = collect_hashes(root)
    return CheckReport(
        changed=tuple(sorted(p for p in frozen if p in now and frozen[p] != now[p])),
        missing=tuple(sorted(p for p in frozen if p not in now)),
        added=tuple(sorted(p for p in now if p not in frozen)),
    )


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def is_clean(root: Path) -> bool | None:
    """git 작업 트리가 깨끗한가. git이 없으면 None."""
    status = _git(root, "status", "--porcelain")
    return None if status is None else status == ""


def _next_version(version: str) -> str:
    major, minor = version.split(".")
    return f"{major}.{int(minor) + 1}"


def build_manifest(
    root: Path,
    *,
    expert_review: str,
    test_runs: int = 0,
    previous: dict[str, Any] | None = None,
    revise_reason: str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """매니페스트를 만든다. 로더 검증을 통과한 골든셋만 동결할 수 있다."""
    golden = root / "data" / "golden"
    cases = load_golden(golden)  # 스키마 위반이면 ValueError
    now = collect_hashes(root)
    day = (today or date.today()).isoformat()
    revisions: list[dict[str, Any]] = list(previous["revisions"]) if previous else []
    version = INITIAL_VERSION
    if previous is not None:
        if not revise_reason:
            raise ValueError("이미 동결돼 있다. 개정하려면 사유(--revise)가 필요하다")
        old: dict[str, str] = previous["files"]
        revisions.append(
            {
                "from": previous["version"],
                "date": day,
                "reason": revise_reason,
                "changed": sorted(p for p in now if p in old and old[p] != now[p]),
                "added": sorted(p for p in now if p not in old),
                "removed": sorted(p for p in old if p not in now),
            }
        )
        version = _next_version(previous["version"])
    return {
        "version": version,
        "frozen_at": previous["frozen_at"] if previous else day,
        "updated_at": day,
        "git_parent_commit": _git(root, "rev-parse", "HEAD"),
        "counts": {
            "all": len(cases),
            "dev": sum(c.split == "dev" for c in cases),
            "test": sum(c.split == "test" for c in cases),
        },
        "test_runs": test_runs,
        "status": {"F2_human_labeling": "unmet", "expert_review_R3_G1": expert_review},
        "limitations": list(DEFAULT_LIMITATIONS),
        "files": now,
        "revisions": revisions,
    }


def write_manifest(root: Path, manifest: dict[str, Any]) -> Path:
    """매니페스트를 파일로 쓴다."""
    path = root / MANIFEST_PATH
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    """CLI 진입점. 종료 코드: 0 성공 · 1 불일치 · 2 실행 거부."""
    parser = argparse.ArgumentParser(prog="carenote.evaluation.freeze", description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="매니페스트와 현재 파일을 비교")
    mode.add_argument("--write", action="store_true", help="매니페스트를 쓴다 (동결 또는 개정)")
    parser.add_argument("--expert-review", choices=["pending", "done"], default=None)
    parser.add_argument("--test-runs", type=int, default=0, help="동결 시점까지 test 실행 횟수")
    parser.add_argument("--revise", default=None, metavar="REASON", help="동결 후 개정 사유")
    parser.add_argument("--allow-dirty", action="store_true", help="git 작업 트리가 더러워도 허용")
    args = parser.parse_args(argv)
    base = root or Path.cwd()

    if args.check:
        try:
            report = check(base)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(report.format())
        return 0 if report.ok else 1

    if args.expert_review is None:
        print("--write에는 --expert-review pending|done이 필요하다", file=sys.stderr)
        return 2
    if not args.allow_dirty and is_clean(base) is False:
        print("git 작업 트리가 깨끗하지 않다. 커밋한 뒤 다시 실행한다 (--allow-dirty로 우회 가능).",
              file=sys.stderr)  # fmt: skip
        return 2
    previous = load_manifest(base)
    if previous is not None and not args.revise:
        print(f"이미 동결돼 있다(v{previous['version']}). 개정은 --revise \"사유\"로 한다.",
              file=sys.stderr)  # fmt: skip
        return 2
    try:
        manifest = build_manifest(
            base,
            expert_review=args.expert_review,
            test_runs=args.test_runs,
            previous=previous,
            revise_reason=args.revise,
        )
    except ValueError as exc:
        print(f"동결 거부: {exc}", file=sys.stderr)
        return 2
    path = write_manifest(base, manifest)
    print(f"매니페스트 v{manifest['version']} 저장: {path} (파일 {len(manifest['files'])}개)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
