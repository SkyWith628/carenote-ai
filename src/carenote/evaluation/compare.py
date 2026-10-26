"""두 평가 보고서의 짝지은 비교.

    python -m carenote.evaluation.compare A.json B.json

같은 케이스 집합에서 F1(B) - F1(A)와 케이스 단위 부트스트랩 95% 구간을 낸다.
구간이 0을 포함하면 이 골든셋으로는 두 구성을 구분할 수 없다는 뜻이다.
보고서는 `--out`으로 저장한 것이어야 하며(per_case 포함), 이전 형식은 다시 채점해야 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from carenote.evaluation.metrics import Counts
from carenote.evaluation.stats import Interval, paired_f1_difference


def select_per_case(
    doc: dict[str, Any],
    *,
    exclude_review_required: bool = False,
    recoverable: bool = False,
    name: str = "보고서",
) -> dict[str, Counts]:
    """보고서 dict에서 케이스별 TP/FP/FN을 고른다.

    - `recoverable`: 복원 불가 필드를 뺀 집계(`per_case_recoverable`)를 쓴다.
    - `exclude_review_required`: 현장 검토 대기 케이스(R3·G1)를 뺀다(헤드라인 비교용).
    """
    key = "per_case_recoverable" if recoverable else "per_case"
    raw = doc.get(key)
    if not raw:
        raise ValueError(f"{name}: {key}가 없다. 현재 러너로 다시 채점한 보고서가 필요하다")
    skip: set[str] = set()
    if exclude_review_required:
        if "review_required_ids" not in doc:
            raise ValueError(f"{name}: review_required_ids가 없다. 현재 러너로 다시 채점해야 한다")
        skip = set(doc["review_required_ids"])
    return {k: Counts(tp=v["tp"], fp=v["fp"], fn=v["fn"]) for k, v in raw.items() if k not in skip}


def load_per_case(
    path: Path, *, exclude_review_required: bool = False, recoverable: bool = False
) -> dict[str, Counts]:
    """보고서 JSON 파일에서 케이스별 TP/FP/FN을 읽는다."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    return select_per_case(
        doc,
        exclude_review_required=exclude_review_required,
        recoverable=recoverable,
        name=str(path),
    )


def format_difference(label_a: str, label_b: str, diff: Interval) -> str:
    """차이와 구간을 한 줄로."""
    if diff.point is None:
        return f"{label_b} - {label_a}: 계산 불가"
    if diff.low is None or diff.high is None:
        return f"{label_b} - {label_a}: {diff.point * 100:+.1f}%p (구간 없음: 케이스 부족)"
    verdict = "구분 가능" if diff.low > 0 or diff.high < 0 else "구분 불가(구간이 0 포함)"
    return (
        f"{label_b} - {label_a}: {diff.point * 100:+.1f}%p "
        f"[95% {diff.low * 100:+.1f} ~ {diff.high * 100:+.1f}] → {verdict}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(prog="carenote.evaluation.compare", description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument(
        "--exclude-review-required",
        action="store_true",
        help="현장 검토 대기 케이스(R3·G1)를 뺀 헤드라인 비교",
    )
    parser.add_argument(
        "--recoverable", action="store_true", help="복원 불가 필드를 뺀 집계로 비교"
    )
    args = parser.parse_args(argv)
    opts = {
        "exclude_review_required": args.exclude_review_required,
        "recoverable": args.recoverable,
    }
    try:
        diff = paired_f1_difference(load_per_case(args.a, **opts), load_per_case(args.b, **opts))
    except ValueError as exc:
        print(f"비교 불가: {exc}", file=sys.stderr)
        return 2
    print(format_difference(args.a.stem, args.b.stem, diff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
