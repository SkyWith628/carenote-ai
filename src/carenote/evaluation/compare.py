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

from carenote.evaluation.metrics import Counts
from carenote.evaluation.stats import Interval, paired_f1_difference


def load_per_case(path: Path) -> dict[str, Counts]:
    """보고서 JSON에서 케이스별 TP/FP/FN을 읽는다."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    raw = doc.get("per_case")
    if not raw:
        raise ValueError(f"{path}: per_case가 없다. 현재 러너로 다시 채점한 보고서가 필요하다")
    return {k: Counts(tp=v["tp"], fp=v["fp"], fn=v["fn"]) for k, v in raw.items()}


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
    args = parser.parse_args(argv)
    try:
        diff = paired_f1_difference(load_per_case(args.a), load_per_case(args.b))
    except ValueError as exc:
        print(f"비교 불가: {exc}", file=sys.stderr)
        return 2
    print(format_difference(args.a.stem, args.b.stem, diff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
