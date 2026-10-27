"""Sentinel LLM 보조 단계 dev 실행기 (설계: docs/sentinel-llm-assist-design.md 7장).

dev 분할만 읽는다: 골든셋 dev 30건과 R-set dev 20건. test는 어떤 경로로도 읽지 않는다.
API 호출은 --approve-cost가 추정 최대 이상일 때만 한다. --dry-run은 호출하지 않는다.

  python -m carenote.evaluation.sentinel_llm_run --sentinel-llm haiku --dry-run
  python -m carenote.evaluation.sentinel_llm_run --sentinel-llm haiku --approve-cost 0.06 \\
      --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

from carenote.config import Settings, get_settings
from carenote.evaluation.loader import load_golden
from carenote.evaluation.schema import FieldKey, Flag
from carenote.extraction.llm_extractor import ParseClient, build_client
from carenote.extraction.pricing import PRICES_AS_OF, BudgetExceeded, CostGuard, cost_usd, price_for
from carenote.extraction.sentinel import RuleSet, load_rules, scan_text
from carenote.extraction.sentinel_llm import (
    SYSTEM_PROMPT,
    AssistResult,
    SentinelAssist,
    build_user_message,
    merge_signals,
)
from carenote.generation.spec import load_specs

EXIT_REFUSED = 2
EXIT_BUDGET = 3
MODEL_IDS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
DEFAULT_CACHE = Path(".cache/sentinel_llm")
CHARS_PER_TOKEN = 1.5
SCHEMA_OVERHEAD_TOKENS = 400
INPUT_HIGH_FACTOR = 1.3
OUTPUT_TOKENS_LOW, OUTPUT_TOKENS_HIGH = 150, 1500


@dataclass(frozen=True)
class Item:
    """채점 단위. group은 R-set의 표현 층(lexicon·novel·circ·neg)이나 골든(golden-pos·neg)."""

    item_id: str
    origin: str  # rset | golden
    group: str
    gold_risk: bool
    gray: bool
    transcript: str


def wilson(k: int, n: int, z: float = 1.96) -> tuple[int, int]:
    """Wilson 95% 구간(퍼센트, 정수). n=0이면 (0, 0)."""
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - a) / d * 100), round((c + a) / d * 100)


def load_items(golden: Path, rset_dir: Path, source: str) -> list[Item]:
    """dev 항목만 읽는다. R-set은 스펙의 split이 dev인 것만, 후보 transcript가 없으면 ValueError."""
    items: list[Item] = []
    if source in ("both", "golden-dev"):
        for case in load_golden(golden, "dev"):
            note = case.field(FieldKey.SPECIAL_NOTES)
            risk = note.value is not None and note.flag is Flag.RISK
            items.append(
                Item(
                    case.id,
                    "golden",
                    "golden-pos" if risk else "golden-neg",
                    risk,
                    False,
                    case.transcript,
                )
            )
    if source in ("both", "rset-dev"):
        for spec in load_specs(rset_dir / "specs"):
            if spec.split != "dev":
                continue
            path = rset_dir / "candidates" / f"{spec.spec_id}.json"
            if not path.exists():
                raise ValueError(f"R-set 후보가 없다: {spec.spec_id}")
            transcript = str(json.loads(path.read_text(encoding="utf-8")).get("transcript") or "")
            note = spec.gold[FieldKey.SPECIAL_NOTES]
            risk = note.value is not None and note.flag is Flag.RISK
            if not risk:
                group = "neg"
            elif spec.field_review_required:
                group = "circ"
            else:
                group = "novel" if "표현 novel" in spec.policy else "lexicon"
            gray = bool(spec.gray_fields) or spec.spec_id == "K26"
            items.append(Item(spec.spec_id, "rset", group, risk, gray, transcript))
    return items


def estimate(assist: SentinelAssist, items: list[Item], model: str) -> tuple[int, float, float]:
    """(호출 예정 수, 비용 하한, 비용 상한). 캐시에 있는 항목은 호출하지 않는다."""
    price = price_for(model)
    if price is None:
        raise ValueError(f"단가가 등록되지 않은 모델: {model}")
    pending = [i for i in items if not assist.is_cached(i.transcript)]
    low_in = sum(
        int((len(SYSTEM_PROMPT) + len(build_user_message(i.transcript))) / CHARS_PER_TOKEN)
        + SCHEMA_OVERHEAD_TOKENS
        for i in pending
    )
    high_in = int(low_in * INPUT_HIGH_FACTOR)
    n = len(pending)
    return (
        n,
        cost_usd(price, low_in, OUTPUT_TOKENS_LOW * n),
        cost_usd(price, high_in, OUTPUT_TOKENS_HIGH * n),
    )


def _rate(rows: list[dict[str, Any]], group: str, key: str) -> str:
    sel = [r for r in rows if r["group"] == group]
    k = sum(1 for r in sel if r[key])
    lo, hi = wilson(k, len(sel))
    return f"{k}/{len(sel)} (구간 {lo}~{hi}%)"


def summarize(rows: list[dict[str, Any]]) -> list[str]:
    """그룹별 규칙 단독 · LLM 단독 · 합집합 포착/오발동."""
    lines = ["그룹 | 규칙 단독 | LLM 단독 | 규칙 ∪ LLM", "---|---|---|---"]
    labels = {
        "lexicon": "양성 어휘 근접(R-set)",
        "novel": "양성 어휘 밖(R-set)",
        "circ": "정황 양성(별도)",
        "neg": "음성 오발동(R-set)",
        "golden-pos": "골든 dev 양성",
        "golden-neg": "골든 dev 음성 오발동",
    }
    for group, label in labels.items():
        if any(r["group"] == group for r in rows):
            lines.append(
                f"{label} | {_rate(rows, group, 'rules')} | {_rate(rows, group, 'llm')} "
                f"| {_rate(rows, group, 'union')}"
            )
    gray = [r for r in rows if r["group"] == "neg" and r["gray"]]
    sel = [r for r in rows if r["group"] == "neg" and not r["gray"]]
    if gray:
        k = sum(1 for r in sel if r["union"])
        lo, hi = wilson(k, len(sel))
        lines.append(f"R-set 음성 gray 제외 오발동(합집합): {k}/{len(sel)} (구간 {lo}~{hi}%)")
    return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carenote.evaluation.sentinel_llm_run", description=__doc__
    )
    parser.add_argument("--sentinel-llm", choices=sorted(MODEL_IDS), required=True, dest="model")
    parser.add_argument("--source", choices=["both", "golden-dev", "rset-dev"], default="both")
    parser.add_argument("--golden", type=Path, default=Path("data/golden"))
    parser.add_argument("--rset-dir", type=Path, default=Path("data/golden/rset"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approve-cost", type=float, default=None, metavar="USD")
    return parser


def _row(item: Item, rules: RuleSet, result: AssistResult) -> dict[str, Any]:
    rule_signals = scan_text(item.transcript, rules)
    union = merge_signals(rule_signals, result.signals)
    return {
        "id": item.item_id,
        "origin": item.origin,
        "group": item.group,
        "gold_risk": item.gold_risk,
        "gray": item.gray,
        "rules": bool(rule_signals),
        "llm": bool(result.signals),
        "union": bool(union),
        "status": result.status,
        "reason": result.reason,
        "llm_signals": [{"category": s.category, "quote": s.quote} for s in result.signals],
        "rejected": [
            {"check": r.check, "quote": r.quote, "reason": r.reason} for r in result.rejected
        ],
    }


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    client_factory: Callable[[Settings], ParseClient] = build_client,
) -> int:
    """CLI 진입점. 종료 코드 0 성공 · 2 거부 · 3 비용 한도 중단."""
    args = _build_parser().parse_args(argv)
    settings = settings or get_settings()
    model = MODEL_IDS[args.model]
    price = price_for(model)
    if price is None:
        print(f"단가가 등록되지 않은 모델: {model}", file=sys.stderr)
        return EXIT_REFUSED
    rules = load_rules()
    try:
        items = load_items(args.golden, args.rset_dir, args.source)
    except (ValueError, OSError) as exc:
        print(f"입력 오류: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if not items:
        print("채점할 항목이 없다.", file=sys.stderr)
        return EXIT_REFUSED

    class _NoClient:
        @property
        def messages(self) -> object:
            raise RuntimeError("dry-run 추정에서 API 클라이언트를 쓰면 안 된다")

    probe = SentinelAssist(
        _NoClient(),  # type: ignore[arg-type]
        model=model,
        effort=args.effort,
        rules=rules,
        cache_dir=args.cache_dir,
        run_tag=args.run_tag,
    )
    pending, low, high = estimate(probe, items, model)
    print(
        f"[dry-run] Sentinel LLM 보조 {model} · effort {args.effort} · dev {len(items)}건\n"
        f"  캐시 적중 {len(items) - pending}건, API 호출 예정 {pending}건\n"
        f"  예상 비용 ${low:.3f} ~ ${high:.3f} (단가 기준일 {PRICES_AS_OF}, 글자 수 기반 추정)"
    )
    if args.dry_run:
        print("  (dry-run: API를 호출하지 않았다)")
        return 0
    approved: float | None = args.approve_cost
    if approved is None or approved <= 0:
        print("\n실행 거부: --approve-cost <USD>가 필요하다.", file=sys.stderr)
        return EXIT_REFUSED
    if high > approved:
        print(
            f"\n실행 거부: 추정 최대 ${high:.3f}가 승인액 ${approved:.2f}를 넘는다.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if pending and settings.anthropic_api_key is None:
        print("\n실행 거부: ANTHROPIC_API_KEY가 없다.", file=sys.stderr)
        return EXIT_REFUSED

    guard = CostGuard(approved, price)
    assist = SentinelAssist(
        client_factory(settings) if pending else _NoClient(),  # type: ignore[arg-type]
        model=model,
        effort=args.effort,
        rules=rules,
        cache_dir=args.cache_dir,
        guard=guard,
        run_tag=args.run_tag,
    )
    rows: list[dict[str, Any]] = []
    try:
        for item in items:
            rows.append(_row(item, rules, assist.assess(item.transcript)))
    except BudgetExceeded as exc:
        print(f"\n중단: {exc}. 끝낸 호출은 캐시에 저장됐다.", file=sys.stderr)
        return EXIT_BUDGET
    rejected = Counter(r["check"] for row in rows for r in row["rejected"])
    unavailable = [row["id"] for row in rows if row["status"] == "unavailable"]
    print("\n".join(["", *summarize(rows)]))
    print(
        f"\n검증 거절: {dict(sorted(rejected.items())) or '없음'}"
        f" · 호출 실패(미실행): {unavailable or '없음'}"
    )
    print(f"사용량: {assist.usage_summary()}")
    if args.out is not None:
        args.out.write_text(
            json.dumps(
                {"model": model, "usage": assist.usage_summary(), "rows": rows},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
