"""프롬프트 ablation 드라이버: 변형 × 케이스 × 반복을 한 프로세스·한 비용 한도로 돌린다.

    python -m carenote.evaluation.ablate --extractor llm-b1 --model claude-haiku-5-5 \\
        --targets gs-003:blood_pressure --variants v3 v4 v4-drop:2 v4-only:2 --runs 20 \\
        --dry-run
    ... --approve-cost 0.20

변형 표기: `v3` · `v4` · `v4-drop:N`(v4에서 편집 N만 뺌) · `v4-only:N`(v3에 편집 N만 얹음) ·
`v4-edits:1,3,4`(v3에 그 편집들만). 한 호출이 끝날 때마다 누적 비용을 세고 승인액에 닿으면
멈춘다. 반복은 변형들을 번갈아 돌려(1회차 전 변형 → 2회차 …) 중간에 멈춰도 균형이 맞는다.
각 반복은 서로 다른 `--run-tag`(ab1, ab2, …)를 써서 캐시를 공유하지 않고, 끝난 호출은 캐시에
남아 같은 명령을 다시 실행하면 이어서 진행된다. 평가는 dev 케이스만 한다(test는 거부).

종료 코드: 0 성공 · 2 거부(승인·키·단가·입력 오류) · 3 비용 한도로 중단
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from carenote.config import Settings, get_settings
from carenote.evaluation.loader import load_golden
from carenote.evaluation.metrics import CORRECT_OUTCOMES, field_outcomes
from carenote.evaluation.schema import GoldenCase
from carenote.extraction.estimate import estimate_run
from carenote.extraction.llm_extractor import LLMExtractor, ParseClient, build_client
from carenote.extraction.pricing import PRICES_AS_OF, BudgetExceeded, CostGuard, price_for
from carenote.extraction.prompts import (
    ALL_V4_EDITS,
    PromptVersion,
    Variant,
    normalize_edits,
)

DEFAULT_GOLDEN = Path("data/golden")
EXIT_REFUSED = 2
EXIT_BUDGET = 3
RESTORE_P = 0.05


@dataclass(frozen=True)
class VariantSpec:
    """프롬프트 변형 하나."""

    label: str
    version: PromptVersion
    edits: tuple[int, ...] | None


def parse_variant(text: str) -> VariantSpec:
    """`v4-drop:2` 같은 표기를 해석한다. 잘못된 표기는 ValueError."""
    if text in ("v1", "v2", "v3", "v4"):
        return VariantSpec(text, text, None)  # type: ignore[arg-type]
    head, _, arg = text.partition(":")
    try:
        numbers = tuple(int(x) for x in arg.split(",") if x.strip())
    except ValueError as exc:
        raise ValueError(f"변형 표기가 잘못됐다: {text}") from exc
    if head == "v4-drop" and len(numbers) == 1:
        chosen = tuple(e for e in ALL_V4_EDITS if e not in numbers)
    elif head == "v4-only" and len(numbers) == 1:
        chosen = numbers
    elif head == "v4-edits" and numbers:
        chosen = numbers
    else:
        raise ValueError(f"변형 표기가 잘못됐다: {text}")
    return VariantSpec(text, "v4", normalize_edits(chosen))


def fisher_one_sided(
    a_correct: int, a_n: int, b_correct: int, b_n: int, *, b_higher: bool
) -> float:
    """B 묶음이 A보다 더 많이(b_higher) 또는 더 적게 맞은 것이 우연일 확률(초기하분포 단측)."""
    total_c, total_n = a_correct + b_correct, a_n + b_n
    denom = math.comb(total_n, b_n)

    def pmf(k: int) -> int:
        if k < 0 or k > b_n or total_c - k > a_n or total_c - k < 0:
            return 0
        return math.comb(total_c, k) * math.comb(total_n - total_c, b_n - k)

    ks = range(b_correct, b_n + 1) if b_higher else range(0, b_correct + 1)
    return sum(pmf(k) for k in ks) / denom


@dataclass
class Tally:
    """변형 × 목표 필드의 반복 결과."""

    outcomes: list[str] = field(default_factory=lambda: list[str]())
    failures: int = 0

    @property
    def n(self) -> int:
        """유효 호출 수(실패 제외)."""
        return len(self.outcomes)

    @property
    def correct(self) -> int:
        """정답 횟수."""
        return sum(o in CORRECT_OUTCOMES for o in self.outcomes)


def parse_targets(items: list[str]) -> list[tuple[str, str]]:
    """`gs-003:blood_pressure` 목록을 (케이스, 필드)로."""
    result: list[tuple[str, str]] = []
    for item in items:
        case, _, fld = item.partition(":")
        if not case or not fld:
            raise ValueError(f"목표 형식이 잘못됐다(케이스:필드): {item}")
        result.append((case, fld))
    return result


def run_ablation(
    cases: dict[str, GoldenCase],
    targets: list[tuple[str, str]],
    variants: list[VariantSpec],
    runs: int,
    *,
    client: ParseClient,
    extractor_variant: Variant,
    model: str,
    effort: str,
    cache_dir: Path | None,
    guard: CostGuard,
) -> tuple[dict[str, dict[tuple[str, str], Tally]], bool]:
    """변형을 번갈아 돌려 목표 필드의 정오를 센다.

    비용 한도에 닿으면 거기까지의 결과와 `True`(중단됨)를 돌려준다.
    """
    tallies = {v.label: {t: Tally() for t in targets} for v in variants}
    case_ids = sorted({c for c, _ in targets})
    try:
        _run_replicates(
            tallies, cases, targets, variants, runs, case_ids, client, extractor_variant,
            model, effort, cache_dir, guard,
        )  # fmt: skip
    except BudgetExceeded:
        return tallies, True
    return tallies, False


def _run_replicates(
    tallies: dict[str, dict[tuple[str, str], Tally]],
    cases: dict[str, GoldenCase],
    targets: list[tuple[str, str]],
    variants: list[VariantSpec],
    runs: int,
    case_ids: list[str],
    client: ParseClient,
    extractor_variant: Variant,
    model: str,
    effort: str,
    cache_dir: Path | None,
    guard: CostGuard,
) -> None:
    for replicate in range(1, runs + 1):
        for spec in variants:
            extractor = LLMExtractor(
                client,
                variant=extractor_variant,
                model=model,
                effort=effort,
                cache_dir=cache_dir,
                guard=guard,
                prompt_version=spec.version,
                run_tag=f"ab{replicate}",
                prompt_edits=spec.edits,
            )
            for case_id in case_ids:
                before = extractor.stats.failures
                predictions = extractor.extract(cases[case_id].transcript)
                failed = extractor.stats.failures > before
                outcomes = field_outcomes(cases[case_id], predictions)
                for case, fld in targets:
                    if case != case_id:
                        continue
                    tally = tallies[spec.label][(case, fld)]
                    if failed:
                        tally.failures += 1
                    else:
                        tally.outcomes.append(outcomes[fld])


def format_table(
    tallies: dict[str, dict[tuple[str, str], Tally]], variants: list[VariantSpec]
) -> str:
    """변형별 정답 횟수와 기준 대비 단측 p. v3·v4가 목록에 있으면 원인 후보 규칙을 적용한다."""
    lines: list[str] = []
    for target in next(iter(tallies.values())):
        name = f"{target[0]}:{target[1]}"
        lines.append(f"[{name}]")
        base = tallies.get("v3", {}).get(target)
        new = tallies.get("v4", {}).get(target)
        midpoint: float | None = None
        if base and new and base.n and new.n:
            midpoint = (base.correct / base.n + new.correct / new.n) / 2
        for spec in variants:
            t = tallies[spec.label][target]
            rate = "–" if t.n == 0 else f"{t.correct / t.n * 100:.0f}%"
            note = f" · 실패 {t.failures}" if t.failures else ""
            extras: list[str] = []
            if base and t.n and base.n and spec.label != "v3":
                p_down = fisher_one_sided(base.correct, base.n, t.correct, t.n, b_higher=False)
                extras.append(f"v3보다 낮을 p={p_down:.3f}")
            if new and t.n and new.n and spec.label != "v4":
                p_up = fisher_one_sided(new.correct, new.n, t.correct, t.n, b_higher=True)
                extras.append(f"v4보다 높을 p={p_up:.3f}")
                if (
                    spec.label.startswith("v4-drop")
                    and midpoint is not None
                    and t.correct / t.n >= midpoint
                    and p_up <= RESTORE_P
                ):
                    extras.append("→ 원인 후보(복원에 필요한 편집)")
            tail = f"  ({'; '.join(extras)})" if extras else ""
            lines.append(f"  {spec.label:<14} {t.correct}/{t.n} ({rate}){note}{tail}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="carenote.evaluation.ablate", description=__doc__)
    parser.add_argument("--extractor", choices=["llm-b0", "llm-b1"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None)
    parser.add_argument("--targets", nargs="+", required=True, metavar="CASE:FIELD")
    parser.add_argument("--variants", nargs="+", required=True, metavar="VARIANT")
    parser.add_argument("--runs", type=int, default=20, help="변형마다 반복 횟수")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="JSON 결과 저장 경로")
    parser.add_argument("--dry-run", action="store_true", help="API 없이 호출 수와 예상 비용만")
    parser.add_argument("--approve-cost", type=float, default=None, metavar="USD")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    client_factory: Callable[[Settings], ParseClient] = build_client,
) -> int:
    """CLI 진입점."""
    args = _build_parser().parse_args(argv)
    settings = settings or get_settings()
    model = args.model or settings.extractor_model
    effort = args.effort or settings.extractor_effort
    extractor_variant: Variant = "b1" if args.extractor == "llm-b1" else "b0"
    try:
        targets = parse_targets(args.targets)
        variants = [parse_variant(v) for v in args.variants]
        if len({v.label for v in variants}) != len(variants):
            raise ValueError("변형이 중복됐다")
        if args.runs < 1:
            raise ValueError("--runs는 1 이상이어야 한다")
        dev_cases = {c.id: c for c in load_golden(args.golden, "dev")}
        unknown = sorted({c for c, _ in targets} - set(dev_cases))
        if unknown:
            raise ValueError(f"dev 분할에 없는 케이스(test는 쓸 수 없다): {unknown}")
        bad = [
            f"{c}:{f}" for c, f in targets if f not in {x.key.value for x in dev_cases[c].fields}
        ]
        if bad:
            raise ValueError(f"알 수 없는 필드: {bad}")
    except ValueError as exc:
        print(f"실행 거부: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    price = price_for(model)
    if price is None:
        print(f"단가가 등록되지 않은 모델이라 비용을 추정할 수 없다: {model}", file=sys.stderr)
        return EXIT_REFUSED

    case_ids = sorted({c for c, _ in targets})
    cache_dir = args.cache_dir or Path(settings.extraction_cache_dir)

    class _NoClient:
        @property
        def messages(self) -> object:
            raise RuntimeError("dry-run에서 API 클라이언트를 쓰면 안 된다")

    cost_low = cost_high = 0.0
    for spec in variants:
        probe = LLMExtractor(
            _NoClient(),  # type: ignore[arg-type]
            variant=extractor_variant,
            model=model,
            effort=effort,
            cache_dir=cache_dir,
            prompt_version=spec.version,
            run_tag="ab1",
            prompt_edits=spec.edits,
        )
        est = estimate_run(probe, [dev_cases[c] for c in case_ids], price)
        cost_low += est.cost_low * args.runs
        cost_high += est.cost_high * args.runs
    calls = len(variants) * len(case_ids) * args.runs
    print(
        f"[ablate] 모델 {model} · 변형 {len(variants)}개 × 케이스 {len(case_ids)}개 × "
        f"{args.runs}회 = 호출 최대 {calls}건 (캐시 적중은 비용 0)\n"
        f"  예상 비용 ${cost_low:.3f} ~ ${cost_high:.3f} (단가 기준일 {PRICES_AS_OF}, "
        "글자 수 기반 추정이라 오차가 크다)"
    )
    if args.dry_run:
        print("  (dry-run: API를 호출하지 않았다)")
        return 0
    approved: float | None = args.approve_cost
    if approved is None or approved <= 0:
        print("\n실행 거부: --approve-cost <USD>가 필요하다.", file=sys.stderr)
        return EXIT_REFUSED
    if cost_high > approved:
        print(
            f"\n실행 거부: 추정 최대 ${cost_high:.3f}가 승인액 ${approved:.2f}를 넘는다. "
            "누적 한도로 멈추므로 승인액이 최대 추정에 못 미쳐도 실제로는 끝날 수 있지만, "
            "추정 최대 이상으로 승인해야 시작한다.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if settings.anthropic_api_key is None:
        print("\n실행 거부: ANTHROPIC_API_KEY가 없다 (.env에 넣는다).", file=sys.stderr)
        return EXIT_REFUSED

    guard = CostGuard(approved, price)
    tallies, stopped = run_ablation(
        dev_cases,
        targets,
        variants,
        args.runs,
        client=client_factory(settings),
        extractor_variant=extractor_variant,
        model=model,
        effort=effort,
        cache_dir=cache_dir,
        guard=guard,
    )
    if stopped:
        print(
            f"\n중단: 승인 한도 ${approved:.2f}에 도달. 끝난 호출은 캐시에 있어 같은 명령(승인액을 "
            "올려)으로 이어서 진행된다. 아래는 거기까지의 결과다.",
            file=sys.stderr,
        )
    print(format_table(tallies, variants))
    print(f"누적 비용 ${guard.spent_usd:.4f} (승인 ${approved:.2f})")
    if args.out is not None:
        payload = {
            "model": model,
            "extractor": args.extractor,
            "runs": args.runs,
            "cost_usd": round(guard.spent_usd, 6),
            "variants": {
                spec.label: {
                    "version": spec.version,
                    "edits": list(spec.edits) if spec.edits else None,
                    "targets": {
                        f"{c}:{f}": {
                            "correct": t.correct,
                            "n": t.n,
                            "failures": t.failures,
                            "outcomes": t.outcomes,
                        }
                        for (c, f), t in tallies[spec.label].items()
                    },
                }
                for spec in variants
            },
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return EXIT_BUDGET if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
