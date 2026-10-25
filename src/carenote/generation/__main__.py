"""생성 CLI.

dry-run:  python -m carenote.generation --pilot --dry-run
파일럿:   python -m carenote.generation --pilot --approve-cost 0.50
검수 시트: python -m carenote.generation --render-review docs/reports/v2-review.md

비용이 드는 실행은 --approve-cost(승인할 최대 USD)와 API 키가 모두 있어야 시작한다.
추정 최대 비용이 승인액을 넘으면 거부하고, 실행 중 누적 비용이 승인액에 도달하면 중단한다.
종료 코드: 0 성공 · 2 실행 거부 · 3 비용 한도로 중단
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from carenote.config import Settings, get_settings
from carenote.extraction.llm_extractor import ParseClient, build_client
from carenote.extraction.pricing import (
    PRICES,
    PRICES_AS_OF,
    BudgetExceeded,
    CostGuard,
    price_for,
)
from carenote.generation.candidates import (
    apply_batch_checks,
    blind_ids,
    blind_label_template,
    compare_blind,
    format_blind_report,
    load_candidates,
    render_blind_sheet,
    render_review_sheet,
    write_candidate,
)
from carenote.generation.diversity import assign_diversity
from carenote.generation.estimate import estimate_generation, format_estimate
from carenote.generation.generator import DEFAULT_MAX_ATTEMPTS, Generator
from carenote.generation.spec import Spec, load_specs

DEFAULT_SPEC_DIR = Path("data/golden/specs")
DEFAULT_OUT_DIR = Path("data/golden/candidates/v2")
DEFAULT_GOLDEN = Path("data/golden")
PILOT_IDS = ("H3", "S1", "L1")
EXIT_REFUSED = 2
EXIT_BUDGET = 3


class _NoClient:
    """dry-run용. 호출되면 버그이므로 즉시 실패한다."""

    @property
    def messages(self) -> object:
        raise RuntimeError("dry-run에서 API 클라이언트를 쓰면 안 된다")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="carenote.generation", description=__doc__)
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--spec", action="append", default=None, help="spec_id (여러 번 지정 가능)")
    parser.add_argument("--pilot", action="store_true", help=f"파일럿 {', '.join(PILOT_IDS)}")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approve-cost", type=float, default=None, metavar="USD")
    parser.add_argument("--render-review", type=Path, default=None, metavar="PATH")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="--render-review와 함께: 정답을 숨긴 라벨링 시트와 빈 라벨 파일을 만든다",
    )
    parser.add_argument("--compare-blind", type=Path, default=None, metavar="LABELS_JSON")
    return parser


def select_specs(specs: list[Spec], args: argparse.Namespace) -> list[tuple[Spec, bool]]:
    """생성할 (스펙, 비교용 여부) 목록. 직접 작성 스펙은 명시한 경우에만 비교용으로 포함."""
    by_id = {s.spec_id: s for s in specs}
    wanted: list[str] = list(PILOT_IDS) if args.pilot else list(args.spec or [])
    if wanted:
        unknown = [w for w in wanted if w not in by_id]
        if unknown:
            raise SystemExit(f"알 수 없는 spec_id: {unknown}")
        return [(by_id[w], by_id[w].author == "human") for w in wanted]
    return [
        (s, False)
        for s in specs
        if s.author == "llm" and (args.split == "all" or s.split == args.split)
    ]


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    client_factory: Callable[[Settings], ParseClient] = build_client,
) -> int:
    """CLI 진입점. settings·client_factory는 테스트에서 바꿔 끼운다."""
    args = _build_parser().parse_args(argv)
    settings = settings or get_settings()
    specs = load_specs(args.spec_dir)

    if args.compare_blind is not None:
        key_path = args.out_dir.parent / "blind_key.json"
        key = json.loads(key_path.read_text(encoding="utf-8"))
        labels = json.loads(args.compare_blind.read_text(encoding="utf-8"))
        print(format_blind_report(compare_blind(labels, key, specs)))
        return 0

    if args.render_review is not None and args.blind:
        ids = [str(d["spec_id"]) for d in load_candidates(args.out_dir) if d.get("transcript")]
        key = blind_ids(ids)
        args.render_review.parent.mkdir(parents=True, exist_ok=True)
        args.render_review.write_text(render_blind_sheet(args.out_dir, key), encoding="utf-8")
        template = args.render_review.with_suffix(".labels.json")
        template.write_text(
            json.dumps(blind_label_template(key), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.out_dir.parent / "blind_key.json").write_text(
            json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"블라인드 시트 저장: {args.render_review}\n라벨 파일: {template}")
        return 0

    if args.render_review is not None:
        args.render_review.parent.mkdir(parents=True, exist_ok=True)
        args.render_review.write_text(render_review_sheet(args.out_dir, specs), encoding="utf-8")
        print(f"검수 시트 저장: {args.render_review}")
        return 0

    selected = select_specs(specs, args)
    if not selected:
        print("생성할 스펙이 없다.", file=sys.stderr)
        return EXIT_REFUSED
    model = args.model or settings.generator_model
    effort = args.effort or settings.generator_effort
    price = price_for(model)
    if price is None:
        print(
            f"단가가 등록되지 않은 모델이라 비용을 추정할 수 없다: {model}\n"
            f"등록된 모델: {', '.join(PRICES)}",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    cache_dir = args.cache_dir or Path(settings.generation_cache_dir)
    diversity = assign_diversity(specs)  # 전체 스펙 기준이라 부분 실행도 같은 배정
    probe = Generator(
        _NoClient(),  # type: ignore[arg-type]
        model=model,
        effort=effort,
        cache_dir=cache_dir,
        max_attempts=args.max_attempts,
        diversity=diversity,
    )
    estimate = estimate_generation(probe, [s for s, _ in selected], price)
    print(format_estimate(model, effort, estimate, PRICES_AS_OF))
    comparison = [s.spec_id for s, cmp in selected if cmp]
    if comparison:
        print(f"  직접 작성 스펙 {comparison}은 비교용으로만 생성한다 (pilot 폴더).")
    if args.dry_run:
        print("  (dry-run: API를 호출하지 않았다)")
        return 0

    approved: float | None = args.approve_cost
    if approved is None or approved <= 0:
        print(
            "\n실행 거부: 비용 승인이 없다. 위 추정을 확인한 뒤 "
            "--approve-cost <USD>로 승인할 최대 금액을 지정한다.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if estimate.cost_high > approved:
        print(
            f"\n실행 거부: 추정 최대 비용 ${estimate.cost_high:.3f}가 승인액 ${approved:.2f}를 "
            "넘는다. --spec으로 줄이거나 승인액을 올린다.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if settings.anthropic_api_key is None:
        print("\n실행 거부: ANTHROPIC_API_KEY가 없다 (.env에 넣는다).", file=sys.stderr)
        return EXIT_REFUSED

    guard = CostGuard(approved, price)
    generator = Generator(
        client_factory(settings),
        model=model,
        effort=effort,
        cache_dir=cache_dir,
        guard=guard,
        max_attempts=args.max_attempts,
        diversity=diversity,
    )
    try:
        for spec, is_comparison in selected:
            result = generator.generate(spec, comparison_only=is_comparison)
            path = write_candidate(args.out_dir, result, spec, generator)
            warns = [c.code for c in result.checks if c.level == "warn"]
            print(
                f"{result.spec_id:<4} {result.status:<12} 시도 {result.attempts}회"
                f" 경고 {warns or '-'} → {path}"
            )
    except BudgetExceeded as exc:
        print(
            f"\n중단: {exc}. 끝낸 호출은 캐시에 있어 승인액을 올려 다시 실행하면 이어진다.",
            file=sys.stderr,
        )
        return EXIT_BUDGET

    summary = apply_batch_checks(args.out_dir, args.golden)
    for sid, problems in summary.items():
        for problem in problems:
            print(f"  {sid} 배치 점검 {problem}")
    usage = generator.usage_summary()
    print(
        f"usage: api_calls={usage['api_calls']} cache_hits={usage['cache_hits']}"
        f" in={usage['input_tokens']} out={usage['output_tokens']}"
        f" cost≈${guard.spent_usd:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
