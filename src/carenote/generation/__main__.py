"""생성 CLI.

dry-run:  python -m carenote.generation --pilot --dry-run
파일럿:   python -m carenote.generation --pilot --approve-cost 0.50
검수 시트: python -m carenote.generation --render-review docs/reports/v2-review.md

비용이 드는 실행은 --approve-cost(승인할 최대 USD)와 API 키가 모두 있어야 시작한다.
추정 최대 비용이 승인액을 넘으면 거부하고, 실행 중 누적 비용이 승인액에 도달하면 중단한다.
종료 코드: 0 성공 · 2 실행 거부 · 3 비용 한도로 중단 · 4 API 연속 실패로 중단
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
from carenote.generation.blind_labeler import (
    BlindLabeler,
    LabeledCase,
    estimate_labeling,
    to_label_file,
)
from carenote.generation.candidates import (
    BlindReport,
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
from carenote.generation.generator import DEFAULT_MAX_ATTEMPTS, ApiUnavailable, Generator
from carenote.generation.promote import promote
from carenote.generation.spec import Spec, load_specs

DEFAULT_SPEC_DIR = Path("data/golden/specs")
DEFAULT_OUT_DIR = Path("data/golden/candidates/v2")
DEFAULT_GOLDEN = Path("data/golden")
PILOT_IDS = ("H3", "S1", "L1")
EXIT_REFUSED = 2
EXIT_BUDGET = 3
EXIT_API = 4


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
    parser.add_argument(
        "--compare-blind",
        type=Path,
        nargs="+",
        default=None,
        metavar="LABELS_JSON",
        help="라벨 파일(들)을 정답과 비교한다. 여러 개를 주면 라운드별 결과와 합산을 낸다",
    )
    parser.add_argument(
        "--exclude-spec",
        action="append",
        default=[],
        metavar="SPEC_ID",
        help="--compare-blind에서 뺄 스펙(교체된 케이스의 옛 라벨). 여러 번 지정 가능",
    )
    parser.add_argument(
        "--include-unrecoverable",
        action="store_true",
        help="--compare-blind에 복원 불가 필드(recoverable=false)도 포함한다 (기본은 제외)",
    )
    parser.add_argument(
        "--model-label",
        type=Path,
        default=None,
        metavar="OUT_JSON",
        help="정답을 모르는 새 맥락의 모델이 --blind-sheet의 후보를 라벨한다 (비용이 든다)",
    )
    parser.add_argument("--blind-sheet", type=Path, default=None, metavar="SHEET_MD")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="후보+직접 작성분을 data/golden/v2_dev.jsonl, v2_test.jsonl로 승격한다 (API 없음)",
    )
    parser.add_argument("--human-dir", type=Path, default=Path("data/golden/human"))
    return parser


def _blind_key_path(out_dir: Path, labels_or_template: Path, *, for_write: bool = False) -> Path:
    """블라인드 키 파일 경로. 라벨 파일 이름(`v2-blind2.labels.json`)마다 키를 따로 둔다.

    첫 라운드(`v2-blind`)는 호환을 위해 기존 `blind_key.json`을 쓴다. 키는 후보 폴더 밖에 두어
    후보로 오인되지 않게 하고, 새 라운드가 이전 라운드의 키를 덮어쓰지 않게 한다.
    """
    stem = (
        labels_or_template.name.removesuffix(".json")
        .removesuffix(".labels")
        .removesuffix(".model-labels")
        .removesuffix(".md")
    )
    legacy = out_dir.parent / "blind_key.json"
    per_round = out_dir.parent / f"blind_key.{stem}.json"
    if stem == "v2-blind":
        return legacy
    if for_write or per_round.exists():
        return per_round
    return legacy


def _run_model_label(
    args: argparse.Namespace,
    settings: Settings,
    client_factory: Callable[[Settings], ParseClient],
) -> int:
    """블라인드 시트의 후보를 정답을 모르는 모델이 라벨한다 (승인·비용 가드는 생성과 같다)."""
    if args.blind_sheet is None:
        print("--blind-sheet가 필요하다 (예: docs/reports/v2-blind2.md)", file=sys.stderr)
        return EXIT_REFUSED
    key_path = _blind_key_path(args.out_dir, args.blind_sheet.with_suffix(".labels.json"))
    key: dict[str, str] = json.loads(key_path.read_text(encoding="utf-8"))
    transcripts = {
        blind: str(d["transcript"])
        for d in load_candidates(args.out_dir)
        if (blind := key.get(str(d["spec_id"]))) is not None and d.get("transcript")
    }
    model = args.model or settings.generator_model
    effort = args.effort or settings.generator_effort
    price = price_for(model)
    if price is None:
        print(f"단가가 등록되지 않은 모델: {model}", file=sys.stderr)
        return EXIT_REFUSED
    cache_dir = args.cache_dir or Path(settings.generation_cache_dir).parent / "blind_labels"
    no_client: ParseClient = _NoClient()  # type: ignore[assignment]
    probe = BlindLabeler(no_client, model=model, effort=effort, cache_dir=cache_dir)
    pending, cached, low, high = estimate_labeling(probe, list(transcripts.values()), price)
    print(
        f"[dry-run] 블라인드 라벨러 {model} · effort {effort}\n"
        f"  케이스 {len(transcripts)}건 중 캐시 적중 {cached}건, API 호출 예정 {pending}건\n"
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
    if settings.anthropic_api_key is None:
        print("\n실행 거부: ANTHROPIC_API_KEY가 없다.", file=sys.stderr)
        return EXIT_REFUSED

    guard = CostGuard(approved, price)
    labeler = BlindLabeler(
        client_factory(settings), model=model, effort=effort, cache_dir=cache_dir, guard=guard
    )
    labeled: dict[str, LabeledCase] = {}
    try:
        for blind in sorted(transcripts):
            case = labeler.label(transcripts[blind])
            if case is None:
                print(f"{blind} 라벨 실패: {labeler.stats.last_error}")
                continue
            labeled[blind] = case
    except ApiUnavailable as exc:
        print(f"\n중단: {exc}", file=sys.stderr)
        return EXIT_API
    except BudgetExceeded as exc:
        print(f"\n중단: {exc}. 끝낸 호출은 캐시에 있다.", file=sys.stderr)
        return EXIT_BUDGET
    out = to_label_file(labeled, f"{model} (블라인드, 새 맥락, 정답 미노출)")
    args.model_label.parent.mkdir(parents=True, exist_ok=True)
    args.model_label.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    usage = labeler.usage_summary()
    print(
        f"라벨 {len(labeled)}/{len(transcripts)}건 저장: {args.model_label}\n"
        f"usage: api_calls={usage['api_calls']} cache_hits={usage['cache_hits']}"
        f" in={usage['input_tokens']} out={usage['output_tokens']} cost≈${guard.spent_usd:.4f}"
    )
    return 0


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

    if args.model_label is not None:
        return _run_model_label(args, settings, client_factory)

    if args.promote:
        try:
            done = promote(specs, args.out_dir, args.human_dir, args.golden)
        except ValueError as exc:
            print(f"승격 실패: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        for split, ids in done.items():
            print(f"{split}: {len(ids)}건 → {args.golden / f'v2_{split}.jsonl'}")
        return 0

    if args.compare_blind is not None:
        reports: list[BlindReport] = []
        for path in args.compare_blind:
            key = json.loads(_blind_key_path(args.out_dir, path).read_text(encoding="utf-8"))
            labels = json.loads(path.read_text(encoding="utf-8"))
            report = compare_blind(
                labels,
                key,
                specs,
                exclude_specs=frozenset(args.exclude_spec),
                exclude_unrecoverable=not args.include_unrecoverable,
            )
            reports.append(report)
            if len(args.compare_blind) > 1:
                print(f"== {path.name}")
            print(format_blind_report(report))
        if len(reports) > 1:
            pooled = reports[0]
            for extra in reports[1:]:
                pooled = pooled + extra
            print("== 합산 (참고: 통과 판정은 동결 절차의 F2 정의를 따른다)")
            print(format_blind_report(pooled))
        return 0

    if args.render_review is not None and args.blind:
        ids = [str(d["spec_id"]) for d in load_candidates(args.out_dir) if d.get("transcript")]
        if args.spec:  # 표본 지정: --spec으로 고른 후보만 시트에 넣는다
            missing = sorted(set(args.spec) - set(ids))
            if missing:
                print(f"후보가 없는 spec_id: {missing}", file=sys.stderr)
                return EXIT_REFUSED
            ids = [i for i in ids if i in set(args.spec)]
        key = blind_ids(ids)
        args.render_review.parent.mkdir(parents=True, exist_ok=True)
        args.render_review.write_text(render_blind_sheet(args.out_dir, key), encoding="utf-8")
        template = args.render_review.with_suffix(".labels.json")
        template.write_text(
            json.dumps(blind_label_template(key), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        key_file = _blind_key_path(args.out_dir, template, for_write=True)
        key_file.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"블라인드 시트 저장: {args.render_review} ({len(key)}건)\n라벨 파일: {template}")
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
            if result.status == "api_failure":
                # 실패한 호출로 기존 후보를 덮어쓰거나 빈 껍데기 파일을 남기지 않는다
                print(f"{result.spec_id:<4} api_failure  {result.error}")
                continue
            path = write_candidate(args.out_dir, result, spec, generator)
            warns = [c.code for c in result.checks if c.level == "warn"]
            print(
                f"{result.spec_id:<4} {result.status:<12} 시도 {result.attempts}회"
                f" 경고 {warns or '-'} → {path}"
            )
    except ApiUnavailable as exc:
        print(
            f"\n중단: {exc}. 끝낸 호출은 캐시에 있어 원인을 해결한 뒤 다시 실행하면 이어진다.",
            file=sys.stderr,
        )
        return EXIT_API
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
