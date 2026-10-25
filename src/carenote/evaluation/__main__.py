"""평가 CLI.

기준선:  python -m carenote.evaluation --extractor oracle|null
LLM 실측: python -m carenote.evaluation --extractor llm-b0|llm-b1 --dry-run
          python -m carenote.evaluation --extractor llm-b1 --approve-cost 1.00

분할: --split dev(기본)|test|all. test는 최종 보고용이라 LLM 실행에 --final-report가 필요하다.
구성 간 비교: python -m carenote.evaluation.compare A.json B.json

LLM 실행은 비용이 든다. --dry-run은 네트워크를 쓰지 않고 예상 비용만 보여준다.
실제 실행은 API 키와 --approve-cost(승인할 최대 USD)가 모두 있어야 시작하고,
추정 최대 비용이 승인액을 넘거나 실행 중 누적 비용이 승인액에 도달하면 멈춘다.

종료 코드: 0 성공 · 2 실행 거부(키·승인·단가·test 누수 가드) · 3 비용 한도로 중단
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from carenote.config import Settings, get_settings
from carenote.evaluation.extractor import Extractor, NullExtractor, OracleExtractor
from carenote.evaluation.loader import SplitChoice, load_golden
from carenote.evaluation.metrics import Metrics, Ratio
from carenote.evaluation.runner import EvalReport, evaluate
from carenote.evaluation.schema import GoldenCase
from carenote.extraction.estimate import estimate_run, format_estimate
from carenote.extraction.flagger import FlaggingExtractor
from carenote.extraction.llm_extractor import LLMExtractor, ParseClient, build_client
from carenote.extraction.pricing import (
    PRICES,
    PRICES_AS_OF,
    BudgetExceeded,
    CostGuard,
    price_for,
)
from carenote.extraction.prompts import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_VERSIONS,
    PromptVersion,
    Variant,
)

DEFAULT_GOLDEN = Path("data/golden")
EXIT_REFUSED = 2
EXIT_BUDGET = 3
LLM_CHOICES = {"llm-b0": "b0", "llm-b1": "b1"}


class _NoClient:
    """dry-run용. 호출되면 버그이므로 즉시 실패한다."""

    @property
    def messages(self) -> object:
        raise RuntimeError("dry-run에서 API 클라이언트를 쓰면 안 된다")


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _ratio(r: Ratio) -> str:
    return f"{_pct(r.value)} ({r.num}/{r.den})"


def _row(label: str, m: Metrics) -> str:
    return (
        f"{label:<18}{_pct(m.f1):>8}  {_ratio(m.ungrounded):<16}{_ratio(m.hallucinated):<16}"
        f"{_ratio(m.numeric_error):<16}{_ratio(m.risk_recall):<16}{_ratio(m.confirm_recall)}"
    )


def _abstain_line(m: Metrics) -> str:
    return f"기권 {_ratio(m.abstained)} · 오답 {_ratio(m.wrong_value)}"


def format_report(report: EvalReport) -> str:
    """사람이 읽는 표."""
    header = (
        f"{'':<18}{'F1':>8}  {'근거없음':<16}{'환각':<16}{'수치오류':<16}"
        f"{'위험recall':<16}{'확인recall'}"
    )
    lines = [f"cases: {report.case_count} (split={report.split})", header]
    lines.append(_row("overall", report.overall))
    if (ci := report.f1_ci) is not None and ci.low is not None and ci.high is not None:
        span = f"{_pct(ci.low).strip()} ~ {_pct(ci.high).strip()}"
        lines.append(f"  F1 95% 구간(케이스 부트스트랩): {span}")
    lines.append("  " + _abstain_line(report.overall))
    if report.recoverable is not None:
        lines.append(_row("복원가능 필드만", report.recoverable))
    if report.strict is not None:
        lines.append(_row("복원가능·gray 제외", report.strict))
    lines.append("-- 유형별")
    lines += [_row(name, m) for name, m in report.by_category.items()]
    if len(report.by_probe) > 1:
        lines.append("-- 프로브별")
        lines += [_row(name, m) for name, m in report.by_probe.items()]
    if report.usage:
        u = report.usage
        lines.append(
            f"usage: calls={u.get('calls')} cache_hits={u.get('cache_hits')} "
            f"failures={u.get('failures')} demoted={u.get('demoted_fields')} "
            f"in={u.get('input_tokens')} out={u.get('output_tokens')} "
            f"latency={float(str(u.get('latency_s') or 0)):.1f}s "
            f"cost≈${u.get('estimated_cost_usd')}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="carenote.evaluation", description=__doc__)
    parser.add_argument("--extractor", choices=["oracle", "null", *LLM_CHOICES], required=True)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="dev",
        help="평가할 분할. 기본 dev. test·all은 최종 보고용이라 LLM 실행에 --final-report 필요",
    )
    parser.add_argument(
        "--final-report",
        action="store_true",
        help="test 분할을 포함한 LLM 실행을 허용한다 (최종 ablation 보고 때만 쓴다)",
    )
    parser.add_argument("--out", type=Path, default=None, help="JSON 보고서 저장 경로")
    parser.add_argument("--model", default=None, help="기본값은 설정(CARENOTE_EXTRACTOR_MODEL)")
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=None,
        help="기본값은 설정(CARENOTE_EXTRACTOR_EFFORT)",
    )
    parser.add_argument(
        "--prompt",
        choices=PROMPT_VERSIONS,
        default=DEFAULT_PROMPT_VERSION,
        help="추출 프롬프트 버전 (기본 v1: 기존 보고서 재현용). v2는 라벨 정책 반영판",
    )
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N건만 실행")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 예상 비용만 출력")
    parser.add_argument(
        "--approve-cost",
        type=float,
        default=None,
        metavar="USD",
        help="승인하는 최대 비용(USD). 없으면 LLM 실행을 거부한다",
    )
    return parser


def _select_cases(golden: Path, limit: int | None, split: SplitChoice) -> list[GoldenCase]:
    cases = load_golden(golden, split)
    return cases if limit is None else cases[:limit]


def _run_llm(
    args: argparse.Namespace,
    settings: Settings,
    client_factory: Callable[[Settings], ParseClient],
) -> int:
    model = args.model or settings.extractor_model
    effort = args.effort or settings.extractor_effort
    variant: Variant = "b1" if LLM_CHOICES[args.extractor] == "b1" else "b0"
    prompt_version: PromptVersion = args.prompt
    price = price_for(model)
    if price is None:
        print(
            f"단가가 등록되지 않은 모델이라 비용을 추정할 수 없다: {model}\n"
            f"등록된 모델: {', '.join(PRICES)} (pricing.py에 추가한 뒤 사용)",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    if args.split != "dev" and not args.final_report:
        print(
            f"실행 거부: --split {args.split}는 test 케이스를 포함한다. test는 프롬프트·규칙을 "
            "고치는 데 쓰면 안 되므로 최종 보고 때만 --final-report와 함께 실행한다 "
            "(실행 횟수를 docs/decisions.md에 기록).",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    cases = _select_cases(args.golden, args.limit, args.split)
    cache_dir = args.cache_dir or Path(settings.extraction_cache_dir)
    probe = LLMExtractor(
        _NoClient(),  # type: ignore[arg-type]
        variant=variant,
        model=model,
        effort=effort,
        cache_dir=cache_dir,
        prompt_version=prompt_version,
    )
    estimate = estimate_run(probe, cases, price)
    print(format_estimate(model, effort, estimate, PRICES_AS_OF))
    print(f"  프롬프트 {prompt_version} · 변형 {variant}")
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
            "넘는다. --limit으로 줄이거나 승인액을 올린다.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if settings.anthropic_api_key is None:
        print("\n실행 거부: ANTHROPIC_API_KEY가 없다 (.env에 넣는다).", file=sys.stderr)
        return EXIT_REFUSED

    guard = CostGuard(approved, price)
    llm = LLMExtractor(
        client_factory(settings),
        variant=variant,
        model=model,
        effort=effort,
        cache_dir=cache_dir,
        guard=guard,
        prompt_version=prompt_version,
    )
    extractor: Extractor = FlaggingExtractor(llm)
    try:
        report = evaluate(extractor, cases, args.split)
    except BudgetExceeded as exc:
        print(
            f"\n중단: {exc}. 끝낸 호출은 캐시에 저장됐으니 승인액을 올려 다시 실행하면 "
            "이어서 진행된다.",
            file=sys.stderr,
        )
        return EXIT_BUDGET
    _emit(report, args.out)
    return 0


def _emit(report: EvalReport, out: Path | None) -> None:
    print(format_report(report))
    if out is not None:
        out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    client_factory: Callable[[Settings], ParseClient] = build_client,
) -> int:
    """CLI 진입점. settings·client_factory는 테스트에서 바꿔 끼운다."""
    args = _build_parser().parse_args(argv)
    if args.extractor in LLM_CHOICES:
        return _run_llm(args, settings or get_settings(), client_factory)

    cases = _select_cases(args.golden, args.limit, args.split)
    extractor: Extractor = OracleExtractor(cases) if args.extractor == "oracle" else NullExtractor()
    _emit(evaluate(extractor, cases, args.split), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
