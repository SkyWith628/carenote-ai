"""방문 1건당 호출 수·비용·지연 실측 (blueprint 8.1 표, 설계: docs/cost-latency-design.md).

운영 구성(Extractor B1 v4 → Sentinel LLM 보조 → Verifier → Re-listener 폴백)을 dev 케이스에
캐시 없이 실제로 호출해 단계별 호출 수·토큰·비용·지연을 잰다. 모든 API 호출을 `TimedClient`가
감싸 호출마다 시간을 기록한다. Extractor와 Sentinel, 검증 단계(전 필드)는 스레드로 병렬 실행해
실제 벽시계 시간을 재고, 순차 실행이었을 시간은 호출별 지연의 합으로 구한다.

  python -m carenote.evaluation.costlatency --dry-run --limit 10
  python -m carenote.evaluation.costlatency --limit 10 --approve-cost 0.15 --out OUT.json

종료 코드: 0 성공 · 2 실행 거부 · 3 비용 한도로 중단
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from carenote.config import Settings, get_settings
from carenote.drafts.api import SENTINEL_EFFORT, SENTINEL_MODEL
from carenote.drafts.pipeline import DraftPipeline
from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.loader import load_golden
from carenote.evaluation.schema import GoldenCase
from carenote.extraction.estimate import estimate_run
from carenote.extraction.flagger import FlaggingExtractor
from carenote.extraction.llm_extractor import (
    LLMExtractor,
    MessagesAPI,
    ParseClient,
    ParseResponse,
    build_client,
)
from carenote.extraction.pricing import PRICES_AS_OF, BudgetExceeded, cost_usd, price_for
from carenote.extraction.relistener import (
    LLMReadback,
    TextRelistener,
    estimate_relistener_run,
)
from carenote.extraction.schema import LLMExtractionB1
from carenote.extraction.sentinel import load_rules
from carenote.extraction.sentinel_llm import LLMSentinelOutput, SentinelAssist
from carenote.extraction.verifier import (
    LLMVerdict,
    Verifier,
    VerifyingExtractor,
    estimate_verifier_run,
    needs_verification,
)

DEFAULT_GOLDEN = Path("data/golden")
RUN_TAG = "cl1"
EXIT_REFUSED = 2
EXIT_BUDGET = 3
SENTINEL_OUT_HIGH = 1500
COMPONENTS = ("extractor", "sentinel", "verifier", "relistener")
_FORMAT_TO_COMPONENT: dict[type[BaseModel], str] = {
    LLMExtractionB1: "extractor",
    LLMSentinelOutput: "sentinel",
    LLMVerdict: "verifier",
    LLMReadback: "relistener",
}


@dataclass(frozen=True)
class CallRecord:
    """API 호출 하나의 기록."""

    case_id: str
    component: str
    model: str
    seconds: float
    input_tokens: int
    output_tokens: int
    cost: float


class TimedClient:
    """호출마다 지연·토큰·비용을 기록하고, 누적 비용이 한도에 닿으면 다음 호출 전에 멈춘다."""

    def __init__(self, inner: ParseClient, limit_usd: float) -> None:
        self._inner = inner
        self._limit = limit_usd
        self._lock = Lock()
        self.records: list[CallRecord] = []
        self.spent = 0.0
        self.case_id = ""
        self.messages: MessagesAPI = self  # type: ignore[assignment]

    def parse(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
        output_format: type[BaseModel],
        output_config: dict[str, str],
    ) -> ParseResponse:
        with self._lock:
            if self.spent >= self._limit:
                raise BudgetExceeded(
                    f"승인 한도 ${self._limit:.2f}에 도달 (사용 ${self.spent:.4f})"
                )
            case_id = self.case_id
        started = time.perf_counter()
        response = self._inner.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_format=output_format,
            output_config=output_config,
        )
        seconds = time.perf_counter() - started
        price = price_for(model)
        usage = response.usage
        cost = cost_usd(price, usage.input_tokens, usage.output_tokens) if price else 0.0
        with self._lock:
            self.spent += cost
            self.records.append(
                CallRecord(
                    case_id,
                    _FORMAT_TO_COMPONENT.get(output_format, "other"),
                    model,
                    seconds,
                    usage.input_tokens,
                    usage.output_tokens,
                    cost,
                )
            )
        return response


@dataclass
class VisitTiming:
    """한 방문의 단계별 벽시계 시간(병렬 실행 실측)."""

    case_id: str
    stage1_s: float = 0.0  # Extractor ‖ Sentinel
    stage2_s: float = 0.0  # 위험 필드 전체의 Verifier ‖ Re-listener
    total_s: float = 0.0
    targets: int = 0


@dataclass
class Measurement:
    """실측 결과."""

    visits: list[VisitTiming] = field(default_factory=lambda: [])
    records: list[CallRecord] = field(default_factory=lambda: [])


def run_visit(
    transcript: str,
    case_id: str,
    llm: LLMExtractor,
    assist: SentinelAssist,
    verifier: Verifier,
    relistener: TextRelistener,
    pool: ThreadPoolExecutor,
) -> VisitTiming:
    """방문 1건을 운영과 같은 순서로 실행한다 (1단계 병렬 → 2단계 병렬)."""
    started = time.perf_counter()
    fields_future = pool.submit(llm.extract, transcript)
    pool.submit(assist.assess, transcript).result()
    fields: list[PredictedField] = fields_future.result()
    stage1 = time.perf_counter() - started
    targets = [f for f in fields if needs_verification(f)]
    started2 = time.perf_counter()
    futures = [pool.submit(verifier.verify, transcript, f) for f in targets]
    futures += [pool.submit(relistener.verify, transcript, f) for f in targets]
    for future in futures:
        future.result()
    stage2 = time.perf_counter() - started2
    return VisitTiming(case_id, stage1, stage2, stage1 + stage2, len(targets))


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarize(m: Measurement) -> dict[str, Any]:
    """구성별(누적) 방문당 호출 수·비용·지연을 요약한다.

    순차 지연 = 방문별 호출 지연의 합. 병렬 지연 = 실측 벽시계 시간(단계 사이는 순차).
    구성 행의 지연은 그 구성까지의 단계만 더한 값이다.
    """
    visits = len(m.visits)
    by_comp: dict[str, list[CallRecord]] = {c: [] for c in COMPONENTS}
    for r in m.records:
        by_comp.setdefault(r.component, []).append(r)

    def per_visit(component: str, attr: str) -> float:
        return sum(getattr(r, attr) for r in by_comp[component]) / visits if visits else 0.0

    calls = {c: len(by_comp[c]) / visits if visits else 0.0 for c in COMPONENTS}
    cost = {c: per_visit(c, "cost") for c in COMPONENTS}
    seq = {c: per_visit(c, "seconds") for c in COMPONENTS}
    stage1 = [v.stage1_s for v in m.visits]
    stage2 = [v.stage2_s for v in m.visits]
    total = [v.total_s for v in m.visits]
    # 병렬 지연: 추출만 = Extractor 호출 시간, +Sentinel = 1단계 벽시계, 이후 = 총 벽시계
    ext_only = [
        sum(r.seconds for r in by_comp["extractor"] if r.case_id == v.case_id) for v in m.visits
    ]
    cumulative: list[tuple[str, tuple[str, ...]]] = [
        ("Extractor(B1 v4)", ("extractor",)),
        ("+ Sentinel LLM 보조", ("extractor", "sentinel")),
        ("+ Verifier", ("extractor", "sentinel", "verifier")),
        ("+ Re-listener 폴백", COMPONENTS),
    ]
    rows: list[dict[str, Any]] = []
    for name, comps in cumulative:
        if comps == ("extractor",):
            parallel = statistics.fmean(ext_only) if ext_only else 0.0
        elif comps == ("extractor", "sentinel"):
            parallel = statistics.fmean(stage1) if stage1 else 0.0
        elif comps == COMPONENTS:
            parallel = statistics.fmean(total) if total else 0.0  # 실측 벽시계
        else:
            # 일부 검증 단계만 켠 구성은 실측하지 않았다. 1단계 + 가장 느린 호출로 추정한다
            # (전체 구성에서 이 모델의 추정은 실측 벽시계와 맞는지 검증됨: 테스트·문서 참고).
            extra = [
                max(
                    (r.seconds for r in by_comp["verifier"] if r.case_id == v.case_id),
                    default=0.0,
                )
                for v in m.visits
            ]
            parallel = (
                statistics.fmean(s + e for s, e in zip(stage1, extra, strict=True))
                if stage1
                else 0.0
            )
        rows.append(
            {
                "config": name,
                "calls_per_visit": round(sum(calls[c] for c in comps), 2),
                "cost_per_visit_usd": round(sum(cost[c] for c in comps), 5),
                "latency_sequential_s": round(sum(seq[c] for c in comps), 2),
                "latency_parallel_s": round(parallel, 2),
                "latency_parallel_measured": comps
                in (("extractor",), ("extractor", "sentinel"), COMPONENTS),
            }
        )
    call_stats: dict[str, dict[str, float]] = {}
    for c in COMPONENTS:
        secs = [r.seconds for r in by_comp[c]]
        if secs:
            call_stats[c] = {
                "n": float(len(secs)),
                "mean_s": round(statistics.fmean(secs), 2),
                "p50_s": round(_percentile(secs, 0.5), 2),
                "p95_s": round(_percentile(secs, 0.95), 2),
                "max_s": round(max(secs), 2),
            }
    return {
        "visits": visits,
        "rows": rows,
        "per_call": call_stats,
        "wall_clock_s": {
            "total_mean": round(statistics.fmean(total), 2) if total else 0.0,
            "total_p95": round(_percentile(total, 0.95), 2) if total else 0.0,
            "stage1_mean": round(statistics.fmean(stage1), 2) if stage1 else 0.0,
            "stage2_mean": round(statistics.fmean(stage2), 2) if stage2 else 0.0,
        },
        "targets_per_visit": round(statistics.fmean(v.targets for v in m.visits), 2)
        if m.visits
        else 0.0,
        "total_cost_usd": round(sum(r.cost for r in m.records), 5),
        "prices_as_of": PRICES_AS_OF,
    }


def format_summary(s: dict[str, Any]) -> str:
    """사람이 읽는 표."""
    lines = [
        f"방문 {s['visits']}건 · 방문당 위험 필드 {s['targets_per_visit']}개"
        f" · 총 ${s['total_cost_usd']}"
    ]
    lines.append(
        f"{'구성(누적)':<22}{'호출/건':>8}{'비용/건':>12}{'지연 순차':>10}{'지연 병렬':>10}"
    )
    for r in s["rows"]:
        cost = "$" + format(r["cost_per_visit_usd"], ".4f")
        lines.append(
            f"{r['config']:<22}{r['calls_per_visit']:>8}{cost:>12}"
            f"{r['latency_sequential_s']:>9}s{r['latency_parallel_s']:>9}s"
        )
    for c, st in s["per_call"].items():
        lines.append(
            f"  {c:<11} n={int(st['n'])} 평균 {st['mean_s']}s p50 {st['p50_s']}s p95 {st['p95_s']}s"
            f" 최대 {st['max_s']}s"
        )
    w = s["wall_clock_s"]
    lines.append(f"  실측 벽시계(병렬): 평균 {w['total_mean']}s p95 {w['total_p95']}s")
    return "\n".join(lines)


class _NoClient:
    """dry-run용. 호출되면 버그이므로 즉시 실패한다."""

    @property
    def messages(self) -> object:
        raise RuntimeError("dry-run에서 API 클라이언트를 쓰면 안 된다")


def _select(golden: Path, limit: int | None, case_ids: str | None) -> list[GoldenCase]:
    cases = load_golden(golden, "dev")
    if case_ids:
        wanted = {c.strip() for c in case_ids.split(",") if c.strip()}
        unknown = sorted(wanted - {c.id for c in cases})
        if unknown:
            raise ValueError(f"dev에 없는 케이스: {unknown}")
        cases = [c for c in cases if c.id in wanted]
    return cases if limit is None else cases[:limit]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="carenote.evaluation.costlatency", description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--limit", type=int, default=10, help="앞에서 N건 (기본 10)")
    parser.add_argument("--case-ids", default=None, metavar="ID,ID")
    parser.add_argument(
        "--production",
        action="store_true",
        help="구성 요소별 모델 대신 운영 경로(DraftPipeline, LangGraph)의 end-to-end 지연을 잰다",
    )
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 예상 비용만 출력")
    parser.add_argument("--approve-cost", type=float, default=None, metavar="USD")
    parser.add_argument("--out", type=Path, default=None, help="JSON 보고서 저장 경로")
    return parser


def _estimate_high(settings: Settings, cases: list[GoldenCase]) -> tuple[float, list[str]]:
    """네트워크 없이 추정 최대 비용과 구성별 설명. 추출 캐시(태그 없음)로 위험 필드 수를 안다."""
    extractor_price = price_for(settings.extractor_model)
    sentinel_price = price_for(SENTINEL_MODEL)
    if extractor_price is None or sentinel_price is None:
        raise ValueError("단가가 등록되지 않은 모델이 있다")
    cached = LLMExtractor(
        _NoClient(),  # type: ignore[arg-type]
        variant="b1",
        model=settings.extractor_model,
        effort=settings.extractor_effort,
        cache_dir=Path(settings.extraction_cache_dir),
    )
    fresh = LLMExtractor(
        _NoClient(),  # type: ignore[arg-type]
        variant="b1",
        model=settings.extractor_model,
        effort=settings.extractor_effort,
        cache_dir=None,
        run_tag=RUN_TAG,
    )
    ext = estimate_run(fresh, cases, extractor_price)
    sent_in = sum(int(len(c.transcript) / 1.5) + 1200 for c in cases)
    sent_high = cost_usd(sentinel_price, int(sent_in * 1.3), SENTINEL_OUT_HIGH * len(cases))
    transcripts = [c.transcript for c in cases]
    flagging = FlaggingExtractor(cached, None)
    ver = estimate_verifier_run(
        Verifier(
            _NoClient(),  # type: ignore[arg-type]
            model=settings.verifier_model,
            effort=settings.verifier_effort,
            run_tag=RUN_TAG,
        ),
        flagging,
        cached.is_cached,
        transcripts,
        settings.verifier_model,
    )
    rel = estimate_relistener_run(
        TextRelistener(
            _NoClient(),  # type: ignore[arg-type]
            model=settings.relistener_model,
            effort=settings.relistener_effort,
            run_tag=RUN_TAG,
        ),
        flagging,
        cached.is_cached,
        transcripts,
        settings.relistener_model,
    )
    notes = [
        f"Extractor {settings.extractor_model} ${ext.cost_low:.3f}~${ext.cost_high:.3f}"
        f" ({ext.to_call}건)",
        f"Sentinel {SENTINEL_MODEL} ~${sent_high:.3f}",
        f"Verifier ${ver.cost_low:.4f}~${ver.cost_high:.4f} ({ver.calls}건)",
        f"Re-listener ${rel.cost_low:.4f}~${rel.cost_high:.4f} ({rel.calls}건)",
    ]
    return ext.cost_high + sent_high + ver.cost_high + rel.cost_high, notes


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    client_factory: Callable[[Settings], ParseClient] = build_client,
) -> int:
    """CLI 진입점. settings·client_factory는 테스트에서 바꿔 끼운다."""
    args = _build_parser().parse_args(argv)
    settings = settings or get_settings()
    try:
        cases = _select(args.golden, args.limit, args.case_ids)
        high, notes = _estimate_high(settings, cases)
    except ValueError as exc:
        print(f"실행 거부: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"[dry-run] 방문 {len(cases)}건 · 캐시 없이 전 구성 호출 (단가 기준일 {PRICES_AS_OF})")
    for note in notes:
        print(f"  {note}")
    print(f"  합계 최대 ${high:.3f}")
    if args.dry_run:
        print("  (dry-run: API를 호출하지 않았다)")
        return 0
    approved: float | None = args.approve_cost
    if approved is None or approved <= 0:
        print("\n실행 거부: 비용 승인이 없다. --approve-cost <USD>로 지정한다.", file=sys.stderr)
        return EXIT_REFUSED
    if high > approved:
        print(
            f"\n실행 거부: 추정 최대 비용 ${high:.3f}가 승인액 ${approved:.2f}를 넘는다. "
            "--limit으로 줄이거나 승인액을 올린다.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if settings.anthropic_api_key is None:
        print("\n실행 거부: ANTHROPIC_API_KEY가 없다.", file=sys.stderr)
        return EXIT_REFUSED

    timed = TimedClient(client_factory(settings), approved)
    client: ParseClient = timed  # type: ignore[assignment]
    llm = LLMExtractor(
        client,
        variant="b1",
        model=settings.extractor_model,
        effort=settings.extractor_effort,
        cache_dir=None,
        run_tag=RUN_TAG,
    )
    assist = SentinelAssist(
        client,
        model=SENTINEL_MODEL,
        effort=SENTINEL_EFFORT,
        rules=load_rules(),
        run_tag=RUN_TAG,
    )
    verifier = Verifier(
        client,
        model=settings.verifier_model,
        effort=settings.verifier_effort,
        run_tag=RUN_TAG,
    )
    relistener = TextRelistener(
        client,
        model=settings.relistener_model,
        effort=settings.relistener_effort,
        run_tag=RUN_TAG,
    )
    if args.production:
        return _run_production(args, cases, timed, llm, assist, verifier, relistener)
    measurement = Measurement()
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            for case in cases:
                timed.case_id = case.id
                measurement.visits.append(
                    run_visit(case.transcript, case.id, llm, assist, verifier, relistener, pool)
                )
    except BudgetExceeded as exc:
        print(f"\n중단: {exc}. 지금까지의 기록만 요약한다.", file=sys.stderr)
        measurement.records = list(timed.records)
        print(format_summary(summarize(measurement)))
        return EXIT_BUDGET
    measurement.records = list(timed.records)
    summary = summarize(measurement)
    print(format_summary(summary))
    if args.out is not None:
        payload = {"summary": summary, "calls": [r.__dict__ for r in measurement.records]}
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _run_production(
    args: argparse.Namespace,
    cases: list[GoldenCase],
    timed: TimedClient,
    llm: LLMExtractor,
    assist: SentinelAssist,
    verifier: Verifier,
    relistener: TextRelistener,
) -> int:
    """운영 경로(`/drafts`와 같은 DraftPipeline, LangGraph)를 방문마다 한 번 실행해 벽시계를 잰다.

    그래프는 사람 검토 interrupt까지 실행하므로 결정 대기 상태까지 포함한 시간이다.
    """
    extractor = VerifyingExtractor(
        FlaggingExtractor(llm, load_rules(), assist),
        {"verifier": verifier, "relistener": relistener},
    )
    pipeline = DraftPipeline(extractor, InMemorySaver())
    walls: list[float] = []
    try:
        for case in cases:
            timed.case_id = case.id
            started = time.perf_counter()
            pipeline.create(case.transcript)
            walls.append(time.perf_counter() - started)
    except BudgetExceeded as exc:
        print(f"\n중단: {exc}", file=sys.stderr)
        return EXIT_BUDGET
    visits = len(walls)
    sequential = sum(r.seconds for r in timed.records) / visits
    summary = {
        "mode": "production",
        "path": "DraftPipeline(LangGraph)",
        "visits": visits,
        "wall_mean_s": round(statistics.fmean(walls), 2),
        "wall_p95_s": round(_percentile(walls, 0.95), 2),
        "wall_max_s": round(max(walls), 2),
        "sequential_sum_s": round(sequential, 2),
        "calls_per_visit": round(len(timed.records) / visits, 2),
        "cost_per_visit_usd": round(sum(r.cost for r in timed.records) / visits, 5),
        "total_cost_usd": round(sum(r.cost for r in timed.records), 5),
        "walls_s": [round(w, 2) for w in walls],
    }
    print(
        f"운영 경로 {visits}건: 벽시계 평균 {summary['wall_mean_s']}s p95 {summary['wall_p95_s']}s "
        f"최대 {summary['wall_max_s']}s · 순차로 돌렸을 합 {summary['sequential_sum_s']}s\n"
        f"  호출 {summary['calls_per_visit']}회/건 · ${summary['cost_per_visit_usd']}/건 "
        f"· 총 ${summary['total_cost_usd']}"
    )
    if args.out is not None:
        payload = {"summary": summary, "calls": [r.__dict__ for r in timed.records]}
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
