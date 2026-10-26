"""평가 러너: extractor를 골든셋에 돌려 지표를 낸다."""

from __future__ import annotations

from dataclasses import dataclass, field

from carenote.evaluation.extractor import Extractor, SupportsUsage
from carenote.evaluation.metrics import Counts, Metrics, Ratio, Subset, field_outcomes, score_case
from carenote.evaluation.schema import GoldenCase
from carenote.evaluation.stats import Interval, bootstrap_f1

UNTAGGED = "v1"


def probe_of(case: GoldenCase) -> str:
    """프로브 코드(태그의 첫 항목). v1처럼 태그가 없으면 `v1`."""
    return case.tags[0] if case.tags else UNTAGGED


@dataclass(frozen=True)
class EvalReport:
    """전체·유형별·프로브별 지표와 신뢰구간."""

    case_count: int
    overall: Metrics
    by_category: dict[str, Metrics]
    usage: dict[str, object] | None = None
    split: str = "all"
    by_probe: dict[str, Metrics] = field(default_factory=lambda: dict[str, Metrics]())
    # 텍스트로 복원 불가능한 필드를 뺀 지표 / 거기에 논쟁적(gray) 필드까지 뺀 지표
    recoverable: Metrics | None = None
    strict: Metrics | None = None
    f1_ci: Interval | None = None
    # 헤드라인(사전 등록 docs/test-final-report-plan.md): 현장 검토 대기 케이스(review_required,
    # R3·G1)를 뺀 집계. 주 지표는 복원가능 필드만의 지표(headline)이고 복원 불가 포함과
    # gray 제외 지표도 함께 낸다.
    review_required_ids: tuple[str, ...] = ()
    headline: Metrics | None = None
    headline_all_fields: Metrics | None = None
    headline_strict: Metrics | None = None
    headline_ci: Interval | None = None
    per_case_recoverable: dict[str, Counts] = field(default_factory=lambda: dict[str, Counts]())
    # 케이스별 집계. 구성 간 짝지은 비교(compare)에 쓴다.
    per_case: dict[str, Counts] = field(default_factory=lambda: dict[str, Counts]())
    # 케이스·필드별 채점 결과(tp/fn/fpfn/fp/tn). 실행 간 변동 분석(variance)에 쓴다.
    per_field: dict[str, dict[str, str]] = field(
        default_factory=lambda: dict[str, dict[str, str]]()
    )

    def to_dict(self) -> dict[str, object]:
        """JSON 직렬화용 dict."""

        def ratio(r: Ratio) -> dict[str, object]:
            return {"num": r.num, "den": r.den, "value": r.value}

        def metrics(m: Metrics) -> dict[str, object]:
            return {
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "ungrounded": ratio(m.ungrounded),
                "hallucinated": ratio(m.hallucinated),
                "numeric_error": ratio(m.numeric_error),
                "risk_recall": ratio(m.risk_recall),
                "confirm_recall": ratio(m.confirm_recall),
                "abstained": ratio(m.abstained),
                "wrong_value": ratio(m.wrong_value),
                "silent_wrong": ratio(m.silent_wrong),
                "over_confirm": ratio(m.over_confirm),
            }

        return {
            "case_count": self.case_count,
            "split": self.split,
            "overall": metrics(self.overall),
            "recoverable": metrics(self.recoverable) if self.recoverable else None,
            "strict": metrics(self.strict) if self.strict else None,
            "f1_ci": (
                {"point": self.f1_ci.point, "low": self.f1_ci.low, "high": self.f1_ci.high}
                if self.f1_ci
                else None
            ),
            "review_required_ids": list(self.review_required_ids),
            "headline": metrics(self.headline) if self.headline else None,
            "headline_all_fields": (
                metrics(self.headline_all_fields) if self.headline_all_fields else None
            ),
            "headline_strict": metrics(self.headline_strict) if self.headline_strict else None,
            "headline_ci": (
                {
                    "point": self.headline_ci.point,
                    "low": self.headline_ci.low,
                    "high": self.headline_ci.high,
                }
                if self.headline_ci
                else None
            ),
            "per_case_recoverable": {
                k: {"tp": c.tp, "fp": c.fp, "fn": c.fn}
                for k, c in sorted(self.per_case_recoverable.items())
            },
            "by_category": {k: metrics(v) for k, v in self.by_category.items()},
            "by_probe": {k: metrics(v) for k, v in self.by_probe.items()},
            "per_case": {
                k: {"tp": c.tp, "fp": c.fp, "fn": c.fn} for k, c in sorted(self.per_case.items())
            },
            "per_field": {k: dict(v) for k, v in sorted(self.per_field.items())},
            "usage": self.usage,
        }


def _total(counts: list[Counts]) -> Counts:
    total = Counts()
    for c in counts:
        total = total + c
    return total


def evaluate(extractor: Extractor, cases: list[GoldenCase], split: str = "all") -> EvalReport:
    """모든 케이스를 추출·채점해 보고서를 만든다. 추출은 케이스당 한 번만 한다."""
    predictions = {case.id: extractor.extract(case.transcript) for case in cases}

    def score(subset: Subset) -> dict[str, Counts]:
        return {c.id: score_case(c, predictions[c.id], subset) for c in cases}

    full, recoverable, strict = score("all"), score("recoverable"), score("strict")
    excluded = {c.id for c in cases if c.review_required}
    kept = [c.id for c in cases if c.id not in excluded]
    per_category: dict[str, list[Counts]] = {}
    per_probe: dict[str, list[Counts]] = {}
    for case in cases:
        per_category.setdefault(case.category.value, []).append(full[case.id])
        per_probe.setdefault(probe_of(case), []).append(full[case.id])
    return EvalReport(
        case_count=len(cases),
        overall=Metrics.from_counts(_total(list(full.values()))),
        by_category={k: Metrics.from_counts(_total(v)) for k, v in sorted(per_category.items())},
        usage=extractor.usage_summary() if isinstance(extractor, SupportsUsage) else None,
        split=split,
        by_probe={k: Metrics.from_counts(_total(v)) for k, v in sorted(per_probe.items())},
        recoverable=Metrics.from_counts(_total(list(recoverable.values()))),
        strict=Metrics.from_counts(_total(list(strict.values()))),
        f1_ci=bootstrap_f1(list(full.values())),
        per_case=full,
        review_required_ids=tuple(sorted(excluded)),
        headline=Metrics.from_counts(_total([recoverable[i] for i in kept])),
        headline_all_fields=Metrics.from_counts(_total([full[i] for i in kept])),
        headline_strict=Metrics.from_counts(_total([strict[i] for i in kept])),
        headline_ci=bootstrap_f1([recoverable[i] for i in kept]),
        per_case_recoverable=recoverable,
        per_field={c.id: field_outcomes(c, predictions[c.id]) for c in cases},
    )
