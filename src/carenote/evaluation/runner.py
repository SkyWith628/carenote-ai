"""평가 러너: extractor를 골든셋에 돌려 지표를 낸다."""

from __future__ import annotations

from dataclasses import dataclass, field

from carenote.evaluation.extractor import Extractor, SupportsUsage
from carenote.evaluation.metrics import Counts, Metrics, Ratio, Subset, score_case
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
    # 케이스별 집계. 구성 간 짝지은 비교(compare)에 쓴다.
    per_case: dict[str, Counts] = field(default_factory=lambda: dict[str, Counts]())

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
            "by_category": {k: metrics(v) for k, v in self.by_category.items()},
            "by_probe": {k: metrics(v) for k, v in self.by_probe.items()},
            "per_case": {
                k: {"tp": c.tp, "fp": c.fp, "fn": c.fn} for k, c in sorted(self.per_case.items())
            },
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
    )
