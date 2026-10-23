"""평가 러너: extractor를 골든셋에 돌려 지표를 낸다."""

from __future__ import annotations

from dataclasses import dataclass

from carenote.evaluation.extractor import Extractor
from carenote.evaluation.metrics import Counts, Metrics, Ratio, score_case
from carenote.evaluation.schema import GoldenCase


@dataclass(frozen=True)
class EvalReport:
    """전체와 유형별 지표."""

    case_count: int
    overall: Metrics
    by_category: dict[str, Metrics]

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
            }

        return {
            "case_count": self.case_count,
            "overall": metrics(self.overall),
            "by_category": {k: metrics(v) for k, v in self.by_category.items()},
        }


def evaluate(extractor: Extractor, cases: list[GoldenCase]) -> EvalReport:
    """모든 케이스를 추출·채점해 보고서를 만든다."""
    total = Counts()
    per_category: dict[str, Counts] = {}
    for case in cases:
        counts = score_case(case, extractor.extract(case.transcript))
        total = total + counts
        key = case.category.value
        per_category[key] = per_category.get(key, Counts()) + counts
    return EvalReport(
        case_count=len(cases),
        overall=Metrics.from_counts(total),
        by_category={k: Metrics.from_counts(v) for k, v in sorted(per_category.items())},
    )
