"""CLI: `python -m carenote.evaluation --extractor oracle|null [--out report.json]`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from carenote.evaluation.extractor import Extractor, NullExtractor, OracleExtractor
from carenote.evaluation.loader import load_golden
from carenote.evaluation.metrics import Metrics, Ratio
from carenote.evaluation.runner import EvalReport, evaluate

DEFAULT_GOLDEN = Path("data/golden")


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _ratio(r: Ratio) -> str:
    return f"{_pct(r.value)} ({r.num}/{r.den})"


def _row(label: str, m: Metrics) -> str:
    return (
        f"{label:<18}{_pct(m.f1):>8}  {_ratio(m.ungrounded):<16}{_ratio(m.hallucinated):<16}"
        f"{_ratio(m.numeric_error):<16}{_ratio(m.risk_recall):<16}{_ratio(m.confirm_recall)}"
    )


def format_report(report: EvalReport) -> str:
    """사람이 읽는 표."""
    header = (
        f"{'':<18}{'F1':>8}  {'근거없음':<16}{'환각':<16}{'수치오류':<16}"
        f"{'위험recall':<16}{'확인recall'}"
    )
    lines = [f"cases: {report.case_count}", header, _row("overall", report.overall)]
    lines += [_row(name, m) for name, m in report.by_category.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(prog="carenote.evaluation")
    parser.add_argument("--extractor", choices=["oracle", "null"], required=True)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--out", type=Path, default=None, help="JSON 보고서 저장 경로")
    args = parser.parse_args(argv)

    cases = load_golden(args.golden)
    extractor: Extractor = OracleExtractor(cases) if args.extractor == "oracle" else NullExtractor()
    report = evaluate(extractor, cases)
    print(format_report(report))
    if args.out is not None:
        args.out.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
