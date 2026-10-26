"""실행 간 변동 분석.

같은 (모델, 변형, 프롬프트, 골든셋 분할)을 여러 번 돌린 보고서들(`--out`, `--run-tag`로 반복)을
받아 모델의 실행 간 변동을 잰다. 설계 `docs/run-variance-design.md`.

    python -m carenote.evaluation.variance run0.json run1.json run2.json ...
    python -m carenote.evaluation.variance X0.json X1.json ... --vs Y0.json Y1.json ...

`--vs`는 두 프롬프트(또는 구성)의 반복 실행 묶음을 사전에 정한 규칙으로 비교한다:
평균 F1 차이가 σ_run 합성값의 2배보다 크고, 케이스를 재표집한 실행 평균 결과의 95% 구간이
0을 포함하지 않아야 "차이가 있다"고 쓴다. API를 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from carenote.evaluation.metrics import CORRECT_OUTCOMES

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 0
SIGMA_MULTIPLE = 2.0
ALPHA = 0.05

Report = dict[str, Any]


def load_reports(paths: list[Path]) -> list[Report]:
    """보고서 JSON들을 읽는다."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _require(reports: list[Report], key: str) -> None:
    for i, report in enumerate(reports):
        if not report.get(key):
            raise ValueError(
                f"{i + 1}번째 보고서에 {key}가 없다. 현재 러너로 다시 채점한 보고서가 필요하다"
            )


def _f1(report: Report) -> float:
    value = report["overall"]["f1"]
    if value is None:
        raise ValueError("F1이 없는 보고서(분모 0)는 분석할 수 없다")
    return float(value)


def _same_fields(reports: list[Report]) -> list[tuple[str, str]]:
    keys = [sorted((c, f) for c, fields in r["per_field"].items() for f in fields) for r in reports]
    if any(k != keys[0] for k in keys[1:]):
        raise ValueError("보고서들의 케이스·필드 집합이 다르다 (같은 분할이어야 한다)")
    return keys[0]


@dataclass(frozen=True)
class VarianceSummary:
    """같은 설정을 K번 돌린 결과의 변동 요약."""

    runs: int
    f1s: tuple[float, ...]
    mean_f1: float
    sd_f1: float | None
    min_f1: float
    max_f1: float
    mean_pairwise_abs_diff: float | None
    fields: int
    unstable: tuple[str, ...]
    stable_errors: tuple[str, ...]
    majority_f1: float | None

    @property
    def instability_rate(self) -> float:
        """한 번이라도 정오가 달라진 필드의 비율."""
        return len(self.unstable) / self.fields if self.fields else 0.0


def _f1_from_counts(tp: float, fp: float, fn: float) -> float | None:
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    if precision is None or recall is None:
        return None if precision is None and recall is None else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


_OUTCOME_COUNTS = {
    "tp": (1, 0, 0),
    "fn": (0, 0, 1),
    "fpfn": (0, 1, 1),
    "fp": (0, 1, 0),
    "tn": (0, 0, 0),
}


def analyze(reports: list[Report]) -> VarianceSummary:
    """같은 설정의 반복 실행 보고서(2개 이상)를 분석한다."""
    if len(reports) < 2:
        raise ValueError("변동을 재려면 같은 설정의 실행이 2번 이상 필요하다")
    _require(reports, "per_field")
    pairs = _same_fields(reports)
    f1s = tuple(_f1(r) for r in reports)
    unstable: list[str] = []
    stable_errors: list[str] = []
    majority = [0, 0, 0]
    for case, field in pairs:
        outcomes = [r["per_field"][case][field] for r in reports]
        correct = [o in CORRECT_OUTCOMES for o in outcomes]
        label = f"{case} {field}"
        if any(correct) and not all(correct):
            unstable.append(label)
        elif not any(correct):
            stable_errors.append(label)
        ranked = sorted(Counter(outcomes).items(), key=lambda kv: (-kv[1], kv[0]))
        tp, fp, fn = _OUTCOME_COUNTS[ranked[0][0]]
        majority[0] += tp
        majority[1] += fp
        majority[2] += fn
    diffs = [abs(a - b) for a, b in combinations(f1s, 2)]
    return VarianceSummary(
        runs=len(reports),
        f1s=f1s,
        mean_f1=statistics.fmean(f1s),
        sd_f1=statistics.stdev(f1s),
        min_f1=min(f1s),
        max_f1=max(f1s),
        mean_pairwise_abs_diff=statistics.fmean(diffs),
        fields=len(pairs),
        unstable=tuple(unstable),
        stable_errors=tuple(stable_errors),
        majority_f1=_f1_from_counts(*(float(x) for x in majority)),
    )


def _averaged_counts(reports: list[Report]) -> dict[str, tuple[float, float, float]]:
    """케이스마다 실행들의 TP/FP/FN 평균."""
    _require(reports, "per_case")
    ids = sorted(reports[0]["per_case"])
    if any(sorted(r["per_case"]) != ids for r in reports[1:]):
        raise ValueError("보고서들의 케이스 집합이 다르다")
    k = len(reports)
    return {
        i: (
            sum(r["per_case"][i]["tp"] for r in reports) / k,
            sum(r["per_case"][i]["fp"] for r in reports) / k,
            sum(r["per_case"][i]["fn"] for r in reports) / k,
        )
        for i in ids
    }


@dataclass(frozen=True)
class Verdict:
    """두 반복 실행 묶음의 비교 결과 (Y − X)."""

    mean_diff: float
    sigma_combined: float | None
    threshold: float | None
    ci_low: float | None
    ci_high: float | None
    passes_sigma_rule: bool
    passes_interval_rule: bool

    @property
    def different(self) -> bool:
        """두 규칙을 모두 통과했을 때만 차이가 있다고 쓴다."""
        return self.passes_sigma_rule and self.passes_interval_rule


def compare_runs(
    xs: list[Report],
    ys: list[Report],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Verdict:
    """X 묶음 대비 Y 묶음 F1 차이를 사전 규칙으로 판정한다. 각 묶음은 2회 이상이어야 한다."""
    sx, sy = analyze(xs), analyze(ys)
    mean_diff = sy.mean_f1 - sx.mean_f1
    assert sx.sd_f1 is not None and sy.sd_f1 is not None
    sigma = math.sqrt(sx.sd_f1**2 + sy.sd_f1**2)
    threshold = SIGMA_MULTIPLE * sigma
    ax, ay = _averaged_counts(xs), _averaged_counts(ys)
    if set(ax) != set(ay):
        raise ValueError("두 묶음의 케이스 집합이 다르다")
    ids = sorted(ax)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        picked = [ids[rng.randrange(len(ids))] for _ in ids]
        fx = _f1_from_counts(*(sum(ax[i][j] for i in picked) for j in range(3)))
        fy = _f1_from_counts(*(sum(ay[i][j] for i in picked) for j in range(3)))
        if fx is not None and fy is not None:
            samples.append(fy - fx)
    samples.sort()
    low = samples[round(ALPHA / 2 * (len(samples) - 1))] if samples else None
    high = samples[round((1 - ALPHA / 2) * (len(samples) - 1))] if samples else None
    return Verdict(
        mean_diff=mean_diff,
        sigma_combined=sigma,
        threshold=threshold,
        ci_low=low,
        ci_high=high,
        passes_sigma_rule=abs(mean_diff) > threshold,
        passes_interval_rule=low is not None and high is not None and (low > 0 or high < 0),
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def format_summary(summary: VarianceSummary, label: str = "") -> str:
    """사람이 읽는 요약."""
    head = f"[{label}] " if label else ""
    lines = [
        f"{head}실행 {summary.runs}회 · F1 {', '.join(_pct(x) for x in summary.f1s)}",
        f"  평균 {_pct(summary.mean_f1)} · σ_run {_pct(summary.sd_f1)} · "
        f"범위 {_pct(summary.min_f1)}~{_pct(summary.max_f1)} · "
        f"쌍별 평균 |ΔF1| {_pct(summary.mean_pairwise_abs_diff)}",
        f"  불안정 필드 {len(summary.unstable)}/{summary.fields} "
        f"({summary.instability_rate * 100:.1f}%) · 안정 오류 {len(summary.stable_errors)}개 · "
        f"다수결 F1 {_pct(summary.majority_f1)}",
    ]
    if summary.unstable:
        lines.append("  불안정: " + ", ".join(summary.unstable))
    if summary.stable_errors:
        lines.append("  안정 오류: " + ", ".join(summary.stable_errors))
    return "\n".join(lines)


def format_verdict(verdict: Verdict) -> str:
    """판정 한 줄 요약."""
    if verdict.ci_low is None or verdict.ci_high is None or verdict.threshold is None:
        return "비교 불가"
    word = "차이 있음" if verdict.different else "구분 불가"
    return (
        f"Y − X = {verdict.mean_diff * 100:+.1f}%p · 기준 ±{verdict.threshold * 100:.1f}%p "
        f"(σ합성 {_pct(verdict.sigma_combined)}의 {SIGMA_MULTIPLE:g}배) "
        f"{'통과' if verdict.passes_sigma_rule else '미달'} · "
        f"실행 평균 95% 구간 [{verdict.ci_low * 100:+.1f}, {verdict.ci_high * 100:+.1f}] "
        f"{'통과' if verdict.passes_interval_rule else '0 포함'} → {word}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점. 종료 코드 0 성공 · 2 입력 오류."""
    parser = argparse.ArgumentParser(prog="carenote.evaluation.variance", description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="같은 설정의 반복 실행 보고서(2개 이상)")
    parser.add_argument("--vs", type=Path, nargs="+", default=None, help="비교할 다른 묶음")
    args = parser.parse_args(argv)
    try:
        xs = load_reports(args.runs)
        print(format_summary(analyze(xs), "X"))
        if args.vs is not None:
            ys = load_reports(args.vs)
            print(format_summary(analyze(ys), "Y"))
            print(format_verdict(compare_runs(xs, ys)))
    except (ValueError, OSError) as exc:
        print(f"분석 불가: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
