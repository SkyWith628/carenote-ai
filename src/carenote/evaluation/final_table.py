"""test 최종 보고의 표와 사전 등록 비교 (`docs/test-final-report-plan.md`).

    python -m carenote.evaluation.final_table --dir docs/reports
    python -m carenote.evaluation.final_table --dir docs/reports --kr1

보고서 이름 규칙: `test-final-{prompt}-{model}-{variant}.json` (예: `test-final-v4-haiku-b1.json`),
모두 `--split test --final-report --out`으로 저장한 것이다. 없는 구성은 "없음"으로 두고 있는 것만
표로 만든다. 표와 비교는 헤드라인(현장 검토 대기 케이스 R3·G1 제외, 복원가능 필드)을 쓴다.
API를 쓰지 않는다(`--kr1`은 캐시만 읽는다).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from carenote.config import get_settings
from carenote.evaluation.compare import select_per_case
from carenote.evaluation.extractor import Extractor, PredictedField
from carenote.evaluation.loader import load_golden
from carenote.evaluation.runner import evaluate
from carenote.evaluation.schema import FieldKey, GoldenCase
from carenote.evaluation.stats import Interval, paired_f1_difference
from carenote.extraction.flagger import FlaggingExtractor, flag_field
from carenote.extraction.llm_extractor import LLMExtractor

PROMPTS = ("v4", "v1")
MODELS = ("sonnet", "haiku")
VARIANTS = ("b0", "b1")
MODEL_IDS = {"sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
# 사전 등록: dev Haiku 반복 실행의 F1 σ_run(0.4~0.6%p의 대표값). Sonnet 변동은 측정하지 않았다.
SIGMA_RUN_PROXY = 0.005
SIGMA_MULTIPLE = 2.0

Config = tuple[str, str, str]  # (prompt, model, variant)
Report = dict[str, Any]


def report_path(directory: Path, config: Config) -> Path:
    """구성의 보고서 경로."""
    prompt, model, variant = config
    return directory / f"test-final-{prompt}-{model}-{variant}.json"


def all_configs() -> list[Config]:
    """사전 등록 구성 8개(계층 1 v4 먼저, 계층 2 v1)."""
    return [(p, m, v) for p in PROMPTS for m in MODELS for v in VARIANTS]


def load_available(directory: Path) -> dict[Config, Report]:
    """디렉터리에 있는 구성의 보고서를 읽는다."""
    found: dict[Config, Report] = {}
    for config in all_configs():
        path = report_path(directory, config)
        if path.exists():
            found[config] = json.loads(path.read_text(encoding="utf-8"))
    return found


def _label(config: Config) -> str:
    prompt, model, variant = config
    return f"{prompt} {model.capitalize()} {variant.upper()}"


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.1f}%"


def _ratio(block: dict[str, Any]) -> str:
    return f"{block['num']}/{block['den']}"


def _block(report: Report, key: str) -> dict[str, Any]:
    """보고서의 하위 dict. 없거나 null이면 빈 dict."""
    value = report.get(key)
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _headline_row(config: Config, report: Report) -> str:
    head = _block(report, "headline")
    if not head:
        return f"| {_label(config)} | 헤드라인 없음(현재 러너로 다시 채점 필요) |"
    ci = _block(report, "headline_ci")
    low = cast(float | None, ci.get("low"))
    high = cast(float | None, ci.get("high"))
    interval = f" [{_pct(low)}~{_pct(high)}]" if low is not None else ""
    all_fields = cast(float | None, _block(report, "headline_all_fields").get("f1"))
    cost = cast(float | None, _block(report, "usage").get("estimated_cost_usd"))
    overall_f1 = cast(float | None, _block(report, "overall").get("f1"))
    f1 = cast(float | None, head.get("f1"))
    return (
        f"| {_label(config)} | **{_pct(f1)}**{interval} | {_pct(all_fields)} "
        f"| {_pct(overall_f1)} | {_ratio(head['silent_wrong'])} "
        f"| {_ratio(head['over_confirm'])} | {_ratio(head['risk_recall'])} "
        f"| {_ratio(head['confirm_recall'])} | {_ratio(head['hallucinated'])} "
        f"| {_ratio(head['numeric_error'])} | {_ratio(head['ungrounded'])} "
        f"| {'–' if cost is None else f'${cost:.3f}'} |"
    )


def format_table(reports: dict[Config, Report]) -> str:
    """구성 × 지표 표."""
    excluded = sorted(
        {tuple(cast(list[str], r.get("review_required_ids") or [])) for r in reports.values()}
    )
    note = ", ".join(", ".join(e) for e in excluded if e) or "없음"
    lines = [
        f"헤드라인 = 현장 검토 대기 케이스({note})를 뺀 케이스, 복원가능 필드만. "
        "횟수 지표(위험·확인 recall 등)는 분모가 작아 비율의 유의성을 주장하지 않는다.",
        "",
        "| 구성 | 헤드라인 F1 [95%] | 전체 필드 F1 | 30건 전체 F1 | 조용한 오답 | 과다 확인 "
        "| 위험 recall | 확인 recall | 환각 | 수치 오류 | 근거 없는 값 | 비용 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for config in all_configs():
        if config in reports:
            lines.append(_headline_row(config, reports[config]))
        else:
            lines.append(f"| {_label(config)} | 없음 |")
    return "\n".join(lines)


@dataclass(frozen=True)
class Comparison:
    """사전 등록 비교 하나(Y − X)."""

    name: str
    x: Config
    y: Config
    diff: Interval
    threshold: float
    sonnet_involved: bool
    verdict: str


def comparison_pairs() -> list[tuple[str, Config, Config]]:
    """사전 등록 비교 12개: B1−B0, Sonnet−Haiku, v4−v1."""
    pairs: list[tuple[str, Config, Config]] = []
    for p in PROMPTS:
        for m in MODELS:
            pairs.append(("B1−B0", (p, m, "b0"), (p, m, "b1")))
    for p in PROMPTS:
        for v in VARIANTS:
            pairs.append(("Sonnet−Haiku", (p, "haiku", v), (p, "sonnet", v)))
    for m in MODELS:
        for v in VARIANTS:
            pairs.append(("v4−v1", ("v1", m, v), ("v4", m, v)))
    return pairs


def judge(diff: Interval, sonnet_involved: bool) -> tuple[str, float]:
    """사전 등록 판정 규칙. (라벨, 변동 기준 임계값).

    - Haiku만 낀 비교: 구간이 0을 포함하지 않고 |Δ| > 2σ합성이면 "차이 있음".
    - Sonnet이 낀 비교(변동 미측정): 구간만 0을 포함하지 않으면 "방향만".
    - 그 밖은 "구분 불가".
    """
    threshold = SIGMA_MULTIPLE * math.sqrt(2) * SIGMA_RUN_PROXY
    if diff.point is None or diff.low is None or diff.high is None:
        return "비교 불가", threshold
    interval_ok = diff.low > 0 or diff.high < 0
    if sonnet_involved:
        return ("방향만(Sonnet 변동 미측정)" if interval_ok else "구분 불가"), threshold
    return ("차이 있음" if interval_ok and abs(diff.point) > threshold else "구분 불가"), threshold


def compare_configs(reports: dict[Config, Report]) -> list[Comparison]:
    """있는 구성끼리 사전 등록 비교를 계산한다(헤드라인 케이스, 복원가능 필드)."""
    results: list[Comparison] = []
    for name, x, y in comparison_pairs():
        if x not in reports or y not in reports:
            continue
        opts: dict[str, bool] = {"exclude_review_required": True, "recoverable": True}
        diff = paired_f1_difference(
            select_per_case(reports[x], name=_label(x), **opts),
            select_per_case(reports[y], name=_label(y), **opts),
        )
        sonnet = "sonnet" in (x[1], y[1])
        verdict, threshold = judge(diff, sonnet)
        results.append(Comparison(name, x, y, diff, threshold, sonnet, verdict))
    return results


def format_comparisons(comparisons: list[Comparison]) -> str:
    """비교 표."""
    if not comparisons:
        return "사전 등록 비교: 짝이 되는 보고서가 아직 없다."
    lines = [
        "사전 등록 비교(Y − X, 헤드라인·복원가능). 조건: 짝지은 95% 구간이 0을 포함하지 않고 "
        f"|Δ| > {SIGMA_MULTIPLE:g}σ합성(≈{comparisons[0].threshold * 100:.1f}%p). "
        "Sonnet이 낀 비교는 변동을 재지 않아 방향만 쓴다.",
        "",
        "| 비교 | X | Y | Y−X | 95% 구간 | 판정 |",
        "|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        d = c.diff
        interval = (
            "–" if d.low is None or d.high is None else f"[{d.low * 100:+.1f}, {d.high * 100:+.1f}]"
        )
        point = "–" if d.point is None else f"{d.point * 100:+.1f}%p"
        lines.append(
            f"| {c.name} | {_label(c.x)} | {_label(c.y)} | {point} | {interval} | **{c.verdict}** |"
        )
    return "\n".join(lines)


# -- KR-1 규칙 전후 (같은 모델 출력, 오프라인 재채점) ------------------------------------


class WithoutKr1Rules:
    """transcript를 넘기지 않아 R1·R2 없이 예전 규칙만으로 flag를 붙인다."""

    def __init__(self, inner: Extractor) -> None:
        self._inner = inner

    def extract(self, transcript: str) -> list[PredictedField]:
        """추출 후 예전 방식의 flag를 계산한다."""
        by_key = {f.key: f for f in self._inner.extract(transcript)}
        fields = [by_key.get(key, PredictedField(key=key)) for key in FieldKey]
        return [f.model_copy(update={"flag": flag_field(f)}) for f in fields]


@dataclass(frozen=True)
class Kr1Row:
    """KR-1 규칙 전·후 안전 지표(헤드라인 케이스)."""

    silent_before: str
    silent_after: str
    over_before: str
    over_after: str
    confirm_before: str
    confirm_after: str
    f1_same: bool


def kr1_before_after(cases: list[GoldenCase], inner: Extractor) -> Kr1Row:
    """같은 추출 결과에 KR-1 규칙이 없을 때와 있을 때의 안전 지표를 비교한다."""
    before = evaluate(WithoutKr1Rules(inner), cases, "test")
    after = evaluate(FlaggingExtractor(inner), cases, "test")
    hb, ha = before.headline, after.headline
    assert hb is not None and ha is not None
    return Kr1Row(
        f"{hb.silent_wrong.num}/{hb.silent_wrong.den}",
        f"{ha.silent_wrong.num}/{ha.silent_wrong.den}",
        f"{hb.over_confirm.num}/{hb.over_confirm.den}",
        f"{ha.over_confirm.num}/{ha.over_confirm.den}",
        f"{hb.confirm_recall.num}/{hb.confirm_recall.den}",
        f"{ha.confirm_recall.num}/{ha.confirm_recall.den}",
        hb.f1 == ha.f1,
    )


class _NoClient:
    @property
    def messages(self) -> object:
        raise RuntimeError("--kr1은 캐시만 읽는다. API 클라이언트를 쓰면 안 된다")


def format_kr1(rows: dict[Config, Kr1Row | None]) -> str:
    """KR-1 전후 표."""
    lines = [
        "KR-1 규칙 전·후(같은 모델 출력, 헤드라인 케이스, 캐시 재채점):",
        "",
        "| 구성 | 조용한 오답 전 → 후 | 과다 확인 전 → 후 | 확인 recall 전 → 후 | F1 불변 |",
        "|---|---|---|---|---|",
    ]
    for config in all_configs():
        row = rows.get(config)
        if row is None:
            lines.append(f"| {_label(config)} | 캐시 없음 |")
            continue
        lines.append(
            f"| {_label(config)} | {row.silent_before} → {row.silent_after} "
            f"| {row.over_before} → {row.over_after} "
            f"| {row.confirm_before} → {row.confirm_after} | {'예' if row.f1_same else '아니오'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점. 종료 코드 0 성공 · 2 입력 오류."""
    parser = argparse.ArgumentParser(prog="carenote.evaluation.final_table", description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("docs/reports"))
    parser.add_argument("--kr1", action="store_true", help="KR-1 규칙 전·후 재채점(캐시만)")
    parser.add_argument("--golden", type=Path, default=Path("data/golden"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        reports = load_available(args.dir)
        if not reports:
            print(f"{args.dir}에 test-final-*.json 보고서가 없다", file=sys.stderr)
            return 2
        print(format_table(reports))
        print()
        print(format_comparisons(compare_configs(reports)))
        if args.kr1:
            cases = load_golden(args.golden, "test")
            cache = args.cache_dir or Path(get_settings().extraction_cache_dir)
            rows: dict[Config, Kr1Row | None] = {}
            for config in reports:
                prompt, model, variant = config
                inner = LLMExtractor(
                    _NoClient(),  # type: ignore[arg-type]
                    variant=variant,  # type: ignore[arg-type]
                    model=MODEL_IDS[model],
                    effort=get_settings().extractor_effort,
                    cache_dir=cache,
                    prompt_version=prompt,  # type: ignore[arg-type]
                )
                rows[config] = (
                    kr1_before_after(cases, inner)
                    if all(inner.is_cached(c.transcript) for c in cases)
                    else None
                )
            print()
            print(format_kr1(rows))
    except (ValueError, OSError) as exc:
        print(f"표 생성 불가: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
