"""후보 저장, 배치 점검(H·K), 사람 검수 시트."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from carenote.evaluation.loader import load_golden
from carenote.generation.checks import (
    check_h_duplicates,
    check_k_closings,
    check_k_openings,
)
from carenote.generation.generator import GenerationResult, Generator
from carenote.generation.prompts import GENERATION_PROMPT_VERSION
from carenote.generation.spec import Spec

LABELING_SAMPLE_SIZE = 10


def candidate_path(out_dir: Path, result: GenerationResult) -> Path:
    """직접 작성 스펙의 비교용 생성물은 pilot 폴더에 따로 둔다."""
    base = out_dir / "pilot" if result.comparison_only else out_dir
    return base / f"{result.spec_id}.json"


def write_candidate(
    out_dir: Path, result: GenerationResult, spec: Spec, generator: Generator
) -> Path:
    """후보를 JSON으로 저장한다. 실패한 후보도 사람이 볼 수 있도록 마지막 시도를 남긴다."""
    path = candidate_path(out_dir, result)
    path.parent.mkdir(parents=True, exist_ok=True)
    case = result.case
    doc: dict[str, Any] = {
        "spec_id": result.spec_id,
        "status": result.status,
        "comparison_only": result.comparison_only,
        "transcript": case.transcript if case else None,
        "reference_transcript": case.reference_transcript if case else None,
        "evidence": ({item.key.value: item.quotes for item in case.evidence} if case else {}),
        "notes": case.notes if case else "",
        "generator": {
            "model": generator.model,
            "effort": generator.effort,
            "prompt_version": GENERATION_PROMPT_VERSION,
            "attempts": result.attempts,
        },
        "checks": [c.model_dump() for c in result.checks],
        "batch_checks": [],
        "review": {"status": "pending", "date": None, "note": ""},
        "spec": {"probe": spec.probe, "split": spec.split, "label_risk": spec.label_risk},
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_candidates(out_dir: Path) -> list[dict[str, Any]]:
    """후보 JSON 전부 (pilot 폴더 제외)."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*.json"))]


def apply_batch_checks(out_dir: Path, golden_dir: Path | None) -> dict[str, list[str]]:
    """저장된 후보 전체에 H(중복)와 K(문두)를 적용해 파일에 기록하고 요약을 돌려준다."""
    docs = [d for d in load_candidates(out_dir) if d.get("transcript")]
    items = [(str(d["spec_id"]), str(d["transcript"])) for d in docs]
    existing = (
        [c.transcript for c in load_golden(golden_dir)]
        if golden_dir is not None and golden_dir.exists()
        else []
    )
    h = check_h_duplicates(items, existing)
    k = check_k_openings(items)
    k_end = check_k_closings(items)
    summary: dict[str, list[str]] = {}
    for doc in docs:
        sid = str(doc["spec_id"])
        results = [h[sid], k[sid], k_end[sid]]
        doc["batch_checks"] = [r.model_dump() for r in results]
        (out_dir / f"{sid}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        problems = [f"[{r.code}] {r.level}: {r.message}" for r in results if r.level != "pass"]
        if problems:
            summary[sid] = problems
    return summary


def pick_labeling_sample(specs: list[Spec], seed: int = 0) -> list[str]:
    """독립 라벨링 표본: label_risk가 low가 아닌 스펙 전부 + 무작위로 채워 최소 10건."""
    llm = [s for s in specs if s.author == "llm"]
    chosen = [s.spec_id for s in llm if s.label_risk != "low"]
    rest = sorted(s.spec_id for s in llm if s.label_risk == "low")
    random.Random(seed).shuffle(rest)
    chosen += rest[: max(0, LABELING_SAMPLE_SIZE - len(chosen))]
    return sorted(chosen)


def render_review_sheet(out_dir: Path, specs: list[Spec]) -> str:
    """사람 검수용 마크다운: 후보 transcript, 정답, 점검 결과를 케이스별로 모은다."""
    by_id = {s.spec_id: s for s in specs}
    docs = load_candidates(out_dir)
    sample = set(pick_labeling_sample(specs))
    lines = [
        "# 골든셋 v2 후보 검수 시트",
        "",
        f"후보 {len(docs)}건. 독립 라벨링 표본(★)은 transcript만 보고 먼저 라벨한 뒤 비교한다.",
        "",
    ]
    for doc in docs:
        sid = str(doc["spec_id"])
        spec = by_id.get(sid)
        star = " ★" if sid in sample else ""
        lines += [f"## {sid}{star} · {doc['status']} · 시도 {doc['generator']['attempts']}회"]
        if spec is not None:
            lines.append(
                f"{spec.title} · 프로브 {spec.probe} · {spec.split} · 라벨 위험도 {spec.label_risk}"
                + (" · 현장 확인 필요" if spec.field_review_required else "")
            )
        lines += ["", "**transcript**", "", f"> {doc['transcript']}", ""]
        if doc.get("reference_transcript"):
            lines += ["**reference**", "", f"> {doc['reference_transcript']}", ""]
        if spec is not None:
            lines += ["| 필드 | 정답 | flag | 후보 |", "|---|---|---|---|"]
            for key, gold in spec.gold.items():
                lines.append(
                    f"| {key.value} | {gold.value or '(없음)'} | {gold.flag.value} |"
                    f" {', '.join(gold.candidates) or '-'} |"
                )
            lines.append("")
        problems = [
            f"[{c['code']}] {c['level']}: {c['message']}"
            for c in doc["checks"] + doc["batch_checks"]
            if c["level"] != "pass"
        ]
        lines += (
            (["**점검**"] + [f"- {p}" for p in problems] + [""])
            if problems
            else ["점검: 모두 통과", ""]
        )
        if doc.get("notes"):
            lines += [f"생성기 메모: {doc['notes']}", ""]
        lines += ["검수: ☐ 승인  ☐ 반려 (사유: )", ""]
    return "\n".join(lines)
