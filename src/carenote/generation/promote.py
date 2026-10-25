"""후보와 직접 작성분을 골든셋 v2 JSONL(`v2_dev.jsonl`, `v2_test.jsonl`)로 승격한다.

정답은 스펙의 `gold`가 유일한 출처다. 후보에서는 transcript·reference_transcript·근거 인용만
가져오고, 직접 작성분은 사용자가 쓴 transcript와 `evidence.json`의 인용을 쓴다. 승격은 동결이
아니다. 동결(sha256·태그)은 현장 검토 대기 케이스(R3, G1)를 포함한 뒤 따로 한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carenote.evaluation.schema import Category, FieldKey, GoldenCase
from carenote.generation.spec import Spec

PROBE_ORDER = "HCDNLSVGIR"
FIRST_V2_NUMBER = 21
CATEGORY_BY_PROBE: dict[str, Category] = {
    "H": Category.MISSING,
    "C": Category.CORRECTION,
    "D": Category.DISTRACTOR,
    "N": Category.CONTEXT_TRAP,
    "L": Category.LONG_NOISY,
    "S": Category.STT_ERROR,
    "V": Category.VOCAB_BOUNDARY,
    "G": Category.DIALECT,
    "I": Category.INJECTION,
    "R": Category.RISK_SIGNAL,
}
# 프로브보다 구체적인 유형이 있는 스펙.
CATEGORY_OVERRIDES: dict[str, Category] = {"H2": Category.BLANKET}


class PromotionError(ValueError):
    """승격할 수 없는 입력."""


@dataclass(frozen=True)
class Source:
    """케이스 하나의 원문 자료."""

    transcript: str
    reference_transcript: str | None
    evidence: dict[str, list[str]]


def spec_sort_key(spec: Spec) -> tuple[int, int]:
    """프로브 순서, 번호 순서."""
    return PROBE_ORDER.index(spec.probe), int(spec.spec_id[1:])


def assign_ids(specs: list[Spec]) -> dict[str, str]:
    """스펙을 프로브·번호 순으로 정렬해 gs-021부터 연속 id를 준다 (결정적)."""
    ordered = sorted(specs, key=spec_sort_key)
    return {s.spec_id: f"gs-{FIRST_V2_NUMBER + i:03d}" for i, s in enumerate(ordered)}


def collect_sources(candidate_dir: Path, human_dir: Path, specs: list[Spec]) -> dict[str, Source]:
    """스펙마다 원문 자료를 모은다. 사람 작성 스펙은 human_dir, 나머지는 후보 폴더."""
    human_texts: dict[str, str] = json.loads(
        (human_dir / "transcripts.json").read_text(encoding="utf-8")
    )
    human_evidence: dict[str, dict[str, list[str]]] = json.loads(
        (human_dir / "evidence.json").read_text(encoding="utf-8")
    )
    sources: dict[str, Source] = {}
    for spec in specs:
        if spec.author == "human":
            if spec.spec_id not in human_texts or spec.spec_id not in human_evidence:
                raise PromotionError(f"{spec.spec_id}: 직접 작성 transcript 또는 근거가 없다")
            sources[spec.spec_id] = Source(
                human_texts[spec.spec_id], None, human_evidence[spec.spec_id]
            )
            continue
        path = candidate_dir / f"{spec.spec_id}.json"
        if not path.exists():
            raise PromotionError(f"{spec.spec_id}: 후보 파일이 없다")
        doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        fails = [c["code"] for c in doc["checks"] + doc["batch_checks"] if c["level"] == "fail"]
        if doc["status"] != "passed" or doc["comparison_only"] or fails or not doc["transcript"]:
            raise PromotionError(
                f"{spec.spec_id}: 승격 불가 후보 (status={doc['status']}, fail={fails})"
            )
        sources[spec.spec_id] = Source(
            doc["transcript"], doc["reference_transcript"], doc["evidence"]
        )
    return sources


def build_case(spec: Spec, source: Source, case_id: str) -> GoldenCase:
    """스펙 정답과 원문 자료로 GoldenCase를 만든다. 스키마 위반이면 PromotionError."""
    unrecoverable: set[FieldKey] = (
        {e.field for e in spec.stt.errors if not e.recoverable_from_text} if spec.stt else set()
    )
    fields: list[dict[str, Any]] = []
    for key in FieldKey:
        gold = spec.gold[key]
        quotes = source.evidence.get(key.value, [])
        if (gold.value is None and not gold.candidates) and quotes:
            raise PromotionError(f"{spec.spec_id}/{key.value}: 정답이 비었는데 근거 인용이 있다")
        fields.append(
            {
                "key": key.value,
                "value": gold.value,
                "source": gold.source.value,
                "evidence": [{"text": q} for q in quotes],
                "flag": gold.flag.value,
                "candidates": list(gold.candidates),
                "recoverable": key not in unrecoverable,
            }
        )
    category = CATEGORY_OVERRIDES.get(spec.spec_id, CATEGORY_BY_PROBE[spec.probe])
    payload: dict[str, Any] = {
        "id": case_id,
        "category": category.value,
        "synthetic": True,
        "transcript": source.transcript,
        "reference_transcript": source.reference_transcript,
        "injection_attempt": category is Category.INJECTION,
        "fields": fields,
        "notes": spec.title,
        "split": spec.split,
        "tags": [spec.probe, spec.spec_id],
        "spec_id": spec.spec_id,
        "author": spec.author,
        "gray_fields": [k.value for k in spec.gray_fields],
        "label_risk": spec.label_risk,
        "review_required": spec.field_review_required,
    }
    try:
        return GoldenCase.model_validate(payload)
    except ValueError as exc:
        raise PromotionError(f"{spec.spec_id}({case_id}): {exc}") from exc


def promote(
    specs: list[Spec], candidate_dir: Path, human_dir: Path, out_dir: Path
) -> dict[str, list[str]]:
    """40건을 만들어 `v2_dev.jsonl`, `v2_test.jsonl`로 쓴다. 분할별 id 목록을 돌려준다.

    하나라도 실패하면 아무것도 쓰지 않는다.
    """
    ids = assign_ids(specs)
    sources = collect_sources(candidate_dir, human_dir, specs)
    cases = [build_case(s, sources[s.spec_id], ids[s.spec_id]) for s in specs]
    cases.sort(key=lambda c: c.id)
    result: dict[str, list[str]] = {}
    for split in ("dev", "test"):
        chosen = [c for c in cases if c.split == split]
        lines = [json.dumps(c.model_dump(mode="json"), ensure_ascii=False) for c in chosen]
        (out_dir / f"v2_{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        result[split] = [c.id for c in chosen]
    return result
