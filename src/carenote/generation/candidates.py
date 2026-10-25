"""후보 저장, 배치 점검(H·K), 사람 검수 시트."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from carenote.evaluation.loader import load_golden
from carenote.evaluation.schema import FieldKey
from carenote.generation.checks import (
    check_h_duplicates,
    check_k_closings,
    check_k_openings,
    check_k_time_openings,
    check_k_weather_openings,
    check_p_repeated_phrases,
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
            "diversity": (
                d.to_payload() if (d := generator.diversity_for(result.spec_id)) else None
            ),
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
    k_time = check_k_time_openings(items)
    k_weather = check_k_weather_openings(items)
    phrase_items = [
        (
            str(d["spec_id"]),
            {
                FieldKey(key): quotes[0]
                for key, quotes in dict(d.get("evidence") or {}).items()
                if quotes
            },
        )
        for d in docs
    ]
    p_repeat = check_p_repeated_phrases(phrase_items)
    summary: dict[str, list[str]] = {}
    for doc in docs:
        sid = str(doc["spec_id"])
        results = [h[sid], k[sid], k_end[sid], k_time[sid], k_weather[sid], p_repeat[sid]]
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


# -- 블라인드 검수 (독립 라벨링) ---------------------------------------------------

BLIND_AGREEMENT_TARGET = 0.95
_FIELD_ORDER = (
    "visit_start",
    "blood_pressure",
    "pulse",
    "meal_intake",
    "medication_taken",
    "fall_event",
    "mood",
    "special_notes",
)

BLIND_GUIDE = """\
## 라벨링 방법

transcript만 보고 8개 필드를 라벨한다. 정답은 보이지 않는다. 판단 기준은 `docs/label-policy.md`다.

| 필드 | 값 |
|---|---|
| visit_start | `HH:MM` (24시간). 시각이 아니면 비움 |
| blood_pressure | `수축기/이완기`. 번복·불확실하면 value 비우고 candidates에 해석들 |
| pulse | 숫자 |
| meal_intake | 전량 · 절반 · 소량 · 미섭취 · 미제공 |
| medication_taken | 복용 · 미복용 · 중복 복용 의심 |
| fall_event | 없음 · 있음 |
| mood | 양호 · 저조 (어휘 밖이면 비움) |
| special_notes | 위 항목에 못 들어가는 관찰·호소·사건을 짧게. 없으면 비움 |

- `source`: 직접 말하면 `stated`, 말에서 해석하면 `inferred`, 말하지 않았으면 `missing` (value 비움)
- `flag`: 기본 `ok`. 위험 필드(혈압·맥박·복약·낙상)가 `missing`이거나 candidates가 있거나
  불가능한 수치면 `needs_confirm`.
  미복용·중복 복용 의심·낙상 있음·위험한 특이사항이면 `risk`
- 추측("드셨겠죠", "아마 130쯤")은 사실이 아니다. **추측한 값은 value에 넣지 않고 비운다**
  (source `missing`, 위험 필드면 flag `needs_confirm`).
  해석이 갈리면 `inferred`로 표시하고 메모를 남겨도 된다
- 낙상: 오늘은 없었어도 최근 7일 안에 낙상 보고가 있으면 `있음`이다
- 복약 거부 보고는 `미복용`과 별개로 특이사항에도 적는다 (비워 두면 오류다)
- 특이사항에는 건강·안전 이상 징후 외에 측정을 못 한 사유, 확인·질문을 못 한 항목,
  약 변경·재측정 예정도 적는다.
  가사·환경 변화·일상 잡담·사소한 일화는 적지 않는다
- 기분은 웃음·표정·기색을 직접 말했을 때만 `inferred`다 (대화가 활발했다는 이유만으로는 비움)
- 기분 관찰에 "~것 같아요" 같은 추측 표현이 붙어도 화자가 직접 본 기분·표정이면 `inferred`로 적는다
  (저조: "기운이 없어 보인다", "표정이 어둡다" / 양호: "웃으며", "밝은 표정")
- 번복이 해소되지 않아 숫자가 두 가지로 읽히면 값을 비우고
  **후보 칸에 그 숫자들을 적는다** (그냥 비우지 않는다). 해석할 숫자가 없으면 후보 없이 비운다
- 어제·평소 수치를 오늘 수치와 견주는 말은 특이사항에 적지 않는다
- 라벨을 끝내기 전에 낙상과 복약 이야기가 있는지 transcript를
  한 번 더 찾아본다 (말하지 않았을 때만 비운다)
- 특이사항이 가슴 답답함·호흡곤란·말 어눌함·한쪽 힘 빠짐·낙상 후 통증·며칠째 식사 거부·
  갑작스러운 부종·출혈 같은 위험 신호면 그 필드의 flag를 `risk`로 한다 (`docs/label-policy.md`)

라벨은 `*.labels.json`의 각 케이스에 채운다 (value는 문자열, 없으면 빈 문자열).
채운 뒤 `python -m carenote.generation --compare-blind <labels.json>`으로 정답과 비교한다.
"""


def blind_ids(spec_ids: list[str], seed: int = 0) -> dict[str, str]:
    """spec_id를 B01, B02…로 가려서 순서도 섞는다 (id만으로 프로브를 짐작하지 못하게)."""
    shuffled = sorted(spec_ids)
    random.Random(seed).shuffle(shuffled)
    return {sid: f"B{i + 1:02d}" for i, sid in enumerate(shuffled)}


def render_blind_sheet(out_dir: Path, key: dict[str, str]) -> str:
    """정답·제목·프로브·위험도를 숨기고 transcript만 보여 주는 라벨링 시트."""
    docs = {str(d["spec_id"]): d for d in load_candidates(out_dir) if d.get("transcript")}
    lines = ["# 골든셋 v2 후보 블라인드 라벨링", "", BLIND_GUIDE]
    for sid, blind in sorted(key.items(), key=lambda kv: kv[1]):
        doc = docs[sid]
        lines += [f"## {blind}", "", f"> {doc['transcript']}", ""]
    return "\n".join(lines)


def blind_label_template(key: dict[str, str]) -> dict[str, Any]:
    """사용자가 채울 빈 라벨 파일."""

    def empty() -> dict[str, Any]:
        return {"value": "", "source": "", "flag": "", "candidates": [], "memo": ""}

    return {
        "version": 1,
        "labeler": "",
        "cases": {blind: {f: empty() for f in _FIELD_ORDER} for blind in sorted(key.values())},
    }


def _norm(value: object) -> str:
    return "".join(str(value or "").split())


def _value_agrees(field: str, gold: Any, label: dict[str, Any]) -> bool:
    raw_candidates = cast(list[object], label.get("candidates") or [])
    label_candidates = {_norm(c) for c in raw_candidates}
    gold_candidates = {_norm(c) for c in gold.candidates}
    if gold_candidates:
        return label_candidates == gold_candidates and not _norm(label.get("value"))
    if label_candidates:
        return False
    if field == "special_notes":  # 자유 문장은 값 유무만 비교 (러너와 같은 규칙)
        return bool(_norm(label.get("value"))) == bool(_norm(gold.value))
    return _norm(label.get("value")) == _norm(gold.value)


@dataclass
class BlindReport:
    """블라인드 라벨과 정답의 비교 결과."""

    total: int = 0
    value_agree: int = 0
    flag_agree: int = 0
    unlabeled: int = 0
    gray_total: int = 0
    gray_value_agree: int = 0
    disagreements: list[str] = field(default_factory=lambda: list[str]())
    uncertain: list[str] = field(default_factory=lambda: list[str]())

    @property
    def value_rate(self) -> float | None:
        """값 일치율."""
        return None if self.total == 0 else self.value_agree / self.total

    @property
    def flag_rate(self) -> float | None:
        """flag 일치율."""
        return None if self.total == 0 else self.flag_agree / self.total

    @property
    def value_rate_excluding_gray(self) -> float | None:
        """회색 필드(정책상 갈릴 수 있는 필드)를 뺀 값 일치율."""
        rest = self.total - self.gray_total
        return None if rest == 0 else (self.value_agree - self.gray_value_agree) / rest


def compare_blind(labels: dict[str, Any], key: dict[str, str], specs: list[Spec]) -> BlindReport:
    """사람 라벨을 스펙 정답과 비교한다. 값 일치와 flag 일치를 따로 센다."""
    by_id = {s.spec_id: s for s in specs}
    report = BlindReport()
    for sid, blind in sorted(key.items(), key=lambda kv: kv[1]):
        spec = by_id[sid]
        case = labels.get("cases", {}).get(blind, {})
        for fld in _FIELD_ORDER:
            gold = spec.gold[FieldKey(fld)]
            label = case.get(fld, {})
            gray = FieldKey(fld) in spec.gray_fields
            report.total += 1
            report.gray_total += gray
            if not (label.get("source") or label.get("flag") or _norm(label.get("value"))):
                report.unlabeled += 1
            value_ok = _value_agrees(fld, gold, label)
            flag_ok = (label.get("flag") or "") == gold.flag.value
            report.value_agree += value_ok
            report.flag_agree += flag_ok
            report.gray_value_agree += value_ok and gray
            if label.get("uncertain"):
                gold_text = gold.value or ", ".join(gold.candidates) or "(없음)"
                label_text = label.get("value") or "(없음)"
                report.uncertain.append(
                    f"{sid} {fld}: 정답 {gold_text} · 라벨 {label_text}"
                    f" · {'불일치' if not value_ok else '일치'} · 이유: {label.get('memo') or '-'}"
                )
            if not value_ok or not flag_ok:
                cand_list = cast(list[object], label.get("candidates") or [])
                shown_candidates = ", ".join(str(c) for c in cand_list)
                shown = label.get("value") or shown_candidates or "(없음)"
                gold_shown = gold.value or ", ".join(gold.candidates) or "(없음)"
                report.disagreements.append(
                    f"{sid} {fld}: 정답 {gold_shown}|{gold.flag.value}"
                    f" ↔ 라벨 {shown}|{label.get('flag') or '-'}"
                    + (" [gray]" if gray else "")
                    + (" [모호]" if label.get("uncertain") else "")
                    + (f" 메모: {label['memo']}" if label.get("memo") else "")
                )
    return report


def format_blind_report(report: BlindReport) -> str:
    """사람이 읽는 비교 요약."""

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v * 100:.1f}%"

    ok = (report.value_rate or 0) >= BLIND_AGREEMENT_TARGET
    lines = [
        f"필드 {report.total}개 (미작성 {report.unlabeled}개)",
        f"값 일치율 {pct(report.value_rate)} (목표 {BLIND_AGREEMENT_TARGET:.0%}): "
        f"{'달성' if ok else '미달'}",
        f"  회색 필드 제외 {pct(report.value_rate_excluding_gray)}"
        f" · flag 일치율 {pct(report.flag_rate)}",
    ]
    lines += [f"- {d}" for d in report.disagreements]
    if report.uncertain:
        lines += [f"라벨러가 애매하다고 표시한 필드 {len(report.uncertain)}개:"]
        lines += [f"  · {u}" for u in report.uncertain]
    return "\n".join(lines)
