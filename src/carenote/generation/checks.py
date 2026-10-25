"""생성된 transcript의 자동 점검 A~K (API 호출 없음, 설계 4장).

A 스키마 · B 근거 · C 길이 · D 개인정보 · E 수치 존재 · F 모호 수사 · G 라벨 누설 ·
H 중복(배치) · I STT 메타 · J 인젝션 · K 문두 다양성(배치)
"""

from __future__ import annotations

import re
from collections import Counter

from carenote.evaluation.schema import (
    PII_PATTERNS,
    Evidence,
    ExpectedField,
    FieldKey,
)
from carenote.generation.numerals import (
    extract_hours,
    extract_numbers,
    find_ambiguous_numerals,
    hour_matches,
)
from carenote.generation.schema import CheckResult, GeneratedCase
from carenote.generation.spec import Spec

LENGTH_TOLERANCE = 0.10
DUPLICATE_THRESHOLD = 0.5
OPENING_WORDS = 6
OPENING_MAX_SHARED = 3

_ENGLISH_LABELS = re.compile(r"\b(needs_confirm|missing|stated|inferred|candidates)\b")
_KOREAN_LABEL_WARN = ("미복용", "미섭취", "미제공", "중복 복용 의심", "전량")
_BP_LIKE = re.compile(r"\d{2,3}\s*(?:에|/)\s*\d{2,3}")
_INJECTION_MARKERS = ("시스템", "지시", "무시", "적어", "출력", "JSON", "{")


def _ok(code: str) -> CheckResult:
    return CheckResult(code=code, level="pass")


def _fail(code: str, message: str) -> CheckResult:
    return CheckResult(code=code, level="fail", message=message)


def _warn(code: str, message: str) -> CheckResult:
    return CheckResult(code=code, level="warn", message=message)


def _numbers_source(spec: Spec, case: GeneratedCase) -> str:
    """S 프로브의 정답은 참고 발화(reference) 기준이다."""
    if spec.stt is not None and case.reference_transcript:
        return case.reference_transcript
    return case.transcript


def check_a_schema(spec: Spec, case: GeneratedCase) -> CheckResult:
    """정답 필드를 ExpectedField 규칙(누락·후보·근거)으로 조립해 본다."""
    quotes: dict[FieldKey, list[str]] = {}
    for item in case.evidence:
        if item.key in quotes:
            return _fail("A", f"evidence에 {item.key.value}가 중복됨")
        quotes[item.key] = item.quotes
    problems: list[str] = []
    for key, gold in spec.gold.items():
        evidence = tuple(Evidence(text=q) for q in quotes.get(key, []) if q.strip())
        try:
            ExpectedField(
                key=key,
                value=gold.value,
                source=gold.source,
                evidence=evidence,
                flag=gold.flag,
                candidates=gold.candidates,
            )
        except ValueError as exc:
            problems.append(f"{key.value}: {exc}")
    if problems:
        return _fail("A", "; ".join(problems))
    return _ok("A")


def check_b_evidence(spec: Spec, case: GeneratedCase) -> CheckResult:
    """모든 인용이 transcript의 부분 문자열이어야 한다."""
    bad = [
        f"{item.key.value}: {q!r}"
        for item in case.evidence
        for q in item.quotes
        if q not in case.transcript
    ]
    if bad:
        return _fail("B", "transcript에 없는 인용: " + "; ".join(bad))
    return _ok("B")


def check_c_length(spec: Spec, case: GeneratedCase) -> CheckResult:
    """길이 범위. ±10% 안이면 경고, 그 밖이면 실패."""
    low, high = spec.length_chars
    length = len(case.transcript)
    if low <= length <= high:
        return _ok("C")
    if low * (1 - LENGTH_TOLERANCE) <= length <= high * (1 + LENGTH_TOLERANCE):
        return _warn("C", f"길이 {length}자가 목표 {low}~{high}자를 약간 벗어남")
    return _fail("C", f"길이 {length}자가 목표 {low}~{high}자 밖")


def check_d_pii(spec: Spec, case: GeneratedCase) -> CheckResult:
    """주민번호·휴대전화 패턴이 없어야 한다."""
    texts = [case.transcript, case.reference_transcript or ""]
    if any(p.search(t) for p in PII_PATTERNS for t in texts):
        return _fail("D", "개인정보 패턴 발견")
    return _ok("D")


def _required_numbers(spec: Spec) -> list[tuple[FieldKey, str, set[int]]]:
    """정답에서 transcript에 나타나야 하는 숫자 (필드, 설명, 숫자 집합)."""
    required: list[tuple[FieldKey, str, set[int]]] = []
    for key in (FieldKey.BLOOD_PRESSURE, FieldKey.PULSE):
        gold = spec.gold[key]
        values = list(gold.candidates) if gold.candidates else ([gold.value] if gold.value else [])
        for value in values:
            nums = {int(n) for n in re.findall(r"\d+", value)}
            if nums:
                required.append((key, value, nums))
    return required


def check_e_numbers(spec: Spec, case: GeneratedCase) -> CheckResult:
    """정답의 수치·시각이 transcript에 나타나는지, 정답이 비어 있는 혈압 모양 수치가 새지 않는지."""
    text = _numbers_source(spec, case)
    found = extract_numbers(text)
    problems: list[str] = []
    for key, value, nums in _required_numbers(spec):
        missing = sorted(nums - found)
        if missing:
            problems.append(f"{key.value} {value}: {missing} 가 transcript에 없음")

    visit = spec.gold[FieldKey.VISIT_START].value
    if visit:
        hour, minute = (int(x) for x in visit.split(":"))
        if not hour_matches(hour, extract_hours(text) | found):
            problems.append(f"visit_start {visit}: 시각이 transcript에 없음")
        if minute == 30 and "반" not in text and 30 not in found and "삼십" not in text:
            problems.append(f"visit_start {visit}: '30분/반' 표현이 없음")
    if problems:
        return _fail("E", "; ".join(problems))

    bp_gold = spec.gold[FieldKey.BLOOD_PRESSURE]
    if bp_gold.value is None and not bp_gold.candidates and _BP_LIKE.search(case.transcript):
        return _warn("E", "혈압이 정답에 없는데 혈압 모양의 수치가 transcript에 있음")
    return _ok("E")


def check_f_ambiguous_numerals(spec: Spec, case: GeneratedCase) -> CheckResult:
    """'칠십이'처럼 70+이/72로 둘 다 읽히는 표현은 경고한다 (v1 gs-010·015 교훈)."""
    found = find_ambiguous_numerals(case.transcript)
    if found:
        return _warn("F", f"모호한 한글 수사: {sorted(set(found))}")
    return _ok("F")


def check_g_label_leak(spec: Spec, case: GeneratedCase) -> CheckResult:
    """영문 라벨은 실패, 정답 어휘가 라벨 형태로 나오면 경고."""
    text = case.transcript
    if _ENGLISH_LABELS.search(text):
        return _fail("G", "영문 라벨 어휘가 transcript에 있음")
    leaked = [w for w in _KOREAN_LABEL_WARN if w in text]
    if leaked:
        return _warn("G", f"라벨 어휘 의심: {leaked}")
    return _ok("G")


def check_i_stt(spec: Spec, case: GeneratedCase) -> CheckResult:
    """S 프로브는 참고 발화와 오인식 쌍이 반영돼야 하고, 다른 프로브는 참고 발화가 없어야 한다."""
    if spec.stt is None:
        if case.reference_transcript is not None:
            return _fail("I", "S 프로브가 아닌데 reference_transcript가 있음")
        return _ok("I")
    reference = case.reference_transcript
    if not reference:
        return _fail("I", "reference_transcript가 없음")
    if reference == case.transcript:
        return _fail("I", "reference와 transcript가 같음")
    problems = [
        f"{e.field.value}: heard {e.heard!r}가 transcript에 없음"
        for e in spec.stt.errors
        if e.heard not in case.transcript
    ]
    problems += [
        f"{e.field.value}: reference {e.reference!r}가 reference_transcript에 없음"
        for e in spec.stt.errors
        if e.reference not in reference
    ]
    if problems:
        return _fail("I", "; ".join(problems))
    return _ok("I")


def check_j_injection(spec: Spec, case: GeneratedCase) -> CheckResult:
    """I 프로브는 지시문이 들어 있어야 하고, 정답 특이사항에 지시 내용이 옮겨지지 않아야 한다."""
    if spec.probe != "I":
        return _ok("J")
    if not any(marker in case.transcript for marker in _INJECTION_MARKERS):
        return _fail("J", "삽입된 지시문이 transcript에 없음")
    notes = spec.gold[FieldKey.SPECIAL_NOTES].value or ""
    if any(marker in notes for marker in ("시스템", "무시", "JSON")):
        return _fail("J", "정답 특이사항에 지시 내용이 반영됨")
    return _ok("J")


_CANDIDATE_CHECKS = (
    check_a_schema,
    check_b_evidence,
    check_c_length,
    check_d_pii,
    check_e_numbers,
    check_f_ambiguous_numerals,
    check_g_label_leak,
    check_i_stt,
    check_j_injection,
)


def run_candidate_checks(spec: Spec, case: GeneratedCase) -> list[CheckResult]:
    """후보 한 건에 대한 점검 A~G, I, J."""
    return [check(spec, case) for check in _CANDIDATE_CHECKS]


def failures(results: list[CheckResult]) -> list[CheckResult]:
    """실패한 점검만."""
    return [r for r in results if r.level == "fail"]


def _shingles(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[i : i + n] for i in range(max(len(compact) - n + 1, 0))}


def jaccard(a: str, b: str) -> float:
    """문자 3-gram Jaccard 유사도."""
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def check_h_duplicates(
    items: list[tuple[str, str]], existing_texts: list[str]
) -> dict[str, CheckResult]:
    """배치 점검: (spec_id, transcript) 목록이 서로, 그리고 기존 텍스트와 너무 비슷하지 않은가.

    같은 쌍에서는 목록 뒤쪽 항목이 실패 처리된다.
    """
    results: dict[str, CheckResult] = {}
    for i, (spec_id, text) in enumerate(items):
        worst = 0.0
        others = [t for _, t in items[:i]] + existing_texts
        for other in others:
            worst = max(worst, jaccard(text, other))
        if worst >= DUPLICATE_THRESHOLD:
            results[spec_id] = _fail("H", f"다른 텍스트와 유사도 {worst:.2f}")
        else:
            results[spec_id] = _ok("H")
    return results


def check_k_openings(items: list[tuple[str, str]]) -> dict[str, CheckResult]:
    """배치 점검: 첫 6어절이 같은 후보가 4건 이상이면 해당 후보에 경고."""
    openings = {sid: " ".join(text.split()[:OPENING_WORDS]) for sid, text in items}
    counts = Counter(openings.values())
    results: dict[str, CheckResult] = {}
    for sid, opening in openings.items():
        if counts[opening] > OPENING_MAX_SHARED:
            results[sid] = _warn("K", f"문두 '{opening}'가 {counts[opening]}건에서 반복됨")
        else:
            results[sid] = _ok("K")
    return results
