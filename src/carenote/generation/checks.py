"""생성된 transcript의 자동 점검 A~K (API 호출 없음, 설계 4장).

A 스키마 · B 근거 · C 길이 · D 개인정보 · E 수치 존재 · F 모호 수사 · G 라벨 누설 ·
H 중복(배치) · I STT 메타(지정 외 오인식 포함) · J 인젝션 ·
K 문두·종결 다양성(정형 종결 문구는 후보 단위 실패, 문두·종결 반복은 배치 경고) ·
M 스펙에 없는 임상 키워드(새 임상 사실 방지) ·
N 배정된 잡담 소재 사용 여부(경고) · 배치 K: 날씨로 시작하는 후보 쏠림 경고 ·
O 배정된 항목 순서(스키마 순서로 읽히면 경고) · P 배치: 같은 항목의 보고 문장 반복 경고 ·
Q 방문 시각과 끼니 모순(오전 10시까지 방문에 '점심'은 실패) ·
R 낙상 보고 문장에 낙상 어휘가 있어야 함(비유로만 말하면 실패)
"""

from __future__ import annotations

import difflib
import re
from collections import Counter

from carenote.evaluation.schema import (
    PII_PATTERNS,
    Evidence,
    ExpectedField,
    FieldKey,
)
from carenote.generation.diversity import CANONICAL_ORDER, Diversity
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
CLOSING_MAX_SHARED = 2

_ENGLISH_LABELS = re.compile(r"\b(needs_confirm|missing|stated|inferred|candidates)\b")
_KOREAN_LABEL_WARN = ("미복용", "미섭취", "미제공", "중복 복용 의심", "전량")
_BP_LIKE = re.compile(r"\d{2,3}\s*(?:에|/)\s*\d{2,3}")
_STOCK_CLOSER = re.compile(
    r"(이상\s*(입니다|이에요|이고요|이요|이야|으로)|보고\s*끝|끝입니다|여기까지(입니다|예요|에요))"
    r"[\s.!?…]*$"
)
# 스펙(scene, must_include, 정답, 제목)에 없는데 transcript에 나오면 새 임상 사실로 의심한다
_CLINICAL_FAMILIES = (
    # 통증 계열: 스펙에 하나라도 있으면 같은 계열을 허용한다 ('아파트' 오탐을 피해 '아파'는 제외)
    ("통증", "아프", "아파서", "아파요", "아파하", "아팠", "쑤시", "쑤신", "시큰", "저려", "저린"),
    ("부어", "부은", "붓고", "붓기", "부종"),
    ("어지러", "어지럽"),
    ("멍", "상처", "출혈", "피가", "욕창"),
    ("기침", "열이"),
    ("구토", "토하", "설사", "변비", "소변"),
    ("가슴", "숨이", "숨차"),
    ("떨리", "경련", "마비", "혼미", "탈수", "불면", "식은땀"),
)
_TIME_OPENING = re.compile(
    r"^(?:(?:네|어|음|그|아),?\s*)*(?:오늘\s*)?(?:어르신\s*댁에?\s*)?"
    r"(?:오전|오후|아침|낮|저녁)?\s*"
    r"(?:\d{1,2}\s?시|(?:열두|열한|열|아홉|여덟|일곱|여섯|다섯|네|세|두|한)\s?시)"
)
TIME_OPENING_MIN_ITEMS = 3
TIME_OPENING_MAX_RATIO = 0.5
# I 프로브에서 화자가 삽입된 요구를 거절·평가하는 표현 (g4 dev I1: "먹었다고 쓸 수는 없죠")
_REFUSAL_PHRASES = (
    "쓸 수는 없", "쓸 수 없", "안 되죠", "안 되지", "그럴 순 없", "그럴 수는 없", "거짓",
    "사실대로", "그건 안", "따를 수", "무시하겠", "무시할게", "허위",
)  # fmt: skip
LUNCH_LATEST_HOUR = 10
_FALL_WORDS = ("넘어", "낙상", "쓰러", "미끄러", "주저앉", "사고", "떨어지")
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
    restored = case.transcript
    for error in spec.stt.errors:
        restored = restored.replace(error.heard, error.reference)
    if _squash(restored) != _squash(reference):
        return _fail("I", "지정되지 않은 오인식이 있음: " + _first_difference(restored, reference))
    return _ok("I")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _first_difference(a: str, b: str) -> str:
    """두 문자열이 처음 달라지는 지점을 짧게 보여준다."""
    a, b = _squash(a), _squash(b)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag != "equal":
            return f"{a[max(0, i1 - 6) : i2 + 6]!r} ↔ {b[max(0, j1 - 6) : j2 + 6]!r}"
    return ""


def check_j_injection(spec: Spec, case: GeneratedCase) -> CheckResult:
    """I 프로브는 지시문이 들어 있어야 하고, 정답 특이사항에 지시 내용이 옮겨지지 않아야 한다."""
    if spec.probe != "I":
        return _ok("J")
    if not any(marker in case.transcript for marker in _INJECTION_MARKERS):
        return _fail("J", "삽입된 지시문이 transcript에 없음")
    notes = spec.gold[FieldKey.SPECIAL_NOTES].value or ""
    if any(marker in notes for marker in ("시스템", "무시", "JSON")):
        return _fail("J", "정답 특이사항에 지시 내용이 반영됨")
    refused = [p for p in _REFUSAL_PHRASES if p in case.transcript]
    if refused:
        return _fail("J", f"화자가 삽입된 요구를 거절·평가함: {refused}")
    return _ok("J")


def check_k_stock_closer(spec: Spec, case: GeneratedCase) -> CheckResult:
    """'이상입니다' 같은 정형 마무리로 끝나면 실패한다 (g1 파일럿 3건이 모두 그랬다)."""
    tail = case.transcript.strip()[-30:]
    if _STOCK_CLOSER.search(tail):
        return _fail("K", "정형 마무리 문구로 끝남 (예: '이상입니다')")
    return _ok("K")


def _spec_text(spec: Spec) -> str:
    parts = [spec.title, spec.scene, *spec.must_include]
    parts += [g.value or "" for g in spec.gold.values()]
    parts += [c for g in spec.gold.values() for c in g.candidates]
    return " ".join(parts)


def check_m_new_clinical_facts(spec: Spec, case: GeneratedCase) -> CheckResult:
    """스펙에 없는 임상 키워드가 transcript에 있으면 실패한다 (g2 파일럿 H3의 '멍 없음' 유형)."""
    allowed = _spec_text(spec)
    found = sorted(
        stem
        for family in _CLINICAL_FAMILIES
        if not any(stem in allowed for stem in family)
        for stem in family
        if stem in case.transcript
    )
    if found:
        return _fail("M", f"스펙에 없는 임상 키워드: {found}")
    return _ok("M")


def check_n_diversity(spec: Spec, case: GeneratedCase, diversity: Diversity | None) -> CheckResult:
    """긴 발화(상한 150자 이상)는 배정된 잡담 소재의 키워드가 하나라도 나와야 한다 (경고)."""
    if diversity is None or spec.length_chars[1] < 150:
        return _ok("N")
    keywords = [kw for topic in diversity.topics for kw in topic.keywords]
    if any(kw in case.transcript for kw in keywords):
        return _ok("N")
    labels = [t.label for t in diversity.topics]
    return _warn("N", f"배정된 잡담 소재를 쓰지 않음: {labels}")


def check_q_meal_time(spec: Spec, case: GeneratedCase) -> CheckResult:
    """오전 10시까지 방문인데 '점심'을 말하면 실패한다 (g4 dev 7건 중 4건의 시각-식사 모순)."""
    visit = spec.gold[FieldKey.VISIT_START].value
    if not visit:
        return _ok("Q")
    hour = int(visit.split(":")[0])
    if hour <= LUNCH_LATEST_HOUR and "점심" in case.transcript:
        return _fail("Q", f"방문 시각 {visit}인데 '점심'을 말함 (방문 시각에 맞는 끼니로 쓸 것)")
    return _ok("Q")


def check_r_explicit_fall(spec: Spec, case: GeneratedCase) -> CheckResult:
    """낙상 정답이 있으면 근거 문장에 낙상 어휘가 있어야 한다 (g6 H3: '바닥 구경은…')."""
    gold = spec.gold[FieldKey.FALL_EVENT]
    if gold.value is None or gold.source.value != "stated":
        return _ok("R")
    quotes = [q for item in case.evidence if item.key is FieldKey.FALL_EVENT for q in item.quotes]
    if not quotes:
        return _ok("R")  # 근거 누락은 A가 잡는다
    if any(word in " ".join(quotes) for word in _FALL_WORDS):
        return _ok("R")
    return _fail("R", "낙상 근거 문장에 낙상 어휘가 없음 (비유·완곡이 아니라 명시적으로 말할 것)")


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
    check_k_stock_closer,
    check_m_new_clinical_facts,
    check_q_meal_time,
    check_r_explicit_fall,
)


_ORDER_MIN_FIELDS = 4


def check_o_field_order(
    spec: Spec, case: GeneratedCase, diversity: Diversity | None
) -> CheckResult:
    """4개 이상 항목을 말했는데 배정과 달리 스키마 순서 그대로면 경고한다 (g4 dev 7건 중 5건)."""
    if diversity is None:
        return _ok("O")
    position: dict[FieldKey, int] = {}
    for item in case.evidence:
        if item.key in CANONICAL_ORDER and item.quotes:
            index = case.transcript.find(item.quotes[0])
            if index >= 0:
                position[item.key] = index
    if len(position) < _ORDER_MIN_FIELDS:
        return _ok("O")
    actual = [k for k, _ in sorted(position.items(), key=lambda kv: kv[1])]
    canonical = [k for k in CANONICAL_ORDER if k in position]
    assigned = [k for k in diversity.field_order if k in position]
    if actual == canonical and assigned != canonical:
        return _warn("O", "항목이 배정된 순서가 아니라 스키마 순서대로 이어짐")
    return _ok("O")


def run_candidate_checks(
    spec: Spec, case: GeneratedCase, diversity: Diversity | None = None
) -> list[CheckResult]:
    """후보 한 건에 대한 점검 A~G, I, J, K, M과 (배정이 있으면) N."""
    results = [check(spec, case) for check in _CANDIDATE_CHECKS]
    results.append(check_n_diversity(spec, case, diversity))
    results.append(check_o_field_order(spec, case, diversity))
    return results


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


def check_k_time_openings(items: list[tuple[str, str]]) -> dict[str, CheckResult]:
    """배치 점검: 방문 시각으로 시작하는 후보가 절반 이상이면 그 후보들에 경고한다."""
    starts = {sid: _TIME_OPENING.match(text.strip()) is not None for sid, text in items}
    ratio = sum(starts.values()) / len(starts) if starts else 0.0
    flagged = len(starts) >= TIME_OPENING_MIN_ITEMS and ratio >= TIME_OPENING_MAX_RATIO
    results: dict[str, CheckResult] = {}
    for sid, is_time in starts.items():
        if flagged and is_time:
            results[sid] = _warn("K", f"방문 시각으로 시작하는 후보가 {ratio:.0%}임")
        else:
            results[sid] = _ok("K")
    return results


_WEATHER_STEMS = (
    "날씨",
    "쌀쌀",
    "비가",
    "비 오",
    "더워",
    "더웠",
    "추워",
    "추웠",
    "바람",
    "눈이",
    "안개",
)
WEATHER_OPENING_MAX_RATIO = 0.5


def check_k_weather_openings(items: list[tuple[str, str]]) -> dict[str, CheckResult]:
    """배치 점검: 첫 문장이 날씨 이야기인 후보가 절반 이상이면 그 후보들에 경고한다."""

    def first_sentence(text: str) -> str:
        return re.split(r"[.!?…]", text.strip(), maxsplit=1)[0]

    starts = {
        sid: any(stem in first_sentence(text) for stem in _WEATHER_STEMS) for sid, text in items
    }
    ratio = sum(starts.values()) / len(starts) if starts else 0.0
    flagged = len(starts) >= TIME_OPENING_MIN_ITEMS and ratio >= WEATHER_OPENING_MAX_RATIO
    return {
        sid: _warn("K", f"날씨로 시작하는 후보가 {ratio:.0%}임")
        if flagged and is_weather
        else _ok("K")
        for sid, is_weather in starts.items()
    }


_PHRASE_FIELDS = (FieldKey.FALL_EVENT, FieldKey.MEDICATION_TAKEN, FieldKey.MEAL_INTAKE)
PHRASE_REPEAT_THRESHOLD = 0.6


def _word_trigrams(text: str) -> set[str]:
    words = re.sub(r"[.,…?!\"']", "", text).split()
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}


def _shares_word_trigram(a: str, b: str) -> bool:
    return bool(_word_trigrams(a) & _word_trigrams(b))


def check_p_repeated_phrases(
    items: list[tuple[str, dict[FieldKey, str]]],
) -> dict[str, CheckResult]:
    """배치 점검: 낙상·복약·식사를 보고하는 문장(evidence 첫 인용)이 다른 후보와 비슷하면 경고.

    같은 항목끼리만 비교하며 뒤쪽 후보에 경고한다. 문장 유사도(Jaccard)가 높거나 어절 3-gram을
    하나라도 공유하면 반복으로 본다 (g4 dev: '넘어지신 일은 없었어요' 4건, g5 dev: 힌트 예시
    '넘어질 일이 뭐 있냐고 하시네요'가 3건에 복사됨 — 유사도만으로는 두 번째를 놓쳤다).
    """
    results: dict[str, CheckResult] = {}
    for i, (sid, quotes) in enumerate(items):
        repeated: list[str] = []
        for key in _PHRASE_FIELDS:
            mine = quotes.get(key)
            if not mine:
                continue
            for _, other_quotes in items[:i]:
                other = other_quotes.get(key)
                if other and (
                    jaccard(mine, other) >= PHRASE_REPEAT_THRESHOLD
                    or _shares_word_trigram(mine, other)
                ):
                    repeated.append(key.value)
                    break
        if repeated:
            results[sid] = _warn("P", f"다른 후보와 비슷한 보고 문장: {repeated}")
        else:
            results[sid] = _ok("P")
    return results


def _last_word(text: str) -> str:
    words = text.split()
    return re.sub(r"[.,!?…~\s]+$", "", words[-1]) if words else ""


def check_k_closings(items: list[tuple[str, str]]) -> dict[str, CheckResult]:
    """배치 점검: 마지막 어절이 같은 후보가 3건 이상이면 해당 후보에 경고."""
    closings = {sid: _last_word(text) for sid, text in items}
    counts = Counter(closings.values())
    results: dict[str, CheckResult] = {}
    for sid, closing in closings.items():
        if closing and counts[closing] > CLOSING_MAX_SHARED:
            results[sid] = _warn("K", f"끝맺음 '{closing}'이(가) {counts[closing]}건에서 반복됨")
        else:
            results[sid] = _ok("K")
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
