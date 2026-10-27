"""특이사항 risk Sentinel과 수치 Rule Engine (설계: docs/rule-engine-sentinel-design.md).

규칙은 `rules/sentinel.yaml`에 있다. 모든 규칙은 flag를 risk로 올리기만 하고 값은 바꾸지 않는다.
전 항목이 provisional(임시)이며 진단이 아니라 "사람이 확인할 위험 신호"를 알리는 용도다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from carenote.evaluation.schema import FieldKey, Flag

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "sentinel.yaml"

GUARD_NAMES = frozenset({"negation", "conditional_advice", "third_party", "past_event"})

# 부정 끝맺음: "~지는 않", "~진 못", "~는 건 없", "~한 일은 없"
_NEG_END = (
    r"(?:(?:지|진)\s?(?:는|도)?\s?(?:않|못)|(?:건|게|것|일|거)[은는]?\s?(?:전혀\s)?(?:없|아니))"
)
# 증상 서술어 바로 뒤의 부정. 문장 속 무관한 "안"과 구분하려고 짧게 본다.
_NEGATION = re.compile(rf"^[가-힣\s]{{0,6}}?(?:는|은)?\s?{_NEG_END}")
# 나열 끝의 부정: "가슴이 답답하거나 숨이 차거나 하는 건 없었다". 거나 목록 뒤에 부정이 와야 하고,
# 쉼표 없이 짧게 이어질 때만 본다(뒤에 오는 다른 말의 '없다'가 새어 들어오지 않게).
_NEGATION_LIST = re.compile(
    rf"^[가-힣\s]{{0,4}}?거나(?:[^.,]{{0,16}}?거나)*[^.,]{{0,12}}?{_NEG_END}"
)
# 가정("~하면")과 조언("연락 달라")이 함께 있을 때. 나열된 뒤 증상에도 닿도록 창을 넓힌다.
_CONDITIONAL = re.compile(r"^[가-힣\s]{0,22}?(?:면|시면)")
# 앞쪽 14자 안의 제3자 주어. 증상의 주인이 이용자가 아니다.
_THIRD_PARTY = re.compile(
    r"(?:남편|아내|부인|아들|딸|따님|아드님|며느리|사위|손주|손자|손녀|동생|언니|형님|친구|이웃|저도|제가|옆집)[^.]{0,12}$"
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TAIL = 14
_COND_TAIL = 26
_PAST_HEAD = 24
_BP = re.compile(r"^(\d{2,3})/(\d{2,3})$")


class RuleError(ValueError):
    """규칙 파일이 잘못되었다."""


@dataclass(frozen=True)
class Signal:
    """Sentinel이 찾은 위험 신호. quote는 transcript의 부분 문자열이다."""

    rule_id: str
    category: str
    severity: Flag
    quote: str
    matched: str
    provisional: bool
    needs_expert: bool
    source: str = "rule"  # rule | llm | rule+llm


@dataclass(frozen=True)
class TextRule:
    """문장 단위 어휘 규칙."""

    rule_id: str
    category: str
    severity: Flag
    provisional: bool
    needs_expert: bool
    guards: frozenset[str]
    patterns: tuple[re.Pattern[str], ...]
    context_patterns: tuple[re.Pattern[str], ...] = ()
    context_window: int = 0


@dataclass(frozen=True)
class NumericRule:
    """수치 임계값 규칙. 수축기·이완기 중 하나라도 기준 이상이면 발동한다."""

    rule_id: str
    field: FieldKey
    severity: Flag
    provisional: bool
    systolic_gte: int | None
    diastolic_gte: int | None


@dataclass(frozen=True)
class RuleSet:
    """로드된 규칙 모음."""

    version: int
    text_rules: tuple[TextRule, ...]
    numeric_rules: tuple[NumericRule, ...]
    advice: re.Pattern[str]
    past: re.Pattern[str]


def _mapping(raw: object, where: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuleError(f"{where}: 매핑이어야 한다")
    return cast(Mapping[str, Any], raw)


def _strings(raw: object, where: str) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str) or not raw:
        raise RuleError(f"{where}: 비어 있지 않은 문자열 목록이어야 한다")
    items = cast(Sequence[object], raw)
    if not all(isinstance(item, str) for item in items):
        raise RuleError(f"{where}: 문자열 목록이어야 한다")
    return tuple(cast(Sequence[str], items))


def _compile(pattern: str, where: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise RuleError(f"{where}: 정규식 오류 {pattern!r} ({exc})") from exc


def _severity(raw: object, where: str) -> Flag:
    try:
        flag = Flag(str(raw))
    except ValueError as exc:
        raise RuleError(f"{where}: severity가 올바르지 않다 ({raw!r})") from exc
    if flag is Flag.OK:
        raise RuleError(f"{where}: severity는 ok일 수 없다 (규칙은 flag를 올리기만 한다)")
    return flag


def _text_rule(raw: object, seen: set[str]) -> TextRule:
    data = _mapping(raw, "text_rules[]")
    rule_id = str(data.get("id", ""))
    if not rule_id:
        raise RuleError("text_rules[]: id가 필요하다")
    if rule_id in seen:
        raise RuleError(f"{rule_id}: id가 중복되었다")
    seen.add(rule_id)
    where = f"text_rules[{rule_id}]"
    guards = frozenset(_strings(data.get("guards", ["negation"]), f"{where}.guards"))
    unknown = guards - GUARD_NAMES
    if unknown:
        raise RuleError(f"{where}: 알 수 없는 가드 {sorted(unknown)}")
    patterns = tuple(
        _compile(p, where) for p in _strings(data.get("patterns"), f"{where}.patterns")
    )
    context_patterns: tuple[re.Pattern[str], ...] = ()
    window = int(data.get("context_window", 0))
    if "context_patterns" in data:
        context_patterns = tuple(
            _compile(p, where)
            for p in _strings(data["context_patterns"], f"{where}.context_patterns")
        )
        if window < 1:
            raise RuleError(f"{where}: context_patterns에는 context_window >= 1이 필요하다")
    return TextRule(
        rule_id=rule_id,
        category=str(data.get("category", rule_id)),
        severity=_severity(data.get("severity", "risk"), where),
        provisional=bool(data.get("provisional", True)),
        needs_expert=bool(data.get("needs_expert", False)),
        guards=guards,
        patterns=patterns,
        context_patterns=context_patterns,
        context_window=window,
    )


def _numeric_rule(raw: object) -> NumericRule:
    data = _mapping(raw, "numeric_rules[]")
    rule_id = str(data.get("id", ""))
    where = f"numeric_rules[{rule_id}]"
    try:
        field = FieldKey(str(data.get("field")))
    except ValueError as exc:
        raise RuleError(f"{where}: field가 올바르지 않다") from exc
    if field is not FieldKey.BLOOD_PRESSURE:
        raise RuleError(f"{where}: 현재 수치 규칙은 blood_pressure만 지원한다")
    systolic = data.get("systolic_gte")
    diastolic = data.get("diastolic_gte")
    if systolic is None and diastolic is None:
        raise RuleError(f"{where}: systolic_gte 또는 diastolic_gte가 필요하다")
    return NumericRule(
        rule_id=rule_id,
        field=field,
        severity=_severity(data.get("severity", "risk"), where),
        provisional=bool(data.get("provisional", True)),
        systolic_gte=None if systolic is None else int(systolic),
        diastolic_gte=None if diastolic is None else int(diastolic),
    )


def parse_rules(text: str) -> RuleSet:
    """YAML 문자열을 규칙 모음으로 바꾼다. 잘못되면 RuleError."""
    try:
        loaded: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RuleError(f"YAML 구문 오류: {exc}") from exc
    data = _mapping(loaded, "최상위")
    seen: set[str] = set()
    text_rules = tuple(
        _text_rule(r, seen) for r in cast(Sequence[object], data.get("text_rules", []))
    )
    numeric_rules = tuple(
        _numeric_rule(r) for r in cast(Sequence[object], data.get("numeric_rules", []))
    )
    return RuleSet(
        version=int(data.get("version", 0)),
        text_rules=text_rules,
        numeric_rules=numeric_rules,
        advice=_compile(str(data.get("advice_words", "연락|전화|알려|말씀")), "advice_words"),
        past=_compile(str(data.get("past_words", "작년|지난해|예전")), "past_words"),
    )


def load_rules(path: Path | None = None) -> RuleSet:
    """규칙 파일을 읽는다. 기본은 저장소의 rules/sentinel.yaml."""
    target = path or DEFAULT_RULES_PATH
    return parse_rules(target.read_text(encoding="utf-8"))


def split_sentences(text: str) -> list[str]:
    """마침표·물음표·느낌표 뒤 공백으로 문장을 나눈다. 부분 문자열 성질을 유지한다."""
    return [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def guarded_span(rules: RuleSet, rule: TextRule, sentence: str, start: int, end: int) -> bool:
    """문장 안 [start, end) 구간에 규칙의 가드 중 하나라도 걸리면 True (신호를 버린다).

    부정·조건은 구간 뒤쪽 문맥, 제3자·과거는 앞쪽 문맥을 본다. 구간 안쪽은 보지 않는다.
    """
    tail = sentence[end : end + _TAIL]
    head = sentence[:start]
    if "negation" in rule.guards and (
        _NEGATION.search(tail) or _NEGATION_LIST.search(sentence[end:])
    ):
        return True
    if (
        "conditional_advice" in rule.guards
        and _CONDITIONAL.search(sentence[end : end + _COND_TAIL])
        and rules.advice.search(sentence[end:])
    ):
        return True
    if "third_party" in rule.guards and _THIRD_PARTY.search(head[-_TAIL:]):
        return True
    return "past_event" in rule.guards and bool(rules.past.search(head[-_PAST_HEAD:]))


def _guarded(rules: RuleSet, rule: TextRule, sentence: str, match: re.Match[str]) -> bool:
    return guarded_span(rules, rule, sentence, match.start(), match.end())


def scan_text(transcript: str, rules: RuleSet) -> list[Signal]:
    """transcript를 문장 단위로 훑어 위험 신호를 모은다 (같은 규칙·같은 문장은 한 번만)."""
    signals: list[Signal] = []
    sentences = split_sentences(transcript)
    for index, sentence in enumerate(sentences):
        for rule in rules.text_rules:
            if rule.context_patterns:
                window = " ".join(sentences[max(0, index - rule.context_window) : index])
                scope = f"{window} {sentence}"
                if not any(p.search(scope) for p in rule.context_patterns):
                    continue
            for pattern in rule.patterns:
                hit = next(
                    (
                        m
                        for m in pattern.finditer(sentence)
                        if not _guarded(rules, rule, sentence, m)
                    ),
                    None,
                )
                if hit is None:
                    continue
                signals.append(
                    Signal(
                        rule_id=rule.rule_id,
                        category=rule.category,
                        severity=rule.severity,
                        quote=sentence.strip(),
                        matched=hit.group(),
                        provisional=rule.provisional,
                        needs_expert=rule.needs_expert,
                    )
                )
                break
    return [s for s in signals if s.quote in transcript]


def numeric_signals(field: FieldKey, value: str | None, rules: RuleSet) -> list[NumericRule]:
    """수치 값이 걸리는 수치 규칙 목록. 형식이 다르거나 값이 없으면 비어 있다."""
    if value is None or field is not FieldKey.BLOOD_PRESSURE:
        return []
    match = _BP.match(value.strip())
    if match is None:
        return []
    systolic, diastolic = int(match.group(1)), int(match.group(2))
    hits: list[NumericRule] = []
    for rule in rules.numeric_rules:
        if rule.field is not field:
            continue
        high_s = rule.systolic_gte is not None and systolic >= rule.systolic_gte
        high_d = rule.diastolic_gte is not None and diastolic >= rule.diastolic_gte
        if high_s or high_d:
            hits.append(rule)
    return hits
