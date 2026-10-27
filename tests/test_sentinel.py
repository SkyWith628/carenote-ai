"""특이사항 risk Sentinel · 수치 Rule Engine · Consensus 단위 테스트.

문장은 모두 합성이다(규칙 정합성만 증명한다). 정책 표의 어휘를 옮긴 규칙이라 일반화 recall의 증거가
아니다. test 분할은 이 규칙의 독립 평가가 아니므로 어떤 테스트도 test 케이스를 읽지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carenote.evaluation.__main__ import EXIT_REFUSED, main
from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.loader import load_golden
from carenote.evaluation.schema import Evidence, FieldKey, Flag, Source
from carenote.extraction.flagger import FlaggingExtractor, flag_field
from carenote.extraction.sentinel import (
    DEFAULT_RULES_PATH,
    RuleError,
    RuleSet,
    load_rules,
    numeric_signals,
    parse_rules,
    scan_text,
    split_sentences,
)

GOLDEN = Path(__file__).parent.parent / "data" / "golden"
RULES = load_rules()


def ids(text: str) -> set[str]:
    return {s.rule_id for s in scan_text(text, RULES)}


# ---- 양성: 분류별 위험 신호 -------------------------------------------------------------------

POSITIVES = [
    ("chest_resp", "오늘은 가슴이 답답하다고 하셨어요."),
    ("chest_resp", "가슴이 너무 아프다고 자꾸 말씀하세요."),
    ("chest_resp", "계단을 조금 올랐더니 숨이 차서 쉬셨어요."),
    ("chest_resp", "호흡 곤란을 호소하셨습니다."),
    ("neuro", "말씀이 평소보다 어눌하셨어요."),
    ("neuro", "발음이 이상하게 들렸습니다."),
    ("neuro", "왼쪽 팔에 힘이 없다고 하셨어요."),
    ("neuro", "오른손에 힘이 빠져서 컵을 놓치셨어요."),
    ("intake_days", "사흘째 식사를 거부하세요."),
    ("intake_days", "이틀 동안 밥을 안 드셨다고 합니다."),
    ("intake_days", "물도 사흘째 안 드시려고 해요."),
    ("body_change", "발이 퉁퉁 부어 있었어요."),
    ("body_change", "얼굴이 갑자기 부었습니다."),
    ("body_change", "다리에 부종이 심했어요."),
    ("body_change", "상처가 곪아 있었습니다."),
    ("circumstance", "팔에 멍이 여러 군데 있었어요."),
    ("circumstance", "누가 때렸다는 말을 하셨습니다."),
    ("circumstance", "아드님이 오시면 말을 안 하고 위축되세요."),
    ("post_fall", "어제 화장실에서 넘어지셨어요. 엉덩이가 많이 아프다고 하세요."),
    ("post_fall", "넘어지신 뒤로 걷기 힘들어하세요."),
    ("post_fall", "지난밤에 쓰러지셨다고 해요. 잠시 후 팔이 붓고 아파했어요."),
]


@pytest.mark.parametrize(("rule_id", "text"), POSITIVES)
def test_positive_signal(rule_id: str, text: str) -> None:
    assert rule_id in ids(text)


# ---- 어려운 음성: 가드가 오탐을 막는다 --------------------------------------------------------

NEGATIVES = [
    # 부정
    "가슴이 답답하지는 않으셨어요.",
    "숨이 차지 않는다고 하셨어요.",
    "말씀이 어눌하지도 않고 또렷하셨어요.",
    "왼쪽 팔에 힘이 없지는 않았어요.",
    "발이 붓지 않았습니다.",
    # 조언·가정
    "혹시 가슴이 답답하면 바로 연락 달라고 안내드렸어요.",
    "숨이 차면 센터로 전화하라고 말씀드렸습니다.",
    # 제3자
    "남편분이 가슴이 답답하다고 하셨어요.",
    "옆집 분은 숨이 차서 병원에 가셨대요.",
    "아드님이 발이 부었다고 하더라고요.",
    # 만성 통증
    "무릎이 쑤시고 아프다고 하셨어요.",
    "허리가 늘 아프다고 하셨어요.",
    # 안경: '맞'이 폭행으로 읽히면 안 된다
    "새로 맞춘 안경이 잘 맞는다고 좋아하셨어요.",
    # 낙상 부정형
    "넘어지신 적은 없다고 하셨고 무릎이 조금 아프다고 했어요.",
    # 기간 없는 식사
    "오늘 점심은 반쯤 드셨어요.",
    "아침에는 식사를 안 드셨어요.",
    # 과거 사건
    "작년에 사흘째 식사를 거부하신 적이 있다고 해요.",
    "예전에 발이 퉁퉁 부은 적이 있었다고 하세요.",
    # 낙상 맥락이 없는 통증
    "엉덩이가 아프다고 하셨어요.",
    # 무관한 문장
    "오늘 날씨가 좋아서 산책을 하셨어요.",
]


@pytest.mark.parametrize("text", NEGATIVES)
def test_negative_no_signal(text: str) -> None:
    assert scan_text(text, RULES) == []


def test_post_fall_requires_fall_within_two_sentences() -> None:
    text = "넘어지셨어요. 괜찮으셨어요. 차를 드셨어요. 엉덩이가 아프다고 하세요."
    assert "post_fall" not in ids(text)


def test_post_fall_links_across_sentences() -> None:
    assert "post_fall" in ids("어제 넘어지셨어요. 그리고 차를 드셨어요. 엉덩이가 아프다고 하세요.")


def test_post_fall_pain_negated_is_not_signal() -> None:
    assert "post_fall" not in ids("어제 넘어지셨어요. 다행히 엉덩이가 아프지는 않대요.")


# ---- 신호 계약 ----------------------------------------------------------------------------------


def test_signal_quote_is_substring_of_transcript() -> None:
    text = "오늘 방문했어요. 발이 퉁퉁 부어 있었어요. 점심은 드셨어요."
    signals = scan_text(text, RULES)
    assert signals
    assert all(s.quote in text for s in signals)
    assert signals[0].quote == "발이 퉁퉁 부어 있었어요."


def test_one_signal_per_rule_per_sentence() -> None:
    signals = scan_text("가슴이 답답하고 호흡 곤란도 있어요.", RULES)
    assert [s.rule_id for s in signals] == ["chest_resp"]


def test_signals_are_provisional_and_circumstance_needs_expert() -> None:
    signals = scan_text("멍이 여러 군데 있고 발이 부었어요.", RULES)
    by_id = {s.rule_id: s for s in signals}
    assert all(s.provisional for s in signals)
    assert by_id["circumstance"].needs_expert is True
    assert by_id["body_change"].needs_expert is False


def test_all_default_rules_are_provisional_risk() -> None:
    assert all(r.provisional and r.severity is Flag.RISK for r in RULES.text_rules)
    assert all(r.provisional for r in RULES.numeric_rules)


def test_empty_and_whitespace_transcripts() -> None:
    assert scan_text("", RULES) == []
    assert scan_text("   \n ", RULES) == []


def test_split_sentences_keeps_substrings() -> None:
    text = "첫째 문장이에요. 둘째 문장이에요! 셋째요?"
    parts = split_sentences(text)
    assert len(parts) == 3
    assert all(p in text for p in parts)


# ---- 로더 ---------------------------------------------------------------------------------------


def test_default_rules_file_loads() -> None:
    assert DEFAULT_RULES_PATH.exists()
    assert RULES.version == 2
    assert {r.rule_id for r in RULES.text_rules} == {
        "chest_resp",
        "neuro",
        "intake_days",
        "body_change",
        "circumstance",
        "post_fall",
    }
    assert [r.rule_id for r in RULES.numeric_rules] == ["bp_high"]


MINIMAL = """
version: 2
text_rules:
  - id: a
    patterns: ['가슴']
"""


def test_minimal_rules_parse_with_defaults() -> None:
    rules = parse_rules(MINIMAL)
    rule = rules.text_rules[0]
    assert rule.severity is Flag.RISK
    assert rule.provisional is True
    assert rule.guards == frozenset({"negation"})


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[1, 2]", "매핑"),
        ("text_rules: [", "YAML"),
        ("text_rules:\n  - patterns: ['x']\n", "id"),
        ("text_rules:\n  - id: a\n", "패턴|patterns"),
        ("text_rules:\n  - id: a\n    patterns: []\n", "patterns"),
        ("text_rules:\n  - id: a\n    patterns: ['(']\n", "정규식"),
        ("text_rules:\n  - id: a\n    patterns: ['x']\n    guards: [oops]\n", "가드"),
        ("text_rules:\n  - id: a\n    patterns: ['x']\n    severity: ok\n", "ok"),
        ("text_rules:\n  - id: a\n    patterns: ['x']\n    severity: boom\n", "severity"),
        (
            "text_rules:\n  - id: a\n    patterns: ['x']\n  - id: a\n    patterns: ['y']\n",
            "중복",
        ),
        (
            "text_rules:\n  - id: a\n    patterns: ['x']\n    context_patterns: ['y']\n",
            "context_window",
        ),
        ("numeric_rules:\n  - id: n\n    field: nope\n    systolic_gte: 1\n", "field"),
        ("numeric_rules:\n  - id: n\n    field: pulse\n    systolic_gte: 1\n", "blood_pressure"),
        ("numeric_rules:\n  - id: n\n    field: blood_pressure\n", "systolic_gte"),
        ("advice_words: '('\n", "advice_words"),
    ],
)
def test_invalid_rules_raise(text: str, message: str) -> None:
    with pytest.raises(RuleError, match=message):
        parse_rules(text)


def test_load_rules_from_custom_path(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    assert load_rules(path).version == 2


def test_custom_rule_fires_and_guard_applies() -> None:
    rules = parse_rules(MINIMAL)
    assert scan_text("가슴이 답답해요.", rules)
    assert scan_text("가슴이 답답하지 않아요.", rules) == []


# ---- 수치 Rule Engine ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("160/90", True),
        ("159/99", False),
        ("159/100", True),
        ("120/80", False),
        ("170/105", True),
        ("100/60", False),
        ("160/100", True),
    ],
)
def test_bp_threshold_boundaries(value: str, expected: bool) -> None:
    assert bool(numeric_signals(FieldKey.BLOOD_PRESSURE, value, RULES)) is expected


@pytest.mark.parametrize("value", [None, "", "고혈압", "160", "160-100"])
def test_bp_threshold_ignores_unparseable(value: str | None) -> None:
    assert numeric_signals(FieldKey.BLOOD_PRESSURE, value, RULES) == []


def test_numeric_rules_ignore_other_fields() -> None:
    assert numeric_signals(FieldKey.PULSE, "200", RULES) == []


# ---- Consensus(FlaggingExtractor) -------------------------------------------------------------


class Fixed:
    """고정된 필드를 돌려주는 extractor."""

    def __init__(self, fields: list[PredictedField]) -> None:
        self._fields = fields

    def extract(self, transcript: str) -> list[PredictedField]:
        return self._fields


def stated(key: FieldKey, value: str, quote: str = "근거") -> PredictedField:
    return PredictedField(
        key=key, value=value, source=Source.STATED, evidence=(Evidence(text=quote),)
    )


def flags_of(fields: list[PredictedField]) -> dict[FieldKey, Flag]:
    return {f.key: f.flag for f in fields}


NOTE_TEXT = "방문했어요. 발이 퉁퉁 부어 있었어요."


def test_sentinel_raises_special_notes_to_risk() -> None:
    inner = Fixed([stated(FieldKey.SPECIAL_NOTES, "발 부종")])
    out = FlaggingExtractor(inner, RULES).extract(NOTE_TEXT)
    assert flags_of(out)[FieldKey.SPECIAL_NOTES] is Flag.RISK


def test_unrecorded_signal_raises_flag_without_inventing_value() -> None:
    out = FlaggingExtractor(Fixed([]), RULES).extract(NOTE_TEXT)
    notes = next(f for f in out if f.key is FieldKey.SPECIAL_NOTES)
    assert notes.flag is Flag.RISK
    assert notes.value is None
    assert notes.source is Source.MISSING


def test_without_rules_behavior_is_unchanged() -> None:
    inner = Fixed([stated(FieldKey.SPECIAL_NOTES, "발 부종")])
    out = FlaggingExtractor(inner).extract(NOTE_TEXT)
    assert flags_of(out)[FieldKey.SPECIAL_NOTES] is Flag.OK


def test_no_signal_leaves_special_notes_ok() -> None:
    inner = Fixed([stated(FieldKey.SPECIAL_NOTES, "무릎 통증")])
    out = FlaggingExtractor(inner, RULES).extract("무릎이 쑤신다고 하셨어요.")
    assert flags_of(out)[FieldKey.SPECIAL_NOTES] is Flag.OK


def test_signals_for_returns_quotes() -> None:
    extractor = FlaggingExtractor(Fixed([]), RULES)
    extractor.extract(NOTE_TEXT)
    signals = extractor.signals_for(NOTE_TEXT)
    assert [s.rule_id for s in signals] == ["body_change"]
    assert extractor.signals_for("다른 문장") == []


def test_signals_for_empty_without_rules() -> None:
    extractor = FlaggingExtractor(Fixed([]))
    extractor.extract(NOTE_TEXT)
    assert extractor.signals_for(NOTE_TEXT) == []


def test_high_bp_is_risk_with_rules_and_ok_without() -> None:
    field = stated(FieldKey.BLOOD_PRESSURE, "170/105")
    assert flag_field(field) is Flag.OK
    assert flag_field(field, None, RULES) is Flag.RISK


def test_normal_bp_stays_ok_with_rules() -> None:
    assert flag_field(stated(FieldKey.BLOOD_PRESSURE, "120/80"), None, RULES) is Flag.OK


def test_needs_confirm_outranks_rule_risk() -> None:
    ambiguous = PredictedField(key=FieldKey.BLOOD_PRESSURE, candidates=("170/105", "130/80"))
    assert flag_field(ambiguous, None, RULES) is Flag.NEEDS_CONFIRM


def test_implausible_bp_stays_needs_confirm_with_rules() -> None:
    assert flag_field(stated(FieldKey.BLOOD_PRESSURE, "300/100"), None, RULES) is Flag.NEEDS_CONFIRM


def test_rules_do_not_change_values() -> None:
    inner = Fixed([stated(FieldKey.BLOOD_PRESSURE, "170/105")])
    base = FlaggingExtractor(inner).extract(NOTE_TEXT)
    with_rules = FlaggingExtractor(inner, RULES).extract(NOTE_TEXT)
    assert [(f.key, f.value, f.source) for f in base] == [
        (f.key, f.value, f.source) for f in with_rules
    ]


def test_other_flags_unchanged_by_rules_on_clean_transcript() -> None:
    inner = Fixed([stated(FieldKey.MEDICATION_TAKEN, "미복용")])
    clean = "오늘은 약을 안 드셨어요."
    assert flags_of(FlaggingExtractor(inner, RULES).extract(clean)) == flags_of(
        FlaggingExtractor(inner).extract(clean)
    )


def test_ruleset_type_exported() -> None:
    assert isinstance(RULES, RuleSet)


# ---- dev 회귀 (test 분할은 읽지 않는다) -------------------------------------------------------


def test_dev_special_notes_risk_recall_and_false_alarms() -> None:
    """프로토타입 v2 재현: 위험 3건 포착, 오발동 0 (튜닝에 쓴 dev라 독립 증거 아님)."""
    cases = load_golden(GOLDEN, "dev")
    caught = missed = false_alarm = 0
    for case in cases:
        gold = case.field(FieldKey.SPECIAL_NOTES)
        is_risk = gold.value is not None and gold.flag is Flag.RISK
        fired = bool(scan_text(case.transcript, RULES))
        caught += is_risk and fired
        missed += is_risk and not fired
        false_alarm += (not is_risk) and fired
    assert (caught, missed, false_alarm) == (3, 0, 0)


def test_dev_bp_threshold_matches_gold_flags() -> None:
    cases = load_golden(GOLDEN, "dev")
    for case in cases:
        gold = case.field(FieldKey.BLOOD_PRESSURE)
        predicted = bool(numeric_signals(FieldKey.BLOOD_PRESSURE, gold.value, RULES))
        assert predicted is (gold.value is not None and gold.flag is Flag.RISK), case.id


# ---- CLI: test 분할 거부 ------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["test", "all"])
def test_cli_refuses_sentinel_on_non_dev(split: str) -> None:
    code = main(["--extractor", "llm-b0", "--sentinel", "--split", split, "--dry-run"])
    assert code == EXIT_REFUSED


def test_cli_refuses_sentinel_for_non_llm_extractor() -> None:
    assert main(["--extractor", "oracle", "--sentinel"]) == EXIT_REFUSED


# ---- 일반화(2026-10-27): R-set dev 원인 분석 기반 -----------------------------------------------
# 아래 문장은 R-set dev의 실패 유형(패턴 모양, 가드 범위)에서 뽑은 변형이다. 같은 유형이라
# 이 테스트와 dev 점수는 in-sample이며 독립 증거가 아니다. 독립 평가는 test R-set 한 번뿐이다.

GENERALIZED_POSITIVES = [
    # 어순: 증상어 사이에 다른 말이 낀다
    ("neuro", "어르신 말씀이 오늘은 자꾸 꼬이셔서 알아듣기 힘들었어요."),
    ("neuro", "말이 평소랑 달리 어눌하셨어요."),
    ("neuro", "말씀이 요즘 들어 많이 어눌해지셨어요."),
    # 기간이 거부 뒤에 오고, 동사가 다양하다
    ("intake_days", "식사를 또 안 하세요, 이게 사흘째예요."),
    ("intake_days", "밥을 계속 못 드세요, 나흘째 그러세요."),
    ("intake_days", "나흘째 밥상을 물리세요."),
    ("intake_days", "물을 입에 안 대세요, 사흘째예요."),
    # 수식어가 둘 이상
    ("body_change", "두 다리가 갑자기 퉁퉁 부어서 바지가 안 올라가요."),
    ("body_change", "발이 오늘 갑자기 많이 부었어요."),
    ("body_change", "발목이 퉁퉁 갑자기 부어 있었어요."),
    # 가드가 새어 나가면 안 된다: 부정·조건이 있어도 다른 증상은 살아 있다
    ("chest_resp", "가슴이 답답하다고 하셨고 약은 아직 안 드셨어요."),
    ("chest_resp", "숨이 차고 입맛은 없으셨어요."),
    ("chest_resp", "혹시 몰라 119 번호를 적어 드렸는데 가슴이 답답하다고 하셨어요."),
    ("chest_resp", "할머니가 가슴이 답답하다고 하셨어요."),
]


@pytest.mark.parametrize(("rule_id", "text"), GENERALIZED_POSITIVES)
def test_generalized_positive_signal(rule_id: str, text: str) -> None:
    assert rule_id in ids(text)


GENERALIZED_NEGATIVES = [
    # 부정이 나열 끝에 온다
    "가슴이 답답하거나 숨이 차거나 하는 건 없으셨어요.",
    "말이 어눌하거나 팔에 힘이 없거나 한 것은 아니었어요.",
    "발이 붓거나 다리가 저린 건 없었어요.",
    # 증상 바로 뒤의 '건 없다'
    "숨이 차는 건 없으셨어요.",
    # 조건절이 첫 증상 뒤에 길게 이어진다
    "혹시 숨이 차거나 가슴이 답답해지면 바로 119에 전화하시라고 했어요.",
    "팔에 힘이 빠지거나 말이 어눌해지면 응급실에 가시라고 안내했어요.",
    # 제3자 호칭 보강
    "이웃 분이 발이 퉁퉁 부었다고 하셨어요.",
    "사위가 숨이 차서 병원에 갔다고 하셨어요.",
    # 절이 다르면 어눌과 말씀을 잇지 않는다
    "말씀은 평소처럼 또렷하셨는데, 혼잣말처럼 노래는 어눌하게 흥얼거리셨어요.",
]


@pytest.mark.parametrize("text", GENERALIZED_NEGATIVES)
def test_generalized_negative_no_signal(text: str) -> None:
    assert scan_text(text, RULES) == []


# ---- 동결 (2026-10-27): test R-set 평가 전에 규칙·가드 코드·검증 코드를 고정한다 ---------------
# 바꾸려면 version을 올리고 새 태그를 만든 뒤 해시를 갱신한다. test R-set 채점 이후에는 금지.

FROZEN_SHA256 = {
    "rules/sentinel.yaml": "6e616747bf698e9ee7ec79bd34a85eefdc5b16e9f5a6bd9474cb32febfd010fd",
    "src/carenote/extraction/sentinel.py": "b0d787bcb30fab9ce0b086d89923591b5249aea5686c296698eea01e4a1ebf98",  # noqa: E501
    "src/carenote/extraction/sentinel_llm.py": "25f07f60eb96dd1669ff7acd31228e7f1316ae2308e3a699a8fe39853ec38640",  # noqa: E501
}


@pytest.mark.parametrize("relative", sorted(FROZEN_SHA256))
def test_sentinel_files_are_frozen(relative: str) -> None:
    import hashlib

    digest = hashlib.sha256((Path(__file__).parent.parent / relative).read_bytes()).hexdigest()
    assert digest == FROZEN_SHA256[relative], f"{relative}가 동결 후 바뀌었다"
    assert RULES.version == 2
