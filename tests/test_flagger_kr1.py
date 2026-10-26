"""KR-1 Flagger 규칙(R1 의문형 표지, R2 만들어진 숫자)과 러너 지표(조용한 오답, 과다 확인)."""

from __future__ import annotations

from pathlib import Path

import pytest

from carenote.evaluation.__main__ import format_report
from carenote.evaluation.extractor import NullExtractor, OracleExtractor, PredictedField
from carenote.evaluation.loader import load_golden
from carenote.evaluation.metrics import score_case
from carenote.evaluation.runner import evaluate
from carenote.evaluation.schema import Evidence, FieldKey, Flag, GoldenCase, Source
from carenote.extraction.flagger import (
    REASON_FABRICATED,
    REASON_HEDGE,
    FlaggingExtractor,
    flag_field,
    flag_reasons,
    hedged_fields,
)

GOLDEN = Path(__file__).parent.parent / "data" / "golden"
BP, PULSE = FieldKey.BLOOD_PRESSURE, FieldKey.PULSE

GS003 = (
    "두 시에 방문했고 혈압은 백삼십 팔십이었나, 아니 백사십 팔십이었나 그래요. "
    "맥박은 70이고 점심은 다 드셨어요."
)
GS009 = (
    "열두 시에 갔고 혈압 150에 90인가 105에 90인가 헷갈리네요. "
    "맥박은 75고요. 점심은 반쯤 드셨고 약도 드셨어요."
)
GS010 = (
    "한 시에 도착. 맥박이 칠십이었나 팔십이었나, 아무튼 괜찮았어요. "
    "혈압은 120에 80이에요. 점심 다 드셨고요."
)


def _f(key: FieldKey, value: str | None, **kw: object) -> PredictedField:
    source = Source.MISSING if value is None else Source.STATED
    return PredictedField(key=key, value=value, source=source, **kw)  # type: ignore[arg-type]


# -- R1: 번복·불확실 의문형 표지 --------------------------------------------------------


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        (GS003, {BP}),
        (GS009, {BP}),
        (GS010, {PULSE}),  # 혈압은 서술형("이에요")이라 걸리지 않는다
        ("혈압은 130에 80이었나 모르겠어요. 맥박은 70이에요.", {BP}),
        ("맥박은 70이에요. 혈압은 130에 80이었나 모르겠어요.", {BP}),  # 문장이 다르면 앞 키워드
        ("오늘 수치는 120에 80인가 그랬어요.", {BP, PULSE}),  # 키워드가 없으면 둘 다
        ("맥박은 72였나 봐요.", {PULSE}),
    ],
)
def test_r1_positive_hedged_numbers(transcript: str, expected: set[FieldKey]) -> None:
    assert hedged_fields(transcript) == frozenset(expected)


@pytest.mark.parametrize(
    "transcript",
    [
        # 해소된 정정(서술형 종결)은 건드리지 않는다
        "혈압은 130에 80, 아니 죄송해요, 140에 90이었어요.",
        "혈압은 130에 80이었는데 아니에요, 140에 90이었어요.",
        "맥박은 아니 팔십이었어요.",
        # 평범한 수치
        "혈압은 130에 80이에요. 맥박은 72예요.",
        "혈압은 백삼십에 팔십이고요 맥박은 칠십이에요.",
        # 의문형이지만 수치가 아님
        "약을 드셨는지 모르겠어요.",
        "무슨 일이었나 싶었어요.",  # '일이'를 12로 읽어도 혈압·맥박 범위가 아니다
        "아이인가 했는데 손주였어요.",
        "혈압약인가 보더라고요.",
        "어르신 성함이 이순자인가 그랬어요.",
        "몇 시인가 물으셨어요.",
        "열 시인가 열한 시인가 헷갈렸어요.",  # 방문 시각의 불확실은 R1 대상이 아니다
        "식사는 반쯤인가 드신 것 같아요.",
        "혈압 재는 걸 잊었는지 모르겠어요.",
        "5인가 6인가 모르겠어요.",  # 혈압·맥박일 수 없는 작은 수
        "",
    ],
)
def test_r1_negative_controls(transcript: str) -> None:
    assert hedged_fields(transcript) == frozenset()


def test_r1_flags_a_single_confirmed_value_and_keeps_the_value() -> None:
    field = _f(BP, "140/82", flag=Flag.OK)
    assert flag_reasons(field, GS003) == [REASON_HEDGE]
    assert flag_field(field, GS003) is Flag.NEEDS_CONFIRM
    assert flag_field(field) is Flag.OK  # transcript 없이는 예전 동작 그대로
    # 값 자체는 flagger가 바꾸지 않는다
    kept = FlaggingExtractor(_Fixed([field])).extract(GS003)
    assert next(f for f in kept if f.key is BP).value == "140/82"


def test_r1_applies_only_to_the_hedged_field() -> None:
    pulse = _f(PULSE, "70")
    assert flag_reasons(pulse, GS003) == []  # 혈압만 의문형이다
    assert flag_field(pulse, GS003) is Flag.OK
    assert flag_reasons(_f(PULSE, "70"), GS010) == [REASON_HEDGE]
    assert flag_reasons(_f(BP, "120/80"), GS010) == []


# -- R2: 만들어진 숫자 ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value", "transcript", "fires"),
    [
        (BP, "140/90", "혈압은 일사공에 구공으로 나왔어요.", False),
        (BP, "120/90", "혈압은 일사공에 구공으로 나왔어요.", True),  # 일사공을 120으로 읽은 오류
        (BP, "130/80", "혈압은 130에 80이에요.", False),
        (BP, "130/85", "혈압은 130에 80이에요.", True),  # 85는 어디에도 없다
        (PULSE, "72", "맥박은 칠십이고 점심은 반 정도예요.", False),  # 70+이고/72 둘 다 읽힘
        (PULSE, "72", "맥박은 60이에요.", True),
        (PULSE, "70", "맥박은 70이에요.", False),
    ],
)
def test_r2_number_must_be_readable_from_the_transcript(
    key: FieldKey, value: str, transcript: str, fires: bool
) -> None:
    reasons = flag_reasons(_f(key, value), transcript)
    assert (REASON_FABRICATED in reasons) is fires
    assert (flag_field(_f(key, value), transcript) is Flag.NEEDS_CONFIRM) is fires


def test_r2_does_not_touch_non_numeric_or_malformed_values() -> None:
    assert flag_reasons(_f(FieldKey.MEAL_INTAKE, "절반"), "점심 반 드셨어요") == []
    assert flag_reasons(_f(BP, "130 80"), "혈압은 130에 80") == []  # 형식 불량은 기존 규칙 몫
    assert flag_field(_f(BP, "130 80"), "혈압은 130에 80") is Flag.NEEDS_CONFIRM


# -- 규칙은 flag를 올리기만 한다 -----------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        _f(BP, None),
        _f(BP, None, candidates=("130/80", "140/80")),
        _f(FieldKey.MEDICATION_TAKEN, "미복용"),
        _f(FieldKey.FALL_EVENT, "있음"),
    ],
)
def test_rules_never_lower_a_flag(field: PredictedField) -> None:
    for transcript in (None, GS003, "혈압은 130에 80이에요."):
        assert flag_field(field, transcript) is flag_field(field)


def test_candidates_pass_through_r1_untouched() -> None:
    cand = _f(BP, None, candidates=("130/80", "140/80"))
    assert flag_reasons(cand, GS003) == []
    assert flag_field(cand, GS003) is Flag.NEEDS_CONFIRM


class _Fixed:
    def __init__(self, fields: list[PredictedField]) -> None:
        self._fields = fields

    def extract(self, transcript: str) -> list[PredictedField]:
        return self._fields


def test_flagging_extractor_passes_the_transcript_to_the_rules() -> None:
    wrong = [_f(BP, "140/82", flag=Flag.OK)]
    hedged = {f.key: f for f in FlaggingExtractor(_Fixed(wrong)).extract(GS003)}
    plain = {f.key: f for f in FlaggingExtractor(_Fixed(wrong)).extract("혈압은 140에 82예요.")}
    assert hedged[BP].flag is Flag.NEEDS_CONFIRM
    assert plain[BP].flag is Flag.OK


# -- 러너 지표 ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cases() -> list[GoldenCase]:
    return load_golden(GOLDEN)


def test_oracle_and_null_have_no_silent_wrong_or_over_confirm(cases: list[GoldenCase]) -> None:
    for extractor in (OracleExtractor(cases), NullExtractor()):
        m = evaluate(extractor, cases).overall
        assert m.silent_wrong.num == 0 and m.over_confirm.num == 0
    oracle = evaluate(OracleExtractor(cases), cases).overall
    assert oracle.silent_wrong.den > 0 and oracle.over_confirm.den > 0
    assert evaluate(NullExtractor(), cases).overall.silent_wrong.den == 0  # 값을 안 내면 분모 0


class _SilentWrong:
    """모든 케이스의 혈압을 틀린 값으로 내고 flag는 ok로 둔다."""

    def extract(self, transcript: str) -> list[PredictedField]:
        return [
            PredictedField(
                key=BP,
                value="1/1",
                source=Source.STATED,
                evidence=(Evidence(text=transcript[:2]),),
                flag=Flag.OK,
            )
        ]


class _OverConfirm:
    """정답 flag가 ok인 혈압 값을 정확히 내지만 needs_confirm으로 올린다."""

    def __init__(self, cases: list[GoldenCase]) -> None:
        self._by_text = {c.transcript: c for c in cases}

    def extract(self, transcript: str) -> list[PredictedField]:
        gold = self._by_text[transcript].field(BP)
        return [
            PredictedField(
                key=BP, value=gold.value, source=Source.STATED, flag=Flag.NEEDS_CONFIRM,
                evidence=gold.evidence,
            )
        ]  # fmt: skip


def test_silent_wrong_counts_wrong_values_that_pass_with_flag_ok(cases: list[GoldenCase]) -> None:
    m = evaluate(_SilentWrong(), cases).overall
    assert m.silent_wrong.den == len(cases) and m.silent_wrong.num == len(cases)
    # 같은 오답이라도 flag가 needs_confirm이면 조용하지 않다
    flagged = evaluate(
        FlaggingExtractor(_SilentWrong()), cases
    ).overall  # 1/1은 범위 밖 → 확인 필요
    assert flagged.silent_wrong.num == 0


def test_over_confirm_counts_gold_ok_fields_raised_to_needs_confirm(
    cases: list[GoldenCase],
) -> None:
    m = evaluate(_OverConfirm(cases), cases).overall
    gold_ok_bp = sum(c.field(BP).flag is Flag.OK and c.field(BP).value is not None for c in cases)
    assert m.over_confirm.num == gold_ok_bp > 0
    assert m.over_confirm.den > m.over_confirm.num  # 분모는 정답 flag가 ok이고 값이 맞은 필드


class _WrongButFlagged:
    """틀린 값을 내지만 needs_confirm으로 올린다(좋은 동작: 과다 확인이 아니다)."""

    def extract(self, transcript: str) -> list[PredictedField]:
        return [
            PredictedField(
                key=BP,
                value="1/1",
                source=Source.STATED,
                evidence=(Evidence(text=transcript[:2]),),
                flag=Flag.NEEDS_CONFIRM,
            )
        ]


def test_flagging_a_wrong_value_is_not_counted_as_over_confirm(cases: list[GoldenCase]) -> None:
    m = evaluate(_WrongButFlagged(), cases).overall
    assert m.over_confirm.num == 0 and m.silent_wrong.num == 0


def test_new_metrics_are_in_case_counts_report_text_and_json(cases: list[GoldenCase]) -> None:
    counts = score_case(cases[0], _SilentWrong().extract(cases[0].transcript))
    assert counts.silent_wrong.num == 1
    report = evaluate(OracleExtractor(cases), cases)
    assert "조용한 오답" in format_report(report) and "과다 확인" in format_report(report)
    data = report.to_dict()
    overall = data["overall"]
    assert isinstance(overall, dict) and "silent_wrong" in overall and "over_confirm" in overall
