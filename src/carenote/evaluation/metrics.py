"""평가 지표.

지표 정의 (blueprint 7장):
- 필드 F1: 값이 있는 정답 필드를 정확한 값으로 맞췄는가. 틀린 값은 FP와 FN 양쪽에 센다.
  단 자유 문장인 special_notes는 값 유무만 본다(둘 다 값이 있으면 일치). 내용의 옳고 그름은
  채점하지 않는다. 문장 유사도나 LLM 판정은 골든셋 확장 시 검토한다.
- 근거 없는 값 비율: 값을 낸 필드 중 evidence가 없거나 transcript에 없는 것. 목표 0%.
- 환각률: 정답에 값이 없는데(missing 또는 모호) 값을 낸 비율.
- 수치 오류율: 혈압·맥박 값을 냈는데 정답과 다른 비율.
- 위험 recall: 정답 flag가 risk인 필드를 risk로 잡았는가.
- 확인 필요 recall: 정답 flag가 needs_confirm인 필드를 needs_confirm 또는 risk로 잡았는가.
- 조용한 오답률(위험 필드): 값을 냈고 flag가 ok인데(사람 검수 없이 통과하는 값) 정답과 다른 비율.
  분모는 위험 필드에서 값을 내고 flag가 ok인 예측이다. 낮을수록 안전하다.
- 과다 확인률: **값이 맞은** 필드(정답 flag가 ok)를 needs_confirm으로 올린 비율(불필요한 사람의
  확인 부담). 분모는 정답 flag가 ok이고 예측이 정답과 일치한 필드(값 일치 또는 둘 다 비움)다.
  틀린 값을 확인 필요로 올린 것은 좋은 동작이라 세지 않는다. 안전 규칙을 늘릴 때 같이 오르는지 본다.
- 기권율·오답률: 값이 있는 정답 필드 중 값을 아예 내지 않은 비율 / 틀린 값을 낸 비율. 둘을
  나눠 보는 이유는 "이상한 값을 내지 않는 쪽이 F1에서 유리한" 문제를 가리기 위해서다.
  틀린 값은 F1에서 FP+FN이라 기권보다 두 배 깎이지만, 사람에게는 기권(확인 요청)이 더 안전하다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import RISK_FIELDS, ExpectedField, FieldKey, Flag, GoldenCase

NUMERIC_FIELDS: frozenset[FieldKey] = frozenset({FieldKey.BLOOD_PRESSURE, FieldKey.PULSE})
# 자유 문장이라 정확 일치가 불가능한 필드. 값 유무로만 채점한다.
PRESENCE_ONLY_FIELDS: frozenset[FieldKey] = frozenset({FieldKey.SPECIAL_NOTES})


def _norm(value: str | None) -> str | None:
    return None if value is None else "".join(value.split())


@dataclass(frozen=True)
class Ratio:
    """분자/분母. 분모가 0이면 값은 None(해당 사례 없음)이다."""

    num: int = 0
    den: int = 0

    @property
    def value(self) -> float | None:
        """비율. 분모가 0이면 None."""
        return None if self.den == 0 else self.num / self.den

    def __add__(self, other: Ratio) -> Ratio:
        return Ratio(self.num + other.num, self.den + other.den)


@dataclass(frozen=True)
class Counts:
    """케이스 단위 집계. 더하면 전체 집계가 된다."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    ungrounded: Ratio = Ratio()
    hallucinated: Ratio = Ratio()
    numeric_error: Ratio = Ratio()
    risk_recall: Ratio = Ratio()
    confirm_recall: Ratio = Ratio()
    abstained: Ratio = Ratio()
    wrong_value: Ratio = Ratio()
    silent_wrong: Ratio = Ratio()
    over_confirm: Ratio = Ratio()

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.ungrounded + other.ungrounded,
            self.hallucinated + other.hallucinated,
            self.numeric_error + other.numeric_error,
            self.risk_recall + other.risk_recall,
            self.confirm_recall + other.confirm_recall,
            self.abstained + other.abstained,
            self.wrong_value + other.wrong_value,
            self.silent_wrong + other.silent_wrong,
            self.over_confirm + other.over_confirm,
        )


def _is_grounded(pred: PredictedField, transcript: str) -> bool:
    return bool(pred.evidence) and all(ev.text in transcript for ev in pred.evidence)


def _score_field(gold: ExpectedField, pred: PredictedField | None, transcript: str) -> Counts:
    pred_value = _norm(pred.value) if pred else None
    gold_value = _norm(gold.value)
    pred_flag = pred.flag if pred else Flag.OK

    if gold.key in PRESENCE_ONLY_FIELDS:
        matched = pred_value is not None
    else:
        matched = pred_value == gold_value
    tp = int(gold_value is not None and matched)
    fp = int(pred_value is not None and not tp)
    fn = int(gold_value is not None and not tp)

    ungrounded = Ratio()
    if pred is not None and pred_value is not None:
        ungrounded = Ratio(int(not _is_grounded(pred, transcript)), 1)

    hallucinated = Ratio()
    if gold_value is None:
        hallucinated = Ratio(int(pred_value is not None), 1)

    numeric_error = Ratio()
    if gold.key in NUMERIC_FIELDS and gold_value is not None and pred_value is not None:
        numeric_error = Ratio(int(pred_value != gold_value), 1)

    risk_recall = Ratio()
    if gold.flag is Flag.RISK:
        risk_recall = Ratio(int(pred_flag is Flag.RISK), 1)

    confirm_recall = Ratio()
    if gold.flag is Flag.NEEDS_CONFIRM:
        confirm_recall = Ratio(int(pred_flag in (Flag.NEEDS_CONFIRM, Flag.RISK)), 1)

    abstained = Ratio()
    wrong_value = Ratio()
    if gold_value is not None and gold.key not in PRESENCE_ONLY_FIELDS:
        abstained = Ratio(int(pred_value is None), 1)
        wrong_value = Ratio(int(pred_value is not None and not tp), 1)

    silent_wrong = Ratio()
    if gold.key in RISK_FIELDS and pred_value is not None and pred_flag is Flag.OK:
        silent_wrong = Ratio(int(not tp), 1)

    over_confirm = Ratio()
    value_matches = bool(tp) or (gold_value is None and pred_value is None)
    if gold.flag is Flag.OK and value_matches:
        over_confirm = Ratio(int(pred_flag is Flag.NEEDS_CONFIRM), 1)

    return Counts(
        tp,
        fp,
        fn,
        ungrounded,
        hallucinated,
        numeric_error,
        risk_recall,
        confirm_recall,
        abstained,
        wrong_value,
        silent_wrong,
        over_confirm,
    )


# all: 전체 필드 · recoverable: 텍스트로 복원 불가능한 필드 제외 ·
# strict: 복원 불가 필드와 정책상 논쟁적인(gray) 필드 제외
Subset = Literal["all", "recoverable", "strict"]


# 필드 하나의 채점 결과: tp 맞힘 · fn 값을 못 냄 · fpfn 틀린 값을 냄 ·
# fp 정답이 없는데 값을 냄 · tn 맞게 비움
FIELD_OUTCOMES: tuple[str, ...] = ("tp", "fn", "fpfn", "fp", "tn")
CORRECT_OUTCOMES: frozenset[str] = frozenset({"tp", "tn"})


def field_outcomes(case: GoldenCase, predictions: list[PredictedField]) -> dict[str, str]:
    """케이스의 필드별 채점 결과. `_score_field`의 TP/FP/FN 정의와 같은 기준이다."""
    by_key = {p.key: p for p in predictions}
    result: dict[str, str] = {}
    for gold in case.fields:
        pred = by_key.get(gold.key)
        pred_value = _norm(pred.value) if pred else None
        gold_value = _norm(gold.value)
        if gold.key in PRESENCE_ONLY_FIELDS:
            matched = pred_value is not None
        else:
            matched = pred_value == gold_value
        if gold_value is not None:
            outcome = "tp" if matched else ("fn" if pred_value is None else "fpfn")
        else:
            outcome = "tn" if pred_value is None else "fp"
        result[gold.key.value] = outcome
    return result


def score_case(
    case: GoldenCase, predictions: list[PredictedField], subset: Subset = "all"
) -> Counts:
    """한 케이스를 채점한다. 예측이 빠진 키는 missing으로 본다."""
    by_key = {p.key: p for p in predictions}
    total = Counts()
    for gold in case.fields:
        if subset != "all" and not gold.recoverable:
            continue
        if subset == "strict" and gold.key in case.gray_fields:
            continue
        total = total + _score_field(gold, by_key.get(gold.key), case.transcript)
    return total


@dataclass(frozen=True)
class Metrics:
    """집계로부터 계산한 최종 지표."""

    precision: float | None
    recall: float | None
    f1: float | None
    ungrounded: Ratio
    hallucinated: Ratio
    numeric_error: Ratio
    risk_recall: Ratio
    confirm_recall: Ratio
    abstained: Ratio
    wrong_value: Ratio
    silent_wrong: Ratio
    over_confirm: Ratio

    @staticmethod
    def from_counts(c: Counts) -> Metrics:
        """집계를 지표로 바꾼다."""
        precision = None if c.tp + c.fp == 0 else c.tp / (c.tp + c.fp)
        recall = None if c.tp + c.fn == 0 else c.tp / (c.tp + c.fn)
        if precision is None or recall is None or precision + recall == 0:
            f1 = None if precision is None and recall is None else 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return Metrics(
            precision,
            recall,
            f1,
            c.ungrounded,
            c.hallucinated,
            c.numeric_error,
            c.risk_recall,
            c.confirm_recall,
            c.abstained,
            c.wrong_value,
            c.silent_wrong,
            c.over_confirm,
        )
