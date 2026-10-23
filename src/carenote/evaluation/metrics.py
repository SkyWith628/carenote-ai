"""평가 지표.

지표 정의 (blueprint 7장):
- 필드 F1: 값이 있는 정답 필드를 정확한 값으로 맞췄는가. 틀린 값은 FP와 FN 양쪽에 센다.
- 근거 없는 값 비율: 값을 낸 필드 중 evidence가 없거나 transcript에 없는 것. 목표 0%.
- 환각률: 정답에 값이 없는데(missing 또는 모호) 값을 낸 비율.
- 수치 오류율: 혈압·맥박 값을 냈는데 정답과 다른 비율.
- 위험 recall: 정답 flag가 risk인 필드를 risk로 잡았는가.
- 확인 필요 recall: 정답 flag가 needs_confirm인 필드를 needs_confirm 또는 risk로 잡았는가.
"""

from __future__ import annotations

from dataclasses import dataclass

from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import ExpectedField, FieldKey, Flag, GoldenCase

NUMERIC_FIELDS: frozenset[FieldKey] = frozenset({FieldKey.BLOOD_PRESSURE, FieldKey.PULSE})


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
        )


def _is_grounded(pred: PredictedField, transcript: str) -> bool:
    return bool(pred.evidence) and all(ev.text in transcript for ev in pred.evidence)


def _score_field(gold: ExpectedField, pred: PredictedField | None, transcript: str) -> Counts:
    pred_value = _norm(pred.value) if pred else None
    gold_value = _norm(gold.value)
    pred_flag = pred.flag if pred else Flag.OK

    tp = int(gold_value is not None and pred_value == gold_value)
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

    return Counts(tp, fp, fn, ungrounded, hallucinated, numeric_error, risk_recall, confirm_recall)


def score_case(case: GoldenCase, predictions: list[PredictedField]) -> Counts:
    """한 케이스를 채점한다. 예측이 빠진 키는 missing으로 본다."""
    by_key = {p.key: p for p in predictions}
    total = Counts()
    for gold in case.fields:
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
        )
