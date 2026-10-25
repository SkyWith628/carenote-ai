"""케이스 단위 부트스트랩 신뢰구간과 구성 간 짝지은 차이.

필드가 아니라 **케이스**를 재표집한다. 한 케이스의 필드들은 같은 transcript를 공유해서
독립이 아니기 때문이다. 난수는 시드를 고정해 같은 입력이면 같은 구간이 나온다.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from carenote.evaluation.metrics import Counts, Metrics

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 0
ALPHA = 0.05


@dataclass(frozen=True)
class Interval:
    """점추정과 95% 구간 (백분위 부트스트랩)."""

    point: float | None
    low: float | None
    high: float | None


def f1_of(counts: Counts) -> float | None:
    """집계의 F1. 정답도 예측도 없으면 None."""
    return Metrics.from_counts(counts).f1


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def _interval(point: float | None, samples: list[float]) -> Interval:
    if point is None or not samples:
        return Interval(point, None, None)
    samples.sort()
    return Interval(point, _percentile(samples, ALPHA / 2), _percentile(samples, 1 - ALPHA / 2))


def _sum(items: Sequence[Counts]) -> Counts:
    total = Counts()
    for item in items:
        total = total + item
    return total


def bootstrap_f1(
    per_case: Sequence[Counts], resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED
) -> Interval:
    """F1의 케이스 단위 부트스트랩 구간."""
    point = f1_of(_sum(per_case))
    if len(per_case) < 2:
        return Interval(point, None, None)
    rng = random.Random(seed)
    n = len(per_case)
    samples: list[float] = []
    for _ in range(resamples):
        value = f1_of(_sum([per_case[rng.randrange(n)] for _ in range(n)]))
        if value is not None:
            samples.append(value)
    return _interval(point, samples)


def paired_f1_difference(
    a: Mapping[str, Counts],
    b: Mapping[str, Counts],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """같은 케이스에서 F1(b) - F1(a)의 부트스트랩 구간. 케이스 집합이 다르면 ValueError.

    구간이 0을 포함하지 않으면 그 차이는 표집 변동만으로 설명하기 어렵다. 포함하면 이
    골든셋으로는 두 구성을 구분할 수 없다.
    """
    if set(a) != set(b):
        only = sorted(set(a) ^ set(b))
        raise ValueError(f"두 보고서의 케이스 집합이 다르다: {only[:5]}")
    ids = sorted(a)
    fa, fb = f1_of(_sum([a[i] for i in ids])), f1_of(_sum([b[i] for i in ids]))
    point = None if fa is None or fb is None else fb - fa
    if len(ids) < 2:
        return Interval(point, None, None)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        picked = [ids[rng.randrange(len(ids))] for _ in ids]
        sa, sb = f1_of(_sum([a[i] for i in picked])), f1_of(_sum([b[i] for i in picked]))
        if sa is not None and sb is not None:
            samples.append(sb - sa)
    return _interval(point, samples)
