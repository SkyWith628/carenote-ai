"""STT 정확도 지표: 문자 오류율(CER)과 수치 보존율.

한국어는 띄어쓰기가 흔들려 단어 단위 WER이 과하게 나빠지므로 CER을 기본으로 쓴다. 비교 전에
유니코드 정규화(NFC)를 하고 공백·문장부호를 없앤다. CER은 "혈압 130에 80"에서 숫자 하나가
틀려도 글자 수 대비 작아 보이므로, 임상적으로 중요한 숫자는 `numbers_preserved`로 따로 본다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from carenote.generation.numerals import extract_numbers

_DROP = re.compile(r"[\s.,!?;:\"'()\[\]{}~·…\-–—]+")


def normalize(text: str) -> str:
    """NFC 정규화 후 공백·문장부호를 제거한다."""
    return _DROP.sub("", unicodedata.normalize("NFC", text))


def edit_distance(a: str, b: str) -> int:
    """레벤슈타인 거리 (삽입·삭제·치환 비용 1)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class CerResult:
    """한 쌍의 CER. 합산(micro)을 위해 거리와 기준 길이를 함께 둔다."""

    distance: int
    ref_length: int

    @property
    def cer(self) -> float:
        """오류율 (기준 길이 0이면 0 또는 1)."""
        if self.ref_length == 0:
            return 0.0 if self.distance == 0 else 1.0
        return self.distance / self.ref_length


def cer(reference: str, hypothesis: str) -> CerResult:
    """정규화한 기준문과 인식문의 문자 오류."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    return CerResult(edit_distance(ref, hyp), len(ref))


def numbers_preserved(reference: str, hypothesis: str) -> tuple[int, int]:
    """기준문의 수치 중 인식문에서도 읽히는 개수와 기준 수치 개수.

    숫자·한자어 수사를 정수로 읽는 `extract_numbers`를 양쪽에 쓴다. 이 읽기는 어미가 붙은 수사를
    여러 방식으로 읽어 인식문 쪽 집합이 넉넉하므로 보존율은 낙관적이다(상한).
    """
    expected = extract_numbers(reference)
    if not expected:
        return 0, 0
    found = extract_numbers(hypothesis)
    return len(expected & found), len(expected)
