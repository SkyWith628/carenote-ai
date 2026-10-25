"""한글 수사 해석. 생성된 transcript에 정답의 숫자가 들어 있는지 확인하는 용도다.

정밀한 번역기가 아니라 점검용이다. 거짓 양성(일상어의 '이'를 숫자로 읽음)은 허용하고
거짓 음성(실제 수사를 놓침)을 줄이는 쪽으로 설계했다.
"""

from __future__ import annotations

import re

_DIGITS = {
    "영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4,
    "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9,
}  # fmt: skip
_UNITS = {"십": 10, "백": 100, "천": 1000}
_SINO_RUN = re.compile(r"[영공일이삼사오육칠팔구십백천]+")
_ARABIC = re.compile(r"\d+")

_NATIVE_HOURS = {
    "열두": 12, "열한": 11, "열": 10, "아홉": 9, "여덟": 8, "일곱": 7,
    "여섯": 6, "다섯": 5, "네": 4, "세": 3, "두": 2, "한": 1,
}  # fmt: skip
_NATIVE_HOUR_RE = re.compile(r"(열두|열한|열|아홉|여덟|일곱|여섯|다섯|네|세|두|한)\s?시")
_ARABIC_HOUR_RE = re.compile(r"(\d{1,2})\s?시")

# "칠십이"처럼 70+이(었나)와 72가 모두 가능한 읽기. 어미/조사가 붙은 '십이'를 잡는다.
_AMBIGUOUS_RE = re.compile(r"[일이삼사오육칠팔구]?십이(?=[었고에예래라야요죠나다]|\s|$)")


def parse_sino(run: str) -> int | None:
    """한자어 수사 문자열을 정수로 바꾼다. 해석할 수 없으면 None.

    '백삼십' → 130, '칠십오' → 75, '일삼공' → 130(숫자 하나씩 읽기), '팔공' → 80.
    """
    if not run or any(ch not in _DIGITS and ch not in _UNITS for ch in run):
        return None
    if not any(ch in _UNITS for ch in run):
        if len(run) < 2:
            return None
        return int("".join(str(_DIGITS[ch]) for ch in run))
    total, current = 0, 0
    for ch in run:
        if ch in _DIGITS:
            current = _DIGITS[ch]
        else:
            total += (current or 1) * _UNITS[ch]
            current = 0
    return total + current


def extract_numbers(text: str) -> set[int]:
    """텍스트의 아라비아 숫자와 한자어 수사(2자 이상 또는 단위 포함)를 정수 집합으로."""
    numbers = {int(m) for m in _ARABIC.findall(text)}
    for match in _SINO_RUN.finditer(text):
        run = match.group()
        # 어미가 붙은 수사('팔공이우다'의 '이')를 떼어 읽도록 모든 연속 부분 문자열을 본다
        for start in range(len(run)):
            for end in range(start + 1, min(len(run), start + 8) + 1):
                part = run[start:end]
                if len(part) < 2 and not any(ch in _UNITS for ch in part):
                    continue
                value = parse_sino(part)
                if value is not None:
                    numbers.add(value)
    return numbers


def extract_hours(text: str) -> set[int]:
    """'열 시', '3시', '오후 세 시' 같은 시각 표현의 시(hour) 숫자 집합."""
    hours = {int(m) for m in _ARABIC_HOUR_RE.findall(text)}
    hours |= {_NATIVE_HOURS[m] for m in _NATIVE_HOUR_RE.findall(text)}
    return hours


def hour_matches(hour24: int, found: set[int]) -> bool:
    """24시간제 시각이 12시간제/24시간제 표현 어느 쪽으로든 나타났는가."""
    twelve = hour24 % 12 or 12
    return hour24 in found or twelve in found


def find_ambiguous_numerals(text: str) -> list[str]:
    """70+이/72처럼 둘로 읽힐 수 있는 한글 수사 표현을 찾는다."""
    return [m.group() for m in _AMBIGUOUS_RE.finditer(text)]
