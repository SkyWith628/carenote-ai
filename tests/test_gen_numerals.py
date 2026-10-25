"""한글 수사 변환기 테스트."""

from __future__ import annotations

import pytest

from carenote.generation.numerals import (
    extract_hours,
    extract_numbers,
    find_ambiguous_numerals,
    hour_matches,
    parse_sino,
)


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        ("백삼십", 130),
        ("칠십오", 75),
        ("일삼공", 130),  # 숫자 하나씩 읽기
        ("팔공", 80),
        ("구공", 90),
        ("일사공", 140),
        ("육십오", 65),
        ("이십", 20),
        ("백", 100),
        ("십", 10),
        ("천이백", 1200),
    ],
)
def test_parse_sino(run: str, expected: int) -> None:
    """한자어 수사를 정수로 바꾼다."""
    assert parse_sino(run) == expected


@pytest.mark.parametrize("run", ["", "가나", "일", "삼쏙"])
def test_parse_sino_rejects_non_numerals(run: str) -> None:
    """수사가 아니거나 한 글자 숫자만 있으면 None."""
    assert parse_sino(run) is None


def test_extract_numbers_mixes_arabic_and_korean() -> None:
    """아라비아 숫자와 한글 수사를 함께 찾고, 어미가 붙은 수사도 떼어 읽는다."""
    found = extract_numbers("혈압은 일삼공에 팔공이우다 맥박은 육십오 그리고 128에 80")
    assert {130, 80, 65, 128} <= found


def test_extract_numbers_dialect_digit_reading() -> None:
    """'일사공에 구공'은 140과 90이다."""
    assert {140, 90} <= extract_numbers("혈압은 일사공에 구공이래요")


def test_extract_hours_handles_native_and_arabic() -> None:
    """고유어 시각과 아라비아 시각을 모두 읽는다."""
    text = "열 시에 갔고 열한 시 반 오후 3시 아홉시쯤 열두 시 한 시"
    assert extract_hours(text) == {10, 11, 3, 9, 12, 1}


@pytest.mark.parametrize(
    ("hour24", "found", "expected"),
    [(15, {3}, True), (15, {15}, True), (9, {9}, True), (0, {12}, True), (10, {11}, False)],
)
def test_hour_matches_accepts_12_and_24_hour_forms(
    hour24: int, found: set[int], expected: bool
) -> None:
    """24시간제 시각은 12시간제 표현으로도 일치한다."""
    assert hour_matches(hour24, found) is expected


def test_ambiguous_numerals_flag_v1_lessons() -> None:
    """'칠십이고', '팔십이었나'처럼 둘로 읽히는 표현을 잡고, 명확한 표현은 놔둔다."""
    assert find_ambiguous_numerals("맥박은 칠십이고 혈압은 팔십이었나") == ["칠십이", "팔십이"]
    assert find_ambiguous_numerals("맥박은 칠십 정도고 팔십오예요") == []
