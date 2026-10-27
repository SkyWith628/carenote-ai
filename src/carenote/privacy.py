"""transcript의 개인식별정보 패턴 마스킹 (blueprint 6장).

음성은 STT 전에는 마스킹할 수 없으므로 STT가 낸 텍스트를 LLM·저장소로 보내기 전에 거친다.
정규식으로 잡을 수 있는 숫자 형식만 다룬다. 한계:
- 한글 수사로 읽은 번호("공일공 일이삼사 …")는 잡지 못한다.
- 이름은 수급자 사전이 있어야 치환할 수 있어 아직 다루지 않는다(blueprint: 수급자A 방식).
따라서 이 단계는 보조 장치이며 "개인정보가 없다"는 근거로 쓰지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# (이름, 패턴, 치환어). 긴 패턴을 먼저 둔다.
_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("rrn", re.compile(r"(?<!\d)\d{6}-?[1-4]\d{6}(?!\d)"), "[주민번호]"),
    ("phone", re.compile(r"(?<!\d)01[016789][-\s]?\d{3,4}[-\s]?\d{4}(?!\d)"), "[전화번호]"),
    ("ltc_no", re.compile(r"(?<![A-Za-z0-9])L\d{10}(?!\d)"), "[인정번호]"),
)


@dataclass(frozen=True)
class MaskResult:
    """마스킹 결과. counts는 종류별 치환 횟수(값은 담지 않는다)."""

    text: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        """치환한 총 횟수."""
        return sum(self.counts.values())


def mask_pii(text: str) -> MaskResult:
    """주민등록번호·휴대전화·장기요양인정번호 형식을 치환어로 바꾼다."""
    counts: dict[str, int] = {}
    for name, pattern, replacement in _PATTERNS:
        text, n = pattern.subn(replacement, text)
        if n:
            counts[name] = n
    return MaskResult(text, counts)
