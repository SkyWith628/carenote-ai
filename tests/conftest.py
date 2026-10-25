"""생성 단계 테스트 공용 픽스처: 실제 스펙과, 점검을 통과하도록 손으로 쓴 후보."""

from __future__ import annotations

from pathlib import Path

import pytest

from carenote.evaluation.schema import FieldKey
from carenote.generation.schema import FieldEvidence, GeneratedCase
from carenote.generation.spec import Spec, load_specs

SPEC_DIR = Path(__file__).parent.parent / "data" / "golden" / "specs"


@pytest.fixture(scope="session")
def all_specs() -> list[Spec]:
    """data/golden/specs의 스펙 40건."""
    return load_specs(SPEC_DIR)


@pytest.fixture(scope="session")
def spec_by_id(all_specs: list[Spec]) -> dict[str, Spec]:
    """spec_id로 찾는 사전."""
    return {s.spec_id: s for s in all_specs}


def _ev(**quotes: list[str]) -> list[FieldEvidence]:
    return [FieldEvidence(key=FieldKey(k), quotes=v) for k, v in quotes.items()]


GOOD_D1 = GeneratedCase(
    transcript=(
        "아홉 시 반에 어르신 댁에 도착했어요. 혈압약이 5밀리짜리 한 알이고 당뇨약은 500 한 알"
        "이라고 하시는데 하루에 두 번 챙겨 드시는 거래요. 오늘 혈압은 128에 80 나왔고요. 맥박은"
        " 74예요. 점심은 남기지 않고 다 드셨어요. 아침 약도 드셨다고 하셔서 약통을 같이 봤어요."
        " 넘어지신 일은 없었어요. 오늘은 별일 없이 방문을 마쳤습니다."
    ),
    reference_transcript=None,
    evidence=_ev(
        visit_start=["아홉 시 반에 어르신 댁에 도착했어요"],
        blood_pressure=["오늘 혈압은 128에 80 나왔고요"],
        pulse=["맥박은 74예요"],
        meal_intake=["점심은 남기지 않고 다 드셨어요"],
        medication_taken=["아침 약도 드셨다고 하셔서"],
        fall_event=["넘어지신 일은 없었어요"],
    ),
    notes="",
)

GOOD_H3 = GeneratedCase(
    transcript=(
        "오전 아홉 시에 들렀어요. 어르신 약은 드셨겠죠, 평소에 워낙 잘 챙겨 드시니까요. 혈압은"
        " 아마 130쯤 나왔을 거예요, 늘 그 정도라서요. 맥박은 제가 직접 쟀는데 76이었어요. 점심은"
        " 반 정도 드시는 걸 제가 봤고요. 넘어지신 일은 없다고 확인했어요. 오늘은 그냥 평소 같았어요"
        " 다른 건 딱히 모르겠네요."
    ),
    reference_transcript=None,
    evidence=_ev(
        visit_start=["오전 아홉 시에 들렀어요"],
        pulse=["맥박은 제가 직접 쟀는데 76이었어요"],
        meal_intake=["점심은 반 정도 드시는 걸 제가 봤고요"],
        fall_event=["넘어지신 일은 없다고 확인했어요"],
    ),
    notes="",
)

GOOD_S1 = GeneratedCase(
    transcript=(
        "오전 열 시에 방문했어요. 혈 압은 130에 80 나왔고 맥박은 72예요. 점심도 다 드 셨고요."
        " 아침 약도 챙겨서 드 셨어요. 어르신은 넘어 지신 적은 없다고 하세요. 오늘도 이야기 나누면서"
        " 방 청소랑 빨래를 조금 도와드렸고 설거지도 같이 했어요. 따님이 다음 주에 오신다고 해서"
        " 어르신이 많이 기다리고 계시더라고요."
    ),
    reference_transcript=(
        "오전 열 시에 방문했어요. 혈압은 130에 80 나왔고 맥박은 72예요. 점심도 다 드셨고요."
        " 아침 약도 챙겨서 드셨어요. 어르신은 넘어지신 적은 없다고 하세요. 오늘도 이야기 나누면서"
        " 방 청소랑 빨래를 조금 도와드렸고 설거지도 같이 했어요. 따님이 다음 주에 오신다고 해서"
        " 어르신이 많이 기다리고 계시더라고요."
    ),
    evidence=_ev(
        visit_start=["오전 열 시에 방문했어요"],
        blood_pressure=["혈 압은 130에 80 나왔고"],
        pulse=["맥박은 72예요"],
        meal_intake=["점심도 다 드 셨고요"],
        medication_taken=["아침 약도 챙겨서 드 셨어요"],
        fall_event=["넘어 지신 적은 없다고 하세요"],
    ),
    notes="",
)

GOOD_L1 = GeneratedCase(
    transcript=(
        "오늘은 날씨가 좀 쌀쌀해서 어르신 댁 가는 길에 목도리를 챙겼어요. 열한 시 반쯤 도착했고요."
        " 들어가자마자 손주 사진부터 보여주시면서 이번에 초등학교 들어간다고 얼마나 자랑하시던지"
        " 몰라요. 그러고 나서 혈압부터 쟀는데 124에 78 나왔어요. 맥박은 72고요. 어르신이 요즘"
        " 반찬 가게 김치가 너무 짜다고 한참 말씀하셨어요. 점심은 밥을 반 공기쯤 드셨고 국은 거의"
        " 다 드셨어요, 그래서 반 정도로 봤어요. 아침 약은 드셨다고 하셔서 약통도 확인했어요."
        " 넘어지신 적은 없다고 하셨고요. 그런데 무릎이 쑤신다고 계속 만지시더라고요, 오래 앉아"
        " 계셔서 그런가 봐요. 그래도 내내 웃으시면서 이야기를 아주 많이 하셨어요. 갈 때쯤에는"
        " 옆집 아주머니가 가져다주신 감을 먹어 보라고 자꾸 챙겨 주셨고, 저도 하나 받아서 나왔어요."
        " 다음 방문은 이틀 뒤라고 말씀드리니까 그때는 따뜻한 차를 끓여 놓으신다고 하셨어요."
    ),
    reference_transcript=None,
    evidence=_ev(
        visit_start=["열한 시 반쯤 도착했고요"],
        blood_pressure=["혈압부터 쟀는데 124에 78 나왔어요"],
        pulse=["맥박은 72고요"],
        meal_intake=["밥을 반 공기쯤 드셨고"],
        medication_taken=["아침 약은 드셨다고 하셔서"],
        fall_event=["넘어지신 적은 없다고 하셨고요"],
        mood=["내내 웃으시면서 이야기를 아주 많이 하셨어요"],
        special_notes=["무릎이 쑤신다고 계속 만지시더라고요"],
    ),
    notes="",
)

GOOD_CASES: dict[str, GeneratedCase] = {
    "D1": GOOD_D1,
    "H3": GOOD_H3,
    "S1": GOOD_S1,
    "L1": GOOD_L1,
}
