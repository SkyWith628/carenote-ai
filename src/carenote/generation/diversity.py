"""케이스 간 다양성 배정: 코드가 결정적으로 정한다.

생성 호출이 서로 독립이라 생성기는 다른 케이스를 모른다. 프롬프트로 "다양하게"라고 부탁하는 대신
전체 스펙 목록에서 spec_id마다 아래를 미리 배정해 사용자 메시지에 넣는다.
- 잡담 소재 2개와 첫 문장 유형 1개 (g4: 따님·반찬·냉장고 모티프 반복 대응)
- 항목 언급 순서 (g5: 7건 중 5건이 스키마 순서대로 읽던 문제 대응)
- 정상 항목(낙상·복약·식사량)의 표현 방식 (g5: "넘어지신 일은 없었어요"가 4건에 반복된 문제 대응)
같은 스펙 집합이면 항상 같은 배정이 나온다 (난수는 spec_id 문자열로 시드한다).
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from carenote.evaluation.schema import FieldKey
from carenote.generation.spec import Spec


@dataclass(frozen=True)
class Topic:
    """비임상 잡담 소재. keywords는 생성물에서 사용 여부를 확인하는 용도다."""

    label: str
    keywords: tuple[str, ...]


TOPICS: tuple[Topic, ...] = (
    Topic("동네 길 공사 소음", ("공사", "소음", "포크레인")),
    Topic("손주 학교 운동회", ("운동회", "학교")),
    Topic("베란다 화분과 꽃", ("화분", "꽃", "베란다")),
    Topic("오래된 사진첩", ("사진첩", "앨범", "옛날 사진")),
    Topic("아침 라디오 프로그램", ("라디오",)),
    Topic("트로트 가수 이야기", ("트로트", "가수")),
    Topic("시장 채소 값", ("채소", "시금치", "배추")),
    Topic("옆집 강아지", ("강아지", "옆집")),
    Topic("새로 나온 과자", ("과자", "간식")),
    Topic("교회나 절 모임", ("교회", "성당", "절에")),
    Topic("동창 모임 소식", ("동창",)),
    Topic("고향 마을 이야기", ("고향",)),
    Topic("예전에 다니던 직장", ("직장", "회사")),
    Topic("저녁 드라마 줄거리", ("드라마",)),
    Topic("이사 간 이웃", ("이사", "이웃")),
    Topic("택배 아저씨", ("택배",)),
    Topic("추석 준비", ("추석", "명절")),
    Topic("김장 계획", ("김장",)),
    Topic("장마철 빨래", ("장마", "빨래")),
    Topic("가을 단풍 구경", ("단풍",)),
    Topic("옛날 노래 흥얼거리기", ("노래", "흥얼")),
    Topic("뜨개질", ("뜨개", "털실")),
    Topic("마을 경로당 소식", ("경로당",)),
    Topic("손녀 결혼 이야기", ("결혼", "손녀")),
    Topic("군대 간 손자", ("군대", "입대")),
    Topic("해외 사는 자녀와 영상통화", ("영상통화", "해외")),
    Topic("새로 맞춘 안경", ("안경",)),
    Topic("멈춘 벽시계", ("벽시계", "시계")),
    Topic("아침 산책길", ("산책",)),
    Topic("공원 비둘기", ("비둘기", "공원")),
    Topic("이불 말리기", ("이불",)),
    Topic("전기장판 꺼내기", ("전기장판",)),
    Topic("보리차 끓이기", ("보리차",)),
    Topic("누룽지 간식", ("누룽지",)),
    Topic("호박죽 쑤기", ("호박죽",)),
    Topic("홈쇼핑 방송", ("홈쇼핑",)),
    Topic("버스 노선 변경", ("버스", "노선")),
    Topic("쌀 가격", ("쌀",)),
    Topic("텃밭 상추", ("텃밭", "상추")),
    Topic("마당의 감나무", ("홍시", "감나무")),
    Topic("동네에 새로 생긴 가게", ("새로 생긴", "가게")),
    Topic("보험 광고 전화", ("보험", "광고")),
    Topic("우편함에 온 우편물", ("우편", "우체")),
    Topic("젊을 때 연애 이야기", ("연애", "중매")),
    Topic("피난 시절 이야기", ("피난", "전쟁")),
    Topic("젊을 때 농사일", ("농사",)),
    Topic("아파트 승강기 점검", ("승강기", "엘리베이터")),
    Topic("겨울 난방비", ("난방비", "난방")),
)

OPENING_TYPES: tuple[str, ...] = (
    "어르신이 문을 열어 주며 먼저 건넨 한마디",
    "도착하기 직전 현관이나 복도에서 있었던 일",
    "지난 방문 때와 달라진 점 한 가지",
    "집 안에 들어섰을 때의 첫인상(냄새, 정돈 상태, 켜 둔 TV 등)",
    "도착하자마자 먼저 한 일(환기, 장 본 것 정리, 전기밥솥 확인 등)",
    "오는 길이나 출근 전에 받은 연락",
    "화자 자신의 개인적인 사정이나 컨디션 한마디",
    "이웃이나 가족과 마주친 일",
    "어르신이 하고 계시던 일(TV, 화분, 뜨개질 등)을 본 장면",
)


# 스키마(체크리스트) 순서. 생성물이 이 순서를 따르지 않도록 배정 순서는 이와 달라야 한다.
CANONICAL_ORDER: tuple[FieldKey, ...] = (
    FieldKey.VISIT_START,
    FieldKey.BLOOD_PRESSURE,
    FieldKey.PULSE,
    FieldKey.MEAL_INTAKE,
    FieldKey.MEDICATION_TAKEN,
    FieldKey.FALL_EVENT,
)

# 정상 보고(정답이 단순 값인 항목)의 말하는 방식. scene·must_include가 정한 항목에는 쓰지 않는다.
PHRASING_POOLS: dict[tuple[FieldKey, str], tuple[str, ...]] = {
    (FieldKey.FALL_EVENT, "없음"): (
        "'낙상'이나 '사고'라는 말을 쓰고 짧게 끝낸다",
        "약·식사 이야기에 이어서 한 문장에 묶어 말한다",
        "어르신의 말을 그대로 옮긴다 (넘어질 일이 뭐 있냐고 하시네요 식으로)",
        "'안 ~셨어요' 부정형으로 말한다",
        "다른 이야기 도중에 곁다리로 한마디 던진다",
    ),
    (FieldKey.MEDICATION_TAKEN, "복용"): (
        "약 종류나 시간대를 함께 말한다 (아침 약은 드셨고 식으로)",
        "챙겨 드린 행동으로 말한다 (물이랑 챙겨 드렸더니 식으로)",
        "다른 항목과 한 문장에 묶어 말한다",
        "어르신의 말을 옮긴다 (약은 먹었다고 하세요 식으로)",
        "한마디로 짧게 말한다",
    ),
    (FieldKey.MEAL_INTAKE, "전량"): (
        "남기신 게 없다는 식으로 말한다",
        "그릇을 싹 비웠다는 식으로 말한다",
        "다 드셨다고 직설적으로 말한다",
    ),
    (FieldKey.MEAL_INTAKE, "절반"): (
        "반쯤 드시고 남기셨다는 식으로 말한다",
        "절반 정도라고 직설적으로 말한다",
        "밥은 반 공기쯤이라는 식으로 말한다",
    ),
    (FieldKey.MEAL_INTAKE, "소량"): (
        "몇 숟가락만이라는 식으로 말한다",
        "조금만 드셨다고 직설적으로 말한다",
        "수저를 금방 내려놓으셨다는 식으로 말한다",
    ),
}


@dataclass(frozen=True)
class Diversity:
    """스펙 한 건에 배정된 다양성 지시."""

    opening_type: str
    topics: tuple[Topic, Topic]
    field_order: tuple[FieldKey, ...]
    phrasing: tuple[tuple[FieldKey, str], ...] = ()

    def to_payload(self) -> dict[str, object]:
        """사용자 메시지에 넣는 형태 (키워드는 숨긴다)."""
        return {
            "opening_type": self.opening_type,
            "chatter_topics": [t.label for t in self.topics],
            "field_order": [k.value for k in self.field_order],
            "phrasing_hints": {k.value: hint for k, hint in self.phrasing},
        }


def _order_for(spec_id: str) -> tuple[FieldKey, ...]:
    """spec_id로 시드한 항목 언급 순서. 스키마 순서와 같으면 다시 섞는다."""
    rng = random.Random(f"order:{spec_id}")
    order = list(CANONICAL_ORDER)
    while tuple(order) == CANONICAL_ORDER:
        rng.shuffle(order)
    return tuple(order)


def _phrasing_for(spec: Spec) -> tuple[tuple[FieldKey, str], ...]:
    """정답이 단순 정상 값인 항목에만 표현 방식을 배정한다."""
    hints: list[tuple[FieldKey, str]] = []
    for key in (FieldKey.MEAL_INTAKE, FieldKey.MEDICATION_TAKEN, FieldKey.FALL_EVENT):
        gold = spec.gold[key]
        pool = PHRASING_POOLS.get((key, gold.value or ""))
        if pool is None or gold.source.value != "stated":
            continue
        rng = random.Random(f"phrase:{key.value}:{spec.spec_id}")
        hints.append((key, pool[rng.randrange(len(pool))]))
    return tuple(hints)


def assign_diversity(specs: Iterable[Spec]) -> dict[str, Diversity]:
    """정렬된 spec_id 순서로 소재 2개, 첫 문장 유형 1개를 배정하고 순서·표현은 spec_id로 시드한다.

    첫 소재는 풀을 순서대로, 둘째 소재는 반 바퀴 뒤에서 고르되 한 바퀴를 돌 때마다 간격을 1씩
    늘려 같은 소재 쌍이 반복되지 않게 한다. 스펙 40건 안에서 한 소재는 최대 2번만 쓰인다.
    """
    ordered = sorted({s.spec_id: s for s in specs}.values(), key=lambda s: s.spec_id)
    half = len(TOPICS) // 2
    assignment: dict[str, Diversity] = {}
    for i, spec in enumerate(ordered):
        first = TOPICS[i % len(TOPICS)]
        second = TOPICS[(i + half + i // half) % len(TOPICS)]
        assignment[spec.spec_id] = Diversity(
            opening_type=OPENING_TYPES[i % len(OPENING_TYPES)],
            topics=(first, second),
            field_order=_order_for(spec.spec_id),
            phrasing=_phrasing_for(spec),
        )
    return assignment
