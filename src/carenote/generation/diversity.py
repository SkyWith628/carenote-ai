"""케이스 간 다양성 배정: 코드가 결정적으로 정한다.

생성 호출이 서로 독립이라 생성기는 다른 케이스를 모른다. 프롬프트로 "다양하게"라고 부탁하는 대신
전체 스펙 목록에서 spec_id마다 아래를 미리 배정해 사용자 메시지에 넣는다.
- 잡담 소재 2개와 첫 문장 유형 1개 (g4: 따님·반찬·냉장고 모티프 반복 대응)
- 항목 언급 순서 (g5: 7건 중 5건이 스키마 순서대로 읽던 문제 대응)
- 정상 항목(낙상·복약·식사량)의 표현 방식, g8부터 분할별 라운드로빈 (g5: 같은 문장 4건 반복)
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
    # 힌트는 말하는 틀만 설명한다. 예시 문장을 넣으면 그대로 복사된다 (g5 dev 3건).
    (FieldKey.FALL_EVENT, "없음"): (
        "'낙상'이나 '사고' 같은 사무적인 단어를 써서 짧게 끝낸다",
        "약이나 식사 이야기에 이어 붙여 한 문장 안에서 지나가듯 말한다",
        "어르신이 한 말로 전한다. 표현은 직접 지어내되 흔한 관용구는 피하고, "
        "낙상이 없었다는 뜻이 그 문장에서 분명히 읽히게 한다",
        "'안 ~셨다' 꼴의 부정 서술로 말한다",
        "다른 이야기 도중에 곁다리로 한마디 던진다",
    ),
    (FieldKey.MEDICATION_TAKEN, "복용"): (
        "약 종류나 시간대를 함께 언급하며 말한다",
        "챙겨 드린 행동을 중심으로 말한다",
        "다른 항목과 한 문장에 묶어 말한다",
        "어르신이 한 말로 전한다. 표현은 직접 지어내되, "
        "약을 드셨다는 뜻이 그 문장에서 분명히 읽히게 한다",
        "아주 짧게 한마디로 말한다",
    ),
    (FieldKey.MEAL_INTAKE, "전량"): (
        "남기신 게 없다는 쪽에서 말한다",
        "그릇이 비었다는 쪽에서 말한다",
        "다 드셨다고 직설적으로 말한다",
    ),
    (FieldKey.MEAL_INTAKE, "절반"): (
        "반쯤 드시고 남겼다는 쪽에서 말한다",
        "절반 정도라고 직설적으로 말한다",
        "밥의 양을 그릇 단위로 어림해서 말한다",
    ),
    (FieldKey.MEAL_INTAKE, "소량"): (
        "숟가락질 횟수로 어림해서 말한다",
        "조금만 드셨다고 직설적으로 말한다",
        "수저를 금방 내려놓으셨다는 쪽에서 말한다",
    ),
}


# 유형마다 구체 장면 5개. 같은 유형이 한 분할에서 여러 번 쓰여도 장면이 겹치지 않게 순번으로 고른다
# (g5: 같은 유형을 받은 두 건이 똑같이 "목이 좀 잠겨서"로 수렴했다). 날씨·시각·냉장고는 쓰지 않는다.
OPENING_SEEDS: tuple[tuple[str, ...], ...] = (
    (
        "오늘은 왜 이렇게 늦었냐고 농담을 건넴",
        "배고프지 않냐고 먼저 물어봄",
        "열쇠를 못 찾겠다며 허둥대심",
        "어제 본 프로그램 이야기를 바로 꺼냄",
        "신발부터 가지런히 놓으라고 챙김",
    ),
    (
        "복도에서 청소하시는 분과 마주침",
        "현관 비밀번호를 잘못 눌러 헤맴",
        "신발장 앞에 못 보던 신발이 놓여 있었음",
        "엘리베이터 안내문을 읽다가 한 대 놓침",
        "현관 센서등이 계속 깜빡거림",
    ),
    (
        "가구 배치가 바뀌어 있었음",
        "달력이 새것으로 바뀌어 있었음",
        "화분이 하나 늘어 있었음",
        "TV 위치가 달라져 있었음",
        "현관 매트가 새것으로 바뀌어 있었음",
    ),
    (
        "TV와 라디오 소리가 동시에 들림",
        "방 안 공기가 답답하게 느껴짐",
        "거실이 말끔하게 정돈되어 있었음",
        "볕이 거실 깊숙이 들어와 있었음",
        "주방에서 음식 냄새가 났음",
    ),
    (
        "창문부터 열어 환기시킴",
        "마른 빨래를 먼저 걷음",
        "쌓인 신문을 정리함",
        "현관 신발을 정리함",
        "전기밥솥 상태를 먼저 확인함",
    ),
    (
        "센터 담당자에게서 일정 변경 전화가 옴",
        "가족이 보낸 부탁 문자를 받음",
        "다른 보호사와 인수인계 통화를 함",
        "약국에서 온 안내 문자를 확인함",
        "어르신이 먼저 안부 전화를 걸어 옴",
    ),
    (
        "어제 잠을 설쳐서 피곤했음",
        "점심을 거르고 오느라 배가 고팠음",
        "신발이 불편해 걷기 힘들었음",
        "버스를 놓칠 뻔함",
        "새로 산 가방이 무거웠음",
    ),
    (
        "옆집 아주머니와 인사를 나눔",
        "경비 아저씨가 당부 한마디를 함",
        "어르신 아드님과 계단에서 마주침",
        "이웃 아이가 먼저 인사를 함",
        "앞 동 어르신과 짧게 대화함",
    ),
    (
        "돋보기를 쓰고 신문을 읽고 계심",
        "화투패를 만지작거리고 계심",
        "TV 앞에서 깜빡 졸고 계심",
        "화분에 물을 주고 계심",
        "전화번호를 찾느라 수첩을 뒤지고 계심",
    ),
)

SEEDS_PER_TYPE = 5


@dataclass(frozen=True)
class Diversity:
    """스펙 한 건에 배정된 다양성 지시."""

    opening_type: str
    opening_seed: str
    topics: tuple[Topic, Topic]
    field_order: tuple[FieldKey, ...]
    phrasing: tuple[tuple[FieldKey, str], ...] = ()

    def to_payload(self) -> dict[str, object]:
        """사용자 메시지에 넣는 형태 (키워드는 숨긴다)."""
        return {
            "opening_type": self.opening_type,
            "opening_scene": self.opening_seed,
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


def _assign_phrasing(
    ordered: list[Spec],
) -> dict[str, tuple[tuple[FieldKey, str], ...]]:
    """정답이 단순 정상 값(stated)인 항목에 표현 방식을 분할별 라운드로빈으로 배정한다.

    같은 분할·같은 항목·같은 값을 가진 스펙끼리 id순으로 풀을 한 칸씩 돌아, 풀 크기 이하의 스펙은
    서로 다른 방식을 받는다 (g7: 무작위 배정으로 세 건이 같은 "어르신이 한 말로 전한다" 힌트를 받아
    같은 발화로 수렴했다).
    """
    counters: dict[tuple[str, FieldKey, str], int] = {}
    result: dict[str, list[tuple[FieldKey, str]]] = {s.spec_id: [] for s in ordered}
    for spec in ordered:
        for key in (FieldKey.MEAL_INTAKE, FieldKey.MEDICATION_TAKEN, FieldKey.FALL_EVENT):
            gold = spec.gold[key]
            pool = PHRASING_POOLS.get((key, gold.value or ""))
            if pool is None or gold.source.value != "stated":
                continue
            slot = (spec.split, key, gold.value or "")
            count = counters.get(slot, 0)
            counters[slot] = count + 1
            result[spec.spec_id].append((key, pool[count % len(pool)]))
    return {sid: tuple(hints) for sid, hints in result.items()}


def assign_diversity(specs: Iterable[Spec]) -> dict[str, Diversity]:
    """소재·첫 문장은 순번으로, 항목 순서·표현은 spec_id로 시드해 배정한다.

    - 소재: 전체 스펙을 id순으로 세워 첫 소재는 풀을 순서대로, 둘째 소재는 반 바퀴 뒤에서 고르되
      한 바퀴를 돌 때마다 간격을 1씩 늘려 쌍이 반복되지 않게 한다. 40건 안에서 한 소재는 최대 2번.
    - 표현 방식: 분할별·항목별·값별 라운드로빈 (`_assign_phrasing`).
    - 첫 문장 유형과 구체 장면: **분할(dev/test)별로** id순 라운드로빈. 같은 분할 안에서 유형이
      고르게 쓰이고, 같은 유형이 다시 나올 때마다 다른 구체 장면을 받는다 (g5에서 같은 유형을 받은
      두 건이 같은 표현으로 수렴한 문제의 대응).
    """
    ordered = sorted({s.spec_id: s for s in specs}.values(), key=lambda s: s.spec_id)
    half = len(TOPICS) // 2
    split_rank: dict[str, int] = {}
    counters: dict[str, int] = {}
    for spec in ordered:
        split_rank[spec.spec_id] = counters.get(spec.split, 0)
        counters[spec.split] = split_rank[spec.spec_id] + 1
    phrasing = _assign_phrasing(ordered)
    assignment: dict[str, Diversity] = {}
    for i, spec in enumerate(ordered):
        rank = split_rank[spec.spec_id]
        type_index = rank % len(OPENING_TYPES)
        seed_index = (rank // len(OPENING_TYPES)) % SEEDS_PER_TYPE
        assignment[spec.spec_id] = Diversity(
            opening_type=OPENING_TYPES[type_index],
            opening_seed=OPENING_SEEDS[type_index][seed_index],
            topics=(
                TOPICS[i % len(TOPICS)],
                TOPICS[(i + half + i // half) % len(TOPICS)],
            ),
            field_order=_order_for(spec.spec_id),
            phrasing=phrasing[spec.spec_id],
        )
    return assignment
