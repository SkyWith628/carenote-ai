"""케이스 간 다양성 배정: 잡담 소재와 첫 문장 유형을 코드가 결정적으로 정한다.

생성 호출이 서로 독립이라 생성기는 다른 케이스를 모른다. 프롬프트로 "다양하게"라고 부탁하는 대신
전체 스펙 목록에서 spec_id마다 소재 2개와 첫 문장 유형 1개를 미리 배정해 사용자 메시지에 넣는다
(g3 파일럿에서 따님·반찬·냉장고 모티프가 3건에 반복된 문제의 대응).
같은 spec_id 집합이면 항상 같은 배정이 나온다.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


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


@dataclass(frozen=True)
class Diversity:
    """스펙 한 건에 배정된 다양성 지시."""

    opening_type: str
    topics: tuple[Topic, Topic]

    def to_payload(self) -> dict[str, object]:
        """사용자 메시지에 넣는 형태 (키워드는 숨긴다)."""
        return {
            "opening_type": self.opening_type,
            "chatter_topics": [t.label for t in self.topics],
        }


def assign_diversity(spec_ids: Iterable[str]) -> dict[str, Diversity]:
    """정렬된 spec_id 순서로 소재 2개와 첫 문장 유형 1개를 배정한다.

    첫 소재는 풀을 순서대로, 둘째 소재는 반 바퀴 뒤에서 고르되 한 바퀴를 돌 때마다 간격을 1씩
    늘려 같은 소재 쌍이 반복되지 않게 한다. 스펙 40건 안에서 한 소재는 최대 2번만 쓰인다.
    """
    ordered = sorted(set(spec_ids))
    half = len(TOPICS) // 2
    assignment: dict[str, Diversity] = {}
    for i, spec_id in enumerate(ordered):
        first = TOPICS[i % len(TOPICS)]
        second = TOPICS[(i + half + i // half) % len(TOPICS)]
        assignment[spec_id] = Diversity(
            opening_type=OPENING_TYPES[i % len(OPENING_TYPES)],
            topics=(first, second),
        )
    return assignment
