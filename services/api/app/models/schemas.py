# =============================================================================
# 케어노트AI — 데이터 모델 (SQLAlchemy 2.0 ORM)
#
# [What] DB 테이블을 파이썬 클래스로 정의한다. 현재는 Draft(초안)와
#        전 테이블 공통 뼈대(Base·타임스탬프·soft delete)만 있다.
#        옛 스코프의 CareRecipient·FillJob 은 2026-10-12 에 제거했다.
#        통합기술설계서 §8 의 Recording·TranscriptSegment·Fact·ReviewIssue·ExperimentRun 은 추가 예정.
# [Why]  6.3 도메인 모델을 코드로 옮긴다. "지금 단순하게 만들어도 버려지는 코드가
#        없다"(기획서 §전제) — 뒤 Phase 들이 이 스키마 위에 얹힌다.
# [How]  SQLAlchemy 2.0 타입주도 스타일(Mapped/mapped_column). 현업에서는
#        믹스인으로 공통 컬럼을 한 곳에 모아 12개 테이블에 중복 없이 상속시킨다.
#
# ※ 아직 pyproject 에 의존성이 없어 import 는 실행되지 않는다. 다음 마이크로스텝:
#   (a) pyproject 에 sqlalchemy/psycopg 추가  (b) migrations/001_init.sql 작성.
# =============================================================================
from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# -----------------------------------------------------------------------------
# Base + 공통 믹스인
#   [Why] 6.3: "전 테이블 공통: created_at / updated_at / soft delete(deleted_at)".
#         상속으로 강제하면 새 테이블이 규칙을 빠뜨릴 수 없다.
# -----------------------------------------------------------------------------
class Base(DeclarativeBase):
    """모든 ORM 모델의 부모. metadata 를 공유해 마이그레이션이 한눈에 잡힌다."""


class TimestampMixin:
    # server_default=func.now(): DB 시계가 진실. 앱 서버 시계와 어긋나도 일관.
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class SoftDeleteMixin:
    # [Why] 요양 도메인은 원본 삭제가 곧 감사 불능. 물리 삭제 대신 deleted_at 만 찍는다.
    #       (LLM 이 안 하는 일: "원본 삭제/수정" — §6.4)
    deleted_at: Mapped[datetime | None] = mapped_column(default=None)


def _pk() -> Mapped[uuid.UUID]:
    """UUID 기본키 헬퍼. 분산·병합에 안전하고 순번 노출(정보유출)이 없다."""
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


# -----------------------------------------------------------------------------
# 열거형 — 상태·문서타입은 자유 문자열이 아니라 닫힌 집합
#   [How] MVP 는 네이티브 PG enum 대신 String + StrEnum 검증(마이그레이션이 가볍다).
# -----------------------------------------------------------------------------
class DraftStatus(StrEnum):
    DRAFT = "draft"            # 생성 직후
    NEEDS_INFO = "needs_info"  # ValidateDraft 가 부족정보 판단 → 되묻기 루프
    IN_REVIEW = "in_review"    # 실무자 검토 중
    APPROVED = "approved"      # 사람이 승인 (HITL 게이트 통과)
    REJECTED = "rejected"


class DocType(StrEnum):
    VISIT_RECORD = "visit_record"  # 방문기록지
    CARE_NOTE = "care_note"        # 급여제공기록/메모 정리
    STATE_CHANGE = "state_change"  # 상태변화 보고


# -----------------------------------------------------------------------------
# Draft — AI 초안 (파이프라인의 산출물, HITL 검토 대상)
# -----------------------------------------------------------------------------
class Draft(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "drafts"

    id: Mapped[uuid.UUID] = _pk()
    # 수급자 개체는 새 설계에 맞춰 다시 정의할 예정이라 FK 없이 ID 만 보관한다.
    recipient_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))

    doc_type: Mapped[str] = mapped_column(String(32))   # DocType
    status: Mapped[str] = mapped_column(String(16), default=DraftStatus.DRAFT)

    # [Why] 문서타입마다 필드가 다르다 → 정형 컬럼 대신 JSONB 한 칸.
    #       Postgres JSONB 는 색인·부분질의가 되어 "정형+반정형 한 곳"(§기술스택) 실현.
    #       구조 검증은 ValidateDraft 노드가 JSON Schema 로 (컬럼 제약 아님).
    content: Mapped[dict] = mapped_column(JSONB, default=dict)

