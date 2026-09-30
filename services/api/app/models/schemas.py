# =============================================================================
# 케어노트AI — 데이터 모델 (SQLAlchemy 2.0 ORM)
#
# [What] DB 테이블을 파이썬 클래스로 정의한다. 이번 스텝은 build order 2단계가
#        콕 집은 핵심 3엔티티 — CareRecipient(수급자)·Draft(초안)·FillJob(자동채움 작업)
#        + 전 테이블 공통 뼈대(Base·타임스탬프·soft delete).
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

from sqlalchemy import ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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


class FillJobStatus(StrEnum):
    PENDING = "pending"      # 승인 직후 생성, 확장 팝업 대기
    CLAIMED = "claimed"      # 확장이 집어감
    DONE = "done"            # read-back 검증까지 완료
    EXPIRED = "expired"      # TTL 만료 (미claim)
    FAILED = "failed"


# -----------------------------------------------------------------------------
# CareRecipient — 수급자
# -----------------------------------------------------------------------------
class CareRecipient(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "care_recipients"

    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(String(100))

    # [보안] 장기요양인정번호는 고유식별정보 → 암호화 컬럼, 키는 앱코드와 분리(§6.3).
    #   지금은 자리만 잡는다. 실제 암호화는 10단계 core/security.py 의
    #   EncryptedString(TypeDecorator) 로 교체 → 평문으로 DB 에 남기지 않는다.
    # TODO(보안): String → EncryptedString 로 교체, 절대 평문 저장 금지.
    ltc_number_enc: Mapped[str | None] = mapped_column(String(255), default=None)

    # 1:N — 한 수급자에게 여러 초안. soft-deleted 포함 여부는 쿼리에서 필터.
    drafts: Mapped[list[Draft]] = relationship(back_populates="recipient")


# -----------------------------------------------------------------------------
# Draft — AI 초안 (파이프라인의 산출물, HITL 검토 대상)
# -----------------------------------------------------------------------------
class Draft(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "drafts"

    id: Mapped[uuid.UUID] = _pk()
    recipient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("care_recipients.id")
    )

    doc_type: Mapped[str] = mapped_column(String(32))   # DocType
    status: Mapped[str] = mapped_column(String(16), default=DraftStatus.DRAFT)

    # [Why] 문서타입마다 필드가 다르다 → 정형 컬럼 대신 JSONB 한 칸.
    #       Postgres JSONB 는 색인·부분질의가 되어 "정형+반정형 한 곳"(§기술스택) 실현.
    #       구조 검증은 ValidateDraft 노드가 JSON Schema 로 (컬럼 제약 아님).
    content: Mapped[dict] = mapped_column(JSONB, default=dict)

    recipient: Mapped[CareRecipient] = relationship(back_populates="drafts")


# -----------------------------------------------------------------------------
# FillJob — 자동채움 작업 (승인→확장 사이의 원자적 배턴)
#   §6.7: "초안 승인 → fill_job 생성(원자적, 짧은 TTL)".
# -----------------------------------------------------------------------------
class FillJob(Base, TimestampMixin):
    __tablename__ = "fill_jobs"

    id: Mapped[uuid.UUID] = _pk()
    draft_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("drafts.id"))

    status: Mapped[str] = mapped_column(String(16), default=FillJobStatus.PENDING)

    # 확장이 폼에 채울 필드 맵({셀렉터/필드명: 값}). 민감정보 최소화 원칙 적용 대상.
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)

    # [Why] 짧은 TTL. 승인 직후 잠깐만 유효 → 유출/오채움 창을 좁힌다.
    #       만료 판정은 앱/워커가 now() > expires_at 로. (soft delete 불필요 → 믹스인 제외)
    expires_at: Mapped[datetime | None] = mapped_column(default=None)
