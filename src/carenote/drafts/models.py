"""초안 응답 모델. 필드 계약은 docs/blueprint.md 4.2를 따른다."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from carenote.evaluation.schema import FieldKey, Flag, Source

MAX_TRANSCRIPT_CHARS = 20_000
MAX_EDIT_CHARS = 200
MAX_REASON_CHARS = 500


class DraftStatus(StrEnum):
    """초안 상태. 사람이 결정(approved·rejected)하면 더 바꿀 수 없다."""

    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class EvidenceSpan(BaseModel):
    """근거 구간. start·end는 transcript 안의 위치이고, 원문에서 못 찾으면 None이다."""

    model_config = ConfigDict(frozen=True)

    text: str
    start: int | None = None
    end: int | None = None


class DraftField(BaseModel):
    """초안 필드 하나."""

    model_config = ConfigDict(frozen=True)

    key: FieldKey
    value: str | None
    source: Source
    evidence: tuple[EvidenceSpan, ...] = ()
    flag: Flag
    candidates: tuple[str, ...] = ()
    # blueprint 4.2 checks. verifier: supported | unsupported | ambiguous | unavailable.
    # 키가 없으면 그 검사를 하지 않았다는 뜻이다(unavailable은 검사가 실패한 것).
    checks: dict[str, str] = Field(default_factory=lambda: {})
    # 사람이 쓴 값이면 True. 근거(evidence)는 비우고 source는 stated, flag는 ok로 둔다(사람이 확정).
    edited: bool = False


class SignalOut(BaseModel):
    """Sentinel이 낸 위험 신호 (원문 일부만 담는다)."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    category: str
    quote: str
    severity: Flag
    source: str


class RevisionRecord(BaseModel):
    """초안의 한 시점. append-only라 기존 revision은 바뀌지 않는다 (blueprint 5장)."""

    model_config = ConfigDict(frozen=True)

    rev: int
    author: Literal["ai", "user"]
    at: datetime
    fields: tuple[DraftField, ...]
    # 이전 revision 대비 값이 바뀐 필드: {필드 키: {"from": 이전 값, "to": 새 값}}
    diff: dict[str, dict[str, str | None]] = Field(default_factory=lambda: {})


class AuditEntry(BaseModel):
    """감사 기록 한 줄 (수정 불가, 본문·식별정보는 담지 않는다)."""

    model_config = ConfigDict(frozen=True)

    at: datetime
    actor: str
    action: str
    detail: str = ""


class Draft(BaseModel):
    """생성된 초안. revision은 append-only 확장을 염두에 둔 번호다."""

    id: str
    status: DraftStatus
    revision: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    transcript: str
    fields: tuple[DraftField, ...]
    signals: tuple[SignalOut, ...] = ()
    revisions: tuple[RevisionRecord, ...] = ()
    audit: tuple[AuditEntry, ...] = ()

    @property
    def flag_counts(self) -> dict[str, int]:
        """flag별 필드 수."""
        counts = {flag.value: 0 for flag in Flag}
        for field in self.fields:
            counts[field.flag.value] += 1
        return counts


class DraftCreate(BaseModel):
    """POST /drafts 요청."""

    transcript: str = Field(min_length=1, max_length=MAX_TRANSCRIPT_CHARS)


class DraftOut(BaseModel):
    """GET·POST 응답."""

    id: str
    status: DraftStatus
    revision: int
    created_at: datetime
    transcript: str
    fields: tuple[DraftField, ...]
    signals: tuple[SignalOut, ...]
    flag_counts: dict[str, int]
    revisions: tuple[RevisionRecord, ...]
    audit: tuple[AuditEntry, ...]


class DraftSummary(BaseModel):
    """목록 항목 (transcript 본문 없음)."""

    id: str
    status: DraftStatus
    created_at: datetime
    flag_counts: dict[str, int]


EditValue = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_EDIT_CHARS)
]


class DecisionRequest(BaseModel):
    """POST /drafts/{id}/decision 요청.

    approve는 flag가 ok가 아닌 필드를 모두 `edits`로 고치거나 `confirmed`로 확인했을 때만 받는다.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    edits: dict[FieldKey, EditValue] = Field(default_factory=lambda: {})
    confirmed: list[FieldKey] = Field(default_factory=lambda: [])
    reason: str | None = Field(default=None, max_length=MAX_REASON_CHARS)
