"""transcript → 초안. 추출·flag 계산은 주입된 extractor가 하고 여기서는 응답 모양만 만든다."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol

from carenote.drafts.models import (
    AuditEntry,
    DecisionRequest,
    Draft,
    DraftField,
    DraftStatus,
    EvidenceSpan,
    RevisionRecord,
    SignalOut,
)
from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import Flag, Source
from carenote.extraction.sentinel import Signal
from carenote.extraction.verifier import FieldVerdict, VerifyingExtractor


def locate(transcript: str, text: str) -> EvidenceSpan:
    """근거 문장이 transcript의 어디인지 찾는다. 못 찾으면 위치 없이 돌려준다."""
    start = transcript.find(text)
    if start < 0:
        return EvidenceSpan(text=text)
    return EvidenceSpan(text=text, start=start, end=start + len(text))


class DraftExtractor(Protocol):
    """초안을 만드는 추출기: FlaggingExtractor 또는 그것을 감싼 VerifyingExtractor."""

    def extract(self, transcript: str) -> list[PredictedField]:
        """flag까지 계산된 필드 목록."""
        ...

    def signals_for(self, transcript: str) -> list[Signal]:
        """마지막 extract의 Sentinel 신호."""
        ...


def to_draft_field(
    transcript: str, field: PredictedField, verdicts: Mapping[str, FieldVerdict] | None = None
) -> DraftField:
    """PredictedField를 응답 필드로 바꾼다. 검사기별 판정은 checks[검사기 이름]에 싣는다."""
    return DraftField(
        key=field.key,
        value=field.value,
        source=field.source,
        evidence=tuple(locate(transcript, e.text) for e in field.evidence),
        flag=field.flag,
        candidates=field.candidates,
        checks={name: v.status for name, v in (verdicts or {}).items()},
    )


def to_signal(signal: Signal) -> SignalOut:
    """Sentinel 신호를 응답 모양으로 바꾼다."""
    return SignalOut(
        rule_id=signal.rule_id,
        category=signal.category,
        quote=signal.quote,
        severity=signal.severity,
        source=signal.source,
    )


def create_draft(transcript: str, extractor: DraftExtractor, *, masked_count: int = 0) -> Draft:
    """추출기를 한 번 실행해 검토 대기 초안을 만든다. masked_count는 마스킹 횟수(감사용)."""
    fields = extractor.extract(transcript)
    verdicts = (
        extractor.all_verdicts_for(transcript) if isinstance(extractor, VerifyingExtractor) else {}
    )
    draft_fields = tuple(
        to_draft_field(transcript, f, {n: v[f.key] for n, v in verdicts.items() if f.key in v})
        for f in fields
    )
    now = datetime.now(UTC)
    return Draft(
        id=uuid.uuid4().hex,
        status=DraftStatus.IN_REVIEW,
        created_at=now,
        transcript=transcript,
        fields=draft_fields,
        signals=tuple(to_signal(s) for s in extractor.signals_for(transcript)),
        revisions=(RevisionRecord(rev=1, author="ai", at=now, fields=draft_fields),),
        audit=(
            AuditEntry(
                at=now,
                actor="system",
                action="created",
                detail=f"masked={masked_count}" if masked_count else "",
            ),
        ),
    )


class DecisionError(Exception):
    """결정을 받을 수 없다. `conflict`는 이미 결정된 초안, 아니면 요청 내용 문제다."""

    def __init__(self, message: str, *, conflict: bool = False, unresolved: tuple[str, ...] = ()):
        super().__init__(message)
        self.message = message
        self.conflict = conflict
        self.unresolved = unresolved


def _human_field(field: DraftField, value: str) -> DraftField:
    """사람이 쓴 값으로 바꾼다. 모델 근거·후보·검사 결과는 이 값에 해당하지 않으므로 비운다."""
    return field.model_copy(
        update={
            "value": value,
            "source": Source.STATED,
            "evidence": (),
            "candidates": (),
            "flag": Flag.OK,
            "checks": {},
            "edited": True,
        }
    )


def decide(draft: Draft, request: DecisionRequest, actor: str = "reviewer") -> Draft:
    """사람의 결정을 반영한 새 초안을 돌려준다 (입력 초안은 바꾸지 않는다).

    approve는 flag가 ok가 아닌 필드마다 사람의 처리(edits 또는 confirmed)를 요구한다. 값을 고치면
    새 revision(author=user)을 추가하고, 고치지 않은 승인은 revision을 늘리지 않는다. 모든 결정은
    감사 기록에 남는다. 감사 기록에는 필드 키만 담고 값·본문·사유 내용은 담지 않는다.
    """
    if draft.status is not DraftStatus.IN_REVIEW:
        raise DecisionError(f"이미 {draft.status.value} 상태라 결정할 수 없다", conflict=True)
    now = datetime.now(UTC)
    if request.decision == "reject":
        entry = AuditEntry(
            at=now,
            actor=actor,
            action="reject",
            detail="사유 있음" if request.reason else "",
        )
        return draft.model_copy(
            update={"status": DraftStatus.REJECTED, "audit": (*draft.audit, entry)}
        )

    handled = set(request.edits) | set(request.confirmed)
    unresolved = tuple(
        f.key.value for f in draft.fields if f.flag is not Flag.OK and f.key not in handled
    )
    if unresolved:
        raise DecisionError(
            "확인 필요·위험 필드를 모두 고치거나 확인해야 승인할 수 있다", unresolved=unresolved
        )

    fields: list[DraftField] = []
    diff: dict[str, dict[str, str | None]] = {}
    for f in draft.fields:
        new_value = request.edits.get(f.key)
        if new_value is not None and new_value != f.value:
            diff[f.key.value] = {"from": f.value, "to": new_value}
            f = _human_field(f, new_value)
        fields.append(f)
    revision = draft.revision
    revisions = draft.revisions
    if diff:
        revision += 1
        revisions = (
            *revisions,
            RevisionRecord(rev=revision, author="user", at=now, fields=tuple(fields), diff=diff),
        )
    edited_keys = ",".join(sorted(diff))
    confirmed_keys = ",".join(sorted(k.value for k in request.confirmed))
    entry = AuditEntry(
        at=now,
        actor=actor,
        action="approve",
        detail=f"edited=[{edited_keys}] confirmed=[{confirmed_keys}]",
    )
    return draft.model_copy(
        update={
            "status": DraftStatus.APPROVED,
            "fields": tuple(fields),
            "revision": revision,
            "revisions": revisions,
            "audit": (*draft.audit, entry),
        }
    )
