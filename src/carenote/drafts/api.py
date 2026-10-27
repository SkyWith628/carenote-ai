"""초안 API: POST /drafts, GET /drafts, GET /drafts/{id}, 검토 화면 GET /review."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from carenote.config import get_settings
from carenote.drafts.models import DecisionRequest, Draft, DraftCreate, DraftOut, DraftSummary
from carenote.drafts.service import DecisionError, DraftExtractor, create_draft, decide
from carenote.drafts.store import InMemoryDraftStore
from carenote.extraction.flagger import FlaggingExtractor
from carenote.extraction.llm_extractor import LLMExtractor, build_client
from carenote.extraction.relistener import TextRelistener
from carenote.extraction.sentinel import load_rules
from carenote.extraction.sentinel_llm import SentinelAssist
from carenote.extraction.verifier import Verifier, VerifyingExtractor
from carenote.privacy import mask_pii

SENTINEL_MODEL = "claude-haiku-5-5"
SENTINEL_EFFORT = "low"
STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter()


@lru_cache
def get_store() -> InMemoryDraftStore:
    """프로세스 하나에서 공유하는 저장소."""
    return InMemoryDraftStore()


@lru_cache
def get_extractor() -> VerifyingExtractor:
    """운영용 추출기: Extractor + 규칙 + Sentinel 보조, 이어서 Verifier·Re-listener를 병렬로.

    API 키가 없으면 503으로 응답하도록 ValueError를 HTTPException으로 바꾼다.
    테스트는 이 의존성을 바꿔 끼운다.
    """
    settings = get_settings()
    try:
        client = build_client(settings)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="추출기를 쓸 수 없다 (API 키 미설정)") from exc
    rules = load_rules()
    llm = LLMExtractor(
        client,
        variant="b1",
        model=settings.extractor_model,
        effort=settings.extractor_effort,
        cache_dir=Path(settings.extraction_cache_dir),
    )
    assist = SentinelAssist(client, model=SENTINEL_MODEL, effort=SENTINEL_EFFORT, rules=rules)
    verifier = Verifier(
        client,
        model=settings.verifier_model,
        effort=settings.verifier_effort,
        cache_dir=Path(settings.verifier_cache_dir),
    )
    relistener = TextRelistener(
        client,
        model=settings.relistener_model,
        effort=settings.relistener_effort,
        cache_dir=Path(settings.relistener_cache_dir),
    )
    return VerifyingExtractor(
        FlaggingExtractor(llm, rules, assist), {"verifier": verifier, "relistener": relistener}
    )


Store = Annotated[InMemoryDraftStore, Depends(get_store)]
Extractor = Annotated[DraftExtractor, Depends(get_extractor)]


def to_out(draft: Draft) -> DraftOut:
    """저장된 초안을 응답으로 바꾼다."""
    return DraftOut(
        id=draft.id,
        status=draft.status,
        revision=draft.revision,
        created_at=draft.created_at,
        transcript=draft.transcript,
        fields=draft.fields,
        signals=draft.signals,
        flag_counts=draft.flag_counts,
        revisions=draft.revisions,
        audit=draft.audit,
    )


@router.post("/drafts", response_model=DraftOut, status_code=201)
def post_draft(body: DraftCreate, store: Store, extractor: Extractor) -> DraftOut:
    """transcript로 초안을 만든다. 추출은 동기 호출이라 스레드풀에서 돈다."""
    try:
        masked = mask_pii(body.transcript)
        draft = create_draft(masked.text, extractor, masked_count=masked.total)
    except HTTPException:
        raise
    except Exception as exc:  # 모델·네트워크 오류. 본문은 로그에 남기지 않는다.
        raise HTTPException(status_code=502, detail="추출에 실패했다") from exc
    store.add(draft)
    return to_out(draft)


@router.get("/drafts", response_model=list[DraftSummary])
def list_drafts(store: Store) -> list[DraftSummary]:
    """최신순 요약 목록."""
    return [
        DraftSummary(id=d.id, status=d.status, created_at=d.created_at, flag_counts=d.flag_counts)
        for d in store.list()
    ]


@router.get("/drafts/{draft_id}", response_model=DraftOut)
def get_draft(draft_id: str, store: Store) -> DraftOut:
    """초안 하나."""
    draft = store.get(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="초안이 없다")
    return to_out(draft)


@router.post("/drafts/{draft_id}/decision", response_model=DraftOut)
def post_decision(draft_id: str, body: DecisionRequest, store: Store) -> DraftOut:
    """사람의 승인·거절을 반영한다. 이미 결정된 초안은 409, 미해결 필드가 있으면 422.

    인증이 아직 없어 행위자는 고정값(reviewer)이다. 인증 도입 전에는 외부에 열면 안 된다.
    """
    try:
        draft = store.update(draft_id, lambda current: decide(current, body))
    except DecisionError as exc:
        if exc.conflict:
            raise HTTPException(status_code=409, detail=exc.message) from exc
        raise HTTPException(
            status_code=422,
            detail={"message": exc.message, "unresolved": list(exc.unresolved)},
        ) from exc
    if draft is None:
        raise HTTPException(status_code=404, detail="초안이 없다")
    return to_out(draft)


@router.get("/review", include_in_schema=False)
def review_page() -> FileResponse:
    """검토 화면(정적 HTML)."""
    return FileResponse(STATIC_DIR / "review.html")
