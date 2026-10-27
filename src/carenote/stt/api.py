"""POST /transcribe: 오디오 바이트 → 텍스트(개인식별정보 패턴 마스킹 적용).

요청 본문이 곧 오디오다(`Content-Type: audio/webm` 등). 오디오는 메모리에서만 다루고 저장·로그하지
않는다. 변환 결과를 검토자가 고친 뒤 `POST /drafts`로 보내는 2단계 흐름이라, STT 오류(특히 숫자)를
사람이 먼저 바로잡을 기회가 있다. 인증이 없으므로 외부에 열면 안 된다.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from carenote.config import Settings, get_settings
from carenote.privacy import mask_pii
from carenote.stt.engine import (
    ALLOWED_AUDIO_TYPES,
    SttBusy,
    SttEngine,
    SttError,
    SttUnavailable,
    normalize_content_type,
)
from carenote.stt.whisper_engine import FasterWhisperEngine

router = APIRouter()


class TranscribeOut(BaseModel):
    """변환 결과. text는 마스킹을 거친 것이다."""

    text: str
    duration_s: float
    language: str
    engine: str
    elapsed_s: float
    masked_count: int


@lru_cache
def get_stt_engine() -> SttEngine:
    """운영용 엔진(faster-whisper). 테스트는 이 의존성을 바꿔 끼운다."""
    s = get_settings()
    return FasterWhisperEngine(
        model=s.stt_model,
        device=s.stt_device,
        compute_type=s.stt_compute_type,
        language=s.stt_language,
        max_seconds=s.stt_max_seconds,
        download_root=s.stt_download_root,
        local_files_only=s.stt_local_files_only,
    )


Engine = Annotated[SttEngine, Depends(get_stt_engine)]
Config = Annotated[Settings, Depends(get_settings)]


async def _read_limited(request: Request, limit: int) -> bytes:
    """본문을 limit 바이트까지만 읽는다. 넘으면 413 (Content-Length가 없어도 막는다)."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"오디오가 너무 크다 (최대 {limit}바이트)")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/transcribe", response_model=TranscribeOut)
async def transcribe(
    request: Request, response: Response, engine: Engine, settings: Config
) -> TranscribeOut:
    """오디오를 받아 적는다. 415 형식 · 413 크기 · 400 빈 본문 · 422 음성 없음 · 503 엔진 불가."""
    response.headers["Cache-Control"] = "no-store"
    content_type = normalize_content_type(request.headers.get("content-type"))
    if content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail="지원하지 않는 오디오 형식이다")
    limit = settings.stt_max_audio_bytes
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail=f"오디오가 너무 크다 (최대 {limit}바이트)")
    audio = await _read_limited(request, limit)
    if not audio:
        raise HTTPException(status_code=400, detail="오디오가 비어 있다")
    try:
        result = await run_in_threadpool(engine.transcribe, audio, content_type)
    except SttBusy as exc:
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"Retry-After": "10"}
        ) from exc
    except SttUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SttError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    masked = mask_pii(result.text)
    return TranscribeOut(
        text=masked.text,
        duration_s=result.duration_s,
        language=result.language,
        engine=result.engine,
        elapsed_s=result.elapsed_s,
        masked_count=masked.total,
    )
