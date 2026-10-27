"""STT 엔진 인터페이스. 오디오는 메모리에서만 다루고 저장하지 않는다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# 브라우저 MediaRecorder(webm·ogg·mp4)와 합성 음성(wav·aiff·flac·mp3)을 받는다.
ALLOWED_AUDIO_TYPES = frozenset(
    {
        "audio/webm",
        "audio/ogg",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/aiff",
        "audio/x-aiff",
        "audio/flac",
        "audio/x-flac",
    }
)


class SttError(Exception):
    """오디오를 받아 적을 수 없다 (손상·너무 김·음성 없음 등, 요청 쪽 문제)."""


class SttUnavailable(SttError):
    """엔진을 쓸 수 없다 (패키지·모델이 없음, 로드 실패 등, 서버 쪽 문제)."""


class SttBusy(SttError):
    """다른 변환이 오래 걸려 대기 시간 안에 차례가 오지 않았다."""


@dataclass(frozen=True)
class Transcription:
    """변환 결과. text에는 본문이 들어 있으므로 로그에 남기지 않는다."""

    text: str
    duration_s: float
    language: str
    engine: str
    elapsed_s: float


class SttEngine(Protocol):
    """오디오 바이트를 텍스트로 바꾼다."""

    @property
    def name(self) -> str:
        """엔진·모델 이름 (보고서용)."""
        ...

    def transcribe(self, audio: bytes, content_type: str) -> Transcription:
        """변환한다. 실패는 SttError 계열로 알린다."""
        ...


def normalize_content_type(value: str | None) -> str:
    """`audio/webm;codecs=opus` 같은 값에서 소문자 MIME만 뽑는다."""
    return (value or "").split(";")[0].strip().lower()
