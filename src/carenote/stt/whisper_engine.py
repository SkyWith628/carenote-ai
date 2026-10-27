"""faster-whisper 엔진 (자체 호스팅). 모델은 첫 요청에서 불러온다.

오디오는 `BytesIO`로 넘겨 디스크에 쓰지 않는다(PyAV가 디코딩). 모델은 한 번에 한 요청만 돌리고
(CPU 데모 기준), 다른 요청은 `busy_timeout_s`까지 기다린다. 침묵·잡음에서 생기는 환각을 줄이려고
VAD 필터를 켜고 `condition_on_previous_text`를 끈다.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Iterable
from threading import Lock
from typing import BinaryIO, Protocol

from carenote.stt.engine import SttBusy, SttError, SttUnavailable, Transcription


class SegmentLike(Protocol):
    """faster-whisper Segment의 부분집합."""

    @property
    def text(self) -> str: ...


class InfoLike(Protocol):
    """faster-whisper TranscriptionInfo의 부분집합."""

    @property
    def duration(self) -> float: ...


class WhisperModelLike(Protocol):
    """faster_whisper.WhisperModel의 부분집합. 테스트에서 가짜로 바꾼다."""

    def transcribe(
        self,
        audio: BinaryIO,
        *,
        language: str,
        beam_size: int,
        vad_filter: bool,
        condition_on_previous_text: bool,
    ) -> tuple[Iterable[SegmentLike], InfoLike]: ...


class FasterWhisperEngine:
    """faster-whisper로 한국어 음성을 받아 적는다."""

    def __init__(
        self,
        *,
        model: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "ko",
        max_seconds: float = 120.0,
        download_root: str | None = None,
        local_files_only: bool = False,
        busy_timeout_s: float = 60.0,
        model_factory: Callable[[], WhisperModelLike] | None = None,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._max_seconds = max_seconds
        self._download_root = download_root
        self._local_files_only = local_files_only
        self._busy_timeout_s = busy_timeout_s
        self._factory = model_factory or self._default_factory
        self._model: WhisperModelLike | None = None
        self._run_lock = Lock()

    @property
    def name(self) -> str:
        """엔진·모델 이름."""
        return f"faster-whisper:{self._model_name}"

    def _default_factory(self) -> WhisperModelLike:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found,unused-ignore]
        except (ImportError, OSError) as exc:  # DLL 차단 등 네이티브 로드 실패도 OSError다
            raise SttUnavailable(
                f"faster-whisper를 불러오지 못했다 ({type(exc).__name__}). "
                "`uv add --optional stt faster-whisper`로 설치한다"
            ) from exc
        try:
            return WhisperModel(  # type: ignore[no-any-return,unused-ignore]
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self._download_root,
                local_files_only=self._local_files_only,
            )
        except Exception as exc:  # 모델 다운로드·로드 실패
            raise SttUnavailable(f"모델을 불러오지 못했다 ({type(exc).__name__})") from exc

    def transcribe(self, audio: bytes, content_type: str) -> Transcription:
        """오디오를 받아 적는다. 같은 시각에는 한 요청만 모델을 쓴다."""
        if not self._run_lock.acquire(timeout=self._busy_timeout_s):
            raise SttBusy("다른 음성을 변환하는 중이다. 잠시 뒤 다시 시도한다")
        started = time.perf_counter()
        try:
            if self._model is None:
                self._model = self._factory()
            try:
                segments, info = self._model.transcribe(
                    io.BytesIO(audio),
                    language=self._language,
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                duration = float(info.duration)
                if duration > self._max_seconds:
                    raise SttError(f"음성이 너무 길다 (최대 {self._max_seconds:.0f}초)")
                parts = [s.text.strip() for s in segments]  # 여기서 실제 디코딩이 일어난다
            except SttError:
                raise
            except Exception as exc:  # 손상·미지원 형식. 메시지에 오디오 내용은 없다
                raise SttError(f"오디오를 읽을 수 없다 ({type(exc).__name__})") from exc
        finally:
            self._run_lock.release()
        text = " ".join(p for p in parts if p)
        if not text:
            raise SttError("음성이 감지되지 않았다")
        return Transcription(
            text=text,
            duration_s=round(duration, 2),
            language=self._language,
            engine=self.name,
            elapsed_s=round(time.perf_counter() - started, 2),
        )
