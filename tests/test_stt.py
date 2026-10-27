"""STT: 엔진(가짜 모델), /transcribe 엔드포인트, 개인정보 마스킹. 실제 모델·오디오는 쓰지 않는다."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, BinaryIO

import pytest
from fastapi.testclient import TestClient

from carenote.drafts.api import get_extractor, get_store
from carenote.drafts.store import InMemoryDraftStore
from carenote.evaluation.extractor import PredictedField
from carenote.extraction.flagger import FlaggingExtractor
from carenote.extraction.sentinel import load_rules
from carenote.main import create_app
from carenote.privacy import mask_pii
from carenote.stt.api import get_stt_engine
from carenote.stt.engine import SttBusy, SttError, SttUnavailable, Transcription
from carenote.stt.whisper_engine import FasterWhisperEngine

# -- 개인정보 마스킹 -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected", "kind"),
    [
        ("주민번호 900101-1234567 입니다", "주민번호 [주민번호] 입니다", "rrn"),
        ("9001011234567", "[주민번호]", "rrn"),
        ("전화는 010-1234-5678이에요", "전화는 [전화번호]이에요", "phone"),
        ("01012345678", "[전화번호]", "phone"),
        ("010 1234 5678", "[전화번호]", "phone"),
        ("인정번호 L2012345678", "인정번호 [인정번호]", "ltc_no"),
    ],
)
def test_masks_pii_patterns(text: str, expected: str, kind: str) -> None:
    result = mask_pii(text)
    assert result.text == expected and result.counts == {kind: 1} and result.total == 1


@pytest.mark.parametrize(
    "text", ["혈압 130에 80, 맥박 72", "2026년 10월 28일 3시", "체온 36.5도", "123456 7890123456"]
)
def test_does_not_touch_clinical_numbers(text: str) -> None:
    assert mask_pii(text).text == text


def test_mask_counts_multiple_and_keeps_no_values() -> None:
    result = mask_pii("010-1111-2222 그리고 010-3333-4444, 900101-2345678")
    assert result.counts == {"rrn": 1, "phone": 2}
    assert "1111" not in str(result.counts)


# -- 엔진 ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Seg:
    text: str


@dataclass(frozen=True)
class _Info:
    duration: float


class FakeModel:
    def __init__(
        self,
        segments: list[str] | None = None,
        duration: float = 5.0,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.segments = (
            segments if segments is not None else ["혈압은 130에 80이고요", "약은 드셨어요"]
        )
        self.duration = duration
        self.error = error
        self.delay = delay
        self.calls: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def transcribe(
        self,
        audio: BinaryIO,
        *,
        language: str,
        beam_size: int,
        vad_filter: bool,
        condition_on_previous_text: bool,
    ) -> tuple[list[_Seg], _Info]:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append(
                {
                    "bytes": audio.read(),
                    "language": language,
                    "vad": vad_filter,
                    "cond": condition_on_previous_text,
                }
            )
            time.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return [_Seg(s) for s in self.segments], _Info(self.duration)
        finally:
            with self._lock:
                self.active -= 1


def engine_with(model: FakeModel, **kw: Any) -> FasterWhisperEngine:
    return FasterWhisperEngine(model_factory=lambda: model, **kw)


def test_engine_joins_segments_and_passes_audio_in_memory() -> None:
    model = FakeModel()
    result = engine_with(model).transcribe(b"AUDIO", "audio/webm")
    assert result.text == "혈압은 130에 80이고요 약은 드셨어요"
    assert (result.language, result.duration_s) == ("ko", 5.0)
    assert result.engine == "faster-whisper:large-v3-turbo"
    call = model.calls[0]
    assert call["bytes"] == b"AUDIO" and call["language"] == "ko"
    assert call["vad"] is True and call["cond"] is False  # 환각 억제 설정


def test_engine_loads_model_once() -> None:
    loads: list[int] = []
    model = FakeModel()

    def factory() -> FakeModel:
        loads.append(1)
        return model

    engine = FasterWhisperEngine(model_factory=factory)
    engine.transcribe(b"a", "audio/webm")
    engine.transcribe(b"b", "audio/webm")
    assert len(loads) == 1


def test_engine_rejects_too_long_audio_before_decoding() -> None:
    model = FakeModel(duration=300.0)
    with pytest.raises(SttError, match="너무 길다"):
        engine_with(model, max_seconds=120.0).transcribe(b"a", "audio/webm")


@pytest.mark.parametrize("segments", [[], ["", "  "]])
def test_engine_reports_no_speech(segments: list[str]) -> None:
    with pytest.raises(SttError, match="음성이 감지되지 않았다"):
        engine_with(FakeModel(segments=segments)).transcribe(b"a", "audio/webm")


def test_engine_wraps_decode_errors_without_leaking_details() -> None:
    model = FakeModel(error=RuntimeError("secret transcript body"))
    with pytest.raises(SttError) as err:
        engine_with(model).transcribe(b"a", "audio/webm")
    assert "secret" not in str(err.value) and "RuntimeError" in str(err.value)


def test_engine_factory_failure_is_unavailable() -> None:
    def factory() -> FakeModel:
        raise SttUnavailable("모델 없음")

    with pytest.raises(SttUnavailable):
        FasterWhisperEngine(model_factory=factory).transcribe(b"a", "audio/webm")


def test_engine_serializes_requests_and_times_out_when_busy() -> None:
    model = FakeModel(delay=0.3)
    engine = engine_with(model, busy_timeout_s=0.05)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(engine.transcribe, b"a", "audio/webm")
        time.sleep(0.1)
        second = pool.submit(engine.transcribe, b"b", "audio/webm")
        assert first.result().text
        with pytest.raises(SttBusy):
            second.result()
    assert model.max_active == 1  # 모델은 한 번에 한 요청만 쓴다


def test_default_factory_reports_missing_package_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", None)  # import 실패를 모사
    with pytest.raises(SttUnavailable, match="faster-whisper"):
        FasterWhisperEngine().transcribe(b"a", "audio/webm")


# -- 엔드포인트 ----------------------------------------------------------------------


class StubEngine:
    name = "stub"

    def __init__(self, text: str = "혈압은 130에 80이에요", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.seen: list[tuple[int, str]] = []

    def transcribe(self, audio: bytes, content_type: str) -> Transcription:
        self.seen.append((len(audio), content_type))
        if self.error is not None:
            raise self.error
        return Transcription(self.text, 4.2, "ko", self.name, 0.5)


@pytest.fixture
def stub() -> StubEngine:
    return StubEngine()


@pytest.fixture
def client(stub: StubEngine) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_stt_engine] = lambda: stub
    with TestClient(app) as c:
        yield c


def post_audio(client: TestClient, body: bytes = b"x" * 100, ctype: str = "audio/webm") -> Any:
    return client.post("/transcribe", content=body, headers={"Content-Type": ctype})


def test_transcribe_returns_text_and_metadata_without_caching(
    client: TestClient, stub: StubEngine
) -> None:
    res = post_audio(client, ctype="audio/webm;codecs=opus")
    assert res.status_code == 200
    assert res.json() == {
        "text": "혈압은 130에 80이에요",
        "duration_s": 4.2,
        "language": "ko",
        "engine": "stub",
        "elapsed_s": 0.5,
        "masked_count": 0,
    }
    assert res.headers["cache-control"] == "no-store"
    assert stub.seen == [(100, "audio/webm")]  # 코덱 파라미터는 떼고 전달


def test_transcribe_masks_pii_in_output(client: TestClient, stub: StubEngine) -> None:
    stub.text = "제 번호는 010-1234-5678이에요. 혈압 130에 80"
    body = post_audio(client).json()
    assert "1234" not in body["text"] and "[전화번호]" in body["text"]
    assert body["masked_count"] == 1


@pytest.mark.parametrize("ctype", ["application/json", "text/plain", "", "video/mp4"])
def test_wrong_content_type_is_415(client: TestClient, stub: StubEngine, ctype: str) -> None:
    assert post_audio(client, ctype=ctype).status_code == 415
    assert stub.seen == []


def test_empty_body_is_400(client: TestClient) -> None:
    assert post_audio(client, body=b"").status_code == 400


def test_oversize_is_413_before_engine_runs(client: TestClient, stub: StubEngine) -> None:
    res = post_audio(client, body=b"x" * 4_000_001)
    assert res.status_code == 413 and stub.seen == []


def test_oversize_chunked_body_without_length_is_413(client: TestClient, stub: StubEngine) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(5):
            yield b"x" * 1_000_000

    res = client.post("/transcribe", content=chunks(), headers={"Content-Type": "audio/webm"})
    assert res.status_code == 413 and stub.seen == []


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (SttError("음성이 감지되지 않았다"), 422),
        (SttUnavailable("엔진 없음"), 503),
        (SttBusy("바쁨"), 503),
    ],
)
def test_engine_errors_map_to_statuses(
    client: TestClient, stub: StubEngine, error: Exception, status: int
) -> None:
    stub.error = error
    res = post_audio(client)
    assert res.status_code == status
    if isinstance(error, SttBusy):
        assert res.headers["retry-after"] == "10"


# -- 초안 생성 경로의 마스킹 ---------------------------------------------------------


class _Inner:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def extract(self, transcript: str) -> list[PredictedField]:
        self.seen.append(transcript)
        return []


def test_drafts_mask_pii_before_extraction_and_record_count_only() -> None:
    app = create_app()
    inner = _Inner()
    app.dependency_overrides[get_store] = lambda: InMemoryDraftStore()
    app.dependency_overrides[get_extractor] = lambda: FlaggingExtractor(inner, load_rules())
    with TestClient(app) as c:
        body = c.post("/drafts", json={"transcript": "번호 010-1234-5678. 혈압 130에 80"}).json()
    assert inner.seen == ["번호 [전화번호]. 혈압 130에 80"]  # 추출기는 마스킹된 글만 본다
    assert "1234" not in body["transcript"]
    assert body["audit"][0]["detail"] == "masked=1"  # 감사에는 횟수만, 값은 없다


def test_review_page_has_recording_controls(client: TestClient) -> None:
    html = client.get("/review").text
    assert "/transcribe" in html and "녹음 시작" in html and "MediaRecorder" in html
