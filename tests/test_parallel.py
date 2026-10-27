"""병렬 실행 테스트: 호출이 실제로 겹쳐 돌고, 결과·통계가 순차와 같은지 확인한다."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Barrier, Lock
from typing import cast

from fastapi.testclient import TestClient

from carenote.drafts.api import get_extractor, get_store
from carenote.drafts.store import InMemoryDraftStore
from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import FieldKey, Flag, Source
from carenote.extraction.flagger import FlaggingExtractor
from carenote.extraction.llm_extractor import ParseClient, ParseResponse
from carenote.extraction.relistener import LLMReadback, TextRelistener
from carenote.extraction.sentinel import load_rules
from carenote.extraction.sentinel_llm import AssistResult, SentinelAssist
from carenote.extraction.verifier import FieldVerdict, LLMVerdict, Verifier, VerifyingExtractor
from carenote.main import create_app

TEXT = "혈압은 130에 80이고요. 맥박은 72예요. 약은 드셨어요. 넘어지신 적은 없어요."


def _fields() -> list[PredictedField]:
    values = {
        FieldKey.BLOOD_PRESSURE: "130/80",
        FieldKey.PULSE: "72",
        FieldKey.MEDICATION_TAKEN: "복용",
        FieldKey.FALL_EVENT: "없음",
    }
    return [PredictedField(key=k, value=v, source=Source.STATED) for k, v in values.items()]


class Inner:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def extract(self, transcript: str) -> list[PredictedField]:
        time.sleep(self.delay)
        return _fields()


class SlowChecker:
    """호출마다 delay만큼 걸리고, 동시에 몇 개가 실행됐는지 센다."""

    def __init__(self, status: str = "supported", delay: float = 0.15) -> None:
        self.status = status
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._lock = Lock()

    def verify(self, transcript: str, field_: PredictedField) -> FieldVerdict:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        return FieldVerdict(field_.key, self.status)  # type: ignore[arg-type]

    def usage_summary(self) -> dict[str, object]:
        return {"calls": 0}


def test_all_checkers_and_fields_run_concurrently() -> None:
    a, b = SlowChecker(), SlowChecker()
    ext = VerifyingExtractor(Inner(), {"verifier": a, "relistener": b})
    started = time.perf_counter()
    ext.extract(TEXT)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.15 * 8 * 0.5  # 순차(8회 × 0.15초 = 1.2초)의 절반보다 훨씬 빠르다
    assert a.max_active + b.max_active > 2  # 실제로 겹쳤다


def test_checker_verdicts_are_merged_with_raise_only_semantics() -> None:
    ext = VerifyingExtractor(
        Inner(),
        {"verifier": SlowChecker("supported", 0), "relistener": SlowChecker("ambiguous", 0)},
    )
    flags = {f.key: f.flag for f in ext.extract(TEXT)}
    assert all(
        flag is Flag.NEEDS_CONFIRM
        for key, flag in flags.items()
        if key
        in (FieldKey.BLOOD_PRESSURE, FieldKey.PULSE, FieldKey.MEDICATION_TAKEN, FieldKey.FALL_EVENT)
    )
    verdicts = ext.all_verdicts_for(TEXT)
    assert set(verdicts) == {"verifier", "relistener"}
    assert ext.verdicts_for(TEXT)[FieldKey.PULSE].status == "supported"  # 첫 검사기
    assert ext.verdicts_for(TEXT, "relistener")[FieldKey.PULSE].status == "ambiguous"
    usage = ext.usage_summary()
    assert "verifier" in usage and "relistener" in usage


def test_single_checker_still_works_with_name() -> None:
    ext = VerifyingExtractor(Inner(), SlowChecker("unsupported", 0), name="relistener")
    assert ext.extract(TEXT)[0].flag is Flag.NEEDS_CONFIRM
    assert set(ext.all_verdicts_for(TEXT)) == {"relistener"}


class SlowAssist:
    """Sentinel LLM 보조 대역: delay만큼 걸리고 신호는 없다."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def assess(self, transcript: str) -> AssistResult:
        time.sleep(self.delay)
        return AssistResult(status="ok")


def test_extractor_and_sentinel_assist_overlap() -> None:
    flagging = FlaggingExtractor(Inner(0.2), load_rules(), cast(SentinelAssist, SlowAssist(0.2)))
    started = time.perf_counter()
    result = flagging.extract(TEXT)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.36  # 순차면 0.4초 이상
    assert len(result) == len(FieldKey)  # 모든 필드, 결과는 순차와 같다


def test_failure_in_one_branch_propagates() -> None:
    class Boom:
        def extract(self, transcript: str) -> list[PredictedField]:
            raise RuntimeError("boom")

    flagging = FlaggingExtractor(Boom(), load_rules(), cast(SentinelAssist, SlowAssist(0.0)))
    try:
        flagging.extract(TEXT)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("예외가 전파돼야 한다")


# -- 통계 잠금 -------------------------------------------------------------------


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 10
    output_tokens: int = 5
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class _Client:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.messages = self

    def parse(self, **_: object) -> ParseResponse:
        return cast(ParseResponse, _Response(self.outcome))


def test_stats_are_exact_under_heavy_concurrency() -> None:
    verifier = Verifier(
        cast(ParseClient, _Client(LLMVerdict(verdict="unsupported", quote=""))),
        model="claude-haiku-5-5",
        effort="low",
    )
    relistener = TextRelistener(
        cast(ParseClient, _Client(LLMReadback(value="", quote="", conflict=True))),
        model="claude-haiku-5-5",
        effort="low",
    )
    field_ = _fields()[0]
    barrier = Barrier(32)

    def hit(i: int) -> None:
        barrier.wait()
        for _ in range(50):
            (verifier if i % 2 else relistener).verify(TEXT, field_)

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(hit, range(32)))
    assert verifier.stats.calls == 16 * 50 and verifier.stats.input_tokens == 16 * 50 * 10
    assert relistener.stats.calls == 16 * 50 and relistener.stats.output_tokens == 16 * 50 * 5


# -- API ---------------------------------------------------------------------------


def test_api_exposes_both_checks_per_risk_field() -> None:
    app = create_app()
    app.dependency_overrides[get_store] = lambda: InMemoryDraftStore()
    app.dependency_overrides[get_extractor] = lambda: VerifyingExtractor(
        FlaggingExtractor(Inner(), load_rules()),
        {"verifier": SlowChecker("supported", 0), "relistener": SlowChecker("unsupported", 0)},
    )
    with TestClient(app) as client:
        body = client.post("/drafts", json={"transcript": TEXT}).json()
        by_key = {f["key"]: f for f in body["fields"]}
        assert by_key["blood_pressure"]["checks"] == {
            "verifier": "supported",
            "relistener": "unsupported",
        }
        assert by_key["blood_pressure"]["flag"] == "needs_confirm"
        assert by_key["mood"]["checks"] == {}
        assert "독립 재읽기" in client.get("/review").text
