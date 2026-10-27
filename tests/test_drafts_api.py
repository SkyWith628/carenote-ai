"""POST /drafts · GET /drafts: 가짜 extractor로 API 계약을 검증한다 (API 호출 없음)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from carenote.drafts.api import get_extractor, get_store
from carenote.drafts.store import InMemoryDraftStore
from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import Evidence, FieldKey, Source
from carenote.extraction.flagger import FlaggingExtractor
from carenote.extraction.llm_extractor import ParseClient, ParseResponse
from carenote.extraction.sentinel import load_rules
from carenote.extraction.verifier import LLMVerdict, Verifier, VerifyingExtractor
from carenote.main import create_app

TRANSCRIPT = "어르신 혈압 130에 80이고 약은 드셨어요. 어제 가슴이 답답하다 하셨어요."


class FakeInner:
    """transcript를 보고 정해진 필드를 돌려주는 가짜 추출기."""

    def extract(self, transcript: str) -> list[PredictedField]:
        return [
            PredictedField(
                key=FieldKey.BLOOD_PRESSURE,
                value="130/80",
                source=Source.STATED,
                evidence=(Evidence(text="혈압 130에 80"),),
            ),
            PredictedField(
                key=FieldKey.MEDICATION_TAKEN,
                value="복용",
                source=Source.STATED,
                evidence=(Evidence(text="없는 문장"),),
            ),
        ]


class Boom:
    def extract(self, transcript: str) -> list[PredictedField]:
        raise RuntimeError("secret transcript body must not leak")


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    store = InMemoryDraftStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_extractor] = lambda: FlaggingExtractor(FakeInner(), load_rules())
    with TestClient(app) as c:
        yield c


def test_post_creates_draft_with_all_fields_and_flags(client: TestClient) -> None:
    res = client.post("/drafts", json={"transcript": TRANSCRIPT})
    assert res.status_code == 201
    body = res.json()
    assert body["status"] == "in_review" and body["revision"] == 1
    assert [f["key"] for f in body["fields"]] == [k.value for k in FieldKey]
    by_key = {f["key"]: f for f in body["fields"]}
    assert by_key["blood_pressure"]["flag"] == "ok"
    # 빠진 위험 필드(낙상)는 조용히 ok가 되지 않는다
    assert by_key["fall_event"]["flag"] == "needs_confirm"
    # transcript의 위험 신호가 특이사항을 risk로 올린다
    assert by_key["special_notes"]["flag"] == "risk"
    assert body["signals"] and body["flag_counts"]["risk"] >= 1


def test_evidence_offsets_point_into_transcript(client: TestClient) -> None:
    body = client.post("/drafts", json={"transcript": TRANSCRIPT}).json()
    by_key = {f["key"]: f for f in body["fields"]}
    span = by_key["blood_pressure"]["evidence"][0]
    assert TRANSCRIPT[span["start"] : span["end"]] == "혈압 130에 80"
    missing = by_key["medication_taken"]["evidence"][0]
    assert missing["start"] is None and missing["end"] is None


def test_get_by_id_and_list(client: TestClient) -> None:
    created = client.post("/drafts", json={"transcript": TRANSCRIPT}).json()
    assert client.get(f"/drafts/{created['id']}").json() == created
    listing = client.get("/drafts").json()
    assert [d["id"] for d in listing] == [created["id"]]
    assert "transcript" not in listing[0]


def test_unknown_id_is_404(client: TestClient) -> None:
    assert client.get("/drafts/nope").status_code == 404


@pytest.mark.parametrize("payload", [{}, {"transcript": ""}, {"transcript": "가" * 20_001}])
def test_invalid_body_is_422(client: TestClient, payload: dict[str, str]) -> None:
    assert client.post("/drafts", json=payload).status_code == 422


def test_extractor_failure_is_502_without_leaking_body(client: TestClient) -> None:
    client.app.dependency_overrides[get_extractor] = lambda: Boom()  # type: ignore[attr-defined]
    res = client.post("/drafts", json={"transcript": TRANSCRIPT})
    assert res.status_code == 502
    assert "secret" not in res.text and TRANSCRIPT not in res.text
    assert client.get("/drafts").json() == []


def test_missing_api_key_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_extractor.cache_clear()
    app = create_app()
    app.dependency_overrides[get_store] = lambda: InMemoryDraftStore()
    from carenote.config import Settings

    monkeypatch.setattr(
        "carenote.drafts.api.get_settings", lambda: Settings.model_construct(anthropic_api_key=None)
    )
    res = TestClient(app).post("/drafts", json={"transcript": TRANSCRIPT})
    assert res.status_code == 503


def test_review_page_is_served(client: TestClient) -> None:
    res = client.get("/review")
    assert res.status_code == 200 and "초안 검토" in res.text


# -- Verifier 연결 -----------------------------------------------------------------


class _VerdictClient:
    """검증 요청에 정해진 판정을 돌려주거나 예외를 던진다."""

    def __init__(self, outcome: LLMVerdict | Exception) -> None:
        self.outcome = outcome
        self.messages = self

    def parse(self, **_: object) -> ParseResponse:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return cast(ParseResponse, _VerdictResponse(self.outcome))


@dataclass(frozen=True)
class _VerdictUsage:
    input_tokens: int = 100
    output_tokens: int = 10
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _VerdictResponse:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _VerdictUsage = field(default_factory=_VerdictUsage)


def _with_verifier(client: TestClient, outcome: LLMVerdict | Exception) -> None:
    verifier = Verifier(
        cast(ParseClient, _VerdictClient(outcome)), model="claude-haiku-5-5", effort="low"
    )
    client.app.dependency_overrides[get_extractor] = lambda: VerifyingExtractor(  # type: ignore[attr-defined]
        FlaggingExtractor(FakeInner(), load_rules()), verifier
    )


def test_verifier_result_is_exposed_and_raises_flag(client: TestClient) -> None:
    _with_verifier(client, LLMVerdict(verdict="unsupported", quote=""))
    body = client.post("/drafts", json={"transcript": TRANSCRIPT}).json()
    by_key = {f["key"]: f for f in body["fields"]}
    assert by_key["blood_pressure"]["checks"] == {"verifier": "unsupported"}
    assert by_key["blood_pressure"]["flag"] == "needs_confirm"
    assert by_key["mood"]["checks"] == {}  # 위험 필드가 아니면 검증하지 않는다
    assert body["signals"]  # Sentinel 신호는 Verifier를 거쳐도 전달된다


def test_verifier_supported_keeps_flag(client: TestClient) -> None:
    _with_verifier(client, LLMVerdict(verdict="supported", quote="혈압 130에 80"))
    by_key = {
        f["key"]: f
        for f in client.post("/drafts", json={"transcript": TRANSCRIPT}).json()["fields"]
    }
    assert by_key["blood_pressure"]["checks"] == {"verifier": "supported"}
    assert by_key["blood_pressure"]["flag"] == "ok"


def test_verifier_failure_is_shown_as_unavailable_not_ok(client: TestClient) -> None:
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))
    _with_verifier(client, error)
    res = client.post("/drafts", json={"transcript": TRANSCRIPT})
    assert res.status_code == 201  # 검증 실패가 초안 생성을 막지 않는다
    by_key = {f["key"]: f for f in res.json()["fields"]}
    assert by_key["blood_pressure"]["checks"] == {"verifier": "unavailable"}
    assert by_key["blood_pressure"]["flag"] == "ok"


def test_review_page_shows_verifier_check(client: TestClient) -> None:
    assert "재검증" in client.get("/review").text
