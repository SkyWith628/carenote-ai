"""DraftPipeline(LangGraph) 테스트: 다중 검사기, 결정 재개, API와 그래프의 일관성. API 호출 없음."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from carenote.drafts.api import get_pipeline, get_store
from carenote.drafts.models import DecisionRequest, DraftStatus
from carenote.drafts.pipeline import DraftPipeline
from carenote.drafts.store import InMemoryDraftStore
from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import Evidence, FieldKey, Source
from carenote.extraction.flagger import FlaggingExtractor
from carenote.extraction.sentinel import load_rules
from carenote.extraction.verifier import FieldVerdict, VerdictStatus, VerifyingExtractor
from carenote.main import create_app

TRANSCRIPT = "어르신 혈압 130에 80이고 약은 드셨어요. 기분은 좋으셨어요."


class Inner:
    def extract(self, transcript: str) -> list[PredictedField]:
        return [
            PredictedField(
                key=FieldKey.BLOOD_PRESSURE,
                value="130/80",
                source=Source.STATED,
                evidence=(Evidence(text="혈압 130에 80"),),
            ),
            PredictedField(key=FieldKey.MOOD, value="좋음", source=Source.STATED),
        ]


class Checker:
    """항상 같은 판정을 내는 가짜 검사기."""

    def __init__(self, status: VerdictStatus) -> None:
        self.status: VerdictStatus = status
        self.calls = 0

    def verify(self, transcript: str, field_: PredictedField) -> FieldVerdict:
        self.calls += 1
        return FieldVerdict(key=field_.key, status=self.status)

    def usage_summary(self) -> dict[str, object]:
        return {}


def _extractor(**checkers: Checker) -> VerifyingExtractor:
    return VerifyingExtractor(FlaggingExtractor(Inner(), load_rules()), dict(checkers))


def _pipeline(extractor: VerifyingExtractor | FlaggingExtractor) -> DraftPipeline:
    return DraftPipeline(extractor, InMemorySaver())


def _waiting(pipeline: DraftPipeline, draft_id: str) -> bool:
    return pipeline.is_waiting(draft_id)


def _unresolved(draft: dict[str, Any]) -> list[str]:
    """flag가 ok가 아닌 필드 키 (승인하려면 고치거나 확인해야 한다)."""
    return [f["key"] for f in draft["fields"] if f["flag"] != "ok"]


def test_each_checker_runs_on_each_risk_field_and_is_recorded() -> None:
    a, b = Checker("supported"), Checker("supported")
    draft = _pipeline(_extractor(verifier=a, relistener=b)).create(TRANSCRIPT)
    assert (a.calls, b.calls) == (1, 1)  # 위험 필드는 혈압 하나, 기분은 검증하지 않는다
    bp = next(f for f in draft.fields if f.key is FieldKey.BLOOD_PRESSURE)
    assert bp.checks == {"verifier": "supported", "relistener": "supported"}
    assert bp.flag.value == "ok"


def test_any_checker_can_raise_the_flag() -> None:
    extractor = _extractor(verifier=Checker("supported"), relistener=Checker("unsupported"))
    draft = _pipeline(extractor).create(TRANSCRIPT)
    bp = next(f for f in draft.fields if f.key is FieldKey.BLOOD_PRESSURE)
    assert bp.flag.value == "needs_confirm"
    assert bp.checks["relistener"] == "unsupported"


def test_plain_extractor_has_no_checks() -> None:
    draft = _pipeline(FlaggingExtractor(Inner(), load_rules())).create(TRANSCRIPT)
    assert all(f.checks == {} for f in draft.fields)
    assert draft.status is DraftStatus.IN_REVIEW and draft.revision == 1


def test_graph_waits_until_resumed_and_matches_decision() -> None:
    pipeline = _pipeline(_extractor(verifier=Checker("supported")))
    draft = pipeline.create(TRANSCRIPT)
    assert _waiting(pipeline, draft.id)
    assert pipeline.resume(draft.id, DecisionRequest(decision="approve")) == "approved"
    assert not _waiting(pipeline, draft.id)


def test_reject_resumes_graph_as_rejected() -> None:
    pipeline = _pipeline(_extractor(verifier=Checker("supported")))
    draft = pipeline.create(TRANSCRIPT)
    assert pipeline.resume(draft.id, DecisionRequest(decision="reject")) == "rejected"


def test_resume_without_waiting_graph_returns_none() -> None:
    pipeline = _pipeline(_extractor(verifier=Checker("supported")))
    assert pipeline.resume("missing", DecisionRequest(decision="approve")) is None


# -- API: 규칙은 decide()가, 재개는 통과한 결정만 ----------------------------------


@pytest.fixture
def api() -> Iterator[tuple[TestClient, DraftPipeline]]:
    app = create_app()
    store = InMemoryDraftStore()
    pipeline = _pipeline(_extractor(verifier=Checker("unsupported")))
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as client:
        yield client, pipeline


def test_invalid_decision_keeps_graph_waiting_until_a_valid_one(
    api: tuple[TestClient, DraftPipeline],
) -> None:
    client, pipeline = api
    draft = client.post("/drafts", json={"transcript": TRANSCRIPT}).json()
    path = f"/drafts/{draft['id']}/decision"
    # 혈압이 needs_confirm이라 처리하지 않은 승인은 422이고 그래프는 계속 기다린다
    assert client.post(path, json={"decision": "approve"}).status_code == 422
    assert _waiting(pipeline, draft["id"])
    ok = client.post(path, json={"decision": "approve", "confirmed": _unresolved(draft)})
    assert ok.status_code == 200 and ok.json()["status"] == "approved"
    assert not _waiting(pipeline, draft["id"])
    assert client.post(path, json={"decision": "reject"}).status_code == 409


def test_edit_via_api_creates_revision_and_finishes_graph(
    api: tuple[TestClient, DraftPipeline],
) -> None:
    client, pipeline = api
    draft = client.post("/drafts", json={"transcript": TRANSCRIPT}).json()
    others = [k for k in _unresolved(draft) if k != "blood_pressure"]
    res = client.post(
        f"/drafts/{draft['id']}/decision",
        json={
            "decision": "approve",
            "edits": {"blood_pressure": "140/90"},
            "confirmed": others,
        },
    )
    body = res.json()
    assert res.status_code == 200 and body["revision"] == 2
    assert body["revisions"][-1]["author"] == "user"
    assert not _waiting(pipeline, draft["id"])


def test_reject_via_api_finishes_graph(api: tuple[TestClient, DraftPipeline]) -> None:
    client, pipeline = api
    draft = client.post("/drafts", json={"transcript": TRANSCRIPT}).json()
    res = client.post(f"/drafts/{draft['id']}/decision", json={"decision": "reject"})
    assert res.json()["status"] == "rejected"
    assert not _waiting(pipeline, draft["id"])
