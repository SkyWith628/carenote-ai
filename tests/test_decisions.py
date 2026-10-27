"""POST /drafts/{id}/decision: 승인·거절·수정, revision, 감사 기록, 동시 결정. API 호출 없음."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi.testclient import TestClient

from carenote.drafts.api import get_extractor, get_store
from carenote.drafts.models import DecisionRequest, DraftStatus
from carenote.drafts.service import DecisionError, create_draft, decide
from carenote.drafts.store import InMemoryDraftStore
from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import Evidence, FieldKey, Source
from carenote.extraction.flagger import FlaggingExtractor
from carenote.extraction.sentinel import load_rules
from carenote.main import create_app

TRANSCRIPT = "어르신 혈압 130에 80이고 약은 드셨어요. 기분은 좋으셨어요."


class FakeInner:
    """혈압·복약·기분만 낸다. 맥박·낙상(위험 필드)은 빠져 needs_confirm이 된다."""

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
                evidence=(Evidence(text="약은 드셨어요"),),
            ),
            PredictedField(key=FieldKey.MOOD, value="좋음", source=Source.STATED),
        ]


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    store = InMemoryDraftStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_extractor] = lambda: FlaggingExtractor(FakeInner(), load_rules())
    with TestClient(app) as c:
        yield c


def new_draft(client: TestClient) -> dict[str, Any]:
    body = client.post("/drafts", json={"transcript": TRANSCRIPT}).json()
    assert body["status"] == "in_review" and body["revision"] == 1
    return body


def unresolved(draft: dict[str, Any]) -> list[str]:
    return [f["key"] for f in draft["fields"] if f["flag"] != "ok"]


def decision(client: TestClient, draft_id: str, **body: Any) -> Any:
    return client.post(f"/drafts/{draft_id}/decision", json=body)


# -- 초기 상태 ----------------------------------------------------------------------


def test_new_draft_has_revision_one_by_ai_and_created_audit(client: TestClient) -> None:
    draft = new_draft(client)
    assert [(r["rev"], r["author"]) for r in draft["revisions"]] == [(1, "ai")]
    assert [a["action"] for a in draft["audit"]] == ["created"]


# -- 승인 ----------------------------------------------------------------------------


def test_approve_requires_every_non_ok_field_to_be_handled(client: TestClient) -> None:
    draft = new_draft(client)
    assert {"pulse", "fall_event"} <= set(unresolved(draft))  # 빠진 위험 필드
    res = decision(client, draft["id"], decision="approve")
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert set(detail["unresolved"]) == set(unresolved(draft))
    # 거부된 요청은 아무것도 바꾸지 않는다
    after = client.get(f"/drafts/{draft['id']}").json()
    assert after["status"] == "in_review" and len(after["audit"]) == 1


def test_approve_with_confirmed_only_keeps_values_and_revision(client: TestClient) -> None:
    draft = new_draft(client)
    res = decision(client, draft["id"], decision="approve", confirmed=unresolved(draft))
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "approved" and body["revision"] == 1
    assert body["fields"] == draft["fields"]  # 확인만 했으므로 값은 그대로
    assert [a["action"] for a in body["audit"]] == ["created", "approve"]
    assert len(body["revisions"]) == 1


def test_approve_with_edit_adds_user_revision_with_diff(client: TestClient) -> None:
    draft = new_draft(client)
    others = [k for k in unresolved(draft) if k != "pulse"]
    res = decision(client, draft["id"], decision="approve", edits={"pulse": "72"}, confirmed=others)
    assert res.status_code == 200
    body = res.json()
    assert body["revision"] == 2 and body["status"] == "approved"
    last = body["revisions"][-1]
    assert (last["rev"], last["author"]) == (2, "user")
    assert last["diff"] == {"pulse": {"from": None, "to": "72"}}
    pulse = next(f for f in body["fields"] if f["key"] == "pulse")
    assert pulse["value"] == "72" and pulse["flag"] == "ok" and pulse["edited"] is True
    assert pulse["evidence"] == [] and pulse["candidates"] == [] and pulse["checks"] == {}
    # revision 1(AI)은 그대로 보존된다 (append-only)
    assert body["revisions"][0] == draft["revisions"][0]
    expected = f"edited=[pulse] confirmed=[{','.join(sorted(others))}]"
    assert body["audit"][-1]["detail"] == expected


def test_editing_an_ok_field_is_allowed_and_recorded(client: TestClient) -> None:
    draft = new_draft(client)
    res = decision(
        client,
        draft["id"],
        decision="approve",
        edits={"mood": "우울"},
        confirmed=unresolved(draft),
    )
    body = res.json()
    mood = next(f for f in body["fields"] if f["key"] == "mood")
    assert mood["value"] == "우울" and mood["edited"] is True
    assert body["revisions"][-1]["diff"]["mood"] == {"from": "좋음", "to": "우울"}


def test_edit_equal_to_current_value_is_not_a_change(client: TestClient) -> None:
    draft = new_draft(client)
    res = decision(
        client,
        draft["id"],
        decision="approve",
        edits={"mood": "좋음"},
        confirmed=unresolved(draft),
    )
    body = res.json()
    assert body["revision"] == 1 and len(body["revisions"]) == 1
    assert not next(f for f in body["fields"] if f["key"] == "mood")["edited"]


# -- 거절 ----------------------------------------------------------------------------


def test_reject_needs_no_field_handling_and_keeps_fields(client: TestClient) -> None:
    draft = new_draft(client)
    res = decision(client, draft["id"], decision="reject", reason="녹음 상태 불량")
    body = res.json()
    assert res.status_code == 200 and body["status"] == "rejected"
    assert body["fields"] == draft["fields"] and body["revision"] == 1
    assert body["audit"][-1]["action"] == "reject"
    assert "녹음" not in str(body["audit"])  # 사유 내용은 감사 기록에 담지 않는다


# -- 거부 / 오류 ---------------------------------------------------------------------


@pytest.mark.parametrize("first", ["approve", "reject"])
def test_second_decision_is_409_and_changes_nothing(client: TestClient, first: str) -> None:
    draft = new_draft(client)
    body: dict[str, Any] = {"decision": first}
    if first == "approve":
        body["confirmed"] = unresolved(draft)
    decision(client, draft["id"], **body)
    before = client.get(f"/drafts/{draft['id']}").json()
    for again in ("approve", "reject"):
        assert decision(client, draft["id"], decision=again).status_code == 409
    assert client.get(f"/drafts/{draft['id']}").json() == before


def test_unknown_draft_is_404(client: TestClient) -> None:
    assert decision(client, "nope", decision="reject").status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"decision": "maybe"},
        {"decision": "approve", "edits": {"not_a_field": "x"}},
        {"decision": "approve", "edits": {"pulse": ""}},
        {"decision": "approve", "edits": {"pulse": "   "}},
        {"decision": "approve", "edits": {"pulse": "9" * 201}},
        {"decision": "approve", "confirmed": ["not_a_field"]},
        {"decision": "reject", "reason": "가" * 501},
        {"decision": "reject", "unexpected": 1},
    ],
)
def test_invalid_body_is_422_and_leaves_draft_alone(
    client: TestClient, payload: dict[str, Any]
) -> None:
    draft = new_draft(client)
    assert decision(client, draft["id"], **payload).status_code == 422
    assert client.get(f"/drafts/{draft['id']}").json()["status"] == "in_review"


def test_edit_value_is_whitespace_trimmed(client: TestClient) -> None:
    draft = new_draft(client)
    res = decision(
        client,
        draft["id"],
        decision="approve",
        edits={"pulse": "  72  "},
        confirmed=[k for k in unresolved(draft) if k != "pulse"],
    )
    assert next(f for f in res.json()["fields"] if f["key"] == "pulse")["value"] == "72"


# -- 동시성 / 순수 함수 --------------------------------------------------------------


def test_concurrent_decisions_apply_exactly_once(client: TestClient) -> None:
    draft = new_draft(client)
    body = {"decision": "approve", "confirmed": unresolved(draft)}

    def hit(_: int) -> int:
        return client.post(f"/drafts/{draft['id']}/decision", json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = sorted(pool.map(hit, range(8)))
    assert codes == [200] + [409] * 7
    final = client.get(f"/drafts/{draft['id']}").json()
    assert [a["action"] for a in final["audit"]].count("approve") == 1


def test_decide_does_not_mutate_its_input() -> None:
    draft = create_draft(TRANSCRIPT, FlaggingExtractor(FakeInner(), load_rules()))
    snapshot = draft.model_dump()
    request = DecisionRequest.model_validate(
        {"decision": "approve", "confirmed": [f.key.value for f in draft.fields if f.flag != "ok"]}
    )
    updated = decide(draft, request)
    assert draft.model_dump() == snapshot and updated.status is DraftStatus.APPROVED
    with pytest.raises(DecisionError) as err:
        decide(updated, request)
    assert err.value.conflict


def test_review_page_has_decision_controls(client: TestClient) -> None:
    html = client.get("/review").text
    assert "/decision" in html and "승인" in html and "거절" in html
