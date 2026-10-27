"""LangGraph 그래프 테스트. 가짜 추출기·가짜 클라이언트만 쓰며 API를 호출하지 않는다."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, cast

import anthropic
import httpx2
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import Evidence, FieldKey, Source
from carenote.extraction.flagger import FlaggingExtractor
from carenote.extraction.llm_extractor import ParseClient, ParseResponse
from carenote.extraction.sentinel import load_rules
from carenote.extraction.verifier import LLMVerdict, Verifier
from carenote.orchestration.graph import build_graph

TRANSCRIPT = "어르신 혈압 130에 80이고 약은 드셨어요. 기분은 좋으셨어요."


def _bp() -> PredictedField:
    return PredictedField(
        key=FieldKey.BLOOD_PRESSURE,
        value="130/80",
        source=Source.STATED,
        evidence=(Evidence(text="혈압 130에 80"),),
    )


def _med() -> PredictedField:
    return PredictedField(
        key=FieldKey.MEDICATION_TAKEN,
        value="복용",
        source=Source.STATED,
        evidence=(Evidence(text="약은 드셨어요"),),
    )


def _mood() -> PredictedField:
    return PredictedField(key=FieldKey.MOOD, value="좋음", source=Source.STATED)


class Inner:
    def __init__(self, fields: list[PredictedField]) -> None:
        self.fields = fields

    def extract(self, transcript: str) -> list[PredictedField]:
        return list(self.fields)


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 100
    output_tokens: int = 10
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class Client:
    """검증 요청마다 outcome(판정 또는 예외)을 돌려주고 호출 수를 센다(스레드 안전)."""

    def __init__(self, outcome: LLMVerdict | Exception) -> None:
        self.outcome = outcome
        self.calls = 0
        self._lock = Lock()
        self.messages = self

    def parse(self, **_: object) -> ParseResponse:
        with self._lock:
            self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return cast(ParseResponse, _Response(self.outcome))


def make(
    fields: list[PredictedField],
    outcome: LLMVerdict | Exception,
    saver: InMemorySaver | None = None,
) -> tuple[Any, Client, InMemorySaver]:
    client = Client(outcome)
    verifier = Verifier(cast(ParseClient, client), model="claude-haiku-5-5", effort="low")
    saver = saver or InMemorySaver()
    graph = build_graph(FlaggingExtractor(Inner(fields), load_rules()), verifier, saver)
    return graph, client, saver


def cfg(thread: str = "t1") -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}}


def interrupt_payload(result: dict[str, Any]) -> dict[str, Any]:
    (item,) = result["__interrupt__"]
    return cast(dict[str, Any], item.value)


def by_key(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {i["key"]: i for i in items}


UNSUPPORTED = LLMVerdict(verdict="unsupported", quote="")
SUPPORTED = LLMVerdict(verdict="supported", quote="혈압 130에 80")


def test_runs_to_human_review_and_fans_out_per_risk_field() -> None:
    graph, client, _ = make([_bp(), _med(), _mood()], UNSUPPORTED)
    result = graph.invoke({"transcript": TRANSCRIPT}, cfg())
    payload = interrupt_payload(result)
    assert client.calls == 2  # 혈압·복약만 검증한다
    assert set(payload["verdicts"]) == {"blood_pressure", "medication_taken"}
    fields = by_key(payload["fields"])
    assert [f["key"] for f in payload["fields"]] == [k.value for k in FieldKey]
    assert fields["blood_pressure"]["flag"] == "needs_confirm"  # Consensus가 올렸다
    assert fields["mood"]["flag"] == "ok"
    assert "status" not in result  # 사람이 확정하기 전에는 끝나지 않는다


def test_supported_verdict_keeps_flag() -> None:
    graph, _, _ = make([_bp()], SUPPORTED)
    fields = by_key(interrupt_payload(graph.invoke({"transcript": TRANSCRIPT}, cfg()))["fields"])
    assert fields["blood_pressure"]["flag"] == "ok"


def test_no_verification_targets_goes_straight_to_review() -> None:
    graph, client, _ = make([_mood()], UNSUPPORTED)
    payload = interrupt_payload(graph.invoke({"transcript": TRANSCRIPT}, cfg()))
    assert client.calls == 0 and payload["verdicts"] == {}


def test_verifier_failure_still_reaches_review_as_unavailable() -> None:
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))
    graph, _, _ = make([_bp()], error)
    payload = interrupt_payload(graph.invoke({"transcript": TRANSCRIPT}, cfg()))
    assert payload["verdicts"]["blood_pressure"]["status"] == "unavailable"
    assert by_key(payload["fields"])["blood_pressure"]["flag"] == "ok"


def test_approve_finalizes_without_changes() -> None:
    graph, _, _ = make([_bp()], SUPPORTED)
    first = interrupt_payload(graph.invoke({"transcript": TRANSCRIPT}, cfg()))
    final = graph.invoke(Command(resume={"decision": "approve"}), cfg())
    assert final["status"] == "approved"
    assert final["fields"] == first["fields"]


def test_human_edit_replaces_value_and_clears_evidence() -> None:
    graph, _, _ = make([_bp()], UNSUPPORTED)
    graph.invoke({"transcript": TRANSCRIPT}, cfg())
    final = graph.invoke(
        Command(resume={"decision": "approve", "edits": {"blood_pressure": "140/90"}}), cfg()
    )
    bp = by_key(final["fields"])["blood_pressure"]
    assert final["status"] == "approved"
    assert (bp["value"], bp["flag"], bp["evidence"]) == ("140/90", "ok", [])


def test_reject() -> None:
    graph, _, _ = make([_bp()], SUPPORTED)
    graph.invoke({"transcript": TRANSCRIPT}, cfg())
    final = graph.invoke(Command(resume={"decision": "reject"}), cfg())
    assert final["status"] == "rejected"


@pytest.mark.parametrize("answer", [{}, {"decision": "maybe"}])
def test_unknown_decision_is_rejected_not_approved(answer: dict[str, str]) -> None:
    graph, _, _ = make([_bp()], SUPPORTED)
    graph.invoke({"transcript": TRANSCRIPT}, cfg())
    assert graph.invoke(Command(resume=answer), cfg())["status"] == "rejected"


def test_resume_works_from_a_new_graph_sharing_the_checkpointer() -> None:
    graph, _, saver = make([_bp()], SUPPORTED)
    graph.invoke({"transcript": TRANSCRIPT}, cfg("shared"))
    other, client, _ = make([], SUPPORTED, saver)  # 새 인스턴스(재시작 모사): 재추출 없이 이어간다
    final = other.invoke(Command(resume={"decision": "approve"}), cfg("shared"))
    assert final["status"] == "approved" and client.calls == 0
    assert by_key(final["fields"])["blood_pressure"]["value"] == "130/80"


def test_threads_are_isolated() -> None:
    graph, _, _ = make([_bp()], SUPPORTED)
    graph.invoke({"transcript": TRANSCRIPT}, cfg("a"))
    graph.invoke({"transcript": TRANSCRIPT}, cfg("b"))
    assert graph.invoke(Command(resume={"decision": "reject"}), cfg("a"))["status"] == "rejected"
    assert graph.invoke(Command(resume={"decision": "approve"}), cfg("b"))["status"] == "approved"
