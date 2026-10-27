"""초안 생성 그래프.

extract → verify(위험 필드마다 병렬) → consensus → 사람 검토(interrupt) → finalize.

원칙: LLM은 추출·재검증만, flag 판정은 코드(Consensus), 확정은 사람(interrupt).
상태는 체크포인트에 안전하게 저장되도록 JSON 호환 값(dict·str)만 둔다. 체크포인터는 호출자가 준다.
`human_review` 노드는 resume 때 처음부터 다시 실행되므로 interrupt 앞에서 부작용을 만들지 않는다.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send, interrupt

from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import FieldKey, Flag
from carenote.extraction.sentinel import Signal
from carenote.extraction.verifier import (
    FieldVerdict,
    Verifier,
    apply_verdicts,
    needs_verification,
)

Decision = Literal["approve", "reject"]
Status = Literal["approved", "rejected"]


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """병렬 verify 노드의 결과를 키별로 합친다 (리듀서)."""
    return {**left, **right}


class DraftState(TypedDict, total=False):
    """그래프 상태. fields·verdicts·signals는 JSON 호환 dict다."""

    transcript: str
    fields: list[dict[str, Any]]
    signals: list[dict[str, Any]]
    verdicts: Annotated[dict[str, Any], _merge_dicts]
    decision: Decision
    edits: dict[str, str]
    status: Status


class VerifyInput(TypedDict):
    """Send로 verify 노드에 넘기는 입력 (필드 하나)."""

    transcript: str
    field: dict[str, Any]


class GraphExtractor(Protocol):
    """FlaggingExtractor가 만족하는 추출기."""

    def extract(self, transcript: str) -> list[PredictedField]:
        """flag까지 계산된 필드."""
        ...

    def signals_for(self, transcript: str) -> list[Signal]:
        """마지막 extract의 Sentinel 신호."""
        ...


def _signal_dict(signal: Signal) -> dict[str, Any]:
    return {
        "rule_id": signal.rule_id,
        "category": signal.category,
        "quote": signal.quote,
        "severity": signal.severity.value,
        "source": signal.source,
    }


def _field(data: dict[str, Any]) -> PredictedField:
    return PredictedField.model_validate(data)


def build_graph(
    extractor: GraphExtractor,
    verifier: Verifier,
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph[DraftState, None, DraftState, DraftState]:
    """그래프를 만든다. interrupt를 쓰므로 체크포인터가 필수다."""

    def extract(state: DraftState) -> DraftState:
        transcript = state["transcript"]
        fields = extractor.extract(transcript)
        return {
            "fields": [f.model_dump(mode="json") for f in fields],
            "signals": [_signal_dict(s) for s in extractor.signals_for(transcript)],
        }

    def fan_out(state: DraftState) -> list[Send] | str:
        """검증 대상 위험 필드마다 verify를 병렬로 보낸다. 대상이 없으면 바로 consensus."""
        targets = [f for f in map(_field, state["fields"]) if needs_verification(f)]
        if not targets:
            return "consensus"
        return [
            Send(
                "verify",
                VerifyInput(transcript=state["transcript"], field=f.model_dump(mode="json")),
            )
            for f in targets
        ]

    def verify(item: VerifyInput) -> DraftState:
        field = _field(item["field"])
        verdict = verifier.verify(item["transcript"], field)
        return {"verdicts": {field.key.value: asdict(verdict)}}

    def consensus(state: DraftState) -> DraftState:
        verdicts = {
            FieldKey(key): FieldVerdict(
                key=FieldKey(key),
                status=raw["status"],
                quote=raw["quote"],
                reason=raw["reason"],
            )
            for key, raw in state.get("verdicts", {}).items()
        }
        fields = apply_verdicts([_field(f) for f in state["fields"]], verdicts)
        return {"fields": [f.model_dump(mode="json") for f in fields]}

    def human_review(state: DraftState) -> Command[Literal["finalize"]]:
        """사람이 확정한다. 검토 화면에 줄 내용을 interrupt로 내보내고 결정을 받는다."""
        answer = interrupt(
            {
                "fields": state["fields"],
                "signals": state.get("signals", []),
                "verdicts": state.get("verdicts", {}),
            }
        )
        decision: Decision = "approve" if answer.get("decision") == "approve" else "reject"
        edits = {k: str(v) for k, v in (answer.get("edits") or {}).items()}
        return Command(goto="finalize", update={"decision": decision, "edits": edits})

    def finalize(state: DraftState) -> DraftState:
        """결정을 반영한다. 사람이 고친 값은 flag를 ok로 하고 근거는 비운다(사람이 쓴 값)."""
        if state["decision"] == "reject":
            return {"status": "rejected"}
        edits = state.get("edits", {})
        fields: list[dict[str, Any]] = []
        for raw in state["fields"]:
            if raw["key"] in edits:
                raw = {
                    **raw,
                    "value": edits[raw["key"]],
                    "source": "stated",
                    "evidence": [],
                    "candidates": [],
                    "flag": Flag.OK.value,
                }
            fields.append(raw)
        return {"fields": fields, "status": "approved"}

    builder = StateGraph(DraftState)
    builder.add_node("extract", extract)
    builder.add_node("verify", verify)
    builder.add_node("consensus", consensus)
    builder.add_node("human_review", human_review)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "extract")
    builder.add_conditional_edges("extract", fan_out, ["verify", "consensus"])
    builder.add_edge("verify", "consensus")
    builder.add_edge("consensus", "human_review")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
