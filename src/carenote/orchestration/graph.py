"""초안 생성 그래프.

extract → verify(위험 필드마다 병렬) → consensus → 사람 검토(interrupt) → finalize.

원칙: LLM은 추출·재검증만, flag 판정은 코드(Consensus), 확정은 사람(interrupt).
상태는 체크포인트에 안전하게 저장되도록 JSON 호환 값(dict·str)만 둔다. 체크포인터는 호출자가 준다.
`human_review` 노드는 resume 때 처음부터 다시 실행되므로 interrupt 앞에서 부작용을 만들지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Protocol, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import CompiledStateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.types import Command, Send, interrupt

from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import FieldKey, Flag
from carenote.extraction.sentinel import Signal
from carenote.extraction.verifier import (
    FieldChecker,
    FieldVerdict,
    apply_verdicts,
    needs_verification,
)

Decision = Literal["approve", "reject"]
Status = Literal["approved", "rejected"]


def _merge_verdicts(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """병렬 verify 노드의 결과를 검사기·필드 키별로 합친다 (리듀서)."""
    merged = {name: dict(by_key) for name, by_key in left.items()}
    for name, by_key in right.items():
        merged.setdefault(name, {}).update(by_key)
    return merged


class DraftState(TypedDict, total=False):
    """그래프 상태. fields·verdicts·signals는 JSON 호환 dict다."""

    transcript: str
    fields: list[dict[str, Any]]
    signals: list[dict[str, Any]]
    # 검사기 이름 → 필드 키 → {status, quote, reason}
    verdicts: Annotated[dict[str, dict[str, Any]], _merge_verdicts]
    decision: Decision
    edits: dict[str, str]
    status: Status


class VerifyInput(TypedDict):
    """Send로 verify 노드에 넘기는 입력 (검사기 하나 × 필드 하나)."""

    checker: str
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


def verdicts_from_state(by_key: Mapping[str, Mapping[str, Any]]) -> dict[FieldKey, FieldVerdict]:
    """상태에 저장된 한 검사기의 판정(JSON 호환 dict)을 FieldVerdict로 되돌린다."""
    return {
        FieldKey(key): FieldVerdict(
            key=FieldKey(key), status=raw["status"], quote=raw["quote"], reason=raw["reason"]
        )
        for key, raw in by_key.items()
    }


def resume_command(answer: dict[str, Any] | None) -> Command[Any]:
    """사람의 답으로 interrupt를 재개하는 Command를 만든다.

    LangGraph는 빈 resume 값(`{}`, None)을 "재개 값 없음"으로 보고 interrupt에 다시 멈춘다.
    빈 답이 대기 상태로 남지 않고 거절로 확정되도록, 비어 있으면 승인이 아닌 값으로 바꾼다.
    """
    return Command(resume=answer or {"decision": "invalid"})


def build_graph(
    extractor: GraphExtractor,
    checkers: FieldChecker | Mapping[str, FieldChecker],
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph[DraftState, None, DraftState, DraftState]:
    """그래프를 만든다. interrupt를 쓰므로 체크포인터가 필수다.

    검사기가 여럿이면 (검사기 × 위험 필드) 호출을 한 단계에서 동시에 보낸다. 검사기 하나만 주면
    이름은 "verifier"다.
    """
    named: dict[str, FieldChecker] = (
        dict(checkers) if isinstance(checkers, Mapping) else {"verifier": checkers}
    )

    def extract(state: DraftState) -> DraftState:
        transcript = state.get("transcript", "")
        fields = extractor.extract(transcript)
        return {
            "fields": [f.model_dump(mode="json") for f in fields],
            "signals": [_signal_dict(s) for s in extractor.signals_for(transcript)],
        }

    def fan_out(state: DraftState) -> list[Send] | str:
        """(검사기 × 위험 필드)마다 verify를 병렬로 보낸다. 대상이 없으면 바로 consensus."""
        targets = [f for f in map(_field, state.get("fields", [])) if needs_verification(f)]
        if not targets:
            return "consensus"
        return [
            Send(
                "verify",
                VerifyInput(
                    checker=name,
                    transcript=state.get("transcript", ""),
                    field=f.model_dump(mode="json"),
                ),
            )
            for name in named
            for f in targets
        ]

    def verify(state: VerifyInput) -> DraftState:
        field = _field(state["field"])
        verdict = named[state["checker"]].verify(state["transcript"], field)
        # 체크포인트에는 JSON 호환 값만 둔다 (키는 바깥 dict가 가진다)
        raw = {"status": verdict.status, "quote": verdict.quote, "reason": verdict.reason}
        return {"verdicts": {state["checker"]: {field.key.value: raw}}}

    def consensus(state: DraftState) -> DraftState:
        fields = [_field(f) for f in state.get("fields", [])]
        for by_key in state.get("verdicts", {}).values():
            fields = apply_verdicts(fields, verdicts_from_state(by_key))
        return {"fields": [f.model_dump(mode="json") for f in fields]}

    def human_review(state: DraftState) -> Command[Literal["finalize"]]:
        """사람이 확정한다. 검토 화면에 줄 내용을 interrupt로 내보내고 결정을 받는다."""
        answer = cast(
            dict[str, Any],
            interrupt(
                {
                    "fields": state.get("fields", []),
                    "signals": state.get("signals", []),
                    "verdicts": state.get("verdicts", {}),
                }
            ),
        )
        decision: Decision = "approve" if answer.get("decision") == "approve" else "reject"
        edits = {k: str(v) for k, v in cast(dict[str, Any], answer.get("edits") or {}).items()}
        return Command(goto="finalize", update={"decision": decision, "edits": edits})

    def finalize(state: DraftState) -> DraftState:
        """결정을 반영한다. 사람이 고친 값은 flag를 ok로 하고 근거는 비운다(사람이 쓴 값)."""
        if state.get("decision") == "reject":
            return {"status": "rejected"}
        edits = state.get("edits", {})
        fields: list[dict[str, Any]] = []
        for raw in state.get("fields", []):
            item: dict[str, Any] = raw
            if item["key"] in edits:
                item = {
                    **item,
                    "value": edits[item["key"]],
                    "source": "stated",
                    "evidence": [],
                    "candidates": [],
                    "flag": Flag.OK.value,
                }
            fields.append(item)
        return {"fields": fields, "status": "approved"}

    builder: StateGraph[DraftState, None, DraftState, DraftState] = StateGraph(DraftState)
    builder.add_node("extract", extract)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("verify", verify, input_schema=VerifyInput)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("consensus", consensus)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("human_review", human_review)  # pyright: ignore[reportUnknownMemberType]
    builder.add_node("finalize", finalize)  # pyright: ignore[reportUnknownMemberType]
    builder.add_edge(START, "extract")
    builder.add_conditional_edges("extract", fan_out, ["verify", "consensus"])
    builder.add_edge("verify", "consensus")
    builder.add_edge("consensus", "human_review")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)  # pyright: ignore[reportUnknownMemberType]
