"""초안 파이프라인: LangGraph 그래프로 추출·검증을 돌리고 사람 결정을 받아 재개한다.

규칙(확인해야 승인, append-only revision, 감사 기록)은 `service.decide`가 갖는다. 그래프는
오케스트레이션(병렬 팬아웃, interrupt 대기, 체크포인트)만 맡고 결정 규칙을 다시 구현하지 않는다.
초안 id를 그래프의 thread_id로 쓴다. 체크포인터가 인메모리라 프로세스가 끝나면 대기도 사라진다.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from carenote.drafts.models import DecisionRequest, Draft, SignalOut
from carenote.drafts.service import DraftExtractor, assemble_draft
from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import Flag
from carenote.extraction.verifier import FieldChecker, VerifyingExtractor
from carenote.orchestration.graph import (
    GraphExtractor,
    build_graph,
    resume_command,
    verdicts_from_state,
)


def _config(draft_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": draft_id}}


class DraftPipeline:
    """extractor(와 그것이 감싼 검사기들)로 그래프를 만들고 초안 생성·결정 재개를 맡는다."""

    def __init__(self, extractor: DraftExtractor, checkpointer: BaseCheckpointSaver[Any]) -> None:
        inner: DraftExtractor
        checkers: dict[str, FieldChecker]
        if isinstance(extractor, VerifyingExtractor):
            inner, checkers = cast(DraftExtractor, extractor.inner), extractor.checkers
        else:
            inner, checkers = extractor, {}
        self._graph = build_graph(cast(GraphExtractor, inner), checkers, checkpointer)

    def create(self, transcript: str, *, masked_count: int = 0) -> Draft:
        """그래프를 사람 검토 interrupt까지 실행해 초안을 만든다. 그래프는 결정을 기다린다."""
        draft_id = uuid.uuid4().hex
        state = self._graph.invoke({"transcript": transcript}, _config(draft_id))  # pyright: ignore[reportUnknownMemberType]
        result = cast(dict[str, Any], state)
        fields = [PredictedField.model_validate(raw) for raw in result["fields"]]
        signals = [
            SignalOut(
                rule_id=s["rule_id"],
                category=s["category"],
                quote=s["quote"],
                severity=Flag(s["severity"]),
                source=s["source"],
            )
            for s in result.get("signals", [])
        ]
        verdicts = {n: verdicts_from_state(by_key) for n, by_key in result["verdicts"].items()}
        return assemble_draft(
            draft_id, transcript, fields, signals, verdicts, masked_count=masked_count
        )

    def is_waiting(self, draft_id: str) -> bool:
        """그래프가 사람 결정을 기다리는 중인가."""
        return bool(self._graph.get_state(_config(draft_id)).next)  # pyright: ignore[reportUnknownMemberType]

    def resume(self, draft_id: str, request: DecisionRequest) -> str | None:
        """결정이 검증을 통과한 뒤 그래프를 재개한다. 기다리는 그래프가 없으면 None."""
        config = _config(draft_id)
        if not self.is_waiting(draft_id):
            return None
        answer: dict[str, Any] = {
            "decision": request.decision,
            "edits": {key.value: value for key, value in request.edits.items()},
            "confirmed": [key.value for key in request.confirmed],
        }
        state = self._graph.invoke(resume_command(answer), config)  # pyright: ignore[reportUnknownMemberType]
        return cast(dict[str, Any], state).get("status")
