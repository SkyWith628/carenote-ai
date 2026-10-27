"""인메모리 초안 저장소 (프로세스가 끝나면 사라진다)."""

from __future__ import annotations

import threading
from collections.abc import Callable

from carenote.drafts.models import Draft


class InMemoryDraftStore:
    """스레드 안전한 초안 저장소. 동기 엔드포인트가 스레드풀에서 동시에 접근한다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._drafts: dict[str, Draft] = {}

    def add(self, draft: Draft) -> None:
        """초안을 저장한다. 같은 id가 있으면 ValueError."""
        with self._lock:
            if draft.id in self._drafts:
                raise ValueError(f"이미 있는 초안 id: {draft.id}")
            self._drafts[draft.id] = draft

    def get(self, draft_id: str) -> Draft | None:
        """id로 조회한다."""
        with self._lock:
            return self._drafts.get(draft_id)

    def list(self) -> list[Draft]:
        """최신순 목록."""
        with self._lock:
            return sorted(self._drafts.values(), key=lambda d: d.created_at, reverse=True)

    def update(self, draft_id: str, change: Callable[[Draft], Draft]) -> Draft | None:
        """락 안에서 `change`를 적용해 저장한다 (결정이 겹쳐도 한 번만 반영된다). 없으면 None.

        `change`가 예외를 내면 저장소는 바뀌지 않는다.
        """
        with self._lock:
            current = self._drafts.get(draft_id)
            if current is None:
                return None
            updated = change(current)
            self._drafts[draft_id] = updated
            return updated
