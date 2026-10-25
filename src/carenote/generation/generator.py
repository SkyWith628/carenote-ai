"""transcript 생성기: 스펙 1건당 독립 호출, 자동 점검 실패 시 사유를 덧붙여 재시도.

평가 대상 모델을 쓰지 않는다 (기본 claude-opus-5-5). 호출 비용은 CostGuard로 제한한다.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import anthropic

from carenote.extraction.llm_extractor import ParseClient
from carenote.extraction.pricing import CostGuard
from carenote.generation.checks import failures, run_candidate_checks
from carenote.generation.diversity import Diversity
from carenote.generation.prompts import (
    GENERATION_PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_message,
)
from carenote.generation.schema import CheckResult, GeneratedCase
from carenote.generation.spec import Spec

MAX_TOKENS = 16000
DEFAULT_MAX_ATTEMPTS = 3
_UNUSABLE_STOP_REASONS = frozenset({"refusal", "max_tokens"})

Status = Literal["passed", "needs_human", "api_failure"]


@dataclass
class GenerationStats:
    """호출 집계."""

    specs: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    api_failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class GenerationResult:
    """스펙 한 건의 생성 결과."""

    spec_id: str
    status: Status
    attempts: int
    case: GeneratedCase | None
    checks: list[CheckResult] = field(default_factory=lambda: list[CheckResult]())
    comparison_only: bool = False


class Generator:
    """스펙에서 transcript 후보를 만든다."""

    def __init__(
        self,
        client: ParseClient,
        *,
        model: str,
        effort: str,
        cache_dir: Path | None = None,
        guard: CostGuard | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        diversity: dict[str, Diversity] | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self.effort = effort
        self._cache_dir = cache_dir
        self._guard = guard
        self._max_attempts = max_attempts
        self._diversity = diversity or {}
        self.stats = GenerationStats()

    def usage_summary(self) -> dict[str, object]:
        """사용량 요약."""
        return {
            "model": self.model,
            "effort": self.effort,
            "prompt_version": GENERATION_PROMPT_VERSION,
            **asdict(self.stats),
        }

    def generate(self, spec: Spec, *, comparison_only: bool = False) -> GenerationResult:
        """후보 1건을 만든다. 직접 작성(human) 스펙은 비교용으로만 생성할 수 있다."""
        if spec.author == "human" and not comparison_only:
            raise ValueError(f"{spec.spec_id}는 직접 작성 스펙이라 생성하지 않는다 (비교용만 허용)")
        self.stats.specs += 1
        feedback: str | None = None
        last_case: GeneratedCase | None = None
        last_checks: list[CheckResult] = []
        for attempt in range(1, self._max_attempts + 1):
            case = self._obtain(spec, attempt, feedback)
            if case is None:
                self.stats.api_failures += 1
                return GenerationResult(
                    spec.spec_id, "api_failure", attempt, last_case, last_checks, comparison_only
                )
            checks = run_candidate_checks(spec, case, self._diversity.get(spec.spec_id))
            failed = failures(checks)
            if not failed:
                return GenerationResult(
                    spec.spec_id, "passed", attempt, case, checks, comparison_only
                )
            last_case, last_checks = case, checks
            feedback = "\n".join(f"- [{r.code}] {r.message}" for r in failed)
        return GenerationResult(
            spec.spec_id, "needs_human", self._max_attempts, last_case, last_checks, comparison_only
        )

    # -- 호출과 캐시 -----------------------------------------------------------

    def _obtain(self, spec: Spec, attempt: int, feedback: str | None) -> GeneratedCase | None:
        key = self._cache_key(spec, attempt, feedback)
        cached = self._read_cache(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        generated = self._call(spec, feedback)
        if generated is not None:
            self._write_cache(key, generated)
        return generated

    def _call(self, spec: Spec, feedback: str | None) -> GeneratedCase | None:
        if self._guard is not None:
            self._guard.check()  # BudgetExceeded는 잡지 않고 전체를 멈춘다
        started = time.perf_counter()
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self.request_message(spec, feedback)}],
                output_format=GeneratedCase,
                output_config={"effort": self.effort},
            )
        except anthropic.APIError:
            return None
        finally:
            self.stats.latency_s += time.perf_counter() - started
        self.stats.api_calls += 1
        self.stats.input_tokens += response.usage.input_tokens
        self.stats.output_tokens += response.usage.output_tokens
        if self._guard is not None:
            self._guard.add(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason in _UNUSABLE_STOP_REASONS:
            return None
        parsed = response.parsed_output
        return parsed if isinstance(parsed, GeneratedCase) else None

    def diversity_for(self, spec_id: str) -> Diversity | None:
        """스펙에 배정된 다양성 지시."""
        return self._diversity.get(spec_id)

    def request_message(self, spec: Spec, feedback: str | None = None) -> str:
        """API로 보내는 사용자 메시지 (배정된 다양성 포함)."""
        return build_user_message(spec, feedback, self._diversity.get(spec.spec_id))

    def _cache_key(self, spec: Spec, attempt: int, feedback: str | None) -> str:
        # 가설은 프롬프트에 들어가지 않으므로 키에서 제외해 가설 수정이 캐시를 깨지 않게 한다
        spec_payload = spec.model_dump(mode="json", exclude={"hypothesis", "policy"})
        diversity = self._diversity.get(spec.spec_id)
        payload = json.dumps(
            [
                spec_payload,
                GENERATION_PROMPT_VERSION,
                self.model,
                self.effort,
                attempt,
                feedback or "",
                diversity.to_payload() if diversity else None,
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> GeneratedCase | None:
        if self._cache_dir is None:
            return None
        try:
            return GeneratedCase.model_validate_json(
                (self._cache_dir / f"{key}.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

    def _write_cache(self, key: str, case: GeneratedCase) -> None:
        if self._cache_dir is None:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        (self._cache_dir / f"{key}.json").write_text(case.model_dump_json(), encoding="utf-8")

    def is_cached(self, spec: Spec) -> bool:
        """1차 시도 결과가 캐시에 있는가 (dry-run 추정용)."""
        return self._read_cache(self._cache_key(spec, 1, None)) is not None
