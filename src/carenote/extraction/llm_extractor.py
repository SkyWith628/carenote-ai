"""LLM Extractor (B0: 단일 LLM, B1: + 근거 강제).

transcript만 입력으로 받는다. flag는 여기서 정하지 않는다 (Flagger 담당).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import anthropic
from pydantic import BaseModel

from carenote.config import Settings
from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.schema import Evidence, FieldKey, Source
from carenote.extraction.pricing import CostGuard, cost_usd, price_for
from carenote.extraction.prompts import (
    DEFAULT_PROMPT_VERSION,
    PromptVersion,
    Variant,
    build_system_prompt,
    build_user_message,
    normalize_edits,
)
from carenote.extraction.schema import (
    LLMExtractionB0,
    LLMExtractionB1,
    LLMFieldB0,
    LLMFieldB1,
)

MAX_TOKENS = 16000
# 출력 토큰 한도에 걸리거나 거절된 응답은 일부만 담겨 있을 수 있어 쓰지 않는다.
_UNUSABLE_STOP_REASONS = frozenset({"refusal", "max_tokens"})


class UsageLike(Protocol):
    """응답 사용량 (anthropic.types.Usage의 부분집합)."""

    @property
    def input_tokens(self) -> int: ...
    @property
    def output_tokens(self) -> int: ...
    @property
    def cache_read_input_tokens(self) -> int | None: ...


class ParseResponse(Protocol):
    """messages.parse 응답의 부분집합."""

    @property
    def stop_reason(self) -> str | None: ...
    @property
    def parsed_output(self) -> object | None: ...
    @property
    def usage(self) -> UsageLike: ...


class MessagesAPI(Protocol):
    """테스트에서 가짜로 바꿀 수 있는 최소 인터페이스."""

    def parse(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
        output_format: type[BaseModel],
        output_config: dict[str, str],
    ) -> ParseResponse: ...


class ParseClient(Protocol):
    """anthropic.Anthropic 호환 클라이언트."""

    @property
    def messages(self) -> MessagesAPI: ...


def build_client(settings: Settings) -> ParseClient:
    """설정의 API 키로 실제 클라이언트를 만든다. 키가 없으면 ValueError."""
    if settings.anthropic_api_key is None:
        raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았다 (.env에 넣는다)")
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
    return cast(ParseClient, client)


@dataclass
class ExtractorStats:
    """호출 집계. ablation 표의 비용·지연 열의 원천."""

    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    demoted_fields: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    latency_s: float = 0.0


class LLMExtractor:
    """transcript에서 필드를 추출한다."""

    def __init__(
        self,
        client: ParseClient,
        *,
        variant: Variant,
        model: str,
        effort: str,
        cache_dir: Path | None = None,
        guard: CostGuard | None = None,
        prompt_version: PromptVersion = DEFAULT_PROMPT_VERSION,
        run_tag: str | None = None,
        prompt_edits: tuple[int, ...] | None = None,
    ) -> None:
        self._client = client
        self._run_tag = run_tag
        self._edits = normalize_edits(prompt_edits)
        if self._edits is not None and prompt_version != "v4":
            raise ValueError("prompt_edits는 프롬프트 v4에서만 쓴다")
        self._prompt_version: PromptVersion = prompt_version
        self._variant: Variant = variant
        self._model = model
        self._effort = effort
        self._cache_dir = cache_dir
        self._guard = guard
        self._system = build_system_prompt(variant, prompt_version, self._edits)
        self._schema: type[LLMExtractionB0] | type[LLMExtractionB1] = (
            LLMExtractionB1 if variant == "b1" else LLMExtractionB0
        )
        self.stats = ExtractorStats()

    @property
    def system_prompt(self) -> str:
        """이 변형의 시스템 프롬프트."""
        return self._system

    def is_cached(self, transcript: str) -> bool:
        """이 transcript의 결과가 캐시에 있는가 (dry-run 추정용)."""
        return self._read_cache(self._cache_key(transcript)) is not None

    def usage_summary(self) -> dict[str, object]:
        """보고서용 사용량 요약."""
        price = price_for(self._model)
        cost = (
            None
            if price is None
            else round(cost_usd(price, self.stats.input_tokens, self.stats.output_tokens), 6)
        )
        return {
            "estimated_cost_usd": cost,
            "variant": self._variant,
            "model": self._model,
            "effort": self._effort,
            "prompt_version": self._prompt_version,
            "run_tag": self._run_tag,
            "prompt_edits": list(self._edits) if self._edits is not None else None,
            **asdict(self.stats),
        }

    def extract(self, transcript: str) -> list[PredictedField]:
        """필드를 추출한다. 호출 실패 시 빈 목록(전부 missing으로 채점됨)을 반환한다."""
        key = self._cache_key(transcript)
        extraction = self._read_cache(key)
        if extraction is not None:
            self.stats.cache_hits += 1
        else:
            extraction = self._call(transcript)
            if extraction is None:
                self.stats.failures += 1
                return []
            self._write_cache(key, extraction)
        return self._convert(extraction, transcript)

    # -- API 호출 -------------------------------------------------------------

    def _call(self, transcript: str) -> LLMExtractionB0 | LLMExtractionB1 | None:
        if self._guard is not None:
            self._guard.check()  # BudgetExceeded는 잡지 않고 실행 전체를 멈춘다
        started = time.perf_counter()
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=self._system,
                messages=[{"role": "user", "content": build_user_message(transcript)}],
                output_format=self._schema,
                output_config={"effort": self._effort},
            )
        except anthropic.APIError:
            return None
        finally:
            self.stats.latency_s += time.perf_counter() - started
        self.stats.calls += 1
        self.stats.input_tokens += response.usage.input_tokens
        self.stats.output_tokens += response.usage.output_tokens
        self.stats.cache_read_tokens += response.usage.cache_read_input_tokens or 0
        if self._guard is not None:
            self._guard.add(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason in _UNUSABLE_STOP_REASONS:
            return None
        parsed = response.parsed_output
        if isinstance(parsed, self._schema):
            return parsed
        return None

    # -- 캐시 -----------------------------------------------------------------

    def _cache_key(self, transcript: str) -> str:
        parts: list[str] = [
            self._model,
            self._effort,
            self._variant,
            self._prompt_version,
            transcript,
        ]
        if self._edits is not None:
            # ablation 변형(v4의 일부 편집만 적용). 전부 적용(= v4)이면 키가 그대로다.
            parts.append("edits=" + ",".join(str(e) for e in self._edits))
        if self._run_tag is not None:
            # 같은 입력의 반복 실행(실행 간 변동 측정)을 위해서만 키에 넣는다. 태그가 없으면
            # 키 공식이 이전과 같아 기존 캐시가 그대로 적중한다.
            parts.append(self._run_tag)
        payload = json.dumps(parts, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> LLMExtractionB0 | LLMExtractionB1 | None:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        try:
            return self._schema.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_cache(self, key: str, extraction: LLMExtractionB0 | LLMExtractionB1) -> None:
        if self._cache_dir is None:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        (self._cache_dir / f"{key}.json").write_text(extraction.model_dump_json(), encoding="utf-8")

    # -- 변환과 근거 검증 -------------------------------------------------------

    def _convert(
        self, extraction: LLMExtractionB0 | LLMExtractionB1, transcript: str
    ) -> list[PredictedField]:
        seen: set[FieldKey] = set()
        result: list[PredictedField] = []
        for raw in extraction.fields:
            if raw.key in seen:
                continue
            seen.add(raw.key)
            result.append(self._convert_field(raw, transcript))
        return result

    def _convert_field(self, raw: LLMFieldB0 | LLMFieldB1, transcript: str) -> PredictedField:
        missing = PredictedField(key=raw.key)
        if raw.source == "missing":
            return missing

        value = (raw.value or "").strip() or None
        candidates = tuple(c.strip() for c in raw.candidates if c.strip())
        if len(candidates) >= 2:
            value = None  # 모호한 값은 임의 선택하지 않는다
        elif len(candidates) == 1 and value is None:
            value, candidates = candidates[0], ()
        else:
            candidates = ()
        if value is None and not candidates:
            return missing

        evidence: tuple[Evidence, ...] = ()
        if isinstance(raw, LLMFieldB1):
            quotes = [q for q in raw.evidence if q.strip()]
            if not quotes or not all(q in transcript for q in quotes):
                self.stats.demoted_fields += 1
                return missing  # 근거 강제: 원문에 없는 인용이면 값을 버린다
            evidence = tuple(Evidence(text=q) for q in quotes)

        return PredictedField(
            key=raw.key,
            value=value,
            source=Source(raw.source),
            evidence=evidence,
            candidates=candidates,
        )
