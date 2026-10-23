"""호출 전 비용 추정 (dry-run). 네트워크를 쓰지 않는다.

토큰 수는 글자 수로 어림한 값이다. 한국어 토큰 비율은 모델·문장에 따라 달라 오차가 크다.
정확한 값은 첫 실행의 usage로 확인한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from carenote.evaluation.schema import GoldenCase
from carenote.extraction.llm_extractor import LLMExtractor
from carenote.extraction.pricing import Price, cost_usd
from carenote.extraction.prompts import build_user_message

CHARS_PER_TOKEN = 1.5  # 보수적으로 작게 잡아 토큰 수를 크게 본다
SCHEMA_OVERHEAD_TOKENS = 400  # 구조화 출력 스키마 지시
INPUT_HIGH_FACTOR = 1.3
# 출력에는 구조화 JSON(약 400토큰)과 적응형 추론이 포함된다. 추론량은 effort마다 달라 범위로 둔다.
OUTPUT_TOKENS_LOW = 600
OUTPUT_TOKENS_HIGH = 4000


@dataclass(frozen=True)
class Estimate:
    """실행 전 추정치."""

    cases: int
    cached: int
    to_call: int
    input_tokens_low: int
    input_tokens_high: int
    output_tokens_low: int
    output_tokens_high: int
    cost_low: float
    cost_high: float


def estimate_run(extractor: LLMExtractor, cases: list[GoldenCase], price: Price) -> Estimate:
    """캐시에 없는 케이스만 호출한다고 보고 토큰·비용 범위를 추정한다."""
    pending = [c for c in cases if not extractor.is_cached(c.transcript)]
    system_chars = len(extractor.system_prompt)
    input_low = sum(
        int((system_chars + len(build_user_message(c.transcript))) / CHARS_PER_TOKEN)
        + SCHEMA_OVERHEAD_TOKENS
        for c in pending
    )
    input_high = int(input_low * INPUT_HIGH_FACTOR)
    out_low = OUTPUT_TOKENS_LOW * len(pending)
    out_high = OUTPUT_TOKENS_HIGH * len(pending)
    return Estimate(
        cases=len(cases),
        cached=len(cases) - len(pending),
        to_call=len(pending),
        input_tokens_low=input_low,
        input_tokens_high=input_high,
        output_tokens_low=out_low,
        output_tokens_high=out_high,
        cost_low=cost_usd(price, input_low, out_low),
        cost_high=cost_usd(price, input_high, out_high),
    )


def format_estimate(model: str, effort: str, est: Estimate, prices_as_of: str) -> str:
    """사람이 읽는 추정 요약."""
    return "\n".join(
        [
            f"[dry-run] 모델 {model} · effort {effort}",
            f"  케이스 {est.cases}건 중 캐시 적중 {est.cached}건, API 호출 예정 {est.to_call}건",
            f"  입력 토큰 약 {est.input_tokens_low:,}~{est.input_tokens_high:,}"
            f" · 출력 토큰 약 {est.output_tokens_low:,}~{est.output_tokens_high:,}",
            f"  예상 비용 ${est.cost_low:.3f} ~ ${est.cost_high:.3f}"
            f" (단가 기준일 {prices_as_of}, 글자 수 기반 추정이라 오차가 크다)",
        ]
    )
