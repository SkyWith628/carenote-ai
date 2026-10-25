"""생성 비용 추정 (dry-run). 네트워크를 쓰지 않는다.

토큰 수는 글자 수로 어림한 값이고 Opus의 추론 토큰량은 아직 측정한 적이 없다.
파일럿 실측으로 보정하기 전까지는 범위가 넓다 (설계 8장).
"""

from __future__ import annotations

from dataclasses import dataclass

from carenote.extraction.estimate import CHARS_PER_TOKEN, INPUT_HIGH_FACTOR
from carenote.extraction.pricing import Price, cost_usd
from carenote.generation.generator import Generator
from carenote.generation.prompts import SYSTEM_PROMPT, build_user_message
from carenote.generation.spec import Spec

OUTPUT_TOKENS_LOW = 800
OUTPUT_TOKENS_HIGH = 3000
RETRY_FACTOR_LOW = 1.0
RETRY_FACTOR_HIGH = 1.3  # 평균 시도 횟수. 최악(스펙당 3회)은 CostGuard 한도가 막는다


@dataclass(frozen=True)
class GenerationEstimate:
    """실행 전 추정치."""

    specs: int
    cached: int
    to_call: int
    input_tokens_low: int
    input_tokens_high: int
    output_tokens_low: int
    output_tokens_high: int
    cost_low: float
    cost_high: float


def estimate_generation(
    generator: Generator, specs: list[Spec], price: Price
) -> GenerationEstimate:
    """캐시에 1차 결과가 없는 스펙만 호출한다고 보고 범위를 추정한다."""
    pending = [s for s in specs if not generator.is_cached(s)]
    input_low = sum(
        int((len(SYSTEM_PROMPT) + len(build_user_message(s))) / CHARS_PER_TOKEN) for s in pending
    )
    input_high = int(input_low * INPUT_HIGH_FACTOR * RETRY_FACTOR_HIGH)
    input_low = int(input_low * RETRY_FACTOR_LOW)
    out_low = int(OUTPUT_TOKENS_LOW * len(pending) * RETRY_FACTOR_LOW)
    out_high = int(OUTPUT_TOKENS_HIGH * len(pending) * RETRY_FACTOR_HIGH)
    return GenerationEstimate(
        specs=len(specs),
        cached=len(specs) - len(pending),
        to_call=len(pending),
        input_tokens_low=input_low,
        input_tokens_high=input_high,
        output_tokens_low=out_low,
        output_tokens_high=out_high,
        cost_low=cost_usd(price, input_low, out_low),
        cost_high=cost_usd(price, input_high, out_high),
    )


def format_estimate(model: str, effort: str, est: GenerationEstimate, prices_as_of: str) -> str:
    """사람이 읽는 추정 요약."""
    return "\n".join(
        [
            f"[dry-run] 생성 모델 {model} · effort {effort}",
            f"  스펙 {est.specs}건 중 캐시 적중 {est.cached}건, API 호출 예정 {est.to_call}건",
            f"  입력 토큰 약 {est.input_tokens_low:,}~{est.input_tokens_high:,}"
            f" · 출력 토큰 약 {est.output_tokens_low:,}~{est.output_tokens_high:,}",
            f"  예상 비용 ${est.cost_low:.3f} ~ ${est.cost_high:.3f}"
            f" (단가 기준일 {prices_as_of}, 추론 토큰량 미측정이라 범위가 넓다)",
        ]
    )
