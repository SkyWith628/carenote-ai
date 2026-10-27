"""모델 단가와 비용 가드.

단가는 공식 가격 문서(platform.claude.com/docs/en/about-claude/pricing)로 2026-10-26에
재확인했다(1M 토큰당 USD). 단가가 바뀌었을 수 있으니 실측 전에 공식 가격 문서로 다시 확인한다.
등록되지 않은 모델은 비용을 추정하지 않고 실행을 거부한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """1M 토큰당 USD."""

    input_per_mtok: float
    output_per_mtok: float


PRICES_AS_OF = "2026-10-26"
PRICES: dict[str, Price] = {
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-opus-5-5": Price(4.00, 20.00),
    "claude-haiku-5-5": Price(0.10, 0.50),  # 프롬프트 100K 토큰 이하 기준
}


def price_for(model: str) -> Price | None:
    """등록된 단가. 없으면 None."""
    return PRICES.get(model)


def cost_usd(price: Price, input_tokens: int, output_tokens: int) -> float:
    """토큰 수로 비용을 계산한다. 캐시 읽기 할인은 반영하지 않는다 (보수적)."""
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000


class BudgetExceeded(RuntimeError):
    """승인된 비용 한도에 도달해 호출을 멈췄다."""


class CostGuard:
    """누적 비용이 승인 한도에 도달하면 다음 호출을 막는다.

    한 호출이 끝난 뒤에야 비용을 알 수 있으므로, 한도를 넘는 양은 최대 호출 1회분이다.
    """

    def __init__(self, limit_usd: float, price: Price) -> None:
        self.limit_usd = limit_usd
        self._price = price
        self.spent_usd = 0.0

    def check(self) -> None:
        """한도에 도달했으면 BudgetExceeded."""
        if self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(
                f"승인 한도 ${self.limit_usd:.2f}에 도달 (사용 ${self.spent_usd:.4f})"
            )

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """호출 비용을 누적한다."""
        self.spent_usd += cost_usd(self._price, input_tokens, output_tokens)
