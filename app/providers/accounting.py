"""Per-physical-request usage capture, before answer parsing or validation."""

from hashlib import sha256
import json
from typing import Any

from app.domain.providers import TokenUsage
from app.evaluation.budget import BudgetError, BudgetLedger, PRICES, Price


class RequestAccounting:
    def __init__(self, ledger: BudgetLedger | None, provider: str, model: str,
                 kind: str, role: str, price: Price | None = None):
        self.ledger, self.provider, self.model = ledger, provider, model
        self.kind, self.role = kind, role
        self.price = price or PRICES.get((provider, model))
        self.ids: list[str] = []
        self.usage = TokenUsage()
        self.complete = True

    def reserve(self, payload: dict, max_output: int) -> str | None:
        if self.ledger is None:
            return None
        if self.price is None:
            raise BudgetError("budget_unpriced_model", "no verified price for the selected provider/model")
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        # Text tokenizers cannot use more tokens than UTF-8 bytes. Add framing
        # margin for protocol/message tokens; reject oversized priced tiers.
        bound = len(body) + 4096
        identifier = self.ledger.reserve(
            provider=self.provider, model=self.model, kind=self.kind, role=self.role,
            request_hash=sha256(body).hexdigest(), price=self.price,
            max_input=bound, max_output=max_output,
        )
        self.ids.append(identifier)
        return identifier

    def record(self, identifier: str | None, usage: Any) -> None:
        def field(name, default=None):
            return usage.get(name, default) if isinstance(usage, dict) else getattr(usage, name, default)
        try:
            input_tokens = field("prompt_tokens")
            output_tokens = 0 if self.kind == "embedding" else field("completion_tokens")
            cached = field("prompt_cache_hit_tokens")
            if cached is None:
                details = field("prompt_tokens_details", {})
                cached = details.get("cached_tokens", 0) if isinstance(details, dict) else getattr(details, "cached_tokens", 0)
            if any(type(n) is not int or n < 0 for n in (input_tokens, output_tokens, cached)) or cached > input_tokens:
                raise ValueError("invalid usage")
            if identifier is not None:
                self.ledger.settle(identifier, input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=cached)
            self.usage.prompt_tokens += input_tokens
            self.usage.completion_tokens += output_tokens
            self.usage.total_tokens += input_tokens + output_tokens
            self.usage.cached_input_tokens += cached
        except (ValueError, TypeError, AttributeError):
            self.unknown(identifier)

    def unknown(self, identifier: str | None, outcome: str = "usage_unknown") -> None:
        self.complete = False
        if identifier is not None:
            self.ledger.unknown(identifier, outcome)

    def annotate(self, identifier: str | None, outcome: str) -> None:
        if identifier is not None:
            self.ledger.annotate(identifier, outcome)
