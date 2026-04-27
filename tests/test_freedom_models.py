"""Unit tests for freedom_portfolio.models — no network access."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from freedom_portfolio.models import Portfolio  # noqa: E402


def test_portfolio_parses_api_response():
    """Feed a realistic Tradernet response and assert the model fields."""
    fixture = {
        "ps": {
            "key": "%TESTUSER",
            "acc": [{"s": 100000.0, "curr": "USD", "currval": 1.0}],
            "pos": [
                {
                    "i": "AAPL.US", "q": 10, "s": 1750.0,
                    "mkt_price": 175.0, "open_bal": 1500.0,
                    "profit_close": 250.0, "curr": "USD", "currval": 1.0,
                    "name": "Apple", "name2": "Apple Inc.",
                }
            ],
        }
    }
    portfolio = Portfolio(**fixture["ps"])
    assert portfolio.pos[0].i == "AAPL.US"
    assert portfolio.pos[0].name2 == "Apple Inc."
    assert portfolio.total_pnl == 250.0
    assert portfolio.total_position_value == 1750.0
    assert portfolio.total_cash == 100_000.0


def test_portfolio_tolerates_missing_optional_fields():
    """Position parsing must succeed even when only the required fields exist."""
    portfolio = Portfolio(pos=[{"i": "TSLA.US", "q": 5, "s": 1000, "mkt_price": 200, "open_bal": 900}])
    assert portfolio.pos[0].i == "TSLA.US"
    assert portfolio.pos[0].profit_close is None
    assert portfolio.total_pnl == 0.0   # no profit_close → contributes nothing


def test_portfolio_ignores_unknown_fields():
    """Extra fields in the wire format must not break parsing."""
    portfolio = Portfolio(
        key="X",
        acc=[{"s": 1, "curr": "EUR", "currval": 1.1, "garbage_field": 42}],
        pos=[],
    )
    assert portfolio.acc[0].curr == "EUR"
