"""
Pydantic v2 models for Tradernet portfolio responses.

Field names follow the wire format ('s', 'q', 'i', ...) so a raw Tradernet
response can be parsed with ``Portfolio(**raw["ps"])`` directly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AccountBalance(BaseModel):
    """One currency line from the ``acc`` array."""

    model_config = ConfigDict(extra="ignore")

    s:        float                = Field(default=0.0, description="Free cash in this currency")
    curr:     str                  = Field(default="",  description="Currency code, e.g. USD/KZT/RUR")
    currval:  float                = Field(default=1.0, description="Exchange rate to base currency")
    forecast_in:  float | None = None
    forecast_out: float | None = None


class Position(BaseModel):
    """One holding line from the ``pos`` array."""

    model_config = ConfigDict(extra="ignore")

    i:         str          = Field(description="Ticker, e.g. 'AAPL.US'")
    q:         float        = Field(default=0.0, description="Quantity held")
    s:         float        = Field(default=0.0, description="Position market value (q * mkt_price)")
    mkt_price: float        = Field(default=0.0, description="Current market price")
    open_bal:  float        = Field(default=0.0, description="Cost basis / opening balance")
    profit_close: float | None = Field(default=None, description="Unrealized P&L")
    curr:      str          = Field(default="",  description="Position currency")
    currval:   float        = Field(default=1.0, description="Exchange rate to base currency")
    name:      str | None   = Field(default=None, description="Russian name")
    name2:     str | None   = Field(default=None, description="Latin name")
    t:         int | None   = Field(default=None, description="Security type")
    k:         int | None   = Field(default=None, description="Security kind")
    # Some Tradernet payloads carry an entry-price field instead of (or alongside) open_bal.
    price_a:   float | None = Field(default=None, description="Average entry price")
    bal_price_a: float | None = Field(default=None, description="Balance entry price")


class Portfolio(BaseModel):
    """Top-level portfolio object — the ``ps`` payload."""

    model_config = ConfigDict(extra="ignore")

    key: str                   = Field(default="",   description="Login key returned by API")
    acc: list[AccountBalance]  = Field(default_factory=list)
    pos: list[Position]        = Field(default_factory=list)

    @property
    def total_position_value(self) -> float:
        """Sum of ``s`` across all positions (in their respective currencies)."""
        return sum(p.s for p in self.pos)

    @property
    def total_pnl(self) -> float:
        """Sum of ``profit_close`` across positions where it is reported."""
        return sum(p.profit_close for p in self.pos if p.profit_close is not None)

    @property
    def total_cash(self) -> float:
        """Sum of free cash across all currency accounts (raw ``s``, no FX conversion)."""
        return sum(a.s for a in self.acc)
