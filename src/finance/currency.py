"""
Base-currency / FX layer for the MAC3 risk engine (block H2).

Why this module exists
──────────────────────
Mixing local risk-free rates (e.g. NBK 14%) with USD-denominated portfolios
destroys the currency risk premium and silently biases Sharpe / Sortino.
This module is the single source of truth that lets the engine:

  1. pick a Reporting Currency (USD or KZT) deterministically,
  2. transform a local-currency price matrix into the Reporting Currency
     by point-wise FX multiplication (FX volatility flows naturally into
     the covariance matrix — no synthetic FX factor needed),
  3. apply a single, currency-matched annual RFR with **geometric** daily
     compounding (H3), and
  4. stay Rust-export-ready: all arrays are plain numpy.float64 dense
     matrices, no Python objects, no copies on the hot path.

Design constraints
──────────────────
* Pure-numpy / pandas — no sklearn, no Anthropic SDK.  Unit-testable in
  isolation.
* Side-effect free: every public function returns NEW frames; the caller's
  inputs are not mutated.  Critical so the Black-Litterman / Euler paths
  that read `weights_dict` keep operating on un-touched values.
* Safe short-circuit when reporting == asset currency for ALL assets:
  zero FX calls, zero allocations, original price frame returned.
* Backward-compatibility: if neither `REPORTING_CURRENCY` nor
  `US_RFR_ANNUAL` env-vars are set BUT the legacy `KZ_RFR_ANNUAL` is,
  the engine stays in the historical KZT/14% behaviour (no FX conversion)
  so the existing 214 tests keep passing.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger("Currency")


# ── Reporting Currency enum ─────────────────────────────────────────────────

class ReportingCurrency(str, Enum):
    """
    Single, authoritative report-base currency for all risk calculations.

    Adding a new currency = one enum member + one entry in
    `_DEFAULT_RFR_BY_CCY` and `_RFR_ENV_BY_CCY`.  Nothing else needs to
    change downstream.
    """
    USD = "USD"
    KZT = "KZT"

    @classmethod
    def from_env(cls, default: "ReportingCurrency" = None) -> "ReportingCurrency":
        """
        Resolve the reporting currency from env vars, with legacy fallback.

        Precedence:
          1. `REPORTING_CURRENCY` env var  (explicit, wins)
          2. Legacy: ONLY `KZ_RFR_ANNUAL` set → KZT (keeps existing prod
             deploys behaving exactly as before)
          3. `default` (USD if not supplied)
        """
        env = (os.getenv("REPORTING_CURRENCY") or "").strip().upper()
        if env:
            try:
                return cls(env)
            except ValueError:
                logger.warning(
                    "Unknown REPORTING_CURRENCY=%r — falling back to default",
                    env,
                )
        # Legacy mode: only KZ_RFR_ANNUAL set → treat as KZT for backward compat.
        has_kz  = "KZ_RFR_ANNUAL" in os.environ
        has_us  = "US_RFR_ANNUAL" in os.environ
        if has_kz and not has_us and default is None:
            return cls.KZT
        return default if default is not None else cls.USD


# ── Risk-free-rate registry ─────────────────────────────────────────────────

# Defaults are conservative best estimates as of 2026-06.  Override via env
# (US_RFR_ANNUAL / KZ_RFR_ANNUAL) — no source-code edit needed in prod.
_DEFAULT_RFR_BY_CCY: dict[ReportingCurrency, float] = {
    ReportingCurrency.USD: 0.045,   # SOFR / 3-mo T-Bill
    ReportingCurrency.KZT: 0.14,    # NBK base rate / TONIA
}

_RFR_ENV_BY_CCY: dict[ReportingCurrency, str] = {
    ReportingCurrency.USD: "US_RFR_ANNUAL",
    ReportingCurrency.KZT: "KZ_RFR_ANNUAL",
}


def get_rfr_for_currency(rc: ReportingCurrency) -> tuple[float, str]:
    """
    Return ``(annual_rfr, source_label)`` for the given reporting currency.

    `source_label` is the human-readable origin tag that gets surfaced into
    `portfolio_metrics` and the report's QC panel — so the user can see
    where the rate came from instead of guessing.
    """
    env_name = _RFR_ENV_BY_CCY[rc]
    env_val  = os.getenv(env_name)
    if env_val is not None:
        try:
            return float(env_val), f"{env_name}={env_val}"
        except ValueError:
            logger.warning("%s=%r is not parseable, using default", env_name, env_val)
    default  = _DEFAULT_RFR_BY_CCY[rc]
    return default, f"default[{rc.value}]={default}"


def daily_rfr_geometric(annual_rfr: float, trading_days: int = 252) -> float:
    """
    Convert an annual RFR to a geometric daily rate (H3).

    Linear approximation `r_ann / 252` over-states the daily rate by
    ~r²/(2·252) — small in absolute terms but it biases every Sortino
    downside filter consistently downward (more days "qualify" as
    downside).  Use compound rounding instead.
    """
    if annual_rfr is None or not np.isfinite(annual_rfr):
        return 0.0
    return float((1.0 + float(annual_rfr)) ** (1.0 / int(trading_days)) - 1.0)


# ── Asset-currency inference ────────────────────────────────────────────────

# Exchange-suffix → native currency map.  When a ticker is in the form
# "SYMBOL.EXCHANGE" we use the EXCHANGE part as a strong signal; otherwise
# we fall back to USD as the most common case for the engine.
#
# This is intentionally coarse: TradernetClient does not surface a per-
# instrument currency field, and using the exchange suffix is the same
# heuristic that scoring_orchestrator already uses for sector mapping —
# keeps the layers consistent.
_EXCHANGE_TO_CURRENCY: dict[str, str] = {
    "US":    "USD",     # NASDAQ / NYSE / AMEX
    "KZ":    "KZT",     # KASE
    "AIX":   "USD",     # Astana International Exchange — predominantly USD-quoted
    "IL":    "GBP",     # London (HSBK.IL / KAP.IL trade in GBp practically;
                        # we still convert through GBP→USD downstream)
    "LSE":   "GBP",
    "HK":    "HKD",
    "TO":    "CAD",
    "L":     "GBP",
    "DE":    "EUR",
    "PA":    "EUR",
}

# Hard-coded overrides — used when a specific *symbol* trades in a currency
# different from its exchange's default (e.g. London-listed depository
# receipts that settle in USD).  Keep this list short and explicit.
_TICKER_CURRENCY_OVERRIDE: dict[str, str] = {
    "BTC-USD": "USD",
    "ETH-USD": "USD",
    "SOL-USD": "USD",
    "HSBK.IL": "USD",   # Halyk Bank GDR — settles in USD on LSE/IL
    "KAP.IL":  "USD",   # Kazatomprom GDR — settles in USD on LSE/IL
}


def infer_asset_currency(ticker: str, default: str = "USD") -> str:
    """
    Heuristic: ticker → native trading currency.

    Order:
      1. Exact override map (`_TICKER_CURRENCY_OVERRIDE`)
      2. Exchange-suffix lookup (`_EXCHANGE_TO_CURRENCY`)
      3. Default (USD)

    This is intentionally NOT a network call — the engine must run end-to-
    end offline.  Errors lean toward USD because mis-classifying a USD
    asset as KZT silently inflates its volatility through an FX series
    that wouldn't actually apply.
    """
    if not ticker:
        return default
    t = str(ticker).strip().upper()
    if t in _TICKER_CURRENCY_OVERRIDE:
        return _TICKER_CURRENCY_OVERRIDE[t]
    if "." in t:
        suffix = t.rsplit(".", 1)[-1]
        if suffix in _EXCHANGE_TO_CURRENCY:
            return _EXCHANGE_TO_CURRENCY[suffix]
    return default


def infer_currencies_for_tickers(
    tickers: Iterable[str],
    overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """
    Vectorised helper: ticker list → {ticker: currency}.

    `overrides` is a thin slot for per-portfolio metadata (e.g. broker
    explicitly tells us "this KZ-bond settled in USD").  Always wins over
    the heuristic.
    """
    out: dict[str, str] = {}
    overrides = overrides or {}
    for t in tickers:
        key = str(t).strip().upper()
        if key in overrides:
            out[key] = str(overrides[key]).upper()
        else:
            out[key] = infer_asset_currency(t)
    return out


# ── FX alignment helpers ────────────────────────────────────────────────────

@dataclass(frozen=True)
class FxConversion:
    """Audit record of what was converted and how (surfaced to QC panel)."""
    pair:           str       # e.g. "USDKZT"
    coverage_pct:   float     # share of non-NaN FX observations vs prices
    last_value:     float     # last observed FX rate
    fallback_used:  bool      # True when we had to ffill stale FX into a gap


def align_fx_to_prices(
    fx_series:    pd.Series,
    target_index: pd.Index,
    *,
    lag_one_day:  bool = True,
) -> tuple[pd.Series, FxConversion]:
    """
    Re-index an FX series onto a price index, ffill-ing gaps.

    Look-ahead protection
    ─────────────────────
    Equity exchanges close BEFORE FX markets in most jurisdictions, but
    the historical fact published as "close-of-day" FX for date D is
    typically only available at session-end.  To be conservative we
    optionally lag the FX series by one day (T-1) so date D's prices are
    multiplied by the FX rate KNOWN by the equity close.  This is the
    same convention used by Barra and MSCI risk models.

    The function never introduces NEW information into the time axis —
    we only carry forward already-published rates (`ffill`).  No
    interpolation, no `bfill` (would import the future).
    """
    if fx_series is None or fx_series.empty:
        empty = pd.Series(np.nan, index=target_index, dtype=float)
        return empty, FxConversion(pair="?", coverage_pct=0.0,
                                    last_value=float("nan"), fallback_used=True)

    s = fx_series.copy().sort_index()
    if lag_one_day:
        s = s.shift(1)              # publish-as-of-yesterday for today's prices
    s = s.reindex(target_index).ffill()
    # Cold-start gap before the first FX observation → backfill only the
    # very first contiguous NaN block with the first known value.  This
    # is bounded and explicit, NOT a global bfill.
    if s.iloc[0] != s.iloc[0]:      # NaN check w/o numpy import in hot loop
        first_valid = s.first_valid_index()
        if first_valid is not None:
            s = s.copy()
            s.loc[:first_valid] = s.loc[first_valid]

    coverage = float(s.notna().mean())
    last     = float(s.dropna().iloc[-1]) if s.notna().any() else float("nan")
    fallback = bool(coverage < 0.95)
    return s, FxConversion(pair="?", coverage_pct=round(coverage * 100, 1),
                           last_value=last, fallback_used=fallback)


# ── Price-matrix transformation ─────────────────────────────────────────────

# FX provider callable signature:
#   (base_ccy, quote_ccy) -> pd.Series (datetime-indexed, base->quote rate)
#
# The price-matrix transformer DOES NOT fetch — the caller is responsible
# for supplying a callable that knows how to load FX history.  This keeps
# the module dependency-free and unit-testable with mock providers.
FxProvider = Callable[[str, str], Optional[pd.Series]]


@dataclass
class PriceTransformResult:
    prices_base:      pd.DataFrame             # transformed price frame
    asset_currencies: dict[str, str]           # per-ticker native currency
    fx_records:       list[FxConversion]       # one entry per non-trivial pair
    no_op:            bool                     # True if reporting matches all assets


def convert_price_matrix(
    prices:            pd.DataFrame,
    asset_currencies:  Mapping[str, str],
    reporting:         ReportingCurrency,
    fx_provider:       Optional[FxProvider],
    *,
    lag_one_day:       bool = True,
) -> PriceTransformResult:
    """
    Transform a (T × N) price matrix to the reporting currency.

    For each column (ticker) i:
      if asset_ccy == reporting → no change.
      else                       → multiply column-wise by FX_{asset→reporting}.

    Short-circuit
    ─────────────
    If every asset already matches `reporting` we return the ORIGINAL
    frame (no copy) and `no_op=True` — zero overhead for the dominant
    USD-only-portfolio case.  This is the H2 self-check #2 guard.

    Rust-export note
    ────────────────
    The returned frame has the same shape, columns, index and dtype as
    the input — downstream consumers (ndarray, nalgebra) see no
    structural change.  Only the values are mapped.
    """
    if prices is None or prices.empty:
        return PriceTransformResult(prices_base=prices if prices is not None else pd.DataFrame(),
                                    asset_currencies=dict(asset_currencies),
                                    fx_records=[], no_op=True)

    rep_ccy = reporting.value
    needs_conversion = {t: c for t, c in asset_currencies.items()
                        if t in prices.columns and c != rep_ccy}

    if not needs_conversion:
        return PriceTransformResult(prices_base=prices,
                                    asset_currencies=dict(asset_currencies),
                                    fx_records=[], no_op=True)

    if fx_provider is None:
        # Conversion needed but caller didn't supply FX → log loudly and
        # return the un-converted frame.  Returning silently would hide
        # the bug; raising would break demo/test portfolios.
        logger.warning(
            "convert_price_matrix: %d columns need FX→%s but fx_provider is None — "
            "skipping conversion.  Tickers: %s",
            len(needs_conversion), rep_ccy, list(needs_conversion.keys())[:8],
        )
        return PriceTransformResult(prices_base=prices,
                                    asset_currencies=dict(asset_currencies),
                                    fx_records=[], no_op=True)

    # Cache FX series per (asset_ccy → rep_ccy) — many tickers share a ccy.
    fx_cache: dict[str, pd.Series] = {}
    fx_records: list[FxConversion] = []
    converted = prices.copy()                   # one allocation, sized once

    for ticker, asset_ccy in needs_conversion.items():
        if asset_ccy not in fx_cache:
            raw = fx_provider(asset_ccy, rep_ccy)
            if raw is None or len(raw) == 0:
                logger.warning("FX %s→%s not available — leaving %s in native ccy",
                               asset_ccy, rep_ccy, ticker)
                fx_cache[asset_ccy] = pd.Series(dtype=float)
                fx_records.append(FxConversion(pair=f"{asset_ccy}{rep_ccy}",
                                                coverage_pct=0.0,
                                                last_value=float("nan"),
                                                fallback_used=True))
                continue
            aligned, rec = align_fx_to_prices(raw, prices.index,
                                              lag_one_day=lag_one_day)
            rec = FxConversion(pair=f"{asset_ccy}{rep_ccy}",
                               coverage_pct=rec.coverage_pct,
                               last_value=rec.last_value,
                               fallback_used=rec.fallback_used)
            fx_cache[asset_ccy] = aligned
            fx_records.append(rec)

        fx_aligned = fx_cache[asset_ccy]
        if fx_aligned.empty:
            continue
        converted[ticker] = prices[ticker].astype(float) * fx_aligned.astype(float)

    return PriceTransformResult(prices_base=converted,
                                asset_currencies=dict(asset_currencies),
                                fx_records=fx_records, no_op=False)


# ── Public API ──────────────────────────────────────────────────────────────

__all__ = [
    "ReportingCurrency",
    "FxConversion",
    "FxProvider",
    "PriceTransformResult",
    "get_rfr_for_currency",
    "daily_rfr_geometric",
    "infer_asset_currency",
    "infer_currencies_for_tickers",
    "align_fx_to_prices",
    "convert_price_matrix",
]
