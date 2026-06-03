"""
Parametric stress scenarios using the factor model already fitted by MAC3.

Approach
────────
Each scenario is declared as a SHOCK VECTOR — a per-factor period-decimal
return shock (e.g. ``{"Market": -0.10, "Momentum": -0.15}`` meaning "the
Market factor returns -10% and Momentum -15% over the stress horizon").

The PnL impact on each position is the linear combination of its factor
betas with the shock vector:

    ΔPnL_i / V_i  = Σ_f β_{i,f} · shock_f
    ΔPnL_i       = w_i · Σ_f β_{i,f} · shock_f          (as fraction of total)
    ΔPnL_port    = Σ_i ΔPnL_i                            = w' · B · shock

This is the industry-standard quick-stress technique (used by, among many
others, Goldman Sachs MarketRisk and BlackRock Aladdin).  It is a LINEAR
approximation that assumes the betas remain stable through the shock; this
is acceptable for shocks of single-digit-percent magnitude but should not
be over-interpreted for severe (> 25%) tail events.

Drawdown and recovery estimates layer on top of the gross shock PnL:

  • intra-period max drawdown estimate
        est_dd = port_pct − σ_quarterly          (only when port_pct < 0)
        σ_quarterly = σ_annual · √(1/4)
    Rationale: even when the QUARTER ends at shock_pnl, the WORST DAY
    inside the quarter is typically ~1·σ_quarterly deeper than the close.

  • recovery_months ≈ |max_dd| / expected_monthly_return
    where expected_monthly_return = LONG_RUN_ANN_RETURN / 12 (default 8%/y).
    This is intentionally generic (long-run market average); per-portfolio
    expected returns from Black-Litterman can be substituted by the caller
    if desired.

Catalog Coverage
────────────────
The default catalog defines 7 scenarios.  Five use ONLY factors that the
engine fits today (Market / Momentum / Value / Quality / Size / Rates /
Commodities / EM_Equity / EM_Bond).  Two — "USD +5% rally" and "CPI shock"
— ideally need a dedicated USD-factor (UUP.US) and Inflation-factor
(TIP.US) respectively; until those are added to ``MAC3RiskEngine.
factor_tickers``, they fall back to proxy mappings that are marked
``coverage="proxy"`` so the report can flag them visually.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ── Long-run market average used for recovery-time estimate ─────────────────
# Source: long-run US equity total return ~8-10% nominal; we use 8% as a
# conservative round number.  Caller can override.
DEFAULT_ANN_RETURN_FOR_RECOVERY: float = 0.08

# Drawdown heuristic: intra-period worst-day is roughly 1·σ_period deeper
# than the period close.  σ_period for a quarter = σ_ann · √0.25 = σ_ann / 2.
QUARTER_FRACTION_OF_YEAR: float = 0.25

# ── H4: Convexity cap for per-asset stress impact ────────────────────────────
# Linear β-shock works while shocks are small (single-digit %), but a
# β=2.1 semiconductor name under a "Market -10%, Momentum -15%" scenario
# computes to a -25% one-shot move — and the linear extrapolation
# arithmetic is meaningless at that scale (betas are estimated on
# normal-regime returns, they decompress non-linearly in tails).
#
# We apply a smooth, monotonic, C^∞ saturation above
# CONVEXITY_THRESHOLD (|20%|) that asymptotically approaches HARD_CAP
# (|35%|).  Below the threshold the function is the identity (no
# behaviour change for moderate scenarios).
#
# Functional form (sign-preserving):
#   x' = sign(x) · (T + (C − T) · (1 − exp(−(|x| − T) / (C − T))))     if |x| > T
#   x' = x                                                              otherwise
# Properties:
#   • Continuous in value and derivative at |x| = T (derivative = 1).
#   • Strictly monotonic in x.
#   • lim_{|x|→∞} |x'| = C  (asymptotic; no actual hard kink at C).
CONVEXITY_THRESHOLD: float = 0.20    # below this — linear pass-through
CONVEXITY_HARD_CAP:  float = 0.35    # asymptotic absolute cap

STRESS_TEST_DISCLAIMER: str = (
    "Стресс-сценарии используют линейное приближение β-факторов, "
    "стабильное для шоков до ±20%. Для активов, чья модельная просадка "
    "превышает 20%, применяется выпуклое сглаживание с асимптотой ±35% — "
    "это компенсирует нестабильность бет в хвостах, но в реальном "
    "тейл-событии возможны более глубокие потери. Сценарии — "
    "иллюстративный stress-test, не прогноз."
)


def _convex_cap(x: float,
                threshold: float = CONVEXITY_THRESHOLD,
                hard_cap:  float = CONVEXITY_HARD_CAP) -> float:
    """
    Smooth convex saturation above `threshold`, asymptotic to ±hard_cap.

    Identity for |x| ≤ threshold.  No kink at the threshold (C^1 smooth).
    Exported for unit tests.
    """
    if not math.isfinite(x):
        return x
    a = abs(x)
    if a <= threshold:
        return x
    extra_room = hard_cap - threshold
    if extra_room <= 0:
        return math.copysign(hard_cap, x)
    overshoot = a - threshold
    saturated = extra_room * (1.0 - math.exp(-overshoot / extra_room))
    return math.copysign(threshold + saturated, x)


# ── Default scenario catalog ────────────────────────────────────────────────
# Each scenario declares a shock vector (period decimal returns) keyed by
# factor name.  Factor names MUST match `engine.factor_tickers` keys, which
# in turn match the `Beta_<Name>` columns produced by the regression.
#
# Magnitudes calibrated against realised historical analogues:
#   • Tech sell-off Q2 2022: SPX -16% / NDX -22% → Market -10%, Momentum -15%
#   • Credit blow-out 2008 / 2020 / 2023: HY OAS +200-400 bps → IEF -2%
#   • Fed +50bps surprise: typical equity wobble -2%
#   • Geopolitical risk-off: EM hit hardest (-12%), broad market -7%
#   • Fed cut surprise: equity rally +3%, IEF rally +0.5%
# Magnitudes are documented inline so they can be re-calibrated by hand
# against new historical events.

@dataclass(frozen=True)
class ScenarioSpec:
    name:      str
    shocks:    dict[str, float]               # {factor_name: period_decimal_shock}
    coverage:  str          = "direct"        # "direct" | "proxy"
    note:      str          = ""              # human-readable rationale / source
    category:  str          = "equity"        # "equity" | "rates" | "credit" | "macro" | "geo"


DEFAULT_SCENARIOS: list[ScenarioSpec] = [
    ScenarioSpec(
        name     = "Tech sell-off (как Q2 2022)",
        category = "equity",
        shocks   = {"Market": -0.10, "Momentum": -0.15, "Quality": -0.05},
        note     = "broad equity -10% + momentum factor much harder hit; "
                   "calibrated against SPX -16% / NDX -22% Q2 2022",
    ),
    ScenarioSpec(
        name     = "Credit blow-out (+200 bps HY)",
        category = "credit",
        shocks   = {"Rates": +0.02, "Market": -0.05, "Quality": +0.03,
                    "EM_Bond": -0.08},
        note     = "HY OAS +200 bps → IEF -2% from rate spike; quality bid; "
                   "EM credit hit hardest (2008/2020/2023 analogues)",
    ),
    ScenarioSpec(
        name     = "Fed +50 bps surprise",
        category = "rates",
        shocks   = {"Rates": +0.005, "Market": -0.02},
        note     = "single hawkish surprise; IEF -0.5%, equity wobble -2%",
    ),
    ScenarioSpec(
        name     = "Geopolitical risk-off",
        category = "geo",
        shocks   = {"Market": -0.07, "EM_Equity": -0.12, "EM_Bond": -0.05,
                    "Commodities": +0.05},
        note     = "broad sell-off + EM hit hardest + gold/oil bid; "
                   "calibrated against 2022-02-24 and 2014-Q1 analogues",
    ),
    ScenarioSpec(
        name     = "Fed cut surprise (−50 bps)",
        category = "rates",
        shocks   = {"Rates": -0.005, "Market": +0.03},
        note     = "dovish surprise: rate-sensitive rally; IEF +0.5%, equity +3%",
    ),
    # The next two are PROXY scenarios — they would be more accurate with a
    # dedicated USD factor (UUP.US) and Inflation factor (TIP.US) added to
    # engine.factor_tickers.  Until then, we approximate via EM/Rates betas.
    ScenarioSpec(
        name     = "USD +5% rally",
        category = "macro",
        shocks   = {"EM_Equity": -0.05, "EM_Bond": -0.03, "Commodities": -0.04},
        coverage = "proxy",
        note     = "PROXY (no direct USD factor): strong USD typically -5% EM, "
                   "-3% EM credit, -4% commodities. Would benefit from UUP.US.",
    ),
    ScenarioSpec(
        name     = "CPI shock (+1 пп surprise)",
        category = "macro",
        shocks   = {"Rates": +0.015, "Market": -0.03, "Value": +0.02},
        coverage = "proxy",
        note     = "PROXY (no direct Inflation factor): rates spike +150 bps, "
                   "equity sell-off, value rotates in. Would benefit from TIP.US.",
    ),
]


# ── Core engine ─────────────────────────────────────────────────────────────

def _beta_columns(perf_df: pd.DataFrame) -> list[str]:
    """Return the list of Beta_<Factor> columns actually present in perf_df."""
    return [c for c in perf_df.columns if c.startswith("Beta_")]


def _factor_from_beta_col(col: str) -> str:
    """'Beta_Market' → 'Market'."""
    return col[len("Beta_"):] if col.startswith("Beta_") else col


def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def apply_scenario(perf_df: pd.DataFrame,
                    total_value: float,
                    scenario:    ScenarioSpec,
                    *,
                    port_vol_ann:        float = 0.0,
                    ann_return_baseline: float = DEFAULT_ANN_RETURN_FOR_RECOVERY,
                    ) -> dict:
    """
    Apply a single stress scenario.

    Returns a dict with:
      • name, category, coverage, note            (passthrough from spec)
      • shocks                                    (the input shock vector)
      • port_pct, port_dollar                     (gross shock PnL)
      • max_dd_pct                                (intra-period worst day est.)
      • recovery_months                           (estimate, None for gains)
      • by_asset                                  (list of per-asset contribs)
      • coverage_pct                              (% of shock factors that had
                                                   a matching Beta_ column)
    """
    out: dict = {
        "name":      scenario.name,
        "category":  scenario.category,
        "coverage":  scenario.coverage,
        "note":      scenario.note,
        "shocks":    dict(scenario.shocks),
        "port_pct":           0.0,
        "port_dollar":        0.0,
        "max_dd_pct":         0.0,
        "recovery_months":    None,
        "by_asset":           [],
        "coverage_pct":       0.0,
        "factors_used":       [],
        "factors_missing":    [],
    }
    if perf_df is None or perf_df.empty or total_value <= 0:
        return out

    beta_cols = _beta_columns(perf_df)
    available_factors = {_factor_from_beta_col(c) for c in beta_cols}
    requested_factors = set(scenario.shocks.keys())
    used_factors      = sorted(requested_factors & available_factors)
    missing_factors   = sorted(requested_factors - available_factors)

    out["factors_used"]    = used_factors
    out["factors_missing"] = missing_factors
    out["coverage_pct"]    = (round(len(used_factors) / len(requested_factors) * 100, 1)
                              if requested_factors else 0.0)

    # Per-asset contributions.
    port_pct = 0.0
    by_asset: list[dict] = []
    convexity_applied = 0          # how many positions hit the cap (audit)
    for _, row in perf_df.iterrows():
        ticker = str(row.get("Ticker", "—"))
        cv     = _safe_float(row.get("Current_Value"), 0.0)
        if cv <= 0:
            continue
        w_i = cv / total_value

        # Σ_f β_{i,f} · shock_f for shocks whose factor is present in betas.
        delta_log_ret_raw = 0.0
        for f in used_factors:
            beta_col = f"Beta_{f}"
            beta_val = _safe_float(row.get(beta_col), 0.0)
            delta_log_ret_raw += beta_val * scenario.shocks[f]

        # H4: convex saturation above ±20% with asymptote at ±35%.
        delta_log_ret = _convex_cap(delta_log_ret_raw)
        if abs(delta_log_ret_raw) > CONVEXITY_THRESHOLD:
            convexity_applied += 1
        contrib_pct = w_i * delta_log_ret         # fraction of total portfolio
        port_pct   += contrib_pct
        by_asset.append({
            "ticker":           ticker,
            "weight_pct":       round(w_i * 100, 2),
            "asset_delta_pct":  round(delta_log_ret * 100, 2),    # ΔPnL_i / V_i (after cap)
            "asset_delta_raw":  round(delta_log_ret_raw * 100, 2),# pre-cap, for transparency
            "contrib_pct":      round(contrib_pct * 100, 3),      # of total portfolio
            "contrib_dollar":   round(contrib_pct * total_value, 2),
        })

    # Sort contributions by absolute impact (worst first).
    by_asset.sort(key=lambda d: abs(d["contrib_dollar"]), reverse=True)

    out["port_pct"]    = round(port_pct, 4)              # decimal, e.g. -0.093
    out["port_dollar"] = round(port_pct * total_value, 2)
    out["convexity_applied_n"] = convexity_applied       # H4 audit count

    # Intra-period max drawdown estimate (only for losing scenarios).
    if port_pct < 0:
        sigma_quarter = _safe_float(port_vol_ann, 0.0) * math.sqrt(QUARTER_FRACTION_OF_YEAR)
        est_dd = port_pct - sigma_quarter             # both negative → deeper
        out["max_dd_pct"] = round(est_dd, 4)

        # Recovery: |max_dd| / (annual_return / 12).  Guard against tiny/zero
        # expected return (would produce inf).
        monthly_ret = ann_return_baseline / 12.0
        if monthly_ret > 1e-6:
            out["recovery_months"] = round(abs(est_dd) / monthly_ret, 1)
    else:
        # Positive shock: report the gain itself as max_dd (it's "no drawdown")
        out["max_dd_pct"]      = 0.0
        out["recovery_months"] = None

    out["by_asset"] = by_asset
    return out


def run_stress_scenarios(perf_df:          pd.DataFrame,
                          total_value:      float,
                          port_metrics:     dict,
                          scenarios:        Optional[Iterable[ScenarioSpec]] = None,
                          ann_return_baseline: float = DEFAULT_ANN_RETURN_FOR_RECOVERY,
                          ) -> list[dict]:
    """
    Run the full stress table.

    Args:
        perf_df            : performance table from analyze_all (must contain
                             at least Ticker, Current_Value and Beta_* columns).
        total_value        : total portfolio value.
        port_metrics       : portfolio_metrics dict (used for σ_ann to size
                             intra-period drawdown estimates).
        scenarios          : iterable of ScenarioSpec; defaults to DEFAULT_SCENARIOS.
        ann_return_baseline: long-run annual return used for recovery time.

    Returns:
        list[dict] — one row per scenario in the input order.  Always returns
        a list (possibly empty) so the caller doesn't have to special-case
        None.
    """
    if perf_df is None or perf_df.empty or total_value <= 0:
        return []
    scenarios = list(scenarios) if scenarios is not None else DEFAULT_SCENARIOS
    port_vol_ann = _safe_float((port_metrics or {}).get("Total_Volatility_Ann"), 0.0)
    return [
        apply_scenario(perf_df, total_value, sc,
                        port_vol_ann=port_vol_ann,
                        ann_return_baseline=ann_return_baseline)
        for sc in scenarios
    ]


__all__ = [
    "DEFAULT_ANN_RETURN_FOR_RECOVERY",
    "QUARTER_FRACTION_OF_YEAR",
    "CONVEXITY_THRESHOLD",
    "CONVEXITY_HARD_CAP",
    "STRESS_TEST_DISCLAIMER",
    "ScenarioSpec",
    "DEFAULT_SCENARIOS",
    "apply_scenario",
    "run_stress_scenarios",
    "_convex_cap",
]
