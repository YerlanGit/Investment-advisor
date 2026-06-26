"""
Macro-regime classifier (Recovery / Expansion / Slowdown / Recession).

Inputs are the same factor ETFs already fetched by MAC3RiskEngine — no extra
network calls — so classification is essentially free.

Methodology
-----------
We score the macro regime on two orthogonal axes:

  • Growth axis    : are equities leading vs treasuries?
  • Cycle axis     : are cyclicals leading vs defensives?

Each axis is built from short- and medium-window returns of liquid US ETF
proxies that are already loaded by the MAC3 engine.  The sign and magnitude
of these spreads decide which of the four quadrants the market is in:

                    growth_score > 0      growth_score < 0
   cycle_score>0   ┌──────────────────┐ ┌──────────────────┐
                   │   EXPANSION      │ │   RECOVERY       │
                   │  (risk-on,       │ │  (risk-on,       │
                   │   late cycle)    │ │   early cycle)   │
                   └──────────────────┘ └──────────────────┘
   cycle_score<0   ┌──────────────────┐ ┌──────────────────┐
                   │   SLOWDOWN       │ │   RECESSION      │
                   │  (risk-off,      │ │  (risk-off,      │
                   │   late cycle)    │ │   contraction)   │
                   └──────────────────┘ └──────────────────┘

Confidence (0..1) is the magnitude of (growth_score, cycle_score) clipped at 1.

The classifier is deterministic (no RNG) and therefore safe to call from any
context — two identical price frames always produce the same regime label.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ── BLOCK 3.4 — hard-macro overlay tunables (gated, additive) ────────────────
# The ETF-momentum axes above are the deterministic DEFAULT.  When the env
# flag is set AND a FRED macro pack is supplied to classify(), two hard signals
# join the axis component lists at the SAME scale as the ETF spreads.  Crucially
# each signal blends a LEVEL with a RATE-OF-CHANGE (темпы роста/падения) — the
# regime cares far more about the DIRECTION a macro series is moving than its
# absolute reading: unemployment at 4.1% that is RISING is a slowdown signal,
# the same 4.1% FALLING is an expansion signal.  So:
#   • Real GDP growth : level (vs ~2% trend) ⊕ acceleration (Δ vs prior quarter) → growth axis
#   • Unemployment    : level (vs ~4.5% neutral, inverted) ⊕ −trend (RISING = bad) → cycle axis
# When history is available the two halves blend 50/50; with only a spot value
# the level carries full weight.  Each signal is bounded to ±_MACRO_MAX_NUDGE so
# macro can TILT, never OVERRIDE, the price-based regime.  Off by default → the
# classifier stays byte-identical.
MACRO_OVERLAY_ENV     = "REGIME_MACRO_OVERLAY"
_MACRO_MAX_NUDGE      = 0.05     # ≈ a quarter of the 0.20 magnitude reference
_TREND_GDP_GROWTH     = 2.0      # % SAAR — long-run US real-GDP trend
_GDP_SCALE            = 0.020    # 2pp above trend ⇒ +0.04 growth (level half)
_GDP_TREND_LAG        = 3        # obs back (≈ 3 quarters ≥3 changes) for the GDP trend
_GDP_TREND_SCALE      = 0.015    # +2pp trend over the window ⇒ +0.03 growth (trend half)
_NEUTRAL_UNEMPLOYMENT = 4.5      # % — rough full-employment anchor
_UNEMP_SCALE          = 0.030    # 1.5pp below neutral ⇒ +0.045 cycle (level half)
_UNEMP_TREND_LAG      = 3        # obs back (≈ 3 months ≥3 changes) for the U-rate trend
_UNEMP_TREND_SCALE    = 0.040    # +0.5pp rise over the window ⇒ −0.02 cycle (trend half)
# Минимум наблюдений для расчёта темпа: 4 точки = 3 последовательных изменения
# (требование «анализировать темпы минимум 3 изменений, а не 1»).
_TREND_MIN_POINTS     = 4


def _macro_overlay_enabled() -> bool:
    # BLOCK 2 (2026-06-26): DEFAULT-ON.  The regime confirmation must read macro
    # DYNAMICS (rate-of-change of unemployment / GDP), not just price momentum —
    # so the FRED overlay now runs in production whenever a macro pack is
    # supplied.  Escape hatch: REGIME_MACRO_OVERLAY=0 restores the pure
    # price-momentum classifier.  The overlay is still bounded (±_MACRO_MAX_NUDGE)
    # so macro can only TILT, never hijack, the price-based regime.
    return str(os.getenv(MACRO_OVERLAY_ENV, "on")).strip().lower() not in (
        "0", "false", "no", "off", "")


def _usable_macro(series: object) -> Optional[float]:
    """Return a finite value only for a consulted (ok/stale) FRED series."""
    if not isinstance(series, dict):
        return None
    if series.get("status") not in ("ok", "stale"):
        return None
    v = series.get("value")
    if isinstance(v, (int, float)) and np.isfinite(float(v)):
        return float(v)
    return None


def series_trend(values, lag: int, *, min_points: int = _TREND_MIN_POINTS
                 ) -> tuple[Optional[float], Optional[float], int]:
    """
    Multi-point RATE-OF-CHANGE over ≥3 consecutive changes (анализ темпов).

    Instead of a single latest−prior difference (which a one-off print can
    flip), fit a least-squares line across the last ``lag+1`` observations and
    report the MODELLED change over that window.  This needs ≥ ``min_points``
    points (default 4 ⇒ ≥3 changes); with less history it returns
    ``(None, None, n)`` so the caller falls back to a level-only signal.

    Returns ``(total_change_over_window, slope_per_obs, n_points_used)``.
    """
    vals = [float(v) for v in (values or [])
            if isinstance(v, (int, float)) and np.isfinite(float(v))]
    n = len(vals)
    if n < max(min_points, 2):
        return None, None, n
    window = min(int(lag), n - 1)
    if window < min_points - 1:          # fewer than 3 changes available
        return None, None, n
    seg = vals[-(window + 1):]
    x = np.arange(len(seg), dtype=float)
    slope = float(np.polyfit(x, seg, 1)[0])      # OLS slope per observation
    total = slope * (len(seg) - 1)               # modelled Δ across the window
    return total, slope, len(seg)


def _value_and_trend(series: object, lag: int) -> tuple[Optional[float], Optional[float]]:
    """
    (latest_value, windowed_trend) for a usable FRED series.

    The trend is the slope-based change over ≥3 observations (series_trend) —
    NOT a single-step difference — so the regime reacts to a sustained темп,
    not one noisy print.  history_30d holds the last 30 OBSERVATIONS (≈30
    months for monthly UNRATE, ≈30 quarters for GDP).  trend=None when history
    is too short ⇒ caller falls back to level-only.
    """
    v = _usable_macro(series)
    if v is None:
        return None, None
    hist = (series or {}).get("history_30d") or []
    vals = [h.get("value") for h in hist if isinstance(h, dict)]
    total, _slope, _n = series_trend(vals, lag)
    return v, total


def _blend(level: float, momentum: Optional[float]) -> float:
    """Level alone when no momentum is available; otherwise a 50/50 blend."""
    return level if momentum is None else 0.5 * level + 0.5 * momentum


def _macro_nudges(macro: dict) -> tuple[Optional[float], Optional[float], dict]:
    """
    Map a FRED macro pack to (growth_nudge, cycle_nudge, diagnostics).

    Each nudge fuses a LEVEL and a RATE-OF-CHANGE so the overlay reads macro
    momentum, not just the spot print.  `diagnostics` exposes the raw trend
    components for the QC/CoVe panel.
    """
    g_nudge = c_nudge = None
    diag: dict = {}

    # GDP growth: above-trend level ⊕ quarter-on-quarter acceleration.
    gdp, gdp_tr = _value_and_trend((macro or {}).get("gdp_growth"), _GDP_TREND_LAG)
    if gdp is not None:
        lvl = (gdp - _TREND_GDP_GROWTH) * _GDP_SCALE
        mom = None if gdp_tr is None else gdp_tr * _GDP_TREND_SCALE
        g_nudge = float(np.clip(_blend(lvl, mom), -_MACRO_MAX_NUDGE, _MACRO_MAX_NUDGE))
        if gdp_tr is not None:
            diag["macro_gdp_trend"] = round(gdp_tr, 3)

    # Unemployment: below-neutral level (inverted) ⊕ −trend (RISING U = cooling).
    une, une_tr = _value_and_trend((macro or {}).get("unemployment"), _UNEMP_TREND_LAG)
    if une is not None:
        lvl = (_NEUTRAL_UNEMPLOYMENT - une) * _UNEMP_SCALE
        mom = None if une_tr is None else (-une_tr) * _UNEMP_TREND_SCALE
        c_nudge = float(np.clip(_blend(lvl, mom), -_MACRO_MAX_NUDGE, _MACRO_MAX_NUDGE))
        if une_tr is not None:
            diag["macro_unemployment_trend"] = round(une_tr, 3)

    return g_nudge, c_nudge, diag


# Map of regime → action-relevant guidance.  Used by the Scoring engine
# in pillar A. Fundamentals (Macro Alignment).
REGIME_FAVOURED_SECTORS: dict[str, set[str]] = {
    "Recovery":  {"Finance", "Industrials", "Materials", "Semiconductors",
                  "Consumer", "EM_Kazakhstan"},
    "Expansion": {"Technology", "Semiconductors", "Consumer",
                  "Industrials", "Energy"},
    "Slowdown":  {"Healthcare", "Gold", "Silver", "Commodities"},
    "Recession": {"Gold", "Silver", "Healthcare"},
}

REGIME_PENALISED_SECTORS: dict[str, set[str]] = {
    "Recovery":  {"Healthcare", "Gold"},
    "Expansion": {"Gold", "Silver"},
    "Slowdown":  {"Technology", "Semiconductors", "Finance",
                  "Consumer", "Industrials"},
    "Recession": {"Technology", "Semiconductors", "Finance",
                  "Energy", "EM_Kazakhstan"},
}


@dataclass(frozen=True)
class RegimeReading:
    """One-shot reading of the macro regime."""
    regime: str          # 'Recovery' | 'Expansion' | 'Slowdown' | 'Recession'
    confidence: float    # 0..1
    growth_score: float  # signed strength of equity-vs-bond axis
    cycle_score: float   # signed strength of cyclicals-vs-defensives
    signals: dict        # raw inputs for transparency / footer in PDF

    def as_dict(self) -> dict:
        return {
            "regime":       self.regime,
            "confidence":   round(self.confidence, 3),
            "growth_score": round(self.growth_score, 4),
            "cycle_score":  round(self.cycle_score, 4),
            "signals":      {k: round(v, 4) for k, v in self.signals.items()},
        }


class RegimeClassifier:
    """
    Lightweight, deterministic 4-regime classifier driven entirely by the
    factor ETFs that MAC3RiskEngine already loads.

    Required price columns (any subset works — missing signals fall back to
    neutral 0): SPY.US, IEF.US, IWM.US, EEM.US, DBC.US.

    Optional columns (improve confidence when present):
        XLY.US (Discretionary), XLP.US (Staples), XLF.US (Finance),
        XLV.US (Healthcare).
    """

    # Default lookback windows (trading days).
    SHORT_WIN  = 20    # ~1 month
    MEDIUM_WIN = 60    # ~3 months
    LONG_WIN   = 120   # ~6 months — stabiliser

    def classify(self, prices: pd.DataFrame,
                 macro: Optional[dict] = None) -> Optional[RegimeReading]:
        """
        Return a RegimeReading or None if there's not enough data.
        Caller (UniversalPortfolioManager) treats None as "regime unknown".

        ``macro`` (BLOCK 3.4) is the optional FRED pack from
        MacroFeed.get_regime_drivers().  It is consulted ONLY when
        REGIME_MACRO_OVERLAY=1; otherwise the classifier is the unchanged,
        fully deterministic ETF-momentum model.
        """
        if prices is None or prices.empty or len(prices) < self.MEDIUM_WIN + 5:
            return None

        signals: dict[str, float] = {}

        def _ret(col: str, win: int) -> Optional[float]:
            if col not in prices.columns:
                return None
            series = prices[col].dropna()
            if len(series) < win + 1:
                return None
            return float(series.iloc[-1] / series.iloc[-win - 1] - 1.0)

        # ── Growth axis: equity-leadership and EM appetite ────────────────
        spy_60   = _ret("SPY.US", self.MEDIUM_WIN)
        ief_60   = _ret("IEF.US", self.MEDIUM_WIN)
        eem_60   = _ret("EEM.US", self.MEDIUM_WIN)
        spy_120  = _ret("SPY.US", self.LONG_WIN)

        growth_components = []
        if spy_60 is not None and ief_60 is not None:
            # Equities outperforming bonds → growth on
            growth_components.append(spy_60 - ief_60)
            signals["spy_vs_ief_60d"] = spy_60 - ief_60
        if eem_60 is not None:
            growth_components.append(eem_60)
            signals["eem_60d"] = eem_60
        if spy_120 is not None:
            # Long-window stabiliser (small weight)
            growth_components.append(0.5 * spy_120)
            signals["spy_120d"] = spy_120
        growth_score = float(np.mean(growth_components)) if growth_components else 0.0

        # ── Cycle axis: cyclicals leadership and small-vs-large ────────────
        xly_60 = _ret("XLY.US", self.MEDIUM_WIN)
        xlp_60 = _ret("XLP.US", self.MEDIUM_WIN)
        iwm_60 = _ret("IWM.US", self.MEDIUM_WIN)
        spy_60_for_cycle = spy_60

        cycle_components = []
        if xly_60 is not None and xlp_60 is not None:
            # Discretionary outperforming Staples → cyclical risk-on
            cycle_components.append(xly_60 - xlp_60)
            signals["xly_vs_xlp_60d"] = xly_60 - xlp_60
        if iwm_60 is not None and spy_60_for_cycle is not None:
            # Small caps outperforming large → early-cycle leadership
            cycle_components.append(iwm_60 - spy_60_for_cycle)
            signals["iwm_vs_spy_60d"] = iwm_60 - spy_60_for_cycle
        cycle_score = float(np.mean(cycle_components)) if cycle_components else 0.0

        # ── BLOCK 3.4: optional hard-macro overlay (gated, additive) ──────
        # Fold GDP-growth / unemployment in as extra components at the SAME
        # scale as the ETF spreads, then recompute the axis means.  Because
        # they enter the component LISTS, the magnitude + directional-agreement
        # confidence math below picks them up with no special-casing.  Bounded
        # and off-by-default, so this can tilt but never hijack price action.
        if macro and _macro_overlay_enabled():
            g_nudge, c_nudge, macro_diag = _macro_nudges(macro)
            if g_nudge is not None:
                growth_components.append(g_nudge)
                signals["macro_gdp_growth_nudge"] = round(g_nudge, 4)
            if c_nudge is not None:
                cycle_components.append(c_nudge)
                signals["macro_unemployment_nudge"] = round(c_nudge, 4)
            signals.update(macro_diag)   # raw Δ components for QC transparency
            growth_score = float(np.mean(growth_components)) if growth_components else 0.0
            cycle_score  = float(np.mean(cycle_components))  if cycle_components else 0.0

        # If absolutely no inputs were available, bail.
        if not signals:
            return None

        # ── Quadrant decision ─────────────────────────────────────────────
        # Maps the (growth, cycle) plane to the four regimes exactly as
        # drawn in the module docstring above:
        #   cycle>0  ·  growth>0 → EXPANSION   (risk-on, late cycle)
        #   cycle>0  ·  growth<0 → RECOVERY    (risk-on, early cycle)
        #   cycle<0  ·  growth>0 → SLOWDOWN    (risk-off, late cycle)
        #   cycle<0  ·  growth<0 → RECESSION   (risk-off, contraction)
        # The prior code swapped RECOVERY and SLOWDOWN, so days with
        # SPY>IEF (growth+) and XLY<XLP (cycle−) were mis-labelled "Recovery"
        # instead of "Slowdown" — economically opposite signals.
        if   cycle_score >= 0 and growth_score >= 0:
            regime = "Expansion"
        elif cycle_score >= 0 and growth_score <  0:
            regime = "Recovery"
        elif cycle_score <  0 and growth_score >= 0:
            regime = "Slowdown"
        else:
            regime = "Recession"

        # Confidence is a product of two independent factors:
        #
        #   (a) MAGNITUDE — how far from the origin in the (growth, cycle)
        #       plane is the macro state.  A typical "clear" regime sits at
        #       magnitude ≈ 0.20; below that we should not claim certainty.
        #       Linear ramp 0 → MAGNITUDE_REF (0.20) maps to 0 → 1.0.
        #
        #   (b) DIRECTIONAL AGREEMENT — what share of the underlying signal
        #       components actually point into the assigned quadrant.  If
        #       three out of four 60-day spreads agree the assignment is
        #       robust; if only one agrees, the regime is a coin-flip.
        #
        # Combined: confidence = magnitude_factor × directional_agreement.
        # This avoids the prior failure mode where a barely-positive
        # (growth, cycle) = (0.15, 0.04) blew up to 100% just because
        # magnitude/0.10 saturated the ratio.
        MAGNITUDE_REF = 0.20
        magnitude         = float(np.hypot(growth_score, cycle_score))
        magnitude_factor  = float(min(1.0, magnitude / MAGNITUDE_REF))

        all_components = list(growth_components) + list(cycle_components)
        if all_components:
            # Assigned quadrant signs the engine just chose:
            g_sign = 1.0 if growth_score >= 0 else -1.0
            c_sign = 1.0 if cycle_score  >= 0 else -1.0
            agree  = 0
            total  = 0
            for v in growth_components:
                total += 1
                if (v >= 0 and g_sign > 0) or (v < 0 and g_sign < 0):
                    agree += 1
            for v in cycle_components:
                total += 1
                if (v >= 0 and c_sign > 0) or (v < 0 and c_sign < 0):
                    agree += 1
            directional_agreement = agree / total if total else 0.5
        else:
            directional_agreement = 0.5

        confidence = float(magnitude_factor * directional_agreement)
        # Expose the two factors so the report's QC panel can show WHY
        # confidence is high or low, instead of a single opaque number.
        signals["confidence_magnitude"]   = round(magnitude_factor, 3)
        signals["confidence_directional"] = round(directional_agreement, 3)

        return RegimeReading(
            regime       = regime,
            confidence   = confidence,
            growth_score = growth_score,
            cycle_score  = cycle_score,
            signals      = signals,
        )


__all__ = [
    "RegimeClassifier",
    "RegimeReading",
    "REGIME_FAVOURED_SECTORS",
    "REGIME_PENALISED_SECTORS",
    "series_trend",
]
