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

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


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

    def classify(self, prices: pd.DataFrame) -> Optional[RegimeReading]:
        """
        Return a RegimeReading or None if there's not enough data.
        Caller (UniversalPortfolioManager) treats None as "regime unknown".
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
]
