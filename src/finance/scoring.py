"""
4-pillar Scoring (Fundamentals / Valuations / Technicals / Credit) plus the
robust Z-score utility shared by all four pillars.

Total Score is on the −6..+6 scale.  Pillars contribute:
    F  : −2..+2
    V  : −2..+2
    T  : −2..+2  (clipped sum from technicals.compute_technicals)
    C  : −2..+1  (asymmetric — credit caps the upside)

Action mapping:
    Total ≥ +3       → 'Strong Buy'
    +1 ≤ Total < +3  → 'Buy'
    −1 ≤ Total < +1  → 'Hold'
    −2 ≤ Total < −1  → 'Trim'
            Total < −2 → 'Sell'

A 🔥 Hotspot override exists at the orchestrator level: any asset with
Euler TRC > 20% is forced to Trim (or harsher) regardless of score.

Z-score
-------
We use a *robust* z-score:  z = (x − median) / (1.4826 · MAD)
which is far less sensitive to outliers than mean/std.  Clipped at ±3
so a single bizarre filing cannot explode a downstream score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ── Z-score utility ──────────────────────────────────────────────────────────

def robust_z(value: Optional[float],
             reference: Sequence[float],
             clip: float = 3.0) -> Optional[float]:
    """
    Robust z-score using median + Median Absolute Deviation (MAD).

    Returns None when:
      • value is None / NaN
      • reference has fewer than 5 finite samples
      • MAD is exactly zero (degenerate distribution)
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None

    arr = np.asarray([x for x in reference if x is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return None

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826  # MAD → σ-equivalent
    if mad <= 0.0:
        return None

    z = (v - med) / mad
    return float(np.clip(z, -clip, clip))


# ── Pillar A — Fundamentals score (−2..+2) ───────────────────────────────────

def fundamentals_score(*,
                       roe_z: Optional[float],
                       op_margin_z: Optional[float],
                       debt_to_assets_z: Optional[float],
                       revenue_growth_z: Optional[float],
                       fcf_margin_z: Optional[float] = None,
                       macro_alignment: float = 0.0) -> float:
    """
    Aggregate sector-relative fundamental Z-scores into a -2..+2 contribution.

    Each Z-score component clips at ±1 per metric so no single line item can
    dominate.  Debt/Assets is sign-flipped (lower leverage is better).

      ROE Z   :  +1 / −1
      OpM Z   :  +1 / −1
      D/A Z   :  +0.5 / −1   (asymmetric — high leverage punished harder)
      RG Z    :  +1 / −1
      FCF Z   :  +1 / −1     (only when SEC FCF margin is available)
      Macro   :  ±0.5         (regime alignment from regime.py)

    Final value clipped to -2..+2.
    """
    s = 0.0
    if roe_z is not None:
        if roe_z >  1: s += 1.0
        if roe_z < -1: s -= 1.0
    if op_margin_z is not None:
        if op_margin_z >  1: s += 1.0
        if op_margin_z < -1: s -= 1.0
    if debt_to_assets_z is not None:
        # Lower leverage = better → flip sign.
        if debt_to_assets_z < -0.5: s += 0.5
        if debt_to_assets_z >  1:   s -= 1.0
    if revenue_growth_z is not None:
        if revenue_growth_z >  1: s += 1.0
        if revenue_growth_z < -1: s -= 1.0
    if fcf_margin_z is not None:
        if fcf_margin_z >  1: s += 1.0
        if fcf_margin_z < -1: s -= 1.0
    s += float(np.clip(macro_alignment, -0.5, 0.5))
    return float(np.clip(s, -2.0, 2.0))


# ── Pillar B — Valuations score (−2..+2) ─────────────────────────────────────

def valuations_score(*,
                     pe_history_z: Optional[float] = None,
                     pb_sector_z:  Optional[float] = None,
                     ev_ebitda_z:  Optional[float] = None) -> float:
    """
    Build the Valuation pillar from Z-scores.

      pe_history_z  : Z of current trailing P/E vs the ticker's own 5-year history
      pb_sector_z   : Z of P/B vs sector cross-section
      ev_ebitda_z   : Z of EV/EBITDA vs sector cross-section

    Each metric:  Z < −1.5 → +X cheap;  Z > +1.5 → −X expensive.
    Returns 0 (neutral) when all inputs are None.
    """
    s = 0.0
    if pe_history_z is not None:
        if pe_history_z < -1.5: s += 2.0
        elif pe_history_z < -0.5: s += 1.0
        elif pe_history_z >  1.5: s -= 2.0
        elif pe_history_z >  0.5: s -= 1.0
    if pb_sector_z is not None:
        if pb_sector_z < -1: s += 1.0
        if pb_sector_z >  1: s -= 1.0
    if ev_ebitda_z is not None:
        if ev_ebitda_z < -1: s += 1.0
        if ev_ebitda_z >  1: s -= 1.0
    return float(np.clip(s, -2.0, 2.0))


# ── Pillar C — Technicals score is produced by technicals.compute_technicals ─
# (already in the −2..+2 range; re-exported here for completeness)

def technicals_score_from_reading(reading) -> float:
    if reading is None:
        return 0.0
    return float(np.clip(reading.score, -2.0, 2.0))


# ── Pillar D — Credit score (−2..+1) ─────────────────────────────────────────

def credit_score(*,
                 cds_bps: Optional[float] = None,
                 cds_change_7d: Optional[float] = None,
                 altman_zone: Optional[str] = None,
                 piotroski_f: Optional[int] = None,
                 interest_coverage: Optional[float] = None) -> float:
    """
    Build the Credit pillar from CDS + SEC-derived signals.

    CDS rules (when CDS data is available and quality-gated):
      <  40 bps                              → +1   Safe Haven
      40–90 bps                              →  0   Neutral
      90–150 bps                             → −1   Elevated risk
      ≥ 150 bps OR Δ7d > +20%                → −2   Distress

    Altman zone:
      'Safe'     → +1
      'Grey'     →  0
      'Distress' → −1

    Piotroski F-score (out of 9):
      F ≥ 7 → +0.5
      F ≤ 3 → −0.5

    Interest coverage:
      > 5×  → +0.5
      < 1.5× → −1.0

    Asymmetric: capped at +1, floored at −2.
    """
    s = 0.0

    if cds_bps is not None:
        if cds_change_7d is not None and cds_change_7d > 0.20:
            s -= 2.0
        elif cds_bps >= 150:
            s -= 2.0
        elif cds_bps >= 90:
            s -= 1.0
        elif cds_bps < 40:
            s += 1.0

    if altman_zone == "Safe":
        s += 1.0
    elif altman_zone == "Distress":
        s -= 1.0

    if piotroski_f is not None:
        if piotroski_f >= 7: s += 0.5
        elif piotroski_f <= 3: s -= 0.5

    if interest_coverage is not None:
        if interest_coverage > 5.0:   s += 0.5
        elif interest_coverage < 1.5: s -= 1.0

    return float(np.clip(s, -2.0, 1.0))


# ── Total Score → Action ─────────────────────────────────────────────────────

def action_from_total(total_score: float, *, hotspot: bool = False) -> str:
    """
    Map a clipped total score (-6..+6) to an action label.
    The Hotspot override forces Trim (or worse) on heavily concentrated assets.
    """
    if hotspot and total_score > -1:
        return "Trim"        # forced override — concentration trumps score
    if total_score >= 3:
        return "Strong Buy"
    if total_score >= 1:
        return "Buy"
    if total_score > -1:
        return "Hold"
    if total_score > -2:
        return "Trim"
    return "Sell"


# ── Orchestrator output ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class AssetScore:
    ticker: str
    fundamentals: float
    valuations:   float
    technicals:   float
    credit:       float
    total:        float       # clipped -6..+6
    action:       str
    hotspot:      bool


def total_score(fundamentals: float, valuations: float,
                technicals: float, credit: float) -> float:
    return float(np.clip(fundamentals + valuations + technicals + credit, -6.0, 6.0))


__all__ = [
    "robust_z",
    "fundamentals_score",
    "valuations_score",
    "technicals_score_from_reading",
    "credit_score",
    "total_score",
    "action_from_total",
    "AssetScore",
]
