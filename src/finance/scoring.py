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


# ── Hotspot threshold — SINGLE SOURCE OF TRUTH ──────────────────────────────
# Sprint-5.1 (S4): this 20% Euler-TRC cut-off used to live as TWO independent
# literals (scoring_orchestrator.DEFAULT_HOTSPOT_TRC_PCT and gatekeeper
# DEFAULT_LIMITS["max_euler_risk_pct"]) — tuning one silently desynced the
# other (the 🔥 flag in the report vs the GK-1 audit rule).  Both now import
# THIS constant.
HOTSPOT_TRC_PCT: float = 20.0


# ── Numeric hygiene (BLOCK 4.7) ──────────────────────────────────────────────
# Outlier / Inf / NaN protection shared by the pillar aggregators.  robust_z
# already clips per-metric Z to ±3, but two downstream aggregators trusted
# their inputs:
#   • composite_risk_score relied on Python's min()/max() to clamp — and
#     `max(0.0, float('nan'))` is ORDER-DEPENDENT (returns 0.0 here only by
#     luck of argument order), so a NaN vol/CVaR could silently mis-score.
#   • fundamentals_score added np.clip(macro_alignment, …) straight into the
#     sum — a NaN macro tilt propagated to a NaN pillar and a NaN Total.
# `_finite` coerces any non-finite (NaN/±Inf) value to a safe default so a
# single corrupt upstream number can never poison the score.

def _finite(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(f) else f


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
    # BLOCK 4.7: coerce a non-finite macro tilt to 0 BEFORE it enters the sum —
    # otherwise a NaN regime-alignment poisons the whole pillar (and Total).
    s += float(np.clip(_finite(macro_alignment, 0.0), -0.5, 0.5))
    return float(np.clip(s, -2.0, 2.0))


# ── Pillar B — Valuations score (−2..+2) ─────────────────────────────────────

def valuations_score(*,
                     pe_z:         Optional[float] = None,
                     pb_sector_z:  Optional[float] = None,
                     ev_ebitda_z:  Optional[float] = None,
                     fcf_yield_z:  Optional[float] = None,
                     # Back-compat: older callers used pe_history_z.  Treated
                     # as an alias so we don't break any external caller.
                     pe_history_z: Optional[float] = None) -> float:
    """
    Build the Valuation pillar from Z-scores.

      pe_z         : Z of P/E vs sector (or history, depending on the
                     orchestrator's choice — both use the SAME ±1 scale)
      pb_sector_z  : Z of P/B vs sector cross-section
      ev_ebitda_z  : Z of EV/EBITDA (or EV/EBIT proxy) vs sector
      fcf_yield_z  : Z of FCF YIELD (fcf / market cap) vs sector.  Sprint-5
                     Task 6 quality signal — note the INVERTED direction: a
                     HIGH FCF yield is CHEAP/quality, so a positive z ADDS to
                     the score (mirrors the cheap end of a price multiple).

    "Price" metrics (P/E, P/B, EV/EBITDA) contribute +1 when CHEAP (Z < −1) and
    −1 when EXPENSIVE (Z > +1).  The FCF-yield is a "yield" metric, so its sign
    is flipped (high yield Z > +1 → +1).  Each metric is on the SAME ±1 scale
    so none dominates.  Returns 0 (neutral) when all inputs are None.  Clipped
    to [-2, +2].
    """
    pe = pe_z if pe_z is not None else pe_history_z
    s = 0.0
    if pe is not None:
        if pe < -1: s += 1.0
        if pe >  1: s -= 1.0
    if pb_sector_z is not None:
        if pb_sector_z < -1: s += 1.0
        if pb_sector_z >  1: s -= 1.0
    if ev_ebitda_z is not None:
        if ev_ebitda_z < -1: s += 1.0
        if ev_ebitda_z >  1: s -= 1.0
    if fcf_yield_z is not None:
        # Yield metric — sign inverted vs price multiples (high yield = cheap).
        if fcf_yield_z >  1: s += 1.0
        if fcf_yield_z < -1: s -= 1.0
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
    # When False, the Credit pillar is conceptually not applicable to this
    # asset class (e.g. physical commodities, sovereign-bond ETFs).  The
    # template renders an em-dash for the C cell and DOES NOT include C in
    # the visible total's denominator.  Default True keeps every existing
    # caller / test compatible.
    credit_applicable: bool = True
    # Same idea for the Fundamentals pillar: physical commodities and
    # sovereign-rate ETFs have no financial statements, so F is N/A (em-dash)
    # rather than a misleading regime-tilt number.
    fundamentals_applicable: bool = True


def total_score(fundamentals: float, valuations: float,
                technicals: float, credit: float) -> float:
    return float(np.clip(fundamentals + valuations + technicals + credit, -6.0, 6.0))


# ── Composite Risk Score (0..100) — SINGLE SOURCE OF TRUTH ────────────────────
# Previously duplicated in MAC3RiskEngine._composite_risk_score AND
# simulate._composite_risk_score with a "keep in sync!" comment — a textbook
# DRY violation + drift risk.  Both now delegate here.  Kept sklearn-free so
# the simulator and tests never drag the engine class into scope.

# H4 — risk-mandate calibration.  A conservative investor is more
# tail-sensitive (weights CVaR higher AND uses a tighter CVaR base, so the
# same tail produces a higher perceived-risk score); an aggressive investor
# tolerates more tail (lower CVaR weight + looser base) and cares more about
# raw volatility.  Each row's three weights sum to exactly 1.0.
_RISK_MANDATE_MATRIX: dict[str, dict[str, float]] = {
    #                w_cvar  w_vol  w_erc   cvar_base (divisor → 100)
    "CONSERVATIVE": {"w_cvar": 0.60, "w_vol": 0.30, "w_erc": 0.10, "cvar_base": 0.03},
    # H1 recalibration: MODERATE cvar_base widened 0.05 → 0.065 so the
    # CVaR component no longer dominates the gauge for quality-tech books.
    # The previous 0.05 base pushed long-running portfolios from index 48 → 75
    # for the same risk profile, which read as a phantom spike to users.
    "MODERATE":     {"w_cvar": 0.40, "w_vol": 0.40, "w_erc": 0.20, "cvar_base": 0.065},
    "AGGRESSIVE":   {"w_cvar": 0.20, "w_vol": 0.50, "w_erc": 0.30, "cvar_base": 0.08},
}

# Vol / concentration normalisation scales are mandate-independent.
_VOL_BASE = 0.40    # 40% annual vol  = 100
_ERC_BASE = 50.0    # 50% single-asset concentration = 100


def normalize_risk_mandate(profile) -> str:
    """
    Map a user profile (RU/EN name, or numeric risk-score) to one of the
    three canonical mandates.  Defaults to MODERATE for anything unknown.
    """
    if profile is None:
        return "MODERATE"
    if isinstance(profile, (int, float)):
        s = float(profile)        # onboarding risk-score 6..18
        if s <= 8:  return "CONSERVATIVE"
        if s <= 12: return "MODERATE"
        return "AGGRESSIVE"
    p = str(profile).strip().upper()
    if p in _RISK_MANDATE_MATRIX:
        return p
    if "КОНСЕРВАТ" in p or "CONSERV" in p:
        return "CONSERVATIVE"
    if "АГРЕССИВ" in p or "AGGRESS" in p:
        # "Умеренно-агрессивный" leans MODERATE; pure "Агрессивный" → AGGRESSIVE.
        return "MODERATE" if ("УМЕРЕНН" in p or "MODERAT" in p) else "AGGRESSIVE"
    return "MODERATE"


def composite_risk_score(volatility: float, cvar: float,
                         max_erc_pct: float, mandate: str = "MODERATE") -> int:
    """
    Blend three independent risk signals into a single 0..100 gauge,
    calibrated to the investor's risk mandate (see _RISK_MANDATE_MATRIX).

      • Vol     normalised by 0.40 (40% annual vol = 100)
      • |CVaR|  normalised by mandate cvar_base (0.03 / 0.065 / 0.08)
      • maxERC  normalised by 50   (50% concentration = 100)

    Deterministic — identical inputs + mandate always produce the same
    score.  Unknown mandate falls back to MODERATE.
    """
    m = _RISK_MANDATE_MATRIX.get(str(mandate).strip().upper(),
                                 _RISK_MANDATE_MATRIX["MODERATE"])

    def _norm(x: float, scale: float) -> float:
        return min(100.0, max(0.0, (x / scale) * 100.0)) if scale > 0 else 0.0

    # BLOCK 4.7: sanitize the three inputs to finite values FIRST.  A NaN here
    # used to survive min()/max() in an argument-order-dependent way; +Inf now
    # deterministically saturates the gauge instead of relying on clamp luck.
    s_vol  = _norm(_finite(volatility),       _VOL_BASE)
    s_cvar = _norm(abs(_finite(cvar)),        m["cvar_base"])
    s_conc = _norm(_finite(max_erc_pct),      _ERC_BASE)
    return int(round(m["w_vol"] * s_vol + m["w_cvar"] * s_cvar + m["w_erc"] * s_conc))


# ── Asset-class display label — SINGLE SOURCE OF TRUTH ───────────────────────
# Consolidates the two divergent DISPLAY classifiers that previously lived in
# pdf_payload._classify_asset (suffix-aware, correct) and
# tg_bot._classify_asset (substring `any(x in t)` — buggy: "BNB"⊂"…BNB…",
# "BOND" false-positives, bare ticker → "Акции").  This is the accurate,
# suffix-aware version; both call sites now import it.
#
# NOTE: this is intentionally SEPARATE from:
#   • gatekeeper._classify_to_asset_key  → English limits keys (Stocks_US…)
#   • broker_api._classify_instrument    → instrument TYPE via Tradernet t_field
# Those have different output contracts and purposes; merging them would
# break the limits lookup / broker-metadata typing.

_CASH_BASES       = frozenset({"USD", "EUR", "RUB", "KZT", "CASH"})
_CRYPTO_BASES     = frozenset({"BTC", "ETH", "SOL", "BNB", "DOGE"})
_COMMODITY_BASES  = frozenset({"GLD", "SLV", "GDX", "USO", "DBC", "PDBC",
                               "GOLD", "SILVER", "OIL"})
_BOND_BASES       = frozenset({"TLT", "AGG", "BND", "LQD", "HYG", "IEF",
                               "BIL", "EMB", "SHY"})
# KZ blue chips that may arrive WITHOUT an exchange suffix (the tg_bot path
# saw bare "KSPI"/"KAP"); classify them as KZ equities regardless.
_KZ_BASES         = frozenset({"KAP", "KSPI", "HSBK", "KZTK", "KCEL",
                               "BAST", "HRGL", "KZAP", "KZTO"})


def classify_asset_class(ticker: str) -> str:
    """
    Asset-class label for user-facing tables/cards.

    Returns one of: 'Ден. средства' | 'Крипто' | 'Сырьё' | 'Облигации'
                    | 'Акции KZ' | 'Акции США' | 'Прочее'.
    Suffix-aware and exact-match based — no substring false positives.
    """
    t = (ticker or "").upper().strip()
    base   = t.split(".")[0] if "." in t else t
    suffix = t.rsplit(".", 1)[-1] if "." in t else ""

    if base in _CASH_BASES:
        return "Ден. средства"
    if base in _CRYPTO_BASES or t.endswith("-USD"):
        return "Крипто"
    if base in _COMMODITY_BASES:
        return "Сырьё"
    if base in _BOND_BASES or "BOND" in base or "OVD" in base \
       or t.startswith(("KZ2P", "KZ1P", "XS", "US912")):
        return "Облигации"
    if suffix in {"KZ", "IL"} or t.endswith(".AIX") or "FFSPC" in base \
       or base in _KZ_BASES:
        return "Акции KZ"
    if suffix == "US" or len(base) <= 5:
        return "Акции США"
    return "Прочее"


__all__ = [
    "robust_z",
    "fundamentals_score",
    "valuations_score",
    "technicals_score_from_reading",
    "credit_score",
    "total_score",
    "action_from_total",
    "composite_risk_score",
    "normalize_risk_mandate",
    "classify_asset_class",
    "AssetScore",
]
