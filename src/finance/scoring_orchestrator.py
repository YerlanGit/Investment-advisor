"""
4-pillar scoring orchestrator.

Takes the rich `analyse_all()` result (performance_table with SEC + factor
extensions, technicals from technicals.compute_technicals, regime reading,
and an optional CDS lookup callable) and produces a per-asset AssetScore
that downstream layers (PDF payload, Action Plan, AI narrative) consume.

This module is import-safe — it has no aiogram / cryptography dependency
and can be unit-tested in isolation.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from finance.regime import (
    RegimeReading,
    REGIME_FAVOURED_SECTORS,
    REGIME_PENALISED_SECTORS,
)
from finance.scoring import (
    AssetScore,
    action_from_total,
    credit_score,
    fundamentals_score,
    robust_z,
    technicals_score_from_reading,
    total_score,
    valuations_score,
)
from finance.technicals import TechnicalReading


# Hotspot threshold — must match Gatekeeper Check 1.
DEFAULT_HOTSPOT_TRC_PCT = 20.0


# ── Asset-class filter for the Credit pillar (D) ─────────────────────────────
# CDS / Altman / Piotroski / Interest-coverage do NOT apply to physical
# commodities (Gold/Silver/Oil/broad Commodities ETFs) or sovereign-rate
# instruments (US Treasuries via TLT/IEF/SHY/AGG/BND/BIL).  When the CDS
# feed has nothing for these and SEC EDGAR returns no filings, the legacy
# pipeline emitted ``credit=0`` which is mathematically right but the
# downstream PDF showed a -2.0 row because of stale CDS / fallback paths.
#
# The defensive fix: short-circuit the entire C-pillar to NEUTRAL for these
# classes BEFORE any feed lookup, then exclude C from the Total Score
# normaliser so the asset is not pulled artificially toward "Sell".
_CREDIT_NA_SECTORS: frozenset[str] = frozenset({
    "Commodities", "Gold", "Silver", "Oil", "Bonds",
})
_CREDIT_NA_TICKER_PREFIXES: tuple[str, ...] = (
    # US Treasury / govt-bond ETFs
    "TLT", "IEF", "SHY", "AGG", "BND", "BIL", "GOVT",
    # Pure commodity ETFs
    "GLD", "SLV", "GDX", "USO", "DBC", "UNG",
)


def _is_credit_not_applicable(ticker: str, sector: Optional[str]) -> bool:
    """
    Asset-class guard: True when the Credit pillar (CDS / Altman / Piotroski /
    InterestCoverage) is conceptually meaningless for this position.

    Triggers on either condition (defensive OR — sector classification can
    be sparse on KZ-listed ETFs):
      • sector ∈ {Commodities, Gold, Silver, Oil, Bonds}
      • ticker stem starts with a known sovereign-bond or pure-commodity
        ETF prefix
    """
    s = (sector or "").strip()
    if s in _CREDIT_NA_SECTORS:
        return True
    stem = str(ticker or "").upper().split(".")[0]
    return stem.startswith(_CREDIT_NA_TICKER_PREFIXES)


# CDS lookup callable type:
#   ticker → {bps: float|None, change_7d: float|None, source: str|None,
#             quality: 'A'|'B'|'C'|None}
CDSLookup = Callable[[str], dict]


# ── V-Pillar: absolute sector P/E and P/B benchmarks ─────────────────────────
# Calibrated to S&P 500 sector trailing P/E and P/B medians (2020-2025 avg).
# Used because portfolio cohorts are typically <5 tickers — too small for
# robust_z (which requires ≥5 samples).  Absolute benchmarks are sector-aware,
# stable, and don't depend on portfolio composition.
#
# Scoring direction (same as valuations_score logic):
#   z > +1.5  → expensive  → negative V contribution
#   z > +0.5  → mildly expensive
#   z < -0.5  → cheap
#   z < -1.5  → very cheap → positive V contribution
_SECTOR_PE_BENCHMARKS: dict[str, tuple[float, float]] = {
    # (median_PE, MAD_equivalent_σ)
    "Technology":     (28.0, 10.0),
    "Semiconductors": (25.0,  9.0),
    "Finance":        (14.0,  4.0),
    "Healthcare":     (22.0,  7.0),
    "Energy":         (14.0,  5.0),
    "Consumer":       (22.0,  6.0),
    "Industrials":    (20.0,  6.0),
    "Gold":           (18.0,  6.0),
    "default":        (20.0,  8.0),
}

_SECTOR_PB_BENCHMARKS: dict[str, tuple[float, float]] = {
    # (median_PB, MAD_equivalent_σ)
    "Technology":     (7.0, 3.5),
    "Semiconductors": (6.0, 3.0),
    "Finance":        (1.4, 0.5),
    "Healthcare":     (4.0, 2.0),
    "Energy":         (2.0, 0.8),
    "Consumer":       (5.0, 2.5),
    "Industrials":    (3.0, 1.5),
    "Gold":           (2.5, 1.0),
    "default":        (3.0, 1.5),
}


def _absolute_valuation_z(
    ratio: Optional[float],
    sector: Optional[str],
    benchmarks: dict[str, tuple[float, float]],
    clip: float = 3.0,
) -> Optional[float]:
    """
    Convert a valuation ratio (P/E or P/B) into a z-score using
    sector-specific absolute benchmarks.

    Returns None when ratio is None/NaN/infinite.
    z > 0 → expensive vs sector median; z < 0 → cheap.
    """
    if ratio is None:
        return None
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(r) or r <= 0:
        return None
    key = sector if sector in benchmarks else "default"
    median, sigma = benchmarks[key]
    if sigma <= 0:
        return None
    z = (r - median) / sigma
    return float(np.clip(z, -clip, clip))


def _compute_valuation_ratios(perf: pd.DataFrame) -> None:
    """
    Pre-compute SEC_PE_Ratio and SEC_PB_Ratio columns in-place.

    P/E = Current_Price / EPS   where EPS = SEC_Net_Income / SEC_Shares_Outstanding
          Only computed when net income is positive (loss companies → no P/E).

    P/B = Current_Price * SEC_Shares_Outstanding / SEC_Book_Equity
          Only computed when book equity is strictly positive
          (buyback-heavy firms like AAPL/MSFT may carry negative equity).

    Both ratios set to NaN when any input is missing or invalid.
    """
    pe_col: list[Optional[float]] = []
    pb_col: list[Optional[float]] = []

    for _, row in perf.iterrows():
        price  = row.get("Current_Price")
        ni     = row.get("SEC_Net_Income")
        shares = row.get("SEC_Shares_Outstanding")
        bkeq   = row.get("SEC_Book_Equity")

        # P/E
        pe: Optional[float] = None
        try:
            p  = float(price)
            n  = float(ni)
            sh = float(shares)
            if p > 0 and n > 0 and sh > 0:
                eps = n / sh
                pe  = p / eps
        except (TypeError, ValueError):
            pass
        pe_col.append(pe)

        # P/B
        pb: Optional[float] = None
        try:
            p  = float(price)
            sh = float(shares)
            bk = float(bkeq)
            if p > 0 and sh > 0 and bk > 0:
                bvps = bk / sh
                pb   = p / bvps
        except (TypeError, ValueError):
            pass
        pb_col.append(pb)

    perf["SEC_PE_Ratio"] = pe_col
    perf["SEC_PB_Ratio"] = pb_col


def _macro_alignment(sector: Optional[str],
                     regime: Optional[RegimeReading]) -> float:
    """Map regime + sector to a -0.5..+0.5 macro-alignment bonus/penalty."""
    if regime is None or not sector:
        return 0.0
    favoured  = REGIME_FAVOURED_SECTORS.get(regime.regime, set())
    penalised = REGIME_PENALISED_SECTORS.get(regime.regime, set())
    # Confidence dampens the magnitude — uncertain regime has small effect.
    weight = 0.5 * float(min(1.0, max(0.0, regime.confidence)))
    if sector in favoured:
        return +weight
    if sector in penalised:
        return -weight
    return 0.0


def _sector_z(value: Optional[float],
              ticker_sector: Optional[str],
              perf_table: pd.DataFrame,
              column: str) -> Optional[float]:
    """
    Compute the sector cross-sectional robust z-score for a single value.
    Returns None if the column is missing or the sector cohort is too small.
    """
    if column not in perf_table.columns or value is None:
        return None
    if ticker_sector is None or "Fundamental_Sector" not in perf_table.columns:
        return None
    cohort = perf_table.loc[
        perf_table["Fundamental_Sector"] == ticker_sector, column
    ].dropna().tolist()
    return robust_z(value, cohort)


def score_portfolio(
    perf_table: pd.DataFrame,
    technicals: dict[str, TechnicalReading],
    *,
    regime: Optional[RegimeReading] = None,
    cds_lookup: Optional[CDSLookup] = None,
    hotspot_trc_pct: float = DEFAULT_HOTSPOT_TRC_PCT,
) -> dict[str, AssetScore]:
    """
    Compute AssetScore for every row in `perf_table`.

    The Total Score is clipped to -6..+6.  Hotspot override forces 'Trim'
    when Euler TRC > hotspot_trc_pct.

    Args:
        perf_table   : DataFrame from UniversalPortfolioManager.analyze_all()
                       (the 'performance_table' frame, post-SEC join).
        technicals   : {ticker: TechnicalReading} from compute_technicals.
        regime       : Optional RegimeReading; provides macro alignment.
        cds_lookup   : Optional callable that returns a CDS dict per ticker.
                       When None, the Credit pillar uses only SEC signals
                       (Altman / Piotroski / IntCov).
        hotspot_trc_pct : Cut-off above which an asset is forced to Trim.

    Returns:
        {ticker: AssetScore}
    """
    out: dict[str, AssetScore] = {}
    if perf_table is None or perf_table.empty:
        return out

    if "Ticker" not in perf_table.columns:
        # Defensive — older callers may pass an indexed frame.
        perf = perf_table.reset_index().copy()
    else:
        perf = perf_table.copy()

    # Pre-compute P/E and P/B for all rows so V-pillar z-scores are available
    # during the per-asset loop below.
    _compute_valuation_ratios(perf)

    for _, row in perf.iterrows():
        ticker = str(row.get("Ticker") or "?")
        sector = row.get("Fundamental_Sector") or None

        # Pillar A — Fundamentals (sector cross-sectional Z-scores)
        roe_z = _sector_z(row.get("SEC_ROE"),                  sector, perf, "SEC_ROE")
        opm_z = _sector_z(row.get("SEC_Op_Margin"),            sector, perf, "SEC_Op_Margin")
        dta_z = _sector_z(row.get("SEC_Debt_to_Assets"),       sector, perf, "SEC_Debt_to_Assets")
        rg_z  = _sector_z(row.get("SEC_Revenue_Growth_YoY"),   sector, perf, "SEC_Revenue_Growth_YoY")
        fcf_z = _sector_z(row.get("SEC_FCF_Margin"),           sector, perf, "SEC_FCF_Margin")
        macro_align = _macro_alignment(sector, regime)
        f_score = fundamentals_score(
            roe_z=roe_z, op_margin_z=opm_z,
            debt_to_assets_z=dta_z, revenue_growth_z=rg_z,
            fcf_margin_z=fcf_z, macro_alignment=macro_align,
        )

        # Pillar B — Valuations.
        # P/E and P/B z-scores: both vs absolute sector benchmarks
        # (portfolio cohorts are typically <5 tickers — too small for
        # robust_z which needs ≥5 samples).  Both contribute on the SAME
        # ±1 scale in valuations_score, so neither signal double-counts.
        pe_z = _absolute_valuation_z(
            row.get("SEC_PE_Ratio"), sector, _SECTOR_PE_BENCHMARKS
        )
        pb_z = _absolute_valuation_z(
            row.get("SEC_PB_Ratio"), sector, _SECTOR_PB_BENCHMARKS
        )
        v_score = valuations_score(pe_z=pe_z, pb_sector_z=pb_z, ev_ebitda_z=None)

        # Pillar C — Technicals (already a -2..+2 reading)
        t_reading = technicals.get(ticker)
        t_score = technicals_score_from_reading(t_reading)

        # Pillar D — Credit (SEC layer always; CDS layer when feed is wired)
        # Asset-class guard: commodities and sovereign-rate ETFs have no
        # corporate credit risk, so the C-pillar must stay neutral instead
        # of inheriting a phantom -2.0 from a missing CDS lookup.  The
        # `credit_applicable` flag is propagated to AssetScore so the
        # downstream PDF can render an em-dash and exclude C from the
        # denominator of the user-facing total.
        credit_applicable = not _is_credit_not_applicable(ticker, sector)
        if credit_applicable:
            cds_info = cds_lookup(ticker) if cds_lookup else {}
            bps      = cds_info.get("bps")
            change7  = cds_info.get("change_7d")
            c_score = credit_score(
                cds_bps          = bps,
                cds_change_7d    = change7,
                altman_zone      = row.get("SEC_Altman_Zone"),
                piotroski_f      = int(row.get("SEC_Piotroski_F")) if pd.notna(row.get("SEC_Piotroski_F")) else None,
                interest_coverage= row.get("SEC_Interest_Coverage"),
            )
        else:
            c_score = 0.0

        # Hotspot override
        trc      = float(row.get("Euler_Risk_Contribution_Pct") or 0.0)
        hotspot  = trc > hotspot_trc_pct

        total = total_score(f_score, v_score, t_score, c_score)
        action = action_from_total(total, hotspot=hotspot)

        out[ticker] = AssetScore(
            ticker            = ticker,
            fundamentals      = f_score,
            valuations        = v_score,
            technicals        = t_score,
            credit            = c_score,
            total             = total,
            action            = action,
            hotspot           = hotspot,
            credit_applicable = credit_applicable,
        )

    return out


__all__ = ["score_portfolio", "DEFAULT_HOTSPOT_TRC_PCT"]
