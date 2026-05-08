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


# CDS lookup callable type:
#   ticker → {bps: float|None, change_7d: float|None, source: str|None,
#             quality: 'A'|'B'|'C'|None}
CDSLookup = Callable[[str], dict]


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
        perf = perf_table.reset_index()
    else:
        perf = perf_table

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

        # Pillar B — Valuations (P/E history Z is the only one we have today;
        # P/B and EV/EBITDA can be wired when SEC fields are added later).
        pe_z = None  # placeholder — kept here to make wiring explicit
        v_score = valuations_score(pe_history_z=pe_z, pb_sector_z=None, ev_ebitda_z=None)

        # Pillar C — Technicals (already a -2..+2 reading)
        t_reading = technicals.get(ticker)
        t_score = technicals_score_from_reading(t_reading)

        # Pillar D — Credit (SEC layer always; CDS layer when feed is wired)
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

        # Hotspot override
        trc      = float(row.get("Euler_Risk_Contribution_Pct") or 0.0)
        hotspot  = trc > hotspot_trc_pct

        total = total_score(f_score, v_score, t_score, c_score)
        action = action_from_total(total, hotspot=hotspot)

        out[ticker] = AssetScore(
            ticker       = ticker,
            fundamentals = f_score,
            valuations   = v_score,
            technicals   = t_score,
            credit       = c_score,
            total        = total,
            action       = action,
            hotspot      = hotspot,
        )

    return out


__all__ = ["score_portfolio", "DEFAULT_HOTSPOT_TRC_PCT"]
