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

import functools
from typing import Callable, Optional

import numpy as np
import pandas as pd

from finance.regime import (
    RegimeReading,
    REGIME_FAVOURED_SECTORS,
    REGIME_PENALISED_SECTORS,
)
from finance.scoring import (
    HOTSPOT_TRC_PCT,
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


# Hotspot threshold — re-exported under the legacy name; the VALUE lives in
# finance.scoring.HOTSPOT_TRC_PCT (single source of truth shared with the
# Gatekeeper's GK-1 rule, Sprint-5.1 / S4).
DEFAULT_HOTSPOT_TRC_PCT = HOTSPOT_TRC_PCT


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

    Triggers on any condition (defensive OR — sector classification can
    be sparse on KZ-listed ETFs):
      • sector ∈ {Commodities, Gold, Silver, Oil, Bonds}
      • ticker stem starts with a known sovereign-bond or pure-commodity
        ETF prefix
      • P-6 (audit E1/M-4, 2026-07-13): ticker is a daily-reset leveraged/
        inverse ETP wrapper (CONL, TQQQ, …) — свопная обёртка без
        корпоративной отчётности; без этого guard'а F-пиллар деградировал к
        макро-тилту сектора 'Other' и печатал фантомный фундаментальный
        вердикт (реестр: finance/leveraged.py, dependency-light).

    NaN-safe: a missing sector can arrive as float('nan') (pandas fills
    unmapped Fundamental_Sector cells with NaN, and `NaN or ""` returns NaN
    because NaN is truthy).  We coerce any non-string sector to "" so the
    guard never raises — a single unmapped ticker must NOT crash the whole
    scoring pass.
    """
    if isinstance(sector, str):
        s = sector.strip()
    else:
        s = ""        # None / NaN / float → treat as "no sector"
    if s in _CREDIT_NA_SECTORS:
        return True
    stem = str(ticker or "").upper().split(".")[0]
    if stem.startswith(_CREDIT_NA_TICKER_PREFIXES):
        return True
    try:
        from finance.leveraged import is_leveraged_etp
        return is_leveraged_etp(stem)
    except Exception:       # реестр недоступен → прежнее поведение
        return False


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

# Sprint-5 Task 6 — sector FCF-yield benchmarks (fcf / market cap, decimals).
# Higher yield = cheaper/higher-quality (the V-pillar flips the sign).  These
# are deliberately conservative sector medians; the σ is a robust spread.
_SECTOR_FCF_YIELD_BENCHMARKS: dict[str, tuple[float, float]] = {
    # (median_FCF_yield, σ-equivalent)
    "Technology":     (0.030, 0.020),
    "Semiconductors": (0.030, 0.025),
    "Finance":        (0.050, 0.030),
    "Healthcare":     (0.050, 0.025),
    "Energy":         (0.070, 0.040),
    "Consumer":       (0.050, 0.025),
    "Industrials":    (0.050, 0.025),
    "Gold":           (0.030, 0.030),
    "default":        (0.040, 0.030),
}


def _absolute_signed_z(
    value: Optional[float],
    sector: Optional[str],
    benchmarks: dict[str, tuple[float, float]],
    clip: float = 3.0,
) -> Optional[float]:
    """Z-score for a signed metric (e.g. FCF yield) that may legitimately be
    NEGATIVE — unlike `_absolute_valuation_z`, which rejects ratios ≤ 0 (right
    for P/E and P/B, wrong for a yield that can go negative when a firm burns
    cash).  Returns None only when the value is missing / non-finite.
    z > 0 → richer than the sector median (good for a yield); z < 0 → poorer.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    key = sector if sector in benchmarks else "default"
    median, sigma = benchmarks[key]
    if sigma <= 0:
        return None
    return float(np.clip((v - median) / sigma, -clip, clip))


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
    Pre-compute SEC_PE_Ratio, SEC_PB_Ratio and SEC_FCF_Yield columns in-place.

    P/E = Current_Price / EPS   where EPS = SEC_Net_Income / SEC_Shares_Outstanding
          Only computed when net income is positive (loss companies → no P/E).

    P/B = Current_Price * SEC_Shares_Outstanding / SEC_Book_Equity
          Only computed when book equity is strictly positive
          (buyback-heavy firms like AAPL/MSFT may carry negative equity).

    FCF Yield = SEC_FCF / market cap   where market cap = Price * Shares.
          Sprint-5 Task 6 price-aware quality signal.  CAN be negative (a
          cash-burning firm → negative yield → "expensive"); only None when an
          input is missing or the market cap is non-positive.

    Ratios set to None when any input is missing or invalid.
    """
    pe_col: list[Optional[float]] = []
    pb_col: list[Optional[float]] = []
    fcfy_col: list[Optional[float]] = []

    for _, row in perf.iterrows():
        price  = row.get("Current_Price")
        ni     = row.get("SEC_Net_Income")
        shares = row.get("SEC_Shares_Outstanding")
        bkeq   = row.get("SEC_Book_Equity")
        fcf    = row.get("SEC_FCF")

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

        # FCF Yield = FCF / market cap (can be negative; needs a positive mcap).
        fcfy: Optional[float] = None
        try:
            p  = float(price)
            sh = float(shares)
            fc = float(fcf)
            mcap = p * sh
            if mcap > 0 and np.isfinite(fc):
                fcfy = fc / mcap
        except (TypeError, ValueError):
            pass
        fcfy_col.append(fcfy)

    perf["SEC_PE_Ratio"] = pe_col
    perf["SEC_PB_Ratio"] = pb_col
    perf["SEC_FCF_Yield"] = fcfy_col


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


# ── F-pillar absolute sector benchmarks (fallback for small cohorts) ────────
# Real portfolios rarely carry ≥5 names per sector (the `robust_z` minimum
# cohort size).  Without a fallback, all sector members collapse to the
# same F-score from `macro_alignment` alone — every Tech name reads
# "F=+0.4" regardless of fundamentals.  The benchmarks below let the
# F-pillar still distinguish strong from weak fundamentals when the cohort
# is too small (<5).  Numbers are best-effort sector medians compiled from
# S&P 500 2020–2025 universe; the σ-equivalent column is MAD-ish (robust
# spread), NOT standard deviation, so it survives outliers.
#
# Each entry: sector → {metric: (median, robust_sigma)}
_SECTOR_FUNDAMENTAL_BENCHMARKS: dict[str, dict[str, tuple[float, float]]] = {
    "Technology":     {"SEC_ROE": (0.30, 0.15),  "SEC_Op_Margin": (0.25, 0.10),
                       "SEC_Debt_to_Assets": (0.35, 0.15),
                       "SEC_Revenue_Growth_YoY": (0.10, 0.08),
                       "SEC_FCF_Margin": (0.25, 0.10)},
    "Semiconductors": {"SEC_ROE": (0.20, 0.15),  "SEC_Op_Margin": (0.25, 0.12),
                       "SEC_Debt_to_Assets": (0.30, 0.12),
                       "SEC_Revenue_Growth_YoY": (0.15, 0.12),
                       "SEC_FCF_Margin": (0.20, 0.12)},
    "Finance":        {"SEC_ROE": (0.12, 0.04),  "SEC_Op_Margin": (0.30, 0.10),
                       "SEC_Debt_to_Assets": (0.85, 0.05),
                       "SEC_Revenue_Growth_YoY": (0.05, 0.05),
                       "SEC_FCF_Margin": (0.15, 0.08)},
    "Healthcare":     {"SEC_ROE": (0.18, 0.10),  "SEC_Op_Margin": (0.20, 0.08),
                       "SEC_Debt_to_Assets": (0.45, 0.12),
                       "SEC_Revenue_Growth_YoY": (0.08, 0.05),
                       "SEC_FCF_Margin": (0.15, 0.08)},
    "Energy":         {"SEC_ROE": (0.15, 0.10),  "SEC_Op_Margin": (0.18, 0.10),
                       "SEC_Debt_to_Assets": (0.45, 0.10),
                       "SEC_Revenue_Growth_YoY": (0.05, 0.15),
                       "SEC_FCF_Margin": (0.10, 0.10)},
    "Consumer":       {"SEC_ROE": (0.20, 0.10),  "SEC_Op_Margin": (0.12, 0.06),
                       "SEC_Debt_to_Assets": (0.50, 0.10),
                       "SEC_Revenue_Growth_YoY": (0.06, 0.04),
                       "SEC_FCF_Margin": (0.10, 0.05)},
    "Industrials":    {"SEC_ROE": (0.18, 0.08),  "SEC_Op_Margin": (0.13, 0.05),
                       "SEC_Debt_to_Assets": (0.50, 0.12),
                       "SEC_Revenue_Growth_YoY": (0.06, 0.05),
                       "SEC_FCF_Margin": (0.10, 0.05)},
}


def _absolute_fundamental_z(value: float, sector: str, column: str,
                            clip: float = 3.0) -> Optional[float]:
    """Absolute z vs sector median/MAD.  None if no benchmark for this pair."""
    spec = _SECTOR_FUNDAMENTAL_BENCHMARKS.get(sector, {}).get(column)
    if spec is None:
        return None
    median, sigma = spec
    if sigma <= 0:
        return None
    z = (float(value) - median) / sigma
    return float(np.clip(z, -clip, clip))


# ── Sprint-5.1 (S2): dynamic sector cohort from LIVE SEC filings ────────────
# The static `_SECTOR_FUNDAMENTAL_BENCHMARKS` are "2020-2025 best-effort"
# constants that silently go stale — and because retail books almost never
# hold ≥5 same-sector names, the static table was effectively the ONLY path.
# This block builds a REAL cross-sectional cohort instead: the sector's
# largest US constituents (a static membership list — like an index — whose
# DATA is fetched live from SEC EDGAR CompanyFacts and lru-cached per
# process).  robust_z against this cohort replaces the stale constants;
# the static table remains the LAST resort (logged once per pair).
#
# Cost: ≤7 cached HTTP calls per sector on the first report after boot
# (`_fetch_company_facts` is shared with the portfolio's own SEC scan).
# Kill-switch: env SECTOR_COHORT_DISABLED=1.  The production engine opts in
# via `score_portfolio(dynamic_benchmarks=True)`; library/test callers keep
# the old static behaviour by default (no network in unit tests).
_SECTOR_REPRESENTATIVES: dict[str, tuple[str, ...]] = {
    "Technology":     ("AAPL", "MSFT", "GOOGL", "META", "ORCL", "ADBE", "CRM"),
    "Semiconductors": ("NVDA", "AVGO", "AMD", "TXN", "QCOM", "MU", "INTC"),
    "Finance":        ("JPM", "BAC", "WFC", "GS", "MS", "C"),
    "Healthcare":     ("LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE"),
    "Energy":         ("XOM", "CVX", "COP", "EOG", "SLB", "PSX"),
    "Consumer":       ("PG", "KO", "PEP", "COST", "WMT", "MDLZ"),
    "Industrials":    ("GE", "CAT", "RTX", "UNP", "HON", "DE"),
}

# perf-table column → (which SEC fetcher, dict key in its result)
_COLUMN_TO_SEC_FIELD: dict[str, tuple[str, str]] = {
    "SEC_ROE":                ("crit", "roe"),
    "SEC_Op_Margin":          ("crit", "operating_margin"),
    "SEC_Debt_to_Assets":     ("crit", "debt_to_assets"),
    "SEC_Revenue_Growth_YoY": ("crit", "revenue_growth_yoy"),
    "SEC_FCF_Margin":         ("ext",  "fcf_margin"),
}

# Log the stale-constant fallback only once per (sector, column) pair.
_STATIC_FALLBACK_LOGGED: set[tuple[str, str]] = set()


def _dynamic_sector_cohort(sector: Optional[str],
                           column: str) -> tuple[float, ...]:
    """Live SEC cohort values for (sector, column); () when unavailable.

    Env kill-switch is checked OUTSIDE the cache so flipping
    SECTOR_COHORT_DISABLED at runtime cannot poison cached results.
    """
    import os
    if os.getenv("SECTOR_COHORT_DISABLED") == "1":
        return ()
    if not sector or sector not in _SECTOR_REPRESENTATIVES:
        return ()
    if column not in _COLUMN_TO_SEC_FIELD:
        return ()
    return _dynamic_sector_cohort_cached(sector, column)


@functools.lru_cache(maxsize=64)
def _dynamic_sector_cohort_cached(sector: str, column: str) -> tuple[float, ...]:
    source, field = _COLUMN_TO_SEC_FIELD[column]
    values: list[float] = []
    try:
        from finance import sec_edgar
    except Exception:
        return ()
    for rep in _SECTOR_REPRESENTATIVES[sector]:
        try:
            data = (sec_edgar.get_critical_fundamentals(rep) if source == "crit"
                    else sec_edgar.get_extended_fundamentals(rep))
            v = (data or {}).get(field)
            if v is not None and np.isfinite(float(v)):
                values.append(float(v))
        except Exception as exc:                       # noqa: BLE001
            import logging as _lg
            _lg.getLogger("ScoringOrchestrator").debug(
                "dynamic cohort: %s/%s skipped (%s)", rep, column, exc)
    return tuple(values)


def _sector_z(value: Optional[float],
              ticker_sector: Optional[str],
              perf_table: pd.DataFrame,
              column: str,
              *, dynamic: bool = False) -> Optional[float]:
    """
    Sector-relative z-score with THREE fallbacks:

      1. Cross-sectional robust_z on the IN-PORTFOLIO cohort.  When the
         cohort has ≥5 finite samples this is the most accurate signal.
      2. (Sprint-5.1, when `dynamic=True`) robust_z against the LIVE SEC
         cohort of the sector's largest US constituents
         (`_dynamic_sector_cohort`) — real, current filings instead of
         frozen constants.  Retail books rarely have ≥5 same-sector names,
         so in production this is the path that usually runs.
      3. The ABSOLUTE static benchmark (`_SECTOR_FUNDAMENTAL_BENCHMARKS`,
         2020-2025 medians) — LAST resort only, logged once per pair so a
         stale-constant verdict is никогда не silent.
      4. None when nothing works — the pillar reads "no data" instead of
         fabricating.
    """
    if value is None or not np.isfinite(float(value)):
        return None
    if column in perf_table.columns and ticker_sector and \
       "Fundamental_Sector" in perf_table.columns:
        cohort = perf_table.loc[
            perf_table["Fundamental_Sector"] == ticker_sector, column
        ].dropna().tolist()
        z_rel = robust_z(value, cohort)
        if z_rel is not None:
            return z_rel
    # In-portfolio cohort too small — try the LIVE SEC sector cohort.
    if dynamic and ticker_sector:
        try:
            live = _dynamic_sector_cohort(ticker_sector, column)
            z_live = robust_z(value, live)
            if z_live is not None:
                return z_live
        except Exception as exc:                       # noqa: BLE001
            import logging as _lg
            _lg.getLogger("ScoringOrchestrator").debug(
                "dynamic cohort failed for %s/%s: %s",
                ticker_sector, column, exc)
    # Last resort — static table (provenance-logged once per pair).
    if ticker_sector:
        z_static = _absolute_fundamental_z(value, ticker_sector, column)
        if z_static is not None and dynamic:
            key = (str(ticker_sector), column)
            if key not in _STATIC_FALLBACK_LOGGED:
                _STATIC_FALLBACK_LOGGED.add(key)
                import logging as _lg
                _lg.getLogger("ScoringOrchestrator").info(
                    "F-pillar: static 2020-25 benchmark used for %s/%s "
                    "(live SEC cohort unavailable)", ticker_sector, column)
        return z_static
    return None


def score_portfolio(
    perf_table: pd.DataFrame,
    technicals: dict[str, TechnicalReading],
    *,
    regime: Optional[RegimeReading] = None,
    cds_lookup: Optional[CDSLookup] = None,
    hotspot_trc_pct: float = DEFAULT_HOTSPOT_TRC_PCT,
    dynamic_benchmarks: bool = False,
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
        dynamic_benchmarks : Sprint-5.1 — when True, small-sector F-pillar
                       z-scores fall back to the LIVE SEC cohort of sector
                       leaders before touching the static 2020-25 constants.
                       Off by default (no network in library/unit-test use);
                       the production engine opts in.

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
        # NaN-safe sector: pandas fills unmapped cells with float('nan'),
        # and `NaN or None` returns NaN (NaN is truthy) — which then crashes
        # any string op downstream.  Coerce non-str / NaN to None here, once.
        sector_raw = row.get("Fundamental_Sector")
        sector = sector_raw if isinstance(sector_raw, str) and sector_raw.strip() else None

      # Per-ticker resilience: a single malformed row (bad SEC value, exotic
      # ticker) must never wipe the entire 4-pillar panel + Black-Litterman.
      # Wrap the body; on failure log and skip just that asset.
        try:
            _score_one_asset(out, row, ticker, sector, perf, technicals,
                             regime, cds_lookup, hotspot_trc_pct,
                             dynamic_benchmarks)
        except Exception as exc:                       # noqa: BLE001
            import logging as _lg
            _lg.getLogger("ScoringOrchestrator").warning(
                "Scoring skipped for %s: %s", ticker, exc)
    return out


def _score_one_asset(out, row, ticker, sector, perf, technicals,
                     regime, cds_lookup, hotspot_trc_pct,
                     dynamic_benchmarks: bool = False) -> None:
    """Score a single asset and write the AssetScore into `out` (in place)."""
    # Physical commodities (Gold/Silver/Oil) and sovereign-rate ETFs
    # (TLT/IEF/…) have NO financial statements — the Fundamentals
    # pillar ("Фундамент · отчётность") is conceptually N/A.  Without
    # this guard, F collapsed to the regime tilt alone (macro_alignment)
    # → GLD/SLV showed F = -0.4 purely because Gold/Silver are penalised
    # in Expansion, which reads as a (non-existent) fundamental verdict.
    # The SAME asset classes already have C (credit) marked N/A, so we
    # reuse that guard for a consistent "no corporate financials" rule.
    fundamentals_applicable = not _is_credit_not_applicable(ticker, sector)

    # Pillar A — Fundamentals (sector cross-sectional Z-scores; live SEC
    # cohort fallback when dynamic_benchmarks is on — Sprint-5.1 / S2).
    _dyn = dynamic_benchmarks
    roe_z = _sector_z(row.get("SEC_ROE"),                  sector, perf, "SEC_ROE",                dynamic=_dyn)
    opm_z = _sector_z(row.get("SEC_Op_Margin"),            sector, perf, "SEC_Op_Margin",          dynamic=_dyn)
    dta_z = _sector_z(row.get("SEC_Debt_to_Assets"),       sector, perf, "SEC_Debt_to_Assets",     dynamic=_dyn)
    rg_z  = _sector_z(row.get("SEC_Revenue_Growth_YoY"),   sector, perf, "SEC_Revenue_Growth_YoY", dynamic=_dyn)
    fcf_z = _sector_z(row.get("SEC_FCF_Margin"),           sector, perf, "SEC_FCF_Margin",         dynamic=_dyn)
    if fundamentals_applicable:
        macro_align = _macro_alignment(sector, regime)
        # 4-Pillar #3 — YoY margin-trend momentum (темпы изменения фундаментала).
        # Absent (older payloads / no SEC coverage) → None → F stays unchanged.
        _mom_raw = row.get("SEC_Fundamental_Momentum")
        momentum = float(_mom_raw) if pd.notna(_mom_raw) else None
        f_score = fundamentals_score(
            roe_z=roe_z, op_margin_z=opm_z,
            debt_to_assets_z=dta_z, revenue_growth_z=rg_z,
            fcf_margin_z=fcf_z, macro_alignment=macro_align,
            momentum=momentum,
        )
    else:
        # No financial statements → neutral F; regime tilt is shown
        # separately in the regime section, not smuggled into F.
        f_score = 0.0

    # Pillar B — Valuations.
    # P/E and P/B z-scores: both vs absolute sector benchmarks
    # (portfolio cohorts are typically <5 tickers — too small for
    # robust_z which needs ≥5 samples).  All three signals contribute on the
    # SAME ±1 scale in valuations_score, so none double-counts.
    pe_z = _absolute_valuation_z(
        row.get("SEC_PE_Ratio"), sector, _SECTOR_PE_BENCHMARKS
    )
    pb_z = _absolute_valuation_z(
        row.get("SEC_PB_Ratio"), sector, _SECTOR_PB_BENCHMARKS
    )
    # Sprint-5 Task 6 — add a price-aware FCF-yield quality signal so the V
    # pillar no longer rests on two correlated multiples (P/E + P/B) alone.
    fcf_yield_z = _absolute_signed_z(
        row.get("SEC_FCF_Yield"), sector, _SECTOR_FCF_YIELD_BENCHMARKS
    )
    v_score = valuations_score(pe_z=pe_z, pb_sector_z=pb_z,
                               ev_ebitda_z=None, fcf_yield_z=fcf_yield_z)

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
        ticker                  = ticker,
        fundamentals            = f_score,
        valuations              = v_score,
        technicals              = t_score,
        credit                  = c_score,
        total                   = total,
        action                  = action,
        hotspot                 = hotspot,
        credit_applicable       = credit_applicable,
        fundamentals_applicable = fundamentals_applicable,
    )

    # Mutates `out` in place; no return value (F-11 cleanup).


__all__ = ["score_portfolio", "DEFAULT_HOTSPOT_TRC_PCT"]
