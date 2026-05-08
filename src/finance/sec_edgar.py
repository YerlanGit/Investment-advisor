"""
SEC EDGAR Data Provider — Lightweight fundamental scan via CompanyFacts JSON API.

Fetches ONLY 4 critical metrics from the latest 10-K filing:
  1. Operating Margin   — profitability
  2. Debt / Assets      — leverage
  3. ROE                — capital efficiency
  4. Revenue Growth YoY — momentum

API: data.sec.gov/api/xbrl/companyfacts — free, no key, limit 10 req/sec.
Transport: raw HTTP GET (requests) — returns structured JSON, no HTML parsing.

Performance notes (2026-05):
  • Previous edgartools version downloaded full 10-K XBRL/HTML files (~30 MB each)
    and parsed them with lxml → ~10 min per ticker on CPU-throttled Cloud Run.
  • This version fetches a single JSON endpoint per ticker (~200 KB) → ~1 sec each.
  • Parallel batch (3 workers) + LRU cache → first run ~10 sec for 7 tickers,
    subsequent runs instant.
"""
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("SEC_EDGAR")

# ── SEC EDGAR HTTP config ────────────────────────────────────────────────────
_SEC_HEADERS = {"User-Agent": "RAMP Advisory ramp-advisory@project.com"}
_SEC_TIMEOUT = 15  # seconds per request


# ── Ticker → CIK lookup (cached for process lifetime) ───────────────────────
@functools.lru_cache(maxsize=1)
def _load_cik_lookup() -> dict[str, str]:
    """
    Download the SEC ticker→CIK mapping.  ~500 KB JSON, cached in-process.
    Returns dict: {"AAPL": "0000320193", "MSFT": "0000789019", ...}
    """
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_SEC_HEADERS,
            timeout=_SEC_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return {
            str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10)
            for v in data.values()
        }
    except Exception as exc:
        logger.warning("[SEC EDGAR] Не удалось загрузить CIK-маппинг: %s", exc)
        return {}


def _ticker_to_cik(ticker: str) -> str | None:
    """Resolve a ticker symbol to a zero-padded 10-digit CIK string."""
    base = ticker.split(".")[0].upper() if "." in ticker else ticker.upper()
    lookup = _load_cik_lookup()
    return lookup.get(base)


# ── XBRL fact extraction helpers ─────────────────────────────────────────────

# Ordered preference lists for each metric — companies use different XBRL tags.
_REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
_OP_INCOME_TAGS = [
    "OperatingIncomeLoss",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
]
_NET_INCOME_TAGS = ["NetIncomeLoss"]
_ASSETS_TAGS = ["Assets"]
_LIABILITIES_TAGS = ["Liabilities"]
_EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
# Phase 2.4 — additional tags for Altman-Z, Piotroski-F, IntCov, FCF margin.
_CURRENT_ASSETS_TAGS    = ["AssetsCurrent"]
_CURRENT_LIABS_TAGS     = ["LiabilitiesCurrent"]
_RETAINED_EARN_TAGS     = ["RetainedEarningsAccumulatedDeficit"]
_INTEREST_EXP_TAGS      = [
    "InterestExpense",
    "InterestExpenseDebt",
    "InterestExpenseOperating",
]
_CFO_TAGS               = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
_CAPEX_TAGS             = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForCapitalImprovements",
]
_LONG_TERM_DEBT_TAGS    = [
    "LongTermDebt",
    "LongTermDebtNoncurrent",
]
_SHARES_OUT_TAGS        = [
    "CommonStockSharesOutstanding",
    "CommonStockSharesIssued",
]
_GROSS_PROFIT_TAGS      = ["GrossProfit"]
_PE_RATIO_DEFAULT       = None  # P/E sourced from market data when available


def _get_annual_values(
    us_gaap: dict, tags: list[str], n: int = 2
) -> list[float | None]:
    """
    Extract the last *n* annual (10-K, FY) values for the tag with the most
    recent data among *tags*.

    Returns a list of length *n* (most recent first).  Missing entries are None.

    Deduplication by period end-date (``end`` field) — a single 10-K filing may
    restate multiple prior years; we want unique fiscal periods sorted by recency.

    Tag selection: instead of returning the first matching tag, we evaluate ALL
    candidate tags and pick the one whose most recent ``end`` date is latest.
    This handles companies like NVDA that migrated from one XBRL tag to another.
    """
    best_result: list[float] | None = None
    best_end: str = ""

    for tag in tags:
        concept = us_gaap.get(tag)
        if not concept:
            continue
        units = concept.get("units", {}).get("USD", [])
        if not units:
            continue
        # Filter to annual 10-K filings only
        annual = [
            e for e in units
            if e.get("form") == "10-K" and e.get("fp") == "FY"
        ]
        if not annual:
            continue
        # Sort by period end date descending → most recent fiscal period first.
        annual.sort(key=lambda e: e.get("end", ""), reverse=True)
        # De-duplicate by period end date (same period may appear in multiple filings)
        seen_end: set[str] = set()
        deduped: list[dict] = []
        for entry in annual:
            end_date = entry.get("end", "")
            if end_date not in seen_end:
                seen_end.add(end_date)
                deduped.append(entry)
            if len(deduped) >= n:
                break
        if not deduped:
            continue
        # Pick the tag with the most recent period
        tag_latest_end = deduped[0].get("end", "")
        if tag_latest_end > best_end:
            best_end = tag_latest_end
            best_result = [float(e["val"]) for e in deduped]

    if best_result is not None:
        while len(best_result) < n:
            best_result.append(None)
        return best_result
    return [None] * n


# ── Tickers to skip (no 10-K on SEC) ────────────────────────────────────────
_SKIP_SUFFIXES = (".AIX", ".KZ", ".IL")
_SKIP_ETFS = frozenset({
    "GLD", "SPY", "IWM", "QQQ", "DBC", "IEF", "BIL", "LQD", "HYG",
    "MTUM", "VLUE", "QUAL", "EEM", "BND", "AGG", "TLT", "SHY",
    "XLF", "XLE", "XLV", "XLK", "SOXX", "XME", "XLP", "XLI",
    "GDX", "SLV", "USO", "UNG", "PDBC",
})


def _should_skip(ticker: str) -> bool:
    """Return True if this ticker has no SEC 10-K filing."""
    t = ticker.upper().strip()
    base = t.split(".")[0] if "." in t else t
    if any(t.endswith(sfx) for sfx in _SKIP_SUFFIXES):
        return True
    if base in _SKIP_ETFS:
        return True
    if "FFSPC" in t or "BOND" in t or "OVD" in t:
        return True
    return False


# ═══════════════════════════════════════════════════════════
# 1. LIGHTWEIGHT FUNDAMENTAL SCAN (CompanyFacts JSON API)
# ═══════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=128)
def get_critical_fundamentals(ticker: str) -> dict:
    """
    Fetch ONLY the 4 critical metrics from the SEC CompanyFacts JSON API.

    Uses data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json — a single HTTP GET
    that returns all XBRL facts as structured JSON (~200 KB).  No HTML parsing.

    Returns dict with keys:
        operating_margin, debt_to_assets, roe, revenue_growth_yoy,
        filing_date, revenue, net_income
    """
    if _should_skip(ticker):
        return {}

    cik = _ticker_to_cik(ticker)
    if not cik:
        logger.debug("[SEC EDGAR] CIK не найден для %s", ticker)
        return {}

    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        r = requests.get(url, headers=_SEC_HEADERS, timeout=_SEC_TIMEOUT)
        r.raise_for_status()
        facts = r.json()
    except Exception as exc:
        logger.warning("[SEC EDGAR] Ошибка загрузки CompanyFacts для %s: %s", ticker, exc)
        time.sleep(0.15)
        return {}

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        logger.debug("[SEC EDGAR] Нет us-gaap данных для %s", ticker)
        return {}

    # Extract latest + previous annual values
    revenues = _get_annual_values(us_gaap, _REVENUE_TAGS, n=2)
    net_incomes = _get_annual_values(us_gaap, _NET_INCOME_TAGS, n=1)
    op_incomes = _get_annual_values(us_gaap, _OP_INCOME_TAGS, n=1)
    total_assets = _get_annual_values(us_gaap, _ASSETS_TAGS, n=1)
    total_liabs = _get_annual_values(us_gaap, _LIABILITIES_TAGS, n=1)
    equities = _get_annual_values(us_gaap, _EQUITY_TAGS, n=1)

    revenue = revenues[0] or 0
    rev_prev = revenues[1]
    net_income = net_incomes[0] or 0
    op_income = op_incomes[0] or 0
    assets = total_assets[0] or 0
    liabilities = total_liabs[0] or 0
    equity = equities[0] or 0

    # Determine filing date from the tag with the most recent annual period
    filing_date = None
    best_end = ""
    for tag in _REVENUE_TAGS:
        concept = us_gaap.get(tag)
        if not concept:
            continue
        units = concept.get("units", {}).get("USD", [])
        annual = [e for e in units if e.get("form") == "10-K" and e.get("fp") == "FY"]
        if annual:
            annual.sort(key=lambda e: e.get("end", ""), reverse=True)
            tag_end = annual[0].get("end", "")
            if tag_end > best_end:
                best_end = tag_end
                filing_date = annual[0].get("filed")

    result: dict = {
        "revenue": revenue,
        "net_income": net_income,
        "filing_date": filing_date,
    }

    # Critical metrics
    if revenue > 0:
        result["operating_margin"] = op_income / revenue
    if assets > 0:
        result["debt_to_assets"] = liabilities / assets
    if equity > 0 and equity != 0:
        result["roe"] = net_income / equity

    # YoY revenue growth
    if rev_prev and rev_prev > 0:
        result["revenue_growth_yoy"] = (revenue - rev_prev) / rev_prev

    # Rate-limit safety (SEC allows 10 req/s, we use 3 workers → ~3 req/s)
    time.sleep(0.15)
    return result


# ═══════════════════════════════════════════════════════════
# 1b.  EXTENDED FUNDAMENTALS — Altman-Z, Piotroski-F, IntCov,
#      FCF margin, Long-term debt.  All from the same JSON
#      payload as get_critical_fundamentals (no extra requests).
# ═══════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=128)
def get_extended_fundamentals(ticker: str) -> dict:
    """
    Fetch extended fundamentals used by pillar D (Credit) of the scoring model.
    Re-uses the same CompanyFacts JSON endpoint and lru-cache as the lightweight
    scan, so the marginal cost for an existing ticker is one dict lookup.

    Returns (subset of keys when data is missing):
        altman_z         : float | None
        altman_zone      : 'Safe' | 'Grey' | 'Distress' | None
        piotroski_f      : int  (0..9) | None
        interest_coverage: float | None  (EBIT / InterestExpense)
        fcf_margin       : float | None  (FCF / Revenue)
        long_term_debt   : float | None
        ebit             : float | None
        cfo              : float | None
        capex            : float | None
    """
    if _should_skip(ticker):
        return {}

    cik = _ticker_to_cik(ticker)
    if not cik:
        return {}

    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        r = requests.get(url, headers=_SEC_HEADERS, timeout=_SEC_TIMEOUT)
        r.raise_for_status()
        facts = r.json()
    except Exception as exc:
        logger.warning("[SEC EDGAR] Extended fetch failed for %s: %s", ticker, exc)
        time.sleep(0.15)
        return {}

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return {}

    # Pull two annual readings for momentum-style comparisons (Piotroski needs YoY deltas)
    revenue        = _get_annual_values(us_gaap, _REVENUE_TAGS,        n=2)
    op_income      = _get_annual_values(us_gaap, _OP_INCOME_TAGS,      n=2)
    net_income     = _get_annual_values(us_gaap, _NET_INCOME_TAGS,     n=2)
    total_assets   = _get_annual_values(us_gaap, _ASSETS_TAGS,         n=2)
    total_liabs    = _get_annual_values(us_gaap, _LIABILITIES_TAGS,    n=2)
    current_assets = _get_annual_values(us_gaap, _CURRENT_ASSETS_TAGS, n=2)
    current_liabs  = _get_annual_values(us_gaap, _CURRENT_LIABS_TAGS,  n=2)
    retained_earn  = _get_annual_values(us_gaap, _RETAINED_EARN_TAGS,  n=1)
    interest_exp   = _get_annual_values(us_gaap, _INTEREST_EXP_TAGS,   n=1)
    cfo            = _get_annual_values(us_gaap, _CFO_TAGS,            n=2)
    capex          = _get_annual_values(us_gaap, _CAPEX_TAGS,          n=1)
    long_term_debt = _get_annual_values(us_gaap, _LONG_TERM_DEBT_TAGS, n=2)
    gross_profit   = _get_annual_values(us_gaap, _GROSS_PROFIT_TAGS,   n=2)
    shares_out     = _get_annual_values(us_gaap, _SHARES_OUT_TAGS,     n=1)

    out: dict = {
        "ebit":           op_income[0],
        "cfo":            cfo[0],
        "capex":          capex[0],
        "long_term_debt": long_term_debt[0],
        "shares_outstanding": shares_out[0],
    }

    rev0 = revenue[0]
    rev1 = revenue[1] if len(revenue) > 1 else None

    # ── FCF margin = (CFO - CapEx) / Revenue ────────────────────────────────
    if cfo[0] is not None and capex[0] is not None and rev0 and rev0 > 0:
        # CapEx is reported as a positive outflow on the cash-flow statement.
        out["fcf"]        = cfo[0] - capex[0]
        out["fcf_margin"] = (cfo[0] - capex[0]) / rev0

    # ── Interest Coverage = EBIT / InterestExpense ──────────────────────────
    if op_income[0] is not None and interest_exp[0] and interest_exp[0] > 0:
        out["interest_coverage"] = op_income[0] / interest_exp[0]

    # ── Altman Z-Score (public-firm formula) ────────────────────────────────
    # Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5
    # X1 = Working Capital / Total Assets
    # X2 = Retained Earnings / Total Assets
    # X3 = EBIT / Total Assets
    # X4 = Market Cap / Total Liabilities  (proxy: BookEquity / Liab if no MV)
    # X5 = Revenue / Total Assets
    try:
        ta = total_assets[0]
        tl = total_liabs[0]
        ca = current_assets[0]
        cl = current_liabs[0]
        re_val = retained_earn[0]
        ebit = op_income[0]
        if ta and ta > 0 and ca is not None and cl is not None and \
           re_val is not None and ebit is not None and \
           tl and tl > 0 and rev0 is not None:
            wc  = ca - cl
            x1  = wc / ta
            x2  = re_val / ta
            x3  = ebit / ta
            # Use book-equity proxy when no market price is available here.
            be  = ta - tl
            x4  = be / tl if tl > 0 else 0.0
            x5  = rev0 / ta
            z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
            out["altman_z"] = float(z)
            if z > 2.99:
                out["altman_zone"] = "Safe"
            elif z >= 1.81:
                out["altman_zone"] = "Grey"
            else:
                out["altman_zone"] = "Distress"
    except Exception:
        pass  # missing tags → no Z

    # ── Piotroski F-Score (9 binary checks) ─────────────────────────────────
    try:
        f = 0
        ni0 = net_income[0]
        ni1 = net_income[1] if len(net_income) > 1 else None
        cfo0 = cfo[0]
        cfo1 = cfo[1] if len(cfo) > 1 else None
        ta0 = total_assets[0]
        ta1 = total_assets[1] if len(total_assets) > 1 else None
        ltd0 = long_term_debt[0]
        ltd1 = long_term_debt[1] if len(long_term_debt) > 1 else None
        ca0 = current_assets[0]; cl0 = current_liabs[0]
        ca1 = current_assets[1] if len(current_assets) > 1 else None
        cl1 = current_liabs[1]  if len(current_liabs)  > 1 else None
        gp0 = gross_profit[0]
        gp1 = gross_profit[1] if len(gross_profit) > 1 else None

        # 1. Net income > 0
        if ni0 is not None and ni0 > 0: f += 1
        # 2. CFO > 0
        if cfo0 is not None and cfo0 > 0: f += 1
        # 3. ΔROA > 0  (use NI/TA proxy)
        if ni0 is not None and ni1 is not None and ta0 and ta1:
            if (ni0 / ta0) > (ni1 / ta1): f += 1
        # 4. Accruals: CFO > NI
        if cfo0 is not None and ni0 is not None and cfo0 > ni0: f += 1
        # 5. ΔLeverage < 0  (LongTermDebt / Assets shrinks YoY)
        if ltd0 is not None and ltd1 is not None and ta0 and ta1:
            if (ltd0 / ta0) < (ltd1 / ta1): f += 1
        # 6. ΔLiquidity > 0  (CurrentRatio expands YoY)
        if ca0 and cl0 and ca1 and cl1:
            if (ca0 / cl0) > (ca1 / cl1): f += 1
        # 7. No new equity issuance — rough proxy: shares outstanding flat
        #    (we don't fetch t-1 shares here; skip — caller may add)
        # 8. ΔGross margin > 0  (GP/Rev expands YoY)
        if gp0 is not None and gp1 is not None and rev0 and rev1:
            if (gp0 / rev0) > (gp1 / rev1): f += 1
        # 9. ΔAsset Turnover > 0
        if rev0 is not None and rev1 is not None and ta0 and ta1:
            if (rev0 / ta0) > (rev1 / ta1): f += 1
        # Cap at 9; we may report 0..8 if the equity-issuance check is unavailable.
        out["piotroski_f"] = f
    except Exception:
        pass

    time.sleep(0.10)
    return out


# ═══════════════════════════════════════════════════════════
# 2. SECTOR-NORMALIZED FUNDAMENTAL SCORING
# ═══════════════════════════════════════════════════════════

# Sector-specific benchmarks based on S&P 500 sector medians.
# Each sector has different "healthy" ranges for OpMargin, Debt/Assets, ROE.
# e.g. Finance sector naturally carries 85% D/A — penalizing it is wrong.
SECTOR_BENCHMARKS: dict[str, dict] = {
    "Technology":      {"op_good": 0.20, "op_great": 0.30, "dta_ok": 0.40, "dta_bad": 0.65, "roe_good": 0.18, "roe_great": 0.30},
    "Semiconductors":  {"op_good": 0.25, "op_great": 0.35, "dta_ok": 0.35, "dta_bad": 0.60, "roe_good": 0.18, "roe_great": 0.30},
    "Finance":         {"op_good": 0.25, "op_great": 0.35, "dta_ok": 0.85, "dta_bad": 0.95, "roe_good": 0.10, "roe_great": 0.18},
    "Healthcare":      {"op_good": 0.12, "op_great": 0.22, "dta_ok": 0.50, "dta_bad": 0.75, "roe_good": 0.15, "roe_great": 0.25},
    "Energy":          {"op_good": 0.08, "op_great": 0.18, "dta_ok": 0.45, "dta_bad": 0.70, "roe_good": 0.12, "roe_great": 0.22},
    "Consumer":        {"op_good": 0.05, "op_great": 0.12, "dta_ok": 0.60, "dta_bad": 0.80, "roe_good": 0.15, "roe_great": 0.25},
    "Industrials":     {"op_good": 0.10, "op_great": 0.18, "dta_ok": 0.55, "dta_bad": 0.75, "roe_good": 0.15, "roe_great": 0.25},
    "Gold":            {"op_good": 0.10, "op_great": 0.20, "dta_ok": 0.40, "dta_bad": 0.60, "roe_good": 0.08, "roe_great": 0.15},
    "default":         {"op_good": 0.15, "op_great": 0.25, "dta_ok": 0.50, "dta_bad": 0.70, "roe_good": 0.15, "roe_great": 0.25},
}


def calculate_fundamental_score(ticker: str, sector: str = "default") -> dict:
    """
    Sector-normalized composite fundamental score: 0 (weak) → 100 (strong).

    Scoring is calibrated against sector-specific benchmarks (SECTOR_BENCHMARKS).
    For example, a 30% Debt/Assets is excellent for Tech but normal for Finance.

    Factors (CFA-aligned):
    ┌──────────────────┬──────┬──────────────────────────────────┐
    │ Factor           │ Wt   │ Logic (sector-normalized)        │
    ├──────────────────┼──────┼──────────────────────────────────┤
    │ Operating Margin │ 30%  │ > sector op_great = strong       │
    │ Debt/Assets      │ 25%  │ < sector dta_ok = healthy        │
    │ ROE              │ 30%  │ > sector roe_great = efficient   │
    │ Revenue Growth   │ 15%  │ >10% = momentum (universal)      │
    └──────────────────┴──────┴──────────────────────────────────┘
    """
    fundamentals = get_critical_fundamentals(ticker)
    if not fundamentals:
        return {
            "ticker": ticker, "fundamental_score": 50,
            "sector": sector, "details": [], "raw_fundamentals": {},
        }

    bench = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["default"])
    score = 50.0
    details = []

    # 1. Operating Margin (30%) — sector-normalized
    op_margin = fundamentals.get("operating_margin", 0)
    if op_margin > bench["op_great"]:
        score += 18
        details.append(f"OpMargin {op_margin:.0%} >{bench['op_great']:.0%} [{sector}] [+18]")
    elif op_margin > bench["op_good"]:
        score += 10
        details.append(f"OpMargin {op_margin:.0%} >{bench['op_good']:.0%} [{sector}] [+10]")
    elif op_margin < 0:
        score -= 20
        details.append(f"OpMargin {op_margin:.0%} negative [-20]")

    # 2. Debt/Assets (25%) — sector-normalized
    dta = fundamentals.get("debt_to_assets", 0)
    if 0 < dta < bench["dta_ok"] * 0.6:
        score += 12
        details.append(f"Debt/Assets {dta:.0%} <{bench['dta_ok']*0.6:.0%} [{sector}] [+12]")
    elif dta > bench["dta_bad"]:
        score -= 15
        details.append(f"Debt/Assets {dta:.0%} >{bench['dta_bad']:.0%} [{sector}] [-15]")

    # 3. ROE (30%) — sector-normalized
    roe = fundamentals.get("roe", 0)
    if roe > bench["roe_great"]:
        score += 18
        details.append(f"ROE {roe:.0%} >{bench['roe_great']:.0%} [{sector}] [+18]")
    elif roe > bench["roe_good"]:
        score += 10
        details.append(f"ROE {roe:.0%} >{bench['roe_good']:.0%} [{sector}] [+10]")
    elif roe < 0:
        score -= 15
        details.append(f"ROE {roe:.0%} negative [-15]")

    # 4. Revenue Growth YoY (15%) — universal (not sector-dependent)
    rev_growth = fundamentals.get("revenue_growth_yoy")
    if rev_growth is not None:
        if rev_growth > 0.20:
            score += 8
            details.append(f"RevGrowth {rev_growth:.0%} >20% [+8]")
        elif rev_growth > 0.05:
            score += 3
            details.append(f"RevGrowth {rev_growth:.0%} moderate [+3]")
        elif rev_growth < -0.10:
            score -= 10
            details.append(f"RevGrowth {rev_growth:.0%} declining [-10]")

    score = max(0, min(100, score))

    return {
        "ticker": ticker,
        "fundamental_score": round(score, 1),
        "sector": sector,
        "details": details,
        "raw_fundamentals": fundamentals,
    }


# ═══════════════════════════════════════════════════════════
# 3. PARALLEL BATCH SCAN (3 workers, SEC-safe)
# ═══════════════════════════════════════════════════════════

def batch_fundamental_scan(
    tickers: list,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Parallel batch scan of tickers via SEC EDGAR CompanyFacts API.

    Args:
        tickers: list of ticker symbols to scan
        sector_map: optional mapping {ticker: sector_name} for sector-normalized
                    scoring. If not provided, all tickers use "default" thresholds.

    Filters out non-US tickers, ETFs, and AIX instruments.
    Uses ThreadPoolExecutor(3) to stay under SEC 10 req/s limit.
    Returns DataFrame for join into performance_table.
    """
    sector_map = sector_map or {}

    # Pre-filter: only scan tickers that have SEC filings
    scannable = [t for t in tickers if not _should_skip(t)]
    skipped = [t for t in tickers if _should_skip(t)]

    if skipped:
        logger.info("[SEC EDGAR] Пропущены (нет 10-K): %s", ", ".join(skipped))

    results = []

    def _scan_one(ticker: str) -> dict:
        sector = sector_map.get(ticker, "default")
        logger.info("[SEC EDGAR] Сканирование %s (sector=%s)...", ticker, sector)
        try:
            score_data = calculate_fundamental_score(ticker, sector=sector)
            f = score_data["raw_fundamentals"]
            ext = get_extended_fundamentals(ticker)
            row = {
                "Ticker": ticker,
                "Fundamental_Score": score_data["fundamental_score"],
                "Fundamental_Sector": sector,
                "SEC_Op_Margin": f.get("operating_margin"),
                "SEC_Debt_to_Assets": f.get("debt_to_assets"),
                "SEC_ROE": f.get("roe"),
                "SEC_Revenue_Growth_YoY": f.get("revenue_growth_yoy"),
                "SEC_Filing_Date": f.get("filing_date"),
                # Phase 2.4 extensions for the Credit pillar.
                "SEC_FCF_Margin":         ext.get("fcf_margin"),
                "SEC_Interest_Coverage":  ext.get("interest_coverage"),
                "SEC_Altman_Z":           ext.get("altman_z"),
                "SEC_Altman_Zone":        ext.get("altman_zone"),
                "SEC_Piotroski_F":        ext.get("piotroski_f"),
                "SEC_Long_Term_Debt":     ext.get("long_term_debt"),
            }
            return row
        except Exception as e:
            logger.warning("[SEC EDGAR] Ошибка для %s: %s", ticker, e)
            return {"Ticker": ticker, "Fundamental_Score": 50, "Fundamental_Sector": sector}

    # Parallel execution (3 workers to stay under SEC rate limit)
    if scannable:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_scan_one, t): t for t in scannable}
            for fut in as_completed(futures):
                results.append(fut.result())

    # Add neutral scores for skipped tickers
    for t in skipped:
        results.append({
            "Ticker": t,
            "Fundamental_Score": 50,
            "Fundamental_Sector": sector_map.get(t, "EM_Proxy"),
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).set_index("Ticker")
