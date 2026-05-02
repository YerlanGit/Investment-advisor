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
# 2. COMPOSITE SCORE (4 critical factors only)
# ═══════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=128)
def calculate_fundamental_score(ticker: str) -> dict:
    """
    Composite fundamental score: 0 (weak) → 100 (strong).

    Factors (simplified CFA):
    ┌──────────────────┬──────┬─────────────────────────┐
    │ Factor           │ Wt   │ Logic                   │
    ├──────────────────┼──────┼─────────────────────────┤
    │ Operating Margin │ 30%  │ >15% = strong           │
    │ Debt/Assets      │ 25%  │ <40% = healthy          │
    │ ROE              │ 30%  │ >15% = efficient        │
    │ Revenue Growth   │ 15%  │ >10% = momentum         │
    └──────────────────┴──────┴─────────────────────────┘
    """
    fundamentals = get_critical_fundamentals(ticker)
    if not fundamentals:
        return {"ticker": ticker, "fundamental_score": 50, "details": [], "raw_fundamentals": {}}

    score = 50.0
    details = []

    # 1. Operating Margin (30%)
    op_margin = fundamentals.get("operating_margin", 0)
    if op_margin > 0.25:
        score += 18
        details.append(f"OpMargin {op_margin:.0%} >25% [+18]")
    elif op_margin > 0.15:
        score += 10
        details.append(f"OpMargin {op_margin:.0%} >15% [+10]")
    elif op_margin < 0:
        score -= 20
        details.append(f"OpMargin {op_margin:.0%} negative [-20]")

    # 2. Debt/Assets (25%)
    dta = fundamentals.get("debt_to_assets", 0)
    if 0 < dta < 0.30:
        score += 12
        details.append(f"Debt/Assets {dta:.0%} <30% [+12]")
    elif dta > 0.70:
        score -= 15
        details.append(f"Debt/Assets {dta:.0%} >70% [-15]")

    # 3. ROE (30%)
    roe = fundamentals.get("roe", 0)
    if roe > 0.25:
        score += 18
        details.append(f"ROE {roe:.0%} >25% [+18]")
    elif roe > 0.15:
        score += 10
        details.append(f"ROE {roe:.0%} >15% [+10]")
    elif roe < 0:
        score -= 15
        details.append(f"ROE {roe:.0%} negative [-15]")

    # 4. Revenue Growth YoY (15%)
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
        "details": details,
        "raw_fundamentals": fundamentals,
    }


# ═══════════════════════════════════════════════════════════
# 3. PARALLEL BATCH SCAN (3 workers, SEC-safe)
# ═══════════════════════════════════════════════════════════

def batch_fundamental_scan(tickers: list) -> pd.DataFrame:
    """
    Parallel batch scan of tickers via SEC EDGAR CompanyFacts API.

    Filters out non-US tickers, ETFs, and AIX instruments.
    Uses ThreadPoolExecutor(3) to stay under SEC 10 req/s limit.
    Returns DataFrame for join into performance_table.
    """
    # Pre-filter: only scan tickers that have SEC filings
    scannable = [t for t in tickers if not _should_skip(t)]
    skipped = [t for t in tickers if _should_skip(t)]

    if skipped:
        logger.info("[SEC EDGAR] Пропущены (нет 10-K): %s", ", ".join(skipped))

    results = []

    def _scan_one(ticker: str) -> dict:
        logger.info("[SEC EDGAR] Сканирование %s...", ticker)
        try:
            score_data = calculate_fundamental_score(ticker)
            f = score_data["raw_fundamentals"]
            return {
                "Ticker": ticker,
                "Fundamental_Score": score_data["fundamental_score"],
                "SEC_Op_Margin": f.get("operating_margin"),
                "SEC_Debt_to_Assets": f.get("debt_to_assets"),
                "SEC_ROE": f.get("roe"),
                "SEC_Revenue_Growth_YoY": f.get("revenue_growth_yoy"),
                "SEC_Filing_Date": f.get("filing_date"),
            }
        except Exception as e:
            logger.warning("[SEC EDGAR] Ошибка для %s: %s", ticker, e)
            return {"Ticker": ticker, "Fundamental_Score": 50}

    # Parallel execution (3 workers to stay under SEC rate limit)
    if scannable:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_scan_one, t): t for t in scannable}
            for fut in as_completed(futures):
                results.append(fut.result())

    # Add neutral scores for skipped tickers
    for t in skipped:
        results.append({"Ticker": t, "Fundamental_Score": 50})

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).set_index("Ticker")
