"""
SEC EDGAR Data Provider — Lightweight fundamental scan.

Fetches ONLY 4 critical metrics from the latest 10-K filing:
  1. Operating Margin   — profitability
  2. Debt / Assets      — leverage
  3. ROE                — capital efficiency
  4. Revenue Growth YoY — momentum

API: data.sec.gov — free, no key, limit 10 req/sec.
Library: edgartools v5.30+

Performance notes (2026-05):
  • Previous version downloaded 2 full 10-K XBRL filings per ticker → ~6 min each.
  • This version fetches only the latest 10-K + previous revenue for YoY → ~40 sec each.
  • Parallel batch (3 workers) + LRU cache → first run ~3 min for 7 tickers,
    subsequent runs instant.
"""
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

logger = logging.getLogger("SEC_EDGAR")

# ── Lazy import edgartools ───────────────────────────────────────────────────
_edgar_available = False
try:
    from edgar import set_identity, Company
    set_identity("RAMP Advisory ramp-advisory@project.com")
    _edgar_available = True
except Exception as _edgar_exc:
    logger.warning("edgartools недоступен: %s", _edgar_exc)


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
# 1. LIGHTWEIGHT FUNDAMENTAL SCAN (10-K only)
# ═══════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=128)
def get_critical_fundamentals(ticker: str) -> dict:
    """
    Fetch ONLY the 4 critical metrics from the latest 10-K filing.

    Returns dict with keys:
        operating_margin, debt_to_assets, roe, revenue_growth_yoy,
        filing_date, revenue, net_income
    """
    if not _edgar_available or _should_skip(ticker):
        return {}

    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        if not filings or len(filings) == 0:
            return {}

        tenk = filings[0].obj()
        fin = tenk.financials

        def _safe(fn, default=0):
            try:
                v = fn()
                return float(v) if v is not None else default
            except Exception:
                return default

        revenue = _safe(lambda: fin.get_revenue())
        net_income = _safe(lambda: fin.get_net_income())
        op_income = _safe(lambda: fin.get_operating_income())
        total_assets = _safe(lambda: fin.get_total_assets())
        total_liabilities = _safe(lambda: fin.get_total_liabilities())
        equity = _safe(lambda: fin.get_stockholders_equity())

        facts = {
            "revenue": revenue,
            "net_income": net_income,
            "filing_date": str(filings[0].filing_date),
        }

        # Critical metrics
        if revenue > 0:
            facts["operating_margin"] = op_income / revenue
        if total_assets > 0:
            facts["debt_to_assets"] = total_liabilities / total_assets
        if equity > 0:
            facts["roe"] = net_income / equity

        # YoY revenue growth (requires previous filing — lightweight extraction)
        if len(filings) > 1:
            try:
                prev_fin = filings[1].obj().financials
                rev_prev = _safe(lambda: prev_fin.get_revenue())
                if rev_prev > 0:
                    facts["revenue_growth_yoy"] = (revenue - rev_prev) / rev_prev
            except Exception:
                pass  # Skip YoY if prev filing is corrupted

        time.sleep(0.2)  # SEC rate limit safety (10 req/s)
        return facts

    except Exception as exc:
        logger.warning("[SEC EDGAR] Ошибка для %s: %s", ticker, exc)
        time.sleep(0.2)
        return {}


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
    Parallel batch scan of tickers via SEC EDGAR.

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
