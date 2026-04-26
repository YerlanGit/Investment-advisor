"""
SEC EDGAR Data Provider — Бесплатный фундаментальный анализ.
Источники: 10-K/10-Q (финансы), 13F (институциональные), Form 4 (инсайдеры).
API: data.sec.gov — бесплатно, без ключа, лимит 10 req/sec.
Библиотека: edgartools v5.30+
"""
import logging
import time
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("SEC_EDGAR")

# Ленивый импорт edgartools
_edgar_available = False
try:
    from edgar import set_identity, Company
    set_identity("RAMP Advisory ramp-advisory@project.com")
    _edgar_available = True
except Exception as _edgar_exc:
    logger.warning("edgartools недоступен: %s", _edgar_exc)


def _safe_call(func, default=None):
    """Безопасный вызов SEC API с обработкой ошибок."""
    try:
        return func()
    except Exception as e:
        logger.debug(f"SEC API call failed: {e}")
        return default


# ═══════════════════════════════════════════════════════════
# 1. FUNDAMENTAL SCANNER (10-K / 10-Q)
# ═══════════════════════════════════════════════════════════

def get_sec_fundamentals(ticker: str) -> dict:
    """
    Извлекает ключевые фундаментальные метрики из последнего 10-K.
    Использует edgartools v5 API: financials.get_revenue(), get_net_income() и т.д.
    
    Returns: dict с revenue, net_income, operating_income, total_assets,
             total_liabilities, equity, gross_margin, operating_margin,
             net_margin, debt_to_assets, roe, revenue_per_share
    """
    if not _edgar_available:
        return {}

    def _fetch():
        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        if not filings or len(filings) == 0:
            return {}
        
        tenk = filings[0].obj()
        fin = tenk.financials
        
        # Извлекаем факты через проверенный API
        revenue = _safe_call(lambda: fin.get_revenue(), 0) or 0
        net_income = _safe_call(lambda: fin.get_net_income(), 0) or 0
        op_income = _safe_call(lambda: fin.get_operating_income(), 0) or 0
        total_assets = _safe_call(lambda: fin.get_total_assets(), 0) or 0
        total_liabilities = _safe_call(lambda: fin.get_total_liabilities(), 0) or 0
        equity = _safe_call(lambda: fin.get_stockholders_equity(), 0) or 0
        
        facts = {
            "revenue": float(revenue),
            "net_income": float(net_income),
            "operating_income": float(op_income),
            "total_assets": float(total_assets),
            "total_liabilities": float(total_liabilities),
            "equity": float(equity),
            "filing_date": str(filings[0].filing_date),
        }
        
        # Расчётные метрики (если данные есть)
        if facts["revenue"] > 0:
            facts["operating_margin"] = facts["operating_income"] / facts["revenue"]
            facts["net_margin"] = facts["net_income"] / facts["revenue"]
        
        if facts["total_assets"] > 0:
            facts["debt_to_assets"] = facts["total_liabilities"] / facts["total_assets"]
        
        if equity > 0:
            facts["roe"] = facts["net_income"] / float(equity)
        
        return facts

    result = _safe_call(_fetch, {})
    time.sleep(0.15)  # Rate limit
    return result


# ═══════════════════════════════════════════════════════════
# 2. INSTITUTIONAL TRACKER (13F)
# ═══════════════════════════════════════════════════════════

def get_institutional_sentiment(ticker: str) -> dict:
    """
    Проверяет наличие 13F файлингов для тикера.
    Если фонды активно подают 13F — значит институциональный интерес есть.
    """
    if not _edgar_available:
        return {"institutional_signal": "NO_DATA"}

    def _fetch():
        company = Company(ticker)
        filings = company.get_filings(form="13F-HR")
        count = len(filings) if filings else 0
        
        if count == 0:
            return {"institutional_signal": "NO_FILINGS", "total_13f": 0}
        
        return {
            "institutional_signal": "DATA_AVAILABLE",
            "total_13f": count,
            "latest_13f_date": str(filings[0].filing_date) if count > 0 else "N/A",
        }

    return _safe_call(_fetch, {"institutional_signal": "ERROR"})


# ═══════════════════════════════════════════════════════════
# 3. INSIDER RADAR (Form 4)
# ═══════════════════════════════════════════════════════════

def get_insider_activity(ticker: str) -> dict:
    """
    Анализ Form 4: есть ли инсайдерские сделки?
    Высокая частота Form 4 → активные инсайдерские сделки.
    """
    if not _edgar_available:
        return {"insider_signal": "NO_DATA"}

    def _fetch():
        company = Company(ticker)
        filings = company.get_filings(form="4")
        count = len(filings) if filings else 0
        
        if count == 0:
            return {"insider_signal": "NO_FILINGS", "total_form4": 0}
        
        return {
            "insider_signal": "DATA_AVAILABLE",
            "total_form4": count,
            "latest_form4_date": str(filings[0].filing_date) if count > 0 else "N/A",
        }

    return _safe_call(_fetch, {"insider_signal": "ERROR"})


# ═══════════════════════════════════════════════════════════
# 4. COMPOSITE FUNDAMENTAL SCORE (Факторная интеграция)
# ═══════════════════════════════════════════════════════════

def calculate_fundamental_score(ticker: str) -> dict:
    """
    Композитный фундаментальный скор: 0 (слабый) → 100 (сильный).
    
    Факторы и веса (CFA-методология):
    ┌─────────────────────┬────────┬──────────────────────────────┐
    │ Фактор              │ Вес    │ Логика                       │
    ├─────────────────────┼────────┼──────────────────────────────┤
    │ Operating Margin     │ 25%    │ >15% = сильный бизнес        │
    │ Debt/Assets          │ 20%    │ <40% = здоровый баланс       │
    │ ROE                  │ 25%    │ >15% = эффективное управление│
    │ Net Margin           │ 15%    │ >10% = прибыльный            │
    │ Institutional Signal │ 10%    │ 13F наличие = доверие фондов │
    │ Insider Signal       │  5%    │ Form 4 наличие = активность  │
    └─────────────────────┴────────┴──────────────────────────────┘
    """
    fundamentals = get_sec_fundamentals(ticker)
    insider = get_insider_activity(ticker)
    institutional = get_institutional_sentiment(ticker)

    score = 50.0  # Нейтральная база
    details = []

    # 1. Operating Margin (вес 25%)
    op_margin = fundamentals.get('operating_margin', 0)
    if op_margin > 0.25:
        score += 15
        details.append(f"Op Margin {op_margin:.1%} > 25% [+15]")
    elif op_margin > 0.15:
        score += 8
        details.append(f"Op Margin {op_margin:.1%} > 15% [+8]")
    elif 0 < op_margin < 0.05:
        score -= 10
        details.append(f"Op Margin {op_margin:.1%} < 5% [-10]")
    elif op_margin < 0:
        score -= 20
        details.append(f"Op Margin {op_margin:.1%} отрицательная [-20]")

    # 2. Debt/Assets (вес 20%)
    dta = fundamentals.get('debt_to_assets', 0)
    if 0 < dta < 0.30:
        score += 10
        details.append(f"Debt/Assets {dta:.1%} < 30% [+10]")
    elif 0 < dta < 0.50:
        score += 3
        details.append(f"Debt/Assets {dta:.1%} умеренный [+3]")
    elif dta > 0.70:
        score -= 15
        details.append(f"Debt/Assets {dta:.1%} > 70% [-15]")

    # 3. ROE (вес 25%)
    roe = fundamentals.get('roe', 0)
    if roe > 0.25:
        score += 15
        details.append(f"ROE {roe:.1%} > 25% [+15]")
    elif roe > 0.15:
        score += 8
        details.append(f"ROE {roe:.1%} > 15% [+8]")
    elif roe < 0:
        score -= 15
        details.append(f"ROE {roe:.1%} отрицательный [-15]")

    # 4. Net Margin (вес 15%)
    net_margin = fundamentals.get('net_margin', 0)
    if net_margin > 0.15:
        score += 8
        details.append(f"Net Margin {net_margin:.1%} > 15% [+8]")
    elif net_margin > 0.05:
        score += 3
        details.append(f"Net Margin {net_margin:.1%} > 5% [+3]")
    elif net_margin < 0:
        score -= 10
        details.append(f"Net Margin {net_margin:.1%} убыточный [-10]")

    # 5. Institutional Signal (вес 10%)
    if institutional.get("institutional_signal") == "DATA_AVAILABLE":
        score += 5
        details.append("Институциональный интерес (13F) [+5]")

    # 6. Insider Signal (вес 5%)
    if insider.get("insider_signal") == "DATA_AVAILABLE":
        score += 2
        details.append("Инсайдерская активность (Form 4) [+2]")

    # Clamp
    score = max(0, min(100, score))
    
    return {
        "ticker": ticker,
        "fundamental_score": round(score, 1),
        "details": details,
        "raw_fundamentals": fundamentals,
        "insider_data": insider,
        "institutional_data": institutional,
    }


# ═══════════════════════════════════════════════════════════
# 5. BATCH SCAN (для интеграции в MAC3 pipeline)
# ═══════════════════════════════════════════════════════════

def batch_fundamental_scan(tickers: list) -> pd.DataFrame:
    """
    Батч-сканирование тикеров через SEC EDGAR.
    Возвращает DataFrame для джойна в performance_table.
    """
    results = []
    for ticker in tickers:
        logger.info(f"[SEC EDGAR] Сканирование {ticker}...")
        try:
            score_data = calculate_fundamental_score(ticker)
            row = {
                "Ticker": ticker,
                "Fundamental_Score": score_data["fundamental_score"],
                "SEC_Revenue": score_data["raw_fundamentals"].get("revenue"),
                "SEC_Net_Income": score_data["raw_fundamentals"].get("net_income"),
                "SEC_Op_Margin": score_data["raw_fundamentals"].get("operating_margin"),
                "SEC_Net_Margin": score_data["raw_fundamentals"].get("net_margin"),
                "SEC_Debt_to_Assets": score_data["raw_fundamentals"].get("debt_to_assets"),
                "SEC_ROE": score_data["raw_fundamentals"].get("roe"),
                "SEC_Filing_Date": score_data["raw_fundamentals"].get("filing_date"),
                "Insider_Signal": score_data["insider_data"].get("insider_signal", "N/A"),
                "Institutional_Signal": score_data["institutional_data"].get("institutional_signal", "N/A"),
            }
            results.append(row)
        except Exception as e:
            logger.warning(f"[SEC EDGAR] Ошибка для {ticker}: {e}")
            results.append({"Ticker": ticker, "Fundamental_Score": 50, "Insider_Signal": "ERROR"})
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).set_index("Ticker")
