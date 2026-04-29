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
    Извлекает расширенный набор фундаментальных метрик из 10-K (annual filing).

    Базовые метрики (всегда):
        revenue, net_income, operating_income, total_assets,
        total_liabilities, equity, gross_margin, operating_margin,
        net_margin, debt_to_assets, roe

    Расширенные метрики (добавлены 2026-04 при миграции с yfinance):
        free_cash_flow      — операционный кэш-флоу минус CapEx (качество прибыли)
        fcf_margin          — FCF / Revenue (эффективность преобразования выручки)
        current_ratio       — Current Assets / Current Liabilities (ликвидность)
        quick_ratio         — (Current Assets - Inventory) / Current Liabilities
        interest_coverage   — Operating Income / Interest Expense (покрытие долга)
        rd_intensity        — R&D / Revenue (innovation factor для tech)
        asset_turnover      — Revenue / Total Assets (эффективность активов)
        revenue_growth_yoy  — динамика выручки (год-к-году, требует 2-х филингов)
        net_income_growth_yoy — динамика прибыли (год-к-году)
    """
    if not _edgar_available:
        return {}

    def _fetch():
        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        if not filings or len(filings) == 0:
            return {}

        # Берём последние два 10-K для YoY расчётов
        tenk_latest = filings[0].obj()
        fin = tenk_latest.financials

        # ── Базовые метрики ─────────────────────────────────────────
        revenue = _safe_call(lambda: fin.get_revenue(), 0) or 0
        net_income = _safe_call(lambda: fin.get_net_income(), 0) or 0
        op_income = _safe_call(lambda: fin.get_operating_income(), 0) or 0
        total_assets = _safe_call(lambda: fin.get_total_assets(), 0) or 0
        total_liabilities = _safe_call(lambda: fin.get_total_liabilities(), 0) or 0
        equity = _safe_call(lambda: fin.get_stockholders_equity(), 0) or 0

        facts = {
            "revenue":           float(revenue),
            "net_income":        float(net_income),
            "operating_income":  float(op_income),
            "total_assets":      float(total_assets),
            "total_liabilities": float(total_liabilities),
            "equity":            float(equity),
            "filing_date":       str(filings[0].filing_date),
        }

        # ── Расчётные базовые ──────────────────────────────────────
        if facts["revenue"] > 0:
            facts["operating_margin"] = facts["operating_income"] / facts["revenue"]
            facts["net_margin"]       = facts["net_income"]       / facts["revenue"]
        if facts["total_assets"] > 0:
            facts["debt_to_assets"] = facts["total_liabilities"] / facts["total_assets"]
            facts["asset_turnover"] = facts["revenue"] / facts["total_assets"]
        if equity > 0:
            facts["roe"] = facts["net_income"] / float(equity)

        # ── Расширенные метрики (2026-04) ──────────────────────────
        # Free Cash Flow = CFO - CapEx.  Самый сильный сигнал качества прибыли.
        cfo   = _safe_call(lambda: fin.get_cash_from_operations(), 0) or 0
        capex = _safe_call(lambda: fin.get_capital_expenditure(), 0) or 0
        if cfo:
            fcf = float(cfo) - abs(float(capex))
            facts["free_cash_flow"] = fcf
            if facts["revenue"] > 0:
                facts["fcf_margin"] = fcf / facts["revenue"]

        # Liquidity ratios
        curr_assets = _safe_call(lambda: fin.get_current_assets(), 0) or 0
        curr_liabs  = _safe_call(lambda: fin.get_current_liabilities(), 0) or 0
        inventory   = _safe_call(lambda: fin.get_inventory(), 0) or 0
        if curr_liabs and float(curr_liabs) > 0:
            facts["current_ratio"] = float(curr_assets) / float(curr_liabs)
            facts["quick_ratio"]   = (float(curr_assets) - float(inventory)) / float(curr_liabs)

        # Interest coverage — способность обслуживать долг
        interest_expense = _safe_call(lambda: fin.get_interest_expense(), 0) or 0
        if interest_expense and abs(float(interest_expense)) > 0:
            facts["interest_coverage"] = facts["operating_income"] / abs(float(interest_expense))

        # R&D intensity (важно для tech-портфелей)
        rd = _safe_call(lambda: fin.get_research_and_development(), 0) or 0
        if rd and facts["revenue"] > 0:
            facts["rd_intensity"] = float(rd) / facts["revenue"]

        # ── YoY рост (требует двух филингов) ───────────────────────
        if len(filings) > 1:
            try:
                tenk_prev = filings[1].obj()
                fin_prev  = tenk_prev.financials
                rev_prev = _safe_call(lambda: fin_prev.get_revenue(), 0) or 0
                ni_prev  = _safe_call(lambda: fin_prev.get_net_income(), 0) or 0
                if float(rev_prev) > 0:
                    facts["revenue_growth_yoy"] = (facts["revenue"] - float(rev_prev)) / float(rev_prev)
                if float(ni_prev) != 0:
                    facts["net_income_growth_yoy"] = (facts["net_income"] - float(ni_prev)) / abs(float(ni_prev))
            except Exception as exc:
                logger.debug("YoY вычисление не удалось для %s: %s", ticker, exc)

        return facts

    result = _safe_call(_fetch, {})
    time.sleep(0.15)  # Rate limit (SEC: 10 req/s; ставим 6 с запасом)
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

    # ── Расширенные метрики (2026-04) ──────────────────────────
    # 7. Free Cash Flow margin — качество прибыли
    fcf_margin = fundamentals.get('fcf_margin', 0)
    if fcf_margin > 0.15:
        score += 8
        details.append(f"FCF Margin {fcf_margin:.1%} > 15% [+8]")
    elif fcf_margin > 0.05:
        score += 3
        details.append(f"FCF Margin {fcf_margin:.1%} умеренный [+3]")
    elif 0 > fcf_margin > -0.10:
        score -= 5
        details.append(f"FCF Margin {fcf_margin:.1%} отрицательный [-5]")
    elif fcf_margin <= -0.10:
        score -= 12
        details.append(f"FCF Margin {fcf_margin:.1%} cash burn [-12]")

    # 8. Liquidity (Current Ratio) — стрессоустойчивость
    cr = fundamentals.get('current_ratio', 0)
    if cr > 2.0:
        score += 4
        details.append(f"Current Ratio {cr:.1f} > 2.0 [+4]")
    elif 0 < cr < 1.0:
        score -= 8
        details.append(f"Current Ratio {cr:.1f} < 1.0 (риск ликвидности) [-8]")

    # 9. Interest Coverage — способность обслуживать долг
    ic = fundamentals.get('interest_coverage', 0)
    if ic > 10:
        score += 5
        details.append(f"Interest Coverage {ic:.1f}x > 10 [+5]")
    elif 0 < ic < 2:
        score -= 10
        details.append(f"Interest Coverage {ic:.1f}x < 2 (риск дефолта) [-10]")

    # 10. Revenue Growth YoY — top-line momentum
    rev_growth = fundamentals.get('revenue_growth_yoy', None)
    if rev_growth is not None:
        if rev_growth > 0.20:
            score += 6
            details.append(f"Revenue Growth {rev_growth:.1%} > 20% [+6]")
        elif rev_growth > 0.05:
            score += 2
            details.append(f"Revenue Growth {rev_growth:.1%} умеренный [+2]")
        elif rev_growth < -0.10:
            score -= 8
            details.append(f"Revenue Growth {rev_growth:.1%} <-10% [-8]")

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
            f = score_data["raw_fundamentals"]
            row = {
                "Ticker":               ticker,
                "Fundamental_Score":    score_data["fundamental_score"],
                # Базовые
                "SEC_Revenue":          f.get("revenue"),
                "SEC_Net_Income":       f.get("net_income"),
                "SEC_Op_Margin":        f.get("operating_margin"),
                "SEC_Net_Margin":       f.get("net_margin"),
                "SEC_Debt_to_Assets":   f.get("debt_to_assets"),
                "SEC_ROE":              f.get("roe"),
                "SEC_Filing_Date":      f.get("filing_date"),
                # Расширенные (2026-04)
                "SEC_Free_Cash_Flow":   f.get("free_cash_flow"),
                "SEC_FCF_Margin":       f.get("fcf_margin"),
                "SEC_Current_Ratio":    f.get("current_ratio"),
                "SEC_Quick_Ratio":      f.get("quick_ratio"),
                "SEC_Interest_Coverage": f.get("interest_coverage"),
                "SEC_RD_Intensity":     f.get("rd_intensity"),
                "SEC_Asset_Turnover":   f.get("asset_turnover"),
                "SEC_Revenue_Growth_YoY": f.get("revenue_growth_yoy"),
                "SEC_Net_Income_Growth_YoY": f.get("net_income_growth_yoy"),
                "Insider_Signal":       score_data["insider_data"].get("insider_signal", "N/A"),
                "Institutional_Signal": score_data["institutional_data"].get("institutional_signal", "N/A"),
            }
            results.append(row)
        except Exception as e:
            logger.warning(f"[SEC EDGAR] Ошибка для {ticker}: {e}")
            results.append({"Ticker": ticker, "Fundamental_Score": 50, "Insider_Signal": "ERROR"})
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).set_index("Ticker")
