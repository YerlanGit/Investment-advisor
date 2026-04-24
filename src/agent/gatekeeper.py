"""
Gatekeeper — Детерминированный аудитор портфельных рекомендаций.
Проверяет финальный ответ ИИ перед выводом пользователю.
Работает за 0 мс, стоит $0, никогда не галлюцинирует.
"""
import logging

logger = logging.getLogger("Gatekeeper")


# Дефолтные лимиты (переопределяются профилем пользователя)
DEFAULT_LIMITS = {
    "max_euler_risk_pct": 20.0,        # Макс. вклад одного актива в риск
    "max_single_asset_weight_pct": 15.0, # Макс. вес одного актива
    "max_cvar_daily": -0.05,            # Макс. допустимый CVaR (-5%)
    "min_sharpe": 0.3,                  # Мин. допустимый Sharpe
    "max_portfolio_volatility": 0.40,   # Макс. годовая волатильность (40%)
}


def run_gatekeeper(report: dict, user_limits: dict = None) -> dict:
    """
    Жёсткий детерминированный аудит перед выводом ответа пользователю.
    
    Args:
        report: Словарь из UniversalPortfolioManager.analyze_all()
        user_limits: Пользовательские лимиты (или DEFAULT_LIMITS)
    
    Returns:
        {
            "passed": bool,
            "critical": [...],   # Блокирующие нарушения
            "warnings": [...],   # Предупреждения
            "summary": str       # Краткий текст для LLM
        }
    """
    limits = {**DEFAULT_LIMITS, **(user_limits or {})}
    
    critical = []
    warnings = []
    
    df = report.get("performance_table")
    metrics = report.get("portfolio_metrics", {})
    
    if df is None or df.empty:
        return {"passed": False, "critical": ["Портфель пуст."], "warnings": [], "summary": ""}
    
    # ═══════════════ ПРОВЕРКА 1: Euler Risk ═══════════════
    if "Euler_Risk_Contribution_Pct" in df.columns:
        for _, row in df.iterrows():
            ticker = row.get("Ticker", "?")
            erc = row.get("Euler_Risk_Contribution_Pct", 0)
            if erc > limits["max_euler_risk_pct"]:
                critical.append(
                    f"⛔ EULER RISK: {ticker} генерирует {erc:.1f}% общего риска "
                    f"(лимит: {limits['max_euler_risk_pct']}%). "
                    f"Рекомендация: снизить позицию."
                )
    
    # ═══════════════ ПРОВЕРКА 2: Концентрация капитала ═══════════════
    if "Weight_Pct" in df.columns:
        for _, row in df.iterrows():
            ticker = row.get("Ticker", "?")
            weight = row.get("Weight_Pct", 0)
            if weight > limits["max_single_asset_weight_pct"]:
                warnings.append(
                    f"⚠️ КОНЦЕНТРАЦИЯ: {ticker} = {weight:.1f}% портфеля "
                    f"(лимит: {limits['max_single_asset_weight_pct']}%)."
                )
    
    # ═══════════════ ПРОВЕРКА 3: CVaR (хвостовой риск) ═══════════════
    cvar = metrics.get("CVaR_95_Daily", 0)
    if cvar < limits["max_cvar_daily"]:  # CVaR отрицательный
        critical.append(
            f"⛔ CVaR: Ожидаемый убыток в худшие 5% дней = {cvar:.2%} "
            f"(лимит: {limits['max_cvar_daily']:.2%}). Портфель сверхрискован."
        )
    
    # ═══════════════ ПРОВЕРКА 4: Sharpe Ratio ═══════════════
    sharpe = metrics.get("Sharpe_Ratio", 0)
    if sharpe is not None and sharpe < limits["min_sharpe"]:
        warnings.append(
            f"⚠️ SHARPE: {sharpe:.2f} < {limits['min_sharpe']}. "
            f"Портфель неэффективен (доходность не компенсирует риск)."
        )
    
    # ═══════════════ ПРОВЕРКА 5: Общая волатильность ═══════════════
    vol = metrics.get("Total_Volatility_Ann", 0)
    if vol > limits["max_portfolio_volatility"]:
        critical.append(
            f"⛔ ВОЛАТИЛЬНОСТЬ: {vol:.1%} годовая "
            f"(лимит: {limits['max_portfolio_volatility']:.1%}). Активы разрушают портфель."
        )
    
    # ═══════════════ ПРОВЕРКА 6: ATR Spike (SEC 34-105226 IMS) ═══════════════
    # Если ATR > 3% от цены — актив слишком волатилен для маржинальной торговли
    if "ATR_Pct" in df.columns:
        for _, row in df.iterrows():
            ticker = row.get("Ticker", "?")
            atr_pct = row.get("ATR_Pct", 0)
            if isinstance(atr_pct, (int, float)) and atr_pct > 3.0:
                warnings.append(
                    f"⚠️ IMS/ATR: {ticker} ATR = {atr_pct:.2f}% от цены. "
                    f"Высокая внутридневная волатильность → повышенный маржинальный риск."
                )
    
    # ═══════════════ ПРОВЕРКА 7: Фундаментальный скор (SEC EDGAR) ═══════════════
    if "Fundamental_Score" in df.columns:
        for _, row in df.iterrows():
            ticker = row.get("Ticker", "?")
            fscore = row.get("Fundamental_Score", 50)
            if isinstance(fscore, (int, float)) and fscore < 30:
                warnings.append(
                    f"⚠️ ФУНДАМЕНТАЛ: {ticker} SEC Score = {fscore:.0f}/100. "
                    f"Слабые финансы (маржа, долг или ROE). Рассмотрите сокращение позиции."
                )
    
    # ═══════════════ ФОРМИРОВАНИЕ ИТОГА ═══════════════
    passed = len(critical) == 0
    
    summary_parts = []
    if critical:
        summary_parts.append("КРИТИЧЕСКИЕ НАРУШЕНИЯ (аудитор заблокировал выдачу):")
        summary_parts.extend(critical)
    if warnings:
        summary_parts.append("ПРЕДУПРЕЖДЕНИЯ:")
        summary_parts.extend(warnings)
    if passed and not warnings:
        summary_parts.append("✅ Все проверки пройдены. Портфель в рамках лимитов.")
    
    summary = "\n".join(summary_parts)
    
    if not passed:
        logger.warning(f"Gatekeeper ЗАБЛОКИРОВАЛ выдачу: {len(critical)} критических нарушений.")
    
    return {
        "passed": passed,
        "critical": critical,
        "warnings": warnings,
        "summary": summary
    }
