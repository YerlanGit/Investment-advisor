"""
Gatekeeper — Deterministic portfolio audit (non-blocking).

Checks the final report BEFORE showing to user. Runs in 0 ms, costs $0, never hallucinates.

2026-05 update: Gatekeeper is now ADVISORY-ONLY (non-blocking).
Critical violations are reported as warnings to the user but do NOT prevent
the report from being delivered. This ensures data upload is never blocked.
"""
import logging

logger = logging.getLogger("Gatekeeper")


# Default limits (overridden by user profile)
DEFAULT_LIMITS = {
    "max_euler_risk_pct": 20.0,        # Max single asset risk contribution
    "max_single_asset_weight_pct": 15.0, # Max single asset weight
    "max_cvar_daily": -0.05,            # Max acceptable CVaR (-5%)
    "min_sharpe": 0.3,                  # Min acceptable Sharpe
    "max_portfolio_volatility": 0.40,   # Max annual volatility (40%)
}


def run_gatekeeper(report: dict, user_limits: dict = None) -> dict:
    """
    Advisory-only deterministic audit.

    NEVER blocks report delivery. Returns violations and warnings for
    display in Telegram, but the report PDF is always generated.

    Args:
        report: Dict from UniversalPortfolioManager.analyze_all()
        user_limits: User-specific limits (or DEFAULT_LIMITS)

    Returns:
        {
            "passed": bool,      # True if no critical violations
            "critical": [...],   # Critical violations (advisory, not blocking)
            "warnings": [...],   # Non-critical warnings
            "summary": str       # Brief text for Telegram
        }
    """
    limits = {**DEFAULT_LIMITS, **(user_limits or {})}

    critical = []
    warnings = []

    df = report.get("performance_table")
    metrics = report.get("portfolio_metrics", {})

    if df is None or df.empty:
        return {"passed": True, "critical": [], "warnings": ["Портфель пуст."], "summary": ""}

    # ═══════════════ CHECK 1: Euler Risk ═══════════════
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

    # ═══════════════ CHECK 2: Capital concentration ═══════════════
    if "Weight_Pct" in df.columns:
        for _, row in df.iterrows():
            ticker = row.get("Ticker", "?")
            weight = row.get("Weight_Pct", 0)
            if weight > limits["max_single_asset_weight_pct"]:
                warnings.append(
                    f"⚠️ КОНЦЕНТРАЦИЯ: {ticker} = {weight:.1f}% портфеля "
                    f"(лимит: {limits['max_single_asset_weight_pct']}%)."
                )

    # ═══════════════ CHECK 3: CVaR (tail risk) ═══════════════
    cvar = metrics.get("CVaR_95_Daily", 0)
    if cvar < limits["max_cvar_daily"]:  # CVaR is negative
        critical.append(
            f"⛔ CVaR: Ожидаемый убыток в худшие 5% дней = {cvar:.2%} "
            f"(лимит: {limits['max_cvar_daily']:.2%}). Портфель сверхрискован."
        )

    # ═══════════════ CHECK 4: Sharpe Ratio ═══════════════
    sharpe = metrics.get("Sharpe_Ratio", 0)
    if sharpe is not None and sharpe < limits["min_sharpe"]:
        warnings.append(
            f"⚠️ SHARPE: {sharpe:.2f} < {limits['min_sharpe']}. "
            f"Портфель неэффективен (доходность не компенсирует риск)."
        )

    # ═══════════════ CHECK 5: Total volatility ═══════════════
    vol = metrics.get("Total_Volatility_Ann", 0)
    if vol > limits["max_portfolio_volatility"]:
        critical.append(
            f"⛔ ВОЛАТИЛЬНОСТЬ: {vol:.1%} годовая "
            f"(лимит: {limits['max_portfolio_volatility']:.1%}). Активы разрушают портфель."
        )

    # ═══════════════ CHECK 6: ATR Spike ═══════════════
    if "ATR_Pct" in df.columns:
        for _, row in df.iterrows():
            ticker = row.get("Ticker", "?")
            atr_pct = row.get("ATR_Pct", 0)
            if isinstance(atr_pct, (int, float)) and atr_pct > 3.0:
                warnings.append(
                    f"⚠️ IMS/ATR: {ticker} ATR = {atr_pct:.2f}% от цены. "
                    f"Высокая внутридневная волатильность."
                )

    # ═══════════════ CHECK 7: Fundamental Score ═══════════════
    if "Fundamental_Score" in df.columns:
        for _, row in df.iterrows():
            ticker = row.get("Ticker", "?")
            fscore = row.get("Fundamental_Score", 50)
            if isinstance(fscore, (int, float)) and fscore < 30:
                warnings.append(
                    f"⚠️ ФУНДАМЕНТАЛ: {ticker} SEC Score = {fscore:.0f}/100. "
                    f"Слабые финансы (маржа, долг или ROE)."
                )

    # ═══════════════ RESULT (non-blocking) ═══════════════
    passed = len(critical) == 0

    summary_parts = []
    if critical:
        summary_parts.append("⚠️ РЕКОМЕНДАЦИИ АУДИТОРА:")
        summary_parts.extend(critical)
    if warnings:
        summary_parts.append("ПРЕДУПРЕЖДЕНИЯ:")
        summary_parts.extend(warnings)
    if passed and not warnings:
        summary_parts.append("✅ Все проверки пройдены. Портфель в рамках лимитов.")

    summary = "\n".join(summary_parts)

    if not passed:
        logger.warning("Gatekeeper обнаружил нарушения: %d критических (advisory only).",
                       len(critical))

    return {
        "passed": passed,
        "critical": critical,
        "warnings": warnings,
        "summary": summary
    }
