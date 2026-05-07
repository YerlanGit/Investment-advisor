"""
Gatekeeper — Deterministic portfolio audit (non-blocking).

Checks the final report BEFORE showing to user. Runs in 0 ms, costs $0, never hallucinates.

2026-05 update: Gatekeeper is now ADVISORY-ONLY (non-blocking).
Critical violations are reported as warnings to the user but do NOT prevent
the report from being delivered. This ensures data upload is never blocked.

Checks implemented:
  1. Euler Risk concentration
  2. Capital concentration (single asset weight)
  3. CVaR tail risk (profile-aware threshold)
  4. Sharpe Ratio efficiency
  5. Total portfolio volatility
  6. ATR spike detection
  7. Fundamental score weakness
  8. Mandate compliance (actual allocation vs limits_dict)
  9. Tracking Error vs profile target_te
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

# ── Asset-class classifier for Mandate Compliance (Check 8) ──────────────────
# Maps portfolio tickers to the ASSET_KEYS used in limits_dict.
# This must align with profile_manager.ASSET_KEYS.

_CLASS_CRYPTO = {"BTC", "ETH", "SOL", "BNB", "DOGE", "ADA", "DOT"}
_CLASS_COMMODITIES = {
    "GLD", "SLV", "GDX", "USO", "UNG", "DBC", "PDBC",
    "GOLD", "SILVER", "OIL",
}
_CLASS_BONDS = {
    "TLT", "AGG", "BND", "TNX", "LQD", "HYG", "BIL", "IEF", "SHY", "EMB",
}
_CLASS_KZ = {"KSPI", "HSBK", "KAP", "KZTK", "KCEL", "BAST", "HRGL", "KZAP"}
_CLASS_ETF = {
    "SPY", "QQQ", "IWM", "EEM", "URTH", "VTI", "VOO",
    "MTUM", "VLUE", "QUAL", "XLK", "XLF", "XLE", "XLV",
    "SOXX", "XME", "XLP", "XLI",
}


def _classify_to_asset_key(ticker: str) -> str:
    """Classify a ticker into the asset-class keys used by limits_dict."""
    t = ticker.split(".")[0].upper() if "." in ticker else ticker.upper()
    suffix = ticker.upper().rsplit(".", 1)[-1] if "." in ticker else ""

    if t in _CLASS_CRYPTO or "-USD" in ticker.upper():
        return "Crypto"
    if t in _CLASS_COMMODITIES:
        return "Commodities"
    if t in _CLASS_BONDS:
        return "Bonds"
    # ISIN-pattern bonds from Freedom Finance
    if t.startswith(("KZ2P", "KZ1P", "XS", "US912")):
        return "Bonds"
    if "BOND" in t or "OVD" in t or "FFSPC" in t:
        return "Bonds"
    if t in _CLASS_KZ or suffix in ("KZ", "IL") or ticker.upper().endswith(".AIX"):
        return "Stocks_KZ"
    if t in _CLASS_ETF:
        return "GlobalETFs"
    # Cash-like
    if t in ("USD", "EUR", "CASH", "RUB", "KZT"):
        return "Bonds"  # treat cash as bond-like for allocation purposes
    # Default: US stocks
    return "Stocks_US"


def run_gatekeeper(
    report: dict,
    user_limits: dict = None,
    user_profile: dict = None,
) -> dict:
    """
    Advisory-only deterministic audit.

    NEVER blocks report delivery. Returns violations and warnings for
    display in Telegram, but the report PDF is always generated.

    Args:
        report: Dict from UniversalPortfolioManager.analyze_all()
        user_limits: User-specific limits (or DEFAULT_LIMITS)
        user_profile: Full profile dict from get_profile() — includes
                      limits_dict, target_te, benchmark_ticker, etc.

    Returns:
        {
            "passed": bool,      # True if no critical violations
            "critical": [...],   # Critical violations (advisory, not blocking)
            "warnings": [...],   # Non-critical warnings
            "summary": str       # Brief text for Telegram
        }
    """
    limits = {**DEFAULT_LIMITS, **(user_limits or {})}

    # Scale CVaR threshold by profile's target volatility if available.
    # Aggressive investors accept deeper tail losses.
    if user_profile and "target_volatility" in user_profile:
        target_vol = user_profile["target_volatility"]
        # Scale: Conservative (5%) → -5%, Aggressive (20%) → -8%
        limits["max_cvar_daily"] = -0.05 * (1 + target_vol)

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

    # ═══════════════ CHECK 8: Mandate Compliance ═══════════════
    # Compare actual portfolio weights by asset class against the user's
    # limits_dict from their approved investment mandate.
    if user_profile and "limits_dict" in user_profile:
        mandate_limits = user_profile["limits_dict"]

        # Build actual allocation by asset class
        actual_alloc: dict[str, float] = {}
        ticker_col = "Ticker"
        if ticker_col in df.columns:
            total_value = 0
            for _, row in df.iterrows():
                cv = row.get("Current_Value", 0) or 0
                total_value += cv

            if total_value > 0:
                for _, row in df.iterrows():
                    ticker = row.get(ticker_col, "?")
                    cv = row.get("Current_Value", 0) or 0
                    weight_pct = (cv / total_value) * 100
                    asset_class = _classify_to_asset_key(ticker)
                    actual_alloc[asset_class] = actual_alloc.get(asset_class, 0) + weight_pct

                for asset_class, (lo, hi) in mandate_limits.items():
                    actual = actual_alloc.get(asset_class, 0)
                    if actual > hi + 2:  # 2% tolerance for rounding
                        critical.append(
                            f"⛔ МАНДАТ: {asset_class} = {actual:.0f}% "
                            f"(макс. по мандату: {hi}%). "
                            f"Портфель нарушает утверждённую стратегию."
                        )
                    elif actual < lo - 2 and lo > 0:  # only flag if minimum is meaningful
                        warnings.append(
                            f"⚠️ МАНДАТ: {asset_class} = {actual:.0f}% "
                            f"(мин. по мандату: {lo}%). Недовес класса активов."
                        )

    # ═══════════════ CHECK 9: Tracking Error vs Target ═══════════════
    # If the profile has a target_te, compare the actual TE from benchmark
    # comparison against it. Advisory only — informative, non-blocking.
    if user_profile and "target_te" in user_profile:
        target_te = user_profile["target_te"]
        bm_data = report.get("benchmark_comparison", {})
        profile_bm = bm_data.get("Профильный бенчмарк", {})
        actual_te = profile_bm.get("Tracking_Error")
        if actual_te is not None and actual_te > target_te * 1.5:
            warnings.append(
                f"⚠️ TRACKING ERROR: {actual_te:.1%} (целевой: {target_te:.0%}). "
                f"Портфель значительно отклоняется от бенчмарка мандата."
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
