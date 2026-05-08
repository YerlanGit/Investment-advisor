"""
PDF payload builder — produces the dict consumed by report_basic.html and
report_deep.html (Phase 4).

Lives outside tg_bot.py so it has no aiogram / cryptography imports and can
be unit-tested in isolation.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _format_pnl_pct(x: float) -> str:
    return f"{x*100:+.1f}%"


def _format_pnl_abs(x: float) -> str:
    return f"{x:+,.0f}"


def _classify_asset(ticker: str) -> str:
    """Lightweight asset-class label for the PDF table."""
    t = (ticker or "").upper().strip()
    base = t.split(".")[0] if "." in t else t
    suffix = t.rsplit(".", 1)[-1] if "." in t else ""
    if base in {"USD", "EUR", "RUB", "KZT", "CASH"}:
        return "Ден. средства"
    if base in {"BTC", "ETH", "SOL", "BNB", "DOGE"} or t.endswith("-USD"):
        return "Крипто"
    if base in {"GLD", "SLV", "GDX", "USO", "DBC", "PDBC", "GOLD", "SILVER", "OIL"}:
        return "Сырьё"
    if base in {"TLT", "AGG", "BND", "LQD", "HYG", "IEF", "BIL", "EMB"} or "BOND" in base or "OVD" in base:
        return "Облигации"
    if suffix in {"KZ", "IL"} or t.endswith(".AIX"):
        return "Акции KZ"
    if suffix == "US" or len(base) <= 5:
        return "Акции США"
    return "Прочее"


def _action_color(action: Optional[str]) -> str:
    return {
        "Strong Buy": "pos",
        "Buy":        "pos",
        "Hold":       "neut",
        "Trim":       "warn",
        "Sell":       "neg",
    }.get(action or "", "neut")


def _risk_score_label(score: int) -> str:
    if score < 33:  return "Консервативный"
    if score < 66:  return "Умеренный"
    return "Агрессивный"


# ── Public API ───────────────────────────────────────────────────────────────

TIER_BASE = "base"
TIER_DEEP = "deep"


def build_payload(results: dict, tier: str,
                  ai_summary: Optional[dict] = None) -> dict:
    """
    Build the payload consumed by report_basic.html / report_deep.html.

    Args:
        results    : The dict returned by UniversalPortfolioManager.analyze_all().
        tier       : 'base' or 'deep' — controls which sections are populated.
        ai_summary : Optional AI-narrative payload from advisor_bot:
                     {'verdict': str, 'bullets': list[str], 'action_plan_text': str}
    """
    metrics    = results.get("portfolio_metrics") or {}
    perf_df    = results.get("performance_table")
    total_val  = _safe_float(results.get("total_value"), 1.0) or 1.0

    # ── KPIs ───────────────────────────────────────────────────────────────
    cvar_raw   = _safe_float(metrics.get("CVaR_95_Daily"),         0.0)
    cvar_boot  = metrics.get("CVaR_95_Bootstrap") or {}
    sharpe_raw = _safe_float(metrics.get("Sharpe_Ratio"),          float("nan"))
    sortino_raw= _safe_float(metrics.get("Sortino_Ratio"),         float("nan"))
    var_raw    = _safe_float(metrics.get("VaR_95_Daily"),          0.0)
    mdd_raw    = _safe_float(metrics.get("Max_Drawdown"),          0.0)
    vol_raw    = _safe_float(metrics.get("Total_Volatility_Ann"),  0.0)
    composite  = int(_safe_float(metrics.get("Composite_Risk_Score"),
                                 vol_raw / 0.40 * 100))

    cvar_str    = f"{cvar_raw * 100:.1f}%"
    cvar_lo     = cvar_boot.get("lo95")
    cvar_hi     = cvar_boot.get("hi95")
    cvar_ci_str = ""
    if cvar_lo is not None and cvar_hi is not None:
        cvar_ci_str = f"CI: {cvar_lo*100:.1f}% … {cvar_hi*100:.1f}%"
    sharpe_str  = f"{sharpe_raw:.2f}" if not math.isnan(sharpe_raw) else "—"
    sortino_str = f"{sortino_raw:.2f}" if not math.isnan(sortino_raw) else "—"
    var_str     = f"{var_raw * 100:.1f}%"
    mdd_str     = f"{mdd_raw * 100:.1f}%"
    vol_str     = f"{vol_raw * 100:.1f}%"

    # ── Aggregate P/L since position entry ─────────────────────────────────
    total_pnl  = 0.0
    total_cost = 0.0
    if perf_df is not None and not perf_df.empty:
        if "PnL" in perf_df.columns:
            total_pnl = float(perf_df["PnL"].fillna(0).sum())
        if "Total_Cost" in perf_df.columns:
            total_cost = float(perf_df["Total_Cost"].fillna(0).sum())
    total_return_pct = (total_pnl / total_cost) if total_cost > 0 else 0.0

    # ── Per-asset rows for the master table ────────────────────────────────
    asset_scores = results.get("asset_scores") or {}
    technicals_dict = results.get("technicals") or {}

    assets: list[dict] = []
    if perf_df is not None and not perf_df.empty:
        # Best/Worst within position-to-date P/L.
        best_t, best_pct, best_abs = None, -1e9, 0.0
        worst_t, worst_pct, worst_abs = None, +1e9, 0.0

        for _, row in perf_df.iterrows():
            ticker     = str(row.get("Ticker", "—"))
            cur_val    = _safe_float(row.get("Current_Value"), 0.0)
            weight_pct = cur_val / total_val * 100
            euler      = _safe_float(row.get("Euler_Risk_Contribution_Pct"), 0.0)
            atr_pct    = row.get("ATR_Pct")
            atr_pct    = None if atr_pct is None or (isinstance(atr_pct, float) and math.isnan(atr_pct)) else float(atr_pct)
            mvar       = row.get("Marginal_VaR_Daily")
            mvar       = None if mvar is None or (isinstance(mvar, float) and math.isnan(mvar)) else float(mvar)
            pnl_abs    = _safe_float(row.get("PnL"),        0.0)
            ret_pct    = _safe_float(row.get("Return_Pct"), 0.0)

            sc = asset_scores.get(ticker, {})
            total_score   = sc.get("total")
            action        = sc.get("action") or "Hold"
            hotspot       = bool(sc.get("hotspot"))

            if ret_pct > best_pct:
                best_pct, best_abs, best_t = ret_pct, pnl_abs, ticker
            if ret_pct < worst_pct:
                worst_pct, worst_abs, worst_t = ret_pct, pnl_abs, ticker

            assets.append({
                "ticker":        ticker,
                "weight":        f"{weight_pct:.1f}%",
                "asset_class":   _classify_asset(ticker),
                "euler_risk":    f"{euler:.1f}%",
                "atr_pct":       f"{atr_pct:.2f}%" if atr_pct is not None else "—",
                "mvar_bps":      f"{mvar*10000:.0f}" if mvar is not None else "—",
                "pnl_pct":       _format_pnl_pct(ret_pct),
                "pnl_abs":       _format_pnl_abs(pnl_abs),
                "pnl_color":     "pos" if ret_pct >= 0 else "neg",
                "total_score":   "—" if total_score is None else f"{total_score:+.1f}",
                "score_color":   "pos" if (total_score or 0) > 0 else
                                  ("neg" if (total_score or 0) < 0 else "neut"),
                "action":        action,
                "action_color":  _action_color(action),
                "hotspot":       hotspot,
            })

        best  = {"ticker": best_t  or "—", "pnl_pct": _format_pnl_pct(best_pct  if best_t else 0.0),
                 "pnl_abs": _format_pnl_abs(best_abs)}  if best_t else None
        worst = {"ticker": worst_t or "—", "pnl_pct": _format_pnl_pct(worst_pct if worst_t else 0.0),
                 "pnl_abs": _format_pnl_abs(worst_abs)} if worst_t else None
    else:
        best, worst = None, None

    # ── Risk hotspots (top critical) ───────────────────────────────────────
    hotspots: list[str] = []
    for a in assets:
        if a["hotspot"]:
            hotspots.append(f"{a['ticker']}: TRC {a['euler_risk']}, вес {a['weight']}")
    hotspots = hotspots[:3]

    # ── Sector exposure for the bars chart ─────────────────────────────────
    sector_exposure = results.get("sector_exposure") or {}
    sectors = [
        {"name": s, "weight_pct": float(w) * 100, "weight_str": f"{float(w)*100:.0f}%"}
        for s, w in sector_exposure.items()
    ]

    # ── Macro regime (Cover) ───────────────────────────────────────────────
    regime_block = None
    regime = results.get("regime")
    if regime:
        regime_block = {
            "label":      regime["regime"],
            "confidence": int(round(regime["confidence"] * 100)),
            "growth":     regime["growth_score"],
            "cycle":      regime["cycle_score"],
        }

    # ── Common payload (always rendered) ───────────────────────────────────
    payload: dict = {
        # KPIs
        "cvar":              cvar_str,
        "cvar_ci":           cvar_ci_str,
        "sharpe":            sharpe_str,
        "sortino":           sortino_str,
        "var_95_daily":      var_str,
        "max_drawdown":      mdd_str,
        "volatility":        vol_str,
        "risk_pct":          composite,
        "risk_label":        _risk_score_label(composite),
        # Aggregate P/L since entry
        "pnl_total_abs":     _format_pnl_abs(total_pnl),
        "pnl_total_pct":     _format_pnl_pct(total_return_pct),
        "pnl_total_color":   "pos" if total_return_pct >= 0 else "neg",
        "pnl_best":          best,
        "pnl_worst":         worst,
        # Holdings + risk
        "assets":            assets,
        "hotspots":          hotspots,
        "sectors":           sectors,
        # Regime
        "regime":            regime_block,
        # AI Narrative — placeholder unless caller passed one in
        "ai_verdict":        (ai_summary or {}).get("verdict", ""),
        "ai_bullets":        (ai_summary or {}).get("bullets", []),
        # Tier metadata
        "tier":              tier,
    }

    # ── Deep-tier additions ────────────────────────────────────────────────
    if tier == TIER_DEEP:
        # Benchmarks (annualised excess + IR)
        bm_data   = results.get("benchmark_comparison") or {}
        scenarios = []
        for bm_name, bm in bm_data.items():
            excess_ann = bm.get("Excess_Return_Ann")
            if excess_ann is None:
                excess_ann = _safe_float(bm.get("Excess_Return"), 0.0)
            else:
                excess_ann = _safe_float(excess_ann, 0.0)
            ir = _safe_float(bm.get("Information_Ratio"), 0.0)
            te = _safe_float(bm.get("Tracking_Error"), 0.0)
            scenarios.append({
                "name":    bm_name,
                "excess":  _format_pnl_pct(excess_ann),
                "te":      f"{te*100:.1f}%" if te else "—",
                "ir":      f"{ir:.2f}" if ir else "—",
                "beating": bool(bm.get("Beating_Benchmark")),
                "color":   "pos" if excess_ann >= 0 else "neg",
            })
        payload["scenarios"] = scenarios

        # Score breakdown table — pillar contributions per asset
        score_breakdown = []
        for ticker, sc in (asset_scores or {}).items():
            score_breakdown.append({
                "ticker":       ticker,
                "fundamentals": f"{sc.get('fundamentals', 0):+.1f}",
                "valuations":   f"{sc.get('valuations', 0):+.1f}",
                "technicals":   f"{sc.get('technicals', 0):+.1f}",
                "credit":       f"{sc.get('credit', 0):+.1f}",
                "total":        f"{sc.get('total', 0):+.1f}",
                "action":       sc.get("action", "Hold"),
                "action_color": _action_color(sc.get("action")),
            })
        payload["score_breakdown"] = score_breakdown

        # Action plan (with Buy zone / Sell target / Stop)
        action_plan = results.get("action_plan") or []
        plan_rows = []
        for r in action_plan:
            buy_zone   = r.get("buy_zone")
            sell_zone  = r.get("sell_zone")
            tgt        = r.get("take_target")
            stop       = r.get("stop_loss")
            plan_rows.append({
                "ticker":      r.get("ticker"),
                "action":      r.get("action"),
                "action_color":_action_color(r.get("action")),
                "delta_w_pp":  f"{r.get('delta_w_pp', 0):+.1f}",
                "qty_delta":   r.get("qty_delta") if r.get("qty_delta") is not None else "—",
                "buy_zone":    f"{buy_zone[0]:.2f} – {buy_zone[1]:.2f}" if buy_zone else "—",
                "sell_zone":   f"{sell_zone[0]:.2f} – {sell_zone[1]:.2f}" if sell_zone else "—",
                "take_target": f"{tgt:.2f}"  if tgt  is not None else "—",
                "stop_loss":   f"{stop:.2f}" if stop is not None else "—",
                "reason":      r.get("reason", ""),
            })
        payload["action_plan"] = plan_rows

        # Fundamental layer — SEC-derived columns from perf table
        fundamental_rows = []
        if perf_df is not None and not perf_df.empty:
            for _, row in perf_df.iterrows():
                ticker = str(row.get("Ticker", "—"))
                if str(row.get("Fundamental_Sector") or "default") in ("default", "EM_Proxy"):
                    continue
                fundamental_rows.append({
                    "ticker":   ticker,
                    "roe":      _fmt_pct(row.get("SEC_ROE")),
                    "op_m":     _fmt_pct(row.get("SEC_Op_Margin")),
                    "dta":      _fmt_pct(row.get("SEC_Debt_to_Assets")),
                    "rev_g":    _fmt_pct(row.get("SEC_Revenue_Growth_YoY")),
                    "fcf_m":    _fmt_pct(row.get("SEC_FCF_Margin")),
                    "altman_z": _fmt_num(row.get("SEC_Altman_Z"), digits=2),
                    "altman_zone": row.get("SEC_Altman_Zone") or "—",
                    "piotroski": _fmt_int(row.get("SEC_Piotroski_F")),
                    "int_cov":  _fmt_num(row.get("SEC_Interest_Coverage"), digits=1),
                })
        payload["fundamental_layer"] = fundamental_rows

        # Black-Litterman target-weight summary
        bl_records = results.get("black_litterman") or []
        payload["bl_records"] = [
            {
                "ticker":      r["ticker"],
                "current_w":   f"{r['current_w']*100:.1f}%",
                "target_w":    f"{r['target_w']*100:.1f}%",
                "delta_w_pp":  f"{r['delta_w_pp']:+.1f}",
                "color":       "pos" if r["delta_w_pp"] >= 0 else "neg",
            }
            for r in bl_records
        ]

        # AI deep narrative (full bullets list + action-plan text)
        if ai_summary:
            payload["ai_action_text"] = ai_summary.get("action_plan_text", "")

    return payload


def _fmt_pct(v) -> str:
    try:
        x = float(v)
        if math.isnan(x): return "—"
        return f"{x*100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(v, digits: int = 2) -> str:
    try:
        x = float(v)
        if math.isnan(x): return "—"
        return f"{x:.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_int(v) -> str:
    try:
        x = int(v)
        return str(x)
    except (TypeError, ValueError):
        return "—"


__all__ = ["build_payload", "TIER_BASE", "TIER_DEEP"]
