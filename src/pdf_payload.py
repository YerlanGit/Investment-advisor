"""
PDF payload builder — produces the dict consumed by report_basic.html and
report_deep.html (Phase 4).

Lives outside tg_bot.py so it has no aiogram / cryptography imports and can
be unit-tested in isolation.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
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


# Asset-class label — single source of truth in finance.scoring.
# Aliased to the historical private name so existing call sites are untouched.
from finance.scoring import classify_asset_class as _classify_asset


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


# H1: Russian display label for the risk-mandate badge next to the gauge.
_MANDATE_RU = {
    "CONSERVATIVE": "Консервативный",
    "MODERATE":     "Умеренный",
    "AGGRESSIVE":   "Агрессивный",
}


def _risk_mandate_label(mandate: str) -> str:
    return _MANDATE_RU.get(str(mandate).strip().upper(), "Умеренный")


# Regime quadrant SVG geometry (must match templates/report_deep_v3.html).
#   viewBox 0..310, centre (155,155), axis labels mark ±0.2 at the quadrant
#   edges → 625 px per unit (verified: growth +0.08 → 50 px above centre).
_REGIME_DOT_CENTRE = 155.0
_REGIME_DOT_SCALE  = 625.0
_REGIME_DOT_MIN    = 40.0     # keep the 8px dot fully inside the plot frame
_REGIME_DOT_MAX    = 270.0


def _regime_dot_coords(growth: float, cycle: float) -> tuple[float, float]:
    """Map (growth, cycle) signed scores to SVG (cx, cy) on the regime quadrant.

    cycle drives the X axis (→ right = +), growth drives the Y axis (↑ up = +,
    so a positive growth yields a SMALLER y).  Scores are clamped to ±0.2 (the
    labelled axis range) and the result clamped to the plot frame so an extreme
    reading never paints the dot outside the chart.
    """
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    g = _clamp(float(growth), -0.2, 0.2)
    c = _clamp(float(cycle),  -0.2, 0.2)
    cx = _clamp(_REGIME_DOT_CENTRE + c * _REGIME_DOT_SCALE,
                _REGIME_DOT_MIN, _REGIME_DOT_MAX)
    cy = _clamp(_REGIME_DOT_CENTRE - g * _REGIME_DOT_SCALE,
                _REGIME_DOT_MIN, _REGIME_DOT_MAX)
    return round(cx, 1), round(cy, 1)


def _build_mandate_compliance(perf_df, total_val: float,
                              user_profile: Optional[dict]) -> Optional[dict]:
    """Sprint-5 Task 4 — make the mandate visible in the report.

    Compares the LIVE portfolio's asset-class allocation against the user's
    approved ``limits_dict`` and returns a per-class table the templates render
    so the investor sees BOTH their target profile AND how well the current
    book matches it.  Returns None when there is no profile / no limits (the
    template then hides the panel).  Pure, never raises.
    """
    if not user_profile:
        return None
    limits = user_profile.get("limits_dict") or {}
    if not limits or perf_df is None or getattr(perf_df, "empty", True):
        return None
    try:
        from agent.gatekeeper import _classify_to_asset_key
        from profile_manager import ASSET_DISPLAY
    except Exception:
        return None

    # Actual allocation by asset class (share of NAV, %).
    actual: dict[str, float] = {}
    if "Ticker" in perf_df.columns and total_val:
        for _, row in perf_df.iterrows():
            cv = _safe_float(row.get("Current_Value"), 0.0)
            cls = _classify_to_asset_key(str(row.get("Ticker", "?")))
            actual[cls] = actual.get(cls, 0.0) + (cv / total_val) * 100.0

    rows: list[dict] = []
    breaches = 0
    for key, bounds in limits.items():
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError, IndexError):
            continue
        if lo == 0 and hi == 0:
            continue   # asset class not part of this mandate — skip
        act = round(actual.get(key, 0.0), 1)
        if act > hi + 2:
            status = "over"
            breaches += 1
        elif act < lo - 2:
            status = "under"
        else:
            status = "ok"
        rows.append({
            "key":    key,
            "label":  ASSET_DISPLAY.get(key, key),
            "actual": act,
            "lo":     lo,
            "hi":     hi,
            "status": status,
        })

    if not rows:
        return None
    rows.sort(key=lambda r: r["actual"], reverse=True)
    return {
        "profile_name": user_profile.get("profile_name", "—"),
        "target_vol_pct": round(_safe_float(user_profile.get("target_volatility"), 0.0) * 100, 1),
        "target_te_pct":  round(_safe_float(user_profile.get("target_te"), 0.0) * 100, 1),
        "rows":     rows,
        "breaches": breaches,
        "compliant": breaches == 0,
    }


def _model_display_name(model_id: str) -> str:
    """Convert internal model ID to short human-readable label for PDF."""
    _MAP = {
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
        "claude-haiku-4-5":          "Claude Haiku 4.5",
        "claude-sonnet-4-6":         "Claude Sonnet 4.6",
        "claude-opus-4-7":           "Claude Opus 4.7",
        "fallback":                  "Fallback (нет API)",
    }
    if not model_id:
        return ""
    for key, label in _MAP.items():
        if key in model_id:
            return label
    return model_id


def _build_ai_ideas(stock_picks: dict, tier: str = "base") -> dict:
    """
    Reshape the AI narrative's stock-picks into the idea-card schema the v3
    templates render.

    Input  (ai_narrative.generate_narrative → stock_picks):
        {boost_alpha|rebalance|protect_capital|regime_play:
            {label, desc, picks: [{ticker, name, why, type}]}}
    Output (template ideas grid — buckets flattened in canonical order):
        {risk_reduction|diversification|growth|hedge|rotation:
            [{idea_num, category, priority, title, rationale,
              pipeline, candidates: [{ticker, name, scenario}]}]}

    Pipeline is synthesised from the pick types so the template renders the
    Factor→Regime→RAG (BASE) or Factor→Regime→Stress→RAG (DEEP) notation.
    """
    _MAP = [
        ("boost_alpha",     "growth",          "Рост доходности",   "high"),
        ("rebalance",       "diversification", "Ребалансировка",    "medium"),
        ("protect_capital", "hedge",           "Защита капитала",   "low"),
        ("regime_play",     "rotation",        "Режимная ставка",   "medium"),
    ]
    ideas: dict = {"risk_reduction": [], "diversification": [],
                   "growth": [], "hedge": [], "rotation": []}

    # Pipeline stage templates per tier and idea type
    _PIPELINE_BASE = {
        "boost_alpha":     [("FACTOR", "Momentum + Quality скоринг"),
                            ("REGIME", "Соответствие текущему режиму роста"),
                            ("RAG",    "Подтверждение банковскими отчётами")],
        "rebalance":       [("FACTOR", "4-Pillar Scoring F+V+T+C"),
                            ("REGIME", "Секторная ротация vs режим"),
                            ("RAG",    "Фундаментальный анализ")],
        "protect_capital": [("FACTOR", "Низкая Beta + дивидендная стабильность"),
                            ("REGIME", "Защитные сектора при текущем режиме"),
                            ("RAG",    "Оценка хвостового риска")],
        "regime_play":     [("FACTOR", "Факторный сигнал по режиму"),
                            ("REGIME", "Quadrant: Growth × Cycle"),
                            ("RAG",    "Банковские прогнозы режима")],
    }
    _PIPELINE_DEEP = {
        "boost_alpha":     [("FACTOR", "Momentum + Quality скоринг"),
                            ("REGIME", "Соответствие Growth × Cycle квадранту"),
                            ("STRESS", "Устойчивость при rate shock +200 bps"),
                            ("RAG",    "Подтверждение инвестбанками")],
        "rebalance":       [("FACTOR", "4-Pillar Scoring F+V+T+C"),
                            ("REGIME", "Macro Alignment vs текущий режим"),
                            ("STRESS", "Поведение при equity shock −20%"),
                            ("RAG",    "SEC EDGAR фундаментал")],
        "protect_capital": [("FACTOR", "Низкая Beta, дивидендный аристократ"),
                            ("REGIME", "Защитные сектора Healthcare/Gold"),
                            ("STRESS", "Positive P&L при recession сценарии"),
                            ("RAG",    "CDS + банковские отчёты по риску")],
        "regime_play":     [("FACTOR", "Regime-specific факторный сигнал"),
                            ("REGIME", "Growth-Cycle квадрант + confidence"),
                            ("STRESS", "Сценарный анализ смены режима"),
                            ("RAG",    "Банковские прогнозы режима")],
    }
    pipeline_map = _PIPELINE_DEEP if tier == "deep" else _PIPELINE_BASE

    num = 0
    for src_key, bucket, category, priority in _MAP:
        scenario = (stock_picks or {}).get(src_key) or {}
        picks    = scenario.get("picks") or []
        if not picks:
            continue
        num += 1
        candidates = [
            {"ticker":   str(p.get("ticker", "")),
             "name":     str(p.get("name", "")),
             "scenario": str(p.get("why", ""))}
            for p in picks
        ]
        ideas[bucket].append({
            "idea_num":   f"{num:02d}",
            "category":   category,
            "priority":   priority,
            "title":      str(scenario.get("label") or category),
            "rationale":  str(scenario.get("desc") or ""),
            "pipeline":   pipeline_map.get(src_key, []),
            "candidates": candidates,
        })
    return ideas


def _build_macro_drivers_panel(raw: Optional[dict]) -> dict:
    """
    Adapt MacroFeed.get_regime_drivers() output to the DEEP «Сигналы-драйверы»
    panel schema.

    Engine returns a FLAT dict keyed by series key:
        {yield_curve_10y2y: {series_id, value, as_of, status, unit, label, ...}, ...}
    Template (report_deep_v3.html:2810) reads:
        {series: [{id, name, value, as_of, status, comment}], as_of}
    Without this remap `data.macro_drivers.series` is None → the panel
    falls through to "FRED источник недоступен" even when ALL 4 series
    are fresh (proved by the CoVe lineage rendering them ok same run).
    """
    if not raw:
        return {}
    # Per-unit value formatter — keeps the panel display compact and honest
    # ("+0.18 pp" / "312 bp" / "14.2" / "2.32%").  HY OAS in FRED is reported
    # in percent (3.12) but bond-market convention is basis points (×100).
    def _fmt(value: float, unit: Optional[str], series_id: str) -> str:
        if value is None or not isinstance(value, (int, float)):
            return "—"
        if series_id == "BAMLH0A0HYM2":
            return f"{value * 100:.0f} bp"
        if unit == "pp":
            return f"{'+' if value > 0 else ''}{value:.2f} pp"
        if unit == "%":
            return f"{value:.2f}%"
        if unit == "index":
            return f"{value:.1f}"
        return f"{value:.2f}"

    series_rows: list[dict] = []
    latest_as_of = ""
    for key, row in raw.items():
        if not isinstance(row, dict):
            continue
        series_id = row.get("series_id") or key
        as_of     = str(row.get("as_of") or "")
        series_rows.append({
            "id":      series_id,
            "key":     key,
            "name":    row.get("label") or series_id,
            "value":   _fmt(row.get("value"), row.get("unit"), series_id),
            "as_of":   as_of,
            "status":  row.get("status") or "missing",
            "comment": row.get("note") or "",
        })
        if as_of > latest_as_of:
            latest_as_of = as_of
    return {"series": series_rows, "as_of": latest_as_of}


def _build_expected_effect(raw: Optional[dict]) -> dict:
    """
    Adapt simulate_after_plan() output to the deep template's 8-card schema.

    The engine returns {metrics: {<engine_name>: {before, after, delta,
    improved, delta_pp?}}}; the template's `_ef_card` reads a FLAT dict keyed
    by its OWN card names, each value {before, after, delta_pp, favourable}.
    Without this remap every `_ee.get(card_key)` is None, every card hits the
    falsy guard, and the "Ожидаемый эффект" grid renders empty.

    Key differences bridged here:
      • nesting        — engine wraps metrics under `.metrics`
      • card-name skew — template `vol`/`max_erc_pct` vs engine
                          `volatility_ann`/`max_trc`
      • field name     — engine `improved` vs template `favourable`
      • delta_pp       — engine only sets it for as_pp metrics; for the
                          others (risk_index points, sharpe) fall back to the
                          raw delta, and scale max_trc's fraction to pp.
    """
    metrics = (raw or {}).get("metrics") or {}
    # template card key -> engine metric key
    _KEYMAP = (
        ("risk_index",      "risk_index"),
        ("cvar_95",         "cvar_95"),
        ("sharpe",          "sharpe"),
        ("max_drawdown",    "max_drawdown"),
        ("vol",             "volatility_ann"),
        ("max_erc_pct",     "max_trc"),
        ("it_share",        "it_share"),
        ("expected_return", "expected_return"),
    )
    out: dict = {}
    for tpl_key, eng_key in _KEYMAP:
        cell = metrics.get(eng_key)
        if not isinstance(cell, dict):
            continue
        before, after = cell.get("before"), cell.get("after")
        delta_pp = cell.get("delta_pp")
        if tpl_key == "max_erc_pct":
            # Engine max_trc is ALREADY a percent (0–100); the template card
            # renders it with fmt='pct' (×100).  Rescale to a fraction so it
            # is not double-scaled (17.2% → 1724%); the raw delta is already
            # in percentage points.
            before   = before / 100.0 if before is not None else None
            after    = after  / 100.0 if after  is not None else None
            delta_pp = cell.get("delta")
        elif delta_pp is None:
            # risk_index (points) and sharpe (raw) have no as_pp delta_pp —
            # fall back to the raw delta in its own display unit.
            delta_pp = cell.get("delta")
        out[tpl_key] = {
            "before":     before,
            "after":      after,
            "delta_pp":   delta_pp,
            "favourable": cell.get("improved"),
        }
    # Pass-through the composite verdict computed by simulate_after_plan
    # (improvement / tradeoff / degradation / neutral) — the template
    # uses it to colour-code and label the rebalance honestly.
    verdict = (raw or {}).get("verdict")
    if isinstance(verdict, dict):
        out["verdict"] = {
            "kind":     str(verdict.get("kind", "neutral")),
            "headline": str(verdict.get("headline", "")),
            "worsened": list(verdict.get("worsened") or []),
        }
    return out


def _adapt_period_returns(raw: Optional[dict]) -> dict:
    """
    Adapt compute_period_returns_table() output to the template row schema.

    Engine row: {period, n_days, port_pct, bm_pct, excess_pp} — period is a
    lowercase code ("1m".."12m"/"YTD"), the three returns are decimal floats
    or None.  Template row: {label, portfolio, benchmark, excess} — all
    pre-formatted strings (the renderer does no transform, so the period
    table rendered 6 blank rows from the key/type mismatch).
    """
    _LABELS = {"1m": "1М", "3m": "3М", "6m": "6М", "12m": "12М", "YTD": "YTD"}
    out: dict = {}
    for bm_name, block in (raw or {}).items():
        rows = []
        for r in (block or {}).get("periods", []):
            port, bm, exc = r.get("port_pct"), r.get("bm_pct"), r.get("excess_pp")
            rows.append({
                "label":     _LABELS.get(r.get("period"), r.get("period") or "—"),
                "portfolio": _format_pnl_pct(port) if port is not None else "—",
                "benchmark": _format_pnl_pct(bm)   if bm   is not None else "—",
                "excess":    f"{exc*100:+.1f} пп"  if exc  is not None else "—",
            })
        out[bm_name] = {
            "periods":      rows,
            "window_start": (block or {}).get("window_start"),
            "window_end":   (block or {}).get("window_end"),
        }
    return out


# ── Extreme-value thresholds ────────────────────────────────────────────────
# Used to flag rows that warrant a red icon + AI tooltip.
EXTREME = {
    "atr_pct_high":      3.0,    # %  daily True ATR > 3% of price
    "trc_pct_high":      20.0,   # %  Euler risk contribution
    "sharpe_low":        0.0,    # Sharpe < 0
    "vol_high":          0.30,   # 30% annualised
    "cvar_low":         -0.07,   # -7% one-day CVaR
    "max_dd_low":       -0.20,   # -20% peak-to-trough
    "beta_high":         1.5,    # |Beta_Market|
    "weight_high_pct":   15.0,   # single-asset weight
}


def _flag(value, *, kind: str) -> bool:
    """Return True when `value` is in the 'extreme' tail per `kind`."""
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if math.isnan(v) or math.isinf(v):
        return False
    if kind == "atr_pct":   return v > EXTREME["atr_pct_high"]
    if kind == "trc_pct":   return v > EXTREME["trc_pct_high"]
    if kind == "weight":    return v > EXTREME["weight_high_pct"]
    if kind == "sharpe":    return v < EXTREME["sharpe_low"]
    if kind == "vol":       return v > EXTREME["vol_high"]
    if kind == "cvar":      return v < EXTREME["cvar_low"]
    if kind == "mdd":       return v < EXTREME["max_dd_low"]
    if kind == "beta":      return abs(v) > EXTREME["beta_high"]
    return False


# ── Public API ───────────────────────────────────────────────────────────────

TIER_BASE = "base"
TIER_DEEP = "deep"

# Factor ETFs the engine attempts to load (keep in sync with MAC3RiskEngine).
# SPLV adds the Volatility (low-vol) factor — Step 5 expansion.
_FACTOR_ETFS = [
    "SPY.US", "MTUM.US", "VLUE.US", "QUAL.US", "IWM.US",
    "SPLV.US", "DBC.US", "IEF.US", "EEM.US", "EMB.US",
]

# Benchmark display-name → Tradernet ticker (reverse used to filter scenarios).
_BM_TICKER_TO_NAME: dict[str, str] = {
    "SPY.US":  "S&P 500",
    "QQQ.US":  "Nasdaq 100",
    "IWM.US":  "Russell 2000",
    "EEM.US":  "MSCI EM",
    "EMB.US":  "EM Bonds",
    "AGG.US":  "US Aggregate Bond",
    "URTH.US": "MSCI World",
    "IEF.US":  "US 10Y Treasury",
}


def build_payload(results: dict, tier: str,
                  ai_summary: Optional[dict] = None,
                  user_bench_ticker: Optional[str] = None,
                  prev_snapshot: Optional[dict] = None,
                  regime_rag_confirm: Optional[list] = None,
                  user_profile: Optional[dict] = None) -> dict:
    """
    Build the payload consumed by report_basic.html / report_deep.html.

    Args:
        results      : The dict returned by UniversalPortfolioManager.analyze_all().
        tier         : 'base' or 'deep' — controls which sections are populated.
        ai_summary   : Optional AI-narrative payload from advisor_bot:
                       {'verdict': str, 'bullets': list[str], 'action_plan_text': str}
        user_profile : Optional approved-mandate dict from db_tokenomics.get_profile
                       (profile_name, target_volatility, target_te, limits_dict, …).
                       Sprint-5 Task 4: drives the in-report "соответствие мандату"
                       panel so the questionnaire result is visible, not dead.
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
        # Bare range only — the templates supply their own "CI"/"интервал"
        # label, so prefixing here produced a doubled "CI CI:" in the report.
        cvar_ci_str = f"{cvar_lo*100:.1f}% … {cvar_hi*100:.1f}%"
    sharpe_str  = f"{sharpe_raw:.2f}" if not math.isnan(sharpe_raw) else "—"
    sortino_str = f"{sortino_raw:.2f}" if not math.isnan(sortino_raw) else "—"
    var_str     = f"{var_raw * 100:.1f}%"
    mdd_str     = f"{mdd_raw * 100:.1f}%"
    vol_str     = f"{vol_raw * 100:.1f}%"

    # Risk-free rate the engine used to compute Sharpe / Sortino.
    # Surfaced in the payload so the report's KPI commentary shows the
    # actual RFR (auditable), not a hardcoded template fallback.
    # Source: results.portfolio_metrics.risk_free_rate (echoed by engine
    # at investment_logic.py:1238 from env KZ_RFR_ANNUAL, default 0.14).
    rfr_raw = _safe_float(metrics.get("risk_free_rate"),
                           _safe_float(results.get("risk_free_rate"), 0.14))
    rfr_str = f"{rfr_raw * 100:.0f}%"

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

            # M2: prefer the broker-provided class label (single source of
            # truth from Freedom metadata); fall back to the ticker classifier
            # for demo / proxy rows that have no broker metadata.
            asset_class = (row.get("Asset_Class_Label")
                           or _classify_asset(ticker))
            is_cash     = asset_class == "Ден. средства"

            # Cash is not a tradable risk asset — never carries a 4-pillar
            # score, an action recommendation or a hotspot flag.
            sc = asset_scores.get(ticker, {})
            total_score   = None if is_cash else sc.get("total")
            action        = "—"  if is_cash else (sc.get("action") or "Hold")
            hotspot       = False if is_cash else bool(sc.get("hotspot"))

            # Beta must be NaN-safe across native float, numpy.float64, and
            # pd.NA: the previous `isinstance(x, float) and math.isnan(x)`
            # path silently let `numpy.float64('nan')` through, producing a
            # literal "nan" in the holdings table for some assets while
            # others rendered "—".  np.isfinite handles all three cases.
            beta_mkt_raw = row.get("Beta_Market")
            try:
                if beta_mkt_raw is None or not np.isfinite(float(beta_mkt_raw)):
                    beta_mkt = None
                else:
                    beta_mkt = float(beta_mkt_raw)
            except (TypeError, ValueError):
                beta_mkt = None

            if ret_pct > best_pct:
                best_pct, best_abs, best_t = ret_pct, pnl_abs, ticker
            if ret_pct < worst_pct:
                worst_pct, worst_abs, worst_t = ret_pct, pnl_abs, ticker

            # Extreme-value flags (drive red icons + AI tooltips in template).
            extremes: list[str] = []
            if _flag(weight_pct, kind="weight"):
                extremes.append("вес > 15%")
            if _flag(euler, kind="trc_pct"):
                extremes.append("TRC > 20%")
            if _flag(atr_pct, kind="atr_pct"):
                extremes.append("ATR > 3%")
            assets.append({
                "ticker":        ticker,
                "weight":        f"{weight_pct:.1f}%",
                "weight_pct_num": float(weight_pct),  # raw % for concentration math
                "weight_extreme":_flag(weight_pct, kind="weight"),
                "asset_class":   asset_class,
                "is_cash":       is_cash,
                "euler_risk":    f"{euler:.1f}%",
                "euler_extreme": _flag(euler, kind="trc_pct"),
                "beta":          f"{beta_mkt:.2f}" if beta_mkt is not None else "—",
                "atr_pct":       f"{atr_pct:.2f}%" if atr_pct is not None else "—",
                "atr_extreme":   _flag(atr_pct, kind="atr_pct"),
                "mvar_bps":      f"{mvar*10000:.0f}" if mvar is not None else "—",
                "pnl_pct":       _format_pnl_pct(ret_pct),
                "pnl_abs":       _format_pnl_abs(pnl_abs),
                # Cash positions never carry a real P&L — colour them
                # neutrally so a margin-debt (negative-weight) USD row
                # does not render as "green/profit" (it isn't).
                "pnl_color":     "neut" if is_cash else
                                  ("pos" if ret_pct >= 0 else "neg"),
                "total_score":   "—" if total_score is None else f"{total_score:+.1f}",
                "score_color":   "pos" if (total_score or 0) > 0 else
                                  ("neg" if (total_score or 0) < 0 else "neut"),
                "action":        action,
                "action_color":  _action_color(action),
                "hotspot":       hotspot,
                "extremes":      extremes,
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

    # ── Sector exposure ────────────────────────────────────────────────────
    # Under margin/leverage the LONG sectors sum to >100% of NAV (e.g. 118%
    # when a -17.8% USD margin-debt funds the longs).  Displaying raw NAV %
    # made the legend sum to 118%, which reads as a bug to the user.
    #
    # Fix: present every sector as a share of the INVESTED (long) book, so
    # the breakdown sums to exactly 100%.  This is the standard pie-chart
    # convention and matches the wedge angles (which were already
    # normalised).  `nav_pct` preserves the raw NAV share for any table
    # that needs the leveraged truth; the leverage itself is still shown
    # separately in the "Gross Exposure / Leverage" badge above the pie.
    sector_exposure = results.get("sector_exposure") or {}
    long_only = [(s, float(w)) for s, w in sector_exposure.items() if float(w) > 0]
    _long_sum = sum(w for _, w in long_only)

    if _long_sum > 1e-9:
        # share of long book (sums to 100%)
        sectors = [
            {"name":       s,
             "weight_pct": round(w / _long_sum * 100, 2),
             "weight_str": f"{w / _long_sum * 100:.0f}%",
             "nav_pct":    round(w * 100, 2)}
            for s, w in long_only
        ]
        pie_chart_data = [
            {
                "name":       s,
                "weight_pct": round(w / _long_sum * 100, 2),   # angle share
                "weight_str": f"{w / _long_sum * 100:.0f}%",   # share of long book
                "nav_pct":    round(w * 100, 2),               # raw NAV %
            }
            for s, w in long_only
        ]
    else:
        sectors = []
        pie_chart_data = []

    # ── Concentration metrics (sectors and individual assets) ──────────────
    # Computed on the SAME normalised (share-of-long-book) basis as the
    # display so the warning % matches the legend % exactly.
    sector_pairs = [(s, float(w) / _long_sum) for s, w in long_only] if _long_sum > 1e-9 else []
    sector_concentration = _build_concentration(sector_pairs)
    sector_warnings      = _build_sector_warnings(sector_pairs, cap_pct=SECTOR_CAP_PCT)

    # SSOT for combined-sector headlines (e.g. Tech+Semiconductors).  Compute the
    # super-group rollup ONCE here and surface it as an explicit, labelled
    # warning so the structured panel quotes the same number the AI prose is
    # told to use — killing the BASE 55% vs DEEP 80.8% divergence.
    sector_groups  = build_sector_groups(sector_pairs)
    sector_complex = sector_groups[0] if sector_groups else None
    sector_warnings = sector_warnings + _sector_group_warnings(sector_groups, cap_pct=SECTOR_CAP_PCT)

    asset_pairs = [(a["ticker"], a["weight_pct_num"] / 100.0) for a in assets
                   if a.get("weight_pct_num") is not None]
    asset_concentration = _build_concentration(asset_pairs)

    # ── Risk waterfall: standalone vol per asset vs diversified portfolio ──
    risk_waterfall = _build_risk_waterfall(
        risk_matrix    = results.get("risk_matrix"),
        perf_df        = perf_df,
        total_val      = total_val,
        total_vol_ann  = vol_raw,
    )

    # ── Return series coverage (sparse-history exclusions) ───────────────────
    return_series_coverage = results.get("return_series_coverage") or {}

    # ── Multi-period performance (1м / 3м / 6м / 12м / YTD) ────────────────
    period_returns_table = _adapt_period_returns(results.get("period_returns_table"))

    # ── Stress scenarios (parametric factor shocks, 7-row default catalog) ─
    stress_scenarios = results.get("stress_scenarios") or []

    # ── Expected effect — before/after simulation under BL target weights ──
    expected_effect = _build_expected_effect(results.get("expected_effect"))

    # ── Macro drivers from FRED (5-series pack for DEEP P5) ────────────────
    # Engine emits a flat {key: {value, as_of, status, ...}} dict; the DEEP
    # "Сигналы-драйверы" panel needs {series: [...], as_of}.  Adapt here so
    # the renderer stays schema-agnostic.
    macro_drivers = _build_macro_drivers_panel(results.get("macro_drivers"))

    # ── CoVe data-lineage (runtime status per source) ──────────────────────
    # Build a uniform list[dict] over all data sources consumed by this
    # run — used by the DEEP P4 "CoVe" panel.  Pure-function, never throws.
    try:
        from finance.data_lineage import build_lineage as _build_lineage
        cove_lineage = _build_lineage(results, ai_summary)
    except Exception:
        cove_lineage = []

    # ── Macro regime (Cover) ───────────────────────────────────────────────
    regime_block = None
    regime = results.get("regime")
    if regime:
        signals = regime.get("signals") or {}
        # Build a compact human-readable explainer string from raw signals.
        explainers: list[str] = []
        sv = signals.get("spy_vs_ief_60d")
        if sv is not None:
            sign = "обгоняет" if sv >= 0 else "отстаёт от"
            explainers.append(f"SPY {sign} IEF на {sv*100:+.1f}% за 60 дней (рост vs облигации)")
        xv = signals.get("xly_vs_xlp_60d")
        if xv is not None:
            who = "Discretionary > Staples" if xv >= 0 else "Staples > Discretionary"
            explainers.append(f"{who}: {xv*100:+.1f}% за 60д (цикличные {('on' if xv>=0 else 'off')})")
        iv = signals.get("iwm_vs_spy_60d")
        if iv is not None:
            who = "small-caps лидируют" if iv >= 0 else "large-caps лидируют"
            explainers.append(f"{who}: IWM−SPY {iv*100:+.1f}% за 60д")
        ev = signals.get("eem_60d")
        if ev is not None:
            tone = "EM risk-on" if ev >= 0 else "EM risk-off"
            explainers.append(f"EEM 60д {ev*100:+.1f}% ({tone})")
        # Sprint-5 (R1): the DEEP regime quadrant used to be a HARDCODED SVG dot
        # at (177,110) — it always sat in "Expansion" regardless of the computed
        # regime, visually contradicting the label.  Compute the REAL dot
        # position from growth/cycle here and inject cx/cy into the template.
        g_val = float(regime["growth_score"])
        c_val = float(regime["cycle_score"])
        dot_cx, dot_cy = _regime_dot_coords(g_val, c_val)
        regime_block = {
            "label":      regime["regime"],
            "confidence": int(round(regime["confidence"] * 100)),
            "growth":     g_val,
            "cycle":      c_val,
            "explainers": explainers,
            # SVG dot position (viewBox 0..310; centre at 155,155; 625 px/unit).
            "dot_cx":     dot_cx,
            "dot_cy":     dot_cy,
            "dot_label":  f"Growth {g_val:+.2f} · Cycle {c_val:+.2f}",
        }

    # ── Data quality (loaded factors / benchmarks / SEC) ──────────────────
    history = results.get("history_result")
    available_cols = []
    if history is not None and getattr(history, "data", None) is not None:
        available_cols = list(getattr(history, "data").columns)
    factors_loaded = sum(1 for f in _FACTOR_ETFS if f in available_cols)
    factors_total  = len(_FACTOR_ETFS)
    bm_loaded = len(results.get("benchmark_comparison") or {})
    sec_skipped: list[str] = []
    if perf_df is not None and not perf_df.empty and "Fundamental_Sector" in perf_df.columns:
        sec_skipped = perf_df.loc[
            perf_df["Fundamental_Sector"].astype(str).isin(["default", "EM_Proxy"]),
            "Ticker",
        ].astype(str).tolist()
    data_quality = {
        "factors_loaded":     factors_loaded,
        "factors_total":      factors_total,
        "factors_complete":   factors_loaded == factors_total,
        "benchmarks_loaded":  bm_loaded,
        "sec_skipped":        sec_skipped,
        "data_source_label":  "Daily CLOSE из Tradernet (730d). ATR — OHLC, fallback |ΔClose|.",
    }

    # KPI extreme flags — drive red borders + tooltips in template.
    kpi_extremes = {
        "cvar":         _flag(cvar_raw,    kind="cvar"),
        "sharpe":       _flag(sharpe_raw,  kind="sharpe"),
        "vol":          _flag(vol_raw,     kind="vol"),
        "max_drawdown": _flag(mdd_raw,     kind="mdd"),
    }

    # ── Dollar impact fields ───────────────────────────────────────────────
    cvar_dollar = f"${abs(cvar_raw * total_val):,.0f}" if total_val > 0 else "—"
    mdd_dollar  = f"${abs(mdd_raw  * total_val):,.0f}" if total_val > 0 else "—"
    var_dollar  = f"${abs(var_raw  * total_val):,.0f}" if total_val > 0 else "—"

    # ── Month-over-month risk delta ────────────────────────────────────────
    prev_risk_score: Optional[int] = None
    risk_score_delta: Optional[int] = None
    if prev_snapshot:
        prev_risk_score  = prev_snapshot.get("risk_score")
        if prev_risk_score is not None:
            risk_score_delta = composite - int(prev_risk_score)

    # ── Priority action (first hotspot or first action plan entry) ─────────
    priority_action: Optional[str] = None
    action_plan = results.get("action_plan") or []
    if hotspots:
        priority_action = hotspots[0]
    elif action_plan:
        ap = action_plan[0]
        priority_action = (f"{ap.get('action','Hold')} {ap.get('ticker','')}: "
                           f"{ap.get('reason','')}"[:120])

    # ── AI ideas: reshape stock-picks into the template idea-card schema ───
    ai_ideas    = _build_ai_ideas((ai_summary or {}).get("stock_picks") or {}, tier=tier)
    ideas_count = sum(len(v) for v in ai_ideas.values())

    # ── Fundamental layer — SEC-derived columns from the perf table ────────
    # Built for BOTH tiers: the base report's holdings-detail panel also
    # reads data.fundamental_layer — gating it to deep-only left every base
    # asset showing "н/д".  analyze_all populates SEC_* columns tier-agnostic.
    fundamental_rows: list[dict] = []
    if perf_df is not None and not perf_df.empty:
        for _, row in perf_df.iterrows():
            if str(row.get("Fundamental_Sector") or "default") in ("default", "EM_Proxy"):
                continue
            fundamental_rows.append({
                "ticker":      str(row.get("Ticker", "—")),
                "roe":         _fmt_pct(row.get("SEC_ROE")),
                "op_m":        _fmt_pct(row.get("SEC_Op_Margin")),
                "dta":         _fmt_pct(row.get("SEC_Debt_to_Assets")),
                "rev_g":       _fmt_pct(row.get("SEC_Revenue_Growth_YoY")),
                "fcf_m":       _fmt_pct(row.get("SEC_FCF_Margin")),
                "altman_z":    _fmt_num(row.get("SEC_Altman_Z"), digits=2),
                "altman_zone": row.get("SEC_Altman_Zone") or "—",
                "piotroski":   _fmt_int(row.get("SEC_Piotroski_F")),
                "int_cov":     _fmt_num(row.get("SEC_Interest_Coverage"), digits=1),
            })

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
        "risk_free_rate":    rfr_str,
        "risk_pct":          composite,
        "risk_label":        _risk_score_label(composite),
        # H1: mandate label next to the gauge — the gauge is mandate-
        # calibrated (different CVaR base per profile), and without showing
        # which mandate produced the number the score looks arbitrary.
        "risk_mandate":      results.get("risk_mandate", "MODERATE"),
        "risk_mandate_label": _risk_mandate_label(
                              results.get("risk_mandate", "MODERATE")),
        "kpi_extremes":      kpi_extremes,
        # Dollar impact
        "total_value_usd":   f"${total_val:,.0f}",
        "cvar_dollar":       cvar_dollar,
        "mdd_dollar":        mdd_dollar,
        "var_dollar":        var_dollar,
        # Month-over-month delta
        "prev_risk_score":   prev_risk_score,
        "risk_score_delta":  risk_score_delta,
        # Priority action (top of page 2)
        "priority_action":   priority_action,
        # Aggregate P/L since entry
        "pnl_total_abs":     _format_pnl_abs(total_pnl),
        "pnl_total_pct":     _format_pnl_pct(total_return_pct),
        "pnl_total_color":   "pos" if total_return_pct >= 0 else "neg",
        "pnl_best":          best,
        "pnl_worst":         worst,
        # Engine-emitted portfolio_metrics (H2 reporting currency, RFR,
        # H4 stress disclaimer, etc.).  Must be propagated so the template
        # can render `data.portfolio_metrics.stress_test_disclaimer` and
        # similar QC fields without hand-mapping each one.
        "portfolio_metrics": results.get("portfolio_metrics") or {},
        # Holdings + risk
        "assets":            assets,
        "holdings_count":    len(assets),
        "hotspots":          hotspots,
        "sectors":           sectors,
        # Pie chart wedge data — long-only, normalised to 1.0 for display
        # ONLY.  Other panels keep using `sectors` (real % of NAV).
        "pie_chart_data":    pie_chart_data,
        # Leverage / gross-exposure metrics from the engine — drives the
        # "Gross Exposure / Leverage" badge above the sector pie when the
        # book carries margin debt (USD weight < -0.001).
        "leverage_metrics":  results.get("leverage_metrics") or {
            "is_leveraged": False, "gross_exposure": 1.0,
            "leverage_ratio": 1.0, "long_weight": 1.0,
            "net_exposure": 1.0,   "cash_weight": 0.0,
        },
        # SEC fundamentals — both tiers render a holdings-detail panel.
        "fundamental_layer": fundamental_rows,
        # Concentration & waterfall (new — data ready for next-gen template)
        "sector_concentration": sector_concentration,
        "sector_warnings":      sector_warnings,
        "sector_groups":        sector_groups,    # SSOT super-group rollups
        "sector_complex":       sector_complex,   # top combined group (e.g. Tech-комплекс)
        "asset_concentration":  asset_concentration,
        "risk_waterfall":       risk_waterfall,
        # Multi-period performance table {bm_name: {periods: [...], window_*}}
        "period_returns_table": period_returns_table,
        # Stress scenarios (parametric factor shocks — list of dicts)
        "stress_scenarios":     stress_scenarios,
        # Expected effect — DEEP P4 8-card before/after panel
        "expected_effect":      expected_effect,
        # Macro drivers — FRED 5-series pack for DEEP P5 regime page
        "macro_drivers":        macro_drivers,
        # CoVe data-lineage (runtime status per source — DEEP P4 panel)
        "cove_lineage":         cove_lineage,
        # Regime
        "regime":            regime_block,
        "regime_rag_confirm": regime_rag_confirm or [],
        # Data quality
        "data_quality":      data_quality,
        # Return series coverage (from sparse-history exclusions)
        "return_series_coverage": return_series_coverage,
        # Integrity checks panel — compact ✓/⚠/— list for both BASE and DEEP
        "integrity_checks":  _build_integrity_checks(
            results, ai_summary or {}, data_quality, return_series_coverage
        ),
        # AI Narrative — placeholder unless caller passed one in
        "ai_verdict":            (ai_summary or {}).get("verdict", ""),
        "ai_plain_summary":      (ai_summary or {}).get("plain_summary", ""),
        "ai_bullets":            (ai_summary or {}).get("bullets", []),
        # Raw scenario buckets — kept for the legacy v2 templates.
        "ai_stock_picks":        (ai_summary or {}).get("stock_picks", {}),
        # Idea-card schema consumed by the v3 templates.
        "ai_ideas":              ai_ideas,
        "ideas_count":           ideas_count,
        "used_rag":              bool((ai_summary or {}).get("used_rag")),
        "ai_model_used":         _model_display_name((ai_summary or {}).get("model_used", "")),
        "ai_action_impact":      (ai_summary or {}).get("ai_action_impact", ""),
        # Per-KPI AI notes (CVaR / Sharpe / MaxDD cards) — plain-language.
        "ai_cvar_note":          (ai_summary or {}).get("ai_cvar_note", ""),
        "ai_sharpe_note":        (ai_summary or {}).get("ai_sharpe_note", ""),
        "ai_mdd_note":           (ai_summary or {}).get("ai_mdd_note", ""),
        # Per-section AI commentary (populated by Claude, empty if fallback —
        # the templates hide each block when its comment is empty).
        "ai_risk_comment":           (ai_summary or {}).get("ai_risk_comment", ""),
        "ai_benchmark_comment":      (ai_summary or {}).get("ai_benchmark_comment", ""),
        "ai_performance_comment":    (ai_summary or {}).get("ai_performance_comment", ""),
        "ai_regime_comment":         (ai_summary or {}).get("ai_regime_comment", ""),
        "ai_holdings_comment":       (ai_summary or {}).get("ai_holdings_comment", ""),
        "ai_sector_comment":         (ai_summary or {}).get("ai_sector_comment", ""),
        "ai_factor_comment":         (ai_summary or {}).get("ai_factor_comment", ""),
        "ai_4pillar_comment":        (ai_summary or {}).get("ai_4pillar_comment", ""),
        "ai_stress_comment":         (ai_summary or {}).get("ai_stress_comment", ""),
        "ai_action_comment":         (ai_summary or {}).get("ai_action_comment", ""),
        "ai_effect_comment":         (ai_summary or {}).get("ai_effect_comment", ""),
        # Sprint-5 margin/leverage AI-trigger — non-empty only when the book is
        # margin-funded; templates render it as a red Margin-Call warning card.
        "ai_leverage_warning":       (ai_summary or {}).get("ai_leverage_warning", ""),
        # Sprint-5 Task 4 — mandate compliance panel (target profile + actual
        # allocation vs limits).  None when no approved profile is supplied.
        "mandate_compliance":        _build_mandate_compliance(
                                        perf_df, total_val, user_profile),
        # Structured cross-check of the engine's regime label — DEEP only.
        "regime_confirmation":       (ai_summary or {}).get("regime_confirmation",
                                       {"stance": "", "summary": "", "signals": []}),
        # Tier metadata
        "tier":              tier,
    }

    # ── Benchmark scenarios (BOTH tiers — BASE "Рост против рынка" needs bm_return) ─
    # bm_return = annualised benchmark return for the "Рынок · S&P 500" stat card.
    # Previously built only in TIER_DEEP; now common so BASE template works.
    def _build_scenarios(bm_data: dict, filter_name: Optional[str]) -> list:
        rows = []
        for bm_name, bm in bm_data.items():
            if filter_name and bm_name != filter_name:
                continue
            excess_ann = bm.get("Excess_Return_Ann")
            if excess_ann is None:
                excess_ann = _safe_float(bm.get("Excess_Return"), 0.0)
            else:
                excess_ann = _safe_float(excess_ann, 0.0)
            ir = _safe_float(bm.get("Information_Ratio"), 0.0)
            te = _safe_float(bm.get("Tracking_Error"), 0.0)
            bm_ann  = _safe_float(bm.get("Benchmark_Ann_Return"), float("nan"))
            port_ann = _safe_float(bm.get("Port_Ann_Return"), float("nan"))
            rows.append({
                "name":       bm_name,
                "excess":     _format_pnl_pct(excess_ann),
                "te":         f"{te*100:.1f}%" if te else "—",
                "ir":         f"{ir:.2f}" if ir else "—",
                "beating":    bool(bm.get("Beating_Benchmark")),
                "color":      "pos" if excess_ann >= 0 else "neg",
                # bm_return: used by both BASE and DEEP "Рост против рынка" stat card.
                "bm_return":  f"{bm_ann*100:+.1f}%" if not math.isnan(bm_ann) else "—",
                "port_return": f"{port_ann*100:+.1f}%" if not math.isnan(port_ann) else "",
            })
        return rows

    # Sprint-5 (Task 8 — activate the user's benchmark): the engine always
    # stores the user's chosen benchmark under the canonical name "Профильный
    # бенчмарк" (investment_logic.py:1187).  Prefer that slot so the report's
    # primary benchmark card reflects the ACTUAL selection — including custom
    # tickers (Value/Quality/MSCI World) that are NOT in the standard 5-name
    # comparison set, which the old `_BM_TICKER_TO_NAME` lookup silently
    # dropped (returning an empty scenarios list).
    bm_comparison = results.get("benchmark_comparison") or {}
    if user_bench_ticker and "Профильный бенчмарк" in bm_comparison:
        user_bm_filter: Optional[str] = "Профильный бенчмарк"
    else:
        user_bm_filter = _BM_TICKER_TO_NAME.get(user_bench_ticker) if user_bench_ticker else None
    payload["scenarios"] = _build_scenarios(bm_comparison, user_bm_filter)

    # ── Deep-tier additions ────────────────────────────────────────────────
    if tier == TIER_DEEP:

        # Score breakdown table — pillar contributions per asset.
        # Cash positions are excluded: they carry no fundamentals/technicals
        # and would otherwise show a meaningless zero/negative score row.
        score_breakdown = []
        for ticker, sc in (asset_scores or {}).items():
            if _classify_asset(str(ticker)) == "Ден. средства":
                continue
            credit_applicable = bool(sc.get("credit_applicable", True))
            fund_applicable   = bool(sc.get("fundamentals_applicable", True))
            score_breakdown.append({
                "ticker":       ticker,
                # Asset-class guard: commodities / sovereign bonds have no
                # financial statements → em-dash, not a regime-tilt number
                # masquerading as a fundamental verdict.
                "fundamentals": (f"{sc.get('fundamentals', 0):+.1f}"
                                 if fund_applicable else "—"),
                "fundamentals_na": not fund_applicable,
                "valuations":   f"{sc.get('valuations', 0):+.1f}",
                "technicals":   f"{sc.get('technicals', 0):+.1f}",
                # Same guard for Credit — no corporate credit risk → em-dash.
                "credit":       (f"{sc.get('credit', 0):+.1f}"
                                 if credit_applicable else "—"),
                "credit_na":    not credit_applicable,
                "total":        f"{sc.get('total', 0):+.1f}",
                "action":       sc.get("action", "Hold"),
                "action_color": _action_color(sc.get("action")),
            })
        payload["score_breakdown"] = score_breakdown

        # Action plan (with current Price / Buy zone / Sell target / Stop)
        action_plan = results.get("action_plan") or []
        # Cache current prices from perf_df so the template's `r.price`
        # column actually shows the latest close (was blanket "—" before).
        price_by_ticker: dict = {}
        if perf_df is not None and not perf_df.empty and "Current_Price" in perf_df.columns:
            for _, _row in perf_df.iterrows():
                _tk = str(_row.get("Ticker") or "").strip()
                _px = _row.get("Current_Price")
                if _tk and _px is not None:
                    try:
                        price_by_ticker[_tk] = float(_px)
                    except (TypeError, ValueError):
                        pass

        plan_rows = []
        for r in action_plan:
            if _classify_asset(str(r.get("ticker") or "")) == "Ден. средства":
                continue   # cash carries no Buy/Sell/Stop levels
            buy_zone   = r.get("buy_zone")
            sell_zone  = r.get("sell_zone")
            tgt        = r.get("take_target")
            stop       = r.get("stop_loss")
            px         = price_by_ticker.get(str(r.get("ticker") or ""))
            plan_rows.append({
                "ticker":      r.get("ticker"),
                "action":      r.get("action"),
                "action_color":_action_color(r.get("action")),
                "delta_w_pp":  f"{r.get('delta_w_pp', 0):+.1f}",
                "qty_delta":   r.get("qty_delta") if r.get("qty_delta") is not None else "—",
                # `price` is the field the v3 template renders; payload now
                # supplies it from perf_df.Current_Price so the column is
                # no longer a row of dashes.
                "price":       f"{px:.2f}" if px is not None else "—",
                "buy_zone":    f"{buy_zone[0]:.2f} – {buy_zone[1]:.2f}" if buy_zone else "—",
                "sell_zone":   f"{sell_zone[0]:.2f} – {sell_zone[1]:.2f}" if sell_zone else "—",
                "take_target": f"{tgt:.2f}"  if tgt  is not None else "—",
                # `sell_target` column: the engine sets a profit `take_target`
                # only for BUY actions; for Trim/Sell it produces a `sell_zone`
                # (the price band to sell into).  Show whichever exists so the
                # column is meaningful for every row instead of "—" on sells.
                "sell_target": (f"{tgt:.2f}" if tgt is not None else
                                (f"{sell_zone[0]:.2f} – {sell_zone[1]:.2f}"
                                 if sell_zone else "—")),
                "stop_loss":   f"{stop:.2f}" if stop is not None else "—",
                "reason":      r.get("reason", ""),
            })
        payload["action_plan"] = plan_rows

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

        # AI deep narrative (full bullets list + action-plan text + impact)
        if ai_summary:
            payload["ai_action_text"] = ai_summary.get("action_plan_text", "")
            payload["ai_action_impact"] = ai_summary.get("ai_action_impact", "")

    return payload


# ── Benchmark display — label formatting ─────────────────────────────────────

def te_label(te_val: Optional[float]) -> str:
    """Format Tracking Error with an explicit label to distinguish from Volatility."""
    if te_val is None:
        return "—"
    return f"{te_val * 100:.1f}%"


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


# ── Concentration metrics ─────────────────────────────────────────────────────
# Standard US DOJ Herfindahl thresholds (originally for market structure, but
# the bands have become the de-facto reference for portfolio concentration as
# well — see e.g. Markowitz / Sharpe textbooks).
HHI_BAND_THRESHOLDS = (1500, 2500)   # < 1500: diversified; 1500–2500: moderate; ≥ 2500: concentrated
SECTOR_CAP_PCT      = 40.0           # warn when single sector > this share of portfolio

# ── Sector super-groups (SINGLE SOURCE OF TRUTH for combined-sector headlines)
# The engine intentionally keeps Technology and Semiconductors as DISTINCT
# sectors (they carry a separate SOXX factor).  But the headline "tech
# concentration" must be ONE authoritative number, not something the LLM
# re-aggregates ad-hoc — that mismatch produced 55% in BASE (top granular
# sector) vs 80.8% in DEEP (LLM-lumped Tech+Semi) for the same book.
#
# This map is imported by ai_narrative so the AI prose and the structured
# panel quote the exact same combined figure.  Keep it here as the canonical
# presentation grouping.
SECTOR_SUPERGROUPS: dict[str, tuple[str, ...]] = {
    "Tech-комплекс (Technology+Semiconductors)": ("Technology", "Semiconductors"),
}


def build_sector_groups(weight_pairs: list[tuple[str, float]]) -> list[dict]:
    """Aggregate granular sectors into the canonical super-groups (SSOT).

    `weight_pairs` are (sector, weight_decimal) on the share-of-long-book basis.
    Returns [{name, weight_pct, members}] for every super-group with ≥2 member
    sectors present (a single-member group adds nothing over the granular panel
    and is skipped).  Sorted by combined weight descending.
    """
    wmap = {s: float(w) for s, w in weight_pairs}
    groups: list[dict] = []
    for name, members in SECTOR_SUPERGROUPS.items():
        present = [m for m in members if m in wmap]
        if len(present) < 2:
            continue
        total = sum(wmap[m] for m in present)
        groups.append({
            "name":       name,
            "weight_pct": round(total * 100, 1),
            "members":    {m: round(wmap[m] * 100, 1) for m in present},
        })
    groups.sort(key=lambda g: g["weight_pct"], reverse=True)
    return groups


def _sector_group_warnings(groups: list[dict], cap_pct: float = SECTOR_CAP_PCT) -> list[dict]:
    """Warnings (same shape as `_build_sector_warnings`) for super-groups over cap.

    Surfacing the combined figure here means BOTH templates render it via
    `data.sector_warnings` → the 80.8% Tech-комплекс now appears, labelled, in
    BASE and DEEP identically.
    """
    out: list[dict] = []
    for g in groups:
        w_pct = float(g["weight_pct"])
        if w_pct > cap_pct:
            out.append({
                "sector":     g["name"],
                "weight_pct": round(w_pct, 1),
                "cap_pct":    cap_pct,
                "overage_pp": round(w_pct - cap_pct, 1),
                "is_group":   True,
                "text":       (f"{g['name']}: {w_pct:.0f}% портфеля — превышен "
                               f"мягкий лимит {cap_pct:.0f}% на {w_pct - cap_pct:.0f} п.п."),
            })
    return out


def _hhi_band(points: float) -> str:
    """Map a 0..10000 HHI to a human label (DOJ-style bands)."""
    lo, hi = HHI_BAND_THRESHOLDS
    if points < lo: return "diversified"
    if points < hi: return "moderate"
    return "concentrated"


def _build_concentration(weight_pairs: list[tuple[str, float]]) -> Optional[dict]:
    """
    Build a concentration dict from a list of (label, weight_decimal) pairs.

    weight_decimal must be in [0, 1] (i.e. 0.42 == 42%).  The function does NOT
    normalise: when weights sum to less than 1 (e.g. portfolio has cash that's
    not represented in the sector breakdown) the HHI is computed on the dict
    as-is — that's the correct Herfindahl interpretation of "concentration of
    represented exposure" and matches how prime brokers report it.

    Returns None when the input is empty.
    """
    if not weight_pairs:
        return None
    items = sorted(weight_pairs, key=lambda kv: kv[1], reverse=True)
    hhi_decimal = float(sum(w * w for _, w in items))
    hhi_points  = hhi_decimal * 10000.0
    top1 = items[0]
    top3_sum = float(sum(w for _, w in items[:3]))
    top5_sum = float(sum(w for _, w in items[:5]))
    return {
        "hhi_decimal":   round(hhi_decimal, 4),
        "hhi_points":    int(round(hhi_points)),
        "hhi_band":      _hhi_band(hhi_points),
        "top1_label":    top1[0],
        "top1_pct":      round(top1[1] * 100, 1),
        "top3_pct":      round(top3_sum * 100, 1),
        "top5_pct":      round(top5_sum * 100, 1),
        "top3_labels":   [lbl for lbl, _ in items[:3]],
        "items_count":   len(items),
        "total_weight_pct": round(sum(w for _, w in items) * 100, 1),
    }


def _build_sector_warnings(weight_pairs: list[tuple[str, float]],
                            cap_pct: float = SECTOR_CAP_PCT) -> list[dict]:
    """
    List of {sector, weight_pct, cap_pct, overage_pp} for sectors exceeding the
    soft cap.  Empty list when nothing trips the threshold.
    """
    warnings: list[dict] = []
    for sector, w in weight_pairs:
        w_pct = float(w) * 100.0
        if w_pct > cap_pct:
            warnings.append({
                "sector":     sector,
                "weight_pct": round(w_pct, 1),
                "cap_pct":    cap_pct,
                "overage_pp": round(w_pct - cap_pct, 1),
                # Pre-formatted human string — the template renders this
                # directly (rendering the dict itself leaked a raw repr).
                "text":       (f"{sector}: {w_pct:.0f}% портфеля — превышен "
                               f"мягкий лимит {cap_pct:.0f}% на "
                               f"{w_pct - cap_pct:.0f} п.п."),
            })
    # Sort by overage descending so the worst is first.
    warnings.sort(key=lambda d: d["overage_pp"], reverse=True)
    return warnings


# ── Risk waterfall (standalone vs diversified) ───────────────────────────────

def _build_risk_waterfall(risk_matrix: Optional[pd.DataFrame],
                           perf_df: Optional[pd.DataFrame],
                           total_val: float,
                           total_vol_ann: float) -> Optional[dict]:
    """
    Decompose annualised portfolio volatility into per-asset *standalone*
    contributions and the *diversification benefit*.

    Math (all in annualised decimal units, e.g. 0.142 == 14.2%):
        σ_i              = sqrt(Σ_ii)              # per-asset annual vol (diagonal)
        standalone_i     = w_i * σ_i               # asset's "as if alone" contribution
        Σ standalone     = Σ_i (w_i * σ_i)         # undiversified portfolio vol
        diversified vol  = sqrt(w' Σ w)            # actual portfolio vol (= total_vol_ann)
        diversification_benefit = Σ standalone − diversified vol  ≥ 0

    Sanity guaranteed by Minkowski:
        sqrt(w' Σ w)  ≤  Σ |w_i| * sqrt(Σ_ii)
    so the benefit is non-negative provided weights are non-negative
    (which they are — long-only portfolio).

    Returns None when inputs are missing or the cov matrix is empty.
    """
    if (risk_matrix is None or perf_df is None or
        getattr(risk_matrix, "empty", True) or getattr(perf_df, "empty", True) or
        total_val <= 0):
        return None

    # Build {ticker: weight_decimal} from perf_df.
    weights: dict[str, float] = {}
    for _, row in perf_df.iterrows():
        t = str(row.get("Ticker", "")).strip()
        cv = _safe_float(row.get("Current_Value"), 0.0)
        if t:
            weights[t] = cv / total_val if total_val > 0 else 0.0

    contributions: list[dict] = []
    sum_standalone = 0.0
    for ticker in risk_matrix.index:
        diag = _safe_float(risk_matrix.loc[ticker, ticker], 0.0)
        if diag <= 0 or math.isnan(diag):
            continue
        sigma_i = math.sqrt(diag)              # annualised vol, decimal
        w_i     = weights.get(str(ticker), 0.0)
        contrib = w_i * sigma_i                # decimal pp of total
        sum_standalone += contrib
        contributions.append({
            "ticker":         str(ticker),
            "weight_pct":     round(w_i * 100, 2),
            "standalone_vol_pct": round(sigma_i * 100, 2),     # asset's own annual vol
            "standalone_pp":  round(contrib * 100, 2),         # contribution as if uncorrelated
        })

    if not contributions:
        return None

    # Sort contributions descending for waterfall rendering.
    contributions.sort(key=lambda d: d["standalone_pp"], reverse=True)

    sum_standalone_pp = round(sum_standalone * 100, 2)
    total_vol_pp      = round(total_vol_ann * 100, 2)
    diversification_pp = round(sum_standalone_pp - total_vol_pp, 2)

    # Per-asset share of standalone total (sums to 100%).
    if sum_standalone > 0:
        for c in contributions:
            c["standalone_share_pct"] = round(c["standalone_pp"] / sum_standalone_pp * 100, 1)
    else:
        for c in contributions:
            c["standalone_share_pct"] = 0.0

    # ── H6: plot-ready geometry, computed HERE (was math-in-Jinja) ───────────
    # The template previously merged contributions[4:] into a "+N др." bar,
    # ran cumulative sums via a Jinja `namespace`, and derived the y-axis
    # domain — all presentation-layer math.  Compute it once, here, and hand
    # the template ready arrays + scalars; the template now only maps a value
    # to a pixel via the fixed plot height (legitimate view scaling).
    TOP_N = 4
    if len(contributions) > TOP_N:
        rest = contributions[TOP_N:]
        bars = list(contributions[:TOP_N])
        bars.append({
            "ticker":             f"+{len(rest)} др.",
            "standalone_pp":      round(sum(c["standalone_pp"] for c in rest), 2),
            "standalone_vol_pct": None,
            "weight_pct":         round(sum(c["weight_pct"] for c in rest), 2),
            "is_aggregate":       True,
        })
    else:
        bars = list(contributions)

    # Cumulative offset per bar (for the stacked waterfall).
    cum = 0.0
    for b in bars:
        b["cum_start_pp"] = round(cum, 3)
        cum += b.get("standalone_pp", 0.0) or 0.0
        b["cum_end_pp"]   = round(cum, 3)

    # Dynamic y-axis domain: max(Σstandalone, total, 20%) rounded up to 5%.
    y_top   = max(sum_standalone_pp, total_vol_pp, 20.0)
    y_max_pp = math.ceil(y_top / 5.0) * 5.0 or 20.0
    y_ticks = [round(y_max_pp * f, 1) for f in (0.0, 0.25, 0.50, 0.75, 1.0)]

    return {
        "contributions":     contributions,
        "bars":              bars,          # NEW: pre-aggregated + cumulative
        "y_max_pp":          y_max_pp,      # NEW: axis domain (no Jinja math)
        "y_ticks":           y_ticks,       # NEW: axis tick labels
        "sum_standalone_pp": sum_standalone_pp,
        "total_vol_pp":      total_vol_pp,
        "diversification_pp": max(0.0, diversification_pp),   # clip tiny negative noise to 0
        "diversification_ratio": round(diversification_pp / sum_standalone_pp, 3)
                                  if sum_standalone_pp > 0 else 0.0,
        "method": ("σᵢ = √Σᵢᵢ (annualised) · standalone = wᵢ·σᵢ · "
                   "diversified = √(w'Σw) · benefit = Σstandalone − diversified"),
    }


def _build_integrity_checks(results: dict,
                             ai_summary: dict,
                             data_quality: dict,
                             return_series_coverage: dict) -> list[dict]:
    """
    Compact integrity panel — list[{status, label, detail}].

    Status symbols:
      ✓  data loaded, math computed, within freshness window
      ⚠  partial / degraded (sparse drop, partial factor load, stale cache)
      —  source not used / not applicable for this run
    """
    checks: list[dict] = []

    # 1. Market price data
    history = results.get("history_result")
    hist_data = getattr(history, "data", None) if history is not None else None
    if hist_data is not None and len(hist_data) > 0:
        n_days = len(hist_data)
        checks.append({"status": "✓", "label": "Рыночные данные",
                        "detail": f"Tradernet · {n_days} дней · daily CLOSE"})
    else:
        checks.append({"status": "⚠", "label": "Рыночные данные",
                        "detail": "нет ценовой истории"})

    # 2. Return series coverage (sparse-history exclusions)
    dropped = return_series_coverage.get("dropped", [])
    cov_w   = return_series_coverage.get("covered_weight", 1.0)
    n_ser   = return_series_coverage.get("n_days", 0)
    if dropped:
        checks.append({"status": "⚠", "label": "Серия доходностей",
                        "detail": (f"{cov_w*100:.0f}% портфеля покрыто · "
                                   f"исключены: {', '.join(dropped)}")})
    elif n_ser > 0:
        checks.append({"status": "✓", "label": "Серия доходностей",
                        "detail": f"100% покрытие · {n_ser} торговых дней"})
    else:
        checks.append({"status": "⚠", "label": "Серия доходностей",
                        "detail": "ряд не построен"})

    # 3. Factor model coverage
    f_loaded = data_quality.get("factors_loaded", 0)
    f_total  = data_quality.get("factors_total", 1)
    f_pct    = round(f_loaded / f_total * 100) if f_total else 0
    checks.append({
        "status": "✓" if f_pct >= 80 else "⚠",
        "label":  "Факторная модель",
        "detail": f"Ridge β · {f_loaded}/{f_total} факторов · {f_pct}% покрытие",
    })

    # 4. Math: Euler decomposition presence
    perf = results.get("performance_table")
    has_euler = (perf is not None and not getattr(perf, "empty", True) and
                 "Euler_Risk_Contribution_Pct" in perf.columns)
    checks.append({
        "status": "✓" if has_euler else "⚠",
        "label":  "Euler-декомпозиция",
        "detail": "TRC = w·MCTR/σ_p (Ledoit-Wolf 70/30 blended)" if has_euler
                  else "недоступна",
    })

    # 5. CVaR bootstrap
    metrics = results.get("portfolio_metrics") or {}
    boot    = metrics.get("CVaR_95_Bootstrap") or {}
    has_ci  = boot.get("lo95") is not None
    checks.append({
        "status": "✓" if has_ci else "⚠",
        "label":  "CVaR Bootstrap",
        "detail": ("Politis-Romano · 2000 блоков · 95% CI"
                   if has_ci else "точечная оценка (мало данных)"),
    })

    # 6. RAG (bank research retrieval) — 3-state status from the backend so the
    # panel never shows the misleading binary "не использован" for a portfolio
    # that simply had no matching bank report (or an empty knowledge base).
    rag_status  = (ai_summary or {}).get("rag_status")
    rag_context = (ai_summary or {}).get("rag_context") or ""
    # _fetch_rag_context joins sections with "\n\n"; count non-blank
    # paragraphs as approximate snippet count (header lines included).
    snippets    = (sum(1 for p in rag_context.split("\n\n") if p.strip())
                   if rag_context else 0)
    if rag_status is None:   # back-compat with summaries that predate rag_status
        rag_status = "used" if (ai_summary or {}).get("used_rag") else "no_match"
    if rag_status == "used":
        _rag_icon, _rag_detail = "✓", f"ChromaDB · cosine ≥0.72 · ~{snippets} отрывков"
    elif rag_status == "no_match":
        _rag_icon, _rag_detail = "—", "запрошен — релевантных отчётов не найдено"
    else:  # "unavailable"
        _rag_icon, _rag_detail = "✗", "база недоступна / пуста"
    checks.append({
        "status": _rag_icon,
        "label":  "RAG: банк. отчёты",
        "detail": _rag_detail,
    })

    # 7. AI model attribution
    model = _model_display_name((ai_summary or {}).get("model_used", ""))
    checks.append({
        "status": "✓" if model else "—",
        "label":  "AI-модель",
        "detail": model if model else "не задействована",
    })

    # 8. Reporting currency + RFR provenance (H2 transparency).
    # Surfaces the actual currency the engine used and which env-var (or
    # default) provided the risk-free rate.  Without this the user cannot
    # audit Sharpe / Sortino vs the right benchmark.
    metrics_h2 = results.get("portfolio_metrics") or {}
    rc       = metrics_h2.get("reporting_currency")
    rfr_ann  = metrics_h2.get("risk_free_rate_annual")
    rfr_src  = metrics_h2.get("risk_free_rate_source")
    if rc and rfr_ann is not None:
        # Label already says "Валюта отчёта" — don't repeat "Валюта:" in the
        # detail (was rendering "Валюта отчёта: Валюта: USD").
        detail = f"{rc} · RFR {float(rfr_ann)*100:.2f}%"
        if rfr_src:
            detail += f" · {rfr_src}"
        checks.append({"status": "✓", "label": "Валюта отчёта",
                       "detail": detail})

    # 9. FX conversion audit (only when something was actually converted).
    fx_records = metrics_h2.get("fx_conversion") or []
    if fx_records:
        pairs_str = " · ".join(
            f"{r.get('pair', '?')} ({float(r.get('coverage_pct', 0)):.0f}%)"
            for r in fx_records
        )
        any_fallback = any(r.get("fallback_used") for r in fx_records)
        checks.append({
            "status": "⚠" if any_fallback else "✓",
            "label":  "FX-конверсия",
            "detail": pairs_str,
        })

    return checks


__all__ = ["build_payload", "TIER_BASE", "TIER_DEEP"]
