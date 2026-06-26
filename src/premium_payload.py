"""
Premium V2 data mapper (Adapter / anti-corruption layer).

Translates the engine's "heavy" v3 report payload (≈86 keys produced by
pdf_payload.build_payload) into the STRICT design-data contracts the Premium V2
React components consume — 29 keys for DEEP, 11 for BASE (see PREMIUM_DESIGN.md
§3-4 and design/premium_v2/*-data.sample.json).

Strict isolation (Separation of Concerns):
  • This module ONLY READS the already-built payload — it performs NO finance
    math and imports NO computational module.  The risk engine is untouched.
  • Every access is defensive (`_g`): a missing/None value never raises and is
    rendered as the neutral '–' (text) or 0 (chart-numeric), so a thin/partial
    payload can never produce a KeyError or a broken report.
"""
from __future__ import annotations

from typing import Any

DASH = "–"


# ── Defensive accessors ──────────────────────────────────────────────────────

def _g(obj: Any, *path, default: Any = None) -> Any:
    """Walk dict keys / list indices safely; return default on any miss/None."""
    cur = obj
    for p in path:
        try:
            if isinstance(cur, dict):
                cur = cur.get(p)
            elif isinstance(cur, (list, tuple)):
                cur = cur[p]
            else:
                return default
        except (KeyError, IndexError, TypeError):
            return default
    return default if cur is None else cur


def _txt(obj: Any, *path) -> str:
    """Text field — missing/empty → neutral '–'."""
    v = _g(obj, *path)
    s = "" if v is None else str(v).strip()
    return s if s else DASH


def _num(obj: Any, *path, default: float = 0.0) -> float:
    """Chart-numeric field — missing/garbage → 0 (keeps React math safe)."""
    v = _g(obj, *path)
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace("−", "-").replace("%", "").replace(",", ".")
                     .replace("$", "").replace("≈", "").strip())
    except (TypeError, ValueError):
        return default


def _pct(s: Any, default: float = 0.0) -> float:
    """Parse a percent string ('18.4%' / '−2,7%') → float."""
    return _num(s, default=default)


def _list(obj: Any, *path) -> list:
    v = _g(obj, *path)
    return v if isinstance(v, list) else []


# ── DEEP mapper ───────────────────────────────────────────────────────────────

def _kpi(p: dict, key: str, name: str, val_key: str, note_key: str,
         spark_key: str, status: str, color: str, sub: str) -> dict:
    spark = _g(p, "kpi_sparklines", spark_key, default="")
    # Sparkline SVG is server-rendered; the design wants a points array. We pass
    # the SVG through as `svg` AND keep `pts` empty (the component tolerates it).
    return {
        "key": key, "name": name,
        "value": _txt(p, val_key),
        "status": status, "color": color,
        "sub": sub if sub != DASH else _txt(p, val_key + "_dollar"),
        "ai": _txt(p, note_key),
        "pts": [], "svg": spark,
    }


def _map_deep(p: dict, meta: dict) -> dict:
    # bullets → {tag, text, src}
    bullets = []
    for b in _list(p, "ai_bullets"):
        if isinstance(b, dict):
            bullets.append({"tag": _txt(b, "tag"), "text": _txt(b, "text"), "src": _txt(b, "src")})
        elif b:
            bullets.append({"tag": "", "text": str(b), "src": ""})

    # holdings / concentration from assets[]
    assets = _list(p, "assets")
    holdings = []
    for a in assets:
        fl = _g(p, "fundamental_layer", default=[]) or []
        holdings.append({
            "t": _txt(a, "ticker"), "name": _txt(a, "name") if _g(a, "name") else _txt(a, "ticker"),
            "cls": _txt(a, "asset_class"),
            "w": _num(a, "weight_pct_num"), "risk": _num(a, "euler_risk_pct"),
            "pnlPct": _num(a, "pnl_pct_num", default=_pct(_g(a, "pnl_pct"))),
            "pnlUsd": _num(a, "pnl_abs_num", default=_pct(_g(a, "pnl_abs"))),
            "signal": _txt(a, "action").upper() if _g(a, "action") else DASH,
            "status": "HOTSPOT" if _g(a, "euler_extreme") else "",
            "fund": _g(a, "fundamentals", default={}) or {},
            "note": _txt(a, "note") if _g(a, "note") else "",
        })
    conc = sorted(
        [{"t": h["t"], "w": h["w"], "beta": _num(_find(assets, "ticker", h["t"]), "beta", default=0.0),
          "risk": h["risk"], "status": h["status"]} for h in holdings],
        key=lambda x: x["risk"], reverse=True)[:5]

    # risk waterfall
    wf = _g(p, "risk_waterfall", default={}) or {}
    standalone = [{"t": _txt(c, "ticker"), "v": _num(c, "standalone_pp")}
                  for c in _list(wf, "contributions")[:4]]
    riskDecomp = {
        "standalone": standalone,
        "diversification": -abs(_num(wf, "diversification_pp")),
        "total": _num(wf, "total_vol_pp"),
        "sumStandalone": _num(wf, "sum_standalone_pp"),
    }

    # sectors
    sectors = [{"name": _txt(s, "name"), "pct": round(_num(s, "weight_pct")),
                "warn": bool(_g(s, "warn") or _num(s, "weight_pct") >= 40),
                "hue": "#1c1b1a"} for s in _list(p, "sectors")]

    # factors
    factors = [{"name": _txt(f, "axis"), "port": _num(f, "beta"), "mkt": _num(f, "bench", default=1.0)}
               for f in _list(p, "factor_betas")]

    # 4-pillar scores
    scores = [{"t": _txt(s, "ticker"), "total": _num(s, "total"), "action": _txt(s, "action").upper(),
               "F": _num(s, "fundamentals"), "V": _num(s, "valuations"),
               "T": _num(s, "technicals"), "C": _num(s, "credit"),
               "reason": _txt(s, "reason") if _g(s, "reason") else ""} for s in _list(p, "score_breakdown")]

    # stress
    stress = [{"name": _txt(s, "name"), "pct": round(_num(s, "port_pct") * 100, 1),
               "usd": round(_num(s, "port_dollar")), "dd": round(_num(s, "max_dd_pct") * 100, 1),
               "rec": (f"{_num(s, 'recovery_months')} мес" if _g(s, "recovery_months") else DASH)}
              for s in _list(p, "stress_scenarios")]

    # expected effect → 8 cards
    ee = _g(p, "expected_effect", default={}) or {}
    _ELABELS = [("risk_index", "Индекс риска"), ("vol", "Волатильность"), ("cvar_95", "CVaR 95%"),
                ("max_drawdown", "Max Drawdown"), ("sharpe", "Sharpe"), ("max_erc_pct", "Max TRC"),
                ("it_share", "Доля IT"), ("expected_return", "Ожид. доходность")]
    effect = []
    for key, label in _ELABELS:
        cell = _g(ee, key, default={}) or {}
        fav = _g(cell, "favourable")
        effect.append({"name": label, "before": _txt(cell, "before"), "after": _txt(cell, "after"),
                       "delta": _txt(cell, "delta_pp"),
                       "tone": "pos" if fav is True else ("neg" if fav is False else "flat")})

    # action plan
    plan = [{"t": _txt(a, "ticker"), "action": _txt(a, "action").upper().split()[0] if _g(a, "action") else DASH,
             "price": _num(a, "price"),  # design calls price.toFixed → MUST be numeric
             "target": _txt(a, "sell_target") if _g(a, "sell_target") else _txt(a, "buy_zone"),
             "stop": _txt(a, "stop_loss"), "score": _num(a, "score_total", default=0),
             "hot": bool(_g(a, "hotspot"))} for a in _list(p, "action_plan")]

    # ideas (ai_ideas buckets → 4 cards)
    ideas = _map_ideas(p)

    # regime
    reg = _g(p, "regime", default={}) or {}
    rc = _g(p, "regime_confirmation", default={}) or {}
    drivers = []
    for k, m in (_g(p, "macro_drivers", default={}) or {}).items():
        if isinstance(m, dict) and _g(m, "value") is not None:
            drivers.append({"name": _txt(m, "label"), "val": _txt(m, "value"),
                            "trend": _txt(m, "trend_label") if _g(m, "trend_label") else "",
                            "state": _txt(m, "status"), "tone": "pos"})
    regime = {
        "name": _txt(reg, "label").split()[0] if _g(reg, "label") else DASH,
        "nameRu": _txt(reg, "label_ru") if _g(reg, "label_ru") else _txt(reg, "label"),
        "confidence": round(_num(reg, "confidence") * (100 if _num(reg, "confidence") <= 1 else 1)),
        "confirms": len([s for s in _list(rc, "signals")]) or 0,
        "growth": _num(reg, "growth"), "cycle": _num(reg, "cycle"),
        "dot": {"growth": _num(reg, "growth"), "cycle": _num(reg, "cycle")},
        "ragSignals": [str(x) for x in _list(reg, "explainers")][:4],
        "drivers": drivers,
        "confirm": _txt(rc, "stance") if _g(rc, "stance") else DASH,
        "confirmBullets": [str(x) for x in _list(rc, "signals")][:6],
        "regimeAI": _txt(p, "ai_regime_comment"),
    }

    # CoVe + quality
    cove = [{"st": _cove_st(_g(c, "status")), "title": _txt(c, "name"),
             "meta": " · ".join(x for x in [_txt(c, "source"), _txt(c, "method"), _txt(c, "note")]
                                 if x and x != DASH)} for c in _list(p, "cove_lineage")]
    quality = [f"{_txt(q, 'status')} {_txt(q, 'label')}" for q in _list(p, "integrity_checks")]

    # mandate
    mc = _g(p, "mandate_compliance", default={}) or {}
    mandate = {
        "profile": _txt(p, "risk_mandate_label"),
        "targetVol": _num(mc, "target_vol", default=DASH if not _g(mc, "target_vol") else 0),
        "trackingCap": _g(mc, "tracking_cap", default=DASH),
        "violations": len([r for r in _list(mc, "rows") if _g(r, "state") in ("over", "under")]),
        "rows": [{"label": _txt(r, "label"), "value": _num(r, "value"),
                  "lo": _num(r, "lo"), "hi": _num(r, "hi"), "state": _txt(r, "state")}
                 for r in _list(mc, "rows")],
    }

    return {
        "meta": {**meta, "tier": "DEEP", "engine": "MAC3"},
        "verdict": {"headline": _txt(p, "ai_verdict"), "sub": _txt(p, "ai_plain_summary"),
                    "riskIndex": round(_num(p, "risk_pct")), "riskTier": _txt(p, "risk_label"),
                    # BLOCK 5 — portfolio FORWARD expected annual return shown
                    # next to the risk index ("доходность относительно риска").
                    "expReturn": _txt(p, "expected_return_annual"),
                    "expSharpe": _txt(p, "expected_sharpe"),
                    "summary": _txt(p, "ai_plain_summary"), "bullets": bullets},
        "mandate": mandate,
        "heroStats": [
            {"label": "Позиции", "value": _g(p, "holdings_count", default=DASH), "icon": "briefcase"},
            {"label": "NAV", "value": _txt(p, "total_value_usd"), "icon": "wallet"},
            {"label": "Профиль", "value": _txt(p, "risk_mandate_label"), "icon": "shield", "small": True},
        ],
        "kpis": [
            _kpi(p, "cvar", "CVaR 95%", "cvar", "ai_cvar_note", "cvar_svg", "normal", "#5d7c5c", _txt(p, "cvar_dollar")),
            _kpi(p, "sharpe", "Sharpe Ratio", "sharpe", "ai_sharpe_note", "sharpe_svg", "good", "#caa01a",
                 f"Sortino {_txt(p, 'sortino')}"),
            _kpi(p, "dd", "Max Drawdown", "max_drawdown", "ai_mdd_note", "mdd_svg", "watch", "#c47358", _txt(p, "mdd_dollar")),
        ],
        "concentration": conc, "riskDecomp": riskDecomp,
        "concAI": _txt(p, "ai_risk_comment") if _g(p, "ai_risk_comment") else _txt(p, "ai_holdings_comment"),
        "holdings": holdings, "sectors": sectors,
        "sectorWarn": [str(x) for x in _list(p, "sector_warnings")][:3] or [DASH],
        "holdingsAI": _txt(p, "ai_holdings_comment"),
        "factors": factors, "factorCoverage": _coverage(p), "factorAI": _txt(p, "ai_factor_comment"),
        "scores": scores, "scoresNote": _txt(p, "ai_4pillar_comment"), "scoresAI": _txt(p, "ai_4pillar_comment"),
        "stress": stress, "stressAI": _txt(p, "ai_stress_comment"),
        "effect": effect, "effectVerdict": _txt(ee, "verdict", "headline"), "effectAI": _txt(p, "ai_effect_comment"),
        "actionPlan": plan, "actionAI": _txt(p, "ai_action_comment"),
        "ideas": ideas, "regime": regime, "cove": cove, "quality": quality or [DASH],
    }


# ── BASE mapper ───────────────────────────────────────────────────────────────

def _map_base(p: dict, meta: dict) -> dict:
    assets = _list(p, "assets")
    holdings = [{
        "t": _txt(a, "ticker"), "name": _txt(a, "name") if _g(a, "name") else _txt(a, "ticker"),
        "cls": _txt(a, "asset_class"), "w": _num(a, "weight_pct_num"),
        "beta": _num(a, "beta"), "risk": _num(a, "euler_risk_pct"),
        "pnlPct": _pct(_g(a, "pnl_pct")), "pnlUsd": _pct(_g(a, "pnl_abs")),
        "status": "HOTSPOT" if _g(a, "euler_extreme") else "",
        "signal": _txt(a, "action").upper() if _g(a, "action") else DASH,
        "fund": _g(a, "fundamentals", default={}) or {}, "note": _txt(a, "note") if _g(a, "note") else "",
    } for a in assets]

    hot = (_list(p, "hotspots") or [{}])[0]
    sectors = [{"name": _txt(s, "name"), "pct": round(_num(s, "weight_pct")),
                "warn": bool(_g(s, "warn") or _num(s, "weight_pct") >= 40), "hue": "#1c1b1a"}
               for s in _list(p, "sectors")]
    wf = _g(p, "risk_waterfall", default={}) or {}
    riskDecomp = {
        "standalone": [{"t": _txt(c, "ticker"), "v": _num(c, "standalone_pp")} for c in _list(wf, "contributions")[:4]],
        "diversification": -abs(_num(wf, "diversification_pp")), "total": _num(wf, "total_vol_pp"),
        "sumStandalone": _num(wf, "sum_standalone_pp"),
    }

    def _kpi_obj(label, value_key, unit, frame):
        return {"label": label, "value": _num(p, value_key), "unit": unit,
                "delta": 0, "deltaText": "", "frame": frame}

    return {
        "meta": {**meta, "tier": "BASE", "engine": "MAC3"},
        "verdict": {"headline": _txt(p, "ai_verdict"), "sub": _txt(p, "ai_plain_summary"),
                    "riskIndex": round(_num(p, "risk_pct")),
                    # BLOCK 5 — portfolio FORWARD expected annual return + Sharpe.
                    "expReturn": _txt(p, "expected_return_annual"),
                    "expSharpe": _txt(p, "expected_sharpe"),
                    "riskTrendDelta": _g(p, "risk_score_delta", default=0), "nav": _txt(p, "total_value_usd")},
        "kpis": {
            "cvar": _kpi_obj("CVaR 95%", "cvar", "%", _txt(p, "cvar_dollar")),
            "sharpe": _kpi_obj("Sharpe Ratio", "sharpe", "", f"Sortino {_txt(p, 'sortino')}"),
            "dd": _kpi_obj("Max Drawdown", "max_drawdown", "%", _txt(p, "mdd_dollar")),
            "vol": _kpi_obj("Волатильность", "volatility", "%", "год."),
        },
        "factorPills": [{"label": _txt(f, "axis"), "value": _num(f, "beta"), "accent": "gold",
                         "warn": bool(_g(f, "missing")), "cap": 0} for f in _list(p, "factor_betas")[:5]],
        "heroStats": [
            # BASE Hero icon map only has {briefcase, trendUp, wallet} (no shield).
            {"label": "Позиции", "value": _g(p, "holdings_count", default=DASH), "icon": "briefcase"},
            {"label": "NAV", "value": _txt(p, "total_value_usd"), "icon": "wallet"},
            {"label": "Профиль", "value": _txt(p, "risk_mandate_label"), "icon": "trendUp", "small": True},
        ],
        "topHotspot": {"ticker": _txt(hot, "ticker"), "name": _txt(hot, "ticker"), "sector": DASH,
                       "weight": _num(hot, "weight_pct"), "riskShare": _num(hot, "trc_pct"),
                       "pnlPct": 0, "pnlUsd": 0, "signal": DASH, "note": _txt(hot, "reason")},
        "sectors": sectors, "riskDecomp": riskDecomp, "holdings": holdings,
        "performance": _map_performance(p),
        "ideas": _map_ideas(p, base=True),
    }


# ── shared helpers ────────────────────────────────────────────────────────────

def _coverage(p: dict) -> float:
    dq = _g(p, "data_quality", default={}) or {}
    fl, ft = _num(dq, "factors_loaded"), _num(dq, "factors_total", default=10.0)
    return round(fl / ft * 100, 1) if ft else 0.0


def _cove_st(status: Any) -> str:
    """v3 5-state CoVe status → the design's 3-state {ok, warn, fail}.
    'disabled'/'missing' (intentionally not consulted) → 'warn' (amber), NOT
    'fail' (red ✗) — preserves the «выключено ≠ сломано» semantic."""
    s = str(status or "").strip().lower()
    if s == "ok":
        return "ok"
    if s == "error":
        return "fail"
    return "warn"   # warn / stale / missing / disabled → amber


def _find(rows: list, key: str, val: str) -> dict:
    for r in rows or []:
        if isinstance(r, dict) and str(_g(r, key)) == str(val):
            return r
    return {}


def _map_performance(p: dict) -> dict:
    """v3 period_returns_table (dict-of-benchmarks) → design performance block.
    `periods` MUST be a list ([{label,p,s,d}]); the equity-curve arrays
    (months/port/spx) aren't in the payload, so they're empty (chart degrades)."""
    prt = _g(p, "period_returns_table", default={}) or {}
    first = next(iter(prt.values()), {}) if isinstance(prt, dict) else {}
    periods = [{"label": _txt(r, "label"), "p": _pct(_g(r, "portfolio")),
                "s": _pct(_g(r, "benchmark")), "d": _pct(_g(r, "excess"))}
               for r in _list(first, "periods")]
    # Build a coarse equity curve from the period horizons so PerfChart has
    # real, NaN-free points (the payload has no monthly series).  0 → 1m → 3m …
    _ORD = {"1 мес": 1, "1м": 1, "3 мес": 3, "3м": 3, "6 мес": 6, "6м": 6,
            "12 мес": 12, "12м": 12, "YTD": 9}
    pts = sorted(periods, key=lambda r: _ORD.get(r["label"], 99))
    months = ["старт"] + [r["label"] for r in pts]
    port   = [0.0] + [r["p"] for r in pts]
    spx    = [0.0] + [r["s"] for r in pts]
    return {"months": months, "port": port, "spx": spx,
            "vol": {"port": _num(p, "volatility"), "spx": 0},
            "periods": periods}


def _map_ideas(p: dict, base: bool = False) -> list:
    """ai_ideas buckets → the design idea-card list."""
    ai = _g(p, "ai_ideas", default={}) or {}
    # DEEP `ideaTone` map only has these keys → any other → component crash.
    _TONE = {"growth": "grow", "rotation": "rotation", "hedge": "hedge",
             "risk_reduction": "rebalance", "diversification": "rebalance"}
    out, n = [], 0
    for bucket, cards in ai.items():
        for c in (cards if isinstance(cards, list) else []):
            n += 1
            cands = [{"t": _txt(x, "ticker"), "name": _txt(x, "name"), "why": _txt(x, "scenario") if _g(x, "scenario") else _txt(x, "why")}
                     for x in _list(c, "candidates")]
            item = {
                "n": f"{n:02d}", "cat": _txt(c, "category"), "prio": _txt(c, "priority"),
                "tone": _TONE.get(bucket, "grow"), "title": _txt(c, "title"),
                "lede": _txt(c, "rationale") if _g(c, "rationale") else _txt(c, "lede"),
                "pipeline": [(_txt(s, "stage") + ": " + _txt(s, "detail")) if isinstance(s, dict) else str(s)
                             for s in _list(c, "pipeline")],
            }
            if base:
                item["tickers"] = cands      # list[{t,name,why}] — card reads t.t / t.why
                item["effect"] = []          # list[str] — component does .map
                item["sources"] = []         # list[str] — component does .map
            else:
                item["cands"] = cands
            out.append(item)
    return out[:4]


# ── Public API ────────────────────────────────────────────────────────────────

def build_design_data(payload: dict | None, tier: str = "base",
                      *, user_id: Any = None, generated_at: str | None = None) -> dict:
    """
    Map the engine v3 `payload` → the Premium V2 design contract for `tier`.

    Pure translation: never raises (defensive `_g`), never imports/touches the
    risk engine.  Missing values → '–' (text) / 0 (chart-numeric).
    """
    p = payload or {}
    is_deep = str(tier).lower() == "deep"
    meta = {
        "id": str(user_id) if user_id is not None else _txt(p, "user_id"),
        "aiModel": _txt(p, "ai_model_used"),
        "profile": _txt(p, "risk_mandate_label"),
        "generated": generated_at or DASH,
        "session": generated_at or DASH,
        "nav": _txt(p, "total_value_usd"),
        "positions": _g(p, "holdings_count", default=DASH),
    }
    return _map_deep(p, meta) if is_deep else _map_base(p, meta)


__all__ = ["build_design_data"]
