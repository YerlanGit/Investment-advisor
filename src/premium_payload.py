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
    # The Premium KPI card redraws the trend with the design's own <Sparkline>
    # (theme colour + gradient), so we feed it the REAL numeric series the engine
    # now exposes alongside the SVG (kpi_sparklines.<metric>_pts).  The SVG is
    # kept as a fallback for any payload that only carries the rendered image.
    pts_key = spark_key.replace("_svg", "_pts")
    pts = _g(p, "kpi_sparklines", pts_key, default=None)
    pts = [x for x in pts if isinstance(x, (int, float))] if isinstance(pts, list) else []
    return {
        "key": key, "name": name,
        "value": _txt(p, val_key),
        "status": status, "color": color,
        "sub": sub if sub != DASH else _txt(p, val_key + "_dollar"),
        "ai": _txt(p, note_key),
        "pts": pts, "svg": spark,
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
    fmap = _fund_map(p)
    holdings = []
    for a in assets:
        fund = _fund_for(fmap, a)
        holdings.append({
            "t": _txt(a, "ticker"), "name": _txt(a, "name") if _g(a, "name") else _txt(a, "ticker"),
            "cls": _txt(a, "asset_class"),
            "sector": _txt(a, "sector") if _g(a, "sector") else "",
            "w": _num(a, "weight_pct_num"), "risk": _num(a, "euler_risk_pct"),
            "pnlPct": _num(a, "pnl_pct_num", default=_pct(_g(a, "pnl_pct"))),
            "pnlUsd": _num(a, "pnl_abs_num", default=_pct(_g(a, "pnl_abs"))),
            "signal": _txt(a, "action").upper() if _g(a, "action") else DASH,
            "status": "HOTSPOT" if _g(a, "euler_extreme") else "",
            "fund": fund,
            # Short, rule-based fundamental read-out composed from the SEC metrics
            # (user request: «добавь короткий вывод по Фундаменталу»).  Falls back
            # to the engine note when there's no SEC coverage.
            "fundNote": _fund_verdict(fund) or (_txt(a, "note") if _g(a, "note") else ""),
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

    # sectors — distinct on-brand colour per slice (was a single #1c1b1a for ALL,
    # so the stacked bar + legend squares rendered as one indistinguishable black).
    sectors = [{"name": _txt(s, "name"), "pct": round(_num(s, "weight_pct")),
                "warn": bool(_g(s, "warn") or _num(s, "weight_pct") >= 40),
                "hue": _sector_hue(_txt(s, "name"), i)} for i, s in enumerate(_list(p, "sectors"))]

    # factors
    factors = [{"name": _txt(f, "axis"), "port": _num(f, "beta"), "mkt": _num(f, "bench", default=1.0)}
               for f in _list(p, "factor_betas")]

    # factor-variance decomposition (источники риска + факторные двойники) —
    # additive layer; None hides the sub-block in the Factors section.
    fv = _g(p, "factor_variance", default=None)
    factor_variance = None
    if isinstance(fv, dict) and _list(fv, "rows"):
        factor_variance = {
            "rows": [{"source": _txt(r, "source"), "pct": _num(r, "share_pct"),
                      "drivers": _txt(r, "drivers") if _g(r, "drivers") else ""}
                     for r in _list(fv, "rows")],
            "systematic": _num(fv, "systematic_pct"),
            "idio":       _num(fv, "idio_pct"),
            "twins": [{"pair": _txt(t, "pair_label"), "corr": _num(t, "systematic_corr"),
                       "w": _num(t, "combined_weight_pct")} for t in _list(fv, "twins")],
        }

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
        # The source stores RAW numerics (fractions for %-metrics, points for the
        # index, a ratio for Sharpe); the design's EffectGrid prints before/after/
        # delta as plain strings.  The v3 template formats via _ef_card at render,
        # but the premium mapper used _txt() → raw "0.18692…" leaked into the card.
        # Format per metric type so the cards read 18.7% / 0.34 / 49 like v3.
        effect.append({"name": label,
                       "before": _eff_fmt(key, _g(cell, "before")),
                       "after":  _eff_fmt(key, _g(cell, "after")),
                       "delta":  _eff_delta(key, _g(cell, "delta_pp")),
                       "tone": "pos" if fav is True else ("neg" if fav is False else "flat")})

    # action plan — score + hotspot are NOT on the action_plan rows; they live in
    # score_breakdown (4-Pillar total) and assets (euler hotspot).  The mapper
    # previously read non-existent score_total/hotspot keys → every row showed
    # score 0 and hot=false.  Join them by ticker.
    _score_by_t = {_txt(s, "ticker"): _num(s, "total") for s in _list(p, "score_breakdown")}
    _hot_by_t = {_txt(a, "ticker"): bool(_g(a, "hotspot") or _g(a, "euler_extreme")) for a in assets}
    plan = [{"t": _txt(a, "ticker"), "action": _txt(a, "action").upper().split()[0] if _g(a, "action") else DASH,
             "price": _num(a, "price_num", default=_num(a, "price")),  # §−14 A-1: numeric twin first
             "target": _txt(a, "sell_target") if _g(a, "sell_target") else _txt(a, "buy_zone"),
             "stop": _txt(a, "stop_loss"),
             # Quantity to trade (user request «добавить столбец количество»): the
             # engine ships qty_delta (whole units, signed) and delta_w_pp (target
             # weight change in pp).  We surface both so «сколько продать/сократить»
             # is explicit — units for the order, pp for the portfolio impact.
             "qty": _qty_int(_g(a, "qty_delta")),
             "dw": _num(a, "delta_w_pp"),
             "score": _score_by_t.get(_txt(a, "ticker"), 0.0),
             "hot": _hot_by_t.get(_txt(a, "ticker"), False)} for a in _list(p, "action_plan")]

    # ideas (ai_ideas buckets → 4 cards)
    ideas = _map_ideas(p)

    # regime
    reg = _g(p, "regime", default={}) or {}
    rc = _g(p, "regime_confirmation", default={}) or {}
    # FRED macro-driver chips.  pdf_payload stores the ADAPTED panel
    # ({"series": [{name,value,status,trend_label,...}], "as_of"}), NOT the raw
    # FRED dict — so the old `.items()` walk (expecting {key:{value}}) always
    # yielded an empty list and the «координаты режима» chips never rendered even
    # though FRED was fresh.  Read the `series` list and drop only genuinely
    # missing prints (value formatted as "—").
    # Audit 2026-07-05 (R-9): the state pill used to leak the raw EN token
    # («ok»/«stale») — map to the design's RU labels.
    _drv_state_ru = {"ok": "актуально", "stale": "устарело",
                     "warn": "частично", "missing": "нет данных", "error": "ошибка"}
    drivers = []
    for m in _list(_g(p, "macro_drivers", default={}), "series"):
        if not isinstance(m, dict):
            continue
        val = _txt(m, "value")
        if val in (DASH, "—", "-", ""):
            continue
        st = str(_g(m, "status") or "").strip().lower()
        drivers.append({"name": _txt(m, "name"), "val": val,
                        "trend": _txt(m, "trend_label") if _g(m, "trend_label") else "",
                        "state": _drv_state_ru.get(st, st or "—"),
                        "tone": "pos" if st == "ok" else "warn"})
    _bullets_rc = [_signal_obj(x) for x in _list(rc, "signals")][:6]
    # Audit 2026-07-05 (R-6): the «RAG ·» chips used to show ETF momentum
    # (explainers) while the REAL bank excerpts (regime_rag_confirm) were a dead
    # payload key.  Prefer the true RAG confirmations when the KB returned any;
    # fall back to momentum explainers with an honest label (ragBacked=False).
    _rag_confirm = [str(x) for x in _list(p, "regime_rag_confirm") if str(x).strip()]
    _cons = _g(p, "regime_consistency", default={}) or {}
    regime = {
        "name": _txt(reg, "label").split()[0] if _g(reg, "label") else DASH,
        "nameRu": _txt(reg, "label_ru") if _g(reg, "label_ru") else _txt(reg, "label"),
        "confidence": round(_num(reg, "confidence") * (100 if _num(reg, "confidence") <= 1 else 1)),
        # R-5: «N подтверждающих сигнала» must count CONFIRMING (✓) bullets only,
        # not the contradicting ⚠/✗ ones the same list carries.
        "confirms": len([b for b in _bullets_rc if b.get("ok")]),
        "growth": _num(reg, "growth"), "cycle": _num(reg, "cycle"),
        "dot": {"growth": _num(reg, "growth"), "cycle": _num(reg, "cycle")},
        "ragSignals": (_rag_confirm[:4] if _rag_confirm
                       else [str(x) for x in _list(reg, "explainers")][:4]),
        "ragBacked": bool(_rag_confirm),
        "drivers": drivers,
        # R-2: the drivers header used to hardcode a design-mock date — carry the
        # real macro-pack as_of so the component renders live freshness.
        "driversAsOf": _txt(_g(p, "macro_drivers", default={}) or {}, "as_of"),
        # R-3: the banner paragraph used to render the raw stance token
        # («diverges») — carry the human summary; stance drives the header tone.
        "confirm": _txt(rc, "summary") if _g(rc, "summary") else (
            _txt(rc, "stance") if _g(rc, "stance") else DASH),
        "confirmStance": _txt(rc, "stance") if _g(rc, "stance") else "",
        "confirmBullets": _bullets_rc,
        # R-8: the deterministic FRED↔momentum cross-check (4.D) was v3-only —
        # surface it in premium too.
        "consistency": ({"status": _txt(_cons, "status"), "note": _txt(_cons, "note")}
                        if _cons else None),
        "regimeAI": _txt(p, "ai_regime_comment"),
    }

    # CoVe + quality
    cove = [{"st": _cove_st(_g(c, "status")), "title": _txt(c, "name"),
             "meta": " · ".join(x for x in [_txt(c, "source"), _txt(c, "method"), _txt(c, "note")]
                                 if x and x != DASH)} for c in _list(p, "cove_lineage")]
    quality = [f"{_txt(q, 'status')} {_txt(q, 'label')}" for q in _list(p, "integrity_checks")]

    # mandate — the engine's _build_mandate_compliance emits target_vol_pct /
    # target_te_pct / breaches and per-row {actual, status}.  The previous mapper
    # read non-existent keys (target_vol / tracking_cap / value / state), so the
    # whole panel rendered targetVol/trackingCap='–', every row value=0.0 and
    # state='–'.  Map the REAL keys; targetVol/trackingCap fall back to '–' only
    # when genuinely unset (0 ⇒ not configured).
    mc = _g(p, "mandate_compliance", default={}) or {}
    _tv = _num(mc, "target_vol_pct")
    _tc = _num(mc, "target_te_pct")
    _lev = _leverage_info(p, assets)
    mandate = {
        "profile": _txt(p, "risk_mandate_label"),
        "targetVol": round(_tv, 1) if _tv else DASH,
        "trackingCap": round(_tc, 1) if _tc else DASH,
        "violations": int(_num(mc, "breaches", default=0)),
        # Маржа рендерится ТОЛЬКО при leveraged (правило §−13: без отрицательного
        # кэша упоминания плеча скрыты).
        "leveraged": _lev["on"], "marginPct": _lev["marginPct"],
        "rows": [{"label": _txt(r, "label"), "value": round(_num(r, "actual"), 1),
                  "lo": _num(r, "lo"), "hi": _num(r, "hi"), "state": _txt(r, "status")}
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
        "sectorWarn": [_warn_text(x) for x in _list(p, "sector_warnings")][:3] or [DASH],
        "holdingsAI": _txt(p, "ai_holdings_comment"),
        "factors": factors, "factorCoverage": _coverage(p), "factorAI": _txt(p, "ai_factor_comment"),
        "factorVariance": factor_variance,
        # scoresNote dropped — it duplicated scoresAI verbatim (same
        # ai_4pillar_comment rendered twice).  The AI box keeps the comment once.
        "scores": scores, "scoresNote": "", "scoresAI": _txt(p, "ai_4pillar_comment"),
        "stress": stress, "stressAI": _txt(p, "ai_stress_comment"),
        "effect": effect, "effectVerdict": _txt(ee, "verdict", "headline"), "effectAI": _txt(p, "ai_effect_comment"),
        "actionPlan": plan, "actionAI": _txt(p, "ai_action_comment"),
        "ideas": ideas, "regime": regime, "cove": cove, "quality": quality or [DASH],
    }


# ── BASE mapper ───────────────────────────────────────────────────────────────

def _map_base(p: dict, meta: dict) -> dict:
    assets = _list(p, "assets")
    fmap = _fund_map(p)
    holdings = [{
        "t": _txt(a, "ticker"), "name": _txt(a, "name") if _g(a, "name") else _txt(a, "ticker"),
        "cls": _txt(a, "asset_class"),
        # GICS-style sector (may be ""), separate from the asset class — the
        # holdings sector filters (Технологии/Защитные) read this, not `cls`.
        "sector": _txt(a, "sector") if _g(a, "sector") else "",
        "w": _num(a, "weight_pct_num"),
        "beta": _num(a, "beta_num", default=_num(a, "beta")), "risk": _num(a, "euler_risk_pct"),
        # §−14 A-1: prefer RAW numeric twins; string-parse is the legacy fallback.
        "pnlPct": _num(a, "pnl_pct_num", default=_pct(_g(a, "pnl_pct"))),
        "pnlUsd": _num(a, "pnl_abs_num", default=_pct(_g(a, "pnl_abs"))),
        "status": "HOTSPOT" if _g(a, "euler_extreme") else "",
        "signal": _txt(a, "action").upper() if _g(a, "action") else DASH,
        "fund": _fund_for(fmap, a), "note": _txt(a, "note") if _g(a, "note") else "",
    } for a in assets]

    # Top risk hotspot = the asset with the largest Euler TRC.  The old mapper
    # indexed the `hotspots` list (which is a list of STRINGS, not dicts) as a
    # dict → the whole featured card rendered '–' / 0.  Derive it from assets.
    _risky_h = [a for a in assets if not _g(a, "is_cash")]
    _top = max(_risky_h, key=lambda a: _num(a, "euler_risk_pct"), default={}) if _risky_h else {}
    _top_t = _txt(_top, "ticker")
    _sig_by_t = {_txt(s, "ticker"): _txt(s, "action").upper() for s in _list(p, "score_breakdown")}
    _has_top = bool(_top_t and _top_t != DASH)
    topHotspot = {
        "ticker": _top_t, "name": _top_t, "sector": _txt(_top, "asset_class"),
        "weight": round(_num(_top, "weight_pct_num"), 1),
        "riskShare": round(_num(_top, "euler_risk_pct"), 1),
        "pnlPct": _num(_top, "pnl_pct_num", default=_pct(_g(_top, "pnl_pct"))),
        "pnlUsd": _num(_top, "pnl_abs_num", default=_pct(_g(_top, "pnl_abs"))),
        "signal": _sig_by_t.get(_top_t) or (_txt(_top, "action").upper() if _g(_top, "action") else DASH),
        "note": (f"Наибольший вклад в риск — {round(_num(_top, 'euler_risk_pct'), 1)}% общего риска портфеля"
                 if _has_top else DASH),
    }
    sectors = [{"name": _txt(s, "name"), "pct": round(_num(s, "weight_pct")),
                "warn": bool(_g(s, "warn") or _num(s, "weight_pct") >= 40), "hue": _sector_hue(_txt(s, "name"), i)}
               for i, s in enumerate(_list(p, "sectors"))]
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
                    "riskTier": _txt(p, "risk_label"),   # real tier for the AI-insight card
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
        "factorPills": _base_factor_pills(p, assets, sectors, riskDecomp),
        "heroStats": [
            # BASE Hero icon map only has {briefcase, trendUp, wallet} (no shield).
            {"label": "Позиции", "value": _g(p, "holdings_count", default=DASH), "icon": "briefcase"},
            {"label": "NAV", "value": _txt(p, "total_value_usd"), "icon": "wallet"},
            {"label": "Профиль", "value": _txt(p, "risk_mandate_label"), "icon": "trendUp", "small": True},
        ],
        "topHotspot": topHotspot,
        "sectors": sectors, "riskDecomp": riskDecomp, "holdings": holdings,
        # Показывается ТОЛЬКО при leverage.on (отрицательный кэш) — правило §−13.
        "leverage": _leverage_info(p, assets),
        "performance": _map_performance(p),
        "ideas": _map_ideas(p, base=True),
    }


# ── shared helpers ────────────────────────────────────────────────────────────

def _leverage_info(p: dict, assets: list) -> dict:
    """Margin/leverage indicator — {'on': bool, 'marginPct': float}.

    Business rule (аудит §−13): leverage/долг показывается ТОЛЬКО когда счёт
    реально маржинальный (кэш-баланс отрицательный); на нелевереджёванном
    портфеле все упоминания плеча скрываются.  Источник — engine'овский
    mandate_compliance (leveraged / margin_debt_pct); fallback — знак суммарного
    веса cash-позиций (надёжен даже на тонком payload)."""
    mc = _g(p, "mandate_compliance", default={}) or {}
    lev = bool(_g(mc, "leveraged"))
    margin = _num(mc, "margin_debt_pct")
    if not lev:
        cash_w = sum(_num(a, "weight_pct_num") for a in assets if _g(a, "is_cash"))
        if cash_w < -0.05:                     # отрицательный кэш = маржа
            lev, margin = True, (margin or round(-cash_w, 1))
    return {"on": lev, "marginPct": round(margin, 1) if margin else 0.0}


# Sector names counted as the «tech complex» for the hero Tech-share pill.
_TECH_SECTOR_RE = ("tech", "semicond", "полупровод", "технолог")


def _base_factor_pills(p: dict, assets: list, sectors: list, wf: dict) -> list:
    """BASE hero factor pills.

    Primary source — factor_betas (β по осям).  In production the engine wires
    factor_betas ТОЛЬКО для DEEP (tg_bot `if tier == TIER_DEEP`), so the BASE
    strip rendered EMPTY on live reports (потерянный параметр, аудит §−13).
    Fallback: synthesize the strip from data ALREADY in the payload — tech
    share, hotspot risk share, diversification effect, weighted market beta.
    Every number is real (no template literals)."""
    fb = _list(p, "factor_betas")
    if fb:
        return [{"label": _txt(f, "axis"), "value": _num(f, "beta"), "accent": "gold",
                 "warn": bool(_g(f, "missing")), "cap": 0} for f in fb[:5]]
    pills: list = []
    # 1. Tech-доля (Technology + Semiconductors, % of invested book)
    tech = sum(_num(s, "pct") for s in sectors
               if any(k in str(_g(s, "name") or "").lower() for k in _TECH_SECTOR_RE))
    if tech > 0:
        pills.append({"label": "Tech-доля", "value": round(tech), "cap": 100,
                      "accent": "gold", "warn": tech >= 40, "suffix": "%"})
    # 2. Hotspots — какой % риска сидит в hotspot-позициях (Euler TRC)
    hot = sum(_num(a, "euler_risk_pct") for a in assets if _g(a, "euler_extreme"))
    if hot > 0:
        pills.append({"label": "Hotspots", "value": round(hot), "cap": 100,
                      "accent": "dark", "warn": hot >= 40,
                      "display": f"{hot:.0f}", "suffix": "% риска"})
    # 3. Диверсификация — сколько риска гасится (benefit / Σstandalone)
    ss, dv = _num(wf, "sumStandalone"), abs(_num(wf, "diversification"))
    if ss > 0 and dv > 0:
        eff = round(dv / ss * 100)
        pills.append({"label": "Диверсификация", "value": eff, "cap": 100,
                      "accent": "mute", "warn": False,
                      "display": f"{eff}", "suffix": "% эффект"})
    # 4. Beta — взвешенная рыночная бета рисковой части книги
    bw = [(_num(a, "beta"), _num(a, "weight_pct_num")) for a in assets
          if not _g(a, "is_cash") and _num(a, "weight_pct_num") > 0]
    tot_w = sum(w for _, w in bw)
    if tot_w > 0:
        beta = sum(b * w for b, w in bw) / tot_w
        if beta:
            pills.append({"label": "Beta", "value": round(beta * 100), "cap": 200,
                          "accent": "mute", "warn": abs(beta) > 1.5,
                          "display": f"{beta:.2f}", "raw": True})
    return pills

# Expected-effect cards: which metric keys are stored as FRACTIONS (×100 = %).
# risk_index → integer points; sharpe → bare ratio; everything else → percent.
_EFFECT_PCT = {"vol", "cvar_95", "max_drawdown", "it_share", "expected_return", "max_erc_pct"}


def _eff_to_num(v: Any):
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace("−", "-").replace("%", "")
                     .replace(",", ".").replace("пп", "").strip())
    except (TypeError, ValueError):
        return None


def _eff_fmt(key: str, v: Any) -> str:
    """Format a before/after value for the design's effect card."""
    x = _eff_to_num(v)
    if x is None:
        return DASH
    if key == "risk_index":
        return f"{round(x)}"
    if key == "sharpe":
        return f"{x:.2f}"
    return f"{x * 100:.1f}%"          # %-metric stored as a fraction


def _eff_delta(key: str, v: Any) -> str:
    """Format the delta (already in display units: pp / points / ratio)."""
    x = _eff_to_num(v)
    if x is None:
        return DASH
    if key == "risk_index":
        return f"{x:+.0f} пункта"
    if key == "sharpe":
        return f"{x:+.2f}"
    return f"{x:+.1f} пп"


def _signal_obj(s: Any) -> dict:
    """Parse a regime-confirmation signal «✓/⚠/✗ текст» → {ok, t}.

    The DEEP regime component renders each bullet as an object (b.ok → Check vs
    Warning icon, b.t → text).  The mapper passed the raw STRING, so b.t was
    undefined and the panel showed six icon-only rows with NO text."""
    txt = str(s).strip()
    ok = txt[:1] in ("✓", "✔")
    for ic in ("✓", "✔", "⚠️", "⚠", "✗", "✘", "•"):
        if txt.startswith(ic):
            txt = txt[len(ic):].strip()
            break
    return {"ok": ok, "t": txt}


# Distinct on-brand sector palette (warm cream/gold/ink/sage/rust theme).  A few
# sectors get a SEMANTIC colour (Gold→gold, Silver→silver-grey, Bonds→calm sage);
# the rest cycle through the palette by index so every slice is distinguishable.
_SECTOR_PALETTE = ["#caa01a", "#3a3833", "#5d7c5c", "#c47358", "#9a7a10",
                   "#7a9a78", "#a85a40", "#55524c", "#eac233", "#c4bfb5"]
_SECTOR_NAMED = {
    "gold": "#eac233", "золото": "#eac233",
    "silver": "#9a958c", "серебро": "#9a958c",
    "bonds": "#5d7c5c", "облигации": "#5d7c5c",
    "technology": "#caa01a", "технологии": "#caa01a",
    "semiconductors": "#9a7a10", "полупроводники": "#9a7a10",
    "other": "#c4bfb5", "прочее": "#c4bfb5",
}


def _sector_hue(name: Any, i: int) -> str:
    key = str(name or "").strip().lower()
    if key in _SECTOR_NAMED:
        return _SECTOR_NAMED[key]
    return _SECTOR_PALETTE[i % len(_SECTOR_PALETTE)]


def _warn_text(x: Any) -> str:
    """Sector-warning → its human text.  The engine emits dicts
    {sector, weight_pct, cap_pct, overage_pp, text}; the mapper used to str() the
    whole dict, leaking «{'sector': 'Technology', ...}» into the UI."""
    if isinstance(x, dict):
        return _txt(x, "text")
    return str(x).strip()


def _fund_map(p: dict) -> dict:
    """{ticker → design `fund` object} joined from the engine's SEC-EDGAR layer.

    The SEC fundamentals are NOT on assets[] — they live in a SEPARATE
    `fundamental_layer` list keyed by ticker (roe / op_m / dta / rev_g /
    altman_z).  The mapper previously read a non-existent assets[].fundamentals
    key, so EVERY holding's fundamentals grid was empty — even AAPL / MSFT /
    NVDA whose 4-Pillar F-scores prove SEC data was present.  Map the engine
    keys → the component's {roe,margin,debt,growth,z}; `atr` is added per-holding
    from the asset row (it is a price metric, not an SEC field)."""
    out: dict = {}
    for r in _list(p, "fundamental_layer"):
        t = _txt(r, "ticker")
        if not t or t == DASH:
            continue
        out[t] = {
            "roe":    _txt(r, "roe"),
            "margin": _txt(r, "op_m"),
            "debt":   _txt(r, "dta"),
            "growth": _txt(r, "rev_g"),
            "z":      _txt(r, "altman_z"),
        }
    return out


def _fund_for(fmap: dict, a: Any) -> dict:
    """Per-holding `fund` object — SEC fields from the join (or 'н/д' when the
    instrument has no SEC coverage: ETFs / cash / EM-proxies), `atr` always from
    the asset's own price row."""
    sec = fmap.get(_txt(a, "ticker"), {})
    atr = _txt(a, "atr_pct")
    return {
        "roe":    sec.get("roe", "н/д"),
        "margin": sec.get("margin", "н/д"),
        "debt":   sec.get("debt", "н/д"),
        "growth": sec.get("growth", "н/д"),
        "atr":    atr if atr and atr not in (DASH, "—", "-") else "н/д",
        "z":      sec.get("z", "н/д"),
    }


def _qty_int(v: Any):
    """Action-plan quantity → signed int (units), or None when unavailable."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    try:
        return int(round(float(str(v).replace("−", "-").replace(" ", "").replace(",", "."))))
    except (TypeError, ValueError):
        return None


def _fund_verdict(fund: dict) -> str:
    """Compose a short, rule-based fundamental read-out from the SEC metrics.

    Reads the already-formatted strings on `fund` (roe / margin / debt / growth /
    z), parses the numbers defensively (skips 'н/д'), and returns one plain
    Russian sentence — strengths first, then the main caveat.  Returns '' when no
    SEC field is usable (ETFs / cash / EM proxies), so the caller can fall back to
    the engine note.  This is text composition over existing data — no finance
    math, no engine import (keeps the mapper's anti-corruption boundary)."""
    def f(key):
        v = fund.get(key)
        if v is None or str(v).strip() in ("н/д", DASH, "—", "-", ""):
            return None
        try:
            return float(str(v).replace("−", "-").replace("%", "").replace("+", "")
                         .replace(",", ".").strip())
        except (TypeError, ValueError):
            return None
    roe, margin, debt, growth, z = f("roe"), f("margin"), f("debt"), f("growth"), f("z")
    if all(x is None for x in (roe, margin, debt, growth, z)):
        return ""
    strengths, risks = [], []
    if roe is not None:
        (strengths if roe >= 18 else risks).append(
            f"высокая рентабельность капитала (ROE {roe:.0f}%)" if roe >= 18
            else (f"слабая рентабельность (ROE {roe:.0f}%)" if roe < 8 else None))
    if margin is not None and margin >= 25:
        strengths.append(f"маржинальность {margin:.0f}%")
    if growth is not None:
        (strengths if growth >= 12 else risks).append(
            f"рост выручки +{growth:.0f}% г/г" if growth >= 12
            else (f"выручка снижается ({growth:.0f}% г/г)" if growth < 0 else None))
    if debt is not None:
        if debt >= 60:
            risks.append(f"высокая долговая нагрузка (долг/активы {debt:.0f}%)")
        elif debt <= 25:
            strengths.append("низкий долг")
    if z is not None:
        if z < 1.8:
            risks.append(f"зона риска по Altman-Z ({z:.1f})")
        elif z >= 3:
            strengths.append(f"запас прочности по Altman-Z ({z:.1f})")
    strengths = [s for s in strengths if s]
    risks = [r for r in risks if r]
    parts = []
    if strengths:
        parts.append("Сильные стороны: " + ", ".join(strengths[:3]))
    if risks:
        parts.append(("Риски: " if not strengths else "риски: ") + ", ".join(risks[:2]))
    return (". ".join(parts) + ".") if parts else ""


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

    The chart curve is NOT carried here: the payload has no monthly series, so the
    PerfChart component derives a REAL cumulative curve for the selected window by
    exact nesting of the period endpoints (cum[start..−h] = (1+r_window)/(1+r_h)−1).
    That keeps the chart, the headline and the breakdown table reading the SAME
    numbers, and makes the period selector drive the chart.  We therefore only ship
    `periods` ([{label,p,s,d}]), annualised `vol`, and a `summary` (volatility +
    12-month headline figures for any non-chart consumer)."""
    prt = _g(p, "period_returns_table", default={}) or {}
    first = next(iter(prt.values()), {}) if isinstance(prt, dict) else {}
    periods = []
    for r in _list(first, "periods"):
        # §−14 A-1: numeric twins first; string-parse is the legacy fallback.
        _pp = _num(r, "portfolio_num", default=_pct(_g(r, "portfolio")))
        _ss = _num(r, "benchmark_num", default=_pct(_g(r, "benchmark")))
        _dd = _num(r, "excess_num",    default=_pct(_g(r, "excess")))
        if not _dd:                       # source usually omits excess → derive p−s
            _dd = round(_pp - _ss, 1)
        periods.append({"label": _txt(r, "label"), "p": _pp, "s": _ss, "d": _dd})
    _vol = round(_num(p, "volatility_num", default=_num(p, "volatility")), 1)
    # 12-month headline summary — REAL data (the component reads summary.volPort
    # for the period-independent annualised volatility card).
    _p12 = next((r for r in periods if r["label"] in ("12 мес", "12М", "12м")),
                (periods[-1] if periods else {}))
    summary = {
        "ret":     _p12.get("p", 0.0),
        "spx":     _p12.get("s", 0.0),
        "exc":     _p12.get("d", round(_p12.get("p", 0.0) - _p12.get("s", 0.0), 1)),
        "volPort": _vol,
    }
    return {"vol": {"port": _vol, "spx": 0}, "periods": periods, "summary": summary}


def _pipe_step(s: Any) -> str:
    """Render one idea-pipeline step → its DETAIL string.

    Both idea components (BASE PipelineNode, DEEP PipeNode) supply their OWN
    stage label by position, so the value must be the detail ALONE — the design
    samples store plain strings like 'Momentum + Quality скоринг'.

    The source step may be a {stage,detail} dict OR a (stage, detail) tuple/list.
    The previous mapper only handled the dict and fell back to `str(s)`, leaking a
    raw Python tuple repr — «('RAG', 'Фундаментальный анализ')» — into the card.
    Return the detail (tuple/list → element[1:], dict → detail) so no double
    label and no tuple repr reaches the UI."""
    if isinstance(s, dict):
        return _txt(s, "detail") if _g(s, "detail") else _txt(s, "stage")
    if isinstance(s, (list, tuple)):
        parts = [str(x).strip() for x in s if x is not None and str(x).strip()]
        if len(parts) >= 2:
            return " · ".join(parts[1:])      # drop the stage; card labels it
        return parts[0] if parts else ""
    return str(s)


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
                "pipeline": [_pipe_step(s) for s in _list(c, "pipeline")],
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
