"""
AI narrative generator — produces verdict, plain-language summary, insight bullets,
stock-pick scenarios, and (deep tier only) action-plan commentary for the PDF templates.

Model selection:
  Base tier  → Claude Haiku  (fast, low-cost; 1 200 output tokens)
  Deep tier  → Claude Sonnet (higher quality prose + RAG synthesis; 6 000 tokens)

Falls back to a rule-based summary when the API key is absent or the call fails,
so the PDF is always produced regardless of API availability.

ALL output is in Russian.  Source tags [Quant Engine], [SEC EDGAR], [Regime],
[CDS], [RAG: file] are mandatory on every factual claim.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger("AINarrative")

MAX_TOKENS_BASE = 2_500
MAX_TOKENS_DEEP = 6_000

MODEL_BASE = os.getenv("ANTHROPIC_MODEL_BASE", "claude-haiku-4-5-20251001")
MODEL_DEEP = os.getenv("ANTHROPIC_MODEL_DEEP", "claude-sonnet-4-6")


# ── System prompt ────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    here = os.path.dirname(__file__)
    for path in [
        os.path.join(here, "..", "SYSTEM_PROMPT.md"),
        os.path.join(here, "..", "..", "SYSTEM_PROMPT.md"),
    ]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError:
            continue
    return ("You are an investment-advisory and risk-management assistant. "
            "Output advisory-only text in Russian. Never mention 'RAMP'.")


# ── Compact summary for LLM input ───────────────────────────────────────────

def _summarise_for_prompt(results: dict) -> dict:
    """
    Build a compact, JSON-serialisable view of analyze_all() results.
    Kept small to keep input tokens bounded.
    """
    metrics = results.get("portfolio_metrics") or {}
    perf    = results.get("performance_table")
    perf_rows: list[dict] = []
    if perf is not None and not perf.empty:
        df = perf.copy()
        if "Current_Value" in df.columns:
            df = df.sort_values("Current_Value", ascending=False)
        for _, row in df.head(25).iterrows():
            perf_rows.append({
                "ticker":   str(row.get("Ticker") or "?"),
                "weight":   round(float(row.get("Current_Value") or 0)
                                  / float(results.get("total_value") or 1) * 100, 2),
                "trc_pct":  _safe_round(row.get("Euler_Risk_Contribution_Pct"), 2),
                "atr_pct":  _safe_round(row.get("ATR_Pct"), 2),
                "pnl_pct":  _safe_round((row.get("Return_Pct") or 0) * 100, 2),
                "score":    _safe_round(row.get("Score_Total"), 1),
                "action":   row.get("Score_Action"),
                "sector":   row.get("Fundamental_Sector"),
                "beta_mkt": _safe_round(row.get("Beta_Market"), 2),
                "sec_roe":  _safe_round(row.get("SEC_ROE"), 3),
                "sec_debt": _safe_round(row.get("SEC_Debt_to_Assets"), 3),
                "sec_margin": _safe_round(row.get("SEC_Op_Margin"), 3),
            })
    bm = {
        name: {
            "ir":     _safe_round(d.get("Information_Ratio"), 2),
            "te":     _safe_round((d.get("Tracking_Error") or 0) * 100, 1),
            "excess": _safe_round((d.get("Excess_Return_Ann") or 0) * 100, 1),
        }
        for name, d in (results.get("benchmark_comparison") or {}).items()
    }
    cvar_boot = metrics.get("CVaR_95_Bootstrap") or {}
    return {
        "metrics": {
            "vol_ann":      _safe_round((metrics.get("Total_Volatility_Ann") or 0) * 100, 1),
            "sharpe":       _safe_round(metrics.get("Sharpe_Ratio"), 2),
            "sortino":      _safe_round(metrics.get("Sortino_Ratio"), 2),
            "var_95":       _safe_round((metrics.get("VaR_95_Daily") or 0) * 100, 2),
            "cvar_95":      _safe_round((metrics.get("CVaR_95_Daily") or 0) * 100, 2),
            "cvar_lo95":    _safe_round((cvar_boot.get("lo95") or 0) * 100, 2),
            "cvar_hi95":    _safe_round((cvar_boot.get("hi95") or 0) * 100, 2),
            "max_dd":       _safe_round((metrics.get("Max_Drawdown") or 0) * 100, 2),
            "max_erc_pct":  _safe_round(metrics.get("Max_Euler_Risk_Pct"), 1),
            "composite":    metrics.get("Composite_Risk_Score"),
        },
        "total_value":  _safe_round(results.get("total_value"), 0),
        "regime":       results.get("regime"),
        "sectors":      {k: round(float(v) * 100, 1)
                         for k, v in (results.get("sector_exposure") or {}).items()},
        "benchmarks":   bm,
        "holdings":     perf_rows,
        "action_plan":  (results.get("action_plan") or [])[:8],
        "asset_scores": {t: s for t, s in
                         (results.get("asset_scores") or {}).items()},
    }


def _safe_round(v, digits: int):
    try:
        x = float(v)
        return None if x != x else round(x, digits)
    except (TypeError, ValueError):
        return None


# ── Strip unverified RAG citations ───────────────────────────────────────────

def _strip_unverified_rag_citations(text: str, market_context: str) -> str:
    if not text:
        return text
    if not market_context:
        return re.sub(r"\s*\[RAG:[^\]]*\]", "", text)
    def _keep(match: re.Match) -> str:
        name = match.group(1).strip()
        return match.group(0) if name and name in market_context else ""
    return re.sub(r"\[RAG:\s*([^\]]+)\]", _keep, text)


# ── JSON repair for truncated responses ──────────────────────────────────────

def _repair_truncated_json(raw: str) -> str:
    """
    Repair JSON truncated by max_tokens.  Closes open strings, arrays, objects.
    """
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    repaired = raw.rstrip()
    repaired = repaired.rstrip(',')
    # Close an open string
    if repaired.count('"') % 2 == 1:
        # Find last unmatched quote and truncate partial value
        last_q = repaired.rfind('"')
        # Check if it's a key or value being written
        before = repaired[:last_q].rstrip()
        if before.endswith(':'):
            # Truncated value — close string
            repaired = repaired + '"'
        else:
            # Truncated mid-string — close it
            repaired = repaired + '"'
    # Remove trailing partial key-value (key without value)
    repaired = re.sub(r',\s*"[^"]*"\s*:\s*$', '', repaired)
    # Close open arrays and objects
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    repaired = repaired.rstrip(',')
    repaired += ']' * max(0, open_brackets)
    repaired += '}' * max(0, open_braces)
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass
    # Last resort: try to find the last valid closing brace
    for i in range(len(repaired) - 1, 0, -1):
        if repaired[i] == '}':
            candidate = repaired[:i+1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    return raw  # give up, return original


# ── Narrative prompt (RUSSIAN) ───────────────────────────────────────────────

def _user_prompt(summary: dict, *, tier: str, market_context: str = "",
                 user_profile: str = "Moderate") -> str:
    regime = (summary.get("regime") or {})
    regime_label = regime.get("regime", "unknown")

    rag_block = ""
    rag_rule  = ""
    if market_context:
        ctx_limit = 6000 if tier == "deep" else 2000
        rag_block = (
            "\n\n=== АНАЛИТИКА БАНКОВ (RAG) ===\n"
            f"{market_context[:ctx_limit]}\n"
            "=== КОНЕЦ АНАЛИТИКИ БАНКОВ ==="
        )
        rag_rule = (
            "Если используешь факт из АНАЛИТИКИ БАНКОВ — цитируй "
            "[RAG: <имя_файла>]. Не выдумывай файлы.\n"
        )

    # Build list of tickers that are on Sell/Trim in action_plan
    sell_tickers: set[str] = set()
    for ap in (summary.get("action_plan") or []):
        act = str(ap.get("action", "")).lower()
        if act in ("sell", "trim", "сократить", "продать"):
            sell_tickers.add(str(ap.get("ticker", "")).upper())
    # Also check holdings with action Sell/Trim
    for h in (summary.get("holdings") or []):
        act = str(h.get("action", "")).lower()
        if act in ("sell", "trim", "сократить", "продать"):
            sell_tickers.add(str(h.get("ticker", "")).upper())
    
    contradiction_rule = ""
    if sell_tickers:
        contradiction_rule = (
            f"КРИТИЧНО: тикеры {', '.join(sorted(sell_tickers))} имеют статус Sell/Trim в action_plan. "
            "НЕ предлагай их в stock_picks (ни в boost_alpha, ни в rebalance). "
            "Это противоречит сигналам Quant Engine.\n"
        )

    # ── Base tier: compact prompt, 1 pick per scenario, 3 bullets ──
    if tier != "deep":
        return (
            "Проанализируй портфель. Верни СТРОГО JSON (без текста вне JSON).\n"
            "ВСЕ ТЕКСТЫ НА РУССКОМ.\n\n"
            '{\n'
            '  "verdict": "≤150 знаков — вердикт",\n'
            '  "plain_summary": "≤250 знаков простым языком",\n'
            '  "bullets": ["3 пункта ≤120 знаков каждый с [Источник]"],\n'
            '  "ai_cvar_note": "≤120 знаков — простыми словами про CVaR: сколько примерно теряется в худший день",\n'
            '  "ai_sharpe_note": "≤120 знаков — простыми словами про Sharpe: окупается ли риск доходностью",\n'
            '  "ai_mdd_note": "≤120 знаков — простыми словами про макс. просадку портфеля",\n'
            '  "ai_risk_comment": "≤150 знаков — комментарий к риск-метрикам (CVaR, Vol, MaxDD)",\n'
            '  "ai_regime_comment": "≤120 знаков — комментарий к рыночному режиму",\n'
            '  "ai_holdings_comment": "≤150 знаков — ключевые позиции, hotspots, перевесы",\n'
            '  "ai_benchmark_comment": "≤150 знаков — обгоняет ли портфель рынок и почему",\n'
            '  "stock_picks": {\n'
            '    "boost_alpha": {"label": "Повышение доходности", "desc": "≤80 знаков", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤100 знаков [источник]", "type": "Stock"}]},\n'
            '    "rebalance": {"label": "Ребалансировка", "desc": "≤80 знаков", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤100 знаков", "type": "Stock"}]},\n'
            '    "protect_capital": {"label": "Защита капитала", "desc": "≤80 знаков", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤100 знаков", "type": "Stock"}]}\n'
            '  }\n'
            '}\n\n'
            "ПРАВИЛА: русский язык. Без «RAMP». Каждое число — [Quant Engine]/[SEC EDGAR]/[Regime]/[RAG].\n"
            "ПРОСТОЙ ЯЗЫК: пиши для человека без финансового образования, минимум жаргона, "
            "термины поясняй в скобках, связывай мысли логически.\n"
            f"Риск-профиль: {user_profile}. Режим: {regime_label}.\n"
            "Stock picks: РЕАЛЬНЫЕ АКЦИИ (PLTR, JNJ, KO и т.п.), не только ETF. "
            "why ≤100 знаков с цифрами.\n"
            f"{contradiction_rule}"
            f"{rag_rule}\n"
            f"=== ДАННЫЕ ===\n{json.dumps(summary, ensure_ascii=False)[:7000]}"
            f"{rag_block}"
        )

    # ── Deep tier: full prompt with per-section comments ──
    picks_spec = f"""
"stock_picks": {{
  "boost_alpha": {{
    "label": "Повышение доходности", "desc": "≤100 знаков. Режим: {regime_label}.",
    "picks": [  // 2 идеи: реальные акции роста
      {{"ticker": "...", "name": "...", "why": "≤180 знаков с [источник]", "type": "Stock|ETF"}}
    ]
  }},
  "rebalance": {{
    "label": "Ребалансировка", "desc": "≤100 знаков.",
    "picks": [  // 2 quality-акции
      {{"ticker": "...", "name": "...", "why": "≤180 знаков", "type": "Stock|ETF"}}
    ]
  }},
  "protect_capital": {{
    "label": "Защита капитала", "desc": "≤100 знаков.",
    "picks": [  // 1 защитный инструмент
      {{"ticker": "...", "name": "...", "why": "≤180 знаков", "type": "Stock|ETF|Bond"}}
    ]
  }}
}}"""

    return (
        "Проанализируй портфель и верни СТРОГО JSON (без текста вне JSON, без markdown).\n"
        "ВСЕ ТЕКСТЫ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.\n\n"
        '{\n'
        '  "verdict": "≤180 знаков — общий вердикт",\n'
        '  "plain_summary": "≤300 знаков простым языком",\n'
        '  "bullets": ["5–7 пунктов ≤180 знаков каждый с [Источник]"],\n'
        '  "ai_cvar_note": "≤120 знаков — простыми словами про CVaR ЭТОГО портфеля: '
        'примерно сколько денег теряется в худший день и нормально ли это.",\n'
        '  "ai_sharpe_note": "≤120 знаков — простыми словами про Sharpe: окупается ли '
        'риск доходностью.",\n'
        '  "ai_mdd_note": "≤120 знаков — простыми словами про макс. просадку: насколько '
        'глубоко портфель падал и что это значит для владельца.",\n'
        '  "ai_risk_comment": "≤200 знаков — вывод по риск-метрикам: CVaR, Vol, MaxDD, Sharpe. '
        'Сопоставь с лимитами мандата. Упомяни $ потерь.",\n'
        '  "ai_benchmark_comment": "≤200 знаков — обгоняет ли портфель бенчмарк? '
        'Tracking Error vs Volatility — почему разные. IR интерпретация.",\n'
        '  "ai_regime_comment": "≤200 знаков — режим рынка, как он влияет на портфель, '
        'подтверждается ли банковскими отчётами (если есть RAG).",\n'
        '  "ai_holdings_comment": "≤200 знаков — ключевые hotspots, перевесы, '
        'какие позиции увеличивают хвостовой риск.",\n'
        '  "ai_sector_comment": "≤170 знаков — секторные и СУБ-секторные перекосы '
        '(напр. внутри Tech: софт vs полупроводники) и чем это рискованно.",\n'
        '  "ai_factor_comment": "≤200 знаков — какие факторы доминируют и что это '
        'значит; учти что факторы пересекаются (Momentum/Value/Quality — те же акции '
        'под разными углами), поэтому беты не складываются напрямую.",\n'
        '  "ai_stress_comment": "≤200 знаков — стресс-сценарии: худший сценарий для портфеля, '
        'размер просадки и насколько портфель к нему устойчив.",\n'
        '  "ai_effect_comment": "≤200 знаков — ожидаемый эффект ребалансировки: '
        'как меняются риск/доходность/Sharpe до→после.",\n'
        f'{picks_spec},\n'
        '  "action_plan_text": "≤800 знаков — приоритетные действия (Trim/Sell сначала)",\n'
        '  "ai_action_impact": "≤300 знаков — как изменятся CVaR/Vol/TE при реализации"\n'
        '}\n\n'
        "ПРАВИЛА:\n"
        "- ВСЕ тексты на РУССКОМ. Без «RAMP».\n"
        "- ПРОСТОЙ ЯЗЫК: пиши как другу без финансового образования. Минимум "
        "жаргона; если термин необходим — поясни его в скобках простыми словами. "
        "Связывай мысли логически (причина → следствие → что делать).\n"
        "- Каждое число — [Quant Engine], [SEC EDGAR], [Regime] или [RAG: файл].\n"
        f"- Риск-профиль: {user_profile}.\n"
        "- РЕАЛЬНЫЕ АКЦИИ: PLTR, CRWD, JNJ, COST, KO и т.п. — не только ETF.\n"
        "- why: конкретные цифры (ROE, маржа, Beta, P/E) с [источник].\n"
        f"{contradiction_rule}"
        f"{rag_rule}"
        f"\n=== ДАННЫЕ ПОРТФЕЛЯ ===\n{json.dumps(summary, ensure_ascii=False)[:9000]}"
        f"{rag_block}"
    )


# ── Fallback narrative (no API key / failure path) — RUSSIAN ─────────────────

def _fallback_narrative(results: dict, tier: str) -> dict:
    metrics = results.get("portfolio_metrics") or {}
    regime  = results.get("regime") or {}
    perf    = results.get("performance_table")

    composite  = metrics.get("Composite_Risk_Score") or 0
    risk_label = ("консервативный" if composite < 33 else
                  "умеренный"      if composite < 66 else "агрессивный")
    sharpe     = metrics.get("Sharpe_Ratio")
    s_text     = (f"Sharpe {sharpe:.2f}"
                  if isinstance(sharpe, (int, float)) and sharpe == sharpe else "Sharpe н/д")
    verdict    = (f"Профиль риска: {risk_label} (композит {composite}/100). "
                  f"{s_text}. Режим рынка: {regime.get('regime', 'н/д')} [Regime].")

    bullets: list[str] = []
    if perf is not None and not perf.empty and "Score_Hotspot" in perf.columns:
        hot = perf[perf["Score_Hotspot"] == True]   # noqa: E712
        if not hot.empty:
            t   = str(hot.iloc[0].get("Ticker"))
            trc = float(hot.iloc[0].get("Euler_Risk_Contribution_Pct") or 0)
            bullets.append(f"🔥 Позиция {t} генерирует {trc:.1f}% общего риска портфеля — "
                           f"рекомендуется частичное сокращение [Quant Engine]")
    cvar = metrics.get("CVaR_95_Daily")
    total_val = results.get("total_value") or 0
    if cvar is not None:
        cvar_dollar = abs(cvar) * total_val if total_val else 0
        bullets.append(f"Ожидаемый убыток в худшие 5% дней: {cvar*100:.1f}% "
                       f"(≈${cvar_dollar:,.0f}) [Quant Engine]")
    mdd = metrics.get("Max_Drawdown")
    if mdd is not None:
        mdd_dollar = abs(mdd) * total_val if total_val else 0
        bullets.append(f"Историческая макс. просадка: {mdd*100:.1f}% "
                       f"(≈${mdd_dollar:,.0f}) [Quant Engine]")
    if regime.get("regime"):
        bullets.append(f"Макро-режим: {regime['regime']} "
                       f"(уверенность {int(round(regime.get('confidence', 0)*100))}%) "
                       f"учтён в скоринге [Regime]")
    if tier == "deep":
        bullets.append("Используйте Action Plan: уровни Buy/Sell/Stop рассчитаны "
                       "из ATR (Wilder RMA) и SMA200 [Quant Engine]")

    plain_summary = (
        f"Ваш портфель в {risk_label} зоне риска ({composite}/100). "
        + (f"Доходность с учётом риска (Sharpe) = {sharpe:.2f} — "
           + ("положительная, это хороший знак." if sharpe > 0 else "отрицательная, будьте осторожны.")
           if isinstance(sharpe, (int, float)) and sharpe == sharpe else "")
        + (" Проверьте Action Plan и сократите позиции-hotspots." if tier == "deep" else "")
    )

    regime_label = regime.get("regime", "")
    stock_picks  = _fallback_stock_picks(regime_label, tier)

    action_plan_text = ""
    ai_action_impact = ""
    if tier == "deep":
        action_plan_text = (
            "Приоритет — закрытие концентрационных hotspots (TRC > 20%); "
            "затем точечные покупки по Buy-зонам в активах с Total Score > +1. "
            "Все сделки в пределах 25% оборота NAV [Quant Engine]."
        )
        ai_action_impact = (
            "Сокращение крупнейшего hotspot на 30% снизит Vol на ~1-2 п.п. "
            "и улучшит CVaR на ~0.2 п.п. Добавление защитного ETF (AGG/GLD) "
            "снизит Tracking Error к бенчмарку [Quant Engine]."
        )

    return {
        "verdict":          verdict,
        "plain_summary":    plain_summary[:300],
        "bullets":          bullets[:7 if tier == "deep" else 4],
        "action_plan_text": action_plan_text,
        "ai_action_impact": ai_action_impact,
        "stock_picks":      stock_picks,
        "used_rag":         False,
        "model_used":       "fallback",
    }


def _fallback_stock_picks(regime_label: str, tier: str) -> dict:
    """Rule-based stock picks when Claude is unavailable — includes real stocks."""
    expansion = regime_label in ("Recovery", "Expansion", "")

    if expansion:
        picks_boost = [
            {"ticker": "PLTR", "name": "Palantir Technologies",
             "why": "Рост выручки >30% г/г, высокий momentum 12м, "
                    "бета к рынку ~1.8 — подходит для режима Expansion [SEC EDGAR] [Regime]",
             "type": "Stock"},
        ]
    else:
        picks_boost = [
            {"ticker": "DBC", "name": "Invesco DB Commodity Index ETF",
             "why": "Товарная экспозиция защищает при стагфляции и Slowdown-режимах. "
                    "Низкая корреляция с техно-позициями [Quant Engine] [Regime]",
             "type": "ETF"},
        ]

    picks_balance = [
        {"ticker": "JNJ", "name": "Johnson & Johnson",
         "why": "ROE ~25%, Op. маржа ~25%, Debt/Assets <0.4, "
                "дивиденд-аристократ 62 года подряд — режимо-устойчивый [SEC EDGAR]",
         "type": "Stock"},
    ]

    picks_protect = [
        {"ticker": "KO", "name": "The Coca-Cola Company",
         "why": "Дивиденд-аристократ 61+ лет, Beta ~0.6, "
                "стабильная маржа >28% — защитный актив при любом режиме [SEC EDGAR] [Quant Engine]",
         "type": "Stock"},
    ]

    if tier == "deep":
        if expansion:
            picks_boost.append(
                {"ticker": "CRWD", "name": "CrowdStrike Holdings",
                 "why": "Лидер кибербезопасности, выручка +33% г/г, "
                        "переход на прибыльность, momentum 12м +45% [SEC EDGAR] [Quant Engine]",
                 "type": "Stock"}
            )
        else:
            picks_boost.append(
                {"ticker": "GLD", "name": "SPDR Gold Shares",
                 "why": "Золото как хедж в Slowdown/Recession, "
                        "нулевая корреляция с S&P 500 [Quant Engine] [Regime]",
                 "type": "ETF"}
            )
        picks_balance.append(
            {"ticker": "COST", "name": "Costco Wholesale",
             "why": "ROE ~30%, Op. маржа ~3.5% (стабильная бизнес-модель), "
                    "Revenue Growth ~7% г/г, низкий долг [SEC EDGAR]",
             "type": "Stock"}
        )

    return {
        "boost_alpha":     {"label": "Повышение доходности — повышенный риск",
                            "desc":  "Увеличение потенциальной доходности за счёт акций роста и momentum.",
                            "picks": picks_boost},
        "rebalance":       {"label": "Качественная ребалансировка — сохранение профиля",
                            "desc":  "Улучшение качества портфеля без изменения общего риска.",
                            "picks": picks_balance},
        "protect_capital": {"label": "Защита капитала — снижение риска",
                            "desc":  "Защитное позиционирование при макроэкономической неопределённости.",
                            "picks": picks_protect},
    }


# ── Public API ───────────────────────────────────────────────────────────────

def generate_narrative(results: dict, tier: str = "base",
                       market_context: str = "",
                       user_risk_profile: str = "Moderate") -> dict:
    """
    Returns {verdict, plain_summary, bullets, stock_picks,
             action_plan_text, ai_action_impact, used_rag}.

    Base tier  → Claude Haiku  (1200 tokens, fast)
    Deep tier  → Claude Sonnet (6000 tokens, richer prose + RAG synthesis)
    Falls back deterministically when the API is unavailable or fails.
    All output is in Russian.
    """
    api_key  = os.getenv("ANTHROPIC_API_KEY")
    used_rag = bool(market_context) and tier == "deep"

    if not api_key:
        logger.warning("ANTHROPIC_API_KEY отсутствует — используется fallback-нарратив.")
        out = _fallback_narrative(results, tier)
        out["used_rag"] = False
        out["model_used"] = "fallback"
        return out

    summary = _summarise_for_prompt(results)
    model   = MODEL_DEEP if tier == "deep" else MODEL_BASE
    max_tok = MAX_TOKENS_DEEP if tier == "deep" else MAX_TOKENS_BASE

    logger.info("AI narrative: запрос к %s (tier=%s, max_tokens=%d, rag=%s)",
                model, tier, max_tok, used_rag)
    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model       = model,
            max_tokens  = max_tok,
            temperature = 0.1,
            system      = _build_system_prompt(),
            messages    = [{
                "role": "user",
                "content": _user_prompt(summary, tier=tier,
                                        market_context=market_context,
                                        user_profile=user_risk_profile),
            }],
        )
        usage    = response.usage
        raw      = response.content[0].text.strip()
        logger.info("AI narrative: ответ от %s получен (input=%d tok, output=%d tok, len=%d chars)",
                    model, usage.input_tokens, usage.output_tokens, len(raw))

        first_brace = raw.find("{")
        if first_brace == -1:
            raise ValueError(f"JSON-объект не найден в ответе (raw[:200]={raw[:200]!r})")
        last_brace = raw.rfind("}")
        # Truncation by max_tokens can swallow the closing brace entirely —
        # feed everything from the first '{' to the end into the repair pass.
        json_str = raw[first_brace:last_brace + 1] if last_brace > first_brace else raw[first_brace:]
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            logger.info("AI narrative: JSON невалиден, пробуем repair (output=%d tok, "
                        "closing_brace_found=%s)",
                        usage.output_tokens, last_brace > first_brace)
            repaired = _repair_truncated_json(json_str)
            try:
                parsed = json.loads(repaired)
                logger.info("AI narrative: JSON repair УСПЕШЕН")
            except json.JSONDecodeError as je:
                raise ValueError(
                    f"JSON repair не удался (output={usage.output_tokens} tok, "
                    f"raw_tail={raw[-200:]!r}): {je}"
                ) from je

        verdict       = str(parsed.get("verdict", "")).strip()
        plain_summary = str(parsed.get("plain_summary", "")).strip()
        bullets       = [str(b).strip() for b in (parsed.get("bullets") or []) if str(b).strip()]
        plan_txt      = str(parsed.get("action_plan_text", "")).strip()
        impact_txt    = str(parsed.get("ai_action_impact", "")).strip()
        stock_picks   = parsed.get("stock_picks") or {}

        if not verdict or not bullets:
            raise ValueError(
                f"verdict или bullets пусты (verdict={verdict!r}, bullets_count={len(bullets)})"
            )

        # CoVe: strip RAG citations whose source file is not in market_context.
        bullets       = [_strip_unverified_rag_citations(b, market_context) for b in bullets]
        plan_txt      = _strip_unverified_rag_citations(plan_txt, market_context)
        verdict       = _strip_unverified_rag_citations(verdict, market_context)
        plain_summary = _strip_unverified_rag_citations(plain_summary, market_context)
        impact_txt    = _strip_unverified_rag_citations(impact_txt, market_context)

        # Normalise stock_picks structure.
        stock_picks = _normalise_stock_picks(stock_picks, tier, market_context)
        # Post-generation contradiction check: remove Sell/Trim tickers from buy picks.
        stock_picks = _check_pick_contradictions(stock_picks, results)
        total_picks = sum(len(s.get("picks", [])) for s in stock_picks.values())

        logger.info("AI narrative: SUCCESS model=%s verdict=%d chars bullets=%d picks=%d",
                    model, len(verdict), len(bullets), total_picks)

        # Extract per-section AI comments.  Strip unverified RAG citations so
        # a comment never references a bank report that wasn't retrieved.
        def _comment(key: str, limit: int = 250) -> str:
            txt = str(parsed.get(key, "")).strip()
            return _strip_unverified_rag_citations(txt, market_context)[:limit]

        return {
            "verdict":              verdict[:300],
            "plain_summary":        plain_summary[:400],
            "bullets":              bullets[:7 if tier == "deep" else 4],
            "action_plan_text":     plan_txt[:1000] if tier == "deep" else "",
            "ai_action_impact":     impact_txt[:400] if tier == "deep" else "",
            "stock_picks":          stock_picks,
            "used_rag":             used_rag,
            "model_used":           model,
            "ai_cvar_note":         _comment("ai_cvar_note", 160),
            "ai_sharpe_note":       _comment("ai_sharpe_note", 160),
            "ai_mdd_note":          _comment("ai_mdd_note", 160),
            "ai_risk_comment":      _comment("ai_risk_comment"),
            "ai_benchmark_comment": _comment("ai_benchmark_comment"),
            "ai_regime_comment":    _comment("ai_regime_comment"),
            "ai_holdings_comment":  _comment("ai_holdings_comment"),
            "ai_sector_comment":    _comment("ai_sector_comment", 200),
            "ai_factor_comment":    _comment("ai_factor_comment"),
            "ai_stress_comment":    _comment("ai_stress_comment"),
            "ai_effect_comment":    _comment("ai_effect_comment"),
        }
    except Exception as exc:
        logger.warning("AI narrative FAILED (%s) — используется fallback. Модель: %s",
                       exc, model, exc_info=True)
        out = _fallback_narrative(results, tier)
        out["used_rag"]   = False
        out["model_used"] = "fallback"
        return out


def _normalise_stock_picks(raw: dict, tier: str, market_context: str) -> dict:
    """
    Ensure stock_picks has exactly the three scenario keys.
    Strips unverified RAG citations from pick rationale.
    """
    result: dict = {}
    for key in ("boost_alpha", "rebalance", "protect_capital"):
        scenario = raw.get(key) or {}
        picks    = scenario.get("picks") or []
        clean_picks = []
        for p in picks:
            why = _strip_unverified_rag_citations(str(p.get("why", "")), market_context)
            clean_picks.append({
                "ticker": str(p.get("ticker", "")).upper()[:10],
                "name":   str(p.get("name", ""))[:80],
                "why":    why[:220],
                "type":   str(p.get("type", "Stock"))[:10],
            })
        result[key] = {
            "label": str(scenario.get("label", key))[:80],
            "desc":  str(scenario.get("desc", ""))[:200],
            "picks": clean_picks,
        }
    return result


def _check_pick_contradictions(stock_picks: dict, results: dict) -> dict:
    """
    Post-generation sanity check: remove picks that contradict the action_plan.
    If action_plan says Sell/Trim for a ticker, it should NOT appear in
    boost_alpha or rebalance picks (but protect_capital is OK for hedging).
    """
    sell_tickers: set[str] = set()
    # From action_plan
    for ap in (results.get("action_plan") or []):
        act = str(ap.get("action", "")).lower()
        if act in ("sell", "trim", "сократить", "продать"):
            sell_tickers.add(str(ap.get("ticker", "")).upper().split(".")[0])
    # From performance table
    perf = results.get("performance_table")
    if perf is not None and not perf.empty and "Score_Action" in perf.columns:
        for _, row in perf.iterrows():
            act = str(row.get("Score_Action", "")).lower()
            if act in ("sell", "trim"):
                sell_tickers.add(str(row.get("Ticker", "")).upper().split(".")[0])

    if not sell_tickers:
        return stock_picks

    for scenario_key in ("boost_alpha", "rebalance"):
        scenario = stock_picks.get(scenario_key)
        if not scenario:
            continue
        original = scenario.get("picks", [])
        filtered = [p for p in original
                    if p.get("ticker", "").upper().split(".")[0] not in sell_tickers]
        if len(filtered) < len(original):
            removed = [p["ticker"] for p in original if p not in filtered]
            logger.info("Contradiction check: removed %s from %s (Sell/Trim in action_plan)",
                        removed, scenario_key)
        scenario["picks"] = filtered

    return stock_picks


__all__ = ["generate_narrative"]
