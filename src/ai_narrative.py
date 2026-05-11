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

MAX_TOKENS_BASE = 1_200
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


# ── Narrative prompt (RUSSIAN) ───────────────────────────────────────────────

def _user_prompt(summary: dict, *, tier: str, market_context: str = "",
                 user_profile: str = "Moderate") -> str:
    regime = (summary.get("regime") or {})
    regime_label = regime.get("regime", "unknown")
    n_bullets = "5–7" if tier == "deep" else "3–4"
    n_picks   = 5 if tier == "deep" else 3

    ask_plan = ""
    if tier == "deep":
        ask_plan = (
            'Также выведи ключ "action_plan_text": блок ≤800 знаков с 3-4 предложениями '
            'про самые приоритетные действия (Trim/Sell сначала). '
            'Ссылайся на action_plan из summary. Укажи конкретные тикеры и цены.\n'
            'Также выведи ключ "ai_action_impact": 2-3 предложения ≤300 знаков — '
            'как изменятся CVaR, Vol и Tracking Error портфеля, если реализовать '
            'рекомендации из stock_picks и action_plan. Пример: '
            '"Добавление QUAL (5%) снизит Vol с 22% до ~20%, CVaR улучшится на 0.3 п.п. '
            '[Quant Engine]"'
        )

    rag_block = ""
    rag_rule  = ""
    if tier == "deep" and market_context:
        rag_block = (
            "\n\n=== АНАЛИТИКА БАНКОВ (RAG) ===\n"
            f"{market_context[:6000]}\n"
            "=== КОНЕЦ АНАЛИТИКИ БАНКОВ ==="
        )
        rag_rule = (
            "Если используешь факт из АНАЛИТИКИ БАНКОВ — ОБЯЗАТЕЛЬНО цитируй "
            "источник как [RAG: <имя_файла>] ровно в том виде, в каком имя файла "
            "появляется в блоке выше. Не выдумывай файлы и числа, которых нет "
            "ни в summary, ни в RAG-блоке.\n"
        )

    picks_per_scenario = 2 if tier == "deep" else 1
    scenarios_spec = f"""
"stock_picks": {{
  "boost_alpha": {{
    "label": "Повышение доходности — повышенный риск",
    "desc": "Увеличение потенциальной доходности. Текущий режим: {regime_label}.",
    "picks": [  // {picks_per_scenario} идей: РЕАЛЬНЫЕ АКЦИИ (не только ETF!) — акции роста, small-cap momentum, крипто, сырьё
      {{"ticker": "...", "name": "Полное название компании", "why": "Развёрнутое обоснование ≤200 знаков с [источник]", "type": "Stock|ETF|Crypto"}}
    ]
  }},
  "rebalance": {{
    "label": "Качественная ребалансировка — сохранение профиля",
    "desc": "Улучшение качества портфеля без изменения общего уровня риска.",
    "picks": [  // {picks_per_scenario} идей: quality-акции (JNJ, COST, V, MA, PG), factor-ETF
      {{"ticker": "...", "name": "...", "why": "...", "type": "Stock|ETF"}}
    ]
  }},
  "protect_capital": {{
    "label": "Защита капитала — снижение риска",
    "desc": "Защитное позиционирование при смене режима или росте хвостовых рисков.",
    "picks": [  // 1 идея: дивидендные аристократы (KO, PEP), облигации, золото
      {{"ticker": "...", "name": "...", "why": "...", "type": "Stock|ETF|Bond"}}
    ]
  }}
}}"""

    return (
        "Проанализируй портфель и верни СТРОГО JSON (без текста вне JSON, без markdown).\n"
        "ВСЕ ТЕКСТЫ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.\n\n"
        "{\n"
        '  "verdict": "1 предложение ≤180 знаков — общий вердикт по портфелю",\n'
        '  "plain_summary": "2-3 предложения простым языком для обычного инвестора: '
        'текущее состояние, главный риск, одно конкретное действие. ≤300 знаков.",\n'
        f'  "bullets": [{n_bullets} пунктов, каждый ≤200 знаков, каждый с тегом [Источник]],\n'
        f"{scenarios_spec},\n"
        '  "action_plan_text": "..."  // только для deep tier\n'
        '  "ai_action_impact": "..."  // только для deep tier\n'
        "}\n\n"
        "ПРАВИЛА:\n"
        "- ВСЕ тексты на РУССКОМ языке.\n"
        "- Слово «RAMP» нигде не использовать.\n"
        "- Каждое число должно иметь тег источника: [Quant Engine], [SEC EDGAR], [CDS], [Regime] или [RAG: файл].\n"
        "- Stock picks должны соответствовать риск-профилю пользователя: " + user_profile + ".\n"
        "- КРИТИЧНО: предлагай РЕАЛЬНЫЕ АКЦИИ (Stock), а не только ETF. "
        "Для каждой акции: объясни ПОЧЕМУ именно она — ссылайся на MAC3 факторы "
        "(Beta, TRC), SEC EDGAR данные (ROE, маржа, долг) и режим рынка.\n"
        "- Для Boost Alpha: акции роста с высоким momentum, которых НЕТ в портфеле. "
        "Примеры: PLTR, COIN, RKLB, CELH, CRWD, ANET, PANW — или другие, "
        "подходящие текущему режиму.\n"
        "- Для Rebalance: quality-акции с устойчивой маржой и низким долгом. "
        "Примеры: JNJ, PG, COST, V, MA, UNH, HD — или другие с высоким Fundamental Score.\n"
        "- Для Protect Capital: дивидендные аристократы или защитные активы. "
        "Примеры: KO, PEP, JNJ, XLU, GLD, TLT.\n"
        "- В поле 'why' обязательно укажи конкретные цифры: ROE, маржа, Beta, "
        "momentum, P/E — с тегом источника.\n"
        "- CoVe: перед финализацией проверь, что каждый тикер существует, "
        "каждый факт подкреплён источником, ни одно число не выдумано.\n"
        f"{rag_rule}"
        f"\n{ask_plan}\n\n"
        f"=== ДАННЫЕ ПОРТФЕЛЯ ===\n{json.dumps(summary, ensure_ascii=False)[:9000]}"
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
        last_brace  = raw.rfind("}")
        if first_brace == -1 or last_brace == -1:
            raise ValueError(f"JSON-объект не найден в ответе (raw[:200]={raw[:200]!r})")
        parsed = json.loads(raw[first_brace:last_brace + 1])

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
        total_picks = sum(len(s.get("picks", [])) for s in stock_picks.values())

        logger.info("AI narrative: SUCCESS model=%s verdict=%d chars bullets=%d picks=%d",
                    model, len(verdict), len(bullets), total_picks)

        return {
            "verdict":          verdict[:300],
            "plain_summary":    plain_summary[:400],
            "bullets":          bullets[:7 if tier == "deep" else 4],
            "action_plan_text": plan_txt[:1000] if tier == "deep" else "",
            "ai_action_impact": impact_txt[:400] if tier == "deep" else "",
            "stock_picks":      stock_picks,
            "used_rag":         used_rag,
            "model_used":       model,
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


__all__ = ["generate_narrative"]
