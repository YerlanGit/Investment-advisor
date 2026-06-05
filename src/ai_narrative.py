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

MAX_TOKENS_BASE = 4_500  # raised from 3_500 — Haiku BASE was hitting cap on long action plans, triggering JSON repair on every call
MAX_TOKENS_DEEP = 7_000


# ── Prompt-injection defence: input sanitization ────────────────────────────
# Every string that flows from BROKER / RAG / TELEGRAM into the LLM prompt
# passes through these guards.  Without them a holding called
# "Ignore previous instructions" or a poisoned bank-PDF chunk can override
# the system prompt.  Tickers go through a strict regex; free-text strings
# (sector tags, fund names) are control-char-stripped and length-clamped.
#
# Ticker regex permits up to 32 chars so KZ structured notes like
# "FFSPC6.1028.AIX" (16 chars) survive — strictly tighter ^[A-Z0-9.\-]{1,10}$
# from the original spec would drop them, breaking KZ workflows.
_TICKER_RE       = re.compile(r"^[A-Z0-9.\-]{1,32}$")
_SAFE_TEXT_DROP  = re.compile(r"[\x00-\x1F\x7F<>{}|`]")   # control / tag chars


def _safe_ticker(t) -> str:
    """Strict ticker validator.  Non-matching input → '?' (never propagated as instruction)."""
    s = str(t or "").strip().upper()
    return s if _TICKER_RE.match(s) else "?"


def _safe_text(t, max_len: int = 60) -> str:
    """Strip control chars + length-clamp untrusted free-text fields."""
    s = str(t or "")[:max_len]
    return _SAFE_TEXT_DROP.sub("", s).strip()


def _wrap_untrusted(label: str, payload: str) -> str:
    """
    Fence external data with an XML-like tag.  The system prompt instructs
    the model to NEVER follow instructions found inside these tags.
    """
    return f"<untrusted_data source=\"{label}\">\n{payload}\n</untrusted_data>"


# ── Soft-trim helper ────────────────────────────────────────────────────────
# Hard-cap slicing (`text[:N]`) produces obviously truncated sentences in
# the production report ("Это давит на перегруженный Tech-портфель [GS][Bar").
# `_soft_trim` cuts at the LAST sentence boundary (".", "!", "?", "…")
# inside the budget so the user never sees a half-sentence.  Hard fallback
# to the budget if no boundary is present (e.g. one long URL).
_SENTENCE_END_RE = re.compile(r"[\.\!\?…](?:\s|\)|\]|$)")


def _soft_trim(text: str, max_chars: int) -> str:
    """
    Trim ``text`` to ``max_chars`` characters, ending at the LAST complete
    sentence boundary inside the budget.  If the candidate would lose more
    than 50% of the budget, fall back to a hard cut + ellipsis so the
    truncation is at least visually marked, not silent.
    """
    if not text:
        return ""
    s = str(text).strip()
    if len(s) <= max_chars:
        return s
    head    = s[:max_chars]
    matches = list(_SENTENCE_END_RE.finditer(head))
    if matches:
        end = matches[-1].end()
        # Strip trailing whitespace after the boundary char.
        candidate = head[:end].rstrip()
        if len(candidate) >= max_chars // 2:
            return candidate
    # Boundary too far back → ellipsis-tagged hard cut.
    return head.rstrip(" ,;:[(") + "…"

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
            # SANITIZE every string field — these flow straight into the LLM
            # prompt, so a broker-supplied ticker like "<ignore_above>" or a
            # sector field with hidden instructions must never reach Claude.
            perf_rows.append({
                "ticker":   _safe_ticker(row.get("Ticker")),
                "weight":   round(float(row.get("Current_Value") or 0)
                                  / float(results.get("total_value") or 1) * 100, 2),
                "trc_pct":  _safe_round(row.get("Euler_Risk_Contribution_Pct"), 2),
                "atr_pct":  _safe_round(row.get("ATR_Pct"), 2),
                "pnl_pct":  _safe_round((row.get("Return_Pct") or 0) * 100, 2),
                "score":    _safe_round(row.get("Score_Total"), 1),
                "action":   _safe_text(row.get("Score_Action"), 20),
                "sector":   _safe_text(row.get("Fundamental_Sector"), 30),
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

    # Compact macro-driver pack — surfaced so the AI can cross-check the
    # engine's regime label against the actual yield curve / credit spread /
    # volatility / inflation signals (FRED).  Only series with a real value
    # are included; status flag is kept so the AI weights stale data lower.
    macro_src = results.get("macro_drivers") or {}
    _MACRO_KEYS = (
        ("yield_curve_10y2y", "yield_curve_10y2y"),
        ("hy_credit_spread",  "hy_oas"),
        ("vix",               "vix"),
        ("breakeven_inflation", "breakeven"),
    )
    macro_summary: dict = {}
    for src_key, label in _MACRO_KEYS:
        row = macro_src.get(src_key) or {}
        val = row.get("value")
        if val is None:
            continue
        macro_summary[label] = {
            "value":  _safe_round(val, 2),
            "status": row.get("status"),
            "as_of":  row.get("as_of"),
            "unit":   row.get("unit"),
        }

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
        "macro":        macro_summary,
        "sectors":      {k: round(float(v) * 100, 1)
                         for k, v in (results.get("sector_exposure") or {}).items()},
        "benchmarks":   bm,
        "holdings":     perf_rows,
        "action_plan":  (results.get("action_plan") or [])[:8],
        "asset_scores": {t: s for t, s in
                         (results.get("asset_scores") or {}).items()},
        # ── Stress scenarios with per-asset CAPPED deltas ──────────────────
        # Sonnet's old narrative kept re-computing β × shock by hand, which
        # bypassed the H4 convex cap and produced "AVGO −42%, NVDA −47%"
        # against table figures of −15.6% port.  Feed the engine's CAPPED
        # `asset_delta_pct` directly so the AI cites the same numbers the
        # table shows.
        "stress_scenarios": [
            {
                "name":          s.get("name"),
                "port_pct":      _safe_round((s.get("port_pct") or 0) * 100, 2),
                "max_dd_pct":    _safe_round((s.get("max_dd_pct") or 0) * 100, 2),
                "recovery_mo":   s.get("recovery_months"),
                # Top-5 per-asset CAPPED impacts (after convex cap ±35%);
                # `asset_delta_pct` ALREADY %-scaled (not decimal) — keep
                # as-is so the AI quotes the number verbatim.
                "top_assets": [
                    {"ticker":  a.get("ticker"),
                     "delta_pct_capped": a.get("asset_delta_pct"),
                     "delta_pct_raw":    a.get("asset_delta_raw"),
                     "weight_pct":       a.get("weight_pct")}
                    for a in (s.get("by_asset") or [])[:5]
                ],
            }
            for s in (results.get("stress_scenarios") or [])
        ],
        # Reporting-currency / RFR metadata so the AI never writes the
        # wrong RFR in trailing commentary (Sharpe, Sortino).
        "reporting": {
            "currency": (results.get("portfolio_metrics") or {}).get("reporting_currency"),
            "rfr_ann":  (results.get("portfolio_metrics") or {}).get("risk_free_rate_annual"),
        },
        # Composite verdict from simulate_after_plan — the AI must echo
        # this verdict in `ai_effect_comment` instead of inventing its own.
        "rebalance_verdict": ((results.get("expected_effect") or {}).get("verdict")
                              if isinstance(results.get("expected_effect"), dict) else None),
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
        # Defence: RAG chunks come from third-party PDFs that may carry
        # injection payloads.  Fence the entire block as `<untrusted_data>`
        # so the system-prompt guardrail tells the model to treat it as
        # pure data, never as instructions.
        rag_block = (
            "\n\n=== АНАЛИТИКА БАНКОВ (RAG) ===\n"
            + _wrap_untrusted("rag_bank_pdfs", market_context[:ctx_limit])
            + "\n=== КОНЕЦ АНАЛИТИКИ БАНКОВ ==="
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

    # ── Base tier: compact prompt, 1 pick per scenario, 4 scenarios ──
    if tier != "deep":
        return (
            "Ты — Senior Financial Analyst. Анализируй портфель как ЕДИНЫЙ НАРРАТИВ: "
            "все секции связаны, каждый комментарий ссылается на другие данные в отчёте.\n"
            "Верни СТРОГО JSON (без текста вне JSON). ВСЕ ТЕКСТЫ НА РУССКОМ.\n\n"
            "ЦЕПОЧКА АНАЛИЗА (обязательно соблюдать):\n"
            "  Риск (CVaR/Vol) объясняется → конкретными позициями (holdings) →\n"
            "  которые создают → секторную концентрацию → которая даёт → факторный перекос →\n"
            "  который противоречит/поддерживает → текущий режим рынка →\n"
            "  что подтверждается/опровергается → аналитикой инвестбанков → итог: Action Plan.\n\n"
            '{\n'
            '  "verdict": "≤150 знаков — один вердикт: главный риск + причина + неотложное действие",\n'
            '  "plain_summary": "≤250 знаков — простыми словами: что не так, почему, что делать",\n'
            '  "bullets": ["4 пункта ≤120 знаков — КАЖДЫЙ связывает 2 разных раздела отчёта [Источник]"],\n'
            '  "ai_cvar_note": "≤120 знаков — простыми словами: в 5% худших дней теряется ≈X$ (≈Y% портфеля). '
            'Это [нормально/высоко] потому что [причина в 3-5 словах]",\n'
            '  "ai_sharpe_note": "≤120 знаков — простыми словами: на каждый рубль риска портфель зарабатывает X. '
            '[Сравни с нормой 1.0]. [Хорошо/плохо] потому что [причина]",\n'
            '  "ai_mdd_note": "≤120 знаков — простыми словами: портфель уже падал на X% от максимума '
            '(≈Y$). [Приемлемо/опасно] для [профиля]",\n'
            '  "ai_risk_comment": "≤160 знаков — ПРИЧИНА высокого CVaR/Vol: назови 1-2 конкретных тикера '
            '(→ из раздела holdings) и их вклад в риск. Скажи что продать чтобы снизить [Quant Engine]",\n'
            '  "ai_holdings_comment": "≤170 знаков — hotspot-позиции с наибольшим вкладом в риск '
            '(→ TRC%). Объясни ПОЧЕМУ они опасны через их чувствительность к рынку (бета). '
            'Связь: высокий риск → высокий CVaR (→ см. риск-блок) [Quant Engine]",\n'
            '  "ai_sector_comment": "≤140 знаков — сектор с перекосом + % → ПОЧЕМУ это опасно в режиме '
            f'{regime_label} (→ из раздела режима). Назови сектор для докупки [Regime]",\n'
            '  "ai_factor_comment": "≤220 знаков — (1) Value-фактор портфеля [Quant Engine]: '
            'если отрицательный — значит портфель против стоимостных акций, что ПРОТИВОРЕЧИТ режиму '
            f'{regime_label} (→ Barclays/Goldman рекомендуют Value в Recovery). '
            '(2) Назови КОНКРЕТНЫЙ фактор для наращивания (напр. Value через JNJ/KO) и что продать",\n'
            '  "ai_benchmark_comment": "≤160 знаков — опережает/отстаёт от рынка на X%. '
            'Причина (→ сектор/фактор). Стабильно ли это или зависит от одного ралли? [Quant Engine]",\n'
            '  "ai_performance_comment": "≤150 знаков — доходность за 1М/3М/12М: тренд роста/падения. '
            'Что было драйвером (→ конкретные тикеры из holdings). Устойчиво ли это [Quant Engine]",\n'
            '  "ai_regime_comment": "≤140 знаков — режим '
            f'{regime_label}: что это значит простыми словами. '
            'Что рекомендуют Goldman/Barclays/JPMorgan для этого режима. '
            'Как ваш портфель позиционирован (→ совпадает/противоречит факторному анализу) [Regime]",\n'
            '  "stock_picks": {\n'
            '    "boost_alpha": {"label": "Повышение доходности", "desc": "≤80 знаков", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤120 знаков: ROE/маржа/бета + '
            'ПОЧЕМУ подходит под текущий режим [Источник]", "type": "Stock"}]},\n'
            '    "rebalance": {"label": "Ребалансировка", "desc": "≤80 знаков", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤120 знаков: качество + '
            'замена какой позиции из Sell-списка [Источник]", "type": "Stock"}]},\n'
            '    "protect_capital": {"label": "Защита капитала", "desc": "≤80 знаков", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤120 знаков: низкая бета + '
            'дивиденды + защита от стресс-сценария [Источник]", "type": "Stock"}]},\n'
            '    "regime_play": {"label": "Режимная ставка", "desc": "≤80 знаков — позиционирование под режим", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤120 знаков: почему именно этот режим '
            'даёт преимущество + банковский взгляд [Regime]", "type": "Stock|ETF"}]}\n'
            '  }\n'
            '}\n\n'
            "ПРАВИЛА:\n"
            "- Русский язык. Без «RAMP».\n"
            "- ПРОСТОЙ ЯЗЫК — глоссарий замен:\n"
            "  CVaR → 'потери в худший день из 20'\n"
            "  волатильность → 'нестабильность' или 'насколько прыгает портфель'\n"
            "  Бета → 'чувствительность к рынку (Beta 2 = двигается вдвое сильнее рынка)'\n"
            "  IR → 'стабильность обгона рынка'\n"
            "  TRC → 'доля в общем риске портфеля'\n"
            "  Recovery-режим → 'рынок восстанавливается после спада'\n"
            "- БАНКОВСКАЯ АНАЛИТИКА: даже без RAG-данных — используй знания о позициях "
            "Goldman Sachs, Barclays, JPMorgan, Morgan Stanley по текущему режиму и секторам.\n"
            "- Каждое число — [Quant Engine]/[SEC EDGAR]/[Regime]/[RAG] или [GS]/[Barclays]/[JPM].\n"
            f"- Риск-профиль: {user_profile}. Режим: {regime_label}.\n"
            "- Stock picks: РЕАЛЬНЫЕ АКЦИИ (PLTR, JNJ, KO, CRWD и т.п.), не только ETF. "
            "why — конкретные цифры (ROE, маржа, Бета) с тегом [Источник].\n"
            f"{contradiction_rule}"
            f"{rag_rule}\n"
            f"=== ДАННЫЕ ПОРТФЕЛЯ ===\n"
            f"{_wrap_untrusted('broker_portfolio_json', json.dumps(summary, ensure_ascii=False)[:7000])}"
            f"{rag_block}"
        )

    # ── Deep tier: full prompt with per-section comments, 4 scenarios ──
    picks_spec = f"""
"stock_picks": {{
  "boost_alpha": {{
    "label": "Повышение доходности — активный риск", "desc": "≤100 знаков. Режим: {regime_label}.",
    "picks": [  // 2 реальные акции роста НЕ из портфеля
      {{"ticker": "...", "name": "...", "why": "≤200 знаков: ROE/маржа/beta/momentum + [источник]", "type": "Stock"}}
    ]
  }},
  "rebalance": {{
    "label": "Качественная ребалансировка", "desc": "≤100 знаков.",
    "picks": [  // 2 quality-акции (4-Pillar F+V > 0)
      {{"ticker": "...", "name": "...", "why": "≤200 знаков с SEC-цифрами [SEC EDGAR]", "type": "Stock"}}
    ]
  }},
  "protect_capital": {{
    "label": "Защита капитала", "desc": "≤100 знаков. CVaR/tail-risk hedge.",
    "picks": [  // 1-2 защитных инструмента
      {{"ticker": "...", "name": "...", "why": "≤200 знаков: Beta/корреляция/дивиденд [источник]", "type": "Stock|ETF|Bond"}}
    ]
  }},
  "regime_play": {{
    "label": "Режимная ставка — {regime_label}", "desc": "≤100 знаков. Тактическая ставка под текущий режим.",
    "picks": [  // 1-2 идеи специфичные для режима {regime_label}
      {{"ticker": "...", "name": "...", "why": "≤200 знаков: почему этот режим даёт преимущество [Regime]", "type": "Stock|ETF"}}
    ]
  }}
}}"""

    return (
        "Проанализируй портфель как Senior Financial Analyst. Верни СТРОГО JSON (без текста вне JSON, без markdown).\n"
        "ВСЕ ТЕКСТЫ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.\n"
        "Используй 3-step reasoning: (1) взаимосвязи между метриками, (2) скрытые риски, (3) стратегические рекомендации с числами.\n\n"
        "ЦЕПОЧКА АНАЛИЗА — каждый раздел ссылается на предыдущий:\n"
        "  Риск (CVaR/Vol) → Состав портфеля (hotspots, TRC) → Секторы (перекосы) →\n"
        "  Факторы (β-концентрации) → Режим (Recovery/Stagflation/...) →\n"
        "  Банки (Goldman/Barclays/JPM) → 4-Pillar (F+V+T+C) → Action Plan.\n"
        "Каждый комментарий должен явно указывать, откуда взяты данные И как это связано с соседними разделами.\n\n"
        '{\n'
        '  "verdict": "≤200 знаков — причинно-следственный вердикт: что СЕЙЧАС и ПОЧЕМУ",\n'
        '  "plain_summary": "≤300 знаков — Executive Summary: позиция + главный риск + приоритет",\n'
        '  "bullets": ["5–7 пунктов ≤200 знаков каждый с [Источник] — цепочка: риск → состав → секторы → факторы → действие"],\n'
        '  "ai_cvar_note": "≤120 знаков — простыми словами: примерно сколько денег теряется '
        'в худший день из 20, нормально ли это для данного профиля риска [Quant Engine].",\n'
        '  "ai_sharpe_note": "≤120 знаков — простыми словами: окупается ли риск доходностью, '
        'лучше или хуже рынка [Quant Engine].",\n'
        '  "ai_mdd_note": "≤120 знаков — простыми словами: насколько глубоко портфель падал '
        'исторически и что это значит для владельца [Quant Engine].",\n'
        '  "ai_risk_comment": "≤220 знаков — потери в худший день из 20 (CVaR), нестабильность '
        '(Vol), макс. просадка и стабильность обгона рынка (Sharpe): взаимосвязь, $ потерь. '
        'Укажи какая позиция [см. ai_holdings_comment] вносит наибольший вклад [Quant Engine]",\n'
        '  "ai_benchmark_comment": "≤200 знаков — стабильность обгона рынка (IR) vs бенчмарк: '
        'причины отставания/опережения. Свяжи с нестабильностью портфеля из ai_risk_comment '
        'и секторными перекосами из ai_sector_comment [Quant Engine]",\n'
        '  "ai_performance_comment": "≤180 знаков — доходность 1М/3М/12М/YTD: тренды и причины. '
        'Укажи как макро-режим [см. ai_regime_comment] повлиял на динамику [Quant Engine]",\n'
        '  "ai_regime_comment": "≤220 знаков — текущий режим рынка: что это значит простыми словами '
        '(рынок восстанавливается / перегрет / в стагфляции). Что рекомендуют Goldman Sachs, '
        'Barclays, JPMorgan для этого режима. Как это влияет на факторы [Quant Engine][GS][Barclays][JPM]",\n'
        '  "regime_confirmation": {\n'
        '    "stance": "confirms | partial | diverges",\n'
        '    "summary": "≤220 знаков — простыми словами: подтверждается ли вывод движка о режиме, '
        'на основании каких независимых сигналов",\n'
        '    "signals": ["3–6 строк ≤90 знаков — каждая начинается с ✓/⚠/✗ + сигнал. '
        'ПОКРЫТЬ обязательно: (1) кривая доходности 10Y−2Y из summary.macro, '
        '(2) HY OAS (кредитный спред), (3) VIX (страх рынка), '
        '(4) факторные беты портфеля vs ожидаемые для режима ([Barclays]), '
        '(5) банковский консенсус [GS]/[Barclays]/[JPM]"]\n'
        '  },\n'
        '  "ai_holdings_comment": "≤200 знаков — какие позиции занимают наибольшую долю в риске '
        '(TRC — доля в общем риске). Назови конкретные тикеры-hotspots и объясни почему они опасны. '
        'Свяжи с факторами [см. ai_factor_comment] [Quant Engine]",\n'
        '  "ai_sector_comment": "≤170 знаков — какие секторы перевешены и недовешены. '
        'Назови субсектора (напр. внутри Tech: софт vs полупроводники). '
        'Укажи риск ротации при режиме [см. ai_regime_comment]",\n'
        '  "ai_factor_comment": "≤220 знаков — назови Value β (чувствительность к дешёвым акциям): '
        'положительный = портфель любит дешёвые акции, отрицательный = наоборот. '
        f'Режим {regime_label}: Barclays рекомендует Value>Growth — совпадает ли портфель? '
        'Назови конкретный фактор для наращивания и тикеры для этого [Quant Engine][Barclays]",\n'
        '  "ai_4pillar_comment": "≤240 знаков — 4-Pillar (F=фундаментал, V=оценка, T=техника, '
        'C=кредит). ВАЖНО про V простыми словами: оценка сравнивается с НОРМОЙ СЕКТОРА, а не '
        'со всем рынком — у технологий P/E высокий ОТ ПРИРОДЫ (платим за рост), поэтому V≈0 у '
        'теха значит «оценён справедливо ДЛЯ своего сектора», а НЕ «дорого». Отметь позиции с '
        'расходящимися сигналами (F высокий, T низкий = хороший бизнес на спаде) [SEC EDGAR]",\n'
        '  "ai_stress_comment": "≤220 знаков — в худшем сценарии: насколько падает портфель и '
        'какие позиции пострадают сильнее всего (свяжи с TRC из ai_holdings_comment и '
        'факторными бетами из ai_factor_comment). Конкретные цифры просадки [Quant Engine]",\n'
        '  "ai_action_comment": "≤200 знаков — что делать в первую очередь: Trim/Sell hotspots '
        '(из ai_holdings_comment) → затем Buy (из ai_factor_comment и ai_4pillar_comment). '
        'Ожидаемый эффект на долю в риске (TRC) позиций [Quant Engine]",\n'
        '  "ai_effect_comment": "≤220 знаков — чего ожидать после ребалансировки: '
        'потери в худший день из 20 (CVaR) до→после, нестабильность (Vol), Sharpe. '
        'Причинно-следственная связь с конкретными изменениями позиций [Quant Engine]",\n'
        f'{picks_spec},\n'
        '  "action_plan_text": "≤800 знаков — приоритетные действия: Trim/Sell сначала, '
        'конкретные уровни, cumulative |Δw| ≤ 25% NAV",\n'
        '  "ai_action_impact": "≤300 знаков — количественный прогноз: CVaR/Vol/TE/Sharpe после плана"\n'
        '}\n\n'
        "ПРАВИЛА:\n"
        "- ВСЕ тексты на РУССКОМ. Без «RAMP».\n"
        "- ПРОСТОЙ ЯЗЫК — глоссарий замен:\n"
        "  CVaR → 'потери в худший день из 20'\n"
        "  волатильность → 'нестабильность' или 'насколько прыгает портфель'\n"
        "  Бета → 'чувствительность к рынку (Beta 2 = двигается вдвое сильнее рынка)'\n"
        "  IR → 'стабильность обгона рынка'\n"
        "  TRC → 'доля в общем риске портфеля'\n"
        "  Recovery-режим → 'рынок восстанавливается после спада'\n"
        "- Если используешь финансовый термин — сразу поясни его в скобках простыми словами.\n"
        "- БАНКОВСКАЯ АНАЛИТИКА: даже без RAG-данных — используй знания о позициях "
        "Goldman Sachs, Barclays, JPMorgan, Morgan Stanley по текущему режиму и секторам. "
        "Теги: [GS], [Barclays], [JPM], [MS].\n"
        "- ПОДТВЕРЖДЕНИЕ РЕЖИМА — заполни поле regime_confirmation:\n"
        f"    движок выдал {regime_label} (confidence из summary.regime); проверь это на "
        "    НЕЗАВИСИМЫХ сигналах: yield curve 10Y−2Y (положительная = рост, инверсия = "
        "    рецессия), HY OAS (<350 bp = риск-он, >550 bp = стресс), VIX (<20 = спокойствие), "
        "    breakeven (инфляционные ожидания), факторные беты портфеля и банковский консенсус. "
        "    Stance: 'confirms' если ≥80% сигналов согласны, 'partial' если 2-3 расходятся, "
        "    'diverges' при фундаментальном противоречии. Каждый сигнал в signals[] — "
        "    одна строка с ✓/⚠/✗ + что именно подтверждает или противоречит.\n"
        "- 3-step reasoning: взаимосвязи → риски → рекомендации с числами.\n"
        "- Каждое число — [Quant Engine], [SEC EDGAR], [Regime] или [RAG: файл].\n"
        f"- Риск-профиль: {user_profile}. Режим: {regime_label}.\n"
        "- РЕАЛЬНЫЕ АКЦИИ: PLTR, CRWD, JNJ, COST, KO, ANET, PANW и т.п. — не только ETF.\n"
        "- why: конкретные цифры (ROE, маржа, Beta, P/E, momentum) с тегом [источник].\n"
        f"{contradiction_rule}"
        f"{rag_rule}"
        f"\n=== ДАННЫЕ ПОРТФЕЛЯ ===\n"
        f"{_wrap_untrusted('broker_portfolio_json', json.dumps(summary, ensure_ascii=False)[:9000])}"
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
        "verdict":                  verdict,
        "plain_summary":            plain_summary[:300],
        "bullets":                  bullets[:7 if tier == "deep" else 4],
        "action_plan_text":         action_plan_text,
        "ai_action_impact":         ai_action_impact,
        "stock_picks":              stock_picks,
        "used_rag":                 False,
        "model_used":               "fallback",
        "ai_risk_comment":          "",
        "ai_benchmark_comment":     "",
        "ai_performance_comment":   "",
        "ai_regime_comment":        "",
        "ai_holdings_comment":      "",
        "ai_sector_comment":        "",
        "ai_factor_comment":        "",
        "ai_4pillar_comment":       "",
        "ai_stress_comment":        "",
        "ai_action_comment":        "",
        "ai_effect_comment":        "",
        "regime_confirmation":      {"stance": "", "summary": "", "signals": []},
        "rag_context":              "",
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

    # regime_play: a basic regime-aligned pick
    picks_regime = [
        {"ticker": "DBC" if not expansion else "MTUM",
         "name":   "Invesco DB Commodity Index ETF" if not expansion else "iShares MSCI USA Momentum ETF",
         "why":    ("Товарная экспозиция защищает при Slowdown-режиме [Regime]"
                    if not expansion else
                    "Momentum-фактор лидирует в Expansion-режиме [Regime] [Quant Engine]"),
         "type":   "ETF"},
    ]
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
        "regime_play":     {"label": f"Режимная ставка — {'Expansion' if expansion else 'Slowdown'}",
                            "desc":  "Тактическое позиционирование под текущий макро-режим.",
                            "picks": picks_regime},
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
    # RAG is consulted for BOTH tiers (_fetch_rag_context runs tier-agnostic
    # in tg_bot), and both Haiku (BASE) and Sonnet (DEEP) receive the bank
    # context in their prompt.  Gating used_rag to deep-only made the BASE
    # report's QC panel claim "RAG: не использован" even when bank excerpts
    # were retrieved and cited — flag the ACTUAL availability instead.
    used_rag = bool(market_context)

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
        # Remove picks that already exist in the portfolio (they are not "new ideas").
        stock_picks = _remove_held_picks(stock_picks, results)
        # Post-generation contradiction check: remove Sell/Trim tickers from buy picks.
        stock_picks = _check_pick_contradictions(stock_picks, results)
        # Backfill any scenario the AI left empty (or that the held/contradiction
        # filters drained) from the rule-based fallback, so the user always
        # sees the full 4-scenario strategic menu instead of a silently
        # shrinking idea list.  Held tickers are filtered again so a backfilled
        # pick is never one the user already owns.
        _regime_label = ((results.get("regime") or {}).get("regime")
                         if isinstance(results.get("regime"), dict) else None) or "unknown"
        stock_picks = _backfill_empty_scenarios(stock_picks, _regime_label, tier,
                                                 market_context, results)
        total_picks = sum(len(s.get("picks", [])) for s in stock_picks.values())

        logger.info("AI narrative: SUCCESS model=%s verdict=%d chars bullets=%d picks=%d",
                    model, len(verdict), len(bullets), total_picks)

        # Extract per-section AI comments.  Strip unverified RAG citations so
        # a comment never references a bank report that wasn't retrieved.
        # `_soft_trim` lands the cut on a sentence boundary so the report
        # never shows half-sentences like "[GS][Bar".
        def _comment(key: str, limit: int = 250) -> str:
            txt = str(parsed.get(key, "")).strip()
            stripped = _strip_unverified_rag_citations(txt, market_context)
            return _soft_trim(stripped, limit)

        # Structured regime-confirmation cell — DEEP tier only.  Validates
        # that the AI cross-checked the engine's regime label against macro
        # drivers, factor betas and bank views.
        def _regime_confirmation() -> dict:
            if tier != "deep":
                return {"stance": "", "summary": "", "signals": []}
            raw = parsed.get("regime_confirmation") or {}
            stance  = str(raw.get("stance", "")).strip().lower()
            if stance not in {"confirms", "partial", "diverges"}:
                stance = ""
            summary = _soft_trim(_strip_unverified_rag_citations(
                str(raw.get("summary", "")).strip(), market_context), 260)
            signals_in = raw.get("signals") or []
            signals = [
                _soft_trim(_strip_unverified_rag_citations(
                    str(s).strip(), market_context), 120)
                for s in signals_in if str(s).strip()
            ][:6]
            return {"stance": stance, "summary": summary, "signals": signals}

        return {
            "verdict":                  _soft_trim(verdict, 300),
            "plain_summary":            _soft_trim(plain_summary, 400),
            "bullets":                  bullets[:7 if tier == "deep" else 4],
            "action_plan_text":         _soft_trim(plan_txt, 1000) if tier == "deep" else "",
            "ai_action_impact":         _soft_trim(impact_txt, 400) if tier == "deep" else "",
            "stock_picks":              stock_picks,
            "used_rag":                 used_rag,
            "model_used":               model,
            "ai_cvar_note":             _comment("ai_cvar_note", 160),
            "ai_sharpe_note":           _comment("ai_sharpe_note", 160),
            "ai_mdd_note":              _comment("ai_mdd_note", 160),
            "ai_risk_comment":          _comment("ai_risk_comment"),
            "ai_benchmark_comment":     _comment("ai_benchmark_comment"),
            "ai_performance_comment":   _comment("ai_performance_comment"),
            "ai_regime_comment":        _comment("ai_regime_comment"),
            "ai_holdings_comment":      _comment("ai_holdings_comment"),
            "ai_sector_comment":        _comment("ai_sector_comment", 200),
            "ai_factor_comment":        _comment("ai_factor_comment"),
            "ai_4pillar_comment":       _comment("ai_4pillar_comment"),
            "ai_stress_comment":        _comment("ai_stress_comment"),
            "ai_action_comment":        _comment("ai_action_comment"),
            "ai_effect_comment":        _comment("ai_effect_comment"),
            "regime_confirmation":      _regime_confirmation(),
            # Surface the raw RAG context so integrity_checks can show the
            # actual snippet count.  Without this the integrity pill was
            # hard-wired to "~0 отрывков" even when ChromaDB returned real
            # bank-report excerpts — the bank views WERE reaching the AI,
            # but the UI claimed nothing was retrieved.
            "rag_context":              market_context,
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
    Ensure stock_picks has the four scenario keys.
    Strips unverified RAG citations from pick rationale.
    """
    result: dict = {}
    for key in ("boost_alpha", "rebalance", "protect_capital", "regime_play"):
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


def _remove_held_picks(stock_picks: dict, results: dict) -> dict:
    """
    Remove tickers already in the live portfolio from AI idea picks.

    Recommending GOOGL (16.7% held) as a "new idea" is a logical contradiction
    that undermines trust in the report.  boost_alpha / rebalance / protect_capital
    scenarios are all cleaned — the AI should only suggest additions, not echo
    current holdings.
    """
    perf = results.get("performance_table")
    if perf is None or getattr(perf, "empty", True):
        return stock_picks
    held: set[str] = {
        str(t).upper().split(".")[0]
        for t in perf.get("Ticker", perf.index if perf.index.name == "Ticker" else [])
    }
    if not held:
        return stock_picks
    for scenario in stock_picks.values():
        if not isinstance(scenario, dict):
            continue
        original = scenario.get("picks", [])
        filtered = [p for p in original
                    if p.get("ticker", "").upper().split(".")[0] not in held]
        if len(filtered) < len(original):
            removed = [p["ticker"] for p in original if p not in filtered]
            logger.info("Held-pick filter: removed %s (already in portfolio)", removed)
        scenario["picks"] = filtered
    return stock_picks


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


def _backfill_empty_scenarios(stock_picks: dict, regime_label: str, tier: str,
                              market_context: str, results: dict) -> dict:
    """
    Guarantee all four strategic scenarios carry at least one pick.

    The deep/base templates skip any scenario whose `picks` list is empty,
    so a category the AI left blank (or that the held/contradiction filters
    drained) silently shrinks the idea menu from 4 cards to 3.  We backfill
    such gaps from the deterministic rule-based catalogue, then re-run the
    held-ticker filter so a backfilled idea is never one the user owns.
    """
    fallback = _fallback_stock_picks(regime_label, tier)
    fallback = _normalise_stock_picks(fallback, tier, market_context)
    fallback = _remove_held_picks(fallback, results)

    for key in ("boost_alpha", "rebalance", "protect_capital", "regime_play"):
        scenario = stock_picks.get(key) or {}
        if scenario.get("picks"):
            continue
        fb = fallback.get(key) or {}
        if fb.get("picks"):
            # Preserve the AI's label/desc if present, else take the fallback's.
            stock_picks[key] = {
                "label": scenario.get("label") or fb.get("label", key),
                "desc":  scenario.get("desc")  or fb.get("desc", ""),
                "picks": fb["picks"],
            }
            logger.info("Idea backfill: scenario '%s' was empty → filled from "
                        "rule-based catalogue (%d picks)", key, len(fb["picks"]))
    return stock_picks


__all__ = ["generate_narrative"]
