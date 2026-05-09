"""
AI narrative generator — wraps the Claude API to produce the verdict box,
executive-summary bullets, and (deep tier only) the action-plan commentary
that the PDF templates render.

The module is deterministic when ANTHROPIC_API_KEY is missing or the call
fails: it falls back to a rule-based summary derived purely from the engine
output, so the PDF is always produced.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger("AINarrative")

# Hard caps so we don't blow the API budget per report.
MAX_TOKENS_BASE = 900
MAX_TOKENS_DEEP = 3500
DEFAULT_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


# ── Prompt builders ─────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """Load the project's SYSTEM_PROMPT.md (v3) as the system instruction."""
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "..", "SYSTEM_PROMPT.md"),
        os.path.join(here, "..", "..", "SYSTEM_PROMPT.md"),
    ]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError:
            continue
    return ("You are an investment-advisory and risk-management assistant. "
            "Output advisory-only text. Never mention 'RAMP'.")


def _summarise_for_prompt(results: dict) -> dict:
    """
    Build a compact, JSON-serialisable view of analyze_all() results that
    the LLM consumes.  Stays small to keep input tokens bounded.
    """
    metrics = results.get("portfolio_metrics") or {}
    perf    = results.get("performance_table")
    perf_rows: list[dict] = []
    if perf is not None and not perf.empty:
        # Limit to 25 rows to bound input — sort by Current_Value desc.
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
            })
    bm = {
        name: {
            "ir":     _safe_round(d.get("Information_Ratio"),   2),
            "te":     _safe_round((d.get("Tracking_Error") or 0) * 100, 1),
            "excess": _safe_round((d.get("Excess_Return_Ann") or 0) * 100, 1),
        }
        for name, d in (results.get("benchmark_comparison") or {}).items()
    }
    cvar_boot = metrics.get("CVaR_95_Bootstrap") or {}
    return {
        "metrics": {
            "vol_ann":        _safe_round((metrics.get("Total_Volatility_Ann") or 0) * 100, 1),
            "sharpe":         _safe_round(metrics.get("Sharpe_Ratio"),  2),
            "sortino":        _safe_round(metrics.get("Sortino_Ratio"), 2),
            "var_95":         _safe_round((metrics.get("VaR_95_Daily")  or 0) * 100, 2),
            "cvar_95":        _safe_round((metrics.get("CVaR_95_Daily") or 0) * 100, 2),
            "cvar_lo95":      _safe_round((cvar_boot.get("lo95") or 0) * 100, 2),
            "cvar_hi95":      _safe_round((cvar_boot.get("hi95") or 0) * 100, 2),
            "max_dd":         _safe_round((metrics.get("Max_Drawdown")  or 0) * 100, 2),
            "max_erc_pct":    _safe_round(metrics.get("Max_Euler_Risk_Pct"), 1),
            "composite":      metrics.get("Composite_Risk_Score"),
        },
        "regime":     results.get("regime"),
        "sectors":    {k: round(float(v) * 100, 1)
                       for k, v in (results.get("sector_exposure") or {}).items()},
        "benchmarks": bm,
        "holdings":   perf_rows,
        "action_plan": (results.get("action_plan") or [])[:8],
    }


def _safe_round(v, digits: int):
    try:
        x = float(v)
        if x != x:  # NaN check
            return None
        return round(x, digits)
    except (TypeError, ValueError):
        return None


def _user_prompt(summary: dict, *, tier: str, market_context: str = "") -> str:
    """
    Build the user message that asks for verdict + bullets + (deep) plan.

    When tier == 'deep' AND market_context is non-empty, an extra RAG section
    is appended and the model is INSTRUCTED to cite each insight back as
    [RAG: <filename>].  Citations whose filename does not appear verbatim in
    the provided context are stripped post-hoc to enforce CoVe.
    """
    if tier == "deep":
        n_bullets = "5–7"
        ask_plan  = ("Также выведи ключ \"action_plan_text\": краткий блок (≤ 600 знаков) "
                     "с 2-3 предложениями про самые приоритетные действия (Trim/Sell сначала). "
                     "Опирайся на action_plan из summary.")
    else:
        n_bullets = "3"
        ask_plan  = ""

    rag_block = ""
    rag_rule  = ""
    if tier == "deep" and market_context:
        rag_block = (
            "\n\n=== BANK ANALYTICS (RAG) ===\n"
            f"{market_context[:6000]}\n"
            "=== END BANK ANALYTICS ==="
        )
        rag_rule = (
            "Если в bullets или action_plan_text используешь факт из BANK "
            "ANALYTICS — ОБЯЗАТЕЛЬНО цитируй источник как [RAG: <имя_файла>] "
            "ровно в том виде, в каком имя файла появляется в блоке выше "
            "(между квадратными скобками после даты). Не выдумывай файлы, "
            "которых нет в блоке. Не выдумывай числа, не подкреплённые ни "
            "summary, ни RAG-блоком.\n"
        )

    plain_field = (
        '  "plain_summary": "2-3 предложения простым языком для обычного '
        'инвестора (без жаргона): что сейчас с портфелем, главный риск, '
        'и один конкретный совет что делать. ≤300 знаков.",\n'
    )

    return (
        "Ниже структура портфеля и риск-аналитика. Сформируй СТРОГО JSON-ответ:\n"
        "{\n"
        '  "verdict":  "1 предложение, ≤180 знаков — общий вердикт",\n'
        f'{plain_field}'
        f'  "bullets": [{n_bullets} пунктов, каждый ≤ 180 знаков],\n'
        '  "action_plan_text": "..."  // только для tier=deep\n'
        "}\n\n"
        "Никакого текста вне JSON. Никаких markdown-форматов. Слово \"RAMP\" не использовать.\n"
        "Все числа со ссылкой на источник [Quant Engine] / [SEC EDGAR] / [CDS] / [Regime].\n"
        f"{rag_rule}"
        f"\n{ask_plan}\n\n"
        f"=== SUMMARY ===\n{json.dumps(summary, ensure_ascii=False)[:9000]}"
        f"{rag_block}"
    )


# ── Fallback summary (no API key / failure path) ────────────────────────────

def _fallback_narrative(results: dict, tier: str) -> dict:
    """
    Rule-based summary used when Claude is unavailable.  Only references
    fields directly present in `results` so it can never hallucinate.
    """
    metrics = results.get("portfolio_metrics") or {}
    regime  = results.get("regime") or {}
    perf    = results.get("performance_table")

    # Verdict line
    composite = metrics.get("Composite_Risk_Score") or 0
    risk_label = "консервативный" if composite < 33 else \
                 "умеренный" if composite < 66 else "агрессивный"
    sharpe = metrics.get("Sharpe_Ratio")
    s_text = f"Sharpe {sharpe:.2f}" if isinstance(sharpe, (int, float)) and sharpe == sharpe else "Sharpe н/д"
    verdict = (f"Профиль риска {risk_label} (composite {composite}). {s_text}. "
               f"Режим рынка: {regime.get('regime', 'N/A')}.")

    # Bullets — Hotspot first when present (most important user-facing signal),
    # then tail risk, drawdown, and macro regime.
    bullets: list[str] = []
    if perf is not None and not perf.empty and "Score_Hotspot" in perf.columns:
        hot = perf[perf["Score_Hotspot"] == True]   # noqa: E712
        if not hot.empty:
            t = str(hot.iloc[0].get("Ticker"))
            trc = float(hot.iloc[0].get("Euler_Risk_Contribution_Pct") or 0)
            bullets.append(f"🔥 Hotspot: {t} даёт {trc:.1f}% общего риска — рекомендуется частичное сокращение "
                           f"[Quant Engine]")
    cvar = metrics.get("CVaR_95_Daily")
    if cvar is not None:
        bullets.append(f"Ожидаемый убыток в худшие 5% дней: {cvar*100:.1f}% [Quant Engine]")
    mdd = metrics.get("Max_Drawdown")
    if mdd is not None:
        bullets.append(f"Реализованная максимальная просадка: {mdd*100:.1f}% [Quant Engine]")
    if regime.get("regime"):
        bullets.append(f"Макро-режим {regime['regime']} (уверенность {int(round(regime.get('confidence', 0)*100))}%) "
                       f"учтён в Macro Alignment [Regime]")

    if tier == "deep":
        bullets.append("Используйте Action Plan ниже: уровни Buy / Sell / Stop рассчитаны "
                       "из ATR (Wilder RMA) и SMA200 [Quant Engine]")
        action_plan_text = ("Приоритет — закрытие концентрационных hotspots; затем точечные покупки "
                            "по Buy-зонам в активах с положительным Total Score. "
                            "Все сделки в пределах 25% оборота NAV.")
    else:
        action_plan_text = ""

    plain_summary = (
        f"Ваш портфель сейчас в {risk_label} зоне риска ({composite}/100). "
        + (f"Доходность с поправкой на риск (Sharpe) составляет {sharpe:.2f} — "
           + ("выше нуля, это хороший знак." if sharpe > 0 else "ниже нуля, осторожно.")
           if isinstance(sharpe, (int, float)) and sharpe == sharpe else "")
        + (" Рекомендуем проверить Action Plan и сократить позиции с высоким риском." if tier == "deep" else "")
    )

    return {
        "verdict": verdict,
        "plain_summary": plain_summary[:300],
        "bullets": bullets[:7 if tier == "deep" else 3],
        "action_plan_text": action_plan_text,
        "used_rag": False,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def _strip_unverified_rag_citations(text: str, market_context: str) -> str:
    """
    Remove [RAG: <name>] markers whose <name> does not appear verbatim in the
    market_context block.  Defensive against the model citing files it never
    saw.  When market_context is empty this is a no-op (the prompt forbids
    RAG citations in that case anyway).
    """
    if not text:
        return text
    if not market_context:
        # No RAG was provided → strip any RAG citation entirely.
        import re
        return re.sub(r"\s*\[RAG:[^\]]*\]", "", text)
    import re
    def _keep(match: "re.Match") -> str:
        name = match.group(1).strip()
        return match.group(0) if name and name in market_context else ""
    return re.sub(r"\[RAG:\s*([^\]]+)\]", _keep, text)


def generate_narrative(results: dict, tier: str = "base",
                       market_context: str = "") -> dict:
    """
    Returns {verdict, bullets, action_plan_text, used_rag} for the PDF.

    Args:
        results        : dict from UniversalPortfolioManager.analyze_all().
        tier           : 'base' or 'deep'. Only deep tier consumes RAG.
        market_context : Optional RAG context string.  When non-empty AND
                         tier == 'deep', the prompt asks the model to cite
                         sources back as [RAG: <filename>].  Citations whose
                         filename is not present in market_context are
                         stripped to enforce CoVe.

    Falls back deterministically when the API is unavailable or fails.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    used_rag = bool(market_context) and tier == "deep"

    if not api_key:
        logger.info("ANTHROPIC_API_KEY missing — using fallback narrative.")
        out = _fallback_narrative(results, tier)
        out["used_rag"] = False
        return out

    summary = _summarise_for_prompt(results)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        max_tokens = MAX_TOKENS_DEEP if tier == "deep" else MAX_TOKENS_BASE
        response = client.messages.create(
            model       = DEFAULT_MODEL,
            max_tokens  = max_tokens,
            temperature = 0.1,
            system      = _build_system_prompt(),
            messages    = [{
                "role": "user",
                "content": _user_prompt(summary, tier=tier,
                                          market_context=market_context),
            }],
        )
        raw = response.content[0].text.strip()
        # Extract JSON block defensively (model may wrap in markdown despite the rule).
        first_brace = raw.find("{")
        last_brace  = raw.rfind("}")
        if first_brace == -1 or last_brace == -1:
            raise ValueError("No JSON object found in response")
        parsed = json.loads(raw[first_brace:last_brace + 1])

        verdict      = str(parsed.get("verdict", "")).strip()
        plain_summary= str(parsed.get("plain_summary", "")).strip()
        bullets      = [str(b).strip() for b in (parsed.get("bullets") or []) if str(b).strip()]
        plan_txt     = str(parsed.get("action_plan_text", "")).strip()
        if not verdict or not bullets:
            raise ValueError("verdict or bullets missing")

        # CoVe enforcement: strip any RAG citation that wasn't actually
        # present in the market_context block.
        bullets       = [_strip_unverified_rag_citations(b, market_context) for b in bullets]
        plan_txt      = _strip_unverified_rag_citations(plan_txt,      market_context)
        verdict       = _strip_unverified_rag_citations(verdict,       market_context)
        plain_summary = _strip_unverified_rag_citations(plain_summary, market_context)

        return {
            "verdict":          verdict[:300],
            "plain_summary":    plain_summary[:400],
            "bullets":          bullets[:7 if tier == "deep" else 3],
            "action_plan_text": plan_txt[:800] if tier == "deep" else "",
            "used_rag":         used_rag,
        }
    except Exception as exc:
        logger.warning("AI narrative failed (%s) — using fallback.", exc)
        out = _fallback_narrative(results, tier)
        out["used_rag"] = False
        return out


__all__ = ["generate_narrative"]
