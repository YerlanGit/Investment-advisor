"""
AI narrative generator — produces verdict, plain-language summary, insight bullets,
stock-pick scenarios, and (deep tier) action-plan commentary for the PDF templates.

Model selection:
  Base tier  → Claude Haiku  (fast, low-cost; 900 output tokens)
  Deep tier  → Claude Sonnet (higher quality prose + RAG synthesis; 4 500 tokens)

Falls back to a rule-based summary when the API key is absent or the call fails,
so the PDF is always produced regardless of API availability.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger("AINarrative")

MAX_TOKENS_BASE = 900
MAX_TOKENS_DEEP = 4_500

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
            "Output advisory-only text. Never mention 'RAMP'.")


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


# ── Narrative prompt ─────────────────────────────────────────────────────────

def _user_prompt(summary: dict, *, tier: str, market_context: str = "",
                 user_profile: str = "Moderate") -> str:
    regime = (summary.get("regime") or {})
    regime_label = regime.get("regime", "unknown")
    n_bullets = "5–7" if tier == "deep" else "3"
    n_picks   = 5 if tier == "deep" else 3
    ask_plan  = (
        'Also output key "action_plan_text": concise block (≤600 chars) with '
        '2-3 sentences on the most urgent Trim/Sell actions. Reference action_plan from summary.'
        if tier == "deep" else ""
    )

    rag_block = ""
    rag_rule  = ""
    if tier == "deep" and market_context:
        rag_block = (
            "\n\n=== BANK ANALYTICS (RAG) ===\n"
            f"{market_context[:6000]}\n"
            "=== END BANK ANALYTICS ==="
        )
        rag_rule = (
            "When using a fact from BANK ANALYTICS in bullets, action_plan_text, or stock pick "
            "rationale — cite it as [RAG: <filename>] EXACTLY as the filename appears in the "
            "block above. Do not invent filenames or numbers not supported by summary or RAG. "
            "If a pick cannot be confirmed by RAG or quant data, mark it [NOT CONFIRMED].\n"
        )

    scenarios_spec = f"""
"stock_picks": {{
  "boost_alpha": {{
    "label": "Boost Alpha — Higher Risk",
    "desc": "Increase returns by accepting more risk. Current regime: {regime_label}.",
    "picks": [  // {1 if n_picks == 3 else 2} picks: small-cap momentum, commodity/crypto/PE ETFs
      {{"ticker": "...", "name": "...", "why": "...(≤120 chars, cite source)", "type": "ETF|Stock"}}
    ]
  }},
  "rebalance": {{
    "label": "Maintain Profile — Quality Rebalance",
    "desc": "Improve portfolio quality without changing overall risk level.",
    "picks": [  // {1 if n_picks == 3 else 2} picks: quality stocks aligned with regime
      {{"ticker": "...", "name": "...", "why": "...", "type": "ETF|Stock"}}
    ]
  }},
  "protect_capital": {{
    "label": "Protect Capital — Reduce Risk",
    "desc": "Defensive positioning for regime change or rising macro risk.",
    "picks": [  // 1 pick: dividend stocks, short-duration bonds, gold ETF
      {{"ticker": "...", "name": "...", "why": "...", "type": "ETF|Stock"}}
    ]
  }}
}}"""

    return (
        "Analyze the portfolio and return STRICT JSON (no text outside JSON, no markdown):\n"
        "{\n"
        '  "verdict": "1 sentence ≤180 chars — overall verdict",\n'
        '  "plain_summary": "2-3 plain sentences for a non-expert investor: '
        'current state, main risk, one concrete action. ≤300 chars.",\n'
        f'  "bullets": [{n_bullets} items, each ≤180 chars, each with [Source] tag],\n'
        f"{scenarios_spec},\n"
        '  "action_plan_text": "..."  // deep tier only\n'
        "}\n\n"
        "Rules:\n"
        "- No 'RAMP' anywhere.\n"
        "- Every number needs a source tag: [Quant Engine], [SEC EDGAR], [CDS], [Regime], or [RAG: file].\n"
        "- Stock picks must match user risk profile: " + user_profile + ".\n"
        "- For Boost Alpha: suggest instruments NOT already in the portfolio that increase alpha "
        "(small-cap momentum ETFs like IWM/MTUM, commodity ETFs like DBC/GLD, crypto ETFs like BITO, "
        "private equity like PSP, leverage ETFs matching risk appetite).\n"
        "- For Rebalance: suggest quality stocks/ETFs aligned with current market regime "
        "(factor: QUAL, sector rotation based on regime signals).\n"
        "- For Protect Capital: suggest defensive instruments (AGG, TLT, VIG dividend ETF, "
        "sector ETFs like XLU/XLV, or trim existing overweight positions).\n"
        "- CoVe self-check: before finalizing, verify that every ticker exists, every "
        "fact is sourced, and no number is invented.\n"
        f"{rag_rule}"
        f"\n{ask_plan}\n\n"
        f"=== PORTFOLIO SUMMARY ===\n{json.dumps(summary, ensure_ascii=False)[:9000]}"
        f"{rag_block}"
    )


# ── Fallback narrative (no API key / failure path) ───────────────────────────

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
    verdict    = (f"Risk profile: {risk_label} (composite {composite}/100). "
                  f"{s_text}. Regime: {regime.get('regime', 'N/A')}.")

    bullets: list[str] = []
    if perf is not None and not perf.empty and "Score_Hotspot" in perf.columns:
        hot = perf[perf["Score_Hotspot"] == True]   # noqa: E712
        if not hot.empty:
            t   = str(hot.iloc[0].get("Ticker"))
            trc = float(hot.iloc[0].get("Euler_Risk_Contribution_Pct") or 0)
            bullets.append(f"Position {t} contributes {trc:.1f}% of total portfolio risk — "
                           f"consider partial reduction [Quant Engine]")
    cvar = metrics.get("CVaR_95_Daily")
    if cvar is not None:
        bullets.append(f"Expected loss on the worst 5% of days: {cvar*100:.1f}% [Quant Engine]")
    mdd = metrics.get("Max_Drawdown")
    if mdd is not None:
        bullets.append(f"Historical peak-to-trough drawdown: {mdd*100:.1f}% [Quant Engine]")
    if regime.get("regime"):
        bullets.append(f"Market regime {regime['regime']} "
                       f"({int(round(regime.get('confidence', 0)*100))}% confidence) "
                       f"factored into scoring [Regime]")
    if tier == "deep":
        bullets.append("See Action Plan for buy/sell levels derived from ATR + SMA200 [Quant Engine]")

    plain_summary = (
        f"Your portfolio is in the {risk_label} risk zone ({composite}/100). "
        + (f"Risk-adjusted return (Sharpe) is {sharpe:.2f} — "
           + ("positive, a good sign." if sharpe > 0 else "below zero, use caution.")
           if isinstance(sharpe, (int, float)) and sharpe == sharpe else "")
    )

    regime_label = regime.get("regime", "")
    stock_picks  = _fallback_stock_picks(regime_label, tier)

    return {
        "verdict":          verdict,
        "plain_summary":    plain_summary[:300],
        "bullets":          bullets[:7 if tier == "deep" else 3],
        "action_plan_text": ("Priority: address concentration hotspots first. "
                             "Then add positions at Buy-zone levels. "
                             "Keep cumulative turnover ≤25% NAV."
                             if tier == "deep" else ""),
        "stock_picks":      stock_picks,
        "used_rag":         False,
    }


def _fallback_stock_picks(regime_label: str, tier: str) -> dict:
    """Rule-based stock picks when Claude is unavailable."""
    expansion = regime_label in ("Recovery", "Expansion", "")
    picks_boost = [
        {"ticker": "MTUM", "name": "iShares MSCI USA Momentum Factor ETF",
         "why": "Captures equity momentum premium; outperforms in expansion regimes [Quant Engine]",
         "type": "ETF"},
    ]
    picks_balance = [
        {"ticker": "QUAL", "name": "iShares MSCI USA Quality Factor ETF",
         "why": "High-quality large caps with stable earnings; regime-agnostic [Quant Engine]",
         "type": "ETF"},
    ]
    picks_protect = [
        {"ticker": "AGG",  "name": "iShares Core U.S. Aggregate Bond ETF",
         "why": "Broad US bond exposure; reduces equity concentration risk [Quant Engine]",
         "type": "ETF"},
    ]
    if not expansion:
        picks_boost = [
            {"ticker": "DBC", "name": "Invesco DB Commodity Index ETF",
             "why": "Commodity exposure benefits from stagflation / slowdown regimes [Quant Engine]",
             "type": "ETF"},
        ]

    if tier == "deep":
        picks_boost.append(
            {"ticker": "IWM", "name": "iShares Russell 2000 ETF",
             "why": "Small-cap momentum; adds alpha vs large-cap heavy portfolios [Quant Engine]",
             "type": "ETF"}
        )
        picks_balance.append(
            {"ticker": "VIG", "name": "Vanguard Dividend Appreciation ETF",
             "why": "Consistent dividend growers; low drawdown, quality bias [Quant Engine]",
             "type": "ETF"}
        )

    return {
        "boost_alpha":      {"label": "Boost Alpha — Higher Risk",
                             "desc":  "Increase return potential by adding growth or momentum exposure.",
                             "picks": picks_boost},
        "rebalance":        {"label": "Maintain Profile — Quality Rebalance",
                             "desc":  "Improve portfolio quality without changing overall risk.",
                             "picks": picks_balance},
        "protect_capital":  {"label": "Protect Capital — Reduce Risk",
                             "desc":  "Defensive positioning for regime change or macro uncertainty.",
                             "picks": picks_protect},
    }


# ── Public API ───────────────────────────────────────────────────────────────

def generate_narrative(results: dict, tier: str = "base",
                       market_context: str = "",
                       user_risk_profile: str = "Moderate") -> dict:
    """
    Returns {verdict, plain_summary, bullets, stock_picks, action_plan_text, used_rag}.

    Base tier  → Claude Haiku  (900 tokens, fast)
    Deep tier  → Claude Sonnet (4500 tokens, richer prose + RAG synthesis)
    Falls back deterministically when the API is unavailable or fails.
    """
    api_key  = os.getenv("ANTHROPIC_API_KEY")
    used_rag = bool(market_context) and tier == "deep"

    if not api_key:
        logger.info("ANTHROPIC_API_KEY missing — using fallback narrative.")
        out = _fallback_narrative(results, tier)
        out["used_rag"] = False
        return out

    summary = _summarise_for_prompt(results)
    model   = MODEL_DEEP if tier == "deep" else MODEL_BASE
    max_tok = MAX_TOKENS_DEEP if tier == "deep" else MAX_TOKENS_BASE

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
        raw = response.content[0].text.strip()
        first_brace = raw.find("{")
        last_brace  = raw.rfind("}")
        if first_brace == -1 or last_brace == -1:
            raise ValueError("No JSON object in response")
        parsed = json.loads(raw[first_brace:last_brace + 1])

        verdict       = str(parsed.get("verdict", "")).strip()
        plain_summary = str(parsed.get("plain_summary", "")).strip()
        bullets       = [str(b).strip() for b in (parsed.get("bullets") or []) if str(b).strip()]
        plan_txt      = str(parsed.get("action_plan_text", "")).strip()
        stock_picks   = parsed.get("stock_picks") or {}

        if not verdict or not bullets:
            raise ValueError("verdict or bullets missing")

        # CoVe: strip RAG citations whose source file is not in market_context.
        bullets       = [_strip_unverified_rag_citations(b, market_context) for b in bullets]
        plan_txt      = _strip_unverified_rag_citations(plan_txt, market_context)
        verdict       = _strip_unverified_rag_citations(verdict, market_context)
        plain_summary = _strip_unverified_rag_citations(plain_summary, market_context)

        # Normalise stock_picks structure (both flat list and nested dict are accepted).
        stock_picks = _normalise_stock_picks(stock_picks, tier, market_context)

        return {
            "verdict":          verdict[:300],
            "plain_summary":    plain_summary[:400],
            "bullets":          bullets[:7 if tier == "deep" else 3],
            "action_plan_text": plan_txt[:800] if tier == "deep" else "",
            "stock_picks":      stock_picks,
            "used_rag":         used_rag,
        }
    except Exception as exc:
        logger.warning("AI narrative failed (%s) — using fallback.", exc)
        out = _fallback_narrative(results, tier)
        out["used_rag"] = False
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
                "why":    why[:160],
                "type":   str(p.get("type", "Stock"))[:10],
            })
        result[key] = {
            "label": str(scenario.get("label", key))[:80],
            "desc":  str(scenario.get("desc", ""))[:160],
            "picks": clean_picks,
        }
    return result


__all__ = ["generate_narrative"]
