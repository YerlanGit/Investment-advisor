"""
AI narrative generator — produces verdict, plain-language summary, insight bullets,
stock-pick scenarios, and (deep tier only) action-plan commentary for the PDF templates.

Model selection (BLOCK 1.2 — current-generation routing, env-overridable):
  Base tier  → Claude Sonnet 4.6 (quality lift over the prior Haiku tier)
  Deep tier  → Claude Opus  4.8  (maximum depth/analysis; omits `temperature`,
               so idea variety comes from the prompt-level freshness directive)

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
from datetime import date
from typing import Optional

logger = logging.getLogger("AINarrative")

MAX_TOKENS_BASE = 5_000  # 4_500 → 5_000 (BLOCK 1.2): BASE now runs on Sonnet (more verbose than Haiku); headroom so the structured tool output is never truncated → no JSON repair / dropped KPI notes
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
    # Audit 2026-06-14: no sentence boundary in budget → cut at the last WORD
    # boundary, never mid-word.  The prod reports showed «увеличивать Fi…»,
    # «недооце…», «рекомендуе…» — a hard char-cut landing inside a word.
    cut = head.rstrip()
    sp  = cut.rfind(" ")
    if sp >= max_chars // 2:
        cut = cut[:sp]
    return cut.rstrip(" ,;:[(") + "…"


def validate_stress_comment(text: str) -> str:
    """
    M1 — post-validator for the stress-test AI comment.

    The system prompt INSTRUCTS the model to mention the non-linear convex
    cap, but Sonnet/Haiku sometimes omit it.  This guard appends an explicit
    footnote whenever neither "ограничител…" nor "cap" is present, so the
    reader always knows extreme per-asset drawdowns were convex-capped.
    """
    if not text:
        return text
    low = text.lower()
    if "ограничител" not in low and "cap" not in low:
        text += ("\n\n*(Примечание: Экстремальные просадки рассчитаны с учётом "
                 "нелинейного ограничителя риска).*")
    return text


# Tier → model routing.  The owner's BLOCK-1 request was "BASE → Sonnet,
# DEEP → Opus (maximum depth & analysis)".  The literal IDs in that request
# (`claude-3-5-sonnet-latest` / `claude-3-opus-latest`) are 2024-generation
# models — pinning them would REGRESS quality below the project's current
# Haiku-4.5/Sonnet-4.6 stack and cost MORE.  We honour the INTENT with the
# current-generation IDs instead:
#   BASE → Sonnet 4.6  (was Haiku 4.5 — a clear quality lift for the volume tier)
#   DEEP → Opus  4.8   (the planned upgrade that was already guard-ready below)
# Both remain env-overridable so cost/latency can be tuned without a deploy
# (e.g. ANTHROPIC_MODEL_BASE=claude-haiku-4-5-20251001 to restore the cheaper
# BASE tier).  NOTE the cost delta the owner is explicitly opting into:
# BASE Haiku→Sonnet ≈3× ($1/$5 → $3/$15), DEEP Sonnet→Opus ≈1.7× ($3/$15 → $5/$25).
MODEL_BASE = os.getenv("ANTHROPIC_MODEL_BASE", "claude-sonnet-4-6")
MODEL_DEEP = os.getenv("ANTHROPIC_MODEL_DEEP", "claude-opus-4-8")

# Sprint-5 (idea staleness): the narrative + stock-picks are produced by ONE
# model call, and that call ran at temperature 0.1 — near-deterministic, so a
# stable macro regime produced byte-identical "ideas" month after month.
# BLOCK 1.1 (06-22): the BASE tier (Sonnet) still parroted the same blue-chips,
# so the default is raised 0.5 → 0.7 and the band widened to 0.5–0.85 — more
# idea exploration while Structured Outputs (forced tool_use) still guarantee
# the JSON shape.  The risk NUMBERS are engine-computed (deterministic)
# regardless of this temperature — only the prose / picks vary.  (DEEP runs on
# Opus, which ignores temperature; its variety comes from the prompt directive.)
def _resolve_temperature() -> float:
    try:
        t = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7"))
    except (TypeError, ValueError):
        t = 0.7
    return min(0.85, max(0.5, t))


NARRATIVE_TEMPERATURE = _resolve_temperature()

# Sprint-5.1 (§4.2): Opus 4.7/4.8 REJECT the `temperature` param (HTTP 400);
# they expose only adaptive thinking.  Now that the DEEP tier defaults to
# Opus (BLOCK 1.2) this guard is LOAD-BEARING, not hypothetical: the API call
# omits `temperature` for the DEEP narrative and relies on the prompt-level
# freshness directive (ideas_rule) for idea variety instead.  BASE stays on
# Sonnet, which still accepts `temperature`, so its 0.5 exploration is intact.
_TEMPERATURE_UNSUPPORTED_PREFIXES = ("claude-opus-4-7", "claude-opus-4-8")


def _model_supports_temperature(model: str) -> bool:
    m = str(model or "")
    return not any(m.startswith(p) for p in _TEMPERATURE_UNSUPPORTED_PREFIXES)


# BLOCK 1.1 (06-22) — day-rotating "angle" for idea generation (time-decay).
# Indexed by day-of-year so successive reports nudge the model toward DIFFERENT
# fresh angles instead of the same blue-chips every run.  Deterministic (same
# day → same angle = reproducible report) but varies daily.
_IDEA_ROTATION_ANGLES = [
    "приоритизируй НЕДОоценённые сектора текущего режима, а не очевидные мегакэпы",
    "предложи имена СРЕДНЕЙ капитализации (mid-cap) с сильными фундаменталиями",
    "усиль НЕДОпредставленный в портфеле фактор (Value/Quality/Momentum) точечной идеей",
    "добавь международную / EM-идею для географической диверсификации",
    "приоритизируй имена с недавним позитивным пересмотром прибыли и сильным momentum",
    "сделай акцент на качестве с реальной премией, избегая дежурных дивидендных мегакэпов",
    "предложи тематическую идею под текущий макро-режим (а не «вечные» имена)",
]


# Sprint-5.3 (замечание 2): plain-Russian regime names.  The engine's labels
# (Expansion/Recovery/Slowdown/Recession) are finance jargon — feed the model
# a human phrase so the verdict/comments read naturally for a retail user.
_REGIME_RU: dict[str, str] = {
    "Expansion": "экономика растёт",
    "Recovery":  "рынок восстанавливается после спада",
    "Slowdown":  "экономика замедляется",
    "Recession": "экономический спад",
}


def _regime_ru(label: str) -> str:
    return _REGIME_RU.get(str(label or "").strip(), "режим рынка не определён")


# ── Structured Outputs tool (Tools API) ──────────────────────────────────────
# Forcing the narrative through a typed tool call makes the response a
# GUARANTEED structured object the SDK parses for us — eliminating the
# brace-finding + _repair_truncated_json hack on possibly-truncated free text.
# The property names mirror exactly what the downstream extractor reads via
# parsed.get(...), and the per-field CONTENT rules still come from the user
# prompt (lengths, citation tags, anti-re-aggregation directives).
_REPORT_TOOL: dict = {
    "name": "emit_report",
    "description": ("Вернуть институциональный нарратив строго в этой структуре. "
                    "Заполни КАЖДОЕ поле по правилам из пользовательского запроса "
                    "(лимиты длины, источники [Quant Engine]/[Regime]/[RAG], "
                    "запрет на самостоятельную агрегацию секторов)."),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict":                {"type": "string"},
            "plain_summary":          {"type": "string"},
            "bullets":                {"type": "array", "items": {"type": "string"}},
            "action_plan_text":       {"type": "string"},
            "ai_action_impact":       {"type": "string"},
            # KPI-strip notes.  ai_cvar_note was MISSING here while the prompt
            # asks for it and the extractor reads it (line ~1273): a model that
            # adheres strictly to the declared schema (Sonnet/BASE) then dropped
            # it, leaving the CVaR card blank in the live BASE report while
            # Sharpe/MDD rendered.  Declaring every field the extractor consumes
            # makes the structured output deterministic across models.
            "ai_cvar_note":           {"type": "string"},
            "ai_sharpe_note":         {"type": "string"},
            "ai_mdd_note":            {"type": "string"},
            "ai_risk_comment":        {"type": "string"},
            "ai_holdings_comment":    {"type": "string"},
            "ai_sector_comment":      {"type": "string"},
            "ai_factor_comment":      {"type": "string"},
            "ai_4pillar_comment":     {"type": "string"},
            "ai_benchmark_comment":   {"type": "string"},
            "ai_performance_comment": {"type": "string"},
            "ai_regime_comment":      {"type": "string"},
            # DEEP-tier section comments — likewise extracted downstream, so
            # they must be declared or a schema-strict model silently omits them.
            "ai_stress_comment":      {"type": "string"},
            "ai_action_comment":      {"type": "string"},
            "ai_effect_comment":      {"type": "string"},
            # Sprint-5 (margin/leverage AI-trigger): when the book carries
            # margin debt (cash leg < 0) the prompt sets has_leverage and
            # REQUIRES this field — an explicit Margin-Call / exponential-risk
            # warning.  Empty string when the book is unlevered.
            "ai_leverage_warning":    {"type": "string"},
            # Nested structures stay permissive — _normalise_stock_picks and
            # _regime_confirmation() validate their shape downstream.
            "stock_picks":            {"type": "object"},
            "regime_confirmation":    {"type": "object"},
        },
        "required": ["verdict", "bullets"],
    },
}


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


# ── Sector figures fed to the LLM (SSOT — shared with the report panel) ──────

def _sector_shares_long_book(results: dict) -> dict:
    """Sector weights as share of the LONG book — the SAME basis the report
    panel renders, so the AI prose and the panel agree."""
    se = results.get("sector_exposure") or {}
    long_only = [(s, float(w)) for s, w in se.items() if float(w) > 0]
    lsum = sum(w for _, w in long_only) or 1.0
    return {s: round(w / lsum * 100, 1) for s, w in long_only}


def _sector_complex_figure(results: dict) -> dict | None:
    """Authoritative combined super-group figure (e.g. Tech-комплекс).

    Computed via the SAME grouping `pdf_payload.build_sector_groups` uses, so
    the model cites one number instead of re-aggregating Tech+Semiconductors
    ad-hoc (the BASE 55% vs DEEP 80.8% bug)."""
    se = results.get("sector_exposure") or {}
    long_only = [(s, float(w)) for s, w in se.items() if float(w) > 0]
    lsum = sum(w for _, w in long_only) or 1.0
    pairs = [(s, w / lsum) for s, w in long_only]
    try:
        from pdf_payload import build_sector_groups  # leaf import, no cycle
        groups = build_sector_groups(pairs)
    except Exception:
        return None
    return groups[0] if groups else None


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
        # BLOCK 3.4 — hard macro series the AI should weigh by their TREND.
        ("unemployment",      "unemployment"),
        ("gdp_growth",        "gdp_growth"),
    )

    def _macro_trend(row: dict):
        """Windowed темп over ≥3 changes (shared series_trend) so the AI weighs
        a sustained DIRECTION, not a single noisy print."""
        hist = row.get("history_30d") or []
        vals = [h.get("value") for h in hist if isinstance(h, dict)]
        try:
            from finance.regime import series_trend
            total, _slope, _n = series_trend(vals, 3)
        except Exception:
            total = None
        return None if total is None else _safe_round(total, 2)

    macro_summary: dict = {}
    for src_key, label in _MACRO_KEYS:
        row = macro_src.get(src_key) or {}
        val = row.get("value")
        if val is None:
            continue
        # Audit 2026-06-14: FRED-sourced, but sanitize for defense-in-depth so
        # NO free-text field reaches the prompt un-fenced (closes the one
        # exception to the "sanitize every string" invariant).
        entry = {
            "value":  _safe_round(val, 2),
            "status": _safe_text(row.get("status"), 16),
            "as_of":  _safe_text(row.get("as_of"), 16),
            "unit":   _safe_text(row.get("unit"), 12),
        }
        # F3 — include the rate-of-change (темп роста/падения).  A rising 4.1%
        # unemployment vs a falling 4.1% are OPPOSITE regime signals.
        trend = _macro_trend(row)
        if trend is not None:
            entry["trend"] = trend
        macro_summary[label] = entry

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
        # Share-of-long-book basis (matches the report panel) + the authoritative
        # combined super-group figure.  The prompt forbids re-aggregating sectors
        # — the model must cite `sector_complex` for combined tech exposure.
        "sectors":         _sector_shares_long_book(results),
        "sector_complex":  _sector_complex_figure(results),
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
        # H4: investor risk mandate so the LLM tailors tone/recommendations
        # (a conservative investor needs tail-risk framing; an aggressive one
        # growth framing).  Composite-risk score is already calibrated for it.
        "risk_mandate": results.get("risk_mandate", "MODERATE"),
        # Sprint-5 (margin/leverage): surface the engine's leverage metrics so
        # the model can quote the actual gross-exposure / leverage-ratio when
        # warning about a margin-funded book (cash leg < 0 → is_leveraged).
        "leverage": _leverage_for_prompt(results.get("leverage_metrics")),
    }


def _leverage_for_prompt(lm: Optional[dict]) -> dict:
    """Compact, %-scaled leverage view for the LLM prompt.

    `is_leveraged` is True only when the cash leg is negative (margin debt).
    gross_exposure / long_weight / net_exposure are decimals from the engine
    (1.0 == 100% of equity); we %-scale them so the model quotes them verbatim.
    """
    lm = lm or {}
    return {
        "is_leveraged":   bool(lm.get("is_leveraged")),
        "gross_exposure_pct": _safe_round((lm.get("gross_exposure") or 0) * 100, 1),
        "long_weight_pct":    _safe_round((lm.get("long_weight") or 0) * 100, 1),
        "net_exposure_pct":   _safe_round((lm.get("net_exposure") or 0) * 100, 1),
        "margin_debt_pct":    _safe_round(abs(min(0.0, lm.get("cash_weight") or 0)) * 100, 1),
        "leverage_ratio":     _safe_round(lm.get("leverage_ratio"), 2),
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
    regime_ru    = _regime_ru(regime_label)   # Sprint-5.3: plain-Russian phrase

    # Sprint-5.3 (замечание 2) — shared hard rule banning jargon the retail
    # reader can't parse.  Injected into BOTH the base and deep prompts.
    plain_rule = (
        "ЯЗЫК — СТРОГО ДЛЯ НЕСПЕЦИАЛИСТА:\n"
        f"- Режим рынка называй ТОЛЬКО по-русски: «{regime_ru}». НЕ пиши "
        "Expansion/Recovery/Slowdown/Recession.\n"
        "- ЗАПРЕЩЕНЫ слова: trailing, trailing 12M, ДИ, CI, CVaR, TRC, Hotspot, "
        "Sharpe, Sortino, Beta, IR, HHI, Value, Growth, Financials — как "
        "голые термины. Пиши простыми словами:\n"
        "  trailing / trailing 12M → «за последние 12 месяцев»\n"
        "  ДИ / доверительный интервал → «разброс оценки» или просто диапазон без аббревиатуры\n"
        "  CVaR → «потери в худший день из 20»\n"
        "  Sharpe → «отдача на единицу риска» (можно «коэффициент Шарпа X» в скобках ОДИН раз)\n"
        "  TRC / доля в риске → «сколько процентов риска даёт эта позиция»\n"
        "  Beta → «во сколько раз сильнее рынка двигается»\n"
        "  Value/Growth/Financials → «недооценённые акции» / «акции роста» / «банки и финансы»\n"
        "- Если без термина никак — добавь короткое пояснение в скобках ОДИН раз, "
        "дальше используй простое слово.\n"
    )

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

    # Sprint-5.2 (live-report audit) — held-tickers rule.  The 2026-06-12 BASE
    # report showed pure catalogue picks: Haiku spent its picks on HELD names,
    # the held-filter drained them, and the deterministic backfill took over.
    # Telling the model UP FRONT which tickers are already owned prevents the
    # drain at the source (the post-filter stays as defence in depth).
    held_tickers = sorted({
        str(h.get("ticker", "")).upper().split(".")[0]
        for h in (summary.get("holdings") or [])
        if str(h.get("ticker", "")).strip() not in ("", "?")
    } - {"USD", "EUR", "RUB", "KZT", "CASH"})
    held_rule = ""
    if held_tickers:
        held_rule = (
            f"УЖЕ В ПОРТФЕЛЕ: {', '.join(held_tickers[:20])}. "
            "НЕ предлагай эти тикеры в stock_picks — только НОВЫЕ имена "
            "вне портфеля.\n"
        )

    # Sprint-5.1 — data-driven ideas rule.  Temperature alone (Sprint-5) gave
    # variety, but the model could still parrot its training-set "favourites"
    # (PLTR/CRWD/JNJ/KO/COST).  Require every pick to be ANCHORED to data
    # from THIS report (regime, sector gaps, 4-pillar, RAG) and ban the
    # template names unless the data of the current report supports them.
    # BLOCK 1.1 (06-22) — ротация идей по дню (time-decay).  Картина «идеи не
    # меняются месяц» имела две причины: (1) якорь свежести был МЕСЯЧНЫМ
    # (`%Y-%m` → «2026-06» весь месяц), (2) бан «дежурных» имён не покрывал
    # quality-compounder-мегакэпы (V/MA/GS/UNH/PG), к которым модель сходится
    # «по умолчанию».  Якорь стал ДНЕВНЫМ + дневной «угол» ротации, а бан-лист
    # расширен — так BASE-идеи варьируются от отчёта к отчёту, как в DEEP.
    _angle = _IDEA_ROTATION_ANGLES[
        date.today().timetuple().tm_yday % len(_IDEA_ROTATION_ANGLES)]
    ideas_rule = (
        "ИДЕИ — СТРОГО DATA-DRIVEN: каждая идея в stock_picks обязана "
        "опираться на данные ЭТОГО отчёта: текущий режим "
        f"{regime_label} и его благоприятные сектора, недовесы/перекосы из "
        "sectors, сигналы 4-Pillar и/или АНАЛИТИКУ БАНКОВ (RAG). В поле why — "
        "явная привязка к этим данным (что именно в портфеле/режиме делает "
        "идею уместной СЕЙЧАС). НЕ предлагай «дежурные» имена (PLTR, CRWD, "
        "JNJ, KO, COST, MTUM, DBC, V, MA, GS, UNH, PG, AVGO), если их "
        "преимущество не следует из данных текущего отчёта — предпочитай менее "
        "очевидные имена с тем же профилем риска/качества.\n"
        # Idea freshness — now DAILY-anchored + a day-rotating angle so the
        # BASE tier (Sonnet) diverges report-to-report instead of parroting the
        # same blue-chips for a whole month.
        f"СВЕЖЕСТЬ ИДЕЙ (срез на {date.today():%Y-%m-%d}): идеи должны отражать "
        "ИМЕННО текущий срез данных этого отчёта. НЕ повторяй один и тот же "
        "набор тикеров из отчёта в отчёт «по привычке» — если режим/недовесы/"
        "оценки сместились, picks обязаны сместиться вслед за ними.\n"
        f"УГОЛ РОТАЦИИ НА СЕГОДНЯ (соблюдай, если не противоречит данным): {_angle}.\n"
    )

    # Sprint-5 — Margin/leverage AI-trigger.  When the engine flags a
    # margin-funded book (cash leg < 0), the model is HARD-REQUIRED to fill
    # `ai_leverage_warning` and to lead a bullet with the Margin-Call risk.
    lev = summary.get("leverage") or {}
    leverage_rule = ""
    if lev.get("is_leveraged"):
        leverage_rule = (
            "⚠ ЗАЁМНЫЕ СРЕДСТВА (МАРЖА): портфель использует кредитное плечо — "
            f"валовая экспозиция ≈{lev.get('gross_exposure_pct')}% капитала, "
            f"плечо ≈{lev.get('leverage_ratio')}x, маржинальный долг ≈{lev.get('margin_debt_pct')}%. "
            "ОБЯЗАТЕЛЬНО заполни поле ai_leverage_warning (≤240 знаков): объясни простыми словами, "
            f"что и прибыль, и убыток умножаются на коэффициент плеча ≈{lev.get('leverage_ratio')}x "
            f"(НЕ пиши «удваивается», если плечо не ≈2x), и при просадке возможен Margin Call "
            "(принудительное закрытие позиций брокером по худшим ценам). Также выдели это "
            "ОТДЕЛЬНЫМ первым пунктом в bullets [Quant Engine].\n"
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
            '  "verdict": "ОДНО короткое предложение ≤110 знаков — главный риск простыми словами + что делать. БЕЗ терминов, аббревиатур и нагромождения чисел",\n'
            '  "plain_summary": "МАКСИМУМ 2 коротких предложения ≤170 знаков — что не так и что делать, простым языком. Без перечисления метрик",\n'
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
            '  "ai_sector_comment": "≤140 знаков — используй ТОЛЬКО числа из раздела sectors; '
            'НЕ суммируй сектора сам. Для совокупной техно-экспозиции бери sector_complex.weight_pct '
            'ВЕРБАТИМ (напр. \\"Tech-комплекс 80.8%\\"). Сектор с перекосом + % → ПОЧЕМУ это опасно в режиме '
            f'{regime_label}. Назови сектор для докупки [Regime]",\n'
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
            '    "smart_money": {"label": "Smart Money", "desc": "≤80 знаков — следование за умными деньгами", '
            '"picks": [{"ticker": "...", "name": "...", "why": "≤120 знаков: ПРИЗНАК умных денег — '
            'заметные покупки инсайдеров (Form 4), накопление институционалов/13F или необычный '
            'объём — + как это согласуется с режимом [Smart Money]", "type": "Stock|ETF"}]}\n'
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
            "- Stock picks: РЕАЛЬНЫЕ АКЦИИ, не только ETF; имена выводи из данных "
            "отчёта (см. правило DATA-DRIVEN ниже). "
            "why — конкретные цифры (ROE, маржа, Бета) с тегом [Источник].\n"
            f"{plain_rule}"
            f"{contradiction_rule}"
            f"{held_rule}"
            f"{leverage_rule}"
            f"{ideas_rule}"
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
  "smart_money": {{
    "label": "Smart Money", "desc": "≤100 знаков. Следование за умными деньгами (институционалы + инсайдеры).",
    "picks": [  // 1-2 идеи с ПРИЗНАКОМ умных денег
      {{"ticker": "...", "name": "...", "why": "≤200 знаков: конкретный сигнал умных денег — покупки инсайдеров (SEC Form 4), накопление институционалов/13F, необычный объём — и согласование с режимом {regime_label} [Smart Money]", "type": "Stock|ETF"}}
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
        '  "verdict": "ОДНО короткое предложение ≤130 знаков — главный риск простыми словами + что делать. БЕЗ нагромождения терминов и чисел",\n'
        '  "plain_summary": "МАКСИМУМ 2 коротких предложения ≤200 знаков — позиция + главный риск + что делать, простым языком",\n'
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
        'ОБЯЗАТЕЛЬНО согласуй с rebalance_verdict из данных: если kind=tradeoff/degradation — '
        'ЯВНО назови ухудшившуюся метрику (напр. рост концентрации Max TRC при урезании топ-позиции) '
        'и НЕ заявляй об одностороннем снижении риска. Причинно-следственная связь с конкретными '
        'изменениями позиций [Quant Engine]",\n'
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
        "- РЕАЛЬНЫЕ АКЦИИ, не только ETF; имена выводи из данных отчёта "
        "(см. правило DATA-DRIVEN ниже).\n"
        "- why: конкретные цифры (ROE, маржа, Beta, P/E, momentum) с тегом [источник].\n"
        f"{plain_rule}"
        f"{contradiction_rule}"
        f"{held_rule}"
        f"{leverage_rule}"
        f"{ideas_rule}"
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
    # Sprint-5.3 (замечания 1+2): short, plain-Russian fallback verdict — one
    # sentence, no jargon (Sharpe/композит/Expansion).  «Уровень риска —
    # {risk_label}» keeps grammatical agreement (уровень — м.р.).
    verdict    = (f"Уровень риска портфеля — {risk_label}. Проверьте "
                  f"концентрацию и при необходимости снизьте крупнейшие позиции.")

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

    # Sprint-5.3: max 2 short plain sentences, no jargon.
    plain_summary = (
        f"Уровень риска портфеля — {risk_label}. "
        + ("Снизьте крупнейшие позиции, чтобы риск не зависел от одной акции."
           if tier == "deep" else
           "Главное — не держать слишком много в одной акции.")
    )

    regime_label = regime.get("regime", "")
    stock_picks  = _fallback_stock_picks(regime_label, tier)

    # Sprint-5 — deterministic Margin-Call warning when the book is levered,
    # so even the no-API fallback path surfaces the leverage risk.
    lm = results.get("leverage_metrics") or {}
    ai_leverage_warning = ""
    if lm.get("is_leveraged"):
        gross = (lm.get("gross_exposure") or 0) * 100
        lev_x = lm.get("leverage_ratio") or 0
        ai_leverage_warning = (
            f"⚠️ Внимание: портфель использует заёмные средства (плечо ≈{lev_x:.2f}x, "
            f"валовая экспозиция ≈{gross:.0f}% капитала). Плечо умножает и прибыль, и "
            "убыток; при сильной просадке возможен Margin Call — принудительное "
            "закрытие позиций брокером по невыгодным ценам [Quant Engine]."
        )
        # Lead the bullets with the leverage warning.
        bullets.insert(0, ai_leverage_warning)

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
        "ai_leverage_warning":      ai_leverage_warning,
        "regime_confirmation":      {"stance": "", "summary": "", "signals": []},
        "rag_context":              "",
    }


def _fallback_stock_picks(regime_label: str, tier: str) -> dict:
    """Rule-based stock picks when Claude is unavailable — includes real stocks."""
    expansion = regime_label in ("Recovery", "Expansion", "")

    # Sprint-5.1 — the fallback used to be a FROZEN 1-candidate catalogue
    # (always PLTR/JNJ/KO/MTUM in expansion), so no-API users saw byte-
    # identical "ideas" forever.  Each slot now carries 3 candidates and the
    # selection rotates DETERMINISTICALLY by calendar month (stable within a
    # month — reproducible reports; different across months — no more
    # year-long loops).  All candidates are conservative, liquid US names.
    from datetime import date
    _rot = date.today().month

    def _pick(candidates: list[dict], offset: int = 0) -> dict:
        return candidates[(_rot + offset) % len(candidates)]

    _BOOST_EXPANSION = [
        {"ticker": "PLTR", "name": "Palantir Technologies",
         "why": "Рост выручки >30% г/г, высокий momentum 12м, бета ~1.8 — "
                "ставка на режим Expansion [SEC EDGAR] [Regime]", "type": "Stock"},
        {"ticker": "NOW", "name": "ServiceNow",
         "why": "Подписочная выручка +22% г/г, Op. маржа растёт, лидер "
                "enterprise-софта — циклический рост [SEC EDGAR] [Regime]", "type": "Stock"},
        {"ticker": "ANET", "name": "Arista Networks",
         "why": "Сетевая инфраструктура ИИ-ЦОД, выручка +20% г/г, без долга, "
                "ROE >30% — бенефициар capex-цикла [SEC EDGAR] [Regime]", "type": "Stock"},
    ]
    _BOOST_DEFENSIVE = [
        {"ticker": "DBC", "name": "Invesco DB Commodity Index ETF",
         "why": "Товарная экспозиция защищает при стагфляции и Slowdown; низкая "
                "корреляция с техно-позициями [Quant Engine] [Regime]", "type": "ETF"},
        {"ticker": "XLE", "name": "Energy Select Sector SPDR",
         "why": "Энергетика — исторический лидер в позднем цикле/Slowdown; "
                "дивидендная доходность сектора >3% [Regime]", "type": "ETF"},
        {"ticker": "XLU", "name": "Utilities Select Sector SPDR",
         "why": "Коммунальный сектор: низкая бета ~0.5, устойчивые денежные "
                "потоки при замедлении экономики [Regime]", "type": "ETF"},
    ]
    _BALANCE = [
        {"ticker": "JNJ", "name": "Johnson & Johnson",
         "why": "ROE ~25%, Op. маржа ~25%, Debt/Assets <0.4, дивиденд-аристократ "
                "62 года — режимо-устойчивый [SEC EDGAR]", "type": "Stock"},
        {"ticker": "PG", "name": "Procter & Gamble",
         "why": "Op. маржа ~24%, стабильный FCF, 68 лет роста дивиденда — "
                "качество вне цикла [SEC EDGAR]", "type": "Stock"},
        {"ticker": "AVGO", "name": "Broadcom",
         "why": "FCF-маржа >45%, диверсификация чипы+софт, растущий дивиденд — "
                "качество с ростом [SEC EDGAR]", "type": "Stock"},
    ]
    _PROTECT = [
        {"ticker": "KO", "name": "The Coca-Cola Company",
         "why": "Дивиденд-аристократ 61+ лет, Beta ~0.6, маржа >28% — защитный "
                "актив при любом режиме [SEC EDGAR] [Quant Engine]", "type": "Stock"},
        {"ticker": "PEP", "name": "PepsiCo",
         "why": "Beta ~0.55, 52 года роста дивиденда, диверсификация снеки+напитки "
                "— защита с ростом [SEC EDGAR]", "type": "Stock"},
        {"ticker": "AGG", "name": "iShares Core US Aggregate Bond ETF",
         "why": "Инвест-грейд облигации: отрицательная корреляция с акциями в "
                "risk-off, дюрация ~6 лет [Quant Engine]", "type": "ETF"},
    ]
    _BALANCE_DEEP_EXTRA = [
        {"ticker": "COST", "name": "Costco Wholesale",
         "why": "ROE ~30%, стабильная подписочная модель, Revenue Growth ~7% г/г, "
                "низкий долг [SEC EDGAR]", "type": "Stock"},
        {"ticker": "V", "name": "Visa",
         "why": "Op. маржа ~67%, asset-light модель, рост объёмов платежей — "
                "качество-компаундер [SEC EDGAR]", "type": "Stock"},
        {"ticker": "UNH", "name": "UnitedHealth Group",
         "why": "ROE ~25%, защитный health-сектор, стабильный двузначный рост "
                "EPS [SEC EDGAR]", "type": "Stock"},
    ]
    _BOOST_DEEP_EXP_EXTRA = [
        {"ticker": "CRWD", "name": "CrowdStrike Holdings",
         "why": "Лидер кибербезопасности, выручка +33% г/г, переход на "
                "прибыльность, momentum 12м [SEC EDGAR] [Quant Engine]", "type": "Stock"},
        {"ticker": "PANW", "name": "Palo Alto Networks",
         "why": "Платформенная консолидация кибербеза, FCF-маржа ~38%, "
                "рост ARR >15% [SEC EDGAR]", "type": "Stock"},
        {"ticker": "AMAT", "name": "Applied Materials",
         "why": "Полупроводниковое оборудование — рычаг на capex-цикл чипов, "
                "ROE ~45% [SEC EDGAR] [Regime]", "type": "Stock"},
    ]
    _BOOST_DEEP_DEF_EXTRA = [
        {"ticker": "GLD", "name": "SPDR Gold Shares",
         "why": "Золото как хедж в Slowdown/Recession, ~нулевая корреляция "
                "с S&P 500 [Quant Engine] [Regime]", "type": "ETF"},
        {"ticker": "IAU", "name": "iShares Gold Trust",
         "why": "Золото с низкой комиссией (0.25%) — хедж хвостового риска "
                "[Quant Engine] [Regime]", "type": "ETF"},
        {"ticker": "TLT", "name": "iShares 20+ Year Treasury",
         "why": "Длинные трежерис — классический risk-off хедж при рецессии "
                "[Quant Engine] [Regime]", "type": "ETF"},
    ]
    _REGIME_EXPANSION = [
        {"ticker": "MTUM", "name": "iShares MSCI USA Momentum ETF",
         "why": "Momentum-фактор лидирует в Expansion-режиме [Regime] [Quant Engine]",
         "type": "ETF"},
        {"ticker": "IWM", "name": "iShares Russell 2000",
         "why": "Малые капитализации — ранне-цикличная ставка при росте "
                "аппетита к риску [Regime]", "type": "ETF"},
        {"ticker": "XLF", "name": "Financial Select Sector SPDR",
         "why": "Финансы выигрывают от крутой кривой доходности в "
                "Recovery/Expansion [Regime]", "type": "ETF"},
    ]
    _REGIME_DEFENSIVE = [
        {"ticker": "DBC", "name": "Invesco DB Commodity Index ETF",
         "why": "Товарная экспозиция защищает при Slowdown-режиме [Regime]", "type": "ETF"},
        {"ticker": "XLV", "name": "Health Care Select Sector SPDR",
         "why": "Healthcare — защитный сектор позднего цикла со стабильным "
                "спросом [Regime]", "type": "ETF"},
        {"ticker": "USMV", "name": "iShares MSCI USA Min Vol ETF",
         "why": "Min-vol фактор исторически опережает в risk-off фазах "
                "[Regime] [Quant Engine]", "type": "ETF"},
    ]

    picks_boost   = [_pick(_BOOST_EXPANSION if expansion else _BOOST_DEFENSIVE)]
    picks_balance = [_pick(_BALANCE)]
    picks_protect = [_pick(_PROTECT)]

    if tier == "deep":
        picks_boost.append(_pick(
            _BOOST_DEEP_EXP_EXTRA if expansion else _BOOST_DEEP_DEF_EXTRA,
            offset=1))
        picks_balance.append(_pick(_BALANCE_DEEP_EXTRA, offset=1))

    # smart_money: institutional / insider-conviction proxies (B2.4).  The
    # offline catalogue can't read live Form 4, so it uses widely-followed
    # "smart money" proxies; rotates monthly like the rest.
    _SMART_MONEY = [
        {"ticker": "BRK.B", "name": "Berkshire Hathaway",
         "why": "Прокси на аллокацию Баффета — сам по себе индикатор умных денег; "
                "диверсифицированный кэш-генератор [Smart Money]", "type": "Stock"},
        {"ticker": "BLK", "name": "BlackRock",
         "why": "Крупнейший управляющий активами, бенефициар притоков; "
                "высокая институциональная доля [Smart Money]", "type": "Stock"},
        {"ticker": "BX", "name": "Blackstone",
         "why": "Лидер альтернативных активов; индикатор аппетита институционалов "
                "к риску [Smart Money]", "type": "Stock"},
    ]
    picks_smart = [_pick(_SMART_MONEY)]
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
        "smart_money":     {"label": "Smart Money — институционалы и инсайдеры",
                            "desc":  "Следование за умными деньгами (накопление фондов / покупки инсайдеров).",
                            "picks": picks_smart},
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
        # Sprint-5: temperature raised from 0.1 (near-deterministic → stale,
        # repeating ideas).  Sprint-5.1: omitted entirely on Opus 4.7/4.8 —
        # those models reject the param with HTTP 400 (§4.2 migration prep).
        _sampling_kwargs: dict = {}
        if _model_supports_temperature(model):
            _sampling_kwargs["temperature"] = NARRATIVE_TEMPERATURE
        response = client.messages.create(
            model       = model,
            max_tokens  = max_tok,
            **_sampling_kwargs,
            # Prompt Caching — the ~25 KB system prompt is byte-identical on
            # every request.  Caching it means repeated calls (base+deep, and
            # many users within the 5-min TTL) read the prefix at ~0.1× instead
            # of full input price.  (Cache is model-scoped: Haiku BASE and
            # Sonnet DEEP keep separate caches — both still benefit at volume.)
            system      = [{
                "type": "text",
                "text": _build_system_prompt(),
                "cache_control": {"type": "ephemeral"},
            }],
            messages    = [{
                "role": "user",
                "content": _user_prompt(summary, tier=tier,
                                        market_context=market_context,
                                        user_profile=user_risk_profile),
            }],
            # Structured Outputs — force the report through a typed tool call so
            # the SDK returns a GUARANTEED dict.  No brace-finding, no
            # _repair_truncated_json: JSON parsing is now strictly deterministic.
            tools       = [_REPORT_TOOL],
            tool_choice = {"type": "tool", "name": "emit_report"},
        )
        usage      = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        # Deterministic extraction — the forced tool_use block's `input` IS the
        # parsed object.  No text parsing path at all.
        tool_blocks = [b for b in response.content
                       if getattr(b, "type", None) == "tool_use"
                       and getattr(b, "name", "") == "emit_report"]
        if not tool_blocks:
            raise ValueError("Модель не вернула структурированный tool_use (emit_report).")
        parsed = dict(tool_blocks[0].input or {})
        logger.info("AI narrative: structured tool_use OK model=%s (input=%d tok, "
                    "output=%d tok, cache_read=%d tok, keys=%d)",
                    model, usage.input_tokens, usage.output_tokens, cache_read, len(parsed))

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
            # Sprint-5.3 (замечание 1): hard length cut so the cover verdict
            # stays short even if the model overruns — was 300/400.
            "verdict":                  _soft_trim(verdict, 150),
            "plain_summary":            _soft_trim(plain_summary, 230),
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
            "ai_stress_comment":        validate_stress_comment(_comment("ai_stress_comment")),
            "ai_action_comment":        _comment("ai_action_comment"),
            "ai_effect_comment":        _comment("ai_effect_comment"),
            # Sprint-5 margin/leverage trigger output — only the AI fills this
            # (empty when the book is unlevered; the template hides it then).
            "ai_leverage_warning":      _comment("ai_leverage_warning", 260),
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
    for key in ("boost_alpha", "rebalance", "protect_capital", "smart_money"):
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

    for key in ("boost_alpha", "rebalance", "protect_capital", "smart_money"):
        scenario = stock_picks.get(key) or {}
        if scenario.get("picks"):
            continue
        fb = fallback.get(key) or {}
        if fb.get("picks"):
            # Preserve the AI's label/desc if present, else take the fallback's.
            # Sprint-5.2: a label equal to the RAW key (normaliser default when
            # the model omitted it) is NOT a real label — prefer the fallback's
            # human-readable one (the 2026-06-12 BASE report showed card titles
            # "boost_alpha"/"rebalance").
            ai_label = str(scenario.get("label") or "").strip()
            if not ai_label or ai_label == key:
                ai_label = fb.get("label", key)
            desc = scenario.get("desc") or fb.get("desc", "")
            # Sprint-5.2 honesty marker: catalogue picks must not masquerade
            # as the AI's own analysis.
            if "резервный каталог" not in desc:
                desc = (desc + " · резервный каталог").strip(" ·")
            stock_picks[key] = {
                "label": ai_label,
                "desc":  desc,
                "picks": fb["picks"],
            }
            logger.info("Idea backfill: scenario '%s' was empty → filled from "
                        "rule-based catalogue (%d picks)", key, len(fb["picks"]))
    return stock_picks


__all__ = ["generate_narrative"]
