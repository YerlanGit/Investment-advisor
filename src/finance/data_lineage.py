"""
Runtime CoVe (Chain-of-Verification) data lineage.

Replaces the hard-coded "verified sources" block in report_deep.html with
a data-driven dict the renderer can iterate.  For every metric category
reported on the cover page, we surface:

  • where the number came from (source tag),
  • how it was computed (one-line method),
  • whether the source was actually consulted on THIS run (status flag),
  • how fresh the underlying observation is.

Status taxonomy (mirrors services.macro_data.MacroFeed so the UI uses ONE
visual vocabulary across pipeline modules):

  ok       Source consulted, data within freshness window, no flags
  warn     Source consulted but partial data (e.g. SEC missed N tickers,
            CDS feed has stale gate trips, etc.)
  stale    Cache served / data older than freshness window — still usable
            but should be displayed with a "⚠" badge
  missing  Source intentionally not consulted (e.g. FRED_API_KEY unset)
  error    Source FAILED to deliver any usable data on this run

Design constraints
──────────────────
1. Sklearn-free — runs in any environment.
2. NO additional fetches.  The function reads ONLY the dict already
   produced by analyze_all() plus a couple of helper columns from perf_df.
3. Stable schema.  Every row carries the same keys so the renderer can
   loop without conditionals.
"""
from __future__ import annotations

import math
import os
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd


def _lookback_days() -> int:
    """Актуальное окно истории (кал. дней) из env — чтобы CoVe-лейбл
    «Nd window» не расходился с реальным `HISTORY_LOOKBACK_DAYS`."""
    try:
        return max(90, min(3650, int(os.getenv("HISTORY_LOOKBACK_DAYS", "1825"))))
    except (TypeError, ValueError):
        return 1825


# Per-source freshness thresholds (calendar days).
SEC_FILING_WARN_MONTHS    = 18           # 10-K older than ~1.5y → warn
SEC_FILING_STALE_MONTHS   = 30           # > 2.5y → stale
TRADERNET_FRESH_CAL_DAYS  = 5            # ≈ 3 trading days


# ── Helpers ──────────────────────────────────────────────────────────────────

def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _parse_iso_date(s) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def _months_between(iso_date_str, ref: Optional[date] = None) -> Optional[int]:
    d = _parse_iso_date(iso_date_str)
    if d is None:
        return None
    r = ref or _today_utc()
    return (r.year - d.year) * 12 + (r.month - d.month)


def _row(name: str, source: str, method: str, status: str, *,
          as_of:           Optional[str] = None,
          freshness_days:  Optional[int] = None,
          note:            str           = "") -> dict:
    return {
        "name":           name,
        "source":         source,
        "method":         method,
        "status":         status,
        "as_of":          as_of,
        "freshness_days": freshness_days,
        "note":           note,
    }


# ── Per-source checkers ──────────────────────────────────────────────────────

def _tradernet_status(results: dict, today: date) -> dict:
    """Last close-date freshness from history_result.data."""
    history = results.get("history_result")
    data    = getattr(history, "data", None) if history is not None else None
    if data is None or len(data) == 0:
        return _row(
            name   = "Цены и история активов",
            source = "Tradernet (Freedom Broker)",
            method = f"Daily CLOSE · {_lookback_days()}d window",
            status = "error",
            note   = "no price data loaded",
        )
    last_dt = pd.to_datetime(data.index[-1]).date()
    age = (today - last_dt).days
    if age > TRADERNET_FRESH_CAL_DAYS:
        status, note = "stale", f"last close {age} cal days old"
    else:
        status, note = "ok", ""
    return _row(
        name   = "Цены и история активов",
        source = "Tradernet (Freedom Broker)",
        method = f"Daily CLOSE · {_lookback_days()}d window · ATR via OHLC (fallback |ΔClose|)",
        status = status,
        as_of  = last_dt.isoformat(),
        freshness_days = age,
        note   = note,
    )


def _sec_status(results: dict, today: date) -> list[dict]:
    """
    Two rows: fundamental Z-scores + financial-health metrics.
    Both depend on SEC EDGAR CompanyFacts JSON.
    Status downgrades based on:
      - number of skipped tickers (default/EM_Proxy sector)
      - age of the OLDEST 10-K filing in the universe
    """
    perf = results.get("performance_table")
    skipped: list[str] = []
    oldest_age_m: Optional[int] = None
    oldest_ticker: Optional[str] = None

    if perf is not None and not perf.empty:
        if "Fundamental_Sector" in perf.columns:
            mask = perf["Fundamental_Sector"].astype(str).isin(["default", "EM_Proxy"])
            skipped = perf.loc[mask, "Ticker"].astype(str).tolist()
        if "SEC_Filing_Date" in perf.columns:
            for _, row in perf.iterrows():
                fd = row.get("SEC_Filing_Date")
                age_m = _months_between(fd, today)
                if age_m is None:
                    continue
                if oldest_age_m is None or age_m > oldest_age_m:
                    oldest_age_m = age_m
                    oldest_ticker = str(row.get("Ticker"))

    # Build the status flag.
    notes: list[str] = []
    status = "ok"
    if skipped:
        notes.append(f"{len(skipped)} тикеров без SEC покрытия")
        status = "warn"
    if oldest_age_m is not None:
        if oldest_age_m >= SEC_FILING_STALE_MONTHS:
            notes.append(f"{oldest_ticker} 10-K {oldest_age_m} мес. (просрочено)")
            status = "stale"
        elif oldest_age_m >= SEC_FILING_WARN_MONTHS:
            notes.append(f"{oldest_ticker} 10-K {oldest_age_m} мес. (устарело)")
            status = "warn" if status == "ok" else status

    note_str = " · ".join(notes)
    return [
        _row(
            name   = "Fundamental Z-scores",
            source = "SEC EDGAR CompanyFacts",
            method = "10-K FY · sector-normalised MAD · Group B factors",
            status = status,
            note   = note_str,
        ),
        _row(
            name   = "Altman-Z · Piotroski-F · Interest Coverage",
            source = "SEC EDGAR CompanyFacts",
            method = "разностный расчёт по балансу и P&L (annual filings)",
            status = status,
            note   = note_str,
        ),
    ]


def _cds_status(results: dict) -> dict:
    """
    Read the CDS coverage summary attached by analyze_all.

    `cds_summary` shape: {enabled, checked, loaded, gated_out}.
      • enabled=False           → CDS_DISABLED=1 or feed import failed
      • enabled=True, loaded=0  → all tickers gated out (or no coverage)
      • loaded>0, gated_out>0   → partial coverage (warn)
      • loaded>0, gated_out=0   → full coverage (ok)
    """
    cds_summary = results.get("cds_summary") or {}
    if not cds_summary or cds_summary.get("enabled") is False:
        return _row(
            name   = "CDS spreads (credit signal)",
            source = "FRED HY proxy + WGB sovereign",
            method = "QualityGate: sanity 1–3000 bps · ≤ 3 trading days",
            status = "missing",
            note   = "CDS_DISABLED=1" if cds_summary.get("enabled") is False
                     else "no CDS summary attached",
        )
    n_loaded  = int(cds_summary.get("loaded",    0) or 0)
    n_gated   = int(cds_summary.get("gated_out", 0) or 0)
    n_checked = int(cds_summary.get("checked",   0) or 0)
    if n_loaded == 0 and n_checked > 0:
        status = "missing"
        note   = f"0/{n_checked} tickers cleared the gate"
    elif n_gated > 0:
        status = "warn"
        note   = f"{n_loaded}/{n_checked} loaded · {n_gated} gated out"
    elif n_loaded > 0:
        status = "ok"
        note   = f"{n_loaded}/{n_checked} tickers"
    else:
        status = "missing"
        note   = "no tickers checked"
    return _row(
        name   = "CDS spreads (credit signal)",
        source = "FRED HY proxy + WGB sovereign",
        method = "QualityGate: sanity 1–3000 bps · ≤ 3 trading days",
        status = status,
        note   = note,
    )


def _macro_status(results: dict) -> list[dict]:
    """
    One row per FRED series in macro_drivers — reuses the status field the
    MacroFeed already produced (ok / stale / missing / error).
    """
    macro = results.get("macro_drivers") or {}
    if not macro:
        return [_row(
            name   = "Макро-драйверы (FRED)",
            source = "FRED St. Louis Fed",
            # BLOCK 3.4: PMI is discontinued on FRED; the pack is now yield
            # curve · HY · VIX · breakeven · unemployment · real-GDP growth.
            method = "yield curve · HY spread · VIX · breakeven · "
                     "unemployment · GDP growth",
            status = "missing",
            note   = "FRED_API_KEY not set (see .env.template)",
        )]
    rows: list[dict] = []
    for key, m in macro.items():
        rows.append(_row(
            name   = m.get("label", key),
            source = f"FRED · {m.get('series_id', '?')}",
            method = f"{m.get('publish_cadence', 'daily')} · QualityGate "
                     f"freshness + sanity range",
            status         = m.get("status", "error"),
            as_of          = m.get("as_of"),
            freshness_days = m.get("freshness_days"),
            note           = m.get("note", ""),
        ))
    return rows


def _action_levels_status(results: dict) -> dict:
    plan = results.get("action_plan") or []
    if not plan:
        return _row(
            name   = "Action levels (Buy / Sell / Stop)",
            source = "Quant Engine",
            method = "ATR Wilder RMA + SMA50/200 + RSI(14) + MACD(12,26,9)",
            status = "missing",
            note   = "action plan not computed for this run",
        )
    return _row(
        name   = "Action levels (Buy / Sell / Stop)",
        source = "Quant Engine",
        method = "ATR Wilder RMA + SMA50/200 + RSI(14) + MACD(12,26,9)",
        status = "ok",
        note   = f"{len(plan)} positions",
    )


def _bl_status(results: dict) -> dict:
    bl = results.get("black_litterman")
    if not bl:
        return _row(
            name   = "Black-Litterman target weights",
            source = "Quant Engine",
            method = "Reverse-optimisation prior + score-derived views",
            status = "missing",
            note   = "BL skipped (no scores / empty cov)",
        )
    return _row(
        name   = "Black-Litterman target weights",
        source = "Quant Engine",
        method = "reverse-optimisation prior + score-derived views, "
                 "τ=0.05",
        status = "ok",
        note   = f"{len(bl)} positions reweighted",
    )


def _regime_status(results: dict) -> dict:
    regime = results.get("regime")
    if not regime:
        return _row(
            name   = "Регим-классификатор",
            source = "Quant Engine",
            method = "Growth × Cycle factor returns · 60-day window",
            status = "missing",
        )
    conf = int(round(float(regime.get("confidence", 0)) * 100))
    return _row(
        name   = "Регим-классификатор",
        source = "Quant Engine",
        method = "Growth × Cycle factor returns · 60-day window",
        status = "ok",
        note   = f"{regime.get('regime')} · confidence {conf}%",
    )


def _stress_status(results: dict) -> dict:
    rows = results.get("stress_scenarios") or []
    if not rows:
        return _row(
            name   = "Стресс-сценарии",
            source = "Quant Engine",
            method = "parametric factor shocks · per-asset β · linear PnL",
            status = "missing",
        )
    proxies = sum(1 for r in rows if r.get("coverage") == "proxy")
    return _row(
        name   = "Стресс-сценарии",
        source = "Quant Engine",
        method = "parametric factor shocks · per-asset β · linear PnL",
        status = "ok" if proxies == 0 else "warn",
        note   = (f"{len(rows)} сценариев"
                  + (f" · {proxies} proxy" if proxies else "")),
    )


def _rag_status(ai_summary: Optional[dict]) -> dict:
    """Retrieval row — proves HOW MUCH bank research the KB holds and whether it
    was actually read on this run (docs/chunks inventory + retrieved snippets),
    using the 3-state rag_status instead of the old misleading binary."""
    a = ai_summary or {}
    rag_status = a.get("rag_status")
    if rag_status is None:   # back-compat with summaries predating rag_status
        rag_status = "used" if a.get("used_rag") else "missing"
    docs   = int(a.get("rag_kb_docs", 0) or 0)
    chunks = int(a.get("rag_kb_chunks", 0) or 0)
    ctx    = a.get("rag_context") or ""
    snippets = sum(1 for p in ctx.split("\n\n") if p.strip()) if ctx else 0
    kb = f"база: {docs} отчётов · {chunks} чанков"
    if rag_status == "used":
        status, note = "ok", f"прочитано {snippets} отрывков · {kb}"
    elif rag_status == "no_match":
        status, note = "warn", f"{kb} · релевантных под портфель не найдено"
    else:  # unavailable / missing — база пуста или недоступна
        status = "missing"
        note   = (f"{kb} — база пуста/недоступна, отчёты не читались"
                  if chunks == 0 else f"{kb} · RAG не запрашивался")
    return _row(
        name   = "Bank RAG (выдержки)",
        source = "ChromaDB · GS / MS / JPM PDF reports",
        method = "cosine similarity retrieval (semantic 0.6 ⊕ recency 0.4)",
        status = status,
        note   = note,
    )


def _rag_citation_status(ai_summary: Optional[dict]) -> dict:
    """Audit whether the NARRATIVE actually referenced bank research, and of
    what kind — a verified [RAG:file] citation (report was truly read) vs. bank
    consensus the model surfaced from its own training memory (allowed by the
    prompt even with an empty KB, but NOT proof a report was read).  This is the
    checker the owner asked for: «если ИИ ссылается на отчёты — это чекается»."""
    a = ai_summary or {}
    have_ai = bool(a.get("verdict") or a.get("bullets"))
    name   = "ИИ-цитирование банк-аналитики"
    source = "CoVe · аудит ссылок нарратива"
    method = "[RAG:файл] (проверено RAG) vs. банк-консенсус из знаний модели"
    if not have_ai:
        return _row(name=name, source=source, method=method,
                    status="missing", note="AI не вызывался")
    rag_status = a.get("rag_status") or ("used" if a.get("used_rag") else "missing")
    used       = rag_status == "used"
    file_cites = int(a.get("rag_file_citations", 0) or 0)
    bank_cites = int(a.get("rag_bank_citations", 0) or 0)
    banks      = [str(b) for b in (a.get("rag_cited_banks") or [])]
    banks_str  = ", ".join(banks[:4])
    if file_cites > 0:
        note = f"{file_cites} проверенных [RAG]-цитат из ингестированных отчётов"
        note += f" · {banks_str}" if banks_str else ""
        status = "ok"
    elif bank_cites > 0 and used:
        status = "ok"
        note   = f"ИИ сослался на {bank_cites} банк(ов) ({banks_str}) · база RAG активна"
    elif bank_cites > 0 and not used:
        status = "warn"
        note   = (f"ИИ сослался на {bank_cites} банк(ов) ({banks_str}) из общих знаний "
                  "модели — база RAG пуста, цитаты НЕ подтверждены отчётами")
    else:
        status = "ok"
        note   = "модель не ссылалась на банк-аналитику (источники — Quant Engine/SEC/FRED)"
    return _row(name=name, source=source, method=method, status=status, note=note)


def _ai_status(ai_summary: Optional[dict]) -> dict:
    a = ai_summary or {}
    if not a.get("verdict") and not a.get("bullets"):
        return _row(
            name   = "AI verdict · bullets",
            source = "Anthropic Claude (Sonnet/Opus)",
            method = "advisory only; verdict + plain summary + bullets",
            status = "missing",
            note   = "ANTHROPIC_API_KEY missing or AI call failed",
        )
    # Sprint-5.2: prefixing the RAW model id produced "Anthropic Claude
    # claude-sonnet-4-6" in the prod CoVe panel — use the display name.
    model_id = a.get("model_used") or ""
    try:
        from pdf_payload import _model_display_name
        model_disp = _model_display_name(model_id) or model_id
    except Exception:
        model_disp = model_id
    return _row(
        name   = "AI verdict · bullets",
        source = ("Anthropic · " + model_disp) if model_disp else "Anthropic Claude",
        method = "advisory only; verdict + plain summary + bullets",
        status = "ok",
        note   = "не является ИИР (индивидуальной инвест. рекомендацией)",
    )


def _factor_diagnostic_status(results: dict) -> dict:
    """BLOCK 4.6 — factor-multicollinearity diagnostic row.

    Surfaces κ (condition number) + max|corr| of the factor set so a reader can
    see the structural model was CHECKED for double-counting, not just trusted.
    """
    pm = results.get("portfolio_metrics") or {}
    fd = pm.get("factor_diagnostics") or {}
    if not fd:
        return _row(
            name   = "Факторная независимость (мультиколлинеарность)",
            source = "Quant Engine MAC3",
            method = "корр. факторов: κ (condition number) + max|corr|",
            status = "missing",
            note   = "структурная модель не построена на этом прогоне",
        )
    near  = bool(fd.get("near_collinear"))
    ortho = bool(fd.get("orthogonalized"))
    # BLOCK 3.5: if the hierarchical orthogonalization is ON, a high raw κ is
    # already remediated → treat as OK and label it; otherwise warn on near-
    # collinearity so the operator can enable FACTOR_ORTHOGONALIZE.
    return _row(
        name   = "Факторная независимость (мультиколлинеарность)",
        source = "Quant Engine MAC3",
        method = ("Σ=B·F·Bᵀ+D; "
                  + ("иерархическая ортогонализация (style→core) ON · " if ortho
                     else "диагностика ")
                  + "κ + max|corr|"),
        status = "ok" if (ortho or not near) else "warn",
        note   = (f"факторов {fd.get('n_factors')} · max|corr|={fd.get('max_abs_corr')} "
                  f"· κ={fd.get('condition_number')}"
                  + (" · ортогонализовано" if ortho
                     else (" · близки к коллинеарности (вкл. FACTOR_ORTHOGONALIZE)" if near else ""))),
    )


def _llm_checker_status(ai_summary: Optional[dict],
                        leveraged: bool = False) -> list[dict]:
    """BLOCK 4.8 — explicit LLM verification rows.

    The narrative is advisory, but it passes through deterministic CHECKERS
    before it reaches the user; CoVe must show them so "the AI said so" is
    never the end of the audit trail:

      • Hallucination guard — held-tickers filter (no recommending owned
        names), DATA-DRIVEN ideas rule, and the pick-contradiction filter.
      • Math verification    — leverage-ratio phrasing rule (no "doubles" at
        1.16x), the stress convex-cap mention validator, and the
        no-self-aggregation directive on sector totals.
    """
    a = ai_summary or {}
    have_ai = bool(a.get("verdict") or a.get("bullets"))
    status  = "ok" if have_ai else "missing"
    # Honesty (audit 06-23): these rows reflect that the post-LLM filters are
    # CONFIGURED in the pipeline, not a per-run pass/trip count — so the copy
    # says "настроены" (configured), not "активны" (which over-claimed that they
    # ran AND caught something on this specific run).
    note_h  = ("held-filter + data-driven + contradiction-filter настроены"
               if have_ai else "AI не вызывался")
    # Leverage/debt phrasing is HIDDEN unless the book is actually margin-funded
    # (cash balance < 0): on an unlevered portfolio the «плечо ≈Nx» validator is
    # not applicable, and the rule is to keep every leverage/debt term out of the
    # report when cash is non-negative.  The other two math validators always show.
    _checks_m = (["плечо ≈Nx (без «удваивается»)"] if leveraged else []) + [
        "упоминание выпуклого капа стресса", "запрет ре-агрегации секторов"]
    _note_validators = ("leverage-phrasing + " if leveraged else "") + \
        "stress-cap + no-self-aggregation настроены"
    note_m  = (_note_validators if have_ai else "AI не вызывался")
    return [
        _row(
            name   = "LLM-чекер: контроль галлюцинаций",
            source = "CoVe · post-LLM фильтры",
            method = "held-tickers · DATA-DRIVEN идеи · фильтр противоречий пиков",
            status = status,
            note   = note_h,
        ),
        _row(
            name   = "LLM-чекер: проверка вычислений",
            source = "CoVe · валидаторы нарратива",
            method = " · ".join(_checks_m),
            status = status,
            note   = note_m,
        ),
    ]


def _smart_money_status(results: dict) -> dict:
    """BLOCK 4.8/3.5 — insider (SEC Form 4) lineage row (gated)."""
    try:
        from finance.smart_money import insider_lineage_row
        return insider_lineage_row(results.get("smart_money"))
    except Exception:
        return _row(
            name   = "Smart-Money (инсайдеры SEC Form 4)",
            source = "SEC EDGAR · Form 4",
            method = "90д нетто-поток + кластер покупок",
            status = "missing",
            note   = "слой инсайдеров выключен (по умолчанию)",
        )


def _fx_status(results: dict) -> dict:
    """Sprint-5.4 — currency layer (Base Currency Approach).

    Every price is FX-converted to the reporting currency BEFORE returns and
    covariance, and Sharpe/Sortino use the matching risk-free rate.  This core
    transform sits behind every number in the report yet had no CoVe row — add
    it so the data-lineage picture is complete.  Data: portfolio_metrics
    .fx_conversion / .reporting_currency / .risk_free_rate_source.
    """
    pm      = results.get("portfolio_metrics") or {}
    ccy     = pm.get("reporting_currency") or "USD"
    rfr_src = pm.get("risk_free_rate_source") or "—"
    rfr_ann = pm.get("risk_free_rate_annual")
    rfr_txt = (f"ставка {rfr_ann * 100:.2f}%/год"
               if isinstance(rfr_ann, (int, float)) else "ставка —")
    fx = pm.get("fx_conversion") or []

    if not fx:
        # No conversion records → every asset already priced in `ccy`.
        return _row(
            name   = "Валютный слой (конверсия + ставка)",
            source = f"Base Currency = {ccy} · {rfr_src}",
            method = f"конверсия не требуется (все активы в {ccy}) · {rfr_txt}",
            status = "ok",
            note   = "цены и безрисковая ставка в одной валюте",
        )

    pairs    = ", ".join(str(r.get("pair", "?")) for r in fx[:4])
    fallback = any(r.get("fallback_used") for r in fx)
    min_cov  = min((float(r.get("coverage_pct") or 0) for r in fx), default=100.0)
    status   = "warn" if (fallback or min_cov < 90.0) else "ok"
    notes    = []
    if fallback:
        notes.append("по части дней — T-1 фолбэк курса")
    notes.append(f"покрытие курса ≥{min_cov:.0f}%" if min_cov < 100 else "полное покрытие курса")
    return _row(
        name   = "Валютный слой (конверсия + ставка)",
        source = f"FX-провайдер (FRED) · Base = {ccy} · {rfr_src}",
        method = (f"цены × курс до расчёта риска (лаг T-1) · {pairs} · {rfr_txt}"),
        status = status,
        note   = "; ".join(notes),
    )


# ── Public API ───────────────────────────────────────────────────────────────

def build_lineage(results: dict,
                   ai_summary: Optional[dict] = None,
                   *, today: Optional[date] = None) -> list[dict]:
    """
    Assemble the runtime CoVe table.  Order matters — this is the order
    the report will render.
    """
    today = today or _today_utc()
    rows: list[dict] = []

    # Quant Engine — always-on baseline.
    rows.append(_row(
        name   = "Vol · CVaR · TE · IR · Max DD",
        source = "Quant Engine MAC3",
        method = "Wilder RMA · EWMA hl=63 (λ≈0.99) ⊕ Ledoit-Wolf 70/30 · "
                 "Politis-Romano bootstrap CI",
        status = "ok",
    ))
    rows.append(_row(
        name   = "TRC (Euler decomposition) · MCTR · CVaR",
        source = "Quant Engine MAC3",
        method = "MCTR = Σw/σ_p · ERC%_i = w_i·MCTR_i/σ_p",
        status = "ok",
    ))

    # BLOCK 4.6: factor-independence diagnostic sits next to the risk engine
    # whose betas it qualifies.
    rows.append(_factor_diagnostic_status(results))

    # Прайсы + ATR + benchmark frame.
    rows.append(_tradernet_status(results, today))

    # Sprint-5.4: currency layer (FX conversion + risk-free rate) — sits right
    # after the price source it transforms.
    rows.append(_fx_status(results))

    # SEC EDGAR (2 rows).
    rows.extend(_sec_status(results, today))

    # CDS (when engine exposes the gate summary).
    rows.append(_cds_status(results))

    # Action plan / BL / regime / stress / RAG / AI.
    rows.append(_action_levels_status(results))
    rows.append(_bl_status(results))
    rows.append(_regime_status(results))
    rows.append(_stress_status(results))

    # FRED macro drivers (variable rows depending on catalog).
    rows.extend(_macro_status(results))

    # BLOCK 3.5/4.8: Smart-Money (insider Form-4) layer — gated, shows
    # "missing" until SMART_MONEY_INSIDERS=1 wires a provider.
    rows.append(_smart_money_status(results))

    # Bank RAG (retrieval inventory) + AI-citation audit + AI narrative.
    rows.append(_rag_status(ai_summary))
    rows.append(_rag_citation_status(ai_summary))
    rows.append(_ai_status(ai_summary))

    # BLOCK 4.8: explicit LLM verification rows — the narrative's
    # hallucination + math checkers are part of the audit trail.
    _lev = bool((results.get("leverage_metrics") or {}).get("is_leveraged"))
    rows.extend(_llm_checker_status(ai_summary, leveraged=_lev))

    return rows


__all__ = [
    "build_lineage",
    "SEC_FILING_WARN_MONTHS",
    "SEC_FILING_STALE_MONTHS",
    "TRADERNET_FRESH_CAL_DAYS",
]
