"""
HTML Report Renderer — Jinja2-only (no Chromium, no PDF).

Replaces the previous pdf_generator.py.  The bot now ships HTML reports
via a signed Cloud Storage URL (see services.report_storage) instead of
rendering a PDF — this drops Playwright + Chromium from the runtime
(~1.1 GB image shrinkage) and lets the reports remain interactive,
zoomable and mobile-friendly in the user's own browser.

Two layouts:
  • report_basic.html — BASE-tier report (tier='base')
  • report_deep.html  — DEEP-tier report (tier='deep')

Legacy `report.html` (dark theme) is kept as a fallback when
REPORT_VERSION=v1 in the environment.

The renderer writes the rendered HTML to /tmp by default; callers
typically pass the path to services.report_storage.upload_report() to
push the file to GCS and obtain a signed URL.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
# Reports are temporary by design — /tmp survives only until container
# restart, but each report should already be in GCS by then.
OUTPUT_DIR   = Path(os.getenv("REPORT_LOCAL_DIR", "/tmp/user_reports"))

# Feature flag: 'v2' (default) → light theme basic/deep; 'v1' → legacy report.html
REPORT_VERSION = os.getenv("REPORT_VERSION", "v2").lower()


# Minimal mock fixture for the CLI smoke test below.  Built through
# build_payload so it carries every key the templates expect.  Full
# real-data fixtures live in tests/test_phase4_reporting.py.
def _mock_results() -> dict:
    """Synthetic results dict that build_payload can chew on."""
    import pandas as pd
    perf = pd.DataFrame([
        {"Ticker": "AAPL", "Current_Value": 25000.0, "Total_Cost": 20000.0,
         "PnL": 5000.0, "Return_Pct": 0.25,
         "Euler_Risk_Contribution_Pct": 18.4, "ATR_Pct": 1.4,
         "Fundamental_Sector": "Technology"},
        {"Ticker": "KSPI", "Current_Value": 15000.0, "Total_Cost": 18000.0,
         "PnL": -3000.0, "Return_Pct": -0.167,
         "Euler_Risk_Contribution_Pct": 22.1, "ATR_Pct": 2.1,
         "Fundamental_Sector": "EM_Kazakhstan"},
        {"Ticker": "BND",  "Current_Value": 10000.0, "Total_Cost": 10000.0,
         "PnL": 0.0, "Return_Pct": 0.0,
         "Euler_Risk_Contribution_Pct": 3.2, "ATR_Pct": 0.4,
         "Fundamental_Sector": "Fixed_Income"},
    ])
    return {
        "performance_table": perf,
        "total_value": 50000.0,
        "portfolio_metrics": {
            "CVaR_95_Daily": -0.052, "Sharpe_Ratio": 1.34,
            "Sortino_Ratio": 1.62, "VaR_95_Daily": -0.024,
            "Max_Drawdown": -0.128, "Total_Volatility_Ann": 0.142,
            "Composite_Risk_Score": 62, "Max_Euler_Risk_Pct": 22.1,
            "CVaR_95_Bootstrap": {"point": -0.052, "lo95": -0.07, "hi95": -0.045},
        },
        "sector_exposure": {"Technology": 0.50, "EM_Kazakhstan": 0.30,
                             "Fixed_Income": 0.20},
        "benchmark_comparison": {},
        "asset_scores": {},
    }


def _mock_payload(tier: str = "base") -> dict:
    """
    Build a mock payload via build_payload so every key is present.

    For DEEP tier we additionally inject `factor_radar_svg` and
    `factor_betas` — these are normally produced inside tg_bot.py from
    Beta_<axis> columns in the perf table, so the smoke fixture needs
    to mimic that out-of-band wiring.

    DEEP also synthesises mock values for engine-heavy keys that need
    external services (FRED, stress engine, BL simulate):
      stress_scenarios, expected_effect, macro_drivers, regime,
      action_plan, hotspots, risk_waterfall.
    The shapes match what the live engine emits — same field names and
    types — so the template binds the same way in production.
    """
    # Scenario tier — не проходит через build_payload (у него своя схема
    # `data.scenario`).  Smoke-фикстура строит синтетический сценарный payload
    # напрямую (форма 1:1 с finance/scenario_report.build_scenario_payload).
    if (tier or "").lower() == "scenario":
        return {
            "tier": "scenario",
            "scenario": {
                "available": True,
                "window_days": 1825, "n_obs": 1240,
                "metrics": {"ann_return": 0.183, "vol_cov": 0.116, "vol_gross_ref": 0.214,
                            "sharpe_rfr": 1.19, "beta": 0.65, "div_yield": 0.012,
                            "pe": 24.5, "rfr": 0.045},
                "mctr_rows": [
                    {"display": "NVDA", "weight": 0.148, "sigma_i": 0.46, "rho_ip": 0.82,
                     "mctr": 0.38, "ctrisk": 0.056, "pct_ctr": 27.8},
                    {"display": "MSFT", "weight": 0.175, "sigma_i": 0.28, "rho_ip": 0.71,
                     "mctr": 0.20, "ctrisk": 0.035, "pct_ctr": 17.4},
                    {"display": "GLD", "weight": 0.056, "sigma_i": 0.14, "rho_ip": -0.11,
                     "mctr": -0.015, "ctrisk": -0.001, "pct_ctr": -0.6},
                ],
                "vol_cov": 0.116,
                "regime_survival": [
                    {"regime": "risk_on", "label": "Risk-On / бычий рынок", "avg_pct": 6.2,
                     "n_shocks": 2, "survives": True},
                    {"regime": "rate_shock", "label": "Шок процентных ставок", "avg_pct": -8.1,
                     "n_shocks": 4, "survives": True},
                    {"regime": "risk_off", "label": "Risk-Off / рецессия", "avg_pct": -14.6,
                     "n_shocks": 3, "survives": False},
                ],
                "funding": [
                    {"display": "ORCL", "weight": 0.109, "sharpe": 0.42,
                     "flags": ["Sharpe 0.42 < 0.5", "дублирующийся риск: corr≥0.9 c NVDA"]},
                ],
                "excluded_young": ["FFSPC6.1028.AIX"],
                "backtest": {"ticker": "MSFT", "rule": "Цена выше 200-дневной средней (трендовый фильтр)",
                             "summary": {"n_signals": 41, "hit_rate_63d": 0.68,
                                         "horizons": {"21d": {"mean": 0.021, "median": 0.018, "worst": -0.11, "n": 41},
                                                      "63d": {"mean": 0.058, "median": 0.049, "worst": -0.19, "n": 41},
                                                      "126d": {"mean": 0.112, "median": 0.093, "worst": -0.22, "n": 38}}}},
                "disclaimers": [
                    "Историческая доходность не гарантирует будущих результатов.",
                    "Расчёты подвержены искажению выжившего (survivorship bias).",
                    "Отчёт — результат мат. моделирования и не является ИИР.",
                ],
            },
        }

    from pdf_payload import build_payload
    # Tier-specific model — mirrors src/ai_narrative.py:
    #   MODEL_BASE = claude-haiku-4-5-20251001
    #   MODEL_DEEP = claude-sonnet-4-6
    # The DEEP report's AI commentary is generated by Sonnet (richer
    # reasoning required for stress / regime / scoring narrative).
    _mock_model = ("claude-sonnet-4-6" if tier == "deep"
                    else "claude-haiku-4-5-20251001")
    payload = build_payload(
        _mock_results(),
        tier=tier,
        ai_summary={
            "verdict":        "Портфель в умеренной зоне риска (composite 62/100).",
            "plain_summary":  "За месяц портфель прибавил +5.8%.  Главные риски — "
                              "перевес в технологиях и KSPI hotspot (TRC 22%).",
            "bullets":        ["AAPL +25% от входа — крупнейший вклад",
                                "KSPI просел −16.7%, TRC 22% — пересмотр позиции"],
            "stock_picks":    {},
            "used_rag":       False,
            "model_used":     _mock_model,
            # Per-KPI plain-language notes.
            "ai_cvar_note":   "В худший день из 20 портфель теряет около $2.6K — "
                              "терпимо для умеренного профиля.",
            "ai_sharpe_note": "На каждую единицу риска портфель приносит 1.34 "
                              "доходности — риск окупается.",
            "ai_mdd_note":    "Когда-то портфель просел на 12.8% от пика — "
                              "переживаемо, но держите подушку.",
            # Per-section AI commentary — mock values so the smoke render
            # exercises the data-bound AI blocks in both templates.
            "ai_risk_comment":      "CVaR −5.2% (≈$2.6K) в пределах мандата, но "
                                    "MaxDD −12.8% близко к лимиту.",
            "ai_holdings_comment":  "Два hotspot: AAPL и KSPI дают 40% риска при "
                                    "50% веса — концентрация выше порога.",
            "ai_sector_comment":    "Технологии 50% — перевес против бенчмарка; "
                                    "облигации 20% сглаживают.",
            "ai_benchmark_comment": "Портфель обгоняет S&P 500: IR 0.61 при TE 8.4%.",
            "ai_regime_comment":    "Поздняя экспансия — циклические сектора пока "
                                    "лидируют, но запас хода ограничен.",
            "ai_factor_comment":    "Доминируют Market и Momentum — типичный "
                                    "IT-перевес; Rates-бета отрицательна.",
            "ai_stress_comment":    "Худший сценарий — Equity DM −20%: портфель "
                                    "−15.8%, восстановление ≈6 мес.",
            "ai_effect_comment":    "Ребалансировка снижает индекс риска 62→54 и "
                                    "поднимает Sharpe 1.18→1.32.",
            # Structured regime cross-check — DEEP only; populated to exercise
            # the new confirmation panel in the smoke render.
            "regime_confirmation":  {
                "stance":  "partial",
                "summary": "Движок видит Expansion; макро и беты в основном "
                           "согласны, но HY-спред слегка расширяется — "
                           "ранний сигнал слабости.",
                "signals": [
                    "✓ Кривая 10Y−2Y +0.18 пп — рецессия не сигналит",
                    "⚠ HY OAS 312 bp растёт 3 нед. подряд — кредит сжимается",
                    "✓ VIX 14 — спокойствие, страх не нарастает",
                    "✓ Market β 1.05 — типично для late-Expansion",
                    "⚠ [Barclays] предупреждает: близко к границе со Slowdown",
                ],
            },
        },
    )
    if tier == "deep":
        # Mirror the post-build enrichment done in tg_bot._render_v2 so the
        # new factor + CoVe sections actually populate in the smoke render.
        try:
            from tg_bot import (_build_equity_curve_svg,
                                  _build_factor_radar_svg,
                                  _build_factor_betas_table)
            results = _mock_results()
            payload["equity_curve_svg"] = _build_equity_curve_svg(results)
            payload["factor_radar_svg"] = _build_factor_radar_svg(results)
            payload["factor_betas"]     = _build_factor_betas_table(results)
        except Exception as exc:
            # tg_bot has many heavy deps (aiogram); fall back to a synthetic
            # radar + an 8-row table so the template at least renders cleanly.
            logger.warning("Smoke fallback for radar — tg_bot import failed: %s", exc)
            from pdf_charts import factor_radar_svg
            mock_betas = {
                "Market":     1.18,  "Momentum": 1.05, "Value":      0.25,
                "Quality":    0.62,  "Size":     0.42, "Volatility": 0.88,
                "Commodities":0.15,  "Rates":   -0.30,
            }
            payload["factor_radar_svg"] = factor_radar_svg(mock_betas)
            payload["factor_betas"]     = [
                {"axis": a, "beta": b,
                 "bench":  1.0 if a == "Market" else 0.0,
                 "delta":  round(b - (1.0 if a == "Market" else 0.0), 2),
                 "missing": False}
                for a, b in mock_betas.items()
            ]
        # Synthesise mock stress/effect/macro/regime so the design pages
        # render fully populated.  Engine produces identical shapes in
        # production from real inputs (FRED + simulate_after_plan).
        payload["stress_scenarios"] = _MOCK_STRESS_SCENARIOS
        payload["expected_effect"]  = _MOCK_EXPECTED_EFFECT
        payload["macro_drivers"]    = _MOCK_MACRO_DRIVERS
        payload["regime"]           = _MOCK_REGIME
        payload["action_plan"]      = _MOCK_ACTION_PLAN
        payload["hotspots"]         = _MOCK_HOTSPOTS
        payload["risk_waterfall"]   = _MOCK_RISK_WATERFALL
        payload["score_breakdown"]  = _MOCK_SCORE_BREAKDOWN
    # Shared mock keys for BOTH tiers — engine-correct shapes,
    # so the same template bindings work in production.
    if not payload.get("scenarios"):
        payload["scenarios"] = _MOCK_SCENARIOS
    if not payload.get("period_returns_table"):
        payload["period_returns_table"] = _MOCK_PERIOD_RETURNS
    if not payload.get("risk_waterfall") and tier == "base":
        payload["risk_waterfall"] = _MOCK_RISK_WATERFALL
    # ai_ideas is the v3 idea-card schema; populate the mock when empty so the
    # smoke render exercises the ideas grid.
    if not any((payload.get("ai_ideas") or {}).values()):
        payload["ai_ideas"] = _MOCK_AI_STOCK_PICKS
    if not payload.get("kpi_sparklines"):
        payload["kpi_sparklines"] = _build_mock_sparklines()
    return payload


def _build_mock_sparklines() -> dict:
    """
    Generate 3 mock 12-point sparkline SVGs for the cover KPI strip.

    Uses tg_bot._sparkline_svg if available (production rendering path);
    falls back to a synthetic SVG matching the production output format
    when tg_bot can't import (e.g. local-dev without dotenv).
    """
    # Synthetic 12-month series — realistic banker shapes:
    #   CVaR worsens then recovers (V-shape)
    #   Sharpe steady improvement
    #   MaxDD drawdown then partial recovery
    cvar_pts   = [-0.030, -0.035, -0.041, -0.048, -0.055, -0.060,
                   -0.058, -0.052, -0.047, -0.045, -0.050, -0.052]
    sharpe_pts = [ 0.94,   1.02,   1.10,   1.05,   0.92,   0.88,
                    0.99,   1.12,   1.20,   1.28,   1.30,   1.34]
    mdd_pts    = [-0.040, -0.055, -0.072, -0.095, -0.110, -0.128,
                   -0.115, -0.108, -0.105, -0.100, -0.110, -0.128]
    # Raw points too (scaled) — the Premium V2 KPI cards redraw these with the
    # design's own <Sparkline>; mirrors the production keys from
    # tg_bot._build_kpi_sparklines so the smoke render exercises the same path.
    _pts = {
        "cvar_pts":   [round(x * 100, 2) for x in cvar_pts],
        "sharpe_pts": [round(x, 3) for x in sharpe_pts],
        "mdd_pts":    [round(x * 100, 2) for x in mdd_pts],
    }
    try:
        from tg_bot import _sparkline_svg
        return {
            "cvar_svg":   _sparkline_svg(cvar_pts,   color="#3F8F5F", invert=True),
            "sharpe_svg": _sparkline_svg(sharpe_pts, color="#9A7A10", invert=False),
            "mdd_svg":    _sparkline_svg(mdd_pts,    color="#C0492F", invert=True),
            **_pts,
        }
    except Exception:
        # Inline minimal SVG fallback so the smoke render still shows
        # the sparkline cells with the gradient + polyline.
        def _fallback(values, color):
            vmin, vmax = min(values), max(values)
            rng = vmax - vmin or 1
            xs = [4 + i * 232 / 11 for i in range(12)]
            ys = [4 + 28 - (v - vmin) / rng * 28 for v in values]
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
            return (f'<svg viewBox="0 0 240 36" preserveAspectRatio="none" '
                     f'xmlns="http://www.w3.org/2000/svg">'
                     f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
                     f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3" '
                     f'fill="#FBFAF7" stroke="{color}" stroke-width="1.1"/>'
                     f'</svg>')
        return {
            "cvar_svg":   _fallback(cvar_pts,   "#3F8F5F"),
            "sharpe_svg": _fallback(sharpe_pts, "#9A7A10"),
            "mdd_svg":    _fallback(mdd_pts,    "#C0492F"),
            **_pts,
        }


# ── Mock fixtures — moved to report_mocks.py (§−14 C-7); re-exported here so
# existing imports (tests / CLI) keep working unchanged.
from report_mocks import (            # noqa: E402  (import after helpers)
    _MOCK_STRESS_SCENARIOS, _MOCK_EXPECTED_EFFECT, _MOCK_MACRO_DRIVERS,
    _MOCK_REGIME, _MOCK_ACTION_PLAN, _MOCK_HOTSPOTS, _MOCK_RISK_WATERFALL,
    _MOCK_SCORE_BREAKDOWN, _MOCK_SCENARIOS, _MOCK_PERIOD_RETURNS,
    _MOCK_AI_STOCK_PICKS,
)

MOCK_DATA: dict = {}   # lazily filled by the CLI / smoke entry-point below.


# §−14 C-6: REPORT_DEEP_VERSION / REPORT_BASIC_VERSION env knobs removed — they
# were read into DEEP_VERSION/BASIC_VERSION and never consulted anywhere (v3 is
# the single Jinja design; selection is tier-based in _select_template).

# Premium V2 routing (see PREMIUM_DESIGN.md).  Sprint-1 #1: Premium V2 is now the
# PRODUCTION DEFAULT — the code default flips OFF→ON to match the live deploy,
# which already sets PREMIUM_REPORT_ENABLED=true.  The classic v3 Jinja pipeline
# is RETAINED as the automatic fallback (render_report_html wraps the premium
# path in try/except → v3) and is exercised by the test suite; set
# PREMIUM_REPORT_ENABLED=false to force it.  We deliberately do NOT delete v3:
# 16 test files pin the pdf_payload→v3 contract and it is the resilience net if a
# premium asset is ever missing.  So "Premium is the only production path" is
# realised as "Premium is the default, v3 is the safety fallback".
PREMIUM_REPORT_ENABLED = os.getenv("PREMIUM_REPORT_ENABLED", "true").strip().lower() in (
    "1", "true", "yes", "on")


def _select_template(tier: str) -> str:
    """
    Pick the template name based on tier.

    L-8: the legacy v1 (`report.html`, dark theme) and v2
    (`report_basic.html` / `report_deep.html`) designs were retired and
    deleted.  v3 is now the single production design — it carries the RFR +
    reporting-currency provenance panel (integrity checks) the older
    templates lacked.  Selection is purely tier-based:
        deep → report_deep_v3.html
        base → report_basic_v3.html
    """
    if (tier or "base").lower() == "scenario":
        return "report_scenario_v3.html"
    if (tier or "base").lower() == "deep":
        return "report_deep_v3.html"
    return "report_basic_v3.html"


def _jinja_env() -> Environment:
    return Environment(
        loader     = FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape = select_autoescape(["html"]),
    )


def render_report_html(data_dict: dict | None,
                        user_id:   int | str,
                        report_type: str = "Базовый отчёт",
                        tier:        str = "base",
                        generated_at: str | None = None) -> str:
    """
    Render the Jinja template to an HTML string.

    Args:
        data_dict   : Payload from pdf_payload.build_payload.  When None,
                       a synthetic mock payload is built (smoke-test only).
        user_id     : Telegram user ID (shown in the report header).
        report_type : Human-readable label for the header badge.
        tier        : 'base' | 'deep' — selects the template.
        generated_at: Optional override for the report timestamp; defaults
                       to "now" in UTC+5.

    Returns:
        The rendered HTML as a string.  Does NOT write to disk — pass to
        write_report_html() or services.report_storage.upload_report().
    """
    payload  = data_dict if data_dict is not None else _mock_payload(tier)

    # Resolve the timestamp ONCE so BOTH render paths show it.  Previously only
    # the v3 branch defaulted it (inside template.render); the premium branch
    # passed the raw `generated_at` (None when the caller omits it, as tg_bot
    # does) straight to the mapper → meta.generated / meta.session rendered '–'.
    generated_at = generated_at or datetime.now().strftime("%d.%m.%Y %H:%M UTC+5")

    # ── Routing: Premium V2 (flag) vs classic v3 Jinja ─────────────────────────
    # Feature-flagged so the v3 pipeline below is byte-identical when OFF.  Any
    # failure in the premium path falls back to v3 — a report is always produced.
    # Scenario tier has NO Premium React bundle (focused single-page report) —
    # it always renders via its dedicated Jinja template below.
    if PREMIUM_REPORT_ENABLED and (tier or "base").lower() != "scenario":
        try:
            from premium_payload import build_design_data
            from premium_renderer import render_premium
            design_data = build_design_data(payload, tier, user_id=user_id,
                                             generated_at=generated_at)
            return render_premium(tier, design_data)
        except Exception as exc:   # never break delivery — degrade to v3
            # ERROR (not warning) + stack: the flag is ON, so a fallback here is
            # a real misconfiguration (e.g. missing premium assets) the operator
            # MUST see — it silently masked the v3 output before.
            logger.error("PREMIUM_REPORT_ENABLED=on but premium render FAILED (%s) — "
                         "falling back to v3.  Check src/premium_assets/ is deployed.",
                         exc, exc_info=True)

    template = _jinja_env().get_template(_select_template(tier))
    return template.render(
        data         = payload,
        user_id      = user_id,
        report_type  = report_type,
        generated_at = generated_at or datetime.now().strftime("%d.%m.%Y %H:%M UTC+5"),
    )


def write_report_html(html: str, user_id: int | str, tier: str,
                       output_dir: Path | None = None) -> Path:
    """
    Persist a rendered HTML string to a local file.

    Path: <output_dir>/<user_id>_<YYYY-MM-DD>_<tier>.html  (default dir
    is OUTPUT_DIR = /tmp/user_reports).  Caller is responsible for
    uploading the file to GCS or sending the path to the user.

    Returns the absolute Path that was written.
    """
    out_dir = Path(output_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    path  = out_dir / f"{user_id}_{today}_{tier}.html"
    path.write_text(html, encoding="utf-8")
    logger.info("HTML report written: %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def generate_portfolio_html(data_dict: dict | None,
                              user_id:   int | str,
                              report_type: str = "Базовый отчёт",
                              tier:        str = "base") -> str:
    """
    Convenience wrapper: render + write to disk, returns the file path.

    Mirrors the old generate_portfolio_pdf() signature so callers in
    tg_bot.py only need to swap the function name.
    """
    html = render_report_html(data_dict, user_id, report_type, tier)
    return str(write_report_html(html, user_id, tier))


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for t in ("base", "deep"):
        p = generate_portfolio_html(
            None, user_id="smoke", tier=t,
            report_type="Базовый отчёт" if t == "base" else "Глубокий сценарный анализ",
        )
        print(f"[{t}] → {p}")


__all__ = [
    "render_report_html",
    "write_report_html",
    "generate_portfolio_html",
    "MOCK_DATA",
    "TEMPLATE_DIR",
    "OUTPUT_DIR",
    "REPORT_VERSION",
]
