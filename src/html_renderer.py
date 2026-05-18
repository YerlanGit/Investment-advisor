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
    from pdf_payload import build_payload
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
        payload["stress_scenarios"] = [
            {"name": "Equity DM −20%",  "port_pct": -0.158, "port_dollar":  -7900,
             "max_dd_pct": -0.220, "recovery_months": 6, "tag": "Tech sell-off"},
            {"name": "Equity EM −25%",  "port_pct": -0.075, "port_dollar":  -3750,
             "max_dd_pct": -0.110, "recovery_months": 4, "tag": "EM rout"},
            {"name": "Rates +100 bp",   "port_pct": -0.038, "port_dollar":  -1900,
             "max_dd_pct": -0.062, "recovery_months": 3, "tag": "Duration shock"},
            {"name": "HY Credit +200bp","port_pct": -0.028, "port_dollar":  -1400,
             "max_dd_pct": -0.044, "recovery_months": 3, "tag": "Credit widening"},
            {"name": "USD +10% (proxy)","port_pct":  0.012, "port_dollar":    600,
             "max_dd_pct": -0.018, "recovery_months": 2, "tag": "DXY rally"},
            {"name": "Oil −30%",        "port_pct": -0.014, "port_dollar":   -700,
             "max_dd_pct": -0.025, "recovery_months": 2, "tag": "Energy drawdown"},
            {"name": "CPI +1pp (proxy)","port_pct": -0.022, "port_dollar":  -1100,
             "max_dd_pct": -0.038, "recovery_months": 3, "tag": "Inflation surprise"},
        ]
        payload["expected_effect"] = {
            "risk_index":      {"before": 62,    "after": 54,    "delta_pp": -8,    "favourable": True},
            "cvar_95":         {"before": -0.052,"after": -0.041,"delta_pp": 1.1,   "favourable": True},
            "sharpe":          {"before": 1.18,  "after": 1.32,  "delta_pp": 0.14,  "favourable": True},
            "max_drawdown":    {"before": -0.128,"after": -0.104,"delta_pp": 2.4,   "favourable": True},
            "vol":             {"before": 0.148, "after": 0.126, "delta_pp": -2.2,  "favourable": True},
            "max_erc_pct":     {"before": 0.244, "after": 0.168, "delta_pp": -7.6,  "favourable": True},
            "expected_return": {"before": 0.142, "after": 0.126, "delta_pp": -1.6,  "favourable": False},
            "it_share":        {"before": 0.62,  "after": 0.50,  "delta_pp": -12.0, "favourable": True},
        }
        payload["macro_drivers"] = {
            "as_of":   "2026-05-14",
            "regime":  "Expansion (late)",
            "series": [
                {"id":"T10Y2Y","name":"Yield curve 10Y−2Y","value":"+0.18 pp","as_of":"2026-05-14","status":"ok",
                 "comment":"положительный наклон — рецессия не сигналит"},
                {"id":"BAMLH0A0HYM2","name":"HY OAS (ICE BofA)","value":"312 bp","as_of":"2026-05-14","status":"ok",
                 "comment":"спреды узкие — рынок не закладывает кредитный риск"},
                {"id":"NAPM","name":"ISM Manufacturing PMI","value":"51.4","as_of":"2026-04-01","status":"ok",
                 "comment":"чуть выше 50 — производство расширяется"},
                {"id":"VIXCLS","name":"CBOE VIX","value":"14.2","as_of":"2026-05-14","status":"ok",
                 "comment":"низкая ожидаемая волатильность — комфортная зона"},
                {"id":"T10YIE","name":"10Y Breakeven Inflation","value":"2.34%","as_of":"2026-05-14","status":"ok",
                 "comment":"инфляционные ожидания у цели ФРС 2%"},
            ],
        }
        payload["regime"] = {
            "label":      "Expansion (late)",
            "confidence": 72,
            "growth":     0.08,
            "cycle":      0.04,
            "explainers": [
                "SPY обгоняет IEF на +4.1% за 60 дней (рост vs облигации)",
                "Discretionary > Staples: +2.8% за 60д (цикличные on)",
                "EEM 60д +1.2% (EM risk-on)",
            ],
        }
        payload["action_plan"] = [
            {"ticker":"AAPL","action":"Sell 25%","reason":"HOTSPOT 24% риска; фиксируем часть прибыли (+25%)",
             "price":"$197.40","buy_zone":"$182–188","sell_target":"$215","stop_loss":"$175"},
            {"ticker":"KSPI","action":"Reduce 50%","reason":"TRC 22% при просадке −16.7% — концентрация выше порога",
             "price":"₸123.50","buy_zone":"₸115–119","sell_target":"₸140","stop_loss":"₸108"},
            {"ticker":"BND","action":"Hold","reason":"Hedge против rate shock; держим вес 20%",
             "price":"$72.18","buy_zone":"$71–73","sell_target":"$76","stop_loss":"$69"},
        ]
        payload["hotspots"] = [
            {"ticker":"AAPL","trc_pct":18.4,"reason":"TRC 18.4% при весе 50% — высокая концентрация"},
            {"ticker":"KSPI","trc_pct":22.1,"reason":"TRC 22% — KSPI EM hotspot"},
        ]
        payload["risk_waterfall"] = {
            "structural": 14.8,
            "concentration_add": 4.2,
            "stress_add":        2.4,
            "diversif_credit":  -3.4,
            "final":            18.0,
            "narrative": "Базовая волатильность 14.8% растёт до 18.0% после поправок на "
                         "концентрацию (+4.2) и стресс-сценарии (+2.4), компенсируется "
                         "диверсификационным кредитом (−3.4 pp).",
        }
    return payload


MOCK_DATA: dict = {}   # lazily filled by the CLI / smoke entry-point below.


DEEP_VERSION = os.getenv("REPORT_DEEP_VERSION", "v2").lower()


def _select_template(tier: str) -> str:
    """
    Pick the template name based on tier and feature flag.

    Feature flag layering:
      REPORT_VERSION     v1 → legacy dark theme (single file)
                          v2 → default light theme (basic + deep light)
      REPORT_DEEP_VERSION v2 → legacy hybrid (snapshot-bar + q-blocks)
                          v3 → prototype-grade banker design (cream theme,
                                5 pages, 8-axis radar, banker-grade KPI
                                cards).  In development — flip via env
                                when the v3 port is feature-complete.
    """
    if REPORT_VERSION == "v1":
        return "report.html"
    if (tier or "base").lower() == "deep":
        if DEEP_VERSION == "v3":
            return "report_deep_v3.html"
        return "report_deep.html"
    return "report_basic.html"


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
