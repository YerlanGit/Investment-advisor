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
