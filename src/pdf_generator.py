"""
PDF Generation Module — Playwright + Jinja2 institutional report renderer.

Two layouts are shipped:
  • report_basic.html — 2-page light theme (tier='base')
  • report_deep.html  — 4-page light theme (tier='deep')

Selection is driven by the `tier` argument; legacy `report.html` (dark theme)
is kept as a fallback when REPORT_VERSION=v1 in the environment.

Output path: data/user_reports/<user_id>_<YYYY-MM-DD>_<tier>.pdf
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
OUTPUT_DIR   = Path(__file__).parent.parent / "data" / "user_reports"

# Feature flag: 'v2' (default) → light theme basic/deep; 'v1' → legacy report.html.
REPORT_VERSION = os.getenv("REPORT_VERSION", "v2").lower()

MOCK_DATA: dict = {
    "cvar":             "-5.2%",
    "sharpe":           "1.34",
    "var_95_daily":     "-2.4%",      # quantile of 1-day return distribution
    "max_drawdown":     "-12.8%",     # realised peak-to-trough on equity curve
    "risk_pct":         62,
    "pnl_total_abs":    "+1,245",
    "pnl_total_pct":    "+5.8%",
    "pnl_total_color":  "pos",
    "assets": [
        {"ticker": "AAPL",  "weight": "20%", "asset_class": "Акции США",    "euler_risk": "18.4%",
         "pnl_pct": "+18.2%", "pnl_abs": "+2,145", "pnl_color": "pos"},
        {"ticker": "KSPI",  "weight": "15%", "asset_class": "Акции KZ",     "euler_risk": "22.1%",
         "pnl_pct": "-3.4%",  "pnl_abs": "-320",   "pnl_color": "neg"},
        {"ticker": "BTC",   "weight": "10%", "asset_class": "Крипто",        "euler_risk": "34.7%",
         "pnl_pct": "+24.6%", "pnl_abs": "+1,860", "pnl_color": "pos"},
        {"ticker": "GOLD",  "weight": "25%", "asset_class": "Сырьё",         "euler_risk": "8.2%",
         "pnl_pct": "+6.1%",  "pnl_abs": "+540",   "pnl_color": "pos"},
        {"ticker": "CASH",  "weight": "30%", "asset_class": "Ден. средства", "euler_risk": "0.0%",
         "pnl_pct": "0.0%",   "pnl_abs": "0",      "pnl_color": "pos"},
    ],
    "scenarios": [
        {"name": "Профильный бенчмарк", "probability": "IR: 0.84", "pnl": "+4.1%",  "driver": "✅ Обыгрывает бенчмарк"},
        {"name": "S&P 500",             "probability": "IR: 0.61", "pnl": "+2.7%",  "driver": "✅ Обыгрывает бенчмарк"},
        {"name": "Nasdaq 100",          "probability": "IR: -0.2", "pnl": "-11.3%", "driver": "❌ Отстаёт бенчмарк"},
    ],
}


def _select_template(tier: str) -> str:
    """Pick the template name based on tier and feature flag."""
    if REPORT_VERSION == "v1":
        return "report.html"
    if (tier or "base").lower() == "deep":
        return "report_deep.html"
    return "report_basic.html"


async def generate_portfolio_pdf(
    data_dict: dict | None,
    user_id:   int | str,
    report_type: str = "Базовый отчёт",
    tier:      str = "base",
) -> str:
    """
    Render the appropriate template via Playwright Chromium → A4 PDF.

    Args:
        data_dict:   Portfolio payload. Falls back to MOCK_DATA when None.
        user_id:     Telegram user ID — used for the output filename.
        report_type: Label shown in the report header.
        tier:        'base' or 'deep' — selects template (v2 only).

    Returns:
        Absolute path string to the generated PDF.
    """
    payload = data_dict if data_dict is not None else MOCK_DATA
    template_name = _select_template(tier)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template    = env.get_template(template_name)
    html_string = template.render(
        data        = payload,
        user_id     = user_id,
        report_type = report_type,
        generated_at = datetime.now().strftime("%d.%m.%Y %H:%M UTC+5"),
    )

    today = datetime.now().strftime("%Y-%m-%d")
    output_path = OUTPUT_DIR / f"{user_id}_{today}_{tier}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run Playwright synchronously inside a ThreadPoolExecutor.
    # async_playwright uses asyncio.create_subprocess_exec which calls
    # get_child_watcher() — not implemented in aiogram's event loop policy.
    # The sync API launched from a thread bypasses this entirely.
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        await loop.run_in_executor(pool, _render_pdf_sync, html_string, str(output_path))

    logger.info("PDF сгенерирован: %s", output_path)
    return str(output_path)


def _render_pdf_sync(html_string: str, output_path: str) -> None:
    """
    Blocking Playwright render — runs Chromium in a child subprocess so that
    Chromium's SIGTRAP on --single-process exit (Cloud Run / gVisor) does not
    propagate to the parent aiogram process and cause container restarts.
    """
    # Inline runner script passed via stdin to avoid temp-file race conditions.
    runner_code = r"""
import sys, json
from playwright.sync_api import sync_playwright
args_cfg = json.loads(sys.stdin.read())
html_string  = args_cfg["html"]
output_path  = args_cfg["output"]
with sync_playwright() as pw:
    browser = pw.chromium.launch(
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--no-zygote",
            "--single-process",
            "--disable-extensions",
        ],
    )
    page = browser.new_page()
    page.set_content(html_string, wait_until="networkidle", timeout=30000)
    page.pdf(
        path=output_path, format="A4", print_background=True,
        margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"},
    )
    browser.close()
"""
    payload = json.dumps({"html": html_string, "output": output_path})
    result = subprocess.run(
        [sys.executable, "-c", runner_code],
        input=payload.encode(),
        capture_output=True,
        timeout=90,
    )
    # Non-zero exit is expected when Chromium raises SIGTRAP on --single-process
    # cleanup (signal 5 → exit code 133 on Linux).  The PDF is already written
    # at that point — only raise if the output file is missing.
    if result.returncode not in (0, 133) and not Path(output_path).exists():
        stderr = result.stderr.decode(errors="replace")[:600]
        raise RuntimeError(f"PDF subprocess failed (rc={result.returncode}): {stderr}")


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    async def _smoke() -> None:
        for tier in ("base", "deep"):
            path = await generate_portfolio_pdf(
                None, user_id="test_user", tier=tier,
                report_type=("Базовый отчёт" if tier == "base"
                             else "Глубокий сценарный анализ"),
            )
            print(f"[{tier}] {path}")

    asyncio.run(_smoke())
