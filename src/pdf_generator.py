"""
PDF Generation Module — Playwright + Jinja2 institutional report renderer.
Playwright renders the HTML template through a real Chromium engine, producing
pixel-perfect dark-mode PDFs that xhtml2pdf could not match.

Output path: data/user_reports/<user_id>_report.pdf
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
OUTPUT_DIR   = Path(__file__).parent.parent / "data" / "user_reports"

MOCK_DATA: dict = {
    "cvar":         "-5.2%",
    "sharpe":       "1.34",
    "max_drawdown": "-12.8%",
    "risk_pct":     62,
    "assets": [
        {"ticker": "AAPL",  "weight": "20%", "asset_class": "Акции США",    "euler_risk": "18.4%"},
        {"ticker": "KSPI",  "weight": "15%", "asset_class": "Акции KZ",     "euler_risk": "22.1%"},
        {"ticker": "BTC",   "weight": "10%", "asset_class": "Крипто",        "euler_risk": "34.7%"},
        {"ticker": "GOLD",  "weight": "25%", "asset_class": "Сырьё",         "euler_risk": "8.2%"},
        {"ticker": "CASH",  "weight": "30%", "asset_class": "Ден. средства", "euler_risk": "0.0%"},
    ],
    "scenarios": [
        {"name": "Профильный бенчмарк", "probability": "IR: 0.84", "pnl": "+4.1%",  "driver": "✅ Обыгрывает бенчмарк"},
        {"name": "S&P 500",             "probability": "IR: 0.61", "pnl": "+2.7%",  "driver": "✅ Обыгрывает бенчмарк"},
        {"name": "Nasdaq 100",          "probability": "IR: -0.2", "pnl": "-11.3%", "driver": "❌ Отстаёт бенчмарк"},
    ],
}


async def generate_portfolio_pdf(
    data_dict: dict | None,
    user_id:   int | str,
    report_type: str = "Базовый отчёт",
) -> str:
    """
    Render report.html via Playwright Chromium → A4 PDF.

    Args:
        data_dict:   Portfolio payload. Falls back to MOCK_DATA when None.
        user_id:     Telegram user ID — used for the output filename.
        report_type: Label shown in the report header.

    Returns:
        Absolute path string to the generated PDF.
    """
    payload = data_dict if data_dict is not None else MOCK_DATA

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template    = env.get_template("report.html")
    html_string = template.render(
        data        = payload,
        user_id     = user_id,
        report_type = report_type,
        generated_at = datetime.now().strftime("%d.%m.%Y %H:%M UTC+5"),
    )

    output_path = OUTPUT_DIR / f"{user_id}_report.pdf"
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
    """Blocking Playwright render — must be called from a thread, not the event loop."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        page = browser.new_page()
        page.set_content(html_string, wait_until="networkidle", timeout=30_000)
        page.pdf(
            path             = output_path,
            format           = "A4",
            print_background = True,
            margin           = {
                "top":    "14mm",
                "bottom": "14mm",
                "left":   "12mm",
                "right":  "12mm",
            },
        )
        browser.close()


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    async def _smoke() -> None:
        path = await generate_portfolio_pdf(None, user_id="test_user")
        print(f"PDF: {path}")

    asyncio.run(_smoke())
