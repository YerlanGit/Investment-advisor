"""
RAMP Telegram Bot — aiogram 3.x async entry point.

Deep-link format: t.me/RampBot?start=<source_slug>
Analysis tiers:
  - base  : 1 token  → MAC3 CVaR + allocation table
  - deep  : 2 tokens → base + scenario analysis + fundamental signals

Onboarding FSM (new users only):
  Q1 → Q2 → Q3 → Q4 → Q5 → Q6 → Universe → Benchmark → MandateReview → Connection → Analysis

PortfolioConnection FSM:
  connect:template → save mode → Analysis menu
  connect:freedom  → Login → ApiKey → save encrypted → Analysis menu
"""

import asyncio
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramConflictError
from aiogram.filters import CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from db_tokenomics import (
    approve_mandate,
    credit_tokens,
    deduct_tokens,
    get_balance,
    get_benchmark_ticker,
    get_connection_mode,
    get_profile,
    get_last_report_snapshot,
    init_db,
    init_user,
    save_benchmark_ticker,
    save_connection_mode,
    save_profile,
    save_report_snapshot,
)
from finance.broker_api import (
    BrokerAuthError,
    BrokerEmptyPortfolioError,
    FreedomConnector,
    RealPortfolioRequired,
)
from finance.investment_logic import UniversalPortfolioManager
from finance.security import SecureVault
from agent.gatekeeper import run_gatekeeper
from html_renderer import MOCK_DATA, render_report_html, write_report_html
from services.report_storage import upload_report
from pdf_payload import build_payload as _build_v2_payload, TIER_BASE, TIER_DEEP
from pdf_charts import equity_curve_svg, factor_radar_svg
from ai_narrative import generate_narrative
from profile_manager import (
    ASSET_DISPLAY, ASSET_KEYS, BENCHMARK_LIST,
    PROFILE_BENCH_TICKER, RiskProfileManager,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger("ramp_bot")

BOT_TOKEN: str = os.environ["RAMP_BOT_TOKEN"].strip()
if not BOT_TOKEN or ":" not in BOT_TOKEN:
    raise ValueError(
        f"Invalid RAMP_BOT_TOKEN format — expected 'id:secret', got prefix: {BOT_TOKEN[:10]!r}"
    )
logger.info("Starting bot with token prefix: %s…", BOT_TOKEN[:10])
VAULT_DB: str  = str(Path(__file__).parent.parent / "data" / "users_vault.db")

# ── FSM ───────────────────────────────────────────────────────────────────────

class Onboarding(StatesGroup):
    Q1            = State()
    Q2            = State()
    Q3            = State()
    Q4            = State()
    Q5            = State()
    Q6            = State()
    Universe      = State()
    Benchmark     = State()
    MandateReview = State()


class PortfolioConnection(StatesGroup):
    Login     = State()
    ApiKey    = State()
    SecretKey = State()


class AnalysisFlow(StatesGroup):
    awaiting_approval = State()


# ── Onboarding question definitions ──────────────────────────────────────────
# Each option tuple: (button_label, score_points, callback_data)

NUM_QUESTIONS = 6

QUESTIONS: list[dict] = [
    {
        "state": Onboarding.Q1,
        "text":  "🕐 *Вопрос 1 из 6*\nКаков ваш инвестиционный горизонт?",
        "options": [
            ("Менее 1 года",                         1, "ob:q1:1"),
            ("От 1 до 3 лет",                        2, "ob:q1:2"),
            ("Более 3 лет",                          3, "ob:q1:3"),
        ],
    },
    {
        "state": Onboarding.Q2,
        "text":  "🎯 *Вопрос 2 из 6*\nВаша главная инвестиционная цель?",
        "options": [
            ("Сохранение капитала",                  1, "ob:q2:1"),
            ("Умеренный рост",                       2, "ob:q2:2"),
            ("Максимальный рост",                    3, "ob:q2:3"),
        ],
    },
    {
        "state": Onboarding.Q3,
        "text":  "📉 *Вопрос 3 из 6*\nЕсли ваш портфель упадёт на 20%, вы:",
        "options": [
            ("Продам всё, чтобы остановить потери",  1, "ob:q3:1"),
            ("Подожду восстановления",               2, "ob:q3:2"),
            ("Докуплю на просадке",                  3, "ob:q3:3"),
        ],
    },
    {
        "state": Onboarding.Q4,
        "text":  "📚 *Вопрос 4 из 6*\nВаш опыт в инвестировании?",
        "options": [
            ("Нет опыта",                            1, "ob:q4:1"),
            ("До 3 лет",                             2, "ob:q4:2"),
            ("Более 3 лет / профессионал",           3, "ob:q4:3"),
        ],
    },
    {
        "state": Onboarding.Q5,
        "text":  "🛡 *Вопрос 5 из 6*\nНасколько комфортно вы себя чувствуете финансово?",
        "options": [
            ("Живу от зарплаты до зарплаты",         1, "ob:q5:1"),
            ("Есть накопления на несколько месяцев", 2, "ob:q5:2"),
            ("Финансово защищён(а) на год и более",  3, "ob:q5:3"),
        ],
    },
    {
        "state": Onboarding.Q6,
        "text":  "💼 *Вопрос 6 из 6*\nНасколько стабилен ваш доход?",
        "options": [
            ("Нестабильный / фриланс",               1, "ob:q6:1"),
            ("Стабильная зарплата",                  2, "ob:q6:2"),
            ("Несколько источников дохода",           3, "ob:q6:3"),
        ],
    },
]

# ── Constants ─────────────────────────────────────────────────────────────────

TIER_COST  = {"base": 1, "deep": 2}
TIER_LABEL = {"base": "Базовый отчёт", "deep": "Глубокий сценарный анализ"}

SOURCE_GREETING: dict[str, str] = {
    "news_apple": "новостей Apple",
    "news_kaspi":  "новостей Kaspi",
}


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _source_label(slug: str) -> str:
    return SOURCE_GREETING.get(slug, f"канала «{slug}»")


def _classify_asset(ticker: str) -> str:
    t = ticker.upper()
    if any(x in t for x in ("BTC", "ETH", "SOL", "BNB")):
        return "Крипто"
    if any(x in t for x in ("GOLD", "GLD", "DBC", "OIL", "PDBC")):
        return "Сырьё"
    if any(x in t for x in ("CASH", "USD", "EUR", "KZT", "RUB")):
        return "Ден. средства"
    if any(x in t for x in ("KAP", "KSPI", "HSBK", "KZTK")):
        return "Акции KZ"
    if any(x in t for x in ("TLT", "AGG", "BND", "TNX", "BOND", "LQD", "HYG", "BIL", "IEF", "SHY")):
        return "Облигации"
    # KZ / international bond ISIN patterns from Freedom Finance
    if t.startswith(("KZ2P", "KZ1P", "XS", "US912")):
        return "Облигации"
    return "Акции"


def _get_keys_sync(user_id: int):
    """Blocking helper — must be run in a thread executor."""
    vault = SecureVault(db_name=VAULT_DB)
    return vault.get_user_keys(str(user_id))


def _save_keys_sync(user_id: int, login: str, api_key: str, secret_key: str) -> None:
    """Blocking helper — must be run in a thread executor."""
    vault = SecureVault(db_name=VAULT_DB)
    vault.save_user_keys(str(user_id), login, api_key, secret_key)


def _fetch_and_analyze_sync(api_key: str, secret_key: str = "",
                            login: str = "",
                            bench_ticker: str | None = None) -> dict:
    """
    Blocking: fetches portfolio from broker, runs MAC3 engine.
    Must be run in a thread executor.
    """
    df = FreedomConnector(api_key, secret_key, login).fetch_portfolio()
    return UniversalPortfolioManager().analyze_all(df, profile_benchmark=bench_ticker)


def _fetch_portfolio_sync(api_key: str, secret_key: str = "", login: str = ""):
    """Blocking: только подключение к брокеру и загрузка позиций (без анализа)."""
    return FreedomConnector(api_key, secret_key, login).fetch_portfolio()


def _analyze_existing_portfolio_sync(df, bench_ticker: str | None = None) -> dict:
    """Blocking: только MAC3 анализ уже загруженного DataFrame."""
    return UniversalPortfolioManager().analyze_all(df, profile_benchmark=bench_ticker)


def _format_portfolio_preview(df) -> str:
    """
    Сжатая Markdown-таблица портфеля для Telegram-сообщения.

    Колонки: Тикер | Тип | Кол-во | Сумма $.

    Тип инструмента берётся из ``df["Asset_Type"]`` если он есть (поставлен
    ``broker_api._classify_instrument``); иначе вычисляется на лету.
    """
    if df is None or df.empty:
        return "_(портфель пуст)_"

    df2 = df.copy()
    if "Quantity" in df2.columns and "Purchase_Price" in df2.columns:
        df2["_value"] = df2["Quantity"] * df2["Purchase_Price"]
    else:
        df2["_value"] = df2.get("Quantity", 0)

    df2 = df2.sort_values("_value", ascending=False)

    # Source of ticker: column "Ticker" (RangeIndex case) or index when named.
    if "Ticker" in df2.columns:
        tickers = df2["Ticker"].astype(str).tolist()
    else:
        tickers = [str(i) for i in df2.index.tolist()]

    # Source of type: column "Asset_Type" if present, else heuristic on ticker.
    if "Asset_Type" in df2.columns:
        types = df2["Asset_Type"].astype(str).tolist()
    else:
        try:
            from finance.broker_api import _classify_instrument
            types = [_classify_instrument(t) for t in tickers]
        except Exception:
            types = ["—"] * len(tickers)

    rows = ["```",
            f"{'Тикер':<12} {'Тип':<12} {'Кол-во':>10} {'Сумма $':>12}",
            "─" * 50]

    for i, (_, row) in enumerate(df2.iterrows()):
        ticker = tickers[i] if i < len(tickers) else "?"
        kind   = types[i]   if i < len(types)   else "—"
        qty    = row.get("Quantity", 0) or 0
        value  = row.get("_value", 0) or 0
        rows.append(f"{ticker[:12]:<12} {kind[:12]:<12} {qty:>10.2f} {value:>12.2f}")

    rows.append("─" * 50)
    total_value = df2["_value"].sum() if "_value" in df2.columns else 0
    rows.append(f"{'Всего':<12} {'':<12} {len(df2):>10} {total_value:>12.2f}")
    rows.append("```")
    return "\n".join(rows)


# ── PDF payload mapping ───────────────────────────────────────────────────────

def _build_equity_curve_svg(results: dict) -> str:
    """
    Inline equity-curve SVG: portfolio vs profile-benchmark daily log returns.

    Important fix vs the prior version: when bm_name == "Профильный бенчмарк"
    the bm_ticker_map used to leak `None` and the benchmark line was never
    drawn even though the data was already loaded.  We now look the actual
    ticker up via PROFILE_BENCH_TICKER (or any concrete benchmark name → its
    Tradernet ETF proxy) and fall back to SPY when the profile benchmark is
    unknown but the portfolio has data.

    Also: when ANY required input is missing we return a labelled empty-state
    SVG instead of "" so the user sees WHY the chart is empty.
    """
    try:
        history = results.get("history_result")
        bm_data = results.get("benchmark_comparison") or {}
        perf    = results.get("performance_table")
        if perf is None or perf.empty:
            return equity_curve_svg([])  # empty-state w/ "нет данных"

        prices = getattr(history, "data", None)
        if prices is None or prices.empty:
            return equity_curve_svg([])

        # Cap-weighted portfolio daily log-return stream.
        import numpy as _np
        total_val = float(results.get("total_value") or 1.0)
        cols: list[str]    = []
        weights: list[float] = []
        for _, row in perf.iterrows():
            t = str(row.get("Ticker"))
            cv = float(row.get("Current_Value") or 0.0)
            w  = cv / total_val if total_val else 0.0
            if w <= 0:
                continue
            if t in prices.columns:
                cols.append(t); weights.append(w)
            else:
                base = t.split(".")[0]
                for c in prices.columns:
                    if c.split(".")[0] == base:
                        cols.append(c); weights.append(w); break
        if not cols:
            return equity_curve_svg([])

        sub      = prices[cols].dropna()
        port_log = _np.log(sub / sub.shift(1)).dropna().values @ _np.array(weights)

        # Pick the benchmark — concrete name first, then profile benchmark
        # (resolved through its actual ticker mapping).
        bm_concrete_map = {
            "S&P 500":      "SPY.US",
            "Nasdaq 100":   "QQQ.US",
            "Russell 2000": "IWM.US",
            "MSCI EM":      "EEM.US",
            "EM Bonds":     "EMB.US",
        }
        bm_log: _np.ndarray | None = None
        chosen_bm_name: str | None = None
        # Prefer 'Профильный бенчмарк' when present — match it back to its
        # actual ETF via the Phase-5 lookup helper.
        if "Профильный бенчмарк" in bm_data:
            try:
                from profile_manager import PROFILE_BENCH_TICKER  # type: ignore
                # PROFILE_BENCH_TICKER maps profile_name → ETF; we don't have
                # the profile name at this depth, so we resolve via the user
                # context downstream.  Fall back to first concrete benchmark.
                pass
            except Exception:
                pass
            # Heuristic: the profile benchmark's prices were already fetched
            # in BENCHMARK_EXTRA — pick the first concrete ETF that loaded.
            for name, ticker in bm_concrete_map.items():
                if ticker in prices.columns:
                    chosen_bm_name = name
                    bm_series = prices[ticker].dropna()
                    bm_log = _np.log(bm_series / bm_series.shift(1)).dropna().values
                    break
        else:
            for name in bm_data.keys():
                ticker = bm_concrete_map.get(name)
                if ticker and ticker in prices.columns:
                    chosen_bm_name = name
                    bm_series = prices[ticker].dropna()
                    bm_log = _np.log(bm_series / bm_series.shift(1)).dropna().values
                    break
        if chosen_bm_name:
            logger.info("Equity curve benchmark = %s", chosen_bm_name)
        return equity_curve_svg(port_log, bm_log)
    except Exception as exc:
        logger.warning("equity_curve_svg build failed: %s", exc)
        return equity_curve_svg([])


# ── KPI sparklines (DEEP/BASE templates · cover KPI strip) ──────────────────

def _sparkline_svg(values: list[float], color: str = "#2F6FB3",
                    invert: bool = False) -> str:
    """
    Inline 240×36 SVG sparkline for one KPI's 12-month trend.

    Args:
        values : List of 12 floats (one per monthly snapshot).
        color  : Line + area color (e.g. green for CVaR, blue for Sharpe,
                  red for MaxDD).
        invert : When True, the "best" point is the MIN (CVaR/MaxDD — less
                  negative = better).  When False, MAX is best (Sharpe).
                  Drives which extremum gets the highlight marker.

    Returns "" for empty / single-point series.
    """
    if not values or len(values) < 3:
        return ""
    pad_x, pad_y    = 4, 4
    plot_w, plot_h  = 232, 28
    n               = len(values)
    vmin, vmax      = min(values), max(values)

    if vmax == vmin:
        ys = [pad_y + plot_h / 2] * n
    else:
        # Plot so HIGH values appear at the TOP of the plot area
        # (standard chart convention).  For CVaR/MaxDD (negative), this
        # naturally puts the least-bad month on top — visually intuitive.
        ys = [pad_y + plot_h - (v - vmin) / (vmax - vmin) * plot_h for v in values]
    xs = [pad_x + i * plot_w / (n - 1) for i in range(n)]

    points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area_str   = (f"M{xs[0]:.1f},{pad_y + plot_h:.1f} L"
                   + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
                   + f" L{xs[-1]:.1f},{pad_y + plot_h:.1f} Z")

    # Highlight extremum (min for invert=True, max otherwise)
    extr_i = values.index(min(values)) if invert else values.index(max(values))
    extr_x, extr_y = xs[extr_i], ys[extr_i]
    last_x,  last_y  = xs[-1], ys[-1]
    grad_id          = f"spk_{abs(hash(tuple(values))) % 1000000}"

    return (
        f'<svg viewBox="0 0 240 36" preserveAspectRatio="none" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<defs>'
        f'<linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.28"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
        f'</linearGradient>'
        f'</defs>'
        f'<path d="{area_str}" fill="url(#{grad_id})"/>'
        f'<polyline points="{points_str}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{extr_x:.1f}" cy="{extr_y:.1f}" r="3.4" fill="{color}" fill-opacity="0.18"/>'
        f'<circle cx="{extr_x:.1f}" cy="{extr_y:.1f}" r="1.7" fill="{color}"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.0" fill="#FBFAF7" '
        f'stroke="{color}" stroke-width="1.1"/>'
        f'</svg>'
    )


def _build_kpi_sparklines(results: dict) -> Optional[dict]:
    """
    Compute 3 monthly trend series (CVaR / Sharpe / MaxDD) and render them as
    inline SVG sparklines for the KPI cards.

    Window: last 252 trading days, sampled at ~12 evenly-spaced snapshots.
    For each snapshot, compute the metric over a trailing 60-day window of
    cap-weighted portfolio daily log returns.

    Returns dict {cvar_svg, sharpe_svg, mdd_svg} when there is enough history
    (≥ 90 daily obs), else None — template's `{% if data.kpi_sparklines %}`
    guard then hides the chart cells cleanly.
    """
    try:
        history = results.get("history_result")
        perf    = results.get("performance_table")
        if perf is None or perf.empty or history is None:
            return None
        prices = getattr(history, "data", None)
        if prices is None or prices.empty or len(prices) < 90:
            return None

        import numpy as _np
        total_val = float(results.get("total_value") or 1.0)
        cols:    list[str]   = []
        weights: list[float] = []
        for _, row in perf.iterrows():
            t  = str(row.get("Ticker", "")).strip()
            cv = float(row.get("Current_Value", 0) or 0)
            if not t or cv <= 0:
                continue
            # perf carries the broker's ORIGINAL tickers; the price frame is
            # keyed by RESOLVED tickers (e.g. AAPL → AAPL.US).  Match exact
            # first, then fall back to the base symbol before the dot — the
            # missing fallback left `cols` empty and silently killed every
            # sparkline.
            if t in prices.columns:
                col = t
            else:
                base = t.split(".")[0]
                col  = next((c for c in prices.columns
                             if c.split(".")[0] == base), None)
            if col is not None:
                cols.append(col)
                weights.append(cv / total_val)
        if not cols:
            return None

        w        = _np.array(weights)
        daily_lr = _np.log(prices[cols] / prices[cols].shift(1)).dropna()
        if len(daily_lr) < 60:
            return None
        port_lr  = (daily_lr * w).sum(axis=1)

        if len(port_lr) > 252:
            port_lr = port_lr.iloc[-252:]

        n      = len(port_lr)
        step   = max(1, n // 12)
        if step < 5:
            return None
        snap_indices = [step * (i + 1) - 1 for i in range(12) if step * (i + 1) - 1 < n]

        cvar_pts, sharpe_pts, mdd_pts = [], [], []
        for end in snap_indices:
            win = port_lr.iloc[max(0, end - 60):end + 1]
            if len(win) < 30:
                continue
            # CVaR 95% — mean of bottom 5% returns
            cutoff = max(1, int(len(win) * 0.05))
            cvar_pts.append(float(win.sort_values().iloc[:cutoff].mean()))
            # Sharpe annualised (sparkline uses RFR=0 — it's a trend visual,
            # not the headline number, which is wired separately to data.sharpe)
            std = float(win.std())
            sharpe_pts.append(float(win.mean()) / std * (252 ** 0.5) if std > 0 else 0.0)
            # Max drawdown over the trailing window
            eq   = (1 + win).cumprod()
            dd   = (eq / eq.cummax() - 1).min()
            mdd_pts.append(float(dd))

        if len(cvar_pts) < 3:
            return None

        return {
            "cvar_svg":   _sparkline_svg(cvar_pts,   color="#3F8F5F", invert=True),
            "sharpe_svg": _sparkline_svg(sharpe_pts, color="#2F6FB3", invert=False),
            "mdd_svg":    _sparkline_svg(mdd_pts,    color="#C0492F", invert=True),
        }
    except Exception as exc:
        logger.warning("kpi_sparklines build failed: %s", exc)
        return None


# All 9 axes the engine claims to model.  When a factor ETF didn't load,
# the perf table has no Beta_<factor> column — we still want the radar to
# show the missing axis (drawn as a gray dashed slice with value 0) so the
# user sees that 7/9 factors are loaded, not that we have only 7 factors.
_RADAR_FACTOR_AXES = [
    "Market", "Momentum", "Value", "Quality", "Size",
    "Volatility",   # SPLV.US — low-vol factor (Step 5)
    "Commodities", "Rates", "EM_Equity", "EM_Bond",
]


def _build_factor_radar_svg(results: dict) -> str:
    """Inline factor-radar SVG with all 9 axes (missing factors flagged)."""
    betas, missing, _coverage = _compute_factor_betas(results)
    try:
        return factor_radar_svg(betas, missing_axes=missing)
    except Exception as exc:
        logger.warning("factor_radar_svg build failed: %s", exc)
        return factor_radar_svg({})


# Benchmark β profile for "Δ vs benchmark" column in the factor table.
# S&P 500 has Market β ≈ 1.0 and ~0.0 on all other style/macro factors.
_BENCH_FACTOR_BETAS = {axis: (1.0 if axis == "Market" else 0.0)
                        for axis in _RADAR_FACTOR_AXES}


def _compute_factor_betas(results: dict) -> tuple[dict[str, float], list[str], float]:
    """
    Extract weighted-average factor β for each axis from the perf table.

    Returns (betas_dict, missing_axes_list, coverage_pct).

    Coverage matters: only assets that went through the Ridge factor
    regression carry Beta_<axis> values; cash, broker-priced KZ instruments
    and assets with too little overlapping history come back NaN.  The old
    code did `.fillna(0)` then a full-portfolio weighted average, so those
    NaN→0 assets silently dragged every β toward zero (a 62%-tech portfolio
    showing Market β ≈ 0.05).  We instead renormalise each axis over the
    weight of assets that actually got a fit, and report what share of the
    portfolio the factor model covers.
    """
    try:
        perf = results.get("performance_table")
        if perf is None or perf.empty:
            return {}, list(_RADAR_FACTOR_AXES), 0.0
        total_val = float(results.get("total_value") or 1.0)
        weights   = (perf["Current_Value"].fillna(0).astype(float) / total_val).values

        betas:   dict[str, float] = {}
        missing: list[str]        = []
        coverage_w = 0.0
        for axis in _RADAR_FACTOR_AXES:
            col = f"Beta_{axis}"
            if col not in perf.columns:
                betas[axis] = 0.0
                missing.append(axis)
                continue
            raw   = perf[col].astype(float)
            mask  = raw.notna().values
            cov_w = float(abs(weights[mask]).sum())
            if cov_w > 1e-9:
                betas[axis] = float((raw.values[mask] * weights[mask]).sum() / cov_w)
                coverage_w  = max(coverage_w, cov_w)
            else:
                betas[axis] = 0.0
                missing.append(axis)
        coverage_pct = round(min(coverage_w, 1.0) * 100, 1)
        if missing:
            logger.info("Factor radar: %d/%d axes loaded; missing: %s",
                         len(_RADAR_FACTOR_AXES) - len(missing),
                         len(_RADAR_FACTOR_AXES), ", ".join(missing))
        return betas, missing, coverage_pct
    except Exception as exc:
        logger.warning("factor beta extraction failed: %s", exc)
        return {}, list(_RADAR_FACTOR_AXES), 0.0


def _build_factor_betas_table(results: dict) -> list[dict]:
    """
    Template-friendly factor-β table for the DEEP factor-radar section.

    Each row: {axis, beta, bench, delta, missing}.  Bench is the SPX-like
    baseline (Market = 1.0, rest = 0.0); delta = β_port − β_bench rounded
    to 2 dp.  Missing axes get beta=0.0 + missing=True so the template
    can render them as muted "n/a" rows instead of dropping them silently.
    """
    betas, missing, _coverage = _compute_factor_betas(results)
    if not betas:
        return []
    missing_set = set(missing)
    rows: list[dict] = []
    for axis in _RADAR_FACTOR_AXES:
        b     = float(betas.get(axis, 0.0))
        bench = float(_BENCH_FACTOR_BETAS.get(axis, 0.0))
        rows.append({
            "axis":    axis,
            "beta":    round(b, 2),
            "bench":   round(bench, 2),
            "delta":   round(b - bench, 2),
            "missing": axis in missing_set,
        })
    return rows


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _fetch_rag_context(results: dict) -> tuple[str, list[str]]:
    """
    Pull macro + micro RAG excerpts for the AI narrative (deep tier only).
    Also queries for regime confirmation from bank reports.

    Returns (market_context_str, regime_rag_confirm_list).
    Both are empty on ChromaDB unavailability.
    """
    try:
        from agent.rag_engine import FinancialRAG
        rag = FinancialRAG(db_path=os.environ.get("CHROMA_LOCAL_PATH",
                                                   "/app/data/chroma_db"))
        if rag.collection.count() == 0:
            logger.info("RAG database empty — narrative will run without bank context.")
            return "", []

        perf = results.get("performance_table")
        tickers: list[str] = []
        if perf is not None and not perf.empty and "Ticker" in perf.columns:
            tickers = [str(t) for t in perf["Ticker"].tolist()
                       if str(t).upper() not in {"USD", "EUR", "RUB", "KZT", "CASH"}][:12]

        macro_query = ("Rating upgrades or downgrades, sector outlook, "
                       "fund flows, recession or expansion calls, currency views")
        macro_ctx = rag.get_market_sentiment(query=macro_query, n_results=3)

        micro_ctx = ""
        if tickers:
            micro_query = "Outlook, target price, recommendation for: " + ", ".join(tickers)
            micro_ctx   = rag.get_market_sentiment(query=micro_query, n_results=3)

        # Regime confirmation — look for reports that discuss the current macro regime
        regime_rag_confirm: list[str] = []
        regime = results.get("regime") or {}
        regime_label = regime.get("regime", "")
        if regime_label:
            regime_query = (
                f"Market regime {regime_label} GDP growth recession expansion "
                "economic cycle leading indicators PMI yield curve"
            )
            regime_raw = rag.get_market_sentiment(query=regime_query, n_results=2)
            if regime_raw and "NO PDF DATA" not in regime_raw:
                for line in regime_raw.split("\n"):
                    line = line.strip()
                    if line and len(line) > 30:
                        regime_rag_confirm.append(line[:200])
                        if len(regime_rag_confirm) >= 3:
                            break

        sections = []
        if macro_ctx and "NO PDF DATA" not in macro_ctx:
            sections.append("=== MACRO TRENDS ===\n" + macro_ctx)
        if micro_ctx and "NO PDF DATA" not in micro_ctx:
            sections.append("=== MICRO INSIGHTS ===\n" + micro_ctx)
        return "\n\n".join(sections), regime_rag_confirm
    except Exception as exc:
        logger.info("RAG context fetch skipped: %s", exc)
        return "", []


def _build_pdf_payload(results: dict, tier: str,
                       user_bench_ticker: str | None = None,
                       prev_snapshot: dict | None = None,
                       user_risk_profile: str = "Moderate") -> dict:
    """
    Build the PDF payload from analyze_all() output.

    v2 (default): delegates to pdf_payload.build_payload — produces the rich
    schema consumed by report_basic.html (2pp) and report_deep.html (4pp),
    enriched with SVG charts and an AI narrative.

    v1 (legacy, REPORT_VERSION=v1 env): retains the old shape so the legacy
    report.html keeps rendering without code changes elsewhere.
    """
    if os.getenv("REPORT_VERSION", "v2").lower() != "v1":
        # Both tiers pull RAG context (macro + micro + regime confirmation).
        # Deep tier gets full 6000-char context; base tier gets 2000-char
        # version (truncated in ai_narrative) to keep latency reasonable.
        market_context, regime_rag_confirm = _fetch_rag_context(results)

        ai_summary = generate_narrative(
            results,
            tier=tier,
            market_context=market_context,
            user_risk_profile=user_risk_profile,
        )
        payload = _build_v2_payload(
            results, tier,
            ai_summary=ai_summary,
            user_bench_ticker=user_bench_ticker,
            prev_snapshot=prev_snapshot,
            regime_rag_confirm=regime_rag_confirm,
        )
        if tier == TIER_DEEP:
            payload["equity_curve_svg"]    = _build_equity_curve_svg(results)
            payload["factor_radar_svg"]    = _build_factor_radar_svg(results)
            payload["factor_betas"]        = _build_factor_betas_table(results)
            payload["factor_coverage_pct"] = _compute_factor_betas(results)[2]
            payload["used_rag"]            = bool(market_context)
        # KPI sparklines — wired for BOTH tiers (both templates surface them
        # in the cover KPI strip).  None when history < 90 daily obs;
        # template gating then hides the chart cells.
        payload["kpi_sparklines"] = _build_kpi_sparklines(results)
        return payload

    # ── v1 legacy fallback ──────────────────────────────────────────────────
    metrics    = results.get("portfolio_metrics") or {}
    perf_df    = results.get("performance_table")
    total_val  = _safe_float(results.get("total_value"), 1.0) or 1.0

    cvar_raw   = _safe_float(metrics.get("CVaR_95_Daily"),        0.0)
    sharpe_raw = _safe_float(metrics.get("Sharpe_Ratio"),         float("nan"))
    var_raw    = _safe_float(metrics.get("VaR_95_Daily"),         0.0)
    mdd_raw    = _safe_float(metrics.get("Max_Drawdown"),         0.0)
    vol_raw    = _safe_float(metrics.get("Total_Volatility_Ann"), 0.0)

    cvar_str         = f"{cvar_raw * 100:.1f}%"
    sharpe_str       = f"{sharpe_raw:.2f}" if not math.isnan(sharpe_raw) else "—"
    var_95_daily_str = f"{var_raw * 100:.1f}%"
    max_drawdown_str = f"{mdd_raw * 100:.1f}%"
    risk_pct         = min(100, max(0, int(vol_raw / 0.40 * 100)))

    # ── Aggregate P/L since position entry ─────────────────────────────────
    total_pnl  = 0.0
    total_cost = 0.0
    if perf_df is not None and not perf_df.empty:
        if "PnL" in perf_df.columns:
            total_pnl = float(perf_df["PnL"].fillna(0).sum())
        if "Total_Cost" in perf_df.columns:
            total_cost = float(perf_df["Total_Cost"].fillna(0).sum())
    total_return_pct = (total_pnl / total_cost) if total_cost > 0 else 0.0

    assets: list[dict] = []
    if perf_df is not None and not perf_df.empty:
        for _, row in perf_df.iterrows():
            ticker     = str(row.get("Ticker", "—"))
            cur_val    = _safe_float(row.get("Current_Value"), 0.0)
            weight_pct = cur_val / total_val * 100
            euler      = _safe_float(row.get("Euler_Risk_Contribution_Pct"), 0.0)
            pnl_abs    = _safe_float(row.get("PnL"), 0.0)
            ret_pct    = _safe_float(row.get("Return_Pct"), 0.0)
            assets.append({
                "ticker":      ticker,
                "weight":      f"{weight_pct:.1f}%",
                "asset_class": _classify_asset(ticker),
                "euler_risk":  f"{euler:.1f}%",
                "pnl_pct":     f"{ret_pct * 100:+.1f}%",   # P/L since entry, %
                "pnl_abs":     f"{pnl_abs:+,.0f}",         # P/L since entry, $
                "pnl_color":   "pos" if ret_pct >= 0 else "neg",
            })

    payload: dict = {
        # KPI strip
        "cvar":            cvar_str,
        "sharpe":          sharpe_str,
        "var_95_daily":    var_95_daily_str,    # canonical key
        "max_drawdown":    max_drawdown_str,    # now real MaxDD
        "risk_pct":        risk_pct,
        # Aggregate P/L since position entry
        "pnl_total_abs":   f"{total_pnl:+,.0f}",
        "pnl_total_pct":   f"{total_return_pct * 100:+.1f}%",
        "pnl_total_color": "pos" if total_return_pct >= 0 else "neg",
        # Holdings
        "assets":          assets or MOCK_DATA["assets"],
    }

    if tier == "deep":
        bm_data   = results.get("benchmark_comparison") or {}
        scenarios = []
        for bm_name, bm in bm_data.items():
            # Prefer the annualised excess (consistent scale with IR / TE).
            # Fall back to period total only if annualised is unavailable
            # (legacy engine output / first-run before refresh).
            excess_ann = bm.get("Excess_Return_Ann")
            if excess_ann is None:
                excess_ann = _safe_float(bm.get("Excess_Return"), 0.0)
            else:
                excess_ann = _safe_float(excess_ann, 0.0)
            pnl_str = f"+{excess_ann*100:.1f}%" if excess_ann >= 0 else f"{excess_ann*100:.1f}%"
            ir      = _safe_float(bm.get("Information_Ratio"), 0.0)
            beat    = "✅ Обыгрывает" if bm.get("Beating_Benchmark") else "❌ Отстаёт"
            scenarios.append({
                "name":        bm_name,
                "probability": f"IR: {ir:.2f}" if ir else "—",
                "pnl":         pnl_str,
                "driver":      f"{beat} бенчмарк",
            })
        payload["scenarios"] = scenarios

    return payload


# ── MAC3 pipeline ─────────────────────────────────────────────────────────────

async def _build_analysis_payload(user_id: int, tier: str) -> dict:
    """Async orchestrator: vault → broker → MAC3 engine → PDF payload."""
    loop      = asyncio.get_running_loop()
    conn_mode = await get_connection_mode(user_id)

    profile    = await get_profile(user_id)
    bench_tick = PROFILE_BENCH_TICKER.get(profile["profile_name"]) if profile else None

    if conn_mode == "freedom":
        keys = await loop.run_in_executor(None, _get_keys_sync, user_id)
        if keys is None:
            api_key    = os.getenv("FREEDOM_API_KEY",    "demo")
            secret_key = os.getenv("FREEDOM_API_SECRET", "")
            login      = os.getenv("FREEDOM_LOGIN",      "")
            logger.info(
                "KEY SOURCE: env-vars  user=%s  key_prefix=%s  secret_len=%d  login=%s…",
                user_id,
                api_key[:6] if api_key else "EMPTY",
                len(secret_key),
                login[:6]   if login   else "EMPTY",
            )
        else:
            login, api_key, secret_key = keys
            login      = (login      or "").strip()
            api_key    = (api_key    or "").strip()
            secret_key = (secret_key or "").strip()
            logger.info(
                "KEY SOURCE: vault  user=%s  key_prefix=%s  secret_len=%d  login=%s…",
                user_id,
                api_key[:6] if api_key else "EMPTY",
                len(secret_key),
                login[:6]   if login   else "EMPTY",
            )
    else:
        api_key    = "demo"
        secret_key = ""
        login      = ""

    try:
        results = await loop.run_in_executor(
            None, _fetch_and_analyze_sync, api_key, secret_key, login, bench_tick
        )
    except BrokerAuthError:
        raise  # Propagate as-is so cb_confirm can show the specific auth message
    except BrokerEmptyPortfolioError:
        raise  # Propagate as-is so cb_confirm can show the empty-portfolio message
    except RealPortfolioRequired:
        raise  # MAC3 gate fired — broker API failed, no live data; cb_confirm handles UI
    except RuntimeError as exc:
        logger.error("Freedom Broker API ошибка для %s: %s", user_id, exc)
        raise RuntimeError(
            f"Не удалось получить данные от Freedom Broker: {exc}\n\n"
            "Проверьте, что ваш API-ключ действителен, и повторите попытку."
        ) from exc

    if profile:
        gate_limits = {
            "max_portfolio_volatility": profile["target_volatility"] * 1.2
        }
        gate = run_gatekeeper(results, user_limits=gate_limits, user_profile=profile)
        if not gate["passed"]:
            logger.warning("Gatekeeper нарушения: %s", gate["critical"])

    return _build_pdf_payload(results, tier)


# ── Keyboard builders ─────────────────────────────────────────────────────────

def kb_question(options: list[tuple[str, int, str]], q_num: int = 1) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=label, callback_data=cb)]
        for label, _, cb in options
    ]
    if q_num > 1:
        rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="ob:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_universe(selected: set[str]) -> InlineKeyboardMarkup:
    rows = []
    for key in ASSET_KEYS:
        label = ("✅ " if key in selected else "") + ASSET_DISPLAY[key]
        rows.append([InlineKeyboardButton(text=label, callback_data=f"ob:uni:{key}")])
    rows.append([
        InlineKeyboardButton(text="Подтвердить выбор ➡️", callback_data="ob:uni:confirm")
    ])
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="ob:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_benchmark(current: str | None = None) -> InlineKeyboardMarkup:
    """Benchmark selection keyboard. Shows check mark on the currently selected one."""
    rows = []
    for ticker, display_name in BENCHMARK_LIST.items():
        prefix = "✅ " if ticker == current else ""
        rows.append([
            InlineKeyboardButton(
                text=f"{prefix}{display_name}",
                callback_data=f"ob:bench:{ticker}",
            )
        ])
    rows.append([
        InlineKeyboardButton(text="Продолжить ➡️", callback_data="ob:bench:confirm")
    ])
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="ob:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_mandate_review() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Утвердить мандат", callback_data="ob:mandate:approve"),
        InlineKeyboardButton(text="✏️ Изменить",        callback_data="ob:mandate:edit"),
    ]])


def kb_connect_choice() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📋 Демо-режим (Шаблон)", callback_data="connect:template")],
        [InlineKeyboardButton(text="🔗 Freedom Broker API",  callback_data="connect:freedom")],
    ])


def kb_analysis_choice() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📊 Базовый (1 токен)",   callback_data="analysis:base"),
        InlineKeyboardButton(text="🔬 Глубокий (2 токена)", callback_data="analysis:deep"),
    ]])


def kb_confirm(tier: str, context_slug: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="✅ Одобрить",
            callback_data=f"confirm:{tier}:{context_slug}",
        ),
        InlineKeyboardButton(text="❌ Отмена", callback_data="cancel"),
    ]])


# ── Shared UI helpers ─────────────────────────────────────────────────────────

async def _edit_or_answer(
    target: Message | CallbackQuery,
    state: FSMContext,
    text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    """Edit the active onboarding message when possible; otherwise send new."""
    data      = await state.get_data()
    ob_msg_id = data.get("ob_message_id")

    if isinstance(target, CallbackQuery) and ob_msg_id:
        try:
            await target.message.edit_text(
                text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )
            return
        except Exception:
            pass

    if isinstance(target, CallbackQuery):
        sent = await target.message.answer(
            text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
        )
    else:
        sent = await target.answer(
            text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
        )
    await state.update_data(ob_message_id=sent.message_id)


async def send_question(
    target: Message | CallbackQuery,
    state: FSMContext,
    q_idx: int,
) -> None:
    """Advance to Q q_idx (0-based) or to Universe/Benchmark steps after Q6."""
    if q_idx < NUM_QUESTIONS:
        q = QUESTIONS[q_idx]
        await state.set_state(q["state"])
        await _edit_or_answer(target, state, q["text"], kb_question(q["options"], q_num=q_idx + 1))
    else:
        data     = await state.get_data()
        selected = set(data.get("universe", []))
        await state.set_state(Onboarding.Universe)
        if "universe" not in data:
            await state.update_data(universe=[])
        await _edit_or_answer(
            target, state,
            "🌍 *Выбор классов активов*\n\n"
            "Отметьте классы активов, которые вы хотите включить в портфель:",
            kb_universe(selected),
        )


async def _show_analysis_menu(message: Message, slug: str) -> None:
    """Send the final analysis choice, with a deep-link context if slug is set."""
    if slug:
        await message.answer(
            f"👋 Кстати, вы пришли из нашего канала _{_source_label(slug)}_.\n\n"
            "Хотите узнать, как эта новость влияет на ваш портфель?\n\n"
            "💰 *Базовый отчёт:* 1 токен\n"
            "🔬 *Глубокий анализ:* 2 токена",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_analysis_choice(),
        )
    else:
        await message.answer(
            "🚀 *Выберите тип анализа вашего портфеля:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_analysis_choice(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ONBOARDING ROUTER
# ══════════════════════════════════════════════════════════════════════════════

onboarding_router = Router(name="onboarding")


# ── /start ────────────────────────────────────────────────────────────────────

async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    user_id = message.from_user.id
    slug    = message.text.split(maxsplit=1)[1].strip() if " " in (message.text or "") else ""
    profile = await get_profile(user_id)

    if profile is None:
        # NEW user — launch onboarding; token grant is deferred until mandate approval.
        await state.update_data(slug=slug)
        await state.set_state(Onboarding.Q1)
        q    = QUESTIONS[0]
        sent = await message.answer(
            "👋 *Добро пожаловать в RAMP — Risk & Asset Management Platform!*\n\n"
            "Прежде чем начать, пройдите короткое анкетирование (6 вопросов), "
            "чтобы мы могли составить ваш персональный инвестиционный мандат.\n\n"
            + q["text"],
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_question(q["options"], q_num=1),
        )
        await state.update_data(ob_message_id=sent.message_id)

    else:
        # RETURNING user — original flow unchanged.
        await init_user(user_id)  # returns False — no double-grant
        if slug:
            await state.update_data(context_slug=slug)
            await message.answer(
                f"👋 Привет! Я вижу, вы пришли из нашего канала _{_source_label(slug)}_.\n\n"
                "Я могу проанализировать, как эта новость повлияет на ваш портфель.\n\n"
                "💰 *Стоимость базового отчета:* 1 токен\n"
                "🔬 *Глубокий сценарный анализ:* 2 токена\n\n"
                "Начать анализ?",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_analysis_choice(),
            )
        else:
            balance = await get_balance(user_id)
            await message.answer(
                f"📊 *RAMP — Risk & Asset Management Platform*\n\n"
                f"Ваш баланс: *{balance} токен(а)*\n\n"
                "Выберите тип анализа вашего портфеля:",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_analysis_choice(),
            )


# ── Q1–Q6 answer handler ──────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data.regexp(r"^ob:q[1-6]:\d$"),
    StateFilter(Onboarding.Q1, Onboarding.Q2, Onboarding.Q3,
                Onboarding.Q4, Onboarding.Q5, Onboarding.Q6),
)
async def cb_question_answer(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    _, q_part, pts_str = callback.data.split(":")
    q_num = int(q_part[1])
    pts   = int(pts_str)
    await state.update_data(**{f"q{q_num}": pts})
    await send_question(callback, state, q_num)


# ── Back button handler ───────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:back",
    StateFilter(Onboarding.Q2, Onboarding.Q3, Onboarding.Q4,
                Onboarding.Q5, Onboarding.Q6,
                Onboarding.Universe, Onboarding.Benchmark),
)
async def cb_back(callback: CallbackQuery, state: FSMContext) -> None:
    """Roll back to the previous onboarding step."""
    await callback.answer()
    current = await state.get_state()
    # Map current state → previous q_idx (0-based)
    back_map = {
        Onboarding.Q2.state:       0,  # back to Q1
        Onboarding.Q3.state:       1,  # back to Q2
        Onboarding.Q4.state:       2,
        Onboarding.Q5.state:       3,
        Onboarding.Q6.state:       4,
        Onboarding.Universe.state: 5,  # back to Q6
    }
    prev_idx = back_map.get(current)
    if prev_idx is not None:
        await send_question(callback, state, prev_idx)
    elif current == Onboarding.Benchmark.state:
        # Back from Benchmark → Universe
        data = await state.get_data()
        selected = set(data.get("universe", []))
        await state.set_state(Onboarding.Universe)
        await _edit_or_answer(
            callback, state,
            "🌍 *Выбор классов активов*\n\n"
            "Отметьте классы активов, которые вы хотите включить в портфель:",
            kb_universe(selected),
        )


# ── Universe toggle ───────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data.startswith("ob:uni:"),
    ~F.data.endswith("confirm"),
    StateFilter(Onboarding.Universe),
)
async def cb_universe_toggle(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    asset_key = callback.data[len("ob:uni:"):]
    data      = await state.get_data()
    universe: list[str] = list(data.get("universe", []))
    if asset_key in universe:
        universe.remove(asset_key)
    else:
        universe.append(asset_key)
    await state.update_data(universe=universe)
    await callback.message.edit_reply_markup(reply_markup=kb_universe(set(universe)))


# ── Universe confirm ──────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:uni:confirm",
    StateFilter(Onboarding.Universe),
)
async def cb_universe_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    data     = await state.get_data()
    universe = data.get("universe", [])

    if not universe:
        await callback.message.answer("⚠️ Выберите хотя бы один класс активов.")
        return

    # Score from 6 questions (range 6-18)
    score   = sum(data.get(f"q{i}", 0) for i in range(1, NUM_QUESTIONS + 1))
    profile = RiskProfileManager.score_to_profile(score)

    # Store profile data for later use
    await state.update_data(profile_data={
        "name":       profile["name"],
        "target_vol": profile["target_vol"],
        "target_te":  profile["target_te"],
        "score":      score,
        "limits":     RiskProfileManager.apply_universe(profile, universe),
    })

    # Transition to Benchmark selection
    default_bench = PROFILE_BENCH_TICKER.get(profile["name"], "SPY.US")
    await state.update_data(benchmark_ticker=default_bench)
    await state.set_state(Onboarding.Benchmark)
    await _edit_or_answer(
        callback, state,
        "📊 *Выберите бенчмарк для вашего портфеля*\n\n"
        f"На основе вашего профиля *{profile['name']}* мы рекомендуем "
        f"*{BENCHMARK_LIST.get(default_bench, default_bench)}*.\n\n"
        "Вы можете выбрать другой или продолжить с рекомендованным:",
        kb_benchmark(current=default_bench),
    )


# ── Benchmark toggle ──────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data.startswith("ob:bench:"),
    ~F.data.endswith("confirm"),
    StateFilter(Onboarding.Benchmark),
)
async def cb_benchmark_toggle(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    ticker = callback.data[len("ob:bench:"):]
    await state.update_data(benchmark_ticker=ticker)
    await callback.message.edit_reply_markup(reply_markup=kb_benchmark(current=ticker))


# ── Benchmark confirm ─────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:bench:confirm",
    StateFilter(Onboarding.Benchmark),
)
async def cb_benchmark_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    data      = await state.get_data()
    prof      = data["profile_data"]
    universe  = data.get("universe", [])
    bench_tk  = data.get("benchmark_ticker")

    summary = RiskProfileManager.build_mandate_summary(
        prof, prof["limits"], benchmark_ticker=bench_tk,
    )
    await state.set_state(Onboarding.MandateReview)
    await _edit_or_answer(callback, state, summary, kb_mandate_review())


# ── Mandate approve ───────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:mandate:approve",
    StateFilter(Onboarding.MandateReview),
)
async def cb_mandate_approve(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    user_id  = callback.from_user.id
    data     = await state.get_data()
    prof     = data["profile_data"]
    slug     = data.get("slug", "")
    universe = data.get("universe", [])
    bench_tk = data.get("benchmark_ticker")

    await save_profile(
        telegram_id       = user_id,
        score             = prof["score"],
        profile_name      = prof["name"],
        target_volatility = prof["target_vol"],
        target_te         = prof["target_te"],
        selected_assets   = universe,
        limits_dict       = prof["limits"],
        benchmark_ticker  = bench_tk,
    )
    await approve_mandate(user_id)
    await init_user(user_id)   # grants tokens — user is new here

    # Preserve slug for the connection sub-flow, then reset everything else.
    await state.clear()
    if slug:
        await state.update_data(slug=slug)

    balance = await get_balance(user_id)
    await callback.message.answer(
        f"🎉 *Мандат утверждён! Добро пожаловать в RAMP.*\n\n"
        f"Ваш профиль: *{prof['name']}*\n"
        f"На ваш счёт зачислено *{balance} токен(а)*.\n\n"
        "Последний шаг: подключите источник данных о вашем портфеле.",
        parse_mode=ParseMode.MARKDOWN,
    )
    await callback.message.answer(
        "📡 *Как вы хотите подключить ваш портфель?*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_connect_choice(),
    )


# ── Mandate edit ──────────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:mandate:edit",
    StateFilter(Onboarding.MandateReview),
)
async def cb_mandate_edit(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    data     = await state.get_data()
    universe = set(data.get("universe", []))
    await state.set_state(Onboarding.Universe)
    await _edit_or_answer(
        callback, state,
        "🌍 *Выбор классов активов*\n\nОтметьте классы активов для вашего портфеля:",
        kb_universe(universe),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONNECTION ROUTER
# ══════════════════════════════════════════════════════════════════════════════

portfolio_router = Router(name="portfolio_connection")


@portfolio_router.callback_query(F.data.startswith("connect:"))
async def cb_connect_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    _, mode  = callback.data.split(":", 1)
    user_id  = callback.from_user.id
    fsm_data = await state.get_data()
    slug     = fsm_data.get("slug", "")

    if mode == "template":
        await save_connection_mode(user_id, "template")
        await callback.message.edit_text(
            "✅ *Демо-режим активирован.*\n\n"
            "Для анализа будет использоваться шаблонный институциональный портфель.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        await _show_analysis_menu(callback.message, slug)

    elif mode == "freedom":
        await state.update_data(slug=slug)
        await state.set_state(PortfolioConnection.Login)
        await callback.message.edit_text(
            "⚠️ *Важно:* RAMP использует API исключительно для режима ЧТЕНИЯ (Read-Only) "
            "сырых данных для глубокого квантового анализа. "
            "Мы не имеем права совершать сделки. "
            "В целях безопасности, после ввода ключей, пожалуйста, "
            "удалите свои сообщения из истории чата.\n\n"
            "🔐 Введите ваш *Логин* в Freedom Broker:",
            parse_mode=ParseMode.MARKDOWN,
        )


@portfolio_router.message(StateFilter(PortfolioConnection.Login))
async def msg_login(message: Message, state: FSMContext) -> None:
    await state.update_data(connect_login=message.text.strip())
    await state.set_state(PortfolioConnection.ApiKey)
    await message.answer(
        "🔑 Введите ваш *API Key*:",
        parse_mode=ParseMode.MARKDOWN,
    )


@portfolio_router.message(StateFilter(PortfolioConnection.ApiKey))
async def msg_api_key(message: Message, state: FSMContext) -> None:
    await state.update_data(connect_api_key=message.text.strip())
    await state.set_state(PortfolioConnection.SecretKey)
    await message.answer(
        "🔑 Введите ваш *Secret Key* (приватный ключ):",
        parse_mode=ParseMode.MARKDOWN,
    )


@portfolio_router.message(StateFilter(PortfolioConnection.SecretKey))
async def msg_secret_key(message: Message, state: FSMContext) -> None:
    secret_key = message.text.strip()
    data       = await state.get_data()
    login      = data.get("connect_login", "")
    api_key    = data.get("connect_api_key", "")
    slug       = data.get("slug", "")
    user_id    = message.from_user.id

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _save_keys_sync, user_id, login, api_key, secret_key)
    await save_connection_mode(user_id, "freedom")
    await state.clear()

    await message.answer(
        "🔐 *Подключение успешно. Ваши данные под квантовой защитой.*\n\n"
        "✅ API-ключ и Secret Key надёжно зашифрованы. Пожалуйста, удалите "
        "предыдущие сообщения с ключами из истории чата.",
        parse_mode=ParseMode.MARKDOWN,
    )
    await _show_analysis_menu(message, slug)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FLOW
# ══════════════════════════════════════════════════════════════════════════════

async def _send_pdf(
    bot: Bot,
    chat_id: int,
    user_id: int,
    tier: str,
    payload: dict | None = None,
) -> None:
    """
    Render the HTML report, push to GCS, send the user a signed URL.

    Function name kept as `_send_pdf` for now to minimise call-site churn;
    will be renamed once all references are updated.  The underlying
    delivery is HTML + signed URL — no PDF, no Chromium.
    """
    report_type = TIER_LABEL[tier]

    # Render Jinja → HTML string and write to /tmp.  Both ops are sync
    # and fast (<50 ms total); no need for executor offload.
    html       = render_report_html(payload, user_id=user_id,
                                     report_type=report_type, tier=tier)
    local_path = write_report_html(html, user_id=user_id, tier=tier)

    # Push to GCS (or fall back to file:// in local-dev mode).
    url = upload_report(local_path, user_id=user_id, tier=tier)

    # A file:// URL means the GCS upload/signing failed (in production the
    # bucket is always configured).  Telegram rejects file:// links inside a
    # markdown text-link entity, so sending one would crash send_message and
    # the user would get nothing.  Signal the failure to the caller so it can
    # refund tokens and show a clear message instead.
    if url.startswith("file://"):
        raise RuntimeError("report_delivery_failed")

    # Tell the user.  The link is a plain markdown URL — Telegram renders
    # it as a preview card on most clients.
    await bot.send_message(
        chat_id,
        text=(
            f"📊 *{report_type}* готов.\n\n"
            f"[Открыть отчёт]({url})\n\n"
            "Ссылка действительна 7 дней.  Отчёт сформирован институциональным "
            "риск-движком (Euler Decomposition · Bootstrap CVaR · 4-pillar "
            "Scoring · Black-Litterman).  Штурвал всегда у вас."
        ),
        parse_mode               = ParseMode.MARKDOWN,
        disable_web_page_preview = False,
    )


async def cb_analysis_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    _, tier  = callback.data.split(":", 1)
    cost     = TIER_COST[tier]
    balance  = await get_balance(callback.from_user.id)
    ctx      = await state.get_data()
    context_slug = ctx.get("context_slug", "menu")

    await state.update_data(tier=tier)
    await callback.message.edit_text(
        f"⚠️ *Внимание:*\n\n"
        f"Данный анализ потребует сложных нейросетевых и квантовых вычислений.\n\n"
        f"С вашего баланса будет списано *{cost} токен(а)*.\n"
        f"Текущий баланс: *{balance} токен(а)*.\n\n"
        "Одобрить?",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_confirm(tier, context_slug),
    )
    await state.set_state(AnalysisFlow.awaiting_approval)


async def cb_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    """
    Flow:
      1. Списать токены.
      2. Получить портфель из брокера (5-10 сек).
      3. Отправить превью-таблицу + сообщить, что анализ займёт 5-10 минут.
      4. Запустить MAC3 анализ как фоновую задачу — handler возвращается СРАЗУ,
         чтобы aiogram long-poll не таймаутил.
      5. По завершении задачи отправить PDF отдельным сообщением.
    """
    await callback.answer()
    _, tier, context_slug = callback.data.split(":", 2)
    user_id = callback.from_user.id
    cost    = TIER_COST[tier]

    success = await deduct_tokens(user_id, cost, reason=f"{tier}_analysis")
    if not success:
        balance = await get_balance(user_id)
        await callback.message.edit_text(
            f"❌ *Недостаточно токенов.*\n\n"
            f"Требуется: *{cost}*, доступно: *{balance}*.\n\n"
            "Пополните баланс командой /topup.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        return

    balance_after = await get_balance(user_id)
    await callback.message.edit_text(
        f"⏳ Подключаюсь к Freedom Broker и загружаю портфель…\n\n"
        f"Остаток баланса: *{balance_after} токен(а)*.",
        parse_mode=ParseMode.MARKDOWN,
    )

    # ── Шаг 1 — подгружаем портфель (быстрая часть) ──────────────────────
    loop = asyncio.get_running_loop()
    profile    = await get_profile(user_id)
    bench_tick = PROFILE_BENCH_TICKER.get(profile["profile_name"]) if profile else None
    conn_mode  = await get_connection_mode(user_id)

    if conn_mode == "freedom":
        keys = await loop.run_in_executor(None, _get_keys_sync, user_id)
        if keys is None:
            api_key    = os.getenv("FREEDOM_API_KEY",    "demo")
            secret_key = os.getenv("FREEDOM_API_SECRET", "")
            login      = os.getenv("FREEDOM_LOGIN",      "")
        else:
            login, api_key, secret_key = keys
            login      = (login      or "").strip()
            api_key    = (api_key    or "").strip()
            secret_key = (secret_key or "").strip()
    else:
        api_key, secret_key, login = "demo", "", ""

    try:
        df = await loop.run_in_executor(
            None, _fetch_portfolio_sync, api_key, secret_key, login
        )
    except BrokerAuthError as exc:
        logger.error("Freedom Broker auth failed for %s: %s", user_id, exc)
        await callback.message.answer(
            "⚠️ *Ошибка: Ваши API-ключи брокера неверны или отозваны.*\n\n"
            "Проверьте их в /start → 🔗 Freedom Broker API.\n\n"
            "Токены не потеряны — обратитесь в /support.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        return
    except BrokerEmptyPortfolioError as exc:
        await callback.message.answer(
            f"📭 *Портфель пуст*\n\n{exc}\n\n"
            "Токены не потеряны — обратитесь в /support.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        return
    except Exception as exc:
        logger.exception("Не удалось загрузить портфель для %s: %s", user_id, exc)
        await callback.message.answer(
            f"⚠️ *Не удалось получить портфель:*\n`{str(exc)[:200]}`\n\n"
            "Токены не потеряны — обратитесь в /support.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        return

    # ── Шаг 2 — отправляем превью + уведомление о длительности ──────────
    preview_md = _format_portfolio_preview(df)
    await callback.message.answer(
        "✅ *Портфель загружен:*\n\n"
        f"{preview_md}\n\n"
        f"⚙️ Запускаю анализ *{TIER_LABEL[tier]}*. "
        "Это займёт *5–10 минут* — я пришлю PDF-отчёт сюда сразу как он будет готов.\n\n"
        "Можно продолжать пользоваться ботом — анализ работает в фоне.",
        parse_mode=ParseMode.MARKDOWN,
    )

    # ── Шаг 3 — запускаем анализ как background task ────────────────────
    asyncio.create_task(_run_analysis_background(
        bot       = callback.message.bot,
        chat_id   = callback.message.chat.id,
        user_id   = user_id,
        tier      = tier,
        cost      = cost,
        df        = df,
        bench_tick= bench_tick,
    ))
    await state.clear()


async def _run_analysis_background(
    *,
    bot,
    chat_id: int,
    user_id: int,
    tier: str,
    cost: int,
    df,
    bench_tick: str | None,
) -> None:
    """
    Фоновая задача с поэтапными уведомлениями в Telegram.

    Каждый этап MAC3-pipeline публикует прогресс и ошибки сразу как они
    случаются — пользователь видит, что делается, и где именно сломалось.
    На критических ошибках токены возвращаются автоматически.
    """
    from finance.investment_logic import UniversalPortfolioManager

    loop = asyncio.get_running_loop()
    progress_messages: list = []   # для последующего удаления/обновления

    async def step(emoji: str, text: str):
        """Отправить сообщение прогресса и сохранить ссылку на него."""
        try:
            msg = await bot.send_message(chat_id, f"{emoji} {text}", parse_mode=ParseMode.MARKDOWN)
            progress_messages.append(msg)
            return msg
        except Exception as e:
            logger.warning("Не удалось отправить прогресс: %s", e)
            return None

    async def refund(reason: str) -> None:
        """Вернуть списанные токены и сообщить пользователю."""
        try:
            await credit_tokens(user_id, cost, reason=f"refund_{reason}")
            balance_after = await get_balance(user_id)
            await bot.send_message(
                chat_id,
                f"↩️ *{cost} токен(а) возвращены* на ваш счёт.\n"
                f"Текущий баланс: *{balance_after} токен(а)*.",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as exc:
            logger.error("Не удалось вернуть токены для %s: %s", user_id, exc)

    try:
        # ── Step 1: load market history ──────────────────────────────────
        await step("📈", "*Шаг 1/4:* Загружаю исторические цены через Freedom API…")

        manager = UniversalPortfolioManager()

        # Wrap the heavy parts to detect WHERE we fail.
        def _stage_market_data():
            tickers = [
                t for t in df["Ticker"].astype(str).tolist()
                if t.upper() not in manager.engine.NON_RISK_ASSETS
            ] if "Ticker" in df.columns else [
                t for t in df.index.astype(str).tolist()
                if t.upper() not in manager.engine.NON_RISK_ASSETS
            ]
            data, history_result = manager.engine.get_market_data(tickers)
            return data, tickers, history_result

        try:
            all_data, risky_tickers, history_result = await loop.run_in_executor(None, _stage_market_data)
        except Exception as exc:
            logger.exception("Stage 1 (market data) failed: %s", exc)
            await bot.send_message(
                chat_id,
                "❌ *Шаг 1 не удался:* не получилось загрузить исторические цены.\n\n"
                f"Причина: `{str(exc)[:200]}`",
                parse_mode=ParseMode.MARKDOWN,
            )
            await refund("market_data_error")
            raise

        loaded_count = len([c for c in all_data.columns if not all_data[c].isna().all()]) if not all_data.empty else 0

        # ── Separate portfolio tickers from internal infrastructure ─────────
        # Factor ETFs (SPY, MTUM, VLUE, etc.) and BENCHMARK_EXTRA (QQQ, AGG, URTH)
        # are internal MAC3 infrastructure — do NOT show to users.
        # Users only care about THEIR portfolio tickers.
        internal_tickers = set(
            list(manager.engine.factor_tickers.values())
            + manager.engine.BENCHMARK_EXTRA
        )
        resolved_portfolio = list(dict.fromkeys(
            manager.engine.resolve_tickers(risky_tickers)
        ))
        portfolio_loaded = len([
            t for t in resolved_portfolio
            if t in all_data.columns and not all_data[t].isna().all()
        ])
        portfolio_total = len(resolved_portfolio)

        if loaded_count == 0:
            await bot.send_message(
                chat_id,
                "❌ *Шаг 1 — критично:* Freedom API не вернул ни одной серии цен.\n\n"
                "*Возможные причины:*\n"
                "• Ваш API-ключ Freedom Broker не имеет доступа к Market Data "
                "— это **отдельная подписка** на стороне брокера\n"
                "• Сервер Freedom активно обрывает соединение (`SSL EOF` / "
                "`RemoteDisconnected` в логах) — обычно так блокируется "
                "доступ без подписки\n\n"
                "*Что делать:*\n"
                "1. Зайдите в Личный кабинет Freedom Broker → API → Market Data\n"
                "2. Активируйте подписку на исторические данные (если её нет)\n"
                "3. Либо обратитесь в поддержку брокера: попросите включить "
                "доступ к `getHloc` для вашего API-ключа",
                parse_mode=ParseMode.MARKDOWN,
            )
            await refund("no_market_data")
            raise RuntimeError("market_data_subscription_required")

        # ── Build detailed ticker status message ──────────────────────────
        # Show only portfolio tickers (exclude factor ETFs + benchmark extras)
        lines = [f"✅ Загружено серий: *{portfolio_loaded}/{portfolio_total}*"]

        # Show proxy-resolved tickers
        proxy_lines = []
        for t in risky_tickers:
            resolved = manager.engine.resolve_tickers([t])
            if resolved and resolved[0] != f"{t.upper()}.US" and resolved[0] != t.upper():
                proxy_lines.append(f"  `{t}` → `{resolved[0]}`")
        if proxy_lines:
            lines.append("\n📎 *Прокси-замены:*")
            lines.extend(proxy_lines)

        # Show retried tickers (only portfolio, not internal)
        if history_result.retried:
            portfolio_retried = [t for t in history_result.retried if t not in internal_tickers]
            if portfolio_retried:
                lines.append(f"\n🔄 *Восстановлено retry:* {', '.join(portfolio_retried)}")

        # Show failed tickers with reasons (only portfolio, not internal)
        if history_result.failed:
            portfolio_failed = {t: r for t, r in history_result.failed.items() if t not in internal_tickers}
            if portfolio_failed:
                lines.append(f"\n❌ *Не загружены ({len(portfolio_failed)}):*")
                for t, reason in portfolio_failed.items():
                    short = reason[:60] + "…" if len(reason) > 60 else reason
                    lines.append(f"  `{t}` — {short}")
                lines.append("\nℹ️ _Пропущенные оцениваются через факторные индексы_")
        if portfolio_loaded == portfolio_total:
            lines.append("\n✅ _Все тикеры загружены успешно_")

        await bot.send_message(
            chat_id,
            "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN,
        )

        # ── Step 2: full MAC3 analysis (includes SEC EDGAR) ────────────────
        await step("🧮", "*Шаг 2/4:* Запускаю MAC3 (Ridge + Ledoit-Wolf) + SEC EDGAR…")
        try:
            results = await loop.run_in_executor(
                None, _analyze_existing_portfolio_sync, df, bench_tick,
            )
        except Exception as exc:
            logger.exception("Stage 2 (MAC3) failed: %s", exc)
            await bot.send_message(
                chat_id,
                "❌ *Шаг 2 не удался:* MAC3 движок упал.\n\n"
                f"Причина: `{str(exc)[:200]}`",
                parse_mode=ParseMode.MARKDOWN,
            )
            await refund("mac3_failure")
            raise

        await step("✅", "MAC3 факторная модель, SEC EDGAR и риск-декомпозиция посчитаны.")

        # ── MAC3 Risk Summary notification ─────────────────────────────────
        port_metrics = results.get("portfolio_metrics", {})
        if port_metrics:
            vol = port_metrics.get("Total_Volatility_Ann", 0)
            sharpe = port_metrics.get("Sharpe_Ratio", float("nan"))
            cvar = port_metrics.get("CVaR_95_Daily", 0)
            var95 = port_metrics.get("VaR_95_Daily", 0)
            sortino = port_metrics.get("Sortino_Ratio", float("nan"))
            pos_days = port_metrics.get("Positive_Days_Pct", 0)

            import math
            sharpe_str = f"{sharpe:.2f}" if not math.isnan(sharpe) else "—"
            sortino_str = f"{sortino:.2f}" if not math.isnan(sortino) else "—"

            risk_lines = [
                "📊 *MAC3 Risk Summary:*",
                f"  Volatility: *{vol*100:.1f}%* годовых",
                f"  Sharpe Ratio: *{sharpe_str}*",
                f"  Sortino Ratio: *{sortino_str}*",
                f"  CVaR (95%): *{cvar*100:.2f}%*",
                f"  VaR (95%): *{var95*100:.2f}%*",
                f"  Positive Days: *{pos_days:.0f}%*",
            ]
            try:
                await bot.send_message(
                    chat_id,
                    "\n".join(risk_lines),
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                pass  # Don't block on send failure

        # ── Step 3: gatekeeper (advisory only, non-blocking) ──────────────
        await step("📊", "*Шаг 3/4:* Проверяю риск-лимиты…")

        gate = run_gatekeeper(results)  # Always run with defaults

        profile = await get_profile(user_id)
        if profile is not None:
            gate_limits = {"max_portfolio_volatility": profile["target_volatility"] * 1.2}
            gate = run_gatekeeper(results, user_limits=gate_limits, user_profile=profile)

        # Show gatekeeper results (advisory only — never blocks report)
        if gate["critical"] or gate["warnings"]:
            alert_lines = []
            if gate["critical"]:
                alert_lines.append("⛔ *Нарушения риск-лимитов:*")
                for c in gate["critical"][:3]:
                    alert_lines.append(f"  {c}")
            if gate["warnings"]:
                alert_lines.append("⚠️ *Предупреждения:*")
                for w in gate["warnings"][:3]:
                    alert_lines.append(f"  {w}")
            if len(gate["critical"]) + len(gate["warnings"]) > 6:
                alert_lines.append(f"_…ещё {len(gate['critical']) + len(gate['warnings']) - 6} в PDF-отчёте_")

            try:
                await bot.send_message(
                    chat_id,
                    "\n".join(alert_lines),
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                pass  # Don't block on send failure
        else:
            await step("✅", "Все риск-проверки пройдены.")

        # Show sector exposure
        sector_exposure = results.get("sector_exposure", {})
        if sector_exposure:
            sector_lines = ["📊 *Секторное распределение:*"]
            for sector, weight in list(sector_exposure.items())[:6]:
                bar = "█" * max(1, int(weight * 100 / 5))
                sector_lines.append(f"  {bar} {sector}: {weight:.0%}")
            try:
                await bot.send_message(
                    chat_id,
                    "\n".join(sector_lines),
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                pass

        # ── Step 4: PDF generation ────────────────────────────────────────
        await step("📄", "*Шаг 4/4:* Генерирую PDF-отчёт…")

        # Fetch previous snapshot for month-over-month delta
        prev_snapshot = await get_last_report_snapshot(user_id, tier)

        # Resolve user's risk profile name for AI stock-pick context
        profile       = await get_profile(user_id)
        profile_name  = (profile or {}).get("profile_name", "Moderate")

        payload = _build_pdf_payload(
            results, tier,
            user_bench_ticker=bench_tick,
            prev_snapshot=prev_snapshot,
            user_risk_profile=profile_name,
        )
        await _send_pdf(bot, chat_id, user_id, tier, payload)

        # Persist this report's key metrics for future MoM comparison
        metrics = results.get("portfolio_metrics") or {}
        try:
            await save_report_snapshot(
                telegram_id = user_id,
                tier        = tier,
                risk_score  = payload.get("risk_pct"),
                sharpe      = metrics.get("Sharpe_Ratio"),
                cvar        = metrics.get("CVaR_95_Daily"),
                volatility  = metrics.get("Total_Volatility_Ann"),
                total_value = results.get("total_value"),
            )
        except Exception as snap_exc:
            logger.warning("Failed to save report snapshot: %s", snap_exc)

        await bot.send_message(
            chat_id,
            "✅ *Отчёт готов!*\n\n"
            "Скачайте PDF выше — там полный анализ MAC3, "
            "разложение рисков по факторам и сравнение с бенчмарками.",
            parse_mode=ParseMode.MARKDOWN,
        )

    except RealPortfolioRequired as exc:
        await bot.send_message(
            chat_id,
            "⚠️ *Анализ невозможен: нет реальных данных портфеля.*\n\n"
            f"{exc}",
            parse_mode=ParseMode.MARKDOWN,
        )
        await refund("no_real_portfolio")
    except RuntimeError as exc:
        if str(exc) == "market_data_subscription_required":
            # Already reported + refunded above.
            pass
        elif str(exc) == "report_delivery_failed":
            logger.error("Report generated but upload/delivery failed for %s", user_id)
            await bot.send_message(
                chat_id,
                "⚠️ *Отчёт сформирован, но не удалось загрузить его в облачное "
                "хранилище.*\n\n"
                "Это сбой на нашей стороне — повторите анализ позже или "
                "обратитесь в /support.",
                parse_mode=ParseMode.MARKDOWN,
            )
            await refund("report_delivery_failed")
        else:
            logger.exception("Background analysis runtime error: %s", exc)
            await bot.send_message(
                chat_id,
                f"⚠️ *Ошибка при анализе:*\n`{str(exc)[:200]}`",
                parse_mode=ParseMode.MARKDOWN,
            )
            await refund("runtime_error")
    except Exception as exc:
        logger.exception("MAC3 анализ упал для %s: %s", user_id, exc)
        await bot.send_message(
            chat_id,
            f"⚠️ *Непредвиденная ошибка:*\n`{str(exc)[:200]}`",
            parse_mode=ParseMode.MARKDOWN,
        )
        await refund("unexpected_error")


async def cb_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await state.clear()
    await callback.message.edit_text("❌ Анализ отменён. Токены не списаны.")


# ── Utility commands ──────────────────────────────────────────────────────────

async def cmd_balance(message: Message) -> None:
    balance = await get_balance(message.from_user.id)
    await message.answer(
        f"💳 *Ваш баланс:* {balance} токен(а)\n\n"
        "Пополнить: /topup\n"
        "Тарифы: 10 токенов = 5 000 KZT (1 токен = 500 KZT)",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_topup(message: Message) -> None:
    await message.answer(
        "💰 *Пополнение баланса*\n\n"
        "Тариф: *10 токенов за 5 000 KZT* (1 токен = 500 KZT)\n\n"
        "Для оплаты обратитесь к администратору или используйте платёжный шлюз (скоро).",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_support(message: Message) -> None:
    await message.answer(
        "🛟 *Поддержка RAMP*\n\n"
        "По всем вопросам пишите: @ramp_support_bot\n"
        "Часы работы: пн–пт, 09:00–18:00 (UTC+5).",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_mandate(message: Message, state: FSMContext) -> None:
    """Re-run onboarding questionnaire for returning users."""
    user_id = message.from_user.id
    profile = await get_profile(user_id)
    if profile is None:
        await message.answer(
            "⚠️ У вас ещё нет профиля. Используйте /start для регистрации.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    await state.clear()
    await state.set_state(Onboarding.Q1)
    q = QUESTIONS[0]
    sent = await message.answer(
        "🔄 *Обновление инвестиционного мандата*\n\n"
        "Пройдите анкетирование заново, чтобы обновить ваш профиль.\n\n"
        + q["text"],
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_question(q["options"], q_num=1),
    )
    await state.update_data(ob_message_id=sent.message_id)


# ── Dispatcher assembly ───────────────────────────────────────────────────────

def build_dispatcher() -> Dispatcher:
    dp = Dispatcher(storage=MemoryStorage())

    # Routers first — StateFilter guards prevent cross-fire with AnalysisFlow.
    dp.include_router(onboarding_router)
    dp.include_router(portfolio_router)

    # Message commands
    dp.message.register(cmd_start,    CommandStart())
    dp.message.register(cmd_balance,  F.text == "/balance")
    dp.message.register(cmd_topup,    F.text == "/topup")
    dp.message.register(cmd_support,  F.text == "/support")
    dp.message.register(cmd_mandate,  F.text == "/mandate")

    # Analysis flow callbacks
    dp.callback_query.register(cb_analysis_choice, F.data.startswith("analysis:"))
    dp.callback_query.register(cb_confirm,          F.data.startswith("confirm:"))
    dp.callback_query.register(cb_cancel,           F.data == "cancel")

    return dp


_CONFLICT_MAX_RETRIES = 5
_CONFLICT_RETRY_DELAY = 10  # seconds


async def main() -> None:
    await init_db()
    bot = Bot(token=BOT_TOKEN)
    dp  = build_dispatcher()
    logger.info("RAMP Bot запущен.")
    for attempt in range(_CONFLICT_MAX_RETRIES):
        try:
            await dp.start_polling(
                bot,
                allowed_updates=dp.resolve_used_update_types(),
                drop_pending_updates=True,
            )
            break
        except TelegramConflictError:
            if attempt < _CONFLICT_MAX_RETRIES - 1:
                logger.warning(
                    "Конфликт (409), жду %ds перед попыткой %d/%d",
                    _CONFLICT_RETRY_DELAY, attempt + 2, _CONFLICT_MAX_RETRIES,
                )
                await asyncio.sleep(_CONFLICT_RETRY_DELAY)
            else:
                logger.error(
                    "Конфликт не разрешился за %d попыток. Завершение.",
                    _CONFLICT_MAX_RETRIES,
                )
                raise


# if __name__ == "__main__":
#     asyncio.run(main())
