"""Визуальные блоки отчёта: инлайн-SVG и таблица факторных бет (Арх-5).

Роль : построить из `results{}` готовые к вставке в шаблон строки SVG и
       template-friendly таблицы. Чистая презентация — ни одного числа тут
       не РОЖДАЕТСЯ: серии считает `finance/portfolio_series`, примитивы
       рисует `pdf_charts`, а этот слой их только сшивает.
Слой : **L3 (Report)**. Смотрит вниз — в `pdf_charts` (L3) и `finance/*` (L2);
       наверх, в `tg_bot`, не смотрит НИКОГДА.
Док  : `docs/ARCHITECTURE_FOR_AGENTS.md` §3.1.

Почему модуль появился
----------------------
Эти функции жили в `tg_bot.py` (слой доставки, L4), а нужны были слою отчёта:
`html_renderer` импортировал их ОБРАТНО ВВЕРХ — четыре приватных имени
(`_sparkline_svg`, `_build_equity_curve_svg`, `_build_factor_radar_svg`,
`_build_factor_betas_table`). Это нарушало правило «импорты смотрят только
вниз» и делало рендер отчёта заложником модуля с Telegram-клавиатурами:
чтобы отрисовать спарклайн, тянулся весь `aiogram`.

`tg_bot` продолжает пользоваться ими под прежними именами (алиасы в конце
его импортов), поэтому ни один вызов и ни один тест не изменились.
"""

from __future__ import annotations

import logging
from typing import Optional

from pdf_charts import equity_curve_svg, factor_radar_svg

logger = logging.getLogger("ReportCharts")

__all__ = [
    "RADAR_FACTOR_AXES",
    "BENCH_FACTOR_BETAS",
    "sparkline_svg",
    "build_kpi_sparklines",
    "build_equity_curve_svg",
    "compute_factor_betas",
    "build_factor_radar_svg",
    "build_factor_betas_table",
]


# Все 10 осей, которые движок заявляет как моделируемые. Когда факторный ETF
# не загрузился, в perf-таблице нет колонки Beta_<factor> — ось всё равно
# показывается (серым пунктиром со значением 0), чтобы пользователь видел
# «загружено 7 из 10», а не «у нас всего 7 факторов».
RADAR_FACTOR_AXES = [
    "Market", "Momentum", "Value", "Quality", "Size",
    "Volatility",   # SPLV.US — low-vol factor (Step 5)
    "Commodities", "Rates", "EM_Equity", "EM_Bond",
]

# S&P 500 BASELINE when benchmark betas are unavailable (B1 2026-07-17).
# The PRIMARY source for the "Δ vs benchmark" column is now
# results["benchmark_factor_profile"] — the REAL factor betas of the user's
# mandate benchmark fitted by the engine (investment_logic.fit_factor_betas).
# This constant (Market β ≈ 1.0, rest ≈ 0 — exactly the S&P 500 profile) is
# kept as the explicit graceful fallback: no benchmark chosen / no history →
# the column and its label degrade to S&P 500 together, never mismatched.
BENCH_FACTOR_BETAS = {axis: (1.0 if axis == "Market" else 0.0)
                      for axis in RADAR_FACTOR_AXES}


def build_equity_curve_svg(results: dict) -> str:
    """
    Inline equity-curve SVG: portfolio vs benchmark daily log returns.

    SoC (Sprint-1 #3): the cap-weighted return math + benchmark selection now
    live in `finance.portfolio_series.compute_equity_curve_series`.  This
    helper is pure UI — it asks the finance layer for the numeric series and
    renders them (or a labelled empty-state SVG when the data is missing).
    """
    try:
        from finance.portfolio_series import compute_equity_curve_series
        port_log, bm_log, _bm_name = compute_equity_curve_series(results)
        if port_log is None or len(port_log) == 0:
            return equity_curve_svg([])  # empty-state w/ "нет данных"
        return equity_curve_svg(port_log, bm_log)
    except Exception as exc:
        logger.warning("equity_curve_svg build failed: %s", exc)
        return equity_curve_svg([])


# ── KPI sparklines (DEEP/BASE templates · cover KPI strip) ──────────────────

def sparkline_svg(values: list[float], color: str = "#9A7A10",
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


def build_kpi_sparklines(results: dict) -> Optional[dict]:
    """
    Render the CVaR / Sharpe / MaxDD trend series as inline SVG sparklines.

    SoC (Sprint-1 #3): the rolling-window MATH now lives in
    `finance.portfolio_series.compute_kpi_trend_series`; this helper is pure
    UI — it renders the returned numeric series as SVGs and passes the scaled
    raw points through for the Premium V2 KPI cards (which redraw with their own
    <Sparkline>).  Returns None when there is not enough history so the
    template's `{% if data.kpi_sparklines %}` guard hides the cells cleanly.
    """
    try:
        from finance.portfolio_series import compute_kpi_trend_series
        series = compute_kpi_trend_series(results)
        if not series:
            return None
        cvar_pts   = series["cvar_pts"]
        sharpe_pts = series["sharpe_pts"]
        mdd_pts    = series["mdd_pts"]
        return {
            "cvar_svg":   sparkline_svg(cvar_pts,   color="#3F8F5F", invert=True),
            "sharpe_svg": sparkline_svg(sharpe_pts, color="#9A7A10", invert=False),
            "mdd_svg":    sparkline_svg(mdd_pts,    color="#C0492F", invert=True),
            # Raw series too — the Premium V2 KPI cards redraw these with the
            # design's own <Sparkline>.  Scaled ×100 so the React component plots
            # them as percent-points (CVaR/MaxDD) and a ratio (Sharpe).
            "cvar_pts":   [round(x * 100, 2) for x in cvar_pts],
            "sharpe_pts": [round(x, 3) for x in sharpe_pts],
            "mdd_pts":    [round(x * 100, 2) for x in mdd_pts],
        }
    except Exception as exc:
        logger.warning("kpi_sparklines build failed: %s", exc)
        return None


def build_factor_radar_svg(results: dict) -> str:
    """Inline factor-radar SVG with all axes (missing factors flagged)."""
    betas, missing, _coverage = compute_factor_betas(results)
    try:
        return factor_radar_svg(betas, missing_axes=missing)
    except Exception as exc:
        logger.warning("factor_radar_svg build failed: %s", exc)
        return factor_radar_svg({})


def compute_factor_betas(results: dict) -> tuple[dict[str, float], list[str], float]:
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
            return {}, list(RADAR_FACTOR_AXES), 0.0
        total_val = float(results.get("total_value") or 1.0)
        weights   = (perf["Current_Value"].fillna(0).astype(float) / total_val).values

        betas:   dict[str, float] = {}
        missing: list[str]        = []
        coverage_w = 0.0
        for axis in RADAR_FACTOR_AXES:
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
                         len(RADAR_FACTOR_AXES) - len(missing),
                         len(RADAR_FACTOR_AXES), ", ".join(missing))
        return betas, missing, coverage_pct
    except Exception as exc:
        logger.warning("factor beta extraction failed: %s", exc)
        return {}, list(RADAR_FACTOR_AXES), 0.0


def build_factor_betas_table(results: dict) -> list[dict]:
    """
    Template-friendly factor-β table for the DEEP factor-radar section.

    Each row: {axis, beta, bench, delta, missing}.  Bench is the REAL factor-β
    of the user's mandate benchmark (results["benchmark_factor_profile"],
    B1 2026-07-17) with a graceful fallback to the S&P 500 constant
    (Market = 1.0, rest = 0.0) when the profile is absent; delta =
    β_port − β_bench rounded to 2 dp.  Missing axes get beta=0.0 +
    missing=True so the template can render them as muted "n/a" rows
    instead of dropping them silently.
    """
    betas, missing, _coverage = compute_factor_betas(results)
    if not betas:
        return []
    bench_profile = results.get("benchmark_factor_profile") or {}
    bench_betas   = (bench_profile.get("betas")
                     if isinstance(bench_profile, dict) else None) or None
    missing_set = set(missing)
    rows: list[dict] = []
    for axis in RADAR_FACTOR_AXES:
        b = float(betas.get(axis, 0.0))
        if bench_betas:
            bench = float(bench_betas.get(axis, 0.0))
        else:
            bench = float(BENCH_FACTOR_BETAS.get(axis, 0.0))
        rows.append({
            "axis":    axis,
            "beta":    round(b, 2),
            "bench":   round(bench, 2),
            "delta":   round(b - bench, 2),
            "missing": axis in missing_set,
        })
    return rows
