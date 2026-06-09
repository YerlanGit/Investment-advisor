"""
Inline SVG generators for the PDF report.

We render charts as raw SVG strings rather than rasterising via matplotlib —
SVG embeds cleanly inside Playwright/Chromium, scales without artefacts on
A4 print, and adds zero binary dependencies (PIL, cairo, kaleido).

Three charts are exposed today:

  • equity_curve_svg(port, bm)      — line chart, portfolio vs benchmark
  • sector_pie_svg(sectors)          — donut chart of sector exposure
  • factor_radar_svg(beta_dict)      — radar (spider) chart of factor betas

The light theme uses navy as the primary stroke and a muted gold accent —
designed to read well on white A4 paper.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


# ── Light theme palette (kept in sync with templates/report_basic.html) ──────
NAVY      = "#0F4C81"
NAVY_HI   = "#1E6BB8"
GOLD      = "#B7873B"
POS_COLOR = "#047857"
NEG_COLOR = "#B91C1C"
NEUT      = "#94A3B8"
GRID      = "#E2E8F0"
TEXT_1    = "#0F172A"
TEXT_2    = "#475569"
PIE_COLORS = [
    "#0F4C81", "#1E6BB8", "#B7873B", "#047857", "#B91C1C",
    "#7C3AED", "#0891B2", "#A16207", "#65A30D", "#94A3B8",
]


# ── Equity curve ─────────────────────────────────────────────────────────────

def equity_curve_svg(port_daily_log_returns: Iterable[float],
                     bm_daily_log_returns: Optional[Iterable[float]] = None,
                     *,
                     width: int = 720, height: int = 200,
                     padding: int = 28) -> str:
    """
    Two-line equity curve from daily log-return streams.  Both series are
    aligned to start at 100 (end-of-day before observation 0).
    """
    p_arr = np.asarray(list(port_daily_log_returns), dtype=float)
    if p_arr.size == 0:
        return _empty_chart(width, height, "нет данных")

    # M-4: one NaN/Inf log-return would make cumsum→exp all-NaN and every SVG
    # coordinate "nan" (an invisible chart).  Replace non-finite values with 0
    # (a flat step) so the curve always renders.
    p_arr = np.where(np.isfinite(p_arr), p_arr, 0.0)

    p_eq = 100.0 * np.exp(np.cumsum(p_arr))
    p_eq = np.concatenate([[100.0], p_eq])

    if bm_daily_log_returns is not None:
        b_arr = np.asarray(list(bm_daily_log_returns), dtype=float)
        b_arr = np.where(np.isfinite(b_arr), b_arr, 0.0)   # M-4: same guard
        if b_arr.size:
            b_eq = 100.0 * np.exp(np.cumsum(b_arr))
            b_eq = np.concatenate([[100.0], b_eq])
            min_len = min(len(p_eq), len(b_eq))
            p_eq, b_eq = p_eq[-min_len:], b_eq[-min_len:]
        else:
            b_eq = None
    else:
        b_eq = None

    y_min = float(min(p_eq.min(), b_eq.min() if b_eq is not None else p_eq.min()))
    y_max = float(max(p_eq.max(), b_eq.max() if b_eq is not None else p_eq.max()))
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0

    def _scale_x(i: int, n: int) -> float:
        return padding + i * (width - 2 * padding) / max(1, n - 1)

    def _scale_y(v: float) -> float:
        return height - padding - (v - y_min) / (y_max - y_min) * (height - 2 * padding)

    def _path(arr: np.ndarray) -> str:
        if arr.size == 0:
            return ""
        parts = [f"M {_scale_x(0, len(arr)):.1f} {_scale_y(arr[0]):.1f}"]
        for i in range(1, len(arr)):
            parts.append(f"L {_scale_x(i, len(arr)):.1f} {_scale_y(arr[i]):.1f}")
        return " ".join(parts)

    bm_path = _path(b_eq) if b_eq is not None else None
    pt_path = _path(p_eq)

    # Y-axis grid lines (3 lines: min, mid, max).
    grid_lines = ""
    for v in (y_min, (y_min + y_max) / 2.0, y_max):
        y = _scale_y(v)
        grid_lines += (
            f'<line x1="{padding}" y1="{y:.1f}" x2="{width-padding}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="0.5" />'
            f'<text x="{padding-4}" y="{y+3:.1f}" text-anchor="end" '
            f'font-size="9" fill="{TEXT_2}" font-family="Inter,Arial">{v:.0f}</text>'
        )

    legend = (
        f'<g font-family="Inter,Arial" font-size="9" fill="{TEXT_2}">'
        f'<line x1="{width-180}" y1="14" x2="{width-160}" y2="14" stroke="{NAVY}" stroke-width="2" />'
        f'<text x="{width-156}" y="17">Портфель</text>'
    )
    if bm_path:
        legend += (
            f'<line x1="{width-95}" y1="14" x2="{width-75}" y2="14" stroke="{GOLD}" '
            f'stroke-width="2" stroke-dasharray="3 2" />'
            f'<text x="{width-71}" y="17">Бенчмарк</text>'
        )
    legend += "</g>"

    bm_line = ""
    if bm_path:
        bm_line = (f'<path d="{bm_path}" fill="none" stroke="{GOLD}" '
                   f'stroke-width="1.6" stroke-dasharray="3 2" />')

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">'
        f'{grid_lines}'
        f'{bm_line}'
        f'<path d="{pt_path}" fill="none" stroke="{NAVY}" stroke-width="1.8" />'
        f'{legend}'
        f'</svg>'
    )


# ── Sector donut ─────────────────────────────────────────────────────────────

def sector_pie_svg(sectors: dict[str, float],
                   *, size: int = 220, inner: int = 64) -> str:
    """
    Donut chart from {sector_name: weight (0..1 or 0..100)}.

    If values are clearly percentages (>1.0), we normalise.
    """
    if not sectors:
        return _empty_chart(size, size, "нет данных")
    items = list(sectors.items())
    total = sum(float(v) for _, v in items)
    if total <= 0:
        return _empty_chart(size, size, "нет данных")
    weights = [(name, float(v) / total) for name, v in items]

    cx, cy, r = size / 2, size / 2, size / 2 - 6
    parts = []
    angle0 = -math.pi / 2.0
    for i, (name, w) in enumerate(weights):
        angle1 = angle0 + 2 * math.pi * w
        x0 = cx + r * math.cos(angle0); y0 = cy + r * math.sin(angle0)
        x1 = cx + r * math.cos(angle1); y1 = cy + r * math.sin(angle1)
        large_arc = 1 if (angle1 - angle0) > math.pi else 0
        color = PIE_COLORS[i % len(PIE_COLORS)]
        parts.append(
            f'<path d="M {cx} {cy} L {x0:.1f} {y0:.1f} A {r} {r} 0 {large_arc} 1 {x1:.1f} {y1:.1f} Z" '
            f'fill="{color}" />'
        )
        angle0 = angle1
    # Inner cutout for donut
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{inner}" fill="white" />')

    # Legend on the right
    leg_x = size + 8
    leg_y = 10
    legend_parts = []
    for i, (name, w) in enumerate(weights):
        color = PIE_COLORS[i % len(PIE_COLORS)]
        legend_parts.append(
            f'<rect x="{leg_x}" y="{leg_y + i*16}" width="10" height="10" fill="{color}" />'
            f'<text x="{leg_x+14}" y="{leg_y + i*16 + 9}" '
            f'font-family="Inter,Arial" font-size="10" fill="{TEXT_1}">'
            f'{_xml_escape(name)} · {w*100:.0f}%</text>'
        )
    full_w = size + 160
    # M-3: grow the canvas vertically so the legend never clips past the
    # viewBox.  Each legend row is 16px tall starting at leg_y; with a fixed
    # `size` height anything beyond ~13 sectors used to be cut off.
    full_h = max(size, leg_y + len(weights) * 16 + 4)
    return (
        f'<svg width="{full_w}" height="{full_h}" viewBox="0 0 {full_w} {full_h}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'{"".join(parts)}{"".join(legend_parts)}'
        f'</svg>'
    )


# ── Factor radar ─────────────────────────────────────────────────────────────

def factor_radar_svg(betas: dict[str, float],
                     *, size: int = 240,
                     missing_axes: list[str] | None = None) -> str:
    """
    Radar chart of factor betas.  Each axis is one factor; values are clipped
    to [-1.5, 1.5] for a balanced visual.

    Args:
        betas        : Dict of {axis_name: beta}.  Pass 0.0 for missing factors
                       to keep the axis on the chart (better for the user
                       than a polygon that silently drops sides).
        missing_axes : Optional list of axis names whose data was unavailable
                       — these axes are drawn with a gray dashed marker so
                       the user can see N/N axes loaded.
    """
    if not betas:
        return _empty_chart(size, size, "нет данных")
    labels = list(betas.keys())
    values = [float(v) for v in betas.values()]
    n = len(labels)
    if n < 3:
        return _empty_chart(size, size, "недостаточно факторов")
    missing_set = set(missing_axes or [])
    cx, cy = size / 2, size / 2
    r_max = size / 2 - 28
    grid_levels = (-1.5, -0.75, 0.0, 0.75, 1.5)

    def _pt(idx: int, val: float) -> tuple[float, float]:
        clipped = max(-1.5, min(1.5, val))
        # Map [-1.5, 1.5] → [0, r_max], with 0 at the centre.
        radius = (clipped + 1.5) / 3.0 * r_max
        angle = -math.pi / 2.0 + 2 * math.pi * idx / n
        return cx + radius * math.cos(angle), cy + radius * math.sin(angle)

    # Concentric grid
    parts = []
    for lvl in grid_levels:
        radius = (lvl + 1.5) / 3.0 * r_max
        ring = []
        for i in range(n):
            angle = -math.pi / 2.0 + 2 * math.pi * i / n
            ring.append(f"{cx + radius*math.cos(angle):.1f},{cy + radius*math.sin(angle):.1f}")
        parts.append(f'<polygon points="{" ".join(ring)}" fill="none" stroke="{GRID}" stroke-width="0.5" />')

    # Axis spokes + labels (missing axes get a dashed gray spoke + ⚠ marker)
    for i, label in enumerate(labels):
        is_missing = label in missing_set
        ax_x = cx + r_max * math.cos(-math.pi / 2.0 + 2 * math.pi * i / n)
        ax_y = cy + r_max * math.sin(-math.pi / 2.0 + 2 * math.pi * i / n)
        spoke_dash = ' stroke-dasharray="2 2"' if is_missing else ""
        spoke_color = NEUT if is_missing else GRID
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{ax_x:.1f}" y2="{ax_y:.1f}" '
            f'stroke="{spoke_color}" stroke-width="0.6"{spoke_dash} />'
        )
        # Label outside the ring
        lab_r = r_max + 14
        lab_x = cx + lab_r * math.cos(-math.pi / 2.0 + 2 * math.pi * i / n)
        lab_y = cy + lab_r * math.sin(-math.pi / 2.0 + 2 * math.pi * i / n)
        prefix = "⚠ " if is_missing else ""
        text_color = NEUT if is_missing else TEXT_2
        parts.append(
            f'<text x="{lab_x:.1f}" y="{lab_y+3:.1f}" text-anchor="middle" '
            f'font-family="Inter,Arial" font-size="9" fill="{text_color}">'
            f'{_xml_escape(prefix + label)}</text>'
        )

    # Data polygon — only draw if at least 3 axes have non-zero, real values.
    real_axes = [i for i, lbl in enumerate(labels) if lbl not in missing_set]
    if len(real_axes) >= 3:
        pts = [f"{x:.1f},{y:.1f}" for x, y in (_pt(i, v) for i, v in enumerate(values))]
        parts.append(
            f'<polygon points="{" ".join(pts)}" fill="{NAVY_HI}" fill-opacity="0.20" '
            f'stroke="{NAVY}" stroke-width="1.5" />'
        )

    # Footnote when any axis is missing
    if missing_set:
        parts.append(
            f'<text x="{cx}" y="{size - 6}" text-anchor="middle" '
            f'font-family="Inter,Arial" font-size="8" fill="{NEUT}">'
            f'⚠ {len(missing_set)}/{n} осей: данные ETF недоступны</text>'
        )

    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_chart(w: int, h: int, msg: str) -> str:
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="100%" height="100%" fill="{GRID}" fill-opacity="0.3" />'
        f'<text x="{w/2}" y="{h/2}" text-anchor="middle" font-family="Inter,Arial" '
        f'font-size="11" fill="{TEXT_2}">{_xml_escape(msg)}</text>'
        f'</svg>'
    )


def _xml_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                  .replace(">", "&gt;").replace('"', "&quot;"))


__all__ = ["equity_curve_svg", "sector_pie_svg", "factor_radar_svg"]
