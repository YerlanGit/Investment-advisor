"""
Premium V2 report renderer — assembles a self-contained DEEP/BASE premium report
HTML from the pre-built static assets (compiled React + compiled JSX components +
compiled Tailwind CSS) plus an injected data object.

Design source: design/premium_v2/ (React + Tailwind, see docs/report/PREMIUM_DESIGN.md).
The component bundles are DATA-FREE; the engine's data object is injected at
render time as `window.DEEP` / `window.PORTFOLIO`, so one static build serves
every portfolio.  No CDN, no runtime Babel — only Google Fonts is external
(degrades gracefully to system fonts).

Usage:
    from premium_renderer import render_premium
    html = render_premium("deep", deep_data_dict)   # deep_data_dict matches the
                                                     # design contract (see
                                                     # design/premium_v2/*.sample.json)
"""
from __future__ import annotations

import json
import os

# BLOCK 1 fix: runtime assets live in src/premium_assets/ so they ship inside
# the deployed container (the Dockerfile COPYs src/ but NOT design/).  The old
# path pointed at design/premium_v2/, which is absent in the Cloud Run image →
# render_premium raised FileNotFoundError and the report silently fell back to
# v3 even with PREMIUM_REPORT_ENABLED=true.  Falls back to the design/ source
# dir for local dev where src/premium_assets/ may be stale.
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DIR      = os.path.join(_HERE, "premium_assets")                       # deployed
_DEV_DIR  = os.path.join(_HERE, "..", "design", "premium_v2")           # dev source


def _read(name: str) -> str:
    primary = os.path.join(_DIR, name)
    path = primary if os.path.exists(primary) else os.path.join(_DEV_DIR, name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_sample(tier: str) -> dict:
    """The design's reference data — used for demos / as a render fallback."""
    fname = "deep-data.sample.json" if str(tier).lower() == "deep" else "base-data.sample.json"
    return json.loads(_read(fname))


def render_premium(tier: str, data: dict | None = None) -> str:
    """
    Build the premium report HTML for `tier` ('deep' | 'base') with `data`
    (the design-contract object).  Falls back to the design sample when data
    is None, so the renderer always produces a valid report.
    """
    is_deep = str(tier).lower() == "deep"
    var     = "DEEP" if is_deep else "PORTFOLIO"
    comp    = "deep-components.js" if is_deep else "base-components.js"

    if data is None:
        data = load_sample("deep" if is_deep else "base")

    react      = _read("react.production.min.js")
    react_dom  = _read("react-dom.production.min.js")
    components = _read(comp)
    css_tw     = _read("report.compiled.css")
    css_custom = _read("custom.css")
    data_js    = f"window.{var} = {json.dumps(data, ensure_ascii=False)};"

    return (
        '<!DOCTYPE html><html lang="ru"><head>'
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{var} Report — Premium</title>'
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700'
        '&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
        f"<style>{css_custom}</style><style>{css_tw}</style>"
        '</head><body>'
        '<div id="root" class="preload"></div>'
        f"<script>{react}</script><script>{react_dom}</script>"
        f"<script>{data_js}</script>"
        f"<script>{components}</script>"
        "</body></html>"
    )


__all__ = ["render_premium", "load_sample"]
