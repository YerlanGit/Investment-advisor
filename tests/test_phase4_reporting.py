"""
Phase 4 regression tests — PDF payload, SVG charts, AI narrative fallback,
and Jinja rendering of both v2 templates.

The tests do not invoke Playwright/Chromium — they only verify that the
templates render to valid HTML against a synthetic payload.  Browser
rendering is exercised manually via `python -m src.pdf_generator`.
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── 4.1  pdf_payload.build_payload ───────────────────────────────────────────

class PayloadBuildTest(unittest.TestCase):

    def _results(self) -> dict:
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 2000.0, "Total_Cost": 1500.0,
             "PnL": 500.0, "Return_Pct": 0.3333,
             "Euler_Risk_Contribution_Pct": 12.0, "ATR_Pct": 1.4,
             "Marginal_VaR_Daily": -0.0006, "Score_Total": 3.0,
             "Score_Action": "Strong Buy", "Score_Hotspot": False,
             "Fundamental_Sector": "Technology",
             "SEC_ROE": 0.30, "SEC_Op_Margin": 0.28, "SEC_Debt_to_Assets": 0.32,
             "SEC_Revenue_Growth_YoY": 0.06, "SEC_FCF_Margin": 0.25,
             "SEC_Altman_Z": 6.5, "SEC_Altman_Zone": "Safe",
             "SEC_Piotroski_F": 8, "SEC_Interest_Coverage": 12.0,
             "Beta_Market": 1.05, "Beta_Momentum": 0.20, "Beta_Quality": 0.5},
            {"Ticker": "KSPI", "Current_Value": 1000.0, "Total_Cost": 1200.0,
             "PnL": -200.0, "Return_Pct": -0.1667,
             "Euler_Risk_Contribution_Pct": 22.0, "ATR_Pct": 2.1,
             "Marginal_VaR_Daily": -0.0010, "Score_Total": -1.0,
             "Score_Action": "Trim", "Score_Hotspot": True,
             "Fundamental_Sector": "EM_Kazakhstan"},
        ])
        return {
            "performance_table": perf,
            "total_value": 3000.0,
            "portfolio_metrics": {
                "CVaR_95_Daily": -0.052, "Sharpe_Ratio": 1.18,
                "Sortino_Ratio": 1.6, "VaR_95_Daily": -0.024,
                "Max_Drawdown": -0.128, "Total_Volatility_Ann": 0.142,
                "Composite_Risk_Score": 62, "Max_Euler_Risk_Pct": 22.0,
                "CVaR_95_Bootstrap": {"point": -0.052, "lo95": -0.071, "hi95": -0.046},
            },
            "benchmark_comparison": {
                "Профильный бенчмарк": {
                    "Excess_Return_Ann": 0.032, "Tracking_Error": 0.043,
                    "Information_Ratio": 0.74, "Beating_Benchmark": True,
                },
                "S&P 500": {
                    "Excess_Return_Ann": 0.018, "Tracking_Error": 0.051,
                    "Information_Ratio": 0.35, "Beating_Benchmark": True,
                },
            },
            "sector_exposure": {"Technology": 0.34, "EM_Kazakhstan": 0.22},
            "regime": {"regime": "Slowdown", "confidence": 0.68,
                       "growth_score": -0.012, "cycle_score": -0.018, "signals": {}},
            "asset_scores": {
                "AAPL": {"fundamentals": 1.5, "valuations": 0.5, "technicals": 1.0,
                         "credit": 1.0, "total": 3.0, "action": "Strong Buy",
                         "hotspot": False},
                "KSPI": {"fundamentals": 0.0, "valuations": 1.0, "technicals": -0.5,
                         "credit": -0.5, "total": 0.0, "action": "Hold",
                         "hotspot": True},
            },
            "action_plan": [
                {"ticker": "KSPI", "action": "Trim", "delta_w_pp": -8.0,
                 "qty_delta": -32, "buy_zone": None, "sell_zone": [145.0, 152.0],
                 "take_target": None, "stop_loss": 132.0, "reason": "🔥 Hotspot"},
                {"ticker": "AAPL", "action": "Buy", "delta_w_pp": 3.5,
                 "qty_delta": 12, "buy_zone": [195.0, 202.0], "sell_zone": None,
                 "take_target": 220.0, "stop_loss": 178.0, "reason": "Score +3"},
            ],
            "black_litterman": [
                {"ticker": "AAPL", "current_w": 0.30, "target_w": 0.35,
                 "delta_w_pp": 5.0, "posterior_mu": 0.10},
                {"ticker": "KSPI", "current_w": 0.40, "target_w": 0.32,
                 "delta_w_pp": -8.0, "posterior_mu": -0.02},
            ],
        }

    def test_base_payload_keys(self) -> None:
        from pdf_payload import build_payload
        p = build_payload(self._results(), "base",
                          ai_summary={"verdict": "v", "bullets": ["a", "b", "c"]})
        # Required v2 keys
        for k in ("cvar", "sharpe", "var_95_daily", "max_drawdown", "risk_pct",
                  "pnl_total_pct", "pnl_total_abs", "assets", "hotspots",
                  "sectors", "regime", "ai_verdict", "ai_bullets"):
            self.assertIn(k, p)
        self.assertEqual(p["tier"], "base")
        # Hotspot present in hotspots list
        self.assertTrue(any("KSPI" in h for h in p["hotspots"]))
        # var_95 ≠ max_drawdown numerically
        self.assertNotEqual(p["var_95_daily"], p["max_drawdown"])

    def test_deep_payload_has_extra_blocks(self) -> None:
        from pdf_payload import build_payload
        p = build_payload(self._results(), "deep",
                          ai_summary={"verdict": "v", "bullets": ["a"], "action_plan_text": "do it"})
        for k in ("scenarios", "score_breakdown", "action_plan",
                  "fundamental_layer", "bl_records", "ai_action_text"):
            self.assertIn(k, p)
        self.assertEqual(p["tier"], "deep")
        # Scenarios use Excess_Return_Ann
        self.assertTrue(any("3.2%" in s["excess"] for s in p["scenarios"]))


# ── 4.2  SVG charts ─────────────────────────────────────────────────────────

class ChartsSVGTest(unittest.TestCase):

    def test_equity_curve_renders(self) -> None:
        from pdf_charts import equity_curve_svg
        rng = np.random.default_rng(0)
        port = rng.normal(0.0008, 0.011, 200)
        bm   = rng.normal(0.0006, 0.011, 200)
        svg = equity_curve_svg(port, bm)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("Портфель", svg)
        self.assertIn("Бенчмарк", svg)

    def test_equity_curve_handles_empty(self) -> None:
        from pdf_charts import equity_curve_svg
        svg = equity_curve_svg([])
        self.assertIn("нет данных", svg)

    def test_sector_pie_renders(self) -> None:
        from pdf_charts import sector_pie_svg
        svg = sector_pie_svg({"Tech": 0.40, "Finance": 0.30, "Cash": 0.30})
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("Tech", svg)
        # Donut center
        self.assertIn("circle", svg)

    def test_factor_radar_renders(self) -> None:
        from pdf_charts import factor_radar_svg
        svg = factor_radar_svg({"Market": 0.85, "Momentum": 0.31, "Value": 0.05,
                                  "Quality": 0.20, "Size": -0.10,
                                  "Rates": -0.18, "EM_Equity": 0.42})
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("Market", svg)
        # Polygon for the data
        self.assertIn("polygon", svg)


# ── 4.3  AI narrative fallback ──────────────────────────────────────────────

class NarrativeFallbackTest(unittest.TestCase):

    def test_fallback_when_no_api_key(self) -> None:
        # Force fallback path by clearing API key.
        prev = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            from ai_narrative import generate_narrative
            results = {
                "portfolio_metrics": {
                    "CVaR_95_Daily": -0.05, "Sharpe_Ratio": 1.0,
                    "Composite_Risk_Score": 55, "Max_Drawdown": -0.1,
                },
                "regime": {"regime": "Expansion", "confidence": 0.6},
                "performance_table": pd.DataFrame([
                    {"Ticker": "AAPL", "Score_Hotspot": False,
                     "Euler_Risk_Contribution_Pct": 10},
                    {"Ticker": "KSPI", "Score_Hotspot": True,
                     "Euler_Risk_Contribution_Pct": 22},
                ]),
            }
            out = generate_narrative(results, tier="base")
            self.assertIn("verdict", out)
            self.assertIn("bullets", out)
            self.assertGreaterEqual(len(out["bullets"]), 1)
            # Hotspot bullet must mention KSPI
            self.assertTrue(any("KSPI" in b for b in out["bullets"]))
        finally:
            if prev is not None:
                os.environ["ANTHROPIC_API_KEY"] = prev


# ── 4.4  Jinja render of both templates ─────────────────────────────────────

class JinjaRenderTest(unittest.TestCase):

    def _render(self, template_name: str, payload: dict) -> str:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        tdir = SRC / "templates"
        env  = Environment(loader=FileSystemLoader(str(tdir)),
                           autoescape=select_autoescape(["html"]))
        return env.get_template(template_name).render(
            data        = payload,
            user_id     = "tester",
            report_type = "Тест",
            generated_at= datetime.now().strftime("%d.%m.%Y %H:%M"),
        )

    def test_basic_template_renders(self) -> None:
        from pdf_payload import build_payload
        from tests.test_phase4_reporting import PayloadBuildTest
        results = PayloadBuildTest()._results()
        payload = build_payload(results, "base",
                                ai_summary={"verdict": "ok", "bullets": ["a", "b"]})
        html = self._render("report_basic.html", payload)
        self.assertIn("PORTFOLIO INTELLIGENCE", html)
        self.assertNotIn("RAMP",                 html)
        self.assertIn("Базовый отчёт".lower() if False else "Тест", html)
        self.assertIn("AAPL",                    html)
        # No leftover dark-theme tokens
        self.assertNotIn("#0D1117", html)
        self.assertIn("#0F4C81",    html)

    def test_deep_template_renders(self) -> None:
        from pdf_payload import build_payload
        from tests.test_phase4_reporting import PayloadBuildTest
        results = PayloadBuildTest()._results()
        payload = build_payload(results, "deep",
                                ai_summary={"verdict": "ok", "bullets": ["a"],
                                              "action_plan_text": "do it now"})
        html = self._render("report_deep.html", payload)
        self.assertIn("PORTFOLIO INTELLIGENCE", html)
        self.assertNotIn("RAMP",                 html)
        # Deep-only sections present
        self.assertIn("Action Plan",             html)
        self.assertIn("Black-Litterman",         html)
        self.assertIn("Fundamental Layer",       html)
        self.assertIn("CoVe",                    html)


if __name__ == "__main__":
    unittest.main()
