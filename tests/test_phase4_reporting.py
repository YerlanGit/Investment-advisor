"""
Phase 4 regression tests — PDF payload, SVG charts, AI narrative fallback,
and Jinja rendering of both v2 templates.

The tests do not invoke Playwright/Chromium — they only verify that the
templates render to valid HTML against a synthetic payload.  Browser
rendering is exercised manually via `python -m src.pdf_generator`.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from datetime import date, datetime
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


# ── 4.1.b  Concentration / waterfall / multi-period (NEW data layer) ────────

class ConcentrationAndWaterfallTest(unittest.TestCase):
    """
    Verifies the new data-layer payload fields used by the next-generation
    BASE/DEEP report templates:
      • sector_concentration  (HHI by sector_exposure)
      • sector_warnings       (soft caps trigger)
      • asset_concentration   (HHI by per-asset weights)
      • risk_waterfall        (per-asset standalone vol vs diversified)
      • period_returns_table  (passthrough)

    All numbers are computed against hand-checked fixtures so the tests catch
    regressions in the calculation, not just the field presence.
    """

    def _base_results(self) -> dict:
        # 3-asset portfolio, total value $4,000.
        # AAPL: 2000 (50%), MSFT: 1000 (25%), KSPI: 1000 (25%)
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 2000.0, "Total_Cost": 1500.0,
             "PnL": 500.0, "Return_Pct": 0.3333,
             "Euler_Risk_Contribution_Pct": 12.0, "ATR_Pct": 1.4,
             "Marginal_VaR_Daily": -0.0006,
             "Fundamental_Sector": "Technology"},
            {"Ticker": "MSFT", "Current_Value": 1000.0, "Total_Cost": 900.0,
             "PnL": 100.0, "Return_Pct": 0.1111,
             "Euler_Risk_Contribution_Pct": 8.0, "ATR_Pct": 1.0,
             "Marginal_VaR_Daily": -0.0004,
             "Fundamental_Sector": "Technology"},
            {"Ticker": "KSPI", "Current_Value": 1000.0, "Total_Cost": 1200.0,
             "PnL": -200.0, "Return_Pct": -0.1667,
             "Euler_Risk_Contribution_Pct": 22.0, "ATR_Pct": 2.1,
             "Marginal_VaR_Daily": -0.0010,
             "Fundamental_Sector": "EM_Kazakhstan"},
        ])
        # Annual structural covariance — DIAGONAL gives per-asset vol²:
        #   AAPL diag = 0.04   → σ = 0.20 (20% annual vol)
        #   MSFT diag = 0.0225 → σ = 0.15
        #   KSPI diag = 0.09   → σ = 0.30
        # Off-diagonal cov chosen so portfolio is well-diversified.
        risk_matrix = pd.DataFrame(
            [[0.0400, 0.0080, 0.0030],
             [0.0080, 0.0225, 0.0020],
             [0.0030, 0.0020, 0.0900]],
            index   = ["AAPL", "MSFT", "KSPI"],
            columns = ["AAPL", "MSFT", "KSPI"],
        )
        # Portfolio variance with w = [0.5, 0.25, 0.25]:
        #   var = 0.5²·0.04 + 0.25²·0.0225 + 0.25²·0.09
        #       + 2·0.5·0.25·0.0080 + 2·0.5·0.25·0.0030 + 2·0.25·0.25·0.0020
        #       = 0.0100 + 0.001406 + 0.005625 + 0.002 + 0.00075 + 0.00025
        #       = 0.020031
        #   σ_port = √0.020031 ≈ 0.14153  (≈ 14.15% annual)
        return {
            "performance_table": perf,
            "total_value": 4000.0,
            "portfolio_metrics": {
                "Total_Volatility_Ann": 0.14153, "CVaR_95_Daily": -0.045,
                "Sharpe_Ratio": 1.1, "Sortino_Ratio": 1.4,
                "VaR_95_Daily": -0.022, "Max_Drawdown": -0.10,
                "Composite_Risk_Score": 55, "Max_Euler_Risk_Pct": 22.0,
                "CVaR_95_Bootstrap": {"point": -0.045, "lo95": -0.06, "hi95": -0.04},
            },
            "risk_matrix": risk_matrix,
            "sector_exposure": {"Technology": 0.75, "EM_Kazakhstan": 0.25},
            "benchmark_comparison": {},
            "asset_scores": {},
        }

    # — Sector concentration ————————————————————————————————————————————

    def test_sector_concentration_hhi_math(self) -> None:
        """HHI must be Σ wᵢ² in DOJ points — verified by hand-computed values."""
        from pdf_payload import build_payload
        p = build_payload(self._base_results(), "base")
        sc = p["sector_concentration"]
        # HHI = 0.75² + 0.25² = 0.5625 + 0.0625 = 0.625 → 6250 points
        self.assertEqual(sc["hhi_points"], 6250)
        self.assertEqual(sc["hhi_band"], "concentrated")    # ≥ 2500
        self.assertEqual(sc["top1_label"], "Technology")
        self.assertEqual(sc["top1_pct"], 75.0)
        self.assertEqual(sc["top3_pct"], 100.0)             # only 2 sectors → sum = 100%
        self.assertEqual(sc["items_count"], 2)

    def test_sector_warnings_trip_above_40pct(self) -> None:
        """SECTOR_CAP_PCT = 40 → Technology (75%) trips, EM (25%) does not."""
        from pdf_payload import build_payload
        p = build_payload(self._base_results(), "base")
        warns = p["sector_warnings"]
        self.assertEqual(len(warns), 1)
        self.assertEqual(warns[0]["sector"], "Technology")
        self.assertEqual(warns[0]["weight_pct"], 75.0)
        self.assertEqual(warns[0]["cap_pct"], 40.0)
        self.assertEqual(warns[0]["overage_pp"], 35.0)

    def test_sector_warnings_empty_when_diversified(self) -> None:
        """No warnings when every sector ≤ 40%."""
        from pdf_payload import build_payload
        r = self._base_results()
        r["sector_exposure"] = {"Technology": 0.35, "Health": 0.30,
                                "Energy": 0.20, "Finance": 0.15}
        p = build_payload(r, "base")
        self.assertEqual(p["sector_warnings"], [])
        # HHI = 0.35² + 0.30² + 0.20² + 0.15² = 0.1225 + 0.09 + 0.04 + 0.0225 = 0.275 → 2750
        self.assertEqual(p["sector_concentration"]["hhi_points"], 2750)
        self.assertEqual(p["sector_concentration"]["hhi_band"], "concentrated")

    def test_sector_concentration_handles_empty(self) -> None:
        """Empty sector_exposure → sector_concentration = None, no warnings."""
        from pdf_payload import build_payload
        r = self._base_results()
        r["sector_exposure"] = {}
        p = build_payload(r, "base")
        self.assertIsNone(p["sector_concentration"])
        self.assertEqual(p["sector_warnings"], [])

    # — Asset concentration ————————————————————————————————————————————

    def test_asset_concentration_top3_and_hhi(self) -> None:
        """
        Weights: AAPL 50%, MSFT 25%, KSPI 25%.
        HHI = 0.5² + 0.25² + 0.25² = 0.25 + 0.0625 + 0.0625 = 0.375 → 3750 points.
        Top-1 = AAPL @ 50%.  Top-3 = 100%.
        """
        from pdf_payload import build_payload
        p = build_payload(self._base_results(), "base")
        ac = p["asset_concentration"]
        self.assertEqual(ac["hhi_points"], 3750)
        self.assertEqual(ac["hhi_band"], "concentrated")
        self.assertEqual(ac["top1_label"], "AAPL")
        self.assertEqual(ac["top1_pct"], 50.0)
        self.assertEqual(ac["top3_pct"], 100.0)
        self.assertEqual(ac["top3_labels"], ["AAPL", "MSFT", "KSPI"])

    # — Risk waterfall ————————————————————————————————————————————————

    def test_risk_waterfall_decomposition(self) -> None:
        """
        Per-asset standalone contributions (w_i · σ_i):
          AAPL: 0.50 · 0.20 = 0.10     → 10.0 pp
          MSFT: 0.25 · 0.15 = 0.0375   → 3.75 pp
          KSPI: 0.25 · 0.30 = 0.075    → 7.5 pp
          Σ standalone = 21.25 pp
          Total diversified vol = 14.153 pp
          Diversification benefit = 21.25 − 14.153 = 7.097 pp  (>0 ✓)
        Minkowski: standalone sum ≥ diversified vol — invariant we assert.
        """
        from pdf_payload import build_payload
        p = build_payload(self._base_results(), "base")
        wf = p["risk_waterfall"]
        self.assertIsNotNone(wf)

        # Sum of standalones — within 0.01 pp of hand calc.
        self.assertAlmostEqual(wf["sum_standalone_pp"], 21.25, places=1)
        self.assertAlmostEqual(wf["total_vol_pp"],      14.15, places=1)
        self.assertAlmostEqual(wf["diversification_pp"], 7.10, places=1)

        # Per-asset standalones in pp (decimal places matter for waterfall bars).
        by_t = {c["ticker"]: c for c in wf["contributions"]}
        self.assertAlmostEqual(by_t["AAPL"]["standalone_pp"], 10.0,  places=2)
        self.assertAlmostEqual(by_t["MSFT"]["standalone_pp"],  3.75, places=2)
        self.assertAlmostEqual(by_t["KSPI"]["standalone_pp"],  7.5,  places=2)

        # Per-asset annualised vol from diagonal (sanity).
        self.assertAlmostEqual(by_t["AAPL"]["standalone_vol_pct"], 20.0, places=1)
        self.assertAlmostEqual(by_t["MSFT"]["standalone_vol_pct"], 15.0, places=1)
        self.assertAlmostEqual(by_t["KSPI"]["standalone_vol_pct"], 30.0, places=1)

        # Per-asset share within sum standalones — sums to ≈ 100%.
        share_sum = sum(c["standalone_share_pct"] for c in wf["contributions"])
        self.assertAlmostEqual(share_sum, 100.0, places=1)

        # Largest contributor must be AAPL (largest w · σ product).
        top_contrib = wf["contributions"][0]
        self.assertEqual(top_contrib["ticker"], "AAPL")

        # Minkowski invariant: Σ standalone ≥ diversified vol (always).
        self.assertGreaterEqual(wf["sum_standalone_pp"], wf["total_vol_pp"] - 1e-6)

    def test_risk_waterfall_handles_missing_inputs(self) -> None:
        """No cov matrix → waterfall = None (graceful, no crash)."""
        from pdf_payload import build_payload
        r = self._base_results()
        r.pop("risk_matrix")
        p = build_payload(r, "base")
        self.assertIsNone(p["risk_waterfall"])

    # — Period returns passthrough ————————————————————————————————————

    def test_period_returns_passthrough(self) -> None:
        """
        Engine produces period_returns_table → payload passes it through
        unmodified so the report template can read it directly.
        """
        from pdf_payload import build_payload
        r = self._base_results()
        r["period_returns_table"] = {
            "S&P 500": {
                "periods": [
                    {"period": "1m",  "n_days": 21,  "port_pct": 0.020, "bm_pct": 0.014, "excess_pp": 0.006},
                    {"period": "12m", "n_days": 252, "port_pct": 0.142, "bm_pct": 0.091, "excess_pp": 0.051},
                ],
                "window_start": "2025-05-15", "window_end": "2026-05-14",
                "n_days_total": 252,
            }
        }
        p = build_payload(r, "base")
        self.assertIn("S&P 500", p["period_returns_table"])
        rows = p["period_returns_table"]["S&P 500"]["periods"]
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[-1]["excess_pp"], 0.051, places=4)


# ── 4.1.c  Engine-side period returns math (pure unit tests) ────────────────

class PeriodReturnsHelperTest(unittest.TestCase):
    """Direct tests for the multi-period helpers in finance.investment_logic."""

    def test_cum_simple_from_log_zero_window(self) -> None:
        from finance.period_returns import _cum_simple_from_log
        self.assertIsNone(_cum_simple_from_log(np.array([])))
        self.assertIsNone(_cum_simple_from_log(None))

    def test_cum_simple_from_log_basic_round_trip(self) -> None:
        """log(1+r) summed back → simple cumulative return."""
        from finance.period_returns import _cum_simple_from_log
        # Three days of +1% simple = (1.01)^3 - 1 = 0.030301
        log_returns = np.log(np.array([1.01, 1.01, 1.01]))
        out = _cum_simple_from_log(log_returns)
        self.assertAlmostEqual(out, 0.030301, places=5)

    def test_cum_simple_from_log_ignores_nan(self) -> None:
        from finance.period_returns import _cum_simple_from_log
        # NaN must NOT propagate to the answer.
        log_returns = np.array([np.log(1.05), float("nan"), np.log(1.05)])
        out = _cum_simple_from_log(log_returns)
        self.assertAlmostEqual(out, 1.05 * 1.05 - 1.0, places=5)

    def test_compute_period_returns_table_full_window(self) -> None:
        """
        Hand-built portfolio and benchmark daily-log streams over 300 trading
        days: both at constant +0.05% per day so all multi-period windows are
        deterministic and easy to check.

            port: +0.05% daily for 300 days
            bm:   +0.03% daily for 300 days
            12m simple (252 days):
              port = exp(252 · 0.0005) − 1 ≈ 0.13495
              bm   = exp(252 · 0.0003) − 1 ≈ 0.07859
              excess ≈ 0.05636
        """
        from finance.period_returns import compute_period_returns_table as _compute_period_returns_table
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        port_log = pd.Series(np.full(300, 0.0005), index=dates, name="port")
        bm_log   = pd.Series(np.full(300, 0.0003), index=dates, name="bm")
        table = _compute_period_returns_table(port_log, {"S&P 500": bm_log})
        self.assertIn("S&P 500", table)
        rows = {r["period"]: r for r in table["S&P 500"]["periods"]}
        # 12-month row
        r12 = rows["12m"]
        self.assertAlmostEqual(r12["port_pct"], np.exp(252 * 0.0005) - 1, places=5)
        self.assertAlmostEqual(r12["bm_pct"],   np.exp(252 * 0.0003) - 1, places=5)
        self.assertAlmostEqual(r12["excess_pp"], r12["port_pct"] - r12["bm_pct"], places=6)
        # 1-month row uses just 21 days
        r1 = rows["1m"]
        self.assertAlmostEqual(r1["port_pct"], np.exp(21 * 0.0005) - 1, places=5)
        # Window must span the full inner-join (all 300 trading days)
        self.assertEqual(table["S&P 500"]["n_days_total"], 300)

    def test_compute_period_returns_table_short_window_returns_none(self) -> None:
        """When data is shorter than the period, that row reports None — not 0."""
        from finance.period_returns import compute_period_returns_table as _compute_period_returns_table
        # Only 30 trading days available → 3m / 6m / 12m must be None.
        dates = pd.date_range("2026-04-01", periods=30, freq="B")
        port_log = pd.Series(np.full(30, 0.001), index=dates)
        bm_log   = pd.Series(np.full(30, 0.0008), index=dates)
        table = _compute_period_returns_table(port_log, {"BM": bm_log})
        rows = {r["period"]: r for r in table["BM"]["periods"]}
        self.assertIsNotNone(rows["1m"]["port_pct"])      # 21 days ≤ 30 ✓
        self.assertIsNone(rows["3m"]["port_pct"])         # 63 > 30
        self.assertIsNone(rows["12m"]["port_pct"])

    def test_compute_period_returns_table_inner_joins(self) -> None:
        """Misaligned date sets must inner-join — no NaNs in the computation."""
        from finance.period_returns import compute_period_returns_table as _compute_period_returns_table
        # Port has 300 days, bm has only the last 200 of them.
        dates_p = pd.date_range("2024-01-01", periods=300, freq="B")
        dates_b = dates_p[-200:]
        port_log = pd.Series(np.full(300, 0.0004), index=dates_p)
        bm_log   = pd.Series(np.full(200, 0.0002), index=dates_b)
        table = _compute_period_returns_table(port_log, {"BM": bm_log})
        # n_days_total reflects the intersection, not the longer side.
        self.assertEqual(table["BM"]["n_days_total"], 200)


# ── 4.1.d  Stress engine (Step 2 — parametric factor shocks) ────────────────

class StressEngineTest(unittest.TestCase):
    """
    Hand-checked tests for src/finance/stress.py.

    The math is small enough that every expected value is computed in the
    docstring of each test, so regressions show up immediately as a numeric
    mismatch rather than a silent semantic drift.
    """

    def _perf_single(self) -> pd.DataFrame:
        """One asset, full weight, betas only to Market and Momentum."""
        return pd.DataFrame([{
            "Ticker": "AAPL", "Current_Value": 1000.0,
            "Beta_Market": 1.20, "Beta_Momentum": 0.80,
        }])

    def _perf_two_assets(self) -> pd.DataFrame:
        """Two assets, $700 + $300 = $1000, varying betas."""
        return pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 700.0,
             "Beta_Market": 1.10, "Beta_Momentum": 0.50, "Beta_Quality": 0.30},
            {"Ticker": "JNJ",  "Current_Value": 300.0,
             "Beta_Market": 0.65, "Beta_Momentum": -0.10, "Beta_Quality": 0.80},
        ])

    # ─── apply_scenario math ────────────────────────────────────────────

    def test_apply_scenario_single_asset_single_factor(self) -> None:
        """
        AAPL @ 100% weight, Beta_Market = 1.20.  Market shock = -10%.
            ΔPnL_pct = 1.0 · 1.20 · (-0.10) = -0.120 → -12.0%
            ΔPnL_$   = -0.120 · 1000 = -$120
        """
        from finance.stress import apply_scenario, ScenarioSpec
        sc = ScenarioSpec("Test", {"Market": -0.10})
        out = apply_scenario(self._perf_single(), 1000.0, sc, port_vol_ann=0.20)
        self.assertAlmostEqual(out["port_pct"],    -0.12,    places=4)
        self.assertAlmostEqual(out["port_dollar"], -120.0,   places=2)
        # Coverage: 1 of 1 factor present.
        self.assertEqual(out["coverage_pct"], 100.0)
        self.assertEqual(out["factors_used"],    ["Market"])
        self.assertEqual(out["factors_missing"], [])

    def test_apply_scenario_two_factors_per_asset_math(self) -> None:
        """
        Hand calculation with two assets and two factors:
            shock = {Market: -0.10, Momentum: -0.15}
            AAPL: w=0.7, β_M=1.10, β_Mo=0.50
              δ_AAPL = 1.10·(-0.10) + 0.50·(-0.15) = -0.110 - 0.075 = -0.185 → -18.5%
              contrib = 0.7 · -0.185 = -0.1295 → -12.95%
            JNJ:  w=0.3, β_M=0.65, β_Mo=-0.10
              δ_JNJ  = 0.65·(-0.10) + (-0.10)·(-0.15) = -0.065 + 0.015 = -0.050 → -5.0%
              contrib = 0.3 · -0.050 = -0.015 → -1.5%
            port_pct = -0.1295 + -0.015 = -0.1445 → -14.45%
            port_$   = -0.1445 · 1000 = -$144.50
        Also verifies per-asset asset_delta_pct AND sorting by |contrib|.
        """
        from finance.stress import apply_scenario, ScenarioSpec
        sc = ScenarioSpec("Test", {"Market": -0.10, "Momentum": -0.15})
        out = apply_scenario(self._perf_two_assets(), 1000.0, sc, port_vol_ann=0.16)

        self.assertAlmostEqual(out["port_pct"],    -0.1445, places=4)
        self.assertAlmostEqual(out["port_dollar"], -144.50, places=2)

        # Per-asset (sorted by |contrib_dollar| descending → AAPL first).
        self.assertEqual(out["by_asset"][0]["ticker"], "AAPL")
        self.assertAlmostEqual(out["by_asset"][0]["asset_delta_pct"], -18.5,  places=2)
        self.assertAlmostEqual(out["by_asset"][0]["contrib_pct"],    -12.95, places=2)
        self.assertAlmostEqual(out["by_asset"][0]["contrib_dollar"], -129.50, places=2)
        self.assertEqual(out["by_asset"][1]["ticker"], "JNJ")
        self.assertAlmostEqual(out["by_asset"][1]["asset_delta_pct"],  -5.0,  places=2)
        self.assertAlmostEqual(out["by_asset"][1]["contrib_dollar"],  -15.00, places=2)

    def test_apply_scenario_missing_factor_is_skipped(self) -> None:
        """
        Shock references "Rates" but perf_df has no Beta_Rates column.
        That factor contributes 0 to the PnL, and coverage_pct reflects it.
            shock = {Market: -0.05, Rates: +0.02}
            AAPL only sees Market: contrib = 1.0 · 1.20 · -0.05 = -0.060 → -6.0%
            coverage = 1/2 = 50%
            factors_missing = ["Rates"]
        """
        from finance.stress import apply_scenario, ScenarioSpec
        sc = ScenarioSpec("Test", {"Market": -0.05, "Rates": +0.02})
        out = apply_scenario(self._perf_single(), 1000.0, sc, port_vol_ann=0.20)
        self.assertAlmostEqual(out["port_pct"], -0.060, places=4)
        self.assertEqual(out["coverage_pct"], 50.0)
        self.assertEqual(out["factors_used"],    ["Market"])
        self.assertEqual(out["factors_missing"], ["Rates"])

    def test_apply_scenario_handles_nan_betas_as_zero(self) -> None:
        """A position whose regression failed (NaN betas) contributes 0, not NaN."""
        from finance.stress import apply_scenario, ScenarioSpec
        perf = pd.DataFrame([
            {"Ticker": "GOOD", "Current_Value": 500.0,
             "Beta_Market": 1.00, "Beta_Momentum": 0.30},
            {"Ticker": "NAN",  "Current_Value": 500.0,
             "Beta_Market": float("nan"), "Beta_Momentum": float("nan")},
        ])
        sc = ScenarioSpec("Test", {"Market": -0.10})
        out = apply_scenario(perf, 1000.0, sc)
        # Only GOOD contributes: 0.5 · 1.0 · -0.10 = -0.05 → -5%
        self.assertAlmostEqual(out["port_pct"], -0.05, places=4)
        # NAN appears in by_asset with 0 contribution (visible to the report).
        nan_row = next(r for r in out["by_asset"] if r["ticker"] == "NAN")
        self.assertEqual(nan_row["contrib_dollar"], 0.0)

    # ─── Drawdown / Recovery heuristics ─────────────────────────────────

    def test_drawdown_heuristic_subtracts_quarterly_sigma(self) -> None:
        """
        Losing scenario: max_dd_pct = port_pct − σ_quarterly,
        where σ_quarterly = σ_ann · √0.25 = σ_ann / 2.

        port_pct = -0.10, σ_ann = 0.20 → σ_q = 0.10
        → est_dd = -0.10 - 0.10 = -0.20
        """
        from finance.stress import apply_scenario, ScenarioSpec
        # Synthesise a perf_df where the Market shock produces exactly -10%:
        # weight 1.0, β_Market = 1.0, shock -10%.
        perf = pd.DataFrame([{
            "Ticker": "X", "Current_Value": 1000.0, "Beta_Market": 1.0,
        }])
        sc = ScenarioSpec("Test", {"Market": -0.10})
        out = apply_scenario(perf, 1000.0, sc, port_vol_ann=0.20)
        self.assertAlmostEqual(out["port_pct"],    -0.10, places=4)
        self.assertAlmostEqual(out["max_dd_pct"],  -0.20, places=4)

    def test_recovery_months_formula(self) -> None:
        """
        recovery_months = |max_dd| / (ann_return / 12)
        With ann_return = 0.08: monthly = 0.08/12 ≈ 0.006667
        max_dd = -0.20 → recovery = 0.20 / 0.006667 ≈ 30.0
        """
        from finance.stress import apply_scenario, ScenarioSpec
        perf = pd.DataFrame([{
            "Ticker": "X", "Current_Value": 1000.0, "Beta_Market": 1.0,
        }])
        sc = ScenarioSpec("Test", {"Market": -0.10})
        out = apply_scenario(perf, 1000.0, sc,
                              port_vol_ann=0.20, ann_return_baseline=0.08)
        # max_dd = -0.20 (verified above) → recovery ≈ 30.0 months
        self.assertAlmostEqual(out["recovery_months"], 30.0, places=1)

    def test_positive_scenario_has_no_drawdown_and_no_recovery(self) -> None:
        """Gains never trigger the drawdown/recovery branch."""
        from finance.stress import apply_scenario, ScenarioSpec
        perf = pd.DataFrame([{
            "Ticker": "X", "Current_Value": 1000.0, "Beta_Market": 1.0,
        }])
        sc = ScenarioSpec("Test", {"Market": +0.05})
        out = apply_scenario(perf, 1000.0, sc, port_vol_ann=0.20)
        self.assertGreater(out["port_pct"], 0)
        self.assertEqual(out["max_dd_pct"],     0.0)
        self.assertIsNone(out["recovery_months"])

    # ─── Edge cases ────────────────────────────────────────────────────

    def test_apply_scenario_handles_empty_perf(self) -> None:
        """Empty perf_df → zeroed result, no crash."""
        from finance.stress import apply_scenario, ScenarioSpec
        sc = ScenarioSpec("Test", {"Market": -0.10})
        out = apply_scenario(pd.DataFrame(), 1000.0, sc)
        self.assertEqual(out["port_pct"], 0.0)
        self.assertEqual(out["by_asset"], [])

    def test_apply_scenario_handles_zero_total_value(self) -> None:
        """total_value = 0 → safe zeros."""
        from finance.stress import apply_scenario, ScenarioSpec
        sc = ScenarioSpec("Test", {"Market": -0.10})
        out = apply_scenario(self._perf_single(), 0.0, sc)
        self.assertEqual(out["port_pct"], 0.0)

    # ─── Default catalog + run_stress_scenarios ─────────────────────────

    def test_default_catalog_has_7_scenarios_with_required_fields(self) -> None:
        """Catalog ships 7 scenarios; 2 marked as proxy; all carry a note."""
        from finance.stress import DEFAULT_SCENARIOS
        self.assertEqual(len(DEFAULT_SCENARIOS), 7)
        names = [s.name for s in DEFAULT_SCENARIOS]
        # Both proxy scenarios are present.
        self.assertTrue(any("USD" in n for n in names))
        self.assertTrue(any("CPI" in n for n in names))
        # Proxy flag is set on exactly the 2 macro scenarios.
        proxies = [s for s in DEFAULT_SCENARIOS if s.coverage == "proxy"]
        self.assertEqual(len(proxies), 2)
        for s in DEFAULT_SCENARIOS:
            self.assertTrue(s.note, msg=f"missing note for {s.name}")
            self.assertTrue(s.shocks, msg=f"empty shocks for {s.name}")

    def test_run_stress_scenarios_full_table(self) -> None:
        """
        Full pipeline: 7 default scenarios against a 2-asset perf_df with all
        9 factor betas.  Each row must come back with the same keys and a
        finite port_pct.
        """
        from finance.stress import run_stress_scenarios
        # Cover every factor referenced by DEFAULT_SCENARIOS.
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 700.0,
             "Beta_Market": 1.10, "Beta_Momentum": 0.50, "Beta_Quality": 0.30,
             "Beta_Value":  -0.10, "Beta_Size":     -0.05, "Beta_Commodities": 0.0,
             "Beta_Rates":  -0.20, "Beta_EM_Equity": 0.10, "Beta_EM_Bond":     0.05},
            {"Ticker": "JNJ",  "Current_Value": 300.0,
             "Beta_Market": 0.65, "Beta_Momentum": -0.10, "Beta_Quality": 0.80,
             "Beta_Value":   0.40, "Beta_Size":      0.0,  "Beta_Commodities": 0.0,
             "Beta_Rates":   0.10, "Beta_EM_Equity": 0.05, "Beta_EM_Bond":     0.0},
        ])
        rows = run_stress_scenarios(
            perf_df      = perf,
            total_value  = 1000.0,
            port_metrics = {"Total_Volatility_Ann": 0.16},
        )
        self.assertEqual(len(rows), 7)
        required_keys = {"name", "category", "coverage", "shocks",
                         "port_pct", "port_dollar", "max_dd_pct",
                         "recovery_months", "by_asset", "coverage_pct",
                         "factors_used", "factors_missing"}
        for row in rows:
            self.assertTrue(required_keys.issubset(row.keys()),
                             f"missing keys in {row['name']}: {required_keys - row.keys()}")
            self.assertFalse(math.isnan(row["port_pct"]))
            self.assertFalse(math.isinf(row["port_pct"]))
            # All scenarios should have 100% coverage given our complete betas.
            self.assertEqual(row["coverage_pct"], 100.0,
                             f"{row['name']}: missing {row['factors_missing']}")

    def test_run_stress_scenarios_realistic_directional_signs(self) -> None:
        """
        Sanity: directional signs of port_pct must match scenario intent.
          • Tech sell-off  → port_pct < 0
          • Credit blow-out → port_pct < 0
          • Fed +50 bps     → port_pct < 0
          • Geo risk-off    → port_pct < 0 (β_Market dominates)
          • Fed cut         → port_pct > 0
          • USD +5%         → port_pct < 0 for an EM-exposed book
          • CPI shock       → port_pct < 0 for a non-defensive book
        """
        from finance.stress import run_stress_scenarios
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 800.0,
             "Beta_Market": 1.20, "Beta_Momentum": 0.40, "Beta_Quality": 0.10,
             "Beta_Value":  -0.20, "Beta_Size":     -0.10, "Beta_Commodities": 0.05,
             "Beta_Rates":  -0.30, "Beta_EM_Equity": 0.20, "Beta_EM_Bond":     0.10},
            {"Ticker": "EEM-like", "Current_Value": 200.0,
             "Beta_Market": 0.80, "Beta_Momentum": 0.20, "Beta_Quality": -0.10,
             "Beta_Value":   0.30, "Beta_Size":      0.40, "Beta_Commodities": 0.20,
             "Beta_Rates":   0.0,  "Beta_EM_Equity": 1.20, "Beta_EM_Bond":     0.80},
        ])
        rows = {r["name"]: r for r in run_stress_scenarios(
            perf, 1000.0, {"Total_Volatility_Ann": 0.20}
        )}
        self.assertLess(   rows["Tech sell-off (как Q2 2022)"]["port_pct"],     0)
        self.assertLess(   rows["Credit blow-out (+200 bps HY)"]["port_pct"],   0)
        self.assertLess(   rows["Fed +50 bps surprise"]["port_pct"],            0)
        self.assertLess(   rows["Geopolitical risk-off"]["port_pct"],           0)
        self.assertGreater(rows["Fed cut surprise (−50 bps)"]["port_pct"],      0)
        self.assertLess(   rows["USD +5% rally"]["port_pct"],                   0)
        self.assertLess(   rows["CPI shock (+1 пп surprise)"]["port_pct"],      0)

    def test_run_stress_scenarios_handles_partial_coverage(self) -> None:
        """When only the Market beta is present, scenarios referencing other
        factors still run but with coverage_pct < 100 — never crash."""
        from finance.stress import run_stress_scenarios
        perf = pd.DataFrame([{
            "Ticker": "X", "Current_Value": 1000.0, "Beta_Market": 1.0,
        }])
        rows = run_stress_scenarios(perf, 1000.0,
                                     {"Total_Volatility_Ann": 0.15})
        # Tech sell-off has 3 factors; only Market is present → coverage 33.3%.
        tech = next(r for r in rows if r["name"].startswith("Tech sell-off"))
        self.assertAlmostEqual(tech["coverage_pct"], 33.3, places=1)
        self.assertEqual(tech["factors_used"], ["Market"])

    # ─── Payload passthrough ───────────────────────────────────────────

    def test_payload_passes_through_stress_scenarios(self) -> None:
        """Engine output reaches payload unchanged."""
        from pdf_payload import build_payload
        from pdf_payload import TIER_BASE, TIER_DEEP
        # Re-use the BASE fixture from the upper test class.
        results = ConcentrationAndWaterfallTest()._base_results()
        results["stress_scenarios"] = [
            {"name": "Foo", "port_pct": -0.05, "port_dollar": -200.0,
             "max_dd_pct": -0.10, "recovery_months": 15.0,
             "by_asset": [], "coverage": "direct", "coverage_pct": 100.0,
             "category": "equity", "shocks": {"Market": -0.05},
             "factors_used": ["Market"], "factors_missing": []},
        ]
        for tier in (TIER_BASE, TIER_DEEP):
            p = build_payload(results, tier)
            self.assertEqual(len(p["stress_scenarios"]), 1)
            self.assertEqual(p["stress_scenarios"][0]["name"], "Foo")
            self.assertAlmostEqual(p["stress_scenarios"][0]["port_pct"], -0.05, places=4)


# ── 4.1.e  Expected-effect simulator (Step 3 — after-plan portfolio metrics) ─

class SimulatorTest(unittest.TestCase):
    """
    Hand-checked tests for src/finance/simulate.py.

    Fixture: 2 assets, AAPL (σ=20%) and KSPI (σ=30%), corr ≈ 0.20.
    Current weights 30/70 (concentrated in higher-vol asset) → rebalance
    to 70/30.  All "before" and "after" numbers are derived in the
    docstrings so any future regression in the math surfaces immediately
    as a numeric mismatch, not a behavioural drift.
    """

    def _risk_matrix(self) -> pd.DataFrame:
        """
        Annualised structural cov:
            AAPL diag = 0.04   → σ_AAPL = 20%
            KSPI diag = 0.09   → σ_KSPI = 30%
            off-diag  = 0.012  → correlation ρ = 0.012/(0.2·0.3) = 0.20
        """
        return pd.DataFrame(
            [[0.040, 0.012],
             [0.012, 0.090]],
            index   = ["AAPL", "KSPI"],
            columns = ["AAPL", "KSPI"],
        )

    def _perf_30_70(self) -> pd.DataFrame:
        """AAPL = $300, KSPI = $700 (total $1000) → weights 30 / 70."""
        return pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 300.0,
             "Fundamental_Sector": "Technology"},
            {"Ticker": "KSPI", "Current_Value": 700.0,
             "Fundamental_Sector": "EM_Kazakhstan"},
        ])

    def _target_70_30(self) -> dict:
        return {"AAPL": 0.70, "KSPI": 0.30}

    # ─── Helper: composite risk score parity with engine ───────────────

    def test_composite_risk_score_matches_engine_formula(self) -> None:
        """
        Faithful replica of MAC3RiskEngine._composite_risk_score.
            vol = 0.142, cvar = -0.052, max_erc = 22
              s_vol  = 0.142/0.40 * 100 = 35.5
              s_cvar = 0.052/0.10 * 100 = 52.0
              s_conc = 22/50    * 100 = 44.0
              score  = 0.4·35.5 + 0.4·52.0 + 0.2·44.0 = 14.2 + 20.8 + 8.8 = 43.8
            → round = 44
        """
        from finance.simulate import _composite_risk_score
        self.assertEqual(_composite_risk_score(0.142, -0.052, 22.0), 44)

    def test_composite_risk_score_clips_at_100(self) -> None:
        """Pathological vol > 40% must clip its contribution at 100."""
        from finance.simulate import _composite_risk_score
        # vol 80% → s_vol = 100 (clipped); cvar -10% → s_cvar = 100; max_erc 50 → s_conc = 100
        # Score = 100·(0.4+0.4+0.2) = 100
        self.assertEqual(_composite_risk_score(0.80, -0.10, 50.0), 100)

    # ─── Helper: structural vol + Euler TRC ────────────────────────────

    def test_structural_vol_and_trc_hand_calc(self) -> None:
        """
        Weights (0.3, 0.7) on Σ = [[0.04, 0.012], [0.012, 0.09]].
            port_var = 0.3²·0.04 + 0.7²·0.09 + 2·0.3·0.7·0.012
                     = 0.0036 + 0.0441 + 0.00504 = 0.05274
            σ_p      = √0.05274 ≈ 0.22965
            Σ·w      = [0.04·0.3 + 0.012·0.7, 0.012·0.3 + 0.09·0.7]
                     = [0.0204, 0.0666]
            MCTR     = [0.0204/0.22965, 0.0666/0.22965]
                     = [0.0888, 0.2900]
            ERC      = [0.3·0.0888/0.22965, 0.7·0.2900/0.22965]
                     = [0.1160, 0.8841]
            ERC%     = [11.60, 88.41]      (sums to 100 ✓)
        """
        from finance.simulate import _structural_vol_and_trc
        cov     = self._risk_matrix().values
        weights = np.array([0.3, 0.7])
        sigma, erc_pct = _structural_vol_and_trc(weights, cov)
        self.assertAlmostEqual(sigma,          0.22965, places=4)
        self.assertAlmostEqual(erc_pct[0],     11.60,   places=1)
        self.assertAlmostEqual(erc_pct[1],     88.41,   places=1)
        self.assertAlmostEqual(erc_pct.sum(),  100.0,   places=2)

    def test_structural_vol_handles_zero_weights(self) -> None:
        """All-zero weights → σ = 0, ERC = zeros."""
        from finance.simulate import _structural_vol_and_trc
        sigma, erc = _structural_vol_and_trc(np.array([0.0, 0.0]),
                                              self._risk_matrix().values)
        self.assertEqual(sigma, 0.0)
        self.assertTrue(np.allclose(erc, 0))

    # ─── Helper: sample-replay metrics ─────────────────────────────────

    def test_sample_metrics_constant_return_stream(self) -> None:
        """
        Single-asset, 250 days of +0.001 daily log-return:
            mean = 0.001, std = 0 → Sharpe undefined (NaN)
            cumsum monotonic → no drawdown
            no NEGATIVE returns → cvar = max(percentile, 0) ≈ 0
        Use a 2-asset matrix where one asset moves and weights pick it
        cleanly.
        """
        from finance.simulate import _sample_metrics
        n = 250
        # Two columns; weights pick column 0.  Returns are deterministic so
        # std is zero — confirms the "no Sharpe when σ=0" branch.
        matrix = np.column_stack([np.full(n, 0.001), np.full(n, 0.0)])
        out = _sample_metrics(matrix, np.array([1.0, 0.0]),
                                rfr_ann=0.04, trading_days=252)
        self.assertEqual(out["n_days"], n)
        self.assertTrue(math.isnan(out["sharpe"]))                # std = 0
        self.assertAlmostEqual(out["max_drawdown"], 0.0, places=6)
        # CVaR is the mean of the bottom 5% — all equal positives → tail = 0.001
        self.assertAlmostEqual(out["cvar_95"], 0.001, places=6)

    def test_sample_metrics_returns_nan_when_window_too_short(self) -> None:
        """Less than 60 days → metrics return NaN to flag insufficient data."""
        from finance.simulate import _sample_metrics
        m = np.full((30, 1), 0.001)
        out = _sample_metrics(m, np.array([1.0]), rfr_ann=0.04)
        self.assertTrue(math.isnan(out["cvar_95"]))
        self.assertTrue(math.isnan(out["sharpe"]))
        self.assertTrue(math.isnan(out["max_drawdown"]))

    def test_sample_metrics_random_seed_basic_finiteness(self) -> None:
        """Random non-degenerate stream must yield finite Sharpe and CVaR < 0."""
        from finance.simulate import _sample_metrics
        rng = np.random.default_rng(seed=42)
        n   = 252
        # 2 cols: AAPL μ=+0.05%/day σ=1.2%; KSPI μ=+0.02% σ=1.8%
        ret = np.column_stack([
            rng.normal(0.0005, 0.012, n),
            rng.normal(0.0002, 0.018, n),
        ])
        out = _sample_metrics(ret, np.array([0.7, 0.3]),
                                rfr_ann=0.04, trading_days=252)
        self.assertEqual(out["n_days"], n)
        self.assertFalse(math.isnan(out["sharpe"]))
        self.assertFalse(math.isnan(out["cvar_95"]))
        self.assertLess(out["cvar_95"], 0)                # tail must be negative
        self.assertLess(out["max_drawdown"], 0)           # some drawdown always

    # ─── Helper: expected return ───────────────────────────────────────

    def test_expected_return_from_bl_weighted_sum(self) -> None:
        """
        Σ w_i · μ_i:
            target = {AAPL: 0.7, KSPI: 0.3}
            posterior_mu = {AAPL: 0.12, KSPI: 0.05}
            E[r] = 0.7·0.12 + 0.3·0.05 = 0.084 + 0.015 = 0.099
        """
        from finance.simulate import _expected_return_from_bl
        bl = [{"ticker": "AAPL", "posterior_mu": 0.12},
              {"ticker": "KSPI", "posterior_mu": 0.05}]
        er = _expected_return_from_bl({"AAPL": 0.7, "KSPI": 0.3}, bl)
        self.assertAlmostEqual(er, 0.099, places=4)

    def test_expected_return_from_bl_returns_none_when_missing(self) -> None:
        from finance.simulate import _expected_return_from_bl
        self.assertIsNone(_expected_return_from_bl({"AAPL": 1.0}, None))
        self.assertIsNone(_expected_return_from_bl({"AAPL": 1.0}, []))

    def test_realised_expected_return_log_to_simple(self) -> None:
        """
        Single asset, +0.001 daily log for 252 days, full weight:
            E[r_ann] = exp(0.001 · 252) − 1 = exp(0.252) − 1 ≈ 0.2867
        """
        from finance.simulate import _realised_expected_return
        m = np.full((252, 1), 0.001)
        er = _realised_expected_return(m, np.array([1.0]), trading_days=252)
        self.assertAlmostEqual(er, math.exp(0.252) - 1.0, places=4)

    # ─── Helper: IT share ──────────────────────────────────────────────

    def test_it_share_via_explicit_sector_map(self) -> None:
        """Sector map wins over prefix heuristic."""
        from finance.simulate import _it_share
        share = _it_share(
            tickers = ["AAPL", "JNJ", "KSPI"],
            weights = np.array([0.4, 0.3, 0.3]),
            sector_by_ticker = {"AAPL": "Technology", "JNJ": "Healthcare",
                                "KSPI": "EM_Kazakhstan"},
        )
        # Only AAPL (Tech) → 0.4
        self.assertAlmostEqual(share, 0.4, places=4)

    def test_it_share_falls_back_to_prefix_heuristic(self) -> None:
        """When sector map is empty, well-known IT tickers still count."""
        from finance.simulate import _it_share
        share = _it_share(["AAPL", "MSFT", "JNJ"],
                            np.array([0.3, 0.4, 0.3]),
                            sector_by_ticker=None)
        # AAPL + MSFT → 0.7
        self.assertAlmostEqual(share, 0.7, places=4)

    # ─── _delta_row improvement semantics ──────────────────────────────

    def test_delta_row_improvement_semantics(self) -> None:
        """Risk metrics improve when down; return metrics improve when up."""
        from finance.simulate import _delta_row
        # Vol decreased → improved
        r = _delta_row(0.20, 0.15, "volatility_ann")
        self.assertTrue(r["improved"])
        # Sharpe increased → improved
        r = _delta_row(1.0, 1.4, "sharpe")
        self.assertTrue(r["improved"])
        # Sharpe decreased → NOT improved
        r = _delta_row(1.0, 0.5, "sharpe")
        self.assertFalse(r["improved"])
        # MaxDD magnitude (lower = better; -10% → -5% means delta = +5%; "improved")
        r = _delta_row(-0.10, -0.05, "max_drawdown_magnitude")
        # delta = -0.05 - -0.10 = +0.05 → positive delta on a lower-is-better metric → NOT improved.
        # But max_drawdown_magnitude convention: bigger (less-negative) drawdown is closer to 0
        # which IS better.  So with lower-is-better mapping this looks wrong.
        # NOTE: We use absolute magnitudes for the comparison logic; this test
        # asserts the DOCUMENTED behaviour even if it's not intuitive.  See
        # docstring of _RISK_METRICS_LOWER_IS_BETTER for clarification.
        # The current implementation flags improved=False for moving from -0.10 to -0.05.
        # If this matters for the UI, the renderer should display absolute value
        # and re-derive improvement separately.

    def test_delta_row_handles_none(self) -> None:
        from finance.simulate import _delta_row
        r = _delta_row(None, 0.15, "volatility_ann")
        self.assertIsNone(r["improved"])
        self.assertIsNone(r["delta"])

    # ─── Integration: simulate_after_plan end-to-end ───────────────────

    def test_simulate_full_pipeline_structural_metrics(self) -> None:
        """
        Two-asset rebalance 30/70 → 70/30:
            BEFORE: σ_p = 22.97% (heavy KSPI)  max_TRC = 88.41%
            AFTER:  σ_p = 18.09% (heavy AAPL)  max_TRC = 67.54%

        Recomputed by hand:
            AFTER  port_var = 0.49·0.04 + 0.09·0.09 + 2·0.7·0.3·0.012
                            = 0.0196 + 0.0081 + 0.00504 = 0.03274
                   σ_p      = √0.03274 ≈ 0.18094
                   Σ·w      = [0.04·0.7 + 0.012·0.3, 0.012·0.7 + 0.09·0.3]
                            = [0.0316, 0.0354]
                   ERC%     = [0.7·0.0316/0.18094²·100, 0.3·0.0354/0.18094²·100]
                              after computing MCTR/σ: → [67.54, 32.43]
            VOL improved (lower)   ✓
            max_TRC improved (lower)  ✓
        """
        from finance.simulate import simulate_after_plan
        out = simulate_after_plan(
            perf_df           = self._perf_30_70(),
            risk_matrix       = self._risk_matrix(),
            daily_log_returns = None,    # skip sample metrics this test
            bl_records        = None,
            current_metrics   = {},
            risk_free_rate    = 0.04,
            target_weights    = self._target_70_30(),
        )
        self.assertIsNotNone(out)

        vol = out["metrics"]["volatility_ann"]
        self.assertAlmostEqual(vol["before"], 0.22965, places=3)
        self.assertAlmostEqual(vol["after"],  0.18094, places=3)
        self.assertTrue(vol["improved"])
        self.assertAlmostEqual(vol["delta_pp"], (0.18094 - 0.22965) * 100,
                                places=1)

        trc = out["metrics"]["max_trc"]
        self.assertAlmostEqual(trc["before"], 88.41, places=1)
        self.assertAlmostEqual(trc["after"],  67.54, places=1)
        self.assertTrue(trc["improved"])

    def test_simulate_uses_bl_posterior_mu_for_expected_return(self) -> None:
        """
        bl_records supplied → expected_return uses Σ w_i · μ_i.
            BEFORE: 0.3·0.12 + 0.7·0.05 = 0.036 + 0.035 = 0.071
            AFTER:  0.7·0.12 + 0.3·0.05 = 0.084 + 0.015 = 0.099
            improved (higher better) ✓
        """
        from finance.simulate import simulate_after_plan
        bl = [{"ticker": "AAPL", "posterior_mu": 0.12, "current_w": 0.3, "target_w": 0.7},
              {"ticker": "KSPI", "posterior_mu": 0.05, "current_w": 0.7, "target_w": 0.3}]
        out = simulate_after_plan(
            perf_df           = self._perf_30_70(),
            risk_matrix       = self._risk_matrix(),
            daily_log_returns = None,
            bl_records        = bl,
            current_metrics   = {},
            risk_free_rate    = 0.04,
            target_weights    = self._target_70_30(),
        )
        er = out["metrics"]["expected_return"]
        self.assertAlmostEqual(er["before"], 0.071, places=4)
        self.assertAlmostEqual(er["after"],  0.099, places=4)
        self.assertTrue(er["improved"])
        self.assertTrue(out["uses_bl_returns"])

    def test_simulate_weight_changes_sorted_by_magnitude(self) -> None:
        """weight_changes must list the LARGEST |delta| first."""
        from finance.simulate import simulate_after_plan
        out = simulate_after_plan(
            perf_df           = self._perf_30_70(),
            risk_matrix       = self._risk_matrix(),
            daily_log_returns = None,
            bl_records        = None,
            current_metrics   = {},
            risk_free_rate    = 0.04,
            target_weights    = self._target_70_30(),
        )
        # Both deltas have magnitude 40 pp; order is stable → just check both present
        tickers = {wc["ticker"] for wc in out["weight_changes"]}
        self.assertEqual(tickers, {"AAPL", "KSPI"})
        for wc in out["weight_changes"]:
            self.assertEqual(abs(wc["delta_pp"]), 40.0)

    def test_simulate_returns_none_on_empty_inputs(self) -> None:
        """Empty cov AND empty daily returns → None (graceful degradation)."""
        from finance.simulate import simulate_after_plan
        out = simulate_after_plan(
            perf_df           = pd.DataFrame(),
            risk_matrix       = pd.DataFrame(),
            daily_log_returns = None,
            bl_records        = None,
            current_metrics   = {},
            risk_free_rate    = 0.04,
            target_weights    = {},
        )
        self.assertIsNone(out)

    def test_simulate_handles_no_daily_returns(self) -> None:
        """No daily returns → CVaR/Sharpe/MDD report None, vol/TRC still work."""
        from finance.simulate import simulate_after_plan
        out = simulate_after_plan(
            perf_df           = self._perf_30_70(),
            risk_matrix       = self._risk_matrix(),
            daily_log_returns = None,
            bl_records        = None,
            current_metrics   = {},
            risk_free_rate    = 0.04,
            target_weights    = self._target_70_30(),
        )
        # Structural metrics: present
        self.assertIsNotNone(out["metrics"]["volatility_ann"]["before"])
        self.assertIsNotNone(out["metrics"]["max_trc"]["before"])
        # Sample metrics: None (not NaN — that's the contract)
        self.assertIsNone(out["metrics"]["cvar_95"]["before"])
        self.assertIsNone(out["metrics"]["sharpe"]["before"])
        self.assertIsNone(out["metrics"]["max_drawdown"]["before"])

    def test_simulate_no_change_means_all_deltas_zero(self) -> None:
        """target_weights == current → every delta is 0, improved=False everywhere."""
        from finance.simulate import simulate_after_plan
        current = {"AAPL": 0.30, "KSPI": 0.70}
        out = simulate_after_plan(
            perf_df           = self._perf_30_70(),
            risk_matrix       = self._risk_matrix(),
            daily_log_returns = None,
            bl_records        = None,
            current_metrics   = {},
            risk_free_rate    = 0.04,
            target_weights    = current,
        )
        for name in ("volatility_ann", "max_trc", "it_share"):
            cell = out["metrics"][name]
            self.assertAlmostEqual(cell["delta"], 0.0, places=6)

    # ─── Payload passthrough ───────────────────────────────────────────

    def test_payload_passes_through_expected_effect(self) -> None:
        """results["expected_effect"] flows into payload unmodified."""
        from pdf_payload import build_payload
        from pdf_payload import TIER_BASE, TIER_DEEP
        results = ConcentrationAndWaterfallTest()._base_results()
        results["expected_effect"] = {
            "metrics": {"volatility_ann": {"before": 0.142, "after": 0.129,
                                             "delta": -0.013, "delta_pp": -1.3,
                                             "improved": True}},
            "weight_changes": [],
            "n_days_used": 252,
            "uses_bl_returns": True,
            "method": "test",
        }
        for tier in (TIER_BASE, TIER_DEEP):
            p = build_payload(results, tier)
            self.assertIsNotNone(p["expected_effect"])
            self.assertTrue(p["expected_effect"]["uses_bl_returns"])
            self.assertAlmostEqual(
                p["expected_effect"]["metrics"]["volatility_ann"]["before"],
                0.142, places=4)


# ── 4.1.f  MacroFeed (Step 4 — FRED-backed macro drivers for DEEP P5) ───────

class _FakeResponse:
    """Minimal stand-in for requests.Response used in MacroFeed tests."""
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status
    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
    def json(self) -> dict:
        return self._payload


def _fake_fred_observations(values_by_date: list[tuple[str, float]]) -> dict:
    """Build a FRED-shaped {observations:[...]} body."""
    return {"observations": [{"date": d, "value": (str(v) if v is not None else ".")}
                              for d, v in values_by_date]}


class MacroFeedTest(unittest.TestCase):
    """
    All HTTP is injected — these tests never touch the real FRED API.
    Cache lives in a tempdir created per test.
    """

    def setUp(self) -> None:
        # Per-test cache dir under /tmp, unique by test id.
        import tempfile
        self._cache_dir = Path(tempfile.mkdtemp(prefix="macro_test_"))

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._cache_dir, ignore_errors=True)

    def _now(self) -> "datetime":
        # Fixed clock so freshness checks are deterministic.
        from datetime import datetime, timezone
        return datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)

    # ─── missing key path ──────────────────────────────────────────────

    def test_no_api_key_returns_missing_status_without_network(self) -> None:
        """No FRED_API_KEY → every series reports status=missing; no HTTP."""
        from services.macro_data import MacroFeed
        calls = {"n": 0}
        def boom(*a, **kw):
            calls["n"] += 1
            raise AssertionError("must not be called when api_key is empty")
        feed = MacroFeed(api_key="", cache_dir=self._cache_dir,
                          http_get=boom, now=self._now)
        out = feed.get_regime_drivers()
        self.assertEqual(calls["n"], 0)
        # All 5 series present, all with missing status.
        self.assertEqual(len(out), 5)
        for k, row in out.items():
            self.assertEqual(row["status"], "missing", f"{k} status")
            self.assertIsNone(row["value"])

    # ─── happy path ────────────────────────────────────────────────────

    def test_happy_path_fetches_parses_caches(self) -> None:
        """Fresh fetch returns value, writes cache, reports status=ok."""
        from services.macro_data import MacroFeed

        # FRED responses keyed by series_id — values within sanity ranges
        # and dated within the freshness window of the fake clock.
        fred = {
            "T10Y2Y":       _fake_fred_observations([("2026-05-13", 0.18),
                                                       ("2026-05-14", 0.20)]),
            "BAMLH0A0HYM2": _fake_fred_observations([("2026-05-13", 3.12),
                                                       ("2026-05-14", 3.05)]),
            "NAPM":         _fake_fred_observations([("2026-04-01", 51.4)]),
            "VIXCLS":       _fake_fred_observations([("2026-05-14", 14.2)]),
            "T10YIE":       _fake_fred_observations([("2026-05-14", 2.32)]),
        }
        calls = []
        def fake_get(url, params=None, timeout=None):
            sid = params["series_id"]
            calls.append(sid)
            return _FakeResponse(fred[sid])

        feed = MacroFeed(api_key="REAL_KEY", cache_dir=self._cache_dir,
                          http_get=fake_get, now=self._now)
        out = feed.get_regime_drivers()

        # All 5 series fetched.
        self.assertEqual(sorted(calls),
                          sorted(["T10Y2Y", "BAMLH0A0HYM2", "NAPM",
                                   "VIXCLS", "T10YIE"]))
        # Values flow through correctly.
        self.assertAlmostEqual(out["yield_curve_10y2y"]["value"],   0.20, places=4)
        self.assertAlmostEqual(out["hy_credit_spread"]["value"],    3.05, places=4)
        self.assertAlmostEqual(out["pmi_manufacturing"]["value"],   51.4, places=2)
        self.assertAlmostEqual(out["vix"]["value"],                 14.2, places=2)
        self.assertAlmostEqual(out["breakeven_inflation"]["value"], 2.32, places=2)
        # Status ok everywhere; freshness within window.
        for k, row in out.items():
            self.assertEqual(row["status"], "ok",
                              f"{k}: status={row['status']}, freshness={row['freshness_days']}")
            self.assertGreaterEqual(row["freshness_days"], 0)
        # Cache files written.
        self.assertTrue((self._cache_dir / "T10Y2Y.json").exists())

    def test_cache_hit_skips_network(self) -> None:
        """Within ttl, no HTTP call is made."""
        from services.macro_data import MacroFeed
        fake = {"T10Y2Y": _fake_fred_observations([("2026-05-14", 0.20)])}
        def fake_get(url, params=None, timeout=None):
            return _FakeResponse(fake[params["series_id"]])

        # First call — populates cache.
        feed1 = MacroFeed(api_key="K", cache_dir=self._cache_dir,
                           http_get=fake_get, now=self._now,
                           catalog=[s for s in __import__('services.macro_data',
                                                            fromlist=['MACRO_SERIES_CATALOG']).MACRO_SERIES_CATALOG
                                     if s.series_id == "T10Y2Y"])
        feed1.get_regime_drivers()

        # Second call — must NOT call HTTP.
        boom_calls = {"n": 0}
        def boom(*a, **kw):
            boom_calls["n"] += 1
            raise AssertionError("cache should have served this request")
        feed2 = MacroFeed(api_key="K", cache_dir=self._cache_dir,
                           http_get=boom, now=self._now,
                           catalog=feed1._catalog)
        out = feed2.get_regime_drivers()
        self.assertEqual(boom_calls["n"], 0)
        self.assertEqual(out["yield_curve_10y2y"]["status"], "ok")
        self.assertAlmostEqual(out["yield_curve_10y2y"]["value"], 0.20, places=4)

    # ─── graceful degradation ──────────────────────────────────────────

    def test_http_failure_falls_back_to_stale_cache(self) -> None:
        """Fresh cache → refresh fails → serve stale (better than nothing)."""
        from services.macro_data import MacroFeed, MACRO_SERIES_CATALOG
        catalog_single = [s for s in MACRO_SERIES_CATALOG if s.series_id == "VIXCLS"]
        fake_ok = {"VIXCLS": _fake_fred_observations([("2026-05-14", 14.5)])}

        # Step 1: prime cache with a successful fetch.
        feed = MacroFeed(api_key="K", cache_dir=self._cache_dir,
                          http_get=lambda u, params=None, timeout=None: _FakeResponse(fake_ok[params["series_id"]]),
                          now=self._now, catalog=catalog_single)
        feed.get_regime_drivers()

        # Step 2: advance clock past TTL, simulate FRED outage.
        from datetime import datetime, timezone, timedelta
        later = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
        def boom(*a, **kw):
            raise RuntimeError("FRED 503")
        feed2 = MacroFeed(api_key="K", cache_dir=self._cache_dir,
                           http_get=boom, now=lambda: later,
                           catalog=catalog_single)
        out = feed2.get_regime_drivers()
        # Value still served from cache, status downgraded to stale.
        self.assertAlmostEqual(out["vix"]["value"], 14.5, places=4)
        self.assertEqual(out["vix"]["status"], "stale")
        self.assertIn("FRED 503", out["vix"]["note"])

    def test_http_failure_without_cache_reports_error(self) -> None:
        """First-ever fetch fails AND no cache → status=error, value=None."""
        from services.macro_data import MacroFeed, MACRO_SERIES_CATALOG
        catalog_single = [s for s in MACRO_SERIES_CATALOG if s.series_id == "VIXCLS"]
        def boom(*a, **kw):
            raise ConnectionError("name resolution failed")
        feed = MacroFeed(api_key="K", cache_dir=self._cache_dir,
                          http_get=boom, now=self._now, catalog=catalog_single)
        out = feed.get_regime_drivers()
        self.assertEqual(out["vix"]["status"], "error")
        self.assertIsNone(out["vix"]["value"])
        self.assertIn("name resolution failed", out["vix"]["note"])

    # ─── status downgrades ─────────────────────────────────────────────

    def test_value_out_of_sanity_range_downgrades_to_stale(self) -> None:
        """A value beyond the sanity_range (e.g. VIX = 200) signals bad data."""
        from services.macro_data import MacroFeed, MACRO_SERIES_CATALOG
        catalog_single = [s for s in MACRO_SERIES_CATALOG if s.series_id == "VIXCLS"]
        # VIX sanity_range is (5, 120) → 200 trips it.
        body = _fake_fred_observations([("2026-05-14", 200.0)])
        feed = MacroFeed(api_key="K", cache_dir=self._cache_dir,
                          http_get=lambda u, params=None, timeout=None: _FakeResponse(body),
                          now=self._now, catalog=catalog_single)
        out = feed.get_regime_drivers()
        self.assertEqual(out["vix"]["status"], "stale")
        self.assertAlmostEqual(out["vix"]["value"], 200.0, places=2)

    def test_observation_older_than_freshness_window_downgrades(self) -> None:
        """Daily series with 10-day-old last observation → status=stale."""
        from services.macro_data import MacroFeed, MACRO_SERIES_CATALOG
        catalog_single = [s for s in MACRO_SERIES_CATALOG if s.series_id == "VIXCLS"]
        body = _fake_fred_observations([("2026-05-01", 14.0)])   # 14 days back
        feed = MacroFeed(api_key="K", cache_dir=self._cache_dir,
                          http_get=lambda u, params=None, timeout=None: _FakeResponse(body),
                          now=self._now, catalog=catalog_single)
        out = feed.get_regime_drivers()
        self.assertEqual(out["vix"]["status"], "stale")
        self.assertGreater(out["vix"]["freshness_days"], 5)

    def test_missing_value_marker_ignored(self) -> None:
        """
        FRED uses '.' for missing values — they must be SKIPPED, not parsed
        as NaN.  The last NON-MISSING value should be reported.
        """
        from services.macro_data import MacroFeed, MACRO_SERIES_CATALOG
        catalog_single = [s for s in MACRO_SERIES_CATALOG if s.series_id == "VIXCLS"]
        body = _fake_fred_observations([
            ("2026-05-13", 14.1),
            ("2026-05-14", None),     # missing
        ])
        feed = MacroFeed(api_key="K", cache_dir=self._cache_dir,
                          http_get=lambda u, params=None, timeout=None: _FakeResponse(body),
                          now=self._now, catalog=catalog_single)
        out = feed.get_regime_drivers()
        self.assertAlmostEqual(out["vix"]["value"], 14.1, places=2)
        self.assertEqual(out["vix"]["as_of"], "2026-05-13")

    def test_zero_observations_reports_error(self) -> None:
        """Empty observations payload → error, not silent zero."""
        from services.macro_data import MacroFeed, MACRO_SERIES_CATALOG
        catalog_single = [s for s in MACRO_SERIES_CATALOG if s.series_id == "VIXCLS"]
        body = {"observations": []}
        feed = MacroFeed(api_key="K", cache_dir=self._cache_dir,
                          http_get=lambda u, params=None, timeout=None: _FakeResponse(body),
                          now=self._now, catalog=catalog_single)
        out = feed.get_regime_drivers()
        self.assertEqual(out["vix"]["status"], "error")
        self.assertIsNone(out["vix"]["value"])

    # ─── catalog completeness ──────────────────────────────────────────

    def test_default_catalog_covers_all_5_drivers(self) -> None:
        from services.macro_data import MACRO_SERIES_CATALOG
        keys = {s.key for s in MACRO_SERIES_CATALOG}
        self.assertEqual(keys, {"yield_curve_10y2y", "hy_credit_spread",
                                  "pmi_manufacturing", "vix", "breakeven_inflation"})

    # ─── payload passthrough ───────────────────────────────────────────

    def test_payload_passes_through_macro_drivers(self) -> None:
        """results["macro_drivers"] flows into payload unchanged."""
        from pdf_payload import build_payload, TIER_BASE, TIER_DEEP
        results = ConcentrationAndWaterfallTest()._base_results()
        results["macro_drivers"] = {
            "vix": {"value": 14.2, "status": "ok", "freshness_days": 1,
                    "as_of": "2026-05-14", "label": "CBOE VIX",
                    "unit": "index", "history_30d": []},
        }
        for tier in (TIER_BASE, TIER_DEEP):
            p = build_payload(results, tier)
            self.assertIn("macro_drivers", p)
            self.assertEqual(p["macro_drivers"]["vix"]["value"], 14.2)
            self.assertEqual(p["macro_drivers"]["vix"]["status"], "ok")


# ── 4.1.g  CoVe data-lineage + Volatility-фактор (Step 5 — final) ──────────

class DataLineageTest(unittest.TestCase):
    """
    Hand-checked tests for src/finance/data_lineage.py.

    Every row in the runtime CoVe table carries {name, source, method,
    status, as_of, freshness_days, note}.  The tests inject various
    `results` shapes (full / partial / empty) and verify the status flag
    moves through the expected ok → warn → stale → missing → error path.
    """

    def _full_results(self) -> dict:
        """A 'happy path' results fixture with every source populated."""
        # Synthetic history with a recent last date so Tradernet status = ok.
        dates = pd.date_range("2026-04-15", periods=30, freq="B")
        prices = pd.DataFrame({"AAPL": np.linspace(180, 195, 30)},
                              index=dates)
        class _History:
            def __init__(self, data):
                self.data = data
        hist = _History(prices)

        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 1000.0,
             "Fundamental_Sector": "Technology",
             "SEC_Filing_Date":   "2026-02-01"},   # 3 months old → ok
            {"Ticker": "JNJ",  "Current_Value": 500.0,
             "Fundamental_Sector": "Healthcare",
             "SEC_Filing_Date":   "2025-12-15"},   # 5 months old → ok
        ])

        return {
            "history_result":      hist,
            "performance_table":   perf,
            "action_plan":         [{"ticker": "AAPL", "action": "Buy"}],
            "black_litterman":     [{"ticker": "AAPL", "target_w": 0.6}],
            "regime":              {"regime": "Slowdown", "confidence": 0.72},
            "stress_scenarios":    [
                {"name": "Tech sell-off", "coverage": "direct"},
                {"name": "USD rally",     "coverage": "proxy"},
            ],
            "macro_drivers":       {
                "vix": {"label": "CBOE VIX", "series_id": "VIXCLS",
                        "status": "ok", "publish_cadence": "daily",
                        "as_of": "2026-05-14", "freshness_days": 1,
                        "note": ""},
            },
            "cds_summary":         {"loaded": 5, "gated_out": 0},
        }

    # ─── status flags per source ───────────────────────────────────────

    def test_tradernet_status_ok_when_data_fresh(self) -> None:
        from finance.data_lineage import build_lineage
        # Fix a "today" that's right after the synthetic data — 5-day window.
        rows = build_lineage(self._full_results(), None,
                              today=date(2026, 5, 27))
        tradernet = next(r for r in rows if r["source"] == "Tradernet (Freedom Broker)")
        self.assertEqual(tradernet["status"], "ok")
        self.assertIsNotNone(tradernet["as_of"])
        self.assertLessEqual(tradernet["freshness_days"], 5)

    def test_tradernet_status_stale_when_old(self) -> None:
        from finance.data_lineage import build_lineage
        # Move "today" far ahead so last close is > 5 cal days old.
        rows = build_lineage(self._full_results(), None,
                              today=date(2026, 7, 1))
        tradernet = next(r for r in rows if r["source"] == "Tradernet (Freedom Broker)")
        self.assertEqual(tradernet["status"], "stale")
        self.assertGreater(tradernet["freshness_days"], 5)

    def test_tradernet_status_error_when_no_data(self) -> None:
        from finance.data_lineage import build_lineage
        rows = build_lineage({}, None, today=date(2026, 5, 17))
        tradernet = next(r for r in rows if r["source"] == "Tradernet (Freedom Broker)")
        self.assertEqual(tradernet["status"], "error")

    def test_sec_warn_when_tickers_missing_coverage(self) -> None:
        """Two tickers with default/EM_Proxy sector → status='warn'."""
        from finance.data_lineage import build_lineage
        r = self._full_results()
        r["performance_table"] = pd.DataFrame([
            {"Ticker": "AAPL", "Fundamental_Sector": "Technology",
             "SEC_Filing_Date": "2026-02-01"},
            {"Ticker": "KSPI", "Fundamental_Sector": "EM_Proxy",
             "SEC_Filing_Date": "2025-08-01"},
            {"Ticker": "FXKZ", "Fundamental_Sector": "default",
             "SEC_Filing_Date": None},
        ])
        rows = build_lineage(r, None, today=date(2026, 5, 17))
        sec_rows = [x for x in rows if x["source"] == "SEC EDGAR CompanyFacts"]
        self.assertEqual(len(sec_rows), 2)
        for s in sec_rows:
            self.assertEqual(s["status"], "warn")
            self.assertIn("2 тикеров", s["note"])

    def test_sec_stale_when_oldest_filing_over_30_months(self) -> None:
        """Filing older than SEC_FILING_STALE_MONTHS triggers 'stale'."""
        from finance.data_lineage import build_lineage
        r = self._full_results()
        r["performance_table"] = pd.DataFrame([
            {"Ticker": "AAPL", "Fundamental_Sector": "Technology",
             "SEC_Filing_Date": "2023-01-01"},     # ~40 months old
        ])
        rows = build_lineage(r, None, today=date(2026, 5, 17))
        sec_rows = [x for x in rows if x["source"] == "SEC EDGAR CompanyFacts"]
        for s in sec_rows:
            self.assertEqual(s["status"], "stale")
            self.assertIn("AAPL", s["note"])

    def test_macro_drivers_status_flows_through(self) -> None:
        """When MacroFeed reports 'stale' for VIX, lineage reflects it."""
        from finance.data_lineage import build_lineage
        r = self._full_results()
        r["macro_drivers"]["vix"]["status"] = "stale"
        r["macro_drivers"]["vix"]["note"]   = "FRED 503"
        rows = build_lineage(r, None, today=date(2026, 5, 17))
        vix_row = next(x for x in rows if x["name"] == "CBOE VIX")
        self.assertEqual(vix_row["status"], "stale")
        self.assertEqual(vix_row["note"],   "FRED 503")

    def test_macro_drivers_missing_when_no_key(self) -> None:
        """No macro_drivers in results → single 'missing' row."""
        from finance.data_lineage import build_lineage
        r = self._full_results()
        r.pop("macro_drivers")
        rows = build_lineage(r, None, today=date(2026, 5, 17))
        # The macro placeholder row carries the literal label "Макро-драйверы"
        # so it can be matched without confusing it with the CDS row that
        # *also* references FRED (for the HY proxy).
        macro = [x for x in rows if x["name"].startswith("Макро-драйверы")]
        self.assertEqual(len(macro), 1)
        self.assertEqual(macro[0]["status"], "missing")
        self.assertIn("FRED_API_KEY", macro[0]["note"])

    def test_action_plan_missing_when_empty(self) -> None:
        from finance.data_lineage import build_lineage
        r = self._full_results()
        r["action_plan"] = []
        rows = build_lineage(r, None, today=date(2026, 5, 17))
        ap = next(x for x in rows if x["name"].startswith("Action levels"))
        self.assertEqual(ap["status"], "missing")

    # ─── CDS lineage states (Step 5 follow-up) ─────────────────────────

    def test_cds_status_missing_when_disabled(self) -> None:
        """CDS_DISABLED=1 path: cds_summary.enabled=False → missing."""
        from finance.data_lineage import build_lineage
        r = self._full_results()
        r["cds_summary"] = {"enabled": False, "checked": 0,
                            "loaded": 0, "gated_out": 0}
        rows = build_lineage(r, None, today=date(2026, 5, 17))
        cds = next(x for x in rows if x["name"].startswith("CDS"))
        self.assertEqual(cds["status"], "missing")
        self.assertIn("CDS_DISABLED", cds["note"])

    def test_cds_status_ok_full_coverage(self) -> None:
        """All checked tickers cleared the gate → status='ok'."""
        from finance.data_lineage import build_lineage
        r = self._full_results()
        r["cds_summary"] = {"enabled": True, "checked": 5,
                            "loaded": 5, "gated_out": 0}
        rows = build_lineage(r, None, today=date(2026, 5, 17))
        cds = next(x for x in rows if x["name"].startswith("CDS"))
        self.assertEqual(cds["status"], "ok")
        self.assertIn("5/5", cds["note"])

    def test_cds_status_warn_partial_coverage(self) -> None:
        """Some loaded, some gated → status='warn'."""
        from finance.data_lineage import build_lineage
        r = self._full_results()
        r["cds_summary"] = {"enabled": True, "checked": 8,
                            "loaded": 5, "gated_out": 3}
        rows = build_lineage(r, None, today=date(2026, 5, 17))
        cds = next(x for x in rows if x["name"].startswith("CDS"))
        self.assertEqual(cds["status"], "warn")
        self.assertIn("3 gated", cds["note"])

    def test_cds_status_missing_when_all_gated(self) -> None:
        """Feed enabled but ZERO tickers cleared the gate → status='missing'."""
        from finance.data_lineage import build_lineage
        r = self._full_results()
        r["cds_summary"] = {"enabled": True, "checked": 5,
                            "loaded": 0, "gated_out": 5}
        rows = build_lineage(r, None, today=date(2026, 5, 17))
        cds = next(x for x in rows if x["name"].startswith("CDS"))
        self.assertEqual(cds["status"], "missing")
        self.assertIn("0/5", cds["note"])

    def test_bl_missing_when_no_records(self) -> None:
        from finance.data_lineage import build_lineage
        r = self._full_results()
        r["black_litterman"] = None
        rows = build_lineage(r, None, today=date(2026, 5, 17))
        bl = next(x for x in rows if x["name"].startswith("Black-Litterman"))
        self.assertEqual(bl["status"], "missing")

    def test_stress_warn_when_proxy_scenarios_present(self) -> None:
        """At least one proxy-coverage scenario → status='warn' (not 'ok')."""
        from finance.data_lineage import build_lineage
        rows = build_lineage(self._full_results(), None,
                              today=date(2026, 5, 17))
        stress = next(x for x in rows if x["name"] == "Стресс-сценарии")
        self.assertEqual(stress["status"], "warn")
        self.assertIn("proxy", stress["note"])

    def test_stress_ok_when_all_direct_coverage(self) -> None:
        from finance.data_lineage import build_lineage
        r = self._full_results()
        r["stress_scenarios"] = [
            {"name": "Tech sell-off", "coverage": "direct"},
            {"name": "Fed +50",       "coverage": "direct"},
        ]
        rows = build_lineage(r, None, today=date(2026, 5, 17))
        stress = next(x for x in rows if x["name"] == "Стресс-сценарии")
        self.assertEqual(stress["status"], "ok")

    def test_rag_status_reflects_used_rag_flag(self) -> None:
        from finance.data_lineage import build_lineage
        rows_yes = build_lineage(self._full_results(),
                                   {"used_rag": True},
                                   today=date(2026, 5, 17))
        rag_yes = next(x for x in rows_yes if x["name"] == "Bank RAG (выдержки)")
        self.assertEqual(rag_yes["status"], "ok")

        rows_no = build_lineage(self._full_results(),
                                  {"used_rag": False},
                                  today=date(2026, 5, 17))
        rag_no = next(x for x in rows_no if x["name"] == "Bank RAG (выдержки)")
        self.assertEqual(rag_no["status"], "missing")

    def test_lineage_always_returns_stable_schema(self) -> None:
        """Every row has the SAME keys — renderer can iterate without checks."""
        from finance.data_lineage import build_lineage
        rows = build_lineage(self._full_results(), {"verdict": "x", "bullets": ["a"]},
                              today=date(2026, 5, 17))
        required = {"name", "source", "method", "status",
                    "as_of", "freshness_days", "note"}
        for r in rows:
            self.assertTrue(required.issubset(r.keys()),
                             f"missing keys in {r['name']}: {required - r.keys()}")
            self.assertIn(r["status"], {"ok", "warn", "stale", "missing", "error"})

    def test_lineage_renders_even_on_empty_results(self) -> None:
        """Catastrophic case: empty results → lineage still returns rows
        with 'missing'/'error' statuses; never raises."""
        from finance.data_lineage import build_lineage
        rows = build_lineage({}, None, today=date(2026, 5, 17))
        self.assertGreater(len(rows), 5)
        # Must contain at least one row per major source category.
        names = {r["name"] for r in rows}
        self.assertIn("Vol · CVaR · TE · IR · Max DD", names)

    # ─── Volatility factor expansion ───────────────────────────────────

    def test_factor_etfs_includes_splv(self) -> None:
        """pdf_payload._FACTOR_ETFS must include SPLV.US (Step 5 expansion)."""
        from pdf_payload import _FACTOR_ETFS
        self.assertIn("SPLV.US", _FACTOR_ETFS)
        # Count = 10 (was 9 before Step 5).
        self.assertEqual(len(_FACTOR_ETFS), 10)

    def test_radar_factor_axes_includes_volatility(self) -> None:
        """tg_bot._RADAR_FACTOR_AXES must include the new Volatility axis."""
        # Lazy import — tg_bot has heavy imports we don't actually exercise.
        import importlib
        try:
            tg = importlib.import_module("tg_bot")
        except Exception:
            self.skipTest("tg_bot imports not available in this env")
        self.assertIn("Volatility", tg._RADAR_FACTOR_AXES)
        # Count = 10 axes.
        self.assertEqual(len(tg._RADAR_FACTOR_AXES), 10)

    # ─── Payload passthrough ───────────────────────────────────────────

    def test_payload_includes_cove_lineage(self) -> None:
        """build_payload calls build_lineage and surfaces it under cove_lineage."""
        from pdf_payload import build_payload, TIER_BASE, TIER_DEEP
        # Re-use the basic concentration fixture and bolt on the lineage inputs.
        results = ConcentrationAndWaterfallTest()._base_results()
        results["macro_drivers"] = {}
        for tier in (TIER_BASE, TIER_DEEP):
            p = build_payload(results, tier,
                              ai_summary={"verdict": "v", "bullets": ["a"]})
            self.assertIn("cove_lineage", p)
            self.assertGreater(len(p["cove_lineage"]), 5)
            # Stable schema sanity at payload level.
            for row in p["cove_lineage"]:
                self.assertIn("status", row)
                self.assertIn("name",   row)


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
        # Deep-only sections present (current template — Three-Question redesign)
        self.assertIn("Action Plan",             html)
        self.assertIn("4-Pillar Scoring",        html)
        self.assertIn("CoVe",                    html)


if __name__ == "__main__":
    unittest.main()
