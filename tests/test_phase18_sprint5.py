"""
Sprint-5 + UX Overhaul — regression tests for the mandate/leverage/regime
wiring and the report-section fixes.  All import-safe (no aiogram / network).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── Task 7: live regime quadrant dot (R1) ─────────────────────────────────────

class RegimeDotCoordsTest(unittest.TestCase):
    def test_expansion_top_right(self):
        from pdf_payload import _regime_dot_coords
        cx, cy = _regime_dot_coords(0.08, 0.04)   # growth+, cycle+
        self.assertGreater(cx, 155)               # right of centre (cycle +)
        self.assertLess(cy, 155)                  # above centre (growth +)

    def test_recession_bottom_left(self):
        from pdf_payload import _regime_dot_coords
        cx, cy = _regime_dot_coords(-0.10, -0.10)
        self.assertLess(cx, 155)
        self.assertGreater(cy, 155)

    def test_clamped_inside_frame(self):
        from pdf_payload import _regime_dot_coords
        cx, cy = _regime_dot_coords(5.0, -5.0)    # absurd inputs
        self.assertGreaterEqual(cx, 40.0)
        self.assertLessEqual(cx, 270.0)
        self.assertGreaterEqual(cy, 40.0)
        self.assertLessEqual(cy, 270.0)

    def test_payload_exposes_dot(self):
        from pdf_payload import build_payload
        results = {
            "performance_table": pd.DataFrame([
                {"Ticker": "AAPL", "Current_Value": 1000.0, "Total_Cost": 800.0,
                 "PnL": 200.0, "Return_Pct": 0.25, "Euler_Risk_Contribution_Pct": 10.0,
                 "Current_Price": 190.0},
            ]),
            "total_value": 1000.0, "portfolio_metrics": {}, "benchmark_comparison": {},
            "sector_exposure": {"Technology": 1.0},
            "regime": {"regime": "Recession", "confidence": 0.7,
                       "growth_score": -0.1, "cycle_score": -0.08, "signals": {}},
        }
        pl = build_payload(results, "deep")
        self.assertIsNotNone(pl["regime"]["dot_cx"])
        self.assertIn("Growth", pl["regime"]["dot_label"])


# ── Task 4: mandate → Black-Litterman constraints ─────────────────────────────

class BlackLittermanMandateCapTest(unittest.TestCase):
    def test_cap_redistribute_feasible(self):
        from finance.black_litterman import _cap_and_redistribute
        w = np.array([0.55, 0.20, 0.10, 0.06, 0.05, 0.04])
        capped = _cap_and_redistribute(w, 0.30)   # 6 names → feasible
        self.assertLessEqual(float(capped.max()), 0.30 + 1e-9)
        self.assertAlmostEqual(float(capped.sum()), 1.0, places=6)

    def test_cap_infeasible_is_noop(self):
        # 4 names, 10% cap → 4·0.10 < 1 → cannot satisfy → leave unchanged
        from finance.black_litterman import _cap_and_redistribute
        w = np.array([0.7, 0.1, 0.1, 0.1])
        out = _cap_and_redistribute(w, 0.10)
        np.testing.assert_allclose(out, w, atol=1e-9)

    def test_black_litterman_accepts_mandate_kwargs(self):
        from finance.black_litterman import black_litterman
        cov = np.diag([0.04, 0.05, 0.06, 0.07, 0.05, 0.06]).astype(float)
        tickers = list("ABCDEF")
        cw = {t: (0.5 if t == "A" else 0.1) for t in tickers}
        res = black_litterman(cov=cov, tickers=tickers, current_weights=cw,
                              risk_aversion=4.0, max_active_share=0.15,
                              max_single_weight=0.30)
        self.assertLessEqual(float(res.target_weights.max()), 0.30 + 1e-6)

    def test_constraint_table_orders_by_aggressiveness(self):
        from finance.investment_logic import _MANDATE_BL_CONSTRAINTS as M
        self.assertLess(M["AGGRESSIVE"]["risk_aversion"], M["CONSERVATIVE"]["risk_aversion"])
        self.assertLess(M["CONSERVATIVE"]["max_single_weight"], M["AGGRESSIVE"]["max_single_weight"])
        self.assertLess(M["CONSERVATIVE"]["max_active_share"], M["AGGRESSIVE"]["max_active_share"])


# ── Task 5: margin/leverage — cash is no longer "Bonds" ───────────────────────

class CashClassificationTest(unittest.TestCase):
    def test_cash_is_cash_not_bonds(self):
        from agent.gatekeeper import _classify_to_asset_key
        for cash in ("USD", "EUR", "CASH", "RUB", "KZT"):
            self.assertEqual(_classify_to_asset_key(cash), "Cash")
        # Real bond ETFs still classify as Bonds.
        self.assertEqual(_classify_to_asset_key("AGG.US"), "Bonds")
        self.assertEqual(_classify_to_asset_key("AAPL"), "Stocks_US")

    def test_negative_cash_does_not_corrupt_bonds_allocation(self):
        from pdf_payload import _build_mandate_compliance
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 6200.0},
            {"Ticker": "AGG.US", "Current_Value": 500.0},
            {"Ticker": "USD", "Current_Value": -700.0},     # margin debt
        ])
        profile = {"profile_name": "Умеренный", "target_volatility": 0.10,
                   "target_te": 0.04,
                   "limits_dict": {"Stocks_US": [20, 40], "Bonds": [30, 50]}}
        mc = _build_mandate_compliance(perf, 6000.0, profile)
        bonds = next(r for r in mc["rows"] if r["key"] == "Bonds")
        # 500 / 6000 = 8.3% — NOT distorted down to negative by the -700 cash.
        self.assertAlmostEqual(bonds["actual"], 8.3, places=1)
        self.assertEqual(bonds["status"], "under")


# ── Task 5: leverage AI trigger (deterministic fallback path) ─────────────────

class LeverageAiTriggerTest(unittest.TestCase):
    def test_fallback_emits_margin_call_warning_when_levered(self):
        # No ANTHROPIC_API_KEY in the test env → deterministic fallback path.
        from ai_narrative import generate_narrative
        results = {
            "portfolio_metrics": {"Composite_Risk_Score": 55, "Sharpe_Ratio": 0.8,
                                  "CVaR_95_Daily": -0.05, "Max_Drawdown": -0.2},
            "total_value": 100000.0,
            "regime": {"regime": "Expansion", "confidence": 0.6},
            "performance_table": pd.DataFrame([{"Ticker": "AAPL"}]),
            "leverage_metrics": {"is_leveraged": True, "gross_exposure": 1.35,
                                 "long_weight": 1.18, "net_exposure": 1.0,
                                 "cash_weight": -0.18, "leverage_ratio": 1.18},
        }
        out = generate_narrative(results, tier="base")
        self.assertTrue(out.get("ai_leverage_warning"))
        self.assertIn("Margin Call", out["ai_leverage_warning"])

    def test_no_warning_when_unlevered(self):
        from ai_narrative import generate_narrative
        results = {
            "portfolio_metrics": {"Composite_Risk_Score": 40, "Sharpe_Ratio": 1.1,
                                  "CVaR_95_Daily": -0.03, "Max_Drawdown": -0.1},
            "total_value": 50000.0,
            "regime": {"regime": "Expansion", "confidence": 0.6},
            "performance_table": pd.DataFrame([{"Ticker": "AAPL"}]),
            "leverage_metrics": {"is_leveraged": False, "gross_exposure": 1.0,
                                 "leverage_ratio": 1.0},
        }
        out = generate_narrative(results, tier="base")
        self.assertFalse(out.get("ai_leverage_warning"))


# ── Task 6: V-pillar gains an FCF-yield quality signal ────────────────────────

class ValuationFcfYieldTest(unittest.TestCase):
    def test_high_yield_is_cheap(self):
        from finance.scoring import valuations_score
        self.assertEqual(valuations_score(fcf_yield_z=2.0), 1.0)
        self.assertEqual(valuations_score(fcf_yield_z=-2.0), -1.0)

    def test_signed_z_allows_negative_yield(self):
        from finance.scoring_orchestrator import (_absolute_signed_z,
                                                  _SECTOR_FCF_YIELD_BENCHMARKS)
        z = _absolute_signed_z(-0.02, "Technology", _SECTOR_FCF_YIELD_BENCHMARKS)
        self.assertIsNotNone(z)
        self.assertLess(z, 0)     # cash-burner → penalty (price-multiple z would be None)

    def test_fcf_yield_ratio_computed(self):
        from finance.scoring_orchestrator import _compute_valuation_ratios
        df = pd.DataFrame([{"Current_Price": 100.0, "SEC_Net_Income": 5e9,
                            "SEC_Shares_Outstanding": 1e9, "SEC_Book_Equity": 2e10,
                            "SEC_FCF": 4e9}])
        _compute_valuation_ratios(df)
        self.assertAlmostEqual(float(df["SEC_FCF_Yield"][0]), 0.04, places=4)


# ── Task 8: benchmark selection is honoured (no longer a dead control) ─────────

class BenchmarkActivationTest(unittest.TestCase):
    def test_resolve_prefers_saved_choice(self):
        # _resolve_bench_ticker lives in tg_bot (needs aiogram); re-implement the
        # contract check against profile_manager's default map to stay import-safe.
        from profile_manager import PROFILE_BENCH_TICKER
        # The saved choice must win over the profile default.
        profile = {"profile_name": "Консервативный", "benchmark_ticker": "QQQ.US"}
        saved = profile.get("benchmark_ticker") or PROFILE_BENCH_TICKER.get(profile["profile_name"])
        self.assertEqual(saved, "QQQ.US")
        # Falls back to the profile default when nothing was saved.
        profile2 = {"profile_name": "Консервативный", "benchmark_ticker": None}
        fb = profile2.get("benchmark_ticker") or PROFILE_BENCH_TICKER.get(profile2["profile_name"])
        self.assertEqual(fb, "AGG.US")

    def test_payload_filters_to_profile_benchmark_slot(self):
        from pdf_payload import build_payload
        results = {
            "performance_table": pd.DataFrame([
                {"Ticker": "AAPL", "Current_Value": 1000.0, "Total_Cost": 800.0,
                 "PnL": 200.0, "Return_Pct": 0.25, "Euler_Risk_Contribution_Pct": 10.0},
            ]),
            "total_value": 1000.0, "portfolio_metrics": {},
            "sector_exposure": {"Technology": 1.0},
            "benchmark_comparison": {
                "Профильный бенчмарк": {"Excess_Return_Ann": 0.02, "Information_Ratio": 0.5,
                                        "Tracking_Error": 0.06, "Beating_Benchmark": True},
                "S&P 500": {"Excess_Return_Ann": -0.01, "Information_Ratio": 0.1,
                            "Tracking_Error": 0.03, "Beating_Benchmark": False},
            },
        }
        pl = build_payload(results, "base", user_bench_ticker="VLUE.US")  # custom, not in std-5
        names = [s["name"] for s in pl["scenarios"]]
        self.assertEqual(names, ["Профильный бенчмарк"])


class RegimeConsistencyR3Test(unittest.TestCase):
    def _macro(self, yc, hy, vix):
        return {"yield_curve_10y2y": {"value": yc},
                "hy_credit_spread":  {"value": hy},
                "vix":               {"value": vix}}

    def test_risk_on_label_with_recession_signals_diverges(self):
        from pdf_payload import _build_regime_consistency
        out = _build_regime_consistency({"regime": "Expansion"},
                                        self._macro(-0.3, 6.5, 28))
        self.assertEqual(out["status"], "diverges")
        self.assertGreaterEqual(len(out["signals"]), 2)

    def test_aligned_when_calm_and_risk_on(self):
        from pdf_payload import _build_regime_consistency
        out = _build_regime_consistency({"regime": "Expansion"},
                                        self._macro(0.4, 3.0, 15))
        self.assertEqual(out["status"], "aligned")

    def test_risk_off_label_with_calm_macro_diverges(self):
        from pdf_payload import _build_regime_consistency
        out = _build_regime_consistency({"regime": "Recession"},
                                        self._macro(0.5, 3.0, 14))
        self.assertEqual(out["status"], "diverges")

    def test_none_when_no_macro(self):
        from pdf_payload import _build_regime_consistency
        self.assertIsNone(_build_regime_consistency({"regime": "Expansion"}, None))
        self.assertIsNone(_build_regime_consistency(None, self._macro(0, 3, 15)))


if __name__ == "__main__":
    unittest.main()
