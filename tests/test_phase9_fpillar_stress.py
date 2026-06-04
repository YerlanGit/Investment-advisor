"""
Phase 9 — fixes for the 2026-06-04 second scan:

  • F-pillar collapse: small cohorts (<5) crashed robust_z → all sector
    members got only `macro_alignment` (visible: 8 stocks at F=+0.4).
    Adds absolute sector benchmarks as a fallback when the cohort is small.
  • Stress disclaimer wiring: `data.portfolio_metrics` was never in the
    payload dict → the Jinja `{% if data.portfolio_metrics ... %}` block
    silently dropped.  Now propagated explicitly.
  • Stress summary for the AI: previous summary lacked per-asset capped
    deltas, so Sonnet kept re-computing β × shock by hand (producing
    "AVGO −42%" against the table's −15.6% Tech sell-off).  We now feed
    `stress_scenarios[].top_assets[].delta_pct_capped` directly so the
    AI cites the engine.
  • Reporting / rebalance_verdict fields surfaced to the AI summary.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── F-pillar absolute benchmarks fallback ───────────────────────────────────

class FPillarFallbackTest(unittest.TestCase):

    def test_small_cohort_falls_back_to_absolute(self) -> None:
        """One Tech holding → cohort=1 → robust_z None → fall back to absolute."""
        from finance.scoring_orchestrator import _sector_z
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Fundamental_Sector": "Technology", "SEC_ROE": 0.30},
        ])
        # Cohort has 1 sample, robust_z returns None.  Absolute benchmark says
        # Technology median ROE = 0.30 → z = 0.
        z = _sector_z(0.30, "Technology", perf, "SEC_ROE")
        self.assertIsNotNone(z, "absolute benchmark must kick in for small cohort")
        self.assertAlmostEqual(z, 0.0, places=2)

    def test_strong_roe_yields_positive_z(self) -> None:
        from finance.scoring_orchestrator import _sector_z
        perf = pd.DataFrame([
            {"Ticker": "ORCL", "Fundamental_Sector": "Technology", "SEC_ROE": 0.60},
        ])
        z = _sector_z(0.60, "Technology", perf, "SEC_ROE")
        # Median 0.30, σ 0.15, ROE 0.60 → z = +2.0 ⇒ +1 contribution to F.
        self.assertGreater(z, 1.5)

    def test_unknown_sector_returns_none(self) -> None:
        """If sector not in the benchmark catalogue → None (no fabrication)."""
        from finance.scoring_orchestrator import _sector_z
        perf = pd.DataFrame([
            {"Ticker": "X", "Fundamental_Sector": "EM_Kazakhstan", "SEC_ROE": 0.15},
        ])
        self.assertIsNone(_sector_z(0.15, "EM_Kazakhstan", perf, "SEC_ROE"))

    def test_relative_path_still_used_when_cohort_large(self) -> None:
        """≥5-name cohort → relative z trumps the absolute fallback."""
        from finance.scoring_orchestrator import _sector_z
        roes = [0.10, 0.12, 0.14, 0.16, 0.18, 0.50]
        perf = pd.DataFrame([
            {"Ticker": f"T{i}", "Fundamental_Sector": "Technology", "SEC_ROE": r}
            for i, r in enumerate(roes)
        ])
        # Absolute benchmark would give roughly +1.3 for ROE=0.50.
        # The relative z over a cohort with median ≈0.15 will be FAR higher
        # — proving the relative path is taken.
        z = _sector_z(0.50, "Technology", perf, "SEC_ROE")
        self.assertGreater(z, 2.5)


# ── Stress disclaimer wiring ────────────────────────────────────────────────

class PayloadPortfolioMetricsTest(unittest.TestCase):

    def test_portfolio_metrics_surfaced(self) -> None:
        from pdf_payload import build_payload
        results = {
            "performance_table": pd.DataFrame([
                {"Ticker": "AAPL", "Current_Value": 1000.0, "Total_Cost": 800.0,
                 "PnL": 200.0, "Return_Pct": 0.25,
                 "Euler_Risk_Contribution_Pct": 10.0},
            ]),
            "total_value": 1000.0,
            "portfolio_metrics": {
                "stress_test_disclaimer": "TEST DISCLAIMER TEXT",
                "reporting_currency": "USD",
                "risk_free_rate_annual": 0.045,
                "Sharpe_Ratio": 1.4,
            },
            "benchmark_comparison": {},
            "sector_exposure": {"Technology": 1.0},
        }
        payload = build_payload(results, "base")
        # The whole portfolio_metrics dict must be propagated, not flattened.
        self.assertIn("portfolio_metrics", payload)
        pm = payload["portfolio_metrics"]
        self.assertEqual(pm.get("stress_test_disclaimer"), "TEST DISCLAIMER TEXT")
        self.assertEqual(pm.get("reporting_currency"), "USD")


# ── AI summary: stress scenarios with capped deltas + reporting + verdict ───

class AISummaryStressTest(unittest.TestCase):

    def test_summary_includes_capped_per_asset_stress(self) -> None:
        from ai_narrative import _summarise_for_prompt
        results = {
            "portfolio_metrics": {
                "Sharpe_Ratio": 1.4, "Total_Volatility_Ann": 0.241,
                "reporting_currency": "USD", "risk_free_rate_annual": 0.045,
                "CVaR_95_Bootstrap": {},
            },
            "total_value": 12_400.0,
            "performance_table": pd.DataFrame(),
            "stress_scenarios": [
                {"name": "Tech sell-off", "port_pct": -0.156,
                 "max_dd_pct": -0.276, "recovery_months": 41.4,
                 "by_asset": [
                     {"ticker": "AVGO", "weight_pct": 11.8,
                      "asset_delta_pct": -31.6, "asset_delta_raw": -42.2},
                     {"ticker": "NVDA", "weight_pct": 7.2,
                      "asset_delta_pct": -32.4, "asset_delta_raw": -46.4},
                 ]},
            ],
            "expected_effect": {
                "verdict": {"kind": "tradeoff",
                            "headline": "compromise headline", "worsened": ["max_trc"]},
            },
        }
        out = _summarise_for_prompt(results)
        self.assertIn("stress_scenarios", out)
        scenarios = out["stress_scenarios"]
        self.assertEqual(len(scenarios), 1)
        # Capped (not raw) value reaches the AI.
        avgo = scenarios[0]["top_assets"][0]
        self.assertEqual(avgo["ticker"], "AVGO")
        self.assertEqual(avgo["delta_pct_capped"], -31.6)
        self.assertEqual(avgo["delta_pct_raw"],    -42.2)
        # Reporting metadata threaded through.
        self.assertEqual(out["reporting"]["currency"], "USD")
        self.assertAlmostEqual(out["reporting"]["rfr_ann"], 0.045)
        # Rebalance verdict echoed so ai_effect_comment can cite it verbatim.
        self.assertEqual(out["rebalance_verdict"]["kind"], "tradeoff")


if __name__ == "__main__":
    unittest.main()
