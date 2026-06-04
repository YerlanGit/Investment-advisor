"""
Phase 8 — regression fixes surfaced by the 2026-06-04 report scan:

  #1 NaN-sector crash in the Credit-pillar guard wiped the ENTIRE 4-pillar
     panel + Black-Litterman (one unmapped ticker → AttributeError →
     score_portfolio returned {}).  Fixed + per-ticker resilience.
  #2 Sector breakdown summed to 118% under leverage → normalise to 100%
     (share of long book).
  #3 AI idea scenarios silently shrank from 4 → 3 when a category emptied
     → backfill from the rule-based catalogue.

Network-free.
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


# ── #1: NaN-sector must not crash scoring ───────────────────────────────────

class NaNSectorResilienceTest(unittest.TestCase):

    def test_guard_handles_nan_sector(self) -> None:
        from finance.scoring_orchestrator import _is_credit_not_applicable
        # The exact failure mode: pandas NaN sector (float).
        self.assertFalse(_is_credit_not_applicable("FFSPC6.1028.AIX", np.nan))
        self.assertFalse(_is_credit_not_applicable("AAPL", float("nan")))
        self.assertFalse(_is_credit_not_applicable("AAPL", None))
        # String sectors still classify correctly.
        self.assertTrue(_is_credit_not_applicable("GLD", "Gold"))

    def test_score_portfolio_survives_nan_sector_row(self) -> None:
        """One unmapped (NaN sector) ticker must NOT empty the whole result."""
        from finance.scoring_orchestrator import score_portfolio
        from finance.regime import RegimeReading
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Fundamental_Sector": "Technology",
             "Euler_Risk_Contribution_Pct": 7.0},
            {"Ticker": "FFSPC6.1028.AIX", "Fundamental_Sector": np.nan,
             "Euler_Risk_Contribution_Pct": 0.5},
            {"Ticker": "GLD", "Fundamental_Sector": "Gold",
             "Euler_Risk_Contribution_Pct": 1.4},
            {"Ticker": "TLT", "Fundamental_Sector": "Bonds",
             "Euler_Risk_Contribution_Pct": 1.4},
        ])
        reg = RegimeReading(regime="Expansion", confidence=0.5,
                            growth_score=0.08, cycle_score=0.04, signals={})
        out = score_portfolio(perf_table=perf, technicals={}, regime=reg,
                              cds_lookup=lambda t: {})
        # All four rows scored — the NaN row no longer crashes the pass.
        self.assertEqual(len(out), 4)
        self.assertIn("FFSPC6.1028.AIX", out)
        self.assertIn("AAPL", out)

    def test_one_bad_row_does_not_wipe_others(self) -> None:
        """Per-ticker resilience: a row that raises is skipped, rest survive."""
        from finance.scoring_orchestrator import score_portfolio
        # Euler_Risk_Contribution_Pct = object that breaks float() → that
        # ticker is skipped, others still score.
        class _Bad:
            def __float__(self): raise ValueError("boom")
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Fundamental_Sector": "Technology",
             "Euler_Risk_Contribution_Pct": 7.0},
            {"Ticker": "BAD", "Fundamental_Sector": "Technology",
             "Euler_Risk_Contribution_Pct": _Bad()},
        ])
        out = score_portfolio(perf_table=perf, technicals={}, regime=None,
                              cds_lookup=None)
        self.assertIn("AAPL", out)            # good row survived
        self.assertNotIn("BAD", out)          # bad row skipped, not fatal


# ── #2: Sector breakdown normalises to 100% ─────────────────────────────────

class SectorNormalisationTest(unittest.TestCase):

    def test_levered_sectors_sum_to_100(self) -> None:
        from pdf_payload import build_payload
        # Longs sum to 118% (leverage); USD short is separate cash, not a sector.
        results = {
            "performance_table": pd.DataFrame([
                {"Ticker": "AAPL", "Current_Value": 1000.0, "Total_Cost": 800.0,
                 "PnL": 200.0, "Return_Pct": 0.25,
                 "Euler_Risk_Contribution_Pct": 10.0},
            ]),
            "total_value": 1000.0,
            "portfolio_metrics": {},
            "benchmark_comparison": {},
            "sector_exposure": {
                "Technology":    0.62,
                "Semiconductors":0.23,
                "Bonds":         0.17,
                "Gold":          0.07,
                "EM_Kazakhstan": 0.05,
                "Silver":        0.04,
            },   # sums to 1.18
        }
        payload = build_payload(results, "base")
        sectors = payload["sectors"]
        total = sum(s["weight_pct"] for s in sectors)
        self.assertAlmostEqual(total, 100.0, places=1)
        # Technology, originally 62% of NAV, becomes 62/118 ≈ 52.5% of book.
        tech = next(s for s in sectors if s["name"] == "Technology")
        self.assertAlmostEqual(tech["weight_pct"], 62.0 / 118.0 * 100, places=1)
        # nav_pct preserves the raw leveraged share for any other consumer.
        self.assertAlmostEqual(tech["nav_pct"], 62.0, places=1)

    def test_pie_legend_matches_bars(self) -> None:
        from pdf_payload import build_payload
        results = {
            "performance_table": pd.DataFrame([
                {"Ticker": "AAPL", "Current_Value": 1000.0, "Total_Cost": 800.0,
                 "PnL": 200.0, "Return_Pct": 0.25,
                 "Euler_Risk_Contribution_Pct": 10.0},
            ]),
            "total_value": 1000.0, "portfolio_metrics": {}, "benchmark_comparison": {},
            "sector_exposure": {"Technology": 0.70, "Bonds": 0.48},  # 1.18
        }
        payload = build_payload(results, "base")
        bars = {s["name"]: s["weight_pct"] for s in payload["sectors"]}
        pie  = {p["name"]: p["weight_pct"] for p in payload["pie_chart_data"]}
        for name in bars:
            self.assertAlmostEqual(bars[name], pie[name], places=2)
        self.assertAlmostEqual(sum(pie.values()), 100.0, places=1)


# ── #3: AI idea scenarios are backfilled when empty ─────────────────────────

class IdeaBackfillTest(unittest.TestCase):

    def test_empty_scenario_is_backfilled(self) -> None:
        from ai_narrative import _backfill_empty_scenarios
        # AI produced only 2 of 4 scenarios with picks.
        stock_picks = {
            "boost_alpha":     {"label": "Рост", "desc": "", "picks": [
                {"ticker": "CRWD", "name": "Crowdstrike", "why": "x", "type": "Stock"}]},
            "rebalance":       {"label": "Ребаланс", "desc": "", "picks": [
                {"ticker": "COST", "name": "Costco", "why": "y", "type": "Stock"}]},
            "protect_capital": {"label": "Защита", "desc": "", "picks": []},
            "regime_play":     {"label": "Режим", "desc": "", "picks": []},
        }
        results = {"performance_table": pd.DataFrame(), "action_plan": []}
        out = _backfill_empty_scenarios(stock_picks, "Expansion", "deep", "", results)
        # All four scenarios now carry picks.
        for key in ("boost_alpha", "rebalance", "protect_capital", "regime_play"):
            self.assertTrue(out[key]["picks"],
                            f"{key} should have picks after backfill")

    def test_non_empty_scenarios_unchanged(self) -> None:
        from ai_narrative import _backfill_empty_scenarios
        original_pick = {"ticker": "PLTR", "name": "Palantir", "why": "z", "type": "Stock"}
        stock_picks = {
            "boost_alpha":     {"label": "Рост", "desc": "", "picks": [original_pick]},
            "rebalance":       {"label": "Ребаланс", "desc": "", "picks": [original_pick]},
            "protect_capital": {"label": "Защита", "desc": "", "picks": [original_pick]},
            "regime_play":     {"label": "Режим", "desc": "", "picks": [original_pick]},
        }
        results = {"performance_table": pd.DataFrame(), "action_plan": []}
        out = _backfill_empty_scenarios(stock_picks, "Expansion", "deep", "", results)
        # Nothing backfilled — original picks preserved.
        self.assertEqual(out["boost_alpha"]["picks"][0]["ticker"], "PLTR")


if __name__ == "__main__":
    unittest.main()
