"""
Phase 10 — fixes from the 2026-06-04 Phase-9 report scan:

  #1 GLD/SLV showed F = −0.4 (regime penalty leaking into the
     Fundamentals pillar — commodities have no financial statements).
     Now F is marked not-applicable (em-dash) for commodities + sovereign
     bonds, mirroring the Credit-pillar guard.
  #2 AI model-attribution chip rendered on every AI comment block
     (template CSS — checked via render smoke test elsewhere).
  QC double-label "Валюта отчёта: Валюта: USD" → de-duplicated.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FundamentalsApplicableGuardTest(unittest.TestCase):

    def _score(self, ticker, sector, **sec):
        from finance.scoring_orchestrator import score_portfolio
        from finance.regime import RegimeReading
        row = {"Ticker": ticker, "Fundamental_Sector": sector,
               "Euler_Risk_Contribution_Pct": 5.0, **sec}
        reg = RegimeReading(regime="Expansion", confidence=0.74,
                            growth_score=0.08, cycle_score=0.04, signals={})
        out = score_portfolio(perf_table=pd.DataFrame([row]),
                              technicals={}, regime=reg, cds_lookup=lambda t: {})
        return out[ticker]

    def test_commodity_f_is_not_applicable(self) -> None:
        gld = self._score("GLD", "Gold")
        slv = self._score("SLV", "Silver")
        self.assertFalse(gld.fundamentals_applicable)
        self.assertFalse(slv.fundamentals_applicable)
        # F is neutral 0.0, NOT the −0.4 regime penalty that used to leak in.
        self.assertEqual(gld.fundamentals, 0.0)
        self.assertEqual(slv.fundamentals, 0.0)

    def test_sovereign_bond_f_is_not_applicable(self) -> None:
        tlt = self._score("TLT", "Bonds")
        self.assertFalse(tlt.fundamentals_applicable)
        self.assertEqual(tlt.fundamentals, 0.0)

    def test_equity_keeps_real_fundamentals(self) -> None:
        orcl = self._score("ORCL", "Technology", SEC_ROE=0.608, SEC_Op_Margin=0.30)
        self.assertTrue(orcl.fundamentals_applicable)
        # Strong ROE → positive F via the absolute sector benchmark fallback.
        self.assertGreater(orcl.fundamentals, 0.5)


class PayloadFundamentalsNaTest(unittest.TestCase):

    def test_score_breakdown_marks_fundamentals_na(self) -> None:
        from pdf_payload import build_payload
        results = {
            "performance_table": pd.DataFrame([
                {"Ticker": "GLD", "Current_Value": 1000.0, "Total_Cost": 900.0,
                 "PnL": 100.0, "Return_Pct": 0.11,
                 "Euler_Risk_Contribution_Pct": 5.0},
            ]),
            "total_value": 1000.0,
            "portfolio_metrics": {},
            "benchmark_comparison": {},
            "sector_exposure": {"Gold": 1.0},
            "asset_scores": {
                "GLD": {"fundamentals": 0.0, "valuations": 0.0, "technicals": 0.5,
                        "credit": 0.0, "total": 0.5, "action": "Hold",
                        "hotspot": False, "credit_applicable": False,
                        "fundamentals_applicable": False},
            },
        }
        payload = build_payload(results, "deep")
        sb = payload.get("score_breakdown") or []
        self.assertTrue(sb)
        gld = next(r for r in sb if r["ticker"] == "GLD")
        self.assertEqual(gld["fundamentals"], "—")
        self.assertTrue(gld["fundamentals_na"])
        self.assertEqual(gld["credit"], "—")


class CurrencyLabelTest(unittest.TestCase):

    def test_currency_qc_not_double_labelled(self) -> None:
        from pdf_payload import _build_integrity_checks
        checks = _build_integrity_checks(
            results={"portfolio_metrics": {
                "reporting_currency": "USD",
                "risk_free_rate_annual": 0.045,
                "risk_free_rate_source": "default[USD]=0.045",
            }},
            ai_summary={}, data_quality={}, return_series_coverage={},
        )
        cur = next((c for c in checks if c["label"] == "Валюта отчёта"), None)
        self.assertIsNotNone(cur)
        # Detail must NOT start with another "Валюта:" (the old double-label).
        self.assertNotIn("Валюта:", cur["detail"])
        self.assertIn("USD", cur["detail"])
        self.assertIn("RFR", cur["detail"])


if __name__ == "__main__":
    unittest.main()
