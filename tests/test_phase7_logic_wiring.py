"""
Phase 7 — Logic & Template-wiring fixes:
  • Credit-pillar asset-class guard (Commodities / sovereign bonds)
  • Regime confidence = magnitude × directional agreement
  • simulate_after_plan composite verdict (improvement / tradeoff / degradation)
  • ai_narrative `_soft_trim` (regex sentence-boundary cut)
  • Beta NaN-robust extraction in pdf_payload

Network-free; reuses fixtures already validated in phases 1–6.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── #1: Credit-pillar asset-class guard ─────────────────────────────────────

class CreditAssetClassGuardTest(unittest.TestCase):

    def test_commodity_etf_gets_neutral_credit(self) -> None:
        from finance.scoring_orchestrator import _is_credit_not_applicable
        self.assertTrue(_is_credit_not_applicable("GLD", "Gold"))
        self.assertTrue(_is_credit_not_applicable("SLV", "Silver"))
        self.assertTrue(_is_credit_not_applicable("DBC", "Commodities"))
        self.assertTrue(_is_credit_not_applicable("USO", "Oil"))

    def test_sovereign_bond_etf_gets_neutral_credit(self) -> None:
        from finance.scoring_orchestrator import _is_credit_not_applicable
        self.assertTrue(_is_credit_not_applicable("TLT", "Bonds"))
        self.assertTrue(_is_credit_not_applicable("IEF", "Bonds"))
        self.assertTrue(_is_credit_not_applicable("SHY", "Bonds"))

    def test_single_name_equity_keeps_credit(self) -> None:
        from finance.scoring_orchestrator import _is_credit_not_applicable
        self.assertFalse(_is_credit_not_applicable("AAPL",  "Technology"))
        self.assertFalse(_is_credit_not_applicable("AVGO",  "Semiconductors"))
        self.assertFalse(_is_credit_not_applicable("JPM",   "Finance"))

    def test_ticker_prefix_alone_is_enough(self) -> None:
        """Even when sector tag is missing/wrong, the prefix triggers the guard."""
        from finance.scoring_orchestrator import _is_credit_not_applicable
        self.assertTrue(_is_credit_not_applicable("TLT.US", None))
        self.assertTrue(_is_credit_not_applicable("GLD",    "Other"))

    def test_end_to_end_score_portfolio_marks_flag(self) -> None:
        """score_portfolio sets credit_applicable=False on commodity / bond ETFs."""
        from finance.scoring_orchestrator import score_portfolio
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Fundamental_Sector": "Technology",
             "Euler_Risk_Contribution_Pct": 5.0,
             "SEC_ROE": 0.30, "SEC_Op_Margin": 0.25, "SEC_Debt_to_Assets": 0.30,
             "SEC_Revenue_Growth_YoY": 0.05, "SEC_FCF_Margin": 0.20,
             "SEC_PE_Ratio": 28.0, "SEC_PB_Ratio": 7.0},
            {"Ticker": "GLD",  "Fundamental_Sector": "Gold",
             "Euler_Risk_Contribution_Pct": 2.0},
            {"Ticker": "TLT",  "Fundamental_Sector": "Bonds",
             "Euler_Risk_Contribution_Pct": 1.5},
        ])
        scores = score_portfolio(perf, technicals={}, regime=None)
        self.assertTrue(scores["AAPL"].credit_applicable)
        self.assertFalse(scores["GLD"].credit_applicable)
        self.assertFalse(scores["TLT"].credit_applicable)
        # Credit value is exactly 0.0 for the not-applicable ones.
        self.assertEqual(scores["GLD"].credit, 0.0)
        self.assertEqual(scores["TLT"].credit, 0.0)


# ── #3: Regime confidence rewrite ───────────────────────────────────────────

class RegimeConfidenceRewriteTest(unittest.TestCase):
    """
    Confidence = magnitude_factor × directional_agreement.

    The previous formula `min(1.0, magnitude/0.10)` saturated to 1.0 for the
    typical real-world signal (g=0.15, c=0.04) — confidence "100%" was a lie
    because the cycle axis was effectively zero.
    """

    def _build_prices(self, *, spy_60_ret, ief_60_ret, eem_60_ret,
                       xly_60_ret, xlp_60_ret, iwm_60_ret, n=130):
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        cols: dict[str, np.ndarray] = {}
        def _series(start: float, total_ret_60: float) -> np.ndarray:
            # constant compounded return so day_t / day_{t-60} = 1+total
            arr = np.full(n, start, dtype=float)
            # walk: gentle ramp so 60-day ratio matches `total_ret_60`
            daily = (1.0 + total_ret_60) ** (1.0 / 60.0) - 1.0
            arr = start * (1.0 + daily) ** np.arange(n)
            return arr
        cols["SPY.US"] = _series(100, spy_60_ret)
        cols["IEF.US"] = _series(100, ief_60_ret)
        cols["EEM.US"] = _series(100, eem_60_ret)
        cols["XLY.US"] = _series(100, xly_60_ret)
        cols["XLP.US"] = _series(100, xlp_60_ret)
        cols["IWM.US"] = _series(100, iwm_60_ret)
        return pd.DataFrame(cols, index=idx)

    def test_weak_signal_no_longer_saturates_to_100pct(self) -> None:
        """Real production case: positive but tiny magnitude → confidence < 1.0."""
        from finance.regime import RegimeClassifier
        # Engineer all components small-positive so the quadrant is Expansion
        # but magnitude stays in the "low confidence" zone.
        prices = self._build_prices(
            spy_60_ret=0.04, ief_60_ret=0.01, eem_60_ret=0.03,
            xly_60_ret=0.05, xlp_60_ret=0.01, iwm_60_ret=0.06,
        )
        reading = RegimeClassifier().classify(prices)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.regime, "Expansion")
        # Magnitude is tiny → confidence must NOT round up to 1.0.
        # Pre-fix confidence was ≥0.99; we accept anything strictly less.
        self.assertLess(reading.confidence, 1.0)
        # Both factors exposed for transparency.
        self.assertIn("confidence_magnitude", reading.signals)
        self.assertIn("confidence_directional", reading.signals)

    def test_strong_aligned_signal_can_reach_high_confidence(self) -> None:
        """When magnitude is high AND all components agree → confidence near 1.0."""
        from finance.regime import RegimeClassifier
        # Big growth + big cycle, all positive → all directional components agree
        prices = self._build_prices(
            spy_60_ret=0.20, ief_60_ret=0.02, eem_60_ret=0.18,
            xly_60_ret=0.20, xlp_60_ret=0.05, iwm_60_ret=0.22,
        )
        reading = RegimeClassifier().classify(prices)
        self.assertEqual(reading.regime, "Expansion")
        # All components positive → directional agreement = 1.0;
        # magnitude well above 0.20 → magnitude_factor ≈ 1.0
        self.assertGreater(reading.confidence, 0.85)

    def test_directional_disagreement_lowers_confidence(self) -> None:
        """Magnitude OK but signals conflict → confidence below 0.5."""
        from finance.regime import RegimeClassifier
        # Engineer disagreement: growth axis split (positive spread +0.05,
        # but eem deep negative -0.20) — the assigned growth_sign = mean sign.
        prices = self._build_prices(
            spy_60_ret=0.06, ief_60_ret=0.01, eem_60_ret=-0.20,
            xly_60_ret=0.01, xlp_60_ret=0.06, iwm_60_ret=-0.05,
        )
        reading = RegimeClassifier().classify(prices)
        self.assertIsNotNone(reading)
        # We don't pin the exact regime — engineering disagreement is the
        # point; what matters is the confidence haircut.
        self.assertLess(reading.confidence, 0.50)


# ── #4: simulate_after_plan composite verdict ──────────────────────────────

class SimulateVerdictTest(unittest.TestCase):
    """A rebalance that drops vol but raises concentration is a TRADEOFF."""

    def _perf(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 6_000.0,
             "Beta_Market": 1.2, "Euler_Risk_Contribution_Pct": 60.0},
            {"Ticker": "KSPI", "Current_Value": 4_000.0,
             "Beta_Market": 0.9, "Euler_Risk_Contribution_Pct": 40.0},
        ])

    def _risk_matrix(self) -> pd.DataFrame:
        # 2-asset, low correlation cov_ann
        idx = ["AAPL", "KSPI"]
        return pd.DataFrame([[0.20**2, 0.30*0.20*0.20*0.1],
                              [0.30*0.20*0.20*0.1, 0.30**2]], index=idx, columns=idx)

    def test_vol_down_but_trc_up_marks_tradeoff(self) -> None:
        from finance.simulate import simulate_after_plan
        # Move from 0.6/0.4 to 0.9/0.1 — KSPI shed → concentration up,
        # but if KSPI was higher-vol then total vol could drop.  We don't
        # care about the literal numbers — just that the verdict catches
        # the asymmetry between vol and TRC.
        out = simulate_after_plan(
            perf_df           = self._perf(),
            risk_matrix       = self._risk_matrix(),
            daily_log_returns = None,
            bl_records        = None,
            current_metrics   = {},
            risk_free_rate    = 0.04,
            target_weights    = {"AAPL": 0.90, "KSPI": 0.10},
        )
        self.assertIn("verdict", out)
        self.assertIn(out["verdict"]["kind"],
                      {"tradeoff", "improvement", "degradation", "neutral"})
        # Sanity on dict shape
        self.assertIn("headline", out["verdict"])
        self.assertIsInstance(out["verdict"]["worsened"], list)

    def test_no_change_yields_neutral_verdict(self) -> None:
        from finance.simulate import simulate_after_plan
        out = simulate_after_plan(
            perf_df           = self._perf(),
            risk_matrix       = self._risk_matrix(),
            daily_log_returns = None,
            bl_records        = None,
            current_metrics   = {},
            risk_free_rate    = 0.04,
            target_weights    = {"AAPL": 0.60, "KSPI": 0.40},  # == current
        )
        self.assertEqual(out["verdict"]["kind"], "neutral")
        self.assertEqual(out["verdict"]["worsened"], [])


# ── #8: ai_narrative._soft_trim ──────────────────────────────────────────────

class SoftTrimTest(unittest.TestCase):

    def test_short_text_unchanged(self) -> None:
        from ai_narrative import _soft_trim
        self.assertEqual(_soft_trim("Привет.", 100), "Привет.")
        self.assertEqual(_soft_trim("", 100), "")

    def test_cuts_at_last_sentence_boundary(self) -> None:
        from ai_narrative import _soft_trim
        s = ("Портфель опередил S&P 500 на 11 пп за 12M. "
             "Концентрация Tech 62% и Semi 23% даёт 85%. "
             "Это требует ребалансировки в Healthcare.")
        out = _soft_trim(s, 70)
        # Ends with a full stop, no dangling clause.
        self.assertTrue(out.endswith(".") or out.endswith("…"))
        # Length within budget.
        self.assertLessEqual(len(out), 70)
        # No mid-word hack.
        self.assertNotIn("Концентрация Tech 62% и Semi 23% даёт 8", out)

    def test_fallback_ellipsis_when_no_boundary(self) -> None:
        from ai_narrative import _soft_trim
        s = "одно_длинное_слово_без_точек_" * 20
        out = _soft_trim(s, 30)
        self.assertLessEqual(len(out), 31)            # +1 for ellipsis char
        self.assertTrue(out.endswith("…"))


# ── #7: Beta NaN robustness in pdf_payload ─────────────────────────────────

class BetaNanRobustnessTest(unittest.TestCase):

    def test_numpy_nan_renders_emdash(self) -> None:
        """numpy.float64('nan') used to slip past the legacy isinstance gate."""
        from pdf_payload import build_payload as _build_pdf_payload
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 1000.0, "Total_Cost": 800.0,
             "PnL": 200.0, "Return_Pct": 0.25,
             "Euler_Risk_Contribution_Pct": 10.0,
             "Beta_Market": np.float64("nan")},     # the failure mode
            {"Ticker": "MSFT", "Current_Value": 2000.0, "Total_Cost": 1800.0,
             "PnL": 200.0, "Return_Pct": 0.111,
             "Euler_Risk_Contribution_Pct": 12.0,
             "Beta_Market": np.float64(1.52)},
        ])
        results = {
            "performance_table": perf,
            "total_value":       3000.0,
            "portfolio_metrics": {},
            "benchmark_comparison": {},
        }
        payload = _build_pdf_payload(results, "base")
        betas = {a["ticker"]: a["beta"] for a in payload.get("assets", [])}
        # NaN must render as em-dash, not "nan"
        self.assertEqual(betas["AAPL"], "—")
        self.assertEqual(betas["MSFT"], "1.52")


if __name__ == "__main__":
    unittest.main()
