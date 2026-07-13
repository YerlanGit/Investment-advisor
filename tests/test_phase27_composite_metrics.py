"""
Phase 27 regression tests — Sharpe/return correctness for books with young
listings and leveraged ETPs (F-20 / F-21 / F-22 / F-23):

  F-22  Realized metrics (Sharpe numerator, Annualised_Return, VaR/CVaR,
        MaxDD) are computed on the MASKED COMPOSITE portfolio series over the
        full price panel — not on the intersection window that shrinks to the
        youngest kept listing.  On full panels the legacy path is kept
        bit-for-bit (composite adds no history → no numeric churn).

  F-21  KPI sparklines consume the engine's precomputed composite series
        (results["port_log_returns"]) instead of rebuilding a row-dropna'd
        joint window that one <60-day listing (SPCX) nulled entirely.

  F-23  Daily-reset leveraged ETPs (CONL/XNDU/TQQQ…): the forward β·μ panel
        retains the NEGATIVE part of the Ridge intercept — the contractual
        variance drag −½k(k−1)σ² — so forecasts no longer overstate them.
        Ordinary names / positive alphas pass through untouched.

  F-20  Action-plan rows for names the sparse-guard excluded carry an
        explicit reason («вне модели: история < 60 торг. дней») instead of a
        bare qty-less SELL.

All tests are deterministic, synthetic and offline.
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

os.environ.setdefault("KZ_RFR_ANNUAL", "0.14")
os.environ["CDS_DISABLED"] = "1"


def _staggered_engine_data(n_days: int = 500, young_days: int = 120):
    """Factor ETFs with full history + one elder asset (full) + one young
    listing (last `young_days` only).  Mirrors the live CONL/FFSPC6 shape."""
    from finance.investment_logic import MAC3RiskEngine

    engine = MAC3RiskEngine()
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    data = {}
    for i, etf in enumerate(engine.factor_tickers.values()):
        r = np.random.default_rng(100 + i).normal(0.0004, 0.010, n_days)
        data[etf] = 100.0 * np.exp(np.cumsum(r))
    data["ELDER.US"] = 100.0 * np.exp(np.cumsum(
        np.random.default_rng(7).normal(0.0005, 0.012, n_days)))
    young = np.full(n_days, np.nan)
    young[-young_days:] = 30.0 * np.exp(np.cumsum(
        np.random.default_rng(8).normal(0.0012, 0.02, young_days)))
    data["YOUNG.US"] = young
    return engine, pd.DataFrame(data, index=idx)


# ═══════════════ F-22: composite basis for realized metrics ═════════════════

class CompositeRealizedBasisTest(unittest.TestCase):

    def test_staggered_book_keeps_elder_history(self) -> None:
        engine, df = _staggered_engine_data()
        _, _, metrics = engine.calculate_structural_risk(
            df, ["ELDER.US", "YOUNG.US"],
            {"ELDER.US": 0.6, "YOUNG.US": 0.4})
        # Regression window honestly shrinks to the young listing…
        self.assertLess(engine._last_regression_nobs, 252)
        # …but the REALIZED panel keeps the elder's full history (composite).
        self.assertGreaterEqual(metrics.get("realized_window_days", 0), 450)
        series = engine._last_port_log_returns
        self.assertIsNotNone(series)
        self.assertGreaterEqual(len(series), 450)
        # Metrics stay finite and sane on the longer window.
        self.assertTrue(np.isfinite(metrics["Sharpe_Ratio"]))
        self.assertTrue(np.isfinite(metrics["Annualised_Return"]))
        self.assertLessEqual(metrics["Max_Drawdown"], 0.0)
        self.assertLess(metrics["CVaR_95_Daily"], 0.0)

    def test_full_panel_equals_legacy_path(self) -> None:
        """No staggering → composite adds no history → the series must be the
        EXACT legacy `a_data @ weights` (bit-for-bit, no numeric churn)."""
        engine, df = _staggered_engine_data(young_days=500)   # both full
        w = {"ELDER.US": 0.6, "YOUNG.US": 0.4}
        _, _, metrics = engine.calculate_structural_risk(
            df, ["ELDER.US", "YOUNG.US"], w)
        series = engine._last_port_log_returns
        rets = np.log(df / df.shift(1)).dropna()
        expected = (rets["ELDER.US"] * 0.6 + rets["YOUNG.US"] * 0.4).values
        self.assertEqual(len(series), len(expected))
        # atol 1e-15: the manual Σw·r reference differs from the engine's
        # matmul only by float op-ordering ulps — semantics identical.
        np.testing.assert_allclose(series.values, expected, rtol=0, atol=1e-15)
        self.assertEqual(metrics["realized_window_days"], len(expected))

    def test_cash_dilution_preserved_on_composite(self) -> None:
        """Σw < 1 (cash in the book) must still dilute the composite series —
        full-coverage days equal a_data @ raw weights exactly."""
        engine, df = _staggered_engine_data()
        _, _, _ = engine.calculate_structural_risk(
            df, ["ELDER.US", "YOUNG.US"],
            {"ELDER.US": 0.3, "YOUNG.US": 0.2})       # 50% cash
        series = engine._last_port_log_returns
        rets = np.log(df / df.shift(1))
        # Probe a late date where BOTH names trade: composite == Σ w·r.
        probe = df.index[-5]
        want = float(rets.loc[probe, "ELDER.US"] * 0.3
                     + rets.loc[probe, "YOUNG.US"] * 0.2)
        self.assertAlmostEqual(float(series.loc[probe]), want, places=12)
        # Probe an early date (young absent): composite convention — the
        # INVESTED sleeve (Σw = 0.5 of NAV) is renormalised across the names
        # trading that day, so it sits fully in the elder → 0.5 · r_elder.
        # (Same convention as the period-returns table / equity curve; the
        # missing name is NOT treated as extra cash.)
        early = df.index[10]
        want_early = float(rets.loc[early, "ELDER.US"] * 0.5)
        self.assertAlmostEqual(float(series.loc[early]), want_early, places=12)


# ═══════════════ F-21: sparklines read the composite series ═════════════════

class SparklineSourceTest(unittest.TestCase):

    def test_uses_precomputed_series(self) -> None:
        from finance.portfolio_series import compute_kpi_trend_series

        rng = np.random.default_rng(5)
        series = pd.Series(rng.normal(0.0005, 0.01, 300),
                           index=pd.date_range("2025-01-01", periods=300,
                                               freq="B"))
        out = compute_kpi_trend_series({"port_log_returns": series})
        self.assertIsNotNone(out)
        self.assertGreaterEqual(len(out["cvar_pts"]), 3)
        self.assertEqual(len(out["cvar_pts"]), len(out["sharpe_pts"]))
        self.assertTrue(all(p <= 0 for p in out["cvar_pts"]))
        self.assertTrue(all(p <= 0 for p in out["mdd_pts"]))

    def test_short_series_falls_back_then_none(self) -> None:
        from finance.portfolio_series import compute_kpi_trend_series
        # 30 days < 90 → precomputed series rejected; no history_result → None.
        short = pd.Series(np.zeros(30))
        self.assertIsNone(compute_kpi_trend_series({"port_log_returns": short}))
        self.assertIsNone(compute_kpi_trend_series({}))    # phase-20 pin

    def test_legacy_fallback_survives_thin_listing(self) -> None:
        """Fallback path (no precomputed series): a <60-day column must be
        FILTERED, not allowed to null the joint window (the SPCX bug)."""
        from finance.portfolio_series import compute_kpi_trend_series

        idx = pd.date_range("2024-06-01", periods=400, freq="B")
        rng = np.random.default_rng(3)
        old = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 400)))
        thin = np.full(400, np.nan)
        thin[-40:] = 25 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 40)))
        prices = pd.DataFrame({"OLD.US": old, "THIN.US": thin}, index=idx)

        class _H:
            data = prices

        perf = pd.DataFrame([
            {"Ticker": "OLD",  "Current_Value": 9000.0},
            {"Ticker": "THIN", "Current_Value": 1000.0},
        ])
        out = compute_kpi_trend_series({
            "history_result": _H(), "performance_table": perf,
            "total_value": 10000.0,
        })
        self.assertIsNotNone(out, "thin listing must not null the sparklines")
        self.assertGreaterEqual(len(out["cvar_pts"]), 3)


# ═══════════════ F-23: leveraged-ETP variance drag ══════════════════════════

class LeveragedDragTest(unittest.TestCase):

    def test_negative_alpha_retained_for_etp_only(self) -> None:
        from finance.investment_logic import apply_leveraged_drag

        exp = np.array([0.001, 0.001, 0.001])
        out = apply_leveraged_drag(
            exp,
            ["CONL", "AAPL", "XNDU.US"],
            [-0.0008, -0.0008, -0.0005])
        self.assertAlmostEqual(out[0], 0.0002, places=10)   # CONL dragged
        self.assertAlmostEqual(out[1], 0.001,  places=10)   # AAPL untouched
        self.assertAlmostEqual(out[2], 0.0005, places=10)   # suffix stripped

    def test_positive_alpha_never_added(self) -> None:
        from finance.investment_logic import apply_leveraged_drag

        out = apply_leveraged_drag(
            np.array([0.001]), ["TQQQ"], [+0.002])
        self.assertAlmostEqual(out[0], 0.001, places=10)

    def test_env_registry_extension(self) -> None:
        from finance.investment_logic import _is_leveraged_etp

        self.assertTrue(_is_leveraged_etp("CONL.US"))
        self.assertFalse(_is_leveraged_etp("NEWLEV"))
        os.environ["LEVERAGED_ETP_EXTRA"] = "NEWLEV, other"
        try:
            self.assertTrue(_is_leveraged_etp("NEWLEV.US"))
            self.assertTrue(_is_leveraged_etp("OTHER"))
        finally:
            del os.environ["LEVERAGED_ETP_EXTRA"]

    def test_nan_alpha_is_safe(self) -> None:
        from finance.investment_logic import apply_leveraged_drag

        out = apply_leveraged_drag(
            np.array([0.001]), ["CONL"], [float("nan")])
        self.assertAlmostEqual(out[0], 0.001, places=10)


# ═══════════════ F-20: uncovered names annotated in the plan ════════════════

class ActionPlanUncoveredTest(unittest.TestCase):

    def _perf(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"Ticker": "SPCX", "Current_Price": 148.77, "Quantity": 4,
             "ATR_Absolute": 3.2},
            {"Ticker": "AAPL", "Current_Price": 314.92, "Quantity": 5,
             "ATR_Absolute": 4.1},
        ])

    def test_uncovered_row_carries_reason(self) -> None:
        from finance.action_plan import build_action_plan

        rows = build_action_plan(
            perf_table=self._perf(),
            asset_scores={"SPCX": {"total": -2.0, "action": "SELL"},
                          "AAPL": {"total": 0.0,  "action": "HOLD"}},
            technicals_map={},
            bl_records=None,
            portfolio_value=14438.0,
            uncovered={"SPCX"},
        )
        by_t = {r.ticker: r for r in rows}
        self.assertIn("вне модели", by_t["SPCX"].reason)
        self.assertNotIn("вне модели", by_t["AAPL"].reason)

    def test_default_behaviour_unchanged(self) -> None:
        from finance.action_plan import build_action_plan

        rows = build_action_plan(
            perf_table=self._perf(),
            asset_scores={"SPCX": {"total": -2.0, "action": "SELL"}},
            technicals_map={}, bl_records=None, portfolio_value=14438.0,
        )
        self.assertTrue(all("вне модели" not in r.reason for r in rows))


if __name__ == "__main__":
    unittest.main()
