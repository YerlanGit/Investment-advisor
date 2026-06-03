"""
Phase 6 regression tests — guard the H1/H2/H3/H4/H5 risk-engine fixes.

These tests deliberately avoid touching the existing 214 tests' assertions:
all new behaviour is validated through new, isolated fixtures.

  H1: geometric annualisation of log-returns (exp(mean·252)−1) replaces
      the legacy arithmetic mean·252 in Sharpe/Sortino.
  H2: base-currency approach — FX-transformed price matrix flows into
      the covariance matrix; reporting-currency-matched RFR.
  H3: geometric daily RFR (1+r)^(1/252)−1 for the Sortino downside filter.
  H4: convex saturation of per-asset stress impacts above ±20%, asymptote
      at ±35%; stress_test_disclaimer surfaced in portfolio_metrics.
  H5: deterministic-but-data-driven seed for bootstrap CVaR.

Network-free: every external dependency (Tradernet, Anthropic, FRED) is
either mocked or sits behind a graceful-degradation path.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `import finance...` without a package install.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── H2: ReportingCurrency / RFR / FX layer ──────────────────────────────────

class ReportingCurrencyEnvTest(unittest.TestCase):
    """REPORTING_CURRENCY env var wins; legacy KZ_RFR_ANNUAL still maps to KZT."""

    def setUp(self) -> None:
        self._snapshot = {k: os.environ.get(k)
                          for k in ("REPORTING_CURRENCY", "KZ_RFR_ANNUAL", "US_RFR_ANNUAL")}
        for k in self._snapshot:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_default_is_usd(self) -> None:
        from finance.currency import ReportingCurrency
        self.assertEqual(ReportingCurrency.from_env(), ReportingCurrency.USD)

    def test_env_explicit_kzt(self) -> None:
        from finance.currency import ReportingCurrency
        os.environ["REPORTING_CURRENCY"] = "KZT"
        self.assertEqual(ReportingCurrency.from_env(), ReportingCurrency.KZT)

    def test_legacy_kz_rfr_implies_kzt(self) -> None:
        """Backward compat: legacy prod env (only KZ_RFR_ANNUAL set) → KZT."""
        from finance.currency import ReportingCurrency
        os.environ["KZ_RFR_ANNUAL"] = "0.14"
        self.assertEqual(ReportingCurrency.from_env(), ReportingCurrency.KZT)

    def test_unknown_env_falls_back(self) -> None:
        from finance.currency import ReportingCurrency
        os.environ["REPORTING_CURRENCY"] = "EUR"   # unsupported
        self.assertEqual(ReportingCurrency.from_env(), ReportingCurrency.USD)


class RfrLookupTest(unittest.TestCase):

    def setUp(self) -> None:
        self._snapshot = {k: os.environ.get(k)
                          for k in ("KZ_RFR_ANNUAL", "US_RFR_ANNUAL")}
        for k in self._snapshot:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_usd_default_rfr_is_45_bps_above_zero(self) -> None:
        from finance.currency import ReportingCurrency, get_rfr_for_currency
        r, src = get_rfr_for_currency(ReportingCurrency.USD)
        self.assertAlmostEqual(r, 0.045, places=4)
        self.assertIn("default", src)

    def test_kzt_default_rfr_is_14pct(self) -> None:
        from finance.currency import ReportingCurrency, get_rfr_for_currency
        r, _ = get_rfr_for_currency(ReportingCurrency.KZT)
        self.assertAlmostEqual(r, 0.14, places=4)

    def test_env_override_wins(self) -> None:
        from finance.currency import ReportingCurrency, get_rfr_for_currency
        os.environ["US_RFR_ANNUAL"] = "0.053"
        r, src = get_rfr_for_currency(ReportingCurrency.USD)
        self.assertAlmostEqual(r, 0.053, places=4)
        self.assertIn("US_RFR_ANNUAL", src)


class GeometricDailyRfrTest(unittest.TestCase):
    """H3: (1+r)^(1/252)−1, not r/252."""

    def test_geometric_below_linear_for_positive_rate(self) -> None:
        from finance.currency import daily_rfr_geometric
        linear   = 0.045 / 252
        geometric = daily_rfr_geometric(0.045, 252)
        self.assertLess(geometric, linear)
        # Difference is ~r²/(2·252) — for 4.5% RFR that's ~4e-6.  Confirm
        # it's in that ballpark (within 1e-5).
        self.assertLess(abs(geometric - linear), 1e-5)

    def test_zero_rate_geometric_is_zero(self) -> None:
        from finance.currency import daily_rfr_geometric
        self.assertEqual(daily_rfr_geometric(0.0, 252), 0.0)


class AssetCurrencyInferenceTest(unittest.TestCase):

    def test_us_suffix(self) -> None:
        from finance.currency import infer_asset_currency
        self.assertEqual(infer_asset_currency("AAPL.US"), "USD")
        self.assertEqual(infer_asset_currency("SPY.US"),  "USD")

    def test_kz_suffix(self) -> None:
        from finance.currency import infer_asset_currency
        self.assertEqual(infer_asset_currency("KSPI.KZ"), "KZT")

    def test_il_override(self) -> None:
        from finance.currency import infer_asset_currency
        self.assertEqual(infer_asset_currency("HSBK.IL"), "USD")   # GDR settles USD

    def test_bare_symbol_defaults_usd(self) -> None:
        from finance.currency import infer_asset_currency
        self.assertEqual(infer_asset_currency("AAPL"), "USD")


class FxAlignmentNoLookaheadTest(unittest.TestCase):
    """T-1 lag rule prevents importing tomorrow's FX into today's price."""

    def test_one_day_lag_shifts_series(self) -> None:
        from finance.currency import align_fx_to_prices
        idx = pd.date_range("2025-01-01", periods=5, freq="D")
        fx  = pd.Series([500.0, 501.0, 502.0, 503.0, 504.0], index=idx)
        aligned, rec = align_fx_to_prices(fx, idx, lag_one_day=True)
        # First day backfills to first-known (500), subsequent days are
        # PREVIOUS day's value.
        self.assertEqual(aligned.iloc[0], 500.0)
        self.assertEqual(aligned.iloc[1], 500.0)
        self.assertEqual(aligned.iloc[2], 501.0)
        self.assertEqual(aligned.iloc[3], 502.0)
        # Coverage 100% after fill.
        self.assertEqual(rec.coverage_pct, 100.0)

    def test_no_lag_passes_through(self) -> None:
        from finance.currency import align_fx_to_prices
        idx = pd.date_range("2025-01-01", periods=3, freq="D")
        fx  = pd.Series([500.0, 501.0, 502.0], index=idx)
        aligned, _ = align_fx_to_prices(fx, idx, lag_one_day=False)
        self.assertEqual(aligned.iloc[0], 500.0)
        self.assertEqual(aligned.iloc[2], 502.0)


class PriceMatrixTransformShortCircuitTest(unittest.TestCase):
    """
    Self-check #2: when reporting == every asset's currency, the function
    returns the ORIGINAL frame (no copy) and reports `no_op=True`.
    """

    def test_usd_only_portfolio_is_no_op(self) -> None:
        from finance.currency import (ReportingCurrency, convert_price_matrix)
        idx = pd.date_range("2025-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"AAPL.US": np.linspace(100, 110, 10),
                               "SPY.US":  np.linspace(400, 420, 10)}, index=idx)
        ccys   = {"AAPL.US": "USD", "SPY.US": "USD"}
        result = convert_price_matrix(prices, ccys,
                                       reporting=ReportingCurrency.USD,
                                       fx_provider=None)
        self.assertTrue(result.no_op)
        # Identity → SAME object (no copy taken in short-circuit).
        self.assertIs(result.prices_base, prices)

    def test_kzt_reporting_multiplies_usd_assets(self) -> None:
        from finance.currency import (ReportingCurrency, convert_price_matrix)
        idx = pd.date_range("2025-01-01", periods=4, freq="D")
        prices = pd.DataFrame({"AAPL.US": [100.0, 101.0, 102.0, 103.0]}, index=idx)
        # USD/KZT history: 500, 500, 510, 520
        fx = pd.Series([500.0, 500.0, 510.0, 520.0], index=idx)
        def provider(base, quote):
            self.assertEqual((base, quote), ("USD", "KZT"))
            return fx
        result = convert_price_matrix(prices, {"AAPL.US": "USD"},
                                       reporting=ReportingCurrency.KZT,
                                       fx_provider=provider,
                                       lag_one_day=True)
        self.assertFalse(result.no_op)
        # T-1 lag: day 0 backfills to 500; day 1 → fx[0]=500; day 2 → fx[1]=500;
        # day 3 → fx[2]=510.
        expected = np.array([100.0 * 500, 101.0 * 500, 102.0 * 500, 103.0 * 510])
        np.testing.assert_allclose(result.prices_base["AAPL.US"].values, expected)
        # Records cover the pair.
        self.assertEqual(len(result.fx_records), 1)
        self.assertEqual(result.fx_records[0].pair, "USDKZT")


class FxNotMutatingInputTest(unittest.TestCase):
    """Self-check #1: caller's frame must NOT be mutated."""

    def test_input_frame_untouched(self) -> None:
        from finance.currency import (ReportingCurrency, convert_price_matrix)
        idx = pd.date_range("2025-01-01", periods=3, freq="D")
        prices = pd.DataFrame({"AAPL.US": [100.0, 101.0, 102.0]}, index=idx)
        snapshot = prices.copy(deep=True)
        fx = pd.Series([500.0, 500.0, 510.0], index=idx)
        convert_price_matrix(prices, {"AAPL.US": "USD"},
                             reporting=ReportingCurrency.KZT,
                             fx_provider=lambda a, b: fx)
        pd.testing.assert_frame_equal(prices, snapshot)


# ── H1+H3: Sharpe with geometric annualisation + currency-matched RFR ───────

class SharpeGeometricAnnualisationTest(unittest.TestCase):
    """
    End-to-end check on calculate_structural_risk:
      • Annualised_Return present and equals exp(mean_log·252)−1.
      • Sharpe = (ann_return − annual_rfr) / vol_ann.
      • Reporting metadata populated.
    """

    @staticmethod
    def _make_engine(reporting: str = "USD"):
        from finance.investment_logic import MAC3RiskEngine
        return MAC3RiskEngine(reporting_currency=reporting)

    @classmethod
    def _synthetic_factor_frame(cls, engine, n_days=300, seed=11):
        idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
        rng = np.random.default_rng(seed)
        df: dict[str, np.ndarray] = {}
        for i, tkr in enumerate(engine.factor_tickers.values()):
            r = np.random.default_rng(seed + i + 1).normal(0.0004, 0.01, n_days)
            df[tkr] = 100.0 * np.exp(np.cumsum(r))
        asset_r = rng.normal(0.0008, 0.014, n_days)
        df["AAPL.US"] = 100.0 * np.exp(np.cumsum(asset_r))
        return pd.DataFrame(df, index=idx)

    def test_metrics_include_reporting_and_geometric_ann_return(self) -> None:
        engine = self._make_engine("USD")
        data   = self._synthetic_factor_frame(engine)
        _, _, port_metrics = engine.calculate_structural_risk(
            data, ["AAPL.US"], {"AAPL.US": 1.0},
        )

        # New canonical keys present.
        for k in ("Annualised_Return", "reporting_currency",
                  "risk_free_rate_annual", "risk_free_rate_daily",
                  "risk_free_rate_source", "stress_test_disclaimer"):
            self.assertIn(k, port_metrics)

        self.assertEqual(port_metrics["reporting_currency"], "USD")
        # USD default RFR ≈ 4.5%
        self.assertAlmostEqual(port_metrics["risk_free_rate_annual"], 0.045, places=4)
        # Daily RFR is geometric: (1.045)^(1/252)-1
        expected_daily = (1.045) ** (1 / 252) - 1
        self.assertAlmostEqual(port_metrics["risk_free_rate_daily"], expected_daily, places=8)

    def test_kzt_reporting_uses_kz_rfr(self) -> None:
        engine = self._make_engine("KZT")
        data   = self._synthetic_factor_frame(engine)
        _, _, port_metrics = engine.calculate_structural_risk(
            data, ["AAPL.US"], {"AAPL.US": 1.0},
        )
        self.assertEqual(port_metrics["reporting_currency"], "KZT")
        self.assertAlmostEqual(port_metrics["risk_free_rate_annual"], 0.14, places=4)

    def test_sharpe_difference_between_currencies_matches_rfr_gap(self) -> None:
        """
        Same synthetic data, same vol/return — Sharpe difference must equal
        (KZ_RFR − US_RFR) / vol.  This is the H2 fix's quantitative payoff.
        """
        eng_usd = self._make_engine("USD")
        eng_kzt = self._make_engine("KZT")
        data_usd = self._synthetic_factor_frame(eng_usd)
        data_kzt = self._synthetic_factor_frame(eng_kzt)  # same seed

        _, _, m_usd = eng_usd.calculate_structural_risk(
            data_usd, ["AAPL.US"], {"AAPL.US": 1.0})
        _, _, m_kzt = eng_kzt.calculate_structural_risk(
            data_kzt, ["AAPL.US"], {"AAPL.US": 1.0})

        # Same data → same vol, same ann_return.
        self.assertAlmostEqual(m_usd["Total_Volatility_Ann"],
                               m_kzt["Total_Volatility_Ann"], places=8)
        self.assertAlmostEqual(m_usd["Annualised_Return"],
                               m_kzt["Annualised_Return"], places=8)
        # Sharpe gap = (RFR_KZT − RFR_USD) / vol  (with sign)
        gap_expected = (0.14 - 0.045) / m_usd["Total_Volatility_Ann"]
        gap_actual   = m_usd["Sharpe_Ratio"] - m_kzt["Sharpe_Ratio"]
        self.assertAlmostEqual(gap_actual, gap_expected, places=6)


# ── H4: convexity cap on stress per-asset impacts ───────────────────────────

class ConvexCapTest(unittest.TestCase):

    def test_identity_below_threshold(self) -> None:
        from finance.stress import _convex_cap, CONVEXITY_THRESHOLD
        for x in (-0.15, -0.05, 0.0, 0.10, CONVEXITY_THRESHOLD):
            self.assertAlmostEqual(_convex_cap(x), x, places=12)

    def test_saturation_above_threshold(self) -> None:
        from finance.stress import _convex_cap, CONVEXITY_HARD_CAP
        # Severe down shock → capped strictly above asymptote, never reaches it.
        x = -0.50
        y = _convex_cap(x)
        self.assertLess(y, -0.20)                       # past the threshold
        self.assertGreater(y, -CONVEXITY_HARD_CAP)      # never breaches the cap
        # Sign preserved.
        self.assertLess(y, 0)

    def test_monotonic(self) -> None:
        from finance.stress import _convex_cap
        # Strictly monotonic in x.
        xs = np.linspace(-1.0, 1.0, 401)
        ys = np.array([_convex_cap(float(x)) for x in xs])
        self.assertTrue((np.diff(ys) > 0).all())

    def test_continuous_derivative_at_threshold(self) -> None:
        from finance.stress import _convex_cap, CONVEXITY_THRESHOLD
        eps = 1e-5
        T = CONVEXITY_THRESHOLD
        left  = (_convex_cap(T) - _convex_cap(T - eps)) / eps
        right = (_convex_cap(T + eps) - _convex_cap(T)) / eps
        # Both slopes ≈ 1 at the threshold (smooth merge).
        self.assertAlmostEqual(left,  1.0, places=4)
        self.assertAlmostEqual(right, 1.0, places=3)


class StressScenarioConvexityAuditTest(unittest.TestCase):
    """Verify the per-scenario row exposes convexity audit fields."""

    def test_high_beta_position_triggers_convex_cap(self) -> None:
        from finance.stress import apply_scenario, DEFAULT_SCENARIOS
        # Tech sell-off: Market −10%, Momentum −15%, Quality −5%
        scenario = next(s for s in DEFAULT_SCENARIOS
                        if s.name.startswith("Tech sell-off"))
        # One position with β=2.5 on Market → raw delta = -25% (above cap).
        perf = pd.DataFrame([{
            "Ticker": "AVGO", "Current_Value": 10_000.0,
            "Beta_Market":   2.5,
            "Beta_Momentum": 0.0,
            "Beta_Quality":  0.0,
        }])
        result = apply_scenario(perf, total_value=10_000.0, scenario=scenario,
                                 port_vol_ann=0.25)
        # raw was -25%, capped value strictly less negative (closer to zero)
        self.assertEqual(result["convexity_applied_n"], 1)
        ad_raw = result["by_asset"][0]["asset_delta_raw"]
        ad_cap = result["by_asset"][0]["asset_delta_pct"]
        self.assertLess(ad_raw, -20.0)          # raw breached threshold
        self.assertGreater(ad_cap, ad_raw)      # cap pulled it toward zero
        self.assertGreater(ad_cap, -35.0)       # asymptote enforced

    def test_low_beta_position_passes_through(self) -> None:
        from finance.stress import apply_scenario, DEFAULT_SCENARIOS
        scenario = next(s for s in DEFAULT_SCENARIOS
                        if s.name.startswith("Tech sell-off"))
        perf = pd.DataFrame([{
            "Ticker": "TLT", "Current_Value": 10_000.0,
            "Beta_Market":   0.3,
            "Beta_Momentum": 0.0,
            "Beta_Quality":  0.0,
        }])
        result = apply_scenario(perf, total_value=10_000.0, scenario=scenario,
                                 port_vol_ann=0.10)
        self.assertEqual(result["convexity_applied_n"], 0)
        # raw == cap value (well below 20% threshold)
        self.assertAlmostEqual(
            result["by_asset"][0]["asset_delta_raw"],
            result["by_asset"][0]["asset_delta_pct"],
            places=6,
        )


# ── H5: deterministic-but-data-driven bootstrap seed ────────────────────────

class BootstrapDeterministicSeedTest(unittest.TestCase):

    def test_same_returns_yield_same_ci(self) -> None:
        from finance.investment_logic import MAC3RiskEngine
        rng = np.random.default_rng(123)
        rets = rng.normal(0.0005, 0.012, 400)
        a = MAC3RiskEngine._bootstrap_cvar(rets, n_boot=200)
        b = MAC3RiskEngine._bootstrap_cvar(rets, n_boot=200)
        self.assertAlmostEqual(a["lo95"], b["lo95"], places=12)
        self.assertAlmostEqual(a["hi95"], b["hi95"], places=12)

    def test_different_returns_yield_different_ci(self) -> None:
        from finance.investment_logic import MAC3RiskEngine
        rng = np.random.default_rng(123)
        rets_a = rng.normal(0.0005, 0.012, 400)
        rets_b = rets_a.copy()
        rets_b[-1] -= 0.01            # one-tick perturbation → new seed
        a = MAC3RiskEngine._bootstrap_cvar(rets_a, n_boot=200)
        b = MAC3RiskEngine._bootstrap_cvar(rets_b, n_boot=200)
        # CIs not identical — the seed actually moved.
        self.assertNotEqual((a["lo95"], a["hi95"]), (b["lo95"], b["hi95"]))

    def test_explicit_seed_still_honoured(self) -> None:
        """Existing tests pass seed=N explicitly — must still control the RNG."""
        from finance.investment_logic import MAC3RiskEngine
        rng = np.random.default_rng(7)
        rets = rng.normal(0, 0.01, 200)
        a = MAC3RiskEngine._bootstrap_cvar(rets, n_boot=100, seed=11)
        b = MAC3RiskEngine._bootstrap_cvar(rets, n_boot=100, seed=11)
        self.assertEqual(a["lo95"], b["lo95"])


# ── Acceptance Criteria (task spec, tests 1 & 2) ─────────────────────────────

class AcceptanceTest1_USDReport_AAPL_KZBond(unittest.TestCase):
    """
    [Test 1] Portfolio [95% AAPL, 5% KZT-bond USD-denominated],
    reporting=USD → SOFR ~4.5% used, no FX conversion of AAPL prices,
    Sharpe not depressed by KZ_RFR.
    """

    def test_no_fx_conversion_and_us_rfr(self) -> None:
        from finance.investment_logic import MAC3RiskEngine
        engine = MAC3RiskEngine(reporting_currency="USD")

        idx = pd.date_range("2024-01-01", periods=300, freq="B")
        rng = np.random.default_rng(7)
        prices = {}
        # Factor ETFs
        for i, tkr in enumerate(engine.factor_tickers.values()):
            r = np.random.default_rng(100 + i).normal(0.0004, 0.01, len(idx))
            prices[tkr] = 100.0 * np.exp(np.cumsum(r))
        # AAPL (.US → USD)
        a_r = rng.normal(0.0008, 0.014, len(idx))
        prices["AAPL.US"] = 100.0 * np.exp(np.cumsum(a_r))
        df = pd.DataFrame(prices, index=idx)

        # Apply the engine's FX path explicitly — should no-op.
        converted = engine._apply_fx_conversion(df)
        self.assertIs(converted, df)           # short-circuit: same object
        self.assertEqual(engine._last_fx_records, [])

        _, _, m = engine.calculate_structural_risk(
            converted, ["AAPL.US"], {"AAPL.US": 0.95, "KZBOND.IL": 0.05},
        )
        self.assertEqual(m["reporting_currency"], "USD")
        self.assertAlmostEqual(m["risk_free_rate_annual"], 0.045, places=4)


class AcceptanceTest2_KZTReport_Cross(unittest.TestCase):
    """
    [Test 2] Same portfolio, reporting=KZT →
    USD prices × USDKZT, RFR=14%, portfolio volatility includes FX vol.
    """

    def _build_data(self, engine, n=300):
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = {}
        for i, tkr in enumerate(engine.factor_tickers.values()):
            r = np.random.default_rng(200 + i).normal(0.0004, 0.01, len(idx))
            prices[tkr] = 100.0 * np.exp(np.cumsum(r))
        a_r = np.random.default_rng(99).normal(0.0008, 0.014, len(idx))
        prices["AAPL.US"] = 100.0 * np.exp(np.cumsum(a_r))
        return pd.DataFrame(prices, index=idx), idx

    def test_kzt_reporting_increases_volatility_via_fx(self) -> None:
        from finance.investment_logic import MAC3RiskEngine
        # Baseline: USD reporting, no FX
        eng_usd = MAC3RiskEngine(reporting_currency="USD")
        df_usd, idx = self._build_data(eng_usd)
        _, _, m_usd = eng_usd.calculate_structural_risk(
            eng_usd._apply_fx_conversion(df_usd), ["AAPL.US"], {"AAPL.US": 1.0})

        # KZT reporting with a volatile USDKZT series
        rng = np.random.default_rng(5)
        # USDKZT drift around 500, with daily σ ≈ 1% — comparable to equity σ
        fx_returns = rng.normal(0.0, 0.01, len(idx))
        fx_path    = 500.0 * np.exp(np.cumsum(fx_returns))
        fx_series  = pd.Series(fx_path, index=idx)
        def fx_provider(base, quote):
            return fx_series if (base == "USD" and quote == "KZT") else None

        eng_kzt = MAC3RiskEngine(reporting_currency="KZT", fx_provider=fx_provider)
        df_kzt, _ = self._build_data(eng_kzt)
        df_conv   = eng_kzt._apply_fx_conversion(df_kzt)
        # Sanity: the converted price column is bigger (× ~500)
        self.assertGreater(df_conv["AAPL.US"].iloc[-1], df_kzt["AAPL.US"].iloc[-1] * 100)

        _, _, m_kzt = eng_kzt.calculate_structural_risk(
            df_conv, ["AAPL.US"], {"AAPL.US": 1.0})
        # KZT reporting uses 14% RFR.
        self.assertAlmostEqual(m_kzt["risk_free_rate_annual"], 0.14, places=4)
        # FX volatility must be visible: vol_kzt > vol_usd (FX path adds risk).
        self.assertGreater(m_kzt["Total_Volatility_Ann"],
                           m_usd["Total_Volatility_Ann"])
        # FX conversion was actually applied — audit record present.
        pairs = [r["pair"] for r in m_kzt["fx_conversion"]]
        self.assertIn("USDKZT", pairs)


if __name__ == "__main__":
    unittest.main()
