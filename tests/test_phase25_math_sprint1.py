# -*- coding: utf-8 -*-
"""
Sprint-1 math-correctness fixes (audit docs/AUDIT_360_2026-07-10.md):

  F-1  stress shocks are transformed into the RESIDUAL factor space when the
       engine orthogonalizes its factors — the stress table is INVARIANT to
       FACTOR_ORTHOGONALIZE (the headline requirement of the fix);
  F-4  the Sharpe/Sortino estimator basis is surfaced to metrics + QC panel;
  F-6  leading NaNs survive the math firewall (no bfill look-ahead) and a
       young listing cannot collapse the structural model's common window;
  F-7  LSE pence quotes (GBX) are scaled ÷100 before the GBP cross-rate.

Run:  python -m pytest tests/test_phase25_math_sprint1.py -q
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Shared synthetic market fixture ──────────────────────────────────────────

def _build_engine_data(n_days: int = 500):
    """Factor ETFs + two assets whose returns are exact linear combos of the
    factor returns (+ tiny idiosyncratic noise) — so both the raw and the
    orthogonalized regressions recover the same fitted subspace and the F-1
    invariance identity holds to float precision."""
    from finance.investment_logic import MAC3RiskEngine

    eng = MAC3RiskEngine(reporting_currency="USD")
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")

    factor_returns: dict[str, np.ndarray] = {}
    prices: dict[str, np.ndarray] = {}
    market = np.random.default_rng(1).normal(0.0004, 0.010, n_days)
    for i, (fname, tkr) in enumerate(eng.factor_tickers.items()):
        if fname == "Market":
            r = market
        else:
            # Correlated with market (β≈0.8) + own component — realistic
            # collinearity so the orthogonalization has real work to do.
            own = np.random.default_rng(50 + i).normal(0.0002, 0.006, n_days)
            r = 0.8 * market + own
        factor_returns[fname] = r
        prices[tkr] = 100.0 * np.exp(np.cumsum(r))

    rng = np.random.default_rng(9)
    a_r = (1.2 * factor_returns["Market"] + 0.6 * factor_returns["Momentum"]
           + rng.normal(0.0, 0.004, n_days))
    b_r = (0.9 * factor_returns["Market"] - 0.4 * factor_returns["Rates"]
           + rng.normal(0.0, 0.004, n_days))
    prices["AAA.US"] = 100.0 * np.exp(np.cumsum(a_r))
    prices["BBB.US"] = 100.0 * np.exp(np.cumsum(b_r))
    return eng, pd.DataFrame(prices, index=idx)


def _perf_from_factor_df(factor_df: pd.DataFrame, values: dict[str, float]) -> pd.DataFrame:
    perf = factor_df.reset_index().rename(columns={"index": "Ticker"})
    perf["Current_Value"] = perf["Ticker"].map(values)
    return perf


class _OrthoEnv:
    """Context manager pinning FACTOR_ORTHOGONALIZE for one engine run."""

    def __init__(self, value: str | None):
        self.value = value

    def __enter__(self):
        self._prev = os.environ.get("FACTOR_ORTHOGONALIZE")
        if self.value is None:
            os.environ.pop("FACTOR_ORTHOGONALIZE", None)
        else:
            os.environ["FACTOR_ORTHOGONALIZE"] = self.value

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop("FACTOR_ORTHOGONALIZE", None)
        else:
            os.environ["FACTOR_ORTHOGONALIZE"] = self._prev


# ── F-1: residualize_shocks unit behaviour ───────────────────────────────────

class ResidualizeShocksTest(unittest.TestCase):

    def test_identity_without_betas(self) -> None:
        from finance.stress import residualize_shocks
        shocks = {"Market": -0.10, "Momentum": -0.15}
        self.assertEqual(residualize_shocks(shocks, {}), shocks)
        self.assertEqual(residualize_shocks(shocks, None), shocks)

    def test_named_child_loses_parent_leg(self) -> None:
        from finance.stress import residualize_shocks
        out = residualize_shocks(
            {"Market": -0.10, "Momentum": -0.15},
            {"Momentum": {"Market": 1.0}},
        )
        # −15% raw MTUM = −10%·β(=1) market leg + −5% residual momentum.
        self.assertAlmostEqual(out["Momentum"], -0.05, places=12)
        self.assertAlmostEqual(out["Market"], -0.10, places=12)  # parent kept

    def test_unnamed_child_gets_counter_leg(self) -> None:
        from finance.stress import residualize_shocks
        # Scenario says raw Value moves 0 while Market −10% ⇒ market-neutral
        # value must move +β·10% for the raw semantics to hold.
        out = residualize_shocks({"Market": -0.10}, {"Value": {"Market": 0.9}})
        self.assertAlmostEqual(out["Value"], 0.09, places=12)

    def test_input_never_mutated(self) -> None:
        from finance.stress import residualize_shocks
        shocks = {"Market": -0.10, "Momentum": -0.15}
        residualize_shocks(shocks, {"Momentum": {"Market": 1.0}})
        self.assertEqual(shocks, {"Market": -0.10, "Momentum": -0.15})


# ── F-1: the headline requirement — stress invariance to orthogonalization ──

class StressOrthoInvarianceTest(unittest.TestCase):
    """The identity w'·B_resid·shock_resid == w'·B_raw·shock_raw is exact for
    OLS; Ridge(α=0.001) on a deliberately-collinear fixture leaves sub-0.5пп
    estimator noise.  The double-count BUG the fix removes is an order of
    magnitude larger, so the test asserts both sides: with the transform the
    runs agree tightly, without it the orthogonalized run drifts visibly."""

    # Moderate shocks — every per-asset delta stays below the ±20% convexity
    # threshold, so the comparison is pure linear algebra (no cap kink).
    def _probe_scenario(self):
        from finance.stress import ScenarioSpec
        return ScenarioSpec(
            name="probe (linear range)",
            shocks={"Market": -0.08, "Momentum": -0.06, "Rates": +0.01},
        )

    def _run(self, ortho: bool, use_transform: bool) -> float:
        from finance.stress import run_stress_scenarios
        eng, data = _build_engine_data()
        with _OrthoEnv("1" if ortho else "0"):
            cov, fdf, metrics = eng.calculate_structural_risk(
                data, ["AAA.US", "BBB.US"],
                {"AAA.US": 0.6, "BBB.US": 0.4},
            )
        perf = _perf_from_factor_df(fdf, {"AAA.US": 60_000.0, "BBB.US": 40_000.0})
        betas = getattr(eng, "_last_ortho_betas", {}) or {}
        if ortho:
            self.assertTrue(betas, "orthogonalized run must expose child→parent betas")
        else:
            self.assertEqual(betas, {}, "raw run must not carry stale betas")
        rows = run_stress_scenarios(
            perf, 100_000.0, metrics,
            scenarios=[self._probe_scenario()],
            ortho_betas=betas if use_transform else {},
        )
        return float(rows[0]["port_pct"])

    def test_port_pct_invariant_to_orthogonalization(self) -> None:
        raw        = self._run(ortho=False, use_transform=False)  # legacy truth
        orth_fixed = self._run(ortho=True,  use_transform=True)   # F-1 path
        orth_buggy = self._run(ortho=True,  use_transform=False)  # pre-fix path

        # (i) With the residual transform the two engine modes agree to
        # within Ridge-vs-OLS estimator noise.  Measured on this fixture:
        # fixed_gap ≈ 0.0056 @ n=500 (→ 0.0036 @ n=1250 — vanishes with n,
        # i.e. it is the α=0.001 shrinkage differing between the raw and the
        # residualized design, NOT a residual double-count).
        self.assertAlmostEqual(raw, orth_fixed, delta=0.008,
                               msg=f"raw {raw} vs fixed {orth_fixed}")
        # (ii) The transform is load-bearing: skipping it (the pre-F-1
        # behaviour) drifts by the double-counted parent leg — measured
        # bug_gap ≈ 0.032, ~6× the estimator noise.
        bug_gap   = abs(raw - orth_buggy)
        fixed_gap = abs(raw - orth_fixed)
        self.assertGreater(bug_gap, 0.015,
                           "fixture must make the double-count bug visible")
        self.assertLess(fixed_gap, bug_gap / 3.0,
                        f"fix must shrink the gap: fixed {fixed_gap} vs bug {bug_gap}")

    def test_orth_run_flags_residualization(self) -> None:
        from finance.stress import run_stress_scenarios
        eng, data = _build_engine_data()
        with _OrthoEnv("1"):
            cov, fdf, metrics = eng.calculate_structural_risk(
                data, ["AAA.US", "BBB.US"],
                {"AAA.US": 0.6, "BBB.US": 0.4},
            )
        perf = _perf_from_factor_df(fdf, {"AAA.US": 60_000.0, "BBB.US": 40_000.0})
        rows = run_stress_scenarios(perf, 100_000.0, metrics,
                                    ortho_betas=eng._last_ortho_betas)
        tech = next(r for r in rows if "Tech sell-off" in r["name"])
        self.assertTrue(tech["residualized"])
        # Raw (display) vector is untouched; the applied one differs.
        self.assertAlmostEqual(tech["shocks"]["Momentum"], -0.15)
        self.assertNotAlmostEqual(
            tech["shocks_applied"]["Momentum"], -0.15, places=3)
        # β(Momentum→Market) ≈ 0.8 in the fixture ⇒ residual momentum shock
        # ≈ −0.15 − 0.8·(−0.10) = −0.07 (± regression noise).
        self.assertAlmostEqual(tech["shocks_applied"]["Momentum"], -0.07,
                               delta=0.02)


# ── F-6: firewall look-ahead + sparse guard live in test_phase14_refactor ───
# (MathFirewallTest was updated in place; the structural sparse-asset guard is
# covered there too so the pre-existing suite keeps the behaviour pinned.)


# ── F-7: GBX (pence) handling ────────────────────────────────────────────────

class GbxCurrencyTest(unittest.TestCase):

    def test_lse_suffixes_infer_pence(self) -> None:
        from finance.currency import infer_asset_currency
        self.assertEqual(infer_asset_currency("VOD.L"),  "GBX")
        self.assertEqual(infer_asset_currency("FOO.IL"), "GBX")
        # USD-settling GDR overrides stay pinned (H2 behaviour).
        self.assertEqual(infer_asset_currency("HSBK.IL"), "USD")
        self.assertEqual(infer_asset_currency("KAP.IL"),  "USD")

    def test_pence_scaled_before_cross_rate(self) -> None:
        from finance.currency import ReportingCurrency, convert_price_matrix
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"VOD.L": np.full(len(idx), 25_000.0)}, index=idx)

        def fx(base: str, quote: str):
            # The provider must be asked for POUNDS, never for "GBX".
            assert base == "GBP" and quote == "USD", (base, quote)
            return pd.Series(1.25, index=idx)

        res = convert_price_matrix(
            prices, {"VOD.L": "GBX"}, ReportingCurrency.USD, fx)
        # 25 000 GBX = £250 → ×1.25 = $312.50 (a raw-GBP pipe would say $31 250).
        self.assertAlmostEqual(float(res.prices_base["VOD.L"].iloc[-1]),
                               312.50, places=6)
        self.assertEqual(res.fx_records[0].pair, "GBPUSD")

    def test_no_cross_rate_still_drops_h1(self) -> None:
        from finance.currency import ReportingCurrency, convert_price_matrix
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        prices = pd.DataFrame({"VOD.L": np.full(len(idx), 25_000.0),
                               "AAPL.US": np.full(len(idx), 180.0)}, index=idx)
        res = convert_price_matrix(
            prices, {"VOD.L": "GBX", "AAPL.US": "USD"},
            ReportingCurrency.USD, lambda b, q: None)
        self.assertNotIn("VOD.L", res.prices_base.columns)   # H-1: no mixing
        self.assertIn("AAPL.US", res.prices_base.columns)

    def test_fred_registry_carries_gbp_cross(self) -> None:
        from services.fx_feed import FRED_FX_SERIES
        self.assertIn(("GBP", "USD"), FRED_FX_SERIES)
        self.assertIn(("USD", "GBP"), FRED_FX_SERIES)
        series, sign = FRED_FX_SERIES[("GBP", "USD")]
        self.assertEqual(series, "DEXUSUK")
        self.assertGreater(sign, 0)     # USD-per-GBP is the direct quote


# ── F-4: Sharpe basis note surfaced end-to-end ───────────────────────────────

class SharpeBasisNoteTest(unittest.TestCase):

    def test_engine_emits_note(self) -> None:
        eng, data = _build_engine_data(n_days=300)
        _, _, metrics = eng.calculate_structural_risk(
            data, ["AAA.US"], {"AAA.US": 1.0})
        note = metrics.get("sharpe_basis_note", "")
        self.assertIn("EWMA", note)
        self.assertIn("геометрическая", note)

    def test_integrity_panel_row(self) -> None:
        from pdf_payload import _build_integrity_checks
        checks = _build_integrity_checks(
            results={"portfolio_metrics": {"sharpe_basis_note": "x"}},
            ai_summary={}, data_quality={}, return_series_coverage={},
        )
        labels = [c["label"] for c in checks]
        self.assertIn("Базис Sharpe/Sortino", labels)

    def test_no_row_without_note(self) -> None:
        from pdf_payload import _build_integrity_checks
        checks = _build_integrity_checks(
            results={"portfolio_metrics": {}},
            ai_summary={}, data_quality={}, return_series_coverage={},
        )
        labels = [c["label"] for c in checks]
        self.assertNotIn("Базис Sharpe/Sortino", labels)


if __name__ == "__main__":
    unittest.main()
