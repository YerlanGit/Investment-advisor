"""
Phase 14 — P2/P3 refactor verification:
  H1/H2 consolidation, H7/M1 math robustness, H6 waterfall geometry,
  H3 PII TTL.
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


# ── H1/H2: single source of truth ──────────────────────────────────────────

class ConsolidationTest(unittest.TestCase):

    def test_classify_asset_class_canonical(self) -> None:
        from finance.scoring import classify_asset_class
        cases = {
            "GLD": "Сырьё", "SLV": "Сырьё",
            "TLT": "Облигации", "AGG": "Облигации",
            "AAPL.US": "Акции США", "MSFT": "Акции США",
            "KSPI.KZ": "Акции KZ", "KSPI": "Акции KZ",
            "FFSPC6.1028.AIX": "Акции KZ",
            "USD": "Ден. средства", "CASH": "Ден. средства",
            "BTC-USD": "Крипто", "ETH": "Крипто",
        }
        for tk, exp in cases.items():
            self.assertEqual(classify_asset_class(tk), exp, tk)

    def test_no_substring_false_positive(self) -> None:
        """The old bot `any(x in t)` mis-fired; canonical must be exact."""
        from finance.scoring import classify_asset_class
        # "ABOND" must NOT match "BOND" substring as a real bond unless it's
        # a genuine bond ticker.  "DBOND.US" base contains BOND → Облигации
        # is acceptable; but a normal equity must not.
        self.assertEqual(classify_asset_class("AAPL.US"), "Акции США")

    def test_pdf_and_bot_use_same_classifier(self) -> None:
        import pdf_payload
        from finance.scoring import classify_asset_class
        self.assertIs(pdf_payload._classify_asset, classify_asset_class)

    def test_composite_risk_score_single_source(self) -> None:
        from finance.scoring import composite_risk_score
        from finance.investment_logic import MAC3RiskEngine
        from finance.simulate import _composite_risk_score
        for args in [(0.05, -0.02, 5.0), (0.20, -0.02, 5.0), (0.40, -0.05, 30.0)]:
            base = composite_risk_score(*args)
            self.assertEqual(MAC3RiskEngine._composite_risk_score(*args), base)
            self.assertEqual(_composite_risk_score(*args), base)


# ── H7/M1: math robustness ─────────────────────────────────────────────────

class MathFirewallTest(unittest.TestCase):

    def test_firewall_strips_inf(self) -> None:
        """F-6 semantics: ±Inf never survives; interior/trailing gaps are
        ffill-ed; LEADING NaNs are PRESERVED by design — back-filling them
        copied an asset's first quote over its pre-listing period (look-ahead
        bias that dragged young assets' σ/β toward zero)."""
        from finance.investment_logic import MAC3RiskEngine
        eng = MAC3RiskEngine()
        df = pd.DataFrame({"A": [1.0, np.inf, 3.0], "B": [np.nan, 2.0, -np.inf]})
        out = eng.math_firewall(df)
        arr = out.to_numpy()
        self.assertFalse(np.isinf(arr).any(), "no Inf may survive the firewall")
        # Interior Inf→NaN gaps are carried over by ffill…
        self.assertEqual(out["A"].tolist(), [1.0, 1.0, 3.0])
        self.assertEqual(out["B"].iloc[2], 2.0)      # trailing -inf → ffill(2.0)
        # …but the pre-listing (leading) NaN must stay NaN (no look-ahead).
        self.assertTrue(np.isnan(out["B"].iloc[0]),
                        "leading NaN must survive — bfill was a look-ahead")

    def test_firewall_sparse_asset_guard(self) -> None:
        """F-6: a young listing (< MIN_OVERLAP_TDAYS finite prices) is dropped
        from the STRUCTURAL model instead of collapsing the common dropna
        window for the whole book."""
        from finance.investment_logic import MAC3RiskEngine
        eng = MAC3RiskEngine()
        idx = pd.date_range("2023-01-02", periods=400, freq="B")
        rng = np.random.default_rng(11)
        prices = {}
        for i, tkr in enumerate(eng.factor_tickers.values()):
            r = np.random.default_rng(300 + i).normal(0.0004, 0.01, len(idx))
            prices[tkr] = 100.0 * np.exp(np.cumsum(r))
        prices["AAA.US"] = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, len(idx))))
        # Young listing: only the last 20 sessions have quotes.
        young = np.full(len(idx), np.nan)
        young[-20:] = 50.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 20)))
        prices["NEWIPO.US"] = young
        data = pd.DataFrame(prices, index=idx)

        cov, fdf, metrics = eng.calculate_structural_risk(
            data, ["AAA.US", "NEWIPO.US"],
            {"AAA.US": 0.7, "NEWIPO.US": 0.3},
        )
        # The young asset is excluded from the structural model…
        self.assertNotIn("NEWIPO.US", list(cov.index))
        self.assertIn("AAA.US", list(cov.index))
        # …and the surviving window was NOT collapsed to the IPO's 20 days.
        self.assertGreater(len(getattr(eng, "_last_port_log_returns", [])), 300)


# ── H6: waterfall geometry precomputed in backend ──────────────────────────

class WaterfallGeometryTest(unittest.TestCase):

    def _wf(self):
        from pdf_payload import _build_risk_waterfall
        idx = ["A", "B", "C", "D", "E", "F"]
        # diagonal cov (annual variance) → σ_i = sqrt(diag)
        cov = pd.DataFrame(np.diag([0.04, 0.09, 0.01, 0.16, 0.02, 0.03]),
                           index=idx, columns=idx)
        perf = pd.DataFrame([{"Ticker": t, "Current_Value": 1000.0} for t in idx])
        return _build_risk_waterfall(cov, perf, total_val=6000.0, total_vol_ann=0.20)

    def test_bars_aggregate_beyond_top4(self) -> None:
        wf = self._wf()
        self.assertIsNotNone(wf)
        bars = wf["bars"]
        # 6 contributions → top-4 + 1 aggregate "+N др." bar = 5 bars.
        self.assertEqual(len(bars), 5)
        self.assertTrue(bars[-1].get("is_aggregate"))
        self.assertTrue(bars[-1]["ticker"].startswith("+"))

    def test_cumulative_offsets_present_and_monotonic(self) -> None:
        wf = self._wf()
        cum = [b["cum_end_pp"] for b in wf["bars"]]
        self.assertEqual(cum, sorted(cum), "cum_end_pp must be non-decreasing")
        # last cum_end ≈ sum_standalone_pp
        self.assertAlmostEqual(cum[-1], wf["sum_standalone_pp"], places=1)

    def test_y_axis_domain_and_ticks(self) -> None:
        wf = self._wf()
        self.assertGreaterEqual(wf["y_max_pp"], wf["sum_standalone_pp"])
        self.assertEqual(len(wf["y_ticks"]), 5)
        self.assertEqual(wf["y_ticks"][0], 0.0)
        self.assertEqual(wf["y_ticks"][-1], wf["y_max_pp"])


# ── H3: PII TTL ─────────────────────────────────────────────────────────────

class PIITTLTest(unittest.TestCase):

    def test_default_ttl_and_cache_headers(self) -> None:
        import importlib, services.report_storage as rs
        importlib.reload(rs)
        self.assertLessEqual(rs.DEFAULT_TTL_HOURS, 48)
        self.assertIn("no-store", rs.CACHE_CONTROL)
        self.assertIn("private", rs.CACHE_CONTROL)
        self.assertNotIn("public", rs.CACHE_CONTROL)


if __name__ == "__main__":
    unittest.main()
