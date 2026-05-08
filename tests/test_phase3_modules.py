"""
Phase 3 regression tests — CDS feed, Black-Litterman, Action Plan.

Network calls (FRED, WGB) are NOT exercised here; we monkey-patch the
free-layer providers with deterministic fixtures so the tests are fast and
hermetic.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── 3.1  CDS feed + QualityGate ──────────────────────────────────────────────

class CDSQualityGateTest(unittest.TestCase):

    def test_rejects_out_of_range(self) -> None:
        from finance.cds_feed import CDSQualityGate
        g = CDSQualityGate()
        ok, q, _ = g.validate(bps=0.5, timestamp=datetime.now(timezone.utc))
        self.assertFalse(ok); self.assertEqual(q, "C")
        ok, q, _ = g.validate(bps=5000, timestamp=datetime.now(timezone.utc))
        self.assertFalse(ok); self.assertEqual(q, "C")

    def test_rejects_stale(self) -> None:
        from finance.cds_feed import CDSQualityGate
        g = CDSQualityGate()
        ok, _, reason = g.validate(
            bps=120, timestamp=datetime.now(timezone.utc) - timedelta(days=10),
        )
        self.assertFalse(ok)
        self.assertIn("stale", reason)

    def test_downgrades_on_disagreement(self) -> None:
        from finance.cds_feed import CDSQualityGate
        g = CDSQualityGate()
        ok, q, _ = g.validate(
            bps=100, timestamp=datetime.now(timezone.utc),
            alternative_bps=200, base_quality="A",
        )
        self.assertTrue(ok)        # value still passes range/staleness
        self.assertEqual(q, "B")    # but quality downgraded one letter


class CDSFeedCacheTest(unittest.TestCase):
    """Sanity check the CDSFeed can be instantiated with a temp cache and
    survives provider failures gracefully."""

    def test_no_provider_returns_none(self) -> None:
        from finance.cds_feed import CDSFeed
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["CDS_CACHE_PATH"] = os.path.join(tmp, "cache.sqlite")
            feed = CDSFeed(cache_path=os.environ["CDS_CACHE_PATH"])
            # Replace providers with one that always returns None.
            feed._providers = [("noop", lambda t: None)]
            self.assertIsNone(feed.get_spread("AAPL"))

    def test_cache_returns_after_first_hit(self) -> None:
        from finance.cds_feed import CDSFeed, CDSPoint
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "c.sqlite")
            feed = CDSFeed(cache_path=cache)
            now = datetime.now(timezone.utc)
            def _provider(t: str):
                return CDSPoint(ticker=t, bps=120.0, source="test",
                                 timestamp=now, quality="B", change_7d=0.05)
            feed._providers = [("test", _provider)]
            p1 = feed.get_spread("XYZ")
            self.assertIsNotNone(p1)
            self.assertEqual(p1.bps, 120.0)
            # Replace provider with raising one — cache must still serve.
            feed._providers = [("boom", lambda t: (_ for _ in ()).throw(RuntimeError("no")))]
            p2 = feed.get_spread("XYZ")
            self.assertIsNotNone(p2)
            self.assertEqual(p2.bps, 120.0)


# ── 3.2  Black-Litterman ─────────────────────────────────────────────────────

class BlackLittermanTest(unittest.TestCase):

    def test_no_views_returns_prior(self) -> None:
        from finance.black_litterman import black_litterman
        cov = np.array([[0.04, 0.01], [0.01, 0.05]])
        res = black_litterman(
            cov=cov, tickers=["A", "B"],
            current_weights={"A": 0.6, "B": 0.4},
        )
        # Without views the target should be ≈ current (within turnover noise).
        self.assertEqual(set(res.tickers), {"A", "B"})
        self.assertAlmostEqual(float(res.target_weights.sum()), 1.0, places=6)
        np.testing.assert_allclose(res.target_weights, res.current_weights, atol=1e-3)

    def test_positive_view_pulls_target_up(self) -> None:
        from finance.black_litterman import black_litterman
        cov = np.array([[0.04, 0.01], [0.01, 0.05]])
        # Relative view: A outperforms B by +10% annualised, high confidence.
        # Relative views are the canonical BL semantics — absolute Q values
        # have to compete with the equilibrium prior π = δ·Σ·w which already
        # encodes a meaningful expected return.
        P = np.array([[1.0, -1.0]])
        Q = np.array([0.10])
        conf = np.array([0.9])
        res = black_litterman(
            cov=cov, tickers=["A", "B"],
            current_weights={"A": 0.5, "B": 0.5},
            views_P=P, views_Q=Q, view_confidence=conf,
        )
        self.assertGreater(res.target_weights[0], 0.5)   # A up
        self.assertLess(res.target_weights[1],    0.5)   # B down
        self.assertAlmostEqual(float(res.target_weights.sum()), 1.0, places=6)
        # Active share respects the soft cap.
        self.assertLessEqual(float(np.abs(res.delta_weights).sum() / 2.0), 0.25 + 1e-6)

    def test_views_from_scores(self) -> None:
        from finance.black_litterman import views_from_scores
        scores = {
            "A": {"total":  4.0},
            "B": {"total": -3.0},
            "C": {"total":  0.0},
        }
        P, Q, conf = views_from_scores(scores, ["A", "B", "C"])
        self.assertEqual(P.shape, (2, 3))
        self.assertEqual(Q.shape, (2,))
        self.assertEqual(conf.shape, (2,))
        # B's view direction must be negative.
        b_index = int(np.argmax(P[:, 1]))
        self.assertLess(Q[b_index], 0.0)


# ── 3.3  Action Plan ────────────────────────────────────────────────────────

class ActionPlanLevelsTest(unittest.TestCase):

    def test_buy_levels_anchored_to_sma_and_atr(self) -> None:
        from finance.action_plan import compute_levels
        out = compute_levels(
            action="Buy", price=200.0, atr_abs=4.0,
            sma50=205.0, sma100=210.0, sma200=190.0,
            high_52w=220.0, rsi=55.0, macd_below_zero=False,
        )
        # Buy zone within [SMA50 - 1·ATR, SMA50] = [201, 205]
        lo, hi = out["buy_zone"]
        self.assertAlmostEqual(lo, 201.0, places=6)
        self.assertAlmostEqual(hi, 205.0, places=6)
        # Take target ≥ price + 3·ATR = 212
        self.assertGreaterEqual(out["take_target"], 212.0)
        # Stop ≥ max(price - 2·ATR=192, SMA200=190) = 192
        self.assertAlmostEqual(out["stop_loss"], 192.0, places=6)

    def test_hot_rsi_shifts_buy_zone_lower(self) -> None:
        from finance.action_plan import compute_levels
        cool = compute_levels(
            action="Buy", price=200.0, atr_abs=4.0,
            sma50=205.0, sma100=210.0, sma200=190.0,
            high_52w=220.0, rsi=50.0, macd_below_zero=False,
        )
        hot = compute_levels(
            action="Buy", price=200.0, atr_abs=4.0,
            sma50=205.0, sma100=210.0, sma200=190.0,
            high_52w=220.0, rsi=80.0, macd_below_zero=False,
        )
        self.assertLess(hot["buy_zone"][0], cool["buy_zone"][0])

    def test_trim_emits_sell_zone_only(self) -> None:
        from finance.action_plan import compute_levels
        out = compute_levels(
            action="Trim", price=180.0, atr_abs=3.0,
            sma50=175.0, sma100=170.0, sma200=160.0,
            high_52w=190.0, rsi=72.0, macd_below_zero=False,
        )
        self.assertIsNone(out["buy_zone"])
        self.assertIsNotNone(out["sell_zone"])
        self.assertIsNotNone(out["stop_loss"])

    def test_hold_emits_only_stop(self) -> None:
        from finance.action_plan import compute_levels
        out = compute_levels(
            action="Hold", price=120.0, atr_abs=2.0,
            sma50=118.0, sma100=115.0, sma200=110.0,
            high_52w=130.0, rsi=55.0, macd_below_zero=False,
        )
        self.assertIsNone(out["buy_zone"])
        self.assertIsNone(out["sell_zone"])
        self.assertIsNone(out["take_target"])
        self.assertIsNotNone(out["stop_loss"])


class ActionPlanBuilderTest(unittest.TestCase):

    def test_builder_respects_turnover_cap(self) -> None:
        from finance.action_plan import build_action_plan, MAX_TRADE_BLOCK_PORTFOLIO_PCT
        # Build a tiny perf table.
        perf = pd.DataFrame([
            {"Ticker": "A", "Current_Price": 100.0, "Quantity": 10, "ATR_Absolute": 2.0},
            {"Ticker": "B", "Current_Price": 50.0,  "Quantity": 20, "ATR_Absolute": 1.0},
            {"Ticker": "C", "Current_Price": 25.0,  "Quantity": 40, "ATR_Absolute": 0.5},
        ])
        scores = {
            "A": {"action": "Strong Buy", "total": +4, "hotspot": False},
            "B": {"action": "Trim",       "total": -2, "hotspot": True},
            "C": {"action": "Sell",       "total": -3, "hotspot": False},
        }
        # Each BL view requests 15pp shift → cumulative 45pp > 25% cap.
        bl_records = [
            {"ticker": "A", "delta_w_pp": +15.0, "current_w": 0.30, "target_w": 0.45, "posterior_mu": 0.10},
            {"ticker": "B", "delta_w_pp": -15.0, "current_w": 0.40, "target_w": 0.25, "posterior_mu": -0.05},
            {"ticker": "C", "delta_w_pp": -15.0, "current_w": 0.30, "target_w": 0.15, "posterior_mu": -0.04},
        ]
        rows = build_action_plan(
            perf_table=perf, asset_scores=scores, technicals_map={},
            bl_records=bl_records, portfolio_value=10_000.0,
        )
        # First two rows should keep their non-zero delta (cumulative 30pp >
        # 25pp cap, so the 3rd one is demoted to Hold).
        self.assertEqual(rows[0].action, "Sell")        # priority sells first
        last = rows[-1]
        self.assertEqual(last.delta_w_pp, 0.0)         # demoted
        self.assertIn("turnover cap", last.reason)
        # Cumulative |delta| of executed rows ≤ 25%.
        executed = sum(abs(r.delta_w_pp) / 100.0 for r in rows if r.delta_w_pp != 0.0)
        self.assertLessEqual(executed, MAX_TRADE_BLOCK_PORTFOLIO_PCT + 1e-9)


if __name__ == "__main__":
    unittest.main()
