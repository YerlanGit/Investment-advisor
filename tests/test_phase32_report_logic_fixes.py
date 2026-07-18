"""
Phase 32 — качество логики и связности DEEP-отчёта (аудит 2026-07-18).

Закрывает четыре находки владельца по живому отчёту (r/148046720):

1. **Risk-index занижен.** Композитный риск-гейдж (vol/CVaR/single-name-ERC)
   игнорировал структурные/хвостовые сигналы — 73%-tech книга с плечом и
   исторической просадкой −43.5% читалась «48 · Умеренный». Добавлены
   BOUNDED аггравторы (MaxDD · сектор · плечо), которые могут ТОЛЬКО повышать
   гейдж; чистая диверсифицированная книга без плеча не меняется (обратная
   совместимость по построению). Verdict и Effect «до/после» используют один
   набор аггравторов → согласованы.

2. **Action Plan: действие ≠ количество.** «Действие» — из 4-Pillar-скоринга,
   «Количество» — из Black-Litterman. При расхождении знака показывался
   абсурдный «+1 шт» под SELL. Теперь при противоречии qty=None + пометка.

3. **Effect: что продаётся/покупается.** Панель показывала лишь список
   тикеров; теперь — явная разбивка Продать/Купить с Δпп (маппер
   `effectActions`).
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("RAMP_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")

from finance.scoring import composite_risk_score  # noqa: E402


class CompositeAggravatorTest(unittest.TestCase):
    """composite_risk_score: bounded aggravators, backward-compatible."""

    # Live-audit inputs: vol 20.3%, CVaR −3.5% daily, max single-name ERC 16.2%.
    VOL, CVAR, ERC = 0.203, -0.035, 16.2

    def _base(self):
        return composite_risk_score(self.VOL, self.CVAR, self.ERC, "MODERATE")

    def test_backward_compatible_without_aggravators(self):
        """No aggravator args → identical to the historical 3-signal score."""
        # Pinned: the live report showed 48 for these inputs under MODERATE.
        self.assertEqual(self._base(), 48)

    def test_none_aggravators_are_noop(self):
        self.assertEqual(
            composite_risk_score(self.VOL, self.CVAR, self.ERC, "MODERATE",
                                 max_drawdown=None, leverage_ratio=None,
                                 sector_top_pct=None),
            self._base())

    def test_max_drawdown_raises_monotonically(self):
        base = self._base()
        shallow = composite_risk_score(self.VOL, self.CVAR, self.ERC, "MODERATE",
                                       max_drawdown=-0.15)   # below 20% threshold
        deep = composite_risk_score(self.VOL, self.CVAR, self.ERC, "MODERATE",
                                    max_drawdown=-0.435)
        deeper = composite_risk_score(self.VOL, self.CVAR, self.ERC, "MODERATE",
                                      max_drawdown=-0.70)
        self.assertEqual(shallow, base)            # < 20% DD → no penalty
        self.assertGreater(deep, base)
        self.assertGreaterEqual(deeper, deep)
        self.assertLessEqual(deeper - base, 15)    # capped at +15

    def test_sector_concentration_raises(self):
        base = self._base()
        self.assertEqual(                          # ≤50% → no penalty
            composite_risk_score(self.VOL, self.CVAR, self.ERC, "MODERATE",
                                 sector_top_pct=45.0), base)
        self.assertGreater(
            composite_risk_score(self.VOL, self.CVAR, self.ERC, "MODERATE",
                                 sector_top_pct=73.0), base)

    def test_leverage_raises(self):
        base = self._base()
        self.assertEqual(                          # unlevered → no penalty
            composite_risk_score(self.VOL, self.CVAR, self.ERC, "MODERATE",
                                 leverage_ratio=1.0), base)
        self.assertGreater(
            composite_risk_score(self.VOL, self.CVAR, self.ERC, "MODERATE",
                                 leverage_ratio=1.3), base)

    def test_combined_pushes_live_book_out_of_moderate(self):
        """The live book (−43.5% DD, 73% tech, 1.05x leverage) should read
        clearly higher than 48 — into the elevated/aggressive zone (≥60)."""
        score = composite_risk_score(self.VOL, self.CVAR, self.ERC, "MODERATE",
                                     max_drawdown=-0.435, sector_top_pct=73.0,
                                     leverage_ratio=1.05)
        self.assertGreaterEqual(score, 60)
        self.assertLessEqual(score, 100)

    def test_clean_book_unchanged(self):
        """A diversified, unlevered book with a shallow drawdown gets ZERO
        aggravators → identical to the base score (no false inflation)."""
        clean = composite_risk_score(0.10, -0.02, 8.0, "MODERATE",
                                     max_drawdown=-0.12, sector_top_pct=35.0,
                                     leverage_ratio=1.0)
        base  = composite_risk_score(0.10, -0.02, 8.0, "MODERATE")
        self.assertEqual(clean, base)

    def test_score_stays_bounded_0_100(self):
        hi = composite_risk_score(0.60, -0.20, 60.0, "AGGRESSIVE",
                                  max_drawdown=-0.90, sector_top_pct=100.0,
                                  leverage_ratio=3.0)
        self.assertLessEqual(hi, 100)
        self.assertGreaterEqual(hi, 0)


class EngineAggravatorWiringTest(unittest.TestCase):
    """MAC3RiskEngine._composite_risk_score forwards the aggravators."""

    def test_static_method_forwards_kwargs(self):
        from finance.investment_logic import MAC3RiskEngine
        base = MAC3RiskEngine._composite_risk_score(0.203, -0.035, 16.2, "MODERATE")
        agg = MAC3RiskEngine._composite_risk_score(
            0.203, -0.035, 16.2, "MODERATE",
            max_drawdown=-0.435, sector_top_pct=73.0, leverage_ratio=1.05)
        self.assertGreater(agg, base)


class SimulateConsistencyTest(unittest.TestCase):
    """simulate_after_plan accepts mandate + leverage and applies the same
    aggravators so effect «before» tracks the verdict gauge (no 70-vs-48 desync)."""

    def test_simulate_signature_accepts_new_kwargs(self):
        import inspect
        from finance.simulate import simulate_after_plan
        params = inspect.signature(simulate_after_plan).parameters
        self.assertIn("mandate", params)
        self.assertIn("leverage_ratio", params)

    def test_top_sector_share_helper(self):
        import numpy as np
        from finance.simulate import _top_sector_share
        tickers = ["A", "B", "C", "D"]
        w = np.array([0.4, 0.3, 0.2, 0.1])
        sec = {"A": "Technology", "B": "Technology", "C": "Health", "D": "Energy"}
        # Tech = 0.4+0.3 = 0.7 of 1.0 long book.
        self.assertAlmostEqual(_top_sector_share(tickers, w, sec), 0.7, places=6)
        # Empty → 0.0
        self.assertEqual(_top_sector_share([], np.array([]), {}), 0.0)


if __name__ == "__main__":
    unittest.main()
