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


class EffectSideFromActionTest(unittest.TestCase):
    """2026-07-18: the Effect «Продать/Купить» side must follow the 4-Pillar
    ACTION, not the Black-Litterman Δw sign — else a TRIM/SELL name with a
    positive BL Δw was shown (and simulated) as a BUY, contradicting the
    Action Plan chip and the AI comment."""

    def _rows(self):
        # MSFT: TRIM but BL +3.37 ; AAOI: SELL but BL +0.38 ; ORCL: SELL −8.65
        return [
            {"ticker": "ORCL", "action": "Sell", "delta_w_pp": -8.65},
            {"ticker": "MSFT", "action": "Trim", "delta_w_pp": +3.37},
            {"ticker": "AAOI", "action": "Sell", "delta_w_pp": +0.38},
        ]

    def test_side_and_move_follow_action_not_bl_sign(self):
        from finance.simulate import high_priority_target_weights
        cur = {"ORCL": 0.16, "MSFT": 0.20, "AAOI": 0.12, "NVDA": 0.15}
        # A held BL buy NOT in the plan → eligible for reinvest of freed weight.
        bl = [{"ticker": "NVDA", "action": "Buy", "delta_w_pp": +5.0}]
        target, tickers, actions = high_priority_target_weights(cur, self._rows(), bl)
        by = {a["ticker"]: a for a in actions}
        # All three plan names are SELL/TRIM → side sell, delta negative, and the
        # simulated target weight goes DOWN (not up, as BL sign would have done).
        for t in ("ORCL", "MSFT", "AAOI"):
            self.assertEqual(by[t]["side"], "sell", t)
            self.assertLess(by[t]["delta_pp"], 0, t)
            self.assertLess(target[t], cur[t], t)
        # MSFT magnitude preserved from |BL Δw| but sign flipped to the action.
        self.assertAlmostEqual(by["MSFT"]["delta_pp"], -3.37, places=2)
        # Freed weight reinvested into the held BL buy (NVDA) → a real buy side.
        self.assertIn("NVDA", by)
        self.assertEqual(by["NVDA"]["side"], "buy")

    def test_genuine_buy_action_stays_buy(self):
        from finance.simulate import high_priority_target_weights
        cur = {"ORCL": 0.16, "GOOGL": 0.05}
        rows = [
            {"ticker": "ORCL",  "action": "Sell", "delta_w_pp": -6.0},
            {"ticker": "GOOGL", "action": "Buy",  "delta_w_pp": +4.0},
        ]
        _t, _tk, actions = high_priority_target_weights(cur, rows, None)
        by = {a["ticker"]: a for a in actions}
        self.assertEqual(by["GOOGL"]["side"], "buy")
        self.assertGreater(by["GOOGL"]["delta_pp"], 0)


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

    def test_top_sector_share_uses_supergroup(self):
        """Technology + Semiconductors combine into the tech-complex — the
        aggravator must see the CORRELATED 73%, not the biggest single sector."""
        import numpy as np
        from finance.simulate import _top_sector_share
        tickers = ["A", "B", "C"]
        w = np.array([0.59, 0.14, 0.27])
        sec = {"A": "Technology", "B": "Semiconductors", "C": "Fixed Income"}
        self.assertAlmostEqual(_top_sector_share(tickers, w, sec), 0.73, places=6)


class SectorConcentrationSSOTTest(unittest.TestCase):
    """asset_taxonomy.top_sector_concentration_pct — one SSOT for the combined
    tech figure shared by the report headline AND the risk gauge."""

    def test_supergroup_beats_single_sector(self):
        from finance.asset_taxonomy import top_sector_concentration_pct
        se = {"Technology": 0.59, "Semiconductors": 0.14, "Fixed Income": 0.12,
              "Commodities": 0.08, "Communication": 0.07}
        # 59% single vs 73% Tech-complex → the complex wins.
        self.assertAlmostEqual(top_sector_concentration_pct(se), 73.0, places=1)

    def test_empty_and_single(self):
        from finance.asset_taxonomy import top_sector_concentration_pct
        self.assertEqual(top_sector_concentration_pct({}), 0.0)
        self.assertAlmostEqual(
            top_sector_concentration_pct({"Health": 0.6, "Energy": 0.4}), 60.0, places=1)

    def test_gauge_uses_complex(self):
        from finance.scoring import composite_risk_score
        single = composite_risk_score(0.203, -0.035, 16.2, "MODERATE",
                                      max_drawdown=-0.435, sector_top_pct=59.0,
                                      leverage_ratio=1.05)
        complex_ = composite_risk_score(0.203, -0.035, 16.2, "MODERATE",
                                        max_drawdown=-0.435, sector_top_pct=73.0,
                                        leverage_ratio=1.05)
        self.assertGreater(complex_, single)
        self.assertGreaterEqual(complex_, 66)   # tech-complex → «Агрессивный» band


class RegimeConfidenceSofteningTest(unittest.TestCase):
    """2026-07-18: at very low regime confidence (<25%) the prompt must tell
    the model NOT to base recommendations/ideas on the regime label."""

    def _prompt(self, conf_pct):
        from ai_narrative import _user_prompt
        summary = {"regime": {"regime": "Expansion", "confidence_pct": conf_pct}}
        return _user_prompt(summary, tier="deep")

    def test_low_confidence_injects_softening_rule(self):
        p = self._prompt(8)
        self.assertIn("СЛАБЫЙ СИГНАЛ", p)
        self.assertIn("8%", p)
        self.assertIn("НЕ делай режим ОСНОВОЙ", p)

    def test_high_confidence_no_softening_rule(self):
        p = self._prompt(74)
        self.assertNotIn("СЛАБЫЙ СИГНАЛ", p)

    def test_threshold_boundary(self):
        self.assertNotIn("СЛАБЫЙ СИГНАЛ", self._prompt(25))   # 25 = floor, not < 25
        self.assertIn("СЛАБЫЙ СИГНАЛ", self._prompt(24))
        # Missing confidence → no crash, no rule.
        from ai_narrative import _user_prompt
        p = _user_prompt({"regime": {"regime": "Expansion"}}, tier="deep")
        self.assertNotIn("СЛАБЫЙ СИГНАЛ", p)


if __name__ == "__main__":
    unittest.main()
