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
        # L-17: реинвест-кандидат обязан иметь рейтинг Buy в ПЛАНЕ (здесь —
        # отложенный турновер-капом ряд, Δw=0) + положительный BL Δw.
        rows = self._rows() + [{"ticker": "NVDA", "action": "Buy", "delta_w_pp": 0.0}]
        bl = [{"ticker": "NVDA", "action": "Buy", "delta_w_pp": +5.0}]
        target, tickers, actions = high_priority_target_weights(cur, rows, bl)
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


class ReinvestDeconcentrationTest(unittest.TestCase):
    """2026-07-18: freed weight from selling the over-weight tech must NOT be
    poured back into more tech (old reinvest pushed IT-share and risk UP,
    contradicting «привести портфель к мандату»).  De-concentrate: buy held
    diversifiers OUTSIDE the top sector, else park as cash."""

    def _book(self):
        cur = {"ORCL": 0.16, "MSFT": 0.20, "AAOI": 0.12, "NVDA": 0.15,
               "GOOGL": 0.08, "META": 0.06, "AAPL": 0.11, "GLD": 0.06,
               "SLV": 0.05, "MRK": 0.01}
        sector = {"ORCL": "Technology", "MSFT": "Technology", "AAOI": "Technology",
                  "NVDA": "Semiconductors", "GOOGL": "Technology", "META": "Technology",
                  "AAPL": "Technology", "GLD": "Commodities", "SLV": "Commodities",
                  "MRK": "Health Care"}
        rows = [{"ticker": "ORCL", "action": "Sell", "delta_w_pp": -8.65},
                {"ticker": "AAPL", "action": "Trim", "delta_w_pp": -1.95}]
        bl = [{"ticker": "GOOGL", "action": "Buy", "delta_w_pp": 7.65},
              {"ticker": "NVDA",  "action": "Buy", "delta_w_pp": 7.62},
              {"ticker": "MRK",   "action": "Buy", "delta_w_pp": 2.0}]
        return cur, sector, rows, bl

    def _it(self, w, sector):
        return sum(v for t, v in w.items()
                   if sector.get(t) in ("Technology", "Semiconductors"))

    def test_reinvest_avoids_top_sector_when_over_concentrated(self):
        from finance.simulate import high_priority_target_weights
        cur, sector, rows, bl = self._book()
        target, _tk, actions = high_priority_target_weights(
            cur, rows, bl, sector_by_ticker=sector)
        buys = {a["ticker"] for a in actions if a["side"] == "buy"}
        # No tech / semiconductor reinvest — only diversifier (MRK) and/or cash.
        self.assertFalse({"GOOGL", "NVDA", "META"} & buys)
        # IT share must DROP toward the mandate, not rise.
        self.assertLess(self._it(target, sector), self._it(cur, sector))

    def test_cash_pseudo_move_when_no_diversifier(self):
        from finance.simulate import high_priority_target_weights
        # All held names are tech → nothing to reinvest into → freed weight cash.
        cur = {"ORCL": 0.3, "MSFT": 0.3, "NVDA": 0.4}
        sector = {"ORCL": "Technology", "MSFT": "Technology", "NVDA": "Semiconductors"}
        rows = [{"ticker": "ORCL", "action": "Sell", "delta_w_pp": -8.0}]
        bl = [{"ticker": "MSFT", "action": "Buy", "delta_w_pp": 5.0}]  # tech → blocked
        _t, _tk, actions = high_priority_target_weights(
            cur, rows, bl, sector_by_ticker=sector)
        cash = [a for a in actions if a.get("is_cash")]
        self.assertTrue(cash, "freed weight should park as cash, not buy tech")
        self.assertNotIn("MSFT", {a["ticker"] for a in actions if a["side"] == "buy"
                                  and not a.get("is_cash")})

    def test_no_sector_data_keeps_legacy_reinvest(self):
        """Without sector data we can't de-concentrate → legacy behaviour."""
        from finance.simulate import high_priority_target_weights
        cur = {"ORCL": 0.2, "NVDA": 0.2}
        rows = [{"ticker": "ORCL", "action": "Sell", "delta_w_pp": -6.0},
                # L-17: план рейтит NVDA Buy (отложен турновер-капом) —
                # только такой кандидат остаётся легитимным реинвестом.
                {"ticker": "NVDA", "action": "Buy", "delta_w_pp": 0.0}]
        bl = [{"ticker": "NVDA", "action": "Buy", "delta_w_pp": 5.0}]
        _t, _tk, actions = high_priority_target_weights(cur, rows, bl)  # no sector map
        self.assertIn("NVDA", {a["ticker"] for a in actions if a["side"] == "buy"})


class ReinvestEligibilityTest(unittest.TestCase):
    """L-13 (2026-07-19): реинвест не должен «покупать» имена, которые модель
    честно не симулирует (blocklist: sparse/broker-priced прокси, напр. AIX-нота
    FFSPC — стала Max-TRC 36.3% в live) или которые добавляют плечо (реестровые
    ETP: XNDU, CONL)."""

    def _base(self):
        cur = {"ORCL": 0.20, "FFSPC6.1028.AIX": 0.02, "XNDU": 0.02,
               "MRK": 0.05, "GLD": 0.10}
        sector = {"ORCL": "Technology", "FFSPC6.1028.AIX": "Other",
                  "XNDU": "Other", "MRK": "Health Care", "GLD": "Commodities"}
        rows = [{"ticker": "ORCL", "action": "Sell", "delta_w_pp": -8.0},
                # L-17: у легитимного кандидата (MRK) есть план-рейтинг Buy
                # (отложен турновер-капом); FFSPC/XNDU план рейтит HOLD —
                # они отваливаются и по conviction-гейту, и по блок-листам.
                {"ticker": "MRK",  "action": "Buy",  "delta_w_pp": 0.0},
                {"ticker": "FFSPC6.1028.AIX", "action": "Hold", "delta_w_pp": 0.0},
                {"ticker": "XNDU", "action": "Hold", "delta_w_pp": 0.0}]
        bl = [{"ticker": "FFSPC6.1028.AIX", "action": "Buy", "delta_w_pp": 9.0},
              {"ticker": "XNDU", "action": "Buy", "delta_w_pp": 8.0},
              {"ticker": "MRK",  "action": "Buy", "delta_w_pp": 2.0}]
        return cur, sector, rows, bl

    def test_blocklist_and_leveraged_excluded(self):
        from finance.simulate import high_priority_target_weights
        cur, sector, rows, bl = self._base()
        _t, _tk, actions = high_priority_target_weights(
            cur, rows, bl, sector_by_ticker=sector,
            reinvest_blocklist={"FFSPC6.1028.AIX"})
        buys = {a["ticker"] for a in actions
                if a["side"] == "buy" and not a.get("is_cash")}
        self.assertNotIn("FFSPC6.1028.AIX", buys)   # blocklist (вне модели)
        self.assertNotIn("XNDU", buys)              # leveraged registry
        self.assertIn("MRK", buys)                  # честный диверсификатор

    def test_all_ineligible_goes_to_cash(self):
        from finance.simulate import high_priority_target_weights
        cur, sector, rows, bl = self._base()
        bl2 = [b for b in bl if b["ticker"] != "MRK"]   # остались только непригодные
        _t, _tk, actions = high_priority_target_weights(
            cur, rows, bl2, sector_by_ticker=sector,
            reinvest_blocklist={"FFSPC6.1028.AIX"})
        self.assertTrue(any(a.get("is_cash") for a in actions))
        self.assertFalse({"FFSPC6.1028.AIX", "XNDU"} &
                         {a["ticker"] for a in actions if a["side"] == "buy"
                          and not a.get("is_cash")})


class UnderlyingClassificationTest(unittest.TestCase):
    """L-14 (2026-07-19): плечевые ETP классифицируются по UNDERLYING —
    CONL (2× Coinbase) = крипто-экспозиция для мандат-панели."""

    def test_conl_is_crypto(self):
        from agent.gatekeeper import _classify_to_asset_key
        self.assertEqual(_classify_to_asset_key("CONL"), "Crypto")
        self.assertEqual(_classify_to_asset_key("CONL.US"), "Crypto")

    def test_plain_names_unchanged(self):
        from agent.gatekeeper import _classify_to_asset_key
        self.assertEqual(_classify_to_asset_key("AAPL"), "Stocks_US")
        self.assertEqual(_classify_to_asset_key("GLD"), "Commodities")
        self.assertEqual(_classify_to_asset_key("TLT"), "Bonds")
        self.assertEqual(_classify_to_asset_key("KSPI.KZ"), "Stocks_KZ")
        self.assertEqual(_classify_to_asset_key("USD"), "Cash")

    def test_mandate_guard_blocks_conl_for_conservative(self):
        """Мандатный гард идей теперь ловит и плечевые крипто-обёртки."""
        from ai_narrative import _remove_mandate_banned_picks
        picks = {"boost_alpha": {"picks": [{"ticker": "CONL"}, {"ticker": "NVDA"}]}}
        out = _remove_mandate_banned_picks(
            picks, {"limits_dict": {"Crypto": [0, 0]}})
        self.assertEqual([p["ticker"] for p in out["boost_alpha"]["picks"]],
                         ["NVDA"])


class AiStyleRuleTest(unittest.TestCase):
    """L-16 (2026-07-19): анти-шаблон и запрет «счёт»/«ярлык» в обоих промптах."""

    def test_style_rule_in_both_tiers(self):
        from ai_narrative import _user_prompt
        for tier in ("base", "deep"):
            p = _user_prompt({}, tier=tier)
            self.assertIn("БЕЗ ПОВТОРОВ И ЖАРГОНА", p)
            self.assertIn("сводная оценка 4-Pillar", p)
            self.assertIn("оценка фазы цикла", p)


class ForwardSharpeTest(unittest.TestCase):
    """2026-07-18: the Effect Sharpe is FORWARD (er/vol), not historical replay,
    so it can't triple while risk rises (look-ahead)."""

    def _sim(self, target_weights, er_map, headline_sharpe=0.69):
        import numpy as np
        import pandas as pd
        from finance.simulate import simulate_after_plan
        tickers = ["A", "B", "C"]
        idx = pd.date_range("2024-01-01", periods=300, freq="B")
        rng = np.random.default_rng(3)
        rets = pd.DataFrame({t: rng.normal(0.0004, 0.012, 300) for t in tickers}, index=idx)
        cov = (rets.cov() * 252)
        perf = pd.DataFrame({"Ticker": tickers, "Current_Value": [500, 300, 200],
                             "Fundamental_Sector": ["Technology", "Health Care", "Energy"]})
        bl = [{"ticker": t, "delta_w_pp": 0.0, "posterior_mu": er_map[t],
               "n_views": 1, "current_w": w, "target_w": w}
              for t, w in [("A", 0.5), ("B", 0.3), ("C", 0.2)]]
        return simulate_after_plan(
            perf_df=perf, risk_matrix=cov, daily_log_returns=rets, bl_records=bl,
            current_metrics={"Sharpe_Ratio": headline_sharpe, "CVaR_95_Daily": -0.03,
                             "Max_Drawdown": -0.2},
            risk_free_rate=0.045, target_weights=target_weights,
            sector_by_ticker={"A": "Technology", "B": "Health Care", "C": "Energy"})

    def test_sharpe_delta_is_bounded_and_anchored(self):
        # Even with a wildly optimistic BL μ on the target, the displayed Sharpe
        # must stay anchored to the headline and the delta bounded (no ×3 jump).
        res = self._sim({"A": 0.2, "B": 0.5, "C": 0.3},
                        er_map={"A": 0.05, "B": 0.9, "C": 0.06})
        self.assertIsNotNone(res)
        sh = res["metrics"]["sharpe"]
        self.assertAlmostEqual(sh["before"], 0.69, delta=0.01)   # anchored to headline
        self.assertLessEqual(abs(sh["after"] - sh["before"]), 0.6 + 1e-6)  # bounded delta


if __name__ == "__main__":
    unittest.main()


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
        self.assertIn("СЛАБЫЙ АГРЕГАТНЫЙ ЯРЛЫК", p)
        self.assertIn("8%", p)
        # Honest, confident read of the ACTUAL signals — not «ignore the regime».
        self.assertIn("читай их УВЕРЕННО", p)
        self.assertIn("НЕ в самих сигналах", p)

    def test_high_confidence_no_softening_rule(self):
        p = self._prompt(74)
        self.assertNotIn("СЛАБЫЙ АГРЕГАТНЫЙ ЯРЛЫК", p)

    def test_threshold_boundary(self):
        self.assertNotIn("СЛАБЫЙ АГРЕГАТНЫЙ ЯРЛЫК", self._prompt(25))   # 25 = floor, not < 25
        self.assertIn("СЛАБЫЙ АГРЕГАТНЫЙ ЯРЛЫК", self._prompt(24))
        # Missing confidence → no crash, no rule.
        from ai_narrative import _user_prompt
        p = _user_prompt({"regime": {"regime": "Expansion"}}, tier="deep")
        self.assertNotIn("СЛАБЫЙ АГРЕГАТНЫЙ ЯРЛЫК", p)


if __name__ == "__main__":
    unittest.main()
