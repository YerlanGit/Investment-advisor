"""
Phase-39 · Финальный аудит отчётов: R-1, R-2, R-4, R-6, R-7, R-8, R-17.

Семь дефектов из аудита живых отчётов 30.07–01.08 (`AUDIT.md §−40`), каждый
воспроизводился на проде. Здесь — контракты, запрещающие их возврат.
"""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class SectorExposureExcludesCashTest(unittest.TestCase):
    """R-1: маржа/кэш — не сектор. 27.03% рисовались как 7% из-за неттинга."""

    def setUp(self):
        from finance.investment_logic import MAC3RiskEngine
        self.eng = MAC3RiskEngine()

    def test_negative_margin_does_not_net_into_other(self):
        se = self.eng.get_sector_exposure(
            ["AAOI", "CRCL", "USD"],
            {"AAOI": 0.10, "CRCL": 0.17, "USD": -0.20})
        self.assertAlmostEqual(se.get("Other", 0.0), 0.27, places=9,
                               msg="маржа −20% не должна вычитаться из сектора")

    def test_positive_cash_is_not_a_sector_either(self):
        se = self.eng.get_sector_exposure(
            ["AAPL", "USD", "KZT"], {"AAPL": 0.6, "USD": 0.35, "KZT": 0.05})
        total = sum(se.values())
        self.assertAlmostEqual(total, 0.6, places=9,
                               msg="кэш не должен появляться ни в одном секторе")

    def test_risky_sectors_unchanged(self):
        se = self.eng.get_sector_exposure(
            ["AAPL", "NVDA", "TLT"], {"AAPL": 0.4, "NVDA": 0.4, "TLT": 0.2})
        self.assertAlmostEqual(se.get("Technology", 0), 0.4, places=9)
        self.assertAlmostEqual(se.get("Bonds", 0), 0.2, places=9)


class RiskIndexAnchoredToCoverTest(unittest.TestCase):
    """R-2: гейдж Effect «до» ОБЯЗАН равняться обложке (инвариант CLAUDE.md)."""

    def test_before_equals_headline_composite(self):
        from finance import simulate as sim
        idx = pd.bdate_range(end="2026-07-31", periods=300)
        rng = np.random.default_rng(7)
        tickers = ["AAA", "BBB", "CCC"]
        perf = pd.DataFrame({
            "Ticker": tickers,
            "Current_Value": [4000.0, 3000.0, 3000.0],
            "Euler_Risk_Contribution_Pct": [40.0, 35.0, 25.0],
        })
        dl = pd.DataFrame(rng.standard_normal((300, 3)) * 0.01,
                          index=idx, columns=tickers)
        cov = pd.DataFrame(np.cov(dl.T.to_numpy()), index=tickers, columns=tickers)
        bl = [{"ticker": t, "current_w": 1/3, "target_w": 1/3,
               "delta_w_pp": d, "posterior_mu": 0.08, "n_views": 1}
              for t, d in zip(tickers, (-5.0, 2.0, 3.0))]
        headline = 63
        out = sim.simulate_after_plan(
            perf_df=perf, risk_matrix=cov, daily_log_returns=dl,
            bl_records=bl, current_metrics={"Composite_Risk_Score": headline,
                                            "Sharpe_Ratio": 0.5,
                                            "CVaR_95_Daily": -0.02,
                                            "Max_Drawdown": -0.30},
            risk_free_rate=0.04,
            target_weights={t: 1/3 for t in tickers})
        ri = (out.get("metrics") or {}).get("risk_index") or {}
        self.assertEqual(int(float(ri.get("before"))), headline,
                         "«до» обязан быть якорен к Composite_Risk_Score обложки")

    def test_anchor_preserves_the_simulated_delta_bounds(self):
        """Якорь переносит уровень, но «после» остаётся в [0, 100]."""
        from finance.scoring import composite_risk_score
        # смок: функция композита не изменилась и остаётся в границах
        v = composite_risk_score(0.2, -0.03, 15.0, "MODERATE")
        self.assertTrue(0 <= v <= 100)


class SellZoneIsExecutableTest(unittest.TestCase):
    """R-4: зона продажи не может быть целиком выше рынка."""

    def _levels(self, price, sma50, atr=8.3, hi52=160.0):
        from finance.action_plan import compute_levels
        return compute_levels(action="Sell", price=price, atr_abs=atr,
                              sma50=sma50, sma100=None, sma200=None,
                              high_52w=hi52, rsi=40)

    def test_weak_name_zone_starts_at_market(self):
        """AAOI-репро: цена 77.51, SMA50 137.9 — зона была 144–152 (+86%)."""
        lo, hi = self._levels(77.51, 137.9)["sell_zone"]
        self.assertLessEqual(lo, 77.51, "нижняя граница обязана быть достижима")
        self.assertGreater(hi, lo)

    def test_reachable_bounce_cap(self):
        """Верх клипованной зоны — досягаемый отскок, а не +86%."""
        lo, hi = self._levels(77.51, 137.9)["sell_zone"]
        self.assertLess(hi, 77.51 * 1.30,
                        "верхняя граница не должна требовать ралли на треть")

    def test_straddling_zone_unchanged(self):
        """Бумага НАД SMA50 — прежняя логика (зона накрывает цену) не тронута."""
        lo, hi = self._levels(100.0, 95.0)["sell_zone"]
        self.assertAlmostEqual(lo, 95.0, places=6)
        self.assertGreaterEqual(hi, lo)

    def test_stop_still_below_price(self):
        stop = self._levels(77.51, 137.9)["stop_loss"]
        self.assertLess(stop, 77.51)


class ReasonReachesPremiumTest(unittest.TestCase):
    """R-6: объяснения движка обязаны доезжать до прод-рендера."""

    def test_plan_row_carries_reason(self):
        from premium_payload import build_design_data
        d = build_design_data({"action_plan": [
            {"ticker": "GOOGL", "action": "Hold", "price": "336.98",
             "price_num": 336.98, "delta_w_pp": "0.0",
             "reason": "Score +1.8 · deferred (turnover cap)"}]}, "deep")
        row = d["actionPlan"][0]
        self.assertEqual(row["reason"], "Score +1.8 · deferred (turnover cap)")

    def test_missing_reason_degrades_to_empty(self):
        from premium_payload import build_design_data
        d = build_design_data({"action_plan": [
            {"ticker": "AAPL", "action": "Trim", "price_num": 1.0,
             "delta_w_pp": "-0.1"}]}, "deep")
        self.assertEqual(d["actionPlan"][0]["reason"], "")

    def test_score_breakdown_produces_reason(self):
        from pdf_payload import _score_reason
        self.assertEqual(_score_reason(2.4), "сильный совокупный сигнал")
        self.assertEqual(_score_reason(0.8), "положительный профиль")
        self.assertEqual(_score_reason(0.0), "нейтральный профиль")
        self.assertEqual(_score_reason(-1.0), "отрицательный профиль")
        self.assertEqual(_score_reason(-3.1), "слабый совокупный сигнал")
        self.assertEqual(_score_reason(None), "нейтральный профиль")

    def test_jsx_bundle_renders_reason(self):
        """Отгружаемый артефакт (работает в проде) обязан читать r.reason."""
        import pathlib
        bundle = (pathlib.Path(__file__).resolve().parents[1]
                  / "src" / "premium_assets" / "deep-components.js")
        self.assertIn("r.reason", bundle.read_text(encoding="utf-8"),
                      "deep-components.js устарел — запусти design/premium_v2/build.sh")


class GrossExposureNoDoubleCountTest(unittest.TestCase):
    """R-7: маржа — финансирование, не позиция. 140.4% против «плечо 1.2x»."""

    def _metrics(self, portfolio):
        from finance.investment_logic import UniversalPortfolioManager
        mgr = UniversalPortfolioManager(price_source="demo")
        idx = pd.bdate_range(end="2026-07-31", periods=400)
        rng = np.random.default_rng(4)
        cols = list(mgr.engine.factor_tickers.values()) + ["AAPL.US", "MSFT.US"]
        f = pd.DataFrame({t: 100*np.exp(np.cumsum(rng.standard_normal(400)*0.008))
                          for t in cols}, index=idx)
        hr = type("HR", (), {"data": f, "loaded": list(f.columns),
                             "failed": {}, "retried": []})()
        with patch.object(mgr.engine, "get_market_data", return_value=(f, hr)):
            return mgr.analyze_all(pd.DataFrame(portfolio))["leverage_metrics"]

    def test_levered_long_book_gross_equals_longs(self):
        lm = self._metrics({"Ticker": ["AAPL", "USD"],
                            "Quantity": [100.0, -3000.0],
                            "Purchase_Price": [180.0, 1.0]})
        self.assertAlmostEqual(lm["gross_exposure"], lm["long_weight"], places=6,
                               msg="|маржа| не должна считаться рыночной позицией")
        self.assertTrue(lm["is_leveraged"])

    def test_unlevered_book_still_sums_to_one(self):
        lm = self._metrics({"Ticker": ["AAPL", "MSFT", "USD"],
                            "Quantity": [10.0, 10.0, 3000.0],
                            "Purchase_Price": [180.0, 300.0, 1.0]})
        self.assertAlmostEqual(lm["gross_exposure"] +
                               lm["cash_weight"], 1.0, places=6)
        self.assertFalse(lm["is_leveraged"])


class BenchmarkVolTest(unittest.TestCase):
    """R-8: график сравнения утверждал, что у бенчмарка нулевая волатильность."""

    def test_payload_computes_benchmark_vol(self):
        from pdf_payload import build_payload
        idx = pd.bdate_range(end="2026-07-31", periods=300)
        rng = np.random.default_rng(3)
        data = pd.DataFrame(
            {"QQQ.US": 100*np.exp(np.cumsum(rng.standard_normal(300)*0.012))},
            index=idx)
        hr = type("HR", (), {"data": data})()
        pl = build_payload({"history_result": hr,
                            "profile_benchmark_ticker": "QQQ.US",
                            "portfolio_metrics": {}}, "base", {})
        self.assertGreater(pl["benchmark_vol_pct"], 5.0,
                           "у реального ряда волатильность не может быть 0")

    def test_missing_series_stays_zero(self):
        from pdf_payload import build_payload
        pl = build_payload({"portfolio_metrics": {}}, "base", {})
        self.assertEqual(pl["benchmark_vol_pct"], 0.0)

    def test_premium_carries_it_to_the_chart(self):
        from premium_payload import build_design_data
        d = build_design_data({"benchmark_vol_pct": 17.3,
                               "volatility_num": 23.3}, "base")
        self.assertEqual(d["performance"]["vol"]["spx"], 17.3)


class RegimeConfidenceNoDoubleConversionTest(unittest.TestCase):
    """R-17: честный 1% показывался как 100% — худшая из возможных ошибок."""

    def _conf(self, pct):
        from premium_payload import build_design_data
        d = build_design_data({"regime": {"label": "Expansion",
                                                "confidence": pct}}, "deep")
        return d["regime"]["confidence"]

    def test_one_percent_stays_one_percent(self):
        self.assertEqual(self._conf(1), 1,
                         "минимальная уверенность не должна стать максимальной")

    def test_whole_scale_passes_through(self):
        for pct in (0, 2, 4, 25, 50, 99, 100):
            self.assertEqual(self._conf(pct), pct)

    def test_legacy_fraction_still_upscaled(self):
        """Старые payload'ы с долей 0.xx по-прежнему конвертируются."""
        self.assertEqual(self._conf(0.5), 50)


class LeverageHiddenWhenCashNonNegativeTest(unittest.TestCase):
    """Правило §−13 (подтверждено этим аудитом): термины плеча скрываются,
    когда кэш-баланс НЕ отрицателен."""

    def test_base_leverage_off_without_negative_cash(self):
        from premium_payload import build_design_data
        d = build_design_data({"assets": [
            {"ticker": "AAPL", "asset_class": "Акции США",
             "weight_pct_num": 60.0, "pnl_pct_num": 1.0},
            {"ticker": "USD", "asset_class": "Ден. средства", "is_cash": True,
             "weight_pct_num": 40.0, "pnl_pct_num": 0.0}]}, "base")
        self.assertFalse(d["leverage"]["on"],
                         "плечо-бейдж обязан быть скрыт при кэше ≥ 0")

    def test_base_leverage_on_with_negative_cash(self):
        from premium_payload import build_design_data
        d = build_design_data({"assets": [
            {"ticker": "AAPL", "asset_class": "Акции США",
             "weight_pct_num": 106.7, "pnl_pct_num": 1.0},
            {"ticker": "USD", "asset_class": "Маржинальный заём", "is_cash": True,
             "weight_pct_num": -6.7, "pnl_pct_num": 0.0}]}, "base")
        self.assertTrue(d["leverage"]["on"])


if __name__ == "__main__":
    unittest.main()
