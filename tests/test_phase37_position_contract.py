"""
Phase-37 · Контракт входного портфеля: склейка дубликатов + гард количества.

Снимает два дефекта сквозного аудита (`AUDIT_360_2026-07-30.md §2.1/§2.2`) и
закрывает предпосылку Фазы 4: `PHASE_04 §3.2` требует «дубликат тикера →
суммирование, средневзвешенная цена покупки, результат склейки показывается
пользователю» и МОЛЧА предполагает, что склейка уже есть. Её не было.

A-1. Дубликат позиции (два лота, два счёта, один ISIN на двух биржах) доезжал до
     `a_data.columns = valid_originals`, давал две одноимённые колонки, и Ridge
     падал с `DuplicateError` из недр sklearn — отчёт не строился ВООБЩЕ.

A-2. `Quantity = NaN` не роняло отчёт: оно протекало в веса и делало `NaN` ВСЕ
     головные метрики при внешне нормальной стоимости портфеля. Хуже падения —
     падение хотя бы видно.

Склейка живёт в ДВИЖКЕ, а не в парсере ручного ввода: дефект есть и на
freedom-пути, а парсер Фазы 4 должен опираться на общий контракт, а не заводить
свою копию правил.
"""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from finance.investment_logic import UniversalPortfolioManager

warnings.filterwarnings("ignore")


def _run(portfolio, extra, last=None):
    mgr = UniversalPortfolioManager(price_source="demo")
    eng = mgr.engine
    idx = pd.bdate_range(end="2026-07-31", periods=400)
    rng = np.random.default_rng(4)
    cols = list(eng.factor_tickers.values()) + list(extra)
    f = pd.DataFrame(
        {t: 100.0 * np.exp(np.cumsum(rng.standard_normal(400) * 0.008))
         for t in cols}, index=idx)
    for t, v in (last or {}).items():
        f.loc[f.index[-1], t] = v
    hr = type("HR", (), {"data": f, "loaded": list(f.columns),
                         "failed": {}, "retried": []})()
    with patch.object(eng, "get_market_data", return_value=(f, hr)):
        return mgr.analyze_all(pd.DataFrame(portfolio))


class NormalizePositionsUnitTest(unittest.TestCase):
    """Чистая единица — без сети и матриц."""

    def setUp(self):
        # Контракт входного фрейма — зона ответственности менеджера портфеля
        # (он же владеет `analyze_all`), а не риск-движка.
        self.eng = UniversalPortfolioManager(price_source="demo")

    def test_weighted_average_purchase_price(self):
        df = pd.DataFrame({"Ticker": ["AAPL", "AAPL"],
                           "Quantity": [10.0, 5.0],
                           "Purchase_Price": [180.0, 200.0]})
        out, merged, dropped = self.eng._normalize_positions(df)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["Quantity"]), 15.0)
        self.assertAlmostEqual(float(out.iloc[0]["Purchase_Price"]),
                               (10 * 180 + 5 * 200) / 15)
        self.assertEqual(merged, [("AAPL", 2)])
        self.assertEqual(dropped, [])

    def test_three_lots_merge(self):
        df = pd.DataFrame({"Ticker": ["X", "X", "X"], "Quantity": [1.0, 2.0, 3.0],
                           "Purchase_Price": [10.0, 20.0, 30.0]})
        out, merged, _ = self.eng._normalize_positions(df)
        self.assertAlmostEqual(float(out.iloc[0]["Quantity"]), 6.0)
        self.assertAlmostEqual(float(out.iloc[0]["Purchase_Price"]),
                               (1*10 + 2*20 + 3*30) / 6)
        self.assertEqual(merged, [("X", 3)])

    def test_unique_frame_is_untouched(self):
        df = pd.DataFrame({"Ticker": ["A", "B"], "Quantity": [1.0, 2.0],
                           "Purchase_Price": [10.0, 20.0]})
        out, merged, dropped = self.eng._normalize_positions(df)
        self.assertEqual(len(out), 2)
        self.assertEqual((merged, dropped), ([], []))

    def test_long_and_short_netting_to_zero_does_not_divide(self):
        """Σqty ≈ 0 — делить нельзя; позиция остаётся, цена = простое среднее."""
        df = pd.DataFrame({"Ticker": ["Z", "Z"], "Quantity": [5.0, -5.0],
                           "Purchase_Price": [10.0, 20.0]})
        out, merged, _ = self.eng._normalize_positions(df)
        self.assertAlmostEqual(float(out.iloc[0]["Quantity"]), 0.0)
        self.assertAlmostEqual(float(out.iloc[0]["Purchase_Price"]), 15.0)
        self.assertEqual(merged, [("Z", 2)])

    def test_broker_price_survives_the_merge(self):
        df = pd.DataFrame({"Ticker": ["N", "N"], "Quantity": [1.0, 1.0],
                           "Purchase_Price": [100.0, 100.0],
                           "Broker_Current_Price": [np.nan, 103.5]})
        out, _, _ = self.eng._normalize_positions(df)
        self.assertAlmostEqual(float(out.iloc[0]["Broker_Current_Price"]), 103.5)

    def test_non_numeric_quantity_is_dropped_and_named(self):
        df = pd.DataFrame({"Ticker": ["A", "B", "C"],
                           "Quantity": [np.nan, 10.0, "мусор"],
                           "Purchase_Price": [1.0, 2.0, 3.0]})
        out, _, dropped = self.eng._normalize_positions(df)
        self.assertEqual(list(out["Ticker"]), ["B"])
        self.assertEqual(sorted(t for t, _ in dropped), ["A", "C"])
        self.assertTrue(all("количеств" in r for _, r in dropped))

    def test_ticker_whitespace_is_stripped_before_grouping(self):
        """«AAPL» и «AAPL » — одна бумага, иначе склейка не сработает."""
        df = pd.DataFrame({"Ticker": ["AAPL", " AAPL "], "Quantity": [1.0, 1.0],
                           "Purchase_Price": [10.0, 20.0]})
        out, merged, _ = self.eng._normalize_positions(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(merged, [("AAPL", 2)])

    def test_empty_frame_is_safe(self):
        out, merged, dropped = self.eng._normalize_positions(pd.DataFrame())
        self.assertEqual((merged, dropped), ([], []))


class DuplicateNoLongerBreaksTheReportTest(unittest.TestCase):
    """A-1 сквозь весь пайплайн."""

    def test_report_builds_and_pnl_is_right(self):
        res = _run({"Ticker": ["AAPL", "AAPL", "USD"],
                    "Quantity": [10.0, 5.0, 5000.0],
                    "Purchase_Price": [180.0, 200.0, 1.0]},
                   ["AAPL.US"], {"AAPL.US": 200.0})
        perf = res["performance_table"].set_index("Ticker")
        self.assertIn("AAPL", perf.index)
        # 15 шт по 200 = 3000; база 10×180 + 5×200 = 2800 → P&L = +200
        self.assertAlmostEqual(float(perf.loc["AAPL", "PnL"]), 200.0, places=2)
        self.assertAlmostEqual(res["total_value"], 8000.0, places=2)

    def test_merge_is_disclosed_in_results(self):
        res = _run({"Ticker": ["AAPL", "AAPL", "USD"],
                    "Quantity": [10.0, 5.0, 5000.0],
                    "Purchase_Price": [180.0, 200.0, 1.0]},
                   ["AAPL.US"], {"AAPL.US": 200.0})
        self.assertEqual(res["merged_positions"], [{"ticker": "AAPL", "rows": 2}])

    def test_merge_chip_reaches_the_report(self):
        from pdf_payload import _build_integrity_checks
        checks = _build_integrity_checks(
            {"merged_positions": [{"ticker": "AAPL", "rows": 2}]}, {}, {}, {})
        chip = next((c for c in checks if "Склейка" in c["label"]), None)
        self.assertIsNotNone(chip)
        self.assertIn("AAPL", chip["detail"])

    def test_no_chip_without_duplicates(self):
        from pdf_payload import _build_integrity_checks
        checks = _build_integrity_checks({}, {}, {}, {})
        self.assertIsNone(next((c for c in checks if "Склейка" in c["label"]), None))


class NaNQuantityNoLongerPoisonsMetricsTest(unittest.TestCase):
    """A-2 сквозь весь пайплайн — самый коварный из двух."""

    def _res(self):
        return _run({"Ticker": ["AAPL", "MSFT", "USD"],
                     "Quantity": [np.nan, 10.0, 5000.0],
                     "Purchase_Price": [180.0, 300.0, 1.0]},
                    ["AAPL.US", "MSFT.US"])

    def test_headline_metrics_are_finite(self):
        pm = self._res()["portfolio_metrics"]
        for key in ("Total_Volatility_Ann", "Annualised_Return", "Sharpe_Ratio"):
            v = pm.get(key)
            self.assertIsNotNone(v, f"{key} отсутствует")
            self.assertTrue(np.isfinite(float(v)), f"{key} = {v}")

    def test_dropped_row_is_named(self):
        self.assertEqual(self._res()["dropped_rows"],
                         [{"ticker": "AAPL", "reason": "нечисловое количество"}])

    def test_dropped_chip_reaches_the_report(self):
        from pdf_payload import _build_integrity_checks
        checks = _build_integrity_checks(
            {"dropped_rows": [{"ticker": "AAPL", "reason": "нечисловое количество"}]},
            {}, {}, {})
        chip = next((c for c in checks if "Отброшенные" in c["label"]), None)
        self.assertIsNotNone(chip)
        self.assertIn("AAPL", chip["detail"])

    def test_healthy_portfolio_drops_nothing(self):
        res = _run({"Ticker": ["AAPL", "MSFT", "USD"], "Quantity": [10.0, 10.0, 5000.0],
                    "Purchase_Price": [180.0, 300.0, 1.0]}, ["AAPL.US", "MSFT.US"])
        self.assertEqual(res["dropped_rows"], [])
        self.assertEqual(res["merged_positions"], [])


if __name__ == "__main__":
    unittest.main()
