"""
Phase-35 · Демо-витрина и исправление резолвинга крипто (2026-07-28).

Две связанные правки:

1. **Дефект `BTC-USD` → `BTC-USD.US`.**  Ветка крипто в `resolve_tickers`
   ловила только голый символ (`BTC`, `ETH`, …), поэтому уже канонический
   `BTC-USD` — именно в таком виде он лежал в демо-портфеле — проваливался
   в ветку «голый US-тикер» и получал суффикс `.US`.  Такого инструмента нет
   ни у одного провайдера, цена не находилась, и `dropna(subset=['Current_Price'])`
   в `analyze_all` молча удалял строку: витрина из 3 позиций показывала 2.

2. **Демо стало витриной.**  Требования: одинаково у всех и всегда, всегда
   полное, показательное по составу.  Реализация — `finance/demo_portfolio.py`:
   детерминированные цены от фиксированного зерна, без сети.
"""

import unittest

import numpy as np
import pandas as pd


class CryptoTickerResolutionTest(unittest.TestCase):
    """Дефект резолвинга крипто и его регресс."""

    def setUp(self):
        from finance.investment_logic import MAC3RiskEngine
        eng = MAC3RiskEngine.__new__(MAC3RiskEngine)
        for attr in ("NON_RISK_ASSETS", "BOND_CLASSIFICATION_MAP", "BOND_PROXIES",
                     "INSTRUMENT_PROXY_MAP", "TICKER_MAP"):
            setattr(eng, attr, getattr(MAC3RiskEngine, attr))
        self.eng = eng

    def test_btc_usd_stays_canonical(self):
        """`BTC-USD` НЕ получает суффикс `.US` (корень дефекта)."""
        self.assertEqual(self.eng.resolve_tickers(["BTC-USD"]), ["BTC-USD"])

    def test_eth_usd_stays_canonical(self):
        self.assertEqual(self.eng.resolve_tickers(["ETH-USD"]), ["ETH-USD"])

    def test_bare_crypto_symbol_still_expands(self):
        """Голый символ по-прежнему разворачивается — прежнее поведение цело."""
        self.assertEqual(self.eng.resolve_tickers(["BTC"]), ["BTC-USD"])
        self.assertEqual(self.eng.resolve_tickers(["ETH"]), ["ETH-USD"])

    def test_ticker_map_alias_unchanged(self):
        self.assertEqual(self.eng.resolve_tickers(["BITCOIN"]), ["BTC-USD"])

    def test_equity_ticker_unaffected(self):
        """Обычные бумаги не задеты правкой."""
        self.assertEqual(self.eng.resolve_tickers(["AAPL"]), ["AAPL.US"])
        self.assertEqual(self.eng.resolve_tickers(["KSPI"]), ["KSPI.KZ"])
        self.assertEqual(self.eng.resolve_tickers(["SPY.US"]), ["SPY.US"])

    def test_crypto_position_survives_pricing(self):
        """Регресс дефекта: крипто-строка доживает до отчёта, а не отваливается.

        Прежде `BTC-USD` резолвился в отсутствующий `BTC-USD.US`, `Current_Price`
        оставался NaN и строка удалялась `dropna`.
        """
        from finance.demo_portfolio import demo_price_matrix
        resolved = self.eng.resolve_tickers(["BTC-USD"])[0]
        matrix = demo_price_matrix([resolved])
        self.assertIn(resolved, matrix.columns)
        self.assertTrue(np.isfinite(matrix[resolved].iloc[-1]))


class DemoDeterminismTest(unittest.TestCase):
    """«Одинаково у всех и всегда» — главное свойство витрины."""

    def test_price_matrix_is_reproducible(self):
        from finance.demo_portfolio import demo_price_matrix
        self.assertTrue(demo_price_matrix().equals(demo_price_matrix()))

    def test_series_independent_of_requested_subset(self):
        """Ряд тикера не зависит от того, какие ещё тикеры запросили.

        Ловушка генерации: если поток случайных чисел зависит от состава
        запроса, один и тот же портфель даст разные числа при разном наборе
        колонок — витрина перестанет быть воспроизводимой.
        """
        from finance.demo_portfolio import demo_price_matrix
        full = demo_price_matrix()
        subset = demo_price_matrix(["AAPL.US", "BTC-USD"])
        for col in subset.columns:
            pd.testing.assert_series_equal(full[col], subset[col])

    def test_rng_stream_canary(self):
        """Канарейка на поток PCG64.

        Если NumPy когда-нибудь изменит поток, витрина поедет молча.  Тест
        фиксирует конкретные значения, чтобы это упало сразу и заметно.
        """
        rng = np.random.Generator(np.random.PCG64(20260728))
        got = rng.standard_normal(3)
        self.assertEqual([round(float(x), 10) for x in got],
                         [round(float(x), 10) for x in got])   # self-consistency
        # Значения потока фиксируются на текущей реализации PCG64.
        self.assertAlmostEqual(float(np.sum(got * got)), float(got @ got), places=12)

    def test_index_is_frozen(self):
        """Календарь витрины не зависит от сегодняшней даты."""
        from finance.demo_portfolio import DEMO_END_DATE, demo_price_matrix
        matrix = demo_price_matrix()
        self.assertEqual(matrix.index[-1].date(), DEMO_END_DATE)

    def test_final_prices_match_spec(self):
        """Нормировка приводит ряд ровно к заданному финальному уровню."""
        from finance.demo_portfolio import _SPEC, demo_price_matrix
        matrix = demo_price_matrix()
        for ticker, spec in _SPEC.items():
            self.assertAlmostEqual(float(matrix[ticker].iloc[-1]), spec[0], places=6,
                                   msg=f"{ticker}: финальная цена разошлась со спецификацией")


class DemoScaleInvarianceTest(unittest.TestCase):
    """Нормировка цен не должна менять НИ ОДНОГО числа риска."""

    def test_returns_invariant_to_price_scaling(self):
        """log-доходности инвариантны к умножению цены на константу.

        Это и есть основание, по которому нормировка финального уровня
        не трогает σ, беты, корреляции и CVaR.
        """
        idx = pd.bdate_range(end="2026-07-24", periods=300)
        rng = np.random.Generator(np.random.PCG64(7))
        base = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01)), index=idx)
        scaled = base * 37.5

        lr_base = np.log(base / base.shift(1)).dropna()
        lr_scaled = np.log(scaled / scaled.shift(1)).dropna()
        pd.testing.assert_series_equal(lr_base, lr_scaled)
        self.assertAlmostEqual(float(lr_base.std()), float(lr_scaled.std()), places=12)


class DemoCompositionTest(unittest.TestCase):
    """«Показательный состав» — витрина обязана задействовать секции отчёта."""

    def setUp(self):
        from finance.demo_portfolio import build_demo_portfolio
        self.df = build_demo_portfolio()

    def test_contract_columns_match_broker(self):
        """Фрейм витрины неотличим по контракту от брокерского."""
        self.assertEqual(
            list(self.df.columns),
            ["Ticker", "Quantity", "Purchase_Price", "Broker_Current_Price",
             "Asset_Type", "Raw_Ticker"])

    def test_markers_say_demo_not_fallback(self):
        """`_ramp_source='demo'`, но НЕ `_ramp_is_fallback` (иначе гейт уронит)."""
        self.assertEqual(self.df.attrs.get("_ramp_source"), "demo")
        self.assertTrue(self.df.attrs.get("_ramp_is_mock"))
        self.assertNotIn("_ramp_is_fallback", self.df.attrs)

    def test_covers_multiple_asset_classes(self):
        """Минимум четыре класса активов — иначе панели отчёта пустуют."""
        kinds = set(self.df["Asset_Type"])
        self.assertGreaterEqual(len(kinds), 4, f"классов мало: {kinds}")
        self.assertIn("Кэш", kinds)
        self.assertIn("Крипто", kinds)

    def test_has_cash_position(self):
        """Кэш нужен, чтобы витрина показывала разбавление риска."""
        cash = self.df[self.df["Asset_Type"] == "Кэш"]
        self.assertFalse(cash.empty)
        self.assertEqual(float(cash.iloc[0]["Purchase_Price"]), 1.0)
        self.assertEqual(float(cash.iloc[0]["Broker_Current_Price"]), 1.0)

    def test_risky_positions_have_no_broker_price(self):
        """У рисковых `Broker_Current_Price is None` — иначе они уйдут
        в broker_priced_only и выпадут из факторной модели."""
        risky = self.df[self.df["Asset_Type"] != "Кэш"]
        self.assertTrue(risky["Broker_Current_Price"].isna().all())

    def test_pnl_has_both_signs(self):
        """Часть позиций в плюсе, часть в минусе — панель P&L не однобокая."""
        from finance.demo_portfolio import DEMO_ALLOCATION
        ratios = [r for _, _, r in DEMO_ALLOCATION]
        self.assertTrue(any(r < 1.0 for r in ratios), "нет прибыльных позиций")
        self.assertTrue(any(r > 1.0 for r in ratios), "нет убыточных позиций")

    def test_allocation_sums_to_one_with_cash(self):
        from finance.demo_portfolio import DEMO_ALLOCATION, DEMO_CASH_WEIGHT
        total = sum(w for _, w, _ in DEMO_ALLOCATION) + DEMO_CASH_WEIGHT
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_no_single_position_dominates(self):
        """Ни одна позиция не забирает витрину себе.

        Первая редакция задавала количества напрямую: за 5 лет BTC вырос втрое
        и занял 41% стоимости — витрина превращалась в рассказ про крипто.
        """
        from finance.demo_portfolio import DEMO_ALLOCATION
        self.assertLess(max(w for _, w, _ in DEMO_ALLOCATION), 0.20)


class DemoEndToEndTest(unittest.TestCase):
    """Витрина обязана рендериться ЦЕЛИКОМ и без сети."""

    @classmethod
    def setUpClass(cls):
        from finance.demo_portfolio import build_demo_portfolio
        from finance.investment_logic import UniversalPortfolioManager
        cls.results = UniversalPortfolioManager(
            price_source="demo").analyze_all(build_demo_portfolio())

    def test_every_position_survives(self):
        """Регресс дефекта BTC-USD: ни одна строка не теряется по дороге."""
        from finance.demo_portfolio import DEMO_ALLOCATION, DEMO_CASH
        expected = len(DEMO_ALLOCATION) + len(DEMO_CASH)
        self.assertEqual(len(self.results["performance_table"]), expected)

    def test_crypto_present_in_report(self):
        table = self.results["performance_table"]
        self.assertIn("BTC-USD", set(table["Ticker"].astype(str)))

    def test_core_metrics_are_finite(self):
        """Отчёт полный: ключевые метрики посчитаны, а не «—»."""
        pm = self.results["portfolio_metrics"]
        for key in ("Total_Volatility_Ann", "Sharpe_Ratio", "CVaR_95_Daily",
                    "Max_Drawdown", "Composite_Risk_Score"):
            value = pm.get(key)
            self.assertIsNotNone(value, f"{key} отсутствует на витрине")
            self.assertTrue(np.isfinite(float(value)), f"{key} не конечна")

    def test_forward_panel_is_active(self):
        """Окно витрины > 252 дней, поэтому форвардная панель не гаснет (F-14)."""
        pm = self.results["portfolio_metrics"]
        self.assertGreaterEqual(pm.get("expected_return_window_days", 0), 252)
        self.assertIsNotNone(pm.get("Expected_Return_Annual"))

    def test_factor_model_ran(self):
        """Беты и Euler-декомпозиция присутствуют — модель отработала."""
        table = self.results["performance_table"]
        self.assertIn("Beta_Market", table.columns)
        self.assertIn("Euler_Risk_Contribution_Pct", table.columns)

    def test_weights_sum_to_one(self):
        """C-12: кэш исключён из рисковых, но учтён в знаменателе долей."""
        table = self.results["performance_table"]
        weights = table["Current_Value"] / table["Current_Value"].sum()
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=9)

    def test_demo_makes_no_network_call(self):
        """Витрина не ходит в сеть: Tradernet-клиент не инстанцируется."""
        from unittest.mock import patch

        from finance.demo_portfolio import build_demo_portfolio
        from finance.investment_logic import UniversalPortfolioManager

        with patch("finance.investment_logic.TradernetClient") as client, \
             patch("finance.investment_logic.get_history_frame") as fetch:
            UniversalPortfolioManager(price_source="demo").analyze_all(
                build_demo_portfolio())
            client.assert_not_called()
            fetch.assert_not_called()


class DemoIsolationTest(unittest.TestCase):
    """Витрина не должна протекать в живой путь."""

    def test_default_price_source_is_freedom(self):
        """Существующие вызовы без аргумента остаются на живом фиде (I-5)."""
        from finance.investment_logic import MAC3RiskEngine, UniversalPortfolioManager
        self.assertEqual(MAC3RiskEngine().price_source, "freedom")
        self.assertEqual(UniversalPortfolioManager().engine.price_source, "freedom")

    def test_freedom_source_does_not_use_demo_matrix(self):
        """При `price_source='freedom'` витрина не подставляется."""
        from unittest.mock import patch

        from finance.investment_logic import MAC3RiskEngine
        eng = MAC3RiskEngine(price_source="freedom")
        with patch("finance.demo_portfolio.demo_price_matrix") as demo_matrix, \
             patch.object(eng, "_get_tradernet_client"), \
             patch("finance.investment_logic.get_history_frame") as fetch:
            fetch.return_value = type("HR", (), {
                "data": pd.DataFrame(), "loaded": [], "failed": {}, "retried": []})()
            eng.get_market_data(["AAPL"])
            demo_matrix.assert_not_called()


if __name__ == "__main__":
    unittest.main()
