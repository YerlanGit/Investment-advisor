"""
Phase-36 · Прокси неликвидных бумаг: EM вместо US IG credit (2026-07-29, H-1).

Замысел слоя (заложен изначально): у неликвидной бумаги нет пригодного ценового
ряда — либо истории нет, либо цена «сглажена» редкими сделками
(appraisal-smoothing).  Оценивать по ней риск нельзя, поэтому в ФАКТОРНУЮ МОДЕЛЬ
подставляется ликвидный индекс с похожим профилем; цена и P&L берутся из
реального тикера, так что стоимость портфеля не искажается.

Что исправлено в этом раунде:

1. **Класс прокси.**  AIX-ноты Freedom проксировались на `LQD.US` — американский
   investment-grade корпоративный кредит.  Развитый рынок занижает риск
   казахстанской структурной ноты (у EM-долга шире спред, выше волатильность,
   другая реакция на risk-off).  Прокси переведены на EM: `VWOB.US` / `VWO.US`.
   Там же KZ-суверен: `KZ_GOV_BOND` вёл на `BIL.US` (US T-Bills, риск ≈ 0).

2. **Прокси больше НЕ ДАЁТ ЦЕНУ.**  Ветка цены брала `all_data[resolved]` для
   любого резолвинга, включая подмену, поэтому нота получала последнюю цену
   прокси-ETF: купленная по 100 и стоящая у брокера 103.5 оценивалась в 65
   (цена VWOB) — минус треть стоимости портфеля и −35% «убытка», которого не
   было (с прежним LQD ≈ 110 знак был обратный, фабриковалась прибыль).

3. **Дублированные колонки.**  `needed_cols = [*факторы, *resolved]` без
   дедупликации ронял отчёт ЦЕЛИКОМ, если resolved-тикер совпадал с факторным.
   Это происходило не только на прокси: любой пользователь, ДЕРЖАЩИЙ SPY (или
   IWM/EEM/EMB/IEF/DBC/MTUM/VLUE/QUAL/SPLV), не получал отчёт вообще.

4. **Точность раскрытия.**  `proxy_map` собирался эвристикой «резолв ≠
   TICKER.US», поэтому нормализация формата (`KSPI`→`KSPI.KZ`) выглядела как
   подмена бумаги.  Теперь единый источник — `MAC3RiskEngine.proxy_for`.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from finance.investment_logic import MAC3RiskEngine, UniversalPortfolioManager


def _price_frame(engine, extra=(), periods=400, seed=7, last=None):
    """Синтетический ценовой куб: все факторы + дополнительные колонки."""
    idx = pd.bdate_range(end="2026-07-24", periods=periods)
    rng = np.random.default_rng(seed)
    cols = list(engine.factor_tickers.values()) + list(extra)
    frame = pd.DataFrame(
        {t: 100.0 * np.exp(np.cumsum(rng.standard_normal(periods) * 0.008))
         for t in cols}, index=idx)
    for ticker, price in (last or {}).items():
        frame.loc[frame.index[-1], ticker] = price
    return frame


def _analyzed(mgr, frame, portfolio):
    """Прогон analyze_all на подготовленном кубе цен."""
    hr = type("HR", (), {"data": frame, "loaded": list(frame.columns),
                         "failed": {}, "retried": []})()
    with patch.object(mgr.engine, "get_market_data", return_value=(frame, hr)):
        return mgr.analyze_all(portfolio)


class ProxyTargetsAreEmTest(unittest.TestCase):
    """Прокси должны отражать EM-природу неликвидных KZ-бумаг."""

    def test_aix_notes_proxy_to_em_debt(self):
        pm = MAC3RiskEngine.INSTRUMENT_PROXY_MAP
        self.assertEqual(pm["AIX_SPC"], "VWOB.US")
        self.assertEqual(pm["KZ_CORPORATE_BOND"], "VWOB.US")

    def test_aix_default_is_not_risk_free(self):
        """Прежний BIL.US (US T-Bills) занижал риск неизвестной AIX-бумаги до нуля."""
        self.assertNotEqual(MAC3RiskEngine.INSTRUMENT_PROXY_MAP["AIX_DEFAULT"],
                            "BIL.US")

    def test_em_equity_fallback_is_em(self):
        self.assertEqual(MAC3RiskEngine.INSTRUMENT_PROXY_MAP["EM_EQUITY_FALLBACK"],
                         "VWO.US")

    def test_no_us_ig_credit_left_in_proxy_map(self):
        """LQD.US — развитый рынок, для KZ-бумаг он не подходит."""
        self.assertNotIn("LQD.US",
                         set(MAC3RiskEngine.INSTRUMENT_PROXY_MAP.values()))

    def test_kz_sovereign_is_em_not_us_tbills(self):
        self.assertEqual(MAC3RiskEngine.BOND_CLASSIFICATION_MAP["KZ_GOV_BOND"], "EM")
        self.assertEqual(MAC3RiskEngine.BOND_PROXIES["EM"], "VWOB.US")

    def test_us_treasury_stays_risk_free(self):
        """Правка не должна задеть настоящие US Treasuries."""
        self.assertEqual(MAC3RiskEngine.BOND_CLASSIFICATION_MAP["US_TREASURY"], "RF")


class ProxyDistinctFromFactorsTest(unittest.TestCase):
    """Прокси не должен совпадать с факторным тикером.

    После дедупликации колонок совпадение больше не роняет отчёт, но остаётся
    экономически неверным: бета бумаги к «своему же» фактору вышла бы ровно 1.0
    при нулевом остаточном риске — модель заявила бы, что у неликвидной
    структурной ноты НЕТ специфического риска, то есть прямо обратное правде.
    """

    def test_no_proxy_equals_a_factor_ticker(self):
        factors = set(MAC3RiskEngine().factor_tickers.values())
        collision = set(MAC3RiskEngine.INSTRUMENT_PROXY_MAP.values()) & factors
        self.assertEqual(collision, set(), f"пересечение: {collision}")

    def test_no_bond_proxy_equals_a_factor_ticker(self):
        factors = set(MAC3RiskEngine().factor_tickers.values())
        collision = set(MAC3RiskEngine.BOND_PROXIES.values()) & factors
        self.assertEqual(collision, set(), f"пересечение: {collision}")


class ProxyResolutionTest(unittest.TestCase):
    """Резолвинг подставляет прокси и не задевает остальные ветки."""

    def setUp(self):
        eng = MAC3RiskEngine.__new__(MAC3RiskEngine)
        for attr in ("NON_RISK_ASSETS", "BOND_CLASSIFICATION_MAP", "BOND_PROXIES",
                     "INSTRUMENT_PROXY_MAP", "TICKER_MAP"):
            setattr(eng, attr, getattr(MAC3RiskEngine, attr))
        self.eng = eng

    def test_aix_note_resolves_to_em_debt(self):
        self.assertEqual(self.eng.resolve_tickers(["FFSPC6.1028.AIX"]), ["VWOB.US"])

    def test_ffspc_without_suffix_also_proxied(self):
        self.assertEqual(self.eng.resolve_tickers(["FFSPC0824"]), ["VWOB.US"])

    def test_ordinary_tickers_unaffected(self):
        self.assertEqual(
            self.eng.resolve_tickers(["AAPL", "KSPI", "BTC-USD", "SPY.US"]),
            ["AAPL.US", "KSPI.KZ", "BTC-USD", "SPY.US"])

    def test_kz_gov_bond_now_uses_em_debt(self):
        self.assertEqual(self.eng.resolve_tickers(["KZ_GOV_BOND"]), ["VWOB.US"])

    def test_kz_named_bond_goes_to_em_not_us_ig(self):
        """Слово «BOND» не говорит, ЧЕЙ это долг — KZ-маркер решает."""
        self.assertEqual(self.eng.resolve_tickers(["KAZ_CORP_BOND"]), ["VWOB.US"])
        self.assertEqual(self.eng.resolve_tickers(["OVD_2030"]), ["VWOB.US"])

    def test_unmarked_bond_keeps_us_ig(self):
        """Без KZ-маркера прежнее поведение сохраняется (US IG)."""
        self.assertEqual(self.eng.resolve_tickers(["SOMEBOND"]), ["LQD.US"])


class ProxyForIsSingleSourceTest(unittest.TestCase):
    """`proxy_for` — SSOT: отличает ПОДМЕНУ бумаги от нормализации формата."""

    def test_format_normalization_is_not_a_proxy(self):
        for ticker in ("AAPL", "KSPI", "BITCOIN", "BTC-USD", "SPY.US", "HALYK"):
            self.assertIsNone(MAC3RiskEngine.proxy_for(ticker),
                              f"{ticker} — та же бумага, не подмена")

    def test_illiquid_names_are_proxies(self):
        for ticker in ("FFSPC6.1028.AIX", "KZ_GOV_BOND", "KASPI_BOND", "OVD_2030"):
            self.assertIsNotNone(MAC3RiskEngine.proxy_for(ticker))

    def test_cash_is_not_a_proxy(self):
        for ticker in ("USD", "KZT", "CASH"):
            self.assertIsNone(MAC3RiskEngine.proxy_for(ticker))

    def test_resolve_tickers_agrees_with_proxy_for(self):
        """Резолвинг и раскрытие обязаны читать одну функцию."""
        eng = MAC3RiskEngine()
        for ticker in ("FFSPC6.1028.AIX", "KZ_GOV_BOND", "KASPI_BOND", "AAPL", "KSPI"):
            proxy = eng.proxy_for(ticker)
            if proxy is not None:
                self.assertEqual(eng.resolve_tickers([ticker]), [proxy])


class ProxyIsDisclosedTest(unittest.TestCase):
    """H-1: подмена на прокси обязана быть ВИДНА пользователю.

    Пользователь ввёл структурную ноту, а риск считается по EM-индексу.  Без
    явного раскрытия он не поймёт, почему у его ноты «чужая» бета.
    """

    def test_prefetch_exposes_proxy_map(self):
        from finance.investment_logic import MarketDataPreview
        self.assertIn("proxy_map", MarketDataPreview.__dataclass_fields__)

    def test_proxy_map_names_the_substitution(self):
        mgr = UniversalPortfolioManager(price_source="demo")
        frame = _price_frame(mgr.engine, extra=["VWOB.US"], periods=300)
        hr = type("HR", (), {"data": frame, "loaded": ["VWOB.US"],
                             "failed": {}, "retried": []})()
        with patch.object(mgr.engine, "get_market_data", return_value=(frame, hr)):
            preview = mgr.prefetch_market_data(["FFSPC6.1028.AIX"])
        self.assertEqual(preview.proxy_map.get("FFSPC6.1028.AIX"), "VWOB.US")

    def test_format_normalization_not_reported_as_substitution(self):
        """`KSPI`→`KSPI.KZ` — не подмена; в списке подмен её быть не должно."""
        mgr = UniversalPortfolioManager(price_source="demo")
        frame = _price_frame(mgr.engine, extra=["KSPI.KZ", "AAPL.US"], periods=300)
        hr = type("HR", (), {"data": frame, "loaded": list(frame.columns),
                             "failed": {}, "retried": []})()
        with patch.object(mgr.engine, "get_market_data", return_value=(frame, hr)):
            preview = mgr.prefetch_market_data(["KSPI", "AAPL", "BITCOIN"])
        self.assertEqual(preview.proxy_map, {})

    def test_results_expose_substitutions_and_cost_pricing(self):
        mgr = UniversalPortfolioManager(price_source="demo")
        frame = _price_frame(mgr.engine, extra=["AAPL.US", "VWOB.US"])
        port = pd.DataFrame({
            "Ticker": ["AAPL", "FFSPC6.1028.AIX", "USD"],
            "Quantity": [10.0, 1000.0, 5000.0],
            "Purchase_Price": [180.0, 100.0, 1.0],
        })
        res = _analyzed(mgr, frame, port)
        self.assertEqual(res["proxy_substitutions"],
                         {"FFSPC6.1028.AIX": "VWOB.US"})
        # нет ни своей истории, ни цены брокера → по цене покупки
        self.assertEqual(res["priced_at_cost"], ["FFSPC6.1028.AIX"])


class ProxyNeverPricesThePositionTest(unittest.TestCase):
    """🔴 Прокси влияет ТОЛЬКО на риск-модель, не на стоимость портфеля."""

    def _portfolio(self, broker_price=None):
        port = pd.DataFrame({
            "Ticker": ["AAPL", "FFSPC6.1028.AIX", "USD"],
            "Quantity": [10.0, 1000.0, 5000.0],
            "Purchase_Price": [180.0, 100.0, 1.0],
        })
        if broker_price is not None:
            port["Broker_Current_Price"] = [np.nan, broker_price, 1.0]
        return port

    def test_broker_price_wins_over_proxy_price(self):
        """Нота стоит 103.5 у брокера; прокси VWOB стоит 65 — берём 103.5."""
        mgr = UniversalPortfolioManager(price_source="demo")
        frame = _price_frame(mgr.engine, extra=["AAPL.US", "VWOB.US"],
                             last={"VWOB.US": 65.0, "AAPL.US": 200.0})
        res = _analyzed(mgr, frame, self._portfolio(broker_price=103.5))
        # 10×200 + 1000×103.5 + 5000
        self.assertAlmostEqual(res["total_value"], 110_500.0, places=2)
        # P&L: AAPL +200, нота +3500 — никакой фабрикации по чужому ETF
        self.assertAlmostEqual(res["total_portfolio_pnl"], 3_700.0, places=2)

    def test_without_broker_price_falls_back_to_cost_not_proxy(self):
        """Нет цены брокера → цена покупки (P&L = 0), а НЕ цена прокси."""
        mgr = UniversalPortfolioManager(price_source="demo")
        frame = _price_frame(mgr.engine, extra=["AAPL.US", "VWOB.US"],
                             last={"VWOB.US": 65.0, "AAPL.US": 200.0})
        res = _analyzed(mgr, frame, self._portfolio())
        # 10×200 + 1000×100 (цена покупки) + 5000
        self.assertAlmostEqual(res["total_value"], 107_000.0, places=2)
        self.assertAlmostEqual(res["total_portfolio_pnl"], 200.0, places=2)

    def test_proxied_name_stays_in_the_factor_model(self):
        """Ради этого прокси и существует — исключать бумагу из модели нельзя."""
        mgr = UniversalPortfolioManager(price_source="demo")
        frame = _price_frame(mgr.engine, extra=["AAPL.US", "VWOB.US"])
        res = _analyzed(mgr, frame, self._portfolio(broker_price=103.5))
        self.assertIn("FFSPC6.1028.AIX", list(res["risk_matrix"].index))

    def test_proxy_absent_from_portfolio_rows(self):
        """Строки портфеля держат ОРИГИНАЛЬНЫЕ тикеры — прокси там не появляется."""
        from finance.demo_portfolio import build_demo_portfolio
        tickers = set(build_demo_portfolio()["Ticker"].astype(str))
        self.assertNotIn("VWOB.US", tickers)
        self.assertNotIn("VWO.US", tickers)

    def test_sector_uses_original_ticker_not_proxy(self):
        eng = MAC3RiskEngine()
        self.assertEqual(eng.get_ticker_sector("FFSPC6.1028.AIX"), "EM_Kazakhstan")


class DuplicateColumnsDoNotBreakTheModelTest(unittest.TestCase):
    """Регресс-тест на дефект, который ронял отчёт целиком."""

    def _run(self, holding):
        mgr = UniversalPortfolioManager(price_source="demo")
        frame = _price_frame(mgr.engine, extra=["AAPL.US", "VWOB.US"])
        port = pd.DataFrame({
            "Ticker": ["AAPL", holding, "USD"],
            "Quantity": [10.0, 50.0, 5000.0],
            "Purchase_Price": [180.0, 400.0, 1.0],
        })
        return _analyzed(mgr, frame, port)

    def test_holding_a_factor_etf_still_builds_a_report(self):
        """SPY — самый популярный ETF в мире; портфель с ним не строил отчёт."""
        for holding in ("SPY", "IWM", "EEM", "EMB", "IEF", "DBC"):
            with self.subTest(holding=holding):
                res = self._run(holding)
                self.assertIn(holding, list(res["risk_matrix"].index))

    def test_factor_holding_gets_sane_market_beta(self):
        """У SPY бета к рынку ≈ 1, остальные ≈ 0 — модель не выродилась."""
        res = self._run("SPY")
        perf = res["performance_table"].set_index("Ticker")
        self.assertAlmostEqual(float(perf.loc["SPY", "Beta_Market"]), 1.0, delta=0.1)
        self.assertLess(abs(float(perf.loc["SPY", "Beta_Momentum"])), 0.1)

    def test_two_illiquid_names_sharing_one_proxy(self):
        """AIX-нота + KZ_GOV_BOND → оба VWOB.US: дубликат не должен ронять модель."""
        mgr = UniversalPortfolioManager(price_source="demo")
        frame = _price_frame(mgr.engine, extra=["AAPL.US", "VWOB.US"])
        port = pd.DataFrame({
            "Ticker": ["AAPL", "FFSPC6.1028.AIX", "KZ_GOV_BOND", "USD"],
            "Quantity": [10.0, 1000.0, 500.0, 5000.0],
            "Purchase_Price": [180.0, 100.0, 98.0, 1.0],
        })
        res = _analyzed(mgr, frame, port)
        idx = list(res["risk_matrix"].index)
        self.assertIn("FFSPC6.1028.AIX", idx)
        self.assertIn("KZ_GOV_BOND", idx)

    def test_metrics_stay_finite_with_zero_residual_risk(self):
        """Нулевой остаточный риск (бумага = свой же фактор) не даёт NaN/inf."""
        res = self._run("SPY")
        bad = {k: v for k, v in res["portfolio_metrics"].items()
               if isinstance(v, float) and not np.isfinite(v)}
        self.assertEqual(bad, {})
        self.assertFalse(np.isnan(res["risk_matrix"].to_numpy()).any())


if __name__ == "__main__":
    unittest.main()
