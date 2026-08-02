"""−37 · Base Currency Approach: валюта сделки → базовая валюта отчёта.

Дефект (замерен на живом портфеле 2026-08-02): движок складывал ЧИСЛА из
разных валют. Строка кэша `KZT 2 500 000` пиннится в цену 1.0 и входила в
портфель как **$2 500 000** — раздувая стоимость в 320 раз вместе с весами,
концентрацией, риск-гейджем и мандатной панелью. Отдельно `Purchase_Price`
KASE-бумаги (12 000 ₸) сравнивался с долларовой текущей ценой → доходность
−99.8%.

Что здесь зафиксировано:
  1. Стоимость позиции ВСЕГДА в базовой валюте (USD) — `Current_Value`,
     `Total_Cost`, `total_value`, веса.
  2. Цена, пришедшая из МАТРИЦЫ, уже сконвертирована `_apply_fx_conversion` —
     второй множитель применять нельзя (регресс двойной конверсии).
  3. Цена брокера / цена покупки / кэш — конвертируются здесь.
  4. Единый текущий курс сокращается в доходности: `Return_Pct` остаётся
     доходностью В ВАЛЮТЕ ИНСТРУМЕНТА. Это КОНВЕНЦИЯ, и она обязана быть
     раскрыта (`results["fx_converted_rows"]` → чип в отчёте).
  5. Валюта сделки доезжает до карточки позиции в обоих тирах.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance.investment_logic import MAC3RiskEngine, UniversalPortfolioManager

KZT_USD = 0.00192308          # 1 ₸ ≈ $0.00192308  (≈ 520 ₸/$)


def _price_frame(engine, extra=(), periods=400, seed=11, last=None):
    idx = pd.bdate_range(end="2026-07-31", periods=periods)
    rng = np.random.default_rng(seed)
    cols = list(engine.factor_tickers.values()) + list(extra)
    frame = pd.DataFrame(
        {t: 100.0 * np.exp(np.cumsum(rng.standard_normal(periods) * 0.008))
         for t in cols}, index=idx)
    for ticker, price in (last or {}).items():
        frame.loc[frame.index[-1], ticker] = price
    return frame


def _analyzed(mgr, frame, portfolio):
    hr = type("HR", (), {"data": frame, "loaded": list(frame.columns),
                         "failed": {}, "retried": []})()
    with patch.object(mgr.engine, "get_market_data", return_value=(frame, hr)):
        return mgr.analyze_all(portfolio)


def _row(res, ticker):
    """Строка позиции из `performance_table` (в results нет отдельного df)."""
    perf = res["performance_table"]
    return perf.set_index("Ticker").loc[ticker]


def _demo_mgr():
    """Менеджер с ЯВНО закреплённой базовой валютой USD.

    Без пина сюда протекает чужой модульный `os.environ.setdefault(
    "KZ_RFR_ANNUAL", …)` из четырёх других suite-ов (legacy-путь
    `ReportingCurrency.from_env()` → KZT), и весь этот файл проходит в
    одиночку, но падает в полном прогоне.  Валюта отчёта — вход задачи,
    а не окружение.
    """
    with patch.dict(os.environ, {"REPORTING_CURRENCY": "USD"}, clear=False):
        return UniversalPortfolioManager(price_source="demo")


def _mgr_with_fx(rate=KZT_USD):
    """Менеджер с детерминированным FX-провайдером KZT→USD."""
    mgr = _demo_mgr()

    def _fx(base, quote):
        if base == "KZT" and quote == "USD":
            return pd.Series([rate] * 30,
                             index=pd.bdate_range(end="2026-07-31", periods=30))
        return None

    mgr.engine.fx_provider = _fx
    mgr.engine._fx_rate_cache = {}
    return mgr


# ──────────────────────────────────────────────────────────────────────────
# 1. Курс
# ──────────────────────────────────────────────────────────────────────────
class FxRateToBaseTest(unittest.TestCase):
    """`fx_rate_to_base` — единственный источник множителя для строк портфеля."""

    def test_base_currency_is_identity(self):
        self.assertEqual(_mgr_with_fx().engine.fx_rate_to_base("USD"), 1.0)

    def test_empty_currency_is_identity(self):
        """Пустая валюта = «не указана» → строка уже в базовой (обратная совместимость)."""
        eng = _mgr_with_fx().engine
        self.assertEqual(eng.fx_rate_to_base(""), 1.0)
        self.assertEqual(eng.fx_rate_to_base(None), 1.0)

    def test_provider_rate_is_multiplicative(self):
        """Конвенция та же, что у `convert_price_matrix`: цена × rate."""
        self.assertAlmostEqual(_mgr_with_fx().engine.fx_rate_to_base("KZT"),
                               KZT_USD, places=8)

    def test_case_and_whitespace_insensitive(self):
        eng = _mgr_with_fx().engine
        self.assertAlmostEqual(eng.fx_rate_to_base(" kzt "), KZT_USD, places=8)

    def test_missing_rate_returns_none_not_one(self):
        """Нет курса → None. Молчаливая 1.0 вернула бы исходный дефект."""
        mgr = _demo_mgr()
        mgr.engine.fx_provider = lambda b, q: None
        mgr.engine._fx_rate_cache = {}
        self.assertIsNone(mgr.engine.fx_rate_to_base("KZT"))

    def test_rate_is_cached_provider_called_once(self):
        """Курс дёргается один раз на отчёт — по строке кэша и по каждой бумаге."""
        calls = []

        mgr = _demo_mgr()

        def _fx(base, quote):
            calls.append((base, quote))
            return pd.Series([KZT_USD] * 5,
                             index=pd.bdate_range(end="2026-07-31", periods=5))

        mgr.engine.fx_provider = _fx
        mgr.engine._fx_rate_cache = {}
        for _ in range(4):
            mgr.engine.fx_rate_to_base("KZT")
        self.assertEqual(len(calls), 1, f"провайдер вызван {len(calls)} раз")

    def test_failed_lookup_is_cached_too(self):
        """Отрицательный результат тоже кэшируется — иначе N сетевых попыток."""
        calls = []
        mgr = _demo_mgr()
        mgr.engine.fx_provider = lambda b, q: calls.append(1) or None
        mgr.engine._fx_rate_cache = {}
        for _ in range(3):
            mgr.engine.fx_rate_to_base("KZT")
        self.assertEqual(len(calls), 1)


# ──────────────────────────────────────────────────────────────────────────
# 2. Кэш в неба́зовой валюте — исходный инцидент
# ──────────────────────────────────────────────────────────────────────────
class NonBaseCashTest(unittest.TestCase):
    """🔴 2 500 000 ₸ входили в портфель как $2 500 000."""

    def _portfolio(self, currency="KZT"):
        return pd.DataFrame({
            "Ticker": ["AAPL", "KZT"],
            "Quantity": [10.0, 2_500_000.0],
            "Purchase_Price": [180.0, 1.0],
            "Currency": ["USD", currency],
        })

    def test_kzt_cash_converted_to_usd(self):
        mgr = _mgr_with_fx()
        frame = _price_frame(mgr.engine, extra=["AAPL.US"], last={"AAPL.US": 300.0})
        res = _analyzed(mgr, frame, self._portfolio())
        # 10×300 = $3 000 акций + 2 500 000 × 0.00192308 ≈ $4 807.69 кэша
        self.assertAlmostEqual(res["total_value"], 3_000.0 + 2_500_000 * KZT_USD,
                               places=2)

    def test_without_conversion_the_book_would_be_320x(self):
        """Замер дефекта: та же книга без колонки Currency."""
        mgr = _mgr_with_fx()
        frame = _price_frame(mgr.engine, extra=["AAPL.US"], last={"AAPL.US": 300.0})
        broken = self._portfolio().drop(columns=["Currency"])
        res = _analyzed(mgr, frame, broken)
        self.assertAlmostEqual(res["total_value"], 2_503_000.0, places=2)
        # …и это ровно в ~320 раз больше правды
        self.assertGreater(res["total_value"] / (3_000.0 + 2_500_000 * KZT_USD), 300)

    def test_weights_follow_the_converted_value(self):
        """Вес — производная стоимости; если стоимость врёт, врёт вся панель."""
        mgr = _mgr_with_fx()
        frame = _price_frame(mgr.engine, extra=["AAPL.US"], last={"AAPL.US": 300.0})
        res = _analyzed(mgr, frame, self._portfolio())
        w = float(_row(res, "AAPL")["Current_Value"]) / float(res["total_value"])
        self.assertAlmostEqual(w, 3_000.0 / (3_000.0 + 2_500_000 * KZT_USD), places=4)

    def test_disclosure_row_present(self):
        res = _analyzed(_mgr_with_fx(),
                        _price_frame(_mgr_with_fx().engine, extra=["AAPL.US"]),
                        self._portfolio())
        rows = res["fx_converted_rows"]
        self.assertTrue(any(r["ticker"] == "KZT" and r["currency"] == "KZT"
                            for r in rows), rows)
        self.assertEqual(res["reporting_currency"], "USD")

    def test_usd_rows_are_not_disclosed(self):
        """Базовая валюта — не конверсия; чип не должен появляться на ровном месте."""
        mgr = _mgr_with_fx()
        frame = _price_frame(mgr.engine, extra=["AAPL.US"])
        res = _analyzed(mgr, frame, self._portfolio(currency="USD"))
        self.assertEqual(res["fx_converted_rows"], [])


# ──────────────────────────────────────────────────────────────────────────
# 3. Цена покупки и двойная конверсия
# ──────────────────────────────────────────────────────────────────────────
class PurchasePriceConversionTest(unittest.TestCase):

    def test_matrix_priced_row_is_not_converted_twice(self):
        """Цена из матрицы уже прошла `_apply_fx_conversion`.

        Регресс: второй множитель дал бы цену × rate² ≈ 0.0019 от истины.
        """
        mgr = _mgr_with_fx()
        frame = _price_frame(mgr.engine, extra=["KMGZ.KZ"], last={"KMGZ.KZ": 25.0})
        port = pd.DataFrame({
            "Ticker": ["KMGZ.KZ"], "Quantity": [100.0],
            "Purchase_Price": [12_000.0], "Currency": ["KZT"],
        })
        res = _analyzed(mgr, frame, port)
        row = _row(res, "KMGZ.KZ")
        self.assertAlmostEqual(row["Current_Price"], 25.0, places=6)
        # 12 000 ₸ × 0.00192308 ≈ $23.08 → ×100 шт ≈ $2 307.69
        self.assertAlmostEqual(row["Total_Cost"],
                               100 * 12_000 * KZT_USD, places=2)

    def test_broker_priced_row_is_converted(self):
        """Цена брокера приходит в валюте инструмента — её конвертировать НАДО."""
        mgr = _mgr_with_fx()
        frame = _price_frame(mgr.engine, extra=["VWOB.US"])
        port = pd.DataFrame({
            "Ticker": ["KZ_GOV_BOND_2030"], "Quantity": [100.0],
            "Purchase_Price": [12_000.0],
            "Broker_Current_Price": [13_000.0],
            "Currency": ["KZT"],
        })
        res = _analyzed(mgr, frame, port)
        row = _row(res, "KZ_GOV_BOND_2030")
        self.assertAlmostEqual(row["Current_Price"],
                               13_000 * KZT_USD, places=6)
        self.assertAlmostEqual(row["Current_Value"],
                               100 * 13_000 * KZT_USD, places=2)

    def test_single_rate_cancels_in_return(self):
        """Задокументированная конвенция: доходность — в валюте инструмента.

        (P_now·r − P_buy·r)/(P_buy·r) = (P_now − P_buy)/P_buy.
        13 000/12 000 − 1 = 8.33% и в тенге, и после пересчёта.
        """
        mgr = _mgr_with_fx()
        frame = _price_frame(mgr.engine, extra=["VWOB.US"])
        port = pd.DataFrame({
            "Ticker": ["KZ_GOV_BOND_2030"], "Quantity": [100.0],
            "Purchase_Price": [12_000.0],
            "Broker_Current_Price": [13_000.0],
            "Currency": ["KZT"],
        })
        res = _analyzed(mgr, frame, port)
        self.assertAlmostEqual(_row(res, "KZ_GOV_BOND_2030")["Return_Pct"],
                               13_000 / 12_000 - 1, places=6)

    def test_missing_rate_leaves_row_untouched_and_logs(self):
        """Без курса строка остаётся в своей валюте — но НЕ объявляется конвертированной."""
        mgr = _demo_mgr()
        mgr.engine.fx_provider = lambda b, q: None
        mgr.engine._fx_rate_cache = {}
        frame = _price_frame(mgr.engine, extra=["AAPL.US"], last={"AAPL.US": 300.0})
        port = pd.DataFrame({
            "Ticker": ["AAPL", "KZT"], "Quantity": [10.0, 2_500_000.0],
            "Purchase_Price": [180.0, 1.0], "Currency": ["USD", "KZT"],
        })
        res = _analyzed(mgr, frame, port)
        self.assertEqual(res["fx_converted_rows"], [])


# ──────────────────────────────────────────────────────────────────────────
# 4. Контракт брокера
# ──────────────────────────────────────────────────────────────────────────
class BrokerCurrencyContractTest(unittest.TestCase):
    """Freedom/Tradernet отдаёт валюту позиции — она обязана доехать до движка."""

    def test_dataframe_columns_include_currency(self):
        src = Path("src/finance/broker_api.py").read_text(encoding="utf-8")
        self.assertIn('"Currency"', src)

    def test_position_currency_reaches_dataframe(self):
        import inspect
        from finance.broker_api import FreedomConnector
        body = inspect.getsource(FreedomConnector._to_dataframe)
        self.assertIn("Currency", body)
        self.assertIn("p.curr", body)


# ──────────────────────────────────────────────────────────────────────────
# 5. Отчёт: валюта видна пользователю
# ──────────────────────────────────────────────────────────────────────────
class ReportSurfacesCurrencyTest(unittest.TestCase):

    def test_fmt_purchase_price_shows_both_currencies(self):
        from pdf_payload import _fmt_purchase_price
        out = _fmt_purchase_price(23.08, "KZT", "USD",
                                  [{"ticker": "X", "currency": "KZT", "rate": KZT_USD}])
        self.assertIn("23.08 USD", out)
        self.assertIn("₸", out)
        self.assertIn("12 00", out)          # ≈ 12 000 ₸ (пробел-разделитель)

    def test_fmt_purchase_price_base_currency_is_plain(self):
        from pdf_payload import _fmt_purchase_price
        self.assertEqual(_fmt_purchase_price(180.0, "USD", "USD", []), "180.00 USD")

    def test_fmt_purchase_price_no_rate_degrades_to_base(self):
        from pdf_payload import _fmt_purchase_price
        out = _fmt_purchase_price(23.08, "KZT", "USD", [])
        self.assertEqual(out, "23.08 USD")

    def test_fmt_purchase_price_none_is_dash(self):
        from pdf_payload import _fmt_purchase_price
        self.assertEqual(_fmt_purchase_price(None, "KZT", "USD", []), "—")

    def test_asset_row_carries_currency_keys(self):
        src = Path("src/pdf_payload.py").read_text(encoding="utf-8")
        for key in ('"currency"', '"fx_converted"', '"purchase_price"',
                    '"purchase_price_num"'):
            self.assertIn(key, src, f"нет ключа {key} в строке позиции")

    def test_premium_mapper_forwards_currency_both_tiers(self):
        src = Path("src/premium_payload.py").read_text(encoding="utf-8")
        self.assertEqual(src.count('"buyPrice": _txt(a, "purchase_price")'), 2,
                         "ключ buyPrice должен быть в _map_base И _map_deep")
        self.assertEqual(src.count('"fxConverted": bool(_g(a, "fx_converted"))'), 2)

    def test_compiled_bundles_render_buy_price(self):
        """Пин на ОТГРУЖАЕМЫЙ артефакт — именно он крутится в проде."""
        for name in ("base-components.js", "deep-components.js"):
            js = Path("src/premium_assets") / name
            self.assertIn("buyPrice", js.read_text(encoding="utf-8"),
                          f"{name} собран без цены покупки — запусти build.sh")

    def test_jsx_sources_match_the_bundles(self):
        jsx = Path("design/premium_v2/portfolio-holdings.jsx")
        if not jsx.exists():
            self.skipTest("design/ отсутствует (Docker deploy-gate)")
        self.assertIn("h.buyPrice", jsx.read_text(encoding="utf-8"))
        deep = Path("design/premium_v2/deep/deep-holdings.jsx")
        self.assertIn("h.buyPrice", deep.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
