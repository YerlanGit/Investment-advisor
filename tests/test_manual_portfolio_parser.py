"""Ручной ввод: разбор текста и валюта сделки (MP-04 · ЧК-04.2, ЧК-04.3).

Раунд: `AUDIT §−71`, фаза Manual Portfolio 04
(`docs/roadmap/manual_portfolio/PHASE_04_MANUAL_SOURCE.md`).

Что охраняется
--------------
**ЧК-04.2 — три категории разбора.** Парсер обязан вернуть принятое,
распознанное по-своему и отвергнутое РАЗДЕЛЬНО, и никогда не бросать на
пользовательском вводе. Молчаливая потеря строки — худший исход ручного ввода:
портфель посчитается, отчёт построится, а позиции в нём не будет, и признака
этому не появится нигде.

**ЧК-04.3 — валюта сделки доезжает до стоимости.** Одна и та же позиция,
введённая в тенге и в долларах, обязана дать РАЗНЫЙ `Total_Cost`. Обратное —
это дефект F-2 из плана: `CASH:KZT 2 500 000` входил в книгу как 2 500 000 у.е.
и получал вес 99.6% вместо ≈32%, утаскивая за собой TRC, CVaR, беты и
композитный риск-индекс. Ни одного визуального признака.

Проверка идёт СКВОЗНОЙ, а не на парсере: конвертацию делает движок
(Base Currency Approach, `§−45`), а парсер лишь проставляет колонку `Currency`.
Тест на парсере доказал бы, что колонка заполнена, — но не что число изменилось.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import golden_support as gs  # noqa: E402
from finance.investment_logic import MAC3RiskEngine  # noqa: E402
from finance.manual_portfolio import (  # noqa: E402
    CONTRACT_COLUMNS,
    parse_portfolio_text,
)

#: Ввод с НАМЕРЕННО битой строкой — чтобы все три категории были непусты.
MIXED_INPUT = """
# принятые
AAPL 10 150.50
каспи 100 104,5 KZT
CASH:USD 5000

# отвергнутые
BADLINE
AAPL -3 10
CASH 5000
"""


class ParseReportCategoriesTest(unittest.TestCase):
    """ЧК-04.2: valid / auto_resolved / failed — три непустые категории."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = MAC3RiskEngine()
        cls.report = parse_portfolio_text(MIXED_INPUT, cls.engine)

    def test_all_three_categories_are_populated(self) -> None:
        self.assertTrue(self.report.valid, "нет ни одной принятой позиции")
        self.assertTrue(self.report.auto_resolved,
                        "синоним «каспи» не попал в auto_resolved — "
                        "пользователь не узнает, как мы его поняли")
        self.assertTrue(self.report.failed, "битые строки не попали в failed")

    def test_errors_name_the_line_number_and_reason(self) -> None:
        """Без номера строки пользователь не найдёт, что чинить."""
        for line_no, reason in self.report.failed:
            with self.subTest(line=line_no):
                self.assertIsInstance(line_no, int)
                self.assertGreater(line_no, 0)
                self.assertTrue(str(reason).strip(), "пустая причина отказа")

    def test_parser_never_raises_on_user_input(self) -> None:
        """Любой мусор — это `errors`, а не исключение."""
        for junk in ("", "   ", "#только комментарий", "\x00\x01",
                     "A" * 400, "AAPL", "AAPL 10", ";;;;", "1 2 3 4 5 6"):
            with self.subTest(junk=junk[:20]):
                report = parse_portfolio_text(junk, self.engine)
                self.assertIsNotNone(report)

    def test_short_sales_are_refused_explicitly(self) -> None:
        """Отрицательный вес молча ломает Euler и composite_risk_score."""
        report = parse_portfolio_text("AAPL -3 10", self.engine)
        self.assertFalse(report.valid)
        self.assertIn("корот", report.failed[0][1].lower())

    def test_dataframe_matches_the_broker_contract(self) -> None:
        """Колонки и их ПОРЯДОК — как у брокерского пути."""
        df = self.report.to_dataframe()
        self.assertEqual(list(df.columns), CONTRACT_COLUMNS)
        self.assertEqual(df.attrs.get("_ramp_source"), "manual")
        for marker in ("_ramp_is_mock", "_ramp_is_fallback"):
            self.assertNotIn(marker, df.attrs,
                             f"ручной портфель реальный — {marker} включил бы "
                             "гейт fallback-мока и подменил книгу демо-витриной")

    def test_risky_rows_have_no_broker_price(self) -> None:
        """`Broker_Current_Price=None` — цена придёт из матрицы.

        Подстановка сюда цены покупки включила бы ветку `broker_priced_only`:
        позиция вылетела бы из факторной модели, а P&L стал бы тождественно
        нулевым (`PHASE_04 §2`).
        """
        df = self.report.to_dataframe()
        risky = df[df["Asset_Type"] != "Кэш"]
        self.assertTrue(risky["Broker_Current_Price"].isna().all())


class TradeCurrencyReachesCostTest(unittest.TestCase):
    """ЧК-04.3: та же позиция в ₸ и в $ даёт РАЗНУЮ стоимость."""

    #: Один и тот же портфель, отличается только валютой позиции и кэша.
    TEXT_KZT = "KSPI 200 45000 KZT\nCASH:KZT 2250000\n"
    TEXT_USD = "KSPI 200 45000 USD\nCASH:USD 2250000\n"

    @classmethod
    def _run(cls, text: str) -> dict:
        """Прогнать текст через ВЕСЬ конвейер на офлайн-фикстуре."""
        scenario = dict(gs.SCENARIOS["manual"])
        scenario["text"] = text
        saved = gs.SCENARIOS.get("_probe")
        gs.SCENARIOS["_probe"] = scenario
        try:
            return gs.run_analyze_all("_probe")
        finally:
            if saved is None:
                gs.SCENARIOS.pop("_probe", None)
            else:                                       # pragma: no cover
                gs.SCENARIOS["_probe"] = saved

    @classmethod
    def setUpClass(cls) -> None:
        cls.res_kzt = cls._run(cls.TEXT_KZT)
        cls.res_usd = cls._run(cls.TEXT_USD)

    @staticmethod
    def _row(results: dict, ticker: str):
        """Строка таблицы по тикеру.

        `performance_table` держит `Ticker` КОЛОНКОЙ, а индекс у неё
        порядковый — искать по индексу нельзя.
        """
        table = results["performance_table"]
        found = table[table["Ticker"].astype(str) == ticker]
        assert not found.empty, f"строка {ticker!r} не найдена в отчёте"
        return found.iloc[0]

    @classmethod
    def _cost(cls, results: dict, ticker: str) -> float:
        return float(cls._row(results, ticker)["Total_Cost"])

    def test_kzt_and_usd_give_different_total_cost(self) -> None:
        """Главный инвариант чекпойнта — и главный дефект, если он нарушен."""
        cost_kzt = self._cost(self.res_kzt, "KSPI")
        cost_usd = self._cost(self.res_usd, "KSPI")
        self.assertNotAlmostEqual(
            cost_kzt, cost_usd, places=2,
            msg="стоимость позиции в тенге совпала со стоимостью в долларах — "
                "значит валюта сделки НЕ доехала до чисел (дефект F-2: книга "
                "получает вес, завышенный в ~450 раз, вместе с ним уезжают "
                "TRC, CVaR, беты и композитный риск-индекс)",
        )
        rate = gs.SCENARIOS["manual"]["fx_rates"]["KZT"]
        self.assertAlmostEqual(
            cost_kzt, cost_usd * rate, places=4,
            msg="конвертация выполнена НЕ тем курсом, что задан фикстурой",
        )

    def test_foreign_cash_is_converted_not_taken_at_face_value(self) -> None:
        """`CASH:KZT 2 250 000` — это ≈$5 000, а не $2 250 000."""
        value = float(self._row(self.res_kzt, "KZT")["Current_Value"])
        rate = gs.SCENARIOS["manual"]["fx_rates"]["KZT"]
        self.assertAlmostEqual(value, 2250000 * rate, places=2)

    def test_conversion_is_disclosed_not_silent(self) -> None:
        """Конвертация обязана быть НАЗВАНА — иначе она невидима (D-2)."""
        rows = self.res_kzt["fx_converted_rows"]
        self.assertTrue(rows, "реестр сконвертированных строк пуст")
        self.assertTrue(all(r.get("rate") for r in rows),
                        "в реестре нет курса — раскрытие неполное")
        self.assertFalse(
            self.res_usd["fx_converted_rows"],
            "долларовая книга не требует конверсии, а реестр непуст — "
            "значит в него попадают строки, которых не переводили (D-2)",
        )


class MixedCurrencyLotsAreRefusedTest(unittest.TestCase):
    """ЧК-04.3, вторая половина: склейка не должна усреднять разные валюты."""

    def test_same_ticker_in_two_currencies_is_an_error(self) -> None:
        """Склейка идёт ДО конверсии — смешение дало бы тихую чепуху.

        `_normalize_positions` берёт валюту ПЕРВОЙ строки группы и усредняет
        цены всех лотов. Для брокера это безопасно (у бумаги одна валюта), для
        ручного ввода — нет. Единственный способ выполнить требование плана
        «конверсия ДО склейки», не переставляя стадии: не пропускать в склейку
        то, что склеивать нельзя.
        """
        engine = MAC3RiskEngine()
        report = parse_portfolio_text(
            "AAPL 10 150 USD\nAAPL 5 60000 KZT\n", engine)
        self.assertEqual(len(report.valid), 1, "обе строки приняты — валюты смешались")
        self.assertTrue(report.failed)
        reason = report.failed[0][1].lower()
        self.assertIn("kzt", reason)
        self.assertIn("usd", reason)

    def test_same_ticker_in_one_currency_is_left_to_the_engine(self) -> None:
        """Обычный дубликат парсер НЕ трогает: склейка живёт в движке (§−41)."""
        engine = MAC3RiskEngine()
        report = parse_portfolio_text("AAPL 10 150\nAAPL 5 160\n", engine)
        self.assertEqual(len(report.valid), 2)
        self.assertFalse(report.failed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
