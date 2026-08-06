"""Ручной ввод: pre-flight и валюта торговли в плане (MP-04 · ЧК-04.4, ЧК-04.5).

Раунд: `AUDIT §−71`, фаза Manual Portfolio 04.

**ЧК-04.4 — покрытие известно ДО списания токена.** Смысл проверки не в
скорости: токен списывается после доставки, но время пользователя тратится
раньше. Молча построить книгу, где у трети позиций нет ценовой истории, —
худший исход: отчёт выглядит полным, а половина его основана на цене покупки.
Отсюда требование: проверка идёт по КЭШУ Фазы 1 и не делает ни одного сетевого
запроса.

**ЧК-04.5 — уровни в валюте торговли.** Показывать владельцу KASE-бумаги «$25»,
когда заявку он выставляет в тенге, бесполезно. Движок отдаёт ОБА значения,
отчёт выбирает — иначе обратная конвертация расползётся по шаблонам и нарушит
инвариант «числа считаются в `finance/`».
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import golden_support as gs  # noqa: E402
from finance.data_lineage import build_lineage  # noqa: E402
from finance.investment_logic import MAC3RiskEngine  # noqa: E402
from finance.manual_portfolio import (  # noqa: E402
    parse_portfolio_text,
    preflight_coverage,
)

PREFLIGHT_TEXT = """
AAPL 10 150
TLT 5 95
NOHIST 3 20 USD
KSPI 100 104,5 KZT
CASH:USD 500
"""


class PreflightCoverageTest(unittest.TestCase):
    """ЧК-04.4: что известно о ценах ДО запуска анализа."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = MAC3RiskEngine()
        cls.report = parse_portfolio_text(PREFLIGHT_TEXT, cls.engine)

    def _coverage(self, cached: dict):
        calls: list[tuple] = []

        def lookup(tickers, days):
            calls.append((tuple(tickers), days))
            return {t: cached.get(t) for t in tickers}

        cov = preflight_coverage(self.report, self.engine,
                                 days=1825, cached_lookup=lookup)
        return cov, calls

    def test_known_and_unknown_are_separated(self) -> None:
        cov, _ = self._coverage({"AAPL.US": 1200, "TLT.US": 1200,
                                 "KSPI.KZ": 900, "NOHIST.US": None})
        self.assertEqual(cov.covered, ["AAPL", "KSPI", "TLT"])
        self.assertEqual(cov.unknown, ["NOHIST"])
        self.assertFalse(cov.is_complete)

    def test_cash_is_not_counted_as_a_priced_position(self) -> None:
        """У кэша цены не бывает — его отсутствие в кэше не «дыра»."""
        cov, _ = self._coverage({"AAPL.US": 1, "TLT.US": 1,
                                 "KSPI.KZ": 1, "NOHIST.US": 1})
        self.assertNotIn("USD", cov.covered + cov.unknown)

    def test_cost_share_is_reported_per_currency_not_mixed(self) -> None:
        """Складывать тенге с долларами без курса — это дефект F-2 в миниатюре.

        Курса на этом шаге ещё нет (он потребовал бы сети), поэтому единственный
        честный ответ — доля ОТДЕЛЬНО по каждой валюте. Одно смешанное число
        выглядело бы точнее, а было бы неверным.
        """
        cov, _ = self._coverage({"AAPL.US": 1200, "TLT.US": 1200,
                                 "KSPI.KZ": 900, "NOHIST.US": None})
        shares = cov.cost_share_by_currency
        self.assertEqual(set(shares), {"USD", "KZT"})
        self.assertAlmostEqual(shares["KZT"], 1.0, places=6)
        # USD: покрыто 10·150 + 5·95 = 1975 из 1975 + 3·20 = 2035
        self.assertAlmostEqual(shares["USD"], 1975.0 / 2035.0, places=6)

    def test_lookup_is_asked_once_and_only_for_resolved_tickers(self) -> None:
        """Ни одного лишнего обращения — и ни одного сетевого."""
        _, calls = self._coverage({"AAPL.US": 1, "TLT.US": 1,
                                   "KSPI.KZ": 1, "NOHIST.US": 1})
        self.assertEqual(len(calls), 1, "кэш опрошен больше одного раза")
        asked, days = calls[0]
        self.assertEqual(days, 1825)
        self.assertEqual(set(asked), {"AAPL.US", "TLT.US", "KSPI.KZ", "NOHIST.US"})

    def test_unknown_is_not_promised_to_be_missing(self) -> None:
        """Пустой кэш значит «не спрашивали», а не «истории нет».

        Поле называется `unknown`, а не `missing`, намеренно: обещать
        пользователю, что бумага не найдётся у провайдера, мы не вправе —
        её просто ещё не качали.
        """
        cov, _ = self._coverage({})
        self.assertEqual(cov.covered, [])
        # Четыре рисковые позиции текста; строка кэша сюда не считается.
        self.assertEqual(cov.unknown, ["AAPL", "KSPI", "NOHIST", "TLT"])


class ManualRunSurfacesTest(unittest.TestCase):
    """ЧК-04.5: уровни в валюте торговли и строка CoVe об источнике."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.res_manual = gs.run_analyze_all("manual")
        cls.res_broker = gs.run_analyze_all("base")

    def _row(self, results: dict, ticker: str) -> dict:
        for row in results["action_plan"]:
            if str(row.get("ticker")) == ticker:
                return row
        self.fail(f"строка плана {ticker!r} не найдена")

    def test_foreign_position_carries_local_levels(self) -> None:
        """Тенговая бумага получает цену в тенге рядом с базовой."""
        row = self._row(self.res_manual, "KSPI")
        self.assertEqual(row["currency"], "KZT")
        self.assertIsNotNone(row["price_local"])
        rate = gs.SCENARIOS["manual"]["fx_rates"]["KZT"]
        # Сравнение ОТНОСИТЕЛЬНОЕ: в словарь оба числа уходят округлёнными до
        # 4 знаков, поэтому точное равенство здесь означало бы проверку
        # округления, а не конверсии.
        self.assertAlmostEqual(row["price_local"] / (row["price"] / rate), 1.0,
                               places=5)
        self.assertGreater(row["price_local"], row["price"],
                           "цена в тенге обязана быть БОЛЬШЕ цены в долларах")

    def test_base_currency_position_has_no_local_duplicate(self) -> None:
        """Дублировать доллары долларами незачем — это шум в карточке."""
        row = self._row(self.res_manual, "AAPL")
        self.assertIsNone(row["currency"])
        self.assertIsNone(row["price_local"])

    def test_broker_path_is_unaffected(self) -> None:
        """Брокерская книга без валютных строк не получает локальных уровней."""
        for row in self.res_broker["action_plan"]:
            with self.subTest(ticker=row.get("ticker")):
                self.assertIsNone(row["price_local"])

    def test_manual_source_is_disclosed_in_cove(self) -> None:
        """Читатель обязан узнать, что состав книги ввёл человек."""
        rows = build_lineage(self.res_manual)
        manual = [r for r in rows if "Ручной ввод" in str(r.get("source", ""))]
        self.assertEqual(len(manual), 1, "строки об источнике состава нет в CoVe")
        note = str(manual[0].get("note", ""))
        self.assertIn("не подтвержден", note.replace("ё", "е"),
                      "строка не называет границу проверяемого")

    def test_broker_run_has_no_manual_row(self) -> None:
        """Строка условная: на брокерском пути её быть не должно."""
        rows = build_lineage(self.res_broker)
        self.assertFalse([r for r in rows if "Ручной ввод" in str(r.get("source", ""))])

    def test_portfolio_source_reaches_the_contract(self) -> None:
        self.assertEqual(self.res_manual["portfolio_source"], "manual")
        self.assertEqual(self.res_broker["portfolio_source"], "freedom")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
