"""MP-08 · замер покрытия Stooq и судьба непокрытой бумаги (ЧК-08.1, ЧК-08.2).

Раунд: `AUDIT §−73`, фаза Manual Portfolio 08
(`docs/roadmap/manual_portfolio/PHASE_08_STOOQ_CONVENTION.md`).

Что здесь охраняется
--------------------
**ЧК-08.1 — список бумаг замера собирается ИЗ КОДА.** Держать его руками в
markdown нельзя: за две недели после написания `PHASE_08 §2.5` список разошёлся
с движком трижды. Тесты ниже проверяют не «список правильный», а «список
ВЫВЕДЕН» — то есть новый факторный ETF или новый прокси попадут в замер сами.

**ЧК-08.2 — что происходит с бумагой, которой у провайдера нет.** Ответ на этот
вопрос нельзя было придумать: он ИЗМЕРЕН на конвейере, и измерение показало
дефект (`ManualPositionWithoutPricesTest`).

Сетевой части здесь нет и быть не может: `stooq.com` закрыт сетевой политикой
(403 на CONNECT, перепроверено 2026-08-06), а `STOOQ_API_KEY` добывается
человеком через капчу. Проверяется всё, что проверяется офлайн: универсум,
формы символа, разбор CSV — включая тело отказа, которое Stooq отдаёт с
кодом 200.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_SCRIPT = _ROOT / "scripts" / "stooq_coverage_probe.py"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tests"))


def _load_probe_module():
    """Скрипт живёт в `scripts/`, которого НЕТ в деплой-образе.

    Тот же приём, что у `test_repo_hygiene`: отсутствие каталога — это
    `skipTest`, иначе деплой-гейт падает на файле, который в образ не входит.
    """
    if not _SCRIPT.exists():
        raise unittest.SkipTest("scripts/ отсутствует (Docker deploy-gate)")
    if "stooq_coverage_probe" in sys.modules:
        return sys.modules["stooq_coverage_probe"]
    spec = importlib.util.spec_from_file_location("stooq_coverage_probe", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Регистрация в `sys.modules` ДО исполнения обязательна: `@dataclass`
    # разрешает аннотации через `sys.modules[cls.__module__].__dict__`, и без
    # записи модуль-владелец не находится (`'NoneType' has no attribute
    # '__dict__'` на импорте, а не на вызове).
    sys.modules["stooq_coverage_probe"] = module
    spec.loader.exec_module(module)
    return module


class UniverseIsDerivedFromCodeTest(unittest.TestCase):
    """Список замера обязан следовать за движком, а не за документом."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.probe = _load_probe_module()
        cls.universe = cls.probe.build_universe()
        cls.by_ticker = {p.ticker: p for p in cls.universe}

    def test_every_factor_etf_is_probed(self) -> None:
        """Группа A решает go/no-go запуска — пропуск фактора недопустим."""
        from finance.investment_logic import MAC3RiskEngine

        for name, ticker in MAC3RiskEngine().factor_tickers.items():
            with self.subTest(factor=name):
                self.assertIn(ticker, self.by_ticker)

    def test_every_proxy_is_probed(self) -> None:
        """Без прокси облигационная часть книги не считается вовсе."""
        from finance.investment_logic import MAC3RiskEngine as E

        for ticker in set(E.BOND_PROXIES.values()) | set(E.INSTRUMENT_PROXY_MAP.values()):
            with self.subTest(proxy=ticker):
                self.assertIn(ticker, self.by_ticker)

    def test_kz_tickers_come_from_the_live_dictionary(self) -> None:
        """🔴 Список плана §2.5 УЖЕ устарел: в нём 4 KZ-бумаги, в словаре 6.

        `KMGZ.KZ` и `KEGC.KZ` добавлены в `TICKER_MAP` раундом `§−70`, а
        markdown-таблица плана об этом не знает. Замер, идущий по документу,
        не проверил бы две бумаги, которые пользователь введёт первыми.
        """
        from finance.investment_logic import MAC3RiskEngine as E

        expected = {t for t in E.TICKER_MAP.values() if t.endswith((".KZ", ".IL"))}
        probed = {p.ticker for p in self.universe if p.group.startswith("C")}
        self.assertEqual(probed, expected)
        self.assertGreater(len(expected), 4,
                           "словарь синонимов сократился — проверьте, не потеряны "
                           "ли KZ-бумаги")

    def test_new_engine_entry_reaches_the_probe_automatically(self) -> None:
        """Главная проверка: список именно ВЫВОДИТСЯ, а не переписан руками."""
        from unittest.mock import patch

        from finance.investment_logic import MAC3RiskEngine as E

        patched = dict(E.INSTRUMENT_PROXY_MAP)
        patched["FRESH_CATEGORY"] = "ZZZZ.US"
        with patch.object(E, "INSTRUMENT_PROXY_MAP", patched):
            tickers = {p.ticker for p in self.probe.build_universe()}
        self.assertIn("ZZZZ.US", tickers,
                      "новая запись движка не доехала до замера — значит список "
                      "захардкожен и разойдётся при первой же правке")

    def test_demo_tickers_are_probed_in_engine_form(self) -> None:
        """Вход для ★-решения «переводить ли витрину на Stooq»."""
        probed = {p.ticker for p in self.universe if p.group.startswith("F")}
        self.assertIn("AAPL.US", probed)
        self.assertIn("BTC-USD", probed, "крипто демо-портфеля выпало из замера")
        self.assertIn("KSPI.KZ", probed)


class SymbolCandidatesTest(unittest.TestCase):
    """Формы символа Stooq — ГИПОТЕЗЫ, которые проверяет замер."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.probe = _load_probe_module()

    def test_us_ticker_has_exactly_one_form(self) -> None:
        """`aapl.us` — единственная форма, о которой задание говорит уверенно."""
        self.assertEqual(self.probe.stooq_candidates("AAPL.US"), ["aapl.us"])
        self.assertEqual(self.probe.stooq_candidates("AAPL"), ["aapl.us"])

    def test_kz_ticker_gets_several_hypotheses(self) -> None:
        """Суффиксы Stooq для KZ/IL НЕ проверены — решает замер, не код."""
        cands = self.probe.stooq_candidates("KSPI.KZ")
        self.assertIn("kspi.kz", cands)
        self.assertGreater(len(cands), 1)

    def test_candidates_are_unique(self) -> None:
        """Дубликат кандидата — лишний запрос при ограниченной дневной квоте."""
        for ticker in ("KAP.IL", "KSPI.KZ", "BTC-USD", "AAPL.US"):
            with self.subTest(ticker=ticker):
                cands = self.probe.stooq_candidates(ticker)
                self.assertEqual(len(cands), len(set(cands)))

    def test_crypto_keeps_the_engine_form_first(self) -> None:
        self.assertEqual(self.probe.stooq_candidates("BTC-USD")[0], "btc-usd")


class CsvParsingTest(unittest.TestCase):
    """Тело отказа Stooq приходит с кодом 200 и НЕ является CSV."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.probe = _load_probe_module()

    def test_valid_csv_is_counted(self) -> None:
        body = ("Date,Open,High,Low,Close,Volume\n"
                "2021-01-04,1,2,0.5,1.5,100\n"
                "2021-01-05,1,2,0.5,1.6,120\n")
        n, first, last, err = self.probe.parse_csv(body)
        self.assertIsNone(err)
        self.assertEqual((n, first, last), (2, "2021-01-04", "2021-01-05"))

    def test_quota_message_is_not_mistaken_for_absence(self) -> None:
        """🔴 Иначе исчерпанная квота выглядит как «бумаги нет».

        Таблица покрытия, составленная в такой день, объявила бы половину
        книги непокрытой — и решение «для какой аудитории manual имеет смысл»
        было бы принято по ложным данным.
        """
        n, _f, _l, err = self.probe.parse_csv("Exceeded the daily hits limit")
        self.assertEqual(n, 0)
        self.assertIsNotNone(err)
        self.assertIn("не CSV", err)

    def test_empty_body_is_an_error_not_zero_rows(self) -> None:
        _n, _f, _l, err = self.probe.parse_csv("")
        self.assertIsNotNone(err)

    def test_header_only_csv_has_no_observations(self) -> None:
        n, _f, _l, err = self.probe.parse_csv("Date,Open,High,Low,Close,Volume\n")
        self.assertEqual(n, 0)
        self.assertIsNotNone(err)


class ProbeLoopTest(unittest.TestCase):
    """Перебор форм символа — без сети, на подставном `http_get`."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.probe = _load_probe_module()

    def _probe(self, ticker: str, answers: dict):
        asked: list[str] = []

        def fake_get(_url, params):
            asked.append(params["s"])
            return answers.get(params["s"], "Exceeded the daily hits limit")

        p = self.probe.Probe(group="C. KZ / IL", ticker=ticker, purpose="",
                             candidates=self.probe.stooq_candidates(ticker))
        return self.probe.probe_one(p, "KEY", http_get=fake_get), asked

    def test_first_working_form_wins_and_stops_the_loop(self) -> None:
        rows = "Date,Close\n" + "\n".join(
            f"2021-01-{d:02d},1" for d in range(1, 29))
        result, asked = self._probe("KSPI.KZ", {"kspi.kz": rows})
        self.assertEqual(result.found_as, "kspi.kz")
        self.assertEqual(result.observations, 28)
        self.assertEqual(asked, ["kspi.kz"], "перебор не остановился на найденном")

    def test_short_series_is_not_counted_as_covered(self) -> None:
        """Ряд короче окна бесполезен для ковариации — это не «есть»."""
        rows = "Date,Close\n2021-01-04,1\n"
        result, _ = self._probe("KSPI.KZ", {"kspi.kz": rows})
        self.assertTrue(result.found_as)
        self.assertFalse(result.ok)

    def test_all_forms_failing_reports_the_reasons(self) -> None:
        result, asked = self._probe("KSPI.KZ", {})
        self.assertIsNone(result.found_as)
        self.assertIn("kspi.kz", result.error)
        self.assertGreater(len(asked), 1, "проверена только одна форма из нескольких")

    def test_api_key_never_appears_in_the_report(self) -> None:
        """S-5: ключ не печатается ни в таблице, ни в тексте ошибки."""
        result, _ = self._probe("KSPI.KZ", {})
        table = self.probe.render_table([result])
        self.assertNotIn("KEY", table)


class ConventionProbeTest(unittest.TestCase):
    """События берутся из `KNOWN_SPLITS`, а не из задания (ошибка E-4)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.probe = _load_probe_module()
        cls.events = cls.probe.convention_probes()

    def test_googl_split_date_comes_from_code(self) -> None:
        """🔴 Задание указывает 2022-06-06 — это дата сплита AMZN.

        Ступеньки GOOGL на 2022-06-06 нет НИ ПРИ КАКОЙ конвенции, потому что
        события там не было. Исполнитель, следующий заданию буквально, получил
        бы «ступеньки нет» → «ряд скорректирован» — вывод, не зависящий от
        того, что Stooq на самом деле отдаёт.
        """
        from datetime import date

        googl = [(t, d) for t, d, _ in self.events if t == "GOOGL.US"]
        self.assertEqual(googl, [("GOOGL.US", date(2022, 7, 18))])
        amzn = [(t, d) for t, d, _ in self.events if t == "AMZN.US"]
        self.assertEqual(amzn, [("AMZN.US", date(2022, 6, 6))])

    def test_dividend_and_control_probes_are_present(self) -> None:
        """Сплиты отличают RAW от скорректированного; дивиденды — TR от SPLIT."""
        tickers = {t for t, _d, _w in self.events}
        self.assertIn("SPY.US", tickers, "нет проверки на дивидендную конвенцию")
        self.assertIn("BRK-B.US", tickers, "нет контрольной бездивидендной бумаги")


class ManualPositionWithoutPricesTest(unittest.TestCase):
    """🔴 ЧК-08.2 — ИЗМЕРЕНО: бумага без ряда исчезает из портфеля МОЛЧА.

    Замер (2026-08-06, книга из шести позиций, у одной нет ценового ряда):

        введено         : AAPL 20·150, TLT 30·95, GLD 20·180, QQQ 10·400,
                          IWM 15·200, NOSTOOQ 100·500, CASH:USD 5000
        стоимость книги : $71 450, из них NOSTOOQ — $50 000 (70%)
        в отчёте        : AAPL, GLD, IWM, QQQ, TLT, USD  — NOSTOOQ НЕТ
        total_value     : $20 641
        чекеры          : МОЛЧАТ

    Почему молчат: лестница цен (`_stage_prices`) имеет три ветки — матрица,
    цена брокера, цена покупки для ПРОКСИРОВАННОЙ бумаги. Обычная бумага без
    ряда не подходит ни под одну, `Current_Price` остаётся NaN, и строку
    удаляет `dropna`. C-8 считает покрытие по стоимости УЖЕ выживших строк,
    поэтому потерянные 70% для него не существуют.

    Для брокерского пути дефект почти не виден: `mkt_price` от Freedom ловит
    такую бумагу второй веткой. Для `manual` цены брокера нет ПО ОПРЕДЕЛЕНИЮ,
    поэтому непокрытие Stooq — не край, а штатный исход, и MP-09 обязана
    закрыть это ДО включения флага.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import golden_support as gs

        cls.gs = gs
        scenario = dict(gs.SCENARIOS["manual"])
        scenario["text"] = (
            "AAPL 20 150\nTLT 30 95\nGLD 20 180\nQQQ 10 400\nIWM 15 200\n"
            "NOSTOOQ 100 500\nCASH:USD 5000\n")
        # NOSTOOQ.US в матрице НЕТ — ровно то, что даст Stooq для непокрытой бумаги.
        scenario["tickers"] = ["AAPL.US", "TLT.US", "GLD.US", "QQQ.US", "IWM.US",
                               "SPY.US", "AGG.US", "EEM.US", "EMB.US", "IEF.US",
                               "MTUM.US", "VLUE.US", "QUAL.US", "SPLV.US",
                               "DBC.US", "URTH.US"]
        gs.SCENARIOS["_mp08_uncovered"] = scenario
        cls.results = gs.run_analyze_all("_mp08_uncovered")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.gs.SCENARIOS.pop("_mp08_uncovered", None)

    def _tickers(self) -> set[str]:
        table = self.results["performance_table"]
        return set(table["Ticker"].astype(str))

    def test_position_disappears_from_the_report(self) -> None:
        """Фиксация ФАКТИЧЕСКОГО поведения — вход для решения ЧК-08.2."""
        self.assertNotIn("NOSTOOQ", self._tickers())

    def test_loss_is_invisible_in_every_disclosure_channel(self) -> None:
        """Ни один из трёх реестров раскрытия о потере не знает."""
        self.assertNotIn("NOSTOOQ", self.results["priced_at_cost"])
        self.assertNotIn("NOSTOOQ", self.results.get("broker_priced_only") or [])
        self.assertNotIn("NOSTOOQ",
                         (self.results.get("model_uncovered") or {}).get("names", []))

    def test_portfolio_value_silently_shrinks(self) -> None:
        """Стоимость книги падает на стоимость потерянной позиции."""
        self.assertLess(float(self.results["total_value"]), 30_000.0)

    @unittest.expectedFailure
    def test_uncovered_value_reaches_c8_SPEC_FOR_MP09(self) -> None:
        """СПЕЦИФИКАЦИЯ для MP-09, сегодня НЕ выполняется (осознанно).

        Требование: позиция без ценового ряда обязана либо остаться в книге по
        цене покупки (как проксированная — `priced_at_cost`), либо доехать до
        C-8 как непокрытая СТОИМОСТЬ. Молчаливое удаление недопустимо: отчёт
        строится по ДРУГОМУ портфелю, и ни одного признака этого в нём нет.

        Решение принято до реализации (`PHASE_08 §2.6`), поэтому тест написан
        сейчас и падает намеренно: он станет зелёным ровно тогда, когда MP-09
        закроет дыру, и не даст «забыть» её при включении флага.
        """
        report = self.results.get("data_quality") or {}
        findings = {f.get("id") for f in (report.get("findings") or [])}
        self.assertTrue(
            {"C-8", "C-9"} & findings,
            "потеря 70% стоимости книги не дошла ни до одного чекера")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
