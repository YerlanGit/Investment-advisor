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


class BulkDumpTest(unittest.TestCase):
    """Замер по bulk-выгрузке `stooq.com/db/` — БЕЗ сети и без ключа.

    Находка владельца 2026-08-09: всю историю можно скачать одним архивом, без
    API. Это снимает ОБА препятствия `PHASE_08 §0` разом — капчу и сетевую
    политику, — потому что после скачивания замер идёт по локальному файлу.

    🔴 Формат архива живьём не проверялся, поэтому здесь проверяется НЕ
    «читаем известный формат», а «формат ОПРЕДЕЛЯЕТСЯ и переживает оба
    известных варианта шапки». Захардкодить одну из них означало бы повторить
    E-4: процедура дала бы уверенный ответ, не зависящий от данных.
    """

    #: «Человеческий» CSV — как у эндпоинта выгрузки по тикеру.
    HEADER_PLAIN = "Date,Open,High,Low,Close,Volume"
    #: ASCII-экспорт — второй известный вариант шапки Stooq.
    HEADER_ASCII = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"

    @classmethod
    def setUpClass(cls) -> None:
        cls.probe = _load_probe_module()

    def _rows_plain(self, n: int = 5, start: float = 100.0) -> str:
        out = [self.HEADER_PLAIN]
        for i in range(n):
            price = start + i
            out.append(f"2024-01-{i + 1:02d},{price},{price},{price},{price},1000")
        return "\n".join(out)

    def _rows_ascii(self, n: int = 5, start: float = 100.0) -> str:
        out = [self.HEADER_ASCII]
        for i in range(n):
            price = start + i
            out.append(f"AAPL.US,D,2024010{i + 1},000000,"
                       f"{price},{price},{price},{price},1000,0")
        return "\n".join(out)

    def _make_dir_dump(self, files: dict) -> Path:
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="stooq-dump-"))
        for rel, body in files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        return root

    def _make_zip_dump(self, files: dict) -> Path:
        import tempfile
        import zipfile

        path = Path(tempfile.mkdtemp(prefix="stooq-zip-")) / "d_us_txt.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for rel, body in files.items():
                zf.writestr(rel, body)
        return path

    def test_index_finds_ticker_regardless_of_directory_layout(self) -> None:
        """Раскладка каталогов у выгрузки своя — индексируем ИМЯ ФАЙЛА."""
        for layout in ("data/daily/us/nasdaq stocks/1/aapl.us.txt",
                       "d_us_txt/data/daily/us/aapl.us.txt",
                       "aapl.us.txt"):
            with self.subTest(layout=layout):
                root = self._make_dir_dump({layout: self._rows_plain()})
                index = self.probe.DumpIndex(root)
                self.assertIn("aapl.us", index)
                self.assertIsNotNone(index.member_for("aapl.us"))

    def test_zip_and_directory_are_equivalent(self) -> None:
        """Оператор не обязан решать, распаковывать ли архив."""
        files = {"data/daily/us/aapl.us.txt": self._rows_plain()}
        for maker in (self._make_dir_dump, self._make_zip_dump):
            with self.subTest(kind=maker.__name__):
                index = self.probe.DumpIndex(maker(files))
                self.assertEqual(len(index), 1)
                self.assertIn("aapl.us", index)

    def test_format_is_discovered_for_both_known_headers(self) -> None:
        """🔴 Формат ОПРЕДЕЛЯЕТСЯ, а не утверждается."""
        cases = {
            "плоский CSV": (self._rows_plain(), "date", "close", "YYYY-MM-DD"),
            "ASCII-экспорт": (self._rows_ascii(), "<date>", "<close>", "YYYYMMDD"),
        }
        for name, (body, date_col, close_col, pattern) in cases.items():
            with self.subTest(header=name):
                index = self.probe.DumpIndex(
                    self._make_dir_dump({"x/aapl.us.txt": body}))
                fmt = self.probe.sniff_format(index)
                self.assertEqual(fmt.date_column, date_col)
                self.assertEqual(fmt.close_column, close_col)
                self.assertEqual(fmt.date_pattern, pattern)
                self.assertEqual(fmt.members_seen, 1)

    def test_series_parses_under_both_headers_identically(self) -> None:
        """Разные шапки — один и тот же ряд: иначе конвенция считалась бы по-разному."""
        parsed = []
        for body in (self._rows_plain(), self._rows_ascii()):
            index = self.probe.DumpIndex(self._make_dir_dump({"a/aapl.us.txt": body}))
            fmt = self.probe.sniff_format(index)
            parsed.append(self.probe.parse_dump_series(body, fmt))
        self.assertEqual(parsed[0], parsed[1])
        self.assertEqual(parsed[0][0], ("2024-01-01", 100.0))

    def test_coverage_probe_reports_observations_and_period(self) -> None:
        index = self.probe.DumpIndex(self._make_dir_dump(
            {"a/aapl.us.txt": self._rows_plain(n=4)}))
        fmt = self.probe.sniff_format(index)
        p = self.probe.Probe(group="F. Демо", ticker="AAPL.US", purpose="",
                             candidates=self.probe.stooq_candidates("AAPL.US"))
        result = self.probe.probe_one_dump(p, [index], fmt)
        self.assertEqual(result.found_as, "aapl.us")
        self.assertEqual(result.observations, 4)
        self.assertEqual(result.first_date, "2024-01-01")
        self.assertEqual(result.member, "a/aapl.us.txt")

    def test_short_series_is_not_counted_as_covered(self) -> None:
        """Ряд короче окна движка бесполезен для ковариации — это не «есть»."""
        index = self.probe.DumpIndex(self._make_dir_dump(
            {"a/aapl.us.txt": self._rows_plain(n=3)}))
        fmt = self.probe.sniff_format(index)
        p = self.probe.Probe(group="A. Факторы", ticker="AAPL.US", purpose="",
                             candidates=["aapl.us"])
        self.assertFalse(self.probe.probe_one_dump(p, [index], fmt).ok)

    def test_missing_ticker_is_reported_not_crashed(self) -> None:
        index = self.probe.DumpIndex(self._make_dir_dump(
            {"a/aapl.us.txt": self._rows_plain()}))
        fmt = self.probe.sniff_format(index)
        p = self.probe.Probe(group="C. KZ / IL", ticker="KSPI.KZ", purpose="",
                             candidates=self.probe.stooq_candidates("KSPI.KZ"))
        result = self.probe.probe_one_dump(p, [index], fmt)
        self.assertIsNone(result.found_as)
        self.assertIn("не найден", result.error)

    def test_several_dumps_are_searched_in_order(self) -> None:
        """US и world — разные пакеты; бумага может лежать в любом."""
        us = self.probe.DumpIndex(self._make_dir_dump(
            {"a/aapl.us.txt": self._rows_plain()}))
        world = self.probe.DumpIndex(self._make_dir_dump(
            {"b/kspi.kz.txt": self._rows_plain()}))
        fmt = self.probe.sniff_format(us)
        p = self.probe.Probe(group="C. KZ / IL", ticker="KSPI.KZ", purpose="",
                             candidates=self.probe.stooq_candidates("KSPI.KZ"))
        self.assertEqual(self.probe.probe_one_dump(p, [us, world], fmt).found_as,
                         "kspi.kz")

    def test_split_step_is_detected_as_raw_series(self) -> None:
        """Ступенька в дату сплита → ряд СЫРОЙ. Даты — из `KNOWN_SPLITS`."""
        from datetime import date, timedelta

        rows = ["Date,Open,High,Low,Close,Volume"]
        split_day = date(2024, 6, 10)                 # NVDA 10:1, из кода
        for i in range(-4, 0):
            d = split_day + timedelta(days=i)
            rows.append(f"{d.isoformat()},1000,1000,1000,1000,1")
        for i in range(0, 4):
            d = split_day + timedelta(days=i)
            rows.append(f"{d.isoformat()},100,100,100,100,1")
        index = self.probe.DumpIndex(self._make_dir_dump(
            {"a/nvda.us.txt": "\n".join(rows)}))
        fmt = self.probe.sniff_format(index)
        nvda = [e for e in self.probe.check_convention([index], fmt)
                if e["ticker"] == "NVDA.US"]
        self.assertEqual(len(nvda), 1)
        self.assertAlmostEqual(nvda[0]["ratio"], 10.0, places=1)
        self.assertIn("RAW", nvda[0]["verdict"])

    def test_absent_step_is_reported_as_adjusted(self) -> None:
        """Ровный ряд через дату сплита → ряд уже скорректирован."""
        from datetime import date, timedelta

        rows = ["Date,Open,High,Low,Close,Volume"]
        split_day = date(2024, 6, 10)
        for i in range(-4, 4):
            d = split_day + timedelta(days=i)
            rows.append(f"{d.isoformat()},100,100,100,100,1")
        index = self.probe.DumpIndex(self._make_dir_dump(
            {"a/nvda.us.txt": "\n".join(rows)}))
        fmt = self.probe.sniff_format(index)
        nvda = [e for e in self.probe.check_convention([index], fmt)
                if e["ticker"] == "NVDA.US"][0]
        self.assertIn("скорректирован", nvda["verdict"])

    def test_garbage_lines_are_skipped_not_fatal(self) -> None:
        body = (self._rows_plain(n=3) + "\n"
                "2024-01-09,,,,,\n"           # пустая цена
                "мусор\n"
                "2024-01-10,1,1,1,7.5,1\n")
        index = self.probe.DumpIndex(self._make_dir_dump({"a/aapl.us.txt": body}))
        fmt = self.probe.sniff_format(index)
        rows = self.probe.parse_dump_series(body, fmt)
        self.assertEqual(rows[-1], ("2024-01-10", 7.5))
        self.assertEqual(len(rows), 4)

    def test_empty_dump_does_not_crash_format_sniffing(self) -> None:
        index = self.probe.DumpIndex(self._make_dir_dump({}))
        fmt = self.probe.sniff_format(index)
        self.assertIsNone(fmt.header)
        self.assertIn("определить не удалось", self.probe.render_format_note(fmt))

    def test_no_network_call_is_possible_in_dump_mode(self) -> None:
        """Гарантия офлайна: путь выгрузки не трогает `_http_get` вовсе."""
        from unittest.mock import patch

        index = self.probe.DumpIndex(self._make_dir_dump(
            {"a/aapl.us.txt": self._rows_plain()}))
        fmt = self.probe.sniff_format(index)
        p = self.probe.Probe(group="F. Демо", ticker="AAPL.US", purpose="",
                             candidates=["aapl.us"])
        with patch.object(self.probe, "_http_get",
                          side_effect=AssertionError("сетевой вызов в режиме --dump")):
            self.probe.probe_one_dump(p, [index], fmt)
            self.probe.check_convention([index], fmt)


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
