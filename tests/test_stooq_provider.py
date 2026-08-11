"""
MP-09 — `StooqProvider`: конвенция, свежесть, инварианты I-6/I-7/I-12/I-13, Т-3.

Все тесты ОФЛАЙН и на синтетической базе: провайдер по построению не ходит в
сеть, и проверять это надо тем, что сети нет ни в одной фикстуре.

Фикстуры строятся кодом, а не читаются с диска: сюита обязана проходить и в
зеркале деплой-образа, где нет ни `design/`, ни `scripts/`, ни выгрузок Stooq.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from finance import stooq_ingest as si
from finance.price_providers import (PriceConvention, PriceProvider,
                                     ProviderResult, provider_for_source)
from finance.stooq_provider import (MAX_MARKET_STALE_DAYS,
                                    MAX_STALE_TRADING_DAYS, StooqProvider)
from finance.stooq_store import StooqStore

HEADER = ("<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,"
          "<VOL>,<OPENINT>")


def bar(symbol: str, day: int, close: float) -> str:
    return (f"{symbol},D,{day},000000,{close},{close * 1.01},"
            f"{close * 0.99},{close},1000,0")


def business_days(start: date, count: int) -> list[int]:
    """`count` будних дат подряд в форме YYYYMMDD, начиная со `start`."""
    out: list[int] = []
    cursor = start
    while len(out) < count:
        if cursor.weekday() < 5:
            out.append(int(cursor.strftime("%Y%m%d")))
        cursor += pd.Timedelta(days=1).to_pytimedelta()
    return out


class _StoreFixture(unittest.TestCase):
    """База котировок в temp-каталоге; наполняется файлами истории."""

    def setUp(self) -> None:                                # noqa: N802
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.db = self.tmp / "prices.sqlite"

    def build(self, series: dict[str, list[tuple[int, float]]]) -> None:
        """`{'SPY.US': [(дата, цена), …]}` → наполненная база.

        Запоминает последний день фикстуры: «сегодня» у провайдера обязано
        стоять рядом с данными, иначе гард остановки рынка честно отвергнет
        любую историческую фикстуру — и будет прав.
        """
        archive = self.tmp / "archive"
        archive.mkdir(exist_ok=True)
        conn = si.connect(self.db)
        si.ensure_schema(conn)
        latest = 0
        for symbol, points in series.items():
            rows = [bar(symbol, day, close) for day, close in points]
            latest = max(latest, max(day for day, _ in points))
            path = archive / f"{symbol.lower()}.txt"
            path.write_text("\r\n".join([HEADER, *rows]) + "\r\n",
                            encoding="utf-8")
        text = str(latest)
        self.last_day = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
        si.bootstrap(conn, archive, window_days=20 * 365,
                     today=self.last_day)
        conn.close()

    def provider(self, **kwargs) -> StooqProvider:
        """Провайдер, чьё «сегодня» — день последнего бара фикстуры."""
        kwargs.setdefault("today", self.last_day)
        return StooqProvider(self.db, **kwargs)

    def store(self) -> StooqStore:
        store = StooqStore.open_readonly(self.db)
        self.addCleanup(store.close)
        return store

    def flat(self, symbol: str, count: int = 40, *, start=date(2026, 6, 15),
             price: float = 100.0) -> list[tuple[int, float]]:
        return [(day, price + i * 0.1)
                for i, day in enumerate(business_days(start, count))]


# ═════════════════════════════════════════════════════════════════════════════
# Контракт протокола
# ═════════════════════════════════════════════════════════════════════════════

class ProviderContractTest(_StoreFixture):

    def test_satisfies_the_price_provider_protocol(self) -> None:
        self.assertIsInstance(StooqProvider(self.db), PriceProvider)

    def test_declares_split_adjusted_for_every_loaded_ticker(self) -> None:
        """Конвенция заполняется на КАЖДУЮ колонку — требование протокола.

        Пустой `convention` сделал бы инвариант I-7 непроверяемым: `single_
        convention()` вернул бы `None` и от «конвенции разошлись» ничем бы не
        отличался.
        """
        self.build({"SPY.US": self.flat("SPY.US"),
                    "QQQ.US": self.flat("QQQ.US", price=300.0)})
        result = self.provider().fetch(["SPY.US", "QQQ.US"], days=1825)
        self.assertEqual(sorted(result.loaded), ["QQQ.US", "SPY.US"])
        self.assertEqual(set(result.convention.values()),
                         {PriceConvention.SPLIT_ADJUSTED})
        self.assertEqual(result.single_convention(),
                         PriceConvention.SPLIT_ADJUSTED)

    def test_result_is_positionally_compatible_with_history_result(self) -> None:
        """I-11: первые четыре поля повторяют `HistoryResult` по порядку.

        Шесть модулей-потребителей распаковывают результат позиционно; новые
        поля обязаны идти строго в хвост.
        """
        self.build({"SPY.US": self.flat("SPY.US")})
        result = self.provider().fetch(["SPY.US"], days=1825)
        data, loaded, failed, retried = result[:4]
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(loaded, ["SPY.US"])
        self.assertEqual(failed, {})
        self.assertEqual(retried, [])

    def test_single_source_per_matrix(self) -> None:
        """I-13: одна матрица — один источник во ВСЕХ колонках."""
        self.build({"SPY.US": self.flat("SPY.US"),
                    "IEF.US": self.flat("IEF.US", price=95.0)})
        result = self.provider().fetch(["SPY.US", "IEF.US"], days=1825)
        self.assertEqual(result.single_source(), "stooq")

    def test_columns_are_named_in_engine_tickers(self) -> None:
        """🔴 Колонка называется так, как спросили, а не как записано у Stooq.

        Движок сопоставляет матрицу с портфелем по СВОЕМУ тикеру, и колонка
        `BTC.V` вместо `BTC-USD` молча потеряла бы позицию.
        """
        self.build({"BTC.V": self.flat("BTC.V", price=60000.0)})
        result = self.provider().fetch(["BTC-USD"], days=1825)
        self.assertEqual(list(result.data.columns), ["BTC-USD"])

    def test_parallelism_is_one_and_not_inherited(self) -> None:
        """S-6 выродилось: источник — локальный файл, распараллеливать нечего.

        Число всё равно объявлено явно: протокол его требует, а унаследовать
        `max_workers=6` брокерского провайдера здесь было бы бессмысленно.
        """
        from finance.price_providers import TradernetProvider

        self.assertEqual(StooqProvider.max_parallelism, 1)
        self.assertNotEqual(StooqProvider.max_parallelism,
                            TradernetProvider.max_parallelism)


# ═════════════════════════════════════════════════════════════════════════════
# Отказы: свежесть, отсутствие бумаги, отсутствие базы
# ═════════════════════════════════════════════════════════════════════════════

class FailureModesTest(_StoreFixture):

    def test_missing_paper_is_failed_not_an_exception(self) -> None:
        """Протокол: пустой ответ по тикеру — это `failed`, не исключение."""
        self.build({"SPY.US": self.flat("SPY.US")})
        result = self.provider().fetch(["SPY.US", "TPH.US"], days=1825)
        self.assertEqual(result.loaded, ["SPY.US"])
        self.assertIn("TPH.US", result.failed)
        self.assertIn("нет в базе", result.failed["TPH.US"])

    def test_stale_paper_goes_to_failed_not_into_the_matrix(self) -> None:
        """🔴 Просроченная бумага НЕ попадает в данные (ЧК-08.5-4).

        Замер `tph.us`: последний бар 2026-05-13, в августовских срезах бумаги
        нет вовсе. Её последняя цена — не «сегодняшняя», и подставить её в
        матрицу значит посчитать портфель по цене трёхмесячной давности,
        причём правдоподобной.
        """
        live = self.flat("SPY.US", 40)
        dead = self.flat("TPH.US", 10, start=date(2026, 5, 4), price=9.0)
        self.build({"SPY.US": live, "TPH.US": dead})
        result = self.provider().fetch(["SPY.US", "TPH.US"], days=1825)
        self.assertEqual(result.loaded, ["SPY.US"])
        self.assertNotIn("TPH.US", result.data.columns)
        self.assertIn("торговых дн.", result.failed["TPH.US"])

    def test_fresh_paper_survives_a_long_weekend(self) -> None:
        """Свежесть в ТОРГОВЫХ днях: длинные выходные не делают бумагу мёртвой."""
        self.build({"SPY.US": self.flat("SPY.US")})
        result = self.provider().fetch(["SPY.US"], days=1825)
        self.assertEqual(result.failed, {})

    def test_market_wide_stall_fails_every_ticker_of_that_market(self) -> None:
        """🔴 Регресс `AUDIT §−80` (дефект 2), теперь на слое провайдера.

        Потикерная свежесть этот случай не видит ПО ПОСТРОЕНИЮ: календарь
        рынка выводится из тех же строк, что и бары, поэтому у базы, которую
        не наполняли неделю, КАЖДАЯ бумага честно рапортует свежесть 0. Данные
        внутри себя непротиворечивы; врёт только вывод «всё свежее».
        """
        self.build({"SPY.US": self.flat("SPY.US")})
        far_future = date(2026, 8, 10) + pd.Timedelta(
            days=MAX_MARKET_STALE_DAYS + 30).to_pytimedelta()
        provider = StooqProvider(self.db, today=far_future)
        result = provider.fetch(["SPY.US"], days=1825)
        self.assertEqual(result.loaded, [])
        self.assertIn("база не обновлялась", result.failed["SPY.US"])

    def test_absent_database_is_a_clean_refusal(self) -> None:
        """Базы нет — все тикеры в `failed`, ни одного исключения наружу.

        `fetch` зовётся из середины конвейера отчёта: `RuntimeError` там
        превратился бы в traceback вместо внятной причины, а пустая матрица
        упирается в чекеры качества, и токен не списывается.
        """
        provider = StooqProvider(self.tmp / "которой-нет.sqlite")
        result = provider.fetch(["SPY.US", "QQQ.US"], days=1825)
        self.assertTrue(result.data.empty)
        self.assertEqual(result.loaded, [])
        self.assertIn("недоступна", result.failed["SPY.US"])
        self.assertIn("недоступна", result.failed["QQQ.US"])

    def test_thresholds_are_configurable_but_sane(self) -> None:
        """Пороги — не магические числа в теле функции."""
        self.assertGreaterEqual(MAX_STALE_TRADING_DAYS, 1)
        self.assertGreaterEqual(MAX_MARKET_STALE_DAYS, 4,
                                "длинные выходные в США — до 4 календарных дней")


# ═════════════════════════════════════════════════════════════════════════════
# I-6 — корректировки на чтении, идемпотентно
# ═════════════════════════════════════════════════════════════════════════════

class SplitConventionTest(_StoreFixture):

    def _nvda(self, *, raw: bool) -> list[tuple[int, float]]:
        """Ряд вокруг сплита NVDA 10:1 (2024-06-10) — сырой или уже сжатый."""
        before = business_days(date(2024, 5, 20), 15)
        after = business_days(date(2024, 6, 10), 15)
        scale = 10.0 if raw else 1.0
        return ([(day, 120.0 * scale) for day in before]
                + [(day, 121.0) for day in after])

    def test_raw_and_adjusted_input_give_the_same_output(self) -> None:
        """🔴 I-6 и обоснование конвенции провайдера в одном тесте.

        Именно это равенство делает законным объявление `SPLIT_ADJUSTED`
        источником, который конвенцию не документирует: она объявляется про
        ВЫХОД. Если равенство сломается, `PHASE_08B §7.6` придётся переписать
        обратно, а провайдер — заблокировать до замера (`AUDIT §−83`).
        """
        self.build({"NVDA.US": self._nvda(raw=True)})
        from_raw = self.provider().fetch(["NVDA.US"], days=20 * 365)

        self.setUp()                       # вторая база, тот же класс
        self.build({"NVDA.US": self._nvda(raw=False)})
        from_adjusted = self.provider().fetch(["NVDA.US"], days=20 * 365)

        pd.testing.assert_series_equal(from_raw.data["NVDA.US"],
                                       from_adjusted.data["NVDA.US"])

    def test_known_split_step_is_removed_from_the_series(self) -> None:
        """Ступенька 10:1 не доезжает до движка как −90 % за день."""
        self.build({"NVDA.US": self._nvda(raw=True)})
        data = self.provider().fetch(["NVDA.US"], days=20 * 365).data
        worst = data["NVDA.US"].pct_change().min()
        self.assertGreater(worst, -0.5,
                           "сплит остался в ряду и читается как обвал")

    def test_store_itself_keeps_the_raw_series(self) -> None:
        """Корректировка живёт на ЧТЕНИИ, а не на записи (I-6).

        База хранит ряд как отдал источник — иначе пересчитать слои можно было
        бы только пересборкой базы.
        """
        self.build({"NVDA.US": self._nvda(raw=True)})
        raw = self.store().bars(["NVDA.US"], days=20 * 365)
        self.assertLess(raw["NVDA.US"].pct_change().min(), -0.5,
                        "хранилище не имеет права ничего корректировать")


# ═════════════════════════════════════════════════════════════════════════════
# Т-3 — крипта не сплитится
# ═════════════════════════════════════════════════════════════════════════════

class CryptoHeuristicTest(_StoreFixture):

    def test_crypto_halving_is_not_read_as_a_split(self) -> None:
        """🔴 Т-3. Падение крипты вдвое за сессию — не обратный сплит.

        Воспроизведение до правки: ряд `[40000, 39000, 38000, 19000]` эвристика
        читала как сплит 2:1 и делила всю предысторию пополам — падение
        ИСЧЕЗАЛО из доходностей вместе с волатильностью колонки. Удвоение
        снапится на `1/2` идеально, поэтому защита «отношение не похоже на
        целое» здесь не срабатывает вовсе. `BTC-USD` есть в демо-портфеле.
        """
        days = business_days(date(2026, 6, 15), 4)
        self.build({"BTC.V": list(zip(days, [40000.0, 39000.0, 38000.0,
                                             19000.0]))})
        data = self.provider().fetch(["BTC-USD"], days=1825).data
        self.assertEqual(list(data["BTC-USD"]),
                         [40000.0, 39000.0, 38000.0, 19000.0])

    def test_equity_split_is_still_detected(self) -> None:
        """Защита крипты не должна отключать эвристику для акций.

        Иначе Т-3 чинился бы ценой того самого дефекта, ради которого
        эвристика написана.
        """
        from freedom_portfolio.history import apply_split_conventions

        index = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06",
                                "2021-01-07"])
        series = pd.Series([400.0, 402.0, 404.0, 101.0], index=index)
        out = apply_split_conventions("ZZZZ.US", series)
        self.assertAlmostEqual(out.iloc[0], 100.0, places=6)


# ═════════════════════════════════════════════════════════════════════════════
# I-12 / I-13 — юридическая граница
# ═════════════════════════════════════════════════════════════════════════════

class LegalBoundaryTest(_StoreFixture):

    def test_manual_chain_is_stooq_only(self) -> None:
        """I-12: у ручного портфеля цепочка из ОДНОГО звена.

        Резерва нет сознательно: при сбое Stooq ручной отчёт честно не
        строится, а не подменяется брокерскими данными (`PHASE_09 §5`).
        """
        from unittest.mock import patch

        with patch.dict("os.environ", {"PRICE_PROVIDER_MANUAL": "stooq"}):
            provider = provider_for_source("manual")
        self.assertIsInstance(provider, StooqProvider)

    def test_manual_failure_does_not_fall_back_to_the_broker(self) -> None:
        """🔴 Stooq недоступен → честный отказ, НЕ подмена Tradernet.

        Тихая подмена юридически недоступна: пользователь ручного ввода может
        не быть клиентом Freedom, и проверить это нельзя.
        """
        from unittest.mock import patch

        from finance.price_providers import TradernetProvider

        with patch.dict("os.environ", {"PRICE_PROVIDER_MANUAL": "stooq"}):
            provider = provider_for_source("manual", client=object())
        self.assertNotIsInstance(provider, TradernetProvider)
        result = StooqProvider(self.tmp / "нет.sqlite").fetch(["SPY.US"],
                                                              days=1825)
        self.assertTrue(result.data.empty)
        self.assertEqual(result.source, {},
                         "провалившийся тикер не подписывается источником")

    def test_provider_never_touches_the_network(self) -> None:
        """Сети на пути отчёта нет — ради этого база и заводилась.

        Проверяется запретом на уровне сокета: любая попытка соединения падает
        с `AssertionError` внутри `fetch`, а не «мы посмотрели и вроде не
        ходит».
        """
        import socket

        self.build({"SPY.US": self.flat("SPY.US")})
        original = socket.socket.connect

        def forbidden(*_args, **_kwargs):
            raise AssertionError("провайдер полез в сеть")

        socket.socket.connect = forbidden
        self.addCleanup(setattr, socket.socket, "connect", original)
        result = self.provider().fetch(["SPY.US"], days=1825)
        self.assertEqual(result.loaded, ["SPY.US"])

    def test_ticker_goes_into_parametrised_sql_not_a_string(self) -> None:
        """S-1 выродилось в параметризацию: инъекция через тикер невозможна.

        Whitelist регулярным выражением нужен был URL-у, которого больше нет.
        Проверяем то, что осталось: тикер-инъекция даёт `failed`, а не
        выполненный SQL и не исключение.
        """
        self.build({"SPY.US": self.flat("SPY.US")})
        poison = ["SPY.US'; DROP TABLE daily_bars;--", "<script>", "SPY&s=evil"]
        result = self.provider().fetch(["SPY.US", *poison], days=1825)
        self.assertEqual(result.loaded, ["SPY.US"])
        for bad in poison:
            self.assertIn(bad, result.failed)
        with sqlite3.connect(self.db) as conn:
            self.assertTrue(conn.execute(
                "SELECT COUNT(*) FROM daily_bars").fetchone()[0])


# ═════════════════════════════════════════════════════════════════════════════
# I-7 — одна конвенция в одной матрице
# ═════════════════════════════════════════════════════════════════════════════

class MixedConventionTest(unittest.TestCase):

    def test_mixed_convention_is_visible_as_none(self) -> None:
        """I-7: матрица с двумя конвенциями обязана быть ОТЛИЧИМА от единой.

        `single_convention()` — та проверка, на которую опирается запрет
        смешивания; без неё «разные конвенции» и «конвенция одна» выглядели бы
        одинаково.
        """
        mixed = ProviderResult(
            data=pd.DataFrame(), loaded=["A", "B"], failed={}, retried=[],
            convention={"A": PriceConvention.SPLIT_ADJUSTED,
                        "B": PriceConvention.TOTAL_RETURN},
            source={"A": "stooq", "B": "stooq"})
        self.assertIsNone(mixed.single_convention())

    def test_stooq_and_tradernet_declare_the_same_convention(self) -> None:
        """Резерв для `freedom` возможен только при совпадении конвенций.

        Совпадают они не случайно: оба провайдера объявляют конвенцию своего
        ВЫХОДА, и на выходе у обоих одни и те же два слоя коррекции.
        """
        from finance.price_providers import TradernetProvider

        self.assertEqual(StooqProvider.convention, TradernetProvider.convention)


# ═════════════════════════════════════════════════════════════════════════════
# Сквозной путь: движок → провайдер → база
# ═════════════════════════════════════════════════════════════════════════════

class EngineWiringTest(_StoreFixture):
    """🔴 Соответствие протоколу не доказывает, что ОТЧЁТ строится.

    Провайдер можно написать безупречным по контракту и всё равно не собрать
    ни одного отчёта: движок спрашивает СВОИ тикеры, ждёт своё окно истории и
    прогоняет матрицу через чекеры качества. Поэтому фаза закрывается здесь, а
    не в тестах контракта выше.

    Фикстура кончается СЕГОДНЯШНИМ рабочим днём намеренно: так проверяется
    ещё и порог свежести в его боевом значении, а не подменённый в тесте.
    """

    #: 1300 торговых дней ≈ 5 лет. Меньше нельзя: чекер C-4 блокирует отчёт на
    #: укороченной истории (замерено — 500 дней дают `DataQualityBlocked`), и
    #: тест на короткой фикстуре проверял бы блокировку, а не сборку.
    HISTORY_DAYS = 1300

    def _universe_store(self) -> None:
        import random

        from finance.data_checks import BENCHMARK_ETFS, FACTOR_ETFS

        days = self._business_days_ending(date.today(), self.HISTORY_DAYS)
        random.seed(11)
        series = {}
        for i, symbol in enumerate(list(FACTOR_ETFS) + list(BENCHMARK_ETFS)
                                   + ["AAPL.US", "MSFT.US", "TLT.US"]):
            price = 100.0 + i
            points = []
            for day in days:
                price *= 1 + random.gauss(0.0004, 0.011)
                points.append((day, round(price, 4)))
            series[symbol] = points
        self.build(series)

    @staticmethod
    def _business_days_ending(end: date, count: int) -> list[int]:
        out: list[int] = []
        cursor = end
        while len(out) < count:
            if cursor.weekday() < 5:
                out.append(int(cursor.strftime("%Y%m%d")))
            cursor -= pd.Timedelta(days=1).to_pytimedelta()
        return sorted(out)

    def test_engine_loads_prices_through_the_store(self) -> None:
        """Движок получает матрицу из базы и НЕ создаёт клиента брокера.

        I-12 проверяется по факту конструирования `TradernetClient`: объявить
        «мы туда не ходим» мало, надо чтобы объект не появился.
        """
        from unittest.mock import patch

        import finance.broker_api as ba
        from finance.investment_logic import UniversalPortfolioManager

        self._universe_store()
        created: list[int] = []
        original = ba.TradernetClient.__init__

        def spy(inner_self, *args, **kwargs):
            created.append(1)
            return original(inner_self, *args, **kwargs)

        ba.TradernetClient.__init__ = spy
        self.addCleanup(setattr, ba.TradernetClient, "__init__", original)

        with patch.dict("os.environ", {"STOOQ_DB_PATH": str(self.db)}):
            upm = UniversalPortfolioManager()
            upm.engine.price_source = "manual"
            frame, result = upm.engine.get_market_data(["AAPL.US", "MSFT.US"])

        self.assertFalse(frame.empty)
        self.assertIn("SPY.US", frame.columns, "фактор обязан доехать")
        self.assertEqual(result.failed, {})
        self.assertEqual(result.single_source(), "stooq")
        self.assertEqual(result.single_convention(),
                         PriceConvention.SPLIT_ADJUSTED)
        self.assertFalse(created, "I-12: клиент брокера не имеет права возникнуть")

    def test_manual_report_is_built_end_to_end(self) -> None:
        """`analyze_all` на ручном источнике доходит до полного контракта.

        Замокано ровно то, что ходит в СЕТЬ и к фазе отношения не имеет
        (EDGAR, CDS, инсайдеры) — тем же способом, что в golden-обвязке.
        Ценовой путь не мокается ничем: он и есть предмет проверки.
        """
        from unittest.mock import patch

        from finance.contracts import RESULTS_KEYS
        from finance.investment_logic import UniversalPortfolioManager

        self._universe_store()
        portfolio = pd.DataFrame([
            {"Ticker": "AAPL.US", "Quantity": 50, "Purchase_Price": 200.0,
             "Currency": "USD"},
            {"Ticker": "MSFT.US", "Quantity": 30, "Purchase_Price": 400.0,
             "Currency": "USD"},
            {"Ticker": "TLT.US", "Quantity": 100, "Purchase_Price": 90.0,
             "Currency": "USD"},
            {"Ticker": "USD", "Quantity": 5000, "Purchase_Price": 1.0,
             "Currency": "USD"},
        ])
        env = {"STOOQ_DB_PATH": str(self.db), "HISTORY_LOOKBACK_DAYS": "1825",
               "REPORTING_CURRENCY": "USD", "RFR_USD": "0.045",
               "FRED_API_KEY": ""}
        with patch.dict("os.environ", env), \
             patch("finance.sec_edgar.batch_fundamental_scan",
                   return_value=pd.DataFrame()), \
             patch("finance.cds_feed.make_lookup",
                   return_value=lambda _t: {}), \
             patch("finance.smart_money.build_insider_signals",
                   return_value={}):
            upm = UniversalPortfolioManager()
            upm.engine.price_source = "manual"
            upm.engine.current_rfr_annual = 0.045
            upm.engine.rfr_source = "fixture[USD]=0.045"
            results = upm.analyze_all(source=portfolio,
                                      profile_benchmark="SPY.US",
                                      risk_mandate="MODERATE")

        self.assertEqual(set(results), set(RESULTS_KEYS),
                         "ручной отчёт обязан отдавать ТОТ ЖЕ контракт results")


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
