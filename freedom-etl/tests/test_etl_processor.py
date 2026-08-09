"""T-1 конвейер: планирование окон, часовые пояса бирж, отсев, идемпотентность.

БД подменена журналирующим двойником. Это не упрощение ради скорости: он
запоминает ПОРЯДОК вызовов, а порядок здесь — инвариант («сначала свечи, потом
курсор»), который на живом PostgreSQL проверялся бы только падением процесса
в нужный миллисекунд.
"""

from __future__ import annotations

import unittest
from datetime import date, datetime
from decimal import Decimal

from config.config import Settings
from services.etl_processor import (
    EtlProcessor,
    exchange_timezone,
    to_candles,
    trade_date_of,
)
from services.tradernet_client import RawCandle, TradernetError

# 2026-08-05 20:00 UTC — внутри торговой сессии США.
TS_AUG5_2000_UTC = 1785960000
# 2026-08-06 01:00 UTC — уже 6-е по UTC, но всё ещё 5-е в Нью-Йорке.
TS_AUG6_0100_UTC = 1785978000


def candle(ts: int, close: float = 100.0, volume: float | None = 1000.0,
           *, open_=None, high=None, low=None) -> RawCandle:
    return RawCandle(ts=ts, open=open_, high=high, low=low, close=close, volume=volume)


def settings(**overrides) -> Settings:
    base = dict(initial_lookback_days=1825, resync_overlap_days=3,
                include_current_day=False, batch_size=50)
    base.update(overrides)
    return Settings(**base)


# ── Тестовые двойники ────────────────────────────────────────────────────────


class FakeDb:
    def __init__(self, tickers: list[str], state: dict[str, date | None] | None = None) -> None:
        self._tickers = tickers
        self._state = state or {}
        self.calls: list[str] = []
        self.written: list = []
        self.cursors: list[tuple[str, date]] = []

    async def fetch_tracked_tickers(self) -> list[str]:
        return list(self._tickers)

    async def fetch_sync_state(self) -> dict[str, date | None]:
        return dict(self._state)

    async def upsert_candles(self, candles) -> int:
        self.calls.append("candles")
        self.written.extend(candles)
        return len(candles)

    async def upsert_sync_status(self, states) -> int:
        self.calls.append("sync_status")
        self.cursors.extend(states)
        return len(states)


class FakeClient:
    def __init__(self, responses: dict | list, batch_size: int = 50) -> None:
        self._responses = responses
        self.batch_size = batch_size
        self.requests: list[list[str]] = []

    async def fetch_daily_candles(self, tickers, date_from, date_to):
        self.requests.append(list(tickers))
        if isinstance(self._responses, list):
            result = self._responses.pop(0)
        else:
            result = self._responses
        if isinstance(result, Exception):
            raise result
        return {t: result[t] for t in tickers if t in result}


# ── Часовые пояса ────────────────────────────────────────────────────────────


class ExchangeTimezoneTest(unittest.TestCase):

    def test_us_suffix_maps_to_new_york(self) -> None:
        self.assertEqual(str(exchange_timezone("AAPL.US")), "America/New_York")

    def test_kz_suffix_maps_to_almaty(self) -> None:
        self.assertEqual(str(exchange_timezone("KEGC.KZ")), "Asia/Almaty")

    def test_unknown_suffix_falls_back_to_utc_without_raising(self) -> None:
        """Один незнакомый тикер не должен ронять прогон по всему универсуму."""
        self.assertEqual(str(exchange_timezone("XYZ.MOEX")), "UTC")

    def test_trade_date_uses_exchange_not_utc(self) -> None:
        """🔴 Ядро требования «учитывать часовой пояс биржи».

        В 01:00 UTC 6 августа в Нью-Йорке ещё 5-е. Взять `.date()` в UTC —
        значит записать сессию 5-го числом 6-го: ряд останется правдоподобным
        и будет неверным.
        """
        self.assertEqual(
            trade_date_of(TS_AUG6_0100_UTC, exchange_timezone("AAPL.US")),
            date(2026, 8, 5),
        )
        self.assertEqual(
            datetime.utcfromtimestamp(TS_AUG6_0100_UTC).date(),
            date(2026, 8, 6),
            "предпосылка теста: по UTC это уже другой день",
        )

    def test_kz_and_us_can_disagree_on_the_same_instant(self) -> None:
        """Один и тот же миг — разные торговые даты на разных биржах."""
        self.assertEqual(
            trade_date_of(TS_AUG5_2000_UTC, exchange_timezone("AAPL.US")),
            date(2026, 8, 5))
        self.assertEqual(
            trade_date_of(TS_AUG5_2000_UTC, exchange_timezone("HSBK.KZ")),
            date(2026, 8, 6))


# ── Transform ────────────────────────────────────────────────────────────────


class TransformTest(unittest.TestCase):

    def test_close_is_converted_to_decimal_without_binary_tail(self) -> None:
        """`Decimal(19.99)` = 19.98999…; в NUMERIC(12,4) это другая цена."""
        candles, _ = to_candles("AAPL.US", [candle(TS_AUG5_2000_UTC, 19.99)],
                                max_date=date(2026, 8, 6))
        self.assertEqual(candles[0].close, Decimal("19.9900"))

    def test_zero_volume_rows_are_dropped(self) -> None:
        """Торгов не было — биржа перенесла вчерашнее закрытие; это ложный нулевой доход."""
        candles, rejected = to_candles(
            "AAPL.US", [candle(TS_AUG5_2000_UTC, 100.0, volume=0)],
            max_date=date(2026, 8, 6))
        self.assertEqual(candles, [])
        self.assertEqual(rejected, 1)

    def test_missing_volume_is_kept(self) -> None:
        """🔴 `None` ≠ `0`: «не сообщили» не значит «торгов не было».

        Часть инструментов приходит вообще без поля объёма, и трактовка
        `None` как нуля выбросила бы всю их историю целиком.
        """
        candles, rejected = to_candles(
            "AAPL.US", [candle(TS_AUG5_2000_UTC, 100.0, volume=None)],
            max_date=date(2026, 8, 6))
        self.assertEqual(len(candles), 1)
        self.assertIsNone(candles[0].volume)
        self.assertEqual(rejected, 0)

    def test_rows_without_close_are_dropped(self) -> None:
        candles, rejected = to_candles(
            "AAPL.US", [candle(TS_AUG5_2000_UTC, close=None)],
            max_date=date(2026, 8, 6))
        self.assertEqual(candles, [])
        self.assertEqual(rejected, 1)

    def test_non_positive_close_is_dropped(self) -> None:
        candles, _ = to_candles("AAPL.US", [candle(TS_AUG5_2000_UTC, 0.0)],
                                max_date=date(2026, 8, 6))
        self.assertEqual(candles, [])

    def test_dates_after_cutoff_are_dropped(self) -> None:
        """Незакрытый день не пишется: внутри сессии `close` — последняя сделка."""
        candles, rejected = to_candles(
            "AAPL.US", [candle(TS_AUG5_2000_UTC, 100.0)],
            max_date=date(2026, 8, 4))
        self.assertEqual(candles, [])
        self.assertEqual(rejected, 1)

    def test_dates_before_window_are_dropped(self) -> None:
        candles, _ = to_candles("AAPL.US", [candle(TS_AUG5_2000_UTC, 100.0)],
                                max_date=date(2026, 8, 6), min_date=date(2026, 8, 6))
        self.assertEqual(candles, [])

    def test_duplicate_dates_keep_the_last_version(self) -> None:
        """В перекрытии ресинка вторая версия свежее — брокер уточняет её после клиринга."""
        candles, _ = to_candles(
            "AAPL.US",
            [candle(TS_AUG5_2000_UTC, 100.0), candle(TS_AUG6_0100_UTC, 101.0)],
            max_date=date(2026, 8, 6))
        self.assertEqual(len(candles), 1, "обе свечи — один торговый день в Нью-Йорке")
        self.assertEqual(candles[0].close, Decimal("101.0000"))

    def test_output_is_sorted_by_date(self) -> None:
        candles, _ = to_candles(
            "AAPL.US",
            [candle(TS_AUG5_2000_UTC + 86400, 101.0), candle(TS_AUG5_2000_UTC, 100.0)],
            max_date=date(2026, 8, 10))
        self.assertEqual([c.trade_date for c in candles],
                         sorted(c.trade_date for c in candles))

    def test_price_outside_numeric_precision_is_rejected_not_fatal(self) -> None:
        """Мусор от брокера не должен ронять транзакцию всего батча."""
        candles, rejected = to_candles(
            "AAPL.US", [candle(TS_AUG5_2000_UTC, 1e12)], max_date=date(2026, 8, 6))
        self.assertEqual(candles, [])
        self.assertEqual(rejected, 1)

    def test_ohlc_fields_are_carried_through(self) -> None:
        candles, _ = to_candles(
            "AAPL.US",
            [candle(TS_AUG5_2000_UTC, 194.0, open_=191.0, high=195.0, low=190.0)],
            max_date=date(2026, 8, 6))
        bar = candles[0]
        self.assertEqual(
            (bar.open, bar.high, bar.low, bar.close),
            (Decimal("191.0000"), Decimal("195.0000"),
             Decimal("190.0000"), Decimal("194.0000")),
        )


# ── Планирование окон ────────────────────────────────────────────────────────


class PlanBatchesTest(unittest.TestCase):

    def setUp(self) -> None:
        self.today = date(2026, 8, 9)

    def processor(self, **cfg) -> EtlProcessor:
        return EtlProcessor(FakeDb([]), FakeClient({}), settings(**cfg))

    def test_new_ticker_gets_the_full_lookback(self) -> None:
        plan = self.processor().plan_batches(["AAPL.US"], {}, today=self.today)
        self.assertEqual(len(plan), 1)
        start, tickers = plan[0]
        self.assertEqual(tickers, ["AAPL.US"])
        self.assertEqual((self.today - start).days, 1825)

    def test_known_ticker_resumes_with_overlap(self) -> None:
        """Перекрытие назад — чтобы забрать уточнённые брокером свечи."""
        plan = self.processor().plan_batches(
            ["AAPL.US"], {"AAPL.US": date(2026, 8, 6)}, today=self.today)
        start, _ = plan[0]
        self.assertEqual(start, date(2026, 8, 4))   # 6 + 1 − 3

    def test_uptodate_ticker_is_skipped_when_overlap_is_off(self) -> None:
        """Без перекрытия курсор на T-1 означает «делать нечего» — запрос не шлём."""
        plan = self.processor(resync_overlap_days=0).plan_batches(
            ["AAPL.US"], {"AAPL.US": date(2026, 8, 8)}, today=self.today)
        self.assertEqual(plan, [], "пустой интервал запрашивать незачем")

    def test_uptodate_ticker_is_still_refreshed_inside_the_overlap(self) -> None:
        """С перекрытием тикер на T-1 ПЕРЕзапрашивается — в этом и смысл перекрытия.

        Брокер уточняет последние свечи после клиринга (объём, иногда
        закрытие). Пропустить их значит навсегда сохранить черновую версию
        дня: курсор уже прошёл эту дату и назад не вернётся.
        Цена вопроса нулевая — те же тикеры, то же окно, тот же батч.
        """
        plan = self.processor(resync_overlap_days=3).plan_batches(
            ["AAPL.US"], {"AAPL.US": date(2026, 8, 8)}, today=self.today)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0][0], date(2026, 8, 6), "окно = последние 3 дня")

    def test_tickers_sharing_a_cursor_land_in_one_bucket(self) -> None:
        """Установившийся режим: один бакет — значит обычные батчи по 50."""
        state = {t: date(2026, 8, 6) for t in ("AAPL.US", "MSFT.US", "TSLA.US")}
        plan = self.processor().plan_batches(list(state), state, today=self.today)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0][1], ["AAPL.US", "MSFT.US", "TSLA.US"])

    def test_new_ticker_does_not_drag_the_batch_into_five_years(self) -> None:
        """🔴 Смысл группировки: свежая бумага не заставляет перекачивать остальные."""
        state = {"AAPL.US": date(2026, 8, 6), "MSFT.US": date(2026, 8, 6)}
        plan = self.processor().plan_batches(
            ["AAPL.US", "MSFT.US", "NEW.US"], state, today=self.today)
        self.assertEqual(len(plan), 2, "должно быть два окна, а не одно общее")
        by_start = dict(plan)
        self.assertEqual(by_start[date(2026, 8, 4)], ["AAPL.US", "MSFT.US"])
        self.assertEqual(by_start[self.today - __import__("datetime").timedelta(days=1825)],
                         ["NEW.US"])

    def test_buckets_are_sorted_by_start_date(self) -> None:
        state = {"AAPL.US": date(2026, 8, 6)}
        plan = self.processor().plan_batches(["AAPL.US", "NEW.US"], state, today=self.today)
        self.assertEqual([s for s, _ in plan], sorted(s for s, _ in plan))

    def test_ticker_order_inside_bucket_is_deterministic(self) -> None:
        """Иначе «упал батч №7» неразбираемо: батч №7 каждый раз разный."""
        state = {t: date(2026, 8, 6) for t in ("ZZZ.US", "AAA.US", "MMM.US")}
        plan = self.processor().plan_batches(
            ["ZZZ.US", "AAA.US", "MMM.US"], state, today=self.today)
        self.assertEqual(plan[0][1], ["AAA.US", "MMM.US", "ZZZ.US"])

    def test_include_current_day_moves_the_cutoff(self) -> None:
        plan = self.processor(include_current_day=True).plan_batches(
            ["AAPL.US"], {"AAPL.US": date(2026, 8, 8)}, today=self.today)
        self.assertEqual(len(plan), 1, "с include_current_day сегодняшний день запрашивается")


# ── Полный прогон ────────────────────────────────────────────────────────────


class RunDailySyncTest(unittest.IsolatedAsyncioTestCase):

    def setUp(self) -> None:
        self.today = date(2026, 8, 7)   # cutoff = 2026-08-06

    async def test_candles_are_written_before_the_cursor_moves(self) -> None:
        """🔴 Обратный порядок теряет данные молча и навсегда.

        Курсор только растёт: объявив день загруженным до записи, за ним
        уже никто не вернётся, а дыра всплывёт месяцы спустя.
        """
        db = FakeDb(["AAPL.US"], {"AAPL.US": date(2026, 8, 4)})
        client = FakeClient({"AAPL.US": [candle(TS_AUG5_2000_UTC, 194.0)]})
        report = await EtlProcessor(db, client, settings()).run_daily_sync(today=self.today)

        self.assertEqual(db.calls, ["candles", "sync_status"])
        self.assertEqual(report.candles_written, 1)
        self.assertTrue(report.ok)

    async def test_cursor_advances_to_the_last_written_date(self) -> None:
        db = FakeDb(["AAPL.US"], {"AAPL.US": date(2026, 8, 4)})
        client = FakeClient({"AAPL.US": [candle(TS_AUG5_2000_UTC, 194.0)]})
        await EtlProcessor(db, client, settings()).run_daily_sync(today=self.today)
        self.assertEqual(db.cursors, [("AAPL.US", date(2026, 8, 5))])

    async def test_rerun_writes_the_same_rows_not_duplicates(self) -> None:
        """Идемпотентность: повтор за тот же день даёт тот же ключ, а не вторую строку."""
        state = {"AAPL.US": date(2026, 8, 4)}
        bars = {"AAPL.US": [candle(TS_AUG5_2000_UTC, 194.0)]}

        first_db = FakeDb(["AAPL.US"], dict(state))
        await EtlProcessor(first_db, FakeClient(bars), settings()).run_daily_sync(today=self.today)

        second_db = FakeDb(["AAPL.US"], dict(state))
        await EtlProcessor(second_db, FakeClient(bars), settings()).run_daily_sync(today=self.today)

        first_keys = [(c.ticker, c.trade_date) for c in first_db.written]
        second_keys = [(c.ticker, c.trade_date) for c in second_db.written]
        self.assertEqual(first_keys, second_keys)
        self.assertEqual(len(set(second_keys)), len(second_keys), "ключи внутри прогона уникальны")

    async def test_ticker_without_data_does_not_move_its_cursor(self) -> None:
        """Иначе делистинг/опечатка навсегда «загрузили» бы пустой интервал."""
        db = FakeDb(["AAPL.US", "GONE.US"], {"AAPL.US": date(2026, 8, 4),
                                             "GONE.US": date(2026, 8, 4)})
        client = FakeClient({"AAPL.US": [candle(TS_AUG5_2000_UTC, 194.0)]})
        report = await EtlProcessor(db, client, settings()).run_daily_sync(today=self.today)

        self.assertEqual([t for t, _ in db.cursors], ["AAPL.US"])
        self.assertIn("GONE.US", report.tickers_no_data)

    async def test_failed_batch_does_not_abort_the_run(self) -> None:
        """Отказ по 50 тикерам не должен уносить остальные 450."""
        db = FakeDb(["A.US", "B.US"], {"A.US": date(2026, 8, 4), "B.US": date(2026, 8, 4)})
        client = FakeClient([TradernetError("429 forever")], batch_size=1)
        client._responses.append({"B.US": [candle(TS_AUG5_2000_UTC, 50.0)]})

        report = await EtlProcessor(db, client, settings()).run_daily_sync(today=self.today)

        self.assertFalse(report.ok)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.candles_written, 1, "второй батч всё равно записан")

    async def test_failed_batch_leaves_its_cursor_untouched(self) -> None:
        db = FakeDb(["A.US"], {"A.US": date(2026, 8, 4)})
        client = FakeClient([TradernetError("boom")], batch_size=1)
        await EtlProcessor(db, client, settings()).run_daily_sync(today=self.today)
        self.assertEqual(db.cursors, [], "курсор упавшего батча не двигается")

    async def test_empty_universe_is_reported_not_crashed(self) -> None:
        report = await EtlProcessor(FakeDb([]), FakeClient({}), settings()).run_daily_sync(
            today=self.today)
        self.assertEqual(report.tickers_total, 0)
        self.assertTrue(report.ok)

    async def test_batches_respect_the_client_batch_size(self) -> None:
        tickers = [f"T{i:02d}.US" for i in range(5)]
        db = FakeDb(tickers, {t: date(2026, 8, 4) for t in tickers})
        client = FakeClient({t: [candle(TS_AUG5_2000_UTC, 10.0)] for t in tickers},
                            batch_size=2)
        report = await EtlProcessor(db, client, settings()).run_daily_sync(today=self.today)

        self.assertEqual([len(r) for r in client.requests], [2, 2, 1])
        self.assertEqual(report.batches, 3)

    async def test_future_candles_are_filtered_before_load(self) -> None:
        """Свеча за незакрытый день не должна доехать до БД."""
        db = FakeDb(["AAPL.US"], {"AAPL.US": date(2026, 8, 4)})
        # 2026-08-07 20:00 UTC — это «сегодня», позже cutoff = 08-06.
        client = FakeClient({"AAPL.US": [candle(TS_AUG5_2000_UTC + 2 * 86400, 195.0)]})
        report = await EtlProcessor(db, client, settings()).run_daily_sync(today=self.today)

        self.assertEqual(db.written, [])
        self.assertEqual(report.rows_rejected, 1)


if __name__ == "__main__":
    unittest.main()
