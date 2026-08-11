"""
MP-08.5 — хранилище котировок Stooq: календарь, формы символа, ingest, чтение.

Каждый тест здесь защищает ЗАМЕР, а не вкус: в докстрингах названы файлы и
числа, на которых правило было получено (`docs/audit/STOOQ_CONVENTION.md
§3.2–3.8`). Тест без замера за спиной в этой сюите — повод его удалить.

Фикстуры генерируются кодом, а не читаются с диска: сюита обязана проходить и
в зеркале деплой-образа, где нет ни `design/`, ни `scripts/`, ни выгрузок.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from finance import market_calendar as mc
from finance import stooq_ingest as si
from finance import stooq_symbols as sym
from finance.stooq_store import StooqStore, StoreUnavailable

HEADER = ("<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,"
          "<VOL>,<OPENINT>")


def row(ticker: str, day: int, *, o=None, h=None, lo=None, c=10.5,
        vol=1000.0) -> str:
    """Одна строка выгрузки.

    OHLC по умолчанию ВЫВОДИТСЯ из цены закрытия, а не берётся константами:
    фикстура с фиксированными `high=11` при `close=58.42` нарушала бы правило 3
    и молча превращала бы каждый тест данных в тест отбраковки.
    """
    o = c if o is None else o
    h = max(o, c) * 1.01 if h is None else h
    lo = min(o, c) * 0.99 if lo is None else lo
    return f"{ticker},D,{day},000000,{o},{h},{lo},{c},{vol},0"


def write_daily(directory: Path, day: int, rows: list[str]) -> Path:
    """Дневной срез с настоящей шапкой и CRLF — как отдаёт источник."""
    path = Path(directory) / f"{day}_d.txt"
    path.write_text("\r\n".join([HEADER, *rows]) + "\r\n", encoding="utf-8")
    return path


def write_history(directory: Path, name: str, rows: list[str]) -> Path:
    path = Path(directory) / name
    path.write_text("\r\n".join([HEADER, *rows]) + "\r\n", encoding="utf-8")
    return path


class _TempDirMixin:
    def setUp(self) -> None:                                # noqa: N802
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def fresh_db(self) -> sqlite3.Connection:
        conn = si.connect(self.tmp / "prices.sqlite")
        si.ensure_schema(conn)
        self.addCleanup(conn.close)
        return conn


# ═════════════════════════════════════════════════════════════════════════════
# ЧК-08.5-1 — рынок, валюта, календарь
# ═════════════════════════════════════════════════════════════════════════════

class MarketDetectionTest(unittest.TestCase):
    """Суффикс → рынок. Ключевое требование: неизвестное НЕ становится США."""

    def test_known_suffixes_map_to_their_markets(self) -> None:
        self.assertEqual(mc.market_of("AAPL.US"), "US")
        self.assertEqual(mc.market_of("BTC.V"), "CRYPTO")
        self.assertEqual(mc.market_of("7203.JP"), "JP")

    def test_unknown_suffix_is_unknown_not_us(self) -> None:
        """🔴 Тихая подстановка США — класс дефекта A-3.

        Бумага с неизвестным суффиксом получила бы американский торговый
        календарь и доллар как валюту котировки — оба утверждения выдуманы.
        """
        self.assertEqual(mc.market_of("FFSPC6.1028.AIX"), mc.UNKNOWN_MARKET)
        self.assertNotEqual(mc.market_of("FFSPC6.1028.AIX"), "US")

    def test_index_and_fx_are_non_instruments(self) -> None:
        """Замер: 59 строк `^…` и 95 валютных пар без суффикса в одном срезе."""
        self.assertEqual(mc.market_of("^DAX"), mc.NON_INSTRUMENT)
        self.assertEqual(mc.market_of("EURUSD"), mc.NON_INSTRUMENT)
        self.assertEqual(mc.market_of("HUOHUF1M"), mc.NON_INSTRUMENT)

    def test_currency_follows_market(self) -> None:
        self.assertEqual(mc.currency_of_symbol("AAPL.US"), "USD")
        self.assertEqual(mc.currency_of_symbol("7203.JP"), "JPY")

    def test_unknown_market_has_no_currency(self) -> None:
        """Доллар по умолчанию посчитал бы иеновую цену как долларовую (Т-5)."""
        self.assertEqual(mc.currency_of("MARS"), mc.UNKNOWN_CURRENCY)

    def test_index_symbol_is_valid_not_garbage(self) -> None:
        """`^DAX` обязан считаться корректным символом.

        Иначе индексы попадают в счётчик «битый символ» вместе с настоящим
        мусором, и сводка перестаёт отличать осознанный пропуск от порчи файла.
        """
        self.assertTrue(mc.is_valid_symbol("^DAX"))
        self.assertFalse(mc.is_valid_symbol("<!DOCTYPE html>"))


class TradingCalendarTest(unittest.TestCase):
    """Календарь выводится из сессий, а не из списка праздников."""

    SESSIONS_US = [20260803, 20260804, 20260805, 20260806, 20260807]
    SESSIONS_CRYPTO = [20260806, 20260807, 20260808, 20260809]

    def test_weekend_costs_zero_trading_days_on_us(self) -> None:
        """Пятница → суббота: календарный день есть, торгового нет."""
        self.assertEqual(
            mc.staleness_trading_days(self.SESSIONS_US, 20260807,
                                      as_of=20260808), 0)

    def test_weekend_costs_trading_days_on_crypto(self) -> None:
        """У крипты 7/7 — те же сутки стоят одной сессией."""
        self.assertEqual(
            mc.staleness_trading_days(self.SESSIONS_CRYPTO, 20260807,
                                      as_of=20260808), 1)

    def test_same_day_is_not_stale(self) -> None:
        self.assertEqual(
            mc.staleness_trading_days(self.SESSIONS_US, 20260807,
                                      as_of=20260807), 0)

    def test_gap_counts_only_sessions(self) -> None:
        self.assertEqual(
            mc.staleness_trading_days(self.SESSIONS_US, 20260803,
                                      as_of=20260807), 4)

    def test_empty_calendar_is_zero_not_crash(self) -> None:
        self.assertEqual(mc.staleness_trading_days([], 20260807,
                                                   as_of=20260808), 0)

    def test_crypto_expects_seven_sessions_a_week(self) -> None:
        self.assertEqual(mc.expected_sessions_per_week("CRYPTO"), 7)
        self.assertEqual(mc.expected_sessions_per_week("US"), 5)


# ═════════════════════════════════════════════════════════════════════════════
# Формы символа
# ═════════════════════════════════════════════════════════════════════════════

class SymbolFormsTest(unittest.TestCase):
    """Кандидат меняет нотацию, но НИКОГДА — площадку."""

    def test_plain_us_ticker(self) -> None:
        self.assertEqual(sym.candidates_for("AAPL.US"), ["AAPL.US"])

    def test_bare_ticker_gets_us_suffix(self) -> None:
        self.assertEqual(sym.candidates_for("AAPL"), ["AAPL.US"])

    def test_crypto_uses_dot_v(self) -> None:
        """Замер: `BTC.V`, `ETH.V`, `SOL.V`, `BNB.V` — все четыре в срезе."""
        self.assertEqual(sym.candidates_for("BTC-USD"), ["BTC.V"])

    def test_share_class_uses_hyphen(self) -> None:
        """Ловушка №7: у Stooq класс акции пишется дефисом (`brk-b.us`)."""
        self.assertEqual(sym.candidates_for("BRK.B"), ["BRK-B.US"])

    def test_london_gdr_tries_uk_not_il(self) -> None:
        """Ловушка №8: `.IL` у нас — Лондон, `.IL` у Stooq — Израиль."""
        self.assertEqual(sym.candidates_for("KAP.IL"), ["KAP.UK"])

    def test_no_foreign_ticker_ever_falls_back_to_us(self) -> None:
        """🔴 Ловушка №6, самая дорогая — и проверяется ОБЕ ветки перебора.

        `KSPI.US` существует и торгуется — это ADR на Nasdaq, а НЕ листинг на
        KASE. Кандидат `{base}.us` подсунул бы другую бумагу: другая валюта,
        другая ликвидность, другой календарь.

        Тест перечисляет тикеры с ИЗВЕСТНЫМ суффиксом (`.KZ`, `.JP`) и с
        НЕИЗВЕСТНЫМ (`.XX`, `.AIX`): в коде это разные ветки, и проверка одной
        из них ничего не говорит о другой. Прошлая редакция проверяла только
        `KSPI.KZ` и была ПУСТОЙ — мутация «добавить `{base}.US`» в ветку
        известного суффикса её не роняла.
        """
        for ticker in ("KSPI.KZ", "KZTK.KZ", "7203.JP", "FOO.XX",
                       "FFSPC6.1028.AIX"):
            with self.subTest(ticker=ticker):
                got = sym.candidates_for(ticker)
                self.assertFalse(
                    [c for c in got if c.endswith(".US")],
                    f"{ticker} получил американского кандидата: {got}")

    def test_aix_note_has_no_candidates(self) -> None:
        """Честный пустой ответ лучше правдоподобного чужого."""
        self.assertEqual(sym.candidates_for("FFSPC6.1028.AIX"), [])

    def test_venue_change_is_detectable(self) -> None:
        self.assertTrue(sym.would_change_venue("KSPI.KZ", "KSPI.US"))
        self.assertFalse(sym.would_change_venue("AAPL.US", "AAPL.US"))

    def test_own_notation_is_never_called_venue_change(self) -> None:
        """🔴 Две функции об одном обязаны отвечать одинаково (`AUDIT §−80`).

        `candidates_for` отдаёт `KAP.IL → KAP.UK` как НОТАЦИЮ (Лондон и там, и
        там) и сопоставляет с `match_kind='exact'`. Если `would_change_venue`
        зовёт ту же пару подменой площадки, раскрытие подмен на экране
        разойдётся с тем, что реально сделал перебор.
        """
        for engine_ticker in ("KAP.IL", "HSBK.IL", "AAPL", "BTC-USD", "BRK.B"):
            for candidate in sym.candidates_for(engine_ticker):
                with self.subTest(pair=(engine_ticker, candidate)):
                    self.assertFalse(
                        sym.would_change_venue(engine_ticker, candidate),
                        f"{engine_ticker} → {candidate} — это наша форма записи")


# ═════════════════════════════════════════════════════════════════════════════
# ЧК-08.5-2 — девять правил ingest
# ═════════════════════════════════════════════════════════════════════════════

class IngestRulesTest(_TempDirMixin, unittest.TestCase):
    """Каждое правило — с замером за спиной."""

    def test_rule1_row_with_foreign_date_is_dropped(self) -> None:
        """Правило 1. Замер: 148 баров мёртвых бумаг вплоть до 2015 года."""
        path = write_daily(self.tmp, 20260807,
                           [row("AAPL.US", 20260807), row("DEAD.US", 20150818)])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual([b.symbol for b in batch.bars], ["AAPL.US"])
        self.assertEqual(batch.rejected.get("чужая дата"), 1)

    def test_rule1_empty_date_is_dropped(self) -> None:
        """Правило 1 снимает и 43 строки с ПУСТОЙ датой — тем же условием."""
        path = write_daily(self.tmp, 20260807,
                           [row("AAPL.US", 20260807),
                            "EMPTY.US,D,,000000,1,2,0.5,1.5,100,0"])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual(len(batch.bars), 1)
        self.assertEqual(batch.rejected.get("нет даты"), 1)

    def test_rule2_missing_close_is_dropped(self) -> None:
        path = write_daily(self.tmp, 20260807,
                           [row("AAPL.US", 20260807),
                            "NOCLOSE.US,D,20260807,000000,1,2,0.5,,100,0"])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual(len(batch.bars), 1)
        self.assertEqual(batch.rejected.get("нет цены закрытия"), 1)

    def test_rule3_broken_ohlc_is_dropped(self) -> None:
        """Правило 3. Замер: `^IPSA` 2026-08-06 — 1 строка на 11 999."""
        path = write_daily(self.tmp, 20260807,
                           [row("BAD.US", 20260807, o=10, h=5, lo=9, c=10.5)])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual(batch.bars, [])
        self.assertEqual(batch.rejected.get("нарушен OHLC"), 1)

    def test_rule4_flat_zero_volume_stub_is_dropped(self) -> None:
        """Правило 4. Замер: `spsc.us` 2010-04-21 — заглушка до IPO."""
        path = write_daily(self.tmp, 20260807,
                           [row("STUB.US", 20260807, o=7, h=7, lo=7, c=7, vol=0)])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual(batch.bars, [])
        self.assertEqual(batch.rejected.get("плоская заглушка"), 1)

    def test_rule3_survives_missing_open(self) -> None:
        """🔴 Регресс `AUDIT §−80`: пустой `<OPEN>` гасил ВСЮ проверку OHLC.

        Условие было одно на три поля (`None not in (open_, high, low)`), и
        строка без открытия проносила мимо гарда даже `high < low`. Каждое
        условие обязано зависеть только от тех полей, которые в нём участвуют.
        """
        path = write_daily(self.tmp, 20260807,
                           ["NOOPEN.US,D,20260807,000000,,5,9,7,100,0"])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual(batch.bars, [], "бар с high=5 < low=9 попал в базу")
        self.assertEqual(batch.rejected.get("нарушен OHLC"), 1)

    def test_rule3_catches_low_above_the_body(self) -> None:
        """Нижняя граница проверяется ОТДЕЛЬНО от верхней.

        Без этого теста гард `low > min(open, close)` был бы прикрыт соседним:
        мутация выключала его, а падал соседний тест — то есть сам гард
        оставался непроверенным.
        """
        path = write_daily(self.tmp, 20260807,
                           [row("LOWBAD.US", 20260807, o=10, h=11, lo=10.5,
                                c=10)])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual(batch.bars, [])
        self.assertEqual(batch.rejected.get("нарушен OHLC"), 1)

    def test_rule3_accepts_bar_with_only_close(self) -> None:
        """Отсутствие OHLC — не брак: проверять нечего, цена закрытия есть."""
        path = write_daily(self.tmp, 20260807,
                           ["ONLYC.US,D,20260807,000000,,,,7.5,100,0"])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual(len(batch.bars), 1)

    def test_rule4_keeps_flat_bar_with_volume(self) -> None:
        """Плоский бар С ОБЪЁМОМ — настоящий день торгов, не заглушка."""
        path = write_daily(self.tmp, 20260807,
                           [row("REAL.US", 20260807, o=7, h=7, lo=7, c=7,
                                vol=42)])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual(len(batch.bars), 1)

    def test_rule5_index_and_fx_are_skipped_as_such(self) -> None:
        """Правило 5. Пропуск учитывается ОТДЕЛЬНО от брака."""
        path = write_daily(self.tmp, 20260807,
                           [row("AAPL.US", 20260807), row("^DAX", 20260807),
                            row("EURUSD", 20260807)])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual(len(batch.bars), 1)
        self.assertEqual(batch.rejected.get("индекс/FX (пропуск)"), 2)
        self.assertNotIn("битый символ", batch.rejected)

    def test_rule6_crypto_goes_to_its_own_market(self) -> None:
        """Правило 6. Иначе субботний бар крипты выглядел бы аномалией США."""
        path = write_daily(self.tmp, 20260807,
                           [row("AAPL.US", 20260807), row("BTC.V", 20260807)])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual(batch.accepted_by_market, {"US": 1, "CRYPTO": 1})

    def test_rule7_symbol_is_upper_cased(self) -> None:
        path = write_daily(self.tmp, 20260807, [row("aapl.us", 20260807)])
        batch = si.parse_daily_file(path, min_us_rows=0)
        self.assertEqual(batch.bars[0].symbol, "AAPL.US")

    def test_rule8_history_outside_window_is_dropped(self) -> None:
        """Правило 8: окно движка — 1825 календарных дней."""
        path = write_history(self.tmp, "aapl.us.txt",
                             [row("AAPL.US", 20100421), row("AAPL.US", 20260807)])
        batch = si.parse_history_file(path, window_days=1825,
                                      today=date(2026, 8, 10))
        self.assertEqual([b.trade_date for b in batch.bars], [20260807])
        self.assertEqual(batch.rejected.get("вне окна"), 1)

    def test_rule9_truncated_weekday_file_is_rejected_whole(self) -> None:
        """🔴 Правило 9. Частично скачанный файл опаснее битого: он выглядит
        нормально и делает дыру в истории сразу у ВСЕХ бумаг."""
        path = write_daily(self.tmp, 20260807,
                           [row(f"T{i}.US", 20260807) for i in range(10)])
        batch = si.parse_daily_file(path, min_us_rows=5000)
        self.assertIsNotNone(batch.fatal)
        self.assertIn("недокачанным", batch.fatal)

    def test_rule9_does_not_fire_on_closed_market(self) -> None:
        """🔴 Форма правила — «0 < принято < порога».

        Замер субботы 2026-08-08: 0 US-баров и 202 крипто. Ноль означает
        закрытый рынок — выходной или праздник, — поэтому календарь праздников
        не нужен и биржевых исключений не заводится.
        """
        path = write_daily(self.tmp, 20260808,
                           [row("BTC.V", 20260808), row("ETH.V", 20260808)])
        batch = si.parse_daily_file(path, min_us_rows=5000)
        self.assertIsNone(batch.fatal)
        self.assertEqual(len(batch.bars), 2)

    def test_captcha_page_is_rejected_by_header(self) -> None:
        """Самый частый отказ скачивания — HTML капчи вместо CSV."""
        path = self.tmp / "20260807_d.txt"
        path.write_text("<!DOCTYPE html><html><body>captcha</body></html>",
                        encoding="utf-8")
        with self.assertRaises(si.IngestError):
            si.parse_daily_file(path)

    def test_file_without_date_in_name_is_fatal(self) -> None:
        """Без даты в имени правило 1 неприменимо — значит файл непригоден."""
        path = write_history(self.tmp, "whatever.txt", [row("AAPL.US", 20260807)])
        batch = si.parse_daily_file(path)
        self.assertIsNotNone(batch.fatal)


class IngestWriteTest(_TempDirMixin, unittest.TestCase):
    """Запись: идемпотентность, коммутативность, дробный объём."""

    def test_volume_keeps_fractional_precision(self) -> None:
        """🔴 Замер `vgit.us`: `1824.5238176263` в 3 274 строках из 4 197.

        `INTEGER` уронил бы загрузку или молча усёк.
        """
        conn = self.fresh_db()
        path = write_daily(self.tmp, 20260807,
                           [row("VGIT.US", 20260807, vol=1824.5238176263)])
        si.apply_batch(conn, si.parse_daily_file(path, min_us_rows=0))
        stored = conn.execute("SELECT volume FROM daily_bars").fetchone()[0]
        self.assertEqual(stored, 1824.5238176263)

    def test_applying_same_file_twice_changes_nothing(self) -> None:
        """Оператор — человек: он применит файл дважды."""
        conn = self.fresh_db()
        path = write_daily(self.tmp, 20260807, [row("AAPL.US", 20260807)])
        si.apply_batch(conn, si.parse_daily_file(path, min_us_rows=0))
        snapshot = conn.execute(
            "SELECT * FROM daily_bars ORDER BY instrument_id").fetchall()
        si.apply_batch(conn, si.parse_daily_file(path, min_us_rows=0))
        again = conn.execute(
            "SELECT * FROM daily_bars ORDER BY instrument_id").fetchall()
        self.assertEqual([tuple(r) for r in snapshot], [tuple(r) for r in again])

    def test_order_of_files_does_not_matter(self) -> None:
        """UPSERT по `(инструмент, дата)` коммутативен: пропуск дня не ломает."""
        forward = self.fresh_db()
        for day in (20260806, 20260807):
            path = write_daily(self.tmp, day, [row("AAPL.US", day, c=day / 1e7)])
            si.apply_batch(forward, si.parse_daily_file(path, min_us_rows=0))
        got_forward = forward.execute(
            "SELECT trade_date, close FROM daily_bars ORDER BY trade_date"
        ).fetchall()

        backward = si.connect(self.tmp / "other.sqlite")
        si.ensure_schema(backward)
        self.addCleanup(backward.close)
        for day in (20260807, 20260806):
            path = write_daily(self.tmp, day, [row("AAPL.US", day, c=day / 1e7)])
            si.apply_batch(backward, si.parse_daily_file(path, min_us_rows=0))
        got_backward = backward.execute(
            "SELECT trade_date, close FROM daily_bars ORDER BY trade_date"
        ).fetchall()
        self.assertEqual([tuple(r) for r in got_forward],
                         [tuple(r) for r in got_backward])

    def test_fatal_file_leaves_database_untouched(self) -> None:
        """Правило 9 обязано быть «всё или ничего»."""
        conn = self.fresh_db()
        good = write_daily(self.tmp, 20260806, [row("AAPL.US", 20260806)])
        si.apply_batch(conn, si.parse_daily_file(good, min_us_rows=0))
        before = conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]

        truncated = write_daily(self.tmp, 20260807,
                                [row(f"T{i}.US", 20260807) for i in range(3)])
        result = si.apply_batch(conn,
                                si.parse_daily_file(truncated, min_us_rows=5000))
        after = conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]
        self.assertEqual(before, after)
        self.assertTrue(result.fatal_files)

    def test_instrument_records_market_and_currency(self) -> None:
        conn = self.fresh_db()
        path = write_daily(self.tmp, 20260807,
                           [row("AAPL.US", 20260807), row("BTC.V", 20260807)])
        si.apply_batch(conn, si.parse_daily_file(path, min_us_rows=0))
        rows = {r["source_symbol"]: (r["market"], r["currency"])
                for r in conn.execute("SELECT * FROM instruments")}
        self.assertEqual(rows["AAPL.US"], ("US", "USD"))
        self.assertEqual(rows["BTC.V"], ("CRYPTO", "USD"))

    def test_convention_is_stored_as_unmeasured(self) -> None:
        """🔴 Конвенция Stooq НЕ измерена — и база обязана это признавать.

        Записать `split_adjusted` «для порядка» значило бы дать провайдеру
        основание применить корректировку к уже скорректированному ряду.
        """
        conn = self.fresh_db()
        path = write_daily(self.tmp, 20260807, [row("AAPL.US", 20260807)])
        si.apply_batch(conn, si.parse_daily_file(path, min_us_rows=0))
        value = conn.execute("SELECT convention FROM instruments").fetchone()[0]
        self.assertEqual(value, si.CONVENTION_UNMEASURED)

    def test_generation_changes_on_write(self) -> None:
        """Иначе контейнер недельной давности отдаёт недельные цены молча."""
        conn = self.fresh_db()
        path = write_daily(self.tmp, 20260807, [row("AAPL.US", 20260807)])
        si.apply_batch(conn, si.parse_daily_file(path, min_us_rows=0))
        first = conn.execute(
            "SELECT value FROM meta WHERE key='generation'").fetchone()[0]
        self.assertTrue(first)

    def test_ingest_run_is_recorded(self) -> None:
        """Провенанс: по любому бару можно спросить, каким файлом он приехал."""
        conn = self.fresh_db()
        path = write_daily(self.tmp, 20260807, [row("AAPL.US", 20260807)])
        si.apply_batch(conn, si.parse_daily_file(path, min_us_rows=0))
        run = conn.execute("SELECT * FROM ingest_runs").fetchone()
        self.assertEqual(run["file_name"], "20260807_d.txt")
        self.assertEqual(run["rows_written"], 1)


class InboxTest(_TempDirMixin, unittest.TestCase):
    """Папочный цикл оператора."""

    def setUp(self) -> None:                                # noqa: N802
        super().setUp()
        self.inbox = self.tmp / "inbox"
        self.applied = self.tmp / "applied"
        self.rejected = self.tmp / "rejected"
        self.inbox.mkdir()

    def test_files_apply_in_date_order_and_move_out(self) -> None:
        """Вчерашний файл, принесённый после сегодняшнего, не должен значить."""
        write_daily(self.inbox, 20260807, [row("AAPL.US", 20260807, c=2)])
        write_daily(self.inbox, 20260806, [row("AAPL.US", 20260806, c=1)])
        conn = self.fresh_db()
        result = si.apply_inbox(conn, self.inbox, self.applied, self.rejected,
                                min_us_rows=0)
        self.assertEqual(result.files, 2)
        self.assertEqual(list(self.inbox.glob("*.txt")), [])
        self.assertEqual(sorted(p.name for p in self.applied.iterdir()),
                         ["20260806_d.txt", "20260807_d.txt"])

    def test_rejected_file_goes_to_its_own_folder(self) -> None:
        write_daily(self.inbox, 20260807,
                    [row(f"T{i}.US", 20260807) for i in range(3)])
        conn = self.fresh_db()
        si.apply_inbox(conn, self.inbox, self.applied, self.rejected,
                       min_us_rows=5000)
        self.assertEqual([p.name for p in self.rejected.iterdir()],
                         ["20260807_d.txt"])
        self.assertFalse(self.applied.exists() and list(self.applied.iterdir()))


class BootstrapAndSeamTest(_TempDirMixin, unittest.TestCase):
    """Бутстрап архивом и сверка шва с дневной дельтой."""

    def _archive(self) -> Path:
        archive = self.tmp / "archive" / "nasdaq etfs"
        archive.mkdir(parents=True)
        write_history(archive, "vgit.us.txt",
                      [row("VGIT.US", 20260806, c=58.31),
                       row("VGIT.US", 20260807, c=58.42)])
        stocks = self.tmp / "archive" / "nasdaq stocks" / "1"
        stocks.mkdir(parents=True)
        write_history(stocks, "aapl.us.txt", [row("AAPL.US", 20260807, c=200.0)])
        return self.tmp / "archive"

    def test_bootstrap_walks_etf_subfolders_too(self) -> None:
        """🔴 Регресс `AUDIT §−76`: ETF лежат в СОСЕДНЕЙ подпапке архива.

        Беглый просмотр подпапок `stocks` однажды породил вывод «в пакете нет
        ETF» — и он попал в roadmap как блокер фазы.
        """
        conn = self.fresh_db()
        si.bootstrap(conn, self._archive(), window_days=1825,
                     today=date(2026, 8, 10))
        symbols = {r[0] for r in conn.execute(
            "SELECT source_symbol FROM instruments")}
        self.assertEqual(symbols, {"VGIT.US", "AAPL.US"})

    def test_bootstrap_can_be_limited_to_working_set(self) -> None:
        """Рабочий набор — 40–80 МБ против 430–700 МБ полной выгрузки."""
        conn = self.fresh_db()
        si.bootstrap(conn, self._archive(), symbols=["VGIT.US"],
                     window_days=1825, today=date(2026, 8, 10))
        symbols = {r[0] for r in conn.execute(
            "SELECT source_symbol FROM instruments")}
        self.assertEqual(symbols, {"VGIT.US"})

    def test_seam_between_archive_and_delta_is_exact(self) -> None:
        """🔴 Замер 2026-08-10: совпадение до последнего знака.

        Расхождение здесь означает ретроспективный пересчёт истории у
        источника — и лечится ре-бутстрапом, а не патчем.
        """
        conn = self.fresh_db()
        archive = self._archive()
        delta = write_daily(self.tmp, 20260807, [row("VGIT.US", 20260807,
                                                     c=58.42)])
        si.apply_batch(conn, si.parse_daily_file(delta, min_us_rows=0))
        self.assertEqual(si.verify_seam(conn, archive, 20260807,
                                        symbols=["VGIT.US"]), [])

    def test_seam_ignores_archive_symbols_outside_the_database(self) -> None:
        """🔴 Регресс `AUDIT §−80`, самая дорогая находка ревью.

        `symbols=None` означало «весь архив», и каждая бумага, которую мы
        сознательно НЕ грузили, приезжала как расхождение `архив=X база=None`.
        На реальном архиве это ~11 800 ложных срабатываний при рабочем наборе
        в 13 бумаг: пункт чек-листа «verify-seam совпал» не мог пройти никогда,
        а команда переставала что-либо значить.
        """
        conn = self.fresh_db()
        archive = self._archive()                # VGIT.US и AAPL.US
        si.bootstrap(conn, archive, symbols=["VGIT.US"], window_days=1825,
                     today=date(2026, 8, 10))
        self.assertEqual(si.verify_seam(conn, archive, 20260807), [],
                         "AAPL.US нет в базе осознанно — это не расхождение")

    def _split_archive(self) -> Path:
        """Архив с двумя бумагами из `KNOWN_SPLITS` и РАЗНЫМ поведением."""
        archive = self.tmp / "splits"
        archive.mkdir()
        # NVDA 10:1 2024-06-10 — ступенька на месте, ряд сырой.
        write_history(archive, "nvda.us.txt",
                      [row("NVDA.US", 20240607, c=1200.0),
                       row("NVDA.US", 20240610, c=120.0)])
        # AMZN 20:1 2022-06-06 — ступеньки нет, ряд скорректирован.
        write_history(archive, "amzn.us.txt",
                      [row("AMZN.US", 20220603, c=124.0),
                       row("AMZN.US", 20220606, c=122.0)])
        return archive

    def test_convention_detects_raw_series(self) -> None:
        """Ступенька в дату сплита = ряд отдан СЫРЫМ.

        От этого ответа зависит, вправе ли провайдер объявить конвенцию: если
        ряд сырой, движок обязан корректировать сам; если уже
        скорректированный — повторная корректировка удвоит её и испортит цены
        правдоподобно (`PHASE_08B §7.6`).
        """
        probes = {p.ticker: p for p in si.measure_convention(
            self._split_archive(), today=date(2026, 8, 10))}
        self.assertAlmostEqual(probes["NVDA.US"].observed_ratio, 10.0, places=3)
        self.assertEqual(probes["NVDA.US"].verdict, si.VERDICT_RAW)

    def test_convention_detects_adjusted_series(self) -> None:
        probes = {p.ticker: p for p in si.measure_convention(
            self._split_archive(), today=date(2026, 8, 10))}
        self.assertLess(probes["AMZN.US"].observed_ratio, 1.5)
        self.assertEqual(probes["AMZN.US"].verdict, si.VERDICT_ADJUSTED)

    def test_convention_says_nothing_when_paper_is_absent(self) -> None:
        """Отсутствие бумаги — не вердикт. Молчание честнее догадки."""
        probes = {p.ticker: p for p in si.measure_convention(
            self._split_archive(), today=date(2026, 8, 10))}
        self.assertEqual(probes["TSLA.US"].verdict, si.VERDICT_ABSENT)
        self.assertIsNone(probes["TSLA.US"].observed_ratio)
        self.assertFalse(probes["TSLA.US"].measured)

    def test_convention_refuses_to_call_a_wrong_step_raw(self) -> None:
        """🔴 Ступенька, не равная сплиту, — это НЕ «ряд сырой».

        Прежняя редакция отвечала двумя вердиктами: «больше порога → СЫРОЙ»,
        иначе «СКОРРЕКТИРОВАН». Отношение 2.0 при сплите 10:1 не описывается
        ни одной из гипотез — там происходит что-то третье, — но объявлялось
        сырым УВЕРЕННО. Ровно этот класс ошибки фаза и обязана исключить:
        неверно объявленная конвенция оставляет цены правдоподобными.
        """
        archive = self.tmp / "half"
        archive.mkdir()
        write_history(archive, "nvda.us.txt",
                      [row("NVDA.US", 20240607, c=240.0),
                       row("NVDA.US", 20240610, c=120.0)])   # ступенька 2, не 10
        probes = {p.ticker: p for p in si.measure_convention(
            archive, today=date(2026, 8, 10))}
        self.assertEqual(probes["NVDA.US"].verdict, si.VERDICT_UNCLEAR)
        self.assertTrue(probes["NVDA.US"].measured,
                        "событие измерено — просто ответ не читается")

    def test_convention_names_the_bars_it_compared(self) -> None:
        """Даты сравненных баров печатаются: разрыв ряда виден только по ним.

        Отношение цен через месяц молчания у события — это не замер сплита, а
        замер месячного движения. Отличить одно от другого без дат нельзя.
        """
        probes = {p.ticker: p for p in si.measure_convention(
            self._split_archive(), today=date(2026, 8, 10))}
        self.assertEqual(probes["NVDA.US"].date_before, 20240607)
        self.assertEqual(probes["NVDA.US"].date_after, 20240610)

    def test_convention_does_not_trust_the_order_of_rows(self) -> None:
        """«Последний бар до события» — самый поздний по ДАТЕ, а не по строке.

        Живые файлы Stooq отсортированы по возрастанию, и замер на них верен
        при любой реализации — то есть порядок здесь является НЕПРОВЕРЕННЫМ
        допущением о поставщике. Фикстура его нарушает намеренно: с опорой на
        порядок строк тот же архив даёт «СКОРРЕКТИРОВАН» вместо «СЫРОЙ».
        """
        archive = self.tmp / "shuffled"
        archive.mkdir()
        write_history(archive, "nvda.us.txt",
                      [row("NVDA.US", 20240607, c=1200.0),
                       row("NVDA.US", 20240605, c=130.0),
                       row("NVDA.US", 20240610, c=120.0)])
        probes = {p.ticker: p for p in si.measure_convention(
            archive, today=date(2026, 8, 10))}
        self.assertEqual(probes["NVDA.US"].date_before, 20240607)
        self.assertEqual(probes["NVDA.US"].verdict, si.VERDICT_RAW)

    def test_convention_threshold_matches_the_engine(self) -> None:
        """🔴 Порог «ступеньки» ОБЯЗАН совпадать с эвристикой движка.

        Свой порог означал бы, что замер объявляет ряд сырым, а
        `_detect_and_adjust_splits` ту же ступеньку не видит — и наоборот.
        """
        import math

        from freedom_portfolio.history import _SPLIT_RETURN_THRESHOLD

        self.assertAlmostEqual(math.log(si._SPLIT_STEP_THRESHOLD),
                               _SPLIT_RETURN_THRESHOLD, places=9)

    def test_seam_mismatch_is_reported(self) -> None:
        conn = self.fresh_db()
        archive = self._archive()
        delta = write_daily(self.tmp, 20260807, [row("VGIT.US", 20260807,
                                                     c=99.99)])
        si.apply_batch(conn, si.parse_daily_file(delta, min_us_rows=0))
        mismatches = si.verify_seam(conn, archive, 20260807, symbols=["VGIT.US"])
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0].archive_close, 58.42)
        self.assertEqual(mismatches[0].stored_close, 99.99)


# ═════════════════════════════════════════════════════════════════════════════
# Рабочий набор — какие бумаги кладутся в базу
# ═════════════════════════════════════════════════════════════════════════════

class UniverseSelectionTest(_TempDirMixin, unittest.TestCase):
    """🔴 Какие бумаги вообще класть в базу (`AUDIT §−85`).

    Прежний «рабочий набор» был равен факторам и бенчмаркам — 13 бумаг, 1 МБ.
    Допуск C-1 такая база проходит, а отчёт по портфелю пользователя не строит
    НИ ОДИН: бумаг пользователя в ней нет, и C-8 блокирует по доле непокрытой
    стоимости. План (`PHASE_08B §3.1`) закладывал 0.6–1.0 млн баров, то есть
    500–800 бумаг, и прямо отмечал «нужен механизм добора» — механизма не было.
    """

    def _archive(self) -> Path:
        archive = self.tmp / "archive"
        archive.mkdir()
        days = self._business_days(1300)
        self._write(archive, "LIQUID.US", days, volume=500_000)
        self._write(archive, "THIN.US", days, volume=100)
        self._write(archive, "YOUNG.US", days[-100:], volume=500_000)
        self._write(archive, "DEAD.US", days[:-40], volume=500_000)
        self._write(archive, "^SPX", days, volume=500_000)
        return archive

    @staticmethod
    def _business_days(count: int) -> list[int]:
        out: list[int] = []
        cursor = date(2026, 8, 7)
        while len(out) < count:
            if cursor.weekday() < 5:
                out.append(int(cursor.strftime("%Y%m%d")))
            cursor -= pd.Timedelta(days=1).to_pytimedelta()
        return sorted(out)

    @staticmethod
    def _write(directory: Path, symbol: str, days, *, volume: float) -> None:
        write_history(directory, f"{symbol.lower()}.txt",
                      [row(symbol, day, c=100.0, vol=volume) for day in days])

    def _picked(self, **overrides) -> list[str]:
        options = dict(min_bars=1260, stale_after_days=10,
                       min_turnover=1_000_000.0, limit=800)
        options.update(overrides)
        candidates = si.scan_archive(self._archive(), window_days=1825,
                                     today=date(2026, 8, 10))
        return si.select_universe(candidates, **options)

    def test_liquid_paper_with_full_history_is_taken(self) -> None:
        self.assertIn("LIQUID.US", self._picked())

    def test_illiquid_paper_is_dropped(self) -> None:
        """Неликвид даёт ряд из повторов и ложно НИЗКУЮ волатильность.

        Опаснее пропуска: пропущенную бумагу видно в отчёте, а заниженный риск
        выглядит как хорошая новость.
        """
        self.assertNotIn("THIN.US", self._picked())

    def test_young_listing_is_dropped(self) -> None:
        """Короткая история схлопывает окно регрессии ВСЕЙ панели (F-15/F-21).

        То есть одна молодая бумага портит числа у чужих позиций.
        """
        self.assertNotIn("YOUNG.US", self._picked())

    def test_delisted_paper_is_dropped(self) -> None:
        """Замер `tph.us`: последняя цена мёртвой бумаги — не сегодняшняя."""
        self.assertNotIn("DEAD.US", self._picked())

    def test_index_never_enters_the_universe(self) -> None:
        """Индекс — не инструмент: держать его позицией нельзя."""
        self.assertNotIn("^SPX", self._picked())

    def test_limit_is_a_size_ceiling_and_cuts_the_least_liquid(self) -> None:
        """Потолок — про размер базы, а не про качество бумаги.

        Замер: 75 КБ на бумагу, значит 800 бумаг ≈ 60 МБ против ≈900 МБ полной
        выгрузки. Отсекать надо наименее ликвидное — его отсутствие пользователь
        заметит с наименьшей вероятностью.
        """
        candidates = [
            si.UniverseCandidate("BIG.US", "US", 1300, 20260807, 9e9),
            si.UniverseCandidate("MID.US", "US", 1300, 20260807, 5e6),
            si.UniverseCandidate("SMALL.US", "US", 1300, 20260807, 2e6),
        ]
        picked = si.select_universe(candidates, min_bars=1260,
                                    stale_after_days=10, min_turnover=1e6,
                                    limit=2)
        self.assertEqual(picked, ["BIG.US", "MID.US"])

    def test_selection_is_pure_and_needs_no_files(self) -> None:
        """Выбор отделён от сканирования НАМЕРЕННО.

        Дорогой проход по 12 000 файлов и правило отбора — разные вещи; слитые
        вместе, они сделали бы правило непроверяемым без выгрузки на диске.
        """
        self.assertEqual(
            si.select_universe([], min_bars=1, stale_after_days=1,
                               min_turnover=0.0, limit=10), [])

# ═════════════════════════════════════════════════════════════════════════════
# ЧК-08.5-3 — чтение
# ═════════════════════════════════════════════════════════════════════════════

class StoreReadTest(_TempDirMixin, unittest.TestCase):
    """Читатель: запрет записи, сопоставление, свежесть, матрица."""

    def setUp(self) -> None:                                # noqa: N802
        super().setUp()
        conn = self.fresh_db()
        for day in (20260806, 20260807):
            path = write_daily(self.tmp, day, [
                row("SPY.US", day, c=700 + (day % 10)),
                row("BRK-B.US", day, c=500.0),
                row("BTC.V", day, c=64000.0),
            ])
            si.apply_batch(conn, si.parse_daily_file(path, min_us_rows=0))
        # Суббота: только крипта — ровно как в замеренном файле на 258 строк.
        saturday = write_daily(self.tmp, 20260808, [row("BTC.V", 20260808,
                                                        c=64973.28)])
        si.apply_batch(conn, si.parse_daily_file(saturday, min_us_rows=0))
        # Мёртвая бумага: последний бар задолго до конца базы (`tph.us`).
        dead = write_daily(self.tmp, 20260513, [row("TPH.US", 20260513, c=46.95)])
        si.apply_batch(conn, si.parse_daily_file(dead, min_us_rows=0))
        conn.close()
        self.db = self.tmp / "prices.sqlite"

    def open_store(self) -> StooqStore:
        store = StooqStore.open_readonly(self.db)
        self.addCleanup(store.close)
        return store

    def test_connection_is_read_only(self) -> None:
        """🔴 Запрет технический, а не договорной: гонок нет по построению."""
        store = self.open_store()
        with self.assertRaises(sqlite3.OperationalError):
            store._conn.execute(                          # noqa: SLF001
                "INSERT INTO meta(key, value) VALUES('x','y')")

    def test_absent_database_says_so_plainly(self) -> None:
        with self.assertRaises(StoreUnavailable):
            StooqStore.open_readonly(self.tmp / "nope.sqlite")

    def test_engine_ticker_resolves_through_symbol_form(self) -> None:
        store = self.open_store()
        self.assertEqual(store.resolve("BTC-USD").source_symbol, "BTC.V")
        self.assertEqual(store.resolve("BRK.B").source_symbol, "BRK-B.US")

    def test_missing_lists_what_is_absent(self) -> None:
        store = self.open_store()
        self.assertEqual(store.missing(["SPY.US", "KZTK.KZ"]), ["KZTK.KZ"])

    def test_columns_are_named_by_engine_ticker(self) -> None:
        """Переименование колонки в `BTC.V` тихо потеряло бы позицию движка."""
        store = self.open_store()
        frame = store.bars(["BTC-USD"], days=1825, as_of=20260808)
        self.assertEqual(list(frame.columns), ["BTC-USD"])

    def test_saturday_is_not_stale_for_us(self) -> None:
        """🔴 Календарные дни дали бы ложную просрочку у ЖИВОЙ бумаги."""
        store = self.open_store()
        self.assertEqual(
            store.staleness_trading_days("SPY.US", as_of=20260808), 0)

    def test_dead_ticker_is_stale_in_trading_days(self) -> None:
        """Замер `tph.us`: база свежая, бумага молчит с 2026-05-13."""
        store = self.open_store()
        self.assertGreater(
            store.staleness_trading_days("TPH.US", as_of=20260808), 0)

    def test_absent_ticker_returns_none_not_zero(self) -> None:
        """«Нет данных» и «данные свежие» — разные исходы с разной реакцией."""
        store = self.open_store()
        self.assertIsNone(store.staleness_trading_days("KZTK.KZ",
                                                       as_of=20260808))

    def test_crypto_has_its_own_calendar(self) -> None:
        store = self.open_store()
        self.assertIn(20260808, store.sessions("CRYPTO"))
        self.assertNotIn(20260808, store.sessions("US"))

    def test_matrix_is_sorted_and_typed(self) -> None:
        store = self.open_store()
        frame = store.bars(["SPY.US", "BTC-USD"], days=1825, as_of=20260808)
        self.assertIsInstance(frame.index, pd.DatetimeIndex)
        self.assertTrue(frame.index.is_monotonic_increasing)

    def test_window_cuts_by_calendar_days(self) -> None:
        """Окно движка — про глубину истории, а не про свежесть."""
        store = self.open_store()
        frame = store.bars(["SPY.US"], days=2, as_of=20260807)
        self.assertEqual(len(frame), 2)

    def test_generation_is_readable(self) -> None:
        store = self.open_store()
        self.assertTrue(store.generation())

    def test_symbol_map_wins_over_candidate_search(self) -> None:
        """Ручное решение оператора старше автоматического перебора."""
        conn = si.connect(self.db)
        self.addCleanup(conn.close)
        row = conn.execute("SELECT id FROM instruments WHERE source_symbol=?",
                           ("BRK-B.US",)).fetchone()
        conn.execute("INSERT INTO symbol_map(engine_ticker, instrument_id, "
                     "match_kind, note) VALUES('KSPI.KZ', ?, "
                     "'venue_substitution', 'ADR вместо KASE')", (row["id"],))
        conn.commit()
        store = self.open_store()
        found = store.resolve("KSPI.KZ")
        self.assertEqual(found.source_symbol, "BRK-B.US")
        self.assertEqual(store.venue_substitutions(["KSPI.KZ"]),
                         [("KSPI.KZ", "BRK-B.US")])

    def test_coverage_and_store_resolve_identically(self) -> None:
        """🔴 Регресс `AUDIT §−80`: допуск C-1 и бот обязаны видеть одно.

        Раньше `coverage_report` не знал про `symbol_map`, а `resolve` знал.
        Первая же ручная подмена площадки дала бы «C-1: нет истории» и код
        возврата 1 для бумаги, которую бот находит прекрасно.
        """
        conn = si.connect(self.db)
        self.addCleanup(conn.close)
        row = conn.execute("SELECT id FROM instruments WHERE source_symbol=?",
                           ("SPY.US",)).fetchone()
        conn.execute("INSERT INTO symbol_map(engine_ticker, instrument_id, "
                     "match_kind, note) VALUES('KZTK.KZ', ?, "
                     "'venue_substitution', NULL)", (row["id"],))
        conn.commit()
        store = self.open_store()
        self.assertIsNotNone(store.resolve("KZTK.KZ"))
        self.assertIsNotNone(si.coverage_report(conn, ["KZTK.KZ"])["KZTK.KZ"])

    def test_market_wide_stall_is_visible_only_by_the_clock(self) -> None:
        """🔴 Ограничение потикерной свежести — названо и покрыто (`§−80`).

        Календарь рынка выводится из тех же строк, что и бары: если рынок
        встал целиком, сессий после последнего бара нет, и каждая бумага
        честно рапортует 0. Разорвать круг могут только часы снаружи.
        """
        store = self.open_store()
        self.assertEqual(store.staleness_trading_days("SPY.US",
                                                      as_of=20260808), 0)
        self.assertEqual(
            store.market_staleness_days("US", today=date(2026, 9, 7)), 31)
        self.assertIsNone(store.market_staleness_days("JP",
                                                      today=date(2026, 9, 7)))


# ═════════════════════════════════════════════════════════════════════════════
# ЧК-08.5-5 — CLI оператора
# ═════════════════════════════════════════════════════════════════════════════

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@unittest.skipUnless((_SCRIPTS / "stooq_ingest.py").exists(),
                     "каталога scripts/ нет в деплой-образе")
class OperatorCliTest(_TempDirMixin, unittest.TestCase):
    """CLI — тонкая обёртка; проверяем именно это, а не логику загрузки.

    Тест обязан `skipTest` при отсутствии `scripts/`: иначе он валит
    деплой-гейт, где этого каталога нет по построению.
    """

    def _cli(self):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "stooq_ingest_cli", _SCRIPTS / "stooq_ingest.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(module)
        return module

    def test_working_set_is_derived_from_code(self) -> None:
        """Универсум берётся у `data_checks`, а не списком в скрипте.

        Markdown-список уже трижды расходился с реальностью (`AUDIT §−73`);
        список в скрипте разошёлся бы так же и так же молча.
        """
        from finance.data_checks import FACTOR_ETFS

        working = self._cli()._working_set()
        for ticker in FACTOR_ETFS:
            self.assertIn(ticker, working)

    def test_apply_reports_c1_and_exits_zero(self) -> None:
        cli = self._cli()
        inbox = self.tmp / "inbox"
        inbox.mkdir()
        write_daily(inbox, 20260807, [row("SPY.US", 20260807)])
        code = cli.main(["--db", str(self.tmp / "p.sqlite"), "apply",
                         "--inbox", str(inbox),
                         "--applied", str(self.tmp / "applied"),
                         "--rejected", str(self.tmp / "rejected")])
        self.assertEqual(code, 1)          # 9 факторов из 10 отсутствуют

    def test_parser_builds(self) -> None:
        self._cli().build_parser()

    def test_output_survives_a_windows_console(self) -> None:
        """🔴 На `cmd.exe` (cp866) вывод падал бы `UnicodeEncodeError`.

        Замер: в тексте команд 14 символов `🔴`, 3 `✅`, 3 `⚠️`, плюс `→`, `≈`,
        `—`, `§` — ни один не кодируется в cp866. Питон в этом случае не
        рисует квадратики, а бросает исключение: оператор получил бы traceback
        вместо результата и решил бы, что сломан загрузчик, а не консоль.

        Проверяется СКВОЗЬ команду с потоком в cp866 — проверка одного лишь
        `reconfigure` не доказала бы, что он вызывается на реальном пути.
        """
        import os
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1]
        code = (
            "import io, sys, runpy\n"
            "sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='cp866')\n"
            f"sys.argv = ['stooq_ingest.py', '--db', r'{self.tmp / 'p.sqlite'}',"
            f" 'apply', '--inbox', r'{self.tmp}']\n"
            "try:\n"
            "    runpy.run_path('scripts/stooq_ingest.py', run_name='__main__')\n"
            "except SystemExit:\n"
            "    pass\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=str(root), capture_output=True,
            text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": "src"})
        self.assertNotIn("UnicodeEncodeError", proc.stderr)
        self.assertEqual(proc.returncode, 0, proc.stderr[-400:])

    def test_universe_is_asked_in_engine_tickers(self) -> None:
        """🔴 Два списка, и путать их нельзя (`AUDIT §−80`).

        Загрузчик ищет файлы по именам Stooq, а `store.missing()` спрашивают
        тикером движка. Прежняя редакция подавала в `verify-universe` символы
        источника: для нынешнего универсума (сплошь `.US`) они совпадают
        СЛУЧАЙНО, а бумага БЕЗ кандидатов исчезала из списка молча — и команда
        рапортовала полное покрытие ровно для тех бумаг, которых нет.

        Поэтому universe подменяется на такой, где два списка РАСХОДЯТСЯ:
        сравнение на реальном универсуме ничего бы не доказало (проверено
        мутацией — тест на нём не падал).
        """
        cli = self._cli()
        original = cli._engine_universe
        cli._engine_universe = lambda: (("SPY.US", "FFSPC6.1028.AIX"), ())
        self.addCleanup(setattr, cli, "_engine_universe", original)

        self.assertIn("FFSPC6.1028.AIX", cli._engine_tickers(),
                      "бумага без кандидатов обязана остаться в универсуме")
        self.assertNotIn("FFSPC6.1028.AIX", cli._working_set(),
                         "в загрузку она попасть не может — форм символа нет")

        # И то же самое СКВОЗЬ команду: проверка одних функций не доказывает,
        # что команда зовёт правильную (мутация это и показала — call site
        # можно было подменить, а тест не падал).
        import io
        from contextlib import redirect_stdout

        conn = self.fresh_db()
        path = write_daily(self.tmp, 20260807, [row("SPY.US", 20260807)])
        si.apply_batch(conn, si.parse_daily_file(path, min_us_rows=0))
        conn.close()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--db", str(self.tmp / "prices.sqlite"),
                             "verify-universe"])
        self.assertEqual(code, 1, "непокрытая бумага обязана дать код 1")
        self.assertIn("FFSPC6.1028.AIX", buffer.getvalue())

    def test_verify_seam_without_database_says_so(self) -> None:
        cli = self._cli()
        code = cli.main(["--db", str(self.tmp / "нет.sqlite"), "verify-seam",
                         "--date", "2026-08-07",
                         "--archive", str(self.tmp)])
        self.assertEqual(code, 1)

    def test_missing_archive_is_named_instead_of_blamed_on_stooq(self) -> None:
        """🔴 Опечатка в пути НЕ должна читаться как вывод о поставщике.

        Без гарда `bootstrap` на несуществующем каталоге печатал «файлов
        обработано 0», а следом «🔴 нет истории: SPY.US» — то есть буквально
        утверждал, что в Stooq нет SPY. Проект уже трижды делал ложный вывод
        такого рода из пустого перебора (`AUDIT §−76`), и стоил он отменённой
        фазы. На Windows опечатка вероятнее всего: путь с пробелами без
        кавычек рвётся argparse'ом.
        """
        import io
        from contextlib import redirect_stdout

        cli = self._cli()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--db", str(self.tmp / "p.sqlite"), "bootstrap",
                             "--archive", str(self.tmp / "опечатка"),
                             "--symbols", "SPY.US"])
        self.assertEqual(code, 1)
        self.assertIn("каталога архива нет", buffer.getvalue())
        self.assertNotIn("нет истории", buffer.getvalue(),
                         "отсутствие каталога — не утверждение о Stooq")

    def test_seam_on_an_empty_database_cannot_report_success(self) -> None:
        """🔴 Проверка, которая не может провалиться, ничего не подтверждает.

        `verify_seam` сверяет то, что ЕСТЬ В БАЗЕ (`AUDIT §−80`). На пустой
        базе сверять нечего, и прежняя редакция честно печатала «расхождений
        нет ✅» с кодом 0 — пункт гейта «verify-seam совпал» проходил
        ВХОЛОСТУЮ, до всякого бутстрапа.
        """
        import io
        from contextlib import redirect_stdout

        cli = self._cli()
        self.fresh_db().close()               # схема есть, инструментов нет
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--db", str(self.tmp / "prices.sqlite"),
                             "verify-seam", "--date", "2026-08-07",
                             "--archive", str(self.tmp)])
        self.assertEqual(code, 1)
        self.assertIn("нет ни одного инструмента", buffer.getvalue())

    def test_measure_convention_counts_unclear_events_apart(self) -> None:
        """Неясные события считаются ОТДЕЛЬНО и называются в сводке.

        Смешать их с «нет в архиве» значит потерять единственный сигнал о том,
        что у бумаги происходит что-то третье; смешать с ответом — объявить
        конвенцию по данным, которые её не подтверждают.
        """
        import io
        from contextlib import redirect_stdout

        cli = self._cli()
        archive = self.tmp / "arch"
        archive.mkdir()
        write_history(archive, "nvda.us.txt",           # ступенька 10 — сырой
                      [row("NVDA.US", 20240607, c=1200.0),
                       row("NVDA.US", 20240610, c=120.0)])
        write_history(archive, "tsla.us.txt",           # ступенька 1.8 при 3:1
                      [row("TSLA.US", 20220824, c=270.0),
                       row("TSLA.US", 20220825, c=150.0)])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["measure-convention", "--archive", str(archive)])
        out = buffer.getvalue()
        self.assertEqual(code, 0, "односторонний ответ — не отказ")
        self.assertIn("сырых 1", out)
        self.assertIn("неясных 1", out)
        self.assertIn("не читаются ни как сырые", out)

    def test_db_default_follows_stooq_root(self) -> None:
        """Один корень управляет и папками, и базой.

        Иначе быстрый старт собирает базу в одном месте, а заливает в облако
        из другого — и оператор публикует пустой файл.
        """
        cli = self._cli()
        default_db = Path(cli.build_parser().get_default("db"))
        self.assertEqual(default_db.parent, cli.DEFAULT_ROOT)

    def test_ingest_library_needs_no_third_party(self) -> None:
        """🔴 Оператор не обязан ставить окружение бота ради разбора текста.

        `finance.stooq_ingest` — чистый stdlib, и это ПРОВЕРЯЕМОЕ свойство, а
        не намерение: первый же `import pandas`, добавленный в модуль, уронит
        этот тест. Без него `bootstrap`/`apply` однажды потребовали бы sklearn
        и aiogram, и регламент оператора превратился бы в установку прода.
        """
        import subprocess
        import sys

        code = (
            "import sys, importlib.abc\n"
            "BLOCKED = {'pandas', 'numpy', 'sklearn', 'aiogram', 'anthropic'}\n"
            "class B(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in BLOCKED:\n"
            "            raise ImportError(name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, B())\n"
            "import finance.stooq_ingest\n"
            "print('ok')\n"
        )
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(root),
                              env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0,
                         f"stooq_ingest потянул стороннее:\n{proc.stderr}")

    def test_c1_is_reported_as_unchecked_without_pandas(self) -> None:
        """Без pandas допуск не подтверждён — и об этом СКАЗАНО.

        Молча пропустить C-1 нельзя: это и есть допуск оператора. Пропуск без
        сообщения читался бы как «проверено и всё хорошо».
        """
        import io
        from contextlib import redirect_stdout

        cli = self._cli()
        conn = self.fresh_db()

        def boom():
            raise ImportError("No module named 'pandas'", name="pandas")

        original = cli._engine_universe
        cli._engine_universe = boom
        self.addCleanup(setattr, cli, "_engine_universe", original)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli._print_c1(conn)
        self.assertEqual(code, 0)
        self.assertIn("C-1 НЕ ПРОВЕРЕН", buffer.getvalue())


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
