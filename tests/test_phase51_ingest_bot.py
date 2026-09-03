"""
Бот-загрузчик котировок: публикация с CAS, гарды, приём файла (IB-0…IB-4).

🔴 Имя файла — `test_phase51_*` НАМЕРЕННО.  Деплой-гейт в `cloudbuild.yaml`
запускает `unittest discover -p "test_phase*.py"`, и под этот шаблон не
попадают ни `test_stooq_*`, ни `test_manual_*`, ни `test_layering`.  То есть
весь ценовой путь сегодня проверяется GitHub CI, но НЕ проверяется гейтом,
который решает судьбу образа (находка `IB-F1`, `AUDIT §1`).  Пока шаблон не
расширен, единственный способ попасть в гейт — назваться фазой.

Что здесь проверяется и чего здесь нет
──────────────────────────────────────
Ни один тест не ходит в сеть и не поднимает Telegram.  Хранилище — локальный
каталог (офлайн-бэкенд), aiogram-часть проверяется отдельным классом и
`skipTest`-ится там, где библиотеки нет: логика бота обязана быть проверяемой
без неё, ради этого политика доступа и вынесена в `ingest_access`.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ingest_access as access                                  # noqa: E402
from finance import stooq_ingest as si                          # noqa: E402
from services import quote_ingest as qi                         # noqa: E402
from services.quote_publisher import (Cursor, LocalQuotePublisher,  # noqa: E402
                                      PublisherUnavailable, UploadResult,
                                      publisher_from_env)

_HAS_AIOGRAM = importlib.util.find_spec("aiogram") is not None

_HEADER = ("<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,"
           "<VOL>,<OPENINT>\n")


def _history_text(symbol: str, days) -> str:
    rows = [_HEADER]
    for offset, day in enumerate(days):
        price = 100 + offset
        rows.append(f"{symbol},D,{day},000000,{price},{price + 1},"
                    f"{price - 1},{price},1000,0\n")
    return "".join(rows)


def _daily_text(day: int, symbols) -> str:
    rows = [_HEADER]
    for symbol in symbols:
        rows.append(f"{symbol},D,{day},000000,10,11,9,10.5,500,0\n")
    return "".join(rows)


def _bars(db_path: Path) -> list[tuple]:
    """Содержимое таблицы фактов — то, что обязано быть идемпотентным."""
    conn = si.connect(db_path, read_only=True)
    try:
        return [tuple(r) for r in conn.execute(
            "SELECT instrument_id, trade_date, close FROM daily_bars "
            "ORDER BY instrument_id, trade_date")]
    finally:
        conn.close()


class _IngestCase(unittest.TestCase):
    """Общая обвязка: собранная бутстрапом база + локальное хранилище."""

    universe = ("SPY.US", "AAPL.US")
    seed_days = (20260810, 20260811, 20260812)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ib-test-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = self.root / "store"
        self.store.mkdir()
        self.db = self.store / "prices.sqlite"

        conn = si.connect(self.db)
        si.ensure_schema(conn)
        for symbol in self.universe:
            path = self.root / f"{symbol.lower()}.txt"
            path.write_text(_history_text(symbol, self.seed_days),
                            encoding="utf-8")
            si.apply_batch(conn, si.parse_history_file(path, window_days=9000),
                           kind="bootstrap", allow_new=True)
        conn.close()

        self.publisher = LocalQuotePublisher(self.store)
        # Правило 9 меряет ПОЛНЫЙ дневной срез США (порог 5 000 баров).
        # Синтетический файл на две строки под него не подходит по построению,
        # поэтому порог снимается ЯВНО и только там, где проверяется не он.
        patcher = mock.patch.object(si, "MIN_US_ROWS", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def daily(self, day: int, symbols=None) -> Path:
        path = self.root / f"{day}_d.txt"
        path.write_text(_daily_text(day, symbols or self.universe),
                        encoding="utf-8")
        return path

    def instruments(self) -> int:
        conn = si.connect(self.db, read_only=True)
        try:
            return int(conn.execute(
                "SELECT COUNT(*) AS n FROM instruments").fetchone()["n"])
        finally:
            conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# IB-1 — публикация с compare-and-swap
# ═════════════════════════════════════════════════════════════════════════════

class PublisherContractTest(_IngestCase):

    def test_download_reports_generation_and_size(self) -> None:
        target = self.root / "copy.sqlite"
        snapshot = self.publisher.download(target)
        self.assertTrue(target.exists())
        self.assertGreater(snapshot.generation, 0)
        self.assertEqual(snapshot.size, target.stat().st_size)

    def test_missing_database_is_unavailable_not_created(self) -> None:
        """🔴 Отсутствие базы — отказ, а НЕ повод создать пустую.

        Пустая база, уехавшая в хранилище поверх настоящей, стирает труд
        оператора и при этом выглядит успешной операцией.
        """
        empty = LocalQuotePublisher(self.root / "nowhere")
        with self.assertRaises(PublisherUnavailable):
            empty.download(self.root / "x.sqlite")
        self.assertFalse((self.root / "nowhere" / "prices.sqlite").exists())

    def test_upload_publishes_when_generation_matches(self) -> None:
        snapshot = self.publisher.download(self.root / "copy.sqlite")
        result = self.publisher.upload(self.root / "copy.sqlite",
                                       if_generation_match=snapshot.generation)
        self.assertTrue(result.published)
        self.assertFalse(result.conflict)

    def test_generation_strictly_grows_on_every_publish(self) -> None:
        """Без монотонности CAS не отличит «не менялось» от «менялось дважды».

        На файловой системе с грубым разрешением времени две публикации подряд
        могли бы получить одно и то же `mtime`, поэтому поколение сдвигается
        принудительно, а не берётся из часов на удачу.
        """
        seen = []
        for _ in range(3):
            snapshot = self.publisher.download(self.root / "copy.sqlite")
            result = self.publisher.upload(self.root / "copy.sqlite",
                                           if_generation_match=snapshot.generation)
            seen.append(result.generation)
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(len(set(seen)), 3)

    def test_generation_grows_even_when_the_clock_stands_still(self) -> None:
        """🔴 Мутация показала, что защита НЕДОСТИЖИМА обычным путём.

        `time.time_ns()` почти всегда больше предыдущего значения, поэтому
        ветка принудительного сдвига в нормальном прогоне не исполняется — и
        тест на монотонность её не проверял, а лишь не падал. Здесь часы
        останавливаются намеренно: так ведёт себя файловая система с грубым
        разрешением времени, ради которой ветка и написана.
        """
        fixed = 1_700_000_000_000_000_000
        seen = []
        with mock.patch("services.quote_publisher.time.time_ns",
                        return_value=fixed):
            for _ in range(3):
                snapshot = self.publisher.download(self.root / "copy.sqlite")
                seen.append(self.publisher.upload(
                    self.root / "copy.sqlite",
                    if_generation_match=snapshot.generation).generation)
        self.assertEqual(len(set(seen)), 3, f"поколения совпали: {seen}")
        self.assertEqual(seen, sorted(seen))

    def test_conflict_is_a_result_not_an_exception(self) -> None:
        snapshot = self.publisher.download(self.root / "copy.sqlite")
        self.publisher._stamp(snapshot.generation)      # кто-то залил мимо бота
        result = self.publisher.upload(self.root / "copy.sqlite",
                                       if_generation_match=snapshot.generation)
        self.assertIsInstance(result, UploadResult)
        self.assertTrue(result.conflict)
        self.assertFalse(result.published)

    def test_cursor_round_trip(self) -> None:
        cursor = Cursor().with_applied(20260813, generation=7,
                                       last_run={"file": "a.txt"})
        self.publisher.write_cursor(cursor)
        back = self.publisher.read_cursor()
        self.assertEqual(back.applied_dates, (20260813,))
        self.assertEqual(back.generation, 7)
        self.assertEqual(back.last_run["file"], "a.txt")

    def test_broken_cursor_reads_as_absent_not_as_zero_dates(self) -> None:
        """«Не смог прочитать журнал» и «ничего не применял» — разные ответы.

        Второе означало бы, что откат невозможен по определению, и гард на
        него замолчал бы навсегда.
        """
        self.publisher.cursor_path.write_text("{не json", encoding="utf-8")
        with self.assertLogs("services.quote_publisher", level="WARNING") as log:
            self.assertIsNone(self.publisher.read_cursor())
        # Молча вернуть «нет курсора» тоже нельзя: оператор обязан узнать, что
        # журнал испорчен, иначе гард на откат замолчит незаметно для него.
        self.assertIn("нечитаем", "\n".join(log.output))

    def test_absent_cursor_is_none(self) -> None:
        self.assertIsNone(self.publisher.read_cursor())

    def test_archive_keeps_the_source_file(self) -> None:
        source = self.daily(20260813)
        self.publisher.archive_source(source, source.name)
        self.assertTrue((self.store / "inbox" / source.name).exists())


class BackendSelectionTest(unittest.TestCase):

    def test_default_backend_is_offline_local(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QUOTES_BACKEND", None)
            self.assertEqual(publisher_from_env().name, "local")

    def test_gcs_without_bucket_refuses_instead_of_falling_back(self) -> None:
        """🔴 Опечатка в конфигурации не должна выглядеть как рабочий режим.

        Молчаливый откат в офлайн означал бы сервис, который «работает» и
        ничего никуда не публикует.
        """
        with mock.patch.dict(os.environ,
                             {"QUOTES_BACKEND": "gcs", "QUOTES_BUCKET": ""}):
            with self.assertRaises(PublisherUnavailable):
                publisher_from_env()

    def test_unknown_backend_refuses(self) -> None:
        with mock.patch.dict(os.environ, {"QUOTES_BACKEND": "s3"}):
            with self.assertRaises(PublisherUnavailable):
                publisher_from_env()


# ═════════════════════════════════════════════════════════════════════════════
# IB-2 — цикл применения и четыре гарда
# ═════════════════════════════════════════════════════════════════════════════

class ApplyDailyTest(_IngestCase):

    def test_happy_path_writes_bars_and_publishes(self) -> None:
        outcome = qi.apply_daily(self.daily(20260813), actor="1",
                                 publisher=self.publisher)
        self.assertTrue(outcome.ok, outcome.reason)
        self.assertTrue(outcome.published)
        self.assertEqual(outcome.result.rows_written, 2)
        self.assertEqual(
            [d for _, d, _ in _bars(self.db) if d == 20260813].count(20260813), 2,
            "бары дня не доехали до таблицы фактов")
        self.assertEqual(self.publisher.read_cursor().applied_dates, (20260813,))

    def test_repeat_leaves_the_fact_table_identical(self) -> None:
        """Идемпотентность — про ТАБЛИЦУ ФАКТОВ, а не про байты файла.

        🔴 Побитового совпадения здесь нет и быть не должно: `ingest_runs`
        записывает КАЖДЫЙ прогон, а `meta.generation` меняется при каждом
        применении — ровно ради того, чтобы по любому бару можно было спросить,
        каким файлом он приехал. Требовать неизменности файла значило бы
        требовать отсутствия провенанса.
        """
        source = self.daily(20260813)
        qi.apply_daily(source, actor="1", publisher=self.publisher)
        before = _bars(self.db)
        outcome = qi.apply_daily(source, actor="1", publisher=self.publisher)
        self.assertTrue(outcome.ok)
        self.assertEqual(_bars(self.db), before)

    def test_delta_never_creates_instruments(self) -> None:
        """`allow_new=False` — главное свойство дельты (`AUDIT §−88`)."""
        before = self.instruments()
        outcome = qi.apply_daily(
            self.daily(20260813, ["SPY.US", "AAPL.US", "ZZZZ.US"]),
            actor="1", publisher=self.publisher)
        self.assertTrue(outcome.ok)
        self.assertEqual(self.instruments(), before)
        self.assertEqual(outcome.result.rejected.get("вне рабочего набора"), 1)

    def test_partial_download_leaves_the_store_untouched(self) -> None:
        """Правило 9: неполный будний файл отвергается ЦЕЛИКОМ.

        §−105 развёл ДВЕ причины отказа; здесь проверяется ИНВАРИАНТ, общий
        для обеих: база и её поколение не тронуты. Какая именно причина
        названа — предмет `test_rule9_*` в `test_stooq_price_store`.
        """
        with mock.patch.object(si, "MIN_US_ROWS", 5000):
            before_gen = self.publisher._generation()
            before_bars = _bars(self.db)
            outcome = qi.apply_daily(self.daily(20260814), actor="1",
                                     publisher=self.publisher)
        self.assertFalse(outcome.ok)
        self.assertIn("База не тронута", outcome.reason)
        self.assertEqual(self.publisher._generation(), before_gen)
        self.assertEqual(_bars(self.db), before_bars)

    def test_file_without_date_in_name_is_refused_for_the_daily_parser(self) -> None:
        source = self.root / "srez.txt"
        source.write_text(_daily_text(20260813, self.universe), encoding="utf-8")
        outcome = qi.apply_daily(source, actor="1", publisher=self.publisher)
        self.assertFalse(outcome.ok)
        self.assertIn("дата", outcome.reason.lower())

    def test_captcha_page_instead_of_csv_is_refused(self) -> None:
        source = self.root / "20260813_d.txt"
        source.write_text("<html>are you human?</html>", encoding="utf-8")
        outcome = qi.apply_daily(source, actor="1", publisher=self.publisher)
        self.assertFalse(outcome.ok)
        self.assertIn("<TICKER>", outcome.reason)

    def test_conflict_refuses_loudly_and_does_not_retry(self) -> None:
        """🔴 412 при ОДНОМ операторе — нарушение инварианта, а не невезение.

        Тихий повтор стёр бы единственный признак того, что базу трогали мимо
        бота, поэтому проверяется не только отказ, но и то, что попытка
        публикации была РОВНО ОДНА.
        """
        calls: list[int] = []
        before = _bars(self.db)

        def spy(src, *, if_generation_match):
            calls.append(if_generation_match)
            return UploadResult(published=False, conflict=True,
                                reason="поколение изменилось")

        with mock.patch.object(self.publisher, "upload", spy):
            outcome = qi.apply_daily(self.daily(20260813), actor="1",
                                     publisher=self.publisher)
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.conflict)
        self.assertEqual(len(calls), 1, "конфликт не должен приводить к ретраю")
        self.assertIn("вручную", outcome.reason)
        self.assertEqual(_bars(self.db), before,
                         "при конфликте хранилище обязано остаться прежним")

    def test_failed_publish_tells_the_operator_the_change_is_lost(self) -> None:
        """Свойство read-modify-write, которое нельзя замалчивать."""
        with mock.patch.object(self.publisher, "upload",
                               lambda src, *, if_generation_match:
                               UploadResult(published=False, reason="сеть")):
            outcome = qi.apply_daily(self.daily(20260813), actor="1",
                                     publisher=self.publisher)
        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.conflict)
        self.assertIn("ещё раз", outcome.reason)

    def test_empty_store_refuses_before_reading_the_file(self) -> None:
        """Порядок гардов доказывается ОТСУТСТВУЮЩИМ файлом.

        Если бы разбор шёл первым, мы получили бы ошибку чтения файла; раз
        причина про пустую базу — гард отработал раньше.
        """
        empty_dir = self.root / "empty"
        empty_dir.mkdir()
        conn = si.connect(empty_dir / "prices.sqlite")
        si.ensure_schema(conn)
        conn.close()
        outcome = qi.apply_daily(self.root / "нет-такого.txt", actor="1",
                                 publisher=LocalQuotePublisher(empty_dir))
        self.assertFalse(outcome.ok)
        self.assertIn("ноль инструментов", outcome.reason)

    def test_garbage_object_is_not_turned_into_a_database(self) -> None:
        junk_dir = self.root / "junk"
        junk_dir.mkdir()
        (junk_dir / "prices.sqlite").write_bytes(b"not a database at all")
        outcome = qi.apply_daily(self.daily(20260813), actor="1",
                                 publisher=LocalQuotePublisher(junk_dir))
        self.assertFalse(outcome.ok)
        self.assertIn("не похож на базу", outcome.reason)

    def test_database_without_the_fact_table_is_refused(self) -> None:
        """Мутация вскрыла дыру: проверок ДВЕ, а тест бил только по второй.

        Файл может быть законным SQLite и даже иметь `instruments` — так
        выглядит база, недоделанная прерванным бутстрапом. Без пробы по
        `daily_bars` она проходила бы гард и уезжала в хранилище поверх
        рабочей. Отдельный тест нужен именно потому, что вторая проба
        прикрывала первую и та могла быть удалена незамеченной (`§−80` §7.8).
        """
        half = self.root / "half"
        half.mkdir()
        conn = sqlite3.connect(str(half / "prices.sqlite"))
        conn.execute("CREATE TABLE instruments (id INTEGER PRIMARY KEY, "
                     "source TEXT, source_symbol TEXT, market TEXT, "
                     "currency TEXT, convention TEXT)")
        conn.commit()
        conn.close()
        outcome = qi.apply_daily(self.daily(20260813), actor="1",
                                 publisher=LocalQuotePublisher(half))
        self.assertFalse(outcome.ok)
        self.assertIn("не похож на базу", outcome.reason)

    def test_provenance_names_which_operation_wrote_the_bars(self) -> None:
        """`ingest_runs.kind` — не украшение: по бару надо уметь спросить, чем он приехал.

        Дельта и добор различаются правом заводить бумагу, и если журнал
        называет их одинаково, разобрать постфактум, откуда в базе взялась
        бумага, будет нечем.
        """
        qi.apply_daily(self.daily(20260813), actor="1", publisher=self.publisher)
        source = self.root / "nvda.us.txt"
        source.write_text(_history_text("NVDA.US", (20260810,)), encoding="utf-8")
        qi.apply_history(source, actor="1", publisher=self.publisher)
        conn = si.connect(self.db, read_only=True)
        try:
            kinds = [r["kind"] for r in conn.execute(
                "SELECT kind FROM ingest_runs ORDER BY id")]
        finally:
            conn.close()
        self.assertIn("apply", kinds)
        self.assertIn("backfill", kinds)

    def test_size_collapse_blocks_publication(self) -> None:
        """База, усохшая после дельты, не публикуется: дельта умеет только расти."""
        class Shrinking(LocalQuotePublisher):
            def download(self, dest):
                snapshot = super().download(dest)
                return snapshot.__class__(path=snapshot.path,
                                          generation=snapshot.generation,
                                          size=snapshot.size * 10)

        outcome = qi.apply_daily(self.daily(20260813), actor="1",
                                 publisher=Shrinking(self.store))
        self.assertFalse(outcome.ok)
        self.assertIn("усохла", outcome.reason)

    def test_day_with_zero_written_bars_is_not_recorded_in_the_cursor(self) -> None:
        """Иначе бот сам себе сфабрикует «пропавший» день и позовёт его чинить."""
        outcome = qi.apply_daily(self.daily(20260813, ["ZZZZ.US"]), actor="1",
                                 publisher=self.publisher)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.result.rows_written, 0)
        self.assertEqual(self.publisher.read_cursor().applied_dates, ())


class RollbackDetectionTest(_IngestCase):

    def _drop(self, day: int) -> None:
        conn = si.connect(self.db)
        conn.execute("DELETE FROM daily_bars WHERE trade_date=?", (day,))
        conn.commit()
        conn.close()

    def test_missing_days_are_reported_after_a_rebootstrap(self) -> None:
        qi.apply_daily(self.daily(20260813), actor="1", publisher=self.publisher)
        self._drop(20260813)
        state = qi.status(publisher=self.publisher)
        self.assertEqual(state.missing_dates, (20260813,))
        self.assertIn("20260813", qi.format_status(state))

    def test_rollback_warns_but_does_not_block(self) -> None:
        """🔴 Отступление от `PLAN §5.3`, и оно обоснованное.

        Применить сегодняшний файл к откатившейся базе безвредно — он добавляет
        сегодняшние бары. Блокировка наказывала бы оператора за законную
        квартальную операцию и не давала бы двигаться, пока он не дошлёт
        пропущенное. Лечит откат пересылка файлов, а не запрет работать.
        """
        qi.apply_daily(self.daily(20260813), actor="1", publisher=self.publisher)
        self._drop(20260813)
        outcome = qi.apply_daily(self.daily(20260814), actor="1",
                                 publisher=self.publisher)
        self.assertTrue(outcome.ok, outcome.reason)
        self.assertEqual(outcome.missing_dates, (20260813,))
        self.assertTrue(any("ре-бутстрап" in w for w in outcome.warnings))

    def test_forward_divergence_from_the_cli_is_not_an_alarm(self) -> None:
        """CLI остаётся штатным запасным путём — при одном операторе он весь резерв.

        Курсор, считающий свой снимок единственной истиной, объявил бы день,
        применённый мимо бота, аномалией.
        """
        qi.apply_daily(self.daily(20260813), actor="1", publisher=self.publisher)
        conn = si.connect(self.db)                       # «оператор через CLI»
        si.apply_batch(conn, si.parse_daily_file(self.daily(20260814)),
                       kind="apply")
        conn.close()
        state = qi.status(publisher=self.publisher)
        self.assertEqual(state.missing_dates, ())


class ApplyHistoryTest(_IngestCase):

    def test_history_file_creates_the_instrument(self) -> None:
        source = self.root / "msft.us.txt"
        source.write_text(_history_text("MSFT.US", (20260810, 20260811)),
                          encoding="utf-8")
        before = self.instruments()
        outcome = qi.apply_history(source, actor="1", publisher=self.publisher)
        self.assertTrue(outcome.ok, outcome.reason)
        self.assertEqual(self.instruments(), before + 1)
        self.assertEqual(outcome.result.instruments_added, 1)

    def test_symbol_comes_from_the_file_body_not_from_its_name(self) -> None:
        """🔴 Имя решает ровно один вопрос: история это или дневной срез.

        Символ читается из колонки `<TICKER>` — так же, как его читает
        `bootstrap`. Вывод символа из имени сломался бы на первом же файле,
        который оператор переименовал.
        """
        source = self.root / "renamed-by-hand.txt"
        source.write_text(_history_text("NVDA.US", (20260810,)), encoding="utf-8")
        qi.apply_history(source, actor="1", publisher=self.publisher)
        conn = si.connect(self.db, read_only=True)
        try:
            symbols = {r["source_symbol"] for r in conn.execute(
                "SELECT source_symbol FROM instruments")}
        finally:
            conn.close()
        self.assertIn("NVDA.US", symbols)

    def test_history_window_matches_the_engine_lookback(self) -> None:
        """Другая глубина у добранной бумаги схлопнула бы окно всей панели."""
        source = self.root / "old.us.txt"
        source.write_text(_history_text("OLD.US", (20000101, 20260810)),
                          encoding="utf-8")
        with mock.patch.dict(os.environ, {"HISTORY_LOOKBACK_DAYS": "1825"}):
            outcome = qi.apply_history(source, actor="1",
                                       publisher=self.publisher,
                                       today=date(2026, 8, 20))
        self.assertTrue(outcome.ok, outcome.reason)
        self.assertEqual(outcome.result.rejected.get("вне окна"), 1)
        self.assertEqual(outcome.result.rows_written, 1)


# ═════════════════════════════════════════════════════════════════════════════
# IB-4 — приём файла: лимиты ДО скачивания
# ═════════════════════════════════════════════════════════════════════════════

class UploadClassificationTest(unittest.TestCase):

    def test_daily_slice_is_recognised_by_its_name(self) -> None:
        decision = qi.classify_upload("20260818_d.txt", 700_000)
        self.assertEqual(decision.kind, "daily")
        self.assertEqual(decision.trade_date, 20260818)

    def test_history_file_is_recognised_by_absence_of_a_date(self) -> None:
        self.assertEqual(qi.classify_upload("aapl.us.txt", 120_000).kind,
                         "history")

    def test_oversized_file_is_refused_by_metadata_alone(self) -> None:
        """Проверка по описанию документа: скачивать, чтобы отказать, незачем.

        `/tmp` на Cloud Run — это оперативная память, и загрузка архива туда
        стоила бы контейнеру ровно столько, сколько он весит.
        """
        decision = qi.classify_upload("20260818_d.txt", qi.MAX_UPLOAD_BYTES + 1)
        self.assertIsNone(decision.kind)
        self.assertIn("потолка", decision.reason)

    def test_archive_is_refused_with_an_explanation(self) -> None:
        decision = qi.classify_upload("d_us_txt.zip", 1000)
        self.assertIsNone(decision.kind)
        self.assertIn("архив", decision.reason)

    def test_foreign_extension_is_refused(self) -> None:
        self.assertIsNone(qi.classify_upload("portfolio.xlsx", 1000).kind)

    def test_unknown_size_does_not_block(self) -> None:
        """Telegram не всегда сообщает размер; отсутствие — не превышение."""
        self.assertEqual(qi.classify_upload("20260818_d.txt", None).kind, "daily")


class SummaryTest(_IngestCase):

    def test_summary_carries_every_counter_of_the_batch(self) -> None:
        outcome = qi.apply_daily(
            self.daily(20260813, ["SPY.US", "AAPL.US", "ZZZZ.US"]),
            actor="1", publisher=self.publisher)
        text = qi.format_summary(outcome)
        self.assertIn("строк прочитано", text)
        self.assertIn("баров записано", text)
        self.assertIn("вне рабочего набора", text)
        self.assertIn("поколение", text)

    def test_refusal_summary_names_the_reason(self) -> None:
        outcome = qi.apply_daily(self.root / "нет.txt", actor="1",
                                 publisher=LocalQuotePublisher(self.root / "no"))
        text = qi.format_summary(outcome)
        self.assertIn("НЕ применён", text)
        self.assertIn("бутстрап", text.lower())

    def test_c1_reports_real_numbers_when_the_engine_lists_are_available(self) -> None:
        """🔴 В деплой-образе pandas ЕСТЬ, значит в проде идёт именно эта ветка.

        Без подмены списков она не исполняется здесь ни разу, и сводка
        проверялась бы только в состоянии «не проверено» — то есть в том,
        которого в проде не бывает.
        """
        with mock.patch.object(qi, "_engine_universe",
                               return_value=(("SPY.US", "MSFT.US"), ("AAPL.US",))):
            outcome = qi.apply_daily(self.daily(20260813), actor="1",
                                     publisher=self.publisher)
        self.assertTrue(outcome.c1.checked)
        self.assertEqual((outcome.c1.factors_ok, outcome.c1.factors_total), (1, 2))
        self.assertEqual(outcome.c1.missing_factors, ("MSFT.US",))
        text = qi.format_summary(outcome)
        self.assertIn("C-1 факторы ........... 1/2 🔴", text)
        self.assertIn("нет истории: MSFT.US", text)

    def test_c1_is_either_a_number_or_an_explicit_not_checked(self) -> None:
        """🔴 Молчание про C-1 читалось бы как «проверено, всё хорошо».

        В профиле `STRICT` (ручной ввод) потеря ЛЮБОГО факторного ETF даёт
        `BLOCK`, то есть пользователь получит отказ вместо отчёта.
        """
        outcome = qi.apply_daily(self.daily(20260813), actor="1",
                                 publisher=self.publisher)
        text = qi.format_summary(outcome)
        self.assertTrue("C-1 факторы" in text or "C-1 НЕ ПРОВЕРЕН" in text,
                        f"строки допуска нет вовсе:\n{text}")


class PartialDayIsNamedTest(_IngestCase):
    """🔴 `§−113`. Пять дней подряд срез клал 631 бар в базу из 803 бумаг.

    Сводка печатала «баров записано 631» и «C-1 факторы 10/10 ✅», и оба
    утверждения были правдой — ни одно из них не было ОТВЕТОМ. У числа не было
    знаменателя, а у допуска — свежести: все тринадцать тикеров C-1 замерли на
    первом же дне и продолжали числиться зелёными.

    Тот же класс, что `§−90` A-3 («пустая коллекция ≠ всё хорошо») и `§−90`
    («статус — вердикт, а не оформление»): факт был, вывода не было.
    """

    def test_written_bars_carry_a_denominator(self) -> None:
        outcome = qi.apply_daily(self.daily(20260813, ["SPY.US"]),
                                 actor="1", publisher=self.publisher)
        text = qi.format_summary(outcome)
        self.assertEqual((outcome.universe_total, outcome.missed_total), (2, 1))
        self.assertIn("1 из 2 бумаг базы", text)

    def test_papers_without_a_bar_are_named_not_counted(self) -> None:
        """Имя позволяет открыть файл; число не говорит даже, где смотреть."""
        outcome = qi.apply_daily(self.daily(20260813, ["SPY.US"]),
                                 actor="1", publisher=self.publisher)
        text = qi.format_summary(outcome)
        self.assertIn("без бара за 20260813", text)
        self.assertIn("AAPL.US", text)
        self.assertEqual(outcome.missed, ("AAPL.US",))

    def test_a_full_day_says_nothing_extra(self) -> None:
        """Молчание на здоровом дне обязательно: строка, которая есть всегда,
        перестаёт читаться раньше, чем понадобится."""
        outcome = qi.apply_daily(self.daily(20260813), actor="1",
                                 publisher=self.publisher)
        self.assertEqual(outcome.missed_total, 0)
        self.assertNotIn("без бара за", qi.format_summary(outcome))

    def test_the_plateau_stays_visible_after_the_spike_detector_goes_quiet(
            self) -> None:
        """🔴 Главное свойство. Детектор `§−107` ловит СКАЧОК отбраковки: на
        второй одинаковый день он молчит, потому что первый уже стал нормой.
        Знаменатель обязан говорить КАЖДЫЙ раз — иначе провал 800 → 631
        держится сколько угодно дней под зелёной сводкой.
        """
        first = qi.apply_daily(self.daily(20260813, ["SPY.US"]),
                               actor="1", publisher=self.publisher)
        second = qi.apply_daily(self.daily(20260814, ["SPY.US"]),
                                actor="1", publisher=self.publisher)
        self.assertTrue(second.ok)
        self.assertEqual(second.warnings, (),
                         "детектор всплеска на плато молчит — это его свойство")
        self.assertIn("AAPL.US", qi.format_summary(second))
        self.assertIn("1 из 2 бумаг базы", qi.format_summary(second))
        self.assertIn("AAPL.US", qi.format_summary(first))

    def test_a_stale_factor_is_not_a_green_tick(self) -> None:
        """Фактор В БАЗЕ, но замерший, к следующему отчёту так же непригоден,
        как отсутствующий: в профиле STRICT это BLOCK, а не деградация."""
        with mock.patch.object(qi, "_engine_universe",
                               return_value=(("SPY.US",), ())):
            outcome = qi.apply_daily(self.daily(20260813, ["AAPL.US"]),
                                     actor="1", publisher=self.publisher)
        self.assertEqual(outcome.c1.factors_ok, 1)      # история есть
        self.assertTrue(outcome.c1.complete)            # состав полон
        self.assertFalse(outcome.c1.usable)             # но допуск НЕ пройден
        self.assertEqual(outcome.c1.stale_factors, ("SPY.US",))
        text = qi.format_summary(outcome)
        self.assertIn("C-1 факторы ........... 1/1 🔴", text)
        self.assertIn("ПРОТУХЛИ", text)
        self.assertIn("SPY.US 20260812", text)

    def test_a_fresh_factor_keeps_the_green_tick(self) -> None:
        """Обратная мутация: строгость не должна кричать на здоровой базе."""
        with mock.patch.object(qi, "_engine_universe",
                               return_value=(("SPY.US",), ())):
            outcome = qi.apply_daily(self.daily(20260813), actor="1",
                                     publisher=self.publisher)
        self.assertTrue(outcome.c1.usable)
        text = qi.format_summary(outcome)
        self.assertIn("C-1 факторы ........... 1/1 ✅", text)
        self.assertNotIn("ПРОТУХЛИ", text)

    def test_a_market_that_did_not_trade_is_not_called_missing(self) -> None:
        """🔴 Рынок без единого бара за день был ЗАКРЫТ, а не сломан.

        Форма условия та же, что у правила 9 (`0 < принято < порога`): ноль
        означает выходной. Без оговорки про рынок праздник в США печатал бы
        «без бара 803 бумаги», то есть самая громкая тревога приходилась бы на
        самый штатный день.
        """
        conn = si.connect(self.db)
        path = self.root / "btc.v.txt"
        path.write_text(_history_text("BTC.V", self.seed_days), encoding="utf-8")
        si.apply_batch(conn, si.parse_history_file(path, window_days=9000),
                       kind="bootstrap", allow_new=True)
        conn.close()

        outcome = qi.apply_daily(self.daily(20260813, ["SPY.US", "AAPL.US"]),
                                 actor="1", publisher=self.publisher)
        self.assertEqual(outcome.universe_total, 3)     # BTC.V в базе есть
        self.assertEqual(outcome.missed_total, 0)       # но крипта не торговала
        self.assertNotIn("BTC.V", qi.format_summary(outcome))

    def test_status_says_how_many_papers_got_the_last_day(self) -> None:
        """`MAX(trade_date)` двигает ОДНА бумага, поэтому «последний день» без
        доли свежих не отличает полный день от пятой его части."""
        qi.apply_daily(self.daily(20260813, ["SPY.US"]), actor="1",
                       publisher=self.publisher)
        state = qi.status(publisher=self.publisher)
        market = state.markets[0]
        self.assertEqual((market.instruments, market.fresh), (2, 1))
        text = qi.format_status(state)
        self.assertIn("свежих 1", text)
        self.assertIn("без бара за 20260813", text)


class MangledDailyNameTest(_IngestCase):
    """🔴 `§−114`. Живой инцидент 31.08: браузер сохранил повторную загрузку
    дневного среза как `20260831_d 2.txt`. Строгое имя не совпало, бот принял
    файл за ИСТОРИЮ — а история имеет право заводить бумаги — и завёл 11 281
    бумагу с одним баром каждая, опубликовав базу. Это `§−88` через другую
    дверь: дельту тогда закрыли, вход истории остался открыт.

    Две линии защиты, обе fail-closed:
      * по ИМЕНИ — ещё до скачивания (`classify_upload`);
      * по СОДЕРЖИМОМУ — правило 10 в парсере истории: одна бумага, иначе
        файл не история. Вторая линия не зависит от формы искажения имени.
    """

    def test_a_duplicate_download_name_is_refused_with_the_right_name(self) -> None:
        for name in ("20260831_d 2.txt", "20260831_d (1).txt",
                     "20260831_d_2.txt", "20260831_d-copy.txt"):
            decision = qi.classify_upload(name, 700_000)
            self.assertIsNone(decision.kind, name)
            self.assertIn("20260831_d.txt", decision.reason, name)
            self.assertIn("ИСТОРИИ", decision.reason, name)

    def test_a_real_history_name_is_still_history(self) -> None:
        """Обратная мутация: имя бумаги не начинается с восьми цифр."""
        self.assertEqual(qi.classify_upload("aapl.us.txt", 100_000).kind, "history")
        self.assertEqual(qi.classify_upload("brk-b.us.txt", 100_000).kind, "history")

    def test_a_strict_daily_name_is_still_daily(self) -> None:
        self.assertEqual(qi.classify_upload("20260831_d.txt", 700_000).kind, "daily")

    def test_a_multi_symbol_file_is_refused_as_history_and_the_base_is_untouched(
            self) -> None:
        """🔴 Главная проверка: даже если имя прошло, содержимое не пройдёт."""
        source = self.root / "renamed-by-hand.txt"          # имя — «история»
        source.write_text(_daily_text(20260813, ["MSFT.US", "NVDA.US", "AMD.US"]),
                          encoding="utf-8")
        before = self.instruments()
        generation = self.publisher.download(self.root / "g.sqlite").generation

        outcome = qi.apply_history(source, actor="1", publisher=self.publisher)

        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.store_touched)
        self.assertIn("3 разных бумаг", outcome.reason)
        self.assertIn("20260813_d.txt", outcome.reason)     # дата — из файла
        self.assertEqual(self.instruments(), before)          # ни одной новой
        self.assertEqual(
            self.publisher.download(self.root / "g2.sqlite").generation,
            generation)                                       # не публиковалась

    def test_a_single_symbol_history_still_enrols_the_paper(self) -> None:
        """Обратная мутация: штатный добор одной бумаги не сломан."""
        source = self.root / "msft.us.txt"
        source.write_text(_history_text("MSFT.US", (20260810, 20260811)),
                          encoding="utf-8")
        outcome = qi.apply_history(source, actor="1", publisher=self.publisher)
        self.assertTrue(outcome.ok, outcome.reason)
        self.assertEqual(outcome.result.instruments_added, 1)


# ═════════════════════════════════════════════════════════════════════════════
# IB-3 — доступ
# ═════════════════════════════════════════════════════════════════════════════

class AccessPolicyTest(unittest.TestCase):
    """Политика проверяется ВСЕГДА — она не зависит от aiogram намеренно."""

    def test_empty_list_means_nobody(self) -> None:
        """🔴 Зеркальная противоположность `tg_bot.WhitelistMiddleware`.

        Тот при пустом списке пропускает всех — осознанное решение для беты.
        Здесь пустая переменная не должна открывать запись в базу цен.
        """
        with mock.patch.dict(os.environ, {access.ENV_NAME: ""}):
            self.assertFalse(access.is_admin(148046720))
            self.assertFalse(access.configured())

    def test_unset_variable_means_nobody(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(access.ENV_NAME, None)
            self.assertFalse(access.is_admin(1))

    def test_listed_admin_passes_and_others_do_not(self) -> None:
        with mock.patch.dict(os.environ, {access.ENV_NAME: "148046720, 42"}):
            self.assertTrue(access.is_admin(148046720))
            self.assertTrue(access.is_admin(42))
            self.assertFalse(access.is_admin(43))
            self.assertFalse(access.is_admin(None))

    def test_separators_match_the_main_bot(self) -> None:
        self.assertEqual(access.parse_admin_ids("1;2, 3"), frozenset({1, 2, 3}))

    def test_garbage_is_dropped_not_interpreted(self) -> None:
        self.assertEqual(access.parse_admin_ids("abc, 7, "), frozenset({7}))

    def test_stranger_is_logged_as_a_security_event(self) -> None:
        """Легитимный отправитель один — значит любое иное обращение заметно."""
        with self.assertLogs("ingest_access", level="WARNING") as captured:
            access.note_stranger(999, "сообщение")
        self.assertIn("999", "\n".join(captured.output))


@unittest.skipUnless(_HAS_AIOGRAM, "aiogram не установлен (офлайн-разработка)")
class BotWiringTest(unittest.TestCase):

    def test_dispatcher_builds_without_token_or_network(self) -> None:
        import ingest_bot                                # noqa: PLC0415
        self.assertIsNotNone(ingest_bot.build_dispatcher())

    def test_middleware_stops_a_stranger_before_the_handler(self) -> None:
        import asyncio                                   # noqa: PLC0415
        import ingest_bot                                # noqa: PLC0415

        seen: list[str] = []

        class _User:
            id = 999

        class _Event:
            from_user = _User()

            async def answer(self, text, **_kw):
                seen.append(text)

        async def handler(event, data):                  # pragma: no cover
            seen.append("HANDLER")
            return "reached"

        with mock.patch.dict(os.environ, {access.ENV_NAME: "1"}), \
                self.assertLogs("ingest_access", level="WARNING"):
            result = asyncio.run(
                ingest_bot.AdminOnlyMiddleware()(handler, _Event(), {}))
        self.assertIsNone(result)
        self.assertNotIn("HANDLER", seen)
        self.assertEqual(seen, [access.DENIAL_TEXT])

    def test_middleware_lets_the_admin_through(self) -> None:
        import asyncio                                   # noqa: PLC0415
        import ingest_bot                                # noqa: PLC0415

        class _User:
            id = 148046720

        class _Event:
            from_user = _User()

        async def handler(event, data):
            return "reached"

        with mock.patch.dict(os.environ, {access.ENV_NAME: "148046720"}):
            result = asyncio.run(
                ingest_bot.AdminOnlyMiddleware()(handler, _Event(), {}))
        self.assertEqual(result, "reached")

    def test_main_refuses_to_start_without_admins(self) -> None:
        """Fail-fast повторяет приём `M-9`: тихая деградация скрыла бы ошибку."""
        import asyncio                                   # noqa: PLC0415
        import ingest_bot                                # noqa: PLC0415

        with mock.patch.object(ingest_bot, "BOT_TOKEN", "123:abc"), \
                mock.patch.dict(os.environ, {access.ENV_NAME: ""}):
            with self.assertRaises(RuntimeError):
                asyncio.run(ingest_bot.main())

    def test_main_refuses_to_start_without_a_token(self) -> None:
        import asyncio                                   # noqa: PLC0415
        import ingest_bot                                # noqa: PLC0415

        with mock.patch.object(ingest_bot, "BOT_TOKEN", ""):
            with self.assertRaises(RuntimeError):
                asyncio.run(ingest_bot.main())



class LongOperationNoticeTest(unittest.TestCase):
    """🔴 `§−115`. Живой случай: оператор нажал «выполнить чистку» и получил
    МОЛЧАНИЕ на шесть минут при таймауте 300 с.

    Механика: `asyncio.to_thread` не отменяется. `wait_for` снимал ожидание,
    печатал «не уложилась» как ИТОГ — а поток продолжал удалять и мог
    опубликовать базу уже после этого сообщения. Хуже того, выход из
    `async with _LOCK` отпускал замок поверх ЖИВОЙ записи: следующая команда
    шла параллельно. Данные спасал CAS, картина у оператора — нет.

    Теперь таймаут это ПРЕДУПРЕЖДЕНИЕ, а не результат: ожидание продолжается,
    значит замок держится, а настоящий итог приходит следом.
    """

    def _fake_message(self, sent):
        class _Msg:
            async def answer(self, text, **_kw):
                sent.append(text)
        return _Msg()

    def test_a_slow_operation_gets_a_notice_and_then_the_real_result(self) -> None:
        import asyncio                                   # noqa: PLC0415
        import ingest_bot                                # noqa: PLC0415

        sent: list[str] = []

        async def work():
            await asyncio.sleep(0.15)
            return "ИТОГ"

        async def scenario():
            with mock.patch.object(ingest_bot, "APPLY_TIMEOUT_S", 0.02):
                return await ingest_bot._await_with_notice(
                    self._fake_message(sent), work(), notice="ЖДИТЕ")

        result = asyncio.run(scenario())
        self.assertEqual(result, "ИТОГ")                 # итог НЕ потерян
        self.assertEqual(len(sent), 1)                   # ровно одно уведомление
        self.assertIn("ЖДИТЕ", sent[0])

    def test_a_fast_operation_says_nothing_extra(self) -> None:
        """Обратная мутация: уведомление на каждой операции читать перестанут."""
        import asyncio                                   # noqa: PLC0415
        import ingest_bot                                # noqa: PLC0415

        sent: list[str] = []

        async def work():
            return "ИТОГ"

        async def scenario():
            with mock.patch.object(ingest_bot, "APPLY_TIMEOUT_S", 5):
                return await ingest_bot._await_with_notice(
                    self._fake_message(sent), work(), notice="ЖДИТЕ")

        self.assertEqual(asyncio.run(scenario()), "ИТОГ")
        self.assertEqual(sent, [])

    def test_an_error_after_the_notice_still_reaches_the_caller(self) -> None:
        """Отказ, случившийся ПОСЛЕ предупреждения, обязан дойти как отказ."""
        import asyncio                                   # noqa: PLC0415
        import ingest_bot                                # noqa: PLC0415

        sent: list[str] = []

        async def work():
            await asyncio.sleep(0.15)
            raise RuntimeError("облако отказало")

        async def scenario():
            with mock.patch.object(ingest_bot, "APPLY_TIMEOUT_S", 0.02):
                await ingest_bot._await_with_notice(
                    self._fake_message(sent), work(), notice="ЖДИТЕ")

        with self.assertRaises(RuntimeError):
            asyncio.run(scenario())
        self.assertEqual(len(sent), 1)

    def test_the_lock_is_held_until_the_work_really_finishes(self) -> None:
        """🔴 Главное свойство: замок нельзя отпускать поверх живой записи."""
        import asyncio                                   # noqa: PLC0415
        import ingest_bot                                # noqa: PLC0415

        order: list[str] = []
        sent: list[str] = []

        async def work():
            await asyncio.sleep(0.15)
            order.append("работа закончилась")
            return "ИТОГ"

        async def holder():
            async with ingest_bot._LOCK:
                await ingest_bot._await_with_notice(
                    self._fake_message(sent), work(), notice="ЖДИТЕ")

        async def rival():
            await asyncio.sleep(0.05)                    # уже после таймаута
            async with ingest_bot._LOCK:
                order.append("вторая команда вошла")

        async def scenario():
            with mock.patch.object(ingest_bot, "APPLY_TIMEOUT_S", 0.02):
                await asyncio.gather(holder(), rival())

        asyncio.run(scenario())
        self.assertEqual(order, ["работа закончилась", "вторая команда вошла"])


# ═════════════════════════════════════════════════════════════════════════════
# IB-5/IB-6 — диагностика
# ═════════════════════════════════════════════════════════════════════════════

_FACTORS = ("SPY.US", "AAPL.US")
_BENCH = ("MSFT.US",)


class UniverseReportTest(_IngestCase):
    """Списки факторов подменяются: сам `data_checks` тянет pandas."""

    def test_missing_factor_is_named_and_marks_the_gate_incomplete(self) -> None:
        with mock.patch.object(qi, "_engine_universe",
                               return_value=(_FACTORS, _BENCH)):
            report = qi.universe_report(publisher=self.publisher)
        self.assertTrue(report.ok)
        self.assertEqual(report.factors_ok, 2)
        self.assertEqual(report.benchmarks_ok, 0)      # MSFT.US в базе нет
        text = qi.format_universe(report)
        self.assertIn("MSFT.US", text)
        self.assertIn("НЕТ В БАЗЕ", text)

    def test_complete_coverage_is_reported_as_such(self) -> None:
        with mock.patch.object(qi, "_engine_universe",
                               return_value=(_FACTORS, ())):
            report = qi.universe_report(publisher=self.publisher)
        self.assertTrue(report.complete)
        self.assertIn("✅", qi.format_universe(report))

    def test_depth_is_reported_not_just_presence(self) -> None:
        """Наличие бары не равно пригодности: обрывочная история схлопывает окно.

        Одна молодая бумага обнуляет окно регрессии ВСЕЙ панели (F-15/F-21),
        поэтому в отчёте число баров, а не галочка.
        """
        with mock.patch.object(qi, "_engine_universe",
                               return_value=(_FACTORS, ())):
            report = qi.universe_report(publisher=self.publisher)
        self.assertEqual({p.bars for p in report.factors}, {3})
        self.assertIn("3 баров", qi.format_universe(report))

    def test_a_factor_that_stopped_updating_loses_the_green_tick(self) -> None:
        """🔴 `§−113`. Именно эта команда вскрыла дефект — и сама же печатала
        «10/10 ✅» над тринадцатью строками с датой позавчера.

        Свежесть меряется последним днём РЫНКА В БАЗЕ, а не часами: иначе
        выходные и праздники давали бы тревогу на здоровой базе.
        """
        qi.apply_daily(self.daily(20260813, ["AAPL.US"]), actor="1",
                       publisher=self.publisher)
        with mock.patch.object(qi, "_engine_universe",
                               return_value=(_FACTORS, ())):
            report = qi.universe_report(publisher=self.publisher)
        stale = {p.ticker for p in report.factors if p.stale}
        self.assertEqual(stale, {"SPY.US"})             # AAPL.US обновился
        self.assertTrue(report.complete)                # состав полон
        self.assertFalse(report.usable)                 # допуск НЕ пройден
        text = qi.format_universe(report)
        self.assertIn("ПРОТУХ", text)
        self.assertIn("факторы 2/2 🔴", text)

    def test_a_fresh_universe_keeps_the_green_tick(self) -> None:
        """Обратная мутация: строгость не должна кричать на здоровой базе."""
        qi.apply_daily(self.daily(20260813), actor="1", publisher=self.publisher)
        with mock.patch.object(qi, "_engine_universe",
                               return_value=(_FACTORS, ())):
            report = qi.universe_report(publisher=self.publisher)
        self.assertTrue(report.usable)
        self.assertNotIn("ПРОТУХ", qi.format_universe(report))

    def test_without_pandas_it_refuses_instead_of_copying_the_list(self) -> None:
        """🔴 Вторая копия списка факторов однажды разошлась бы с первой молча."""
        with mock.patch.object(qi, "_engine_universe",
                               side_effect=ImportError("pandas")):
            report = qi.universe_report(publisher=self.publisher)
        self.assertFalse(report.ok)
        self.assertIn("data_checks", report.reason)


class CheckTickerTest(_IngestCase):

    def test_known_ticker_reports_its_source_form(self) -> None:
        probe = qi.check_ticker("SPY.US", publisher=self.publisher)
        self.assertTrue(probe.found)
        self.assertEqual(probe.source_symbol, "SPY.US")
        self.assertEqual(probe.bars, 3)
        self.assertFalse(probe.substituted)

    def test_a_paper_behind_its_market_is_told_so(self) -> None:
        """«Последний бар 20260824» само по себе не говорит, отстала бумага или
        нет: ответ даёт только сравнение с последним днём её рынка."""
        qi.apply_daily(self.daily(20260813, ["AAPL.US"]), actor="1",
                       publisher=self.publisher)
        probe = qi.check_ticker("SPY.US", publisher=self.publisher)
        self.assertTrue(probe.stale)
        self.assertEqual((probe.last_bar, probe.market_latest),
                         (20260812, 20260813))
        self.assertIn("ОТСТАЛА", qi.format_probe(probe))

    def test_a_paper_level_with_its_market_is_not_alarmed_about(self) -> None:
        probe = qi.check_ticker("SPY.US", publisher=self.publisher)
        self.assertFalse(probe.stale)
        self.assertNotIn("ОТСТАЛА", qi.format_probe(probe))

    def test_unknown_ticker_says_so_and_suggests_the_fix(self) -> None:
        probe = qi.check_ticker("NOSUCH.US", publisher=self.publisher)
        self.assertFalse(probe.found)
        self.assertIn("файл истории", qi.format_probe(probe))

    def test_venue_substitution_is_surfaced_not_hidden(self) -> None:
        """🔴 Подмена площадки меняет цену И валюту — она обязана быть видимой.

        Прокси меняет бумагу ради факторной модели, оставляя цену настоящей;
        смена площадки подменяет саму цену. `KSPI.KZ`, «найденный» как ADR на
        Nasdaq, — другая бумага (ловушка №6).
        """
        conn = si.connect(self.db)
        iid = conn.execute(
            "SELECT id FROM instruments WHERE source_symbol='SPY.US'"
        ).fetchone()["id"]
        conn.execute("INSERT INTO symbol_map(engine_ticker, instrument_id, "
                     "match_kind, note) VALUES (?,?,?,?)",
                     ("KAP.IL", iid, "venue_substitution", "тест"))
        conn.commit()
        conn.close()
        probe = qi.check_ticker("KAP.IL", publisher=self.publisher)
        self.assertTrue(probe.substituted)
        self.assertIn("ПОДМЕНА ПЛОЩАДКИ", qi.format_probe(probe))


class PruneTest(_IngestCase):

    def _add_thin(self) -> None:
        thin = self.root / "thin.us.txt"
        thin.write_text(_history_text("THIN.US", (20260812,)), encoding="utf-8")
        conn = si.connect(self.db)
        si.apply_batch(conn, si.parse_history_file(thin, window_days=9000),
                       kind="bootstrap", allow_new=True)
        conn.close()

    def test_without_the_core_it_refuses_instead_of_deleting(self) -> None:
        """🔴 Пустой `keep` — это «не знаем, что защищать», а не «нечего защищать».

        Чистка без ядра срезала бы факторные ETF, а в профиле STRICT потеря
        ЛЮБОГО фактора даёт BLOCK — ручной тир перестал бы отдавать отчёты.
        """
        with mock.patch.object(qi, "_working_set",
                               side_effect=ImportError("pandas")):
            outcome = qi.prune(dry_run=False, actor="1",
                               publisher=self.publisher)
        self.assertFalse(outcome.ok)
        self.assertIn("ядро", outcome.reason)

    def test_dry_run_shows_the_victims_and_touches_nothing(self) -> None:
        self._add_thin()
        before = _bars(self.db)
        generation = self.publisher._generation()
        with mock.patch.object(qi, "_working_set", return_value=list(self.universe)):
            outcome = qi.prune(dry_run=True, actor="1", publisher=self.publisher)
        self.assertTrue(outcome.ok)
        self.assertIn("THIN.US", outcome.removed)
        self.assertFalse(outcome.published)
        self.assertEqual(_bars(self.db), before)
        self.assertEqual(self.publisher._generation(), generation)
        self.assertIn("НЕ изменена", qi.format_prune(outcome))

    def test_real_prune_publishes_even_though_the_base_shrinks(self) -> None:
        """🔴 Единственная операция, где гард обвала размера НЕ применяется.

        `prune_thin_instruments` завершается `VACUUM`; сжатие файла — это её
        работа, а не признак поломки. Гард существует для дельты, которая
        умеет только добавлять.
        """
        self._add_thin()
        with mock.patch.object(qi, "_working_set", return_value=list(self.universe)):
            outcome = qi.prune(dry_run=False, actor="1", publisher=self.publisher)
        self.assertTrue(outcome.ok, outcome.reason)
        self.assertTrue(outcome.published)
        conn = si.connect(self.db, read_only=True)
        try:
            left = {r["source_symbol"] for r in conn.execute(
                "SELECT source_symbol FROM instruments")}
        finally:
            conn.close()
        self.assertNotIn("THIN.US", left)
        self.assertEqual(left, set(self.universe))

    def test_core_survives_even_below_the_threshold(self) -> None:
        """Фактор с обрезанной историей чинят ре-бутстрапом, а не удалением."""
        thin_core = self.root / "spycut.txt"
        thin_core.write_text(_history_text("SPY.US", (20260812,)), encoding="utf-8")
        conn = si.connect(self.db)
        conn.execute("DELETE FROM daily_bars WHERE instrument_id IN "
                     "(SELECT id FROM instruments WHERE source_symbol='SPY.US') "
                     "AND trade_date < 20260812")
        conn.commit()
        conn.close()
        with mock.patch.object(qi, "_working_set", return_value=["SPY.US"]):
            outcome = qi.prune(dry_run=True, actor="1", publisher=self.publisher)
        self.assertNotIn("SPY.US", outcome.removed)


# ═════════════════════════════════════════════════════════════════════════════
# IB-7 — напоминания
# ═════════════════════════════════════════════════════════════════════════════

def _state(**kw):
    base = dict(ok=True, storage="тест")
    base.update(kw)
    return qi.StoreStatus(**base)


class MissingAnswerTest(_IngestCase):
    """🔴 «Пустая коллекция ≠ всё хорошо» — правило проекта (`§−90` A-3).

    Список пропавших дат пуст и когда всё в порядке, и когда базу прочитать не
    удалось. Первая редакция печатала на оба случая зелёное «всё на месте»:
    самый громкий отказ выглядел как самый спокойный ответ.
    """

    def test_unavailable_store_is_not_reported_as_all_clear(self) -> None:
        state = qi.status(publisher=LocalQuotePublisher(self.root / "no"))
        text = qi.format_missing(state)
        self.assertIn("не могу сказать", text)
        self.assertNotIn("✅", text)

    def test_healthy_store_with_nothing_missing_says_so(self) -> None:
        text = qi.format_missing(qi.status(publisher=self.publisher))
        self.assertIn("✅", text)

    def test_the_last_operation_is_described_by_its_own_kind(self) -> None:
        """🔴 `/status` печатал «prune (None баров)»: шаблон один на все
        операции, а чистка баров не пишет — у неё ключ `removed`.

        `None`, выведенный как число, — это артефакт шаблона вместо факта, тот
        же класс, что «пустая коллекция ≠ всё хорошо» (`§−90` A-3).
        """
        prune_run = {"file": "prune", "kind": "prune", "removed": 11281}
        text = qi.format_status(_state(last_run=prune_run))
        self.assertIn("prune (удалено бумаг: 11281)", text)
        self.assertNotIn("None баров", text)          # ровно то, что печаталось

        apply_run = {"file": "20260831_d.txt", "kind": "daily",
                     "rows_written": 800}
        self.assertIn("20260831_d.txt (800 баров)",
                      qi.format_status(_state(last_run=apply_run)))

    def test_missing_days_are_listed_as_files_to_resend(self) -> None:
        qi.apply_daily(self.daily(20260813), actor="1", publisher=self.publisher)
        conn = si.connect(self.db)
        conn.execute("DELETE FROM daily_bars WHERE trade_date=20260813")
        conn.commit()
        conn.close()
        text = qi.format_missing(qi.status(publisher=self.publisher))
        self.assertIn("20260813_d.txt", text)


class ReminderTest(unittest.TestCase):

    def test_healthy_base_produces_SILENCE(self) -> None:
        """🔴 Молчание — штатный исход, и это главное свойство напоминаний.

        Сообщение, приходящее каждый день независимо от состояния, перестают
        читать за неделю — и тогда оно не сработает в тот единственный раз,
        когда было нужно.
        """
        state = _state(markets=(qi.MarketState("US", 20260819, 5, 50,
                                               stale_days=1, days_left=6),))
        self.assertIsNone(qi.build_reminder(state))

    def test_approaching_block_warns_with_the_number_of_days(self) -> None:
        state = _state(markets=(qi.MarketState("US", 20260814, 5, 50,
                                               stale_days=6, days_left=1),))
        text = qi.build_reminder(state)
        self.assertIn("1 дн.", text)

    def test_already_blocked_is_reported_as_such(self) -> None:
        state = _state(markets=(qi.MarketState("US", 20260810, 5, 50,
                                               stale_days=10, days_left=-3),))
        self.assertIn("УЖЕ заблокирован", qi.build_reminder(state))

    def test_missing_days_outrank_the_countdown(self) -> None:
        """Пропавшие дни — про потерю данных, обратный отсчёт — про свежесть."""
        state = _state(missing_dates=(20260813, 20260814),
                       markets=(qi.MarketState("US", 20260819, 5, 50,
                                               stale_days=1, days_left=6),))
        text = qi.build_reminder(state)
        self.assertIn("20260813", text)
        self.assertIn("перезаливали", text)

    def test_unavailable_store_is_an_alarm(self) -> None:
        self.assertIn("недоступна",
                      qi.build_reminder(_state(ok=False, reason="нет базы")))

    def test_threshold_comes_from_the_provider_not_from_a_literal(self) -> None:
        """Свой порог означал бы: бот обещает один срок, отчёт блокирует по другому."""
        self.assertLessEqual(qi.REMIND_DAYS_LEFT, 7)


class TaskEndpointTest(unittest.TestCase):
    """Плановая проверка (IB-7) закрыта дважды; здесь — внутренний рубеж."""

    def setUp(self) -> None:
        import ingest_entrypoint                        # noqa: PLC0415
        self.ie = ingest_entrypoint
        self.calls: list[int] = []

    async def _task(self):
        self.calls.append(1)
        return "тихо"

    def _run(self, method, headers):
        import asyncio                                  # noqa: PLC0415
        return asyncio.run(self.ie._handle_task(method, headers, self._task))

    def test_request_line_and_headers_are_parsed(self) -> None:
        method, path, headers = self.ie._parse(
            b"POST /tasks/check?x=1 HTTP/1.1\r\nX-Ingest-Task-Token: s\r\n\r\n")
        self.assertEqual((method, path.split("?")[0]), ("POST", "/tasks/check"))
        self.assertEqual(headers["x-ingest-task-token"], "s")

    def test_without_a_secret_the_route_is_OFF(self) -> None:
        """🔴 Незакрытая ручка шлёт сообщения владельцу — это канал для шума
        в единственном канале связи оператора."""
        with mock.patch.dict(os.environ, {"INGEST_TASK_TOKEN": ""}), \
                self.assertLogs("ombri.ingest.entrypoint", level="WARNING") as log:
            reply = self._run("POST", {"x-ingest-task-token": "s"})
        self.assertIn(b"503", reply.split(b"\r\n")[0])
        self.assertIn("отключён", "\n".join(log.output))
        self.assertEqual(self.calls, [])

    def test_wrong_secret_is_refused(self) -> None:
        with mock.patch.dict(os.environ, {"INGEST_TASK_TOKEN": "right"}), \
                self.assertLogs("ombri.ingest.entrypoint", level="WARNING") as log:
            reply = self._run("POST", {"x-ingest-task-token": "wrong"})
        self.assertIn(b"401", reply.split(b"\r\n")[0])
        self.assertIn("неверным секретом", "\n".join(log.output))
        self.assertEqual(self.calls, [])

    def test_get_is_refused_so_a_crawler_cannot_fire_it(self) -> None:
        with mock.patch.dict(os.environ, {"INGEST_TASK_TOKEN": "right"}):
            reply = self._run("GET", {"x-ingest-task-token": "right"})
        self.assertIn(b"405", reply.split(b"\r\n")[0])
        self.assertEqual(self.calls, [])

    def test_correct_secret_runs_the_check(self) -> None:
        with mock.patch.dict(os.environ, {"INGEST_TASK_TOKEN": "right"}):
            reply = self._run("POST", {"x-ingest-task-token": "right"})
        self.assertIn(b"200", reply.split(b"\r\n")[0])
        self.assertEqual(self.calls, [1])


# ═════════════════════════════════════════════════════════════════════════════
# IB-8 — деплой
# ═════════════════════════════════════════════════════════════════════════════

class DeployStepTest(unittest.TestCase):
    """Шаг деплоя загрузчика не имеет права трогать главного бота."""

    def setUp(self) -> None:
        try:
            import yaml                                 # noqa: PLC0415
        except ImportError:
            self.skipTest("PyYAML не установлен")
        path = Path(__file__).resolve().parents[1] / "cloudbuild.yaml"
        # 🔴 `§−99`. `cloudbuild.yaml` лежит в КОРНЕ РЕПО и в образ не копируется
        # (`Dockerfile` берёт только `src/`, `tests/`, `SYSTEM_PROMPT.md` и
        # requirements). Деплой-гейт гоняет `unittest discover` ВНУТРИ образа —
        # без этой проверки пять тестов падали там `FileNotFoundError`, сборка
        # краснела и НЕ ДЕПЛОИЛСЯ НИКТО, включая главный бот. Ровно тот случай,
        # ради которого правило «читаешь файл вне образа → skipTest» и написано;
        # `test_phase16_sprint2` держит ту же охрану на том же файле.
        if not path.is_file():
            self.skipTest("cloudbuild.yaml не попадает в образ (деплой-гейт); "
                          "репо-локальный прогон проверит его целиком")
        self.doc = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_ingest_step_is_a_noop_until_the_owner_enables_it(self) -> None:
        self.assertEqual(self.doc["substitutions"]["_INGEST_SERVICE"], "")

    def test_main_bot_env_is_untouched(self) -> None:
        """🔴 `--set-env-vars` заменяет ВЕСЬ набор переменных.

        Эта грабля уже однажды выключила Premium V2 в проде: выставленный
        руками флаг стирался следующим деплоем, и прод месяцами отдавал старый
        шаблон отчёта.
        """
        deploy = next(s for s in self.doc["steps"] if s["id"] == "deploy")
        env = next(a for a in deploy["args"] if a.startswith("--set-env-vars"))
        for expected in ("PREMIUM_REPORT_ENABLED=true",
                         "TOKENOMICS_DB_PATH=/mnt/state/tokenomics.db",
                         "STOOQ_DB_PATH=/mnt/state/stooq/prices.sqlite"):
            self.assertIn(expected, env)

    def test_loader_refuses_to_deploy_without_its_own_service_account(self) -> None:
        """🔴 Без своего SA всё ограничение прав — слова.

        `--service-account` в шаге не задавался, и Cloud Run брал дефолтный
        Compute-SA, у которого уже есть objectAdmin на бакете с `tokenomics.db`
        и `users_vault.db`. То есть утверждение «у загрузчика нет прав на
        балансы» было ЛОЖНЫМ как реализовано — ни отдельный бакет, ни условие
        IAM этого не меняли: сервис бежал с полным доступом.
        """
        step = next(s for s in self.doc["steps"] if s["id"] == "deploy-ingest-bot")
        body = "\n".join(step["args"])
        self.assertIn("--service-account=", body)
        self.assertIn("_INGEST_SA", body)
        self.assertIn("ОТКАЗ", body, "пустой SA обязан останавливать деплой")
        self.assertEqual(self.doc["substitutions"]["_INGEST_SA"], "",
                         "SA заводится осознанно, а не достаётся по умолчанию")

    def test_loader_has_room_for_the_base_in_tmpfs(self) -> None:
        """🔴 `§−116`. На Cloud Run `/tmp` — это ОПЕРАТИВНАЯ память.

        Цикл загрузчика держит там копию базы (сейчас ~55 МБ и растёт), а
        выгрузка обратно буферизует её ещё раз; `/prune` вдобавок пишет
        `VACUUM`-копию. Вместе с numpy/pandas это ~300 МБ на дельте и больше
        на чистке. При 512Mi запас исчезает МОЛЧА: контейнер убивают, ответ в
        чат не уходит, и симптом неотличим от зависания.

        Пин нужен, потому что `gcloud run deploy` задаёт память КАЖДЫЙ раз:
        поднятая руками в консоли, она вернулась бы к значению из файла
        следующим деплоем — та же грабля, что стирала env главного бота.
        """
        step = next(s for s in self.doc["steps"] if s["id"] == "deploy-ingest-bot")
        body = "\n".join(step["args"])
        self.assertIn("--memory=1Gi", body)
        self.assertNotIn("--memory=512Mi", body)

    def test_loader_token_secret_is_a_substitution_named_by_the_owner(self) -> None:
        """Имя секрета — подстановка `_INGEST_BOT_TOKEN_SECRET` (`§−100`).

        Имя секрета живёт НЕ в репозитории, а в Secret Manager: переименовали
        там — поменяли одну подстановку, не трогая шаг деплоя. Литерал в
        `--set-secrets` означал бы правку файла на каждый переезд, а этот шаг
        стоит рядом с деплоем главного бота.
        """
        self.assertEqual(
            self.doc["substitutions"]["_INGEST_BOT_TOKEN_SECRET"],
            "OMBRI_INGEST_BOT_TOKEN")
        step = next(s for s in self.doc["steps"] if s["id"] == "deploy-ingest-bot")
        script = "\n".join(step["args"])
        self.assertIn(
            "OMBRI_INGEST_BOT_TOKEN=${_INGEST_BOT_TOKEN_SECRET}:latest", script,
            "слева от `=` — имя переменной В КОНТЕЙНЕРЕ, справа — имя СЕКРЕТА; "
            "их путают, и тогда бот читает пустоту при живом секрете")

    def test_the_rename_did_not_touch_the_other_secrets(self) -> None:
        """🔴 Прочие секреты главного бота переездами не затронуты.

        `--set-secrets` заменяет ВЕСЬ набор привязок: опечатка здесь означает
        не «не переименовали», а «сервис остался без ключа». Токен вынесен в
        подстановку (`§−101`, свой тест ниже), остальные три остаются
        литералами и обязаны остаться на месте.
        """
        deploy = next(s for s in self.doc["steps"] if s["id"] == "deploy")
        secrets = next(a for a in deploy["args"] if a.startswith("--set-secrets"))
        for expected in ("FINTECH_MASTER_KEY=FINTECH_MASTER_KEY:latest",
                         "ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest",
                         "FREEDOM_API_KEY=FREEDOM_API_KEY:latest"):
            self.assertIn(expected, secrets)

    def test_writer_and_reader_address_the_same_object_name(self) -> None:
        """Префикс писателя и путь читателя обязаны совпасть (`§−100`).

        🔴 Владелец свёл оба бота в ОДИН бакет (`_QUOTES_BUCKET=ramp-bot-state`)
        с условием IAM на префикс `stooq/`, и этим ЗАМКНУЛ кольцо: загрузчик
        публикует `gs://ramp-bot-state/stooq/prices.sqlite`, а отчётный бот
        читает `/mnt/state/stooq/prices.sqlite` — тот же объект через gcsfuse.

        Держится это равенство на трёх независимо редактируемых строках:
        `QUOTES_PREFIX` у загрузчика, `STOOQ_DB_PATH` у отчётного бота и
        `DB_OBJECT_NAME` в коде. Разъехались — кольцо размыкается МОЛЧА: оба
        бота живы, у каждого «свой» файл, и оператор неделю шлёт срезы в
        пустоту, пока отчёт не заблокируется по свежести. Поэтому равенство —
        тест, а не договорённость.

        Пинится ИМЯ ОБЪЕКТА, а не бакет: бакет — решение владельца (общий с
        условием IAM либо отдельный), и тест его не навязывает.
        """
        from services.quote_publisher import DB_OBJECT_NAME  # noqa: PLC0415

        ingest = next(s for s in self.doc["steps"]
                      if s["id"] == "deploy-ingest-bot")
        body = "\n".join(ingest["args"])
        env = dict(kv.split("=", 1) for kv in
                   re.search(r"--set-env-vars=(\S+)", body).group(1).split(","))
        written = env["QUOTES_PREFIX"] + DB_OBJECT_NAME

        deploy = next(s for s in self.doc["steps"] if s["id"] == "deploy")
        mount = next(a for a in deploy["args"]
                     if a.startswith("--add-volume-mount=")).split("mount-path=")[1]
        main_env = dict(
            kv.split("=", 1) for kv in
            next(a for a in deploy["args"]
                 if a.startswith("--set-env-vars=")).split("=", 1)[1].split(","))
        read = main_env["STOOQ_DB_PATH"][len(mount):].lstrip("/")

        self.assertEqual(
            written, read,
            f"загрузчик пишет объект {written!r}, а отчётный бот читает "
            f"{read!r}. В общем бакете это РАЗНЫЕ файлы, и кольцо разомкнуто")

    def test_loader_gets_its_own_bucket_not_the_state_one(self) -> None:
        """Радиус поражения писателя не должен включать балансы и ключи брокера.

        🔴 Пинится ДЕФОЛТ репозитория и отсутствие тома состояния — не то, что
        реально приедет в прод: `_QUOTES_BUCKET` можно переопределить в
        Build Trigger. Вариант «тот же бакет + условие IAM на префикс `stooq/`»
        делает это осознанно, и компенсирующий контроль там — условие, а не
        эта проверка. Тест говорит: по умолчанию репозиторий предлагает
        раздельные бакеты.
        """
        self.assertNotEqual(self.doc["substitutions"]["_QUOTES_BUCKET"],
                            self.doc["substitutions"]["_STATE_BUCKET"])
        body = self._step_body()
        grant = self._granting_part(body)
        # 🔴 Проверяются ОБА написания. Мутация показала, что проверка одного
        # лишь ЗНАЧЕНИЯ пропускает ссылку на подстановку `${_STATE_BUCKET}` —
        # то есть ровно ту форму, в которой её и написали бы в YAML.
        #
        # 🔴 §−103: проверка сузилась со ВСЕГО шага до его ВЫДАЮЩЕЙ части —
        # вызова `gcloud run deploy`. Прежняя редакция падала на ДИАГНОСТИКЕ:
        # шаг печатает, замкнуто ли кольцо, и для этого обязан НАЗВАТЬ оба
        # бакета. Гейт, срабатывающий на объяснении, учит удалять объяснения
        # (`§−97` E-9) — здесь он потребовал бы убрать ровно ту проверку, что
        # ловит разомкнутое кольцо. Радиус поражения задаётся флагами
        # `--set-env-vars` / `--add-volume` / `--service-account`, и они все
        # внутри `grant`; `echo` не выдаёт ничего.
        self.assertNotIn(self.doc["substitutions"]["_STATE_BUCKET"], grant)
        self.assertNotIn("_STATE_BUCKET", grant)
        self.assertIn("ingest_entrypoint.py", body)

    def test_loader_does_not_mount_the_state_volume(self) -> None:
        body = self._step_body()
        grant = self._granting_part(body)
        self.assertNotIn("--add-volume", body)      # тома нет НИГДЕ в шаге
        self.assertNotIn("/mnt/state", grant)

    def test_ring_is_reported_at_deploy_time(self) -> None:
        """§−103 · деплой обязан СКАЗАТЬ, замкнуто ли кольцо.

        Полный путь объекта состоит из трёх частей — бакет, префикс, имя, — а
        `test_object_name_matches_what_the_report_bot_reads` пинит только две
        последние: бакет сознательно оставлен решением владельца. Из-за этого
        дефолт репозитория (`_QUOTES_BUCKET` ≠ `_STATE_BUCKET`) даёт РАЗОМКНУТОЕ
        кольцо, и ни один гейт этого не ловит. Цена молчания измерена в `§−100`:
        срезы применяются, оба бота живы, а отчёт читает другой файл.

        Раз доказать равенство нельзя, деплой обязан хотя бы ПРОИЗНЕСТИ вердикт
        там, где конфигурация становится живой.
        """
        body = self._step_body()
        self.assertIn("_QUOTES_BUCKET", body)
        self.assertIn("_STATE_BUCKET", body)
        self.assertIn("КОЛЬЦО ЗАМКНУТО", body)
        self.assertIn("КОЛЬЦО РАЗОМКНУТО", body)
        # Вердикт обязан быть УСЛОВНЫМ, а не печататься всегда одинаково.
        self.assertRegex(body, r'if\s+\[\s+"\$\{_QUOTES_BUCKET\}"\s*=\s*"\$\{_STATE_BUCKET\}"\s+\]')

    def _step_body(self) -> str:
        step = next(s for s in self.doc["steps"] if s["id"] == "deploy-ingest-bot")
        return "\n".join(step["args"])

    @staticmethod
    def _granting_part(body: str) -> str:
        """Часть шага, которая РЕАЛЬНО что-то выдаёт: вызов `gcloud run deploy`.

        Всё до него — проверки и печать; выдать доступ `echo` не может.
        """
        idx = body.find("gcloud run deploy")
        assert idx >= 0, "в шаге нет вызова `gcloud run deploy`"
        return body[idx:]



# ═════════════════════════════════════════════════════════════════════════════
# «Базы нет» ≠ «бумаги нет»
# ═════════════════════════════════════════════════════════════════════════════

class StoreSilenceIsNotAnAnswerTest(_IngestCase):
    """🔴 Инвариант не новый: он записан в `manual_portfolio.PreflightCoverage`.

    «В базе этой бумаги нет» и «базы нет» — разные утверждения, и второе не
    даёт права говорить про бумагу ничего. Первая редакция `check_ticker`
    сливала их в `found=False`, и ответ советовал «пришлите файл истории» —
    совет, который применять некуда, когда базы нет вовсе.
    """

    def test_absent_store_is_reported_as_unanswered(self) -> None:
        probe = qi.check_ticker("SPY.US",
                                publisher=LocalQuotePublisher(self.root / "no"))
        self.assertFalse(probe.answered)
        self.assertFalse(probe.found)

    def test_absent_store_does_not_advise_sending_a_history_file(self) -> None:
        probe = qi.check_ticker("SPY.US",
                                publisher=LocalQuotePublisher(self.root / "no"))
        text = qi.format_probe(probe)
        self.assertIn("хранилище недоступно", text)
        self.assertNotIn("Пришлите файл истории", text)

    def test_present_store_answers_and_the_two_cases_differ(self) -> None:
        known = qi.check_ticker("SPY.US", publisher=self.publisher)
        unknown = qi.check_ticker("NOSUCH.US", publisher=self.publisher)
        self.assertTrue(known.answered and known.found)
        self.assertTrue(unknown.answered)
        self.assertFalse(unknown.found)
        self.assertIn("Пришлите файл истории", qi.format_probe(unknown))


class PruneDoesNotFabricateARollbackTest(_IngestCase):
    """Чистка не вправе оставить в журнале дату, которую сама и удалила.

    Иначе следующая проверка пошлёт оператору на телефон «база откатилась» и
    отправит искать файлы, которых бот лишил базу по его же команде. Ложная
    тревога дороже пропущенной: после неё перестают верить настоящей.
    """

    def test_dates_removed_by_prune_leave_the_cursor(self) -> None:
        thin = self.root / "thin.us.txt"
        thin.write_text(_history_text("THIN.US", (20260901,)), encoding="utf-8")
        conn = si.connect(self.db)
        si.apply_batch(conn, si.parse_history_file(thin, window_days=99999,
                                                   today=date(2026, 9, 2)),
                       kind="bootstrap", allow_new=True)
        conn.close()
        self.publisher.write_cursor(
            Cursor().with_applied(20260901, generation=1))

        with mock.patch.object(qi, "_working_set", return_value=list(self.universe)):
            outcome = qi.prune(dry_run=False, actor="1", publisher=self.publisher)
        self.assertTrue(outcome.ok, outcome.reason)
        self.assertIn("THIN.US", outcome.removed)
        self.assertEqual(self.publisher.read_cursor().applied_dates, ())
        self.assertEqual(qi.status(publisher=self.publisher).missing_dates, ())

    def test_a_real_rollback_still_survives_a_prune(self) -> None:
        """Пропавшее ДО чистки — чужой откат, и он обязан остаться тревогой.

        🔴 Первая редакция этого теста была ПУСТОЙ, и вскрыла её мутация.
        Чистить было нечего (все бумаги проходили порог), `prune` выходил
        раньше записи курсора — то есть тест проверял, что нетронутый журнал
        остался нетронутым. Здесь база СОДЕРЖИТ обрывочную бумагу, поэтому
        путь записи курсора исполняется по-настоящему.
        """
        thin = self.root / "thin.us.txt"
        thin.write_text(_history_text("THIN.US", (20260812,)), encoding="utf-8")
        conn = si.connect(self.db)
        si.apply_batch(conn, si.parse_history_file(thin, window_days=9000),
                       kind="bootstrap", allow_new=True)
        conn.close()
        self.publisher.write_cursor(
            Cursor().with_applied(20250101, generation=1))   # в базе такого нет

        with mock.patch.object(qi, "_working_set", return_value=list(self.universe)):
            outcome = qi.prune(dry_run=False, actor="1", publisher=self.publisher)
        self.assertTrue(outcome.published, "чистка не дошла до записи курсора — "
                                           "тест снова стал бы пустым")
        self.assertIn(20250101, self.publisher.read_cursor().applied_dates)


# ═════════════════════════════════════════════════════════════════════════════
# IB-4 — ядро бота: приём документа целиком
# ═════════════════════════════════════════════════════════════════════════════

class _Doc:
    def __init__(self, name, size):
        self.file_name = name
        self.file_size = size


class _Sent:
    def __init__(self):
        self.deleted = False

    async def delete(self):
        self.deleted = True


class _FakeBot:
    """Телеграм, которого нет: пишет заготовленный файл вместо `getFile`."""

    def __init__(self, payload: str):
        self.payload = payload
        self.downloads: list = []

    async def download(self, document, destination):
        self.downloads.append(Path(destination))
        Path(destination).write_text(self.payload, encoding="utf-8")


class _FakeMessage:
    def __init__(self, bot, document=None, text=None, user_id=1):
        self.bot = bot
        self.document = document
        self.text = text
        self.from_user = type("U", (), {"id": user_id})()
        self.answers: list = []

    async def answer(self, text, **kwargs):
        self.answers.append((text, kwargs))
        return _Sent()

    async def edit_reply_markup(self, **_kw):
        return None


class RejectionAnomalyTest(_IngestCase):
    """§−107 · отказ, которого РАНЬШЕ НЕ БЫЛО, обязан быть назван тревогой.

    🔴 Живой разбор 25.08. Срез за 24.08 дал «чужая дата» 166 и «нет цены
    закрытия» 43, а в ЧЕТЫРЁХ предыдущих прогонах таких строк не было НИ ОДНОЙ;
    «индекс/FX» вырос с ~57 до 154; записался 631 бар вместо привычных 800.
    Сводка всё это честно печатала — и всё равно читалась как успех: `✅
    применён`. Чтобы понять, что копия файла битая, оператору пришлось вручную
    поднять `ingest_runs` и сравнить пять прогонов глазами.

    Сравнивать было С ЧЕМ: история лежит в той же базе. Факт был, вывода из
    него никто не делал — тот же класс, что «пустая коллекция ≠ всё хорошо»
    (`§−90` A-3).
    """

    _HDR = ("<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,"
            "<OPENINT>")

    def _store(self):
        from services.quote_publisher import LocalQuotePublisher  # noqa: PLC0415
        root = self.root / "anom"
        root.mkdir(parents=True, exist_ok=True)
        db = root / "prices.sqlite"
        conn = si.connect(db)
        si.ensure_schema(conn)
        cache: dict = {}
        for t in self._tickers():
            iid = si._instrument_id(conn, t, cache, allow_new=True)
            conn.execute(
                "INSERT INTO daily_bars(instrument_id,trade_date,open,high,"
                "low,close,volume) VALUES(?,?,?,?,?,?,?)",
                (iid, 20260817, 10.0, 10.1, 9.9, 10.0, 1000.0))
        si._bump_generation(conn)
        conn.commit()
        conn.close()
        return root, LocalQuotePublisher(root)

    @staticmethod
    def _tickers():
        return [f"T{i:03d}.US" for i in range(40)]

    def _file(self, root: Path, day: int, *, stale: int = 0,
              noclose: int = 0, indices: int = 5) -> str:
        rows = []
        for i, t in enumerate(self._tickers()):
            d = day - 1 if i < stale else day
            close = "" if stale <= i < stale + noclose else "10.05"
            rows.append(f"{t},D,{d},000000,10,10.1,9.9,{close},1000,0")
        rows += [f"^IDX{i},D,{day},000000,1,1,1,1,0,0" for i in range(indices)]
        path = root / f"{day}_d.txt"
        path.write_text("\r\n".join([self._HDR, *rows]) + "\r\n",
                        encoding="utf-8")
        return str(path)

    def _run(self, pub, root, day, **kw):
        with mock.patch.object(si, "MIN_US_ROWS", 0):
            return qi.apply_daily(self._file(root, day, **kw), actor="op",
                                  publisher=pub)

    def test_new_rejection_category_raises_a_warning(self):
        root, pub = self._store()
        for day in (20260818, 20260819, 20260820, 20260821):
            out = self._run(pub, root, day)
            self.assertEqual(out.warnings, (), f"{day}: тревога на здоровом файле")
        out = self._run(pub, root, 20260824, stale=12, noclose=4)
        text = "\n".join(out.warnings)
        self.assertIn("чужая дата", text)
        self.assertIn("не было НИ ОДНОЙ", text)
        self.assertIn("перекачайте", text,
                      "тревога обязана говорить, ЧТО делать, а не только что не так")

    def test_steady_rejections_are_not_flagged(self):
        """Индексы отбраковываются КАЖДЫЙ день — это норма, а не тревога."""
        root, pub = self._store()
        for day in (20260818, 20260819, 20260820, 20260821, 20260824):
            out = self._run(pub, root, day, indices=5)
            self.assertEqual(out.warnings, (),
                             f"{day}: постоянный отказ принят за аномалию")

    def test_spike_against_history_is_flagged(self):
        root, pub = self._store()
        for day in (20260818, 20260819, 20260820, 20260821):
            self._run(pub, root, day, indices=5)
        out = self._run(pub, root, 20260824, indices=30)      # 6× от нормы
        self.assertTrue(any("индекс/FX" in w for w in out.warnings),
                        f"всплеск не назван: {out.warnings}")

    def test_history_is_the_RECENT_runs_not_the_oldest(self):
        """«Норма» — это ПОСЛЕДНИЕ прогоны, а не первые попавшиеся.

        Разница видна только когда прогонов больше окна: если брать самые
        старые, то давно ушедшая аномалия навсегда останется «нормой» и
        заглушит настоящую. Здесь три старых прогона шумные, пять свежих —
        чистые, и всплеск обязан быть назван по СВЕЖИМ.
        """
        root, pub = self._store()
        day = 20260801
        for _ in range(3):                     # старая «норма»: шумно
            self._run(pub, root, day, indices=30); day += 1
        for _ in range(5):                     # свежая норма: чисто
            self._run(pub, root, day, indices=5); day += 1
        out = self._run(pub, root, day, indices=30)
        self.assertTrue(
            any("индекс/FX" in w for w in out.warnings),
            "всплеск не назван — сверка смотрит на СТАРЫЕ прогоны вместо свежих: "
            f"{out.warnings}")

    def test_silent_when_there_is_no_history_to_compare_with(self):
        """Одному прогону верить не за что — молчание честнее догадки."""
        root, pub = self._store()
        out = self._run(pub, root, 20260818, stale=12)
        self.assertEqual(out.warnings, ())

    def test_unreadable_history_never_blocks_the_upload(self):
        """Сверка полезна, но заливка от неё зависеть не вправе."""
        root, pub = self._store()
        for day in (20260818, 20260819):
            self._run(pub, root, day)
        with mock.patch.object(qi, "_rejection_anomaly",
                               side_effect=AssertionError("сверка не должна ронять")):
            with self.assertRaises(AssertionError):
                self._run(pub, root, 20260820)
        # а с испорченным JSON истории — молча и успешно
        conn = si.connect(root / "prices.sqlite")
        conn.execute("UPDATE ingest_runs SET rejected='не json'")
        conn.commit()
        conn.close()
        out = self._run(pub, root, 20260821, stale=12)
        self.assertTrue(out.ok, out.reason)


class LocalCliAndBotProduceTheSameBaseTest(_IngestCase):
    """§−106 · у базы ДВА писателя, и они обязаны писать ОДНО И ТО ЖЕ.

    🔴 `CLAUDE.md`: «База котировок Stooq: ПИШУТ двое — `stooq_ingest` у
    оператора и бот-загрузчик». Пути в них РАЗНЫЕ: CLI зовёт `apply_inbox`
    (папка, много файлов, сортировка по дате), бот — `parse_daily_file` +
    `apply_batch` (один файл). До этого раунда совпадение результата было
    ДОПУЩЕНИЕМ: `apply_inbox` тестировался, паритет — нет.

    Цена расхождения: содержимое базы начинает зависеть от того, КТО её
    обновлял, и разойтись пути могут молча — оба «успешны», числа разные.

    Что здесь пинится — ДАННЫЕ (`daily_bars`, `instruments`), а не байты
    файла. Байты совпадать НЕ ОБЯЗАНЫ и не должны: `meta.generation` — это
    CAS-токен, а `ingest_runs.started_at` — журнал операции; обе записи по
    смыслу фиксируют МОМЕНТ, и одинаковыми они были бы только если бы CAS
    не работал.
    """

    def _seed(self, target: Path) -> None:
        """База из двух бумаг с одним днём истории — общий старт обоих путей."""
        conn = si.connect(target)
        si.ensure_schema(conn)
        cache: dict = {}
        for sym in ("AAA.US", "BBB.US"):
            iid = si._instrument_id(conn, sym, cache, allow_new=True)
            conn.execute(
                "INSERT INTO daily_bars(instrument_id,trade_date,open,high,"
                "low,close,volume) VALUES(?,?,?,?,?,?,?)",
                (iid, 20260813, 10.0, 10.1, 9.9, 10.0, 1000.0))
        si._bump_generation(conn)
        conn.commit()
        conn.close()

    @staticmethod
    def _digest(db: Path, table: str, cols: str) -> str:
        import hashlib                                  # noqa: PLC0415
        conn = sqlite3.connect(db)
        try:
            h = hashlib.sha256()
            for r in conn.execute(f"SELECT {cols} FROM {table} ORDER BY {cols}"):
                h.update(repr(tuple(r)).encode())
            return h.hexdigest()
        finally:
            conn.close()

    def test_both_writers_produce_identical_data(self) -> None:
        from services.quote_publisher import LocalQuotePublisher  # noqa: PLC0415

        rows = [f"{t},D,20260814,000000,10,10.1,9.9,10.05,1000,0"
                for t in ("AAA.US", "BBB.US", "ZZZ.US")]
        body = "\r\n".join(
            ["<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,"
             "<OPENINT>", *rows]) + "\r\n"

        # ── путь А: локальный CLI ────────────────────────────────────────
        cli_root = self.root / "cli"
        (cli_root / "inbox").mkdir(parents=True)
        cli_db = cli_root / "prices.sqlite"
        self._seed(cli_db)
        (cli_root / "inbox" / "20260814_d.txt").write_text(body, encoding="utf-8")
        conn = si.connect(cli_db)
        cli_result = si.apply_inbox(conn, cli_root / "inbox",
                                    cli_root / "applied", cli_root / "rejected",
                                    min_us_rows=0)
        conn.close()

        # ── путь Б: бот ──────────────────────────────────────────────────
        bot_root = self.root / "bot"
        bot_root.mkdir(parents=True)
        self._seed(bot_root / "prices.sqlite")
        drop = self.root / "20260814_d.txt"
        drop.write_text(body, encoding="utf-8")
        with mock.patch.object(si, "MIN_US_ROWS", 0):
            outcome = qi.apply_daily(str(drop), actor="bot",
                                     publisher=LocalQuotePublisher(bot_root))
        self.assertTrue(outcome.ok, outcome.reason)

        # ── сравнение ────────────────────────────────────────────────────
        self.assertEqual(cli_result.rows_written, outcome.result.rows_written,
                         "писатели записали РАЗНОЕ число баров")
        for table, cols in (
                ("daily_bars",
                 "instrument_id,trade_date,open,high,low,close,volume"),
                ("instruments",
                 "id,source,source_symbol,market,currency,convention")):
            with self.subTest(table=table):
                self.assertEqual(
                    self._digest(cli_db, table, cols),
                    self._digest(bot_root / "prices.sqlite", table, cols),
                    f"{table} разошлась между локальным CLI и ботом — "
                    "содержимое базы стало зависеть от того, КТО её обновлял")

    def test_neither_writer_admits_new_instruments_on_a_daily_slice(self) -> None:
        """Общее ядро обоих путей: `allow_new=False` на дневной дельте (`§−88`).

        Проверяется у ОБОИХ, потому что параметр задаётся вызывающим, и
        разойтись они могут именно здесь.
        """
        from services.quote_publisher import LocalQuotePublisher  # noqa: PLC0415

        body = ("<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,"
                "<VOL>,<OPENINT>\r\n"
                "ZZZ.US,D,20260814,000000,10,10.1,9.9,10.05,1000,0\r\n")
        for label, run in (("cli", "cli"), ("bot", "bot")):
            with self.subTest(writer=label):
                root = self.root / f"new_{label}"
                root.mkdir(parents=True)
                db = root / "prices.sqlite"
                self._seed(db)
                before = sqlite3.connect(db).execute(
                    "SELECT COUNT(*) FROM instruments").fetchone()[0]
                if run == "cli":
                    (root / "inbox").mkdir()
                    (root / "inbox" / "20260814_d.txt").write_text(
                        body, encoding="utf-8")
                    conn = si.connect(db)
                    si.apply_inbox(conn, root / "inbox", None, None,
                                   min_us_rows=0)
                    conn.close()
                else:
                    drop = root / "20260814_d.txt"
                    drop.write_text(body, encoding="utf-8")
                    with mock.patch.object(si, "MIN_US_ROWS", 0):
                        qi.apply_daily(str(drop), actor="bot",
                                       publisher=LocalQuotePublisher(root))
                after = sqlite3.connect(db).execute(
                    "SELECT COUNT(*) FROM instruments").fetchone()[0]
                self.assertEqual(after, before,
                                 f"{label}: дельта завела новую бумагу")


class SnapshotCacheTest(unittest.TestCase):
    """§−105 · повторная команда не обязана снова тянуть базу по сети.

    🔴 Замер: КАЖДАЯ команда скачивала базу ЦЕЛИКОМ — `/status`, `/universe`,
    `/check`, `/prune` по одному полному скачиванию на нажатие. В GCS это сеть,
    и отсюда «работают, только долго».

    Кэш ключуется ПОКОЛЕНИЕМ, которое GCS отдаёт в метаданных `get_blob` без
    скачивания тела. Три свойства обязаны держаться одновременно, и каждое
    здесь проверяется: тело не качается повторно; смена поколения кэш
    инвалидирует; кэш отдаёт КОПИЮ, потому что `_apply` скачанный файл мутирует.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory(prefix="ombri-cache-")
        self.root = Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)
        self.source = self.root / "src.sqlite"
        self.source.write_bytes(b"BASE" * 2048)
        self.body_calls = 0
        self._env = mock.patch.dict(
            os.environ, {"INGEST_SNAPSHOT_CACHE_DIR": str(self.root / "cache"),
                         "INGEST_SNAPSHOT_CACHE_MAX_MB": "64"})
        self._env.start(); self.addCleanup(self._env.stop)

    def _publisher(self, generation: int = 111):
        from services.quote_publisher import GcsQuotePublisher  # noqa: PLC0415
        outer = self

        class _Blob:
            def __init__(self, gen):
                self.generation = gen
                self.size = outer.source.stat().st_size

            def download_to_filename(self, path):
                outer.body_calls += 1
                shutil.copyfile(outer.source, path)

        class _Bucket:
            def get_blob(self, _name):
                return _Blob(pub.generation)

        pub = GcsQuotePublisher("test-bucket", "stooq/")
        pub.generation = generation
        pub._bucket = lambda: _Bucket()
        return pub

    def test_same_generation_is_served_without_a_second_download(self):
        pub = self._publisher()
        for i in range(3):
            snap = pub.download(self.root / f"d{i}.sqlite")
            self.assertEqual(snap.generation, 111)
        self.assertEqual(self.body_calls, 1,
                         "база скачана повторно при неизменном поколении")

    def test_new_generation_invalidates_the_cache(self):
        pub = self._publisher()
        pub.download(self.root / "a.sqlite")
        pub.generation = 222
        snap = pub.download(self.root / "b.sqlite")
        self.assertEqual(snap.generation, 222)
        self.assertEqual(self.body_calls, 2,
                         "новое поколение обязано вытеснить кэш — иначе под "
                         "запись уедет устаревшая база")

    def test_cache_hands_out_a_copy_not_its_own_file(self):
        """`_apply` мутирует скачанное — выдача файла кэша отравила бы его."""
        pub = self._publisher()
        first = self.root / "work.sqlite"
        pub.download(first)
        original = first.stat().st_size
        first.write_bytes(b"MUTATED")
        second = pub.download(self.root / "again.sqlite")
        self.assertEqual(second.size, original)

    def test_cache_can_be_switched_off_entirely(self):
        """Нулевой потолок = кэш ВЫКЛЮЧЕН (не «без ограничения»)."""
        with mock.patch.dict(os.environ, {"INGEST_SNAPSHOT_CACHE_MAX_MB": "0"}):
            pub = self._publisher()
            pub.download(self.root / "x.sqlite")
            pub.download(self.root / "y.sqlite")
        self.assertEqual(self.body_calls, 2)

    def test_database_above_the_cap_is_not_cached(self):
        """🔴 `/tmp` в Cloud Run — это RAM, у загрузчика её 512 МиБ.

        В пике живут ДВА файла: копия кэша и рабочая копия команды. База выше
        потолка обязана идти мимо кэша, иначе оптимизация роняет контейнер —
        цена ошибки тут выше цены самой оптимизации.
        """
        self.source.write_bytes(b"X" * (2 * 1024 * 1024))    # 2 МБ
        with mock.patch.dict(os.environ, {"INGEST_SNAPSHOT_CACHE_MAX_MB": "1"}):
            pub = self._publisher()
            pub.download(self.root / "big1.sqlite")
            pub.download(self.root / "big2.sqlite")
        self.assertEqual(
            self.body_calls, 2,
            "база выше потолка попала в кэш — защита памяти снята")

    def test_database_below_the_cap_is_cached(self):
        """Парная проверка: иначе тест выше проходил бы и при мёртвом кэше."""
        with mock.patch.dict(os.environ, {"INGEST_SNAPSHOT_CACHE_MAX_MB": "8"}):
            pub = self._publisher()
            pub.download(self.root / "s1.sqlite")
            pub.download(self.root / "s2.sqlite")
        self.assertEqual(self.body_calls, 1)

    def test_broken_cache_dir_never_fails_the_command(self):
        """Сбой кэша — не ошибка: отчёт о базе важнее оптимизации."""
        blocker = self.root / "blocked"
        blocker.write_text("не каталог", encoding="utf-8")
        with mock.patch.dict(os.environ,
                             {"INGEST_SNAPSHOT_CACHE_DIR": str(blocker)}):
            pub = self._publisher()
            snap = pub.download(self.root / "z.sqlite")
        self.assertEqual(snap.generation, 111)
        self.assertTrue((self.root / "z.sqlite").is_file())


class _RejectingMessage(_FakeMessage):
    """Telegram, который отвечает 400 — как настоящий на битой разметке.

    `reject` — предикат по `(text, parse_mode)`. Всё, что он пропустил,
    считается доставленным.
    """

    def __init__(self, reject, **kw):
        super().__init__(_FakeBot(""), **kw)
        self._reject = reject

    async def answer(self, text, **kwargs):
        if self._reject(text, kwargs.get("parse_mode")):
            raise RuntimeError("Bad Request: can't parse entities")
        return await super().answer(text, **kwargs)


@unittest.skipUnless(_HAS_AIOGRAM, "aiogram не установлен (офлайн-разработка)")
class DeliveryNeverGoesSilentTest(_IngestCase):
    """§−104 · МОЛЧАНИЕ — худший ответ бота, чей смысл в доставке сводки.

    🔴 Живой симптом (владелец, 25.08): «`/help`, `/universe`, `/prune` — бот
    молчит и ничего не отвечает», при работающих `/status` и `/missing`.

    Причина структурная: КАЖДЫЙ вызов `message.answer` стоял ВНЕ `try`, включая
    финальную строку `_guarded`. Любой отказ Telegram — 400 на разметке, 400 на
    длине — вылетал из хендлера, aiogram писал его в лог, а оператор не получал
    НИЧЕГО: ни ответа, ни ошибки. Работали ровно те команды, чей текст короткий
    и простой; молчали три с самыми рискованными полезными нагрузками — сырая
    разметка `_HELP`, самые длинные списки и единственная инлайн-клавиатура.
    """

    def test_html_rejected_falls_back_to_plain_text(self):
        import asyncio                                   # noqa: PLC0415
        import ingest_bot                                # noqa: PLC0415
        msg = _RejectingMessage(lambda _t, pm: pm is not None)
        asyncio.run(ingest_bot._deliver(msg, "<pre>сводка</pre>"))
        self.assertTrue(msg.answers, "оператор остался БЕЗ ответа")
        text, kwargs = msg.answers[-1]
        self.assertIsNone(kwargs.get("parse_mode"))
        self.assertIn("сводка", text)
        self.assertNotIn("<pre>", text)

    def test_handler_never_raises_when_delivery_fails(self):
        """Хендлер не имеет права уронить ответ вместе с собой."""
        import asyncio                                   # noqa: PLC0415
        import ingest_bot                                # noqa: PLC0415
        for name, handler in (("/help", ingest_bot.cmd_help),
                              ("/status", ingest_bot.cmd_status),
                              ("/universe", ingest_bot.cmd_universe),
                              ("/prune", ingest_bot.cmd_prune)):
            with self.subTest(command=name):
                msg = _RejectingMessage(lambda _t, _pm: True, text=name)
                try:
                    asyncio.run(handler(msg))
                except Exception as exc:                # noqa: BLE001
                    self.fail(f"{name}: исключение вылетело из хендлера "
                              f"({type(exc).__name__}) → оператор видит молчание")

    def test_every_answer_in_the_module_goes_through_the_ladder(self):
        """Гейт по КОДУ: новая точка доставки обязана идти через `_deliver`.

        Иначе дефект отрастает по одной строке за раз — ровно так он и возник.
        Ловится доставка С РАЗМЕТКОЙ: короткой служебной реплике без неё
        отказать нечем.
        """
        import ingest_bot                                # noqa: PLC0415
        src = Path(ingest_bot.__file__).read_text(encoding="utf-8")
        # Тело самой лестницы исключается: три её `answer` и ЕСТЬ реализация.
        lines, inside = [], False
        for ln in src.splitlines():
            if ln.startswith("async def _deliver("):
                inside = True
                continue
            if inside and ln.startswith(("def ", "async def ", "class ")):
                inside = False
            if not inside:
                lines.append(ln)
        offenders = [ln.strip() for ln in lines
                     if ".answer(" in ln and "_deliver" not in ln
                     and "parse_mode" in ln]
        self.assertEqual(
            offenders, [],
            "доставка с разметкой мимо `_deliver` — она молча умрёт на 400: "
            + "; ".join(offenders))


@unittest.skipUnless(_HAS_AIOGRAM, "aiogram не установлен (офлайн-разработка)")
class RenderBlockBoundsTheResultTest(unittest.TestCase):
    """§−104 · гард длины обязан ограничивать ОТДАННОЕ, а не входное.

    Обрезка шла ДО `html.escape`, а экранирование расширяет: `&` → `&amp;`
    (×5). Замер: 3 900 символов на входе давали **19 554** на выходе — в 4.8
    раза выше потолка Telegram 4096. То есть гард, заведённый ровно ради «не
    превысить длину», её не ограничивал, а 400-й убивает сообщение целиком.
    """

    TELEGRAM_MAX = 4096

    def test_escaped_result_fits_telegram(self):
        import ingest_bot                                # noqa: PLC0415
        for label, body in (("амперсанды", "&" * 4000),
                            ("угловые скобки", "<x>" * 1400),
                            ("кавычки", '"' * 4000),
                            ("обычный текст", "щ" * 4000)):
            with self.subTest(case=label):
                out = ingest_bot.render_block(body)
                self.assertLessEqual(
                    len(out), self.TELEGRAM_MAX,
                    f"{label}: отдано {len(out)} симв. при потолке "
                    f"{self.TELEGRAM_MAX} — Telegram ответит 400")

    def test_truncation_does_not_split_an_html_entity(self):
        """Хвост вида `&am` — сломанная разметка, то есть тот же 400-й."""
        import ingest_bot                                # noqa: PLC0415
        out = ingest_bot.render_block("&" * 4000)
        head = out[len("<pre>"):].split("\n")[0]
        self.assertNotRegex(head, r"&[a-z]{0,5}$")

    def test_short_text_is_untouched(self):
        import ingest_bot                                # noqa: PLC0415
        self.assertEqual(ingest_bot.render_block("привет"), "<pre>привет</pre>")


@unittest.skipUnless(_HAS_AIOGRAM, "aiogram не установлен (офлайн-разработка)")
class DocumentHandlerTest(_IngestCase):
    """🔴 Раньше ядро бота не имело НИ ОДНОГО теста.

    Сводка, лимиты, маршрутизация файла и удаление временной копии
    проверялись только глазами — то есть не проверялись. Здесь Telegram
    подменён целиком: сети нет, токена нет, а путь пройден весь.
    """

    def setUp(self) -> None:
        super().setUp()
        import ingest_bot                                # noqa: PLC0415
        self.ib = ingest_bot
        env = mock.patch.dict(os.environ, {
            "QUOTES_BACKEND": "local",
            "QUOTES_LOCAL_ROOT": str(self.store),
        })
        env.start()
        self.addCleanup(env.stop)

    def _send(self, name, payload, size=1000):
        import asyncio                                   # noqa: PLC0415
        bot = _FakeBot(payload)
        message = _FakeMessage(bot, document=_Doc(name, size))
        asyncio.run(self.ib.on_document(message))
        return bot, message

    def test_daily_file_travels_end_to_end_and_is_published(self) -> None:
        bot, message = self._send("20260813_d.txt",
                                  _daily_text(20260813, self.universe))
        summary = message.answers[-1][0]
        self.assertIn("применён", summary)
        self.assertIn("поколение", summary)
        self.assertEqual(len(_bars(self.db)), 8)
        self.assertEqual(len(bot.downloads), 1)

    def test_oversized_file_is_refused_WITHOUT_downloading_it(self) -> None:
        """🔴 Главное свойство порядка проверок.

        `/tmp` на Cloud Run — это оперативная память; скачать архив, чтобы
        потом отказать, значит заплатить памятью за отказ.
        """
        bot, message = self._send("20260813_d.txt", "неважно",
                                  size=qi.MAX_UPLOAD_BYTES + 1)
        self.assertEqual(bot.downloads, [], "файл скачали, хотя обязаны были "
                                            "отказать по метаданным")
        self.assertIn("не принят", message.answers[-1][0])

    def test_history_file_is_routed_to_the_backfill_path(self) -> None:
        before = self.instruments()
        _bot, message = self._send("nvda.us.txt",
                                   _history_text("NVDA.US", (20260810,)))
        self.assertIn("применён", message.answers[-1][0])
        self.assertEqual(self.instruments(), before + 1)

    def test_the_temporary_copy_is_removed_afterwards(self) -> None:
        bot, _message = self._send("20260813_d.txt",
                                   _daily_text(20260813, self.universe))
        self.assertFalse(bot.downloads[0].exists(),
                         "временный файл остался в /tmp — а это RAM")

    def test_refusal_reaches_the_operator_instead_of_a_traceback(self) -> None:
        with self.assertLogs("ombri.ingest", level="WARNING"):
            _bot, message = self._send("20260813_d.txt", "<html>капча</html>")
        summary = message.answers[-1][0]
        self.assertIn("применён", summary)
        self.assertIn("TICKER", summary)

    def test_every_summary_goes_out_as_escaped_HTML(self) -> None:
        """🔴 Пин на способ доставки, а не на его вид.

        Legacy-Markdown разбирает разметку и внутри блока и отвечает 400-м на
        несбалансированный символ — сводка не доходит ВООБЩЕ. Откат на него
        прошёл бы мимо всех остальных тестов.
        """
        _bot, message = self._send("20260813_d.txt",
                                   _daily_text(20260813, self.universe))
        text, kwargs = message.answers[-1]
        self.assertTrue(text.startswith("<pre>"), text[:40])
        self.assertEqual(str(kwargs.get("parse_mode")).lower().split(".")[-1],
                         "html")

    def test_dangerous_characters_are_escaped_not_rendered(self) -> None:
        self.assertIn("&lt;b&gt;", self.ib.render_block("<b>x</b>"))
        self.assertIn("&amp;", self.ib.render_block("a & b"))

    def test_a_long_summary_is_clipped_rather_than_dropped(self) -> None:
        """Обрезанная сводка лучше не доставленной: лимит Telegram — 4096."""
        rendered = self.ib.render_block("x" * 10_000)
        self.assertLess(len(rendered), 4096)
        self.assertIn("обрезано", rendered)


@unittest.skipUnless(_HAS_AIOGRAM, "aiogram не установлен (офлайн-разработка)")
class PruneConfirmationTest(_IngestCase):
    """Необратимая команда требует второго осознанного действия."""

    def setUp(self) -> None:
        super().setUp()
        import ingest_bot                                # noqa: PLC0415
        self.ib = ingest_bot
        env = mock.patch.dict(os.environ, {
            "QUOTES_BACKEND": "local",
            "QUOTES_LOCAL_ROOT": str(self.store),
        })
        env.start()
        self.addCleanup(env.stop)
        thin = self.root / "thin.us.txt"
        thin.write_text(_history_text("THIN.US", (20260812,)), encoding="utf-8")
        conn = si.connect(self.db)
        si.apply_batch(conn, si.parse_history_file(thin, window_days=9000),
                       kind="bootstrap", allow_new=True)
        conn.close()

    def _callback(self, data):
        import asyncio                                   # noqa: PLC0415
        message = _FakeMessage(_FakeBot(""))

        class _CB:
            def __init__(self, msg):
                self.data = data
                self.message = msg
                self.from_user = type("U", (), {"id": 1})()

            async def answer(self, *a, **k):
                return None

        with mock.patch.object(qi, "_working_set",
                               return_value=list(self.universe)):
            asyncio.run(self.ib.cb_prune(_CB(message)))
        return message

    def test_cancel_leaves_the_base_alone(self) -> None:
        before = _bars(self.db)
        message = self._callback("prune:no")
        self.assertIn("Отменено", message.answers[-1][0])
        self.assertEqual(_bars(self.db), before)

    def test_confirmation_actually_prunes_and_publishes(self) -> None:
        message = self._callback("prune:go")
        self.assertIn("вычищена", message.answers[-1][0])
        conn = si.connect(self.db, read_only=True)
        try:
            left = {r["source_symbol"] for r in conn.execute(
                "SELECT source_symbol FROM instruments")}
        finally:
            conn.close()
        self.assertNotIn("THIN.US", left)


@unittest.skipUnless(_HAS_AIOGRAM, "aiogram не установлен (офлайн-разработка)")
class CommandFailureTest(_IngestCase):
    """Ни одна команда не имеет права уронить polling или отдать traceback."""

    def setUp(self) -> None:
        super().setUp()
        import ingest_bot                                # noqa: PLC0415
        self.ib = ingest_bot

    def test_missing_store_is_explained_not_raised(self) -> None:
        import asyncio                                   # noqa: PLC0415
        message = _FakeMessage(_FakeBot(""), text="/status")
        with mock.patch.dict(os.environ, {
                "QUOTES_BACKEND": "local",
                "QUOTES_LOCAL_ROOT": str(self.root / "nowhere")}):
            asyncio.run(self.ib.cmd_status(message))
        self.assertIn("недоступна", message.answers[-1][0])

    def test_an_unexpected_error_is_caught_and_named(self) -> None:
        import asyncio                                   # noqa: PLC0415
        message = _FakeMessage(_FakeBot(""), text="/status")
        with mock.patch.object(qi, "status", side_effect=RuntimeError("бум")), \
                self.assertLogs("ombri.ingest", level="ERROR"):
            asyncio.run(self.ib.cmd_status(message))
        self.assertIn("бум", message.answers[-1][0])

    def test_check_without_an_argument_explains_the_format(self) -> None:
        import asyncio                                   # noqa: PLC0415
        message = _FakeMessage(_FakeBot(""), text="/check")
        asyncio.run(self.ib.cmd_check(message))
        self.assertIn("/check", message.answers[-1][0])



@unittest.skipUnless(_HAS_AIOGRAM, "aiogram не установлен (офлайн-разработка)")
class DispatcherRoutingTest(unittest.TestCase):
    """Проверка НАСТОЯЩЕЙ маршрутизации, а не только сборки диспетчера.

    Всё остальное здесь зовёт хендлеры напрямую и потому ничего не говорит о
    том, доедет ли до них апдейт. Здесь апдейт скармливается диспетчеру
    целиком: работают фильтры, порядок регистрации и middleware.
    """

    def _update(self, text: str, user_id: int):
        from datetime import datetime                    # noqa: PLC0415
        from aiogram.types import Chat, Message, Update, User   # noqa: PLC0415

        return Update(update_id=1, message=Message(
            message_id=1, date=datetime.now(),
            chat=Chat(id=user_id, type="private"),
            from_user=User(id=user_id, is_bot=False, first_name="op"),
            text=text))

    def _feed(self, text: str, user_id: int, admins: str) -> list:
        import asyncio                                   # noqa: PLC0415
        from aiogram import Bot                          # noqa: PLC0415
        import ingest_bot                                # noqa: PLC0415

        seen: list = []

        async def recorder(message):
            seen.append(message.text)

        # 🔴 `Message.answer` подменяется НЕ для удобства. Гейт деплоя объявлен
        # сетевно-изолированным («no live calls leave the builder»,
        # `cloudbuild.yaml`), а отказ чужому отправляется именно через
        # `answer` — то есть без подмены тест стучался бы в api.telegram.org с
        # заведомо чужим токеном. Подмена заодно делает отказ проверяемым.
        from aiogram.types import Message                # noqa: PLC0415

        refusals: list = []

        async def _answer(self, text, **_kw):
            refusals.append(text)
            return None

        async def run():
            bot = Bot(token="123456:AAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            try:
                with mock.patch.object(ingest_bot, "cmd_status", recorder), \
                        mock.patch.object(Message, "answer", _answer), \
                        mock.patch.dict(os.environ, {access.ENV_NAME: admins}):
                    dispatcher = ingest_bot.build_dispatcher()
                    await dispatcher.feed_update(bot, self._update(text, user_id))
            finally:
                await bot.session.close()

        asyncio.run(run())
        self._refusals = refusals
        return seen

    def test_the_admin_command_reaches_its_handler(self) -> None:
        self.assertEqual(self._feed("/status", 7, "7"), ["/status"])

    def test_a_stranger_never_reaches_the_handler(self) -> None:
        """Гейт стоит ДО хендлера, а не внутри него."""
        with self.assertLogs("ingest_access", level="WARNING"):
            self.assertEqual(self._feed("/status", 999, "7"), [])
        self.assertEqual(self._refusals, [access.DENIAL_TEXT],
                         "чужому не ответили нейтральным отказом")

    def test_an_empty_admin_list_stops_everyone(self) -> None:
        with self.assertLogs("ingest_access", level="WARNING"):
            self.assertEqual(self._feed("/status", 7, ""), [])

    def test_both_buses_carry_the_gate(self) -> None:
        """🔴 Кнопка подтверждения чистки приходит КОЛБЭКОМ.

        Шина без гейта пустила бы чужое нажатие к необратимой операции —
        именно этого не хватало в первой редакции.
        """
        import ingest_bot                                # noqa: PLC0415

        dispatcher = ingest_bot.build_dispatcher()
        for bus in ("message", "callback_query"):
            with self.subTest(bus=bus):
                observer = getattr(dispatcher, bus)
                names = [type(m).__name__
                         for m in getattr(observer.middleware, "_middlewares", [])]
                self.assertIn("AdminOnlyMiddleware", names)


# ═════════════════════════════════════════════════════════════════════════════
# Слои и изоляция от прода
# ═════════════════════════════════════════════════════════════════════════════

class TokenEnvRenameTest(unittest.TestCase):
    """Переезд имени RAMP → OMBRI у ЗАГРУЗЧИКА (`§−100`).

    Владелец переименовал секрет в `OMBRI_INGEST_BOT_TOKEN`. Переезд разнесён
    во времени намеренно: сначала переехал загрузчик, отдельной операцией —
    главный бот (`§−101`). Трогать имя переменной работающего бота из
    пристройки было бы ровно тем срастанием двух ботов, которого весь
    модуль и избегает.

    🔴 Отсюда три обязательства, и каждое проверяется ниже:
    1. бот читает НОВОЕ имя;
    2. ПРЕЖНЕЕ имя всё ещё принимается — иначе полуприменённая настройка
       (код переехал, секрет нет) даёт «токен пуст» на живом секрете;
    3. приём прежнего имени не молчалив: без предупреждения переезд не
       закончится никогда, а «работает» перестанет означать «настроено».
    """

    _KEYS = ("OMBRI_INGEST_BOT_TOKEN", "RAMP_INGEST_BOT_TOKEN")

    def setUp(self) -> None:
        self._prev = {k: os.environ.get(k) for k in self._KEYS}

        def _restore() -> None:
            for key, value in self._prev.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.addCleanup(_restore)
        for key in self._KEYS:
            os.environ.pop(key, None)

    def test_new_name_is_the_primary_one(self) -> None:
        import ingest_bot                                 # noqa: PLC0415

        self.assertEqual(ingest_bot.TOKEN_ENV, "OMBRI_INGEST_BOT_TOKEN")
        os.environ["OMBRI_INGEST_BOT_TOKEN"] = "111:NEW"
        self.assertEqual(ingest_bot.read_bot_token(), "111:NEW")

    def test_legacy_name_still_works(self) -> None:
        import ingest_bot                                 # noqa: PLC0415

        os.environ["RAMP_INGEST_BOT_TOKEN"] = "222:OLD"
        with self.assertLogs("ombri.ingest", level="WARNING") as log:
            self.assertEqual(ingest_bot.read_bot_token(), "222:OLD")
        self.assertIn("OMBRI_INGEST_BOT_TOKEN", "".join(log.output),
                      "приём прежнего имени обязан называть новое — иначе "
                      "оператор не узнает, что именно переименовывать")

    def test_new_name_wins_over_the_legacy_one(self) -> None:
        """Обе заданы — берётся НОВАЯ, без предупреждения."""
        import ingest_bot                                 # noqa: PLC0415

        os.environ["OMBRI_INGEST_BOT_TOKEN"] = "111:NEW"
        os.environ["RAMP_INGEST_BOT_TOKEN"] = "222:OLD"
        self.assertEqual(ingest_bot.read_bot_token(), "111:NEW")

    def test_neither_name_gives_empty_not_an_exception(self) -> None:
        """Пусто — это пусто; отказ печатает `main`, и он объясняет причину."""
        import ingest_bot                                 # noqa: PLC0415

        self.assertEqual(ingest_bot.read_bot_token(), "")

    def test_the_loader_does_not_configure_the_main_bot(self) -> None:
        """🔴 Пристройка не настраивает главный бот — только сверяется с ним.

        Главный бот переехал отдельной операцией (`§−101`) и читает токен
        своим `tg_bot.read_bot_token()`. Загрузчику от него нужно РОВНО одно:
        имена переменных, чтобы поймать общий токен. Оба имени обязаны
        остаться в сверке — прежнее тоже: пока секрет в Secret Manager носит
        старое имя, в контейнере главного бота может оказаться любое из двух,
        и страж не должен ослепнуть.
        """
        import ingest_bot                                 # noqa: PLC0415
        self.assertEqual(set(ingest_bot.MAIN_TOKEN_ENVS),
                         {"OMBRI_BOT_TOKEN", "RAMP_BOT_TOKEN"})
        # …и он именно СВЕРЯЕТСЯ: своего имени переменной главному боту
        # загрузчик не назначает.
        source = (Path(__file__).resolve().parents[1] / "src" / "ingest_bot.py"
                  ).read_text(encoding="utf-8")
        self.assertNotIn("os.environ[", source,
                         "загрузчик не вправе требовать переменные главного бота")


class IsolationTest(unittest.TestCase):
    """Загрузчик обязан быть пристройкой, а не врезкой в работающий бот."""

    _SRC = Path(__file__).resolve().parent.parent / "src"
    _NEW = ("ingest_bot.py", "ingest_entrypoint.py", "ingest_access.py",
            "services/quote_ingest.py", "services/quote_publisher.py")

    def _imports(self, relative: str) -> set[str]:
        import ast                                       # noqa: PLC0415
        tree = ast.parse((self._SRC / relative).read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
        return names

    def test_new_modules_never_import_the_main_bot(self) -> None:
        """Инверсия слоёв: правка онбординга не вправе ломать загрузку цен."""
        for relative in self._NEW:
            with self.subTest(module=relative):
                self.assertNotIn("tg_bot", self._imports(relative))

    def test_the_main_bot_does_not_import_the_loader(self) -> None:
        """Обратная сторона: прод не должен зависеть от новой пристройки.

        Пока этого импорта нет, деплой главного бота физически не может
        сломаться из-за загрузчика.
        """
        for relative in ("tg_bot.py", "entrypoint.py"):
            with self.subTest(module=relative):
                imported = self._imports(relative)
                self.assertNotIn("ingest_bot", imported)
                self.assertNotIn("ingest_access", imported)
                self.assertFalse({"services.quote_ingest"} & imported)

    def test_the_whole_offline_cycle_runs_without_third_party_libraries(self) -> None:
        """🔴 Проверяется РАБОТА без библиотек, а не отсутствие строк `import`.

        Приём взят у `test_ingest_library_needs_no_third_party`: сторонние
        модули блокируются на `sys.meta_path` в отдельном процессе. Скан
        импортов по AST такого не доказывает и вдобавок врёт в обе стороны —
        он считает нарушением ЛЕНИВЫЙ импорт GCS внутри метода (тот в офлайне
        не исполняется) и пропустил бы зависимость, приехавшую транзитивно.

        Прогоняется при этом весь цикл, а не один `import`: собрать базу,
        применить дельту, опубликовать. Офлайн-режим либо работает целиком,
        либо это не режим.
        """
        import subprocess                                # noqa: PLC0415
        import sys as _sys                               # noqa: PLC0415

        code = (
            "import sys, importlib.abc, tempfile, pathlib\n"
            "BLOCKED = {'pandas','numpy','sklearn','aiogram','anthropic','google'}\n"
            "class B(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in BLOCKED:\n"
            "            raise ImportError(name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, B())\n"
            "import ingest_access\n"
            "from finance import stooq_ingest as si\n"
            "from services.quote_publisher import LocalQuotePublisher\n"
            "from services import quote_ingest as qi\n"
            "si.MIN_US_ROWS = 0\n"
            "H = '<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>'\n"
            "root = pathlib.Path(tempfile.mkdtemp())\n"
            "store = root / 'store'; store.mkdir()\n"
            "hist = root / 'spy.us.txt'\n"
            "hist.write_text(H + chr(10) + 'SPY.US,D,20260810,0,10,11,9,10,100,0' + chr(10))\n"
            "conn = si.connect(store / 'prices.sqlite'); si.ensure_schema(conn)\n"
            "si.apply_batch(conn, si.parse_history_file(hist, window_days=9000),"
            " kind='bootstrap', allow_new=True); conn.close()\n"
            "delta = root / '20260813_d.txt'\n"
            "delta.write_text(H + chr(10) + 'SPY.US,D,20260813,0,10,11,9,10.5,500,0' + chr(10))\n"
            "out = qi.apply_daily(delta, actor='1', publisher=LocalQuotePublisher(store))\n"
            "assert out.ok, out.reason\n"
            "assert out.published\n"
            "assert out.result.rows_written == 1\n"
            "assert out.c1 is not None and not out.c1.checked, 'C-1 обязан честно"
            " сказать, что не проверен'\n"
            "print('ok')\n"
        )
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run([_sys.executable, "-c", code], cwd=str(root),
                              env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
                              capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0,
                         f"офлайн-цикл потянул стороннее:\n{proc.stderr}")

    def test_new_top_level_modules_are_known_to_the_layering_gate(self) -> None:
        """🔴 Без записи в `_PROJECT_ROOTS` импорты модуля считаются ВНЕШНИМИ.

        Гейт приватных кросс-импортов прошёл бы мимо новых файлов и остался
        зелёным, ничего не проверив.
        """
        import test_layering                             # noqa: PLC0415

        for name in ("ingest_bot", "ingest_entrypoint", "ingest_access"):
            self.assertIn(name, test_layering._PROJECT_ROOTS)


# ═════════════════════════════════════════════════════════════════════════════
# IB-7 — расписание и радиус писателя (ручная обвязка GCP)
# ═════════════════════════════════════════════════════════════════════════════

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class SchedulerScriptMatchesTheRouteTest(unittest.TestCase):
    """🔴 Скрипт и маршрут обязаны говорить об ОДНОМ адресе.

    Маршрут `POST /tasks/check` готов с IB-7, но расписания не было — и
    напоминание не приходило НИКОГДА. Скрипт эту дыру закрывает; теперь
    появляется вторая, тише первой: путь и заголовок продублированы в bash, а
    дубль расходится молча. Расхождение здесь не даёт ни ошибки, ни симптома —
    планировщик получает 200 «OK» с пробы `GET /`, задание зелёное, а проверка
    не выполняется. Тот же класс, что `§−100`: оба конца живы, кольцо разорвано.
    """

    def setUp(self) -> None:
        path = _SCRIPTS / "setup_ingest_scheduler.sh"
        # `scripts/` в образ не копируется — см. шапку `test_repo_hygiene`.
        if not path.is_file():
            self.skipTest("scripts/ отсутствует (деплой-гейт)")
        self.body = path.read_text(encoding="utf-8")
        import ingest_entrypoint                          # noqa: PLC0415
        self.entry = ingest_entrypoint

    def _assignment(self, name: str) -> str:
        """Значение bash-присваивания `NAME=...` — РАБОЧАЯ строка, а не файл.

        🔴 Проверка «путь встречается где-то в скрипте» была слабой, и мутация
        это доказала: `/tasks/check` стоит в скрипте ТРИЖДЫ — в комментарии, в
        тексте отказа и в самом URI. Подмена в URI оставляла две других копии
        на месте, тест не замечал ничего, а задание стучалось бы по чужому
        адресу. Пробу `GET /` сервис отдаёт 200 на ЛЮБОЙ путь, кроме своего —
        значит задание было бы вечно-зелёным, а проверка не выполнялась бы.
        """
        match = re.search(rf"^{name}=(.+)$", self.body, re.MULTILINE)
        self.assertIsNotNone(match, f"присваивание {name} не найдено")
        return match.group(1)

    def _invocation(self, verb: str) -> str:
        """Строка ВЫЗОВА `gcloud scheduler jobs <verb> http`, а не упоминания.

        🔴 Первая редакция искала подстроку по всему файлу и находила её в
        КОММЕНТАРИИ, объясняющем разницу флагов. Ровно та же ошибка, что уже
        ловилась мутацией на `TASK_PATH`: текст файла — не то же самое, что
        исполняемая строка. Гейт обязан смотреть на вторую.
        """
        # Продолжения строк склеиваются ПЕРЕД поиском: аргументы вызова
        # разнесены по строкам через `\`, и построчный поиск увидел бы только
        # первую из них — то есть проверял бы четверть команды.
        joined = re.sub(r"\\\n\s*", " ", self.body)
        for line in joined.splitlines():
            if f"gcloud scheduler jobs {verb} http" in line \
                    and not line.lstrip().startswith("#"):
                return line
        self.fail(f"вызов jobs {verb} http не найден")

    def test_the_job_targets_the_path_the_service_actually_serves(self) -> None:
        self.assertIn(self.entry.TASK_PATH, self._assignment("TARGET_URI"),
                      "скрипт стучится не туда, куда слушает сервис")

    def test_the_header_name_is_the_one_the_route_checks(self) -> None:
        """Заголовок с опечаткой = 401 на каждом запуске, и молча."""
        self.assertIn(self.entry.TASK_TOKEN_HEADER,
                      self._assignment("HEADER_NAME"))

    def test_the_job_is_built_from_the_pinned_variables(self) -> None:
        """🔴 Пина переменной мало: аргумент обязан брать ИМЕННО её.

        Вторая мутация показала дыру в первой редакции: `--uri=` можно вписать
        мимо `TARGET_URI`, и тогда пин сторожит переменную, которой никто не
        пользуется. Гейт, охраняющий неиспользуемое значение, зелен всегда.
        """
        self.assertIn('--uri="$TARGET_URI"', self.body,
                      "адрес задания собран мимо проверяемой переменной")
        self.assertIn('--headers="${HEADER_NAME}=', self.body,
                      "заголовок собран мимо проверяемой переменной")

    def test_the_job_uses_POST(self) -> None:
        """`_handle_task` отвечает 405 на всё, кроме POST."""
        self.assertIn("--http-method=POST", self.body)

    def test_the_call_is_authenticated(self) -> None:
        """Сервис задеплоен `--no-allow-unauthenticated`: без OIDC — 403."""
        self.assertIn("--oidc-service-account-email", self.body)
        self.assertIn("--oidc-token-audience", self.body)

    def test_the_secret_is_read_from_secret_manager_not_hardcoded(self) -> None:
        self.assertIn("gcloud secrets versions access", self.body)

    def test_the_secret_is_never_printed(self) -> None:
        """🔴 Значение секрета не имеет права попасть в вывод скрипта.

        Оператор гоняет его в Cloud Shell, а вывод уходит в историю сессии и в
        скриншоты. Печатать разрешено ИМЯ секрета, но не значение.
        """
        for line in self.body.splitlines():
            stripped = line.strip()
            if re.match(r"^(echo|info|warn|die)\b", stripped):
                self.assertNotIn(
                    "TASK_TOKEN", stripped,
                    f"значение секрета уходит в вывод: {stripped[:70]}")

    def test_create_and_update_use_THEIR_OWN_header_flag(self) -> None:
        """🔴 Флаг заголовков у `create` и `update` РАЗНЫЙ, и это не мелочь.

        Первая редакция гнала один набор аргументов в обе ветки.
        `gcloud scheduler jobs update http` не знает `--headers` — он отвечает
        `unrecognized arguments` и ПЕЧАТАЕТ нераспознанный аргумент целиком.
        А в этом аргументе лежит секрет. То есть неверный флаг не просто ломал
        обновление задания: он РОНЯЛ СЕКРЕТ В ВЫВОД. Обожглись живьём 26.08 —
        сначала на флаге, потом на утечке из-за него.
        """
        update, create = self._invocation("update"), self._invocation("create")
        self.assertIn("--update-headers=", update,
                      "обновление задания пойдёт с флагом, которого у него нет")
        self.assertNotIn("--headers=", update.replace("--update-headers=", ""),
                         "в ветку обновления попал флаг создания")
        self.assertIn("--headers=", create,
                      "создание задания осталось без заголовка")

    def test_commands_carrying_the_secret_mask_their_output(self) -> None:
        """🔴 «Скрипт не печатает секрет» — недостаточное правило.

        Печатает не скрипт, а СООБЩЕНИЕ ОБ ОШИБКЕ gcloud: неуспешный вызов
        возвращает свои аргументы обратно. Поэтому вызовы, несущие токен,
        обязаны идти через маскирующую обёртку.
        """
        self.assertIn("_masked()", self.body, "обёртки маскировки нет")
        for verb in ("update", "create"):
            self.assertTrue(
                self._invocation(verb).lstrip().startswith("_masked "),
                f"jobs {verb} http вызывается в обход маскировки")

    def test_the_mask_actually_removes_a_real_leak(self) -> None:
        """🔴 Контрольный опыт: обёртка проверяется на СТРОКЕ, которая утекла.

        Регулярка, которая ничего не ловит, — не защита, а её имитация.
        Строка ниже — дословный формат отказа gcloud, наблюдавшийся 26.08.
        """
        import subprocess                                 # noqa: PLC0415

        expr = re.search(r'sed "(s/\$\{HEADER_NAME\}=[^"]+)"', self.body)
        self.assertIsNotNone(expr, "выражение маскировки не найдено")
        # 🔴 Значение СИНТЕТИЧЕСКОЕ, и это часть урока раунда. Первая редакция
        # взяла сюда НАСТОЯЩИЙ токен из утёкшего вывода — он к тому моменту был
        # ротирован и мёртв, но попал в репозиторий и в историю git, где живёт
        # вечно. Тест проверяет РЕГУЛЯРКУ, а ей всё равно, какие именно 64 hex
        # она маскирует. Настоящий секрет не добавлял доказательности — только
        # риск. Формат строки отказа ниже дословный, он и есть предмет проверки.
        secret = "0" * 24 + "deadbeef" + "0" * 32
        leak = ("ERROR: (gcloud.scheduler.jobs.update.http) unrecognized "
                f"arguments: --headers=x-ingest-task-token={secret} "
                "(did you mean '--clear-headers'?)")
        script = (f'HEADER_NAME=x-ingest-task-token\n'
                  f'printf %s "$LEAK" | sed "{expr.group(1)}"\n')
        out = subprocess.run(["bash", "-c", script], capture_output=True,
                             text=True, timeout=60, env={**os.environ, "LEAK": leak})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn(secret, out.stdout, "секрет пережил маскировку")
        self.assertIn("x-ingest-task-token=***", out.stdout,
                      "маскировка не сработала на реальной строке отказа")

    def test_the_default_schedule_runs_on_weekends_too(self) -> None:
        """🔴 Порог свежести — КАЛЕНДАРНЫЙ, значит выходные его тоже съедают.

        `stooq_store.market_staleness_days` считает в календарных днях от
        `today`, а не в торговых сессиях. Расписание «по будням» отдало бы
        двое суток запаса как раз в тот промежуток, когда файл забывают чаще
        всего — в пятницу вечером.
        """
        match = re.search(r'SCHEDULE="\$\{SCHEDULE:-([^}]+)\}"', self.body)
        self.assertIsNotNone(match, "дефолт расписания не найден")
        fields = match.group(1).split()
        self.assertEqual(len(fields), 5, "расписание не пятипольное")
        self.assertEqual(fields[4], "*",
                         "день недели ограничен — выходные останутся без проверки")


class IamVerifierCatchesAWideRadiusTest(unittest.TestCase):
    """🔴 Инструмент проверки сам обязан быть проверен (`§−68`).

    `verify_ingest_iam.sh` — единственный способ узнать, стоит ли условие IAM
    на префиксе `stooq/`: репозиторий это проверить не может, шаг деплоя лишь
    ПЕЧАТАЕТ команду. Если сам скрипт зелёный всегда, он не компенсирующий
    контроль, а его имитация. Поэтому здесь пять политик и ожидаемый вердикт
    по каждой — включая ту, которую легче всего проглядеть: доступ ЕСТЬ,
    условия НЕТ, бот при этом работает штатно.
    """

    _SA = "ramp-ingest@test-project.iam.gserviceaccount.com"
    _MEMBER = f"serviceAccount:{_SA}"
    _NARROW = ("resource.name.startsWith('projects/_/buckets/"
               "ramp-bot-state/objects/stooq/')")

    def setUp(self) -> None:
        self.script = _SCRIPTS / "verify_ingest_iam.sh"
        if not self.script.is_file():
            self.skipTest("scripts/ отсутствует (деплой-гейт)")
        if shutil.which("bash") is None:
            self.skipTest("bash недоступен")
        self.tmp = tempfile.TemporaryDirectory(prefix="ombri-iam-")
        self.root = Path(self.tmp.name)
        stub = self.root / "bin"
        stub.mkdir()
        # Подставной `gcloud`: политики приезжают из файлов, названных в
        # окружении. Ходить в GCP тест не имеет права.
        (stub / "gcloud").write_text(
            "#!/usr/bin/env bash\n"
            'case "$*" in\n'
            '  *"config get-value project"*) echo "test-project" ;;\n'
            # Личность писателя скрипт спрашивает у САМОГО сервиса, а не
            # собирает из имени: угаданное имя дало бы вердикт про чужой
            # аккаунт. Подставной сервис отвечает тем же SA, что в политиках.
            '  *"run services describe"*) echo "$FAKE_SA" ;;\n'
            '  *"storage buckets get-iam-policy"*) cat "$FAKE_BUCKET" ;;\n'
            '  *"projects get-iam-policy"*) cat "$FAKE_PROJECT" ;;\n'
            '  *) echo "{}" ;;\n'
            "esac\n", encoding="utf-8")
        (stub / "gcloud").chmod(0o755)
        self.stub = stub

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _policy(self, name: str, bindings: list) -> str:
        import json                                       # noqa: PLC0415

        path = self.root / f"{name}.json"
        path.write_text(json.dumps({"bindings": bindings}), encoding="utf-8")
        return str(path)

    def _run(self, bucket: list, project: list):
        import subprocess                                 # noqa: PLC0415

        env = dict(os.environ)
        env["PATH"] = f"{self.stub}:{env.get('PATH', '')}"
        env["FAKE_SA"] = self._SA
        env["FAKE_BUCKET"] = self._policy("bucket", bucket)
        env["FAKE_PROJECT"] = self._policy("project", project)
        return subprocess.run(["bash", str(self.script)], env=env,
                              capture_output=True, text=True, timeout=120)

    def test_a_narrow_condition_passes(self) -> None:
        out = self._run([{"role": "roles/storage.objectAdmin",
                          "members": [self._MEMBER],
                          "condition": {"title": "stooq-only",
                                        "expression": self._NARROW}}], [])
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)

    def test_write_access_WITHOUT_a_condition_is_caught(self) -> None:
        """Самый тихий случай: бот работает, симптома нет, радиус — весь бакет."""
        out = self._run([{"role": "roles/storage.objectAdmin",
                          "members": [self._MEMBER]}], [])
        self.assertEqual(out.returncode, 1, "доступ без условия признан узким")
        self.assertIn("БЕЗ УСЛОВИЯ", out.stdout)

    def test_a_condition_naming_the_wrong_bucket_is_caught(self) -> None:
        """Опечатка copy-paste: префикс верный, бакет чужой — не сужает ничего."""
        wrong = self._NARROW.replace("ramp-bot-state", "ramp-bot-quotes")
        out = self._run([{"role": "roles/storage.objectAdmin",
                          "members": [self._MEMBER],
                          "condition": {"title": "oops", "expression": wrong}}], [])
        self.assertEqual(out.returncode, 1, "условие на чужой бакет принято")

    def test_a_project_level_grant_overrides_the_bucket_condition(self) -> None:
        """🔴 Проверка одного бакета этого НЕ ВИДИТ — потому и читаются обе политики."""
        out = self._run(
            [{"role": "roles/storage.objectAdmin", "members": [self._MEMBER],
              "condition": {"title": "stooq-only", "expression": self._NARROW}}],
            [{"role": "roles/storage.admin", "members": [self._MEMBER]}])
        self.assertEqual(out.returncode, 1, "грант на проекте пропущен")
        self.assertIn("ПРОЕКТА", out.stdout)

    def test_no_write_access_at_all_is_a_failure_not_a_success(self) -> None:
        """«Прав нет» — это неработающий загрузчик, а не безопасность."""
        out = self._run([{"role": "roles/storage.objectViewer",
                          "members": [self._MEMBER]}], [])
        self.assertEqual(out.returncode, 1)
        self.assertIn("НЕТ прав", out.stdout)

    def test_the_identity_checked_is_the_one_the_service_actually_runs_as(self) -> None:
        """🔴 Проверять надо ТУ личность, под которой сервис бежит.

        Первая редакция собирала адрес из имени `ramp-ingest@…`. Разойдись оно
        с реальным — скрипт не нашёл бы ни одной привязки и сказал «прав нет
        вовсе»: вердикт fail-closed, но НЕ ТОТ. Оператор пошёл бы выдавать
        права, которые в порядке, а настоящий радиус остался бы непроверенным.
        Здесь сервис бежит под НЕОЖИДАННЫМ аккаунтом, и узкое условие стоит
        именно на нём — скрипт обязан это увидеть.
        """
        odd = "unexpected-writer@test-project.iam.gserviceaccount.com"
        import json                                       # noqa: PLC0415
        import subprocess                                 # noqa: PLC0415

        env = dict(os.environ)
        env["PATH"] = f"{self.stub}:{env.get('PATH', '')}"
        env["FAKE_SA"] = odd
        bucket = self.root / "bucket.json"
        bucket.write_text(json.dumps({"bindings": [
            {"role": "roles/storage.objectAdmin",
             "members": [f"serviceAccount:{odd}"],
             "condition": {"title": "stooq-only", "expression": self._NARROW}}]}),
            encoding="utf-8")
        project = self.root / "project.json"
        project.write_text(json.dumps({"bindings": []}), encoding="utf-8")
        env["FAKE_BUCKET"], env["FAKE_PROJECT"] = str(bucket), str(project)
        out = subprocess.run(["bash", str(self.script)], env=env,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn(odd, out.stdout,
                      "скрипт проверил не ту личность, под которой бежит сервис")


if __name__ == "__main__":                               # pragma: no cover
    unittest.main()
