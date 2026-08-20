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
        """Правило 9: недокачанный будний файл отвергается ЦЕЛИКОМ."""
        with mock.patch.object(si, "MIN_US_ROWS", 5000):
            before_gen = self.publisher._generation()
            before_bars = _bars(self.db)
            outcome = qi.apply_daily(self.daily(20260814), actor="1",
                                     publisher=self.publisher)
        self.assertFalse(outcome.ok)
        self.assertIn("недокачанным", outcome.reason)
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

        with mock.patch.dict(os.environ, {access.ENV_NAME: "1"}):
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
                self.assertLogs("ramp.ingest.entrypoint", level="WARNING") as log:
            reply = self._run("POST", {"x-ingest-task-token": "s"})
        self.assertIn(b"503", reply.split(b"\r\n")[0])
        self.assertIn("отключён", "\n".join(log.output))
        self.assertEqual(self.calls, [])

    def test_wrong_secret_is_refused(self) -> None:
        with mock.patch.dict(os.environ, {"INGEST_TASK_TOKEN": "right"}), \
                self.assertLogs("ramp.ingest.entrypoint", level="WARNING") as log:
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

    def test_loader_gets_its_own_bucket_not_the_state_one(self) -> None:
        """Радиус поражения писателя не должен включать балансы и ключи брокера."""
        self.assertNotEqual(self.doc["substitutions"]["_QUOTES_BUCKET"],
                            self.doc["substitutions"]["_STATE_BUCKET"])
        step = next(s for s in self.doc["steps"] if s["id"] == "deploy-ingest-bot")
        body = "\n".join(step["args"])
        # 🔴 Проверяются ОБА написания. Мутация показала, что проверка одного
        # лишь ЗНАЧЕНИЯ пропускает ссылку на подстановку `${_STATE_BUCKET}` —
        # то есть ровно ту форму, в которой её и написали бы в YAML.
        self.assertNotIn(self.doc["substitutions"]["_STATE_BUCKET"], body)
        self.assertNotIn("_STATE_BUCKET", body)
        self.assertIn("ingest_entrypoint.py", body)

    def test_loader_does_not_mount_the_state_volume(self) -> None:
        step = next(s for s in self.doc["steps"] if s["id"] == "deploy-ingest-bot")
        body = "\n".join(step["args"])
        self.assertNotIn("--add-volume", body)
        self.assertNotIn("/mnt/state", body)


# ═════════════════════════════════════════════════════════════════════════════
# Слои и изоляция от прода
# ═════════════════════════════════════════════════════════════════════════════

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


if __name__ == "__main__":                               # pragma: no cover
    unittest.main()
