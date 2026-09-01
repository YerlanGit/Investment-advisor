"""`§−113` — затычка «долить бумаги, которых нет в дневном срезе».

Зачем она вообще есть
─────────────────────
Дневной пакет АКЦИЙ Stooq не содержит ETF (`STOOQ_CONVENTION §3.6`, замер
повторился в проде 24–28.08). Штатное лечение — качать пакет ETF рядом с
пакетом акций (`OPERATOR_STOOQ §8.0`). Но пока пакета под рукой нет, а срок по
свежести идёт, оператору нужен способ долить недостающие бумаги потикерно.

Что здесь охраняется
────────────────────
1. **Отказ источника приходит ТЕКСТОМ с кодом 200.** «Exceeded the daily hits
   limit» — это не исключение и не пустой ответ; без проверки шапки лимит
   записался бы в дневной файл как данные и уехал бы в базу ценой.
2. **Чужой день не берётся.** Ряд бумаги содержит много дат, а дневной срез —
   ровно одну: правило 1 бота отсеет лишнее, но тогда оператор увидит
   «чужая дата» и пойдёт чинить не то.
3. **Рынок без единого бара за день — закрыт, а не сломан** (та же оговорка,
   что в `quote_ingest._MISSED_WHERE`): иначе крипта тянула бы за собой всю
   американскую секцию в выходной.
4. **Дописанные строки принимает НАСТОЯЩИЙ парсер.** Формат, проверенный
   глазами, — не доказательство; здесь файл прогоняется через
   `parse_daily_file` + `apply_batch` и сверяется цена в базе.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_SCRIPT = _ROOT / "scripts" / "stooq_topup.py"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from finance import stooq_ingest as si                             # noqa: E402

_HEADER = ("<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,"
           "<VOL>,<OPENINT>")
_GOOD_CSV = ("Date,Open,High,Low,Close,Volume\n"
             "2026-08-28,55.1,55.9,54.8,55.4,1234567\n")


def _load():
    """Скрипт живёт в `scripts/`, которого НЕТ в деплой-образе — отсутствие
    каталога обязано быть `skipTest`, иначе падает деплой-гейт."""
    if not _SCRIPT.exists():
        raise unittest.SkipTest("scripts/stooq_topup.py отсутствует (деплой-образ)")
    spec = importlib.util.spec_from_file_location("stooq_topup", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ResponseParsingTest(unittest.TestCase):

    def test_a_good_day_becomes_a_daily_slice_row(self) -> None:
        row = _load().to_row("SPY.US", 20260828, _GOOD_CSV)
        self.assertEqual(
            row, "SPY.US,D,20260828,000000,55.1,55.9,54.8,55.4,1234567,0")

    def test_a_rate_limit_page_never_becomes_data(self) -> None:
        """🔴 Главная проверка: отказ источника — текст с кодом 200."""
        for junk in ("Exceeded the daily hits limit", "", "<html>captcha</html>"):
            self.assertIsNone(_load().to_row("SPY.US", 20260828, junk), junk[:20])

    def test_another_day_is_not_taken(self) -> None:
        other = "Date,Open,High,Low,Close,Volume\n2026-08-27,1,1,1,1,1\n"
        self.assertIsNone(_load().to_row("SPY.US", 20260828, other))


class MissingSelectionTest(unittest.TestCase):

    def setUp(self) -> None:                                       # noqa: N802
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.db = self.root / "prices.sqlite"
        conn = si.connect(self.db)
        si.ensure_schema(conn)
        cache: dict = {}
        for symbol, days in (("AAPL.US", (20260827, 20260828)),
                             ("SPY.US", (20260827,)),
                             ("BTC.V", (20260827,))):
            iid = si._instrument_id(conn, symbol, cache, allow_new=True)
            for day in days:
                conn.execute(
                    "INSERT INTO daily_bars(instrument_id,trade_date,open,high,"
                    "low,close,volume) VALUES(?,?,?,?,?,?,?)",
                    (iid, day, 10.0, 10.1, 9.9, 10.0, 1000.0))
        conn.commit()
        conn.close()

    def test_only_papers_of_markets_that_traded_are_listed(self) -> None:
        self.assertEqual(_load().missing(str(self.db), 20260828), ["SPY.US"])

    def test_a_full_day_lists_nobody(self) -> None:
        self.assertEqual(_load().missing(str(self.db), 20260827), [])

    def test_the_appended_row_is_accepted_by_the_real_parser(self) -> None:
        """Формат, проверенный глазами, — не доказательство."""
        daily = self.root / "20260828_d.txt"
        daily.write_text(
            "\n".join([_HEADER, "AAPL.US,D,20260828,000000,10,11,9,10.5,500,0"])
            + "\n", encoding="utf-8")
        with open(daily, "a", encoding="utf-8") as handle:
            handle.write(_load().to_row("SPY.US", 20260828, _GOOD_CSV) + "\n")

        conn = si.connect(self.db)
        try:
            result = si.apply_batch(
                conn, si.parse_daily_file(daily, min_us_rows=0))
            self.assertEqual(result.rows_written, 2)
            row = conn.execute(
                "SELECT b.trade_date, b.close FROM daily_bars b "
                "JOIN instruments i ON i.id = b.instrument_id "
                "WHERE i.source_symbol='SPY.US' "
                "ORDER BY b.trade_date DESC LIMIT 1").fetchone()
            self.assertEqual(int(row["trade_date"]), 20260828)
            self.assertAlmostEqual(float(row["close"]), 55.4)
        finally:
            conn.close()


if __name__ == "__main__":                                         # pragma: no cover
    unittest.main()
