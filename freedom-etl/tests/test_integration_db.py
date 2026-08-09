"""Интеграционные тесты слоя БД — против ЖИВОГО PostgreSQL.

Запуск:

    ETL_TEST_DSN="postgresql://postgres@/quotes?host=/var/tmp&port=55432" \
        python -m pytest tests/test_integration_db.py -q

Без переменной `ETL_TEST_DSN` весь модуль пропускается — обычный прогон
`pytest tests/` не должен требовать поднятой базы.

Зачем это отдельно от юнит-тестов
─────────────────────────────────
Юнит-тесты проверяют DDL как ТЕКСТ (`"PRIMARY KEY (ticker, trade_date)" in ddl`)
и потому не поймают невалидный SQL, несовпадение типов asyncpg ↔ NUMERIC и —
главное — реальное поведение `ON CONFLICT`. Идемпотентность из ТЗ («повторный
запуск не создаёт дубликатов») доказывается только тем, что после второго
UPSERT в таблице по-прежнему ОДНА строка.

Тесты рассчитаны и на чистый PostgreSQL без расширения TimescaleDB: там
`apply_schema` возвращает False, а данные и запросы работают так же.
"""

from __future__ import annotations

import os
import unittest
from datetime import date
from decimal import Decimal

from db.connection import Database
from db.models import Candle

DSN = os.getenv("ETL_TEST_DSN")


@unittest.skipUnless(DSN, "ETL_TEST_DSN не задан — интеграционные тесты пропущены")
class DatabaseIntegrationTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self) -> None:
        self.db = await Database(DSN).connect()
        await self.db.apply_schema()
        # Чистим только свои тестовые тикеры: база может быть общей.
        await self.db.pool.execute(
            "DELETE FROM daily_candles WHERE ticker LIKE 'ZZTEST%'")
        await self.db.pool.execute(
            "DELETE FROM sync_status WHERE ticker LIKE 'ZZTEST%'")
        await self.db.pool.execute(
            "DELETE FROM tracked_tickers WHERE ticker LIKE 'ZZTEST%'")

    async def asyncTearDown(self) -> None:
        await self.db.close()

    # ── Схема ────────────────────────────────────────────────────────────────

    async def test_apply_schema_is_idempotent(self) -> None:
        """Схема применяется на КАЖДОМ старте контейнера."""
        await self.db.apply_schema()
        await self.db.apply_schema()      # не должно бросить

    async def test_close_is_not_nullable(self) -> None:
        nullable = await self.db.pool.fetchval(
            """SELECT is_nullable FROM information_schema.columns
               WHERE table_name = 'daily_candles' AND column_name = 'close'""")
        self.assertEqual(nullable, "NO")

    async def test_numeric_precision_matches_the_contract(self) -> None:
        row = await self.db.pool.fetchrow(
            """SELECT numeric_precision, numeric_scale FROM information_schema.columns
               WHERE table_name = 'daily_candles' AND column_name = 'close'""")
        self.assertEqual((row["numeric_precision"], row["numeric_scale"]), (12, 4))

    # ── Идемпотентность (ядро ТЗ) ────────────────────────────────────────────

    async def test_rerun_does_not_create_duplicates(self) -> None:
        """🔴 Требование ТЗ: повторный запуск за тот же день не плодит строк."""
        candle = Candle("ZZTEST.US", date(2026, 8, 5), Decimal("191.0000"),
                        Decimal("195.0000"), Decimal("190.0000"),
                        Decimal("194.0000"), 1_000_000)

        await self.db.upsert_candles([candle])
        await self.db.upsert_candles([candle])
        await self.db.upsert_candles([candle])

        self.assertEqual(await self.db.count_candles("ZZTEST.US"), 1)

    async def test_reupsert_refreshes_close(self) -> None:
        """Брокер уточняет свечу после клиринга — DO NOTHING законсервировал бы черновик."""
        await self.db.upsert_candles([
            Candle("ZZTEST.US", date(2026, 8, 5), Decimal("191"), Decimal("195"),
                   Decimal("190"), Decimal("194.0000"), 1_000_000)])
        await self.db.upsert_candles([
            Candle("ZZTEST.US", date(2026, 8, 5), Decimal("191"), Decimal("195"),
                   Decimal("190"), Decimal("194.5000"), 1_100_000)])

        row = await self.db.pool.fetchrow(
            "SELECT close, volume FROM daily_candles WHERE ticker = 'ZZTEST.US'")
        self.assertEqual(row["close"], Decimal("194.5000"))
        self.assertEqual(row["volume"], 1_100_000)

    async def test_partial_reupsert_preserves_existing_ohlc(self) -> None:
        """🔴 Ресинк, отдавший только закрытие, не должен обнулять O/H/L и объём."""
        await self.db.upsert_candles([
            Candle("ZZTEST.US", date(2026, 8, 5), Decimal("191.0000"),
                   Decimal("195.0000"), Decimal("190.0000"),
                   Decimal("194.0000"), 1_000_000)])
        await self.db.upsert_candles([
            Candle("ZZTEST.US", date(2026, 8, 5), None, None, None,
                   Decimal("194.5000"), None)])

        row = await self.db.pool.fetchrow(
            "SELECT open, high, low, volume FROM daily_candles WHERE ticker = 'ZZTEST.US'")
        self.assertEqual(row["open"], Decimal("191.0000"))
        self.assertEqual(row["high"], Decimal("195.0000"))
        self.assertEqual(row["low"], Decimal("190.0000"))
        self.assertEqual(row["volume"], 1_000_000)

    async def test_same_date_different_tickers_coexist(self) -> None:
        """Ключ составной — одна дата у разных бумаг конфликтом не является."""
        await self.db.upsert_candles([
            Candle("ZZTEST.US", date(2026, 8, 5), None, None, None, Decimal("1"), None),
            Candle("ZZTEST2.US", date(2026, 8, 5), None, None, None, Decimal("2"), None),
        ])
        self.assertEqual(await self.db.count_candles("ZZTEST.US"), 1)
        self.assertEqual(await self.db.count_candles("ZZTEST2.US"), 1)

    # ── Курсор ───────────────────────────────────────────────────────────────

    async def test_cursor_advances_forward(self) -> None:
        await self.db.upsert_sync_status([("ZZTEST.US", date(2026, 8, 1))])
        await self.db.upsert_sync_status([("ZZTEST.US", date(2026, 8, 5))])
        state = await self.db.fetch_sync_state()
        self.assertEqual(state["ZZTEST.US"], date(2026, 8, 5))

    async def test_cursor_never_rolls_back(self) -> None:
        """🔴 Откат курсора = бесконечная перезагрузка уже лежащей истории."""
        await self.db.upsert_sync_status([("ZZTEST.US", date(2026, 8, 5))])
        await self.db.upsert_sync_status([("ZZTEST.US", date(2026, 8, 1))])
        state = await self.db.fetch_sync_state()
        self.assertEqual(state["ZZTEST.US"], date(2026, 8, 5),
                         "частичный ответ брокера откатил курсор назад")

    # ── Универсум ────────────────────────────────────────────────────────────

    async def test_seed_is_idempotent(self) -> None:
        await self.db.seed_tickers(["ZZTEST.US", "ZZTEST.KZ"])
        await self.db.seed_tickers(["ZZTEST.US", "ZZTEST.KZ"])
        tickers = [t for t in await self.db.fetch_tracked_tickers() if t.startswith("ZZTEST")]
        self.assertEqual(sorted(tickers), ["ZZTEST.KZ", "ZZTEST.US"])

    async def test_seed_does_not_resurrect_deactivated_ticker(self) -> None:
        """Делистинг помечают вручную; повторный seed не должен его отменять."""
        await self.db.seed_tickers(["ZZTEST.US"])
        await self.db.pool.execute(
            "UPDATE tracked_tickers SET is_active = FALSE WHERE ticker = 'ZZTEST.US'")
        await self.db.seed_tickers(["ZZTEST.US"])

        active = await self.db.fetch_tracked_tickers()
        self.assertNotIn("ZZTEST.US", active)

    async def test_exchange_is_derived_from_suffix(self) -> None:
        await self.db.seed_tickers(["ZZTEST.KZ"])
        exchange = await self.db.pool.fetchval(
            "SELECT exchange FROM tracked_tickers WHERE ticker = 'ZZTEST.KZ'")
        self.assertEqual(exchange, "KZ")

    async def test_tracked_tickers_come_back_sorted(self) -> None:
        """Стабильный порядок делает состав батча воспроизводимым при разборе инцидента."""
        await self.db.seed_tickers(["ZZTESTZ.US", "ZZTESTA.US", "ZZTESTM.US"])
        tickers = [t for t in await self.db.fetch_tracked_tickers() if t.startswith("ZZTEST")]
        self.assertEqual(tickers, sorted(tickers))


if __name__ == "__main__":
    unittest.main()
