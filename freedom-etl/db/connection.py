"""Пул соединений asyncpg и все запросы к хранилищу.

Почему весь SQL заперт в этом модуле
────────────────────────────────────
`etl_processor` — бизнес-логика T-1; если она начнёт сама писать SQL, то
инвариант идемпотентности (UPSERT по `(ticker, trade_date)`) размажется по
двум слоям, и первый же «быстрый фикс» в процессоре вернёт дубликаты. Здесь
граница жёсткая: наружу торчат методы доменного языка (`fetch_sync_state`,
`upsert_candles`), внутрь не заглядывает никто.

Почему пул, а не одно соединение
─────────────────────────────────
Загрузка идёт батчами по 50 тикеров, и запись очередного батча в БД обязана
идти ПАРАЛЛЕЛЬНО ожиданию следующего HTTP-ответа — иначе двухсекундная пауза
между батчами складывается с временем записи, и полный прогон удлиняется
пропорционально числу батчей. Пул из 5 соединений это позволяет, оставаясь
далеко от лимитов PostgreSQL.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable, Sequence

import asyncpg

from db.models import (
    HYPERTABLE_STATEMENTS,
    SCHEMA_STATEMENTS,
    UPSERT_CANDLES_SQL,
    UPSERT_SYNC_STATUS_SQL,
    Candle,
)

logger = logging.getLogger(__name__)

__all__ = ["Database"]


class Database:
    """Тонкая обёртка над `asyncpg.Pool` с доменными операциями ETL."""

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 5,
                 command_timeout: float = 60.0) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._command_timeout = command_timeout
        self._pool: asyncpg.Pool | None = None

    # ── Жизненный цикл ───────────────────────────────────────────────────────

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database.connect() не был вызван")
        return self._pool

    async def connect(self) -> "Database":
        if self._pool is not None:
            return self
        # max_size не может быть меньше min_size — asyncpg падает на этом с
        # невнятной ошибкой, а конфиг приходит из env, где легко разъехаться.
        max_size = max(self._max_size, self._min_size)
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_size,
            max_size=max_size,
            command_timeout=self._command_timeout,
        )
        logger.info("Пул PostgreSQL поднят (min=%d max=%d)", self._min_size, max_size)
        return self

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Пул PostgreSQL закрыт")

    async def __aenter__(self) -> "Database":
        return await self.connect()

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    # ── Схема ────────────────────────────────────────────────────────────────

    async def apply_schema(self) -> bool:
        """Применить DDL. Возвращает True, если гипертаблица TimescaleDB активна.

        Базовый DDL обязателен; шаги TimescaleDB — нет. На чистом PostgreSQL
        (локальная разработка, CI без образа timescale) расширение отсутствует,
        и жёсткое падение здесь означало бы, что схему нельзя даже посмотреть
        без полного стека. Данные при этом сохраняются те же — теряется только
        партиционирование, и об этом честно пишется в лог.
        """
        async with self.pool.acquire() as conn:
            for statement in SCHEMA_STATEMENTS:
                await conn.execute(statement)
        logger.info("Базовая схема применена (%d операторов)", len(SCHEMA_STATEMENTS))

        try:
            async with self.pool.acquire() as conn:
                for statement in HYPERTABLE_STATEMENTS:
                    await conn.execute(statement)
        except asyncpg.PostgresError as exc:
            logger.warning(
                "TimescaleDB недоступна (%s) — daily_candles остаётся обычной "
                "таблицей PostgreSQL. Данные и запросы работают, партиционирования нет.",
                exc,
            )
            return False

        logger.info("Гипертаблица daily_candles готова (chunk = 365 дней)")
        return True

    # ── Универсум тикеров ────────────────────────────────────────────────────

    async def fetch_tracked_tickers(self) -> list[str]:
        """Активные тикеры универсума, в детерминированном порядке.

        Порядок важен не косметически: батчи нарезаются по этому списку, и
        стабильная сортировка делает состав батча воспроизводимым — иначе
        разбор инцидента «батч №7 упал» невозможен, батч №7 каждый раз разный.
        """
        rows = await self.pool.fetch(
            "SELECT ticker FROM tracked_tickers WHERE is_active ORDER BY ticker"
        )
        return [r["ticker"] for r in rows]

    async def seed_tickers(self, tickers: Iterable[str]) -> int:
        """Добавить тикеры в универсум. Идемпотентно; уже существующие не трогает.

        `DO NOTHING`, а не `DO UPDATE`: повторный seed не должен воскрешать
        бумагу, которую руками пометили `is_active = FALSE` (делистинг).
        """
        payload = [(t.strip().upper(), _exchange_of(t)) for t in tickers if t and t.strip()]
        if not payload:
            return 0
        result = await self.pool.executemany(
            """
            INSERT INTO tracked_tickers (ticker, exchange)
            VALUES ($1, $2)
            ON CONFLICT (ticker) DO NOTHING
            """,
            payload,
        )
        logger.info("Seed универсума: подано %d тикеров (%s)", len(payload), result or "ok")
        return len(payload)

    # ── Курсор синхронизации ─────────────────────────────────────────────────

    async def fetch_sync_state(self) -> dict[str, date | None]:
        """`{тикер: последняя загруженная дата}` по всей таблице `sync_status`.

        Читается ОДНИМ запросом, а не по тикеру в цикле: при универсуме в
        500 бумаг поштучное чтение — 500 round-trip'ов ради 500 дат.
        """
        rows = await self.pool.fetch("SELECT ticker, last_synced_date FROM sync_status")
        return {r["ticker"]: r["last_synced_date"] for r in rows}

    async def upsert_sync_status(self, states: Sequence[tuple[str, date]]) -> int:
        if not states:
            return 0
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(UPSERT_SYNC_STATUS_SQL, list(states))
        return len(states)

    # ── Свечи ────────────────────────────────────────────────────────────────

    async def upsert_candles(self, candles: Sequence[Candle]) -> int:
        """Пакетная идемпотентная запись свечей. Возвращает число поданных строк.

        Всё в ОДНОЙ транзакции на батч: при обрыве соединения посреди записи
        в БД не должно остаться половины батча, иначе `sync_status`, который
        обновится следом, объявит загруженным день, которого нет.
        """
        if not candles:
            return 0
        rows = [c.as_row() for c in candles]
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(UPSERT_CANDLES_SQL, rows)
        return len(rows)

    async def count_candles(self, ticker: str | None = None) -> int:
        """Число строк — используется смоук-проверкой и логом итогов прогона."""
        if ticker is None:
            return int(await self.pool.fetchval("SELECT count(*) FROM daily_candles"))
        return int(await self.pool.fetchval(
            "SELECT count(*) FROM daily_candles WHERE ticker = $1", ticker))


def _exchange_of(ticker: str) -> str | None:
    """Биржа по суффиксу тикера Tradernet (`AAPL.US` → `US`, `KEGC.KZ` → `KZ`).

    Только для человекочитаемой колонки `tracked_tickers.exchange`. Решения по
    ней НЕ принимаются — часовой пояс определяется в `etl_processor` своей
    функцией, чтобы «справочная» колонка и логика торговых дат не разъехались
    молча, если суффикс окажется неизвестным.
    """
    _, _, suffix = ticker.strip().upper().partition(".")
    return suffix or None
