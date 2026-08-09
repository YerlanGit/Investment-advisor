"""Схема TimescaleDB: DDL и доменные типы freedom-etl.

Что здесь лежит
───────────────
`daily_candles`   — гипертаблица дневных свечей (OHLCV), ядро хранилища.
`sync_status`     — служебная таблица «докуда доехал тикер» (курсор ETL).
`tracked_tickers` — универсум отслеживаемых бумаг (вход стадии Extract).

Почему `tracked_tickers` добавлена сверх ТЗ
───────────────────────────────────────────
ТЗ описывает две таблицы, но стадия Extract начинается со слов «считать список
всех отслеживаемых тикеров», а читать его неоткуда. Держать универсум в env
нельзя: список меняется чаще, чем передеплоивается сервис, и env не хранит,
КОГДА бумага была добавлена. Поэтому универсум — третья таблица, а `SEED_TICKERS`
остался лишь способом её первичного наполнения (идемпотентного).

Почему первичный ключ именно `(ticker, trade_date)`
────────────────────────────────────────────────────
Это и есть контракт идемпотентности из ТЗ: повторный прогон за тот же день
физически не может создать вторую строку. TimescaleDB требует, чтобы колонка
партиционирования входила в любой UNIQUE/PRIMARY KEY, — `trade_date` входит,
поэтому гипертаблица и естественный ключ не конфликтуют.

Почему чанк — год, а не дефолтные 7 дней
─────────────────────────────────────────
Дефолт TimescaleDB рассчитан на телеметрию с секундным шагом. У дневных свечей
на 7-дневный чанк приходится ≈5 строк на тикер: при универсуме в 500 бумаг
пятилетняя история распалась бы на ~260 чанков, и планировщик тратил бы на их
обход больше, чем на чтение данных. Год держит чанк в разумном размере
(≈250 строк × число тикеров) и совпадает с типичным окном запроса риск-движка.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

__all__ = ["Candle", "SyncState", "SCHEMA_STATEMENTS", "HYPERTABLE_STATEMENTS",
           "UPSERT_CANDLES_SQL", "UPSERT_SYNC_STATUS_SQL"]


# ── Доменные типы ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Candle:
    """Одна дневная свеча, уже приведённая к контракту БД.

    `close` — единственная обязательная цена: ТЗ помечает её NOT NULL, потому
    что на ней держатся все риск-метрики (доходности, σ, VaR). O/H/L и объём
    брокер отдаёт не по каждой бумаге (неликвид, внебиржевые ноты), и терять
    из-за них валидную цену закрытия нельзя.
    """

    ticker: str
    trade_date: date
    open: Decimal | None
    high: Decimal | None
    low: Decimal | None
    close: Decimal
    volume: int | None

    def as_row(self) -> tuple:
        """Кортеж в порядке колонок `UPSERT_CANDLES_SQL` (для `executemany`/`copy`)."""
        return (self.ticker, self.trade_date, self.open, self.high,
                self.low, self.close, self.volume)


@dataclass(frozen=True, slots=True)
class SyncState:
    """Строка `sync_status`: докуда тикер загружен."""

    ticker: str
    last_synced_date: date | None


# ── DDL ──────────────────────────────────────────────────────────────────────
# Каждый оператор идемпотентен (IF NOT EXISTS) — миграции применяются на
# каждом старте, и повторный старт не должен ничего ломать.

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS daily_candles (
        ticker      VARCHAR(20)    NOT NULL,
        trade_date  DATE           NOT NULL,
        open        NUMERIC(12, 4),
        high        NUMERIC(12, 4),
        low         NUMERIC(12, 4),
        close       NUMERIC(12, 4) NOT NULL,
        volume      BIGINT,
        PRIMARY KEY (ticker, trade_date)
    )
    """,
    # Запрос риск-движка всегда имеет вид «ряд по бумаге за окно», то есть
    # WHERE ticker = $1 AND trade_date BETWEEN $2 AND $3. PK (ticker, trade_date)
    # такой запрос уже покрывает; отдельный индекс нужен для ОБРАТНОГО среза
    # «все бумаги за дату» — им ходит проверка полноты дневной загрузки.
    """
    CREATE INDEX IF NOT EXISTS daily_candles_date_idx
        ON daily_candles (trade_date DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS sync_status (
        ticker            VARCHAR(20) PRIMARY KEY,
        last_synced_date  DATE,
        updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tracked_tickers (
        ticker     VARCHAR(20) PRIMARY KEY,
        exchange   VARCHAR(16),
        is_active  BOOLEAN     NOT NULL DEFAULT TRUE,
        added_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
)

# Гипертаблица создаётся ОТДЕЛЬНО от базового DDL: на чистом PostgreSQL без
# расширения эти операторы упадут, и это не должно мешать локальной разработке
# и тестам. `connection.apply_schema` выполняет их в отдельной транзакции и
# при неудаче продолжает с обычной таблицей, записав предупреждение.
HYPERTABLE_STATEMENTS: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE",
    """
    SELECT create_hypertable(
        'daily_candles',
        'trade_date',
        chunk_time_interval => INTERVAL '365 days',
        if_not_exists       => TRUE,
        migrate_data        => TRUE
    )
    """,
)


# ── DML ──────────────────────────────────────────────────────────────────────

# Идемпотентная запись. `DO UPDATE`, а не `DO NOTHING`, намеренно: брокер
# доправляет вчерашнюю свечу после клиринга (уточнённый объём, иногда
# закрытие), и `DO NOTHING` навсегда законсервировал бы первую — черновую —
# версию дня.
#
# COALESCE на O/H/L/volume защищает от обратной беды: если при повторном
# проходе брокер отдал только закрытие, затирать уже сохранённые O/H/L
# значениями NULL нельзя — данные бы деградировали от каждого ресинка.
UPSERT_CANDLES_SQL = """
    INSERT INTO daily_candles (ticker, trade_date, open, high, low, close, volume)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    ON CONFLICT (ticker, trade_date) DO UPDATE SET
        open   = COALESCE(EXCLUDED.open,   daily_candles.open),
        high   = COALESCE(EXCLUDED.high,   daily_candles.high),
        low    = COALESCE(EXCLUDED.low,    daily_candles.low),
        close  = EXCLUDED.close,
        volume = COALESCE(EXCLUDED.volume, daily_candles.volume)
"""

# `GREATEST` не даёт курсору поехать НАЗАД. Иначе один частичный ответ брокера
# (батч отдал не все дни) откатил бы `last_synced_date`, и следующий прогон
# перезапросил бы месяцы уже лежащей истории — тихая деградация в бесконечную
# перезагрузку.
UPSERT_SYNC_STATUS_SQL = """
    INSERT INTO sync_status (ticker, last_synced_date, updated_at)
    VALUES ($1, $2, now())
    ON CONFLICT (ticker) DO UPDATE SET
        last_synced_date = GREATEST(
            EXCLUDED.last_synced_date,
            COALESCE(sync_status.last_synced_date, EXCLUDED.last_synced_date)
        ),
        updated_at = now()
"""
