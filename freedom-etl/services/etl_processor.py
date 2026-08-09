"""Инкрементальная T-1 синхронизация дневных котировок: Extract → Transform → Load.

Контракт стадии
───────────────
Extract    универсум из `tracked_tickers` + курсоры из `sync_status`
           → окна дат → батчи ≤ 50 тикеров → `getHloc`
Transform  UNIX-время → торговая дата в поясе БИРЖИ, отсев мусора, дедупликация
Load       UPSERT в `daily_candles`, затем продвижение `sync_status`

Почему тикеры группируются ПО ДАТЕ, а не просто режутся по 50
──────────────────────────────────────────────────────────────
У `getHloc` одно окно `date_from`/`date_to` на весь батч, а курсоры у тикеров
разные: вчера добавленная бумага требует 5 лет истории, остальные — один день.
Наивная нарезка заставила бы взять минимальную дату по батчу и выкачать 5 лет
на все 50 бумаг: полный прогон вырос бы с секунд до часов, а брокер увидел бы
всплеск, неотличимый от злоупотребления.

Группировка по дате начала делает это бесплатным: в установившемся режиме у
всех тикеров курсор один и тот же, значит бакет ровно один — и нарезка
вырождается в обычные батчи по 50.

Почему `sync_status` двигается ПОСЛЕ записи свечей
───────────────────────────────────────────────────
Обратный порядок теряет данные молча: упади процесс между обновлением курсора
и записью, и день окажется «загруженным», не будучи записанным, — а поскольку
курсор только растёт, никто уже никогда за ним не вернётся. Дыра в ряду
проявится месяцы спустя как необъяснимый скачок доходности.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from config.config import Settings
from db.models import Candle
from services.tradernet_client import RawCandle, TradernetClient, TradernetError, chunked

logger = logging.getLogger(__name__)

__all__ = ["EtlProcessor", "SyncReport", "exchange_timezone", "trade_date_of",
           "to_candles"]


# ── Часовые пояса бирж ───────────────────────────────────────────────────────
# Суффикс тикера Tradernet → пояс биржи. Список намеренно короткий: сюда
# попадают только рынки, которые сервис реально собирает (ТЗ: США и KASE).
# Неизвестный суффикс НЕ угадывается — см. `exchange_timezone`.
_EXCHANGE_TZ: dict[str, str] = {
    "US": "America/New_York",   # NYSE, NASDAQ
    "KZ": "Asia/Almaty",        # KASE, AIX
}
_FALLBACK_TZ = "UTC"

# Кэш объектов ZoneInfo: `ZoneInfo(...)` читает файл базы tzdata, а функция
# зовётся на КАЖДУЮ свечу (сотни тысяч за первичную загрузку).
_TZ_CACHE: dict[str, ZoneInfo] = {}


def exchange_timezone(ticker: str) -> ZoneInfo:
    """Часовой пояс биржи по суффиксу тикера.

    Неизвестный суффикс даёт UTC и предупреждение в лог, а не исключение:
    один незнакомый тикер не должен ронять прогон по всему универсуму. Но и
    молчать нельзя — от пояса зависит, к какому ДНЮ отнесётся свеча, а
    ошибка на день делает ряд доходностей смещённым и правдоподобным
    одновременно, то есть незаметным.
    """
    _, _, suffix = ticker.strip().upper().partition(".")
    name = _EXCHANGE_TZ.get(suffix)
    if name is None:
        logger.warning(
            "Неизвестный суффикс биржи в тикере %r — торговая дата считается по %s. "
            "Добавьте суффикс в _EXCHANGE_TZ, если бумага собирается регулярно.",
            ticker, _FALLBACK_TZ,
        )
        name = _FALLBACK_TZ
    cached = _TZ_CACHE.get(name)
    if cached is None:
        try:
            cached = ZoneInfo(name)
        except ZoneInfoNotFoundError:
            # Образ без пакета tzdata. Падать нельзя, но и делать вид, что
            # даты верны, тоже: UTC-сдвиг относительно Нью-Йорка — 4–5 часов,
            # чего достаточно, чтобы свеча уехала на сутки.
            logger.error(
                "База часовых поясов не найдена (%s) — используется UTC. "
                "Установите пакет tzdata: торговые даты US/KZ будут смещены.", name,
            )
            cached = _TZ_CACHE.setdefault(_FALLBACK_TZ, ZoneInfo("UTC"))
        _TZ_CACHE[name] = cached
    return cached


def trade_date_of(ts: int, tz: ZoneInfo) -> date:
    """UNIX-секунды → календарная торговая дата в поясе биржи.

    Именно здесь ТЗ требует «учитывать часовой пояс биржи». Отметка дневной
    свечи приходит в UTC и для США попадает на 13:30–21:00 UTC; взять от неё
    `.date()` в UTC — почти всегда верно и иногда нет, а «иногда» здесь
    означает целый торговый день не на своём месте.
    """
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(tz).date()


# ── Отчёт о прогоне ──────────────────────────────────────────────────────────


@dataclass
class SyncReport:
    """Итоги одного прогона. Возвращается наружу и печатается в лог."""

    tickers_total: int = 0
    tickers_synced: int = 0
    tickers_no_data: list[str] = field(default_factory=list)
    batches: int = 0
    candles_written: int = 0
    rows_rejected: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return (
            f"тикеров {self.tickers_synced}/{self.tickers_total} · "
            f"батчей {self.batches} · свечей {self.candles_written} · "
            f"отсеяно строк {self.rows_rejected} · "
            f"без данных {len(self.tickers_no_data)} · "
            f"ошибок {len(self.errors)} · {self.duration_seconds:.1f} с"
        )


# ── Transform ────────────────────────────────────────────────────────────────


def to_candles(
    ticker: str,
    raw: Iterable[RawCandle],
    *,
    max_date: date,
    min_date: date | None = None,
    drop_zero_volume: bool = True,
) -> tuple[list[Candle], int]:
    """Привести сырые свечи к контракту БД. Возвращает `(свечи, отсеяно строк)`.

    Отсеивается:

    * свеча без `close` или с `close <= 0` — цены не существует, а колонка
      объявлена NOT NULL; такая строка не «неточная», она пустая;
    * свеча с явным `volume == 0` — торгов в этот день не было, биржа
      перенесла вчерашнее закрытие. Оставить её значит подарить ряду лишний
      НУЛЕВОЙ доход, который занижает σ и все производные от неё метрики;
    * свеча позже `max_date` — незакрытый день (см. `include_current_day`);
    * свеча раньше `min_date`, если он задан.

    🔴 `volume is None` — НЕ то же самое, что `volume == 0`. Часть инструментов
    (неликвид, внебиржевые ноты) приходит вообще без поля объёма, и трактовать
    «не сообщили» как «торгов не было» значит выбросить всю их историю целиком.
    Такие свечи остаются, объём пишется NULL.

    Дедупликация — «побеждает последняя»: в перекрытии ресинка тот же день
    приходит дважды, и вторая версия свежее (брокер уточняет её после клиринга).
    """
    tz = exchange_timezone(ticker)
    by_date: dict[date, Candle] = {}
    rejected = 0

    for bar in raw:
        if not bar.ts:
            rejected += 1
            continue

        close = bar.close
        if close is None or close <= 0:
            rejected += 1
            continue

        if drop_zero_volume and bar.volume is not None and bar.volume <= 0:
            rejected += 1
            continue

        trade_date = trade_date_of(bar.ts, tz)
        if trade_date > max_date or (min_date is not None and trade_date < min_date):
            rejected += 1
            continue

        try:
            candle = Candle(
                ticker=ticker,
                trade_date=trade_date,
                open=_to_decimal(bar.open),
                high=_to_decimal(bar.high),
                low=_to_decimal(bar.low),
                close=_to_decimal(close),          # не None: проверено выше
                volume=int(bar.volume) if bar.volume is not None else None,
            )
        except (InvalidOperation, ValueError, OverflowError):
            # Цена вне NUMERIC(12,4) или нечисловой объём — строка не лезет в
            # схему. Ронять из-за неё весь батч нельзя.
            logger.warning("Свеча %s за %s отвергнута: значения вне схемы", ticker, trade_date)
            rejected += 1
            continue

        by_date[trade_date] = candle

    return [by_date[d] for d in sorted(by_date)], rejected


def _to_decimal(value: float | None) -> Decimal | None:
    """float → NUMERIC(12,4).

    Через `str(...)`, а не `Decimal(float)`: прямая конверсия тащит двоичный
    хвост (`Decimal(19.99)` = 19.98999999...), и `quantize` до 4 знаков даёт
    цену, отличающуюся от присланной в последнем разряде.
    """
    if value is None:
        return None
    result = Decimal(str(value)).quantize(Decimal("0.0001"))
    # NUMERIC(12,4) вмещает 8 знаков до точки. Переполнение здесь — почти
    # наверняка мусор от брокера, но проверить дешевле, чем ловить отказ
    # транзакции на всём батче.
    if abs(result) >= Decimal("100000000"):
        raise ValueError(f"Цена {result} не помещается в NUMERIC(12,4)")
    return result


# ── Процессор ────────────────────────────────────────────────────────────────


class EtlProcessor:
    """Оркестратор T-1 синхронизации.

    Держит ТОЛЬКО порядок стадий и решения о датах; SQL живёт в `db.connection`,
    транспорт — в `services.tradernet_client`. Новая логика добавляется
    стадией или помощником, а не веткой внутри `run_daily_sync`.
    """

    def __init__(self, db, client: TradernetClient, settings: Settings) -> None:
        self._db = db
        self._client = client
        self._settings = settings

    # ── Точка входа ──────────────────────────────────────────────────────────

    async def run_daily_sync(self, *, today: date | None = None) -> SyncReport:
        """Догрузить всё, чего не хватает, до T-1 включительно.

        `today` параметризован ради тестов: прогон, зависящий от системных
        часов, невозможно проверить на границе года и на выходных.
        """
        started = time.monotonic()
        report = SyncReport()

        tickers = await self._db.fetch_tracked_tickers()
        report.tickers_total = len(tickers)
        if not tickers:
            logger.warning(
                "Универсум пуст: таблица tracked_tickers не содержит активных "
                "тикеров. Наполните её через SEED_TICKERS или `main.py --seed`.")
            report.duration_seconds = time.monotonic() - started
            return report

        state = await self._db.fetch_sync_state()
        buckets = self.plan_batches(tickers, state, today=today)

        logger.info(
            "Старт синхронизации: %d тикеров, %d окон загрузки, батч ≤ %d",
            len(tickers), len(buckets), self._client.batch_size,
        )

        for start_date, bucket_tickers in buckets:
            for batch in chunked(bucket_tickers, self._client.batch_size):
                await self._process_batch(batch, start_date, report, today=today)

        report.duration_seconds = time.monotonic() - started
        log = logger.info if report.ok else logger.error
        log("Синхронизация завершена: %s", report.summary())
        return report

    # ── Extract: планирование окон ───────────────────────────────────────────

    def plan_batches(
        self,
        tickers: Sequence[str],
        state: Mapping[str, date | None],
        *,
        today: date | None = None,
    ) -> list[tuple[date, list[str]]]:
        """Сгруппировать тикеры по дате начала загрузки.

        Возвращает `[(дата_начала, [тикеры])]`, отсортированный по дате.

        Тикер выпадает из плана, только когда запрашивать НЕЧЕГО — то есть всё
        его окно (уже с учётом перекрытия) лежит за `cutoff`. При включённом
        перекрытии тикер, доехавший до T-1, в план всё же попадает: перекрытие
        для того и нужно, чтобы перезабрать уточнённые брокером свечи, и стоит
        это ноль — те же тикеры, то же окно, тот же батч.
        """
        anchor = today if today is not None else datetime.now(timezone.utc).date()
        settings = self._settings
        buckets: dict[date, list[str]] = {}

        for ticker in tickers:
            cutoff = self._cutoff_for(ticker, today)
            last = state.get(ticker)

            if last is None:
                start = anchor - timedelta(days=settings.initial_lookback_days)
            else:
                # Перекрытие назад: брокер уточняет последние свечи после
                # клиринга, а UPSERT делает перезапрос безопасным.
                start = last + timedelta(days=1) - timedelta(days=settings.resync_overlap_days)
                if start > cutoff:
                    continue    # уже загружено до T-1 — запрашивать нечего

            buckets.setdefault(start, []).append(ticker)

        return [(start, sorted(buckets[start])) for start in sorted(buckets)]

    def _cutoff_for(self, ticker: str, today: date | None) -> date:
        """Последняя дата, которую МОЖНО записать: T-1 в поясе биржи.

        Когда `today` не задан, «сегодня» берётся по часам САМОЙ биржи, а не
        по UTC: в 03:00 Нью-Йорка в Алматы уже следующий календарный день, и
        общий для всех cutoff отрезал бы у KASE последнюю валидную сессию.

        Явно переданный `today` сильнее системных часов и НЕ корректируется
        поясом — иначе тест не смог бы зафиксировать дату (а прошлая версия
        этого метода как раз молча игнорировала аргумент, совпавший с UTC-датой).
        """
        anchor = today if today is not None else datetime.now(exchange_timezone(ticker)).date()
        return anchor if self._settings.include_current_day else anchor - timedelta(days=1)

    # ── Один батч: Extract → Transform → Load ────────────────────────────────

    async def _process_batch(
        self,
        batch: Sequence[str],
        start_date: date,
        report: SyncReport,
        *,
        today: date | None,
    ) -> None:
        anchor = today if today is not None else datetime.now(timezone.utc).date()
        report.batches += 1

        # `date_to` берётся с запасом в сутки: интервал закрывается по времени
        # биржи, а запрос уходит в UTC-подобном представлении. Лишний день
        # всё равно отрежет `max_date` на стадии Transform — там пояс известен.
        date_from = datetime.combine(start_date, datetime.min.time())
        date_to = datetime.combine(anchor + timedelta(days=1), datetime.min.time())

        try:
            fetched = await self._client.fetch_daily_candles(batch, date_from, date_to)
        except TradernetError as exc:
            # Батч изолирован: отказ по 50 тикерам не должен уносить остальные
            # 450. Курсоры этих тикеров не двигаются, поэтому следующий прогон
            # заберёт тот же интервал заново.
            message = f"батч {batch[0]}…{batch[-1]} ({len(batch)} тикеров): {exc}"
            logger.error("Батч не загружен — %s", message)
            report.errors.append(message)
            return

        candles: list[Candle] = []
        cursors: list[tuple[str, date]] = []

        for ticker in batch:
            bars = fetched.get(ticker)
            if not bars:
                report.tickers_no_data.append(ticker)
                continue

            max_date = self._cutoff_for(ticker, today)
            transformed, rejected = to_candles(
                ticker, bars, max_date=max_date, min_date=start_date)
            report.rows_rejected += rejected

            if not transformed:
                report.tickers_no_data.append(ticker)
                continue

            candles.extend(transformed)
            cursors.append((ticker, transformed[-1].trade_date))

        if not candles:
            return

        # Порядок критичен: сначала свечи, потом курсор. См. шапку модуля.
        written = await self._db.upsert_candles(candles)
        await self._db.upsert_sync_status(cursors)

        report.candles_written += written
        report.tickers_synced += len(cursors)
        logger.info(
            "Батч %d: %d тикеров → %d свечей (окно с %s)",
            report.batches, len(cursors), written, start_date,
        )
