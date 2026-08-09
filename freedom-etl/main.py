"""Точка входа freedom-etl: планировщик APScheduler и ручные режимы CLI.

Режимы::

    python main.py                 демон: применить схему и ждать крона
    python main.py --once          один прогон синхронизации и выход
    python main.py --init-db       только применить схему (миграции) и выйти
    python main.py --seed AAPL.US,MSFT.US   добавить тикеры в универсум

Почему расписание задано зоной, а не UTC-часом
──────────────────────────────────────────────
Требование ТЗ — 03:00 EST (пост-маркет США закрыт, клиринг брокера прошёл;
≈12:00 по Астане). Записать это как `0 8 * * *` в UTC нельзя: США переходят
на летнее время, и дважды в год такой крон уезжал бы на час — в марте прогон
начинался бы до окончания клиринга и записывал неполный день.
`CronTrigger(timezone="America/New_York")` держит именно локальные 03:00.

Почему `--once` существует
──────────────────────────
Первичная загрузка (5 лет × весь универсум) идёт десятки минут и запускается
руками, а не ожиданием ночи. Тот же режим — единственный способ догнать
пропущенный прогон после простоя, не дожидаясь следующих суток.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.config import Settings
from db.connection import Database
from services.auth import TradernetAuth
from services.etl_processor import EtlProcessor, SyncReport
from services.tradernet_client import TradernetClient

logger = logging.getLogger("freedom-etl")


def load_dotenv_if_present() -> None:
    """Подтянуть `.env` при локальном запуске.

    Импорт необязательный: в контейнере переменные приходят через `env_file`
    docker-compose, и тащить туда зависимость ради ненужного файла незачем.
    `override=False` — уже выставленное окружение сильнее файла, иначе
    забытый локальный `.env` тихо перебил бы переменные docker-compose.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    # APScheduler на INFO пишет по строке на каждое срабатывание и на каждый
    # пересчёт следующего запуска — в суточном кроне это шум поверх полезного лога.
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


async def run_sync_once(settings: Settings, db: Database) -> SyncReport:
    """Один полный прогон T-1 синхронизации на уже поднятом пуле БД."""
    settings.require_credentials()
    auth = TradernetAuth(settings.public_key, settings.secret_key)

    async with TradernetClient(
        auth,
        base_url=settings.api_base_url,
        batch_size=settings.batch_size,
        batch_delay=settings.batch_delay_seconds,
        max_retries=settings.max_retries,
        backoff_base=settings.backoff_base_seconds,
        backoff_max=settings.backoff_max_seconds,
        timeout=settings.request_timeout_seconds,
    ) as client:
        return await EtlProcessor(db, client, settings).run_daily_sync()


async def _guarded_sync(settings: Settings, db: Database) -> None:
    """Обёртка для крона: исключение НЕ должно убивать планировщик.

    Незаловленная ошибка в job'е APScheduler снимает саму задачу — сервис
    остался бы «живым» контейнером, который больше никогда ничего не загрузит,
    и заметили бы это только по устаревшим данным недели спустя.
    """
    try:
        await run_sync_once(settings, db)
    except Exception:
        logger.exception("Плановая синхронизация упала — расписание сохранено")


async def serve(settings: Settings) -> None:
    """Демон: схема, планировщик, ожидание сигнала остановки."""
    async with Database(
        settings.db_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        command_timeout=settings.db_command_timeout,
    ) as db:
        await db.apply_schema()
        if settings.seed_tickers:
            await db.seed_tickers(settings.seed_tickers)

        scheduler = AsyncIOScheduler(timezone=settings.schedule_timezone)
        scheduler.add_job(
            _guarded_sync,
            trigger=CronTrigger(
                hour=settings.schedule_hour,
                minute=settings.schedule_minute,
                timezone=settings.schedule_timezone,
            ),
            args=[settings, db],
            id="daily_quotes_sync",
            # Пропущенный прогон (контейнер лежал) должен состояться при
            # старте, а не ждать сутки: `misfire_grace_time` даёт на это час.
            misfire_grace_time=3600,
            # Прошлый прогон ещё идёт (первичная загрузка) — второй не
            # запускается: две копии писали бы одни строки и жгли лимит
            # запросов на одном и том же ключе.
            max_instances=1,
            coalesce=True,
        )
        scheduler.start()
        logger.info(
            "Планировщик запущен: ежедневно в %02d:%02d %s. Конфигурация: %s",
            settings.schedule_hour, settings.schedule_minute,
            settings.schedule_timezone, settings.describe(),
        )

        if settings.run_on_start:
            await _guarded_sync(settings, db)

        await _wait_for_shutdown()
        scheduler.shutdown(wait=False)
        logger.info("Остановка по сигналу — планировщик выключен")


async def _wait_for_shutdown() -> None:
    """Ждать SIGTERM/SIGINT.

    SIGTERM обязателен: именно его шлёт `docker stop`, и без обработчика
    контейнер умирал бы по таймауту KILL, оборвав транзакцию записи батча.
    """
    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(
                sig, lambda: stop.done() or stop.set_result(None))
        except NotImplementedError:      # pragma: no cover — Windows
            signal.signal(sig, lambda *_: stop.done() or stop.set_result(None))

    await stop


async def _amain(args: argparse.Namespace) -> int:
    load_dotenv_if_present()
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    if args.init_db or args.seed or args.once:
        async with Database(
            settings.db_url,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            command_timeout=settings.db_command_timeout,
        ) as db:
            await db.apply_schema()

            seed = list(settings.seed_tickers)
            if args.seed:
                seed += [t.strip() for t in args.seed.replace(";", ",").split(",")]
            if seed:
                await db.seed_tickers(seed)

            if args.once:
                report = await run_sync_once(settings, db)
                logger.info("Итог прогона: %s", report.summary())
                # Ненулевой код возврата нужен планировщику снаружи (cron,
                # Cloud Scheduler): без него частичный отказ выглядел бы
                # успехом и не поднял бы алерт.
                return 0 if report.ok else 1
        return 0

    await serve(settings)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="freedom-etl",
        description="Сбор дневных котировок (T-1) Tradernet → TimescaleDB",
    )
    parser.add_argument("--once", action="store_true",
                        help="один прогон синхронизации и выход")
    parser.add_argument("--init-db", action="store_true",
                        help="применить схему БД и выйти")
    parser.add_argument("--seed", metavar="TICKERS",
                        help="добавить тикеры в универсум через запятую")
    args = parser.parse_args()

    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        # Сюда приходит отсутствие ключей из require_credentials — это
        # конфигурационная ошибка, и стек по ней не нужен.
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
