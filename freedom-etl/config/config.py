"""Настройки сервиса из окружения — SSOT конфигурации freedom-etl.

Почему разбор env вынесен в отдельный модуль
────────────────────────────────────────────
Голый `int(os.getenv("X", "50"))` **на уровне модуля** роняет процесс на этапе
ИМПОРТА: опечатка в переменной (`BATCH_SIZE=50шт`) даёт `ValueError` до того,
как заработает логирование, и контейнер не стартует вообще — без внятного
сообщения о причине. В основном приложении это уже стоило разбора инцидента и
закреплено AST-сканером (корневой `CLAUDE.md`, «Разбор env на уровне модуля —
ТОЛЬКО через `env_config.env_int`/`env_float`»). Здесь действует то же правило.

Почему это КОПИЯ хелперов, а не импорт `src/env_config.py`
──────────────────────────────────────────────────────────
`freedom-etl` — ОТДЕЛЬНАЯ единица поставки: свой `Dockerfile`, свой
`requirements.txt`, свой контейнер в `docker-compose.yml`. Каталога `src/`
в её образе нет и быть не должно — иначе ETL потянет за собой aiogram,
numpy и весь риск-движок ради двух функций на 40 строк.

Прецедент в репозитории уже есть и он осознанный: `cloud_function/rag_engine.py`
держится ИДЕНТИЧНЫМ `src/agent/rag_engine.py` ровно потому, что это другая
единица поставки (Cloud Function против Cloud Run). Разница лишь в том, что там
копия обязана совпадать дословно, а здесь — не обязана: контракт узкий
(«мусор в env → warning + дефолт»), и он закреплён тестом `test_config.py`.

Все значения читаются ОДИН раз в `Settings.from_env()` и дальше ходят
объектом. Модуль намеренно stdlib-only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["Settings", "env_int", "env_float", "env_bool", "env_str"]


# ── Примитивы чтения env (fail-soft: процесс не падает) ──────────────────────


def env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


def env_int(name: str, default: int, *,
            lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    """Целое из env с фолбэком и необязательным клампом.

    Мусор в переменной → предупреждение + `default` (процесс не падает).
    `lo`/`hi` защищают от абсурдных, но синтаксически валидных значений:
    `BATCH_SIZE=0` увёл бы ETL в бесконечный цикл пустых батчей, а
    `BATCH_SIZE=5000` — в мгновенный бан по rate-limit брокера.
    """
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        value = int(default)
    else:
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            logger.warning(
                "env %s=%r — не целое число, беру значение по умолчанию %s",
                name, raw, default)
            value = int(default)
    if lo is not None and value < lo:
        logger.warning("env %s=%s ниже допустимого %s — поднимаю", name, value, lo)
        value = lo
    if hi is not None and value > hi:
        logger.warning("env %s=%s выше допустимого %s — опускаю", name, value, hi)
        value = hi
    return value


def env_float(name: str, default: float, *,
              lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    """Дробное из env — та же семантика, что у :func:`env_int`."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        value = float(default)
    else:
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            logger.warning(
                "env %s=%r — не число, беру значение по умолчанию %s",
                name, raw, default)
            value = float(default)
    if lo is not None and value < lo:
        logger.warning("env %s=%s ниже допустимого %s — поднимаю", name, value, lo)
        value = lo
    if hi is not None and value > hi:
        logger.warning("env %s=%s выше допустимого %s — опускаю", name, value, hi)
        value = hi
    return value


def env_bool(name: str, default: bool) -> bool:
    """Флаг из env. Принимает 1/true/yes/on в любом регистре."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ── Настройки ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Settings:
    """Полный набор параметров запуска. Иммутабелен — читается один раз на старте."""

    # ── Учётные данные Tradernet (ОДНА учётная запись на весь сервис) ────────
    public_key: str = ""
    secret_key: str = ""
    api_base_url: str = "https://tradernet.com"

    # ── База данных ──────────────────────────────────────────────────────────
    db_url: str = "postgresql://postgres:postgres@localhost:5432/quotes"
    db_pool_min: int = 1
    db_pool_max: int = 5
    db_command_timeout: float = 60.0

    # ── Батчинг и лимиты (см. README §«Лимиты») ──────────────────────────────
    # 50 — потолок из ТЗ; выше брокер начинает резать ответ молча, отдавая
    # часть тикеров без ошибки, что хуже явного 429.
    batch_size: int = 50
    # Пауза МЕЖДУ батчами. Нижняя граница 2.0 с — жёсткое требование лимитов;
    # клампом lo=2.0 сделано так, что уменьшить её через env нельзя вообще:
    # «ускорить выгрузку» — самый вероятный способ получить бан на весь ключ.
    batch_delay_seconds: float = 2.0
    max_retries: int = 5
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 120.0
    request_timeout_seconds: float = 60.0

    # ── Горизонт загрузки ────────────────────────────────────────────────────
    # Первичная загрузка тикера, о котором в `sync_status` ещё ничего нет.
    # 1825 календарных дней ≈ 5 лет — то же окно, что у риск-движка RAMP
    # (`HISTORY_LOOKBACK_DAYS`), чтобы VaR/σ/Шарп считались на той же истории.
    initial_lookback_days: int = 1825
    # Догрузка уже известного тикера идёт с `last_synced_date + 1`, но с
    # перекрытием назад: брокер иногда доправляет вчерашнюю свечу постфактум
    # (корректировка объёма/закрытия после клиринга). UPSERT делает перезапись
    # безопасной, поэтому дешевле перезапросить 3 дня, чем не увидеть правку.
    resync_overlap_days: int = 3
    # Текущий (незакрытый) день в базу НЕ пишется: до закрытия сессии `close`
    # — это последняя сделка, а не цена закрытия, и она отравила бы ряд, по
    # которому считаются риск-метрики. Включать только для отладки.
    include_current_day: bool = False

    # ── Планировщик ──────────────────────────────────────────────────────────
    # 03:00 America/New_York = после пост-маркета США и клиринга брокера
    # (≈12:00 по Астане). Имя зоны, а НЕ фиксированный UTC-сдвиг: US переходит
    # на летнее время, и `0 8 * * *` в UTC уезжал бы на час дважды в год.
    schedule_hour: int = 3
    schedule_minute: int = 0
    schedule_timezone: str = "America/New_York"
    run_on_start: bool = False

    # ── Прочее ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    # Тикеры, которыми наполняется `tracked_tickers` при старте (idempotent).
    seed_tickers: tuple[str, ...] = field(default_factory=tuple)

    # ── Сборка из окружения ──────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "Settings":
        """Собрать настройки из переменных окружения.

        Пустые ключи API здесь НЕ ошибка: `main.py --init-db` и прогон тестов
        должны работать без секретов. Проверка живёт в :meth:`require_credentials`
        и вызывается только на пути, который реально идёт в сеть.
        """
        raw_seed = env_str("SEED_TICKERS", "")
        seed = tuple(
            t.strip().upper()
            for t in raw_seed.replace(";", ",").split(",")
            if t.strip()
        )

        return cls(
            public_key=env_str("TRADERNET_PUBLIC_KEY") or env_str("FREEDOM_API_KEY"),
            secret_key=env_str("TRADERNET_SECRET_KEY") or env_str("FREEDOM_API_SECRET"),
            api_base_url=env_str("TRADERNET_API_URL", "https://tradernet.com").rstrip("/"),

            db_url=env_str("DB_URL", "postgresql://postgres:postgres@localhost:5432/quotes"),
            db_pool_min=env_int("DB_POOL_MIN", 1, lo=1, hi=32),
            db_pool_max=env_int("DB_POOL_MAX", 5, lo=1, hi=64),
            db_command_timeout=env_float("DB_COMMAND_TIMEOUT", 60.0, lo=5.0, hi=600.0),

            batch_size=env_int("BATCH_SIZE", 50, lo=1, hi=50),
            batch_delay_seconds=env_float("BATCH_DELAY_SECONDS", 2.0, lo=2.0, hi=60.0),
            max_retries=env_int("MAX_RETRIES", 5, lo=1, hi=10),
            backoff_base_seconds=env_float("BACKOFF_BASE_SECONDS", 2.0, lo=1.0, hi=30.0),
            backoff_max_seconds=env_float("BACKOFF_MAX_SECONDS", 120.0, lo=5.0, hi=900.0),
            request_timeout_seconds=env_float("REQUEST_TIMEOUT_SECONDS", 60.0, lo=5.0, hi=600.0),

            initial_lookback_days=env_int("INITIAL_LOOKBACK_DAYS", 1825, lo=1, hi=20000),
            resync_overlap_days=env_int("RESYNC_OVERLAP_DAYS", 3, lo=0, hi=90),
            include_current_day=env_bool("INCLUDE_CURRENT_DAY", False),

            schedule_hour=env_int("SCHEDULE_HOUR", 3, lo=0, hi=23),
            schedule_minute=env_int("SCHEDULE_MINUTE", 0, lo=0, hi=59),
            schedule_timezone=env_str("SCHEDULE_TIMEZONE", "America/New_York"),
            run_on_start=env_bool("RUN_ON_START", False),

            log_level=env_str("LOG_LEVEL", "INFO").upper(),
            seed_tickers=seed,
        )

    # ── Валидация ────────────────────────────────────────────────────────────

    def require_credentials(self) -> None:
        """Упасть ЯВНО, если сетевой путь запущен без ключей.

        Отдельный метод, а не проверка в `from_env`, потому что без него
        сервис молча уходил бы в сеть с пустой подписью и получал `code=12`
        по каждому батчу — диагностика «ключей нет» превращалась бы в
        «брокер отверг 40 запросов».
        """
        missing = [
            name for name, value in (
                ("TRADERNET_PUBLIC_KEY", self.public_key),
                ("TRADERNET_SECRET_KEY", self.secret_key),
            ) if not value
        ]
        if missing:
            raise RuntimeError(
                "Не заданы обязательные переменные окружения: "
                + ", ".join(missing)
                + ". Без них Tradernet отвергнет каждый запрос (code=12)."
            )

    def describe(self) -> str:
        """Однострочный снимок конфигурации для лога старта.

        Секреты НЕ печатаются даже префиксом — только факт наличия и длина
        (тот же приём, что в `src/freedom_portfolio/client.py`).
        """
        return (
            f"api={self.api_base_url} key_present={bool(self.public_key)} "
            f"key_len={len(self.public_key)} secret_len={len(self.secret_key)} "
            f"batch={self.batch_size} delay={self.batch_delay_seconds}s "
            f"lookback={self.initial_lookback_days}d overlap={self.resync_overlap_days}d "
            f"include_today={self.include_current_day} "
            f"cron={self.schedule_hour:02d}:{self.schedule_minute:02d} {self.schedule_timezone}"
        )
