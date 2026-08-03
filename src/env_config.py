"""Чтение числовых настроек из окружения — SSOT (A-7).

Зачем отдельный модуль
──────────────────────
Голый `int(os.getenv("X", "48"))` **на уровне модуля** роняет процесс на этапе
ИМПОРТА: опечатка в переменной Cloud Run (`REPORT_URL_TTL_HOURS=48h`) даёт
`ValueError` до того, как заработает логирование, и контейнер не стартует
вообще — без внятного сообщения о причине. Замерено в аудите (`A-7`): три
таких места (`services/report_storage`, `finance/scenario_engine`,
`entrypoint`), каждое — на пути старта бота.

Внутри функций эта же конструкция уже четырежды обёрнута в `try/except` с
фолбэком — то есть правильный образец в проекте был, но копировался руками.
Здесь он один на всех: неверное значение НЕ роняет процесс, а пишется в лог
и заменяется дефолтом. Это осознанный выбор в пользу доступности: отчёт с
дефолтным TTL лучше, чем неподнявшийся сервис.

Модуль намеренно stdlib-only и лежит в корне `src/`, чтобы его могли
импортировать все слои (`services/`, `finance/`, `entrypoint`) без петель.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["env_int", "env_float"]


def env_int(name: str, default: int, *,
            lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    """Целое из env с фолбэком и необязательным клампом.

    Мусор в переменной → предупреждение + `default` (процесс не падает).
    `lo`/`hi` защищают от абсурдных, но синтаксически валидных значений
    (`REPORT_URL_TTL_HOURS=0` сделал бы ссылку мёртвой в момент выдачи).
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
        value = lo
    if hi is not None and value > hi:
        value = hi
    return value
