"""Флаги и лимиты гибридного портфеля — ФУНКЦИЯМИ, а не константами модуля.

Константа защёлкнулась бы на импорте, и тест не смог бы проверить оба
состояния бота (правило §6.4 плана ручного ввода, образец —
`tg_bot.manual_portfolio_enabled`). Числа из env — только через
`env_config.env_int`: голый `int(os.getenv(...))` роняет импорт (`CLAUDE.md`).
"""

from __future__ import annotations

import os

from env_config import env_int

#: Флаг меню источников и агрегированного отчёта. Дефолт — ВЫКЛЮЧЕН (I-9):
#: при выключенном флаге бот ведёт себя ровно как до гибрида. Флаг требует
#: включённого ручного ввода — это проверяет вызывающий (`tg_bot`), потому что
#: флаг ручного ввода принадлежит ему.
HYBRID_PORTFOLIO_ENV = "HYBRID_PORTFOLIO_ENABLED"

_TRUE = ("1", "true", "yes", "on")


def hybrid_flag_on() -> bool:
    """Сырой флаг гибрида (без учёта флага ручного ввода)."""
    return str(os.getenv(HYBRID_PORTFOLIO_ENV, "off")).strip().lower() in _TRUE


def manual_max_positions() -> int:
    """Потолок позиций СОХРАНЁННОГО ручного портфеля (S-7)."""
    return env_int("MANUAL_MAX_POSITIONS", 50, lo=1, hi=500)


def aggregated_max_positions() -> int:
    """Потолок строк агрегированного фрейма ДО склейки дублей (S-7)."""
    return env_int("AGGREGATED_MAX_POSITIONS", 150, lo=1, hi=1000)


def broker_fetch_budget_s() -> int:
    """Общий бюджет ожидания брокера, секунды (D-8)."""
    return env_int("BROKER_FETCH_BUDGET_S", 60, lo=10, hi=180)


__all__ = [
    "HYBRID_PORTFOLIO_ENV",
    "aggregated_max_positions",
    "broker_fetch_budget_s",
    "hybrid_flag_on",
    "manual_max_positions",
]
