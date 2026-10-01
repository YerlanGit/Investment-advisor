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

#: Режимы раскатки флага (`§−124`). `admins` — ступень между «выключено» и
#: «всем»: фича видна только `ADMIN_USER_IDS`. Нужна, потому что чек-лист
#: включения ручного ввода (`OPERATOR_STOOQ §13.1`) требует отчёта «по своей
#: книге, осмотренного глазами», а без флага в проде владелец не может его
#: построить — и пользователи не должны видеть фичу раньше, чем он это сделал.
FLAG_ON, FLAG_ADMINS, FLAG_OFF = "on", "admins", "off"


def rollout_mode(env_name: str) -> str:
    """`on` | `admins` | `off`. Неизвестное значение — `off` (fail-closed)."""
    raw = str(os.getenv(env_name, FLAG_OFF) or "").strip().lower()
    if raw in _TRUE:
        return FLAG_ON
    if raw in ("admins", "admin"):
        return FLAG_ADMINS
    return FLAG_OFF


def hybrid_flag_on() -> bool:
    """Гибрид включён ДЛЯ ВСЕХ (без учёта флага ручного ввода).

    Пофамильная проверка (`admins`) — в `tg_bot.hybrid_portfolio_enabled`:
    список администраторов принадлежит боту, а не слою данных.
    """
    return rollout_mode(HYBRID_PORTFOLIO_ENV) == FLAG_ON


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
    "FLAG_ADMINS",
    "FLAG_OFF",
    "FLAG_ON",
    "HYBRID_PORTFOLIO_ENV",
    "aggregated_max_positions",
    "broker_fetch_budget_s",
    "hybrid_flag_on",
    "manual_max_positions",
    "rollout_mode",
]
