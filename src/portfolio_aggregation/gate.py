"""Гейт I-15: агрегированный отчёт только после ЖИВОГО фетча по ключам vault.

`docs/roadmap/manual_portfolio/PHASE_00_CONTRACT.md §2, I-15`. Данные Tradernet
для ручных тикеров законны ровно потому, что в ЭТОМ ЖЕ запросе брокер принял
личные ключи пользователя — это доказательство статуса клиента для I-12. Мок,
кэш, сервисный ключ у не-админа доказательством не являются.

Гейт стоит ДО слияния и ДО создания менеджера с `price_source="aggregated"`:
после слияния маркер `_ramp_is_fallback` уже не найти (`pd.concat` теряет
`attrs`), а менеджер с этим источником создаёт клиента Tradernet.
"""

from __future__ import annotations

from typing import Callable

from .sources import KEY_ORIGINS, SourceResult

#: Источник портфеля, под которым ценовой слой отдаёт Tradernet для ВСЕХ колонок.
AGGREGATED_SOURCE = "aggregated"


class AggregatedNotPermitted(RuntimeError):
    """Нет доказательства живого фетча — агрегированный отчёт не строится."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"I-15: агрегированный отчёт невозможен ({reason})")
        self.reason = reason


def live_broker_proof_failure(freedom: SourceResult | None) -> str | None:
    """Почему доказательства НЕТ, либо `None`, если оно есть.

    Проверяется всё, что можно проверить по результату: имя источника, успех,
    признак живого фетча, происхождение ключей и отсутствие маркеров мока.
    """
    if freedom is None:
        return "брокер не запрашивался"
    if freedom.name != "freedom":
        return "источник не брокер"
    if not freedom.ok:
        return f"брокер недоступен: {freedom.failure_reason or 'unknown'}"
    if not freedom.live:
        return "фетч не живой"
    if freedom.key_origin not in KEY_ORIGINS:
        return "ключи не из vault"
    attrs = getattr(freedom.frame, "attrs", {}) or {}
    if attrs.get("_ramp_is_fallback") or attrs.get("_ramp_is_mock"):
        return "в фрейме маркер мока"
    if freedom.frame is None or freedom.frame.empty:
        return "счёт пуст"
    return None


def require_live_broker_fetch(freedom: SourceResult | None) -> None:
    """Поднять `AggregatedNotPermitted`, если доказательства I-15 нет."""
    reason = live_broker_proof_failure(freedom)
    if reason is not None:
        raise AggregatedNotPermitted(reason)


def aggregated_manager(freedom: SourceResult | None, factory: Callable):
    """Создать менеджер с `price_source="aggregated"` — ТОЛЬКО за гейтом.

    `factory` — `UniversalPortfolioManager` (передаётся вызывающим, чтобы слой
    данных не импортировал движок). Без доказательства фабрика не вызывается
    вовсе: клиент Tradernet не создаётся даже лениво.
    """
    require_live_broker_fetch(freedom)
    return factory(price_source=AGGREGATED_SOURCE)


__all__ = [
    "AGGREGATED_SOURCE",
    "AggregatedNotPermitted",
    "aggregated_manager",
    "live_broker_proof_failure",
    "require_live_broker_fetch",
]
