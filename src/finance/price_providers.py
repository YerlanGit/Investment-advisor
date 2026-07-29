"""
Протокол провайдера ценовых данных (Фаза 2 плана ручного портфеля).

Зачем
─────
Источник цен становится сменным, не задевая математику: движок перестаёт знать,
откуда пришла ценовая матрица, и работает с одним контрактом.  Это предпосылка
и для ручного портфеля (`manual` юридически не может ехать на данных Tradernet,
см. `docs/roadmap/manual_portfolio/PHASE_00_CONTRACT.md §3a`), и для устойчивости
(`docs/roadmap/ROADMAP_DATA_RESILIENCE.md §1`).

Два инварианта, вокруг которых собран модуль
────────────────────────────────────────────
**I-11 — duck-совместимость.**  ``ProviderResult`` подставляется на место
``HistoryResult`` без правок у потребителей: объект расходится по шести модулям
(`pdf_payload`, `data_lineage`, `scenario_report`, `portfolio_series`, `tg_bot`,
`investment_logic`) и читается ТОЛЬКО атрибутами.  Поэтому первые четыре поля
сохраняют имена, порядок и семантику ``HistoryResult``, а новые добавлены
строго в хвост и с дефолтами.

**I-13 — один отчёт, один источник.**  Смешивать провайдеров в одной
ковариационной матрице нельзя: у них могут различаться конвенции корректировок,
торговые календари и правила фиксации закрытия.  Failover — только целиком по
отчёту, никогда по тикеру.  ``provider_for_source`` — единственная точка, где
источник портфеля превращается в провайдера, и она же охраняет юридическую
границу (данные Tradernet недопустимы для `manual`).
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import NamedTuple, Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)


class PriceConvention(str, Enum):
    """Конвенция корректировок ряда — обязана объявляться провайдером явно.

    «Не знаю» недопустимо: без конвенции инвариант I-7 (одна конвенция в одной
    матрице) непроверяем.
    """

    RAW = "raw"                        # без корректировок
    SPLIT_ADJUSTED = "split_adjusted"  # текущая конвенция Tradernet
    TOTAL_RETURN = "total_return"      # сплиты + дивиденды
    # Витрина: ряды порождены факторной моделью, а не рынком.  Отдельное
    # значение нужно, чтобы синтетика не притворялась рыночной конвенцией —
    # иначе I-7 «разрешил» бы смешать её с реальными данными.
    SYNTHETIC = "synthetic"


class ProviderResult(NamedTuple):
    """Результат загрузки цен.

    Первые ЧЕТЫРЕ поля повторяют ``HistoryResult`` по имени, порядку и смыслу
    (I-11) — иначе тихо ломаются шесть модулей-потребителей.  Новые поля идут
    строго в хвост и имеют дефолты, поэтому позиционное создание
    ``ProviderResult(df, loaded, failed, retried)`` остаётся валидным.
    """

    data: pd.DataFrame                 # index=DatetimeIndex, columns=tickers
    loaded: list[str]
    failed: dict[str, str]             # ticker → причина
    retried: list[str]
    # ── ниже — то, чего у HistoryResult не было ──
    convention: dict[str, PriceConvention] = {}   # ticker → конвенция
    source: dict[str, str] = {}                   # ticker → имя провайдера

    def single_source(self) -> str | None:
        """Имя источника, если он ровно один (I-13); иначе ``None``."""
        names = {str(v) for v in self.source.values()}
        return names.pop() if len(names) == 1 else None

    def single_convention(self) -> PriceConvention | None:
        """Конвенция, если она ровно одна (I-7); иначе ``None``."""
        kinds = {PriceConvention(v) for v in self.convention.values()}
        return kinds.pop() if len(kinds) == 1 else None


@runtime_checkable
class PriceProvider(Protocol):
    """Контракт источника цен.

    Требования к реализациям:
      • пустой ответ по тикеру — это ``failed[ticker]``, НЕ исключение;
      • ряд отсортирован по дате, без дублей индекса;
      • только ``logger``, никаких ``print``;
      • таймауты и backoff обязательны для сетевых провайдеров;
      • ``convention`` заполняется для КАЖДОГО загруженного тикера.
    """

    name: str
    convention: PriceConvention
    max_parallelism: int               # свой у каждого провайдера

    def fetch(self, tickers: list[str], *, days: int) -> ProviderResult: ...


class ProviderUnavailable(RuntimeError):
    """Источник не может обслужить запрос (не настроен / выключен флагом)."""


def _result_from(data: pd.DataFrame, loaded, failed, retried,
                 *, name: str, convention: PriceConvention) -> ProviderResult:
    """Собрать ``ProviderResult``, проставив источник и конвенцию всем колонкам."""
    loaded = list(loaded)
    return ProviderResult(
        data=data, loaded=loaded, failed=dict(failed), retried=list(retried),
        convention={t: convention for t in loaded},
        source={t: name for t in loaded},
    )


class TradernetProvider:
    """Обёртка над текущим ``freedom_portfolio.history`` — поведение 1:1.

    Ни одна строка логики не переносится: вся работа остаётся в ``history.py``,
    обёртка лишь объявляет метаданные, которых у ``HistoryResult`` нет.

    О конвенции: Tradernet её не документирует, но модуль применяет ``corr=1``,
    затем ``KNOWN_SPLITS``, затем эвристический детектор сплитов — то есть НА
    ВЫХОДЕ ряд скорректирован на сплиты независимо от того, что вернул сервер.
    Дивиденды не корректируются нигде, значит это не ``TOTAL_RETURN``.
    """

    name = "tradernet"
    convention = PriceConvention.SPLIT_ADJUSTED
    max_parallelism = 6                # ровно текущий ThreadPoolExecutor

    def __init__(self, client) -> None:
        self._client = client

    def fetch(self, tickers: list[str], *, days: int) -> ProviderResult:
        from freedom_portfolio import get_history_frame
        hr = get_history_frame(self._client, list(tickers), days=days,
                               max_workers=self.max_parallelism)
        return _result_from(hr.data, hr.loaded, hr.failed, hr.retried,
                            name=self.name, convention=self.convention)


class DemoProvider:
    """Витрина: детерминированные ряды, без сети.

    Существует, чтобы демо шло по тому же контракту, что и остальные источники —
    иначе строка CoVe не сможет подписать витрину, а проверка «один отчёт —
    один источник» не найдёт, что проверять.
    """

    name = "demo"
    convention = PriceConvention.SYNTHETIC
    max_parallelism = 1                # локальная генерация, параллелизм не нужен

    def fetch(self, tickers: list[str], *, days: int) -> ProviderResult:
        from finance.demo_portfolio import demo_price_matrix
        data = demo_price_matrix(list(tickers), days=days)
        missing = [t for t in tickers if t not in data.columns]
        if missing:
            logger.info("Витрина не содержит %d тикер(ов): %s",
                        len(missing), ", ".join(missing))
        return _result_from(data, list(data.columns),
                            {t: "нет в витрине" for t in missing}, [],
                            name=self.name, convention=self.convention)


# ── Выбор провайдера по источнику портфеля ───────────────────────────────────

# I-12: данные Freedom/Tradernet юридически недопустимы в отчёте `manual`
# (пользователь может не быть клиентом брокера, и проверить это нельзя).
_MANUAL_FORBIDDEN = frozenset({"tradernet"})

_REGISTRY: dict[str, type] = {
    "tradernet": TradernetProvider,
    "demo": DemoProvider,
}


def _flag(name: str, default: str) -> str:
    """Флаг читается ФУНКЦИЕЙ, а не константой модуля.

    Константа защёлкнулась бы на импорте, и тест не смог бы её переопределить —
    та же ошибка, которой избегают `beta_shrinkage_enabled` и соседи.
    """
    return str(os.getenv(name, default) or default).strip().lower()


def provider_for_source(source: str, *, client=None) -> PriceProvider:
    """Единственная точка, где источник портфеля превращается в провайдера.

    Здесь же охраняется юридическая граница: попытка выдать `manual` данные
    Tradernet отвергается В РАНТАЙМЕ, а не только в тесте — переменную окружения
    можно выставить в проде, и тихое нарушение недопустимо.  Отказ происходит
    ДО сетевых вызовов и до списания токена.
    """
    src = str(source or "freedom").lower()

    if src == "demo":
        return DemoProvider()

    if src == "manual":
        chosen = _flag("PRICE_PROVIDER_MANUAL", "stooq")
        if chosen in _MANUAL_FORBIDDEN:
            raise ProviderUnavailable(
                f"I-12: источник '{chosen}' недопустим для ручного портфеля — "
                "рыночные данные брокера нельзя показывать не-клиентам."
            )
        impl = _REGISTRY.get(chosen)
        if impl is None:
            # StooqProvider появится в Фазе 9; до этого ручной ввод в проде
            # выключен флагом MANUAL_PORTFOLIO_ENABLED.
            raise ProviderUnavailable(
                f"Провайдер '{chosen}' для ручного портфеля ещё не реализован."
            )
        return impl() if impl is not TradernetProvider else impl(client)

    # freedom / всё остальное — живой брокерский фид
    return TradernetProvider(client)


__all__ = [
    "DemoProvider",
    "PriceConvention",
    "PriceProvider",
    "ProviderResult",
    "ProviderUnavailable",
    "TradernetProvider",
    "provider_for_source",
]
