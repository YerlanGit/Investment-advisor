"""Источники СОСТАВА портфеля: брокер Freedom и ручной ввод (PR-3 §1).

Слой: L1 (данные). Модуль НЕ знает `user_id`: ключи брокера и текст ручного
портфеля достаёт L4 (`tg_bot`) и передаёт сюда значениями — иначе слой данных
импортировал бы vault и БД.

Почему протокол, а не ABC — по образцу `price_providers.PriceProvider`:
источнику достаточно двух атрибутов, наследование ничего не добавляет, а
тестовый дублёр без наследования проще.

🔴 Главное обещание модуля: **fallback-мок брокера наружу не выходит.** При сбое
Tradernet коннектор отдаёт шаблонную книгу с маркером `_ramp_is_fallback`; здесь
она превращается в `ok=False` с ПУСТЫМ фреймом. Маркер нельзя «донести» до
агрегатора: `pd.concat` фреймов с разными `attrs` теряет `attrs` целиком
(проверено), и мок молча стал бы частью «вашего» портфеля (S-1).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

import pandas as pd

# Импорты верхнего уровня, а не ленивые: `load` исполняется на потоке
# executor'а, а первый импорт модуля на потоке — та самая гонка за модульным
# локом, что роняла старт (`§−98`/`§−99`). Здесь всё грузит главный поток
# вместе с `tg_bot`.
from finance.broker_api import (BrokerAuthError, BrokerEmptyPortfolioError,
                                FreedomConnector)
from finance.manual_portfolio import CONTRACT_COLUMNS, parse_portfolio_text

from .flags import manual_max_positions

logger = logging.getLogger(__name__)

#: Причины отказа источника. Первые три — ровно `FreedomConnector.FALLBACK_REASONS`
#: (их тексты пользователю разные, `_broker_outage_advice`), остальные — свои.
FAILURE_REASONS = (
    "waf_block", "api_error", "parse_error",   # fallback-мок брокера (§−94)
    "auth",          # ключи отозваны / неверны
    "timeout",       # превышен бюджет ожидания (D-8)
    "empty",         # счёт пуст / ручной ввод не дал ни одной позиции
    "not_live",      # пришёл мок без причины (демо-ключ, потерянные attrs)
    "error",         # прочее исключение — у результата есть `error_id`
    "too_many",      # ручной портфель длиннее `MANUAL_MAX_POSITIONS` (S-7)
)

#: Кто дал ключи живому фетчу. Это часть доказательства I-15: сервисный ключ
#: не доказывает статус клиента, кроме случая администратора — владельца ключа.
KEY_ORIGIN_VAULT = "vault"
KEY_ORIGIN_ADMIN_SERVICE = "admin_service"
KEY_ORIGINS = (KEY_ORIGIN_VAULT, KEY_ORIGIN_ADMIN_SERVICE)


def count_positions(items) -> int:
    """Сколько РАЗНЫХ бумаг (кэш не считается) — ОДНО правило для правок,
    хранилища и источника (S-7).

    Аудит `§−123`: правки считали разные бумаги, а хранилище и `ManualSource`
    — строки вместе с кэшем. Портфель «50 бумаг + 2 строки кэша» правки
    пропускали, а агрегированный отчёт затем отвергал как «слишком длинный».
    Принимает и `ParsedPosition` (`.ticker`), и `edits.Entry` (`.key`).
    """
    keys = set()
    for item in items or []:
        if getattr(item, "is_cash", False):
            continue
        keys.add(str(getattr(item, "key", None) or getattr(item, "ticker", "")))
    return len(keys)


def empty_frame() -> pd.DataFrame:
    """Пустой контрактный фрейм — то, что получает потребитель при отказе."""
    return pd.DataFrame(columns=CONTRACT_COLUMNS)


@dataclass
class SourceResult:
    """Ответ источника. НИКОГДА не исключение — отказ это `ok=False`.

    `frame` — контракт `CONTRACT_COLUMNS` (+ доп. колонки брокера). При
    `ok=False` фрейм ВСЕГДА пустой: мок не имеет права доехать даже до превью.
    """

    name: str
    frame: pd.DataFrame
    ok: bool
    failure_reason: Optional[str] = None
    #: Код для поддержки при `failure_reason in {"error", fallback-причины}`.
    error_id: Optional[str] = None
    #: Только для брокера: откуда ключи (`vault` / `admin_service`).
    key_origin: Optional[str] = None
    #: True ровно для УСПЕШНОГО живого фетча брокера в этом запросе (I-15).
    live: bool = False
    #: Счётчики без содержимого — их можно логировать (S-3).
    detail: dict = field(default_factory=dict)

    @property
    def positions(self) -> int:
        return 0 if self.frame is None else int(len(self.frame))


def _failed(name: str, reason: str, *, error_id: Optional[str] = None,
            key_origin: Optional[str] = None, **detail) -> SourceResult:
    return SourceResult(name=name, frame=empty_frame(), ok=False,
                        failure_reason=reason, error_id=error_id,
                        key_origin=key_origin, detail=dict(detail))


@runtime_checkable
class PortfolioSource(Protocol):
    """Контракт источника состава. `load` блокирующий — звать в executor."""

    name: str

    def load(self) -> SourceResult: ...


class FreedomSource:
    """Обёртка над `FreedomConnector.fetch_portfolio` — поведение коннектора 1:1.

    `key_origin` передаёт L4: только он знает, взяты ключи из vault
    пользователя или это сервисные ключи администратора.
    """

    name = "freedom"

    def __init__(self, api_key: str, secret_key: str = "", login: str = "", *,
                 key_origin: str = KEY_ORIGIN_VAULT,
                 connector_factory: Optional[Callable] = None) -> None:
        if key_origin not in KEY_ORIGINS:
            raise ValueError(f"неизвестное происхождение ключей: {key_origin!r}")
        self._api_key = str(api_key or "").strip()
        self._secret_key = str(secret_key or "").strip()
        self._login = str(login or "").strip()
        self._key_origin = key_origin
        self._factory = connector_factory

    def __repr__(self) -> str:                       # ключи в repr не печатаются (S-3)
        return f"FreedomSource(key_origin={self._key_origin!r})"

    def _connector(self):
        if self._factory is not None:
            return self._factory(self._api_key, self._secret_key, self._login)
        return FreedomConnector(self._api_key, self._secret_key, self._login)

    def load(self) -> SourceResult:
        origin = self._key_origin
        if not self._api_key:
            # Пустой ключ коннектор подменил бы СЕРВИСНЫМ из env — то есть чужим
            # портфелем. Отказываем здесь, до коннектора.
            return _failed(self.name, "auth", key_origin=origin)
        try:
            df = self._connector().fetch_portfolio()
        except BrokerAuthError:
            return _failed(self.name, "auth", key_origin=origin)
        except BrokerEmptyPortfolioError:
            return _failed(self.name, "empty", key_origin=origin)
        except Exception as exc:                       # noqa: BLE001
            # Пользователю — только код; текст исключения может нести тело
            # ответа брокера (`client._decode`), поэтому в лог — тип, не текст.
            error_id = uuid.uuid4().hex[:12]
            logger.error("HYBRID: брокер упал [%s]: %s", error_id, type(exc).__name__)
            return _failed(self.name, "error", error_id=error_id, key_origin=origin)

        attrs = getattr(df, "attrs", {}) or {}
        if attrs.get("_ramp_is_fallback") or attrs.get("_ramp_is_mock") \
                or attrs.get("_ramp_source") == "demo":
            reason = str(attrs.get("_ramp_fallback_reason") or "not_live")
            if reason not in FAILURE_REASONS:
                reason = "not_live"
            error_id = uuid.uuid4().hex[:12]
            logger.error("HYBRID: брокер отдал fallback-мок reason=%s [%s] — "
                         "мок отброшен, наружу не выходит.", reason, error_id)
            return _failed(self.name, reason, error_id=error_id, key_origin=origin)
        if df is None or df.empty:
            return _failed(self.name, "empty", key_origin=origin)
        return SourceResult(name=self.name, frame=df, ok=True,
                            key_origin=origin, live=True,
                            detail={"positions": int(len(df))})


class ManualSource:
    """Обёртка над `parse_portfolio_text(...).to_dataframe()` — разбор 1:1.

    `engine` — экземпляр `MAC3RiskEngine`: канон тикера спрашивается у него
    (`canonical_ticker`, не `resolve_tickers` — инвариант `CLAUDE.md`).
    """

    name = "manual"

    def __init__(self, text: str, engine, *, max_positions: Optional[int] = None) -> None:
        self._text = str(text or "")
        self._engine = engine
        self._max = max_positions

    def __repr__(self) -> str:                       # текст портфеля не печатается (S-3)
        return f"ManualSource(bytes={len(self._text.encode('utf-8'))})"

    def load(self) -> SourceResult:
        if not self._text.strip():
            return _failed(self.name, "empty", parsed=0, errors=0)
        report = parse_portfolio_text(self._text, self._engine)
        if not report.valid:
            return _failed(self.name, "empty", parsed=0, errors=len(report.failed))
        limit = self._max if self._max is not None else manual_max_positions()
        if count_positions(report.valid) > limit:
            return _failed(self.name, "too_many", parsed=len(report.valid),
                           errors=len(report.failed), limit=limit)
        return SourceResult(name=self.name, frame=report.to_dataframe(), ok=True,
                            detail={"parsed": len(report.valid),
                                    "errors": len(report.failed)})


async def load_with_budget(source: PortfolioSource, budget_s: float) -> SourceResult:
    """Загрузить источник в executor с потолком ожидания (D-8).

    Поток executor'а по таймауту не отменяется — вызов без побочных эффектов,
    а его поздний результат просто отбрасывается. Пользователь получает
    `failure_reason="timeout"`, а не зависший интерфейс.
    """
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, source.load)
    try:
        return await asyncio.wait_for(future, timeout=budget_s)
    except asyncio.TimeoutError:
        logger.warning("HYBRID: источник %s не ответил за %ss — таймаут.",
                       getattr(source, "name", "?"), budget_s)
        return _failed(str(getattr(source, "name", "?")), "timeout",
                       key_origin=getattr(source, "_key_origin", None))


__all__ = [
    "FAILURE_REASONS",
    "FreedomSource",
    "KEY_ORIGINS",
    "KEY_ORIGIN_ADMIN_SERVICE",
    "KEY_ORIGIN_VAULT",
    "ManualSource",
    "PortfolioSource",
    "SourceResult",
    "count_positions",
    "empty_frame",
    "load_with_budget",
]
