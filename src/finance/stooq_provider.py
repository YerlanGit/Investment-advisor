"""
Провайдер цен поверх локальной базы Stooq (MP-09, ЧК-08.5-4).

Чем он оказался, и чем НЕ оказался
──────────────────────────────────
Фаза проектировалась как живой HTTP-провайдер: запрос к `stooq.com` на каждый
тикер в момент построения отчёта, отсюда весь блок требований S-1…S-8.  Затем
выяснилось, что `stooq.com/db/` отдаёт историю целиком одним архивом, и форма
фазы изменилась (`PHASE_09`, врезка «ПЕРЕСМОТР 2026-08-09»).  Сегодня провайдер
**не ходит в сеть вообще**: он читает SQLite, собранную оператором.

Что из этого следует по пунктам:

* **S-1** (whitelist тикера), **S-2** (редиректы), **S-3** (таймаут и потолок
  тела), **S-4** (запрет `read_csv(url)`), **S-5** (маска `apikey`) — на горячем
  пути их нет, потому что нет ни URL, ни ключа, ни тела ответа.  Инъекция через
  тикер невозможна: имя уходит в **параметризованный** SQL, а не в строку;
* **S-6** (параллелизм), **S-8** (циркуит-брейкер) — нечего ограничивать и
  нечего размыкать: один локальный файл, открытый `mode=ro`;
* **S-7** (кэш) — остаётся у слоя выше и продолжает ключеваться провайдером.

Конвенция
─────────
`SPLIT_ADJUSTED` — и объявляется она про **ВЫХОД** провайдера, а не про то, что
лежит в файле.  `apply_split_conventions` идемпотентна: оба её слоя сами ищут
ступеньку и без неё молчат, поэтому сырой и уже скорректированный ряд дают
побитово один результат (`AUDIT §−83`).  Ровно на этом основании ту же
конвенцию объявляет `TradernetProvider`, чей источник её тоже не документирует.

🔴 Ось дивидендов (`SPLIT_ADJUSTED` против `TOTAL_RETURN`) не измерена ни у
Stooq, ни у Tradernet.  Это записанное ограничение, а не забытый шаг:
`STOOQ_CONVENTION §4`.

Слой
────
L1 Data.  Провайдер НЕ знает про отчёт, про aiogram и про источник портфеля;
решение «кому какой провайдер положен» принимает `price_providers`.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from env_config import env_int
from finance.price_providers import PriceConvention, ProviderResult
from finance.stooq_store import StooqStore, StoreUnavailable

logger = logging.getLogger(__name__)

#: Сколько ТОРГОВЫХ дней рынка бумага вправе молчать, оставаясь пригодной.
#: Пять — неделя торгов: бумага, пропустившая её целиком, либо делистингована,
#: либо перестала торговаться, и её последняя цена уже не «сегодняшняя».
#: Замер `tph.us`: последний бар 2026-05-13, в августовских срезах бумаги нет.
MAX_STALE_TRADING_DAYS = env_int("STOOQ_MAX_STALE_TRADING_DAYS", 5, lo=0, hi=250)

#: Сколько КАЛЕНДАРНЫХ дней вправе не обновляться рынок целиком.
#:
#: 🔴 Отдельный порог, потому что потикерная свежесть этот случай не видит ПО
#: ПОСТРОЕНИЮ (`AUDIT §−80`, дефект 2): календарь рынка выводится из тех же
#: строк, что и бары, поэтому у не наполняемой неделю базы КАЖДАЯ бумага честно
#: рапортует свежесть 0.  Разорвать круг могут только часы снаружи.
#: Семь дней покрывают длинные выходные (максимум 4 календарных дня подряд без
#: торгов в США) и оставляют оператору запас на один пропущенный день.
MAX_MARKET_STALE_DAYS = env_int("STOOQ_MAX_MARKET_STALE_DAYS", 7, lo=1, hi=365)


class StooqProvider:
    """Читатель локальной базы котировок по протоколу `PriceProvider`."""

    name = "stooq"
    convention = PriceConvention.SPLIT_ADJUSTED
    #: Параллелизм не нужен и вреден: источник — один файл SQLite на локальном
    #: диске.  Число объявлено явно, потому что протокол его требует, а
    #: наследовать `max_workers=6` от брокерского провайдера здесь бессмысленно.
    max_parallelism = 1

    def __init__(self, db_path=None, *, store: Optional[StooqStore] = None,
                 today=None) -> None:
        """`store` и `today` — швы для тестов; в проде оба `None`."""
        self._db_path = db_path
        self._store = store
        self._today = today

    # ── протокол ─────────────────────────────────────────────────────────

    def fetch(self, tickers: list[str], *, days: int) -> ProviderResult:
        """Матрица цен из базы.  Сети нет; исключений наружу нет.

        Отсутствие базы — это `failed` по всем тикерам, а не исключение:
        протокол требует, чтобы пустой ответ по тикеру был отказом, а `fetch`
        зовётся из середины конвейера отчёта, где `RuntimeError` превратился бы
        в traceback вместо внятной причины.  Пустая матрица дальше упирается в
        чекеры качества (C-1/C-8), и токен не списывается.
        """
        wanted = [str(t) for t in tickers]
        own_store = self._store is None
        try:
            store = self._store or StooqStore.open_readonly(self._db_path)
        except StoreUnavailable as exc:
            logger.error("База котировок Stooq недоступна: %s", exc)
            return self._all_failed(wanted, f"база котировок недоступна: {exc}")

        try:
            return self._fetch_from(store, wanted, days=days)
        finally:
            if own_store:
                store.close()

    # ── внутреннее ───────────────────────────────────────────────────────

    def _fetch_from(self, store: StooqStore, tickers: list[str], *,
                    days: int) -> ProviderResult:
        failed: dict[str, str] = {}
        resolved = {}
        for ticker in tickers:
            found = store.resolve(ticker)
            if found is None:
                failed[ticker] = "нет в базе котировок Stooq"
                continue
            resolved[ticker] = found

        stalled = self._stalled_markets(store, resolved)
        if stalled:
            # Рынок встал целиком — это отказ ПРОВАЙДЕРА, а не свойство бумаг.
            # Отдать по нему цены значило бы выдать недельной давности снимок
            # за сегодняшний, причём молча: внутри себя такие данные
            # непротиворечивы.
            for ticker, found in list(resolved.items()):
                if found.market in stalled:
                    failed[ticker] = stalled[found.market]
                    resolved.pop(ticker)

        for ticker, found in list(resolved.items()):
            age = store.staleness_trading_days(ticker)
            if age is not None and age > MAX_STALE_TRADING_DAYS:
                last = store.last_bar(ticker)
                failed[ticker] = (
                    f"нет торгов {age} торговых дн. подряд "
                    f"(последний бар {last.trade_date if last else '—'})")
                resolved.pop(ticker)

        data = store.bars(list(resolved), days=days)
        for ticker in list(resolved):
            if ticker not in data.columns:
                failed[ticker] = "в базе нет баров в запрошенном окне"
        data = self._adjusted(data)

        loaded = [t for t in tickers if t in data.columns]
        logger.info("Stooq: загружено %d из %d, поколение базы %s",
                    len(loaded), len(tickers), store.generation())
        return self._result(data, loaded, failed)

    def _stalled_markets(self, store: StooqStore, resolved) -> dict[str, str]:
        """Рынки, которые оператор перестал наполнять.  `{рынок: причина}`."""
        out: dict[str, str] = {}
        for found in resolved.values():
            market = found.market
            if market in out:
                continue
            age = store.market_staleness_days(market, today=self._today)
            if age is not None and age > MAX_MARKET_STALE_DAYS:
                latest = store.latest_date(market)
                out[market] = (
                    f"база не обновлялась {age} дн.: последний день рынка "
                    f"{market} — {latest}")
        for market, reason in out.items():
            logger.error("Stooq: %s", reason)
        return out

    @staticmethod
    def _adjusted(data: pd.DataFrame) -> pd.DataFrame:
        """Привести КАЖДУЮ колонку к конвенции провайдера.

        Корректировка живёт здесь, на чтении, а не в загрузчике: база хранит
        ряд как отдал источник (I-6), и это позволяет пересчитать слои, не
        пересобирая базу.  Функция идемпотентна, поэтому повторный проход по
        уже скорректированному ряду ничего не меняет.
        """
        if data.empty:
            return data
        from freedom_portfolio.history import apply_split_conventions

        out = data.copy()
        for column in out.columns:
            out[column] = apply_split_conventions(str(column), out[column])
        return out

    def _result(self, data: pd.DataFrame, loaded: list[str],
                failed: dict[str, str]) -> ProviderResult:
        return ProviderResult(
            data=data, loaded=list(loaded), failed=dict(failed), retried=[],
            convention={t: self.convention for t in loaded},
            source={t: self.name for t in loaded},
        )

    def _all_failed(self, tickers: list[str], reason: str) -> ProviderResult:
        return ProviderResult(
            data=pd.DataFrame(), loaded=[], retried=[],
            failed={t: reason for t in tickers}, convention={}, source={})


__all__ = ["MAX_MARKET_STALE_DAYS", "MAX_STALE_TRADING_DAYS", "StooqProvider"]
