"""
Чтение локальной базы котировок Stooq (MP-08.5, ЧК-08.5-3).

Роль
────
Единственный способ, которым бот прикасается к базе цен, — и он **только на
чтение**.  Запрет технический, а не договорной: соединение открывается через
URI `mode=ro`, поэтому любая попытка записи поднимает `OperationalError`.
Отсюда следует, что гонок между ботом и загрузчиком нет по построению.

Что модуль СОЗНАТЕЛЬНО не делает
────────────────────────────────
* **не корректирует цены** — конвенция Stooq не измерена
  (`STOOQ_CONVENTION §4`), и применить к уже скорректированному ряду ещё одну
  корректировку значит испортить его молча.  Решение о корректировках
  принимает провайдер (MP-09), когда конвенция станет известна;
* **не ходит в сеть** — ради этого база и заводилась;
* **не решает, что делать с просроченной бумагой** — он лишь честно называет
  её возраст.  Что показать пользователю, решает слой отчёта.

Свежесть — потикерная
─────────────────────
Замер `tph.us`: последний бар 2026-05-13, в срезах за август бумаги нет вовсе,
а база при этом свежая.  Вопрос «когда мы обновлялись» тут даёт неверный
ответ; правильный вопрос — «сколько раз рынок торговал без этой бумаги», и
считается он в ТОРГОВЫХ днях рынка (`finance.market_calendar`).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from finance import market_calendar as mc
from finance.stooq_ingest import SOURCE_NAME, connect, database_path
from finance.stooq_symbols import candidates_for

logger = logging.getLogger(__name__)


class StoreUnavailable(RuntimeError):
    """Базы нет или она непригодна для чтения."""


@dataclass(frozen=True)
class Resolved:
    """Чем именно бумага движка оказалась в базе."""

    engine_ticker: str
    instrument_id: int
    source_symbol: str
    market: str
    currency: str
    match_kind: str


@dataclass(frozen=True)
class LastBar:
    source_symbol: str
    trade_date: int
    close: float


class StooqStore:
    """Читатель базы котировок.  Одно соединение на отчёт."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._sessions: dict[str, list[int]] = {}
        self._resolved: dict[str, Optional[Resolved]] = {}

    # ── жизненный цикл ───────────────────────────────────────────────────

    @classmethod
    def open_readonly(cls, path=None) -> "StooqStore":
        """Открыть базу ТОЛЬКО на чтение.  Отсутствие файла — `StoreUnavailable`.

        Отдельный тип исключения нужен, чтобы вызывающий отличал «база не
        настроена» от «база сломана»: первое — штатное состояние до бутстрапа,
        и провайдер обязан отказать внятно, а не упасть с `sqlite3` в тексте.
        """
        target = Path(path) if path is not None else database_path()
        if not target.exists():
            raise StoreUnavailable(
                f"базы котировок нет по пути {target} — сначала бутстрап "
                "(см. docs/roadmap/manual_portfolio/OPERATOR_STOOQ.md §2)")
        try:
            conn = connect(target, read_only=True)
            conn.execute("SELECT 1 FROM daily_bars LIMIT 1").fetchone()
        except sqlite3.Error as exc:
            raise StoreUnavailable(f"база котировок нечитаема: {exc}") from exc
        return cls(conn)

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:                          # pragma: no cover
            pass

    def __enter__(self) -> "StooqStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ── метаданные ───────────────────────────────────────────────────────

    def generation(self) -> Optional[str]:
        """Отметка поколения базы.

        Её сравнивает контейнер, решая, перекопировать ли файл к себе: без
        такой отметки инстанс, проживший неделю, отдавал бы недельные цены и
        не знал бы об этом (`PHASE_08B §3.2`).
        """
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='generation'").fetchone()
        return str(row["value"]) if row else None

    def instrument_count(self) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) AS n FROM instruments").fetchone()["n"])

    # ── сопоставление тикеров ────────────────────────────────────────────

    def resolve(self, engine_ticker: str) -> Optional[Resolved]:
        """Найти бумагу в базе: сперва по карте, затем по формам символа.

        Порядок важен.  `symbol_map` — это РЕШЕНИЯ человека, включая
        осознанные подмены площадки; автоматический перебор форм таких решений
        принимать не вправе (`stooq_symbols`).  Поэтому карта всегда старше.
        """
        key = str(engine_ticker or "").strip().upper()
        if key in self._resolved:
            return self._resolved[key]

        row = self._conn.execute(
            "SELECT i.id AS id, i.source_symbol AS sym, i.market AS market, "
            "       i.currency AS ccy, m.match_kind AS kind "
            "FROM symbol_map m JOIN instruments i ON i.id = m.instrument_id "
            "WHERE m.engine_ticker = ?", (key,)).fetchone()
        if row is None:
            for candidate in candidates_for(key):
                row = self._conn.execute(
                    "SELECT id, source_symbol AS sym, market, currency AS ccy, "
                    "       'exact' AS kind FROM instruments "
                    "WHERE source=? AND source_symbol=?",
                    (SOURCE_NAME, candidate)).fetchone()
                if row is not None:
                    break

        resolved = None if row is None else Resolved(
            engine_ticker=key, instrument_id=int(row["id"]),
            source_symbol=str(row["sym"]), market=str(row["market"]),
            currency=str(row["ccy"]), match_kind=str(row["kind"]))
        self._resolved[key] = resolved
        return resolved

    def missing(self, engine_tickers: Iterable[str]) -> list[str]:
        """Бумаги, которых в базе нет ни под одной формой символа."""
        return [str(t) for t in engine_tickers if self.resolve(t) is None]

    def venue_substitutions(self,
                            engine_tickers: Iterable[str]) -> list[tuple[str, str]]:
        """Пары «тикер → чужая площадка», которые обязан увидеть пользователь.

        Список отдельный от прокси, потому что это ДРУГАЯ подмена: прокси
        меняет бумагу ради факторной модели, оставляя цену настоящей, а смена
        площадки меняет саму цену и валюту.
        """
        out: list[tuple[str, str]] = []
        for ticker in engine_tickers:
            found = self.resolve(ticker)
            if found is not None and found.match_kind != "exact":
                out.append((str(ticker), found.source_symbol))
        return out

    # ── календарь и свежесть ─────────────────────────────────────────────

    def sessions(self, market: str) -> list[int]:
        """Торговые дни рынка = даты, по которым на нём вообще есть бары.

        Кэшируется на время жизни объекта: за один отчёт календарь не меняется,
        а запрос идёт по всей таблице.
        """
        key = str(market or "").upper()
        if key not in self._sessions:
            rows = self._conn.execute(
                "SELECT DISTINCT b.trade_date AS d FROM daily_bars b "
                "JOIN instruments i ON i.id = b.instrument_id "
                "WHERE i.market = ? ORDER BY d", (key,)).fetchall()
            self._sessions[key] = [int(r["d"]) for r in rows]
        return self._sessions[key]

    def last_bar(self, engine_ticker: str) -> Optional[LastBar]:
        found = self.resolve(engine_ticker)
        if found is None:
            return None
        row = self._conn.execute(
            "SELECT trade_date, close FROM daily_bars WHERE instrument_id=? "
            "ORDER BY trade_date DESC LIMIT 1", (found.instrument_id,)).fetchone()
        if row is None:
            return None
        return LastBar(found.source_symbol, int(row["trade_date"]),
                       float(row["close"]))

    def staleness_trading_days(self, engine_ticker: str, *,
                               as_of: Optional[int] = None) -> Optional[int]:
        """Возраст последнего бара в торговых днях рынка ЭТОЙ бумаги.

        `None` — бумаги в базе нет.  Ноль — бар за `as_of` есть.  Отличать эти
        два ответа обязательно: «нет данных» и «данные свежие» — разные исходы
        с разной реакцией.
        """
        found = self.resolve(engine_ticker)
        if found is None:
            return None
        last = self.last_bar(engine_ticker)
        if last is None:
            return None
        moment = int(as_of) if as_of is not None else self.latest_date(found.market)
        if moment is None:
            return 0
        return mc.staleness_trading_days(self.sessions(found.market),
                                         last.trade_date, as_of=moment)

    def latest_date(self, market: str) -> Optional[int]:
        """Самая свежая дата рынка в базе."""
        sessions = self.sessions(market)
        return sessions[-1] if sessions else None

    # ── собственно цены ──────────────────────────────────────────────────

    def bars(self, engine_tickers: Sequence[str], *, days: int,
             as_of: Optional[int] = None) -> pd.DataFrame:
        """Матрица цен закрытия: index — даты, columns — ТИКЕРЫ ДВИЖКА.

        Колонки названы так, как спросили, а не так, как бумага записана у
        источника: движок сопоставляет матрицу с портфелем по своему тикеру, и
        переименование колонки в `BTC.V` тихо потеряло бы позицию `BTC-USD`.

        Ряды отдаются КАК В БАЗЕ, без корректировок (см. шапку модуля).
        """
        columns: dict[str, pd.Series] = {}
        for ticker in engine_tickers:
            found = self.resolve(ticker)
            if found is None:
                continue
            series = self._series(found, days=days, as_of=as_of)
            if series is not None and not series.empty:
                columns[str(ticker)] = series
        if not columns:
            return pd.DataFrame()
        frame = pd.DataFrame(columns)
        return frame.sort_index()

    def _series(self, found: Resolved, *, days: int,
                as_of: Optional[int]) -> Optional[pd.Series]:
        moment = as_of if as_of is not None else self.latest_date(found.market)
        if moment is None:
            return None
        cutoff = _shift_days(int(moment), -int(days))
        rows = self._conn.execute(
            "SELECT trade_date, close FROM daily_bars "
            "WHERE instrument_id=? AND trade_date > ? AND trade_date <= ? "
            "ORDER BY trade_date", (found.instrument_id, cutoff, int(moment))
        ).fetchall()
        if not rows:
            return None
        index = pd.to_datetime([str(r["trade_date"]) for r in rows],
                               format="%Y%m%d")
        return pd.Series([float(r["close"]) for r in rows], index=index,
                         name=found.engine_ticker)


def _shift_days(yyyymmdd: int, delta: int) -> int:
    """Сдвиг КАЛЕНДАРНОЙ даты в форме `YYYYMMDD`.

    Окно движка задано в календарных днях (`HISTORY_LOOKBACK_DAYS` = 1825 ≈ 5
    лет) — это про глубину истории, а не про свежесть, поэтому здесь
    календарь, а не торговые дни.  Просрочка считается иначе, и это
    сознательно разные величины.
    """
    text = str(int(yyyymmdd))
    anchor = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    shifted = anchor + pd.Timedelta(days=int(delta)).to_pytimedelta()
    return int(shifted.strftime("%Y%m%d"))
