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
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from env_config import env_int
from finance import market_calendar as mc
from finance.stooq_ingest import connect, database_path, resolve_symbol

logger = logging.getLogger(__name__)


class StoreUnavailable(RuntimeError):
    """Базы нет или она непригодна для чтения."""


def local_copy_limits() -> tuple[str, int, int]:
    """`(каталог, нижний порог МБ, верхний порог МБ)` для локальной копии.

    🔴 Читается ФУНКЦИЕЙ, а не константами модуля — тот же приём, что у
    `price_providers._flag`: константа защёлкнулась бы на импорте, и тест не
    смог бы её переопределить, а перезагрузка модуля оставила бы у соседей
    ссылки на старый класс.

    * **нижний порог** — ниже него копия только тратит память: файл целиком
      укладывается в один-два range-запроса;
    * **верхний порог** — выше него копировать ОПАСНО: `/tmp` в Cloud Run это
      оперативная память при лимите 2 GiB, и база, собранная флагом `--all`
      (≈900 МБ), уронила бы контейнер.
    """
    return (os.getenv("STOOQ_LOCAL_COPY_DIR", "/tmp/ramp-stooq"),
            env_int("STOOQ_LOCAL_COPY_MIN_MB", 5, lo=0, hi=8000),
            env_int("STOOQ_LOCAL_COPY_MAX_MB", 400, lo=0, hi=8000))


def _local_copy(source: Path) -> Path:
    """Путь, с которого читать: локальная копия либо сам файл.

    Копия обновляется, когда у исходника изменились размер или время правки —
    то есть после каждой публикации новой базы оператором.  Признак берётся
    `stat`-ом, без открытия SQLite: открыть удалённую базу только ради версии
    значило бы платить сетью за проверку, которая должна быть дешёвой.

    Любой сбой копирования — НЕ ошибка: читаем исходник и говорим об этом в
    лог.  Отчёт важнее оптимизации.
    """
    directory, min_mb, max_mb = local_copy_limits()
    try:
        stat = source.stat()
        size_mb = stat.st_size / (1024 * 1024)
        if size_mb < min_mb:
            return source
        if size_mb > max_mb:
            logger.warning(
                "STOOQ: база %.0f МБ больше потолка копирования %d МБ — читаю "
                "с исходного тома. Если это gcsfuse, отчёт будет медленным; "
                "пересоберите базу без --all (OPERATOR_STOOQ.md §5).",
                size_mb, max_mb)
            return source

        target = Path(directory) / source.name
        stamp = target.with_name(target.name + ".stamp")
        key = f"{stat.st_size}:{int(stat.st_mtime)}"
        if target.exists() and stamp.exists():
            if stamp.read_text(encoding="utf-8").strip() == key:
                return target

        target.parent.mkdir(parents=True, exist_ok=True)
        # Через временный файл и `os.replace`: иначе второй отчёт в том же
        # контейнере может открыть копию на середине записи.
        scratch = target.with_name(target.name + ".part")
        shutil.copyfile(source, scratch)
        os.replace(scratch, target)
        stamp.write_text(key, encoding="utf-8")
        logger.info("STOOQ: база скопирована локально (%.0f МБ) → %s",
                    size_mb, target)
        return target
    except OSError as exc:
        logger.warning("STOOQ: локальная копия не сделана (%s) — читаю с тома",
                       exc)
        return source


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
        target = _local_copy(target)
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

        # Сопоставление живёт в ОДНОМ месте (`stooq_ingest.resolve_symbol`).
        # Вторая копия здесь однажды разошлась бы с той, по которой считается
        # допуск C-1, и расхождение проявилось бы как «бот бумагу видит, а
        # гейт оператора — нет» (`AUDIT §−80`).
        row = resolve_symbol(self._conn, key)
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

        🔴 **Чего этот метод не видит.**  Он отвечает на вопрос «сколько раз
        рынок торговал без этой бумаги», и календарь рынка берёт из тех же
        строк.  Поэтому остановка рынка ЦЕЛИКОМ (оператор перестал грузить
        файлы) даёт ноль у всех бумаг сразу.  Для этого случая есть
        `market_staleness_days` — он считает от часов, а не от базы.
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

    def market_staleness_days(self, market: str, *,
                              today: Optional[date] = None) -> Optional[int]:
        """Возраст ВСЕГО рынка в КАЛЕНДАРНЫХ днях. `None` — рынка в базе нет.

        🔴 Отдельный метод, потому что `staleness_trading_days` этот случай
        поймать НЕ МОЖЕТ по построению (`AUDIT §−80`), и это ограничение надо
        называть, а не прятать. Календарь рынка выводится из тех же строк, что
        и бары: если рынок встал целиком — оператор неделю не грузил файлы, —
        то сессий после последнего бара в базе просто нет, и КАЖДАЯ бумага
        честно рапортует свежесть 0. Данные внутри себя непротиворечивы; врёт
        только вывод «всё свежее».

        Разорвать круг может лишь часы снаружи, поэтому здесь календарные дни
        от `today`, а не торговые из базы. Потребитель обязан спросить ОБА:
        потикерную свежесть — про бумагу, эту — про наполнение базы.
        """
        latest = self.latest_date(market)
        if latest is None:
            return None
        text = str(latest)
        last = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
        return max(0, ((today or date.today()) - last).days)

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
