"""
Загрузчик выгрузки Stooq в локальную базу котировок (MP-08.5, ЧК-08.5-2).

Что это
───────
Библиотека, которая читает файлы со `stooq.com/db/` и кладёт бары в SQLite.
Два вида входа, оба измерены на живых файлах (`docs/audit/STOOQ_CONVENTION.md
§3.2–3.8`):

* **дневной срез** `YYYYMMDD_d.txt` (≈ 700 КБ) — один бар на бумагу за день;
* **файл истории** `aapl.us.txt` из архива `d_us_txt.zip` — вся история бумаги.

Почему библиотека лежит в `src/`, а не в `scripts/`
───────────────────────────────────────────────────
В деплой-образе каталога `scripts/` нет, и тест, который его читает, обязан
`skipTest` — то есть логика загрузчика оказалась бы НЕ ПРОВЕРЕННОЙ во втором,
зеркальном прогоне.  Цена решения — мёртвый модуль в образе (бот его не
импортирует); выгода — те же тесты в обеих линиях.  В `scripts/` остаётся
только разбор аргументов.

Кто пишет и кто читает
──────────────────────
Пишет ТОЛЬКО этот модуль, и только из процесса оператора.  Бот открывает базу
`mode=ro` (`finance.stooq_store`).  Гонок нет по построению, а не по
договорённости.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from env_config import env_int
from finance import market_calendar as mc

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Имя источника.  Одно на всю базу — второго происхождения здесь нет, поэтому
#: колонки `origin` в баре нет (в отличие от витрины Freedom, где I-14 её
#: требует именно из-за соседства двух происхождений).
SOURCE_NAME = "stooq"

#: Конвенция корректировок Stooq **НЕ ИЗМЕРЕНА** (`STOOQ_CONVENTION §4`).
#: Хранится строкой в справочнике, а не подставляется в отчёт: пока замера нет,
#: провайдер обязан отказаться объявлять конвенцию, а не выдумать её.
CONVENTION_UNMEASURED = "unmeasured"

#: Ниже этого числа принятых US-баров будний файл считается НЕДОКАЧАННЫМ.
#: Замер: полные будние срезы дают 11 738 и 11 742 строки US; суббота даёт
#: РОВНО НОЛЬ.  Отсюда форма правила 9 — «0 < принято < порога», а не
#: «принято < порога»: ноль означает закрытый рынок (выходной или праздник) и
#: не требует ни календаря праздников, ни исключений.
MIN_US_ROWS = env_int("STOOQ_MIN_US_ROWS", 5000, lo=0, hi=100_000)

_HEADER_TICKER = "<TICKER>"
_EXPECTED_COLUMNS = ("<TICKER>", "<PER>", "<DATE>", "<OPEN>", "<HIGH>",
                     "<LOW>", "<CLOSE>", "<VOL>")

_DAILY_NAME_RE = re.compile(r"(\d{8})_d\.txt$", re.IGNORECASE)


# ═════════════════════════════════════════════════════════════════════════════
# Типы
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Bar:
    """Один дневной бар в форме источника.  Символ уже в верхнем регистре."""

    symbol: str
    trade_date: int                    # YYYYMMDD — как в файле
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: float
    volume: Optional[float]


@dataclass
class IngestBatch:
    """Разобранный файл: что принято, что отброшено и почему."""

    file_name: str
    file_date: Optional[int]
    bars: list[Bar] = field(default_factory=list)
    rows_read: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    #: Файл целиком непригоден (правило 9 или битая шапка) — причина текстом.
    fatal: Optional[str] = None

    def reject(self, rule: str) -> None:
        self.rejected[rule] = self.rejected.get(rule, 0) + 1

    @property
    def accepted_by_market(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for bar in self.bars:
            market = mc.market_of(bar.symbol)
            out[market] = out.get(market, 0) + 1
        return out


@dataclass
class IngestResult:
    """Итог применения одного или нескольких файлов."""

    files: int = 0
    rows_read: int = 0
    rows_written: int = 0
    instruments_added: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    fatal_files: list[tuple[str, str]] = field(default_factory=list)

    def merge_rejected(self, other: dict[str, int]) -> None:
        for rule, count in other.items():
            self.rejected[rule] = self.rejected.get(rule, 0) + count


@dataclass(frozen=True)
class SeamMismatch:
    """Расхождение бара архива и бара дневного среза за одну дату."""

    symbol: str
    trade_date: int
    archive_close: Optional[float]
    stored_close: Optional[float]


class IngestError(RuntimeError):
    """Файл непригоден целиком — база не тронута."""


# ═════════════════════════════════════════════════════════════════════════════
# Схема
# ═════════════════════════════════════════════════════════════════════════════

_DDL = """
CREATE TABLE IF NOT EXISTS instruments (
    id            INTEGER PRIMARY KEY,
    source        TEXT NOT NULL,
    source_symbol TEXT NOT NULL,
    market        TEXT NOT NULL,
    currency      TEXT NOT NULL,
    convention    TEXT NOT NULL,
    UNIQUE (source, source_symbol)
);

CREATE TABLE IF NOT EXISTS daily_bars (
    instrument_id INTEGER NOT NULL REFERENCES instruments(id),
    trade_date    INTEGER NOT NULL,
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL NOT NULL,
    volume        REAL,
    PRIMARY KEY (instrument_id, trade_date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS symbol_map (
    engine_ticker TEXT PRIMARY KEY,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id),
    match_kind    TEXT NOT NULL,
    note          TEXT
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id           INTEGER PRIMARY KEY,
    started_at   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    file_name    TEXT,
    rows_read    INTEGER,
    rows_written INTEGER,
    rejected     TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(path, *, read_only: bool = False) -> sqlite3.Connection:
    """Соединение с базой котировок.

    `read_only=True` открывает через URI `mode=ro` — это ЗАПРЕТ на уровне
    SQLite, а не соглашение: попытка записи поднимает `OperationalError`.
    Именно так бот и открывает базу, поэтому «бот не пишет» проверяемо.
    """
    target = Path(path)
    if read_only:
        uri = f"file:{target.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Создать таблицы и записать версию схемы. Идемпотентно."""
    conn.executescript(_DDL)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),))
    conn.commit()


def _bump_generation(conn: sqlite3.Connection) -> str:
    """Отметить, что содержимое базы изменилось.

    🔴 Без этого контейнер, скопировавший базу к себе на диск неделю назад,
    отдавал бы недельные цены и НЕ ЗНАЛ БЫ об этом (`PHASE_08B §3.2`).
    Поколение — это то, что бот сравнивает, решая, перекопировать ли файл.
    """
    generation = f"{int(time.time())}"
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('generation', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (generation,))
    return generation


# ═════════════════════════════════════════════════════════════════════════════
# Разбор файлов — девять правил, каждое из замера
# ═════════════════════════════════════════════════════════════════════════════

def file_date_of(path) -> Optional[int]:
    """Дата дневного среза из ИМЕНИ файла: `20260807_d.txt` → `20260807`.

    Имя, а не содержимое — потому что правило 1 сверяет одно с другим.  Взять
    дату из содержимого значило бы сверять файл сам с собой.
    """
    match = _DAILY_NAME_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def _to_float(raw) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value == value else None          # NaN → None


def _read_rows(path) -> Iterator[dict]:
    """Строки файла Stooq. Разделитель и шапка ПРОВЕРЯЮТСЯ, а не предполагаются."""
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
        if _HEADER_TICKER not in first.upper():
            raise IngestError(
                f"{Path(path).name}: нет шапки {_HEADER_TICKER} — это не выгрузка "
                "Stooq (частая причина: скачалась страница капчи вместо CSV)")
        header = [h.strip().upper() for h in first.strip().split(",")]
        missing = [c for c in _EXPECTED_COLUMNS if c not in header]
        if missing:
            raise IngestError(
                f"{Path(path).name}: в шапке нет колонок {', '.join(missing)}")
        handle.seek(0)
        reader = csv.DictReader(handle)
        for row in reader:
            yield {str(k).strip().upper(): v for k, v in row.items() if k}


def _bar_from_row(row: dict, batch: IngestBatch) -> Optional[Bar]:
    """Одна строка → бар, либо `None` с записью причины отбраковки.

    Здесь живут правила 2, 3, 4, 5, 7 (правило 1 — у вызывающего, потому что
    для файла истории оно не применяется; правило 6 — в `market_calendar`).
    """
    symbol = str(row.get(_HEADER_TICKER) or "").strip().upper()   # правило 7
    if not symbol or not mc.is_valid_symbol(symbol):
        batch.reject("битый символ")
        return None

    # Правило 5: индексы и FX-пары не грузим — движок их не запрашивает.
    # Это НЕ брак: отдельный счётчик, иначе сводка перестанет отличать
    # «пропустили осознанно» от «файл плохой».
    market = mc.market_of(symbol)
    if market == mc.NON_INSTRUMENT:
        batch.reject("индекс/FX (пропуск)")
        return None
    if market == mc.UNKNOWN_MARKET:
        batch.reject("неизвестный рынок")
        return None

    close = _to_float(row.get("<CLOSE>"))              # правило 2
    if close is None or close <= 0:
        batch.reject("нет цены закрытия")
        return None

    open_ = _to_float(row.get("<OPEN>"))
    high = _to_float(row.get("<HIGH>"))
    low = _to_float(row.get("<LOW>"))
    volume = _to_float(row.get("<VOL>"))

    # Правило 3: OHLC обязан быть непротиворечив.  Замер: ровно одна такая
    # строка на 11 999 (`^IPSA`, 2026-08-06) — 0.008 %, но молча пропускать её
    # нельзя: битый бар портит ATR и волатильность именно тем, что правдоподобен.
    #
    # 🔴 Проверки РАЗДЕЛЕНЫ по доступности полей (`AUDIT §−80`).  Прежняя
    # редакция гасила всю проверку одним `None not in (open_, high, low)`, и
    # пустой `<OPEN>` пропускал заодно `low <= high` — то есть бар с
    # `high=5, low=9` уезжал в базу.  Каждое условие обязано зависеть только
    # от тех полей, которые в нём участвуют.
    # Отдельной проверки `low <= high` НЕТ, и это не упущение: `close` к этому
    # моменту гарантированно не `None` (правило 2 выше), поэтому `low > high`
    # невозможно без нарушения одной из границ ниже — `low ≤ close ≤ high`.
    # Мутационный прогон это и показал: такая проверка не падала ни на одном
    # входе, то есть была мёртвым кодом, прикрытым живым (`AUDIT §−80`).
    bounds = [v for v in (open_, close) if v is not None]
    if low is not None and low > min(bounds):
        batch.reject("нарушен OHLC")
        return None
    if high is not None and high < max(bounds):
        batch.reject("нарушен OHLC")
        return None

    # Правило 4: синтетическая заглушка до IPO — нулевой объём при полностью
    # плоском баре (`spsc.us`, 2010-04-21).  Настоящий день без сделок в
    # выгрузке выглядит иначе: его просто нет.
    if (volume is not None and volume == 0
            and open_ == high == low == close):
        batch.reject("плоская заглушка")
        return None

    trade_date = _to_float(row.get("<DATE>"))
    if trade_date is None:
        batch.reject("нет даты")
        return None

    return Bar(symbol=symbol, trade_date=int(trade_date), open=open_,
               high=high, low=low, close=close, volume=volume)


def parse_daily_file(path, *, min_us_rows: Optional[int] = None) -> IngestBatch:
    """Дневной срез → батч. Правила 1 и 9 применяются только здесь.

    Правило 1 (дата строки == дата файла) — единственное ЖЁСТКОЕ правило
    фильтра: без него последний бар мёртвой бумаги приезжает как сегодняшняя
    цена.  Замер на сыром файле: 43 строки с пустой датой и 148 баров вплоть
    до 2015 года снимаются им разом, без единого действия человека.
    """
    path = Path(path)
    file_date = file_date_of(path)
    batch = IngestBatch(file_name=path.name, file_date=file_date)
    if file_date is None:
        batch.fatal = ("имя файла не содержит дату в форме YYYYMMDD_d.txt — "
                       "без неё правило «дата строки = дата файла» неприменимо")
        return batch

    for row in _read_rows(path):
        batch.rows_read += 1
        bar = _bar_from_row(row, batch)
        if bar is None:
            continue
        if bar.trade_date != file_date:                # правило 1
            batch.reject("чужая дата")
            continue
        batch.bars.append(bar)

    _apply_partial_download_guard(batch, min_us_rows)
    return batch


def _apply_partial_download_guard(batch: IngestBatch,
                                  min_us_rows: Optional[int]) -> None:
    """Правило 9: недокачанный будний файл отвергается ЦЕЛИКОМ.

    Полностью битый файл виден сразу — его отсеет шапка.  Опасен ЧАСТИЧНЫЙ:
    он выглядит нормальным и создаёт дыру в истории сразу у всех бумаг, а
    дыра в панели тише всего портит ковариацию.

    Форма условия — `0 < принято < порога`.  Ноль означает закрытый рынок:
    замерено на субботе 2026-08-08 (0 US-баров при 202 крипто-барах).  Поэтому
    праздники обрабатываются сами собой и список праздников не нужен.
    """
    threshold = MIN_US_ROWS if min_us_rows is None else int(min_us_rows)
    if threshold <= 0 or batch.file_date is None:
        return
    us_rows = batch.accepted_by_market.get("US", 0)
    if 0 < us_rows < threshold:
        batch.fatal = (
            f"принято всего {us_rows} US-баров при пороге {threshold} — "
            "файл выглядит недокачанным; база не тронута")


def parse_history_file(path, *, window_days: int,
                       today: Optional[date] = None) -> IngestBatch:
    """Файл истории одной бумаги → батч.

    Правило 1 здесь НЕ применяется (в файле истории по определению много дат),
    правило 9 — тоже.  Работает правило 8: берём только окно движка.
    """
    path = Path(path)
    batch = IngestBatch(file_name=path.name, file_date=None)
    cutoff_dt = (today or date.today()) - timedelta(days=int(window_days))
    cutoff = int(cutoff_dt.strftime("%Y%m%d"))

    for row in _read_rows(path):
        batch.rows_read += 1
        bar = _bar_from_row(row, batch)
        if bar is None:
            continue
        if bar.trade_date < cutoff:                    # правило 8
            batch.reject("вне окна")
            continue
        batch.bars.append(bar)
    return batch


# ═════════════════════════════════════════════════════════════════════════════
# Запись
# ═════════════════════════════════════════════════════════════════════════════

def _instrument_id(conn: sqlite3.Connection, symbol: str,
                   cache: dict[str, int]) -> int:
    """id инструмента, создавая справочную запись при первой встрече."""
    if symbol in cache:
        return cache[symbol]
    market = mc.market_of(symbol)
    row = conn.execute(
        "SELECT id FROM instruments WHERE source=? AND source_symbol=?",
        (SOURCE_NAME, symbol)).fetchone()
    if row is None:
        cursor = conn.execute(
            "INSERT INTO instruments(source, source_symbol, market, currency, "
            "convention) VALUES (?,?,?,?,?)",
            (SOURCE_NAME, symbol, market, mc.currency_of(market),
             CONVENTION_UNMEASURED))
        instrument_id = int(cursor.lastrowid)
    else:
        instrument_id = int(row["id"])
    cache[symbol] = instrument_id
    return instrument_id


def apply_batch(conn: sqlite3.Connection, batch: IngestBatch, *,
                kind: str = "apply") -> IngestResult:
    """Записать батч ОДНОЙ транзакцией. Идемпотентно по `(инструмент, дата)`.

    Повторное применение того же файла не меняет базу: `UPSERT` перезаписывает
    бар самим собой.  Порядок применения файлов тоже не важен — операция
    коммутативна по разным датам.
    """
    result = IngestResult(files=1, rows_read=batch.rows_read)
    result.merge_rejected(batch.rejected)

    if batch.fatal:
        result.fatal_files.append((batch.file_name, batch.fatal))
        return result

    before = conn.execute("SELECT COUNT(*) AS n FROM instruments").fetchone()["n"]
    cache: dict[str, int] = {}
    rows = []
    for bar in batch.bars:
        rows.append((_instrument_id(conn, bar.symbol, cache), bar.trade_date,
                     bar.open, bar.high, bar.low, bar.close, bar.volume))
    conn.executemany(
        "INSERT INTO daily_bars(instrument_id, trade_date, open, high, low, "
        "close, volume) VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(instrument_id, trade_date) DO UPDATE SET "
        "open=excluded.open, high=excluded.high, low=excluded.low, "
        "close=excluded.close, volume=excluded.volume", rows)
    after = conn.execute("SELECT COUNT(*) AS n FROM instruments").fetchone()["n"]

    result.rows_written = len(rows)
    result.instruments_added = int(after) - int(before)
    conn.execute(
        "INSERT INTO ingest_runs(started_at, kind, file_name, rows_read, "
        "rows_written, rejected) VALUES (?,?,?,?,?,?)",
        (datetime.utcnow().isoformat(timespec="seconds"), kind,
         batch.file_name, batch.rows_read, result.rows_written,
         json.dumps(batch.rejected, ensure_ascii=False)))
    _bump_generation(conn)
    conn.commit()
    return result


def apply_inbox(conn: sqlite3.Connection, inbox, applied=None,
                rejected_dir=None, *,
                min_us_rows: Optional[int] = None) -> IngestResult:
    """Применить ВСЕ дневные файлы из папки по возрастанию даты.

    Порядок по дате, а не по времени скачивания: оператор может принести
    вчерашний файл после сегодняшнего, и это не должно ничего значить.
    Применённые уезжают в `applied/`, непригодные — в `rejected/`, поэтому
    повторный запуск не перебирает то же самое.
    """
    inbox = Path(inbox)
    total = IngestResult()
    files = sorted((p for p in inbox.glob("*.txt") if file_date_of(p)),
                   key=lambda p: (file_date_of(p) or 0, p.name))
    for path in files:
        try:
            batch = parse_daily_file(path, min_us_rows=min_us_rows)
        except IngestError as exc:
            total.files += 1
            total.fatal_files.append((path.name, str(exc)))
            _move(path, rejected_dir)
            continue
        outcome = apply_batch(conn, batch, kind="apply")
        total.files += outcome.files
        total.rows_read += outcome.rows_read
        total.rows_written += outcome.rows_written
        total.instruments_added += outcome.instruments_added
        total.merge_rejected(outcome.rejected)
        if outcome.fatal_files:
            total.fatal_files.extend(outcome.fatal_files)
            _move(path, rejected_dir)
        else:
            _move(path, applied)
    return total


def _move(path: Path, destination) -> None:
    """Переложить обработанный файл. Отсутствие каталога — не ошибка."""
    if destination is None:
        return
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        path.replace(target_dir / path.name)
    except OSError as exc:                             # pragma: no cover
        logger.warning("не удалось переложить %s: %s", path.name, exc)


# ═════════════════════════════════════════════════════════════════════════════
# Бутстрап и сверка шва
# ═════════════════════════════════════════════════════════════════════════════

def iter_archive_files(archive_dir) -> Iterator[Path]:
    """Файлы истории в распакованном архиве — рекурсивно.

    Рекурсивно, потому что архив раскладывает бумаги по подпапкам биржи и
    ТИПА: `nasdaq stocks/1/…` рядом с `nasdaq etfs/…`.  Именно это разделение
    один раз уже стоило проекту ложного вывода «в пакете нет ETF»
    (`AUDIT §−76`), поэтому обход здесь не выбирает подпапки.
    """
    for path in sorted(Path(archive_dir).rglob("*.txt")):
        if path.is_file():
            yield path


def symbol_of_archive_file(path) -> str:
    """`…/nasdaq etfs/vgit.us.txt` → `VGIT.US`."""
    return Path(path).name.rsplit(".txt", 1)[0].upper()


def bootstrap(conn: sqlite3.Connection, archive_dir, *,
              symbols: Optional[Iterable[str]] = None,
              window_days: Optional[int] = None,
              today: Optional[date] = None) -> IngestResult:
    """Первичное наполнение из архива истории.

    `symbols=None` грузит ВЕСЬ архив; список символов ограничивает загрузку
    рабочим набором.  Рабочий набор — рекомендуемый режим: полная выгрузка US
    даёт 430–700 МБ базы, из которых отчёту нужны десятки (`PHASE_08B §3.1`).
    """
    window = int(window_days if window_days is not None
                 else env_int("HISTORY_LOOKBACK_DAYS", 1825, lo=90, hi=3650))
    wanted = ({str(s).upper() for s in symbols} if symbols is not None else None)
    total = IngestResult()
    for path in iter_archive_files(archive_dir):
        symbol = symbol_of_archive_file(path)
        if wanted is not None and symbol not in wanted:
            continue
        try:
            batch = parse_history_file(path, window_days=window, today=today)
        except IngestError as exc:
            total.files += 1
            total.fatal_files.append((path.name, str(exc)))
            continue
        if not batch.bars:
            total.files += 1
            total.rows_read += batch.rows_read
            total.merge_rejected(batch.rejected)
            continue
        outcome = apply_batch(conn, batch, kind="bootstrap")
        total.files += outcome.files
        total.rows_read += outcome.rows_read
        total.rows_written += outcome.rows_written
        total.instruments_added += outcome.instruments_added
        total.merge_rejected(outcome.rejected)
    return total


def verify_seam(conn: sqlite3.Connection, archive_dir, trade_date: int, *,
                symbols: Optional[Iterable[str]] = None) -> list[SeamMismatch]:
    """Сверить бар архива с баром в базе за одну дату.

    Замер 2026-08-10: совпадение ТОЧНОЕ до последнего знака на всех живых
    бумагах.  Расхождение здесь означает, что источник пересчитал историю
    (ретроспективная корректировка) — и тогда нужен ре-бутстрап, а не патч.

    🔴 `symbols=None` означает «всё, что ЕСТЬ В БАЗЕ», а не «весь архив»
    (`AUDIT §−80`).  Прежняя редакция брала архив целиком, и каждая бумага вне
    рабочего набора попадала в отчёт как расхождение `архив=9.9 база=None`.
    На реальном архиве это ~11 800 ложных срабатываний и ненулевой код
    возврата — то есть пункт чек-листа «verify-seam совпал» не мог пройти
    НИКОГДА, а сама команда переставала что-либо значить.
    """
    wanted = ({str(s).upper() for s in symbols} if symbols is not None
              else stored_symbols(conn))
    mismatches: list[SeamMismatch] = []
    for path in iter_archive_files(archive_dir):
        symbol = symbol_of_archive_file(path)
        if wanted is not None and symbol not in wanted:
            continue
        archive_close = _close_from_history(path, int(trade_date))
        stored = conn.execute(
            "SELECT b.close AS close FROM daily_bars b "
            "JOIN instruments i ON i.id = b.instrument_id "
            "WHERE i.source=? AND i.source_symbol=? AND b.trade_date=?",
            (SOURCE_NAME, symbol, int(trade_date))).fetchone()
        stored_close = float(stored["close"]) if stored is not None else None
        if archive_close is None and stored_close is None:
            continue
        if (archive_close is None or stored_close is None
                or archive_close != stored_close):
            mismatches.append(SeamMismatch(symbol, int(trade_date),
                                           archive_close, stored_close))
    return mismatches


def _close_from_history(path, trade_date: int) -> Optional[float]:
    try:
        for row in _read_rows(path):
            raw = _to_float(row.get("<DATE>"))
            if raw is not None and int(raw) == trade_date:
                return _to_float(row.get("<CLOSE>"))
    except IngestError:
        return None
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Сводка для оператора
# ═════════════════════════════════════════════════════════════════════════════

def resolve_symbol(conn: sqlite3.Connection,
                   engine_ticker: str) -> Optional[sqlite3.Row]:
    """Тикер движка → строка справочника инструментов, либо `None`.

    🔴 **ЕДИНСТВЕННАЯ реализация сопоставления** (`AUDIT §−80`).  Раньше их
    было две: `StooqStore.resolve` спрашивала `symbol_map`, а `coverage_report`
    — нет.  Разойтись они успели бы на первой же ручной подмене площадки:
    допуск C-1 доложил бы «нет истории» и вернул ненулевой код, тогда как бот
    ту же бумагу находил бы прекрасно.

    Живёт здесь, а не в `stooq_store`, потому что этот модуль — чистый stdlib,
    и его может звать загрузчик на голом Python.

    Порядок обязателен: `symbol_map` — это РЕШЕНИЯ человека, включая
    осознанные подмены площадки; автоматический перебор форм таких решений
    принимать не вправе.  Поэтому карта всегда старше.
    """
    from finance.stooq_symbols import candidates_for

    key = str(engine_ticker or "").strip().upper()
    row = conn.execute(
        "SELECT i.id AS id, i.source_symbol AS sym, i.market AS market, "
        "       i.currency AS ccy, m.match_kind AS kind "
        "FROM symbol_map m JOIN instruments i ON i.id = m.instrument_id "
        "WHERE m.engine_ticker = ?", (key,)).fetchone()
    if row is not None:
        return row
    for candidate in candidates_for(key):
        row = conn.execute(
            "SELECT id, source_symbol AS sym, market, currency AS ccy, "
            "       'exact' AS kind FROM instruments "
            "WHERE source=? AND source_symbol=?",
            (SOURCE_NAME, candidate)).fetchone()
        if row is not None:
            return row
    return None


def coverage_report(conn: sqlite3.Connection,
                    tickers: Sequence[str]) -> dict[str, Optional[int]]:
    """`{тикер: дата последнего бара}`; `None` — бумаги в базе нет.

    Это и есть допуск оператора: строка «C-1 факторы 10/10» считается отсюда.
    В профиле `STRICT` (ручной ввод) потеря ЛЮБОГО факторного ETF даёт `BLOCK`
    (`data_checks.check_portfolio_sufficiency`), поэтому «почти все» — не ответ.
    """
    out: dict[str, Optional[int]] = {}
    for ticker in tickers:
        found = resolve_symbol(conn, ticker)
        last: Optional[int] = None
        if found is not None:
            row = conn.execute(
                "SELECT MAX(trade_date) AS d FROM daily_bars "
                "WHERE instrument_id=?", (int(found["id"]),)).fetchone()
            if row is not None and row["d"] is not None:
                last = int(row["d"])
        out[str(ticker)] = last
    return out


#: Вердикты замера.  Константы, а не литералы у потребителя: сводка CLI считает
#: их СРАВНЕНИЕМ, а не поиском подстроки — «СКОРРЕКТИРОВАН» является подстрокой
#: «НЕ СКОРРЕКТИРОВАН», и на такой классификации сводка однажды соврала бы молча.
VERDICT_RAW = "ступенька ЕСТЬ → ряд СЫРОЙ (RAW)"
VERDICT_ADJUSTED = "ступеньки нет → ряд СКОРРЕКТИРОВАН"
VERDICT_UNCLEAR = "ступенька есть, но НЕ равна сплиту → НЕЯСНО"
VERDICT_ABSENT = "нет в архиве"
VERDICT_OUT_OF_RANGE = "событие вне окна ряда"

#: Вердикты, которые НЕ являются измерением: их нельзя ни складывать с
#: ответом, ни считать противоречием ему.
NO_ANSWER_VERDICTS = frozenset({VERDICT_ABSENT, VERDICT_OUT_OF_RANGE})


@dataclass(frozen=True)
class ConventionProbe:
    """Одно событие сплита, проверенное по архиву истории."""

    ticker: str
    event_date: str                    # ISO
    expected_ratio: float
    close_before: Optional[float]
    close_on_after: Optional[float]
    verdict: str
    #: Даты баров, между которыми считалось отношение.  Печатаются оператору:
    #: разрыв ряда у события (бары через месяц друг от друга) делает отношение
    #: бессмысленным, и увидеть это можно только по датам.
    date_before: Optional[int] = None
    date_after: Optional[int] = None

    @property
    def observed_ratio(self) -> Optional[float]:
        if not self.close_before or not self.close_on_after:
            return None
        return self.close_before / self.close_on_after

    @property
    def measured(self) -> bool:
        """Было ли событие вообще измерено (в отличие от «бумаги нет»)."""
        return self.verdict not in NO_ANSWER_VERDICTS


#: Порог «ступеньки» — ТОТ ЖЕ, что у эвристики движка
#: (`history._SPLIT_RETURN_THRESHOLD` = log(1.5)), чтобы слово «ступенька»
#: здесь и там значило одно и то же.  Свой порог означал бы, что замер
#: объявляет ряд сырым, а движок ту же ступеньку не видит.
_SPLIT_STEP_THRESHOLD = 1.5


def _verdict_for(observed: float, expected: float) -> str:
    """Вердикт по отношению цен: СЫРОЙ, СКОРРЕКТИРОВАН или НЕЯСНО.

    🔴 Ответов ТРИ, а не два, и третий — главный.  Отношение сравнивается с
    ДВУМЯ гипотезами: «ступенька размером со сплит» (ряд сырой) и «ступеньки
    нет» (ряд скорректирован).  Если оно не близко ни к одной — например, 2.0
    при сплите 10:1, — то верного ответа среди двух нет, и объявить сырым
    только потому, что «больше порога», значит выдать догадку за замер.
    Именно такую ошибку фаза и обязана исключить (`PHASE_08B §7.6`).

    Близость меряется в ЛОГАРИФМАХ и одним порогом — тем же, которым движок
    отличает ступеньку от новостного движения.  Логарифм здесь не украшение:
    он делает зоны симметричными и снимает вопрос о пересечении, которое при
    сравнении «в разах» возникло бы у сплита 2:1.
    """
    band = math.log(_SPLIT_STEP_THRESHOLD)
    to_raw = abs(math.log(observed) - math.log(expected))
    to_adjusted = abs(math.log(observed))
    if min(to_raw, to_adjusted) > band:
        return VERDICT_UNCLEAR
    return VERDICT_RAW if to_raw <= to_adjusted else VERDICT_ADJUSTED


def measure_convention(archive_dir, *,
                       today: Optional[date] = None) -> list[ConventionProbe]:
    """Измерить конвенцию корректировок Stooq по известным сплитам.

    Вопрос, ради которого всё: отдаёт ли Stooq ряд СЫРЫМ или уже
    скорректированным на сплиты.  Ответ решает, вправе ли провайдер объявить
    `PriceConvention` — а без объявленной конвенции инвариант I-7 непроверяем
    (`PHASE_08B §7.6`).

    Метод прямой: цена закрытия ДО события против цены В день события.
      • отношение ≈ ratio сплита → ступенька на месте, ряд **RAW**;
      • отношение ≈ 1 → ступеньки нет, ряд **скорректирован**.

    🔴 Дивидендную половину вопроса (`SPLIT_ADJUSTED` против `TOTAL_RETURN`)
    так НЕ решить: она требует сверки накопленной доходности с известной
    total-return и остаётся отдельным шагом (`STOOQ_CONVENTION §4`).
    Функция отвечает ровно на то, на что может.

    События берутся из `KNOWN_SPLITS` — единственного места, где они у нас
    записаны.  Свой список здесь был бы вторым справочником фактов о мире, а
    план уже трижды расходился с реальностью именно на таких списках
    (`AUDIT §−73`).
    """
    from freedom_portfolio.history import KNOWN_SPLITS

    by_symbol = {symbol_of_archive_file(p): p
                 for p in iter_archive_files(archive_dir)}
    out: list[ConventionProbe] = []
    for ticker, events in sorted(KNOWN_SPLITS.items()):
        path = by_symbol.get(str(ticker).upper())
        if path is None:
            out.extend(ConventionProbe(ticker, when.isoformat(), float(ratio),
                                       None, None, VERDICT_ABSENT)
                       for when, ratio in events)
            continue
        # Окно берём заведомо шире события: замер не про свежесть.  Файл
        # читается ОДИН раз на бумагу, а не на событие.
        batch = parse_history_file(path, window_days=20 * 365, today=today)
        # Сортировка ЯВНАЯ.  Порядок строк в файле — свойство поставщика, а не
        # наше знание; «последний до события» обязан значить «самый поздний по
        # ДАТЕ», иначе перестановка строк тихо меняет ответ замера.
        ordered = sorted(batch.bars, key=lambda b: b.trade_date)
        for when, ratio in events:
            stamp = int(when.strftime("%Y%m%d"))
            before = [b for b in ordered if b.trade_date < stamp]
            after = [b for b in ordered if b.trade_date >= stamp]
            if not before or not after:
                out.append(ConventionProbe(ticker, when.isoformat(),
                                           float(ratio), None, None,
                                           VERDICT_OUT_OF_RANGE))
                continue
            # Деление безопасно: цена ≤ 0 отбракована правилом 2 при разборе.
            observed = before[-1].close / after[0].close
            out.append(ConventionProbe(
                ticker, when.isoformat(), float(ratio),
                before[-1].close, after[0].close,
                _verdict_for(observed, float(ratio)),
                date_before=before[-1].trade_date,
                date_after=after[0].trade_date))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Рабочий набор: какие бумаги вообще класть в базу
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class UniverseCandidate:
    """Бумага архива, измеренная по трём признакам пригодности."""

    symbol: str
    market: str
    bars: int                          # баров в окне движка
    last_date: int                     # YYYYMMDD последнего бара
    median_turnover: float             # медианный оборот, close × volume


def scan_archive(archive_dir, *, window_days: Optional[int] = None,
                 today: Optional[date] = None,
                 on_progress=None) -> list[UniverseCandidate]:
    """Померить КАЖДУЮ бумагу архива: длина истории, свежесть, оборот.

    Дорогой проход (около 12 000 файлов), поэтому он отделён от выбора: сам
    выбор — чистая функция и проверяется без файлов вовсе.

    Разбор здесь НАМЕРЕННО легче, чем в `parse_history_file`: правила OHLC
    решают, класть ли БАР, а этот проход решает, класть ли БУМАГУ.  Гонять
    полный фильтр ради трёх агрегатов значило бы платить минутами за число,
    которое от него не изменится.

    `on_progress(файлов_пройдено, бумаг_померено)` — необязательный callback для
    оператора.  Именно callback, а не `print`: библиотека в `src/` печатать не
    вправе (протокол провайдера, «только logger»), а проход по 11 842 файлам без
    единого признака жизни выглядит как зависание.
    """
    window = int(window_days if window_days is not None
                 else env_int("HISTORY_LOOKBACK_DAYS", 1825, lo=90, hi=3650))
    cutoff = int(((today or date.today())
                  - timedelta(days=window)).strftime("%Y%m%d"))
    out: list[UniverseCandidate] = []
    seen = 0
    for path in iter_archive_files(archive_dir):
        seen += 1
        if on_progress is not None and seen % 1000 == 0:
            on_progress(seen, len(out))
        symbol = symbol_of_archive_file(path)
        market = mc.market_of(symbol)
        if market in (mc.NON_INSTRUMENT, mc.UNKNOWN_MARKET):
            continue
        last = 0
        turnovers: list[float] = []
        try:
            for row in _read_rows(path):
                day = _to_float(row.get("<DATE>"))
                if day is None or int(day) < cutoff:
                    continue
                close = _to_float(row.get("<CLOSE>"))
                if close is None or close <= 0:
                    continue
                last = max(last, int(day))
                turnovers.append(close * (_to_float(row.get("<VOL>")) or 0.0))
        except IngestError as exc:
            logger.info("STOOQ: %s пропущен при сканировании: %s", symbol, exc)
            continue
        if not turnovers:
            continue
        out.append(UniverseCandidate(symbol, market, len(turnovers), last,
                                     _median(turnovers)))
    return out


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass
class UniverseSelection:
    """Отобранное — и ПОЧЕМУ остальное не отобрано.

    🔴 Счётчики существуют не для красоты (`AUDIT §−87`).  Первая редакция
    возвращала голый список, и когда порог оказался недостижимым, команда
    честно печатала «рабочий набор: 13 бумаг» — не соврав ни словом и не дав
    ни одной зацепки.  Правило проекта «статус обязан называть то, чем его
    можно опровергнуть» относится и к выводу команды, а не только к докам.
    """

    symbols: list[str] = field(default_factory=list)
    measured: int = 0                  # сколько бумаг вообще померено
    too_short: int = 0                 # не хватило истории
    too_stale: int = 0                 # давно не торгуется
    too_thin: int = 0                  # мало оборота
    over_limit: int = 0                # прошли фильтры, но не влезли в потолок
    min_bars_used: int = 0             # порог истории, ФАКТИЧЕСКИ применённый
    busiest_bars: int = 0              # у самой полной бумаги архива


def select_universe(candidates: Iterable[UniverseCandidate], *,
                    min_bars: Optional[int] = None, stale_after_days: int,
                    min_turnover: float, limit: int,
                    as_of: Optional[int] = None) -> list[str]:
    """Совместимый вход: только символы. Разбор причин — `explain_universe`."""
    return explain_universe(candidates, min_bars=min_bars,
                            stale_after_days=stale_after_days,
                            min_turnover=min_turnover, limit=limit,
                            as_of=as_of).symbols


#: Доля от истории САМОЙ ПОЛНОЙ бумаги архива, ниже которой бумага считается
#: молодой.  Доля, а не абсолютное число баров, — потому что абсолютное число
#: уже один раз обнулило универсум: порог 1260 сравнивался со счётчиком, который
#: окно в 1825 календарных дней физически не выдаёт (замер оператора: 1252 бара
#: на бумагу).  Календарь рынка со всеми его праздниками и полуднями знает
#: только сам архив, поэтому эталон берётся оттуда (`AUDIT §−87`).
_MIN_HISTORY_FRACTION = 0.9


def explain_universe(candidates: Iterable[UniverseCandidate], *,
                     min_bars: Optional[int] = None, stale_after_days: int,
                     min_turnover: float, limit: int,
                     as_of: Optional[int] = None) -> UniverseSelection:
    """Отобрать бумаги в базу по ИЗМЕРЕННЫМ признакам, а не по списку.

    🔴 Списка тикеров здесь нет и быть не должно.  Захардкоженный «топ-500
    ликвидных» — это второй справочник фактов о мире (ловушка Т-4): он устареет
    молча, первым же делистингом или IPO, и узнáем мы об этом из отчёта
    пользователя.  Признаки же пересчитываются на каждом ре-бутстрапе сами.

    Три критерия, и каждый отвечает за свой способ сломать отчёт:

    * `min_bars` — короткая история схлопывает окно регрессии для ВСЕЙ панели
      (F-15/F-21), то есть одна молодая бумага портит чужие числа;
    * `stale_after_days` — делистингованная бумага отдала бы последнюю цену как
      сегодняшнюю (замер `tph.us`);
    * `min_turnover` — неликвид даёт ряд из повторов и ложно низкую волатильность.

    `limit` — не критерий качества, а ПОТОЛОК РАЗМЕРА: замер даёт 75 КБ на
    бумагу, значит 800 бумаг ≈ 60 МБ, а полная выгрузка US ≈ 900 МБ, что для
    контейнера с лимитом 2 GiB и `/tmp` в оперативной памяти уже опасно
    (`PHASE_08B §3.1`).  Отсекается наименее ликвидное — то есть то, чьё
    отсутствие пользователь заметит с наименьшей вероятностью.

    🔴 `min_bars=None` означает «спросить у архива» — и это ДЕФОЛТ.  Абсолютное
    число здесь уже стоило универсума: 1260 сравнивалось со счётчиком, который
    окно в 1825 календарных дней не выдаёт никогда (замер: 1252).  Сколько
    торговых дней в окне на самом деле, знает только сам рынок, поэтому
    эталоном берётся самая полная бумага архива.
    """
    pool = list(candidates)
    out = UniverseSelection(measured=len(pool))
    if not pool:
        return out

    out.busiest_bars = max(c.bars for c in pool)
    out.min_bars_used = (int(min_bars) if min_bars
                         else int(out.busiest_bars * _MIN_HISTORY_FRACTION))

    survivors = []
    for candidate in pool:
        if candidate.bars < out.min_bars_used:
            out.too_short += 1
        elif candidate.median_turnover < float(min_turnover):
            out.too_thin += 1
        else:
            survivors.append(candidate)

    newest = int(as_of) if as_of is not None else max(
        (c.last_date for c in pool), default=0)
    if newest:
        horizon = _shift_yyyymmdd(newest, -int(stale_after_days))
        fresh = [c for c in survivors if c.last_date >= horizon]
        out.too_stale = len(survivors) - len(fresh)
        survivors = fresh

    survivors.sort(key=lambda c: (-c.median_turnover, c.symbol))
    out.over_limit = max(0, len(survivors) - int(limit))
    out.symbols = [c.symbol for c in survivors[:int(limit)]]
    return out


def _shift_yyyymmdd(yyyymmdd: int, delta_days: int) -> int:
    text = str(int(yyyymmdd))
    anchor = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    return int((anchor + timedelta(days=int(delta_days))).strftime("%Y%m%d"))


def stored_symbols(conn: sqlite3.Connection) -> set[str]:
    """Символы источника, которые в базе реально есть.

    Нужен `verify_seam`: сверять надо то, что мы загрузили, а не весь архив.
    Бумага, лежащая в архиве и НЕ входящая в рабочий набор, — это не
    расхождение, а осознанный пропуск.
    """
    return {str(r["source_symbol"]).upper() for r in conn.execute(
        "SELECT source_symbol FROM instruments WHERE source=?", (SOURCE_NAME,))}


def database_path(default: Optional[str] = None) -> Path:
    """Где лежит база. `STOOQ_DB_PATH`, иначе репо-локальный `data/`.

    Тот же приём, что у `db_tokenomics.DB_PATH`: в проде путь задаётся env и
    указывает на смонтированный том, локально — на каталог репозитория.
    """
    if default is None:
        default = str(Path(__file__).resolve().parents[2] / "data"
                      / "stooq_prices.sqlite")
    return Path(os.getenv("STOOQ_DB_PATH", default))
