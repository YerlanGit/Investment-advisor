"""
Цикл применения дневной дельты к базе котировок (IB-2).

Что здесь есть и чего здесь НЕТ
──────────────────────────────
Здесь весь цикл «скачать → проверить → применить → опубликовать» и **ни одной
строки aiogram**.  Тот же приём, которым `analyze_all` отделён от `tg_bot`:
логика, отделённая от доставки, проверяется без бота, без токена и без сети.
`ingest_bot` остаётся тонким — он читает сообщение и печатает сводку.

Слой — **L1 Data** (см. шапку `quote_publisher`).

Разбора файлов здесь тоже нет: он в `finance/stooq_ingest.py`, написан фазой
MP-08.5, покрыт тестами и **уже лежит в деплой-образе**.  Девять правил
фильтра — замер, а не выдумка, и переписывать их значило бы завести вторую
редакцию правил, которая однажды разойдётся с первой молча.

Четыре гарда — вместо второй пары глаз
──────────────────────────────────────
Оператор ОДИН (`PLAN §1.2`).  Это не упрощает задачу, а убирает человека,
который заметил бы, что база уехала пустой.  Роль второй пары глаз переходит
машине, и вот её проверки:

1. **валидность** — объект открывается как SQLite и содержит `daily_bars`;
2. **непустота** — инструментов больше нуля;
3. **не откат** — даты из курсора присутствуют в базе;
4. **обвал размера** — после применения база не усохла (дельта умеет только
   добавлять бары).

🔴 Третий гард **предупреждает, но НЕ блокирует**, и это осознанное отступление
от `PLAN §5.3`.  Разбор: применить сегодняшний файл к откатившейся базе
безвредно — он добавляет сегодняшние бары и ничего не портит.  Блокировка же
наказывала бы оператора за законную квартальную операцию и не давала бы ему
двигаться, пока он не дошлёт шесть файлов.  Лечит откат пересылка файлов, а не
запрет работать.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

from env_config import env_int
from finance import stooq_ingest as si
from services.quote_publisher import (Cursor, PublisherUnavailable, QuotePublisher,
                                      publisher_from_env)

logger = logging.getLogger(__name__)

#: Потолок размера присланного файла.  Проверяется ПО МЕТАДАННЫМ, до
#: скачивания: замер дневного среза — 700 КБ, файла истории одной бумаги —
#: сотни КБ.  Два мегабайта дают запас в три раза и отсекают всё остальное,
#: не читая ни байта содержимого (тот же приём, что `PHASE_07 §3`).
MAX_UPLOAD_BYTES = env_int("INGEST_MAX_FILE_BYTES", 2_000_000,
                           lo=10_000, hi=20_000_000)

#: Насколько база вправе усохнуть после применения дельты.  Дневной срез умеет
#: только ДОБАВЛЯТЬ бары, поэтому усадка означает, что что-то пошло не так, а
#: не что данных стало меньше.  Порог не 100 %, потому что SQLite перекладывает
#: страницы и размер может колебаться на единицы процентов.
MIN_SIZE_RATIO = 0.9

_CONVENTIONAL_SUFFIX = ".txt"


# ═════════════════════════════════════════════════════════════════════════════
# Типы
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class UploadDecision:
    """Что это за файл — решается ДО скачивания, по имени и размеру."""

    kind: Optional[str]                 # 'daily' | 'history' | None
    trade_date: Optional[int] = None
    reason: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.kind is not None


@dataclass(frozen=True)
class C1Coverage:
    """Допуск оператора: факторы и бенчмарки на месте.

    `checked=False` — это НЕ «всё хорошо».  Молча пропустить C-1 нельзя ни в
    одном режиме: в профиле `STRICT` (а ручной ввод — это он) потеря ЛЮБОГО
    факторного ETF даёт `BLOCK`, то есть пользователь получит отказ вместо
    отчёта.  Пропуск, напечатанный как пустая строка, читался бы как
    «проверено».
    """

    checked: bool
    factors_ok: int = 0
    factors_total: int = 0
    benchmarks_ok: int = 0
    benchmarks_total: int = 0
    missing_factors: tuple[str, ...] = ()
    note: Optional[str] = None

    @property
    def complete(self) -> bool:
        return self.checked and self.factors_ok == self.factors_total


@dataclass(frozen=True)
class MarketState:
    market: str
    latest: Optional[int]
    instruments: int
    bars: int
    stale_days: Optional[int] = None
    days_left: Optional[int] = None


@dataclass(frozen=True)
class StoreStatus:
    ok: bool
    storage: str
    reason: Optional[str] = None
    generation: Optional[int] = None
    size: int = 0
    instruments: int = 0
    markets: tuple[MarketState, ...] = ()
    c1: Optional[C1Coverage] = None
    missing_dates: tuple[int, ...] = ()
    cursor_at: Optional[str] = None
    last_run: dict = field(default_factory=dict)

    @property
    def days_left(self) -> Optional[int]:
        values = [m.days_left for m in self.markets if m.days_left is not None]
        return min(values) if values else None


@dataclass(frozen=True)
class ApplyOutcome:
    """Итог одной операции.  `ok=False` всегда несёт `reason`."""

    ok: bool
    kind: str
    file_name: str
    reason: Optional[str] = None
    conflict: bool = False
    published: bool = False
    store_touched: bool = False
    generation: Optional[int] = None
    result: Optional[si.IngestResult] = None
    c1: Optional[C1Coverage] = None
    missing_dates: tuple[int, ...] = ()
    days_left: Optional[int] = None
    warnings: tuple[str, ...] = ()


# ═════════════════════════════════════════════════════════════════════════════
# Классификация присланного файла
# ═════════════════════════════════════════════════════════════════════════════

def classify_upload(file_name: str, size_bytes: Optional[int]) -> UploadDecision:
    """Дневной срез, файл истории или отказ — БЕЗ чтения содержимого.

    Различение по имени, а не по содержимому, потому что от него зависит выбор
    парсера: `parse_daily_file` применяет правило 1 («дата строки = дата
    файла»), а у файла истории дат по определению много.  Скормить историю
    дневному парсеру значит отбросить её целиком — молча и правдоподобно.
    """
    name = Path(str(file_name or "")).name
    if not name:
        return UploadDecision(None, reason="у файла нет имени")
    if size_bytes is not None and int(size_bytes) > MAX_UPLOAD_BYTES:
        return UploadDecision(
            None, reason=(f"файл {int(size_bytes) // 1024} КБ больше потолка "
                          f"{MAX_UPLOAD_BYTES // 1024} КБ. Дневной срез Stooq "
                          "весит ≈700 КБ; архив целиком через Telegram не "
                          "проходит и не должен"))
    if name.lower().endswith(".zip"):
        return UploadDecision(
            None, reason=("это архив. Бот применяет РАСПАКОВАННЫЕ дневные "
                          "срезы; сборка базы из архива истории — операция "
                          "оператора на его машине"))
    if not name.lower().endswith(_CONVENTIONAL_SUFFIX):
        return UploadDecision(
            None, reason=f"ожидается файл {_CONVENTIONAL_SUFFIX}, а не {name}")

    trade_date = si.file_date_of(name)
    if trade_date is not None:
        return UploadDecision("daily", trade_date=trade_date)
    return UploadDecision("history")


# ═════════════════════════════════════════════════════════════════════════════
# Чтение базы — только stdlib
# ═════════════════════════════════════════════════════════════════════════════

def _lookback_days() -> int:
    """Окно истории. Читается из того же env, что и `stooq_ingest.bootstrap`.

    Это не копия константы, а чтение ОДНОГО источника правды
    (`HISTORY_LOOKBACK_DAYS`, `CLAUDE.md §Числа и константы`): значение и
    клампы обязаны совпасть с бутстрапом, иначе добранная ботом бумага получит
    другую глубину, чем её соседи, и окно регрессии схлопнется по самой
    короткой (F-15/F-21).
    """
    return env_int("HISTORY_LOOKBACK_DAYS", 1825, lo=90, hi=3650)


def _engine_universe():
    """`(факторы, бенчмарки)` из `data_checks` — SSOT, а не копия.

    Импорт ЛЕНИВЫЙ ровно по той же причине, что в `scripts/stooq_ingest.py`:
    `finance.stooq_ingest` — чистый stdlib, и весь путь применения обязан
    работать без pandas.  Без него команда не падает, а честно говорит, что
    C-1 не проверен.
    """
    from finance.data_checks import BENCHMARK_ETFS, FACTOR_ETFS   # noqa: PLC0415

    return FACTOR_ETFS, BENCHMARK_ETFS


def _market_stale_limit() -> Optional[int]:
    """Сколько КАЛЕНДАРНЫХ дней рынку позволено не обновляться.

    Берётся у `stooq_provider` — того самого места, где порог и применяется.
    Свой литерал здесь означал бы, что бот обещает один срок, а отчёт
    блокируется по другому.  `None` — порог недоступен (нет pandas), и тогда
    срок не печатается вовсе: выдумать его нельзя.
    """
    try:
        from finance.stooq_provider import MAX_MARKET_STALE_DAYS  # noqa: PLC0415
        return int(MAX_MARKET_STALE_DAYS)
    except Exception:                                   # noqa: BLE001
        return None


def _as_date(yyyymmdd: int) -> date:
    text = str(int(yyyymmdd))
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _markets(conn: sqlite3.Connection, *,
             today: Optional[date] = None) -> tuple[MarketState, ...]:
    """Сводка по рынкам: свежесть и сколько осталось до блокировки.

    Это АГРЕГАТ (`MAX`/`COUNT` с группировкой), а не календарь рынка: календарь
    живёт в `StooqStore.sessions` и нужен для потикерной свежести в торговых
    днях.  Вторая реализация календаря здесь была бы ровно тем дефектом, на
    котором проект уже обжигался (`AUDIT §−80`, две реализации сопоставления).
    """
    limit = _market_stale_limit()
    moment = today or date.today()
    rows = conn.execute(
        "SELECT i.market AS market, MAX(b.trade_date) AS latest, "
        "       COUNT(DISTINCT i.id) AS instruments, COUNT(*) AS bars "
        "FROM daily_bars b JOIN instruments i ON i.id = b.instrument_id "
        "GROUP BY i.market ORDER BY i.market").fetchall()
    out: list[MarketState] = []
    for row in rows:
        latest = int(row["latest"]) if row["latest"] is not None else None
        stale = (moment - _as_date(latest)).days if latest is not None else None
        left = (limit - stale) if (limit is not None and stale is not None) else None
        out.append(MarketState(market=str(row["market"]), latest=latest,
                               instruments=int(row["instruments"]),
                               bars=int(row["bars"]),
                               stale_days=stale, days_left=left))
    return tuple(out)


def _dates_present(conn: sqlite3.Connection,
                   dates: Iterable[int]) -> set[int]:
    """Какие из перечисленных дат в базе ЕСТЬ.

    Членство в таблице фактов, а не календарь рынка: вопрос здесь — «легли ли
    бары этого дня», и ответ не зависит от того, какому рынку они принадлежат.
    """
    wanted = sorted({int(d) for d in dates})
    if not wanted:
        return set()
    marks = ",".join("?" * len(wanted))
    rows = conn.execute(
        f"SELECT DISTINCT trade_date AS d FROM daily_bars "
        f"WHERE trade_date IN ({marks})", wanted).fetchall()
    return {int(r["d"]) for r in rows}


def _c1(conn: sqlite3.Connection) -> C1Coverage:
    try:
        factor_etfs, benchmark_etfs = _engine_universe()
    except ImportError as exc:
        name = getattr(exc, "name", None) or "зависимость"
        return C1Coverage(checked=False,
                          note=f"{name} не установлен — допуск не подтверждён")
    coverage = si.coverage_report(conn, list(factor_etfs) + list(benchmark_etfs))
    missing = tuple(t for t in factor_etfs if not coverage.get(t))
    return C1Coverage(
        checked=True,
        factors_ok=len(factor_etfs) - len(missing), factors_total=len(factor_etfs),
        benchmarks_ok=sum(1 for t in benchmark_etfs if coverage.get(t)),
        benchmarks_total=len(benchmark_etfs),
        missing_factors=missing)


# ═════════════════════════════════════════════════════════════════════════════
# Гарды целостности
# ═════════════════════════════════════════════════════════════════════════════

def _open_checked(path: Path) -> sqlite3.Connection:
    """Открыть скачанную базу, убедившись, что это ОНА.

    🔴 Схема здесь НЕ создаётся.  `ensure_schema` на пустом файле сделал бы
    из мусора «правильную пустую базу», и следующий шаг опубликовал бы её
    поверх настоящей — операция, которая стирает труд оператора и при этом
    выглядит успешной.
    """
    conn = si.connect(path)
    try:
        conn.execute("SELECT 1 FROM daily_bars LIMIT 1").fetchone()
        conn.execute("SELECT 1 FROM instruments LIMIT 1").fetchone()
    except sqlite3.Error as exc:
        conn.close()
        raise PublisherUnavailable(
            f"скачанный файл не похож на базу котировок ({exc}). "
            "Бот базу с нуля не создаёт — соберите её бутстрапом.") from exc
    return conn


def _instrument_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM instruments").fetchone()
    return int(row["n"]) if row else 0


def _rollback_warning(conn: sqlite3.Connection,
                      cursor: Optional[Cursor]) -> tuple[tuple[int, ...], list[str]]:
    """Даты, которые бот применял, а в базе их больше нет.

    Расхождение ВПЕРЁД (в базе есть даты, которых нет в курсоре) тревогой не
    считается и здесь не ищется: это оператор применил файл через CLI, а CLI
    остаётся штатным запасным путём на случай, когда бот недоступен.  Курсор,
    считающий свой снимок единственной истиной, сломал бы этот путь — а при
    одном операторе он и есть весь резерв (`PLAN §5.4`).
    """
    if cursor is None or not cursor.applied_dates:
        return (), []
    present = _dates_present(conn, cursor.applied_dates)
    missing = tuple(d for d in cursor.applied_dates if d not in present)
    if not missing:
        return (), []
    listed = " ".join(str(d) for d in missing[:12])
    tail = "" if len(missing) <= 12 else f" … и ещё {len(missing) - 12}"
    return missing, [
        f"⚠️ база моложе моего журнала: не хватает {len(missing)} дн. — "
        f"{listed}{tail}. Похоже, был ре-бутстрап. Перешлите эти файлы из "
        "applied/ — порядок не важен, повтор безвреден."]


# ═════════════════════════════════════════════════════════════════════════════
# Операции
# ═════════════════════════════════════════════════════════════════════════════

def _refuse(kind: str, name: str, reason: str, *,
            conflict: bool = False, warnings: Sequence[str] = ()) -> ApplyOutcome:
    return ApplyOutcome(ok=False, kind=kind, file_name=name, reason=reason,
                        conflict=conflict, warnings=tuple(warnings))


def _apply(path, *, kind: str, actor: str, publisher: Optional[QuotePublisher],
           today: Optional[date]) -> ApplyOutcome:
    """Общий цикл для дневной дельты и файла истории.

    Различий между ними ровно три — парсер, право заводить инструмент и метка
    `kind` в журнале, — поэтому ветка одна, а не две почти одинаковые.
    """
    path = Path(path)
    name = path.name
    publisher = publisher or publisher_from_env()

    with tempfile.TemporaryDirectory(prefix="ramp-ingest-") as tmp:
        local = Path(tmp) / "prices.sqlite"
        try:
            snapshot = publisher.download(local)
        except PublisherUnavailable as exc:
            return _refuse(kind, name, str(exc))

        try:
            conn = _open_checked(local)
        except PublisherUnavailable as exc:
            return _refuse(kind, name, str(exc))

        try:
            if _instrument_count(conn) == 0:
                return _refuse(kind, name,
                               "в базе ноль инструментов — она пустая. "
                               "Бот дописывает бары уже отобранным бумагам и "
                               "состав базы не определяет: это делает бутстрап.")

            cursor = publisher.read_cursor()
            missing, warnings = _rollback_warning(conn, cursor)

            try:
                if kind == "daily":
                    batch = si.parse_daily_file(path)
                else:
                    batch = si.parse_history_file(
                        path, window_days=_lookback_days(), today=today)
            except si.IngestError as exc:
                return _refuse(kind, name, str(exc), warnings=warnings)

            if batch.fatal:
                return _refuse(kind, name, batch.fatal, warnings=warnings)

            result = si.apply_batch(
                conn, batch,
                kind=("apply" if kind == "daily" else "backfill"),
                allow_new=(kind == "history"))
            coverage = _c1(conn)
            markets = _markets(conn, today=today)
        finally:
            conn.close()

        new_size = local.stat().st_size
        if snapshot.size and new_size < snapshot.size * MIN_SIZE_RATIO:
            return _refuse(
                kind, name,
                (f"после применения база усохла с {snapshot.size} до {new_size} "
                 "байт. Дельта умеет только добавлять бары — не публикую."),
                warnings=warnings)

        upload = publisher.upload(local, if_generation_match=snapshot.generation)

    if upload.conflict:
        return ApplyOutcome(
            ok=False, kind=kind, file_name=name, conflict=True,
            store_touched=False, result=result, warnings=tuple(warnings),
            reason=("🔴 база в облаке изменилась, пока я применял файл. "
                    "При одном операторе такого быть не должно — не заливали "
                    f"ли вы её вручную? ({upload.reason}) "
                    "Файл НЕ применён, база не тронута. Пришлите его ещё раз."))
    if not upload.published:
        return ApplyOutcome(
            ok=False, kind=kind, file_name=name, result=result,
            store_touched=False, warnings=tuple(warnings),
            reason=("применено локально, но опубликовать не удалось "
                    f"({upload.reason}). Изменение потеряно — пришлите файл "
                    "ещё раз, повтор безвреден."))

    applied_date = batch.file_date if result.rows_written else None
    publisher.write_cursor((cursor or Cursor()).with_applied(
        applied_date, generation=upload.generation,
        last_run={"file": name, "actor": str(actor), "kind": kind,
                  "rows_written": result.rows_written}))
    publisher.archive_source(path, name)

    days_left = min((m.days_left for m in markets if m.days_left is not None),
                    default=None)
    return ApplyOutcome(
        ok=True, kind=kind, file_name=name, published=True, store_touched=True,
        generation=upload.generation, result=result, c1=coverage,
        missing_dates=missing, days_left=days_left, warnings=tuple(warnings))


def apply_daily(path, *, actor: str, publisher: Optional[QuotePublisher] = None,
                today: Optional[date] = None) -> ApplyOutcome:
    """Применить дневной срез.  Новые бумаги НЕ заводятся.

    `allow_new=False` — главное свойство дельты: состав базы определяет
    бутстрап по измеренным признакам ликвидности.  Дневной срез содержит все
    ~12 000 тикеров США, и без этого различия первый же `apply` затащил бы в
    базу 11 196 бумаг с историей в один день (`AUDIT §−88`).
    """
    return _apply(path, kind="daily", actor=actor, publisher=publisher,
                  today=today)


def apply_history(path, *, actor: str,
                  publisher: Optional[QuotePublisher] = None,
                  today: Optional[date] = None) -> ApplyOutcome:
    """Добрать бумагу файлом истории.  Инструмент заводится (`allow_new=True`).

    Это ровно та пара вызовов, которой пользуется `stooq_ingest.bootstrap`, —
    путь не новый, а уже проверенный.  Право заводить бумагу здесь осознанное:
    оператор прислал историю ИМЕННО ради того, чтобы бумага появилась.

    🔴 Символ берётся из СОДЕРЖИМОГО файла (колонка `<TICKER>`), а не из имени.
    Имя решает ровно один вопрос — история это или дневной срез.
    """
    return _apply(path, kind="history", actor=actor, publisher=publisher,
                  today=today)


def status(*, publisher: Optional[QuotePublisher] = None,
           today: Optional[date] = None) -> StoreStatus:
    """Состояние базы: поколение, рынки, допуск C-1, недостающие даты."""
    publisher = publisher or publisher_from_env()
    with tempfile.TemporaryDirectory(prefix="ramp-status-") as tmp:
        local = Path(tmp) / "prices.sqlite"
        try:
            snapshot = publisher.download(local)
        except PublisherUnavailable as exc:
            return StoreStatus(ok=False, storage=publisher.describe(),
                               reason=str(exc))
        try:
            conn = _open_checked(local)
        except PublisherUnavailable as exc:
            return StoreStatus(ok=False, storage=publisher.describe(),
                               reason=str(exc))
        try:
            cursor = publisher.read_cursor()
            missing, _ = _rollback_warning(conn, cursor)
            return StoreStatus(
                ok=True, storage=publisher.describe(),
                generation=snapshot.generation, size=snapshot.size,
                instruments=_instrument_count(conn),
                markets=_markets(conn, today=today), c1=_c1(conn),
                missing_dates=missing,
                cursor_at=(cursor.at if cursor else None),
                last_run=(dict(cursor.last_run) if cursor else {}))
        finally:
            conn.close()


def missing_dates(*, publisher: Optional[QuotePublisher] = None) -> tuple[int, ...]:
    """Даты, которые бот применял, а в базе их нет.  Вход для `/missing`."""
    return status(publisher=publisher).missing_dates


# ═════════════════════════════════════════════════════════════════════════════
# Сводка человеку
# ═════════════════════════════════════════════════════════════════════════════

def _c1_lines(coverage: Optional[C1Coverage]) -> list[str]:
    """Строка допуска.  Либо число, либо явное «НЕ ПРОВЕРЕН» — третьего нет."""
    if coverage is None:
        return []
    if not coverage.checked:
        return [f"  ⚠️ C-1 НЕ ПРОВЕРЕН: {coverage.note}"]
    mark = "✅" if coverage.complete else "🔴"
    lines = [f"  C-1 факторы ........... "
             f"{coverage.factors_ok}/{coverage.factors_total} {mark}",
             f"  C-1 бенчмарки ......... "
             f"{coverage.benchmarks_ok}/{coverage.benchmarks_total}"]
    lines += [f"    🔴 нет истории: {t}" for t in coverage.missing_factors]
    return lines


def format_summary(outcome: ApplyOutcome) -> str:
    """Сводка в чат.  Форма ТА ЖЕ, что печатает CLI (`OPERATOR_STOOQ §8.1`).

    Совпадение намеренное: регламент и бот обязаны говорить на одном языке,
    иначе оператор, читавший инструкцию, не узнает в ответе бота то же самое
    событие.
    """
    lines: list[str] = []
    if not outcome.ok:
        lines.append(f"🔴 {outcome.file_name} НЕ применён")
        lines.append(f"   {outcome.reason}")
        lines += list(outcome.warnings)
        return "\n".join(lines)

    result = outcome.result
    lines.append(f"✅ применён {outcome.file_name}")
    if result is not None:
        lines.append(f"  строк прочитано ....... {result.rows_read}")
        lines.append(f"  баров записано ........ {result.rows_written}")
        if outcome.kind == "history":
            lines.append(f"  инструментов заведено . {result.instruments_added}")
        if result.rejected:
            lines.append("  отброшено:")
            for rule, count in sorted(result.rejected.items(),
                                      key=lambda kv: -kv[1]):
                lines.append(f"    {rule:.<26} {count}")
        if result.rows_written == 0:
            lines.append("  ⚠️ ни один бар не лёг — день в журнал НЕ записан")
    lines += _c1_lines(outcome.c1)
    lines.append(f"  база опубликована · поколение {outcome.generation}")
    if outcome.days_left is not None:
        lines.append(f"  до блокировки ручного тира: {outcome.days_left} дн.")
    lines += list(outcome.warnings)
    return "\n".join(lines)


def format_status(state: StoreStatus) -> str:
    if not state.ok:
        return f"🔴 база недоступна\n   {state.reason}\n   хранилище: {state.storage}"
    lines = [f"📊 база котировок · {state.storage}",
             f"  поколение ............. {state.generation}",
             f"  размер ................ {state.size / (1024 * 1024):.1f} МБ",
             f"  инструментов .......... {state.instruments}"]
    for market in state.markets:
        age = "—" if market.stale_days is None else f"{market.stale_days} дн."
        lines.append(f"  {market.market:.<22} последний день {market.latest} "
                     f"(возраст {age}, бумаг {market.instruments})")
    if state.days_left is not None:
        mark = "🔴" if state.days_left <= 2 else "  "
        lines.append(f"{mark} до блокировки ручного тира: {state.days_left} дн.")
    lines += _c1_lines(state.c1)
    if state.missing_dates:
        listed = " ".join(str(d) for d in state.missing_dates[:12])
        lines.append(f"  ⚠️ не хватает дней: {len(state.missing_dates)} — {listed}")
    if state.last_run:
        lines.append(f"  последняя операция .... {state.last_run.get('file')} "
                     f"({state.last_run.get('rows_written')} баров)")
    return "\n".join(lines)


__all__ = [
    "ApplyOutcome",
    "C1Coverage",
    "MAX_UPLOAD_BYTES",
    "MarketState",
    "StoreStatus",
    "UploadDecision",
    "apply_daily",
    "apply_history",
    "classify_upload",
    "format_status",
    "format_summary",
    "missing_dates",
    "status",
]
