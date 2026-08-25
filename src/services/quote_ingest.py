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

import json
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

#: Сколько прошлых прогонов берётся за «норму» при сверке отказов.
#: Пять — это рабочая неделя: меньше даёт шум одного плохого дня, больше
#: затягивает в норму то, что уже перестало быть нормой.
REJECTION_HISTORY_RUNS = env_int("INGEST_REJECTION_HISTORY", 5, lo=2, hi=60)

#: Во сколько раз отказ должен превысить исторический максимум, чтобы стать
#: тревогой.  Категория, которой РАНЬШЕ НЕ БЫЛО ВООБЩЕ, тревожна при любом
#: ненулевом числе — там сравнивать не с чем.
REJECTION_SPIKE_FACTOR = env_int("INGEST_REJECTION_SPIKE", 3, lo=2, hi=100)


def _rejection_anomaly(conn: sqlite3.Connection,
                       rejected: dict) -> list[str]:
    """Сверить отказы ЭТОГО файла с недавними прогонами (`§−107`).

    🔴 Повод — живой разбор 25.08. Дневной срез за 24.08 дал «чужая дата» 166
    и «нет цены закрытия» 43, тогда как в ЧЕТЫРЁХ предыдущих прогонах таких
    строк не было НИ ОДНОЙ, а «индекс/FX» вырос с ~57 до 154. Записалось 631
    бара вместо привычных 800. Сводка всё это честно печатала — и всё равно
    выглядела успешной: `✅ применён`. Чтобы понять, что копия файла битая,
    оператору пришлось вручную поднять историю и сравнить пять прогонов.

    Сравнивать было С ЧЕМ: `ingest_runs` лежит в той же базе и хранит разбивку
    отказов JSON-ом. То есть факт был, а вывода из него никто не делал — тот же
    класс, что «пустая коллекция ≠ всё хорошо» (`§−90` A-3).

    Зовётся ДО `apply_batch`, поэтому текущий прогон в историю ещё не попал.
    Любая ошибка чтения истории — молчание, а не отказ: сверка полезна, но
    заливка от неё не зависит.
    """
    if not rejected:
        return []
    try:
        rows = conn.execute(
            "SELECT rejected FROM ingest_runs WHERE kind='apply' "
            "ORDER BY id DESC LIMIT ?", (REJECTION_HISTORY_RUNS,)).fetchall()
    except sqlite3.Error:
        return []
    history: list[dict] = []
    for row in rows:
        try:
            parsed = json.loads(row["rejected"] or "{}")
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            history.append(parsed)
    if len(history) < 2:            # одному прогону верить не за что
        return []

    out: list[str] = []
    for rule, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        seen = [int(h.get(rule, 0) or 0) for h in history]
        peak = max(seen)
        if peak == 0:
            out.append(
                f"⚠️ «{rule}»: {count} строк, а в предыдущих {len(history)} "
                "прогонах таких не было НИ ОДНОЙ. Похоже, копия файла "
                "неполная — перекачайте и пришлите заново (повтор безвреден).")
        elif count >= peak * REJECTION_SPIKE_FACTOR:
            out.append(
                f"⚠️ «{rule}»: {count} строк против {peak} максимум за "
                f"последние {len(history)} прогонов — рост в {count / peak:.1f}×.")
    return out


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

    with tempfile.TemporaryDirectory(prefix="ombri-ingest-") as tmp:
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

            # §−107: сверка ДО записи — иначе текущий прогон попадёт в свою же
            # «норму» и разбавит её.
            warnings.extend(_rejection_anomaly(conn, batch.rejected))

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
    with tempfile.TemporaryDirectory(prefix="ombri-status-") as tmp:
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
    """Даты, которые бот применял, а в базе их нет.

    🔴 Возвращает ТОЛЬКО список и потому не годится для ответа человеку:
    пустой кортеж означает и «всё на месте», и «базу прочитать не удалось».
    Печатать их одинаково — ровно дефект `§−90` A-3 («пустая коллекция ≠ всё
    хорошо»). Ответ в чат строится из `status()`, где это различие сохранено.
    """
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



# ═════════════════════════════════════════════════════════════════════════════
# IB-5/IB-6 — диагностика: покрытие, бумага, чистка
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TickerProbe:
    """Что база знает о тикере ДВИЖКА.

    Спрашивается именно тикером движка, а не формой источника: `resolve_symbol`
    перебирает формы сам (`BRK.B` → `brk-b.us`), и оператор, добравший бумагу,
    обязан иметь возможность проверить ровно то, что спросит отчёт. Прежняя
    редакция `verify-universe` спрашивала символами ИСТОЧНИКА и рапортовала
    полное покрытие для отсутствующих бумаг (`AUDIT §−80`, дефект 5).
    """

    ticker: str
    found: bool
    #: 🔴 Ответило ли хранилище вообще. Различать ОБЯЗАТЕЛЬНО: «в базе этой
    #: бумаги нет» и «базы нет» — разные утверждения, и второе не даёт права
    #: говорить про бумагу ничего. Инвариант не новый: ровно это записано в
    #: `manual_portfolio.PreflightCoverage.answered`, и первая редакция
    #: `check_ticker` его нарушала — отсутствие базы выглядело как «бумаги
    #: нет», а совет «пришлите файл истории» вёл оператора чинить не то.
    answered: bool = True
    reason: Optional[str] = None
    source_symbol: Optional[str] = None
    market: Optional[str] = None
    currency: Optional[str] = None
    match_kind: Optional[str] = None
    last_bar: Optional[int] = None
    bars: int = 0

    @property
    def substituted(self) -> bool:
        return bool(self.found and self.match_kind and self.match_kind != "exact")


@dataclass(frozen=True)
class UniverseReport:
    ok: bool
    reason: Optional[str] = None
    factors: tuple[TickerProbe, ...] = ()
    benchmarks: tuple[TickerProbe, ...] = ()

    @property
    def factors_ok(self) -> int:
        return sum(1 for p in self.factors if p.found and p.bars)

    @property
    def benchmarks_ok(self) -> int:
        return sum(1 for p in self.benchmarks if p.found and p.bars)

    @property
    def complete(self) -> bool:
        return bool(self.factors) and self.factors_ok == len(self.factors)


@dataclass(frozen=True)
class PruneOutcome:
    ok: bool
    dry_run: bool
    reason: Optional[str] = None
    removed: tuple[str, ...] = ()
    kept: int = 0
    bars_removed: int = 0
    min_bars_used: int = 0
    busiest_bars: int = 0
    published: bool = False
    conflict: bool = False
    generation: Optional[int] = None


def _probe(conn: sqlite3.Connection, ticker: str) -> TickerProbe:
    found = si.resolve_symbol(conn, ticker)
    if found is None:
        return TickerProbe(ticker=str(ticker), found=False)
    row = conn.execute(
        "SELECT COUNT(*) AS n, MAX(trade_date) AS last FROM daily_bars "
        "WHERE instrument_id=?", (int(found["id"]),)).fetchone()
    return TickerProbe(
        ticker=str(ticker), found=True, source_symbol=str(found["sym"]),
        market=str(found["market"]), currency=str(found["ccy"]),
        match_kind=str(found["kind"]),
        bars=int(row["n"] or 0) if row else 0,
        last_bar=int(row["last"]) if row and row["last"] is not None else None)


def check_ticker(ticker: str, *,
                 publisher: Optional[QuotePublisher] = None) -> TickerProbe:
    """Есть ли бумага в базе под тикером ДВИЖКА — и под какой формой источника.

    🔴 Команда сверх плана, и вот зачем. Добор бумаги (IB-5) без ответа на этот
    вопрос доделан наполовину: оператор видит «инструмент заведён», но не
    знает, найдёт ли его отчёт. Формы расходятся ровно там, где это опаснее
    всего — `BRK.B` лежит у Stooq как `brk-b.us`, и ни один наивный кандидат не
    совпадает (ловушка №7, `STOOQ_CONVENTION §3.7`).
    """
    publisher = publisher or publisher_from_env()
    with tempfile.TemporaryDirectory(prefix="ombri-check-") as tmp:
        local = Path(tmp) / "prices.sqlite"
        try:
            publisher.download(local)
            conn = _open_checked(local)
        except PublisherUnavailable as exc:
            return TickerProbe(ticker=str(ticker), found=False,
                               answered=False, reason=str(exc))
        try:
            return _probe(conn, str(ticker).strip().upper())
        finally:
            conn.close()


def universe_report(*, publisher: Optional[QuotePublisher] = None) -> UniverseReport:
    """Допуск C-1 с ГЛУБИНОЙ истории, а не только с фактом наличия.

    Наличие бары не гарантирует пригодности: факторная регрессия строится на
    окне, и бумага с обрывочной историей схлопывает окно ВСЕЙ панели
    (F-15/F-21). Поэтому в отчёте число баров, а не галочка.
    """
    publisher = publisher or publisher_from_env()
    try:
        factor_etfs, benchmark_etfs = _engine_universe()
    except ImportError as exc:
        name = getattr(exc, "name", None) or "зависимость"
        return UniverseReport(
            ok=False,
            reason=(f"{name} не установлен — список факторов живёт в "
                    "`data_checks` и берётся ОТТУДА, а не копируется сюда. "
                    "Вторая копия однажды разошлась бы с первой молча."))
    with tempfile.TemporaryDirectory(prefix="ombri-universe-") as tmp:
        local = Path(tmp) / "prices.sqlite"
        try:
            publisher.download(local)
            conn = _open_checked(local)
        except PublisherUnavailable as exc:
            return UniverseReport(ok=False, reason=str(exc))
        try:
            return UniverseReport(
                ok=True,
                factors=tuple(_probe(conn, t) for t in factor_etfs),
                benchmarks=tuple(_probe(conn, t) for t in benchmark_etfs))
        finally:
            conn.close()


def prune(*, dry_run: bool, actor: str,
          publisher: Optional[QuotePublisher] = None) -> PruneOutcome:
    """Вычистить бумаги с обрывочной историей.

    🔴 **Единственная операция, где база УМЕНЬШАЕТСЯ**, поэтому гард обвала
    размера здесь не применяется: `prune_thin_instruments` завершается
    `VACUUM`, и сжатие файла — это её работа, а не признак поломки. Гард
    существует для дельты, которая умеет только добавлять.

    🔴 **Без ядра чистка НЕ запускается.** `keep` — это факторные ETF в формах
    источника; удалить их значило бы уронить допуск C-1, а с ним и ручной тир
    целиком (`data_checks`: в профиле STRICT потеря любого фактора даёт BLOCK).
    Пустой `keep` при недоступном `data_checks` — это не «ничего не защищаем»,
    а «не знаем, что защищать», и разница между ними стоит базы.
    """
    publisher = publisher or publisher_from_env()
    try:
        keep = _working_set()
    except ImportError as exc:
        name = getattr(exc, "name", None) or "зависимость"
        return PruneOutcome(
            ok=False, dry_run=dry_run,
            reason=(f"{name} не установлен — не могу определить ядро, которое "
                    "нельзя удалять. Чистка без него срезала бы факторные ETF "
                    "и заблокировала ручной тир."))

    with tempfile.TemporaryDirectory(prefix="ombri-prune-") as tmp:
        local = Path(tmp) / "prices.sqlite"
        try:
            snapshot = publisher.download(local)
            conn = _open_checked(local)
        except PublisherUnavailable as exc:
            return PruneOutcome(ok=False, dry_run=dry_run, reason=str(exc))
        cursor = publisher.read_cursor() or Cursor()
        try:
            # Даты курсора ДО чистки: то, что пропало раньше, — чужой откат и
            # остаётся тревогой. То, что пропадёт из-за нас, тревогой быть не
            # вправе, иначе бот пошлёт оператору на телефон ложный сигнал
            # «база откатилась» и отправит искать файлы, которые сам же удалил.
            present_before = _dates_present(conn, cursor.applied_dates)
            report = si.prune_thin_instruments(conn, keep=keep, dry_run=dry_run)
            present_after = _dates_present(conn, cursor.applied_dates)
        finally:
            conn.close()
        dropped_by_us = present_before - present_after

        common = dict(removed=tuple(report.removed), kept=report.kept,
                      bars_removed=report.bars_removed,
                      min_bars_used=report.min_bars_used,
                      busiest_bars=report.busiest_bars)
        if dry_run or not report.removed:
            return PruneOutcome(ok=True, dry_run=dry_run, **common)

        upload = publisher.upload(local, if_generation_match=snapshot.generation)

    if upload.conflict:
        return PruneOutcome(
            ok=False, dry_run=False, conflict=True, **common,
            reason=("🔴 база изменилась, пока я чистил. Ничего не опубликовано — "
                    "повторите /prune."))
    if not upload.published:
        return PruneOutcome(ok=False, dry_run=False, **common,
                            reason=f"опубликовать не удалось: {upload.reason}")

    survived = Cursor(
        applied_dates=tuple(d for d in cursor.applied_dates
                            if d not in dropped_by_us),
        generation=cursor.generation, at=cursor.at, last_run=cursor.last_run)
    publisher.write_cursor(survived.with_applied(
        None, generation=upload.generation,
        last_run={"file": "prune", "actor": str(actor), "kind": "prune",
                  "removed": len(report.removed),
                  "dates_dropped": len(dropped_by_us)}))
    return PruneOutcome(ok=True, dry_run=False, published=True,
                        generation=upload.generation, **common)


def _working_set() -> list[str]:
    """Ядро в формах ИСТОЧНИКА — то, что `prune` не удаляет ни при каком пороге.

    🔴 Два разных списка, и путать их нельзя (`AUDIT §−80`). `coverage_report`
    и `_probe` спрашивают тикером ДВИЖКА (`resolve_symbol` сам переберёт
    формы); `prune` сравнивает с `instruments.source_symbol`, то есть с формой
    ИСТОЧНИКА. Подать сюда тикеры движка значило бы не защитить ничего —
    молча, потому что для US-бумаг формы совпадают и ошибка проявилась бы
    только на бумаге с нестандартным суффиксом.
    """
    from finance import stooq_symbols as sym                # noqa: PLC0415

    factor_etfs, benchmark_etfs = _engine_universe()
    out: list[str] = []
    for ticker in list(factor_etfs) + list(benchmark_etfs):
        for candidate in sym.candidates_for(ticker):
            if candidate not in out:
                out.append(candidate)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# IB-7 — напоминания
# ═════════════════════════════════════════════════════════════════════════════

#: За сколько дней до блокировки ручного тира начинать предупреждать.
#: Три календарных дня при пороге 7 означают: после пятничной заливки первое
#: напоминание придёт во ВТОРНИК, если понедельничный файл не применён.
#: Меньше — и выходные давали бы ложную тревогу каждую субботу; больше — и
#: предупреждение перестало бы отличаться от фона.
REMIND_DAYS_LEFT = env_int("INGEST_REMIND_DAYS_LEFT", 3, lo=1, hi=30)


def build_reminder(state: StoreStatus) -> Optional[str]:
    """Текст напоминания — либо `None`, если говорить не о чем.

    🔴 `None` здесь обязателен, и это не оптимизация. Напоминание, приходящее
    каждый день независимо от состояния, перестают читать за неделю — и тогда
    оно не сработает в тот единственный раз, когда было нужно.

    Порог считается от того же числа, которым провайдер БЛОКИРУЕТ отчёт
    (`MAX_MARKET_STALE_DAYS`), а не от отдельной константы: иначе бот обещал
    бы один срок, а ручной тир отказывал по другому.
    """
    if not state.ok:
        return (f"🔴 база котировок недоступна: {state.reason}\n"
                "Ручной тир сейчас не построит ни одного отчёта.")
    if state.missing_dates:
        listed = " ".join(str(d) for d in state.missing_dates[:12])
        tail = "" if len(state.missing_dates) <= 12 else " …"
        return (f"🔴 в базе не хватает {len(state.missing_dates)} дн., которые я "
                f"применял: {listed}{tail}\n"
                "Похоже, базу перезаливали. Перешлите эти файлы — повтор безвреден.")
    left = state.days_left
    if left is not None and left <= 0:
        return ("🔴 ручной тир УЖЕ заблокирован: база не обновлялась дольше "
                "порога. Пришлите свежий дневной срез.")
    if left is not None and left <= REMIND_DAYS_LEFT:
        return (f"⚠️ до блокировки ручного тира {left} дн. "
                "Пришлите свежий дневной срез со stooq.com/db/.")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Сводки диагностики
# ═════════════════════════════════════════════════════════════════════════════

def format_universe(report: UniverseReport) -> str:
    if not report.ok:
        return f"🔴 покрытие не проверено\n   {report.reason}"
    mark = "✅" if report.complete else "🔴"
    lines = [f"🎯 допуск C-1 · факторы {report.factors_ok}/{len(report.factors)} "
             f"{mark} · бенчмарки {report.benchmarks_ok}/{len(report.benchmarks)}"]
    for probe in report.factors + report.benchmarks:
        if not probe.found:
            lines.append(f"  🔴 {probe.ticker:.<12} НЕТ В БАЗЕ")
            continue
        note = ""
        if probe.substituted:
            note = f"  ⚠️ подмена площадки → {probe.source_symbol}"
        lines.append(f"  {probe.ticker:.<12} {probe.bars:>5} баров, "
                     f"последний {probe.last_bar}{note}")
    if not report.complete:
        lines.append("🔴 в профиле STRICT потеря ЛЮБОГО фактора даёт BLOCK — "
                     "ручной тир отдаст отказ вместо отчёта.")
    return "\n".join(lines)


def format_probe(probe: TickerProbe) -> str:
    if not probe.answered:
        # Совет «пришлите файл истории» здесь был бы вредным: применять его
        # некуда, и оператор пошёл бы чинить не то.
        return (f"🔴 не могу ответить про {probe.ticker}: хранилище недоступно.\n"
                f"   {probe.reason}")
    if not probe.found:
        return (f"🔴 {probe.ticker} в базе НЕТ ни под одной формой символа.\n"
                "   Пришлите файл истории этой бумаги из архива — она заведётся.")
    lines = [f"✅ {probe.ticker} найден",
             f"  форма источника ....... {probe.source_symbol}",
             f"  рынок / валюта ........ {probe.market} / {probe.currency}",
             f"  баров в базе .......... {probe.bars}",
             f"  последний бар ......... {probe.last_bar}"]
    if probe.substituted:
        lines.append("  ⚠️ ПОДМЕНА ПЛОЩАДКИ: это другая биржа, другая цена и, "
                     "возможно, другая валюта. Решение ваше, но оно обязано "
                     "быть видимым.")
    return "\n".join(lines)


def format_prune(outcome: PruneOutcome) -> str:
    if not outcome.ok:
        return f"🔴 чистка не выполнена\n   {outcome.reason}"
    lines = [f"  порог истории ......... {outcome.min_bars_used} баров "
             f"(у самой полной бумаги базы — {outcome.busiest_bars})",
             f"  под нож ............... {len(outcome.removed)} бумаг, "
             f"{outcome.bars_removed} баров",
             f"  остаётся .............. {outcome.kept} бумаг"]
    lines += [f"    − {s}" for s in outcome.removed[:15]]
    if len(outcome.removed) > 15:
        lines.append(f"    … и ещё {len(outcome.removed) - 15}")
    if outcome.dry_run:
        lines.append("режим --dry-run: база НЕ изменена.")
    elif outcome.published:
        lines.append(f"✅ база вычищена и опубликована · поколение "
                     f"{outcome.generation}")
    else:
        lines.append("✅ чистить нечего.")
    return "\n".join(lines)


def format_missing(state: StoreStatus) -> str:
    """Ответ на `/missing`.  Принимает СТАТУС, а не список дат.

    🔴 Разница принципиальная и стоила отдельной правки: список сам по себе
    пуст и когда всё в порядке, и когда базу не удалось прочитать. Первая
    редакция печатала на оба случая зелёное «всё на месте» — то есть самый
    громкий отказ выглядел как самый спокойный ответ.
    """
    if not state.ok:
        return (f"🔴 не могу сказать: база недоступна.\n   {state.reason}")
    if not state.missing_dates:
        return "✅ пропавших дней нет: всё, что я применял, в базе на месте."
    listed = "\n".join(f"  {d}_d.txt" for d in state.missing_dates)
    return (f"⚠️ не хватает {len(state.missing_dates)} дн. — перешлите эти "
            f"файлы из applied/:\n{listed}\nПорядок не важен, повтор безвреден.")

__all__ = [
    "ApplyOutcome",
    "C1Coverage",
    "MAX_UPLOAD_BYTES",
    "MarketState",
    "PruneOutcome",
    "REMIND_DAYS_LEFT",
    "StoreStatus",
    "TickerProbe",
    "UniverseReport",
    "UploadDecision",
    "apply_daily",
    "apply_history",
    "build_reminder",
    "check_ticker",
    "classify_upload",
    "format_missing",
    "format_probe",
    "format_prune",
    "format_status",
    "format_summary",
    "format_universe",
    "missing_dates",
    "prune",
    "status",
    "universe_report",
]
