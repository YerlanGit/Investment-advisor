#!/usr/bin/env python3
"""MP-08 · ЧК-08.1 — замер покрытия Stooq и конвенции корректировок.

<!-- nav | area:scripts | code:docs/audit/STOOQ_CONVENTION.md -->

ПОЧЕМУ ЭТО СКРИПТ, А НЕ ТЕСТ
============================
Фаза 08 не выполняется агентом и не выполняется в CI — так записано в
`PHASE_08 §0`, и это перепроверено 2026-08-06:

    $ curl https://stooq.com/q/d/l/?s=aapl.us&i=d
    curl: (56) CONNECT tunnel failed, response 403

Плюс `STOOQ_API_KEY` добывается человеком через капчу. Поэтому замер обязан
выполнить ОПЕРАТОР в окружении с доступом к сети, а не агент по памяти.

Что здесь автоматизировано — ровно то, что автоматизируется без сети:
СПИСОК бумаг, который надо проверить, собирается ИЗ КОДА. Держать его руками
в markdown-таблице нельзя: за две недели после написания `PHASE_08 §2.5`
список успел разойтись с движком трижды (прокси переведены на EM в `§−38`,
KZ-синонимы добавлены в `§−70`, а `VWO.US` из плана вообще не используется —
`EM_EQUITY_FALLBACK` не читает никто).

ДВА ПУТИ ЗАМЕРА, И ВТОРОЙ ЛУЧШЕ
===============================
**(а) API по тикеру** — исходный план: ключ через капчу, запрос на бумагу,
дневная квота. Требует сети и ключа.

**(б) BULK-выгрузка** (`https://stooq.com/db/`, находка владельца 2026-08-09) —
вся история одним архивом, без API и без ключа. Оператор скачивает файл
браузером, дальше замер идёт **ОФЛАЙН** по локальному файлу: ни сети, ни
квоты, ни капчи в момент измерения.

Путь (б) снимает ОБА препятствия `PHASE_08 §0` и потому предпочтителен. Он же
меняет Фазу 09 и шаг 11.7 витрины — разбор в `PHASE_08 §2.7`.

ЗАПУСК
======
    # без сети — показать, ЧТО будет проверяться (работает всегда):
    PYTHONPATH=src python scripts/stooq_coverage_probe.py --dry-run

    # (б) ЗАМЕР ПО BULK-ВЫГРУЗКЕ — сети НЕ требует:
    PYTHONPATH=src python scripts/stooq_coverage_probe.py \
        --dump ~/Downloads/d_us_txt.zip --dump ~/Downloads/d_world_txt.zip \
        --out docs/audit/STOOQ_CONVENTION.md

    # (а) замер по API (нужен доступ к stooq.com и ключ):
    STOOQ_API_KEY=... PYTHONPATH=src python scripts/stooq_coverage_probe.py \
        --out docs/audit/STOOQ_CONVENTION.md

Результат — markdown-таблица «тикер → есть/нет → длина ряда → формат» по шести
группам плюс наблюдения по конвенции корректировок.

🔴 ФОРМАТ ВЫГРУЗКИ НЕ ЗАШИТ, А ОПРЕДЕЛЯЕТСЯ. Ни раскладка каталогов, ни шапка
файла здесь не утверждаются: живого архива у автора этого скрипта не было.
Читатель терпим к обоим известным вариантам шапки и сообщает, что увидел
ФАКТИЧЕСКИ, — наблюдение попадает в артефакт. Утверждать формат по памяти
означало бы ровно ту ошибку, из-за которой в задании появилась E-4.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Optional

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

#: Минимум наблюдений, ниже которого ряд бесполезен для ковариации.
#: 1260 торговых дней ≈ окно `HISTORY_LOOKBACK_DAYS=1825` календарных.
MIN_OBSERVATIONS = 1260

#: Базовый эндпоинт выгрузки. Ключ подставляется параметром, НЕ в путь —
#: и никогда не печатается (S-5: в логах маска).
STOOQ_CSV_URL = "https://stooq.com/q/d/l/"


@dataclass
class Probe:
    """Одна бумага, которую надо проверить."""

    group: str
    ticker: str                  # форма движка (Tradernet): `AAPL.US`
    purpose: str                 # зачем она нужна — и что значит её отсутствие
    candidates: list[str] = field(default_factory=list)   # формы Stooq


@dataclass
class ProbeResult:
    probe: Probe
    found_as: Optional[str] = None
    observations: int = 0
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    error: Optional[str] = None
    #: Путь внутри архива — нужен, чтобы раскладку можно было перепроверить
    #: руками, а не поверить таблице.
    member: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.found_as) and self.observations >= MIN_OBSERVATIONS


@dataclass
class DumpFormat:
    """Что ФАКТИЧЕСКИ обнаружено в выгрузке. Не утверждение, а наблюдение."""

    header: Optional[str] = None
    delimiter: str = ","
    date_column: Optional[str] = None
    date_pattern: Optional[str] = None       # 'YYYY-MM-DD' | 'YYYYMMDD'
    close_column: Optional[str] = None
    members_seen: int = 0
    sample_member: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
# 1. Универсум — СОБИРАЕТСЯ ИЗ КОДА
# ═════════════════════════════════════════════════════════════════════════════

def build_universe() -> list[Probe]:
    """Что обязано найтись у Stooq, чтобы ручной ввод заработал.

    Каждая группа берётся из своего SSOT в коде, а не из списка в документе:
    добавили факторный ETF или прокси — список замера обновится сам, и
    оператор проверит то, что движок действительно спрашивает.
    """
    from finance.demo_portfolio import DEMO_ALLOCATION
    from finance.investment_logic import MAC3RiskEngine as E

    # `factor_tickers` — атрибут ЭКЗЕМПЛЯРА, а не класса: панель факторов
    # задаётся в `__init__`. Конструктор сети не трогает (проверено), поэтому
    # экземпляр здесь безопасен и в офлайне.
    engine = E()

    probes: list[Probe] = []

    def add(group: str, ticker: str, purpose: str) -> None:
        t = str(ticker).upper().strip()
        if any(p.ticker == t and p.group == group for p in probes):
            return
        probes.append(Probe(group=group, ticker=t, purpose=purpose,
                            candidates=stooq_candidates(t)))

    # A. Факторы — стоп-фактор запуска: без них факторной модели нет.
    for name, ticker in engine.factor_tickers.items():
        add("A. Факторы", ticker, f"фактор «{name}» — C-1 BLOCK в профиле STRICT")

    # B. Бенчмарки — DEGRADE: страдает сравнение, не модель.
    for ticker in E.BENCHMARK_EXTRA:
        add("B. Бенчмарки", ticker, "сравнение с бенчмарком и TE")

    # C. KZ/IL — определяют, для какой аудитории `manual` вообще имеет смысл.
    for synonym, ticker in E.TICKER_MAP.items():
        if ticker.endswith((".KZ", ".IL")):
            add("C. KZ / IL", ticker,
                f"KZ-бумага (синоним «{synonym}»); непокрытие → C-8 по стоимости")

    # D. Крипто — есть даже в демо-портфеле.
    for ticker in ("BTC-USD", "ETH-USD"):
        add("D. Крипто", ticker, "крипто-позиция ручного ввода")

    # E. Прокси — на них резолвятся неликвидные бумаги.
    for cat, ticker in sorted(E.BOND_PROXIES.items()):
        add("E. Прокси", ticker, f"прокси облигаций класса {cat}")
    for cat, ticker in sorted(E.INSTRUMENT_PROXY_MAP.items()):
        add("E. Прокси", ticker, f"прокси инструментов «{cat}»")

    # F. Демо — вход для ★-решения «переводить ли витрину на Stooq».
    for ticker, _w, _c in DEMO_ALLOCATION:
        canon = E.canonical_ticker(ticker)
        if canon:
            add("F. Демо", canon, "тикер демо-портфеля")

    return probes


def stooq_candidates(ticker: str) -> list[str]:
    """Формы символа Stooq, которые стоит попробовать для тикера движка.

    🔴 Соответствие форматов — ФАКТ О ВНЕШНЕМ МИРЕ, а не о коде, и здесь оно
    НЕ УТВЕРЖДАЕТСЯ. `aapl.us` — единственная форма, про которую задание
    говорит уверенно; для KZ/IL суффиксы Stooq не проверены (`PHASE_08 §2.5`),
    поэтому кандидатов несколько и решает ЗАМЕР, а не эта функция.

    Здесь же нельзя чинить формат «на будущее»: маппинг, если он окажется
    нетривиальным, живёт в `StooqProvider` (Фаза 09), а не в `resolve_tickers`
    — иначе сломается путь `freedom` (ловушка Т-4, `PHASE_08 §2.5`).
    """
    t = str(ticker).strip().lower()
    if t.endswith("-usd"):                      # крипто: BTC-USD
        base = t[:-4]
        out = [t, f"{base}.v", f"{base}usd"]
    elif "." not in t:
        out = [f"{t}.us"]
    else:
        base, _, suffix = t.rpartition(".")
        if suffix == "us":
            out = [t]
        else:
            # KZ/IL и прочие биржи: форма Stooq неизвестна — пробуем несколько.
            out = [t, base, f"{base}.us"]
    # Порядок сохраняем (он же приоритет), повторы убираем: лишний кандидат —
    # лишний сетевой запрос, а дневная квота Stooq ограничена.
    return list(dict.fromkeys(out))


# ═════════════════════════════════════════════════════════════════════════════
# 2. Замер (нужна сеть)
# ═════════════════════════════════════════════════════════════════════════════

def _http_get(url: str, params: dict) -> str:                # pragma: no cover
    """Минимальный GET. Ключ в логи не попадает НИКОГДА (S-5)."""
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{query}", timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_csv(body: str) -> tuple[int, Optional[str], Optional[str], Optional[str]]:
    """CSV Stooq → (число наблюдений, первая дата, последняя дата, ошибка).

    Тело ошибки у Stooq приходит с кодом 200 и НЕ является CSV («Exceeded the
    daily hits limit»). Отличать его обязательно: иначе исчерпанная квота
    выглядит как «бумаги нет», и таблица покрытия окажется ложной ровно в тот
    день, когда её составляли.
    """
    text = (body or "").strip()
    if not text:
        return 0, None, None, "пустой ответ"
    head = text.splitlines()[0].strip().lower()
    if not head.startswith("date"):
        return 0, None, None, f"не CSV: {text[:120]}"
    rows = list(csv.DictReader(io.StringIO(text)))
    dates = [r.get("Date") for r in rows if r.get("Date")]
    if not dates:
        return 0, None, None, "CSV без строк"
    return len(dates), dates[0], dates[-1], None


def probe_one(probe: Probe, api_key: str,
              http_get: Callable = _http_get) -> ProbeResult:
    """Перебрать формы символа и вернуть первую, давшую ряд."""
    result = ProbeResult(probe=probe)
    errors: list[str] = []
    for candidate in probe.candidates:
        try:
            body = http_get(STOOQ_CSV_URL, {"s": candidate, "i": "d",
                                            "apikey": api_key})
        except Exception as exc:                             # pragma: no cover
            errors.append(f"{candidate}: {type(exc).__name__}")
            continue
        n, first, last, err = parse_csv(body)
        if err:
            errors.append(f"{candidate}: {err}")
            continue
        result.found_as, result.observations = candidate, n
        result.first_date, result.last_date = first, last
        return result
    result.error = "; ".join(errors) if errors else "не найден"
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 2b. BULK-выгрузка (`stooq.com/db/`) — замер БЕЗ сети и без ключа
# ═════════════════════════════════════════════════════════════════════════════

#: Как в шапке файла может называться дата и закрытие. Оба варианта известны из
#: практики Stooq: «человеческий» CSV (`Date,Open,…`) и ASCII-экспорт
#: (`<TICKER>,<PER>,<DATE>,…`). Какой отдаёт bulk — устанавливает ЗАМЕР.
_DATE_KEYS = ("date", "<date>")
_CLOSE_KEYS = ("close", "<close>")


def _norm_symbol(name: str) -> str:
    """`.../daily/us/nasdaq stocks/1/aapl.us.txt` → `aapl.us`.

    Нормализуем ИМЯ ФАЙЛА, а не путь: раскладка каталогов у выгрузки своя и
    заранее неизвестна, а вот имя файла — это и есть символ Stooq. Так индекс
    не зависит от того, как именно архив разложен внутри.
    """
    stem = Path(str(name)).name
    for suffix in (".txt", ".csv"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.strip().lower()


class DumpIndex:
    """Индекс bulk-выгрузки: символ → член архива. Открывается один раз.

    Принимает и `.zip`, и уже распакованный каталог: оператору не нужно решать,
    как именно распаковывать, а тесты обходятся без zip-фикстуры.

    Читает ЛЕНИВО: `namelist()` у zip не разворачивает содержимое, а нам нужны
    ~34 файла из десятков тысяч. Полная распаковка на Cloud Run была бы
    вредна — там `/tmp` это RAM при лимите 2 GiB (`PHASE_07 §3`).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._zip = None
        self._index: dict[str, str] = {}
        if self.path.is_dir():
            for member in self.path.rglob("*"):
                if member.is_file():
                    self._index.setdefault(_norm_symbol(member.name),
                                           str(member.relative_to(self.path)))
        else:
            import zipfile

            self._zip = zipfile.ZipFile(self.path)
            for name in self._zip.namelist():
                if name.endswith("/"):
                    continue
                self._index.setdefault(_norm_symbol(name), name)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, symbol: str) -> bool:
        return str(symbol).lower() in self._index

    def member_for(self, symbol: str) -> Optional[str]:
        return self._index.get(str(symbol).lower())

    def read(self, member: str) -> str:
        if self._zip is not None:
            return self._zip.read(member).decode("utf-8", errors="replace")
        return (self.path / member).read_text(encoding="utf-8", errors="replace")

    def any_member(self) -> Optional[str]:
        return next(iter(self._index.values()), None)


def sniff_format(index: DumpIndex) -> DumpFormat:
    """Определить формат выгрузки по первому же файлу.

    Именно ОПРЕДЕЛИТЬ. Раскладка и шапка bulk-архива в этом репозитории не
    проверялись живьём, поэтому единственный честный ход — прочитать и
    доложить. Результат уходит в артефакт §4 как наблюдение с датой.
    """
    fmt = DumpFormat(members_seen=len(index))
    member = index.any_member()
    if member is None:
        return fmt
    fmt.sample_member = member
    head = index.read(member).lstrip("﻿").splitlines()
    if not head:
        return fmt
    fmt.header = head[0].strip()
    for delim in (",", ";", "\t"):
        if delim in fmt.header:
            fmt.delimiter = delim
            break
    cols = [c.strip().lower() for c in fmt.header.split(fmt.delimiter)]
    fmt.date_column = next((c for c in cols if c in _DATE_KEYS), None)
    fmt.close_column = next((c for c in cols if c in _CLOSE_KEYS), None)
    if len(head) > 1 and fmt.date_column:
        raw = head[1].split(fmt.delimiter)[cols.index(fmt.date_column)].strip()
        fmt.date_pattern = ("YYYY-MM-DD" if "-" in raw
                            else "YYYYMMDD" if raw.isdigit() and len(raw) == 8
                            else f"неизвестен: {raw!r}")
    return fmt


def parse_dump_series(text: str, fmt: DumpFormat) -> list[tuple[str, float]]:
    """Файл выгрузки → `[(дата ISO, close)]`. Мусорные строки пропускаются.

    Возвращает ПАРЫ, а не только счётчик: проверка конвенции (§3) смотрит на
    ступеньку цены в дату сплита, и без самих цен она невозможна.
    """
    lines = (text or "").lstrip("﻿").splitlines()
    if len(lines) < 2:
        return []
    cols = [c.strip().lower() for c in lines[0].split(fmt.delimiter)]
    try:
        i_date = cols.index(fmt.date_column or "")
        i_close = cols.index(fmt.close_column or "")
    except ValueError:
        return []
    out: list[tuple[str, float]] = []
    for line in lines[1:]:
        parts = line.split(fmt.delimiter)
        if len(parts) <= max(i_date, i_close):
            continue
        raw_date, raw_close = parts[i_date].strip(), parts[i_close].strip()
        if len(raw_date) == 8 and raw_date.isdigit():
            raw_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
        try:
            out.append((raw_date, float(raw_close)))
        except ValueError:
            continue                       # строка без цены — не данные
    return out


def probe_one_dump(probe: Probe, indexes: list[DumpIndex],
                   fmt: DumpFormat) -> ProbeResult:
    """Найти бумагу в выгрузке. Перебирает те же формы символа, что и API-путь."""
    result = ProbeResult(probe=probe)
    for candidate in probe.candidates:
        for index in indexes:
            member = index.member_for(candidate)
            if member is None:
                continue
            rows = parse_dump_series(index.read(member), fmt)
            if not rows:
                result.error = f"{candidate}: файл есть, строк нет"
                continue
            result.found_as, result.member = candidate, member
            result.observations = len(rows)
            result.first_date, result.last_date = rows[0][0], rows[-1][0]
            return result
    result.error = result.error or "не найден ни в одной выгрузке"
    return result


def check_convention(indexes: list[DumpIndex], fmt: DumpFormat) -> list[dict]:
    """Есть ли ступенька в дату сплита — по каждому событию `KNOWN_SPLITS`.

    Логика прямая: сравниваем цену закрытия ДО события и В день события.
    Отношение ≈ ratio → ступенька на месте, ряд СЫРОЙ (`RAW`).
    Отношение ≈ 1 → ступеньки нет, ряд скорректирован на сплиты.

    Дивидендную половину вопроса (`SPLIT_ADJUSTED` против `TOTAL_RETURN`) так
    не решить — она требует сверки накопленной доходности `spy.us` с известной
    total-return, и это отдельный шаг оператора (§4 артефакта).
    """
    out: list[dict] = []
    for ticker, when, _what in convention_probes():
        if when is None:
            continue
        candidates = stooq_candidates(ticker)
        rows: list[tuple[str, float]] = []
        found = None
        for candidate in candidates:
            for index in indexes:
                member = index.member_for(candidate)
                if member is not None:
                    rows = parse_dump_series(index.read(member), fmt)
                    found = candidate
                    break
            if rows:
                break
        entry = {"ticker": ticker, "date": when.isoformat(), "found_as": found}
        if not rows:
            entry["verdict"] = "нет в выгрузке"
            out.append(entry)
            continue
        stamp = when.isoformat()
        before = [c for d, c in rows if d < stamp]
        on_after = [c for d, c in rows if d >= stamp]
        if not before or not on_after:
            entry["verdict"] = "событие вне окна ряда"
            out.append(entry)
            continue
        ratio = before[-1] / on_after[0] if on_after[0] else 0.0
        entry["ratio"] = round(ratio, 3)
        # Порог 1.5 — тот же, что у эвристики движка (`_SPLIT_RETURN_THRESHOLD`),
        # чтобы «ступенька» здесь и «ступенька» там значили одно и то же.
        entry["verdict"] = ("ступенька ЕСТЬ → ряд RAW" if ratio > 1.5
                            else "ступеньки нет → ряд скорректирован")
        out.append(entry)
    return out


def render_format_note(fmt: DumpFormat) -> str:
    """Наблюдённый формат — в артефакт, чтобы 11.7 не гадал."""
    if not fmt.header:
        return "Формат выгрузки определить не удалось: архив пуст."
    return "\n".join([
        f"* файлов в выгрузке: **{fmt.members_seen}**",
        f"* пример члена архива: `{fmt.sample_member}`",
        f"* шапка: `{fmt.header}`",
        f"* разделитель: `{fmt.delimiter!r}`",
        f"* колонка даты: `{fmt.date_column}`, формат `{fmt.date_pattern}`",
        f"* колонка закрытия: `{fmt.close_column}`",
    ])


# ═════════════════════════════════════════════════════════════════════════════
# 3. Конвенция корректировок
# ═════════════════════════════════════════════════════════════════════════════

def convention_probes() -> list[tuple[str, Optional[date], str]]:
    """События для §2: `(тикер, дата, что проверяем)`.

    🔴 Даты берутся ИЗ КОДА (`KNOWN_SPLITS`), а не из задания: задание
    указывает для `googl.us` дату **2022-06-06**, тогда как это дата сплита
    AMZN (ошибка E-4). Исполнитель, следующий заданию буквально, искал бы
    ступеньку там, где события не было, не нашёл бы её ни при какой конвенции
    и записал бы «ряд скорректирован» — вывод, не зависящий от данных.
    """
    from freedom_portfolio.history import KNOWN_SPLITS

    out: list[tuple[str, Optional[date], str]] = []
    for ticker, events in sorted(KNOWN_SPLITS.items()):
        for when, ratio in events:
            out.append((ticker, when,
                        f"сплит {ratio:g}:1 — есть ли ступенька ×{ratio:g}"))
    out.append(("SPY.US", None,
                "накопленная доходность против известной total-return SPY"))
    out.append(("BRK-B.US", None,
                "контроль: бездивидендная бумага, расхождений быть не должно"))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4. Отчёт
# ═════════════════════════════════════════════════════════════════════════════

def render_table(results: Iterable[ProbeResult]) -> str:
    """Markdown-таблица «тикер → есть/нет → длина ряда → формат»."""
    lines = ["| Группа | Тикер движка | Найден как | Наблюдений | Период | Вердикт |",
             "|---|---|---|---:|---|---|"]
    for r in results:
        if r.found_as and r.ok:
            verdict = "✅"
        elif r.found_as:
            verdict = f"⚠️ ряд короче {MIN_OBSERVATIONS}"
        else:
            verdict = f"❌ {r.error or 'нет'}"
        period = (f"{r.first_date} … {r.last_date}"
                  if r.first_date and r.last_date else "—")
        lines.append(f"| {r.probe.group} | `{r.probe.ticker}` | "
                     f"{('`' + r.found_as + '`') if r.found_as else '—'} | "
                     f"{r.observations or '—'} | {period} | {verdict} |")
    return "\n".join(lines)


def render_convention(entries: Iterable[dict]) -> str:
    """Таблица «событие → отношение цен → вердикт»."""
    lines = ["| Тикер | Дата сплита | Найден как | Отношение | Вердикт |",
             "|---|---|---|---:|---|"]
    for e in entries:
        lines.append(
            f"| `{e['ticker']}` | {e['date']} | "
            f"{('`' + e['found_as'] + '`') if e.get('found_as') else '—'} | "
            f"{e.get('ratio', '—')} | {e['verdict']} |")
    return "\n".join(lines)


def render_dry_run(probes: Iterable[Probe]) -> str:
    """Что будет проверяться — без единого сетевого запроса."""
    lines = ["| Группа | Тикер движка | Кандидаты Stooq | Зачем |",
             "|---|---|---|---|"]
    for p in probes:
        cands = ", ".join(f"`{c}`" for c in p.candidates)
        lines.append(f"| {p.group} | `{p.ticker}` | {cands} | {p.purpose} |")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:            # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="показать универсум замера без сети")
    parser.add_argument("--dump", type=Path, action="append", default=None,
                        help="bulk-выгрузка stooq.com/db (.zip или каталог); "
                             "можно повторить для нескольких пакетов")
    parser.add_argument("--out", type=Path, default=None,
                        help="куда дописать таблицу результатов")
    args = parser.parse_args(argv)

    probes = build_universe()
    if args.dry_run:
        print(f"# Универсум замера: {len(probes)} бумаг\n")
        print(render_dry_run(probes))
        print("\n# События для конвенции\n")
        for ticker, when, what in convention_probes():
            print(f"* `{ticker}` {when or '—'} — {what}")
        return 0

    if args.dump:
        missing = [p for p in args.dump if not Path(p).exists()]
        if missing:
            print(f"нет такой выгрузки: {missing}", file=sys.stderr)
            return 2
        indexes = [DumpIndex(p) for p in args.dump]
        fmt = sniff_format(indexes[0])
        results = [probe_one_dump(p, indexes, fmt) for p in probes]
        report = "\n\n".join([
            "## Формат выгрузки (НАБЛЮДЁН, не предположен)",
            render_format_note(fmt),
            "## Покрытие",
            render_table(results),
            "## Конвенция: ступенька в дату сплита",
            render_convention(check_convention(indexes, fmt)),
        ])
        print(report)
        failed_a = [r for r in results if r.probe.group.startswith("A") and not r.ok]
        if failed_a:
            print(f"\n🔴 СТОП-ФАКТОР: не покрыто факторов — {len(failed_a)}. "
                  "Решение об исключении фактора из модели принимает владелец "
                  "(PHASE_08 §5), не исполнитель.", file=sys.stderr)
        if args.out:
            args.out.write_text(
                (args.out.read_text(encoding="utf-8") + "\n\n" + report
                 if args.out.exists() else report), encoding="utf-8")
        return 1 if failed_a else 0

    api_key = (os.getenv("STOOQ_API_KEY") or "").strip()
    if not api_key:
        print("STOOQ_API_KEY не задан. Ключ добывается человеком через капчу "
              "(PHASE_08 §0) — замер без него невозможен.", file=sys.stderr)
        return 2

    results = [probe_one(p, api_key) for p in probes]
    table = render_table(results)
    print(table)
    failed_a = [r for r in results if r.probe.group.startswith("A") and not r.ok]
    if failed_a:
        print(f"\n🔴 СТОП-ФАКТОР: не покрыто факторов — {len(failed_a)}. "
              "Решение об исключении фактора из модели принимает владелец "
              "(PHASE_08 §5), не исполнитель.", file=sys.stderr)
    if args.out:
        args.out.write_text(
            args.out.read_text(encoding="utf-8") + "\n\n" + table
            if args.out.exists() else table, encoding="utf-8")
    return 1 if failed_a else 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
