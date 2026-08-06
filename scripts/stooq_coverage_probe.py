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

ЗАПУСК
======
    # без сети — показать, ЧТО будет проверяться (работает всегда):
    PYTHONPATH=src python scripts/stooq_coverage_probe.py --dry-run

    # замер (нужен доступ к stooq.com и ключ):
    STOOQ_API_KEY=... PYTHONPATH=src python scripts/stooq_coverage_probe.py \
        --out docs/audit/STOOQ_CONVENTION.md

Результат — markdown-таблица «тикер → есть/нет → длина ряда → формат» по шести
группам плюс наблюдения по конвенции корректировок.
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

    @property
    def ok(self) -> bool:
        return bool(self.found_as) and self.observations >= MIN_OBSERVATIONS


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
