#!/usr/bin/env python3
"""Загрузчик базы котировок Stooq — интерфейс оператора (MP-08.5, ЧК-08.5-5).

<!-- nav | area:scripts | code:src/finance/stooq_ingest.py,src/finance/stooq_store.py | read-before:наполнение базы котировок вручную -->

🔴 **Здесь только разбор аргументов и печать.** Вся логика — в
`src/finance/stooq_ingest.py`, и это не стилистика: каталога `scripts/` нет в
деплой-образе, а тест, который его читает, обязан `skipTest`. Логика,
оставленная здесь, оказалась бы непроверенной во втором прогоне сюиты.

Регламент оператора — `docs/roadmap/manual_portfolio/OPERATOR_STOOQ.md`.

    export STOOQ_ROOT=~/ramp-stooq
    python scripts/stooq_ingest.py bootstrap --archive $STOOQ_ROOT/archive
    python scripts/stooq_ingest.py apply
    python scripts/stooq_ingest.py verify-seam --date 2026-08-07
    python scripts/stooq_ingest.py verify-universe
    gcloud storage cp $STOOQ_ROOT/prices.sqlite gs://ramp-bot-state/stooq/prices.sqlite

🔴 Скрипт запускается НА КОМПЬЮТЕРЕ ОПЕРАТОРА, а не в облаке. `/mnt/state` в
Cloud Run — это gcsfuse-монтирование бакета, писать в него SQLite нельзя
(блокировки на объектном хранилище не работают), да и оператора с шеллом там
нет. В облако едет один готовый файл базы; см. `PHASE_08B §3.1a`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance import stooq_ingest as si                     # noqa: E402


def _missing_name(exc: ImportError) -> str:
    """Имя недостающего модуля. `ImportError.name` бывает пустым."""
    return exc.name or str(exc) or "зависимость"


def _engine_universe():
    """`(факторы, бенчмарки)` из `data_checks` — SSOT, а не копия в скрипте.

    Импорт ЛЕНИВЫЙ, и это ради оператора. `finance.stooq_ingest` — чистый
    stdlib (замерено), поэтому `bootstrap` и `apply` работают на голом Python
    без единой зависимости. `data_checks` тянет pandas и numpy, и требовать их
    ради двух кортежей-констант значило бы заставить человека ставить
    окружение бота, чтобы разобрать текстовый файл.

    Без pandas команда не падает — она честно говорит, что C-1 не проверен.
    Молча пропустить проверку нельзя: C-1 и есть допуск (§2.4 регламента).
    """
    from finance.data_checks import BENCHMARK_ETFS, FACTOR_ETFS

    return FACTOR_ETFS, BENCHMARK_ETFS


#: Корень рабочих каталогов оператора.  Дефолт РЕПО-ЛОКАЛЬНЫЙ, а не
#: `/mnt/state/stooq`: этот путь существует только внутри контейнера Cloud Run,
#: где запускать загрузчик и незачем, и нечем.  Тот же приём, что у
#: `db_tokenomics.DB_PATH`: в проде путь задаёт env, локально — каталог рядом.
DEFAULT_ROOT = Path(os.getenv(
    "STOOQ_ROOT", str(Path(__file__).resolve().parents[1] / "data" / "stooq")))


def _require_archive(path) -> bool:
    """Каталог архива обязан существовать — иначе команда врёт про Stooq.

    🔴 Без этой проверки опечатка в пути читается как вывод о ПОСТАВЩИКЕ:
    `bootstrap` на несуществующем каталоге печатает «файлов обработано 0» и
    следом «🔴 нет истории: SPY.US», то есть буквально утверждает, что в Stooq
    нет SPY. Проект уже трижды делал ложный вывод такого рода из пустого
    перебора (`AUDIT §−76`), и цена у него — отменённая фаза.

    Особенно вероятна опечатка на Windows: путь с пробелами
    (`C:\\Users\\User\\ramp-stooq\\archive`) без кавычек рвётся argparse'ом.
    """
    if Path(path).is_dir():
        return True
    print(f"🔴 каталога архива нет: {path}")
    print("   Ожидается РАСПАКОВАННЫЙ d_us_txt — внутри лежат папки вида")
    print("   «nasdaq stocks», «nyse etfs». Путь задаётся --archive или")
    print("   переменной STOOQ_ROOT (OPERATOR_STOOQ.md §3.2).")
    print("   Windows: путь с пробелами обязателен в кавычках.")
    return False


def _print_result(title: str, result: si.IngestResult) -> None:
    print(f"{title} завершён")
    print(f"  файлов обработано ..... {result.files}")
    print(f"  строк прочитано ....... {result.rows_read}")
    print(f"  баров записано ........ {result.rows_written}")
    print(f"  инструментов заведено . {result.instruments_added}")
    if result.rejected:
        print("  отброшено:")
        for rule, count in sorted(result.rejected.items(),
                                  key=lambda kv: -kv[1]):
            print(f"    {rule:.<26} {count}")
    for name, reason in result.fatal_files:
        print(f"  🔴 ФАЙЛ ОТВЕРГНУТ ЦЕЛИКОМ: {name} — {reason}")


def _print_c1(conn) -> int:
    """Допуск оператора. Возвращает код выхода: 1, если факторов не 10.

    В профиле `STRICT` (ручной ввод) потеря ЛЮБОГО факторного ETF даёт `BLOCK`
    в `data_checks.check_portfolio_sufficiency` — то есть пользователь получит
    отказ вместо отчёта. Поэтому «почти все» здесь не ответ.
    """
    try:
        factor_etfs, benchmark_etfs = _engine_universe()
    except ImportError as exc:
        print(f"  ⚠️ C-1 НЕ ПРОВЕРЕН: {_missing_name(exc)} не установлен.")
        print("     База наполнена, но допуск не подтверждён. "
              "Поставьте pandas и повторите: pip install pandas")
        return 0
    coverage = si.coverage_report(conn, list(factor_etfs) + list(benchmark_etfs))
    factors = [t for t in factor_etfs if coverage.get(t)]
    benches = [t for t in benchmark_etfs if coverage.get(t)]
    mark = "✅" if len(factors) == len(factor_etfs) else "🔴"
    print(f"  C-1 факторы ........... {len(factors)}/{len(factor_etfs)} {mark}")
    print(f"  C-1 бенчмарки ......... {len(benches)}/{len(benchmark_etfs)}")
    for ticker in factor_etfs:
        if not coverage.get(ticker):
            print(f"    🔴 нет истории: {ticker}")
    return 0 if len(factors) == len(factor_etfs) else 1


def _core_symbols(conn) -> list[str] | None:
    """Факторы и бенчмарки в формах источника. `None` — pandas не установлен."""
    try:
        return _working_set()
    except ImportError as exc:
        print("🔴 не могу определить обязательный набор: "
              f"{_missing_name(exc)} не установлен.")
        print("   Варианты: `pip install pandas` — или задать бумаги явно, "
              "например --symbols SPY.US,QQQ.US")
        print("   Флаг --all НЕ подходит как замена: полная выгрузка это "
              "≈900 МБ, а у Cloud Run /tmp — это RAM при лимите 2 GiB.")
        conn.close()
        return None


def _print_universe(report, args) -> None:
    """Почему бумаг отобрано именно столько — по каждому критерию.

    🔴 Без этой сводки команда однажды напечатала «рабочий набор: 13 бумаг», не
    соврав ни словом и не дав ни одной зацепки: порог истории оказался выше
    того, что окно физически выдаёт, и молча отсеял ВСЁ (`AUDIT §−87`).
    Правило «статус обязан называть то, чем его можно опровергнуть» относится и
    к выводу команды.
    """
    print(f"  померено бумаг ........ {report.measured}")
    print(f"  порог истории ......... {report.min_bars_used} баров "
          f"(у самой полной бумаги архива — {report.busiest_bars})")
    print(f"    короткая история .... {report.too_short}")
    print(f"    давно не торгуются .. {report.too_stale}")
    print(f"    мало оборота ........ {report.too_thin}")
    print(f"    не влезли в потолок . {report.over_limit}")
    print(f"  ОТОБРАНО .............. {len(report.symbols)}")
    if report.symbols:
        return
    if not report.measured:
        print("  🔴 архив не дал НИ ОДНОЙ бумаги — проверьте путь --archive: "
              "внутри должны лежать папки вида «nasdaq stocks».")
        return
    print("  🔴 фильтры отсеяли ВСЁ. Самый частый виновник назван выше по "
          "счётчику; ослабьте соответствующий порог:")
    print(f"     --min-bars (сейчас {args.min_bars or 'авто'}) · "
          f"--stale-days {args.stale_days} · --min-turnover {args.min_turnover:g}")


def cmd_bootstrap(args) -> int:
    if not _require_archive(args.archive):
        return 1
    conn = si.connect(args.db)
    si.ensure_schema(conn)
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif not args.all:
        core = _core_symbols(conn)
        if core is None:
            return 1
        symbols = list(core)
        if args.core_only:
            # 🔴 Осознанный выбор оператора, а не дефолт: такой базой можно
            # проверить допуск C-1, но НЕЛЬЗЯ построить ни одного отчёта —
            # бумаг пользователя в ней нет.
            print(f"ТОЛЬКО ядро: {len(symbols)} бумаг (факторы и бенчмарки). "
                  "Отчёт по портфелю пользователя такой базой не построится.")
        else:
            print(f"ядро: {len(symbols)} бумаг · сканирую архив, чтобы добрать "
                  f"ликвидные (потолок {args.universe_limit})…")

            def progress(files: int, measured: int) -> None:
                print(f"   … прочитано файлов {files}, померено бумаг {measured}",
                      flush=True)

            report = si.explain_universe(
                si.scan_archive(args.archive, window_days=args.window,
                                on_progress=progress),
                min_bars=args.min_bars, stale_after_days=args.stale_days,
                min_turnover=args.min_turnover, limit=args.universe_limit)
            _print_universe(report, args)
            for symbol in report.symbols:
                if symbol not in symbols:
                    symbols.append(symbol)
            print(f"рабочий набор: {len(symbols)} бумаг "
                  f"(≈{len(symbols) * 75 / 1024:.0f} МБ базы)")
    result = si.bootstrap(conn, args.archive, symbols=symbols,
                          window_days=args.window)
    _print_result("bootstrap", result)
    code = _print_c1(conn)
    conn.close()
    return code


def cmd_apply(args) -> int:
    conn = si.connect(args.db)
    si.ensure_schema(conn)
    result = si.apply_inbox(conn, args.inbox, args.applied, args.rejected)
    if result.files == 0:
        print(f"в {args.inbox} нет файлов вида YYYYMMDD_d.txt — нечего применять")
        conn.close()
        return 0
    _print_result("apply", result)
    code = _print_c1(conn)
    conn.close()
    return 1 if result.fatal_files else code


def cmd_backfill(args) -> int:
    if not _require_archive(args.archive):
        return 1
    conn = si.connect(args.db)
    si.ensure_schema(conn)
    symbols = [s.strip().upper() for s in args.ticker.split(",") if s.strip()]
    result = si.bootstrap(conn, args.archive, symbols=symbols,
                          window_days=args.window)
    _print_result("backfill", result)
    conn.close()
    return 0 if result.rows_written else 1


def cmd_verify_seam(args) -> int:
    if not Path(args.db).exists():
        print(f"🔴 базы котировок нет по пути {args.db} — сначала bootstrap "
              "(OPERATOR_STOOQ.md §5)")
        return 1
    if not _require_archive(args.archive):
        return 1
    conn = si.connect(args.db, read_only=True)
    day = int(str(args.date).replace("-", ""))
    # `symbols=None` в библиотеке означает «всё, что есть в базе» — сверять
    # надо загруженное, а не весь архив: бумага вне рабочего набора это
    # осознанный пропуск, а не расхождение (`AUDIT §−80`).
    compared = len(si.stored_symbols(conn))
    if not compared:
        # 🔴 Пустая база даёт «расхождений нет» — гейт проходит ВХОЛОСТУЮ.
        # Проверка, которая не может провалиться, ничего не подтверждает.
        conn.close()
        print("🔴 в базе нет ни одного инструмента — сверять нечего. "
              "Сначала bootstrap (OPERATOR_STOOQ.md §5)")
        return 1
    mismatches = si.verify_seam(conn, args.archive, day)
    conn.close()
    if not mismatches:
        print(f"шов на {day}: сверено бумаг {compared}, расхождений нет ✅")
        return 0
    print(f"🔴 шов на {day}: расхождений {len(mismatches)}")
    for item in mismatches[:20]:
        print(f"    {item.symbol:12s} архив={item.archive_close} "
              f"база={item.stored_close}")
    print("  источник пересчитал историю → нужен ре-бутстрап, а не патч")
    return 1


def cmd_verify_universe(args) -> int:
    try:
        from finance.stooq_store import StooqStore, StoreUnavailable
        universe = _engine_tickers()
    except ImportError as exc:
        print(f"🔴 нужен {_missing_name(exc)}: pip install pandas")
        return 1
    try:
        store = StooqStore.open_readonly(args.db)
    except StoreUnavailable as exc:
        print(f"🔴 {exc}")
        return 1
    missing = store.missing(universe)
    latest = store.latest_date("US")
    print(f"покрытие универсума: {len(universe) - len(missing)}/{len(universe)}")
    print(f"последний торговый день US в базе: {latest}")
    for ticker in missing:
        print(f"    🔴 нет: {ticker}")
    substitutions = store.venue_substitutions(universe)
    for engine_ticker, source_symbol in substitutions:
        print(f"    ⚠️ подмена площадки: {engine_ticker} → {source_symbol}")
    store.close()
    return 1 if missing else 0


def cmd_measure_convention(args) -> int:
    """Замер конвенции корректировок — вход для `STOOQ_CONVENTION §4`."""
    if not _require_archive(args.archive):
        return 1
    try:
        probes = si.measure_convention(args.archive)
    except ImportError as exc:
        print(f"🔴 нужен {_missing_name(exc)}: pip install pydantic")
        return 1
    if not probes:
        print("в KNOWN_SPLITS нет событий — мерить нечего")
        return 1

    def cell(value, fmt: str) -> str:
        return format(value, fmt) if value else "—"

    print(f"{'бумага':10s} {'событие':12s} {'ожид.':>6s} "
          f"{'бар до':>9s} {'цена до':>10s} {'бар после':>10s} {'цена после':>11s} "
          f"{'факт':>7s}  вердикт")
    counts: dict[str, int] = {}
    for probe in probes:
        counts[probe.verdict] = counts.get(probe.verdict, 0) + 1
        print(f"{probe.ticker:10s} {probe.event_date:12s} "
              f"{probe.expected_ratio:>6g} "
              f"{cell(probe.date_before, 'd'):>9s} "
              f"{cell(probe.close_before, '.4f'):>10s} "
              f"{cell(probe.date_after, 'd'):>10s} "
              f"{cell(probe.close_on_after, '.4f'):>11s} "
              f"{cell(probe.observed_ratio, '.3f'):>7s}  {probe.verdict}")

    raw = counts.get(si.VERDICT_RAW, 0)
    adjusted = counts.get(si.VERDICT_ADJUSTED, 0)
    unclear = counts.get(si.VERDICT_UNCLEAR, 0)
    no_answer = sum(n for v, n in counts.items() if v in si.NO_ANSWER_VERDICTS)
    print()
    print(f"итог: сырых {raw} · скорректированных {adjusted} · "
          f"неясных {unclear} · не измерено {no_answer}")

    if raw and adjusted:
        print("🔴 ОТВЕТ ПРОТИВОРЕЧИВ — конвенция не едина по бумагам. "
              "Провайдер объявлять её НЕ вправе: покажите вывод разработке.")
        return 1
    if not raw and not adjusted:
        print("⚠️ ни одного события измерить не удалось — нужен ПОЛНЫЙ архив "
              "истории (бумаги NVDA, AMZN, GOOGL, GOOG, TSLA, SHOP, DXCM).")
        return 1
    if unclear:
        # Не отказ: ответ односторонний. Но «неясно» — это событие, где ряд не
        # похож НИ на сырой, НИ на скорректированный, и потерять его нельзя.
        print(f"🔴 не читаются ни как сырые, ни как скорректированные: "
              f"{unclear} шт. — впишите их в §4 отдельной строкой, "
              "молча отбрасывать нельзя.")
    print("✅ ответ односторонний. Скопируйте вывод целиком — он попадёт в "
          "docs/audit/STOOQ_CONVENTION.md §4 как ЗАМЕР.")
    return 0


def _engine_tickers() -> list[str]:
    """Универсум в терминах ДВИЖКА — ИЗ КОДА, а не списком в скрипте.

    Тот же приём, что в `scripts/stooq_coverage_probe.py`: markdown-список уже
    трижды расходился с реальностью (`AUDIT §−73`), поэтому универсум берётся
    у тех модулей, которые его и определяют.
    """
    factor_etfs, benchmark_etfs = _engine_universe()
    return list(factor_etfs) + list(benchmark_etfs)


def _working_set() -> list[str]:
    """Тот же универсум в терминах ИСТОЧНИКА — для загрузки из архива.

    🔴 Два разных списка, и путать их нельзя (`AUDIT §−80`). Загрузчик ищет
    файлы по именам Stooq (`SPY.US`), а `store.missing()` спрашивают тикером
    движка (`BTC-USD`). Прежняя редакция подавала в `verify-universe` символы
    источника: для US они случайно совпадают, но бумага БЕЗ кандидатов
    (`.AIX`, неизвестный суффикс) исчезала из списка молча — и команда
    рапортовала полное покрытие ровно для тех бумаг, которых нет.
    """
    from finance import stooq_symbols as sym

    out: list[str] = []
    for ticker in _engine_tickers():
        for candidate in sym.candidates_for(ticker):
            if candidate not in out:
                out.append(candidate)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Дефолт базы — ВНУТРИ рабочего корня оператора: один `STOOQ_ROOT`
    # управляет и папками, и базой, иначе быстрый старт собирает базу в одном
    # месте, а заливает в облако из другого (`AUDIT §−80`).
    parser.add_argument("--db",
                        default=os.getenv("STOOQ_DB_PATH",
                                          str(DEFAULT_ROOT / "prices.sqlite")),
                        help="путь к prices.sqlite")
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("bootstrap", help="первичное наполнение из архива")
    boot.add_argument("--archive", default=str(DEFAULT_ROOT / "archive"))
    boot.add_argument("--symbols", default="",
                      help="список символов источника через запятую")
    boot.add_argument("--all", action="store_true",
                      help="грузить весь архив (≈900 МБ — для прода не годится)")
    boot.add_argument("--core-only", action="store_true",
                      help="только факторы и бенчмарки: годится для проверки "
                           "C-1, но отчёт по портфелю НЕ построится")
    boot.add_argument("--universe-limit", type=int, default=800,
                      help="потолок числа бумаг сверх ядра (≈75 КБ на бумагу)")
    boot.add_argument("--min-bars", type=int, default=0,
                      help="минимум баров в окне; 0 — АВТО: 90%% от самой "
                           "полной бумаги архива. Абсолютное число уже один раз "
                           "обнулило универсум (порог 1260 против физических "
                           "1252), поэтому эталон берётся из данных")
    boot.add_argument("--stale-days", type=int, default=10,
                      help="сколько дней бумага вправе молчать, считаясь живой")
    boot.add_argument("--min-turnover", type=float, default=1_000_000.0,
                      help="медианный дневной оборот, close × volume")
    boot.add_argument("--window", type=int, default=1825)
    boot.set_defaults(func=cmd_bootstrap)

    app = sub.add_parser("apply", help="применить дневные срезы из inbox")
    app.add_argument("--inbox", default=str(DEFAULT_ROOT / "inbox"))
    app.add_argument("--applied", default=str(DEFAULT_ROOT / "applied"))
    app.add_argument("--rejected", default=str(DEFAULT_ROOT / "rejected"))
    app.set_defaults(func=cmd_apply)

    back = sub.add_parser("backfill", help="догрузить историю по бумаге")
    back.add_argument("--ticker", required=True)
    back.add_argument("--archive", default=str(DEFAULT_ROOT / "archive"))
    back.add_argument("--window", type=int, default=1825)
    back.set_defaults(func=cmd_backfill)

    seam = sub.add_parser("verify-seam", help="сверить архив с дневной дельтой")
    seam.add_argument("--date", required=True)
    seam.add_argument("--archive", default=str(DEFAULT_ROOT / "archive"))
    seam.set_defaults(func=cmd_verify_seam)

    uni = sub.add_parser("verify-universe", help="покрытие универсума движка")
    uni.set_defaults(func=cmd_verify_universe)

    conv = sub.add_parser("measure-convention",
                          help="сырой ряд или скорректированный (блокер MP-09)")
    conv.add_argument("--archive", default=str(DEFAULT_ROOT / "archive"))
    conv.set_defaults(func=cmd_measure_convention)
    return parser


def _make_console_utf8_safe() -> None:
    """Вывод не должен падать на консоли Windows.

    🔴 Замерено: в тексте команд 14 символов `🔴`, 3 `✅`, 3 `⚠️`, а также `→`,
    `≈`, `—`, `§`. Ни один из них не кодируется в `cp866` (кодировка `cmd.exe`
    в русской Windows) и почти ни один — в `cp1251`. Питон в этом случае не
    печатает «квадратики», а бросает `UnicodeEncodeError` — то есть оператор
    получил бы traceback вместо результата замера, и решил бы, что сломан
    загрузчик, а не консоль.

    `errors="replace"` важнее самой кодировки: даже там, где терминал не умеет
    рисовать эмодзи, команда обязана довести вывод до конца.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):   # pragma: no cover
            pass                                        # перенаправленный поток


def main(argv=None) -> int:
    _make_console_utf8_safe()
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
