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
from finance.data_checks import BENCHMARK_ETFS, FACTOR_ETFS  # noqa: E402
from finance.stooq_store import StooqStore, StoreUnavailable  # noqa: E402

#: Корень рабочих каталогов оператора.  Дефолт РЕПО-ЛОКАЛЬНЫЙ, а не
#: `/mnt/state/stooq`: этот путь существует только внутри контейнера Cloud Run,
#: где запускать загрузчик и незачем, и нечем.  Тот же приём, что у
#: `db_tokenomics.DB_PATH`: в проде путь задаёт env, локально — каталог рядом.
DEFAULT_ROOT = Path(os.getenv(
    "STOOQ_ROOT", str(Path(__file__).resolve().parents[1] / "data" / "stooq")))


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
    coverage = si.coverage_report(conn, list(FACTOR_ETFS) + list(BENCHMARK_ETFS))
    factors = [t for t in FACTOR_ETFS if coverage.get(t)]
    benches = [t for t in BENCHMARK_ETFS if coverage.get(t)]
    mark = "✅" if len(factors) == len(FACTOR_ETFS) else "🔴"
    print(f"  C-1 факторы ........... {len(factors)}/{len(FACTOR_ETFS)} {mark}")
    print(f"  C-1 бенчмарки ......... {len(benches)}/{len(BENCHMARK_ETFS)}")
    for ticker in FACTOR_ETFS:
        if not coverage.get(ticker):
            print(f"    🔴 нет истории: {ticker}")
    return 0 if len(factors) == len(FACTOR_ETFS) else 1


def cmd_bootstrap(args) -> int:
    conn = si.connect(args.db)
    si.ensure_schema(conn)
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif not args.all:
        symbols = _working_set()
        print(f"рабочий набор: {len(symbols)} бумаг "
              f"(полная выгрузка — флаг --all)")
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
    conn = si.connect(args.db)
    si.ensure_schema(conn)
    symbols = [s.strip().upper() for s in args.ticker.split(",") if s.strip()]
    result = si.bootstrap(conn, args.archive, symbols=symbols,
                          window_days=args.window)
    _print_result("backfill", result)
    conn.close()
    return 0 if result.rows_written else 1


def cmd_verify_seam(args) -> int:
    conn = si.connect(args.db, read_only=True)
    day = int(str(args.date).replace("-", ""))
    mismatches = si.verify_seam(conn, args.archive, day)
    conn.close()
    if not mismatches:
        print(f"шов на {day}: расхождений нет ✅")
        return 0
    print(f"🔴 шов на {day}: расхождений {len(mismatches)}")
    for item in mismatches[:20]:
        print(f"    {item.symbol:12s} архив={item.archive_close} "
              f"база={item.stored_close}")
    print("  источник пересчитал историю → нужен ре-бутстрап, а не патч")
    return 1


def cmd_verify_universe(args) -> int:
    try:
        store = StooqStore.open_readonly(args.db)
    except StoreUnavailable as exc:
        print(f"🔴 {exc}")
        return 1
    universe = _working_set()
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


def _working_set() -> list[str]:
    """Рабочий набор — ИЗ КОДА, а не списком в скрипте.

    Тот же приём, что в `scripts/stooq_coverage_probe.py`: markdown-список уже
    трижды расходился с реальностью (`AUDIT §−73`), поэтому универсум берётся
    у тех модулей, которые его и определяют.
    """
    from finance import stooq_symbols as sym

    engine_tickers = list(FACTOR_ETFS) + list(BENCHMARK_ETFS)
    out: list[str] = []
    for ticker in engine_tickers:
        for candidate in sym.candidates_for(ticker):
            if candidate not in out:
                out.append(candidate)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default=str(si.database_path()),
                        help="путь к prices.sqlite")
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("bootstrap", help="первичное наполнение из архива")
    boot.add_argument("--archive", default=str(DEFAULT_ROOT / "archive"))
    boot.add_argument("--symbols", default="",
                      help="список символов источника через запятую")
    boot.add_argument("--all", action="store_true",
                      help="грузить весь архив, а не рабочий набор")
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
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
