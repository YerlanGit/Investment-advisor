#!/usr/bin/env python3
"""Долить бумаги, которых нет в дневном срезе, потикерно со stooq.

Запасной путь на случай, когда дневного пакета ETF под рукой нет. Читает
опубликованную базу, находит бумаги БЕЗ бара за нужный день, тянет по каждой
дневной ряд с stooq и дописывает строки в тот же файл среза — в том же
формате, поэтому дальше это обычный дневной файл для бота.

    python3 topup.py prices.sqlite 20260828 20260828_d.txt

🔴 Это ЗАТЫЧКА, а не процесс. Правильный ежедневный цикл — дневной пакет ETF
рядом с пакетом акций (OPERATOR_STOOQ §8.0).
"""
from __future__ import annotations

import sqlite3
import sys
import time
import urllib.request

URL = "https://stooq.com/q/d/l/?s={sym}&i=d&d1={d}&d2={d}"
PAUSE_SEC = 0.3          # вежливость к источнику; 172 бумаги ≈ минута


def missing(db: str, trade_date: int) -> list[str]:
    """Бумаги базы БЕЗ бара за эту дату — только по рынкам, торговавшим в этот день."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT i.source_symbol FROM instruments i "
            "WHERE i.market IN (SELECT i2.market FROM daily_bars b2 "
            "                   JOIN instruments i2 ON i2.id = b2.instrument_id "
            "                   WHERE b2.trade_date = ?) "
            "  AND NOT EXISTS (SELECT 1 FROM daily_bars b "
            "                  WHERE b.instrument_id = i.id AND b.trade_date = ?) "
            "ORDER BY i.source_symbol", (trade_date, trade_date)).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def to_row(symbol: str, trade_date: int, csv_text: str) -> str | None:
    """CSV одной бумаги → строка формата дневного среза, либо None.

    🔴 Отказ источника приходит ТЕКСТОМ с кодом 200 («Exceeded the daily hits
    limit»), поэтому шапка проверяется, а не предполагается: без этого лимит
    записался бы в файл как данные.
    """
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not lines or not lines[0].lower().startswith("date,"):
        return None
    head = [h.strip().lower() for h in lines[0].split(",")]
    want = ("date", "open", "high", "low", "close")
    if any(c not in head for c in want):
        return None
    idx = {c: head.index(c) for c in head}
    for line in lines[1:]:
        cell = line.split(",")
        if len(cell) < len(head):
            continue
        day = cell[idx["date"]].replace("-", "")
        if day != str(trade_date):
            continue
        vol = cell[idx["volume"]] if "volume" in idx and len(cell) > idx["volume"] else "0"
        return (f"{symbol},D,{day},000000,{cell[idx['open']]},{cell[idx['high']]},"
                f"{cell[idx['low']]},{cell[idx['close']]},{vol or 0},0")
    return None


def fetch(symbol: str, trade_date: int) -> str:
    url = URL.format(sym=symbol.lower(), d=trade_date)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", "replace")


def main() -> int:
    db, day, target = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    names = missing(db, day)
    print(f"без бара за {day}: {len(names)}")
    if not names:
        return 0
    got, lost = [], []
    for n, symbol in enumerate(names, 1):
        try:
            row = to_row(symbol, day, fetch(symbol, day))
        except Exception as exc:                        # noqa: BLE001
            row, exc_text = None, str(exc)
            print(f"  {symbol}: сеть — {exc_text}")
        (got.append(row) if row else lost.append(symbol))
        if n % 25 == 0:
            print(f"  … {n}/{len(names)}")
        time.sleep(PAUSE_SEC)
    if got:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("\n".join(got) + "\n")
    print(f"дописано строк: {len(got)}; не отдал источник: {len(lost)}")
    if lost:
        print("  " + " ".join(lost[:20]) + (" …" if len(lost) > 20 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
