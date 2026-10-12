"""Порог чистки `/prune` — ПО РЫНКУ бумаги, а не по всей базе.

🔴 Порог «90% самой полной бумаги» считался по всей базе. Крипта торгуется 7/7 и за то
же окно набирает ≈ 1.45× баров акций США (≈ 1 825 против ≈ 1 255 за 5 лет), поэтому
одна монета в базе поднимала порог выше всей истории США, и чистка отправляла под нож
ВСЕ акции, кроме защищённого ядра факторов. Найдено тестом бота-загрузчика над Postgres
(OMBRI Ф5): на прежнем коде фикстура теряла AAPL, MSFT, VWO.
"""

from __future__ import annotations

import random
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from finance import stooq_ingest as si
from services import quote_ingest as qi

TODAY = date(2026, 10, 9)
HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"


def _days(n: int, *, weekdays_only: bool) -> list[int]:
    out, d = [], TODAY
    while len(out) < n:
        if not weekdays_only or d.weekday() < 5:
            out.append(int(d.strftime("%Y%m%d")))
        d -= timedelta(days=1)
    return sorted(out)


def _history(directory: Path, symbol: str, days: list[int], rng: random.Random) -> Path:
    price, rows = 100.0, [HEADER]
    for d in days:
        price *= 1 + rng.gauss(0, 0.01)
        rows.append(f"{symbol},D,{d},000000,{price},{price * 1.01},{price * 0.99},{price},1000,0")
    path = directory / f"{symbol.lower()}.txt"
    path.write_text("\r\n".join(rows) + "\r\n", encoding="utf-8")
    return path


class PruneThresholdPerMarketTest(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root, rng = Path(self._tmp.name), random.Random(76)
        archive = root / "archive"
        archive.mkdir()
        for symbol in ("AAPL.US", "MSFT.US", "VWO.US"):
            _history(archive, symbol, _days(600, weekdays_only=True), rng)
        _history(archive, "THIN.US", _days(30, weekdays_only=True), rng)
        _history(archive, "BTC.V", _days(800, weekdays_only=False), rng)
        self.conn = si.connect(root / "prices.sqlite")
        si.ensure_schema(self.conn)
        si.bootstrap(self.conn, archive, window_days=1825, today=TODAY)
        self.addCleanup(self.conn.close)

    def test_one_coin_does_not_doom_the_stocks(self) -> None:
        report, doomed = si.plan_prune(self.conn)
        self.assertEqual(report.removed, ["THIN.US"])
        self.assertEqual(len(doomed), 1)
        self.assertEqual(report.thresholds, {"US": (540, 600), "CRYPTO": (720, 800)})
        # Сводка — по крупнейшему рынку базы.
        self.assertEqual((report.min_bars_used, report.busiest_bars), (540, 600))

    def test_prune_deletes_only_the_thin_paper(self) -> None:
        report = si.prune_thin_instruments(self.conn)
        self.assertEqual(report.removed, ["THIN.US"])
        left = {r["source_symbol"] for r in self.conn.execute(
            "SELECT source_symbol FROM instruments")}
        self.assertEqual(left, {"AAPL.US", "MSFT.US", "VWO.US", "BTC.V"})

    def test_explicit_threshold_applies_to_every_market(self) -> None:
        report, _ = si.plan_prune(self.conn, min_bars=700)
        self.assertEqual(report.removed, ["AAPL.US", "MSFT.US", "THIN.US", "VWO.US"])

    def test_operator_sees_the_threshold_per_market(self) -> None:
        report, _ = si.plan_prune(self.conn)
        text = qi.format_prune(qi.PruneOutcome(
            ok=True, dry_run=True, removed=tuple(report.removed), kept=report.kept,
            bars_removed=report.bars_removed, min_bars_used=report.min_bars_used,
            busiest_bars=report.busiest_bars, thresholds=report.thresholds))
        self.assertIn("CRYPTO 720/800", text)
        self.assertIn("US 540/600", text)


if __name__ == "__main__":
    unittest.main()
