"""`§−129` — C-1: рыночный индекс HY больше не оценивает эмитента.

Решение владельца 05.10 (`REVIEW_2026-10-05` §2.5). Бесплатного CDS эмитента в
подключённых источниках нет; «CDS» каждой бумаги США был индексом ICE BofA US
HY OAS (живой отчёт 02.10: 312 б.п.). Он выше порога 150 б.п. всегда, поэтому
КАЖДАЯ бумага США получала C = −2 — постоянный сдвиг к «Sell», а не оценку.

Сверка — с ИСТИНОЙ, а не с формулой: рыночное число одинаково для всех бумаг
и не вправе изменить НИ ОДНУ оценку; чтение эмитента и страны эмитента —
вправе. Фонд акций эмитента не имеет — его C и F «н/д».
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

_TESTS = Path(__file__).resolve().parent
SRC = _TESTS.parent / "src"
ROOT = _TESTS.parent
for p in (str(SRC), str(_TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

#: Живое значение 02.10 — рыночный индекс, одно число на ВСЕ тикеры.
_HY = {"bps": 312.0, "change_7d": 0.01, "source": "FRED:BAMLH0A0HYM2",
       "quality": "C"}


def _score(rows, lookup, regime=None):
    from finance.scoring_orchestrator import score_portfolio
    perf = pd.DataFrame([{"Euler_Risk_Contribution_Pct": 5.0, **r} for r in rows])
    return score_portfolio(perf, {}, regime=regime, cds_lookup=lookup)


class ScopeOfAReadingTest(unittest.TestCase):

    def test_scope_from_source(self) -> None:
        from finance.cds_feed import cds_scope
        self.assertEqual(cds_scope("FRED:BAMLH0A0HYM2"), "market")
        self.assertEqual(cds_scope("fred:anything"), "market")
        self.assertEqual(cds_scope("WGB:KZ_5Y"), "sovereign")
        self.assertEqual(cds_scope("sp_global"), "issuer")
        self.assertEqual(cds_scope(None), "issuer")

    def test_who_enters_the_issuer_score(self) -> None:
        from finance.cds_feed import enters_issuer_score
        self.assertFalse(enters_issuer_score({}))
        self.assertFalse(enters_issuer_score(None))
        self.assertFalse(enters_issuer_score({"bps": None, "source": "sp"}))
        self.assertFalse(enters_issuer_score(_HY))
        self.assertTrue(enters_issuer_score({"bps": 120.0, "source": "WGB:KZ_5Y"}))
        self.assertTrue(enters_issuer_score({"bps": 80.0, "source": "sp_global"}))
        # Явный охват сильнее источника.
        self.assertFalse(enters_issuer_score({"bps": 80.0, "scope": "market"}))

    def test_lookup_and_point_carry_the_scope(self) -> None:
        """Охват доезжает до скоринга и из провайдера, и из кэша."""
        from finance.cds_feed import CDSFeed, CDSPoint, make_lookup
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            feed = CDSFeed(cache_path=os.path.join(tmp, "c.sqlite"))
            feed._providers = [("fred", lambda t: CDSPoint(
                ticker=t, bps=312.0, source="FRED:BAMLH0A0HYM2",
                timestamp=now, quality="C", change_7d=None))]
            lookup = make_lookup(feed)
            first = lookup("AAPL")
            self.assertEqual(first["scope"], "market")
            feed._providers = []                 # теперь — только кэш
            self.assertEqual(lookup("AAPL")["scope"], "market")
            self.assertEqual(feed.get_spread("AAPL").as_dict()["scope"], "market")


class MarketIndexNeverScoresAnIssuerTest(unittest.TestCase):

    def test_us_stock_no_longer_minus_two(self) -> None:
        sc = _score([{"Ticker": "PEP"}], lambda t: dict(_HY))["PEP"]
        self.assertTrue(sc.credit_applicable)
        self.assertEqual(sc.credit, 0.0)
        # Было: C = −2 → итог −2 → «Sell» у бумаги без единого сигнала.
        self.assertEqual(sc.action, "Hold")

    def test_hy_index_changes_no_score_at_all(self) -> None:
        """Одно рыночное число на всех не вправе изменить НИ ОДНУ оценку."""
        rows = [
            {"Ticker": "PEP", "SEC_Altman_Zone": "Safe", "SEC_Piotroski_F": 8},
            {"Ticker": "FTNT", "SEC_Altman_Zone": "Distress",
             "SEC_Interest_Coverage": 1.2},
            {"Ticker": "JNK"},                    # HY-фонд вне словаря секторов
            {"Ticker": "KSPI"},                   # WGB не ответил → был HY
        ]
        with_hy = _score(rows, lambda t: dict(_HY))
        without = _score(rows, lambda t: {})
        for t in with_hy:
            with self.subTest(ticker=t):
                self.assertEqual(with_hy[t], without[t])

    def test_wide_hy_week_is_not_an_issuer_event(self) -> None:
        """Δ7d > 20% у индекса — рыночное событие, не −2 каждой бумаге."""
        wide = dict(_HY, change_7d=0.35)
        self.assertEqual(_score([{"Ticker": "PEP"}], lambda t: wide)["PEP"].credit, 0.0)

    def test_reporting_signals_still_score(self) -> None:
        """C = только отчётность: Altman/Piotroski/покрытие работают как прежде."""
        from finance.scoring import credit_score
        sc = _score([{"Ticker": "PEP", "SEC_Altman_Zone": "Safe",
                      "SEC_Piotroski_F": 8}], lambda t: dict(_HY))["PEP"]
        self.assertEqual(sc.credit, credit_score(altman_zone="Safe", piotroski_f=8))
        self.assertEqual(sc.credit, 1.0)


class IssuerAndCountryReadingsStillScoreTest(unittest.TestCase):

    def test_issuer_cds_is_scored(self) -> None:
        r = {"bps": 200.0, "change_7d": None, "source": "sp_global", "quality": "A"}
        self.assertEqual(_score([{"Ticker": "PEP"}], lambda t: r)["PEP"].credit, -2.0)

    def test_kz_sovereign_is_scored(self) -> None:
        """Суверенный CDS KZ оставлен решением C-1: потолок страны эмитента."""
        r = {"bps": 120.0, "change_7d": None, "source": "WGB:KZ_5Y", "quality": "C"}
        self.assertEqual(_score([{"Ticker": "KSPI"}], lambda t: r)["KSPI"].credit, -1.0)


class EquityFundHasNoIssuerTest(unittest.TestCase):

    def test_equity_etfs_are_not_applicable(self) -> None:
        from finance.regime import RegimeReading
        reg = RegimeReading(regime="Expansion", confidence=0.9,
                            growth_score=0.1, cycle_score=0.05, signals={})
        rows = [{"Ticker": "SPY", "Fundamental_Sector": "Other"},
                {"Ticker": "XLU.US", "Fundamental_Sector": "Other"},
                # SOXX стоит в словаре как «Semiconductors» — режим Expansion
                # его поощряет, и до C-1 этот наклон печатался как F = +0.45.
                {"Ticker": "SOXX", "Fundamental_Sector": "Semiconductors"},
                {"Ticker": "MTUM", "Fundamental_Sector": "Other"}]
        out = _score(rows, lambda t: dict(_HY), regime=reg)
        for t, sc in out.items():
            with self.subTest(ticker=t):
                self.assertFalse(sc.credit_applicable)
                self.assertFalse(sc.fundamentals_applicable)
                self.assertEqual(sc.credit, 0.0)
                self.assertEqual(sc.fundamentals, 0.0)

    def test_single_stock_keeps_both_pillars(self) -> None:
        sc = _score([{"Ticker": "AAPL", "Fundamental_Sector": "Technology"}],
                    lambda t: {})["AAPL"]
        self.assertTrue(sc.credit_applicable)
        self.assertTrue(sc.fundamentals_applicable)

    def test_predicate_lives_in_the_ssot(self) -> None:
        from finance import asset_taxonomy as tx
        self.assertIn("is_equity_etf", tx.__all__)
        for t in ("SPY", "QQQ.US", "XLU", "SOXX.KZ", "SPLV"):
            self.assertTrue(tx.is_equity_etf(t), t)
        for t in ("AAPL", "TLT", "GLD", "EMB", "KSPI"):
            self.assertFalse(tx.is_equity_etf(t), t)


class ProvenanceCountsOnlyWhatIsScoredTest(unittest.TestCase):
    """V-2: строка CoVe писала «ok N/N», когда у всех было одно рыночное число."""

    def _row(self, summary):
        from finance.data_lineage import _cds_status
        return _cds_status({"cds_summary": summary})

    def test_market_only_is_not_coverage(self) -> None:
        row = self._row({"enabled": True, "checked": 5, "loaded": 0,
                         "market_only": 5, "gated_out": 0})
        self.assertEqual(row["status"], "missing")
        self.assertIn("0/5", row["note"])
        self.assertIn("regime context", row["note"])

    def test_mixed_book_is_partial(self) -> None:
        row = self._row({"enabled": True, "checked": 5, "loaded": 2,
                         "market_only": 3, "gated_out": 0})
        self.assertEqual(row["status"], "warn")
        self.assertIn("2/5", row["note"])

    def test_freshness_text_matches_the_gate(self) -> None:
        from finance.cds_feed import CDSQualityGate
        row = self._row({"enabled": True, "checked": 1, "loaded": 1,
                         "market_only": 0, "gated_out": 0})
        self.assertIn(f"≤ {CDSQualityGate.MAX_STALE_DAYS} calendar days",
                      row["method"])

    def test_engine_tally_on_a_golden_book(self) -> None:
        """Конец в конец: HY на всех бумагах → в оценке 0, ни одна оценка не
        сдвинулась относительно прогона без CDS вовсе."""
        import golden_support as gs
        base = gs.run_analyze_all("base")
        with mock.patch.object(gs, "_fake_cds_lookup",
                               lambda *a, **k: (lambda t: dict(_HY))):
            hy = gs.run_analyze_all("base")
        cs = hy["cds_summary"]
        self.assertEqual(cs["loaded"], 0)
        self.assertEqual(cs["market_only"], cs["checked"])
        self.assertGreater(cs["checked"], 0)
        self.assertEqual(hy["asset_scores"], base["asset_scores"])


class PromptAgreesWithTheEngineTest(unittest.TestCase):

    def test_prompt_says_market_index_is_not_scored(self) -> None:
        txt = (ROOT / "SYSTEM_PROMPT.md").read_text(encoding="utf-8")
        self.assertIn("Рыночный индекс HY в C не входит", txt)
        self.assertNotIn("≤ 3 trading days", txt)


if __name__ == "__main__":
    unittest.main()
