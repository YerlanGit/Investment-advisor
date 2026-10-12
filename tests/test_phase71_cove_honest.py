"""Q-8 — честный CoVe: три строки панели происхождения больше не зеленеют без данных.

1. **SEC (DQ-5).** При сбое SEC EDGAR все поля `SEC_*` пусты: «пропущенных»
   (сектор `default`/`EM_Proxy`) нет, «устаревших» 10-K нет — строка была `ok`
   с пустой заметкой, хотя фундамент не посчитан ни для одной бумаги.
   Знаменатель — бумаги, которые ЖДУТ 10-K (`sec_edgar.expects_annual_filing`,
   то же решение, по которому движок идёт в SEC): ни одной даты → `missing`,
   у части нет → `warn` с тикерами и долей книги. ETF, облигации, крипта, кэш и
   бумаги KASE 10-K не ждут — их «нет даты» сбоем не считается.
2. **ИИ (DQ-6).** Детерминированный фолбэк тоже отдаёт вердикт и тезисы, и
   строка «AI verdict» была `ok` с подписью «Anthropic». Фолбэк → `warn`.
3. **RAG (DQ-7, подпись).** Пустой реестр эмитентов подставлял литерал
   «GS / MS / JPM» — под строкой «база: 0 отчётов». Пустая база → «база пуста»,
   реестр без имён — без имён.
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance.data_lineage import _ai_status, _rag_status, _sec_status  # noqa: E402

TODAY = date(2026, 10, 12)


def _sec(rows: list[dict], total: float = 10_000.0) -> dict:
    perf = pd.DataFrame(rows)
    return _sec_status({"performance_table": perf, "total_value": total}, TODAY)[0]


class SecRowTest(unittest.TestCase):

    def test_sec_outage_is_missing_not_ok(self) -> None:
        """Бумаги ждут 10-K, а дат нет ни у одной — сбой SEC, не «ok»."""
        row = _sec([
            {"Ticker": "AAPL", "Fundamental_Sector": "Technology",
             "Current_Value": 6000.0, "SEC_Filing_Date": None},
            {"Ticker": "JPM", "Fundamental_Sector": "Finance",
             "Current_Value": 4000.0, "SEC_Filing_Date": None},
        ])
        self.assertEqual(row["status"], "missing")
        self.assertIn("ни по одной из 2", row["note"])
        self.assertIn("100% книги по весу", row["note"])

    def test_whole_batch_failure_without_sec_columns_is_missing(self) -> None:
        """`batch_fundamental_scan` упал целиком — колонок `SEC_*` нет вовсе."""
        row = _sec([{"Ticker": "AAPL", "Current_Value": 1000.0},
                    {"Ticker": "MSFT", "Current_Value": 1000.0}])
        self.assertEqual(row["status"], "missing")

    def test_partial_outage_is_warn_and_names_tickers(self) -> None:
        row = _sec([
            {"Ticker": "AAPL", "Fundamental_Sector": "Technology",
             "Current_Value": 7000.0, "SEC_Filing_Date": "2026-02-01"},
            {"Ticker": "MSFT", "Fundamental_Sector": "Technology",
             "Current_Value": 3000.0, "SEC_Filing_Date": None},
        ])
        self.assertEqual(row["status"], "warn")
        self.assertIn("1 из 2 бумаг без отчётности SEC: MSFT", row["note"])
        self.assertIn("30% книги по весу", row["note"])

    def test_book_without_10k_issuers_is_not_an_outage(self) -> None:
        """ETF, облигационный ETF, крипта, кэш и бумага KASE 10-K не ждут."""
        row = _sec([
            {"Ticker": t, "Fundamental_Sector": "Broad_Market", "Current_Value": 100.0,
             "SEC_Filing_Date": None}
            for t in ("SPY", "TLT", "BTC-USD", "USD", "KSPI.KZ")
        ])
        self.assertEqual(row["status"], "ok")
        self.assertNotIn("SEC не вернул", row["note"])

    def test_fresh_filings_stay_ok(self) -> None:
        row = _sec([
            {"Ticker": "AAPL", "Fundamental_Sector": "Technology",
             "Current_Value": 1.0, "SEC_Filing_Date": "2026-02-01"},
            {"Ticker": "SPY", "Fundamental_Sector": "Broad_Market",
             "Current_Value": 1.0, "SEC_Filing_Date": None},
        ])
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["note"], "")

    def test_default_sector_is_not_counted_twice(self) -> None:
        """Бумага с сектором `default` уже названа «без покрытия»."""
        row = _sec([
            {"Ticker": "AAPL", "Fundamental_Sector": "Technology",
             "Current_Value": 1.0, "SEC_Filing_Date": "2026-02-01"},
            {"Ticker": "FXKZ", "Fundamental_Sector": "default",
             "Current_Value": 1.0, "SEC_Filing_Date": None},
        ])
        self.assertEqual(row["status"], "warn")
        self.assertIn("1 тикеров без SEC покрытия", row["note"])
        self.assertNotIn("без отчётности SEC", row["note"])


class AiRowTest(unittest.TestCase):

    def test_deterministic_fallback_is_warn(self) -> None:
        from ai_narrative import _fallback_narrative
        summary = _fallback_narrative({
            "portfolio_metrics": {"Composite_Risk_Score": 55, "Sharpe_Ratio": 0.66},
            "regime": {"regime": "Expansion"},
            "performance_table": pd.DataFrame([{"Ticker": "AAPL"}]),
            "total_value": 10_000.0,
            "leverage_metrics": {"is_leveraged": False}}, "base")
        self.assertTrue(summary["verdict"])            # вердикт есть — и всё же не ИИ
        row = _ai_status(summary)
        self.assertEqual(row["status"], "warn")
        self.assertNotIn("Anthropic", row["source"])
        self.assertIn("без ИИ", row["note"])

    def test_model_answer_stays_ok(self) -> None:
        row = _ai_status({"verdict": "v", "bullets": ["b"],
                          "model_used": "anthropic/claude-opus-4.8"})
        self.assertEqual(row["status"], "ok")

    def test_no_text_stays_missing(self) -> None:
        self.assertEqual(_ai_status({"model_used": "fallback"})["status"], "missing")


class RagRowTest(unittest.TestCase):

    def test_empty_kb_is_labelled_empty(self) -> None:
        row = _rag_status({"rag_status": "unavailable", "rag_kb_docs": 0,
                           "rag_kb_chunks": 0, "rag_kb_banks": []})
        self.assertEqual(row["status"], "missing")
        self.assertEqual(row["source"], "ChromaDB · база пуста")
        for bank in ("GS", "MS", "JPM"):
            self.assertNotIn(bank, row["source"])

    def test_populated_kb_without_named_issuers_names_nobody(self) -> None:
        row = _rag_status({"rag_status": "no_match", "rag_kb_docs": 2,
                           "rag_kb_chunks": 40, "rag_kb_banks": []})
        self.assertEqual(row["source"], "ChromaDB · PDF reports")


if __name__ == "__main__":
    unittest.main()
