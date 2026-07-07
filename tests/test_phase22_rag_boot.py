"""
Phase 22 — RAG boot-ingest fallback (entrypoint).

When the Cloud-Function → Eventarc → ChromaDB path is misconfigured (wrong
bucket region, missing trigger, embedding-model download failure) the STORE
bucket stays empty and RAG never lights up even though PDFs sit in the INBOX.
`entrypoint._boot_ingest_from_inbox` makes the bot self-sufficient: on boot,
if the synced store is empty but the INBOX holds PDFs, it ingests them
in-container (deps + pre-baked model + pinned chromadb) and publishes the store
back to STORE.  These tests are hermetic — GCS + chromadb are mocked.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import entrypoint  # noqa: E402

_HAS_STORAGE = True
try:
    import google.cloud.storage  # noqa: F401
except Exception:  # pragma: no cover - env-dependent
    _HAS_STORAGE = False


class RagBootIngestGateTest(unittest.TestCase):
    def test_env_gate(self) -> None:
        cases = {"0": False, "off": False, "no": False, "false": False, "": False,
                 "on": True, "1": True, "yes": True}
        for val, want in cases.items():
            with mock.patch.dict(os.environ, {"RAG_BOOT_INGEST": val}):
                self.assertEqual(entrypoint._rag_boot_ingest_enabled(), want, val)

    def test_default_on_when_unset(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "RAG_BOOT_INGEST"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(entrypoint._rag_boot_ingest_enabled())

    def test_disabled_is_noop(self) -> None:
        with mock.patch.dict(os.environ, {"RAG_BOOT_INGEST": "0"}), \
             mock.patch("agent.rag_engine.FinancialRAG") as frag:
            entrypoint._boot_ingest_from_inbox()
            frag.assert_not_called()   # never even opens the store


class RagBootIngestBehaviourTest(unittest.TestCase):
    def test_skip_when_store_already_populated(self) -> None:
        fake_rag = mock.MagicMock()
        fake_rag.collection.count.return_value = 42
        with mock.patch.dict(os.environ, {"RAG_BOOT_INGEST": "on"}), \
             mock.patch("agent.rag_engine.FinancialRAG", return_value=fake_rag), \
             mock.patch.object(entrypoint, "_upload_chroma_db_to_store") as up:
            entrypoint._boot_ingest_from_inbox()
            fake_rag.ingest_pdf.assert_not_called()   # nothing to do
            up.assert_not_called()

    @unittest.skipUnless(_HAS_STORAGE, "google-cloud-storage not installed")
    def test_ingests_inbox_pdfs_when_store_empty(self) -> None:
        fake_rag = mock.MagicMock()
        fake_rag.collection.count.return_value = 0        # STORE empty
        fake_rag.ingest_pdf.return_value = 7              # 7 chunks/PDF

        blob = mock.MagicMock()
        blob.name = "goldman_sachs_outlook_Q3_2026.pdf"
        bucket = mock.MagicMock()
        bucket.list_blobs.return_value = [blob]
        client = mock.MagicMock()
        client.bucket.return_value = bucket

        with mock.patch.dict(os.environ, {"RAG_BOOT_INGEST": "on"}), \
             mock.patch("agent.rag_engine.FinancialRAG", return_value=fake_rag), \
             mock.patch("google.cloud.storage.Client", return_value=client), \
             mock.patch.object(entrypoint, "_upload_chroma_db_to_store",
                               return_value=5) as up:
            entrypoint._boot_ingest_from_inbox()

        fake_rag.ingest_pdf.assert_called_once()
        # published the freshly-built store back to STORE
        up.assert_called_once()

    @unittest.skipUnless(_HAS_STORAGE, "google-cloud-storage not installed")
    def test_empty_inbox_does_not_publish(self) -> None:
        fake_rag = mock.MagicMock()
        fake_rag.collection.count.return_value = 0
        bucket = mock.MagicMock()
        bucket.list_blobs.return_value = []               # no PDFs in INBOX
        client = mock.MagicMock()
        client.bucket.return_value = bucket
        with mock.patch.dict(os.environ, {"RAG_BOOT_INGEST": "on"}), \
             mock.patch("agent.rag_engine.FinancialRAG", return_value=fake_rag), \
             mock.patch("google.cloud.storage.Client", return_value=client), \
             mock.patch.object(entrypoint, "_upload_chroma_db_to_store") as up:
            entrypoint._boot_ingest_from_inbox()
            fake_rag.ingest_pdf.assert_not_called()
            up.assert_not_called()

    def test_never_raises_on_failure(self) -> None:
        # A broken rag engine must be swallowed — RAG stays empty, bot boots.
        with mock.patch.dict(os.environ, {"RAG_BOOT_INGEST": "on"}), \
             mock.patch("agent.rag_engine.FinancialRAG",
                        side_effect=RuntimeError("chromadb boom")):
            entrypoint._boot_ingest_from_inbox()   # must not raise


class IngestLogicalFilenameTest(unittest.TestCase):
    """2026-07-05: PDFs are downloaded to NamedTemporaryFile paths, so the tmp
    basename («tmpl3mmhrmf.pdf») used to become the source metadata (leaked
    into the report's RAG chips) and defeated filename-based DATE detection —
    recency then fell back to tmp-file mtime (≈today for ANY old report)."""

    def test_doc_date_prefers_logical_filename(self) -> None:
        from agent.rag_engine import FinancialRAG
        rag = FinancialRAG.__new__(FinancialRAG)   # no chromadb needed
        dt, method = rag._get_doc_date(
            "/tmp/tmpl3mmhrmf.pdf", "no dates in text",
            filename="goldman_sachs_outlook_Q3_2026.pdf")
        self.assertEqual(method, "filename")
        self.assertEqual((dt.year, dt.month), (2026, 7))   # Q3 → июль

    def test_lookback_env_bridge_defaults_to_1825(self) -> None:
        """Фаза 4 §4-б: HISTORY_LOOKBACK_DAYS параметризует окно истории;
        с 2026-07-07 default = 1825 кал.дн ≈ 1260 торговых ≈ 5 лет (целевое
        окно §4-б); env-override сохранён; clamp 90–3650."""
        import inspect
        from finance.investment_logic import MAC3RiskEngine
        src = inspect.getsource(MAC3RiskEngine.get_market_data)
        self.assertIn("HISTORY_LOOKBACK_DAYS", src)
        self.assertIn('"1825"', src)                   # 5-year default
        self.assertIn("period_days: int | None = None", src)

    def test_regime_rag_chips_are_demarkdowned(self) -> None:
        """The regime chips must carry CONTENT, not retrieval headers/markdown."""
        import re as _re
        raw_lines = [
            "--- [2026-06-01] tmpl3mmhrmf.pdf — **Global Direction** (актуальность: 91%) ---",
            "**against spread duration risk** and further tightening ahead",
            "Despite tight spreads, the recent move higher in yields is supportive",
        ]
        # Mirror the tg_bot._fetch_rag_context filter (kept in sync by hand).
        out = []
        for line in raw_lines:
            line = line.strip()
            if not line or line.startswith("---"):
                continue
            line = _re.sub(r"\*\*(.*?)\*\*", r"\1", line)
            line = line.lstrip("#*•- ").strip()
            if len(line) > 30:
                out.append(line[:200])
        self.assertEqual(len(out), 2)                      # header dropped
        self.assertNotIn("**", " ".join(out))              # markdown stripped
        self.assertNotIn("tmpl3mmhrmf", " ".join(out))


if __name__ == "__main__":
    unittest.main()
