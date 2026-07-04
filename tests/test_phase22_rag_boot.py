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


if __name__ == "__main__":
    unittest.main()
