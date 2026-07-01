#!/usr/bin/env python3
"""
Admin CLI — ingest bank analytical PDFs into the RAG store (ChromaDB) and,
optionally, publish the store to GCS so the running bot picks it up.

This is the FAST/manual path that complements the GCS-triggered Cloud Function
(cloud_function/main.py).  Use it to load reports without waiting for the
trigger, or from a laptop / CI job.

Flow
────
  1. Parse each PDF (pymupdf4llm → markdown), section-aware size-bounded chunks,
     auto-detect date + bank + per-chunk tickers  → FinancialRAG.ingest_pdf.
  2. Write to the LOCAL ChromaDB dir (CHROMA_LOCAL_PATH, default ./data/chroma_db).
  3. With --upload: mirror that dir to gs://$CHROMA_BUCKET/$CHROMA_GCS_PREFIX so
     the next bot restart (entrypoint._download_chroma_db) serves the new notes.

Examples
────────
  # ingest one report into the local store
  python scripts/ingest_bank_report.py reports/goldman_outlook_Q3_2026.pdf

  # ingest a folder and publish to GCS for the live bot
  python scripts/ingest_bank_report.py reports/*.pdf --upload

  # override the detected bank label
  python scripts/ingest_bank_report.py note.pdf --bank "Morgan Stanley"

  # list what's already in the store
  python scripts/ingest_bank_report.py --list

Env
───
  CHROMA_LOCAL_PATH  local ChromaDB dir           (default ./data/chroma_db)
  CHROMA_BUCKET      GCS bucket for --upload/--list-remote (default ramp-bot-chroma-db)
  CHROMA_GCS_PREFIX  object prefix                (default chroma_db/)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# Make `agent.rag_engine` importable when run from the repo root.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

CHROMA_LOCAL_PATH = os.environ.get("CHROMA_LOCAL_PATH", "data/chroma_db")
CHROMA_BUCKET     = os.environ.get("CHROMA_BUCKET", "ramp-bot-chroma-db")
CHROMA_GCS_PREFIX = os.environ.get("CHROMA_GCS_PREFIX", "chroma_db/")


def _upload_to_gcs(local_dir: str) -> int:
    """Mirror the local ChromaDB dir → gs://CHROMA_BUCKET/CHROMA_GCS_PREFIX."""
    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage not installed — cannot --upload.", file=sys.stderr)
        return 0
    client = storage.Client()
    bucket = client.bucket(CHROMA_BUCKET)
    n = 0
    for path in Path(local_dir).rglob("*"):
        if path.is_file():
            rel = path.relative_to(local_dir).as_posix()
            bucket.blob(f"{CHROMA_GCS_PREFIX}{rel}").upload_from_filename(str(path))
            n += 1
    print(f"⬆️  Uploaded {n} file(s) → gs://{CHROMA_BUCKET}/{CHROMA_GCS_PREFIX}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest bank PDFs into the RAG store.")
    ap.add_argument("pdfs", nargs="*", help="PDF path(s) or glob(s)")
    ap.add_argument("--bank", default=None, help="Override the detected bank label")
    ap.add_argument("--upload", action="store_true", help="Publish the store to GCS")
    ap.add_argument("--list", action="store_true", help="List ingested documents and exit")
    args = ap.parse_args()

    from agent.rag_engine import FinancialRAG
    rag = FinancialRAG(db_path=CHROMA_LOCAL_PATH)

    if args.list:
        docs = rag.list_documents()
        if not docs:
            print("RAG store is empty.")
            return 0
        print(f"{len(docs)} document(s) in {CHROMA_LOCAL_PATH}:")
        for d in docs:
            print(f"  [{d['date']}] {d['source']} · {d.get('bank','?')} · "
                  f"{d.get('chunks','?')} chunks · date via {d['method']}")
        return 0

    paths: list[str] = []
    for p in args.pdfs:
        paths.extend(sorted(glob.glob(p)) or [p])
    if not paths:
        ap.error("no PDFs given (or use --list)")

    total = 0
    for path in paths:
        if not os.path.isfile(path):
            print(f"⚠️  skip (not found): {path}", file=sys.stderr)
            continue
        meta = {"bank": args.bank} if args.bank else {}
        total += rag.ingest_pdf(path, doc_metadata=meta)

    print(f"\n✅ Ingested {total} chunk(s) from {len(paths)} file(s) into {CHROMA_LOCAL_PATH}")
    if args.upload:
        _upload_to_gcs(CHROMA_LOCAL_PATH)
    else:
        print("ℹ️  Local only. Re-run with --upload to publish to GCS for the live bot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
