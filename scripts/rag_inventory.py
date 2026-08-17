#!/usr/bin/env python3
"""
Admin CLI — ЧТО ИМЕННО лежит в RAG-базе и ЧИТАЕТСЯ ли туда новый отчёт.

Отвечает ровно на два вопроса, которые нельзя ответить по коду:
  1. «доехали ли новые отчёты банков до ChromaDB» — сверка INBOX ↔ store;
  2. «на сколько выросли отчёты/чанки» — дельта к предыдущему замеру.

Зачем отдельный скрипт, если есть `ingest_bank_report.py --list`
──────────────────────────────────────────────────────────────
`--list` печатает документы ЛОКАЛЬНОГО store и ничего не знает ни про INBOX,
ни про дельту.  При разборе «залил 10 отчётов — не выросло» нужны именно они:
что в INBOX, что уже ингестировано, чего не хватает, и на сколько изменились
числа из строки CoVe «база: N отчётов · M чанков».

ТРИ МЕСТА, ГДЕ РВЁТСЯ ЦЕПОЧКА «PDF залит → цифра в отчёте выросла»
─────────────────────────────────────────────────────────────────
  INBOX  gs://ramp-bot-chroma-db-inbox-investadv/      ← PDF кладут СЮДА
  STORE  gs://ramp-bot-chroma-db-investadv/chroma_db/  ← БИНАРНИК ChromaDB

  (1) INBOX → STORE.  Ингест: Cloud Function по событию finalize либо
      `entrypoint._boot_ingest_from_inbox` при старте контейнера.  Порвётся —
      `--inbox` покажет, каких файлов в базе нет.
  (2) STORE → контейнер.  `_download_chroma_db` зовётся ТОЛЬКО из `__main__`,
      то есть синк происходит ОДИН РАЗ НА БУТЕ.  Работающий бот держит копию,
      снятую при старте, и свежий STORE его не догоняет — самый частый случай
      «ингест прошёл, а в отчёте старые цифры».  Лечится рестартом.
  (3) PDF, залитый в STORE по ошибке.  Там его не парсит никто и никогда;
      ищет `--check-store-pdfs`.

Отсюда важное: `--from-gcs` показывает STORE, то есть то, что бот увидит
ПОСЛЕ рестарта, а не обязательно то, что он отдаёт прямо сейчас.

Примеры
───────
  # инвентарь локального store + дельта к прошлому замеру
  python scripts/rag_inventory.py --baseline-docs 28 --baseline-chunks 2984

  # то же по ЖИВОЙ базе бота (скачивает STORE во временный каталог)
  python scripts/rag_inventory.py --from-gcs

  # главное: сверить INBOX со store — доехали ли новые отчёты
  python scripts/rag_inventory.py --from-gcs --inbox --check-store-pdfs

Env
───
  CHROMA_LOCAL_PATH  локальный каталог ChromaDB   (default ./data/chroma_db)
  CHROMA_BUCKET      STORE-бакет                  (default ramp-bot-chroma-db-investadv)
  CHROMA_GCS_PREFIX  префикс объекта              (default chroma_db/)
  RAG_INBOX_BUCKET   INBOX-бакет                  (default ramp-bot-chroma-db-inbox-investadv)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Make `agent.rag_engine` importable when run from the repo root.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

CHROMA_LOCAL_PATH = os.environ.get("CHROMA_LOCAL_PATH", "data/chroma_db")
CHROMA_BUCKET     = os.environ.get("CHROMA_BUCKET", "ramp-bot-chroma-db-investadv")
CHROMA_GCS_PREFIX = os.environ.get("CHROMA_GCS_PREFIX", "chroma_db/")
INBOX_BUCKET      = os.environ.get("RAG_INBOX_BUCKET",
                                   "ramp-bot-chroma-db-inbox-investadv")


# ── Pure helpers (unit-tested; no GCS, no chromadb) ──────────────────────────

def norm_name(name) -> str:
    """Normalised basename — the SAME key `entrypoint._boot_ingest_from_inbox`
    uses to decide whether an INBOX PDF is already ingested.  Diverging from it
    would make this report disagree with what the bot actually does."""
    return os.path.basename(str(name or "")).strip().lower()


def bank_tally(docs: list[dict]) -> list[tuple[str, int, int]]:
    """[(bank, reports, chunks)] — issuers sorted by coverage, «Unknown» last.

    «Unknown» is kept here (unlike the report label): in a diagnostic it is the
    finding itself — an issuer whose name `rag_engine._BANK_PATTERNS` does not
    recognise, so its excerpts reach the report WITHOUT attribution.
    """
    tally: dict[str, list[int]] = {}
    for d in docs or []:
        bank = str((d or {}).get("bank") or "Unknown").strip() or "Unknown"
        row = tally.setdefault(bank, [0, 0])
        row[0] += 1
        row[1] += int((d or {}).get("chunks", 0) or 0)
    def _key(item):
        bank, (n_docs, n_chunks) = item
        return (bank == "Unknown", -n_chunks, bank)   # Unknown always last
    return [(b, v[0], v[1]) for b, v in sorted(tally.items(), key=_key)]


def diff_inbox(inbox_names: list[str], ingested_sources: list[str]) -> dict:
    """Which INBOX PDFs reached the store, and which did not."""
    ingested = {norm_name(s) for s in ingested_sources or []}
    inbox    = [str(n) for n in inbox_names or []]
    missing  = [n for n in inbox if norm_name(n) not in ingested]
    present  = [n for n in inbox if norm_name(n) in ingested]
    # Ingested documents with no PDF behind them in INBOX — either the file was
    # removed after ingestion or it was loaded by the manual CLI.  Not an error,
    # but it explains a store that holds MORE reports than INBOX has files.
    inbox_keys = {norm_name(n) for n in inbox}
    orphans = sorted({norm_name(s) for s in ingested_sources or []} - inbox_keys)
    return {"present": present, "missing": missing, "orphans": orphans}


def format_delta(label: str, current: int, baseline: int | None) -> str:
    """«28 → 38 (+10)» — the answer to «на сколько увеличилось»."""
    if baseline is None:
        return f"{label}: {current}"
    delta = current - baseline
    sign = f"+{delta}" if delta > 0 else (str(delta) if delta < 0 else "±0")
    return f"{label}: {baseline} → {current} ({sign})"


# ── GCS / store access ───────────────────────────────────────────────────────

def _download_store(dest: str) -> int:
    """Mirror gs://CHROMA_BUCKET/CHROMA_GCS_PREFIX → dest.  Mirrors
    `entrypoint._download_chroma_db`, so what we inspect is what the bot sees."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(CHROMA_BUCKET)
    os.makedirs(dest, exist_ok=True)
    n = 0
    for blob in bucket.list_blobs(prefix=CHROMA_GCS_PREFIX):
        rel = blob.name[len(CHROMA_GCS_PREFIX):]
        if not rel or blob.name.endswith("/"):
            continue
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out) or dest, exist_ok=True)
        blob.download_to_filename(out)
        n += 1
    return n


def _list_bucket_pdfs(bucket_name: str) -> list[str]:
    from google.cloud import storage
    client = storage.Client()
    return sorted(b.name for b in client.bucket(bucket_name).list_blobs()
                  if b.name.lower().endswith(".pdf"))


# ── Report ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="RAG store inventory: reports, chunks, banks, INBOX diff.")
    ap.add_argument("--from-gcs", action="store_true",
                    help="inspect the LIVE store (downloads it to a temp dir)")
    ap.add_argument("--inbox", action="store_true",
                    help="compare the INBOX bucket against ingested sources")
    ap.add_argument("--check-store-pdfs", action="store_true",
                    help="flag PDFs sitting in the STORE bucket (wrong bucket)")
    ap.add_argument("--baseline-docs", type=int, default=None,
                    help="previous report count, to print the delta")
    ap.add_argument("--baseline-chunks", type=int, default=None,
                    help="previous chunk count, to print the delta")
    ap.add_argument("--per-report", action="store_true",
                    help="list every ingested report, newest first")
    args = ap.parse_args()

    tmpdir = None
    if args.from_gcs:
        tmpdir = tempfile.mkdtemp(prefix="rag_store_")
        try:
            n = _download_store(tmpdir)
        except Exception as exc:
            print(f"❌ Не смог скачать STORE gs://{CHROMA_BUCKET}/{CHROMA_GCS_PREFIX}: {exc}",
                  file=sys.stderr)
            return 2
        print(f"⬇️  STORE gs://{CHROMA_BUCKET}/{CHROMA_GCS_PREFIX} → {tmpdir} ({n} файлов)")
        if n == 0:
            print("⚠️  STORE ПУСТ — бот стартует с пустым RAG "
                  "(«база пуста/недоступна, отчёты не читались»).")
        db_path = tmpdir
    else:
        db_path = CHROMA_LOCAL_PATH

    from agent.rag_engine import FinancialRAG
    rag = FinancialRAG(db_path=db_path)

    docs   = rag.list_documents()               # real sources (tmp artifacts out)
    all_   = rag.list_documents(include_temp=True)
    chunks = sum(int(d.get("chunks", 0) or 0) for d in docs)

    print(f"\n📚 RAG store: {db_path}")
    print("   " + format_delta("отчётов", len(docs), args.baseline_docs))
    print("   " + format_delta("чанков",  chunks,    args.baseline_chunks))
    artifacts = len(all_) - len(docs)
    if artifacts:
        print(f"   ⚠️  ещё {artifacts} tmp-источник(ов) — артефакты ингеста, "
              f"в счёт не идут (purge_temp_sources их снимет на буте)")

    print("\n🏦 По банкам (отчётов · чанков):")
    for bank, n_docs, n_chunks in bank_tally(docs):
        flag = ("   ← имя эмитента НЕ распознано (rag_engine._BANK_PATTERNS): "
                "выдержки уедут без автора" if bank == "Unknown" else "")
        print(f"   {bank:<18} {n_docs:>3} · {n_chunks:>5}{flag}")

    if args.per_report:
        print("\n📄 Отчёты (свежие сверху):")
        for d in docs:
            print(f"   [{d['date']}] {d['source']} · {d.get('bank','?')} · "
                  f"{d.get('chunks', 0)} чанков · дата из {d['method']}")

    rc = 0
    if args.inbox:
        try:
            inbox_pdfs = _list_bucket_pdfs(INBOX_BUCKET)
        except Exception as exc:
            print(f"\n❌ INBOX gs://{INBOX_BUCKET} недоступен: {exc}", file=sys.stderr)
            return 2
        diff = diff_inbox(inbox_pdfs, [d["source"] for d in docs])
        print(f"\n📥 INBOX gs://{INBOX_BUCKET}: {len(inbox_pdfs)} PDF")
        print(f"   ингестировано: {len(diff['present'])}")
        if diff["missing"]:
            rc = 1
            print(f"   ❌ НЕ ингестировано: {len(diff['missing'])}")
            for name in diff["missing"]:
                print(f"      · {name}")
            print("   → рестартните бот: `_boot_ingest_from_inbox` доберёт "
                  "недостающее в контейнере (выключатель RAG_BOOT_INGEST=0).")
        else:
            print("   ✅ все PDF из INBOX уже в базе")
        if diff["orphans"]:
            print(f"   ℹ️  в базе есть {len(diff['orphans'])} отчёт(ов) без файла "
                  f"в INBOX (залиты вручную либо PDF удалён)")

    if args.check_store_pdfs:
        try:
            stray = _list_bucket_pdfs(CHROMA_BUCKET)
        except Exception as exc:
            print(f"\n❌ STORE gs://{CHROMA_BUCKET} недоступен: {exc}", file=sys.stderr)
            return 2
        if stray:
            rc = 1
            print(f"\n🚨 В STORE-бакете gs://{CHROMA_BUCKET} лежит {len(stray)} PDF — "
                  "их НИКТО не ингестит:")
            for name in stray:
                print(f"      · {name}")
            print(f"   → перелейте их в INBOX:\n"
                  f"      gcloud storage cp gs://{CHROMA_BUCKET}/'*.pdf' "
                  f"gs://{INBOX_BUCKET}/\n"
                  f"   и рестартните бот.")
        else:
            print(f"\n✅ В STORE-бакете PDF нет (там только бинарник ChromaDB).")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
