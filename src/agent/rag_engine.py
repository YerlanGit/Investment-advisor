"""
FinancialRAG — RAG engine с взвешиванием по актуальности (Recency Scoring).

Логика ранжирования:
  Финальный_скор = (1 - distance) * W_sem + recency_score * W_rec
  где:
    distance      — ChromaDB косинусное расстояние (0 = идентично)
    recency_score — нормализованный возраст документа (0.0 = старый, 1.0 = сегодня)
    W_sem = 0.6   — вес семантической схожести
    W_rec = 0.4   — вес актуальности

Дата определяется автоматически:
  1. Из имени файла: goldman_sachs_outlook_Q1_2025.pdf → 2025
  2. Из содержимого PDF (первые 2 страницы): регэксп ищет "January 2025" / "2024-Q3"
  3. Если не нашли — используется дата модификации файла
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone

# chromadb is imported LAZILY inside __init__ (like pymupdf4llm below): the
# pure ingest helpers (_chunk_markdown / _extract_bank / _extract_tickers) and
# module import must not require the heavy vector-store dep — only actually
# opening a collection does.

# pymupdf4llm is needed ONLY for PDF ingestion (ingest_pdf).  The bot's hot
# path only QUERIES ChromaDB, so it is imported LAZILY inside ingest_pdf — a
# missing optional ingest dep must never crash module import and take the
# whole RAG query path down with it (the `No module named 'pymupdf4llm'`
# regression that broke RAG).  It is restored to requirements.txt as well;
# this lazy import is belt-and-suspenders.

logger = logging.getLogger("FinancialRAG")

# ── Recency weights ─────────────────────────────────────────────────────────
W_SEMANTIC = 0.60   # вес семантической близости к запросу
W_RECENCY  = 0.40   # вес свежести документа

# Сколько результатов запрашивать у ChromaDB перед ре-ранжированием
# (берём больше чем нужно, затем отбираем лучшие по combined score)
PREFETCH_MULTIPLIER = 5


class FinancialRAG:
    def __init__(self, db_path: str = "data/chroma_db"):
        import chromadb                                    # lazy (see module top)
        from chromadb.utils import embedding_functions
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        self.embedding_fn  = embedding_functions.DefaultEmbeddingFunction()
        self.collection    = self.chroma_client.get_or_create_collection(
            name="bank_reports",
            embedding_function=self.embedding_fn,
        )

    # ── Date extraction ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_date_from_filename(filename: str) -> datetime | None:
        """
        Пытается вытащить год (и опционально месяц) из имени файла.
        Примеры: goldman_Q1_2025.pdf → 2025-01-01
                 jpmorgan_outlook_march_2024.pdf → 2024-03-01
        """
        # Паттерн: 4-значный год
        year_match = re.search(r'(?<!\d)(20\d{2})(?!\d)', filename)
        if not year_match:
            return None
        year = int(year_match.group(1))

        # Паттерн: квартал Q1-Q4
        q_match = re.search(r'[Qq]([1-4])', filename)
        if q_match:
            month = (int(q_match.group(1)) - 1) * 3 + 1
            return datetime(year, month, 1, tzinfo=timezone.utc)

        # Паттерн: название месяца
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        }
        lower = filename.lower()
        for abbr, num in month_map.items():
            if abbr in lower:
                return datetime(year, num, 1, tzinfo=timezone.utc)

        return datetime(year, 1, 1, tzinfo=timezone.utc)

    @staticmethod
    def _extract_date_from_text(md_text: str) -> datetime | None:
        """
        Ищет даты в первых ~3000 символах PDF (обложка/оглавление).
        Паттерны: 'January 2025', 'Q1 2024', 'March 2024', '2025-03'
        """
        sample = md_text[:3000]

        # 'Month YYYY' или 'Month, YYYY'
        month_names = (
            r'(January|February|March|April|May|June|July|August|'
            r'September|October|November|December)'
        )
        m = re.search(rf'{month_names}[,\s]+(\d{{4}})', sample, re.IGNORECASE)
        if m:
            month_str = m.group(1)[:3].capitalize()
            year      = int(m.group(2))
            month_num = datetime.strptime(month_str, '%b').month
            return datetime(year, month_num, 1, tzinfo=timezone.utc)

        # 'Q1 2024'
        m = re.search(r'Q([1-4])[,\s]+(\d{4})', sample, re.IGNORECASE)
        if m:
            month = (int(m.group(1)) - 1) * 3 + 1
            return datetime(int(m.group(2)), month, 1, tzinfo=timezone.utc)

        # 'YYYY-MM' ISO
        m = re.search(r'\b(20\d{2})-(0[1-9]|1[0-2])\b', sample)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), 1, tzinfo=timezone.utc)

        return None

    def _get_doc_date(self, file_path: str, md_text: str,
                      filename: str | None = None) -> tuple[datetime, str]:
        """
        Returns (datetime, method_used).
        Priority: filename → PDF text → file modification time.

        ``filename`` (2026-07-05): the LOGICAL document name when file_path is
        a NamedTemporaryFile download — otherwise the year/quarter in the real
        name is invisible and recency falls back to tmp-file mtime (≈ «сегодня»
        для любого старого отчёта).
        """
        filename = filename or os.path.basename(file_path)

        dt = self._extract_date_from_filename(filename)
        if dt:
            return dt, "filename"

        dt = self._extract_date_from_text(md_text)
        if dt:
            return dt, "pdf_content"

        # Fallback: дата последнего изменения файла
        mtime = os.path.getmtime(file_path)
        return datetime.fromtimestamp(mtime, tz=timezone.utc), "file_mtime"

    # ── Bank / ticker extraction + section-aware chunking (RAG #в) ────────────

    # Known bank issuers → canonical label (filename + cover-page detection).
    _BANK_PATTERNS = [
        ("Goldman Sachs",  r"goldman|gs\b|\bgs_"),
        # §−14 C-8: bare `\bms\b` dropped — «MS» — обычное сокращение в тексте
        # (Microsoft, миллисекунды); имя банка требует полного «morgan stanley»
        # или файлового префикса ms_.
        ("Morgan Stanley", r"morgan\s*stanley|\bms_"),
        ("JPMorgan",       r"jp\s*morgan|jpm|j\.p\.\s*morgan"),
        ("Bank of America",r"bank\s*of\s*america|bofa|merrill"),
        ("Barclays",       r"barclays"),
        ("UBS",            r"\bubs\b"),
        ("Citi",           r"citi(group|bank)?"),
        ("Wells Fargo",    r"wells\s*fargo"),
        ("Deutsche Bank",  r"deutsche"),
        ("HSBC",           r"hsbc"),
    ]

    @classmethod
    def _extract_bank(cls, filename: str, md_text: str) -> str:
        """Canonical issuing bank from the filename first, then the cover page."""
        hay = f"{filename}\n{md_text[:1500]}".lower()
        for label, pat in cls._BANK_PATTERNS:
            if re.search(pat, hay):
                return label
        return "Unknown"

    @staticmethod
    def _extract_tickers(text: str) -> str:
        """Comma-joined uppercase tickers mentioned in a chunk (for filtering).

        ChromaDB metadata must be scalar, so we store a comma-wrapped string
        (",AAPL,MSFT,") that a `$contains` document filter or a substring check
        can match precisely without partial hits."""
        # $NVDA or bare 2–5-letter all-caps tokens; drop common English words.
        cand = set(re.findall(r"\$?([A-Z]{2,5})\b", text))
        _STOP = {"THE","AND","FOR","USD","EPS","GDP","CEO","CFO","ETF","USA",
                 "Q1","Q2","Q3","Q4","YOY","EBIT","FED","ECB","API","PDF","AI"}
        tickers = sorted(t for t in cand if t not in _STOP)[:25]
        return ("," + ",".join(tickers) + ",") if tickers else ""

    @staticmethod
    def _chunk_markdown(md_text: str, *, max_chars: int = 1200,
                        overlap: int = 150, min_chars: int = 150) -> list[tuple[str, str]]:
        """Section-aware, size-bounded chunks → list of (heading, chunk_text).

        Splits on Markdown headings (keeps the section title as metadata), then
        sub-splits any section longer than `max_chars` into overlapping windows
        so no single embedding is truncated (the old header-only split produced
        multi-page chunks that embedded poorly).  Short fragments are dropped."""
        sections = re.split(r'\n(?=#{1,3}\s)', md_text)
        out: list[tuple[str, str]] = []
        for sec in sections:
            sec = sec.strip()
            if len(sec) < min_chars:
                continue
            first_nl = sec.find("\n")
            heading = (sec[:first_nl] if first_nl > 0 else sec)[:120].lstrip("# ").strip()
            if len(sec) <= max_chars:
                out.append((heading, sec))
                continue
            start = 0
            while start < len(sec):
                piece = sec[start:start + max_chars]
                if len(piece) >= min_chars:
                    out.append((heading, piece))
                if start + max_chars >= len(sec):
                    break
                start += max_chars - overlap
        return out

    # ── Recency scoring ──────────────────────────────────────────────────────

    @staticmethod
    def _recency_score(doc_timestamp: int, half_life_days: int = 365) -> float:
        """
        Экспоненциальный decay: score = exp(-age_days / half_life_days).
        doc_timestamp — Unix epoch seconds (хранится в метаданных).
        Документ сегодняшнего дня → score ~1.0
        Документ год назад → score ~0.37 (1/e)
        Документ 3 года назад → score ~0.08
        """
        now_ts   = time.time()
        age_secs = max(0, now_ts - doc_timestamp)
        age_days = age_secs / 86_400
        import math
        return math.exp(-age_days / half_life_days)

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_pdf(self, file_path: str, doc_metadata: dict | None = None) -> int:
        """
        Reads a PDF, auto-detects its date, and stores all chunks with
        a ``doc_timestamp`` (Unix epoch) in metadata for recency scoring.
        Returns number of chunks ingested.
        """
        if doc_metadata is None:
            doc_metadata = {}

        # 2026-07-05: the Cloud Function / boot-ingest download PDFs into
        # NamedTemporaryFile paths, so basename(file_path) was «tmpl3mmhrmf.pdf»
        # — that garbage became the `source` metadata (leaked into the report's
        # RAG chips) AND broke filename-based date/bank detection, silently
        # degrading recency ranking.  The callers already pass the REAL object
        # name in doc_metadata["filename"] — prefer it.
        filename = os.path.basename(
            str(doc_metadata.get("filename") or os.path.basename(file_path)))
        logger.info("[RAG] Парсинг: %s", filename)
        print(f"[RAG] Парсинг: {filename}")

        try:
            import pymupdf4llm  # lazy: only needed for ingestion (see top)
            md_text = pymupdf4llm.to_markdown(file_path)
        except ImportError as e:
            logger.error("[RAG] pymupdf4llm не установлен — ingest пропущен: %s", e)
            return 0
        except Exception as e:
            logger.error("[RAG] Ошибка чтения PDF %s: %s", filename, e)
            print(f"[RAG] Ошибка: {e}")
            return 0

        # ── Определяем дату документа ──────────────────────────────────────
        doc_dt, method = self._get_doc_date(file_path, md_text, filename=filename)
        doc_ts = int(doc_dt.timestamp())
        print(f"[RAG] Дата документа: {doc_dt.strftime('%Y-%m-%d')} (источник: {method})")

        # ── Section-aware, size-bounded chunking + rich metadata ───────────
        bank = self._extract_bank(filename, md_text)
        sized = self._chunk_markdown(md_text)

        chunks, metadatas, ids = [], [], []
        for i, (heading, chunk) in enumerate(sized):
            meta = doc_metadata.copy()
            meta.update({
                "source":         filename,
                "bank":           doc_metadata.get("bank") or bank,
                "section":        heading or "—",
                "tickers":        self._extract_tickers(chunk),   # ",AAPL,MSFT,"
                "chunk_index":    i,
                "doc_timestamp":  doc_ts,   # ← ключевое поле для ранжирования
                "doc_date_str":   doc_dt.strftime("%Y-%m-%d"),
                "date_method":    method,
            })
            chunks.append(chunk)
            metadatas.append(meta)
            ids.append(f"{filename}_ch{i}")

        if chunks:
            self.collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)
            print(f"[RAG] ✅ Загружено {len(chunks)} блоков (дата: {doc_dt.strftime('%Y-%m')})")
        else:
            print("[RAG] ⚠️  Значимый текст не найден.")

        return len(chunks)

    # ── Query with recency re-ranking ────────────────────────────────────────

    def get_market_sentiment(
        self,
        query: str,
        n_results: int = 3,
        half_life_days: int = 365,
        ticker: str | None = None,
    ) -> str:
        """
        Retrieves n_results chunks ranked by:
          score = semantic_similarity * W_SEMANTIC + recency_score * W_RECENCY

        half_life_days (default 365): через сколько дней документ теряет
        половину своей «актуальности». Уменьшите до 180 для быстрых рынков.

        ticker (RAG #в): when given, SOFT-filter to chunks that actually mention
        the ticker (metadata `tickers` OR document text) so a per-holding query
        pulls notes about THAT name — but falls back to the unfiltered top set
        when the ticker has no coverage (never returns empty just because a
        name isn't in the library).
        """
        if self.collection.count() == 0:
            return "NO PDF DATA AVAILABLE. База отчетов пуста."

        # Запрашиваем больше результатов для ре-ранжирования (шире при фильтре).
        mult = PREFETCH_MULTIPLIER * (3 if ticker else 1)
        prefetch = min(n_results * mult, self.collection.count())

        results = self.collection.query(
            query_texts=[query],
            n_results=prefetch,
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]   # 0 = identical, 2 = opposite

        if not docs:
            return "No relevant information found."

        # ── Re-rank ────────────────────────────────────────────────────────
        scored = []
        for doc, meta, dist in zip(docs, metas, distances):
            sem_score  = 1.0 - (dist / 2.0)   # normalize [0, 1]
            doc_ts     = meta.get("doc_timestamp", 0)
            rec_score  = self._recency_score(doc_ts, half_life_days)
            combined   = sem_score * W_SEMANTIC + rec_score * W_RECENCY
            scored.append((combined, sem_score, rec_score, doc, meta))

        # Сортировка: лучшие наверх
        scored.sort(key=lambda x: x[0], reverse=True)

        # Soft ticker filter — keep only chunks mentioning the name, but fall
        # back to the unfiltered ranking when nothing matches.
        if ticker:
            tk = ticker.upper().split(".")[0]
            def _mentions(item):
                _, _, _, doc, meta = item
                return (f",{tk}," in (meta.get("tickers") or "")) or (tk in doc.upper())
            matched = [s for s in scored if _mentions(s)]
            if matched:
                scored = matched

        top = scored[:n_results]

        # ── Build context string ───────────────────────────────────────────
        context = ""
        for combined, sem, rec, doc, meta in top:
            bank_sec = " · ".join(x for x in [meta.get("bank"), meta.get("section")]
                                  if x and x not in ("—", "Unknown"))
            context += (
                f"\n--- [{meta.get('doc_date_str', '?')}] "
                f"{meta.get('source', '?')}"
                f"{(' — ' + bank_sec) if bank_sec else ''} "
                f"(актуальность: {rec:.0%}, схожесть: {sem:.0%}, "
                f"итог: {combined:.0%}) ---\n"
            )
            context += f"{doc}\n"

        return context

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_documents(self) -> list[dict]:
        """Returns a summary of all ingested documents, sorted by date desc."""
        if self.collection.count() == 0:
            return []

        # Get all metadata (no query needed)
        all_data = self.collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in all_data["metadatas"]:
            src = meta.get("source", "?")
            if src not in seen:
                seen[src] = {
                    "source":     src,
                    "bank":       meta.get("bank", "Unknown"),
                    "date":       meta.get("doc_date_str", "unknown"),
                    "timestamp":  meta.get("doc_timestamp", 0),
                    "method":     meta.get("date_method", "?"),
                    "chunks":     0,
                }
            seen[src]["chunks"] += 1

        return sorted(seen.values(), key=lambda x: x["timestamp"], reverse=True)
