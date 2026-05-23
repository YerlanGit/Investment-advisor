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

import chromadb
import pymupdf4llm
from chromadb.utils import embedding_functions

logger = logging.getLogger("FinancialRAG")

# ── Recency weights ─────────────────────────────────────────────────────────
W_SEMANTIC = 0.60   # вес семантической близости к запросу
W_RECENCY  = 0.40   # вес свежести документа

# Сколько результатов запрашивать у ChromaDB перед ре-ранжированием
# (берём больше чем нужно, затем отбираем лучшие по combined score)
PREFETCH_MULTIPLIER = 5


class FinancialRAG:
    def __init__(self, db_path: str = "data/chroma_db"):
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

    def _get_doc_date(self, file_path: str, md_text: str) -> tuple[datetime, str]:
        """
        Returns (datetime, method_used).
        Priority: filename → PDF text → file modification time.
        """
        filename = os.path.basename(file_path)

        dt = self._extract_date_from_filename(filename)
        if dt:
            return dt, "filename"

        dt = self._extract_date_from_text(md_text)
        if dt:
            return dt, "pdf_content"

        # Fallback: дата последнего изменения файла
        mtime = os.path.getmtime(file_path)
        return datetime.fromtimestamp(mtime, tz=timezone.utc), "file_mtime"

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

        filename = os.path.basename(file_path)
        logger.info("[RAG] Парсинг: %s", filename)
        print(f"[RAG] Парсинг: {filename}")

        try:
            md_text = pymupdf4llm.to_markdown(file_path)
        except Exception as e:
            logger.error("[RAG] Ошибка чтения PDF %s: %s", filename, e)
            print(f"[RAG] Ошибка: {e}")
            return 0

        # ── Определяем дату документа ──────────────────────────────────────
        doc_dt, method = self._get_doc_date(file_path, md_text)
        doc_ts = int(doc_dt.timestamp())
        print(f"[RAG] Дата документа: {doc_dt.strftime('%Y-%m-%d')} (источник: {method})")

        # ── Нарезаем по Markdown-заголовкам ────────────────────────────────
        raw_chunks = re.split(r'\n(?=#{1,3}\s)', md_text)

        chunks, metadatas, ids = [], [], []
        for i, chunk in enumerate(raw_chunks):
            chunk = chunk.strip()
            if len(chunk) < 150:
                continue

            meta = doc_metadata.copy()
            meta.update({
                "source":         filename,
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
    ) -> str:
        """
        Retrieves n_results chunks ranked by:
          score = semantic_similarity * W_SEMANTIC + recency_score * W_RECENCY

        half_life_days (default 365): через сколько дней документ теряет
        половину своей «актуальности». Уменьшите до 180 для быстрых рынков.
        """
        if self.collection.count() == 0:
            return "NO PDF DATA AVAILABLE. База отчетов пуста."

        # Запрашиваем больше результатов для ре-ранжирования
        prefetch = min(n_results * PREFETCH_MULTIPLIER, self.collection.count())

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
        top = scored[:n_results]

        # ── Build context string ───────────────────────────────────────────
        context = ""
        for combined, sem, rec, doc, meta in top:
            context += (
                f"\n--- [{meta.get('doc_date_str', '?')}] "
                f"{meta.get('source', '?')} "
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
                    "date":       meta.get("doc_date_str", "unknown"),
                    "timestamp":  meta.get("doc_timestamp", 0),
                    "method":     meta.get("date_method", "?"),
                }

        return sorted(seen.values(), key=lambda x: x["timestamp"], reverse=True)
