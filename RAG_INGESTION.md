# RAG_INGESTION.md — загрузка аналитических отчётов банков в базу (ChromaDB)

Пошаговая инструкция, как загрузить банковские аналитические записки (Goldman Sachs / Morgan
Stanley / JPMorgan / Barclays / …) в RAG-базу так, чтобы бот их **читал, ранжировал по свежести и
использовал при подборе активов**.

---

## 0. Как это устроено (30 секунд)

```
PDF аналитики банка
      │
      ├── (A) заливка в GCS-бакет ──► Cloud Function (авто-триггер) ─┐
      │                                                             ├─► ChromaDB в GCS
      └── (B) scripts/ingest_bank_report.py --upload ───────────────┘   gs://ramp-bot-chroma-db-investadv/chroma_db/
                                                                             │
                                             при старте контейнера бот тянет ▼
                                             entrypoint._download_chroma_db → /app/data/chroma_db
                                                                             │
                     tg_bot._fetch_rag_context → FinancialRAG.get_market_sentiment(query, ticker=…)
                                                     (semantic 0.6 ⊕ recency 0.4, фильтр по тикеру)
```

- **Хранилище:** ChromaDB, коллекция `bank_reports`, персистится в GCS (переживает рестарт Cloud Run).
- **Ранжирование:** `score = 0.6·семантика + 0.4·свежесть` (экспон. decay, half-life 365 дн).
- **Метаданные чанка:** `source, bank, section, tickers, doc_timestamp, doc_date_str` — для фильтрации
  по банку/тикеру и по дате.

---

## 1. Подготовьте PDF (важно: имя файла)

Дата и банк определяются **сначала из имени файла**, потом из обложки PDF. Дайте файлу «говорящее» имя —
это гарантирует правильную свежесть и метку банка:

```
<bank>_<theme>_<Q#|month>_<YYYY>.pdf
```

Примеры (хорошо):
```
goldman_sachs_equity_outlook_Q3_2026.pdf
morgan_stanley_us_strategy_july_2026.pdf
jpmorgan_rates_2026-07.pdf
barclays_tech_sector_2026.pdf
```

Правила:
- **Год обязателен** (`20YY`) — иначе свежесть берётся из даты файла (менее точно).
- Квартал `Q1..Q4` или месяц (`july`/`march`/…) уточняют дату.
- Имя банка в файле → корректная метка `bank` (можно переопределить `--bank`).

---

## 2A. Способ A — заливка в GCS (постоянно/онлайн, авто-ингест)

Лучший для «обновляются постоянно». Кладёте PDF в бакет — Cloud Function сама парсит и обновляет базу.

```bash
# один файл
gsutil cp goldman_sachs_equity_outlook_Q3_2026.pdf \
    gs://ramp-bot-chroma-db-inbox-investadv/          # бакет-триггер Cloud Function

# пачкой
gsutil -m cp reports/*.pdf gs://ramp-bot-chroma-db-inbox-investadv/
```

- Cloud Function (`cloud_function/main.py`) срабатывает на загрузку, ингестит PDF, пишет ChromaDB в
  `gs://ramp-bot-chroma-db-investadv/chroma_db/`.
- Бот подхватит на следующем рестарте (или запланируйте рестарт/`--no-traffic` ревизию).
- **Онлайн быстро и надёжно:** можно повесить `gsutil cp` на Cloud Scheduler + маленький скрапер витрин
  GS/MS/JPM, либо давать аналитикам прямой upload в inbox-бакет (IAM `roles/storage.objectCreator`).

## 2B. Способ B — админ-CLI (быстро/вручную, без ожидания триггера)

Ингест локально + публикация в GCS одной командой (`scripts/ingest_bank_report.py`):

```bash
# 1) поставить зависимости ингеста (если ещё нет)
pip install pymupdf4llm chromadb google-cloud-storage

# 2) ингест одного отчёта в ЛОКАЛЬНУЮ базу
python scripts/ingest_bank_report.py reports/goldman_outlook_Q3_2026.pdf

# 3) ингест папки + публикация в GCS (бот увидит после рестарта)
python scripts/ingest_bank_report.py reports/*.pdf --upload

# переопределить банк, если не распознан из имени
python scripts/ingest_bank_report.py note.pdf --bank "Morgan Stanley" --upload

# посмотреть, что уже в базе
python scripts/ingest_bank_report.py --list
```

Переменные окружения:
```
CHROMA_LOCAL_PATH   локальная папка ChromaDB     (default ./data/chroma_db)
CHROMA_BUCKET       GCS-бакет для --upload        (default ramp-bot-chroma-db-investadv)
CHROMA_GCS_PREFIX   префикс объекта               (default chroma_db/)
GOOGLE_APPLICATION_CREDENTIALS  сервис-аккаунт с доступом к бакету (для --upload)
```

---

## 3. Проверьте, что загрузилось

```bash
python scripts/ingest_bank_report.py --list
# [2026-07-01] goldman_..._Q3_2026.pdf · Goldman Sachs · 42 chunks · date via filename
```

В отчёте DEEP чекер **«Bank RAG (выдержки)»** станет `ok` (а не `not available`), а в `quality`-строке
`✓ RAG: банк. отчёты`. AI-вердикт начнёт цитировать записки (с датой и банком).

---

## 4. Как записки влияют на выбор активов (что улучшено)

- **Секция-осведомлённый чанкинг** + ограничение размера (≤1200 симв., overlap 150) → эмбеддинги не
  обрезаются, retrieval точнее, чем при старом «резать по заголовкам целиком».
- **Метаданные `bank` / `section` / `tickers`** → контекст для LLM показывает источник и раздел; запрос
  можно сузить по тикеру: `get_market_sentiment(query, ticker="NVDA")` — мягкий фильтр (если тикер не
  покрыт, возвращает общий топ, не пусто).
- **Свежесть (recency)**: свежие записки ранжируются выше; для быстрых рынков уменьшите
  `half_life_days` (например, 180).

---

## 5. Частые проблемы

> **PDF залиты, но отчёт всё равно пуст?** → полный 8-шаговый runbook: **`docs/RAG_TROUBLESHOOTING.md`**
> (сверка имён бакетов INBOX↔STORE, деплой/триггер функции, логи ингеста, бут-синк бота, env-дрейф).

**Два разных бакета — не путать:** PDF льют в **INBOX** `ramp-bot-chroma-db-inbox-investadv`
(его слушает Cloud Function); собранная база ChromaDB лежит в **STORE** `ramp-bot-chroma-db-investadv`
(её бот тянет на буте). PDF, положенный в STORE, никто не ингестит.

| Симптом | Причина | Решение |
|---|---|---|
| `RAG not available` в отчёте | база пуста / не синкнулась / **бот не рестартовали** (синк только на буте) | залить PDF в INBOX (2A/2B), проверить, что функция отработала, **рестартнуть бот**; см. `RAG_TROUBLESHOOTING.md` |
| PDF в GCS, триггера нет | залито НЕ в INBOX-бакет / имена рассинхронены | `gcloud storage ls gs://ramp-bot-chroma-db-inbox-investadv/`; перезалить в INBOX |
| Неверная дата записки | в имени нет года | переименовать по конвенции §1 |
| `bank = Unknown` | банк не распознан | добавить в имя файла или `--bank "…"` |
| `No module named 'pymupdf4llm'` | нет ingest-зависимости | `pip install pymupdf4llm` (нужен только для ингеста) |
| `--upload` падает | нет `google-cloud-storage`/прав | `pip install google-cloud-storage`, задать сервис-аккаунт |

---

**Файлы:** `src/agent/rag_engine.py` (движок RAG), `scripts/ingest_bank_report.py` (админ-CLI),
`cloud_function/` (авто-ингест по GCS-триггеру), `src/entrypoint.py` (синк ChromaDB из GCS на буте),
`docs/RAG_TROUBLESHOOTING.md` (диагностика пустой базы).
