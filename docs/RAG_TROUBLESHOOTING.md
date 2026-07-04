# RAG_TROUBLESHOOTING.md — почему «Bank RAG» пуст, хотя PDF загружены в GCS

Пошаговый runbook: PDF банков лежат в Google Cloud Storage, но отчёт всё равно показывает
`✗ RAG: банк. отчёты` / `Bank RAG (выдержки) · база: 0 отчётов`. Ниже — как за 8 шагов найти
точку обрыва. Выполняйте сверху вниз, первый «красный» шаг = причина.

---

## 0. Как устроен пайплайн (это важно для диагностики)

Пайплайн **двухступенчатый** и использует **ТРИ бакета** (имена — после миграции `-investadv`,
см. §−N в AUDIT / `RAG_INGESTION.md`):

```
(1) PDF кладут сюда  ─►  gs://ramp-bot-chroma-db-inbox-investadv/     ← INBOX (триггер-бакет)
                              │  Eventarc → Cloud Function «ramp-bot-rag-ingest»
                              ▼
        Функция: парсит PDF (pymupdf4llm) → ingest → пишет ChromaDB
                              │
                              ▼
(2) ChromaDB хранится тут ─►  gs://ramp-bot-chroma-db-investadv/chroma_db/   ← STORE
                              │  бот НА БУТЕ: entrypoint._download_chroma_db()
                              ▼
(3) /app/data/chroma_db  (внутри работающего Cloud Run контейнера)
                              │  на запросе: tg_bot._fetch_rag_context → get_market_sentiment
                              ▼
              AI-нарратив + чекер CoVe «Bank RAG (выдержки)»
```

Два **не-очевидных** свойства, которые ломают интуицию:
- **PDF нужно лить в INBOX-бакет, а не в STORE.** INBOX (`…-inbox-investadv`) слушает функция;
  STORE (`…-investadv`) хранит уже собранную базу. PDF, положенный в STORE, никто не ингестит.
- **Бот тянет базу ТОЛЬКО на буте контейнера.** PDF, заингестенный уже после старта текущей
  ревизии Cloud Run, невидим боту до рестарта/новой ревизии.

> ⚠️ **Частая причина №1 после миграции имён.** Если бакеты в GCP и ваши загрузки используют
> СТАРЫЕ имена (`ramp-bot-ingest`, `ramp-bot-chroma-db`), а задеплоенный код смотрит на новые
> `-investadv` — пайплайн «немой». Либо создайте/используйте новые бакеты (шаги ниже), либо
> верните старые имена в коде (см. §9).

---

## 1. Сверка имён: какие бакеты ждёт задеплоенный код и что существует в GCP

```bash
# Что реально существует:
gcloud storage buckets list --format='value(name)' | grep -Ei 'chroma|ingest'

# Ожидаемые задеплоенным кодом (после миграции):
#   INBOX: ramp-bot-chroma-db-inbox-investadv
#   STORE: ramp-bot-chroma-db-investadv
```
- Нет бакета `…-inbox-investadv` или `…-investadv` → **это причина.** Создайте их или см. §9.
- Есть только старые (`ramp-bot-ingest`, `ramp-bot-chroma-db`) → имена рассинхронизированы.

## 2. PDF лежат в INBOX-бакете (а не в STORE и не в старом)?

```bash
gcloud storage ls -l gs://ramp-bot-chroma-db-inbox-investadv/**.pdf
```
- Пусто, а PDF вы видите в другом бакете (`ramp-bot-ingest` / `ramp-bot-chroma-db` /
  `ramp-bot-reports`) → **залито не туда.** Перезалейте в INBOX:
  ```bash
  gcloud storage cp reports/*.pdf gs://ramp-bot-chroma-db-inbox-investadv/
  ```
- Имя файла должно нести год: `goldman_sachs_outlook_Q3_2026.pdf` (иначе свежесть берётся из
  mtime — см. `RAG_INGESTION.md §1`).

## 3. Cloud Function задеплоена и слушает ИМЕННО INBOX?

```bash
gcloud functions describe ramp-bot-rag-ingest --gen2 --region=us-central1 \
  --format='value(eventTrigger.eventFilters)'
# ДОЛЖНО содержать: attribute=bucket, value=ramp-bot-chroma-db-inbox-investadv
```
- Функции нет / триггер на старом бакете → **причина.** Важно: в `cloudbuild.yaml` шаг деплоя
  функции **fail-soft** (`|| echo "WARN…"; exit 0`) — при 403/нехватке прав он молча
  пропускается, и функция может быть НЕ задеплоена или устаревшей. Передеплойте вручную:
  ```bash
  gcloud functions deploy ramp-bot-rag-ingest --gen2 --runtime=python312 \
    --region=us-central1 --source=./cloud_function --entry-point=on_pdf_uploaded \
    --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
    --trigger-event-filters="bucket=ramp-bot-chroma-db-inbox-investadv" \
    --set-env-vars CHROMA_BUCKET=ramp-bot-chroma-db-investadv \
    --memory=1Gi --timeout=300s --max-instances=1 --concurrency=1
  ```
  (`--max-instances=1 --concurrency=1` ОБЯЗАТЕЛЬНЫ — F-4: без сериализации две загрузки
  портят базу.)

## 4. Функция реально запускалась и заингестила чанки?

```bash
gcloud functions logs read ramp-bot-rag-ingest --gen2 --region=us-central1 --limit=80
```
Ищите по каждому PDF цепочку:
- `[Trigger] New PDF: gs://…/<file>.pdf`  — триггер сработал.
- `[Ingest] Added N chunks. Total: M`      — **N должно быть > 0.**
- `[Done] ChromaDB updated in GCS.`         — записал обратно в STORE.

Диагнозы:
- **Нет `[Trigger]` вообще** → триггер не сработал (бакет/деплой из шагов 2–3).
- `[Ingest] Added 0 chunks` → PDF без извлекаемого текста (скан/картинка) ИЛИ нет
  `pymupdf4llm`. Проверьте, что PDF текстовый; см. `cloud_function/requirements.txt`.
- `No module named 'pymupdf4llm'` → добавьте зависимость в функцию и передеплойте.

## 5. ChromaDB реально появился в STORE-бакете?

```bash
gcloud storage ls -lr gs://ramp-bot-chroma-db-investadv/chroma_db/
# Ожидаем: chroma.sqlite3 (не 0 байт) + подпапки сегментов (UUID-каталоги)
```
- Пусто / только `chroma.sqlite3` в 0 байт → ingest не записался (функция упала на upload,
  либо у неё неверный `CHROMA_BUCKET`). Вернитесь к шагу 4 (логи), проверьте env шага 3.

## 6. Бот ПЕРЕЗАПУСКАЛСЯ, чтобы подтянуть базу? (причина №2)

Бот синкает базу **только на буте**. Смотрим бут-логи текущей ревизии:
```bash
gcloud run services logs read ramp-bot --region=us-central1 --limit=200 \
  | grep -i chroma
```
- `ChromaDB synced from gs://ramp-bot-chroma-db-investadv/chroma_db/ → … (N blobs).` и **N>0**
  → база доехала. Если отчёт всё равно пуст — проблема не в синке (см. §7).
- `ChromaDB bucket gs://…-investadv/chroma_db/ is empty — bot starts with empty RAG.`
  → на момент старта бота STORE был пуст (ingest ещё не отработал ⇒ рестартните ПОСЛЕ шага 5).
- `ChromaDB sync failed (continuing anyway): …` → права/сеть; у сервис-аккаунта Cloud Run нет
  `roles/storage.objectViewer` на STORE-бакет.
- Строк про ChromaDB нет вовсе → нет `google-cloud-storage` или ревизия старая.

**Форс-рестарт (новая ревизия тянет свежую базу):**
```bash
gcloud run services update ramp-bot --region=us-central1 \
  --update-env-vars=RAG_REFRESH=$(date +%s)
```

## 7. Env-переменные работающих сервисов совпадают с бакетами?

```bash
gcloud run services describe ramp-bot --region=us-central1 \
  --format='value(spec.template.spec.containers[0].env)' | tr ',' '\n' | grep CHROMA
# CHROMA_BUCKET=ramp-bot-chroma-db-investadv  (или переменная не задана → берётся дефолт кода)

gcloud functions describe ramp-bot-rag-ingest --gen2 --region=us-central1 \
  --format='value(serviceConfig.environmentVariables)' | tr ',' '\n' | grep CHROMA
# CHROMA_BUCKET=ramp-bot-chroma-db-investadv
```
Если у функции и бота `CHROMA_BUCKET` указывают на РАЗНЫЕ бакеты — функция пишет в один,
бот читает из другого. Приведите к одному STORE-бакету.

## 8. Версия ChromaDB не «дрейфует» между функцией и ботом?

Если STORE не пуст (шаг 5 ✓), но бот на буте видит 0 (шаг 6 «is empty» при непустом бакете) —
вероятен **дрейф схемы**: `cloud_function/requirements.txt` пинит `chromadb>=0.5.0` (плавающий),
а бот — hash-lock `chromadb==0.6.3`. Функция могла записать формат новее, который читатель видит
как пустую коллекцию. Запиньте функцию под минор бота:
```
# cloud_function/requirements.txt
chromadb==0.6.*
```

---

## 9. Быстрый путь в обход функции (админ-CLI) — рекомендуется для проверки

Полностью минует INBOX + триггер: ингестит локально и публикует ChromaDB **прямо в STORE**,
который читает бот. Дальше — рестарт.
```bash
pip install pymupdf4llm chromadb google-cloud-storage
export CHROMA_BUCKET=ramp-bot-chroma-db-investadv     # STORE-бакет (куда смотрит бот)
export GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json    # SA с доступом к STORE-бакету

python scripts/ingest_bank_report.py reports/*.pdf --upload
python scripts/ingest_bank_report.py --list            # проверить: банк · дата · N chunks (ЛОКАЛЬНО)

# затем форс-рестарт бота (шаг 6) и перегенерация отчёта
```
Если `--upload` записал файлы, а `gcloud storage ls` (шаг 5) их показывает — проблема была в
функции/триггере (шаги 3–4), а не в чтении.

---

## 10. Как понять, что починилось (критерии приёмки в отчёте)

После наполнения STORE **и рестарта** бота, в DEEP:
- CoVe-строка **`✓ Bank RAG (выдержки)`** → `прочитано N отрывков · база: M отчётов · K чанков`.
- CoVe-строка **`ИИ-цитирование банк-аналитики`** переходит из `⚠ …из знаний модели` в
  `✓ …проверенных [RAG]-цитат из отчётов` (когда ИИ цитирует реально извлечённые записки).
- Футер **`✓ RAG: банк. отчёты`**, а в дисклеймере-источниках появляется `ChromaDB (GS/MS/JPM)`.

---

## Сводка причин (по частоте)

| # | Симптом | Причина | Где чинить |
|---|---|---|---|
| 1 | PDF залиты, триггера нет | залито НЕ в INBOX-бакет / рассинхрон имён после миграции | §1, §2, §9 |
| 2 | STORE непуст, отчёт пуст | бот не рестартовали (синк только на буте) | §6 |
| 3 | триггера нет | функция не задеплоена (fail-soft в cloudbuild проглотил 403) | §3 |
| 4 | `Added 0 chunks` | PDF-скан без текста / нет pymupdf4llm | §4 |
| 5 | функция пишет, бот не видит | разные `CHROMA_BUCKET` у бота и функции | §7 |
| 6 | STORE непуст, бот «is empty» | дрейф версии chromadb функция↔бот | §8 |

**Файлы пайплайна:** `cloud_function/main.py` (ингест по триггеру), `src/entrypoint.py`
(синк STORE→контейнер на буте), `src/agent/rag_engine.py` (движок), `scripts/ingest_bank_report.py`
(админ-CLI), `cloudbuild.yaml` (деплой бота + функции), `RAG_INGESTION.md` (загрузка записок).
