# RAG_TROUBLESHOOTING.md — почему «Bank RAG» пуст, хотя PDF загружены в GCS
<!-- nav | area:rag | code:src/entrypoint.py,src/agent/rag_engine.py | read-before:runbook «RAG пуст, хотя PDF загружены» -->

Пошаговый runbook: PDF банков лежат в Google Cloud Storage, но отчёт всё равно показывает
`✗ RAG: банк. отчёты` / `Bank RAG (выдержки) · база: 0 отчётов`. Ниже — как за 8 шагов найти
точку обрыва. Выполняйте сверху вниз, первый «красный» шаг = причина.

> ## ⭐ Самый надёжный путь (2026-07-04; инкрементальный с 2026-07-09): boot-ingest бота
> Cloud Function → Eventarc — хрупкая цепочка (регион бакета, скачивание embedding-модели,
> fail-soft деплой). Поэтому бот **сам ингестит PDF на буте** прямо в контейнере, где уже есть
> зависимости, «запечённая» embedding-модель и запиненный chromadb, после чего публикует базу
> обратно в STORE. Это **обходит Cloud Function целиком**.
>
> **Инкрементально (2026-07-09, замечание #4 «счётчик базы не обновился после 8 новых отчётов»):**
> раньше boot-ingest срабатывал ТОЛЬКО когда STORE пуст — если в STORE уже были старые отчёты,
> он пропускал новые PDF из INBOX, и `Bank RAG · база: N отчётов` не рос. Теперь он сравнивает
> **имена файлов INBOX против уже ингестированных `source`** и ингестит **только недостающие**,
> даже при непустом STORE. Плюс `_download_chroma_db` зеркалит STORE на КАЖДОМ буте.
>
> **Чистка tmp-артефактов (2026-07-09, замечание R2#6 «48 отчётов вместо 29»):** до фикса 07-05
> `source` мог стать tmp-именем (`tmpXXXX.pdf`), а повторный ингест под настоящим именем оставлял
> старый tmp-«отчёт» в базе → счётчик задваивался. Теперь `list_documents()` их **не считает**
> (`_is_temp_source`), а boot-ingest **физически удаляет** (`purge_temp_sources`) и републикует базу.
> Счётчик читается вживую из РЕАЛЬНЫХ отчётов (`_fetch_rag_context`: docs = `len(list_documents())`,
> chunks = сумма их чанков), поэтому после ингеста/чистки цифра корректна на след. рестарте.
>
> Чтобы им воспользоваться: (1) положите PDF в INBOX
> `gcloud storage cp reports/*.pdf gs://ramp-bot-chroma-db-inbox-investadv/`; (2) **перезапустите
> бот** (новая ревизия Cloud Run — см. §6); (3) в бут-логах будет
> `RAG boot-ingest: store has N chunks · K of M INBOX PDF(s) missing — ingesting in-container` →
> `RAG boot-ingest: ingested K chunks from K new PDF(s)` → `published … blob(s) → …-investadv/…`.
> Выключатель: `RAG_BOOT_INGEST=0`. Бакет INBOX переопределяется `RAG_INBOX_BUCKET`.
> Если и это не сработало — идите по шагам ниже (права SA на бакеты, имена, регион).

---

## 0. Как устроен пайплайн (это важно для диагностики)

Пайплайн **двухступенчатый** и использует **ТРИ бакета** (имена — после миграции `-investadv`,
см. §−N в AUDIT / `RAG_INGESTION.md`):

```
(1) PDF кладут сюда  ─►  gs://ramp-bot-chroma-db-inbox-investadv/     ← INBOX (триггер-бакет)
                          │                              │
        путь A (Cloud Function, опц.)        путь B (bot boot-ingest, надёжный, 07-04)
                          │  Eventarc → «ramp-bot-rag-ingest»          │  entrypoint._boot_ingest_from_inbox
                          ▼                                            │  (если STORE пуст, а в INBOX есть PDF —
        Функция: pymupdf4llm → ingest → ChromaDB                      │   бот ингестит В КОНТЕЙНЕРЕ и публикует)
                          │                                            │
                          ▼                                            ▼
(2) ChromaDB хранится тут ─►  gs://ramp-bot-chroma-db-investadv/chroma_db/   ← STORE
                              │  бот НА БУТЕ: entrypoint._download_chroma_db()
                              ▼
(3) /app/data/chroma_db  (внутри работающего Cloud Run контейнера)
                              │  на запросе: tg_bot._fetch_rag_context → get_market_sentiment
                              ▼
              AI-нарратив + чекер CoVe «Bank RAG (выдержки)»
```

Путь **B** (boot-ingest бота) не зависит от Cloud Function/Eventarc и региона — рекомендуется как
основной. Путь **A** остаётся как авто-триггер для «онлайн-обновлений».

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
- **🌍 РЕГИОН (частая причина для europe-west3).** Eventarc-триггер на бакет должен быть в ТОМ ЖЕ
  регионе, что и бакет. Ваши бакеты — `europe-west3` (Frankfurt), а деплой в коде/этих командах —
  `us-central1`. Функция в `us-central1` **не получит** события бакета из `europe-west3`. Деплойте
  функцию и триггер в **`europe-west3`** (замените `--region=us-central1` → `--region=europe-west3`
  во всех командах ниже), либо просто используйте **путь B** (boot-ingest бота) — ему регион
  безразличен.
- Функции нет / триггер на старом бакете → **причина.** Важно: в `cloudbuild.yaml` шаг деплоя
  функции **fail-soft** (`|| echo "WARN…"; exit 0`) — при 403/нехватке прав он молча
  пропускается, и функция может быть НЕ задеплоена или устаревшей. Передеплойте вручную
  (в правильном регионе):
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
| 0 | **STORE пуст, PDF в INBOX, функция не пишет** | регион функции ≠ регион бакета (europe-west3) / embedding-модель не скачалась в функции / устаревшая копия `rag_engine` в функции | **путь B (boot-ingest бота, §⭐)** — надёжно; или §3 (регион) |
| 1 | PDF залиты, триггера нет | залито НЕ в INBOX-бакет / рассинхрон имён после миграции | §1, §2, §9, путь B |
| 2 | STORE непуст, отчёт пуст | бот не рестартовали (синк только на буте) | §6 |
| 3 | триггера нет | функция не задеплоена (fail-soft в cloudbuild проглотил 403) | §3 |
| 4 | `Added 0 chunks` | PDF-скан без текста / нет pymupdf4llm | §4 |
| 5 | функция пишет, бот не видит | разные `CHROMA_BUCKET` у бота и функции | §7 |
| 6 | STORE непуст, бот «is empty» | дрейф версии chromadb функция↔бот (запинена `==0.6.3`, 07-04) | §8 |

**Файлы пайплайна:** `src/entrypoint.py` (синк STORE→контейнер + **boot-ingest fallback** путь B),
`cloud_function/main.py` (ингест по триггеру, путь A), `src/agent/rag_engine.py` (движок; копия
функции `cloud_function/rag_engine.py` синхронизирована 07-04), `scripts/ingest_bank_report.py`
(админ-CLI), `cloudbuild.yaml` (деплой бота + функции), `RAG_INGESTION.md` (загрузка записок).
