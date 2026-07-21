# INFRA_NETWORKING.md — прод-инциденты Cloud Run: WAF-блок цен + segfault RAG-ингеста
<!-- nav | area:infra | code:cloud_function/,src/entrypoint.py | read-before:прод-инциденты Cloud Run, сеть, egress -->

> Диагностика и фиксы двух прод-ошибок (2026-07-08). Обе — инфраструктурные
> (лимиты/сеть Cloud Run), а не баги логики; код-путь уже деградирует мягко.
> Соответствующие правки закоммичены в `cloudbuild.yaml`, `cloud_function/main.py`
> и `scripts/setup_static_egress.sh`.

---

## 1. `ramp-bot` — Cloudflare-WAF блокирует котировки (пустой набор цен)

**Симптом (логи):**
```
WARNING freedom_portfolio.client — Cloudflare WAF заблокировал запрос к
        https://freedombroker.kz/api/v2/cmd/getHloc (HTTP 403, HTML challenge).
        Вероятно, IP-адрес Cloud Run попал в фильтр.
ERROR   ...RiskEngine — Tradernet вернул пустой набор цен — все тикеры провалились
```

**Причина.** Cloudflare-WAF брокера отдаёт `403 + HTML challenge` на запросы с
**общего пула egress-IP Cloud Run**. Это блокировка **по IP-диапазону**, а не по
заголовкам: клиент (`freedom_portfolio/client.py`) уже шлёт browser-`User-Agent`,
детектит challenge (`_is_cloudflare_block`), ретраит с backoff (`_with_cf_retry`)
и ротирует домены (`freedombroker.kz` → `tradernet.com`). Значит хедеры/ретраи не
помогут — нужен **стабильный статический исходящий IP**, которого нет у общего пула.

**Фикс — статический egress через Serverless VPC + Cloud NAT.** Один запуск:
```bash
./scripts/setup_static_egress.sh
```
Скрипт идемпотентно создаёт VPC-сеть + подсеть, Serverless VPC-коннектор,
зарезервированный статический IP, Cloud Router + Cloud NAT (с этим IP) и
переключает `ramp-bot` на `--vpc-egress=all-traffic`. В конце печатает
статический IP (занесите в вайтлист брокера, если есть такая опция).

**Чтобы CI сохранял привязку** при каждом деплое — задайте подстановку
`_VPC_CONNECTOR` в Build Trigger (имя коннектора, напр. `ramp-egress-conn`).
`cloudbuild.yaml` шаг `configure-egress` тогда применяет коннектор после деплоя;
при пустом `_VPC_CONNECTOR` шаг — no-op (текущее поведение не меняется).

**Ручной эквивалент** (если не хотите скрипт):
```bash
REGION=us-central1
gcloud compute networks create ramp-egress-net --subnet-mode=custom
gcloud compute networks subnets create ramp-egress-subnet \
  --network=ramp-egress-net --region=$REGION --range=10.8.0.0/28
gcloud compute networks vpc-access connectors create ramp-egress-conn \
  --region=$REGION --subnet=ramp-egress-subnet --min-instances=2 --max-instances=3 --machine-type=e2-micro
gcloud compute addresses create ramp-egress-ip --region=$REGION
gcloud compute routers create ramp-egress-router --network=ramp-egress-net --region=$REGION
gcloud compute routers nats create ramp-egress-nat --router=ramp-egress-router --region=$REGION \
  --nat-custom-subnet-ip-ranges=ramp-egress-subnet --nat-external-ip-pool=ramp-egress-ip
gcloud run services update ramp-bot --region=$REGION \
  --vpc-connector=ramp-egress-conn --vpc-egress=all-traffic
```

**Проверка.** `/start` → отчёт формируется; в логах нет `Cloudflare WAF … 403`.
Если WAF всё ещё челленджит новый IP — попросите брокера вайтлистнуть его, либо
поднимите NAT в другом регионе/подсети.

---

## 2. `ingest-pdf-trigger` — Segmentation Fault (signal 11) на ONNX-модели

**Симптом (логи):**
```
[RAG] Парсинг: tmp….pdf
…/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz: 100%|██████████| 79.3M
Uncaught signal: 11, pid=5, tid=29, fault_addr=0.
The request failed because … connection to the instance had an error.
```

**Причина.** `SIGSEGV` возникает ровно на загрузке ONNX-модели эмбеддингов
`all-MiniLM-L6-v2` (79 МБ) внутри `chromadb.DefaultEmbeddingFunction` при
векторизации PDF. На дефолтных **1 GiB / 1 CPU** контейнеру не хватает памяти
(модель + `onnxruntime` + парсинг PDF + запись ChromaDB), а `onnxruntime`
переподписывает потоки CPU под давлением памяти → OOM-kill/segfault.

**Фикс — память/CPU + ограничение потоков.** Уже внесено в:
- `cloudbuild.yaml` (шаг `deploy-rag-function`, функция `ramp-bot-rag-ingest`);
- docstring `cloud_function/main.py` (ручной деплой `ingest-pdf-trigger`).

Параметры: `--memory 4Gi --cpu 2 --timeout 540s` +
`OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, TOKENIZERS_PARALLELISM=false`.

**Починить УЖЕ работающий сервис БЕЗ передеплоя кода** (gen2-функция = Cloud Run
сервис под капотом, лимиты применяются моментально):
```bash
gcloud run services update ingest-pdf-trigger --region=us-central1 \
  --memory=4Gi --cpu=2 --timeout=540 \
  --set-env-vars=OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,TOKENIZERS_PARALLELISM=false
```
(Если у вас CI-функция `ramp-bot-rag-ingest` — тот же `run services update` по её
имени, либо просто дождитесь следующего Cloud Build — правка уже в шаге 5.)

**Проверка.** Загрузите тестовый PDF в `gs://ramp-bot-chroma-db-inbox-investadv/`;
в логах функция должна дойти до `ChromaDB синхронизирован` без `signal 11`.

**Примечание про бота.** У бота есть резервный in-container boot-ingest
(`entrypoint._boot_ingest_from_inbox`) на 2 GiB — он использует **пред-запечённую**
ONNX-модель (без сетевой докачки), поэтому там segfault не воспроизводился. Данный
инцидент — именно у отдельной GCS-триггерной функции, качающей модель на лету.

---

## Сводка правок в репозитории
| Файл | Правка |
|---|---|
| `cloudbuild.yaml` | шаг 5: RAG-функция 1Gi→**4Gi + 2CPU + 540s + thread-env**; новый шаг `configure-egress` (опц. VPC-коннектор через `_VPC_CONNECTOR`) |
| `cloud_function/main.py` | docstring-команда деплоя обновлена (4Gi/2CPU) + прямая команда фикса работающего сервиса |
| `scripts/setup_static_egress.sh` | one-time настройка статического egress (VPC/NAT/IP) |
