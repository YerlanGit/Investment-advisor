#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# setup_ingest_scheduler.sh — плановая проверка базы котировок (IB-7)
#
# ЗАЧЕМ.  Маршрут `POST /tasks/check` в загрузчике готов с IB-7, а расписания
# нет — и напоминание не приходит НИКОГДА.  Цепочка «база не протухла» держится
# на памяти оператора, а первым о протухании узнаёт ПОЛЬЗОВАТЕЛЬ: отказом
# ручного тира по свежести (`stooq_provider.MAX_MARKET_STALE_DAYS`).
# Здесь заводится Cloud Scheduler, который эту ручку дёргает.
#
# 🔴 МОЛЧАНИЕ — ШТАТНЫЙ ИСХОД.  `build_reminder` возвращает `None`, когда всё
# хорошо, и задание отработает вхолостую.  Так и задумано: сообщение, которое
# приходит каждый день независимо от состояния, перестают читать за неделю.
#
# Идемпотентно: повторный запуск обновляет существующее задание, а не плодит
# второе.  Секрет НИКОГДА не печатается в вывод.
#
# Использование:
#   chmod +x scripts/setup_ingest_scheduler.sh
#   ./scripts/setup_ingest_scheduler.sh
#
# Переопределяется через окружение, например:
#   SCHEDULE='0 7 * * *' TIME_ZONE=Europe/Moscow ./scripts/setup_ingest_scheduler.sh
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Конфигурация ──────────────────────────────────────────────────────────────
REGION="${REGION:-us-central1}"
INGEST_SERVICE="${INGEST_SERVICE:-ramp-ingest-bot}"
JOB="${JOB:-ramp-ingest-check}"
# 🔴 Расписание ЕЖЕДНЕВНОЕ, а не по будням.  Порог свежести считается в
# КАЛЕНДАРНЫХ днях (`stooq_store.market_staleness_days`), поэтому за выходные
# запас тает ровно так же, как в будни — а забывают файл чаще всего в пятницу.
# Ежедневный запуск ничего не стоит: при здоровой базе он молчит.
SCHEDULE="${SCHEDULE:-0 9 * * *}"
TIME_ZONE="${TIME_ZONE:-Asia/Almaty}"
TASK_SECRET="${TASK_SECRET:-INGEST_TASK_TOKEN}"
SCHEDULER_SA_NAME="${SCHEDULER_SA_NAME:-ramp-scheduler}"
# Дедлайн щедрый: первая проверка после нового поколения базы скачивает ~54 МБ
# из GCS (снапшот-кэш `§−105` греется только со второго раза).
ATTEMPT_DEADLINE="${ATTEMPT_DEADLINE:-600s}"
# 🔴 Ретраев МАЛО осознанно.  Маршрут отвечает 200 и на внутренней ошибке —
# потому что об ошибке оператор УЖЕ извещён в Telegram самим `run_scheduled_check`.
# Значит ретрай случается только при сетевом сбое, а лишний ретрай после уже
# доставленного сообщения = дубль на телефоне.  Пропущенная проверка ничего не
# стоит: следующая придёт завтра, а запас до блокировки — 3 дня.
MAX_RETRIES="${MAX_RETRIES:-1}"
# ──────────────────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
die()  { echo -e "${RED}❌ $*${NC}"; exit 1; }

command -v gcloud >/dev/null || die "gcloud CLI не найден."
PROJECT_ID=$(gcloud config get-value project 2>/dev/null | tr -d '[:space:]')
[ -n "$PROJECT_ID" ] && [ "$PROJECT_ID" != "(unset)" ] \
  || die "GCP проект не выбран: gcloud config set project <ID>"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
info "Проект: $PROJECT_ID · Регион: $REGION · Сервис: $INGEST_SERVICE"

exists() { eval "$1" >/dev/null 2>&1; }

# ── 0. Включить API ───────────────────────────────────────────────────────────
gcloud services enable cloudscheduler.googleapis.com \
  --project="$PROJECT_ID" --quiet

# ── 1. Сервис должен быть задеплоен: у задания должен быть адрес ──────────────
SERVICE_URL=$(gcloud run services describe "$INGEST_SERVICE" \
  --region="$REGION" --project="$PROJECT_ID" \
  --format='value(status.url)' 2>/dev/null || true)
[ -n "$SERVICE_URL" ] || die "Сервис $INGEST_SERVICE в регионе $REGION не найден.
   Сначала задеплойте загрузчик (подстановка _INGEST_SERVICE в Build Trigger),
   потом заводите расписание: у задания должен быть адрес, куда стучать."
info "Адрес сервиса: $SERVICE_URL"

# ── 2. Секрет маршрута ────────────────────────────────────────────────────────
# 🔴 Без секрета маршрут ОТКЛЮЧЁН (отвечает 503) — так задумано: незакрытая
# ручка, шлющая сообщения владельцу, это канал для чужого шума в единственном
# канале связи оператора.  Значение читается, но НЕ ПЕЧАТАЕТСЯ.
TASK_TOKEN=$(gcloud secrets versions access latest \
  --secret="$TASK_SECRET" --project="$PROJECT_ID" 2>/dev/null || true)
[ -n "$TASK_TOKEN" ] || die "Секрет $TASK_SECRET пуст или недоступен.
   Маршрут /tasks/check без него ОТКЛЮЧЁН, и расписание будет бить в 503.
   Завести:
     openssl rand -hex 32 | gcloud secrets create $TASK_SECRET --data-file=-
   и пересобрать сервис, чтобы секрет доехал в контейнер."
case "$TASK_TOKEN" in
  *,*) die "Секрет $TASK_SECRET содержит запятую.
   gcloud разбирает --headers как список пар через запятую, и такой токен
   доедет обрезанным — маршрут будет отвечать 401. Смените значение секрета
   на безопасный алфавит (например openssl rand -hex 32).";;
esac

# ── 3. Сервис-аккаунт для OIDC ───────────────────────────────────────────────
# ОТДЕЛЬНЫЙ от загрузчика: у того есть право ПИСАТЬ базу, а вызывающему нужно
# ровно одно — постучаться в сервис.  Смешивать личности незачем.
SCHEDULER_SA="${SCHEDULER_SA:-${SCHEDULER_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com}"
if ! exists "gcloud iam service-accounts describe $SCHEDULER_SA --project=$PROJECT_ID"; then
  gcloud iam service-accounts create "$SCHEDULER_SA_NAME" \
    --display-name="Cloud Scheduler → ingest bot" --project="$PROJECT_ID"
  info "Сервис-аккаунт $SCHEDULER_SA создан"
else warn "Сервис-аккаунт $SCHEDULER_SA уже есть"; fi

# Право вызвать — НА СЕРВИСЕ, а не на проекте: сервис задеплоен
# `--no-allow-unauthenticated`, и без этой привязки каждый запуск получит 403.
gcloud run services add-iam-policy-binding "$INGEST_SERVICE" \
  --region="$REGION" --project="$PROJECT_ID" \
  --member="serviceAccount:${SCHEDULER_SA}" \
  --role=roles/run.invoker --quiet >/dev/null
info "roles/run.invoker выдан на сервис $INGEST_SERVICE (не на проект)"

# Планировщику нужно ВЫПУСТИТЬ OIDC-токен от имени этого SA.  Привязка узкая —
# на один сервис-аккаунт, а не на проект.  Идемпотентна; если агент уже имеет
# право через `cloudscheduler.serviceAgent`, повтор ничего не меняет.
SCHEDULER_AGENT="service-${PROJECT_NUMBER}@gcp-sa-cloudscheduler.iam.gserviceaccount.com"
gcloud iam service-accounts add-iam-policy-binding "$SCHEDULER_SA" \
  --project="$PROJECT_ID" \
  --member="serviceAccount:${SCHEDULER_AGENT}" \
  --role=roles/iam.serviceAccountTokenCreator --quiet >/dev/null \
  || warn "Не удалось выдать tokenCreator агенту планировщика — если задание
   упадёт с PERMISSION_DENIED на выпуске токена, выдайте эту роль вручную."

# ── 4. Задание ───────────────────────────────────────────────────────────────
# 🔴 Путь и имя заголовка — НЕ литералы «на всякий случай»: они обязаны
# совпадать с `ingest_entrypoint.TASK_PATH` и `TASK_TOKEN_HEADER`.  Равенство
# закреплено тестом `SchedulerScriptMatchesTheRouteTest`: переименуют маршрут —
# упадёт сюита, а не тихо перестанут приходить напоминания.
TARGET_URI="${SERVICE_URL}/tasks/check"
HEADER_NAME="x-ingest-task-token"

COMMON_ARGS=(
  --location="$REGION"
  --project="$PROJECT_ID"
  --schedule="$SCHEDULE"
  --time-zone="$TIME_ZONE"
  --uri="$TARGET_URI"
  --http-method=POST
  --headers="${HEADER_NAME}=${TASK_TOKEN}"
  --oidc-service-account-email="$SCHEDULER_SA"
  --oidc-token-audience="$SERVICE_URL"
  --attempt-deadline="$ATTEMPT_DEADLINE"
  --max-retry-attempts="$MAX_RETRIES"
)

if exists "gcloud scheduler jobs describe $JOB --location=$REGION --project=$PROJECT_ID"; then
  gcloud scheduler jobs update http "$JOB" "${COMMON_ARGS[@]}" --quiet >/dev/null
  info "Задание $JOB обновлено"
else
  gcloud scheduler jobs create http "$JOB" "${COMMON_ARGS[@]}" \
    --description="Плановая проверка свежести базы котировок (IB-7)" \
    --quiet >/dev/null
  info "Задание $JOB создано"
fi

echo ""
info "ГОТОВО.  Расписание: ${GREEN}${SCHEDULE}${NC} (${TIME_ZONE})"
echo "   Цель:  POST ${TARGET_URI}"
echo "   Вызов: OIDC от ${SCHEDULER_SA}"
echo ""
warn "Секрет маршрута ТЕПЕРЬ ЛЕЖИТ В ДВУХ МЕСТАХ: Secret Manager и конфиг
   задания.  🔴 \`jobs describe\` и \`jobs create\` печатают его ОТКРЫТЫМ —
   не вставляйте их вывод целиком в переписку, тикеты и скриншоты.
   Ротация: новая версия секрета → \`gcloud run services update … --update-secrets\`
   (НЕ --set-secrets: он заменяет ВЕСЬ набор и снесёт токен бота) → этот скрипт."
echo ""
echo "Проверка — разовый прогон прямо сейчас:"
echo "   gcloud scheduler jobs run $JOB --location=$REGION"
echo ""
echo "🔴 НЕ проверяйте по полю status: при живом задании там стоит code:-1,"
echo "   потому что \`-1\` означает «завершённой попытки не записано», а не отказ."
echo "   Приёмка — ЗАПРОС В ЛОГАХ СЕРВИСА с User-Agent планировщика:"
echo "     gcloud logging read \\"
echo "       'resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$INGEST_SERVICE\" AND httpRequest.requestUrl:\"tasks/check\"' \\"
echo "       --limit=10 --freshness=2h --format='value(timestamp,httpRequest.status,httpRequest.userAgent)'"
echo "   Нужна строка со статусом 200 и User-Agent Google-Cloud-Scheduler:"
echo "   ручной curl доказывает исправность МАРШРУТА и молчит о звене"
echo "   «планировщик → сервис», где свои OIDC, аудитория и права."
echo ""
echo "🔴 В Telegram при ЗДОРОВОЙ базе не придёт ничего — молчание штатный исход."
echo "   Второй конец той же проверки — логи бота:"
echo "   gcloud run services logs read $INGEST_SERVICE --region=$REGION --limit=30"
echo "   ищите строку «плановая проверка: всё в порядке, молчу»."
