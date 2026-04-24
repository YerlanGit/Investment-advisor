#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# deploy.sh — одна команда для деплоя RAMP Bot на Google Cloud Run
#
# Использование:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Предварительные требования:
#   1. gcloud CLI установлен и авторизован:
#        gcloud auth login
#        gcloud auth application-default login
#   2. GCP проект выбран:
#        gcloud config set project YOUR_PROJECT_ID
#   3. Секреты созданы в Secret Manager (выполнить один раз):
#        ./deploy.sh --setup-secrets
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Конфигурация (измените под свой проект) ───────────────────────────────────
SERVICE_NAME="ramp-bot"
REGION="us-central1"
REPO_NAME="ramp"
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}✅ $*${NC}"; }
warn()  { echo -e "${YELLOW}⚠️  $*${NC}"; }
error() { echo -e "${RED}❌ $*${NC}"; exit 1; }

# ── 1. Проверка gcloud CLI ────────────────────────────────────────────────────
if ! command -v gcloud &>/dev/null; then
    error "gcloud CLI не найден.\n   Установите: https://cloud.google.com/sdk/docs/install"
fi

# ── 2. Проверка авторизации ───────────────────────────────────────────────────
if ! gcloud auth print-access-token &>/dev/null 2>&1; then
    error "Не авторизован в gcloud.\n\
   Выполните:\n\
     gcloud auth login\n\
     gcloud auth application-default login"
fi

# ── 3. Проверка выбранного проекта ────────────────────────────────────────────
PROJECT_ID=$(gcloud config get-value project 2>/dev/null | tr -d '[:space:]')
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    error "GCP проект не выбран.\n   Выполните: gcloud config set project YOUR_PROJECT_ID"
fi

info "Проект : $PROJECT_ID"
info "Регион : $REGION"
info "Сервис : $SERVICE_NAME"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}"
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")

# ── Режим: первичная настройка секретов (./deploy.sh --setup-secrets) ─────────
if [ "${1:-}" = "--setup-secrets" ]; then
    warn "Режим настройки секретов. Нужен файл .env в корне проекта."
    [ -f .env ] || error "Файл .env не найден."

    gcloud services enable secretmanager.googleapis.com --project="$PROJECT_ID"

    load_secret() {
        local name="$1"
        local value
        value=$(grep "^${name}=" .env | cut -d'=' -f2- | tr -d '"' | tr -d "'")
        [ -z "$value" ] && { warn "  $name не найден в .env, пропускаю."; return; }

        if gcloud secrets describe "$name" --project="$PROJECT_ID" &>/dev/null; then
            echo -n "$value" | gcloud secrets versions add "$name" \
                --data-file=- --project="$PROJECT_ID"
            info "  $name — обновлён"
        else
            echo -n "$value" | gcloud secrets create "$name" \
                --data-file=- --replication-policy=automatic --project="$PROJECT_ID"
            info "  $name — создан"
        fi
    }

    load_secret "RAMP_BOT_TOKEN"
    load_secret "FINTECH_MASTER_KEY"
    load_secret "ANTHROPIC_API_KEY"

    # Разрешить Cloud Run SA читать секреты
    SA_EMAIL="${PROJECT_ID}@appspot.gserviceaccount.com"
    for SECRET in RAMP_BOT_TOKEN FINTECH_MASTER_KEY ANTHROPIC_API_KEY; do
        gcloud secrets add-iam-policy-binding "$SECRET" \
            --member="serviceAccount:${SA_EMAIL}" \
            --role="roles/secretmanager.secretAccessor" \
            --project="$PROJECT_ID" \
            --condition=None &>/dev/null || true
    done

    info "Секреты настроены. Запустите ./deploy.sh для деплоя."
    exit 0
fi

# ── 4. Включение необходимых API ─────────────────────────────────────────────
gcloud services enable \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com \
    --project="$PROJECT_ID" \
    --quiet

# ── 5. Создание Artifact Registry репозитория (если не существует) ────────────
if ! gcloud artifacts repositories describe "$REPO_NAME" \
        --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --project="$PROJECT_ID"
    info "Artifact Registry репозиторий '$REPO_NAME' создан"
fi

# ── 6. Сборка и деплой через Cloud Build ──────────────────────────────────────
info "Запускаю Cloud Build (sha: $GIT_SHA)..."
gcloud builds submit \
    --config=cloudbuild.yaml \
    --substitutions="_REGION=${REGION},_REPO=${REPO_NAME},_SERVICE=${SERVICE_NAME},SHORT_SHA=${GIT_SHA}" \
    --project="$PROJECT_ID" \
    .

# ── 7. URL сервиса ────────────────────────────────────────────────────────────
echo ""
info "Деплой завершён!"
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --format="value(status.url)" 2>/dev/null || echo "(URL недоступен)")
echo -e "   URL сервиса: ${GREEN}${SERVICE_URL}${NC}"
echo ""
warn "Помните: бот использует polling, а не webhooks."
warn "Убедитесь, что min-instances=1 чтобы бот не засыпал."
