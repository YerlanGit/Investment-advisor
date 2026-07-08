#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# setup_static_egress.sh — статический исходящий IP для Cloud Run `ramp-bot`
#
# ЗАЧЕМ.  Cloudflare-WAF брокера (Freedom/Tradernet) отдаёт HTTP 403 + HTML-
# challenge на запросы с ОБЩЕГО пула egress-IP Cloud Run (getHloc /
# getQuotesHistory → «Tradernet вернул пустой набор цен»).  Блокировка по IP,
# а не по заголовкам (клиент уже шлёт browser-User-Agent + ретраит).  Решение —
# прогнать весь исходящий трафик через Serverless VPC connector → Cloud Router →
# Cloud NAT с ЗАРЕЗЕРВИРОВАННЫМ статическим IP, который WAF не челленджит.
#
# Идемпотентно: повторный запуск не пересоздаёт уже существующие ресурсы.
# После успеха печатает статический IP (занесите его в вайтлист у брокера,
# если такая опция есть) и выставляет подстановку _VPC_CONNECTOR для CI.
#
# Использование:
#   chmod +x scripts/setup_static_egress.sh
#   ./scripts/setup_static_egress.sh
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Конфигурация ──────────────────────────────────────────────────────────────
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-ramp-bot}"
NETWORK="${NETWORK:-ramp-egress-net}"
SUBNET="${SUBNET:-ramp-egress-subnet}"
SUBNET_RANGE="${SUBNET_RANGE:-10.8.0.0/28}"          # /28 — минимум для connector
CONNECTOR="${CONNECTOR:-ramp-egress-conn}"
ROUTER="${ROUTER:-ramp-egress-router}"
NAT="${NAT:-ramp-egress-nat}"
STATIC_IP_NAME="${STATIC_IP_NAME:-ramp-egress-ip}"
# ──────────────────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
die()  { echo -e "${RED}❌ $*${NC}"; exit 1; }

command -v gcloud >/dev/null || die "gcloud CLI не найден."
PROJECT_ID=$(gcloud config get-value project 2>/dev/null | tr -d '[:space:]')
[ -n "$PROJECT_ID" ] && [ "$PROJECT_ID" != "(unset)" ] || die "GCP проект не выбран: gcloud config set project <ID>"
info "Проект: $PROJECT_ID · Регион: $REGION · Сервис: $SERVICE"

# helper: существует ли ресурс (по коду возврата describe)
exists() { eval "$1" >/dev/null 2>&1; }

# ── 0. Включить нужные API ────────────────────────────────────────────────────
gcloud services enable vpcaccess.googleapis.com compute.googleapis.com \
  --project="$PROJECT_ID" --quiet

# ── 1. VPC-сеть + подсеть ─────────────────────────────────────────────────────
if ! exists "gcloud compute networks describe $NETWORK --project=$PROJECT_ID"; then
  gcloud compute networks create "$NETWORK" --subnet-mode=custom --project="$PROJECT_ID"
  info "VPC-сеть $NETWORK создана"
else warn "VPC-сеть $NETWORK уже есть"; fi

if ! exists "gcloud compute networks subnets describe $SUBNET --region=$REGION --project=$PROJECT_ID"; then
  gcloud compute networks subnets create "$SUBNET" \
    --network="$NETWORK" --region="$REGION" --range="$SUBNET_RANGE" --project="$PROJECT_ID"
  info "Подсеть $SUBNET ($SUBNET_RANGE) создана"
else warn "Подсеть $SUBNET уже есть"; fi

# ── 2. Serverless VPC Access connector ───────────────────────────────────────
if ! exists "gcloud compute networks vpc-access connectors describe $CONNECTOR --region=$REGION --project=$PROJECT_ID"; then
  gcloud compute networks vpc-access connectors create "$CONNECTOR" \
    --region="$REGION" --subnet="$SUBNET" --subnet-project="$PROJECT_ID" \
    --min-instances=2 --max-instances=3 --machine-type=e2-micro --project="$PROJECT_ID"
  info "VPC-коннектор $CONNECTOR создан"
else warn "VPC-коннектор $CONNECTOR уже есть"; fi

# ── 3. Статический внешний IP ─────────────────────────────────────────────────
if ! exists "gcloud compute addresses describe $STATIC_IP_NAME --region=$REGION --project=$PROJECT_ID"; then
  gcloud compute addresses create "$STATIC_IP_NAME" --region="$REGION" --project="$PROJECT_ID"
  info "Статический IP $STATIC_IP_NAME зарезервирован"
else warn "Статический IP $STATIC_IP_NAME уже есть"; fi
STATIC_IP=$(gcloud compute addresses describe "$STATIC_IP_NAME" --region="$REGION" \
  --project="$PROJECT_ID" --format='value(address)')

# ── 4. Cloud Router + Cloud NAT (привязка статического IP) ────────────────────
if ! exists "gcloud compute routers describe $ROUTER --region=$REGION --project=$PROJECT_ID"; then
  gcloud compute routers create "$ROUTER" --network="$NETWORK" --region="$REGION" --project="$PROJECT_ID"
  info "Cloud Router $ROUTER создан"
else warn "Cloud Router $ROUTER уже есть"; fi

if ! exists "gcloud compute routers nats describe $NAT --router=$ROUTER --region=$REGION --project=$PROJECT_ID"; then
  gcloud compute routers nats create "$NAT" \
    --router="$ROUTER" --region="$REGION" \
    --nat-custom-subnet-ip-ranges="$SUBNET" \
    --nat-external-ip-pool="$STATIC_IP_NAME" --project="$PROJECT_ID"
  info "Cloud NAT $NAT (external IP $STATIC_IP) создан"
else warn "Cloud NAT $NAT уже есть"; fi

# ── 5. Привязать коннектор к Cloud Run + весь egress через VPC ────────────────
gcloud run services update "$SERVICE" \
  --region="$REGION" --project="$PROJECT_ID" \
  --vpc-connector="$CONNECTOR" --vpc-egress=all-traffic --quiet
info "Сервис $SERVICE переключён на статический egress"

echo ""
info "ГОТОВО.  Статический исходящий IP бота: ${GREEN}${STATIC_IP}${NC}"
echo "   • Если у брокера есть вайтлист IP — занесите ${STATIC_IP}."
echo "   • Чтобы CI сохранял привязку при каждом деплое, задайте подстановку в"
echo "     Build Trigger:  _VPC_CONNECTOR=${CONNECTOR}"
echo "     (или разово:  gcloud builds submit --substitutions=_VPC_CONNECTOR=${CONNECTOR},...)"
warn "Проверка: отправьте боту /start → отчёт; в логах не должно быть Cloudflare 403."
