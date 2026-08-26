#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# verify_ingest_iam.sh — узок ли радиус писателя базы котировок (IB-F2)
#
# ЗАЧЕМ.  Загрузчик пишет в ТОТ ЖЕ бакет, где лежат `tokenomics.db` (балансы) и
# `users_vault.db` (ключи брокера).  Кольцо «загрузчик → отчёт» этим и
# замкнулось (`§−103`), то есть выбор осознанный — но компенсирующий контроль
# теперь ЕДИНСТВЕННЫЙ: условие IAM на префикс `stooq/`.
#
# 🔴 РЕПОЗИТОРИЙ ЭТО УСЛОВИЕ ПРОВЕРИТЬ НЕ МОЖЕТ.  Шаг деплоя лишь ПЕЧАТАЕТ
# команду его завести, а права выдаёт оператор руками.  До этого скрипта
# радиус поражения писателя держался на обещании; теперь на него есть ответ
# одной командой.
#
# 🔴 Опасен не отсутствующий доступ, а доступ БЕЗ УСЛОВИЯ: бот при этом
# работает штатно, симптома нет, и разница видна только в JSON-политике.
# Ровно тот класс, что `§−90` A-3 — «пустая коллекция ≠ всё хорошо».
#
# Использование:
#   ./scripts/verify_ingest_iam.sh
#   BUCKET=ramp-bot-state INGEST_SA=ramp-ingest@proj.iam.gserviceaccount.com \
#     ./scripts/verify_ingest_iam.sh
#
# Код возврата: 0 — радиус узок; 1 — найдена дыра (годится для CI/приёмки).
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BUCKET="${BUCKET:-ramp-bot-state}"
PREFIX="${PREFIX:-stooq/}"
INGEST_SA_NAME="${INGEST_SA_NAME:-ramp-ingest}"
INGEST_SERVICE="${INGEST_SERVICE:-ramp-ingest-bot}"
REGION="${REGION:-us-central1}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
die()  { echo -e "${RED}❌ $*${NC}"; exit 1; }

command -v gcloud  >/dev/null || die "gcloud CLI не найден."
command -v python3 >/dev/null || die "python3 не найден."
PROJECT_ID=$(gcloud config get-value project 2>/dev/null | tr -d '[:space:]')
[ -n "$PROJECT_ID" ] && [ "$PROJECT_ID" != "(unset)" ] \
  || die "GCP проект не выбран: gcloud config set project <ID>"
# 🔴 Личность писателя СПРАШИВАЕТСЯ У СЕРВИСА, а не собирается из имени.
# Угаданное имя, разошедшееся с реальным, даёт вердикт «прав нет вовсе» — он
# fail-closed, но НЕ ТОТ: оператор пойдёт чинить права, которые в порядке, а
# настоящий радиус останется непроверенным. Проверять надо ту личность, под
# которой сервис РЕАЛЬНО бежит. Сборка из имени — фолбэк на случай, когда
# сервис ещё не задеплоен.
if [ -z "${INGEST_SA:-}" ]; then
  INGEST_SA=$(gcloud run services describe "$INGEST_SERVICE" \
    --region="$REGION" --project="$PROJECT_ID" \
    --format='value(spec.template.spec.serviceAccountName)' 2>/dev/null || true)
fi
if [ -z "${INGEST_SA:-}" ]; then
  INGEST_SA="${INGEST_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
  warn "Сервис $INGEST_SERVICE не опрошен — личность собрана из имени.
   Если она разойдётся с реальной, вердикт ниже будет про ЧУЖОЙ аккаунт."
fi

echo "Проект:      $PROJECT_ID"
echo "Бакет:       gs://$BUCKET"
echo "Писатель:    $INGEST_SA"
echo "Разрешено:   объекты под $PREFIX — и больше ничего"
echo "───────────────────────────────────────────────────────────"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
gcloud storage buckets get-iam-policy "gs://$BUCKET" --format=json > "$TMP/bucket.json" \
  || die "Не удалось прочитать политику бакета gs://$BUCKET"
# 🔴 Политику ПРОЕКТА тоже: грант на уровне проекта перекрывает условие на
# бакете целиком, и проверка одного бакета его НЕ УВИДИТ.
gcloud projects get-iam-policy "$PROJECT_ID" --format=json > "$TMP/project.json" \
  || die "Не удалось прочитать политику проекта $PROJECT_ID"

BUCKET="$BUCKET" PREFIX="$PREFIX" INGEST_SA="$INGEST_SA" \
BUCKET_POLICY="$TMP/bucket.json" PROJECT_POLICY="$TMP/project.json" \
python3 <<'PY'
import json, os, sys

bucket   = os.environ["BUCKET"]
prefix   = os.environ["PREFIX"]
member   = "serviceAccount:" + os.environ["INGEST_SA"]

# Роли, дающие ПРАВО ПИСАТЬ в бакет.  `owner`/`editor` включены не для полноты:
# именно они чаще всего оказываются выданы «на время отладки» и молча
# обесценивают условие на префиксе.
WRITE_ROLES = {
    "roles/owner", "roles/editor",
    "roles/storage.admin", "roles/storage.objectAdmin",
    "roles/storage.objectCreator", "roles/storage.objectUser",
    "roles/storage.legacyBucketWriter", "roles/storage.legacyBucketOwner",
    "roles/storage.legacyObjectOwner",
}

#: Условие обязано называть ИМЕННО этот путь.  Проверяется полная тройка
#: «бакет + objects + префикс»: условие с верным префиксом, но ЧУЖИМ бакетом —
#: типичная опечатка copy-paste, и она не сужает ничего.
NEEDLE = f"buckets/{bucket}/objects/{prefix}"


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def bindings(policy, want_member):
    for b in policy.get("bindings", []):
        if want_member in (b.get("members") or []):
            yield b


problems, narrow = [], []

for b in bindings(load(os.environ["BUCKET_POLICY"]), member):
    role = b.get("role", "")
    if role not in WRITE_ROLES:
        continue                                    # чтение радиус не расширяет
    cond = b.get("condition") or {}
    expr = str(cond.get("expression", ""))
    if not expr:
        problems.append(
            f"🔴 {role} на бакете БЕЗ УСЛОВИЯ — писатель достаёт до "
            f"tokenomics.db и users_vault.db.")
    elif NEEDLE not in expr:
        problems.append(
            f"🔴 {role}: условие есть, но не подтверждает путь "
            f"«{NEEDLE}». Выражение: {expr}")
    else:
        narrow.append(f"{role} · условие «{cond.get('title') or '—'}»")

for b in bindings(load(os.environ["PROJECT_POLICY"]), member):
    role = b.get("role", "")
    if role in WRITE_ROLES:
        problems.append(
            f"🔴 {role} на уровне ПРОЕКТА — перекрывает условие на бакете "
            f"целиком: писатель достаёт до всего хранилища.")

for line in narrow:
    print(f"  ✅ {line}")
for line in problems:
    print(f"  {line}")

if problems:
    print("\n🔴 РАДИУС ШИРЕ ОБЪЯВЛЕННОГО. Сузить:")
    print(f"""  gcloud storage buckets remove-iam-policy-binding gs://{bucket} \\
    --member={member} --role=<роль без условия>
  gcloud storage buckets add-iam-policy-binding gs://{bucket} \\
    --member={member} --role=roles/storage.objectAdmin \\
    --condition="title=stooq-only,expression=resource.name.startsWith('projects/_/buckets/{bucket}/objects/{prefix}')" """)
    sys.exit(1)

if not narrow:
    print("\n🔴 У писателя НЕТ прав на запись в этот бакет.")
    print("   Это не «безопасно» — это неработающий загрузчик: заливка базы")
    print("   будет отказывать, а ручной тир протухнет по свежести.")
    sys.exit(1)

print("\n✅ Радиус узок: писатель ограничен префиксом и до балансов не достаёт.")
PY
