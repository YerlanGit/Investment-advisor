# Ручная обвязка GCP для бота-загрузчика — что заводится руками и как проверить

<!-- nav | area:roadmap | code:scripts/setup_ingest_scheduler.sh,scripts/verify_ingest_iam.sh,src/ingest_entrypoint.py,src/services/quote_ingest.py | read-before:когда заводишь расписание проверки или проверяешь радиус прав загрузчика в GCP -->

> Статус: 🟡 требует одноразовых операций оператора. Появился в `§−109`, когда
> аудит 26.08 нашёл, что **маршрут плановой проверки готов, а расписания нет** —
> и напоминание не приходит никогда.
>
> Регламент ПЕРВОЙ заливки — [`FIRST_GCS_RUN.md`](FIRST_GCS_RUN.md), это
> отдельный момент чтения. Здесь только то, что репозиторий сделать за вас
> не может.

---

## 0. Что здесь решается за 30 секунд

| Вопрос | Ответ |
|---|---|
| Что осталось незакрытым у загрузчика? | **Расписание** (§2) и **проверка радиуса прав** (§3). Всё остальное доставлено и проверено живым прогоном 25–26.08 |
| Почему это не делает деплой? | Cloud Scheduler и IAM — состояние ПРОЕКТА, а не образа. `cloudbuild.yaml` их не заводит осознанно: сборка не имеет права раздавать права |
| Что будет, если не делать? | Забыли файл на 4 дня → ручной тир отказывает, и первым это видит **пользователь**, а не оператор |
| Сколько это займёт? | Два запуска скрипта, ≈3 минуты |

---

## 0.5 Откуда это запускать

🔴 **Скрипты живут в РЕПОЗИТОРИИ, а Cloud Shell стартует в пустом `~`.**
Строка `./scripts/verify_ingest_iam.sh` без этого уточнения даёт
`No such file or directory` — и это не ошибка оператора, а недосказанность
инструкции (найдено живым запуском 26.08).

**Путь А — с репозиторием** (нужен, если будете гонять скрипты повторно):

```bash
cd ~ && git clone https://github.com/YerlanGit/Investment-advisor.git
cd Investment-advisor && git checkout claude/comprehensive-project-audit-a44y73
chmod +x scripts/verify_ingest_iam.sh scripts/setup_ingest_scheduler.sh
./scripts/verify_ingest_iam.sh
```

**Путь Б — без репозитория.** Обе операции одноразовые, поэтому ручные
последовательности из §2.3 и §3.2 самодостаточны: их можно вставить в Cloud
Shell как есть. Для проверки IAM это не «упрощённый вариант» — та же логика,
включая чтение политики ПРОЕКТА.

**Что сначала.** Узнайте, как сервис называется и под кем бежит — остальное
из этого выводится:

```bash
gcloud run services list --format='table(metadata.name,region)'
gcloud run services describe ramp-ingest-bot --region=us-central1 \
  --format='value(spec.template.spec.serviceAccountName)'
```

🔴 Личность писателя **спрашивается у сервиса**, а не собирается из имени.
Угаданное имя, разошедшееся с реальным, даёт вердикт «прав нет вовсе»: он
fail-closed, но НЕ ТОТ — оператор пойдёт выдавать права, которые в порядке, а
настоящий радиус останется непроверенным. `verify_ingest_iam.sh` спрашивает
сам; в ручном варианте подставьте вывод команды выше.

---

## 1. Что уже сделано и трогать не нужно

Заведено владельцем 23.08 (`§−100`), подтверждено живым прогоном 25–26.08 (`§−103`…`§−107`):

- секреты `OMBRI_INGEST_BOT_TOKEN` и `INGEST_TASK_TOKEN`;
- сервис-аккаунт `ramp-ingest@…` с `objectAdmin` на `ramp-bot-state`
  **с условием** на префикс `stooq/`;
- подстановки в Build Trigger: `_INGEST_SERVICE`, `_INGEST_SA`,
  `_INGEST_ADMIN_IDS`, `_QUOTES_BUCKET=ramp-bot-state`,
  `_INGEST_BOT_TOKEN_SECRET`;
- кольцо «загрузчик → отчёт» замкнуто: оба бота адресуют
  `gs://ramp-bot-state/stooq/prices.sqlite`.

🔴 **Ничего из этого не проверяется репозиторием.** Список выше — запись о том,
что оператор сделал, а не гейт. Раздел §3 существует ровно затем, чтобы
превратить первую строку про условие IAM из обещания в измерение.

---

## 2. Расписание плановой проверки (IB-7)

### 2.1 Что именно не работает без него

`POST /tasks/check` вызывает `ingest_bot.run_scheduled_check` →
`quote_ingest.build_reminder`. Тот **молчит, когда всё хорошо**, и говорит,
когда до блокировки ручного тира осталось ≤ 3 дня
(`INGEST_REMIND_DAYS_LEFT`, порог берётся у
`stooq_provider.MAX_MARKET_STALE_DAYS = 7` календарных дней).

Маршрут готов, задеплоен и закрыт дважды — OIDC снаружи, общий секрет внутри.
**Дёргать его некому.** Значит вся защита от протухания базы сводится к памяти
оператора, а обнаружение — к жалобе пользователя.

### 2.2 Как завести — скриптом

```bash
chmod +x scripts/setup_ingest_scheduler.sh
./scripts/setup_ingest_scheduler.sh
```

Скрипт идемпотентен: повторный запуск обновляет задание, а не заводит второе.
Переопределяется окружением:

| Переменная | Дефолт | Когда менять |
|---|---|---|
| `SCHEDULE` | `0 9 * * *` | 🔴 день недели оставляйте `*` — см. 2.4 |
| `TIME_ZONE` | `Asia/Almaty` | другой часовой пояс оператора |
| `REGION` | `us-central1` | регион сервиса |
| `INGEST_SERVICE` | `ramp-ingest-bot` | если сервис назван иначе |
| `JOB` | `ramp-ingest-check` | имя задания |
| `ATTEMPT_DEADLINE` | `600s` | первая проверка тянет ~54 МБ из GCS |
| `MAX_RETRIES` | `1` | см. 2.5 про дубли |

### 2.3 Как завести — руками, если скрипт запускать не хотите

🔴 **Блок ниже выполняется ЦЕЛИКОМ, а не по одной команде.** Переменные
(`SCHEDULER_SA`, `SERVICE_URL`) объявляются здесь же и в следующей команде уже
нужны. Живой прогон 26.08: оператор выполнил только `service-accounts create`,
переменная осталась пустой, и следующий шаг отказал
`INVALID_ARGUMENT: Invalid service account ()` — пустые скобки и есть подпись
этого случая. Первую строку той ошибки (`For a binding with condition…`)
читать не надо, она generic-подсказка gcloud и к делу не относится.

Если выполняете по шагам — сначала объявите и ПРОВЕРЬТЕ переменные:

```bash
PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
REGION=us-central1
INGEST_SERVICE=ramp-ingest-bot
SCHEDULER_SA="ramp-scheduler@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_AGENT="service-${PROJECT_NUMBER}@gcp-sa-cloudscheduler.iam.gserviceaccount.com"
SERVICE_URL=$(gcloud run services describe "$INGEST_SERVICE" --region="$REGION" \
  --format='value(status.url)')

for v in PROJECT_ID PROJECT_NUMBER REGION INGEST_SERVICE \
         SCHEDULER_SA SCHEDULER_AGENT SERVICE_URL; do
  printf '%-17s = %s\n' "$v" "${!v}"
  [ -n "${!v}" ] || echo "   🔴 ПУСТО — дальше не идите"
done
```

Ни одной строки «ПУСТО» — можно продолжать.

```bash
PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
REGION=us-central1
INGEST_SERVICE=ramp-ingest-bot

gcloud services enable cloudscheduler.googleapis.com

# 1. адрес сервиса
SERVICE_URL=$(gcloud run services describe "$INGEST_SERVICE" \
  --region="$REGION" --format='value(status.url)')

# 2. сервис-аккаунт вызывающего — ОТДЕЛЬНЫЙ от загрузчика
gcloud iam service-accounts create ramp-scheduler \
  --display-name="Cloud Scheduler → ingest bot"
SCHEDULER_SA="ramp-scheduler@${PROJECT_ID}.iam.gserviceaccount.com"

# 3. право вызвать — НА СЕРВИСЕ, не на проекте
gcloud run services add-iam-policy-binding "$INGEST_SERVICE" \
  --region="$REGION" --member="serviceAccount:${SCHEDULER_SA}" \
  --role=roles/run.invoker

# 4. планировщику — право выпустить OIDC-токен от имени этого SA
gcloud iam service-accounts add-iam-policy-binding "$SCHEDULER_SA" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-cloudscheduler.iam.gserviceaccount.com" \
  --role=roles/iam.serviceAccountTokenCreator

# 5. само задание.  🔴 Токен читается из Secret Manager и НЕ печатается.
TASK_TOKEN=$(gcloud secrets versions access latest --secret=INGEST_TASK_TOKEN)
gcloud scheduler jobs create http ramp-ingest-check \
  --location="$REGION" \
  --schedule='0 9 * * *' --time-zone=Asia/Almaty \
  --uri="${SERVICE_URL}/tasks/check" --http-method=POST \
  --headers="x-ingest-task-token=${TASK_TOKEN}" \
  --oidc-service-account-email="$SCHEDULER_SA" \
  --oidc-token-audience="$SERVICE_URL" \
  --attempt-deadline=600s --max-retry-attempts=1
```

**Если gcloud спросит про условие** на шаге 3 (`Condition … [1] None …`) —
выбирайте `None`. Право «постучаться в сервис» сужать нечем, а условие здесь
только сломает вызов.

### 2.3а 🔴 Проверьте маршрут ДО того, как заведёте расписание

Смысл порядка: если маршрут отвечает правильно руками, расписание — уже просто
будильник. Заведёте задание первым — отлаживать придётся через логи
планировщика, где не видно тела ответа.

```bash
TASK_TOKEN=$(gcloud secrets versions access latest --secret=INGEST_TASK_TOKEN)
curl -s -w '\nHTTP %{http_code}\n' -X POST "${SERVICE_URL}/tasks/check" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "x-ingest-task-token: ${TASK_TOKEN}"
```

Ответ маршрута — это ДИАГНОЗ, а не «ок/не ок»:

| Ответ | Что значит |
|---|---|
| `тихо` · 200 | ✅ вся цепочка жива, базе нечего сказать (наблюдалось 26.08) |
| `отправлено N` · 200 | ✅ жива, и вам пришло сообщение: база близка к порогу |
| `неверный секрет` · 401 | секрет в контейнере ≠ секрет в Secret Manager → пересоберите сервис |
| `INGEST_TASK_TOKEN не задан…` · 503 | секрет не доехал в контейнер (деплой без `--set-secrets`) |
| `только POST` · 405 | попали в маршрут, но методом GET |
| `OK` · 200 | 🔴 попали НЕ ТУДА: это проба `GET /`, она отвечает `OK` на любой чужой путь |
| `бот ещё не запущен` · 200 | контейнер поднимается, повторите через полминуты |
| `403` от Cloud Run | вашему аккаунту не хватает `run.invoker` на сервисе |

Строка `OK` — самая коварная: она выглядит успехом. Ровно её и сторожит тест
`SchedulerScriptMatchesTheRouteTest`.

🔴 **Путь `/tasks/check` и заголовок `x-ingest-task-token` — не украшение.**
Оба закреплены тестом `SchedulerScriptMatchesTheRouteTest` против
`ingest_entrypoint.TASK_PATH` / `TASK_TOKEN_HEADER`. Ошибётесь в пути — сервис
ответит **200** (это проба `GET /`, она отвечает 200 на любой путь), задание
будет вечно-зелёным, а проверка не выполнится ни разу.

### 2.4 Почему расписание ЕЖЕДНЕВНОЕ, а не по будням

`stooq_store.market_staleness_days` считает возраст базы в **календарных**
днях, а не в торговых сессиях — иначе застывший рынок рапортовал бы свежесть 0
(`§−80`). Значит выходные съедают запас так же, как будни, а забывают файл
чаще всего в пятницу вечером. Ежедневный запуск при здоровой базе молчит и
потому ничего не стоит. Ограничение дня недели закреплено тестом.

### 2.5 Известные свойства — это решения, а не недоделки

| Свойство | Почему так |
|---|---|
| Маршрут отвечает **200 даже при внутренней ошибке** | об ошибке оператор уже извещён в Telegram самим `run_scheduled_check`. 200 говорит «задача выполнена», и планировщик не ретраит то, что уже доставлено |
| Отсюда: в Cloud Scheduler **не видно** неудачной проверки | канал наблюдения — Telegram, а не консоль. Смотреть в логи сервиса, а не в статус задания |
| Секрет лежит **в двух местах** — Secret Manager и конфиг задания | Cloud Scheduler хранит заголовки в задании; иначе маршрут пришлось бы открыть. Ротация = обновить секрет, пересобрать сервис **и** перезапустить скрипт |
| `MAX_RETRIES=1` | ретрай после уже доставленного сообщения = дубль на телефоне. Пропущенная проверка стоит ноль: следующая придёт завтра, запас — 3 дня |
| Пока `_BOT is None` (первые секунды старта) проверка вернёт «бот ещё не запущен» и промолчит | окно измеряется секундами, а `--min-instances=1` держит контейнер живым |

### 2.6 Проверка

```bash
gcloud scheduler jobs run ramp-ingest-check --location=us-central1
gcloud run services logs read ramp-ingest-bot --region=us-central1 --limit=20
```

🔴 **Успех при здоровой базе — это МОЛЧАНИЕ в Telegram.** Что проверка дошла,
видно строкой в логах: `плановая проверка: всё в порядке, молчу`.
Если хотите увидеть само напоминание — временно поднимите порог:
`INGEST_REMIND_DAYS_LEFT=99` в переменных сервиса, прогнать, вернуть обратно.
🔴 `--set-env-vars` заменяет **весь** набор переменных — пользуйтесь
`--update-env-vars`, иначе снесёте `QUOTES_BUCKET` и бот потеряет базу.

---

## 3. Радиус прав писателя (IB-F2)

### 3.1 Чем это опасно

Загрузчик пишет в тот же бакет, где лежат `tokenomics.db` (балансы) и
`users_vault.db` (ключи брокера). Так замкнулось кольцо (`§−103`) — выбор
осознанный, но компенсирующий контроль остался **единственный**: условие IAM
на префикс `stooq/`.

🔴 **Опасен не отсутствующий доступ, а доступ БЕЗ УСЛОВИЯ.** Бот при этом
работает совершенно штатно: файлы применяются, база публикуется, `/status`
зелёный. Разница видна только в JSON политики. Это тот же класс, что `§−90` A-3
(«пустая коллекция ≠ всё хорошо»): самое опасное состояние выглядит как самое
спокойное.

### 3.2 Проверка одной командой

```bash
chmod +x scripts/verify_ingest_iam.sh
./scripts/verify_ingest_iam.sh
```

Код возврата 0 — радиус узок; 1 — найдена дыра, и скрипт печатает команды, как
её закрыть. Скрипт читает **две** политики:

- политику **бакета** — есть ли условие и называет ли оно нужный путь;
- политику **проекта** — потому что грант уровня проекта (`storage.admin`,
  `editor`, `owner`) перекрывает условие на бакете целиком, и проверка одного
  бакета его **не увидит**.

Что считается дырой:

| Находка | Почему это дыра |
|---|---|
| роль на запись **без условия** | писатель достаёт до балансов и ключей |
| условие называет **чужой бакет** | типичная опечатка copy-paste: сужает не тот путь, то есть не сужает ничего |
| роль на запись **на уровне проекта** | условие на бакете обесценено целиком |
| прав на запись **нет вовсе** | не «безопасно», а неработающий загрузчик: заливка откажет, ручной тир протухнет |

Все четыре сценария плюс здоровый закреплены тестом
`IamVerifierCatchesAWideRadiusTest` — на подставном `gcloud`, без похода в GCP.
Скрипт, который зелен всегда, был бы имитацией контроля, а не контролем.

### 3.2а Та же проверка без репозитория

Вставляется в Cloud Shell целиком. Логика та же, включая политику ПРОЕКТА.

```bash
BUCKET=ramp-bot-state; PREFIX=stooq/
INGEST_SERVICE=ramp-ingest-bot; REGION=us-central1
PROJECT_ID=$(gcloud config get-value project)
INGEST_SA=$(gcloud run services describe "$INGEST_SERVICE" --region="$REGION" \
  --format='value(spec.template.spec.serviceAccountName)')
echo "Писатель (спрошен у сервиса): $INGEST_SA"
gcloud storage buckets get-iam-policy "gs://$BUCKET" --format=json > /tmp/b.json
gcloud projects get-iam-policy "$PROJECT_ID" --format=json > /tmp/p.json
BUCKET="$BUCKET" PREFIX="$PREFIX" MEMBER="serviceAccount:$INGEST_SA" python3 <<'PY'
import json, os, sys
bucket, prefix, member = os.environ["BUCKET"], os.environ["PREFIX"], os.environ["MEMBER"]
WRITE = {"roles/owner","roles/editor","roles/storage.admin","roles/storage.objectAdmin",
         "roles/storage.objectCreator","roles/storage.objectUser",
         "roles/storage.legacyBucketWriter","roles/storage.legacyBucketOwner",
         "roles/storage.legacyObjectOwner"}
NEEDLE = f"buckets/{bucket}/objects/{prefix}"
load = lambda f: json.load(open(f, encoding="utf-8"))
binds = lambda pol: [b for b in pol.get("bindings", []) if member in (b.get("members") or [])]
problems, narrow = [], []
for b in binds(load("/tmp/b.json")):
    if b.get("role") not in WRITE: continue
    cond = b.get("condition") or {}; expr = str(cond.get("expression", ""))
    if not expr:
        problems.append(f"🔴 {b['role']} на бакете БЕЗ УСЛОВИЯ — писатель достаёт до балансов и ключей")
    elif NEEDLE not in expr:
        problems.append(f"🔴 {b['role']}: условие не подтверждает путь «{NEEDLE}». Выражение: {expr}")
    else:
        narrow.append(f"{b['role']} · условие «{cond.get('title') or '—'}»")
for b in binds(load("/tmp/p.json")):
    if b.get("role") in WRITE:
        problems.append(f"🔴 {b['role']} на уровне ПРОЕКТА — перекрывает условие на бакете целиком")
for l in narrow:   print("  ✅", l)
for l in problems: print("  ", l)
if problems: print("\n🔴 РАДИУС ШИРЕ ОБЪЯВЛЕННОГО — см. §3.3"); sys.exit(1)
if not narrow: print("\n🔴 У писателя НЕТ прав на запись — неработающий загрузчик, а не безопасность"); sys.exit(1)
print("\n✅ Радиус узок: писатель ограничен префиксом и до балансов не достаёт")
PY
```

### 3.3 Если условие потерялось

```bash
gcloud storage buckets add-iam-policy-binding gs://ramp-bot-state \
  --member=serviceAccount:ramp-ingest@<PROJECT_ID>.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin \
  --condition="title=stooq-only,expression=resource.name.startsWith('projects/_/buckets/ramp-bot-state/objects/stooq/')"
```

Привязку **без** условия после этого нужно снять отдельно
(`remove-iam-policy-binding`): IAM их не заменяет, а складывает, и широкая
останется действовать рядом с узкой.

---

## 4. Что осталось названным, но не закрытым

- **Ф-6 · окно между `get_blob` и скачиванием** в GCS-бэкенде. Даёт безопасный
  412 («объект изменился»), а не тихую порчу. Оператор один по построению
  (`PLAN §1.2`), поэтому окно практически недостижимо. Записано как
  ограничение, а не как долг.
- **Наблюдаемость плановой проверки** живёт в Telegram и логах, а не в статусе
  задания — см. 2.5. Алертинг на «проверка не выполнялась N дней» не заводился:
  это был бы сторож над сторожем при одном пользователе.

---

## 5. Порядок при переезде в другой проект GCP

1. секреты `OMBRI_INGEST_BOT_TOKEN`, `INGEST_TASK_TOKEN`;
2. сервис-аккаунт загрузчика + условие на `stooq/` → **проверить** §3.2;
3. подстановки в Build Trigger, деплой → прочитать вердикт кольца в логах сборки;
4. бутстрап базы в бакет (архив 0.84 ГБ, с машины оператора — см. `FIRST_GCS_RUN.md §3`);
5. расписание → §2;
6. контрольный прогон: `/status`, `/universe`, разовый `jobs run`.
