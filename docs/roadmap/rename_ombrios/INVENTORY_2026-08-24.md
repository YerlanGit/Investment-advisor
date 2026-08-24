# Опись имени RAMP в репозитории — что где лежит и какой вердикт

<!-- nav | area:roadmap | code:src/tg_bot.py,src/ingest_bot.py,src/entrypoint.py,cloudbuild.yaml | read-before:нужна точная строка, где встречается RAMP, и вердикт по ней -->

> **Статус: 🟢 живой.** Фактическая база под `README.md` (там правила, здесь
> цифры). Снимок сделан на `a0fd670` (`main`, после `§−100`) и обновлён
> после `§−101` — переезда главного бота; изменённые строки помечены.
> Файл НЕ читается в рантайме и ни на что не влияет.

---

## 1. Как считалось (воспроизводимо)

```bash
git rev-parse HEAD                                    # a0fd670
grep -rn "RAMP\|ramp" --include="*.py" src/           # код
grep -rn "RAMP" tests/ | grep -v RAMP_BOT_TOKEN       # пины в тестах
grep -rn "ramp" cloudbuild.yaml deploy.sh scripts/    # инфраструктура
```

Слово `ramp` даёт **четыре несовместимых класса совпадений** — см.
`README.md §0`. Ниже они разведены. Строки указаны на момент снимка;
проверяйте `grep`, а не номер, если файл с тех пор менялся.

---

## 2. 🔴 Контракт провенанса цен — `df.attrs['_ramp_*']`

**32 строки в `src/`, 30 в `tests/`. Вердикт: не трогать никогда.**

| Роль | Файл:строка | Ключи |
|---|---|---|
| Пишет (брокер, фолбэк) | `finance/broker_api.py:221,281-283` | `_ramp_source`, `_ramp_is_fallback`, `_ramp_fallback_reason`, `_ramp_fallback_detail` |
| Пишет (демо) | `finance/demo_portfolio.py:272-273` | `_ramp_is_mock`, `_ramp_source` |
| Пишет (ручной ввод) | `finance/manual_portfolio.py:210` | `_ramp_source` |
| **Читает — гейт движка** | `finance/engine/portfolio_manager.py:331-336` | `_ramp_is_fallback`, `_ramp_is_mock`, `_ramp_source` |
| **Читает — гейт бота** | `tg_bot.py:2705-2713` | все четыре |

Два независимых читателя — это не дублирование, а вторая линия обороны
после живого инцидента 19.07 (комментарий на `tg_bot.py:2699-2703`).
`pdf_payload.py` и `ai_narrative.py` эти ключи **НЕ читают** — провенанс
доезжает до них уже разобранным.

---

## 3. 🔴 Слово `ramp` не про бренд — совпадение букв

**Вердикт: не трогать никогда. Это математика и стандартный термин.**

| Файл:строка | Что это |
|---|---|
| `finance/scoring.py:411` | `def _ramp(x, lo, hi, cap)` — линейная рампа (насыщение) |
| `finance/scoring.py:418,420,422` | три её вызова: drawdown, концентрация сектора, плечо → риск-индекс |
| `finance/regime.py:459` | комментарий «Linear ramp 0 → MAGNITUDE_REF» |
| `tests/test_phase19_block_audit.py:141,157,172,184,201,218` | `_ramp_prices()` — локальный тестовый хелпер |

🔴 Именно из-за этого блока `sed -i 's/ramp/ombri/g'` по репозиторию —
запрещённая операция: он переименует расчётный хелпер риск-индекса.

---

## 4. ⚫ Имена ресурсов GCP — только оператор, в репо только подстановки

**Вердикт (решение владельца 2026-08-24): не трогаем.**

| Имя | Где объявлено | Комментарий |
|---|---|---|
| `ramp-bot` (Cloud Run) | `cloudbuild.yaml:19` `_SERVICE` | Также дефолт в `.github/workflows/Check logs.yml:16,67` — фильтр `resource.labels.service_name` |
| `ramp` (Artifact Registry) | `cloudbuild.yaml:18` `_REPO` | |
| `ramp-bot-state` (бакет) | `cloudbuild.yaml:25` `_STATE_BUCKET` | `tokenomics.db` + `users_vault.db` + `stooq/prices.sqlite` |
| `ramp-bot-quotes` (бакет) | `cloudbuild.yaml:61` `_QUOTES_BUCKET` | **Дефолт в файле ≠ значение в триггере** (там `ramp-bot-state`) — см. §7.1 |
| `ramp-bot-reports` | `cloudbuild.yaml:173` env, `services/report_storage.py:27` | Готовые HTML-отчёты |
| `ramp-bot-chroma-db-investadv` | `entrypoint.py:31`, `cloudbuild.yaml:173` | Снимок ChromaDB |
| `ramp-bot-chroma-db-inbox-investadv` | `entrypoint.py:44` | INBOX для PDF |
| `ramp-bot-rag-ingest` (функция) | `cloudbuild.yaml:310` | |
| `ramp-ingest@…` (SA) | `cloudbuild.yaml:57` `_INGEST_SA` (формат — комментарий на :56, подсказка на :249) | Узкие права: только префикс `stooq/` |
| `ramp-ingest-bot` (Cloud Run) | подстановка `_INGEST_SERVICE` (значение в триггере) | |
| `ramp-egress-{ip,nat,net,router,subnet,conn}` | `scripts/setup_static_egress.sh` | Статический egress под WAF брокера |
| `/tmp/ramp-stooq` | `finance/stooq_store.py:67`, env `STOOQ_LOCAL_COPY_DIR` | Локальная копия базы (не ресурс GCP, но имя пары) |

---

## 5. 🟡/🟢 Бренд и конфигурация — что реально можно менять

### 5.1 Токены (🟡 — синхронно с Secret Manager)

| Переменная | Где читается | Состояние |
|---|---|---|
| `OMBRI_BOT_TOKEN` | `tg_bot.TOKEN_ENV`, привязка `cloudbuild` через `_BOT_TOKEN_SECRET` | Главный бот. **Переехал в `§−101`** |
| `RAMP_BOT_TOKEN` | `tg_bot.LEGACY_TOKEN_ENV`, `deploy.sh`, дефолт `_BOT_TOKEN_SECRET` | Прежнее имя: принимается с предупреждением; секрет в GCP пока носит именно его |
| `OMBRI_INGEST_BOT_TOKEN` | `ingest_bot.py:54` (`TOKEN_ENV`) | Загрузчик. Переехал в `§−100` |
| `RAMP_INGEST_BOT_TOKEN` | `ingest_bot.py:59` (`LEGACY_TOKEN_ENV`) | Принимается с предупреждением |

Отдельно: `tests/` ставят заглушку токена в 13 файлах (`§−101` перевёл их на
`OMBRI_BOT_TOKEN`) — это следствие модульного чтения токена, а не
самостоятельная зависимость. Чтение осталось модульным намеренно: бот без
токена не «работает частично».

### 5.2 Хэндл бота в отчёте (🟢, но с последствием)

| Файл:строка | Значение |
|---|---|
| `branding.bot_username()` | **SSOT (`§−101`)**: env `BOT_USERNAME`, дефолт `Ombri_bot` |
| `pdf_payload.py` | `"bot_username": branding.bot_username()` |
| `premium_payload.py` | фолбэк → `branding.bot_username()` |
| `premium_assets/base-components.js:1973`, `deep-components.js:2556` | фолбэк в собранном бандле |
| `design/premium_v2/portfolio-ideas.jsx:170`, `deep/deep-plan.jsx:12`, `deep/deep-data.jsx:13` | исходники тех же фолбэков |
| `tg_bot.py:4-5` | докстринг формата deep-link |
| `docs/report/REPORT_SECTIONS.md:176`, `docs/bot/TELEGRAM_BOT.md:96-97` | документация |

✅ **Закрыто в `§−101`.** `BOT_USERNAME` теперь задаётся в деплое из
подстановки `_BOT_USERNAME`, а она переключается ТОЛЬКО в паре с
`_BOT_TOKEN_SECRET` — на это стоит тест. До переключения обе подстановки
держат прежние значения, поэтому прод не меняется.

### 5.3 Тексты для пользователя — переведены на SSOT (`§−101`)

Одиннадцать литералов заменены обращением к `branding`:

| Файл | Было | Стало |
|---|---|---|
| `tg_bot` ×4 | «Добро пожаловать в RAMP», «RAMP — Risk & Asset Management Platform», «Мандат утверждён…», «RAMP использует API…» | `branding.project_name()` → **OMBRIOS** |
| `tg_bot` ×3 | «Поддержка RAMP», «RAMP управляется кнопками», «Помощь по RAMP» | `branding.bot_name()` → **OMBRI** |
| `tg_bot` ×1 | `@ramp_support_bot` | `branding.support_contact()` |
| `profile_manager` ×4 | текст мандата | `branding.project_name()` |
| `design/premium_v2/{portfolio-ideas,deep/deep-plan}.jsx` | «Откроется бот RAMP в Telegram» | «Откроется бот OMBRI» + пересборка `build.sh` |
| `agent/advisor_bot`, докстринги `__init__`, `db_tokenomics`, `demo_portfolio`, `risk_engine` | RAMP | OMBRIOS |

Разделение «платформа/бот» — решение владельца: OMBRIOS там, где речь о
продукте, OMBRI там, где о боте, с которым говорит пользователь.

### 5.4 Запрет бренда в ИИ-выводе — называет актуальные имена (`§−101`)

`SYSTEM_PROMPT.md` и `ai_narrative` (3 места) запрещали слово «RAMP».
Теперь запрет называет **OMBRIOS, OMBRI и RAMP**: в `ai_narrative` — через
`branding`, поэтому следующий переезд обновит запрет сам; в
`SYSTEM_PROMPT.md` — литералом, файл читается в рантайме и подстановок не
знает. Основание — `§−97 E-7`: правило, которого нет в промпте, модель не
исполняет; запрет, называющий не то слово, — ровно такой случай.

### 5.5 Логгеры и служебное

`ombri.bot` (`tg_bot`) · `ombri.entrypoint` · `ombri.ingest*` — единая схема
после `§−101`. Проверено: ни один workflow и ни один скрипт не фильтрует по
имени логгера, выгрузка логов идёт по `resource.labels.service_name`.

`finance/sec_edgar.py` шлёт наружу `User-Agent: "OMBRIOS Advisory
ombrios-advisory@project.com"`. ⚠️ Это контакт, а не бренд: SEC ожидает
РАБОЧИЙ адрес, а `@project.com` — заглушка, и она была заглушкой и до
переезда. Требует решения владельца (`§7.5`).

## 6. Что изменено переездом загрузчика (`§−100`) и почему это не задело прод

Коммит `f5df196`, 10 файлов, +421/−68. **Граница переезда — загрузчик и
только он.**

Изменено:

| Файл | Что |
|---|---|
| `src/ingest_bot.py` | `TOKEN_ENV`/`LEGACY_TOKEN_ENV`/`MAIN_TOKEN_ENVS` + `read_bot_token()`; логгер `ramp.ingest` → `ombri.ingest`; «RAMP Ingest» → «OMBRI Ingest»; tmp-префикс `ramp-upload-` → `ombri-upload-` |
| `src/ingest_entrypoint.py` | логгер → `ombri.ingest.entrypoint` |
| `src/services/quote_ingest.py` | tmp-префиксы `ramp-*` → `ombri-*` |
| `cloudbuild.yaml` | новая подстановка `_INGEST_BOT_TOKEN_SECRET: 'OMBRI_INGEST_BOT_TOKEN'`; шаг загрузчика берёт секрет из неё |
| `tests/test_phase51_ingest_bot.py`, `tests/test_phase58_boot_import_race.py` | пины нового имени + пин НЕПРИКОСНОВЕННОСТИ главного |
| `CLAUDE.md`, `docs/audit/*`, `docs/roadmap/ingest_bot/PLAN_2026-08-17.md` | правила и запись раунда |

**Доказательства, что работающий продукт не задет** — не рассуждением, а
проверяемыми фактами:

1. **`src/tg_bot.py` и `src/entrypoint.py` в диффе коммита ОТСУТСТВУЮТ.**
   Проверка: `git show --stat f5df196`.
2. **Секреты главного бота в `cloudbuild.yaml:141` не изменены** —
   `RAMP_BOT_TOKEN=RAMP_BOT_TOKEN:latest,…` как было. На это есть тест:
   `test_phase51_ingest_bot.py:1117` требует буквальную строку, а
   `test_the_rename_did_not_touch_the_main_bot_secrets` падает, если её тронут.
3. **Имя переменной главного бота пинится в коде-как-тексте:**
   `test_phase51_ingest_bot.py:1672` — `assertIn('os.environ["RAMP_BOT_TOKEN"]', source)`.
   То есть «случайно переименовать главный бот» теперь красный тест, а не
   упавший прод.
4. **Фолбэк на прежнее имя** (`read_bot_token`) снимает связь по времени:
   код может уехать раньше секрета или позже, и ни один порядок не роняет бот.
5. **Верификация из коммита:** полный прогон `1861 passed, 5 skipped, 2 xfailed`;
   зеркало деплой-образа `1797 tests, OK, 65 skipped`; 8 из 8 мутаций пойманы.

---

## 7. Находки разбора, требующие решения владельца

### 7.1 Дефолт `_QUOTES_BUCKET` в файле расходится с триггером

`cloudbuild.yaml:61` держит `ramp-bot-quotes`, а триггер переопределяет на
`ramp-bot-state`. Работает именно триггер, и кольцо «загрузчик → отчёт»
замкнуто через него. **Расхождение не ошибка, но ловушка:** читающий файл
видит один бакет, живой сервис пишет в другой. Варианты — привести дефолт к
`ramp-bot-state` (одна строка, поведение не меняется) либо оставить и
задокументировать. Сейчас задокументировано здесь и в `README.md §3`.

### 7.2 Контакт поддержки — ✅ вынесен, ⚠️ аккаунта всё ещё нет

Владелец подтвердил: аккаунта не существует. В `§−101` контакт вынесен в
`branding.support_contact()` с именем `@OMBRI_support_bot` — чтобы завести
его одним действием, когда дойдут руки. **До этого момента команда
«Поддержка» отправляет пользователя в никуда**, и это не переезд сломал:
битым контакт был и раньше.

### 7.3 Бренд в Premium-бандле против запрета бренда в Jinja — осталось как есть

Тесты требуют `assertNotIn("RAMP", html)` для Jinja-фолбэка
(`test_phase4_reporting.py:2016,2029`, `test_phase5_rag_quality.py:287`), а
Premium-бандл показывает «Откроется бот OMBRI в Telegram» — теперь с новым
именем, но показывает.

**Это осознанно, и вот разница.** Запрет адресован МОДЕЛИ: ИИ-нарратив не
должен называть бренд, потому что тон отчёта — банковский, а не рекламный.
Кнопка «Применить идею» — не нарратив, а навигация: пользователю нужно
понимать, куда он сейчас уйдёт. Правило «модель не называет бренд» и факт
«кнопка называет, куда ведёт» не противоречат друг другу.

### 7.4 `BOT_USERNAME` в проде — ✅ закрыто в `§−101`

Задаётся из подстановки `_BOT_USERNAME`, парной к `_BOT_TOKEN_SECRET`.
Полупереключение (одно поменяли, другое нет) ловится тестом
`StagedSwitchIsAtomicTest.test_secret_and_handle_flip_together`.

⚠️ Тест смотрит ФАЙЛ. Значения, заданные в ТРИГГЕРЕ Cloud Build, он не
видит — там пару держит оператор (`MIGRATION_MAIN_BOT.md §3 шаг B`).

### 7.5 `User-Agent` для SEC EDGAR — адрес-заглушка

`finance/sec_edgar.py:32` представляется SEC строкой
`"OMBRIOS Advisory ombrios-advisory@project.com"`.

**Сегодня ничего не сломано, и это надо сказать прямо.** SEC отвергает
запросы БЕЗ заголовка `User-Agent` (403) — а он есть и по форме правильный
(«название + контакт»). Доставляемость адреса SEC не проверяет, поэтому
запросы проходят.

Проблемы две, обе на будущее:

1. **`project.com` — не наш домен.** Это существующий чужой домен, и мы
   приписываем ему свой трафик. Контакт в `User-Agent` — то, по чему SEC
   связывается с источником нагрузки; при жалобе разбираться будут не с нами.
2. **Некуда предупредить.** Политика fair access: лимит **10 запросов/сек**
   на IP, при превышении — блокировка на уровне сети, иногда без
   предупреждения. Контакт нужен ровно для того, чтобы предупреждение
   дошло; на нерабочий адрес оно не дойдёт.

**Насколько это близко.** Далеко: `_fetch_company_facts` кэширован
(`lru_cache`), пул на 3 воркера, один запрос на тикер за процесс, есть
backoff на 429/5xx с учётом `Retry-After`. Реальный темп — единицы запросов
на отчёт против лимита в 10/сек.

**Что сделать.** Заменить адрес на рабочий почтовый ящик, который вы читаете
(годится и личный). Одна строка, без изменения логики. Подставлять чужой или
выдуманный ящик за владельца нельзя — поэтому оставлено решением.

---

## 8. Что изменено переездом ГЛАВНОГО бота (`§−101`) и чем доказано, что прод цел

**Граница переезда:** имя переменной с токеном, имя секрета, хэндл, тексты
для пользователя. **Не тронуто:** бакеты, сервисы Cloud Run, Artifact
Registry, VPC, SA, ключи `df.attrs['_ramp_*']`, `scoring._ramp`, схемы БД.

| Файл | Что |
|---|---|
| `src/branding.py` | **новый**, L0: `project_name` · `bot_name` · `bot_username` · `support_contact`, каждое с env-оверрайдом |
| `src/tg_bot.py` | `read_bot_token()` (`OMBRI_BOT_TOKEN` → фолбэк `RAMP_BOT_TOKEN` → внятный `RuntimeError`); 8 текстов → `branding`; логгер `ombri.bot` |
| `src/profile_manager.py` | 4 строки мандата → `branding.project_name()` |
| `src/pdf_payload.py`, `src/premium_payload.py` | хэндл берётся у `branding`, литералов нет |
| `src/ai_narrative.py`, `SYSTEM_PROMPT.md` | запрет бренда называет OMBRIOS, OMBRI и RAMP |
| `src/entrypoint.py` | логгер `ombri.entrypoint` |
| `src/agent/advisor_bot.py`, `__init__.py`, `db_tokenomics.py`, `demo_portfolio.py`, `risk_engine.py`, `sec_edgar.py` | бренд в докстрингах и `User-Agent` |
| `design/premium_v2/{portfolio-ideas,deep/deep-plan,deep/deep-data}.jsx` | текст и фолбэк-хэндл + **пересборка `build.sh`** |
| `cloudbuild.yaml` | подстановки `_BOT_TOKEN_SECRET`, `_BOT_USERNAME`; привязка `OMBRI_BOT_TOKEN=${_BOT_TOKEN_SECRET}` |
| `deploy.sh`, `.env.template`, `README.md` | новое имя переменной; в шаблоне токена не было ВООБЩЕ — добавлен |
| `tests/test_phase59_brand_rename.py` | **новый**, 17 тестов |
| `tests/test_phase51_ingest_bot.py`, 13 файлов заглушек, `test_layering.py` | пины переведены на новое имя |

### Чем доказано, что работающий продукт цел

1. **Деплой после мержа ничего не переключает.** `_BOT_TOKEN_SECRET`
   указывает на существующий секрет `RAMP_BOT_TOKEN`, `_BOT_USERNAME` — на
   живой хэндл `KEN_investment_bot`. На это стоит тест
   (`test_the_deploy_does_not_bind_a_secret_that_may_not_exist`): привязка к
   несуществующему секрету уронила бы **весь** деплой, и прод остался бы на
   старой ревизии.
2. **Токен читается по обоим именам.** Порядок «код уехал / секрет ещё нет»
   и обратный — оба безопасны (`MainBotTokenEnvRenameTest`, 5 тестов).
3. **Отказ без токена стал внятным.** Было `KeyError` на импорте — симптом
   неотличим от `§−98`; стало `RuntimeError`, называющий оба имени и место,
   откуда токен берётся.
4. **Секрет не течёт в логи** — проверено тестом на тексте предупреждения.
5. **Данные не задеты вовсе.** Схемы БД, ключи шифрования и пути в
   `/mnt/state` не менялись ни одной строкой; состояние ключуется
   `telegram_id` пользователя (§1 `MIGRATION_MAIN_BOT.md`).
6. **Прочие секреты главного бота на месте** — `--set-secrets` заменяет весь
   набор, поэтому три оставшихся литерала пинятся отдельно
   (`test_the_rename_did_not_touch_the_other_secrets`).
7. **Бандлы пересобраны инструментом, а не руками**: в поставляемых
   `src/premium_assets/*.js` не осталось ни `RAMP`, ни `KEN_investment_bot`.
8. **Верификация:** полный прогон и зеркало деплой-образа зелёные;
   **7 из 7 мутаций** по новым гейтам пойманы (полупереключение в обе
   стороны, снятый фолбэк, вернувшийся литерал бренда, убранный хэндл,
   слипшиеся имена платформы и бота, неочищенная `@`).
