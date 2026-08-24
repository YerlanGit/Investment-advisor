# Опись имени RAMP в репозитории — что где лежит и какой вердикт

<!-- nav | area:roadmap | code:src/tg_bot.py,src/ingest_bot.py,src/entrypoint.py,cloudbuild.yaml | read-before:нужна точная строка, где встречается RAMP, и вердикт по ней -->

> **Статус: 🗄 снимок на 2026-08-24.** Это фактическая база под `README.md`
> (там правила, здесь цифры). Снимок делается на коммите `a0fd670`
> (`main`, после мержа PR #180 = раунд `§−100`).
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
| `RAMP_BOT_TOKEN` | `tg_bot.py:110` (**модульный уровень**, `os.environ[...]`), `deploy.sh:81,87`, `cloudbuild.yaml:141` | Главный бот. Фолбэка нет. Переезд — `MIGRATION_MAIN_BOT.md` |
| `OMBRI_INGEST_BOT_TOKEN` | `ingest_bot.py:54` (`TOKEN_ENV`) | Загрузчик. Уже переехал |
| `RAMP_INGEST_BOT_TOKEN` | `ingest_bot.py:59` (`LEGACY_TOKEN_ENV`) | Принимается с предупреждением |
| `OMBRI_BOT_TOKEN` | `ingest_bot.py:66` (`MAIN_TOKEN_ENVS`) | Пока только как имя для сверки «не тот же токен» |

Отдельно: `tests/` ставят `RAMP_BOT_TOKEN` заглушкой в 14 файлах —
это следствие модульного чтения токена, а не самостоятельная зависимость.

### 5.2 Хэндл бота в отчёте (🟢, но с последствием)

| Файл:строка | Значение |
|---|---|
| `pdf_payload.py:1514` | `os.getenv("BOT_USERNAME", "KEN_investment_bot")` |
| `premium_payload.py:1000` | фолбэк-литерал `"KEN_investment_bot"` |
| `premium_assets/base-components.js:1973`, `deep-components.js:2556` | фолбэк в собранном бандле |
| `design/premium_v2/portfolio-ideas.jsx:170`, `deep/deep-plan.jsx:12`, `deep/deep-data.jsx:13` | исходники тех же фолбэков |
| `tg_bot.py:4-5` | докстринг формата deep-link |
| `docs/report/REPORT_SECTIONS.md:176`, `docs/bot/TELEGRAM_BOT.md:96-97` | документация |

🔴 **`BOT_USERNAME` не задан в `cloudbuild.yaml`** — прод работает на
дефолте-литерале. Значит хэндл меняется ДВУМЯ способами: быстро (env в
деплое) и медленно (литерал + пересборка бандла). Ни один тест это значение
не пинит — проверено (`grep -rn "KEN_investment_bot" tests/` пуст).

### 5.3 Тексты для пользователя (🟢)

| Файл:строка | Текст |
|---|---|
| `tg_bot.py:1275` | «Добро пожаловать в RAMP — Risk & Asset Management Platform!» |
| `tg_bot.py:1339` | «RAMP — Risk & Asset Management Platform» |
| `tg_bot.py:1620` | «Мандат утверждён! Добро пожаловать в RAMP.» |
| `tg_bot.py:1712` | «RAMP использует API исключительно для режима ЧТЕНИЯ» |
| `tg_bot.py:3239-3240` | «Поддержка RAMP» + контакт `@ramp_support_bot` ⚠️ см. §7.2 |
| `tg_bot.py:3260` | «RAMP управляется кнопками» |
| `tg_bot.py:3413` | «Помощь по RAMP» |
| `profile_manager.py:179,180,200,202` | четыре строки в тексте мандата |
| `premium_assets/{base,deep}-components.js` + `design/premium_v2/portfolio-ideas.jsx:221` | «Откроется бот RAMP в Telegram» ⚠️ см. §7.3 |
| `agent/advisor_bot.py:36` | заголовок CLI-аудита |

### 5.4 Запрет бренда в ИИ-выводе (🟢, менять ВМЕСТЕ с брендом)

`SYSTEM_PROMPT.md:51` · `ai_narrative.py:313,1454,1627` — четыре места, где
модели запрещено произносить «RAMP». Переименовали бренд, забыли эти
строки → запрет продолжает запрещать НЕ ТО слово.

### 5.5 Логгеры и служебное (🟢)

`tg_bot.py:108` `ramp_bot` · `entrypoint.py:27` `ramp.entrypoint` ·
`__init__.py:1`, `db_tokenomics.py:2`, `finance/engine/risk_engine.py:346`,
`finance/demo_portfolio.py:2` (докстринги) ·
`finance/sec_edgar.py:32` `User-Agent: "RAMP Advisory ramp-advisory@project.com"`
(уходит НАРУЖУ, в SEC EDGAR — это контакт, а не бренд: SEC требует рабочий
адрес; менять только вместе с реальным адресом).

Проверено: **ни один workflow и ни один скрипт не фильтрует по имени
логгера** — выгрузка логов идёт по `resource.labels.service_name`.

---

## 6. Что уже изменено (`§−100`) и почему это не задело прод

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

### 7.2 Контакт поддержки `@ramp_support_bot` (`tg_bot.py:3240`)

Из репозитория нельзя проверить, существует ли такой аккаунт в Telegram.
Если нет — это битый контакт поддержки уже сегодня. Требует ответа
владельца, а не правки кода вслепую.

### 7.3 Бренд в Premium-бандле против запрета бренда в Jinja

Тесты требуют `assertNotIn("RAMP", html)` для Jinja-фолбэка
(`test_phase4_reporting.py:2016,2029`, `test_phase5_rag_quality.py:287`), и
SYSTEM_PROMPT запрещает бренд модели — а Premium-бандл строку «Откроется бот
RAMP в Telegram» показывает. Правило «в отчёте бренда нет» выполняется на
двух путях из трёх. Это не поломка (текст корректный), но это несогласованность
правила, и её стоит закрыть в ту или другую сторону осознанно.

### 7.4 `BOT_USERNAME` не задан в проде

См. §5.2. До переезда главного бота это безвредно (литерал совпадает с живым
хэндлом). В момент переезда это становится единственным местом, откуда
берётся хэндл в кнопке «Применить идею» уже выпущенных отчётов, —
подробности и последствия в `MIGRATION_MAIN_BOT.md §2`.
