# TELEGRAM_BOT.md — полный флоу телеграм-бота: от `/start` до отчёта

> Назначение: единая карта поведения бота (`src/tg_bot.py`, `src/entrypoint.py`) —
> что происходит на каждом шаге, какие состояния FSM, какие колбэки, где списываются
> токены и как собирается отчёт. Для быстрой навигации при доработках UX/флоу.
> Обновлено: **2026-07-17** (B1: `/mandate` — меню точечных правок вместо ре-анкеты,
> `/help`, `setMyCommands`, динамический бенчмарк в факторной секции DEEP;
> **цена токена 2 500 ₸**). Ранее: 2026-07-07 (тир **Scenario Analysis**,
> токен-тариф base 1 / scenario 1 / deep 2, inline-CTA сценарного анализа).

---

## 0. Стек и точка входа

- **Framework:** aiogram 3.x (long-polling, НЕ webhook). `entrypoint.py` поднимает
  бота на Cloud Run, параллельно на boot синхронизирует ChromaDB (RAG) из GCS
  (`_download_chroma_db`) и при пустом сторе — ингестит PDF из INBOX
  (`_boot_ingest_from_inbox`, daemon-thread).
- **Диспетчер:** `build_dispatcher()` собирает middleware → роутеры → команды →
  колбэки. Хранилище FSM — `MemoryStorage` (состояние в памяти процесса;
  max-instances=1, поэтому переживает только рестарт).
- **Единственный инстанс:** Cloud Run `max-instances=1 / concurrency=1`. Фоновый
  анализ идёт как `asyncio.create_task` в polling-процессе — отсюда single-flight
  guard (`_IN_FLIGHT_USERS`), чтобы один юзер не занял воркер параллельными джобами.

---

## 1. Middleware (выполняется ПЕРВЫМ)

**`WhitelistMiddleware`** (`tg_bot.py`) — бета-гейт. Читает список разрешённых
telegram-id из env; апдейты от не-вайтлист юзеров дропаются с одноразовым
уведомлением. Пустой вайтлист = гейт выключен (иначе заблокировал бы всех).
Регистрируется на ОБЕ шины (`message` + `callback_query`), чтобы ни один путь не
обошёл гейт.

---

## 2. `/start` — вход и онбординг

Хендлер `cmd_start` (роутер `onboarding_router`). Развилка **новый / вернувшийся**:

### 2.1. Новый пользователь
1. `state.clear()` → `init_user(user_id)` — регистрирует юзера и начисляет
   **`INITIAL_TOKENS = 10`** («welcome_bonus»; `INSERT … ON CONFLICT DO NOTHING`
   гарантирует однократность бонуса).
2. Запускается **риск-профилирование — 6 вопросов** (`QUESTIONS`, `NUM_QUESTIONS=6`),
   FSM `Onboarding.Q1…Q6`:
   | # | Вопрос |
   |---|--------|
   | 1 | Инвестиционный горизонт |
   | 2 | Главная инвестиционная цель |
   | 3 | Реакция на просадку −20% |
   | 4 | Опыт в инвестировании |
   | 5 | Финансовый комфорт |
   | 6 | Стабильность дохода |
   Каждый вариант несёт `score_points`; сумма → риск-профиль (`profile_manager`).
   Клавиатура — `kb_question` (инлайн, + «⬅️ Назад» через `ob:back`).
3. `Onboarding.Universe` — выбор классов активов (`kb_universe`, мультивыбор,
   тумблеры `ob:uni:<key>`, подтверждение `ob:uni:confirm`).
4. `Onboarding.Benchmark` — рекомендованный профильный бенчмарк
   (`kb_benchmark_compact`), с возможностью раскрыть полный список
   (`ob:bench:expand` → `kb_benchmark`) и подтвердить (`ob:bench:confirm`).
5. `Onboarding.MandateReview` — показ рассчитанного мандата (лимиты классов,
   целевая волатильность); `ob:mandate:approve` / `ob:mandate:edit`.
6. **Подключение портфеля** (`kb_connect_choice`):
   - `connect:template` — демо-режим (шаблонный портфель, без брокера; отчёты по
     демо **бесплатны** — 2026-07-16);
   - `connect:freedom` — ввод ключей Freedom Broker (FSM
     `PortfolioConnection.Login → ApiKey → SecretKey`; это ЕДИНСТВЕННЫЙ
     легитимный free-text ввод — ключи шифруются перед сохранением).
7. → показ меню выбора анализа (`_show_analysis_menu`).

### 2.2. Вернувшийся пользователь
`init_user` (no-op, без повторного бонуса) → сразу меню анализа (или контекст
из deep-link `slug`, если пришёл из промо-канала).

### 2.3. Deep-link «Применить идею» → Сценарный анализ (2026-07-09, #3)
Кнопка **«Применить идею»** в HTML-отчёте (BASE + DEEP) не может списать токен
(статичная страница на GCS). Поэтому она делает **deep-link** в бота:
`t.me/<bot>?start=scn_<n>` (где `<n>` — номер выбранной идеи). `cmd_start`
распознаёт префикс `scn_`, срезает маркер (чтобы он не спутался со slug
промо-канала) и для вернувшегося пользователя сразу показывает подтверждение
**сценарного тира**: «🎯 Сценарный анализ по идее №N — спишется 1 токен»
(`kb_confirm("scenario", "idea")`, состояние `AnalysisFlow.awaiting_approval`).
Дальше — обычный путь `cb_confirm`: **токен списывается в боте только после
готового отчёта**. Логика сценарного тира не меняется — переиспользуется как есть.
Имя бота приходит в отчёт через `pdf_payload.bot_username` (env `BOT_USERNAME`,
default **`KEN_investment_bot`** — живой хэндл бота, R2#4) → `premium_payload.meta.botUsername`.

---

## 3. Выбор тира анализа

`_show_analysis_menu` → `kb_analysis_choice()` — **три тира**:

| Тир | Кнопка (callback) | Токены | Что внутри |
|---|---|---|---|
| **Базовый** | `analysis:base` | **1** | риск-профиль, CVaR/Sharpe/MaxDD, состав, секторы, риск-водопад, идеи ИИ, RAG-наблюдаемость |
| **Сценарный** | `analysis:scenario` | **1** | Euler-MCTR вклад в риск, выживаемость в 3 макро-режимах, слабые звенья (funding), walk-forward бэктест правила. **Детерминирован, 0 LLM-вызовов** |
| **Глубокий** | `analysis:deep` | **2** | всё из базового + факторное разложение (β + декомпозиция дисперсии), 4-Pillar Scoring, стресс-сценарии, рыночный режим, банковская аналитика (RAG), Action Plan, ожидаемый эффект |

Тариф — `TIER_COST = {"base":1, "scenario":1, "deep":2}`; подписи — `TIER_LABEL`.
Тариф применяется к **живому** источнику; отчёты по **демо-портфелю бесплатны**
(`_effective_cost(tier, source)` → 0 при source=demo; 2026-07-16), включая
сценарный отчёт из демо-кэша (маркер `_demo_portfolio` в `_SCENARIO_CACHE`).

`cb_analysis_choice` (`analysis:*`) → показывает предупреждение о списании и
`kb_confirm(tier, slug)` (`confirm:<tier>:<slug>` / `cancel`), состояние
`AnalysisFlow.awaiting_approval`.

---

## 4. Подтверждение и запуск (`cb_confirm`)

Порядок (важно — **billing только после готового отчёта**, H1):
1. **Single-flight guard** (`_try_acquire_user_slot`): второй параллельный запрос
   того же юзера отклоняется.
2. **Read-only проверка баланса** (`get_balance ≥ cost`) — БЕЗ списания. Не хватает
   → сообщение + `/topup`, слот освобождается.
3. **Резолюция источника** (`_resolve_portfolio_source`, ДО баланс-гейта — от неё
   зависит цена). Логируется явно (`PORTFOLIO SOURCE` / `KEY SOURCE` /
   `CONN MODE RECOVERED`), чтобы «ввёл API, но получил демо» был диагностируем
   (инцидент 2026-07-14). Правила (2026-07-16):
   - **ключи в vault всегда побеждают** (`SecureVault.has_user`, existence-check без
     расшифровки) → `freedom` + само-лечение режима, даже если режим потерян/`template`;
   - режим `freedom`, но vault пуст → **сервисный `FREEDOM_API_KEY` — ТОЛЬКО админу**
     (`ADMIN_USER_IDS`); обычному юзеру — «Брокер не подключён, привяжите заново»
     (слот освобождён, токен НЕ списан);
   - явный `template` (юзер сам нажал «Демо-режим»; хранится в `user_connection`) →
     демо-портфель, **отчёт бесплатный** (`_effective_cost`=0 — deduct и баланс-гейт
     пропускаются, работает и при 0 токенов);
   - источник не определён (нет ни выбора, ни ключей) → отчёт НЕ строится:
     «Источник портфеля не выбран», токен не списан — «демо по умолчанию» не существует;
   - `MasterKeyRotatedError` → чистый ре-онбординг без списания.
   Затем `_fetch_portfolio_sync` → обработка `BrokerAuthError` /
   `BrokerEmptyPortfolioError` / прочих (каждая: слот освобождён, токен НЕ списан,
   поддержка-id вместо трейса).
4. Превью портфеля (`_format_portfolio_preview`) + «отчёт через 5–10 мин».
5. `asyncio.create_task(_run_analysis_background(...))` — handler возвращается сразу
   (long-poll не таймаутит).

---

## 5. Фоновый анализ (`_run_analysis_background`)

Одно статус-сообщение, редактируется на месте (`step()`), 4 шага:

- **Шаг 1/4 — рыночные данные.** `manager.prefetch_market_data(candidates)` —
  загрузка истории через Tradernet (**окно 1825 кал.дн ≈ 5 лет**,
  `HISTORY_LOOKBACK_DAYS`), FX-трансформация цен в валюту отчёта. `loaded_count==0`
  → явное сообщение о подписке Market Data + без списания.
- **Шаг 2/4 — MAC3.** `_analyze_existing_portfolio_sync(df, bench, mandate)` →
  `analyze_all()`: факторная модель (Ridge β), ковариация (EWMA⊕Ledoit-Wolf),
  Euler-декомпозиция риска, CVaR bootstrap, SEC EDGAR, 4-Pillar, стресс, режим,
  Black-Litterman. Результат — `results` (см. `investment_logic.analyze_all`).
- **Шаг 3/4 — Gatekeeper** (advisory, non-blocking): проверка риск-лимитов мандата;
  нарушения — только в лог + в панель отчёта, не пугающими сообщениями в чат.
- **Шаг 4/4 — сборка отчёта.** `prev_snapshot` (MoM-дельта) → `_build_pdf_payload`:
  - `base`/`deep` → `pdf_payload.build_payload` (+ RAG-контекст через
    `_fetch_rag_context`, AI-нарратив `generate_narrative`, для DEEP — SVG-графики,
    факторные беты);
  - `scenario` → `finance.scenario_report.build_scenario_payload` (без RAG/ИИ).

**CHECKPOINT 3 — `_send_report`:** рендер HTML (`html_renderer.render_report_html`
→ Premium React для base/deep, свой Jinja для scenario) → запись в /tmp → аплоад в
GCS (`upload_report`) → пользователю подписанная ссылка (живёт 48 ч). Сбой рендера
→ `report_generation_failed`; сбой аплоада (`file://`) → `report_delivery_failed`;
оба → без списания.

**Billing (единственная точка списания):** `deduct_tokens(user_id, cost)` — ТОЛЬКО
после успешной доставки. Затем `save_report_snapshot` (для будущей MoM-дельты) и
финальное сообщение с остатком.

**Scenario-CTA (для base/deep):** после отчёта — `_cache_results_for_scenario`
кладёт `results` в bounded TTL-кэш (1 ч, ≤64 юзера) и бот вешает inline-кнопку
**«🎯 Сценарный анализ этого портфеля (1 токен)»** (`scenario:cached`).

---

## 6. Сценарный анализ одним тапом (`cb_scenario_cached`)

Кнопка под готовым base/deep отчётом. Сценарный отчёт детерминирован и считается
ИЗ КЭШИРОВАННОГО `results` (без повторной загрузки цен / ИИ):
1. `_get_cached_results` — кэш протух (>1 ч) → просьба перезапустить из меню.
2. Single-flight + баланс ≥ 1.
3. `build_scenario_payload` (в executor, чтобы не блокировать event-loop) →
   `_send_report(tier="scenario")` → `deduct_tokens(1, "scenario_analysis")` только
   после доставки. Ошибки — graceful, без списания.

Тот же тир доступен и напрямую из меню (`analysis:scenario`) — тогда считается с
нуля (полный `analyze_all` → scenario payload), тоже 1 токен.

---

## 7. Токеномика (`db_tokenomics.py`)

- Ценообразование (**ВАЖНОЕ изменение 2026-07-17**; было 500 ₸):
  **1 токен = 2 500 KZT · пакет 10 токенов = 25 000 KZT**. Источник правды для
  копирайта — константы `tg_bot.TOKEN_PRICE_KZT` / `TOKEN_PACK_PRICE_KZT`
  (менять там, а не в текстах).
- `INITIAL_TOKENS = 10` — welcome-бонус (однократно).
- `init_user` — регистрация + бонус (idempotent); `deduct_tokens` — списание;
  `credit_tokens` — пополнение; `get_balance` — баланс; `InsufficientFundsError`.
- **Команды** (зарегистрированы в «меню ⋮» Telegram через `set_my_commands`,
  `tg_bot.main`):
  - `/balance` (`cmd_balance`) — текущий баланс + тариф;
  - `/topup` (`cmd_topup`) — инструкция по пополнению;
  - `/grant` (`cmd_grant`) — админ-начисление (гейт `ADMIN_USER_IDS`);
  - `/support` (`cmd_support`) — поддержка;
  - `/help` (`cmd_help`, B1 2026-07-17) — карта тиров/токенов/мандата;
  - `/mandate` (`cmd_mandate`) — меню мандата (см. §7.1).

### 7.1. `/mandate` — меню точечных правок (B1 2026-07-17)

Раньше `/mandate` сразу перезапускал 6-вопросную анкету — чтобы сменить только
бенчмарк, клиент проходил всё заново (первопричина инцидента «сменил бенчмарк,
а отчёт сравнивает с S&P 500»). Теперь `cmd_mandate` показывает текущий мандат
(`_mandate_overview_text`) + инлайн-меню (`kb_mandate_menu`):

| Кнопка | Callback | Что делает |
|---|---|---|
| 🎯 Сменить бенчмарк | `mandate:edit:bench` | `kb_benchmark(current)` → `ob:bench:*` c флагом `edit_mode=True` → по `ob:bench:confirm` — `save_benchmark_ticker` (мгновенно, БЕЗ ре-анкеты и БЕЗ сброса утверждения) → саммари + CTA «Заказать новый отчёт» |
| 🧬 Изменить классы активов | `mandate:edit:universe` | `kb_universe` (мультивыбор `ob:uni:*`, edit-режим) → по confirm — `apply_universe` + `save_profile` (баллы/цели/бенчмарк сохранены) + `approve_mandate` |
| ⚖️ Изменить риск-профиль | `mandate:edit:profile` | «экспертный» выбор из 4 профилей (`mandate:profile:<score>`, опорный балл полосы) → пересчёт vol/TE/лимитов; выбранный бенчмарк НЕ перетирается (дефолт нового профиля — только подсказкой) |
| 🔄 Пройти анкету заново | `mandate:edit:requiz` | прежнее поведение (`Onboarding.Q1`) |
| ⬅️ Закрыть | `mandate:close` | выход в меню анализа |

Guardrails (§5.3 ТЗ): изменения мандата **бесплатны** — в `mandate:*`-флоу нет
ни одного вызова `deduct_tokens`; single-flight, whitelist, deep-link `scn_` и
text-fallback не затронуты. Закрыто тестами
`tests/test_phase31_benchmark_factor_propagation.py::MandateEditNoBillingTest`.

---

## 8. FSM-состояния

| Группа | Состояния | Назначение |
|---|---|---|
| `Onboarding` | `Q1…Q6`, `Universe`, `Benchmark`, `MandateReview` | риск-профилирование + мандат; `Universe`/`Benchmark` переиспользуются `/mandate`-меню с флагом `edit_mode=True` в FSM-данных (confirm-обработчики ветвятся: сохранить в БД и вернуться в саммари, а не идти дальше по онбордингу) |
| `MandateEdit` | `Profile` | ручной выбор риск-профиля из `/mandate` (B1 2026-07-17) |
| `PortfolioConnection` | `Login`, `ApiKey`, `SecretKey` | ввод ключей Freedom Broker (шифруются) |
| `AnalysisFlow` | `awaiting_approval` | подтверждение списания перед анализом |

**Text-fallback (регистрируется ПОСЛЕДНИМ):** любой посторонний free-text вне
активного FSM-состояния (`StateFilter(None)`, не `/команда`) мягко возвращается к
кнопочному флоу. Легитимный free-text (ключи брокера) несёт активное состояние и
не перехватывается.

---

## 9. Реестр колбэков

| Callback prefix | Хендлер | Действие |
|---|---|---|
| `ob:*` (`ob:back`, `ob:uni:*`, `ob:bench:*`, `ob:mandate:*`) | онбординг-роутер | навигация онбординга; в `edit_mode` confirm/back ведут обратно в `/mandate`-саммари |
| `mandate:*` (`mandate:edit:{bench,universe,profile,requiz}`, `mandate:profile:<score>`, `mandate:back/close/report`) | `cb_mandate_action` | меню мандата (B1 2026-07-17; бесплатно, без биллинга) |
| `connect:template` / `connect:freedom` | `cb_connect_choice` | выбор источника портфеля |
| `analysis:{base,scenario,deep}` | `cb_analysis_choice` | выбор тира |
| `confirm:<tier>:<slug>` | `cb_confirm` | запуск анализа |
| `scenario:cached` | `cb_scenario_cached` | сценарный отчёт из кэша |
| `cancel` | `cb_cancel` | отмена (без списания) |

**Deep-link start-параметры** (`/start <param>`, обрабатываются в `cmd_start`):

| Start param | Действие |
|---|---|
| `scn_<n>` | «Применить идею» из отчёта → подтверждение сценарного тира (§2.3) |
| прочий `<slug>` | контекст промо-канала (`_source_label`) |

---

## 10. Диаграмма флоу (ASCII)

```
/start
  │
  ├─ новый ──► init_user (+10 токенов) ──► Q1…Q6 ──► Universe ──► Benchmark
  │                                                                   │
  │                                                            MandateReview
  │                                                                   │
  │                                         connect:template / connect:freedom
  │                                                                   │
  └─ вернувшийся ─────────────────────────────────────► [Меню выбора анализа]
                                                                      │
                          ┌───────────────────┬───────────────────────┤
                     analysis:base       analysis:scenario       analysis:deep
                          │                    │                       │
                          └──────► cb_confirm (single-flight + баланс≥cost, БЕЗ списания)
                                              │
                                   загрузка портфеля (демо/Freedom)
                                              │
                             _run_analysis_background (Шаги 1→4)
                                              │
                               CHECKPOINT 3: render → GCS → ссылка
                                              │
                                     deduct_tokens(cost)  ◄── ЕДИНСТВЕННОЕ списание
                                              │
                       base/deep ──► кэш results + inline-CTA «🎯 Сценарный (1 токен)»
                                              │
                                       scenario:cached ──► отчёт из кэша (−1 токен)
```

---

## 11. Где что менять

- **Кнопки/копирайт меню** → `kb_analysis_choice`, `_show_analysis_menu`, `cmd_start`.
- **Тарифы (в токенах)** → `TIER_COST` / `TIER_LABEL` (`tg_bot.py`).
- **Цена токена в ₸** → `TOKEN_PRICE_KZT` / `TOKEN_PACK_PRICE_KZT` (`tg_bot.py`) —
  копирайт /balance, /topup, /help читает эти константы (2026-07-17: 2 500 ₸).
- **Меню /mandate** → `cmd_mandate` / `cb_mandate_action` / `_mandate_overview_text`
  / `kb_mandate_menu` / `kb_mandate_profile`; edit-ветвления — в
  `cb_benchmark_confirm`, `cb_universe_confirm`, `cb_back` (флаг `edit_mode`).
- **Команды в «меню ⋮» Telegram** → `set_my_commands` в `tg_bot.main`.
- **/help** → `cmd_help` (`tg_bot.py`).
- **Шаги анализа / прогресс** → `_run_analysis_background`.
- **Сборка payload по тиру** → `_build_pdf_payload` (base/deep → `pdf_payload`,
  scenario → `finance/scenario_report`).
- **Бенчмарк в факторной секции DEEP** → движок
  `finance/investment_logic.fit_factor_betas` + `benchmark_factor_profile` →
  `_build_factor_betas_table` (`tg_bot.py`) → `benchmark_name/ticker`
  (`pdf_payload`) → `benchmarkName` (`premium_payload`) →
  `design/premium_v2/deep/deep-factors.jsx` (после правки — `build.sh`!).
- **Рендер/маршрут шаблона** → `html_renderer._select_template` + `render_report_html`.
- **Онбординг-вопросы/скоринг** → `QUESTIONS` (`tg_bot.py`) + `profile_manager`.
- **Токеномика** → `db_tokenomics.py`.
- **Карта секций отчёта** → `REPORT_SECTIONS.md`.
