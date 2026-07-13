# INDEX.md — единая карта проекта RAMP (все файлы, функциональные слои, документация)

> **Единственный навигатор репозитория** (2026-07-10: сюда влит корневой
> `Project_MAP.md` — файл удалён, вся карта живёт здесь).
> Статусы доков: **🟢 живой** · **📌 справочник** · **🗺 roadmap** · **🗄 снимок**
> (историческая диагностика на дату — не обновляется, ценна как след).

**Что это за проект.** RAMP — Telegram-бот, который прогоняет **институциональный
риск-движок MAC3/Barra** по реальному брокерскому портфелю пользователя и отдаёт
banker-grade HTML-отчёт (факторное разложение, хвостовой риск, стресс-тесты,
ИИ-нарратив) на русском. Три тира: **Базовый (1 токен)**, **Сценарный (1 токен)**,
**Глубокий (2 токена)**.

---

## 0. Поток данных (как рождается отчёт)

```
Telegram /start
  → онбординг (риск-профиль, мандат, подключение брокера)   [src/tg_bot.py]
  → загрузка портфеля + цен (Tradernet)                     [src/freedom_portfolio/]
  → analyze_all(): факторная модель, Euler-риск, CVaR,
     4-Pillar, стресс, режим, Black-Litterman               [src/finance/investment_logic.py]
  → build_payload(): движок → «тяжёлый» payload отчёта       [src/pdf_payload.py]
  → build_design_data(): payload → строгий view-контракт     [src/premium_payload.py]
  → рендер HTML (Premium React / Jinja)                      [src/html_renderer.py, premium_renderer.py]
  → загрузка в GCS + подписанная ссылка юзеру               [src/services/report_storage.py]
  ИИ-тексты приходят отдельной веткой:                       [src/ai_narrative.py] → ai_summary
```

**Три способа рендера по тиру** (`html_renderer._select_template` / routing):
- `base` / `deep` → Premium V2 React (бандлы `src/premium_assets/*-components.js`),
  fallback — Jinja `templates/report_{basic,deep}_v3.html`;
- `scenario` → всегда Jinja `templates/report_scenario_v3.html` (минуя Premium).

---

## 1. Корень репозитория

| Файл | Функция |
|---|---|
| `README.md` | Что такое RAMP + запуск. |
| `CLAUDE.md` | Инструкции для агентов: стек, верификация, форма репозитория. |
| `SYSTEM_PROMPT.md` | Системный промпт ИИ-нарратива. **Читается в рантайме** (`ai_narrative`, `agent/advisor_bot`) + копируется Dockerfile → **остаётся в корне**. |
| `Dockerfile` | Образ Cloud Run (digest-pinned base): COPY `requirements` · `src/` · `SYSTEM_PROMPT.md` · `tests/`; non-root user; пре-бейк ONNX-модели Chroma. |
| `cloudbuild.yaml` | CI/CD: build → test-гейт внутри образа → push → deploy Cloud Run (max-instances=1, gcsfuse state volume, секреты из Secret Manager) → deploy RAG Cloud Function. |
| `deploy.sh` | Ручной деплой-скрипт. |
| `requirements.txt` | Зависимости с верхними границами (человеческий intent-файл). |
| `requirements.lock` | Hash-locked резолюция (`--require-hashes`; ставит Docker/CI). Регенерация: `pip-compile --generate-hashes --strip-extras -o requirements.lock requirements.txt`. |
| `.env.template` / `config/.env.example` | Шаблоны env-переменных для локального запуска. |
| `.claude.json` · `.gitignore` · `.dockerignore` | Конфиги агентов / git / docker-контекста. |
| `.github/workflows/python-ci.yml` | CI: pytest-гейт (blocking) + pip-audit и gitleaks (advisory). |
| `.github/workflows/Check logs.yml` | Вспомогательный workflow выгрузки логов. |
| `.github/FUNDING.yml` | GitHub funding-метаданные. |
| `.port_sessions/*.json` | Служебные сессии инструмента (не код). |

---

## 2. `src/` — исполняемый код (все файлы)

### 2.1 Бот / точка входа
| Файл | Функция |
|---|---|
| `tg_bot.py` | **Весь Telegram-флоу**: онбординг-FSM, 3 тира, токеномика, колбэки, single-flight, фоновый анализ, SVG-чарты, admin `/grant`. Только UI/роутинг — серии считает движок. → `docs/TELEGRAM_BOT.md` |
| `entrypoint.py` | Boot Cloud Run: health-сервер + бот (long-polling) + синк ChromaDB из GCS + boot-ingest RAG (daemon-поток). |
| `db_tokenomics.py` | Токеномика (async SQLite, gcsfuse-safe): `init_user` (+10 welcome, exactly-once), `deduct_tokens`/`credit_tokens` (BEGIN IMMEDIATE), профили, снапшоты отчётов (MoM-дельта), fail-fast проверка персистентности (M-9). |
| `profile_manager.py` | Риск-профили онбординга, мандатные лимиты классов активов, список бенчмарков. |
| `batch_reports.py` | Пакетная генерация отчётов (Batch API). |
| `report_mocks.py` | Мок-данные smoke-рендера (вынесены из html_renderer). |
| `test_live_api.py` | Ручной live-смоук Tradernet API (не pytest). |
| `__init__.py` | Пакетные маркеры (также в `finance/`, `agent/`, `services/`, `freedom_portfolio/`). |

### 2.2 Слой отчёта (payload → HTML)
| Файл | Функция |
|---|---|
| `pdf_payload.py` | **Адаптер**: `analyze_all()`-результат → «тяжёлый» payload (~86 ключей); все builder-ы секций, форматирование, integrity/CoVe-панель. |
| `premium_payload.py` | View-map: payload → строгий design-контракт Premium V2 (DEEP ~30 ключей, BASE ~13). Только читает, не считает. |
| `premium_renderer.py` | Инъекция design-data в Premium React бандлы → финальный HTML. |
| `html_renderer.py` | Роутинг тир→шаблон (Premium default, v3 Jinja fallback), smoke-mock, запись в /tmp. |
| `pdf_charts.py` | Инлайн-SVG (equity curve, факторный радар). |
| `ai_narrative.py` | LLM-нарратив (Anthropic API, structured tool-call): вердикт, комментарии секций, идеи, prompt-injection защита, RAG-цитаты. Загружает `SYSTEM_PROMPT.md`. |
| `templates/report_basic_v3.html` | Jinja BASE-отчёт (fallback + test-pinned). |
| `templates/report_deep_v3.html` | Jinja DEEP-отчёт. |
| `templates/report_scenario_v3.html` | Jinja сценарного тира (единственный рендер тира). |
| `templates/_mandate_compliance.html` | Партиал мандатной панели. |
| `premium_assets/{base,deep}-components.js` | Собранные React-бандлы Premium V2 (**не править руками** — `design/premium_v2/build.sh`). |
| `premium_assets/report.compiled.css` · `custom.css` | Скомпилированный Tailwind + ручные стили. |
| `premium_assets/react*.production.min.js` | Вендорный React. |
| `premium_assets/*-data.sample.json` | Сэмплы design-data контрактов. |

### 2.3 `src/finance/` — квант-движок (вся математика)
| Файл | Функция |
|---|---|
| `investment_logic.py` | **Ядро MAC3**: `analyze_all`, Ridge-беты, ковариация EWMA(63)⊕Ledoit-Wolf + PSD-проекция, иерархическая ортогонализация (F-1: экспорт β̂ child→parent), Euler-декомпозиция, bootstrap-CVaR (SHA-256 seed), Marginal VaR, sparse-guard окна (F-6), форвардная E[r] = β·μ без альфы + кламп [−50%,+100%] + гейт панели на окно ≥252 дней (F-14), загрузка цен (lookback 1825). |
| `stress.py` | Параметрические стресс-сценарии: β×шок, convexity cap (20%→асимптота 35%), `residualize_shocks` (F-1 — маппинг raw-шоков в residual-пространство). |
| `scoring.py` | 4-Pillar скоры F/V/T/C, robust-z (MAD), composite risk 0-100 (SSOT), классификатор классов активов, hotspot-порог. |
| `scoring_orchestrator.py` | Оркестратор 4-Pillar: сектора, динамические SEC-когорты, действия Buy/Hold/Trim/Sell. |
| `factor_decomposition.py` | Euler-декомпозиция дисперсии ПО ФАКТОРАМ + «факторные двойники» + unique-risk (аддитивный слой). |
| `simulate.py` | «До/после» ребаланса: sample replay, анкеровка к headline, verdict; high-priority target weights. |
| `black_litterman.py` | BL: prior π=δΣw, posterior He-Litterman (k×k solve), Idzorek-Ω, turnover/per-name капы. |
| `scenario_engine.py` | Сценарный тир: pairwise-ковариация (min_periods 252), MCTR-таблица, funding decision-tree, sizing-капы, walk-forward бэктест (look-ahead guard). |
| `scenario_report.py` | Сборка payload сценарного отчёта из `results` (0 LLM). |
| `regime.py` | Классификатор режима (Growth×Cycle оси + FRED-оверлей level⊕trend, ±0.05). |
| `technicals.py` | RSI-14/MACD/Bollinger-Z/SMA/momentum 12-1/52w-high/volume-confirm (пилар T). |
| `period_returns.py` | Мульти-период (1м/3м/6м/12м/YTD), TE/IR (лог-пространство, pairwise), sparse-robust портфельная серия — маскированная композитная с по-дневной ренормализацией весов, `MIN_DAILY_COVERAGE=0.5` (F-15). |
| `portfolio_series.py` | Временные ряды портфеля (equity curve + KPI-спарклайны); серия — из готового композита `port_log_returns` (F-21), легаси-фолбэк с фильтром <60 дн. |
| `currency.py` | Base-Currency (H2): FX-трансформация цен (лаг T−1), GBX-пенсы ÷100 (F-7), RFR-реестр, H-1 дроп неконвертируемых. |
| `sec_edgar.py` | Фундаментал SEC CompanyFacts: ROE/маржа/долг/рост, Altman-Z, Piotroski-F, FCF. |
| `cds_feed.py` | Кредитный сигнал (CDS-прокси, free-слой). |
| `smart_money.py` | Инсайдеры SEC Form-4 (gated-провайдер). |
| `action_plan.py` | Buy/Sell/Stop уровни (ATR+SMA+52w, мандатные дистанции). |
| `data_lineage.py` | CoVe data-lineage: провенанс каждого числа отчёта (16 строк). |
| `asset_taxonomy.py` | Таксономия классов активов. |
| `broker_api.py` | FreedomConnector: портфель брокера → DataFrame, mock/fallback-гейты. |
| `security.py` | SecureVault: Fernet/MultiFernet-шифрование ключей брокера, ротация мастер-ключа. |
| `setup_vault.py` | CLI первичной закладки ключей в vault. |
| `tool_plugins.py` | Задел под MCP/tool-обёртки движка. |

### 2.4 `src/agent/` — RAG и advisory
| Файл | Функция |
|---|---|
| `rag_engine.py` | RAG (ChromaDB): section-aware чанкинг банковских PDF, метаданные bank/section/tickers, recency-ранжирование, purge tmp-артефактов. |
| `gatekeeper.py` | Advisory-аудит риск-лимитов мандата (9 правил, non-blocking). |
| `advisor_bot.py` | Advisory-обёртка (грузит `SYSTEM_PROMPT.md`). |

### 2.5 `src/services/` — внешние данные
| Файл | Функция |
|---|---|
| `fx_feed.py` | FX-курсы FRED (USD↔KZT, GBP↔USD c F-7); окно 1900 дн; retry+disk-cache; редакция api_key из ошибок. |
| `macro_data.py` | Макро-пак FRED (10Y−2Y, HY OAS, VIX, breakeven, безработица, GDP), 12h disk-cache, 3-state статусы. |
| `report_storage.py` | HTML-отчёт → GCS + подписанная ссылка v4 (TTL 48ч, private no-store). |

### 2.6 `src/freedom_portfolio/` — брокер Freedom/Tradernet
| Файл | Функция |
|---|---|
| `client.py` | Tradernet API-клиент (v2 HMAC-SHA256 → v1 md5 → unsigned fallback-цепочка). |
| `history.py` | Дневные цены `getHloc` (lookback 1825 кал.), retry, pickle-кэш, ffill без bfill (F-6). |
| `auth.py` | Подпись запросов (v1 md5 / v2 HMAC-SHA256). |
| `models.py` | Pydantic-модели портфеля/позиций. |
| `websocket.py` / `display.py` / `__main__.py` | WS-стрим, форматирование, CLI. |

---

## 3. `tests/` — pytest-сьюты (33 файла; CI: `python -m pytest tests/ -q`)

| Файл | Что покрывает |
|---|---|
| `test_phase1_fixes.py` | Ранние фиксы движка. |
| `test_phase2_modules.py` | Модули Фазы-2: режим, техникалс, скоринг. |
| `test_phase3_modules.py` | Модули Фазы-3: Black-Litterman, action plan. |
| `test_phase4_reporting.py` | Payload/отчёт + стресс-каталог. |
| `test_phase5_rag_quality.py` | RAG: чанкинг, метаданные, ранжирование. |
| `test_phase6_currency_h2.py` | Base-Currency H2/H3: FX, RFR, GDR-overrides, acceptance-кейсы. |
| `test_phase6_fx_feed.py` | FRED FX-провайдер (retry, cache, реестр пар). |
| `test_phase7_logic_wiring.py` | Сквозная проводка analyze_all. |
| `test_phase8_report_fixes.py` | Фиксы секций отчёта. |
| `test_phase9_fpillar_stress.py` | F-pillar + стресс-движок. |
| `test_phase10_pillar_chip.py` | 4-Pillar чипы/данные UI. |
| `test_phase12_beta_safety.py` | Беты/защитные гейты. |
| `test_phase13_security.py` | Vault, санитизация, безопасность. |
| `test_phase14_refactor.py` | SSOT-рефакторы, `math_firewall` (F-6: ведущие NaN сохраняются), sparse-guard, waterfall-геометрия. |
| `test_phase15_phase3.py` · `test_phase16_sprint2.py` · `test_phase20_sprint_refactor.py` | Спринтовые рефакторы/фиксы соответствующих фаз. |
| `test_phase17_admin_grant.py` | Admin `/grant` токенов. |
| `test_phase18_sprint5.py` | Мандатные панели, BL-констрейнты, cover-размещение. |
| `test_phase19_block_audit.py` | Ортогонализация факторов, premium-роутинг, блок-аудит. |
| `test_phase21_recos.py` | Рекомендации, numeric-twins контракт. |
| `test_phase22_rag_boot.py` | Boot-ingest RAG (требует `chromadb`). |
| `test_phase23_scenario.py` | Сценарный движок (Панели A/B). |
| `test_phase24_scenario_report.py` | Сценарный отчёт + тариф тиров. |
| `test_phase25_math_sprint1.py` | **Спринт-1 математики**: F-1 инвариантность стресса к ортогонализации, F-4 базис Sharpe, F-7 GBX. |
| `test_phase26_report_fixes.py` | **Post-release hotfix 2026-07-11** (сломанный DEEP-отчёт): F-14 гейт/кламп форвардной E[r] (e2e analyze_all), F-15 маскированная композитная серия, F-16 пропуск None-строк в premium-mapper, F-17 очистка RAG-выдержек. |
| `test_phase27_composite_metrics.py` | **Композитная база метрик** (F-20…F-23): реализованные Sharpe/CAGR/CVaR/MaxDD на композите полной панели (легаси бит-в-бит на полных), спарклайны из `port_log_returns`, drag плечевых ETP (`apply_leveraged_drag`), аннотация вне-модельных имён в Action Plan. |
| `test_factor_decomposition.py` | Факторная декомпозиция дисперсии + двойники. |
| `test_freedom_auth.py` / `_client.py` / `_history.py` / `_models.py` | Tradernet: подпись, клиент, история, модели. |
| `fetch_logs.py` · `query.txt` | Вспомогательные (не тесты). |

---

## 4. Прочие директории

| Путь | Функция |
|---|---|
| `docs/` | **Вся документация** (§5 ниже). |
| `scripts/` | Ручные CLI/smoke: `ingest_bank_report.py` (быстрый RAG-ингест, `--upload`), `ingest_reports.py`, `test_agent.py`, `test_performance.py`, `setup_static_egress.sh` (статический egress-IP для WAF-фикса). НЕ pytest. |
| `design/premium_v2/` | Исходники Premium V2 React: `portfolio-*.jsx` (BASE: app/charts/data/holdings/icons/ideas/overview/performance), `deep/*.jsx` (app/charts/cove/data/factors/holdings/icons/overview/plan/stress-regime), `build.sh` (Tailwind из корня репо + Babel → `src/premium_assets/`), `tailwind.config.js`, dev-шеллы и production-demo HTML. |
| `design/prototype_*.html` · `sample_*.html` | Ранние прототипы/сэмплы дизайна (история). |
| `cloud_function/` | GCS-триггерный RAG-ингест: `main.py` (`on_pdf_uploaded`), `rag_engine.py` (копия `src/agent/rag_engine.py` — держать идентичной), `requirements.txt` (chromadb==0.6.3). |
| `assets/` | Статика README (изображения). |
| `data/` | Локальный кэш (dev). |

---

## 5. Документация (`docs/`) — по темам и значимости

### Tier-0 (начать отсюда)
| Файл | Статус | Назначение |
|---|---|---|
| [`../README.md`](../README.md) | 📌 | Что такое RAMP, пайплайн Engine→Adapter→Template, запуск. |
| [`../CLAUDE.md`](../CLAUDE.md) | 🟢 | Инструкции для агентов: стек, верификация, форма репозитория. |
| [`TELEGRAM_BOT.md`](TELEGRAM_BOT.md) | 🟢 | **Полный флоу бота `/start`→отчёт**: онбординг, 3 тира, токеномика, колбэки, FSM. |
| [`REPORT_SECTIONS.md`](REPORT_SECTIONS.md) | 🟢 | **Карта секций отчётов** BASE/DEEP/Scenario: ключ→builder→движок→шаблон. |

### Отчёты: дизайн и содержание
| Файл | Статус | Назначение |
|---|---|---|
| [`PREMIUM_DESIGN.md`](PREMIUM_DESIGN.md) | 📌 | Дизайн-система Premium V2 (cream/gold/ink), контракты данных, сборка бандлов. |
| [`REPORT_PAGES_BASE.md`](REPORT_PAGES_BASE.md) | 📌 | Постраничный разбор BASE-отчёта. |
| [`REPORT_PAGES_DEEP.md`](REPORT_PAGES_DEEP.md) | 📌 | Постраничный разбор DEEP-отчёта (факторы, 4-Pillar, стресс, режим, CoVe). |
| [`REGIME_SECTION_DEEP.md`](REGIME_SECTION_DEEP.md) | 📌 | Глубокий разбор секции «Рыночный режим». |
| [`REPORT_SECTIONS_AUDIT.md`](REPORT_SECTIONS_AUDIT.md) | 🗄 | Посекционный аудит живых отчётов (снимок 06-14). |

### RAG (банковская аналитика в ChromaDB)
| Файл | Статус | Назначение |
|---|---|---|
| [`RAG_INGESTION.md`](RAG_INGESTION.md) | 📌 | Как загрузить банковские PDF в RAG-базу (пошагово). |
| [`RAG_TROUBLESHOOTING.md`](RAG_TROUBLESHOOTING.md) | 📌 | Runbook «RAG пуст, хотя PDF загружены» — 8 шагов + boot-ingest. |
| [`BANK_RAG_COVE_DIAGNOSIS.md`](BANK_RAG_COVE_DIAGNOSIS.md) | 🗄 | Диагностика Bank-RAG/CoVe (07-03, рекомендации реализованы). |

### LLM / стратегия
| Файл | Статус | Назначение |
|---|---|---|
| [`MCP_STRATEGY.md`](MCP_STRATEGY.md) | 🗺 | MCP/LLM-стратегия: потреблять data-MCP, экспонировать риск-движок. |
| [`LLM_STRATEGY_MULTILINGUAL.md`](LLM_STRATEGY_MULTILINGUAL.md) | 🗺 | Экономика DEEP-вызова, prompt-caching, кросс-язычный RAG. |
| [`../SYSTEM_PROMPT.md`](../SYSTEM_PROMPT.md) | 📌 | Системный промпт AI-нарратива (в корне — рантайм-загрузка). |

### Roadmaps
| Файл | Статус | Назначение |
|---|---|---|
| [`ROADMAP_SCENARIO_TIER.md`](ROADMAP_SCENARIO_TIER.md) | 🗺 | Тир Scenario Analysis. **Ядро + UI/тир реализованы (2026-07-07).** |
| [`ROADMAP_DATA_RESILIENCE.md`](ROADMAP_DATA_RESILIENCE.md) | 🗺 | Устойчивость данных. **§4-б lookback 730→1825 ВКЛЮЧЁН (2026-07-07).** |

### Домены / подсистемы
| Файл | Статус | Назначение |
|---|---|---|
| [`METHODOLOGY_SPARSE_AND_LEVERAGED.md`](METHODOLOGY_SPARSE_AND_LEVERAGED.md) | 📌 | Молодые бумаги (SPCX) и плечевые ETP (CONL) в риск-метриках: композитная конвенция, drag, roadmap пер-активных окон/Vasicek. |
| [`SMART_MONEY.md`](SMART_MONEY.md) | 📌 | Архитектура Smart-Money / инсайдеры (SEC Form 4). |
| [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) | 🟢 | Пер-подсистемные оценки готовности + GA-блокеры. |
| [`INFRA_NETWORKING.md`](INFRA_NETWORKING.md) | 🟢 | Прод-инциденты Cloud Run: WAF-блок цен (статический egress) + segfault RAG-ингеста. Команды фикса. |

### Аудит и экономика
| Файл | Статус | Назначение |
|---|---|---|
| [`AUDIT.md`](AUDIT.md) | 🟢 | Живой институциональный аудит: находки, Было/Стало, дашборд. |
| [`AUDIT_360_2026-07-10.md`](AUDIT_360_2026-07-10.md) | 🗄 | 360°-аудит на c20ef1c: дашборд оценок, инвентаризация всех формул движка, находки F-1…F-13, Action Plan. |
| [`ECONOMICS.md`](ECONOMICS.md) | 🟢 | Экономика: себестоимость генерации по тирам, все затраты, безубыточность, смета масштабирования, рыночные аналоги. |

---

## 6. Быстрый справочник «где менять»

| Хочу изменить… | Иду в… |
|---|---|
| Флоу бота / кнопки / тарифы | `src/tg_bot.py` (→ `docs/TELEGRAM_BOT.md`) |
| Формулу риска / фактор / CVaR | `src/finance/investment_logic.py` |
| Стресс-сценарии / шоки | `src/finance/stress.py` (F-1: raw-каталог residualize-ится автоматически) |
| Скоринг 4-Pillar | `src/finance/scoring.py` |
| Секцию отчёта (данные) | `src/pdf_payload.py` (карта: `docs/REPORT_SECTIONS.md`) |
| Вид отчёта (дизайн) | `design/premium_v2/*.jsx` → `build.sh` → `src/premium_assets/` (или `src/templates/*_v3.html`) |
| Сценарный тир | `src/finance/scenario_report.py` + `templates/report_scenario_v3.html` |
| ИИ-тексты / промпт | `src/ai_narrative.py` + `SYSTEM_PROMPT.md` |
| Токеномику | `src/db_tokenomics.py` |
| Валюты / FX / RFR | `src/finance/currency.py` + `src/services/fx_feed.py` |
| Загрузку банковских PDF в RAG | `scripts/ingest_bank_report.py` (→ `docs/RAG_INGESTION.md`) |
| Lookback / окно истории | `HISTORY_LOOKBACK_DAYS` env (default 1825) → `investment_logic.get_market_data` |

---

## 7. Инварианты (не сломать)

- `SYSTEM_PROMPT.md` — **только в корне** (рантайм-загрузка + Docker COPY).
- Premium-бандлы `src/premium_assets/*` пересобираются из `design/premium_v2/` через `build.sh` (Tailwind-шаг из корня репозитория), НЕ править руками.
- Числа считаются ТОЛЬКО в движке (`finance/*`); формат — в builder (`pdf_payload`); вид — в шаблоне/JSX.
- `src/` и `tests/` меняются вместе (новое поведение = новый тест).
- `cloud_function/rag_engine.py` держится идентичным `src/agent/rag_engine.py`.
- Верификация: `python -m pytest tests/ -q` + smoke-render трёх тиров.

---

## 8. История реорганизаций

| Дата | Что | Почему |
|---|---|---|
| 2026-07-07 | 10 `*.md` из корня → `docs/` (`git mv`, история сохранена); ручные скрипты → `scripts/` | Чистый корень: только `README`, `CLAUDE`, `SYSTEM_PROMPT`. |
| 2026-07-10 | `Project_MAP.md` (корень) влит в этот файл и удалён | Одна карта вместо двух пересекающихся. |
