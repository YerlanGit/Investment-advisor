# Project_MAP.md — карта проекта RAMP (что где лежит и зачем)

> Быстрый навигатор по репозиторию: структура, назначение каждого файла/пакета и
> как устроен поток данных «портфель → отчёт». Держать в корне для быстрого входа.
> Полная тематическая карта документации — `docs/INDEX.md`. Обновлено: 2026-07-08.

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
| `Project_MAP.md` | Этот файл — карта проекта. |
| `Dockerfile` | Образ Cloud Run: COPY `requirements` · `src/` · `SYSTEM_PROMPT.md` · `tests/`. |
| `cloudbuild.yaml` | CI/CD деплой-гейт (прогон pytest перед выкаткой). |
| `deploy.sh` | Ручной деплой-скрипт. |
| `requirements.txt` | Зависимости с верхними границами (intent-файл). |
| `requirements.lock` | Hash-locked резолюция (`--require-hashes`; ставит Docker/CI). |

---

## 2. `src/` — исполняемый код

### 2.1. Бот / точка входа
| Файл | Функция |
|---|---|
| `tg_bot.py` | **Весь Telegram-флоу**: онбординг, 3 тира, токеномика, колбэки, FSM, фоновый анализ. Только UI/роутинг. → `docs/TELEGRAM_BOT.md` |
| `entrypoint.py` | Boot Cloud Run: старт бота (long-polling) + синхронизация ChromaDB из GCS + boot-ingest RAG. |
| `db_tokenomics.py` | Токеномика: `init_user` (+10 welcome), `deduct_tokens`, `credit_tokens`, `get_balance`, снапшоты отчётов (MoM-дельта). |
| `profile_manager.py` | Риск-профили, мандатные лимиты классов активов, бенчмарки. |

### 2.2. Слой отчёта (payload → HTML)
| Файл | Функция |
|---|---|
| `pdf_payload.py` | **Адаптер**: `analyze_all()`-результат → «тяжёлый» payload (~86 ключей); форматирование, builder-ы секций. |
| `premium_payload.py` | View-map: payload → строгий design-контракт (DEEP 30 ключей, BASE 13). Только читает, не считает. |
| `premium_renderer.py` | Инъекция design-data в Premium React бандлы → финальный HTML. |
| `html_renderer.py` | Роутинг тир→шаблон (Premium vs Jinja), smoke-mock, запись в /tmp. |
| `pdf_charts.py` | Инлайн-SVG (equity curve, факторный радар) для отчёта. |
| `report_mocks.py` | Мок-данные для смоук-рендера. |
| `ai_narrative.py` | LLM-нарратив (Anthropic API): вердикт, комментарии секций, идеи. Загружает `SYSTEM_PROMPT.md`. |
| `batch_reports.py` | Пакетная генерация отчётов (Batch API). |
| `templates/` | Jinja-шаблоны: `report_basic_v3.html`, `report_deep_v3.html`, `report_scenario_v3.html`, партиал `_mandate_compliance.html`. |
| `premium_assets/` | Собранные Premium V2 бандлы (JS/CSS) — рантайм-копия из `design/premium_v2` (пересобираются `build.sh`). |

### 2.3. `src/finance/` — квант-движок (вся математика)
| Файл | Функция |
|---|---|
| `investment_logic.py` | **Ядро MAC3**: `analyze_all`, факторная Ridge-модель, ковариация (EWMA⊕Ledoit-Wolf), Euler-декомпозиция риска, CVaR bootstrap, загрузка цен (`get_market_data`, lookback 1825). |
| `scoring.py` / `scoring_orchestrator.py` | 4-Pillar Scoring (Fundamentals/Valuation/Technical/Credit). |
| `factor_decomposition.py` | Декомпозиция дисперсии по факторам (Euler) + «факторные двойники» (additive-слой). |
| `scenario_engine.py` | Движок сценарного тира: σ_p ковариационная, MCTR-таблица, funding-логика, walk-forward бэктест. |
| `scenario_report.py` | Сборка payload сценарного отчёта из `results` (переиспользует `scenario_engine`). |
| `regime.py` | Классификатор рыночного режима (Growth×Cycle + макро-overlay). |
| `stress.py` | Параметрические стресс-сценарии (β×шок с выпуклым капом). |
| `simulate.py` | Симуляция «до/после» ребаланса (ожидаемый эффект). |
| `black_litterman.py` | Black-Litterman целевые веса. |
| `action_plan.py` | Buy/Sell/Stop уровни (ATR+SMA+52w). |
| `technicals.py` | Технические сигналы (SMA/RSI/MACD). |
| `sec_edgar.py` | Фундаментальные данные SEC (ROE, маржа, Altman-Z, Piotroski). |
| `cds_feed.py` | Кредитный сигнал (CDS-прокси). |
| `smart_money.py` | Инсайдерские сделки (SEC Form 4). |
| `period_returns.py` | Мульти-периодная доходность (1м/3м/6м/12м/YTD). |
| `portfolio_series.py` | Расчёт временных рядов портфеля (equity curve). |
| `currency.py` | Base-Currency Approach: FX-трансформация цен до ковариации. |
| `data_lineage.py` | CoVe data-lineage (провенанс каждого числа для QC-панели). |
| `asset_taxonomy.py` | Классификация классов активов. |
| `broker_api.py` | Абстракция брокерского API. |
| `security.py` / `setup_vault.py` | Шифрование ключей брокера. |
| `tool_plugins.py` | Плагины инструментов. |

### 2.4. `src/agent/` — RAG и advisory
| Файл | Функция |
|---|---|
| `rag_engine.py` | RAG-движок (ChromaDB): чанкинг банковских PDF, метаданные bank/section/tickers, ранжирование по свежести. |
| `gatekeeper.py` | Advisory-аудит риск-лимитов мандата (non-blocking). |
| `advisor_bot.py` | Advisory-обёртка (загружает `SYSTEM_PROMPT.md`). |

### 2.5. `src/services/` — внешние данные
| Файл | Функция |
|---|---|
| `fx_feed.py` | FX-курсы (FRED); окно 1900 дней (покрывает ценовое). |
| `macro_data.py` | Макро-серии FRED (10Y−2Y, HY OAS, VIX, breakeven, безработица, GDP). |
| `report_storage.py` | Загрузка HTML-отчёта в GCS + подписанная ссылка (48 ч). |

### 2.6. `src/freedom_portfolio/` — брокер Freedom/Tradernet
| Файл | Функция |
|---|---|
| `client.py` | Tradernet API-клиент. |
| `history.py` | Загрузка дневных цен (`get_candles`/`get_history_frame`, lookback 1825 кал. дн). |
| `auth.py` | Аутентификация брокера. |
| `models.py` | Модели портфеля/позиций. |
| `websocket.py` / `display.py` / `__main__.py` | WS-стрим, форматирование, CLI. |

---

## 3. `tests/` — pytest-сьюты (30 файлов)
CI гоняет `python -m pytest tests/ -q` (baseline **601 passed, 10 skipped**).
Именование: `test_phase*.py` — по фазам разработки; `test_freedom_*.py` — брокер;
`test_factor_decomposition.py`, `test_phase23_scenario.py`, `test_phase24_scenario_report.py` — фичи этой сессии.
`fetch_logs.py` — вспомогательный.

---

## 4. Прочие директории
| Путь | Функция |
|---|---|
| `docs/` | **Вся документация** — начинать с `docs/INDEX.md` (карта по темам/значимости). |
| `scripts/` | Ручные CLI/smoke: `ingest_bank_report.py`, `ingest_reports.py` (RAG-ингест), `test_agent.py`, `test_performance.py` (движок-смоук), `setup_static_egress.sh` (статический egress-IP для WAF-фикса). НЕ pytest. |
| `design/premium_v2/` | Исходники Premium V2 React (`portfolio-*.jsx` = BASE, `deep/*.jsx` = DEEP) + `build.sh` (Tailwind + Babel → бандлы в `src/premium_assets/`). |
| `cloud_function/` | GCS-триггерный RAG-ингест: `main.py`, `rag_engine.py` (идентичен `src/agent/rag_engine.py`), `requirements.txt`. |
| `.github/workflows/` | CI: `python-ci.yml` (pytest-гейт), `Check logs.yml`. |
| `assets/` | Статика (изображения README). |
| `data/` | Локальный кэш (`cache/`). |
| `config/` | Конфиги (при наличии). |

---

## 5. Быстрый справочник «где менять»
| Хочу изменить… | Иду в… |
|---|---|
| Флоу бота / кнопки / тарифы | `src/tg_bot.py` (→ `docs/TELEGRAM_BOT.md`) |
| Формулу риска / фактор / CVaR | `src/finance/investment_logic.py` |
| Скоринг 4-Pillar | `src/finance/scoring.py` |
| Секцию отчёта (данные) | `src/pdf_payload.py` (карта: `docs/REPORT_SECTIONS.md`) |
| Вид отчёта (дизайн) | `design/premium_v2/*.jsx` → `build.sh` → `src/premium_assets/` (или `src/templates/*_v3.html`) |
| Сценарный тир | `src/finance/scenario_report.py` + `templates/report_scenario_v3.html` |
| ИИ-тексты / промпт | `src/ai_narrative.py` + `SYSTEM_PROMPT.md` |
| Токеномику | `src/db_tokenomics.py` |
| Загрузку банковских PDF в RAG | `scripts/ingest_bank_report.py` (→ `docs/RAG_INGESTION.md`) |
| Lookback / окно истории | `HISTORY_LOOKBACK_DAYS` env (default 1825) → `investment_logic.get_market_data` |

---

## 6. Инварианты (не сломать)
- `SYSTEM_PROMPT.md` — **только в корне** (рантайм-загрузка + Docker COPY).
- Premium-бандлы `src/premium_assets/*` пересобираются из `design/premium_v2/` через `build.sh` (Tailwind-шаг из корня репозитория), НЕ править руками.
- Числа считаются ТОЛЬКО в движке (`finance/*`); формат — в builder (`pdf_payload`); вид — в шаблоне/JSX.
- `src/` и `tests/` меняются вместе (новое поведение = новый тест).
- Верификация: `python -m pytest tests/ -q` + smoke-render трёх тиров.
