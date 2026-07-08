# INDEX.md — карта репозитория: что где лежит

> Единая точка входа: структура проекта + вся документация по теме и значимости.
> Статусы доков: **🟢 живой** · **📌 справочник** · **🗺 roadmap** · **🗄 снимок**
> (историческая диагностика на дату — не обновляется, ценна как след).
> Обновлено: 2026-07-07 (реорганизация: корень очищен, доки → `docs/`, скрипты → `scripts/`).

---

## 0. Структура репозитория (верхний уровень)

```
Investment-advisor/
├── README.md            📌 что такое RAMP + запуск (остаётся в корне — конвенция)
├── CLAUDE.md            🟢 инструкции для агентов (остаётся в корне — конвенция)
├── SYSTEM_PROMPT.md     📌 системный промпт ИИ — ЧИТАЕТСЯ В РАНТАЙМЕ + Docker COPY
│                            (ai_narrative._build_system_prompt, agent/advisor_bot) → корень
├── Dockerfile · cloudbuild.yaml · deploy.sh   инфра-деплой
├── requirements.txt · requirements.lock       зависимости (lock = hash-locked)
├── src/                 весь исполняемый код (ниже §1)
├── tests/               pytest-сьюты test_phase*.py (CI: `pytest tests/ -q`)
├── docs/                ВСЯ документация (этот индекс + доки ниже §2-8)
├── scripts/             ручные CLI/smoke-скрипты (не CI)
├── design/premium_v2/   исходники Premium React + build.sh (сборка бандлов)
├── cloud_function/      GCS-триггерный RAG-ингест (ChromaDB из банковских PDF)
├── config/ · assets/ · data/    конфиги, статика, данные
```

## 1. Код (`src/`) — по функциональности

| Слой | Файлы | Назначение |
|---|---|---|
| **Бот / UI** | `tg_bot.py`, `entrypoint.py` | aiogram-флоу, онбординг, тиры, токеномика (→ `docs/TELEGRAM_BOT.md`) |
| **Квант-движок** | `finance/investment_logic.py` (MAC3), `finance/scoring*.py`, `finance/regime.py`, `finance/stress.py`, `finance/simulate.py`, `finance/factor_decomposition.py`, `finance/scenario_engine.py`, `finance/scenario_report.py`, `finance/data_lineage.py` | вся финансовая математика |
| **Слой отчёта** | `pdf_payload.py` (adapter), `premium_payload.py` (view-map), `premium_renderer.py`, `html_renderer.py` (routing), `pdf_charts.py`, `templates/` | payload → HTML |
| **Premium-ассеты** | `premium_assets/` (собранные JS/CSS-бандлы) | рантайм-копия из `design/premium_v2` |
| **LLM-нарратив** | `ai_narrative.py` | генерация текстов отчёта (Anthropic API) |
| **RAG / агент** | `agent/rag_engine.py`, `agent/gatekeeper.py`, `agent/advisor_bot.py` | банковская аналитика, advisory-аудит |
| **Данные / брокер** | `freedom_portfolio/` (Tradernet, `history.py`), `services/` (FX, macro), `db_tokenomics.py`, `profile_manager.py` | цены, FX, токены, профиль |

## 2. Документация — начать отсюда (Tier-0)

| Файл | Статус | Назначение |
|---|---|---|
| [`../README.md`](../README.md) | 📌 | Что такое RAMP, пайплайн Engine→Adapter→Template, запуск. |
| [`../CLAUDE.md`](../CLAUDE.md) | 🟢 | Инструкции для агентов: стек, верификация, форма репозитория. |
| [`TELEGRAM_BOT.md`](TELEGRAM_BOT.md) | 🟢 | **Полный флоу бота `/start`→отчёт**: онбординг, 3 тира, токеномика, колбэки, FSM. |
| [`REPORT_SECTIONS.md`](REPORT_SECTIONS.md) | 🟢 | **Карта секций отчётов** BASE/DEEP/Scenario: ключ→builder→движок→шаблон. |

## 3. Отчёты: дизайн и содержание

| Файл | Статус | Назначение |
|---|---|---|
| [`PREMIUM_DESIGN.md`](PREMIUM_DESIGN.md) | 📌 | Дизайн-система Premium V2 (cream/gold/ink), контракты данных, сборка бандлов. |
| [`REPORT_PAGES_BASE.md`](REPORT_PAGES_BASE.md) | 📌 | Постраничный разбор BASE-отчёта. |
| [`REPORT_PAGES_DEEP.md`](REPORT_PAGES_DEEP.md) | 📌 | Постраничный разбор DEEP-отчёта (факторы, 4-Pillar, стресс, режим, CoVe). |
| [`REGIME_SECTION_DEEP.md`](REGIME_SECTION_DEEP.md) | 📌 | Глубокий разбор секции «Рыночный режим». |
| [`REPORT_SECTIONS_AUDIT.md`](REPORT_SECTIONS_AUDIT.md) | 🗄 | Посекционный аудит живых отчётов (снимок 06-14). |

## 4. RAG (банковская аналитика в ChromaDB)

| Файл | Статус | Назначение |
|---|---|---|
| [`RAG_INGESTION.md`](RAG_INGESTION.md) | 📌 | Как загрузить банковские PDF в RAG-базу (пошагово). |
| [`RAG_TROUBLESHOOTING.md`](RAG_TROUBLESHOOTING.md) | 📌 | Runbook «RAG пуст, хотя PDF загружены» — 8 шагов + boot-ingest. |
| [`BANK_RAG_COVE_DIAGNOSIS.md`](BANK_RAG_COVE_DIAGNOSIS.md) | 🗄 | Диагностика Bank-RAG/CoVe (07-03, рекомендации реализованы). |

## 5. LLM / стратегия

| Файл | Статус | Назначение |
|---|---|---|
| [`MCP_STRATEGY.md`](MCP_STRATEGY.md) | 🗺 | MCP/LLM-стратегия: потреблять data-MCP, экспонировать риск-движок. |
| [`LLM_STRATEGY_MULTILINGUAL.md`](LLM_STRATEGY_MULTILINGUAL.md) | 🗺 | Экономика DEEP-вызова, prompt-caching, кросс-язычный RAG. |
| [`../SYSTEM_PROMPT.md`](../SYSTEM_PROMPT.md) | 📌 | Системный промпт AI-нарратива (в корне — рантайм-загрузка). |

## 6. Roadmaps (планы фаз)

| Файл | Статус | Назначение |
|---|---|---|
| [`ROADMAP_SCENARIO_TIER.md`](ROADMAP_SCENARIO_TIER.md) | 🗺 | Тир Scenario Analysis. **Ядро + UI/тир реализованы (2026-07-07).** |
| [`ROADMAP_DATA_RESILIENCE.md`](ROADMAP_DATA_RESILIENCE.md) | 🗺 | Устойчивость данных. **§4-б lookback 730→1825 ВКЛЮЧЁН (2026-07-07).** |

## 7. Домены / подсистемы

| Файл | Статус | Назначение |
|---|---|---|
| [`SMART_MONEY.md`](SMART_MONEY.md) | 📌 | Архитектура Smart-Money / инсайдеры (SEC Form 4). |
| [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) | 🟢 | Пер-подсистемные оценки готовности + GA-блокеры. |

## 8. Живой аудит

| Файл | Статус | Назначение |
|---|---|---|
| [`AUDIT.md`](AUDIT.md) | 🟢 | Институциональный аудит: находки, Было/Стало, дашборд. |

---

## Реорганизация 2026-07-07 (что переехало)

| Было | Стало | Почему |
|---|---|---|
| 10 `*.md` в корне | корень: только `README`, `CLAUDE`, `SYSTEM_PROMPT` | чистый корень; вся документация в `docs/` |
| `AUDIT/MCP_STRATEGY/PREMIUM_DESIGN/RAG_INGESTION/REPORT_SECTIONS/REPORT_SECTIONS_AUDIT/SMART_MONEY.md` | `docs/…` (`git mv`, история сохранена) | группировка документации |
| `ingest_reports.py`, `test_agent.py`, `test_performance.py` в корне | `scripts/…` (+фикс путей `../src`) | ручные скрипты — отдельно от кода/тестов |

**Инвариант:** `SYSTEM_PROMPT.md` НЕ переносится — читается в рантайме
(`ai_narrative`, `agent/advisor_bot`) и копируется Dockerfile из корня.
`tests/` не тронуты (CI — `pytest tests/ -q`). Дубли по смыслу отсутствуют:
RAG-кластер и LLM-кластер покрывают РАЗНЫЕ грани.

**Опционально (не сделано):** снимки 🗄 (`REPORT_SECTIONS_AUDIT`,
`BANK_RAG_COVE_DIAGNOSIS`) можно вынести в `docs/archive/` — скажите, перенесу
с правкой ссылок.
