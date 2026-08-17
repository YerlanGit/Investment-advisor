# CLAUDE.md

> Инструкция для агентов. **Здесь только ПРАВИЛА.** Журнал изменений, находки и
> «Было/Стало» живут в `docs/audit/AUDIT.md` — не дублируй их сюда.
> Одна мысль = одна строка. Файл читается в КАЖДОЙ сессии, поэтому он короткий.

## Что это

RAMP — Telegram-бот: гоняет институциональный риск-движок MAC3/Barra по реальному
брокерскому портфелю и отдаёт banker-grade HTML-отчёт на русском.
Три тира: **Базовый** (1 токен) · **Сценарный** (1 токен) · **Глубокий** (2 токена).
Стек: Python 3.11 · aiogram 3.x · numpy/pandas/scikit-learn · Anthropic API ·
GCP Cloud Run (long-polling) · Cloud Function (RAG-ингест) · ChromaDB · SQLite на gcsfuse.

## Верификация (обязательна перед каждым пушем)

```bash
PYTHONPATH=src python -m pytest tests/ -q          # → 1670 passed, 2 xfailed
```

- Префикс `PYTHONPATH=src` **ОБЯЗАТЕЛЕН** — без него `import finance…` не находится
  (`conftest.py`/`pyproject.toml` в репо нет; CI задаёт `PYTHONPATH: src`).
- **Прогонов ДВА.** Второй — зеркало деплой-образа, в нём НЕТ каталога `design/`:
  ```bash
  cp -r src tests SYSTEM_PROMPT.md requirements*.txt <tmp>/ && cd <tmp>
  PYTHONPATH=src python -m pytest tests/ -q        # → 1594 passed, 76 skipped, 2 xfailed
  ```
  Зелёный GitHub CI НЕ означает, что деплой пройдёт: CI видит полный чекаут,
  Cloud Build — только образ. Разница 76 тестов — ровно те, что читают
  `design/`, `docs/`, `CLAUDE.md`, `scripts/` и `cloud_function/`, то есть
  отсутствующее в образе.
- Правил `design/*.jsx` → **обязательно** `bash design/premium_v2/build.sh`.
- Смоук-рендер тиров: `html_renderer.render_report_html(None, <user_id>, ...)`.
- `freedom-etl/` гоняется ОТДЕЛЬНО (свои зависимости, в основную сюиту не входит):
  ```bash
  cd freedom-etl && python -m pytest tests/ -q     # → 109 passed, 13 skipped
  ```
  13 skipped — интеграционные; их включает `ETL_TEST_DSN=<dsn живого PostgreSQL>`.

## Инварианты (не сломать)

- Числа считаются ТОЛЬКО в `finance/*`; формат — в `pdf_payload.py`; вид — в шаблоне/JSX.
- `src/` и `tests/` меняются ВМЕСТЕ: новое поведение = новый тест.
- Тест, читающий `design/`, ОБЯЗАН `skipTest` при его отсутствии — иначе падает деплой-гейт.
- Пинить надо ПОСТАВЛЯЕМЫЙ артефакт (`src/premium_assets/*.js`), а не исходник `.jsx`.
- Tailwind в `build.sh` сканирует с **CWD = корень репо** (glob в `tailwind.config.js`
  корне-относительный); из подпапки получится пустой reset-only CSS и отчёт без стилей.
- Новый тикер добавляется ТОЛЬКО в `finance/asset_taxonomy.py` (SSOT ФАКТОВ).
  Решения у потребителей РАЗНЫЕ и сливать их нельзя: `TLT` для мандата «Bonds», для
  `Asset_Type` «ETF» — оба верны. `NoSecondCopyTest` падает на литерале тикера в потребителе.
- Разбор env на уровне модуля — ТОЛЬКО через `env_config.env_int`/`env_float`.
  Голый `int(os.getenv(...))` роняет ИМПОРТ, то есть старт контейнера, ещё до логирования;
  AST-сканер в тестах это запрещает.
- `SYSTEM_PROMPT.md` лежит ТОЛЬКО в корне — он читается в рантайме и копируется Dockerfile.
- `cloud_function/rag_engine.py` держится ИДЕНТИЧНЫМ `src/agent/rag_engine.py`.
- Premium-бандлы `src/premium_assets/*` руками НЕ править — только через `build.sh`.
- Расширил DEEP/BASE-контракт → обнови пин в `tests/test_phase19_block_audit.py`.
- Приватное `_имя` НЕ ходит между модулями: нужное двоим — публичное (`tests/test_layering.py`).
- Слой отчёта не импортирует `tg_bot`: это инверсия и затаскивание `aiogram` в рендер.
- `Ticker` позиции — из `canonical_ticker`, НЕ из `resolve_tickers`: второй отдаёт ПРОКСИ,
  и бумага в портфеле подменится (нота → `VWOB`: цена ETF, чужой тип, склейка разных бумаг).
- Источник портфеля доезжает до `price_source` КАК ЕСТЬ. Свести неизвестный к `freedom` —
  значит молча отдать данные Tradernet не-клиенту Freedom (I-12); отказ делает провайдер.
- Меняешь `analyze_all` → golden-фикстура (`tests/test_contracts_golden.py`) обязана совпасть.
  Расхождение в фазе Арх-3 = ошибка разреза, а НЕ повод обновить эталон.
- `analyze_all` — ОРКЕСТРАТОР: тело ≤ 150 строк, порядок стадий и их полнота пинятся
  (`test_engine_orchestrator.py`). Новая логика = новая СТАДИЯ, а не блок в оркестраторе.
- Эталон СЛЕП к порядку ключей словаря (`normalize` сортирует ради стабильности).
  Порядок, несущий смысл, пинится отдельным тестом — как `test_engine_benchmark_order.py`
  для профильного бенчмарка, чей первый ключ доезжает до подписи карточки (`§−64`).
- `freedom-etl/` — ОТДЕЛЬНАЯ единица поставки: свой образ и свои зависимости.
  В `src/` он не импортируется, `src/` в него не копируется; дублирование
  хелперов env здесь осознанно — тот же случай, что `cloud_function/rag_engine.py`.
- ДВА РАЗНЫХ ПРОЕКТА, не смешивать: `roadmap/manual_portfolio/` — ручной ввод и
  Stooq как его источник; `roadmap/freedom_warehouse/` — Freedom API → своя БД.
  У них разные юридические основания (I-12/I-14), поэтому `manual` не вправе
  читать витрину с `origin='tradernet'` — как и сам Tradernet.
- База котировок Stooq: пишет ТОЛЬКО `stooq_ingest` из процесса оператора, бот
  открывает `mode=ro` и читает ЛОКАЛЬНУЮ КОПИЮ (`stooq_store._local_copy`):
  SQLite поверх gcsfuse — это range-запрос на страницу, сотни на отчёт. Формы символа меняют НОТАЦИЮ, но не ПЛОЩАДКУ: кандидат
  `{base}.US` для иностранной бумаги подсунет ADR вместо листинга (`§−77`).
  Свежесть считается в ТОРГОВЫХ днях рынка бумаги, а не в календарных.
- **База RAG синкается ТОЛЬКО НА БУТЕ** (`_download_chroma_db`/`_boot_ingest_from_inbox`
  зовутся из `__main__` разово): живой контейнер держит копию со старта. Залил PDF в
  INBOX (`…-chroma-db-inbox-investadv`) → **рестартни бот**, иначе печатаются старые
  «N отчётов · M чанков» (`§−93`). Проверка — `scripts/rag_inventory.py --from-gcs --inbox`.
- **Отказ брокера — ТРИ причины** (`_ramp_fallback_reason`): `waf_block` (отбил
  Cloudflare по IP — САМО не пройдёт), `api_error` (транспорт — пройдёт),
  `parse_error` (НАШ дефект разбора). «Пройдёт за 5–15 минут» честно только для
  `api_error`. `_ramp_is_mock`/`_ramp_is_fallback` не трогать — на них гейты (`§−94`).
- **Бот гоняет ONNX-эмбеддинг в СВОЁМ контейнере** (`_boot_ingest_from_inbox`) на
  2Gi/1CPU: deploy-шаг обязан нести тот же кап потоков, что и RAG-функция (`§−94`).
- **Имя эмитента — ОДИН реестр** (`rag_engine.BANK_*`), новый банк = одна строка.
  Копий было четыре и они разошлись: пять банков молча стали `Unknown` на конвенции
  `wells_fargo_*.pdf` — `_` для `\b` СЛОВНЫЙ символ (`§−95`). Реестр в `rag_engine`:
  у `cloud_function/` свой `--source`, `finance/` там нет. Короткие формы
  (`MS`/`GS`) — только имя файла и тег, в прозе нельзя (`§−14` C-8).
- **Отчёт не называет источник, которого нет в базе**: имена — из `rag_banks`/
  `meta.ragBanks`; нечего назвать — молчи. В BASE стояли ТРИ выдуманных отчёта (`§−95`).
- **Статус/тон карточки — ВЕРДИКТ, а не оформление.** Считается в `finance/*` от
  значения и мандата (`scoring.kpi_status`); маппер его ЧИТАЕТ. Литерал статуса в
  `premium_payload` запрещён: так Sharpe 0.56 годами носил ярлык «good» (`§−90`).
  Неизвестный вход → «внимание», НИКОГДА не «ok».
- **Пустая коллекция провенанса ≠ «всё хорошо».** Прежде чем печатать «не
  требуется», спроси у движка, не пустота ли это от ОТКАЗА: пустой `fx_conversion`
  означал и «нечего конвертировать», и «курса не дали» — печаталось одинаково
  зелёным (`§−90` A-3).
- **Один цвет — один факт.** Свежесть данных и направление величины разводятся по
  разным каналам (`tone` против `trendTone`), иначе замедляющийся ВВП зеленеет.
- **Новый ключ design-data обязан иметь потребителя в бандле тира.** `kpis` жил в
  BASE-payload и не рендерился ВООБЩЕ (`§−90` A-2). Тест на это читает
  `src/premium_assets/*.js` — поставляемый артефакт, а не исходник `.jsx`.
- **Запрет в промпте неисполним без факта.** Если правило ссылается на состояние
  движка (погашенная дельта Sharpe), это состояние обязано доехать до модели.
- Мобильная вёрстка правится ПО ЗАМЕРУ (headless на 320/360/390/414), а не на глаз;
  детектор обязан отсеивать содержимое свёрнутых аккордеонов, иначе даёт ложь.
- Крупное изменение → строка «Было/Стало» в `docs/audit/AUDIT.md`.

## Зависимости

- `requirements.txt` — человеческий intent-файл: верхние границы (major-капы).
- `requirements.lock` — hash-locked резолюция; её ставят Docker и CI (`--require-hashes`).
- Менял `requirements.txt` → перегенерируй lock на linux/py3.11:
  ```bash
  pip-compile --generate-hashes --strip-extras -o requirements.lock requirements.txt
  ```
  Точечный апгрейд одного пакета — `--upgrade-package <имя>` (иначе поедут все пины).
- Верхняя граница версии может САМА блокировать security-фикс: если `pip-audit` красный,
  сначала проверь, не отсекает ли cap исправленную версию (`AUDIT §−53`).

## Слои (импорты смотрят только ВНИЗ)

```
L4 Delivery  tg_bot.py · entrypoint.py · db_tokenomics.py · services/report_storage.py
L3 Report    pdf_payload.py → premium_payload.py → premium_renderer.py · html_renderer.py
             ai_narrative.py · pdf_charts.py · report_charts.py · report_mocks.py
             finance/data_lineage.py · finance/scenario_report.py
L2 Engine    finance/engine/{risk_engine,portfolio_manager,market_preview}.py
             (finance/investment_logic.py — ФАСАД) · scoring*.py · stress.py · simulate.py
             black_litterman.py · regime.py · period_returns.py · scenario_engine.py · data_checks.py
L1 Data      freedom_portfolio/* · services/{fx_feed,macro_data}.py · finance/{broker_api,
             price_providers,sec_edgar,cds_feed,currency,demo_portfolio}.py · agent/rag_engine.py
             finance/{stooq_ingest,stooq_store,stooq_provider}.py — база котировок Stooq
L0 Cross     finance/contracts.py · finance/asset_taxonomy.py · finance/leveraged.py · env_config.py
             finance/market_calendar.py (рынок·валюта·календарь) · finance/stooq_symbols.py
```

Три контрактные границы конвейера: `results{}` (36 ключей, `finance/contracts.py`) →
payload (`pdf_payload`) → design-data (DEEP/BASE, пин в `test_phase19_block_audit`).

## Числа и константы — где SSOT

| Что | Источник правды |
|---|---|
| Тариф тиров (base 1 · scenario 1 · deep 2) | `tg_bot.TIER_COST` |
| Цена токена (2 500 ₸; пакет 10 = 25 000 ₸) | `tg_bot.TOKEN_PRICE_KZT` / `TOKEN_PACK_PRICE_KZT` |
| Окно истории (1825 календарных дней ≈ 5 лет) | env `HISTORY_LOOKBACK_DAYS` → `investment_logic.get_market_data` |
| Модели LLM (Sonnet 5 base / Opus 4.8 deep) | env `ANTHROPIC_MODEL_BASE` / `ANTHROPIC_MODEL_DEEP` |
| Риск-индекс 0–100 | `finance/scoring.composite_risk_score` |
| Классификация инструмента | `finance/asset_taxonomy.py` |

- **Текущие модели Anthropic ОТВЕРГАЮТ `temperature` (HTTP 400)** — параметр не
  передаётся (`ai_narrative._TEMPERATURE_UNSUPPORTED_PREFIXES`); разнообразие идей даёт
  промпт-директива свежести + ротация угла по дню.
- Рендер по умолчанию — **Premium V2 React** (`PREMIUM_REPORT_ENABLED`, default true);
  Jinja v3 (`templates/report_*_v3.html`) — авто-фолбэк, он же test-pinned.
- Сценарный тир — отдельный детерминированный отчёт, **0 вызовов LLM**, свой Jinja.
- Демо-отчёты бесплатны; токен списывается ТОЛЬКО после доставки отчёта.

## Навигация по докам

- Старт — **`docs/INDEX.md`**; перед правкой кода смотри **§6: карта КОД → ДОК**.
- У каждого дока на строке 2 машиночитаемый якорь:
  `grep -rl "code:.*<файл>" docs/` находит управляющий документ.
- Авто-подгружаемые вложенные инструкции: `src/finance/CLAUDE.md` (инварианты движка),
  `design/premium_v2/CLAUDE.md` (обязательная пересборка бандлов).
- Какие тесты покрывают модуль — `tests/TEST_MAP.md` (генератор `scripts/gen_test_map.py`).
- Ничто в `docs/` не читается в рантайме: пути в комментариях — документация, не загрузка.
- Текущий архитектурный трек и его фазы — `docs/ARCHITECTURE_FOR_AGENTS.md §4`.

## История

**`docs/audit/AUDIT.md`** — НАВИГАТОР (доска задач · оценка · правила · индекс
раундов · протокол обновления). Читается целиком, он маленький.
Полные записи раундов — `docs/audit/rounds/` (файл по диапазону номеров).
Ссылку вида `§−45` ищи так: `grep -n "^## −45\." docs/audit/rounds/*.md`.
🔴 Нумерация раундов НЕ меняется: на неё ссылаются ~540 мест.
Свежий сквозной аудит — `docs/audit/AUDIT_360_2026-08-17.md`
(там же КАРТА ПРОИСХОЖДЕНИЯ ДАННЫХ: что снаружи, что считается). Предыдущий —
`docs/audit/AUDIT_360_2026-08-12.md` (разбор ОТ АРТЕФАКТА:
живые BASE/DEEP + мобильный замер; там же строгая оценка проекта).
Планы по проектам РАЗДЕЛЕНЫ и смешивать их нельзя (разные основания I-12/I-14):
`docs/roadmap/manual_portfolio/PLAN_2026-08-12.md` · `docs/roadmap/freedom_warehouse/PLAN_2026-08-12.md`.
**Новые факты раунда пиши туда, а сюда — только изменившиеся ПРАВИЛА.**

## Рабочее соглашение

- Мелкие обозримые изменения; не смешивай рефакторинг с правкой поведения в одном PR.
- Общие дефолты — в `.claude.json`; машинные оверрайды — в `.claude/settings.local.json`.
- Не переписывай этот файл автоматически: обновляй осознанно, когда изменились правила.
