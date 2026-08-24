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
PYTHONPATH=src python -m pytest tests/ -q          # → 1861 passed, 5 skipped, 2 xfailed
```

- Префикс `PYTHONPATH=src` **ОБЯЗАТЕЛЕН** (`conftest.py`/`pyproject.toml` нет).
- **Прогонов ДВА.** Второй — зеркало деплой-образа, без `design/`: `cp -r src tests
  SYSTEM_PROMPT.md requirements*.txt <tmp>/ && cd <tmp>`, тот же прогон → 1775 passed,
  91 skipped, 2 xfailed. Зелёный GitHub CI НЕ означает, что деплой пройдёт: CI видит
  полный чекаут, Cloud Build — только образ, и разница — ровно тесты, читающие
  `design/`, `docs/`, `CLAUDE.md`, `scripts/` и `cloud_function/`.
- Правил `design/*.jsx` → **обязательно** `bash design/premium_v2/build.sh`.
- Смоук-рендер тиров: `html_renderer.render_report_html(None, <user_id>, ...)`.
- `freedom-etl/` гоняется ОТДЕЛЬНО (свои зависимости, в основную сюиту не входит):
  `cd freedom-etl && python -m pytest tests/ -q` → 109 passed, 13 skipped
  (интеграционные; включает `ETL_TEST_DSN=<dsn живого PostgreSQL>`).

## Инварианты (не сломать)

- Числа считаются ТОЛЬКО в `finance/*`; формат — в `pdf_payload.py`; вид — в шаблоне/JSX.
- `src/` и `tests/` меняются ВМЕСТЕ: новое поведение = новый тест.
- Тест, читающий `design/`, ОБЯЗАН `skipTest` при его отсутствии — иначе падает деплой-гейт.
- Пинить надо ПОСТАВЛЯЕМЫЙ артефакт (`src/premium_assets/*.js`), а не исходник `.jsx`.
- Tailwind в `build.sh` сканирует с **CWD = корень репо** (glob в `tailwind.config.js`
  корне-относительный); из подпапки получится пустой reset-only CSS и отчёт без стилей.
- Новый тикер — ТОЛЬКО в `finance/asset_taxonomy.py` (SSOT ФАКТОВ); решения у
  потребителей РАЗНЫЕ (`TLT`: мандат «Bonds», `Asset_Type` «ETF» — оба верны).
  `NoSecondCopyTest` падает на литерале тикера в потребителе.
- Старт контейнера роняют ДВЕ вещи. Разбор env на уровне модуля — ТОЛЬКО через
  `env_config.env_int`/`env_float`: голый `int(os.getenv(...))` роняет ИМПОРТ (AST-сканер).
  И ТЯЖЁЛЫЙ импорт на ФОНОВОМ ПОТОКЕ — оба бота: демон RAG рвал numpy у главного
  потока (`§−98`), у загрузчика то же делают потоки `to_thread` (`§−99`). Новый
  ленивый импорт обязан попасть в реестр предзагрузки своего бота
  (`entrypoint._BOOT_INGEST_HEAVY_IMPORTS` / `ingest_bot._WORKER_HEAVY_IMPORTS`).
- У двух ботов РАЗНЫЕ токены: один на двоих — 409 у обоих и упавший ГЛАВНЫЙ бот (`§−99`). Оба на имени OMBRI, прежнее имя принимается с предупреждением.
- Имя СЕКРЕТА — подстановка (`_BOT_TOKEN_SECRET` / `_INGEST_BOT_TOKEN_SECRET`), и её дефолт обязан указывать на СУЩЕСТВУЮЩИЙ секрет: привязка к несуществующему роняет ВЕСЬ деплой (`§−101`).
- `_BOT_TOKEN_SECRET` и `_BOT_USERNAME` переключаются ТОЛЬКО ПАРОЙ: хэндл вшит в статический отчёт, и кнопка «Применить идею» поведёт не к тому боту (`§−101`).
- Имена продукта — ТОЛЬКО `branding.py` (OMBRIOS платформа ≠ OMBRI бот); слово `ramp` — ЧЕТЫРЕ разные сущности, замена по слову ломает `scoring._ramp` (`docs/roadmap/rename_ombrios/`).
- Кольцо «загрузчик → отчёт» держится на равенстве ИМЕНИ ОБЪЕКТА: `QUOTES_PREFIX`
  + `DB_OBJECT_NAME` = путь под точкой монтирования в `STOOQ_DB_PATH`. Разъехались —
  оба бота живы, файлы разные, срезы уходят в пустоту (`§−100`, тест в `phase51`).
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
- Меняешь `analyze_all` → golden-фикстура (`tests/test_contracts_golden.py`) обязана
  совпасть; расхождение в фазе Арх-3 = ошибка разреза, а НЕ повод обновить эталон. Сам
  `analyze_all` — ОРКЕСТРАТОР: тело ≤ 150 строк, порядок стадий и их полнота пинятся
  (`test_engine_orchestrator.py`), новая логика = новая СТАДИЯ, а не блок в оркестраторе.
- Эталон СЛЕП к порядку ключей словаря (`normalize` сортирует ради стабильности).
  Порядок, несущий смысл, пинится отдельным тестом — как `test_engine_benchmark_order.py`
  для профильного бенчмарка, чей первый ключ доезжает до подписи карточки (`§−64`).
- ПРИСТРОЙКИ, которые прод (`entrypoint`, `tg_bot`) НЕ импортирует, иначе его
  деплой начнёт от них зависеть: `ingest_bot`/`ingest_access`/`services/quote_*`
  — бот-загрузчик, дефолт хранилища офлайн (`QUOTES_BACKEND=local`), дельта не
  заводит бумаг (`test_phase51_ingest_bot.IsolationTest`); `freedom-etl/` — свой
  образ и зависимости, `src/` в него не копируется. Дублирование хелперов env
  там осознанно — тот же случай, что `cloud_function/rag_engine.py`.
- ДВА РАЗНЫХ ПРОЕКТА, не смешивать: `roadmap/manual_portfolio/` (ручной ввод, источник
  Stooq) и `roadmap/freedom_warehouse/` (Freedom API → своя БД). Основания разные
  (I-12/I-14): `manual` не вправе читать витрину с `origin='tradernet'`, как и сам Tradernet.
- База котировок Stooq: ПИШУТ двое — `stooq_ingest` у оператора и бот-загрузчик
  (скачал → применил → залил ЦЕЛИКОМ, CAS по поколению; 412 = отказ, не ретрай).
  Отчётный бот открывает `mode=ro` и читает ЛОКАЛЬНУЮ КОПИЮ (`stooq_store._local_copy`): SQLite поверх
  gcsfuse — сотни range-запросов на отчёт. Формы символа меняют НОТАЦИЮ, но не
  ПЛОЩАДКУ: `{base}.US` для иностранной бумаги подсунет ADR (`§−77`). Свежесть —
  в ТОРГОВЫХ днях рынка бумаги, не в календарных.
- Синк ChromaDB и бут-ингест идут ОДИН РАЗ за старт (`entrypoint._main`, ПОСЛЕ
  тяжёлых импортов): залил PDF в INBOX → **рестартни бот**, иначе печатаются старые
  «N отчётов · M чанков» (`§−93`; проверка — `scripts/rag_inventory.py --from-gcs`).
- **Отказ брокера — ТРИ причины** (`_ramp_fallback_reason`): `waf_block` (WAF по IP,
  САМО не пройдёт), `api_error` (пройдёт), `parse_error` (НАШ дефект). «5–15 минут»
  честно только для `api_error`; `_ramp_is_mock`/`_ramp_is_fallback` не трогать (`§−94`).
- **Бот гоняет ONNX-эмбеддинг в СВОЁМ контейнере** на 2Gi/1CPU: deploy-шаг несёт
  тот же кап потоков, что и RAG-функция (`§−94`).
- **Чанк RAG режется по границе фразы**, секцию начинает заголовок ЛЮБОГО
  уровня, а порог отсеивает бессодержательное, а не короткое (`§−97` E-4).
  Отрывки считает `rag_engine.count_snippets` (по заголовкам, не по абзацам).
- **Имя эмитента — ОДИН реестр** (`rag_engine.BANK_*`), новый банк = одна строка.
  Копий было четыре, и пять банков молча стали `Unknown` на конвенции
  `wells_fargo_*.pdf`: `_` для `\b` СЛОВНЫЙ символ (`§−95`). Короткие формы
  (`MS`/`GS`) — только имя файла и тег, в прозе нельзя (`§−14` C-8).
- **Отчёт не называет источник, которого нет в базе**: имена — из `rag_banks`/
  `meta.ragBanks`; нечего назвать — молчи. В BASE стояли ТРИ выдуманных отчёта (`§−95`).
- **Статус/тон карточки — ВЕРДИКТ, а не оформление**: считает `scoring.kpi_status`,
  маппер ЧИТАЕТ. Литерал запрещён — Sharpe 0.56 годами носил «good» (`§−90`);
  неизвестный вход → «внимание», НИКОГДА не «ok».
- **Пустая коллекция провенанса ≠ «всё хорошо»**: пустой `fx_conversion` означал
  и «нечего конвертировать», и «курса не дали» — оба зелёным (`§−90` A-3).
- **Один цвет — один факт.** Свежесть данных и направление величины разводятся по
  разным каналам (`tone` против `trendTone`), иначе замедляющийся ВВП зеленеет.
- **Новый ключ design-data обязан иметь потребителя в бандле тира.** `kpis` жил в
  BASE-payload и не рендерился ВООБЩЕ (`§−90` A-2). Тест на это читает
  `src/premium_assets/*.js` — поставляемый артефакт, а не исходник `.jsx`.
- **Карточка KPI — ОДИН файл на оба тира** (`design/premium_v2/shared-kpi.jsx`):
  две копии уже разошлись, BASE терял график и заметку ИИ (`§−97`).
- **Ось дивидендов Stooq НЕ измерена**: инструмент есть
  (`measure-dividend-axis`), замер за оператором — нужен второй источник.
- **Запрет в промпте неисполним без факта**: состояние движка обязано доехать до
  модели. Обратное тоже — правила вне промпта модель не исполняет (`§−97` E-7).
- **Разность двух чисел модель не считает** — наклон против бенчмарка приезжает
  готовой строкой `summary.factor_tilt_text` (`§−97` E-5).
- Мобильная вёрстка правится ПО ЗАМЕРУ (headless 320/360/390/414), а не на глаз;
  детектор меряет по границе КОМПОНЕНТА — `scrollWidth` обнулён страничным
  `overflow-x: hidden` (`§−97` E-6) — и отсеивает свёрнутые аккордеоны.
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

Три границы конвейера, все три объявлены: `results{}` (`finance/contracts.py`) →
payload (`pdf_payload.PAYLOAD_CONTRACT`) → design-data (`test_phase19_block_audit`).

## Числа и константы — где SSOT

| Что | Источник правды |
|---|---|
| Тариф тиров (base 1 · scenario 1 · deep 2) | `tg_bot.TIER_COST` |
| Цена токена (2 500 ₸; пакет 10 = 25 000 ₸) | `tg_bot.TOKEN_PRICE_KZT` / `TOKEN_PACK_PRICE_KZT` |
| Окно истории (1825 календарных дней ≈ 5 лет) | env `HISTORY_LOOKBACK_DAYS` → `investment_logic.get_market_data` |
| Модели LLM (Sonnet 5 base / Opus 4.8 deep) | env `ANTHROPIC_MODEL_BASE` / `ANTHROPIC_MODEL_DEEP` |
| Риск-индекс 0–100 | `finance/scoring.composite_risk_score` |
| Классификация инструмента | `finance/asset_taxonomy.py` |

- **Модели Anthropic ОТВЕРГАЮТ `temperature` (HTTP 400)** — параметр не передаётся
  (`ai_narrative._TEMPERATURE_UNSUPPORTED_PREFIXES`); разнообразие даёт промпт.
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
- Ничто в `docs/` не читается в рантайме: пути в комментариях — документация.
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

- Мелкие изменения; не смешивай рефакторинг с правкой поведения в одном PR.
- Общие дефолты — в `.claude.json`; машинные оверрайды — в `.claude/settings.local.json`.
- Не переписывай этот файл автоматически: обновляй осознанно, когда изменились правила.
