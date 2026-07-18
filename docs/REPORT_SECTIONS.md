# REPORT_SECTIONS.md — карта секций отчётов BASE и DEEP

> Назначение: **изменить любую секцию отчёта за один заход** — найти строку в этом файле,
> открыть builder и шаблон, поправить, прогнать тест. Поток данных всегда один:
>
> `analyze_all()` (движок, `src/finance/investment_logic.py`)
> → `build_payload()` (адаптер, `src/pdf_payload.py`)
> → Jinja-шаблон (`src/templates/report_basic_v3.html` / `report_deep_v3.html`)
> → HTML по подписанной ссылке.
>
> ИИ-поля приходят отдельной веткой: `generate_narrative()` (`src/ai_narrative.py`) → `ai_summary` → те же payload-ключи.
> Обновлено: **2026-07-17** (B1: мандатный бенчмарк в «Факторном разложении» — `benchmark_factor_profile`/`benchmark_name`/premium `benchmarkName` (DEEP-контракт 32→34 ключа), динамические подписи в `deep-factors.jsx` и v3-фолбэке; ИИ-идеи фильтруются мандатными лимитами 0–0). Ранее: 2026-07-09 (ревью прод-отчётов: BASE→Sonnet 5 + currency-rule; CoVe 24→16 строк; кнопки «Открыть бумагу»/«Применить идею»→Scenario deep-link; regime RAG — банк-источник + ✓/⚠), 2026-07-04 (RAG-наблюдаемость), 2026-06-16 (Sprint 6 · BLOCK 1–4).

## Легенда
- **Ключ** — ключ в `payload` (в шаблоне читается как `data.<ключ>`).
- **Builder** — функция, которая формирует данные секции.
- **Движок** — где считается первичная математика (если есть).
- Поиск секции в шаблоне: `grep -n "<ключ>" src/templates/report_*_v3.html`.

---

## 1. Обложка / KPI (BASE + DEEP)

| Секция | Ключи | Builder | Движок |
|---|---|---|---|
| Вердикт на обложке + инсайты | `ai_verdict`, `ai_bullets[]`, `ai_plain_summary` | прокидка из `ai_summary` | Sprint 5.2: H1 ← `ai_verdict`, `ai_bullets` рендерятся. Sprint 5.3: вердикт ≤1 предложения / summary ≤2, простой язык (soft-trim 150/230). 360°-аудит 06-14: `_soft_trim` режет по границе СЛОВА (не мид-слово) |
| **Соответствие мандату** (в блоке вердикта) | `mandate_compliance{rows,breaches,compliant,leveraged,margin_debt_pct}` | `_build_mandate_compliance` + партиал `_mandate_compliance.html` | Sprint 5.4: панель стоит ПОСЛЕ комментариев ИИ (в `cover-main`); адаптивная карточка с мини-барами (полоса лимита + маркер факта), мобильный `@media`, «допустимое отклонение» вместо «tracking error» |
| Риск-гейдж 0–100 + мандат-бейдж | `risk_pct`, `risk_label`, `risk_mandate_label` | `build_payload` (верх), `_risk_mandate_label` | `scoring.composite_risk_score` (веса по мандату `_RISK_MANDATE_MATRIX`); бейдж = реальное имя профиля при `user_profile` (5.2); `risk_pct` clamp `[0,100]` (360°-аудит 06-14, чтобы стрелка гейджа не перекручивалась) |
| KPI-карточки CVaR / Sharpe / MaxDD | `cvar`, `cvar_ci`, `sharpe`, `sortino`, `max_drawdown`, `volatility`, `*_dollar` | `build_payload` (KPI-блок) | `investment_logic` (bootstrap-CVaR Politis-Romano, Sortino H-4) |
| AI-комментарии к KPI | `ai_cvar_note`, `ai_sharpe_note`, `ai_mdd_note` | прокидка из `ai_summary` | `ai_narrative` (структурированный tool `emit_report`) |
| MoM-дельта риска | `prev_risk_score`, `risk_score_delta` | `build_payload` + `prev_snapshot` | `db_tokenomics.get_last_report_snapshot` |
| KPI-спарклайны | `kpi_sparklines` | `tg_bot._build_kpi_sparklines` | — |

**Как менять:** формулы → движок; формат/подписи → `build_payload`; вид → шаблон (анкор `kpi-grid`).

---

## 2. Рыночный режим и макро-контекст (DEEP; в BASE — только `ai_regime_comment`)

| Под-секция | Ключи | Builder | Примечания |
|---|---|---|---|
| Квадрант Growth×Cycle (SVG) | `regime.dot_cx/dot_cy/dot_label`, `regime.label/growth/cycle/confidence/explainers` | `pdf_payload._regime_dot_coords` + regime-блок в `build_payload` | Точка **живая** (Sprint 5/R1): центр (155,155), 625 px/ед, клэмп в рамку. Сам классификатор: `finance/regime.py` |
| Сигналы-драйверы (FRED) + темп | `macro_drivers.series[].{value,trend_label,trend_dir}` | `_build_macro_drivers_panel` + `_macro_series_trend` | Источник: `services/macro_data.py` (6 серий: 10Y−2Y, HY OAS, VIX, breakeven, **unemployment, GDP**). **F3/2.3:** чип темпа (▲/▼/▬ + Δ) — **OLS-наклон по ≥3 изменениям** (общий `regime.series_trend`), level ⊕ rate-of-change |
| **Smart Money · инсайдеры (SEC Form 4)** | `smart_money.{status,enabled,rows[],headline,hint}` | `_build_smart_money` (+ блок B2.4 в deep-шаблоне) | `finance/smart_money.py` (gated `SMART_MONEY_INSIDERS`); видна всегда: active-таблица или плашка «источник не активирован». Архитектура → `SMART_MONEY.md` |
| Детерминированная сверка FRED↔моментум | `regime_consistency.status/note/signals` | `_build_regime_consistency` | Sprint 5/R3: пороги — инверсия<0, HY>5.5%, VIX>25 |
| AI-подтверждение режима | `regime_confirmation.stance/summary/signals` | прокидка из `ai_summary` | DEEP-only; ✓/⚠/✗ |
| RAG-подтверждение (+ банк + ✓/⚠) | `regime_rag_confirm[]` → `regime.ragSignals[]{text,bank,ok}` | `tg_bot._fetch_rag_context` → `premium_payload` | 2026-07-09 (#5): каждая выдержка несёт **банк-источник** (восстановлен из retrieval-хедера `— БАНК · секция`) + **чек-пойнт** ✓/⚠ (⚠ только когда `confirmStance=diverges`). **R2#3:** выдержка строится из ЦЕЛОГО блока чанка через `_clean_rag_excerpt` (обрезка по границе слова + «…», без «ury…»/«…mod-»), n_results=6 с приоритетом **разных банков** (было 3 выдержки одного JPMorgan). Пустая база → фолбэк на моментум-объяснители (`ragBacked=false`) |
| Детерминированная сверка FRED↔моментум (текст) | `regime_consistency.note` | `_build_regime_consistency` | **R2#2:** нота теперь НАЗЫВАЕТ проверенные сигналы (кривая 10Y−2Y, HY-спред, VIX, инфляц. ожидания + значения), а не безымянное «согласуются» |

**Как менять:** сигналы классификатора → `regime.py` (`SHORT/MEDIUM_WIN`, компоненты осей); пороги сверки → `_build_regime_consistency`; геометрия SVG → `_regime_dot_coords` + анкор `qExpansion` в deep-шаблоне.

> **Sprint 6 / BLOCK 3.4 — макро-обогащение:** FRED-каталог расширен до **6 серий** (+ `UNRATE` безработица, `A191RL1Q225SBEA` Real-GDP SAAR; `macro_data.py`). `RegimeClassifier.classify(prices, macro=…)` имеет **gated overlay** (env `REGIME_MACRO_OVERLAY=1`, OFF по умолч.): GDP→growth, безработица→cycle, аддитивно как доп-компоненты осей, ограничено `±0.05`. Тюнинг → константы `_MACRO_MAX_NUDGE/_TREND_GDP_GROWTH/_NEUTRAL_UNEMPLOYMENT` в `regime.py`.

---

## 3. Состав портфеля / риск-концентрация (BASE + DEEP)

| Под-секция | Ключи | Builder | Движок |
|---|---|---|---|
| Таблица активов (вес, β, TRC, P&L, action) | `assets[]`, `hotspots[]` | `build_payload` (per-asset цикл) | Эйлер-TRC: `calculate_structural_risk`; 🔥 hotspot: TRC > `scoring.HOTSPOT_TRC_PCT` (SSOT, Sprint 5.1/S4) |
| ~~Кнопка «Открыть бумагу»~~ (BASE) | — | — | **R2#7: УДАЛЕНА** — TradingView-ссылки не работали в проде; кнопка убрана из раскрытой строки `portfolio-holdings.jsx` целиком |
| Секторный пай + бейдж плеча | `sectors[]`, `pie_chart_data[]`, `leverage_metrics` | `build_payload` (sector-блок) | Плечо: `investment_logic` (леверидж из отрицательного кэша). Бейдж подписан: Gross=(лонг+\|маржа\|)/капитал, Плечо=лонг/капитал (Sprint 5.1/L1) |
| Концентрация HHI + warnings | `sector_concentration`, `asset_concentration`, `sector_warnings`, `sector_groups`, `sector_complex` | `_build_concentration`, `build_sector_groups` | SSOT супер-группы «Tech-комплекс» |
| Риск-водопад | `risk_waterfall` | `_build_risk_waterfall` | standalone vol vs диверсифицированный |
| Панель «Соответствие мандату» | `mandate_compliance.rows[]/breaches/compliant/leveraged/margin_debt_pct` | `_build_mandate_compliance` (+ партиал `_mandate_compliance.html`) | Sprint 5: классы vs limits_dict; Sprint 5.1/L2: строка маржинального долга, leveraged ⇒ не compliant |
| Margin-Call предупреждение ИИ | `ai_leverage_warning` | прокидка из `ai_summary` | триггер `has_leverage` в промпте |

**Как менять:** классификация классов активов → `agent/gatekeeper._classify_to_asset_key` (кэш = `Cash`, не Bonds); лимиты мандата → `profile_manager._PROFILE_MAP`; вид панели → партиал.

---

## 4. 4-Pillar Scoring (DEEP-таблица; колонки Score в BASE)

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Очки F/V/T/C + Total + Action | `assets[].score/action`, DEEP `score_breakdown` | `build_payload` | `scoring_orchestrator.score_portfolio` → `scoring.py` |
| F-пиллар (ROE/OpM/D-A/RevG/FCF) | — | — | z-каскад `_sector_z`: (1) когорт портфеля ≥5 → (2) **live SEC-когорт лидеров сектора** (`_dynamic_sector_cohort`, Sprint 5.1/S2; off: `SECTOR_COHORT_DISABLED=1`) → (3) статик-таблица 2020-25 (логируется) |
| V-пиллар (P/E, P/B, FCF-yield) | — | — | `_compute_valuation_ratios` + `_SECTOR_*_BENCHMARKS`; FCF-yield — signed-z (Sprint 5) |
| C-пиллар (CDS/Altman/Piotroski) | `fundamental_layer[]` | `build_payload` | `sec_edgar.py`, `cds_feed.py`; N/A-классы → нейтраль + прочерк |
| Hotspot-порог | — | — | `scoring.HOTSPOT_TRC_PCT` = 20.0 — **единственное место тюнинга** (использует и отчёт, и Gatekeeper GK-1) |

**Как менять:** состав live-когорт → `_SECTOR_REPRESENTATIVES`; статик-бенчмарки → `_SECTOR_FUNDAMENTAL_BENCHMARKS` / `_SECTOR_PE/PB/FCF_YIELD_BENCHMARKS`; правила очков → `scoring.fundamentals_score`/`valuations_score`/`credit_score`.

---

## 5. AI Ideas («Идеи ИИ» в BASE / idea-карточки в DEEP)

| Элемент | Ключи | Builder | Источник |
|---|---|---|---|
| Карточки идей (4 сценария) | `ai_ideas{growth/diversification/hedge/rotation}`, `ideas_count` | `_build_ai_ideas` | `ai_summary.stock_picks` (ключи `boost_alpha/rebalance/protect_capital/`**`smart_money`**) — 4-я карточка **Smart Money** (институционалы+инсайдеры) вместо «режима» (06-23); рендерится в bucket `rotation` |
| Генерация пиков (LLM) | — | — | `ai_narrative._user_prompt`: **DATA-DRIVEN** (5.1) + **СВЕЖЕСТЬ ИДЕЙ** (6.2/1.1 — **ДНЕВНОЙ** якорь `YYYY-MM-DD` + дневной `УГОЛ РОТАЦИИ` из `_IDEA_ROTATION_ANGLES`, бан расширен на V/MA/GS/UNH/PG/AVGO). **Routing (2026-07-09): BASE=`claude-sonnet-5`, DEEP=`claude-opus-4-8`** — оба опускают `temperature` (400 при передаче), дисперсию даёт директива + дневной угол; env-override на `claude-sonnet-4-6` возвращает temperature-band 0.5–0.85 |
| Фильтры пиков | — | — | `_remove_held_picks` → `_check_pick_contradictions` → `_backfill_empty_scenarios` |
| Фолбэк-каталог (без API) | — | — | `_fallback_stock_picks`: 3 кандидата/слот + **месячная ротация** (Sprint 5.1) |
| **Кнопка «Применить идею»** (2026-07-09 #3; BASE+DEEP) | `meta.botUsername` (env `BOT_USERNAME`, default **`KEN_investment_bot`** — R2#4) | `pdf_payload.bot_username` → `premium_payload.meta.botUsername`; UI `ApplyIdeaModal` в `portfolio-ideas.jsx` (BASE) и `deep/deep-plan.jsx` (DEEP) | Модал: 4 идеи → выбор → «Сделать Scenario Analysis для идеи «…» — спишется 1 токен» ДА/НЕТ. **ДА** → deep-link `t.me/<bot>?start=scn_<n>`. Статичный HTML НЕ списывает токен — списание в боте: `tg_bot.cmd_start` ловит `scn_`, ведёт в сценарный tier через `kb_confirm("scenario")` → `cb_confirm`. **R2#4:** в DEEP кнопка перенесена из шапки Action Plan в баннер «AI Ideas · 4 идеи» (рядом с самими идеями) |

**Как менять:** правила промпта → `ideas_rule` в `_user_prompt`; каталог → списки `_BOOST_*`/`_BALANCE*`/`_PROTECT`/`_REGIME_*`; пайплайн-подписи карточек → `_PIPELINE_BASE/_PIPELINE_DEEP` в `pdf_payload`; модал/deep-link → `ApplyIdeaModal` + `scenarioDeepLink` в JSX, обработчик `scn_` → `tg_bot.cmd_start`.

---

## 6. Action Plan (DEEP)

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Таблица Buy/Sell/Stop/Δw | `action_plan` (из results) | прокидка | `action_plan.build_action_plan` |
| Уровни (зоны, тейк, стоп) | — | — | `compute_levels`: ATR+SMA+52w-high; **mandate-scale** ATR-дистанций 0.75/1.0/1.25 (`MANDATE_LEVEL_SCALE`, Sprint 5.1/A3); SMA-якоря без масштаба |
| Турновер-кап | — | — | `MAX_TRADE_BLOCK_PORTFOLIO_PCT=0.25` (+ BL `max_active_share` по мандату) |
| AI-комментарий плана | `ai_action_comment`, `action_plan_text`, `ai_action_impact` | прокидка из `ai_summary` | — |

**Как менять:** множители уровней → константы `ATR_*` и `MANDATE_LEVEL_SCALE`; приоритет сортировки → `priority`-словарь в `build_action_plan`.

---

## 7. Ожидаемый эффект (DEEP, 8 карточек)

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Карточки before→after | `expected_effect.{risk_index,vol,cvar_95,sharpe,max_drawdown,max_erc_pct,it_share,expected_return}` | `_build_expected_effect` (флэттенер) | `simulate.simulate_after_plan` под BL-таргетами |
| Цвет дельты | `favourable` | — | `simulate._delta_row`: lower-better = {vol, max_trc, risk_index}; **`it_share` нейтрален** (Sprint 5.1/A2 — направление зависит от мандата); нулевая дельта нейтральна |
| Вердикт ребаланса | `expected_effect.verdict` | прокидка | `simulate` (improvement/tradeoff/degradation/neutral — по vol/TRC/MaxDD) |
| Веса до/после | `weight_changes` | прокидка | BL-таргеты, мандатные ограничения (`_MANDATE_BL_CONSTRAINTS`) |
| Область эффекта (какие позиции меняем) | `expected_effect.high_priority_tickers` → `effectScope[]` / `effectScoped` | `premium_payload._map_deep` | **R2#5:** плашка «Меняем позиции: T1 · T2 …» стала DATA-DRIVEN (была захардкожена «Δ по идеям: MSFT · ORCL …»); показывает тикеры высокоприоритетных строк Action Plan — по ним и посчитан «до/после» |

**Как менять:** направление метрики → `_RISK_METRICS_LOWER_IS_BETTER` / `_NEUTRAL_METRICS`; карточный маппинг → `_KEYMAP` в `_build_expected_effect`; рендер → макрос `_ef_card` (deep-шаблон).

> **Sprint 6 / BLOCK 2.3 — связка Идеи→Action→Эффект:** симуляция идёт на **высокоприоритетных** action-строках (не-deferred Buy/Sell/Trim, `|Δw|>0`), а не на полном BL-векторе — `finance/simulate.high_priority_target_weights()` → `(target, tickers, actions)`. **06-25:** если идея вышла «только продажи» (турновер-кап заполняется продажами первыми), освободившийся вес **реинвестируется** в топ-BL-покупки (только УЖЕ держимые имена → они в ковариации → метрики реально отражают покупку), ≤+12пп на имя, остаток в кэш → панель показывает И что продаёшь, И что покупаешь, и концентрация реально падает. Payload: `expected_effect.high_priority_actions[]` (`{ticker, side=Продать/Купить, delta_pp}`) + `scoped_to_high_priority`. `ai_effect_comment` ОБЯЗАН согласовать с `verdict.kind` (tradeoff → не заявлять односторонне снижение риска).

---

## 8. Бенчмарки и доходность (BASE + DEEP)

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Карточка «vs рынок» | `scenarios[]` (excess, TE, IR, bm_return) | `_build_scenarios` | `_compute_benchmark_stats` (pair inner-join) |
| Выбор главного бенчмарка | фильтр `user_bench_ticker` → слот «Профильный бенчмарк» | `build_payload` | Выбор юзера активирован (Sprint 5/Task 8): `tg_bot._resolve_bench_ticker` |
| Мульти-период 1м/3м/6м/12м/YTD | `period_returns_table` | `_adapt_period_returns` | `period_returns.py` |
| **«Рост против рынка» (BASE) — метка бенчмарка (B1-perf 2026-07-18)** | `performance_benchmark_name` (payload) → premium `performance.benchmarkName` → `portfolio-performance.jsx` (`bench`) / v3 `_bench_keys[0]` | `build_payload` переименовывает ключ «Профильный бенчмарк» в `period_returns_table` в реальное имя (`BENCHMARK_LIST[user_bench_ticker]`) | Цифры карточки и так брались из профильного бенчмарка (первый ключ таблицы), но подпись «S&P 500» была захардкожена в компоненте в 5 местах → отчёт показывал данные Nasdaq под меткой S&P. Метка теперь динамическая; фолбэк (нет профиля) — первый конкретный ключ / «S&P 500». `benchmark_comparison`/gatekeeper/equity-curve читают `results[...]`, а не payload — не затронуты. Тест `BasePerformanceBenchmarkTest` |
| Equity curve / факторный радар | `equity_curve_svg`, `factor_radar_svg`, `factor_betas` | `tg_bot._build_*` | Ridge-беты движка |
| **Бенчмарк в факторной таблице (B1 2026-07-17)** | `benchmark_factor_profile{ticker,name,betas}` (results) → `factor_betas[].bench` → `benchmark_name/benchmark_ticker` (payload) → premium `benchmarkName/benchmarkTicker` | движок `fit_factor_betas` (analyze_all) → `tg_bot._build_factor_betas_table` → `pdf_payload` → `premium_payload` → `deep-factors.jsx` | Столбец сравнения = РЕАЛЬНЫЕ факторные беты мандатного бенчмарка (тот же Ridge-пайплайн: панель `factor_tickers`, ортогонализация BLOCK 3.5, α=0.001, guard ≥max(2K,30) obs). `Наклон Δ = β_портф − β_бенчмарка`. Фолбэк (нет бенчмарка/истории): S&P-константа `(Market=1, rest=0)` **вместе с подписью «S&P 500»** — подпись всегда совпадает с числами (Вариант A ADR; Вариант B «подменить ось Market» отклонён — сломал бы стресс/BL/Euler/κ; Вариант C «только переименовать» отклонён как вредный). Классик-v3 фолбэк-шаблон тоже динамический (`data.benchmark_name`). Тесты: `test_phase31_benchmark_factor_propagation.py` |
| **Источники риска (декомпозиция дисперсии)** + факторные двойники (DEEP) | `factor_variance{rows,systematic_pct,idio_pct,twins}` → premium `factorVariance` | `pdf_payload._build_factor_variance` | **Additive layer** `finance/factor_decomposition.py`: Euler по ФАКТОРАМ (σ²=bᵀFb+wᵀDw, b=Bᵀw), группы «Рыночная бета/Стили/EM/Ставки/Сырьё/Идио»; «двойники» = пары с systematic-corr ≥ 0.90 (`TWIN_CORR_THRESHOLD`); отрицательная доля = фактор-хедж. Хук: `calculate_structural_risk` → `portfolio_metrics["factor_decomposition"]` (graceful {}). AI: `_factor_decomposition_for_prompt` → 4-шаговый `ai_factor_comment` (источник риска → двойники → наклон vs режим → чего не хватает) + детерминированный фолбэк `_fallback_factor_comment` |

---

## 8.1. Логика Action Plan · Effect · риск-гейдж (аудит 2026-07-18)

| Находка | Было | Стало |
|---|---|---|
| **Action Plan: «действие» ≠ «количество»** | Действие — из 4-Pillar Score, количество/Δw — из Black-Litterman; при расхождении знака рисовался абсурдный «+1 шт» под SELL | `build_action_plan`: при противоречии знака BL с действием (Sell/Trim но Δw>0, Buy но Δw<0) → `qty_delta=None` («—») + пометка «BL расходится с сигналом (+Xпп)»; согласующиеся знаки не тронуты; muted BL-Δw честно показывает мнение оптимизатора |
| **Effect: не видно, что продаётся/покупается** | Панель показывала лишь чип «Меняем позиции: A · B · C» | `high_priority_actions` → premium `effectActions` [{t,side,key,dw}] → `EffectGrid`: явные блоки **Продать/сократить** (красный) и **Купить/нарастить** (зелёный) с Δпп; метрики «до/после» ниже — результат именно этих сделок. DEEP-контракт 34→35 |
| **Риск-гейдж занижен** | Композит (vol/CVaR/single-name-ERC) игнорировал плечо, MaxDD и секторную концентрацию → 73%-tech книга с плечом и −43.5% просадкой = «48 · Умеренный» | `composite_risk_score` +bounded аггравторы (opt-in, keyword-only, default None → без изменений): MaxDD 20→50%=+15, сектор 50→75%=+10, плечо 1.0→1.5x=+6; **только повышают** гейдж. `analyze_all` переопределяет composite после sector/leverage; `simulate` получает те же mandate+аггравторы → verdict-гейдж == effect-before (нет рассинхрона). Живая книга 48→70 «Агрессивный». Чистая диверсифицированная без плеча — без изменений. Тесты `test_phase32_report_logic_fixes.py` |
| **ИИ-цепочка не помнила бенчмарк/мандат** | verdict→holdings→factor→stress→effect→action без явной оси бенчмарка/мандата | В «ЦЕПОЧКУ АНАЛИЗА» обоих промптов вплетены бенчмарк (факторный перекос ОТНОСИТЕЛЬНО него) и мандат (концентрация vs лимиты классов = нарушение; Action Plan обязан вести долю обратно в лимит); `mandate_rule` теперь несёт лимиты классов + целевую vol |

## 9. Стресс-тесты (DEEP)

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Сценарии (7 шоков) | `stress_scenarios[]` | прокидка | `stress.py`: β×шок с выпуклым капом T=0.20→C=0.35 |
| AI-комментарий | `ai_stress_comment` | `validate_stress_comment` (гарантия упоминания капа) | — |

---

## 10. Прозрачность / QC (BASE + DEEP)

| Элемент | Ключи | Builder |
|---|---|---|
| Integrity-панель ✓/⚠ | `integrity_checks[]` | `_build_integrity_checks` — RAG 3-state (used/no_match/unavailable) **+ инвентарь базы** (`M отчётов · K чанков`, 2026-07-04) **+ пилл «ИИ↔банк-аналитика»** (⚠ когда ИИ ссылается на банки при пустой базе). Футер красит собственный символ строки (fix двойного `✓`) |
| CoVe data-lineage (**16 строк**, консолидировано 2026-07-09; было 24) | `cove_lineage[]` | `finance/data_lineage.build_lineage`. **Консолидация #6** — объединены похожие по логике: (1) **Риск-метрики + Euler-TRC/MCTR** → 1 строка (одна ковариация MAC3); (2) две SEC-строки (Z-scores + Altman/Piotroski/Coverage) → 1; (3) **6 FRED-серий** → 1 агрегат «Макро-драйверы (FRED)» (бейдж = худший статус, note перечисляет проблемные); (4) два LLM-чекера → 1. Остальные строки: факторная независимость κ+max\|corr\| (4.6), Tradernet-цены, валютный слой FX+ставка, CDS, Action levels, Black-Litterman, режим, стресс, Smart-Money (gated), **Bank RAG** (+ инвентарь база: отчёты/чанки/отрывки), **ИИ-цитирование банк-аналитики** (проверенные [RAG]-цитаты vs. банк-консенсус из памяти), AI-вердикт. **R2#6:** «база: N отчётов/чанков» считает РЕАЛЬНЫЕ отчёты — `list_documents()`/`_fetch_rag_context` исключают tmp-артефактные `source` (`_is_temp_source`), а boot-ingest их физически чистит (`purge_temp_sources`); было 48 вместо 29 из-за pre-fix `tmpXXXX.pdf`-имён |
| Data quality | `data_quality` | `build_payload` (факторы N/10, SEC-пропуски) |
| AI-вердикт/буллеты | `ai_verdict`, `ai_plain_summary`, `ai_bullets` | прокидка из `ai_summary` |

---

## 11. Scenario Analysis (отдельный тир · 1 токен · 2026-07-07)

> Тир `scenario` — **самостоятельный отчёт**, НЕ страница BASE/DEEP. Детерминирован
> (0 LLM-вызовов). Поток: `analyze_all()` → `finance/scenario_report.build_scenario_payload`
> (переиспользует `finance/scenario_engine`) → `templates/report_scenario_v3.html`
> (свой Jinja, `html_renderer` маршрутизирует scenario минуя Premium).

| Панель | Ключи (`data.scenario.*`) | Движок |
|---|---|---|
| A · 5 core-метрик | `metrics{ann_return,vol_cov,vol_gross_ref,sharpe_rfr,beta,div_yield,pe,rfr}` | `scenario_engine.five_metrics` (σ_p ковариационная, RFR-Sharpe, Σwβ) |
| A · Euler-MCTR вклад в риск | `mctr_rows[]{display,weight,sigma_i,rho_ip,mctr,ctrisk,pct_ctr}`, `vol_cov` | `scenario_engine.mctr_table` (MCTR=ρ·σ, Σ CTRisk=σ_p, Σ%=100) |
| A · Выживаемость 3 макро-режима | `regime_survival[]{regime,label,avg_pct,survives}` | `scenario_engine.regime_survival` (группировка 7 шоков; **билдер нормализует `stress.port_pct` долю→%**) |
| A · Funding-звенья | `funding[]{display,weight,sharpe,flags[]}` | `scenario_engine.funding_candidates` (Sharpe<0.5 · доходность<0 · corr-дубль≥0.90 · TRC-несоразмерность) |
| B · Walk-forward бэктест | `backtest{ticker,rule,summary{n_signals,hit_rate_63d,horizons}}` | `scenario_engine.walk_forward` (правило «цена>200-DMA», look-ahead guard, `DISCLAIMERS`) |

**Токен-тариф:** `tg_bot.TIER_COST` = base 1 · scenario 1 · deep 2. Кнопка тира в
`kb_analysis_choice` + inline-CTA `scenario:cached` под BASE/DEEP (кэш `results` per-user).
**Как менять:** математика → `finance/scenario_engine.py`; сборка → `scenario_report.py`;
вид → `report_scenario_v3.html`. Тесты: `tests/test_phase24_scenario_report.py`.

---

## Чек-лист изменения секции
1. Найти секцию здесь → открыть builder (`pdf_payload.py`) и/или движок (`finance/*`).
2. Числа меняются ТОЛЬКО в движке; формат — в builder; вид — в шаблоне.
3. Прогнать: `python -m pytest tests/ -q` + smoke-render обоих тиров.
4. Обновить тест в `tests/test_phase*.py` (новое поведение = новый кейс).
5. Большое изменение — отметить в `AUDIT.md` (Было/Стало).
