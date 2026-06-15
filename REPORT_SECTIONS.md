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
> Обновлено: 2026-06-11 (Sprint 5.1).

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
| Сигналы-драйверы (FRED) | `macro_drivers.series[]` | `_build_macro_drivers_panel` | Источник: `services/macro_data.py` (кривая 10Y−2Y, HY OAS, VIX, breakeven) |
| Детерминированная сверка FRED↔моментум | `regime_consistency.status/note/signals` | `_build_regime_consistency` | Sprint 5/R3: пороги — инверсия<0, HY>5.5%, VIX>25 |
| AI-подтверждение режима | `regime_confirmation.stance/summary/signals` | прокидка из `ai_summary` | DEEP-only; ✓/⚠/✗ |
| RAG-подтверждение | `regime_rag_confirm[]` | `tg_bot._fetch_rag_context` | выдержки банковских PDF |

**Как менять:** сигналы классификатора → `regime.py` (`SHORT/MEDIUM_WIN`, компоненты осей); пороги сверки → `_build_regime_consistency`; геометрия SVG → `_regime_dot_coords` + анкор `qExpansion` в deep-шаблоне.

---

## 3. Состав портфеля / риск-концентрация (BASE + DEEP)

| Под-секция | Ключи | Builder | Движок |
|---|---|---|---|
| Таблица активов (вес, β, TRC, P&L, action) | `assets[]`, `hotspots[]` | `build_payload` (per-asset цикл) | Эйлер-TRC: `calculate_structural_risk`; 🔥 hotspot: TRC > `scoring.HOTSPOT_TRC_PCT` (SSOT, Sprint 5.1/S4) |
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
| Карточки идей (4 сценария) | `ai_ideas{growth/diversification/hedge/rotation}`, `ideas_count` | `_build_ai_ideas` | `ai_summary.stock_picks` |
| Генерация пиков (LLM) | — | — | `ai_narrative._user_prompt`: правило **DATA-DRIVEN** (Sprint 5.1) — идеи обязаны опираться на режим/недовесы/4-Pillar/RAG, «дежурные» имена запрещены без данных; `temperature=0.5` (env `ANTHROPIC_TEMPERATURE`, клэмп 0.4–0.6; на Opus 4.7/4.8 параметр опускается) |
| Фильтры пиков | — | — | `_remove_held_picks` → `_check_pick_contradictions` → `_backfill_empty_scenarios` |
| Фолбэк-каталог (без API) | — | — | `_fallback_stock_picks`: 3 кандидата/слот + **месячная ротация** (Sprint 5.1) |

**Как менять:** правила промпта → `ideas_rule` в `_user_prompt`; каталог → списки `_BOOST_*`/`_BALANCE*`/`_PROTECT`/`_REGIME_*`; пайплайн-подписи карточек → `_PIPELINE_BASE/_PIPELINE_DEEP` в `pdf_payload`.

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

**Как менять:** направление метрики → `_RISK_METRICS_LOWER_IS_BETTER` / `_NEUTRAL_METRICS`; карточный маппинг → `_KEYMAP` в `_build_expected_effect`; рендер → макрос `_ef_card` (deep-шаблон).

---

## 8. Бенчмарки и доходность (BASE + DEEP)

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Карточка «vs рынок» | `scenarios[]` (excess, TE, IR, bm_return) | `_build_scenarios` | `_compute_benchmark_stats` (pair inner-join) |
| Выбор главного бенчмарка | фильтр `user_bench_ticker` → слот «Профильный бенчмарк» | `build_payload` | Выбор юзера активирован (Sprint 5/Task 8): `tg_bot._resolve_bench_ticker` |
| Мульти-период 1м/3м/6м/12м/YTD | `period_returns_table` | `_adapt_period_returns` | `period_returns.py` |
| Equity curve / факторный радар | `equity_curve_svg`, `factor_radar_svg`, `factor_betas` | `tg_bot._build_*` | Ridge-беты движка |

---

## 9. Стресс-тесты (DEEP)

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Сценарии (7 шоков) | `stress_scenarios[]` | прокидка | `stress.py`: β×шок с выпуклым капом T=0.20→C=0.35 |
| AI-комментарий | `ai_stress_comment` | `validate_stress_comment` (гарантия упоминания капа) | — |

---

## 10. Прозрачность / QC (BASE + DEEP)

| Элемент | Ключи | Builder |
|---|---|---|
| Integrity-панель ✓/⚠ | `integrity_checks[]` | `_build_integrity_checks` (RAG 3-state: used/no_match/unavailable) |
| CoVe data-lineage (14 строк) | `cove_lineage[]` | `finance/data_lineage.build_lineage` — источники: Quant Engine, TRC-Euler, Tradernet-цены, **валютный слой FX+ставка** (Sprint 5.4, `_fx_status`), SEC (Z-scores + Altman/Piotroski), CDS, FRED-макро, Action levels, Black-Litterman, режим, стресс, Bank RAG, AI |
| Data quality | `data_quality` | `build_payload` (факторы N/10, SEC-пропуски) |
| AI-вердикт/буллеты | `ai_verdict`, `ai_plain_summary`, `ai_bullets` | прокидка из `ai_summary` |

---

## Чек-лист изменения секции
1. Найти секцию здесь → открыть builder (`pdf_payload.py`) и/или движок (`finance/*`).
2. Числа меняются ТОЛЬКО в движке; формат — в builder; вид — в шаблоне.
3. Прогнать: `python -m pytest tests/ -q` + smoke-render обоих тиров.
4. Обновить тест в `tests/test_phase*.py` (новое поведение = новый кейс).
5. Большое изменение — отметить в `AUDIT.md` (Было/Стало).
