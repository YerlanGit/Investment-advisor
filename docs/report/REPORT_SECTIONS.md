# REPORT_SECTIONS.md — карта секций отчётов BASE и DEEP
<!-- nav | area:report | code:src/pdf_payload.py,src/premium_payload.py,design/premium_v2/ | read-before:ЛЮБАЯ секция отчёта — карта ключ→builder→движок→шаблон -->

> Назначение: **изменить любую секцию отчёта за один заход** — найти секцию в этом файле,
> открыть builder и шаблон/JSX, поправить, прогнать тест. Документ состоит из трёх слоёв:
> **§0** — как данные превращаются в отчёт (конвейер); **§1–§11** — текущее состояние
> каждой секции (ключи → builder → движок); **§8.1** — журнал сквозной логики
> (как менялся отчёт по аудит-раундам: что было, что стало и почему).
>
> Обновлено: **2026-07-19** (раунд 29: conviction-гейт реинвеста, внешняя рука
> глобальных ETF IEF/EEM/EMB с честной расширенной ковариацией, анти-фабрикация
> чисел ИИ; раунд 28: реинвест-eligibility, CONL=крипто по underlying,
> HOTSPOT-чип по условию, анти-шаблонное правило ИИ; §8.1 переработан в журнал раундов).
> Ранее: 2026-07-18 (раунды 25–27: логика Action Plan ↔ Effect ↔ риск-гейдж),
> 2026-07-17 (B1: мандатный бенчмарк во всех секциях + /mandate-меню + тариф 2 500 ₸),
> 2026-07-09 (ревью прод-отчётов, CoVe 24→16), 2026-07-07 (Scenario-тир, lookback 5 лет).

## Оглавление

| § | Секция | Тир |
|---|---|---|
| 0 | Конвейер данных и рендеринга (+что настраивает пользователь) | — |
| 1 | Обложка / KPI / риск-гейдж | BASE + DEEP |
| 2 | Рыночный режим и макро-контекст | DEEP (в BASE — только комментарий) |
| 3 | Состав портфеля / риск-концентрация / мандат-панель | BASE + DEEP |
| 4 | 4-Pillar Scoring | DEEP (колонки Score в BASE) |
| 5 | AI Ideas + правила промптов ИИ | BASE + DEEP |
| 6 | Action Plan | DEEP |
| 7 | Ожидаемый эффект на риск (Effect) | DEEP |
| 8 | Бенчмарки и доходность | BASE + DEEP |
| 8.1 | **Журнал сквозной логики** (аудит-раунды 25–28 + B1) | — |
| 9 | Стресс-тесты | DEEP |
| 10 | Прозрачность / QC (Integrity + CoVe) | BASE + DEEP |
| 11 | Scenario Analysis (отдельный тир) | scenario |

---

## 0. Конвейер данных и рендеринга

### Основной поток (BASE и DEEP)

```
analyze_all()                      движок · src/finance/investment_logic.py
  → build_payload()                адаптер: расчёт → payload-ключи · src/pdf_payload.py
    → build_design_data()          view-map под React · src/premium_payload.py
      → premium_renderer.py        Premium V2: HTML + window.PORTFOLIO/window.DEEP
                                   + скомпилированные бандлы src/premium_assets/*.js
```

- **Premium V2 React — продовый дефолт** (`PREMIUM_REPORT_ENABLED=true` в `html_renderer.py`).
- **Классик v3 Jinja — авто-фолбэк**: `render_report_html` оборачивает Premium в try/except;
  любая ошибка → `templates/report_basic_v3.html` / `report_deep_v3.html`. v3 сознательно
  не удаляется и остаётся под тестами (`PREMIUM_REPORT_ENABLED=false` форсирует его).
- **ИИ-ветка**: `generate_narrative()` (`src/ai_narrative.py`) → `ai_summary` → те же
  payload-ключи (`ai_*`). Модели: BASE=`claude-sonnet-5`, DEEP=`claude-opus-4-8`
  (env-override `ANTHROPIC_MODEL_BASE/_DEEP`); обе НЕ принимают `temperature`.
- **Scenario-тир** идёт мимо Premium: `analyze_all()` →
  `finance/scenario_report.build_scenario_payload` → `templates/report_scenario_v3.html` (§11).

### Фронтенд-контракт Premium V2

- Исходники: `design/premium_v2/*.jsx` (BASE) и `design/premium_v2/deep/*.jsx` (DEEP).
- Сборка: `bash design/premium_v2/build.sh` (Babel + Tailwind, **CWD = корень репо**) →
  артефакты синкаются в `src/premium_assets/*.js`. Расхождение исходника и артефакта
  ловит `CompiledAssetsTest` (skip внутри Docker-образа, где нет `design/`).
- Дизайн-контракт `build_design_data`: **DEEP = 35 ключей, BASE = 13**
  (пин в `test_phase19_block_audit`; расширение контракта = осознанное изменение пина).

### Что настраивает пользователь (и как это доезжает до секций)

| Вход | Где задаётся | На что влияет |
|---|---|---|
| **Мандат** (риск-профиль, лимиты классов, бенчмарк) | онбординг-анкета; правки — `/mandate`-меню (бенчмарк за 2 тапа, классы, риск-профиль, ре-анкета; БЕЗ биллинга) | риск-гейдж (§1), мандат-панель (§3), фильтр ИИ-идей (§5), BL-ограничения плана (§6), деконцентрация Effect (§7), факторный бенчмарк и equity-curve (§8) |
| **Источник портфеля** | `/start` → подключение (freedom / демо) | демо-отчёты бесплатны; неопределённый источник не строит отчёт |
| **Тир** | `kb_analysis_choice` | base 1 · scenario 1 · deep 2 токена; **1 токен = 2 500 ₸, пакет 10 = 25 000 ₸** (SSOT: `tg_bot.TOKEN_PRICE_KZT`/`TOKEN_PACK_PRICE_KZT`) |

### Легенда таблиц §1–§11

- **Ключ** — ключ в `payload` (v3-шаблон читает как `data.<ключ>`; Premium — свой view-map).
- **Builder** — функция, формирующая данные секции. **Движок** — первичная математика.
- Поиск в v3: `grep -n "<ключ>" src/templates/report_*_v3.html`; в Premium:
  `grep -n "<ключ>" src/premium_payload.py design/premium_v2/ -r`.
- Железное правило: **числа — только в движке; формат — в builder; вид — в шаблоне/JSX.**

---

## 1. Обложка / KPI / риск-гейдж (BASE + DEEP)

| Секция | Ключи | Builder | Движок |
|---|---|---|---|
| Вердикт на обложке + инсайты | `ai_verdict`, `ai_bullets[]`, `ai_plain_summary` | прокидка из `ai_summary` | вердикт ≤1 предложения / summary ≤2, простой язык; `_soft_trim` режет по границе слова (150/230) |
| Риск-гейдж 0–100 + мандат-бейдж | `risk_pct`, `risk_label`, `risk_mandate_label` | `build_payload` (верх), `_risk_mandate_label` | `scoring.composite_risk_score` (веса по мандату `_RISK_MANDATE_MATRIX`) **+ bounded аггравторы** (раунды 25–26, opt-in keyword-only, default None → без изменений): MaxDD 20→50% ⇒ +0..15, топ-сектор 50→75% ⇒ +0..10, плечо 1.0→1.5× ⇒ +0..6 — только повышают. Концентрация — по **супергруппе** «Tech-комплекс» (`asset_taxonomy.top_sector_concentration_pct` — SSOT, 73% а не 59%). `simulate` получает те же mandate+аггравторы ⇒ **гейдж на обложке == Effect-«до»** (§7). `risk_pct` clamp `[0,100]` |
| **Соответствие мандату** (в блоке вердикта) | `mandate_compliance{rows,breaches,compliant,leveraged,margin_debt_pct}` | `_build_mandate_compliance` + партиал `_mandate_compliance.html` | классы vs `profile_manager._PROFILE_MAP`; классификация — по **underlying** (§3); панель после комментариев ИИ, мини-бары лимит/факт |
| KPI-карточки CVaR / Sharpe / MaxDD | `cvar`, `cvar_ci`, `sharpe`, `sortino`, `max_drawdown`, `volatility`, `*_dollar` | `build_payload` (KPI-блок) | **Композитная база (F-20…F-23)**: реализованные Sharpe/CAGR/CVaR/MaxDD считаются на full-panel masked composite (единая основа с остальными секциями); bootstrap-CVaR Politis-Romano; χ²/Fisher-ДИ (`finance/inference.py`); флаг `var_reliability` при короткой истории |
| AI-комментарии к KPI | `ai_cvar_note`, `ai_sharpe_note`, `ai_mdd_note` | прокидка из `ai_summary` | структурированный tool `emit_report`; анти-повтор — `style_rule` (§5) |
| MoM-дельта риска | `prev_risk_score`, `risk_score_delta` | `build_payload` + `prev_snapshot` | `db_tokenomics.get_last_report_snapshot` |
| KPI-спарклайны | `kpi_sparklines` | `tg_bot._build_kpi_sparklines` | ряды из `port_log_returns` (композитная база, F-21) |

**Как менять:** формулы → движок (`scoring.py`, `investment_logic.py`); пороги аггравторов →
`_ramp`-константы в `composite_risk_score`; формат/подписи → `build_payload`; вид → анкор `kpi-grid`.

---

## 2. Рыночный режим и макро-контекст (DEEP; в BASE — только `ai_regime_comment`)

| Под-секция | Ключи | Builder | Примечания |
|---|---|---|---|
| Квадрант Growth×Cycle (SVG) | `regime.dot_cx/dot_cy/dot_label`, `regime.label/growth/cycle/confidence/explainers` | `pdf_payload._regime_dot_coords` + regime-блок в `build_payload` | Точка живая: центр (155,155), 625 px/ед, клэмп в рамку. Классификатор: `finance/regime.py` |
| Сигналы-драйверы (FRED) + темп | `macro_drivers.series[].{value,trend_label,trend_dir}` | `_build_macro_drivers_panel` + `_macro_series_trend` | `services/macro_data.py`, 6 серий: 10Y−2Y, HY OAS, VIX, breakeven, unemployment, GDP. Чип темпа (▲/▼/▬ + Δ) — OLS-наклон по ≥3 изменениям, level ⊕ rate-of-change |
| Макро-overlay классификатора | — | — | **DEFAULT-ON** (с 2026-06-26): GDP/безработица/breakeven тилтуют оси Growth×Cycle, каждый сигнал ограничен ±0.05 — «тилт, не захват». Отключение: `REGIME_MACRO_OVERLAY=0` → байт-идентичный чисто ценовой классификатор. Тюнинг → `_MACRO_MAX_NUDGE`/`_TREND_*` в `regime.py` |
| Smart Money · инсайдеры (SEC Form 4) | `smart_money.{status,enabled,rows[],headline,hint}` | `_build_smart_money` | `finance/smart_money.py` (gated `SMART_MONEY_INSIDERS`); видна всегда: таблица или плашка «источник не активирован» |
| Детерминированная сверка FRED↔моментум | `regime_consistency.status/note/signals` | `_build_regime_consistency` | пороги: инверсия<0, HY>5.5%, VIX>25; нота НАЗЫВАЕТ проверенные сигналы со значениями, а не безымянное «согласуются» |
| AI-подтверждение режима | `regime_confirmation.stance/summary/signals` | прокидка из `ai_summary` | DEEP-only; ✓/⚠/✗. **Правило низкой уверенности** (`regime_confidence_rule`, раунд 27): при confidence <25% оценка фазы цикла подаётся осторожно, но КАЖДЫЙ сигнал (макро, рынок, банки, мандат) читается уверенно и конкретно; слова «ярлык»/«агрегатный» — внутренние, читателю — «оценка фазы цикла» |
| RAG-подтверждение (+ банк + ✓/⚠) | `regime_rag_confirm[]` → `regime.ragSignals[]{text,bank,ok}` | `tg_bot._fetch_rag_context` → `premium_payload` | каждая выдержка несёт банк-источник + чек-пойнт ✓/⚠ (⚠ только при `confirmStance=diverges`); выдержка — целый блок чанка через `_clean_rag_excerpt` (обрезка по слову), n_results=6 с приоритетом разных банков; пустая база → фолбэк на моментум-объяснители (`ragBacked=false`) |

**Как менять:** сигналы классификатора → `regime.py` (`SHORT/MEDIUM_WIN`, компоненты осей);
пороги сверки → `_build_regime_consistency`; геометрия SVG → `_regime_dot_coords`;
тон ИИ при низкой уверенности → `regime_confidence_rule` в `ai_narrative._user_prompt`.

---

## 3. Состав портфеля / риск-концентрация / мандат-панель (BASE + DEEP)

| Под-секция | Ключи | Builder | Движок |
|---|---|---|---|
| Таблица активов (вес, β, TRC, P&L, action) | `assets[]`, `hotspots[]` | `build_payload` (per-asset цикл) | Эйлер-TRC: `calculate_structural_risk`; 🔥 hotspot: TRC > `scoring.HOTSPOT_TRC_PCT` (SSOT) |
| Фильтры holdings (Premium, оба тира) | — | `portfolio-holdings.jsx` / `deep/deep-holdings.jsx` | Чип «HOTSPOT» рендерится **только когда hotspot-позиции есть** (`hasHot`, L-15) — раньше пустой фильтр давал «Ничего не подходит». Матчеры Технологии/Защитные читают `sector`+`cls`, толерантны к EN/RU |
| Секторный пай + бейдж плеча | `sectors[]`, `pie_chart_data[]`, `leverage_metrics` | `build_payload` (sector-блок) | Плечо из отрицательного кэша; Gross=(лонг+\|маржа\|)/капитал, Плечо=лонг/капитал. **Реестр плечевых ETP** `finance/leveraged.py` — L, underlying, ER, контрактный drag −½L(L−1)σ_u² |
| Концентрация HHI + warnings | `sector_concentration`, `asset_concentration`, `sector_warnings`, `sector_groups`, `sector_complex` | `_build_concentration`, `build_sector_groups` | SSOT супер-группы: `asset_taxonomy.SECTOR_SUPERGROUPS` («Tech-комплекс» = Technology+Semiconductors) + `top_sector_concentration_pct` — эту же величину видят риск-гейдж (§1) и деконцентрация Effect (§7) |
| Панель «Соответствие мандату» | `mandate_compliance.rows[]/breaches/compliant/leveraged/margin_debt_pct` | `_build_mandate_compliance` (+ партиал `_mandate_compliance.html`) | Классы vs лимиты профиля; строка маржинального долга; leveraged ⇒ не compliant. **Классификация по underlying (L-14)**: `gatekeeper._classify_to_asset_key` сначала рекурсивно резолвит обёртку через `finance/leveraged.etp_info` (CONL→COIN→Криптовалюта, `_CRYPTO_UNDERLYINGS`) и только потом тикер-эвристики — крипто-экспозиция через плечевые ETP видна лимитам |
| Margin-Call предупреждение ИИ | `ai_leverage_warning` | прокидка из `ai_summary` | триггер `has_leverage` в промпте |

**Как менять:** классификация классов → `agent/gatekeeper._classify_to_asset_key`
(кэш = `Cash`, не Bonds; новые крипто-underlying → `_CRYPTO_UNDERLYINGS`; новые обёртки →
реестр `finance/leveraged.py`); лимиты мандата → `profile_manager._PROFILE_MAP`; вид панели → партиал.

---

## 4. 4-Pillar Scoring (DEEP-таблица; колонки Score в BASE)

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Очки F/V/T/C + Total + Action | `assets[].score/action`, DEEP `score_breakdown` | `build_payload` | `scoring_orchestrator.score_portfolio` → `scoring.py`. **Action — SSOT направления** для Action Plan (§6) и Effect (§7) |
| F-пиллар (ROE/OpM/D-A/RevG/FCF) | — | — | z-каскад `_sector_z`: (1) когорт портфеля ≥5 → (2) live SEC-когорт лидеров сектора (`_dynamic_sector_cohort`; off: `SECTOR_COHORT_DISABLED=1`) → (3) статик-таблица 2020-25. **+ bounded ±0.5 fundamental-momentum бонус** (`SEC_Fundamental_Momentum` = YoY-тренд маржи; opt-in, `None` → без эффекта; Piotroski-F остаётся в C-пилларе — двойного счёта нет) |
| V-пиллар (P/E, P/B, FCF-yield) | — | — | `_compute_valuation_ratios` + `_SECTOR_*_BENCHMARKS`; FCF-yield — signed-z |
| C-пиллар (CDS/Altman/Piotroski) | `fundamental_layer[]` | `build_payload` | `sec_edgar.py`, `cds_feed.py`; N/A-классы → нейтраль + прочерк |
| Плечевые обёртки | — | — | LETF-обёртки (CONL/XNDU/…) **исключены из F/C пилларов** (P-8: фундаментал эмитента ноты ≠ фундаментал underlying) |
| Hotspot-порог | — | — | `scoring.HOTSPOT_TRC_PCT` = 20.0 — единственное место тюнинга (отчёт + Gatekeeper GK-1) |

**Как менять:** состав live-когорт → `_SECTOR_REPRESENTATIVES`; статик-бенчмарки →
`_SECTOR_FUNDAMENTAL_BENCHMARKS` / `_SECTOR_PE/PB/FCF_YIELD_BENCHMARKS`; правила очков →
`scoring.fundamentals_score`/`valuations_score`/`credit_score`.

---

## 5. AI Ideas + правила промптов ИИ (BASE + DEEP)

| Элемент | Ключи | Builder | Источник |
|---|---|---|---|
| Карточки идей (4 сценария) | `ai_ideas{growth/diversification/hedge/rotation}`, `ideas_count` | `_build_ai_ideas` | `ai_summary.stock_picks` (`boost_alpha/rebalance/protect_capital/smart_money`); 4-я карточка — Smart Money, рендерится в bucket `rotation` |
| Генерация пиков (LLM) | — | — | `_user_prompt`: DATA-DRIVEN + СВЕЖЕСТЬ ИДЕЙ (дневной якорь `YYYY-MM-DD` + дневной угол ротации `_IDEA_ROTATION_ANGLES`). Routing: BASE=`claude-sonnet-5`, DEEP=`claude-opus-4-8`, оба без `temperature` (400 при передаче) — дисперсию даёт директива+угол; env-override `claude-sonnet-4-6` возвращает temperature-band 0.5–0.85 |
| Пост-фильтры пиков | — | — | `_remove_held_picks` → `_check_pick_contradictions` → `_backfill_empty_scenarios` → **`_remove_mandate_banned_picks`** (B1: класс с лимитом 0–0 в мандате — идея вычёркивается; классификация — по underlying, §3) |
| Фолбэк-каталог (без API) | — | — | `_fallback_stock_picks`: 3 кандидата/слот + месячная ротация |
| Кнопка «Применить идею» (BASE+DEEP) | `meta.botUsername` (env `BOT_USERNAME`, default `KEN_investment_bot`) | `pdf_payload.bot_username` → `premium_payload.meta.botUsername`; UI `ApplyIdeaModal` (`portfolio-ideas.jsx` / `deep/deep-plan.jsx`) | Модал → deep-link `t.me/<bot>?start=scn_<n>`. Статичный HTML НЕ списывает токен — списание в боте: `tg_bot.cmd_start` ловит `scn_` → сценарный тир через `kb_confirm("scenario")` |

### Реестр правил промпта (`ai_narrative._user_prompt`, оба тира — порядок инъекции)

| Правило | Что делает | Появилось |
|---|---|---|
| `plain_rule` | простой язык, расшифровка метрик; глоссарий помечен как **справка, не текст для вставки** | Sprint 5.3 (уточнено L-16) |
| `style_rule` | **анти-шаблон**: каждая мысль/расшифровка цифры — один раз на отчёт (дальше ссылка или другой угол); бан внутренних терминов: «счёт» → «сводная оценка 4-Pillar», «ярлык» → «оценка фазы цикла»; структура длинных полей — 2–3 полных предложения (факт → смысл → действие/мандат/банк-аналитика) | L-16, L-19 |
| `numbers_rule` | **анти-фабрикация**: числа только из данных промпта, пере-округления запрещены (max 1 знак), нет данных → «данных недостаточно» | L-19 (2026-07-19) |
| `currency_rule` | бан выдуманной валюты (баг «8 из каждых 10 рублей» на USD-книге) | 2026-07-09 |
| `benchmark_rule` | ИИ называет бенчмарк клиента по имени; факторный перекос — ОТНОСИТЕЛЬНО него; честность сектор-контекста | B1 (2026-07-17) |
| `mandate_rule` | лимиты классов + целевая vol + запрещённые классы в тексте промпта; «ЦЕПОЧКА АНАЛИЗА» verdict→holdings→factor→stress→effect→action держит бенчмарк и мандат в поле зрения | B1 + раунд 26 |
| `regime_confidence_rule` | при confidence <25%: оценку фазы — осторожно, сигналы — уверенно (см. §2) | раунды 26–27 |

**Как менять:** правила → соответствующая `*_rule`-строка в `_user_prompt` (инъекция в ОБА тира —
проверить оба f-string блока); каталог фолбэка → `_BOOST_*`/`_BALANCE*`/`_PROTECT`/`_REGIME_*`;
модал/deep-link → `ApplyIdeaModal` + обработчик `scn_` в `tg_bot.cmd_start`.

---

## 6. Action Plan (DEEP)

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Таблица Buy/Sell/Stop/Δw | `action_plan` (из results) | прокидка | `action_plan.build_action_plan`. Действие — из 4-Pillar (SSOT), количество/Δw — из Black-Litterman |
| Примирение «действие ↔ количество» (L-1) | `qty_delta`, пометка строки | `build_action_plan` | при противоречии знака BL с действием (Sell/Trim но Δw>0, Buy но Δw<0) → `qty_delta=None` («—») + пометка «BL расходится с сигналом (+Xпп)»; согласующиеся знаки не тронуты |
| Уровни (зоны, тейк, стоп) | — | — | `compute_levels`: ATR+SMA+52w-high; mandate-scale ATR-дистанций 0.75/1.0/1.25 (`MANDATE_LEVEL_SCALE`); SMA-якоря без масштаба |
| Турновер-кап | — | — | `MAX_TRADE_BLOCK_PORTFOLIO_PCT=0.25` (+ BL `max_active_share` по мандату) |
| AI-комментарий плана | `ai_action_comment`, `action_plan_text`, `ai_action_impact` | прокидка из `ai_summary` | обязан вести нарушенную долю обратно в мандатный лимит (`mandate_rule`, §5) |

**Как менять:** множители уровней → константы `ATR_*` и `MANDATE_LEVEL_SCALE`;
приоритет сортировки → `priority`-словарь в `build_action_plan`.

---

## 7. Ожидаемый эффект на риск — Effect (DEEP, 8 карточек)

> **Роль секции (раунды 27–28): Effect = мандатная ребалансировка.** Action Plan даёт
> направление (4-Pillar), Effect — магнитуду и деконцентрацию: продать перегруз →
> реинвестировать в допустимые диверсификаторы или Кэш → показать риск «до/после».
> Обе секции рассказывают ОДНУ историю: привести портфель к риск-профилю и мандату.

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Карточки before→after | `expected_effect.{risk_index,vol,cvar_95,sharpe,max_drawdown,max_erc_pct,it_share,expected_return}` | `_build_expected_effect` (флэттенер) | `simulate.simulate_after_plan(..., mandate, leverage_ratio)` — те же аггравторы, что у гейджа §1 ⇒ `risk_index`-«до» == гейдж на обложке |
| Что именно продаётся/покупается (L-2) | `expected_effect.high_priority_actions[]` (`{ticker, side, delta_pp}`) → premium `effectActions[]{t,side,key,dw}` | `pdf_payload` → `premium_payload._map_deep` | `EffectGrid` (`deep/deep-plan.jsx`): блоки **«Продать / сократить»** (rust) и **«Купить / нарастить»** (sage) с Δпп; Кэш-строка — «В кэш» (`is_cash`). Метрики «до/после» ниже — результат именно этих сделок |
| Таргет-веса высокоприоритетных строк | — | — | `simulate.high_priority_target_weights(current, action_plan, bl, sector_by_ticker, reinvest_blocklist)`: **сторона — из ДЕЙСТВИЯ 4-Pillar** (не знака BL, L-6), магнитуда — \|BL Δw\|, турновер-кап 0.25 |
| Реинвест = деконцентрация (L-10) | — | — | высвобожденный вес идёт ТОЛЬКО в диверсификаторы вне перегруженного топ-сектора (супергруппа-aware, порог `_CONC_FLOOR=0.40`), остаток → **Кэш** (pseudo-move `is_cash`) — IT-доля/концентрация/риск падают, Effect ведёт К мандату |
| Реинвест-eligibility (L-13) | — | — | кандидаты фильтруются: `reinvest_blocklist` (broker-priced-only активы + sparse-dropped, собирает `investment_logic`) + плечевые ETP (`finance/leveraged.is_leveraged_etp`) |
| Conviction-гейт (L-17) | — | — | held-кандидат реинвеста обязан иметь **план-рейтинг Buy/Strong Buy** (типовой случай — Buy, отложенный турновер-капом, Δw=0). Имя с HOLD при пилларах 0.0 (нет данных → нет conviction, напр. неликвидная AIX-нота со сглаженной ценой) НЕ докупается — Effect исполняет план, а не спорит с ним |
| Внешняя рука глобальных ETF (L-18) | `effectActions[].{is_external,name}` | `simulate.EXTERNAL_DIVERSIFIERS` + отбор в `investment_logic` | вес, который некуда деть внутри книги: IEF (гособлигации США 7–10 лет) / EEM (акции EM) / EMB (облигации EM) — порядок по мандату, ≤8пп/имя, ≤3 имён, остаток → Кэш. Кандидаты только из ФАКТОРНОЙ панели (история скачана каждым прогоном — ноль лишних запросов), классы согласованы с мандат-панелью (EEM→GlobalETFs — закрывает нижнюю границу 10% у профилей 9+). Ковариация симуляции расширяется sample-блоком (≥60 дней перекрытия) — ETF несёт реальный риск, «до» == обложка байт-в-байт |
| Sharpe без look-ahead (L-11) | `expected_effect.sharpe` | — | **FORWARD Sharpe** = (er−rfr)/vol из BL-µ и структурной vol (ex-ante база), заякорен к headline-значению, дельта клэмп ±0.6 — реализованный Sharpe «после» не выдумывается |
| Цвет дельты | `favourable` | — | `simulate._delta_row`: lower-better = {vol, max_trc, risk_index}; `it_share` нейтрален; нулевая дельта нейтральна |
| Вердикт ребаланса | `expected_effect.verdict` | прокидка | improvement/tradeoff/degradation/neutral — по vol/TRC/MaxDD; `ai_effect_comment` обязан согласоваться с `verdict.kind` |

**Как менять:** направление метрики → `_RISK_METRICS_LOWER_IS_BETTER` / `_NEUTRAL_METRICS`;
порог деконцентрации → `_CONC_FLOOR`; блок-лист → сборка `_reinvest_block` в
`investment_logic.analyze_all`; карточный маппинг → `_KEYMAP` в `_build_expected_effect`;
вид → `EffectGrid` в `deep/deep-plan.jsx` (после правки — `build.sh`).

---

## 8. Бенчмарки и доходность (BASE + DEEP)

> Бенчмарк клиента (из мандата, меняется в `/mandate` за 2 тапа) — **сквозная сущность**:
> одна и та же величина `results['profile_benchmark_ticker']` питает карточку «vs рынок»,
> метку «Рост против рынка», equity-curve и факторную таблицу. ADR (B1): бенчмарк —
> отдельная сущность со своими бетами (Вариант A); «подменить ось Market» (B) отклонён —
> сломал бы стресс/BL/Euler/κ; «только переименовать» (C) отклонён как вредный.

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Карточка «vs рынок» | `scenarios[]` (excess, TE, IR, bm_return) | `_build_scenarios` | `_compute_benchmark_stats` (pair inner-join) |
| Выбор главного бенчмарка | фильтр `user_bench_ticker` → слот «Профильный бенчмарк» | `build_payload` | `tg_bot._resolve_bench_ticker` |
| Мульти-период 1м/3м/6м/12м/YTD | `period_returns_table` | `_adapt_period_returns` | `period_returns.py` |
| «Рост против рынка» (BASE) — метка бенчмарка (B1-perf) | `performance_benchmark_name` → premium `performance.benchmarkName` → `portfolio-performance.jsx` / v3 `_bench_keys[0]` | `build_payload` переименовывает ключ «Профильный бенчмарк» в реальное имя (`BENCHMARK_LIST[user_bench_ticker]`) | цифры и так были профильного бенчмарка — подпись «S&P 500» была захардкожена в 5 местах; теперь динамическая, фолбэк — первый конкретный ключ / «S&P 500». Тест `BasePerformanceBenchmarkTest` |
| Equity curve (B-1, 2026-07-16) | `equity_curve_svg` | `tg_bot._build_*` | `finance/portfolio_series.compute_equity_curve_series` берёт `results['profile_benchmark_ticker']` (раньше — первый доступный из `_BM_CONCRETE_MAP` = всегда S&P 500). Тесты `test_phase30_benchmark_equity_curve.py` |
| Факторный радар | `factor_radar_svg`, `factor_betas` | `tg_bot._build_*` | Ridge-беты движка. «Рынок (S&P 500)» в факторных подписях — рыночный ФАКТОР модели, не бенчмарк клиента (не баг) |
| **Бенчмарк в факторной таблице (B1)** | `benchmark_factor_profile{ticker,name,betas}` (results) → `factor_betas[].bench` → `benchmark_name/benchmark_ticker` (payload) → premium `benchmarkName/benchmarkTicker` | движок `fit_factor_betas` (analyze_all) → `tg_bot._build_factor_betas_table` → `pdf_payload` → `premium_payload` → `deep-factors.jsx` | столбец сравнения = РЕАЛЬНЫЕ факторные беты мандатного бенчмарка, тот же Ridge-пайплайн (панель `factor_tickers`, ортогонализация BLOCK 3.5, α=0.001, guard ≥max(2K,30) obs). `Наклон Δ = β_портф − β_бенч`. Фолбэк: S&P-константа `(Market=1, rest=0)` ВМЕСТЕ с подписью «S&P 500» — подпись всегда совпадает с числами. v3-фолбэк тоже динамический. Тесты `test_phase31_benchmark_factor_propagation.py` |
| Источники риска (декомпозиция дисперсии) + двойники (DEEP) | `factor_variance{rows,systematic_pct,idio_pct,twins}` → premium `factorVariance` | `pdf_payload._build_factor_variance` | additive layer `finance/factor_decomposition.py`: Euler по ФАКТОРАМ (σ²=bᵀFb+wᵀDw, b=Bᵀw); «двойники» = пары с systematic-corr ≥0.90 (`TWIN_CORR_THRESHOLD`); отрицательная доля = фактор-хедж. AI: 4-шаговый `ai_factor_comment` + детерминированный фолбэк |

**Как менять:** список бенчмарков → `BENCHMARK_LIST`; беты бенчмарка → `fit_factor_betas`
(движок); подписи → `deep-factors.jsx`/`portfolio-performance.jsx` (+ `build.sh`).

---

## 8.1. Журнал сквозной логики отчёта (аудит-раунды)

> Здесь видно, **как менялся проект**: каждый раунд — живой прод-отчёт, найденные
> дефекты логики (Было), фиксы (Стало) и тесты. Текущее состояние секций — в §1–§11;
> этот журнал объясняет «почему так». Полные разборы: `docs/audit/AUDIT.md` §−24…§−28.

### B1 «Мандат → Отчёт» (2026-07-17) — AUDIT §−24

| Было | Стало |
|---|---|
| Смена бенчмарка в мандате не доезжала до «Факторного разложения» (3 слоя хардкода S&P) и до метки BASE-перформанса | Беты бенчмарка тем же Ridge-пайплайном → динамический столбец/легенда/плашка (§8); `/mandate`-меню (бенчмарк за 2 тапа) БЕЗ биллинга; ИИ называет бенчмарк по имени и фильтрует идеи лимитами 0–0. Тесты: phase31 (29) |

### Раунд 25 «Логика отчёта» L-1…L-5 (2026-07-18) — AUDIT §−25

| Было | Стало |
|---|---|
| Action Plan: SELL с «+1 шт» (действие из 4-Pillar, количество из BL — знаки расходились) | L-1: при противоречии знака `qty_delta=None` («—») + пометка «BL расходится с сигналом» (§6) |
| Effect: не видно, ЧТО продаётся/покупается | L-2: `effectActions` → `EffectGrid` с блоками Продать/Купить (§7); DEEP-контракт 34→35 |
| Риск-гейдж занижен: 73%-tech книга с плечом и −43.5% MaxDD = «48 · Умеренный» | L-3: bounded аггравторы composite (§1); живая книга 48→70 «Агрессивный»; verdict==effect-before через `simulate` |
| BASE «Рост против рынка»: данные Nasdaq под меткой «S&P 500» | L-4: динамическая метка (§8, B1-perf) |
| ИИ-цепочка не помнила бенчмарк/мандат | L-5: бенчмарк+мандат вплетены в «ЦЕПОЧКУ АНАЛИЗА» обоих промптов (§5) |

### Раунд 26 follow-up L-6…L-9 (2026-07-18) — AUDIT §−26

| Было | Стало |
|---|---|
| Effect-сторона бралась из знака BL → AAOI (SELL) и MSFT (TRIM) показывались как «Купить» | L-6: сторона — из ДЕЙСТВИЯ 4-Pillar (`high_priority_target_weights`), знак BL только магнитуда (§7) |
| Гейдж считал концентрацию по одиночному сектору (59%), игнорируя Tech-комплекс (73%) | L-7/L-8: супергруппы SSOT `asset_taxonomy` (§1, §3) |
| Режим при confidence 8% тянул уверенный нарратив «экономика растёт» | L-9: `regime_confidence_rule` — ярлык осторожно, сигналы уверенно (§2, §5) |

### Раунд 27 «Effect = мандатная ребалансировка» L-10…L-12 (2026-07-18) — AUDIT §−27

| Было | Стало |
|---|---|
| Реинвест докупал held-тех с highest-BL → IT 73→79%, риск РОС (противоречие мандату) | L-10: деконцентрация — реинвест в диверсификаторы вне топ-сектора / Кэш (§7) |
| Sharpe 0.69→1.89 — look-ahead (реализованный Sharpe таргет-весов на той же истории) | L-11: forward-Sharpe (er/vol), якорь к headline, дельта ±0.6 (§7) |
| Режим-комментарий уклончивый | L-12: честный уверенный разбор всех сигналов (§2) |
| **Live-пруф 19.07 08:25:** Effect продаёт тех → Кэш +4.3пп, IT 73→59%, Sharpe 0.69→0.71, риск 69→67 | подтверждено; прогон вскрыл L-13…L-16 ↓ |

### Раунд 28 follow-up L-13…L-16 (2026-07-19) — AUDIT §−28

| Было | Стало |
|---|---|
| Effect предлагал «Купить FFSPC6.1028.AIX +12пп» (некотируемая нота) и «XNDU +4.1пп» (LETF) — Max TRC 16.2→36.3% | L-13: `reinvest_blocklist` + `is_leveraged_etp`-фильтр; нет кандидатов → Кэш (§7). Тесты `ReinvestEligibilityTest` |
| CONL (2× Coinbase) = «Акции США» → мандат-панель «Крипто 0.0%» | L-14: классификация по underlying через `etp_info` (§3). Тесты `UnderlyingClassificationTest` |
| Чип «HOTSPOT» при 0 hotspot-позиций → пустая таблица | L-15: чип по условию `hasHot` в обоих тирах (§3) |
| «В худший день из 20 теряется ≈3.5%» повторялась в 3+ полях; в тексте «счёт −5.2», «Ярлык» | L-16: `style_rule` — мысль один раз, бан внутренних терминов; глоссарий = справка (§5). Тест `AiStyleRuleTest` |

### Раунд 29 «Conviction + глобальные ETF + честные числа» L-17…L-19 (2026-07-19) — AUDIT §−29

| Было | Стало |
|---|---|
| DEEP 13:03 всё ещё «Купить FFSPC +12пп»: нота обходит блок-лист L-13 честно — она НЕ broker-priced/sparse (Tradernet котирует, corr 0.968 с TLT в панели двойников), но план рейтит её HOLD с пилларами 0.0, а Effect изобретал покупку; vol «после» 15.6% частично фантомная (сглаженная цена ноты занижает риск) | L-17: conviction-гейт — held-докуп только при план-рейтинге Buy/Strong Buy (§7). Тесты `ConvictionGateTest` |
| Свободный вес шёл 100% в Кэш — у ребалансировки не было покупательной стороны | L-18: внешняя рука глобальных ETF (IEF/EEM/EMB из факторной панели, по мандату, ≤8пп/имя) + sample-расширение ковариации в `simulate_after_plan` (§7); имя ETF — в EffectGrid и в `rebalance_actions` промпта. Тесты `ExternalSleeveTest`, `ExtendedCovSimulationTest` |
| «В худший день из 20» — непонятно (жёстко прописано в глоссарии + 4 спеках полей); комментарии телеграфно-короткие; выдуманные цифры/округления не ограничены | L-19: расшифровка CVaR → «средний убыток в редкий плохой день (примерно раз в месяц)» во всех 6 местах; правило структуры 2–3 полных предложений (факт → смысл → действие/мандат/банк-аналитика); `numbers_rule` в обоих тирах (числа только из данных, без пере-округлений); DEEP-капы risk/holdings/action-комментариев 250→400-420. Тесты `PromptQualityTest` |

---

## 9. Стресс-тесты (DEEP)

| Элемент | Ключи | Builder | Движок |
|---|---|---|---|
| Сценарии (7 шоков) | `stress_scenarios[]` | прокидка | `stress.py`: β×шок с выпуклым капом T=0.20→C=0.35. **Плечевые ETP — path-dependent** (P-5, phase28): `finance/leveraged.path_dependent_period_return` — шок underlying прогоняется по пути с daily-reset drag, не линейное β×L |
| AI-комментарий | `ai_stress_comment` | `validate_stress_comment` (гарантия упоминания капа) | — |

---

## 10. Прозрачность / QC (BASE + DEEP)

| Элемент | Ключи | Builder |
|---|---|---|
| Integrity-панель ✓/⚠ | `integrity_checks[]` | `_build_integrity_checks` — RAG 3-state (used/no_match/unavailable) + инвентарь базы (`M отчётов · K чанков`) + пилл «ИИ↔банк-аналитика» (⚠ когда ИИ ссылается на банки при пустой базе) + **caveat-чипы методологии** (phase28): `var_reliability=insufficient_history` при короткой истории, плечевые daily-reset ETP. Футер красит собственный символ строки |
| CoVe data-lineage (**16 строк**; было 24, консолидация 2026-07-09) | `cove_lineage[]` | `finance/data_lineage.build_lineage`. Объединены: Риск-метрики+Euler (одна ковариация MAC3) → 1; две SEC-строки → 1; 6 FRED-серий → 1 агрегат (бейдж = худший статус); два LLM-чекера → 1. Остальные: факторная независимость κ+max\|corr\|, Tradernet-цены, FX+ставка, CDS, Action levels, Black-Litterman, режим, стресс, Smart-Money (gated), Bank RAG (+инвентарь), ИИ-цитирование банк-аналитики ([RAG]-цитаты vs консенсус из памяти), AI-вердикт. «База: N отчётов» считает РЕАЛЬНЫЕ отчёты — tmp-артефакты исключены (`_is_temp_source`) и чистятся на буте (`purge_temp_sources`) |
| Data quality | `data_quality` | `build_payload` (факторы N/10, SEC-пропуски) |
| AI-вердикт/буллеты | `ai_verdict`, `ai_plain_summary`, `ai_bullets` | прокидка из `ai_summary` |

---

## 11. Scenario Analysis (отдельный тир · 1 токен · 2026-07-07)

> Тир `scenario` — **самостоятельный отчёт**, НЕ страница BASE/DEEP. Детерминирован
> (0 LLM-вызовов). Поток: `analyze_all()` → `finance/scenario_report.build_scenario_payload`
> (переиспользует `finance/scenario_engine`) → `templates/report_scenario_v3.html`
> (свой Jinja; `html_renderer` маршрутизирует scenario минуя Premium).

| Панель | Ключи (`data.scenario.*`) | Движок |
|---|---|---|
| A · 5 core-метрик | `metrics{ann_return,vol_cov,vol_gross_ref,sharpe_rfr,beta,div_yield,pe,rfr}` | `scenario_engine.five_metrics` (σ_p ковариационная, RFR-Sharpe, Σwβ) |
| A · Euler-MCTR вклад в риск | `mctr_rows[]{display,weight,sigma_i,rho_ip,mctr,ctrisk,pct_ctr}`, `vol_cov` | `scenario_engine.mctr_table` (MCTR=ρ·σ, Σ CTRisk=σ_p, Σ%=100) |
| A · Выживаемость 3 макро-режима | `regime_survival[]{regime,label,avg_pct,survives}` | `scenario_engine.regime_survival` (группировка 7 шоков; билдер нормализует `stress.port_pct` долю→%) |
| A · Funding-звенья | `funding[]{display,weight,sharpe,flags[]}` | `scenario_engine.funding_candidates` (Sharpe<0.5 · доходность<0 · corr-дубль≥0.90 · TRC-несоразмерность) |
| B · Walk-forward бэктест | `backtest{ticker,rule,summary{n_signals,hit_rate_63d,horizons}}` | `scenario_engine.walk_forward` (правило «цена>200-DMA», look-ahead guard, `DISCLAIMERS`) |

**Токеномика:** `tg_bot.TIER_COST` = base 1 · scenario 1 · deep 2; **1 токен = 2 500 ₸,
пакет 10 = 25 000 ₸** (2026-07-17; SSOT — `TOKEN_PRICE_KZT`/`TOKEN_PACK_PRICE_KZT`,
экономика — `docs/business/ECONOMICS.md`). Демо-отчёты бесплатны, в т.ч. scenario из демо-кэша
(маркер `_demo_portfolio`). Кнопка тира в `kb_analysis_choice` + inline-CTA
`scenario:cached` под BASE/DEEP (кэш `results` per-user).
**Как менять:** математика → `finance/scenario_engine.py`; сборка → `scenario_report.py`;
вид → `report_scenario_v3.html`. Тесты: `tests/test_phase24_scenario_report.py`.

---

## Чек-лист изменения секции

1. Найти секцию здесь → открыть builder (`pdf_payload.py`) и/или движок (`finance/*`).
2. Числа меняются ТОЛЬКО в движке; формат — в builder; вид — в шаблоне/JSX.
3. Правил JSX? → `bash design/premium_v2/build.sh` (иначе `CompiledAssetsTest` красный).
4. Расширил дизайн-контракт? → обновить пин ключей в `test_phase19_block_audit`.
5. Прогнать: `python -m pytest tests/ -q` + smoke-render обоих тиров
   (`html_renderer.render_report_html(None, <user_id>, ...)`).
6. Новое поведение = новый кейс в `tests/test_phase*.py`.
7. Большое изменение — строка в §8.1 (журнал) + `AUDIT.md` (Было/Стало).
