# REPORT_PAGES_DEEP.md — постраничный разбор DEEP-отчёта

> Справочник для инженера: **что показано на каждой странице DEEP-отчёта, откуда берутся
> числа, как считается математика и как читать результат.** Парный файл по BASE-тиру —
> `docs/REPORT_PAGES_BASE.md`. Карта секций — `REPORT_SECTIONS.md`; аудит живых отчётов —
> `REPORT_SECTIONS_AUDIT.md`.

## Поток данных

```
analyze_all()  (src/finance/investment_logic.py)
   → risk_matrix, portfolio_metrics, asset_scores, regime, macro_drivers,
     stress_scenarios, black_litterman, expected_effect, action_plan, factor_scores …
        │
        ▼
build_payload(results, tier="deep", ai_summary, user_profile, …)  (src/pdf_payload.py)
        │   форматирует, строит секционные dict-ы; tier=="deep" добавляет
        │   score_breakdown, action_plan-строки, bl_records.
        ▼
Jinja-шаблон  src/templates/report_deep_v3.html  (+ партиал _mandate_compliance.html)
        ▼
HTML по подписанной ссылке
```

ИИ-поля (`ai_*`, `regime_confirmation`) — отдельная ветка `generate_narrative()`
(`src/ai_narrative.py`). **DEEP использует модель `claude-opus-4-8`** (env `ANTHROPIC_MODEL_DEEP`).
Opus **опускает `temperature`** — дисперсию идей даёт промпт-директива «СВЕЖЕСТЬ ИДЕЙ»
(гард `_model_supports_temperature` — load-bearing). Всё, что в блоках «ИИ-комментарий» /
«ИИ подтверждает режим», — **advisory, не индивидуальная инвестиционная рекомендация**;
числа считает движок, ИИ их комментирует. Пустые ИИ-поля скрывают свои блоки.

> **Про шаблон.** Заголовок файла — «ПРОТОТИП (design-only)». Числовые поля подключены к боту;
> прототипными остаются декоративные SVG (equity-curve на P? и KPI-спарклайны — спрятаны за
> `{% if data.kpi_sparklines %}`). Факторный радар и квадрант режима рендерятся реальными
> данными (`data.factor_radar_svg` от бота, `data.regime.dot_cx/cy` от builder). Пометка
> **[статический SVG]** стоит там, где макет.

DEEP-отчёт — **6 «листов»** (`<div class="sheet">`):

| # | Лист | Содержимое | Якорь для grep |
|---|---|---|---|
| 1 | Обложка · KPI · концентрация (наследует BASE) | вердикт, гейдж, мандат, 3 KPI, TRC-таблица, водопад, QC | `class="cover"`, `kpi-grid` |
| 2 | Что вы держите (наследует BASE + Smart Money) | таблица позиций + SEC-раскрытие, секторный пай, плечо, блок Smart Money (SEC Form 4) | `holdings-layout`, `smart_money` |
| 3 | Факторное разложение · 4-Pillar · стресс | β-радар, 4-Pillar карточки, стресс-таблица 7 шоков | `d3-block`, `factor-layout`, `score-grid`, `stress-table` |
| 4 | Рыночный режим и макро-контекст | квадрант Growth×Cycle, growth/cycle факторы, FRED-сигналы, R3-сверка, ИИ-подтверждение | `regime-layout`, `reg-quadrant-svg`, `macro_drivers` |
| 5 | AI Ideas · Action Plan · Ожидаемый эффект | idea-карточки (4-stage), таблица Buy/Sell/Stop, 8 карт before→after | `ideas-grid`, `action-table`, `effect-grid` |
| 6 | CoVe — верификация источников | data-lineage всех источников с методом и статусом | `cove-grid`, `cove_lineage` |

Лист 1 идентичен BASE (один движок, паритет чисел); лист 2 наследует BASE-состав плюс
DEEP-эксклюзивный блок Smart Money (SEC Form 4). Их детальный разбор по holdings см. в
`docs/REPORT_PAGES_BASE.md` (§1.x, §2.x). Здесь они даны кратко; основной фокус — DEEP-эксклюзив
(блок Smart Money на листе 2 + листы 3–6).

---

## Оглавление

- Лист 1 — Обложка · KPI · концентрация (= BASE §1)
- Лист 2 — Что вы держите (= BASE §2 + DEEP-only блок Smart Money / SEC Form 4)
- Лист 3 — Факторное разложение · 4-Pillar Scoring · стресс
  - 3.A Факторный β-радар (скрытые концентрации)
  - 3.B 4-Pillar Scoring (F/V/T/C)
  - 3.C Стресс-сценарии (7 шоков, выпуклый кап)
- Лист 4 — Рыночный режим и макро-контекст
  - 4.A Квадрант Growth×Cycle + сводка
  - 4.B Growth/Cycle факторы
  - 4.C FRED-сигналы драйверов (+ темп)
  - 4.D Детерминированная сверка FRED↔моментум (R3)
  - 4.E ИИ-подтверждение режима
- Лист 5 — AI Ideas · Action Plan · Ожидаемый эффект
  - 5.A AI Ideas (4-stage pipeline Factor→Regime→Stress→RAG)
  - 5.B Action Plan (уровни Buy/Sell/Stop)
  - 5.C Ожидаемый эффект (8 карт before→after + вердикт)
- Лист 6 — CoVe (Chain-of-Verification)

---

# Лист 1 — Обложка · KPI · концентрация (наследует BASE)

Полностью совпадает с BASE-листом 1 (один и тот же payload, один движок). Кратко:

- **Тех-хедер** — `TIER: DEEP`, `user_id`, `generated_at`.
- **Вердикт** — `data.ai_verdict` (H1), `data.ai_plain_summary`, `data.ai_bullets[:5]`
  (DEEP показывает **5** инсайтов против 4 в BASE).
- **Соответствие мандату** — партиал `_mandate_compliance.html`, ключ `data.mandate_compliance`,
  builder `_build_mandate_compliance`. Логика: фактическая аллокация по классам vs `limits_dict`,
  допуск ±2 п.п., маржинальный долг вне лимитов ⇒ не compliant.
- **Риск-гейдж 0–100** — `data.risk_pct`/`risk_label`/`risk_mandate_label`. Движок:
  `scoring.composite_risk_score` = взвешенный по мандату бленд `vol/0.40`, `|CVaR|/cvar_base`,
  `maxERC/50` (веса из `_RISK_MANDATE_MATRIX`). Стрелка: `rotate=(risk_pct−50)·1.8°`.
- **KPI CVaR/Sharpe/MaxDD** — `data.cvar`+`cvar_ci`+`cvar_dollar`, `data.sharpe`+`sortino`,
  `data.max_drawdown`+`mdd_dollar`. Движок: CVaR — block-bootstrap Politis-Romano (2000 блоков,
  асимметричный 95% CI); Sortino — downside deviation `√(mean(min(excess,0)²))·√252`.
- **Таблица TRC** «Где сосредоточен ваш риск» — `data.assets[]` top-5 по `euler_risk`. Движок:
  Эйлер `ERC%ᵢ = wᵢ·MCTRᵢ/σ_p`, HOTSPOT при `ERC% > 20%` (`HOTSPOT_TRC_PCT`).
- **Риск-водопад** — `data.risk_waterfall`, builder `_build_risk_waterfall`: `standaloneᵢ=wᵢσᵢ`,
  `diversified=√(w'Σw)`, `benefit=Σstandalone−diversified`.
- **Integrity** — `data.integrity_checks[]`, builder `_build_integrity_checks`.

Полные формулы и нюансы — `docs/REPORT_PAGES_BASE.md` §1.1–1.8.

---

# Лист 2 — Что вы держите (наследует BASE)

Наследует BASE-лист 2 (таблица позиций, сектора, ИИ-блок), плюс **DEEP-эксклюзивный блок
Smart Money внизу листа** (B2.4). Кратко:

- **Таблица позиций** — `data.assets[]` (вес, класс, `euler_risk`, P/L%, P/L$, action, hotspot);
  раскрытие по клику — SEC-фундаментал из `data.fundamental_layer[]` (ROE, OpM, Долг/Активы,
  Рост выручки, ATR, Altman-Z) + ИИ-коммент (или rule-based фолбэк). DEEP-фолбэк-текст чуть
  длиннее (добавляет контекст класса актива при `atr_extreme`).
- **Панель секторов** — пай по `data.pie_chart_data` (нормализован к доле длинной книги, сумма
  100%), бейдж плеча `data.leverage_metrics` (`gross_exposure`, `leverage_ratio`), warnings
  `data.sector_warnings[]` (порог 40% + супергруппа Tech-комплекс), Margin-Call
  `data.ai_leverage_warning`.
- **Сводный ИИ-блок** — `data.ai_holdings_comment`, `data.ai_sector_comment`,
  `data.ai_benchmark_comment` (в DEEP добавлен третий абзац «vs Бенчмарк»).
- **Smart Money · Инсайдерские сделки (SEC Form 4)** [DEEP-only] — `data.smart_money`
  (`{status, enabled, rows[], headline, hint}`), builder `_build_smart_money` (источник
  `finance/smart_money.py`, gated `SMART_MONEY_INSIDERS`). 90-дневный нетто-поток инсайдеров +
  кластеры покупок + role-weighted score [−2…+2]. Виден ВСЕГДА: при активном источнике —
  таблица (Тикер · Нетто-поток $ · Покупки/Продажи · Кластер · Score); при выключенном слое —
  нейтральная плашка «источник Form-4 не активирован» (`status='disabled'`).

Детали по holdings — `docs/REPORT_PAGES_BASE.md` §2.1–2.3.

---

# Лист 3 — Факторное разложение · 4-Pillar Scoring · стресс

## 3.A Факторный β-радар (скрытые концентрации)

**Назначение.** Показывает, насколько портфель завязан на глобальные факторы (Market, Momentum,
Value, Quality, Size, Rates, Commodities, EM…) — и где спрятана общая ставка (несколько факторов
в одну сторону = диверсификация ниже, чем кажется).

**Payload-ключи:** `data.factor_radar_svg` (готовый SVG от бота); `data.factor_betas[]` —
строки `{axis, beta, bench, delta, missing}`; `data.factor_coverage_pct`; `data.ai_factor_comment`.

**Builder / источник.** SVG и таблицу собирает бот (`tg_bot._build_*`) из Ridge-бет движка.
Подзаголовок в шаблоне фиксирует методологию: «Ridge β-регрессия (α=0.001) · EWMA-ковариация
hl=63 (λ≈0.99) ⊕ Ledoit-Wolf 70/30 · окно 60 дней». Halflife структурной ковариации в
`calculate_structural_risk` = 63 торговых дня, λ≈0.99 (плавнее RiskMetrics λ=0.94 ≈ hl 11 дн).

**Как работает математика.** Для каждого актива дневные лог-доходности регрессируются Ridge
(`α=0.001`, гасит мультиколлинеарность коррелирующих факторов) на фактор-ETF → β. Колонка
«Рынок» — эталон широкого рынка (S&P 500): у него `β_Market=1.00`, а стилевые `β=0` *по
построению*. Поэтому «Наклон Δ» по стилям = β портфеля — это и есть ваш перекос относительно
рынка. Факторы частично пересекаются (Momentum/Value/Quality — те же акции под разными углами;
EM_Equity ≈ 0.85 скоррелирован с Market) — Ridge интерпретирует β_EM как *дополнительную*
EM-экспозицию сверх объяснённого Market и Rates. `delta = beta − bench`; колонка красится
`delta-pos` (>0.01) / `delta-neg` (<−0.01) / `delta-flat`.

**Как читать.** Большая площадь радара = больше зависимости от рынка. Совпадение направлений по
нескольким факторам = скрытая общая ставка. `Δ` показывает перекос относительно рынка по каждому
фактору. `factor_coverage_pct` — какую долю портфеля покрывает факторная модель.

---

## 3.B 4-Pillar Scoring (F/V/T/C)

**Назначение.** Оценка каждой позиции по 4 столпам: **F** Фундамент (отчётность), **V** Оценка
(дорого/дёшево), **T** Техника (тренд/импульс), **C** Кредитное качество. Каждый столп −2…+2,
итог ∈ [−6, +6] → действие.

**Payload-ключ:** `data.score_breakdown[]` — `{ticker, fundamentals, valuations, technicals,
credit, total, action, fundamentals_na, credit_na}`. Плюс `data.ai_4pillar_comment`.

**Builder.** `tier=="deep"` блок `build_payload`: собирает `score_breakdown` из
`results["asset_scores"]` (кэш исключён). N/A-классы (сырьё, суверенные облигации) → F и C
рисуются прочерком (`fundamentals_na`/`credit_na`), а не числом-«режимным наклоном». Движок —
`scoring_orchestrator.score_portfolio` → `scoring.py`.

**Как работает математика** (`finance/scoring.py`):

*Z-каскад для F* (`_sector_z`): (1) когорта портфеля ≥5 → (2) live SEC-когорта лидеров сектора
(`_dynamic_sector_cohort`) → (3) статик-таблица 2020–25. Z-скор **робастный**:
`z = (x − median) / (1.4826·MAD)`, клип ±3 (нечувствителен к выбросам; None при <5 сэмплах или
нулевом MAD).

- **F (Fundamentals, −2..+2)** `fundamentals_score`: по компонентам ROE-z, OpM-z, D/A-z, RevG-z,
  FCF-z. Каждый даёт ±1 при z>1 / z<−1; D/A — асимметрично (+0.5 при z<−0.5, −1 при z>1, т.к.
  низкий долг лучше); плюс macro-alignment ±0.5 (регим). Сумма клип [−2,+2]. NaN macro → 0
  (`_finite`) до суммы.
- **V (Valuations, −2..+2)** `valuations_score`: P/E-z, P/B-z, EV/EBITDA-z дают +1 когда **дёшево**
  (z<−1) и −1 когда дорого (z>1); FCF-yield — **yield-метрика**, знак инвертирован (высокий yield
  = дёшево → +1). Сравнение с **нормой своего сектора**, не всего рынка: V≈0 у тех-акции = «оценена
  справедливо для технологий».
- **T (Technicals, −2..+2)** — из `compute_technicals` (тренд/импульс), клип [−2,+2].
- **C (Credit, −2..+1, асимметрично)** `credit_score`: CDS (<40 bps → +1; 40–90 → 0; 90–150 → −1;
  ≥150 или Δ7д>+20% → −2); Altman zone (Safe +1 / Grey 0 / Distress −1); Piotroski (F≥7 → +0.5,
  F≤3 → −0.5); Interest coverage (>5× → +0.5, <1.5× → −1). Клип [−2,+1].

**Итог и действие** `action_from_total`: `total = clip(F+V+T+C, −6, +6)`.
`≥+3` Strong Buy · `+1..+3` Buy · `−1..+1` Hold · `−2..−1` Trim · `<−2` Sell. **Hotspot-override**:
позиция с `ERC% > 20%` принудительно ставится Trim (или хуже) независимо от скора.

**Рендер карточки.** Заголовок — тикер + «Итог: ±N» (цвет pos/neg/neut). Каждый столп — мини-бар
от центральной оси 0: ширина `= |score|·25%`, заполнение вправо (pos, зелёный) или влево (neg,
красный). Action-бейдж (act-buy зелёный / act-sell красный / act-trim янтарь / act-hold серый;
SELL обязан читаться красным, не янтарным). Внизу — словесная причина по итогу. Если показано
меньше карт, чем позиций — «остальные в HOLD-зоне без триггеров».

**Как читать.** Зелёные столпы (вправо) — сильная сторона, красные (влево) — слабая. Итог >0 —
совокупно позитивный профиль. Прочерк у F/C — пилляр **неприменим** к классу (нет отчётности /
корп. кредитного риска), а не «ноль». Action — итоговый вердикт по бумаге.

---

## 3.C Стресс-сценарии (7 шоков, выпуклый кап)

**Назначение.** Реакция портфеля на 7 гипотетических шоков, калиброванных по историческим
аналогам. Это **проверка устойчивости, не прогноз**.

**Payload-ключ:** `data.stress_scenarios[]` — `{name, port_pct, port_dollar, max_dd_pct,
recovery_months, tag, coverage}`. Дисклеймер методологии —
`data.portfolio_metrics.stress_test_disclaimer`. Плюс `data.ai_stress_comment`.

**Builder / движок.** Прокидка из `results`; считает `finance/stress.py::run_stress_scenarios`.

**Как работает математика** (`finance/stress.py`). Каждый сценарий — **вектор шоков по факторам**
(периодные десятичные доходности). PnL по позиции — линейная комбинация бет с шоком:

```
ΔPnL_i / V_i = Σ_f β_{i,f} · shock_f          (доходность позиции при шоке)
ΔPnL_port    = Σ_i w_i · (ΔPnL_i / V_i) = w' · B · shock
```

Это индустриальный quick-stress (β×шок). **Выпуклый кап (H4):** линейная экстраполяция честна
для умеренных шоков, но β=2.1 при «Market −10%, Momentum −15%» даёт бессмысленные −25% одним
куском (беты оценены на нормальном режиме, в хвостах декомпрессируются нелинейно). Поэтому при
`|x| > T` применяется гладкое (C^∞) выпуклое насыщение с асимптотой `C`:

```
T = 0.20  (порог — ниже линейный pass-through)
C = 0.35  (асимптотический абсолютный кап)
x' = sign(x)·(T + (C−T)·(1 − exp(−(|x|−T)/(C−T))))   при |x| > T;   x' = x  иначе
```

Свойства: непрерывна по значению и производной в `|x|=T` (производная=1), строго монотонна,
`lim |x'| = C` (без жёсткого кинка). Поэтому β=2.1 под Market −20% показывается **не** как −42%.

**Drawdown и восстановление** поверх gross-PnL:
- `est_dd = port_pct − σ_quarter` (только для убыточных, `σ_quarter = σ_ann·√0.25`) — даже если
  квартал закрывается на shock_pnl, худший день внутри обычно на ~1·σ_quarter глубже.
- `recovery_months` — **геометрическое**: `n = ln(1/(1−|dd|)) / ln(1 + r_monthly)`, где ставка
  восстановления — потенциал доходности самого портфеля, зажатый в коридор 8–18%/год (growth-книга
  восстанавливается быстрее среднерыночных 8%, но не нереалистично быстро). Без довложений.

**Каталог из 7 шоков:** Tech-распродажа (как Q2 2022), кредитный blow-out (+200 bps HY), Fed
+50 bps, гео-риск-офф, Fed cut −50 bps, USD +5% rally (proxy), CPI shock +1 пп (proxy). Два
proxy-сценария помечены (нужны UUP.US/TIP.US-факторы).

**Рендер.** Таблица: Δ Портфель %, Δ Стоимость $, бар «Магнитуда» (нормирован к ±20%:
`bar_w = min(|pct|·100, 20)/20·50`), Drawdown %, Восстановление мес. Дисклеймер про выпуклый кап
обязателен (валидатор M1 гарантирует упоминание в ИИ-комментарии).

**Как читать.** Красные Δ — портфель теряет в сценарии, зелёные — выигрывает. Магнитуда —
относительная тяжесть удара. Восстановление — сколько месяцев нужно отрасти обратно. Помните:
β=2-имя НЕ показывается с −42% из-за капа; это **иллюстрация устойчивости, не предсказание**.

---

# Лист 4 — Рыночный режим и макро-контекст

## 4.A Квадрант Growth×Cycle + сводка

**Назначение.** В какой фазе рынок (Expansion / Recovery / Slowdown / Recession), с уверенностью
модели, плюс «живая» точка на квадранте.

**Payload-ключи:** `data.regime` — `{label, confidence, growth, cycle, explainers[], dot_cx,
dot_cy, dot_label}`. Плюс `data.ai_regime_comment`.

**Builder.** Regime-блок `build_payload` + `_regime_dot_coords(growth, cycle)`. Координаты точки:
центр SVG (155,155), **625 px на единицу** скора, клип скоров к ±0.2 и результата к рамке
[40,270] (экстремальное чтение не выводит точку за чарт). Движок-классификатор — `finance/regime.py`.

**Как работает математика классификатора** (`finance/regime.py::RegimeClassifier.classify`).
Две ортогональные оси из моментума ликвидных US ETF (окно 60 дней, MEDIUM_WIN):

- **Growth-ось** (акции лидируют vs облигации): компоненты `SPY−IEF` (60д), `EEM` (60д),
  `0.5·SPY` (120д стабилизатор) → `growth_score = mean(компоненты)`.
- **Cycle-ось** (цикличные vs защитные): `XLY−XLP` (60д), `IWM−SPY` (60д) →
  `cycle_score = mean(компоненты)`.

**Квадрант** (см. ASCII в модуле):
`cycle≥0 & growth≥0 → Expansion` · `cycle≥0 & growth<0 → Recovery` ·
`cycle<0 & growth≥0 → Slowdown` · `cycle<0 & growth<0 → Recession`.

**Confidence (0..1)** = `magnitude_factor × directional_agreement`:
- `magnitude_factor = min(1, hypot(growth, cycle) / 0.20)` — типичный «чёткий» режим на
  расстоянии ≈0.20 от центра;
- `directional_agreement` = доля компонент-сигналов, указывающих в назначенный квадрант (3 из 4
  спредов согласны → робастно; 1 из 4 → почти монетка).
Это убрало прежний баг, где `(0.15, 0.04)` раздувался в 100%.

**Опциональный macro-overlay (gated, OFF по умолч.,** env `REGIME_MACRO_OVERLAY=1`): когда
включён И подан FRED-пак, GDP-рост (→ growth) и безработица (→ cycle) добавляются как
**дополнительные компоненты** осей на том же масштабе, что ETF-спреды. Каждый сигнал — бленд
**уровня** и **темпа** (rate-of-change: безработица 4.1% растущая ≠ той же падающей), ограничен
`±0.05` (`_MACRO_MAX_NUDGE`) — может **наклонить**, но не **перебить** ценовой режим. По умолчанию
классификатор байт-идентичен (overlay не трогает результат).

**SVG-рендер.** 4 квадранта (Expansion синий — целевая фаза, Recovery зелёный, Slowdown янтарь,
Recession красный); оси Growth (вверх) × Cycle (вправо), ±0.2 по краям. Точка «сейчас» — на
`dot_cx/dot_cy`, с пунктирным лидером от центра и подписью `dot_label` (Growth +X · Cycle +Y).

**Как читать.** Верхний-правый квадрант (Expansion) = здоровый рост, риск-он поздний цикл.
Нижний-левый (Recession) = двойное падение. Точка показывает текущие координаты; чем дальше от
центра, тем «чётче» режим. Уверенность <50% = режим близок к границе, сигнал слабый.

> **Аудит (✅):** точка живая (фикс R1) — `(growth +0.11 / cycle +0.07)` → верхний правый =
> совпадает с ярлыком Expansion. Захардкоженной (177,110) больше нет.

---

## 4.B Growth/Cycle факторы

**Назначение.** Числовое значение двух осей режима с человеческой подписью.

**Payload-ключи:** `data.regime.growth`, `data.regime.cycle`, `data.regime.confidence`,
`data.regime.explainers[]`.

**Рендер.** Карточки «Growth-фактор» / «Cycle-фактор»: значение `%+.2f`, цвет pos (>0.02) /
neg (<−0.02) / neut, подпись по знаку («здоровый рост» / «сжатие» / «цикл. экспансия» / …).
Плюс до 2 explainer-строк (RAG-сигналы из `explainers`, формируются в `build_payload` из
`regime["signals"]`: «SPY обгоняет IEF на +X% за 60 дней», «small-caps лидируют» и т.п.).

**Как читать.** Знак и величина каждой оси прямо определяют квадрант (см. 4.A). Explainers
дают «почему» — какие именно спреды двигают режим.

---

## 4.C FRED-сигналы драйверов (+ темп)

**Назначение.** Независимые макро-индикаторы из FRED (6 серий) с уровнем, темпом и статусом
свежести.

**Payload-ключ:** `data.macro_drivers` — `{series: [{id, name, value, as_of, status, comment,
trend_label, trend_dir}], as_of}`.

**Builder:** `_build_macro_drivers_panel(results["macro_drivers"])` + `_macro_series_trend`
(`pdf_payload.py`). Источник: `services/macro_data.py` (`MacroFeed.get_regime_drivers`).

**6 серий (Sprint 6 / BLOCK 3.4):** 10Y−2Y (наклон кривой), HY OAS (кредитный спред), VIX,
breakeven (инфл. ожидания), **UNRATE (безработица)**, **Real-GDP SAAR**.

**Как работает.** Builder форматирует значение по юниту: HY OAS (`BAMLH0A0HYM2`) → bp (×100),
`pp` → «+X.XX pp», `%` → «X.XX%», index (VIX) → «X.X». **Темп (F3)** `_macro_series_trend`:
по истории серии (`history_30d` — последние 30 наблюдений) считает изменение как **OLS-наклон
по ≥3 последовательным изменениям** (общий `finance.regime.series_trend`, ≥4 точек), а не одношаговую
разницу `vals[-1] − vals[-1-lag]`; lag по cadence (daily→21≈1м, monthly→3≈3м, quarterly→3≈3кв).
Чип вида `▲ +0.12 pp за 1м` (level ⊕ rate-of-change — режим реагирует на **направление**, не только
уровень). Статус (`ok/warn/stale/missing`) маппится на цвет: ok→bull, warn/stale→warn, missing→flat;
подпись «актуально/устарело/частично/нет данных».

**Как читать.** Значение — текущий уровень. Чип-темп — куда движется (▲ растёт, ▼ падает).
Инверсия кривой (10Y−2Y < 0), широкий HY (>5.5%), VIX>25 — классические стресс-сигналы (см. 4.D).
Пустой пак → «FRED источник недоступен».

---

## 4.D Детерминированная сверка FRED↔моментум (R3)

**Назначение.** Согласуются ли моментум-режим (из ETF) и независимые FRED-сигналы — **без LLM**.

**Payload-ключ:** `data.regime_consistency` — `{status, note, signals[]}`.

**Builder:** `_build_regime_consistency(regime, macro_raw)` (`pdf_payload.py`).

**Как работает логика.** `risk_on = label ∈ {Expansion, Recovery}`. Стресс-сигналы (пороги
зеркалят DEEP-промпт): инверсия кривой `yc < 0`; широкий HY `hy > 5.5` (550 bp); страх `vix > 25`.
- `risk_on` И ≥2 стресс-сигнала → `diverges` («режим risk-on по моментуму, но FRED показывает
  стресс… трактовать осторожно»).
- `risk_off` И сигналы спокойны → `diverges` («risk-off по моментуму, но макро спокоен — возможен
  ранний разворот»).
- иначе → `aligned` («Макро-сигналы FRED согласуются с моментум-режимом»).

**Рендер.** Зелёная плашка «✓ Макро-сигналы согласованы» или янтарная «⚠ Расхождение
макро-сигналов» + `note`.

**Как читать.** «Согласованы» — моментум и фундаментальные макро смотрят в одну сторону, режиму
можно доверять. «Расхождение» — позиционирование стоит трактовать осторожно (моментум может
обманывать на развороте).

---

## 4.E ИИ-подтверждение режима

**Назначение.** Структурированная ИИ-сверка режима (confirms/partial/diverges) на основе
макро-драйверов, факторных бет и банковского консенсуса — **DEEP-only, advisory**.

**Payload-ключ:** `data.regime_confirmation` — `{stance, summary, signals[]}`.

**Источник.** Прокидка из `ai_summary`. Рендер: ✓ «ИИ подтверждает режим» (зелёный) / ⚠
«Частичное подтверждение» (янтарь) / ✗ «Сигналы расходятся с моделью» (красный) + summary +
список signals + имя модели. Плюс `data.ai_regime_comment` (свободный текст). Полоса источников
внизу листа: Quant Engine, FRED, SEC EDGAR (earnings revisions), Tradernet, ChromaDB Bank PDF, AI.

**Как читать.** Это **второе мнение** ИИ поверх детерминированной сверки (4.D). ✗ = ИИ видит
противоречие с моделью — повод присмотреться. Не рекомендация к сделке.

---

# Лист 5 — AI Ideas · Action Plan · Ожидаемый эффект

## 5.A AI Ideas (4-stage pipeline)

**Назначение.** Стратегические идеи-карточки (рост · ребаланс · защита · Smart Money) с кандидатами
**вне портфеля** и раскрываемым 4-этапным конвейером отбора (DEEP добавляет стадию STRESS и
β/сектор по каждому кандидату).

**Payload-ключи:** `data.ai_ideas` + `data.ideas_count`. Уплощение бакетов в порядке
`['risk_reduction', 'diversification', 'growth', 'rotation', 'hedge']`. Идея: `category`,
`priority`, `title`, `rationale`, `candidates[]` (`{ticker, name, scenario, beta, sector}`),
`pipeline[]` (4 стадии), `expected_effect[]`, `sources[]`.

**Builder:** `_build_ai_ideas(stock_picks, tier="deep")`. Конвейер из `_PIPELINE_DEEP` —
**4 стадии** (vs 3 в BASE), напр. для `boost_alpha`: FACTOR (Momentum+Quality) → REGIME
(Growth×Cycle квадрант) → **STRESS** (устойчивость при rate shock +200 bps) → RAG (инвестбанки).
Аналогично rebalance/protect_capital/**smart_money** получают свою STRESS-стадию (equity shock −20%,
positive P&L при recession, устойчивость идеи при смене режима). `smart_money` (4-я карточка,
рендерится в bucket `rotation`): FACTOR (институц. потоки + инсайдерские покупки) → REGIME
(накопление умных денег в квадранте) → STRESS → RAG (13F/Form 4 + банковский консенсус).

**Источник пиков.** `ai_summary.stock_picks` от `ai_narrative` — модель **`claude-opus-4-8`**.
Правило DATA-DRIVEN (привязка к портфелю) + СВЕЖЕСТЬ ИДЕЙ (дневной срез `YYYY-MM-DD` + дневной
«угол ротации»). Opus опускает `temperature` → дисперсию даёт директива свежести. Фильтры:
held-tickers → contradiction → backfill (помечается «резервный каталог»).

**Рендер.** Цветная полоса по категории (risk/diversify/rebalance/grow/hedge/rotation —
расширенный маппинг по RU/EN ключам, чтобы новый бакет не терял бейдж). Свёрнуто: title +
rationale + чипы кандидатов + «N кандидатов (не в портфеле)». Раскрыто: 4-stage pipeline,
«Кандидаты: бета · сектор · сценарий» (с чипами β и сектора), «Ожидаемый эффект (3 метрики ·
привязка к 8-card)», источники (chip `[Stress …]` янтарный, `[RAG …]`, прочие — `data`).

**Как читать.** Кандидаты — **не из вашего портфеля** (замена/дополнение). Стадия STRESS
показывает, что идею проверили на устойчивость к шоку. β/сектор по кандидату дают быстрый
контекст. **Аналитические идеи, не рекомендация.**

> **Аудит (✅ DEEP):** образцовый data-driven результат — свежие имена (MSCI, ANET, META, SPGI…)
> с привязкой к портфелю («недовесен в сетевой инфраструктуре AI», «P/E ниже NVDA»). Зацикленность
> ушла.

---

## 5.B Action Plan (уровни Buy/Sell/Stop)

**Назначение.** Перевод стратегических идей в конкретные исполняемые уровни: текущая цена,
Buy zone, Sell target, Stop loss + причина.

**Payload-ключ:** `data.action_plan[]` — `{ticker, action, price, buy_zone, sell_target,
stop_loss, reason, delta_w_pp}`. Плюс `data.ai_action_comment`.

**Builder.** `tier=="deep"` блок `build_payload` (`plan_rows`): берёт `results["action_plan"]`,
подмешивает текущую цену из `perf_df.Current_Price`, форматирует зоны (`f"{lo:.2f} – {hi:.2f}"`).
`sell_target` = `take_target` (для BUY) или `sell_zone` (для Trim/Sell) — чтобы колонка была
осмысленной в каждой строке. Кэш исключён (нет уровней). Движок — `action_plan.build_action_plan`.

**Как работает математика** (`finance/action_plan.py::compute_levels`). Уровни анкорятся **только**
на технические структуры самой бумаги (ATR Wilder RMA, SMA50/100/200, 52w-high, RSI14, MACD) —
без внешних таргет-прайсов, всё воспроизводимо.

- **Buy/Strong Buy:** `buy_lo = SMA50 − 1·ATR`, `buy_hi = SMA50` (при RSI>75 зона сдвигается на
  −0.5 ATR; oversold — до текущей цены; MACD<0 — менее агрессивно). Take-profit =
  `max(SMA200·1.05, price + 3·ATR·ms)` (52w-high только *поднимает* таргет, если основной попал в
  зону сопротивления у хая). Stop = `max(price − 2·ATR·ms, SMA200)`.
- **Trim/Sell:** `sell_zone = (SMA50, SMA50 + 1·ATR)`; stop = `price − 2·ATR·ms`.
- **Hold:** только защитный stop = `max(price − 2.5·ATR·ms, SMA100)`.

**Mandate-scale ATR (A3):** множитель `ms` применяется **только к ATR-дистанциям** stop/take
(SMA-якоря и 52w-high не масштабируются): `MANDATE_LEVEL_SCALE` = Conservative 0.75 / Moderate 1.00
(исторический дефолт) / Aggressive 1.25. Консерватор — теснее стопы и ближе тейки.

**Турновер-кап.** Сортировка по приоритету (`Sell:0, Trim:1, Strong Buy:2, Buy:3, Hold:4`);
кумулятивный `|Δw|` ограничен `MAX_TRADE_BLOCK_PORTFOLIO_PCT = 0.25` (25% NAV за отчёт) — после
исчерпания строка демотируется в Hold с пометкой «· deferred (turnover cap)». (+ BL `max_active_share`
по мандату на стадии BL.)

**Как читать.** Buy zone — диапазон цен для входа; Sell target — цель фиксации; Stop loss — уровень
защиты. Причина — короткое обоснование (Score, Hotspot, RSI). «deferred (turnover cap)» = идея
хорошая, но бюджет ребаланса на этот отчёт исчерпан.

---

## 5.C Ожидаемый эффект (8 карт before→after + вердикт)

**Назначение.** Оценка «до/после» исполнения плана по 8 риск-метрикам — что улучшится, чем
придётся пожертвовать.

**Payload-ключ:** `data.expected_effect` — flat dict `{risk_index, cvar_95, sharpe, max_drawdown,
vol, max_erc_pct, it_share, expected_return}`, каждая = `{before, after, delta_pp, favourable}`;
плюс `verdict`, `scoped_to_high_priority`, `high_priority_tickers[]`, `driver`. И `data.ai_effect_comment`.

**Builder:** `_build_expected_effect(results["expected_effect"])` (флэттенер, `pdf_payload.py`) —
переводит вложенный `{metrics: {<engine_name>: …}}` движка в карточные ключи. Движок —
`finance/simulate.py::simulate_after_plan`.

**Как работает математика** (`finance/simulate.py`). Метрики пересчитываются на новом векторе
весов под BL-таргетами (или high-priority, см. ниже):
- **Vol / Max TRC** — структурно: `σ = √(w'Σw)`, `ERC%ᵢ = wᵢ·MCTRᵢ/σ`, `Max TRC = max(ERC%>0)`
  (только положительные вклады — отрицательный ERC = хедж, не концентрация).
- **CVaR / Sharpe / MaxDD** — **sample-replay** «как если бы новые веса держались всегда»:
  `port_daily_new = daily_log_matrix @ w_new`; CVaR = среднее нижних 5%; Sharpe = `(exp(mean·252)−1
  − RFR)/(σ_daily·√252)` (геометрическая аннуализация, F-7); MaxDD = `min(eq/running_max − 1)`.
  «before» **якорится к headline-метрикам** отчёта (`_anchor`), а симулированная Δ переносится
  на «after» — чтобы панель была консистентна с KPI-полосой.
- **Composite risk index** — та же формула, что гейдж (`0.4·vol + 0.4·|CVaR| + 0.2·maxTRC`).
- **Expected return** — `Σ wᵢ·μᵢ`, где μ — BL-постериор (или realised-фолбэк). ZERO active views ⇒
  отказ (циркулярный π), фолбэк на realised.
- **IT share** — `Σ wᵢ` для Technology (sector lookup + prefix-эвристика).

**Флаг `favourable` (цвет дельты).** «Improved» = движение в желаемую сторону:
`_RISK_METRICS_LOWER_IS_BETTER = {volatility_ann, max_trc, risk_index}` → лучше при уменьшении;
CVaR/MaxDD — отрицательные, улучшение = движение к нулю (положительная Δ); Sharpe/return — лучше
при росте. `it_share` ∈ `_NEUTRAL_METRICS` → `favourable=None` (нейтраль): направление зависит от
мандата (для Aggressive рост tech — не «ухудшение»). **Нулевая дельта → нейтраль.** Флэттенер
дополнительно: если **отображаемая** дельта округляется до нуля (`|delta_pp| < 0.05`),
`favourable=None` (честность дисплея — «+0.0 пп» не горит зелёным).

**Скоуп (BLOCK 2.3).** Симуляция идёт на **высокоприоритетных** action-строках (non-deferred
Buy/Sell/Trim, `|Δw|>0`) — `high_priority_target_weights`, а не на полном BL-векторе. Payload
несёт `scoped_to_high_priority` + `high_priority_tickers` + `driver`
(`high_priority_action_plan` / `bl_target_fallback`). UI помечает: «⚡ Δ только по
высокоприоритетным идеям: <тикеры>».

**Вердикт ребаланса** (`simulate._verdict`): `improvement` (vol↓ без компромиссов) /
`tradeoff` (vol↓ ценой роста концентрации/просадки) / `degradation` (vol↑) / `neutral`
(маржинальный эффект). Рендерится цветной плашкой «Сводный вердикт по плану» + список «ухудшено».

**Рендер 8 карт** (`_ef_card`): before → after с форматом по типу (pct ×100 / int / raw 2dp);
дельта `%+.1f пп` (Sharpe — `%+.2f` без «пп»). Зелёная (`fav`) / красная (`fav is not none`) /
нейтральная. Плюс плашка «Временные оси» (Sharpe/CVaR/MaxDD/Vol — **исторические** trailing;
Expected return — **форвардная** из BL; сравнивать напрямую нельзя).

**Как читать.** Зелёная дельта = метрика улучшилась, красная = ухудшилась, серая = нейтрально/
ноль. Вердикт «Компромисс» честно сообщает: риск снизили, но концентрация/просадка выросли.
`it_share` всегда нейтрален (направление зависит от мандата). «⚡ только по приоритетным идеям» =
дельта относится к конкретным сделкам, не к полной перетряске.

> **Аудит:** «+0.0 пп» раньше горел зелёным (дельта +0.04 округлялась) → фикс: `|delta_pp|<0.05`
> → нейтраль. `it_share` нейтрален вживую (−14.7 пп без окраски). Вердикт «Компромисс» честен
> (vol −3.8 пп improved vs Max TRC +12.2 пп worsened). Кап на ВЕС 20% ≠ кап на ВКЛАД В РИСК —
> TRC может вырасти; TRC-aware ограничение — кандидат в следующую фазу.

---

# Лист 6 — CoVe (Chain-of-Verification)

**Назначение.** Полная data-lineage: каждый показатель прослеживается до первичного источника с
методом расчёта и QualityGate-статусом свежести. Заменяет захардкоженный «verified sources» блок.

**Payload-ключ:** `data.cove_lineage[]` — строки `{name, source, method, status, as_of, note,
freshness_days}`.

**Builder:** `finance/data_lineage.py::build_lineage(results, ai_summary)` — чистая функция, **без
дополнительных запросов** (читает только готовый `results` + пару колонок perf_df). Стабильная
схема (каждая строка — те же ключи). **Порядок строк = порядок рендера.**

**Состояния (статус-таксономия,** зеркалит `MacroFeed`): `ok` (✓, в окне свежести),
`warn`/`stale` (`!`, частично/устаревший кэш), `error` (`✗`, источник СБОЙНУЛ — ошибка),
`missing`/`disabled` (нейтральный `–`, источник намеренно НЕ запрашивался: фича выключена /
ключ не задан). Важно: `missing`/`disabled` НЕ путать с красным `✗` (`error`) — выключенная
фича ≠ сломанный источник.

**Что в lineage (по порядку):**
1. **Vol · CVaR · TE · IR · Max DD** — Quant Engine MAC3; метод «Wilder RMA · EWMA hl=63 (λ≈0.99)
   ⊕ Ledoit-Wolf 70/30 · Politis-Romano bootstrap CI». (Фикс 5.4: подпись EWMA исправлена
   с `λ≈0.94` на `hl=63, λ≈0.99`.)
2. **TRC (Euler) · MCTR · CVaR** — `MCTR = Σw/σ_p · ERC%ᵢ = wᵢ·MCTRᵢ/σ_p`.
3. **Факторная независимость (мультиколлинеарность)** — `_factor_diagnostic_status`: κ (condition
   number) + max|corr| факторов (BLOCK 4.6). `Σ=B·F·Bᵀ+D` несёт полную ковариацию факторов без
   двойного счёта; `warn` при near-collinear (max|corr|>0.95 или κ>30).
4. **Цены и история активов** — Tradernet (Freedom), daily CLOSE 730d, ATR via OHLC; статус по
   возрасту последнего close (`stale` если >5 кал. дней).
5. **Валютный слой (конверсия + ставка)** — `_fx_status` (Sprint 5.4): Base Currency Approach —
   каждая цена конвертируется в валюту отчёта ДО ковариации, Sharpe/Sortino на согласованной RFR.
   `ok` (USD-портфель / полное покрытие) или `warn` (T-1 фолбэк / покрытие <90%).
6. **Fundamental Z-scores** + **Altman-Z · Piotroski-F · Interest Coverage** — SEC EDGAR
   CompanyFacts (2 строки); статус деградирует по числу непокрытых тикеров и возрасту 10-K (warn
   ≥18 мес, stale ≥30 мес).
7. **CDS spreads** — FRED HY proxy + WGB sovereign; QualityGate 1–3000 bps; статус по покрытию.
8. **Action levels** — ATR Wilder RMA + SMA50/200 + RSI(14) + MACD(12,26,9).
9. **Black-Litterman target weights** — reverse-optimisation prior + score-views, τ=0.05.
10. **Регим-классификатор** — Growth×Cycle factor returns, окно 60 дней; note «<режим> ·
    confidence N%».
11. **Стресс-сценарии** — параметрические факторные шоки, per-asset β, linear PnL; `warn` если
    есть proxy-сценарии.
12. **Макро-драйверы (FRED)** — по строке на серию (6 серий: yield curve, HY, VIX, breakeven,
    unemployment, GDP); статус из самого FRED-пака. (Мёртвый «PMI» убран — discontinued.)
13. **Smart-Money (инсайдеры SEC Form 4)** — gated (`_smart_money_status`); `missing` пока
    провайдер не подключён.
14. **Bank RAG (выдержки)** — ChromaDB · GS/MS/JPM PDF; cosine retrieval; `ok`/`missing` по
    `used_rag`.
15. **AI verdict · bullets** — Anthropic · <display-модель> (через `_model_display_name` — фикс 5.2,
    иначе сырой id «claude-sonnet-4-6»); помечен «не является ИИР».
16. **LLM-чекер: контроль галлюцинаций** + **LLM-чекер: проверка вычислений** —
    `_llm_checker_status` (BLOCK 4.8): held-фильтр + DATA-DRIVEN + фильтр противоречий пиков;
    leverage-phrasing (без «удваивается» на 1.16x) + упоминание выпуклого капа стресса + запрет
    ре-агрегации секторов.

**Black-Litterman — для справки** (`finance/black_litterman.py`, формирует BL-таргеты, питающие
Action Plan и Ожидаемый эффект): `π = δ·Σ·w_mkt` (equilibrium prior); посериор по Идзореку
`μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹π + PᵀΩ⁻¹Q]`, `Ω = diag(P·τΣ·Pᵀ)`; целевые веса
`w* = (δΣ)⁻¹μ_BL`, ренормированные к 1; τ=0.05, `max_active_share=0.25` (мягкий кап турновера).
При 0 активных views посериор = голый π (помечается `n_views=0`).

**Рендер.** Двухколоночная сетка: иконка (✓/!/✗) + жирное имя + мета-строка
(`source · method · as_of · note`). Легенда внизу + дисклеймер «Документ не является ИИР».

**Как читать.** ✓ — источник прошёл QualityGate. `!` — частичное покрытие / fallback на
устаревший кэш. ✗ — источник недоступен. Это полный аудиторский след: за каждым числом отчёта
видно, откуда оно и как посчитано. CoVe-строка ≠ оценка качества портфеля — это самоаудит данных.

> **Аудит:** CoVe вырос с 13 до 16+ строк (Sprint 6): + факторная независимость (κ),
> + валютный слой, + Smart-Money (gated), + 2 LLM-чекера. Подпись EWMA исправлена. Модель
> показывается человекочитаемым именем, а не сырым id.
