# PREMIUM_DESIGN.md — Premium V2 report design (Portfolio Premium DEEP/BASE)

> Источник дизайна: `analytics.zip` → `design/premium_v2/` (React + Tailwind).
> Рендер: `src/premium_renderer.py`. Сборка статики: `design/premium_v2/build.sh`.
> **Статус:** самодостаточный премиум-рендер для DEEP и BASE **работает** (302/255 КБ, без CDN,
> рендерится pixel-в-pixel — проверено Playwright). Осталось — **маппер «движок → контракт дизайна»**
> для подачи ЖИВЫХ данных портфеля (сейчас рендерится эталонными данными дизайна).

## 1. Что это за дизайн

Присланный `Portfolio Premium DEEP (standalone).html` — **собранный React-бандл** (пустой `<body>` +
1.4 МБ минифицированного JS). Но в zip пришёл **ИСХОДНИК**: React-компоненты на JSX + Tailwind-тема.

- **Стек:** React 18 + Tailwind CSS (кастомная тема). Компоненты: `design/premium_v2/deep/*.jsx`
  (DEEP, 10 файлов) и `design/premium_v2/portfolio-*.jsx` (BASE, 8 файлов).
- **Дизайн-система (Tailwind theme):** палитра cream `#fbf8f1/#f6f3ea/#efe9d8` · ink `#100f0e…#e2ddd1`
  · gold `#fbe48a/#f5d04e/#eac233/#caa01a/#9a7a10` · sage/rust акценты; шрифты **Manrope + JetBrains Mono**;
  `rounded-4xl`(32px)/`5xl`(40px); премиум-тени `shadow-card/card-lg/dark`; стекло `glass-strong`, `lift`-ховеры;
  фон — радиальный золотой градиент. Конфиг — `design/premium_v2/tailwind.config.js`.
- **Мобайл:** адаптив встроен в Tailwind (префиксы `sm:/lg:/xl:` в компонентах — напр. `hidden lg:flex`
  для десктоп-нав, мобильный FAB-навигатор `<Fab/>`); проверено на 390px.

## 2. Как это собирается в production (без CDN, без Babel-runtime)

Дизайн — «no-build» (React+Babel+Tailwind через CDN, что в проде/headless недопустимо: CDN блокируется,
Babel-в-браузере медленный). Поэтому статика **прекомпилируется** (`build.sh`):

1. **Tailwind → статический CSS** (`report.compiled.css`, 28 КБ) — сканирует JSX, кладёт только нужные утилиты.
2. **JSX → JS** через Babel, **classic-runtime** (`React.createElement`, без `import`), data-free →
   `deep-components.js` / `base-components.js`.
3. **React UMD** (`react(-dom).production.min.js`) вендорится и инлайнится.

`src/premium_renderer.render_premium(tier, data)` собирает самодостаточный HTML:
`<style>{CSS}</style>` + `<div id=root>` + inline React + `window.DEEP={data}` + components. Единственная
внешняя зависимость — Google Fonts (graceful degrade в системные). **Размер ~300 КБ** (vs 1.5 МБ бандла).

## 3. Контракт данных (что подаёт движок)

Эталон — `design/premium_v2/{deep,base}-data.sample.json`. Верхнеуровневые ключи:

- **DEEP (29):** `meta, verdict, mandate, heroStats, kpis, concentration, concAI, sectors, sectorWarn,
  holdings, holdingsAI, factors, factorAI, factorCoverage, scores, scoresAI, scoresNote, riskDecomp,
  stress, stressAI, regime, actionPlan, actionAI, effect, effectAI, effectVerdict, cove, quality, ideas`.
- **BASE (11):** `meta, verdict, kpis, factorPills, heroStats, topHotspot, sectors, riskDecomp,
  holdings, performance, ideas`.

### Маппинг «движок → контракт дизайна» (ОСТАВШАЯСЯ работа)

Движок уже считает все эти данные (см. `REPORT_SECTIONS.md`), но в ДРУГОЙ форме (payload v3-шаблона).
Нужен адаптер `engine_results → design_data` (≈как `pdf_payload.build_payload`, но в форму контракта):

| Контракт дизайна | Источник в движке (payload v3 / results) |
|---|---|
| `meta` | `tier`, `user_id`, `ai_model_used`, `total_value`, `n_positions`, `generated_at` |
| `verdict{headline,sub,riskIndex,riskTier,summary,bullets}` | `ai_verdict`, `ai_plain_summary`, `risk_pct`, `risk_label`, `ai_bullets[]` |
| `mandate{rows[label,value,lo,hi,state]}` | `mandate_compliance.rows` |
| `kpis[{name,value,status,sub,ai,color,pts}]` | `cvar/sharpe/max_drawdown`+`*_dollar`+`ai_*_note`+`kpi_sparklines` |
| `holdings[]`, `concentration`, `riskDecomp` | `assets[]` (вес/β/TRC/PnL/action), `hotspots[]`, `risk_waterfall` |
| `sectors`, `sectorWarn` | `sectors[]`, `sector_warnings`, `sector_complex` |
| `factors`, `factorCoverage`, `factorAI` | `factor_betas`/`factor_scores`, `data_quality`, `ai_factor_comment` |
| `scores`, `scoresAI` | `score_breakdown[]`, `ai_4pillar_comment` |
| `stress`, `stressAI` | `stress_scenarios[]`, `ai_stress_comment` |
| `regime` | `regime{...}`, `macro_drivers`, `regime_confirmation` |
| `actionPlan`, `actionAI` | `action_plan[]`, `ai_action_comment` |
| `effect`, `effectVerdict`, `effectAI` | `expected_effect{...}` (вкл. `high_priority_actions`), `ai_effect_comment` |
| `cove`, `quality` | `cove_lineage[]`, `integrity_checks[]` |
| `ideas` | `ai_ideas{growth/diversification/hedge/rotation(=Smart Money)}` |

## 4. План включения в прод (следующий шаг)

1. Написать `src/premium_payload.py: build_design_data(results, tier) -> dict` (адаптер выше).
2. В `tg_bot`/`html_renderer`: за фичефлагом `PREMIUM_REPORT=1` рендерить
   `premium_renderer.render_premium(tier, build_design_data(results, tier))` вместо v3-шаблона.
   Флаг — чтобы прод-доставка (v3) не ломалась, пока маппер валидируется на проде.
3. QA каждой секции: Playwright-скриншот desktop+mobile + сверка чисел с движком (как в аудитах −4.x).

## 5. Файлы

```
design/premium_v2/
  deep/*.jsx, portfolio-*.jsx     # исходные React-компоненты (дизайн)
  tailwind.config.js              # кастомная тема (cream/gold/ink, шрифты, радиусы, тени)
  build.sh                        # воспроизводимая сборка статики
  report.compiled.css, custom.css # скомпилированный Tailwind + кастомный CSS (фон/glass/lift)
  deep-components.js, base-components.js   # прекомпилированные data-free бандлы
  react(-dom).production.min.js   # вендоренный React UMD
  {deep,base}-data.sample.json    # эталонные данные дизайна (демо/фолбэк)
  {deep,base}_production_demo.html # готовые самодостаточные премиум-отчёты (с эталонными данными)
src/premium_renderer.py           # сборщик отчёта: assets + данные → HTML
```

V1 (синий IBM-Plex) и V2-reskin (gold, текущий прод) сохранены: `report_*_v1_design_prototype.html`,
`report_*_v3.html`. Premium V2 (этот документ) — отдельная ветка, включается флагом после маппера.
