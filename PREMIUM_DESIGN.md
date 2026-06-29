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

---

## Live-report fixes & mobile (2026-06-28) — §−5 в `AUDIT.md`

**Маппер-контракт (`premium_payload.py`) — ключи, которые легко перепутать с источником:**
- `holdings[].fund` ← джойн `fundamental_layer[]` по тикеру (НЕ `assets[].fundamentals`); ETF/кэш → «н/д».
- `regime.drivers` ← `macro_drivers.series[]` (адаптированная панель, НЕ сырой dict); фильтр value=«—».
- `regime.confirmBullets` ← `_signal_obj` парсит `«✓/⚠/✗ текст»` → `{ok,t}` (компонент рисует иконку+текст).
- `sectorWarn` ← `_warn_text` берёт поле `text` (источник — список dict, НЕ строк).
- `mandate` ← `target_vol_pct`/`target_te_pct`/`breaches`/`{actual,status}` (НЕ `target_vol`/`value`/`state`).
- `actionPlan[].score|hot` ← джойн `score_breakdown.total` + `assets.hotspot` (нет на `action_plan[]`).
- `topHotspot` (BASE) ← актив с макс. `euler_risk_pct` (НЕ `hotspots`, который список СТРОК).
- `effect[].before/after/delta` — форматируются per-metric (`_eff_fmt`/`_eff_delta`); источник хранит сырые float.
- идея `pipeline[]` ← `_pipe_step` (деталь-строка; стадию-метку даёт компонент по позиции).
- BASE «AI · {модель}» — из `meta.aiModel`, не хардкод.

**Мобайл (`custom.css`, `@media (max-width:640px)`):**
- НЕ ставить `min-width:0` на ячейки `grid-cols-[…minmax(0,fr)…]` — треки схлопываются и числа НАКЛАДЫВАЮТСЯ.
- Широкие таблицы (holdings/action-plan/stress) обёрнуты в `.mob-scroll-x` → гор. скролл, натуральная ширина;
  на десктопе обёртка инертна (правила только внутри `@media`). Идея-карточки: `TickerCard dark` проп (явные цвета).

**Пересборка статики:** `design/premium_v2/build.sh`; CSS-шаг — Tailwind `content` glob `./design/**/*.jsx` работает
из КОРНЯ репо (build.sh делает `cd design/premium_v2` → запускать tailwind отдельно из корня, либо чинить glob).

## Round 6 (2026-06-28) — template-data, цвета, мобильный UX

**Принцип «нет литералов в компонентах»:** премиум-маппер ЧИТАЕТ payload; React-компоненты НЕ хранят
данных. Любое литерал-число/процент/текст в JSX = template-leak (баг). Найдены и устранены хардкоды:
AI-инсайт (HAIKU/Индекс 62), AI-сводка идей (62% в технологиях), весь Performance-блок
(+14.2%/+5.1пп/S&P +9.1%/vol 14.8% — противоречил реальности). Все → реальные `verdict.*` / `performance.summary`.

**Новые ключи маппера:** `performance.summary {ret,exc,spx,volPort}` (реальные 12-мес); период `d = p − s`;
`verdict.riskTier` (BASE); `sectors[].hue` через `_sector_hue` (палитра, не единый #1c1b1a).

**Мобильный UX (`custom.css`):**
- `.mob-scroll-x` — CSS scroll-shadow (Lea Verou, `background-attachment: local/scroll`, без JS): правая тень
  видна пока есть прокрутка, гаснет в конце. Плюс чип `.swipe-hint` «↔ листайте таблицу» над каждой таблицей.
- `.mob-detail` — раскрытый фундаментал в holdings: `position: sticky; left:0; width:88vw` — выходит из
  гор. скролла строки, показывает все 6 метрик. DEEP-конвейер идей: стек вертикально `<sm`.

**Цвета графиков:** waterfall — отдельно=#1c1b1a, диверсификация=#c47358, итог=#f5d04e (по легенде); `maxV`
масштабируется по ВИДИМОМУ пику (не по полному sum_standalone) — нет пустоты сверху.

## Round 7 (2026-06-28) — кнопки, waterfall-сходимость, пустые боксы

- **Период-селектор** (BASE Performance) теперь функционален: читает выбранный период из `p.periods`
  (метки `1М/3М/6М/12М/YTD`) и обновляет headline + карточки + подсветку строки. Раньше менял state вхолостую.
- **Мёртвые декоративные «↗» кнопки** удалены (нет onClick, не нужны для масштабирования). Кнопки-продукт
  (Пересчитать/Экспорт/Применить идею/nav/FAB) сохранены — это будущие интерактивные функции.
- **Waterfall** «Риск портфеля» теперь СХОДИТСЯ: добавлен бар **«Прочие»** = `sumStandalone − Σпоказанных`,
  чтобы ΣstandAlone + diversification = total (раньше показывались только top-4, диверсификация «висела»).
- **Пустые боксы**: BASE-идеи не несут effect/sources → бокс «Ожидаемый эффект» рендерится только при контенте.
- Бейдж декомпозиции — реальный `riskDecomp.total` (был хардкод 14.2%).

## Round 8 (2026-06-29) — BASE: перенос гейджа, периодный график, чистка

Все правки в BASE-бандле (`portfolio-overview.jsx`, `portfolio-performance.jsx`, `portfolio-charts.jsx`) +
маппер `_map_performance`.

**Обзор (`portfolio-overview.jsx`):**
- **«Индекс риска» перенесён в hero** — компактная горизонтальная карточка `HeroRiskGauge` (гейдж 148px +
  дельта + «Пересчитать») стоит в правой колонке под hero-статами, заполняя ранее пустое пространство справа
  от заголовка. Из нижней сетки `RiskGaugeCard` удалён.
- **«Прочие метрики» + штрих-бокс удалены** (декоративная заглушка без данных).
- **Нижняя сетка перебалансирована** под 3 карточки: `TopHotspot` (lg:3) · `RiskDecomp` (lg:5, шире — водопад
  дышит) · колонка `SectorMix + AIInsight` (lg:4), `items-stretch` → ряд выровнен по высоте.
- **`RiskDecompCard` / `SectorMixCard` приведены в порядок**: единые отступы/типографика, «Итог» справа в шапке,
  подпись «Вклад позиций в волатильность, %»; легенда секторов — цветные точки `s.hue` (совпадают со столбиком),
  табличное выравнивание процентов; бейдж «Перевес» только при наличии `warn`-сектора.
- **`HeroStat` получил `small`** — длинное текстовое значение (мандат-профиль «Умеренно-агрессивный») рендерится
  компактно (`text-base`, перенос), а не 5xl-числом, ломавшим hero. Маппер уже шлёт `heroStats[Профиль].small`.

**График доходности (`portfolio-performance.jsx` + `PerfChart`):**
- **График теперь РЕАГИРУЕТ на период** (раньше брал статичные `p.months/port/spx`). Кривая для выбранного окна
  строится ВЛОЖЕНИЕМ конечных точек периодов — без интерполяции: `cum[старт…−h] = (1+r_окна)/(1+r_h) − 1`.
  Каждая точка — алгебра над теми же числами, что в таблице → **конец графика == headline == строка таблицы**.
- **YTD больше не точка на оси** трейлинг-кривой (был нелогичен) — это самостоятельное ОКНО в селекторе;
  ось показывает `старт · −6М · −3М · −1М · сейчас` (для YTD — span из календарного месяца `meta.generated`).
- **Периоды отсортированы** `1<3<6<12, YTD последним` (в селекторе И таблице; источник иногда шлёт YTD между 3 и 6).
- **`PerfChart` переписан**: домен знако-зависимый и всегда включает базовую линию 0% (убыточное окно рисуется
  корректно — линия вниз, видимый ноль, подписи оси со знаком); конечные подписи значений сдвинуты ВПРАВО от
  маркера внутрь `padR` (раньше обрезались у края) + развод при коллизии port/spx; адаптивная точность тиков.
- **`undefined%` волатильности** устранён: `summary.volPort ?? vol.port`, и карточка показывает `—` при отсутствии.

**Маппер `_map_performance`:** перестал собирать мёртвую кривую `months/port/spx` (компонент строит её сам);
ship-ит только `periods [{label,p,s,d}]`, годовую `vol` и `summary` (для `volPort`). `d = p − s` сверен.

**Мобайл:** факторные пиллы → `grid-cols-2 sm:flex` (аккуратные 2 в ряд на телефоне); `PeriodRow` — `whitespace-nowrap`
на «пп» + тесные отступы `<sm`; hero — `clamp()`-заголовки, `min-w` снижены, правая колонка `w-full <lg`. Гор.
переполнения страницы нет (`scrollWidth == viewport @390`).

**Математика сверена** на реальном payload (NVDA-портфель): водопад сходится (Σstandalone − divers = total),
`excess = p − s` по всем периодам, конец нести-кривой совпадает с таблицей и headline во всех окнах (1М/3М/6М/12М/YTD).
