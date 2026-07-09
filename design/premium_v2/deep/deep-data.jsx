/* DEEP-tier Risk Report — full structured data (real numbers from the engine run U148046720) */

const DEEP = {
  meta: {
    tier: 'DEEP',
    id: 'U148046720',
    engine: 'MAC3',
    aiModel: 'Claude Opus 4.8',
    profile: 'Умеренно-агрессивный',
    generated: '23.06.2026 07:05 UTC+5',
    nav: 13692,
    positions: 11,
    botUsername: 'KEN_investment_bot',
  },

  verdict: {
    headline: 'Слишком много риска в одной акции',
    sub: 'NVDA одна несёт треть всего риска. Сократите её и продайте слабые ORCL и SPCX.',
    riskIndex: 49,            // 0-100
    riskTier: 'Умеренный',
    summary: 'Портфель сильно перекошен в технологии (почти две трети капитала), а NVDA одна несёт 35% риска. Главное сейчас — урезать NVDA и распродать ORCL, SPCX, переложиться в более широкие имена.',
    bullets: [
      { tag:'Риск',    text:'В худший день из 20 теряется ≈2,7% (≈$370 из $13 692), портфель колеблется на 18% в год, исторически падал на 20%.', src:'Quant Engine' },
      { tag:'Состав',  text:'NVDA даёт 35% всего риска при доле 15,5% — главная опасная точка. ORCL — 18% риска при оценке −6 (худший балл).', src:'Quant Engine' },
      { tag:'Секторы', text:'Технологии 49,1% + полупроводники 15,5% = вместе 64,6% — чрезмерный перекос, уязвим к развороту в техах.', src:'Quant Engine' },
      { tag:'Факторы', text:'Чувствительность к рынку зашкаливает — NVDA двигается в 2,3× сильнее рынка, MSFT в 1,56×; портфель усиливает любое падение.', src:'Quant Engine' },
      { tag:'Режим',   text:'Экономика растёт (уверенность 74%), кривая доходности положительная, страх рынка низкий (VIX 16,8) — рост оправдан, но концентрация опасна.', src:'Regime' },
    ],
  },

  // Mandate compliance (Умеренно-агрессивный)
  mandate: {
    profile: 'Умеренно-агрессивный',
    targetVol: 14.0,
    trackingCap: 6.0,
    violations: 1,
    rows: [
      { label:'Акции США',          value:69, lo:30, hi:60, state:'over'  },
      { label:'Облигации',          value:20, lo:10, hi:30, state:'ok'    },
      { label:'Сырьё / Commodities', value:9, lo:0,  hi:15, state:'ok'    },
      { label:'Global ETFs',         value:0, lo:10, hi:40, state:'under' },
      { label:'Крипто',              value:0, lo:0,  hi:5,  state:'ok'    },
    ],
  },

  // hero stat triplet
  heroStats: [
    { label:'Позиции',  value:11,        icon:'briefcase' },
    { label:'NAV',      value:'$13.7K',  icon:'wallet' },
    { label:'Профиль',  value:'Мод.-агр.', icon:'shield', small:true },
  ],

  kpis: [
    { key:'cvar',   name:'CVaR 95%',     value:'−2.7%', status:'normal', sub:'худшие 5% дней · 1 день · ≈$368 · разброс −3,2…−2,2%',
      ai:'В худший день из 20 теряется ≈2,7% (≈$370). Для умеренно-агрессивного профиля — в пределах нормы.',
      color:'#5d7c5c', pts:[12,10,8,7,7,17,17,28,18,18,5,18] },
    { key:'sharpe', name:'Sharpe Ratio', value:'0.45', status:'good', sub:'Sortino 0,61 · отстаёт от рынка на 4,3%',
      ai:'Отдача на единицу риска низкая (0,45) — риск окупается слабо, портфель отстаёт от бенчмарка.',
      color:'#caa01a', pts:[7,4,10,12,24,22,25,28,20,14,15,15] },
    { key:'dd',     name:'Max Drawdown', value:'−20.2%', status:'watch', sub:'пик → дно · ≈ −$2 759',
      ai:'Исторически портфель падал на 20% от пика — для владельца это потеря ≈$2 750 в плохой период, ощутимо, но терпимо.',
      color:'#c47358', pts:[4,4,4,9,13,13,18,22,28,18,17,17] },
  ],

  // risk concentration table (top contributors)
  concentration: [
    { t:'NVDA',  w:15.5, beta:2.30, risk:35.1, status:'HOTSPOT' },
    { t:'ORCL',  w:9.0,  beta:1.46, risk:18.1, status:'NORMAL'  },
    { t:'GOOGL', w:12.7, beta:1.41, risk:15.3, status:'NORMAL'  },
    { t:'MSFT',  w:16.5, beta:1.56, risk:14.6, status:'NORMAL'  },
    { t:'AAPL',  w:11.0, beta:1.36, risk:9.4,  status:'NORMAL'  },
  ],

  // risk waterfall — standalone contributions, diversification, total (annualised vol %)
  riskDecomp: {
    standalone: [
      { t:'NVDA',   v:7.5 },
      { t:'ORCL',   v:4.8 },
      { t:'MSFT',   v:4.1 },
      { t:'GOOGL',  v:4.0 },
      { t:'+6 др.', v:8.5 },
    ],
    diversification: -10.9,
    total: 18.0,
    sumStandalone: 28.9,
  },
  concAI: 'Наибольший вклад в риск даёт NVDA — 35% всего риска портфеля при доле 15,5%. Отдача на риск слабая (Шарп 0,45), портфель прыгает на 18% в год и исторически падал на 20%.',

  // 11 holdings
  holdings: [
    { t:'NVDA',  name:'NVIDIA Corp.',          cls:'Акции США', w:15.5, risk:35.1, pnlPct:6.6,   pnlUsd:131,  signal:'TRIM', status:'HOTSPOT',
      fund:{ roe:'76.3%', margin:'60.4%', debt:'23.9%', growth:'+65.5%', atr:'2.02%', z:'6.57' },
      note:'Крупнейший вклад в риск портфеля — TRC 35,1% при доходности +6,6%. Концентрация выше порога 20%; рассмотрите частичную фиксацию.' },
    { t:'MSFT',  name:'Microsoft Corp.',       cls:'Акции США', w:16.5, risk:14.6, pnlPct:-9.8,  pnlUsd:-246, signal:'SELL', status:'NORMAL',
      fund:{ roe:'29.6%', margin:'45.6%', debt:'44.5%', growth:'+14.9%', atr:'1.93%', z:'2.52' },
      note:'В минусе (−9,8%); вклад в риск 14,6%. Дополнительный триггер: вес > 15%. Совокупный балл −2,1 → SELL.' },
    { t:'GOOGL', name:'Alphabet Inc.',         cls:'Акции США', w:12.7, risk:15.3, pnlPct:16.2,  pnlUsd:241,  signal:'BUY',  status:'NORMAL',
      fund:{ roe:'31.8%', margin:'32.0%', debt:'30.2%', growth:'+15.1%', atr:'2.09%', z:'3.75' },
      note:'Качественная позиция с сильным фундаменталом и техникой. Лучший совокупный балл портфеля +2,4.' },
    { t:'AAPL',  name:'Apple Inc.',            cls:'Акции США', w:11.0, risk:9.4,  pnlPct:45.8,  pnlUsd:471,  signal:'TRIM', status:'NORMAL',
      fund:{ roe:'151.9%', margin:'32.0%', debt:'79.5%', growth:'+6.4%', atr:'1.30%', z:'2.42' },
      note:'Лучшая доходность среди акций США (+45,8%). Оценка дорогая для сектора (V −2), техника сильна — частичная фиксация.' },
    { t:'ORCL',  name:'Oracle Corp.',          cls:'Акции США', w:9.0,  risk:18.1, pnlPct:1.3,   pnlUsd:16,   signal:'SELL', status:'NORMAL',
      fund:{ roe:'40.2%', margin:'30.6%', debt:'0.0%', growth:'+17.3%', atr:'4.36%', z:'н/д' },
      note:'Повышенная внутридневная волатильность (ATR 4,36%) — «дёргается» сильнее среднего. Худший балл портфеля −6 → SELL.' },
    { t:'TLT',   name:'iShares 20+Y Treasury', cls:'Облигации', w:15.7, risk:1.8,  pnlPct:3.1,   pnlUsd:64,   signal:'HOLD', status:'NORMAL',
      fund:{ roe:'н/д', margin:'н/д', debt:'н/д', growth:'н/д', atr:'0.44%', z:'н/д' },
      note:'Длинные госбонды — стабилизатор портфеля. Вес > 15%; следите за динамикой через Action Plan.' },
    { t:'GLD',   name:'SPDR Gold Shares',      cls:'Сырьё',     w:5.6,  risk:1.9,  pnlPct:-7.3,  pnlUsd:-61,  signal:'TRIM', status:'NORMAL',
      fund:{ roe:'н/д', margin:'н/д', debt:'н/д', growth:'н/д', atr:'1.34%', z:'н/д' },
      note:'В минусе (−7,3%); вклад в риск 1,9%. Слабая техника → частичное сокращение.' },
    { t:'FFSPC6.1028.AIX', short:'FFSPC', name:'Freedom SPC (AIX)', cls:'Акции KZ', w:4.8, risk:0.6, pnlPct:8.6, pnlUsd:52, signal:'HOLD', status:'NORMAL',
      fund:{ roe:'н/д', margin:'н/д', debt:'н/д', growth:'н/д', atr:'0.31%', z:'н/д' },
      note:'Локальная KZ-позиция, низкая волатильность. Нейтральный профиль → удерживаем.' },
    { t:'SPCX',  name:'SPAC / Space ETF',      cls:'Акции США', w:4.8,  risk:0.2,  pnlPct:-6.8,  pnlUsd:-48,  signal:'SELL', status:'NORMAL',
      fund:{ roe:'н/д', margin:'н/д', debt:'н/д', growth:'н/д', atr:'3.16%', z:'н/д' },
      note:'Повышенная волатильность (ATR 3,16%) при минусе −6,8%. Совокупный балл −3 → SELL.' },
    { t:'SLV',   name:'iShares Silver Trust',  cls:'Сырьё',     w:3.0,  risk:3.1,  pnlPct:-11.7, pnlUsd:-55,  signal:'SELL', status:'NORMAL',
      fund:{ roe:'н/д', margin:'н/д', debt:'н/д', growth:'н/д', atr:'2.57%', z:'н/д' },
      note:'Самый глубокий минус (−11,7%); слабая техника. Совокупный балл −2 → SELL.' },
    { t:'USD',   name:'Денежные средства',     cls:'Ден. средства', w:1.4, risk:0.0, pnlPct:0.0, pnlUsd:0,   signal:'HOLD', status:'NORMAL', cash:true,
      fund:{ roe:'н/д', margin:'н/д', debt:'н/д', growth:'н/д', atr:'—', z:'н/д' },
      note:'Кэш-буфер. Не несёт рыночного риска.' },
  ],

  // sector mix (by capital)
  sectors: [
    { name:'Technology',    pct:49, warn:true,  hue:'#1c1b1a' },
    { name:'Bonds',         pct:16, warn:false, hue:'#a8a293' },
    { name:'Semiconductors',pct:16, warn:true,  hue:'#caa01a' },
    { name:'Gold',          pct:6,  warn:false, hue:'#c9beac' },
    { name:'Other',         pct:6,  warn:false, hue:'#d8cfbb' },
    { name:'EM Kazakhstan', pct:5,  warn:false, hue:'#e3d9c4' },
    { name:'Silver',        pct:3,  warn:false, hue:'#efe8d4' },
  ],
  sectorWarn: [
    'Technology: 49% портфеля — мягкий лимит 40% превышен на 9 п.п.',
    'Tech-комплекс (Technology + Semiconductors): 65% — лимит 40% превышен на 25 п.п.',
  ],
  holdingsAI: 'Главные точки риска: NVDA даёт 35% всего риска при доле 15,5% (опасная концентрация), ORCL — 18% риска и худший балл −6. Перевешены технологии (вместе с полупроводниками 64,6%), недовешены защитные секторы. Портфель отстаёт от рынка на 4,3% за год.',

  // factor β-radar (10 axes; portfolio β vs market benchmark)
  factors: [
    { name:'Market',      port:1.10,  mkt:1.00 },
    { name:'Momentum',    port:0.05,  mkt:0.00 },
    { name:'Value',       port:-0.23, mkt:0.00 },
    { name:'Quality',     port:0.20,  mkt:0.00 },
    { name:'Size',        port:-0.09, mkt:0.00 },
    { name:'Volatility',  port:-0.43, mkt:0.00 },
    { name:'Commodities', port:0.08,  mkt:0.00 },
    { name:'Rates',       port:0.31,  mkt:0.00 },
    { name:'EM Equity',   port:0.02,  mkt:0.00 },
    { name:'EM Bond',     port:0.04,  mkt:0.00 },
  ],
  factorCoverage: 98.6,
  factorAI: 'Портфель сильно завязан на акции роста (NVDA двигается в 2,3× сильнее рынка, MSFT 1,56×), а не на недооценённые. В растущей экономике Barclays советует value над growth — портфель не совпадает с рекомендацией.',

  // 4-pillar scoring (F/V/T/C, each −2..+2, total ∈ [−6,+6])
  scores: [
    { t:'GOOGL', total:2.4,  action:'BUY',  F:0.9,  V:0.0,  T:2.0,  C:-0.5, reason:'сильный совокупный сигнал' },
    { t:'AAPL',  total:-1.6, action:'TRIM', F:-0.6, V:-2.0, T:2.0,  C:-1.0, reason:'дорогая оценка' },
    { t:'GLD',   total:-1.5, action:'TRIM', F:null, V:0.0,  T:-1.5, C:null, reason:'слабая техника' },
    { t:'TLT',   total:-0.5, action:'HOLD', F:null, V:0.0,  T:-0.5, C:null, reason:'нейтральный профиль' },
    { t:'NVDA',  total:-0.5, action:'TRIM', F:2.0,  V:-2.0, T:0.5,  C:-1.0, reason:'сильный бизнес, дорогая оценка' },
    { t:'SLV',   total:-2.0, action:'SELL', F:null, V:0.0,  T:-2.0, C:null, reason:'слабый сигнал' },
    { t:'MSFT',  total:-2.1, action:'SELL', F:1.4,  V:0.0,  T:-2.0, C:-1.5, reason:'слабая техника + кредит' },
    { t:'SPCX',  total:-3.0, action:'SELL', F:0.0,  V:0.0,  T:-1.0, C:-2.0, reason:'слабый сигнал' },
    { t:'ORCL',  total:-6.0, action:'SELL', F:-0.1, V:-2.0, T:-2.0, C:-2.0, reason:'дорогой, слабая техника, кредит' },
  ],
  scoresNote: 'Показаны 9 из 11 позиций — FFSPC и USD в HOLD-зоне без триггеров.',
  scoresAI: 'GOOGL лучший (+2,4: хороший бизнес, сильная техника). ORCL худший (−6: дорогой, слабая техника, кредитный риск). NVDA: сильный бизнес (фундаментал +2), но оценка −2 — дорого даже для своего сектора.',

  // stress scenarios
  stress: [
    { name:'Tech sell-off (как Q2 2022)', pct:-11.8, usd:-1621, dd:-20.8, rec:'23.7 мес' },
    { name:'Geopolitical risk-off',       pct:-7.7,  usd:-1052, dd:-16.7, rec:'18.5 мес' },
    { name:'Credit blow-out (+200 bps HY)', pct:-4.5, usd:-619, dd:-13.5, rec:'14.7 мес' },
    { name:'CPI shock (+1 пп surprise)',   pct:-3.3,  usd:-446,  dd:-12.2, rec:'13.3 мес' },
    { name:'Fed +50 bps surprise',         pct:-2.0,  usd:-277,  dd:-11.0, rec:'11.8 мес' },
    { name:'USD +5% rally',                pct:-0.5,  usd:-71,   dd:-9.5,  rec:'10.2 мес' },
    { name:'Fed cut surprise (−50 bps)',   pct:3.1,   usd:426,   dd:null,  rec:'—' },
  ],
  stressAI: 'В сценарии распродажи техов (как Q2 2022) портфель теряет 11,85%, просадка до 20,83%. Сильнее всех пострадает NVDA (−29% после ограничителя) из-за чувствительности к рынку 2,3 и доли в риске 35%, далее ORCL −24%, MSFT −15%.',

  // expected effect of action plan (before → after)
  effect: [
    { name:'Индекс риска',        before:'49',    after:'51',    delta:'+2,0 пункта',  tone:'neg' },
    { name:'CVaR 95%',            before:'−2,7%', after:'−2,4%', delta:'+0,3 пп',      tone:'pos' },
    { name:'Sharpe Ratio',        before:'0,45',  after:'0,50',  delta:'+0,05',        tone:'pos' },
    { name:'Max Drawdown',        before:'−20,2%',after:'−18,1%',delta:'+2,0 пп',      tone:'pos' },
    { name:'Волатильность (год.)',before:'18,0%', after:'16,2%', delta:'−1,8 пп',      tone:'pos' },
    { name:'Доля IT-сектора',     before:'64,6%', after:'56,6%', delta:'−8,0 пп',      tone:'pos' },
    { name:'Max TRC (1 позиция)', before:'35,1%', after:'52,8%', delta:'+17,7 пп',     tone:'neg' },
    { name:'Ожид. доходность',    before:'2,2%',  after:'2,6%',  delta:'+0,4 пп',      tone:'pos' },
  ],
  effectScope: ['ORCL', 'SPCX', 'AAPL', 'MSFT', 'NVDA'],
  effectScoped: true,
  effectVerdict: 'Компромисс: волатильность и просадка снижаются, но при урезании NVDA её доля в остаточном риске временно растёт (max TRC ухудшается).',
  effectAI: 'После ребалансировки потери в худший день из 20 снизятся с ≈2,7% к ≈2,2%, волатильность с 18% к ≈16%, отдача на риск вырастет. Главная причина — урезание NVDA и продажа дорогого ORCL уменьшат концентрацию и чувствительность к рынку.',

  // action plan — buy/sell/stop levels
  actionPlan: [
    { t:'ORCL', action:'SELL', price:175.27, target:'188,39 – 196,03', stop:'160,00', score:-6.0, hot:false },
    { t:'SPCX', action:'SELL', price:165.02, target:'176,46 – 181,68', stop:'154,58', score:-3.0, hot:false },
    { t:'MSFT', action:'SELL', price:376.52, target:'412,70 – 419,96', stop:'362,00', score:-2.1, hot:false },
    { t:'SLV',  action:'SELL', price:59.13,  target:'67,84 – 69,36',   stop:'56,09',  score:-2.0, hot:false },
    { t:'AAPL', action:'TRIM', price:300.26, target:'289,72 – 306,27', stop:'292,45', score:-1.6, hot:false },
    { t:'GLD',  action:'TRIM', price:385.69, target:'417,69 – 422,88', stop:'375,32', score:-1.5, hot:false },
    { t:'NVDA', action:'TRIM', price:212.82, target:'209,97 – 217,08', stop:'204,22', score:-0.5, hot:true },
    { t:'GOOGL',action:'HOLD', price:346.46, target:'—',               stop:'331,97', score:2.4,  defer:true },
    { t:'TLT',  action:'HOLD', price:86.07,  target:'—',               stop:'86,73',  score:-0.5, defer:true },
    { t:'FFSPC',action:'HOLD', price:108.61, target:'—',               stop:'107,77', score:0.0,  defer:true },
  ],
  actionAI: 'Сначала продать ORCL (балл −6), SPCX, SLV, MSFT; урезать NVDA (главная точка риска) — её доля в риске упадёт с 35% ближе к 20%. Затем докупить GOOGL и новые недооценённые имена. Концентрация снизится.',

  // AI ideas (candidates outside portfolio)
  ideas: [
    { n:'01', cat:'Рост доходности', prio:'Высокий приоритет', tone:'grow',
      title:'Повышение доходности — активный риск',
      lede:'Растущая экономика → циклические качественные имена вне портфеля, дешевле NVDA при той же теме ИИ.',
      pipeline:['Momentum + Quality скоринг','Соответствие Growth × Cycle квадранту','Устойчивость при rate shock +200 bps','Подтверждение инвестбанками'],
      cands:[
        { t:'ANET', name:'Arista Networks',     why:'Сетевое оборудование для ИИ-дата-центров: ROE ~30%, маржа ~40%, β ~1,1 — рост дешевле NVDA.', src:'SEC EDGAR' },
        { t:'FIX',  name:'Comfort Systems USA', why:'Монтаж под бум дата-центров: выручка +30% г/г, недооценён vs техи; циклический фаворит.', src:'Quant Engine' },
      ] },
    { n:'02', cat:'Ребалансировка', prio:'Средний приоритет', tone:'rebalance',
      title:'Качественная ребалансировка',
      lede:'Качественные недооценённые имена вне техов — снизить зависимость от одного сектора.',
      pipeline:['4-Pillar Scoring F+V+T+C','Macro Alignment vs текущий режим','Поведение при equity shock −20%','SEC EDGAR фундаментал'],
      cands:[
        { t:'MCO',  name:'Moody’s Corporation', why:'Рейтинговый монополист (Barclays): ROE >40%, маржа ~45%; узкие спреды HY поддерживают спрос.', src:'RAG' },
        { t:'SPGI', name:'S&P Global',          why:'Данные и индексы (Barclays): маржа >35%, стабильный рост, низкий долг — качество без перегрева.', src:'RAG' },
      ] },
    { n:'04', cat:'Режимная ставка', prio:'Средний приоритет', tone:'rotation',
      title:'Режимная ставка — экономика растёт',
      lede:'Ставка на широкий рынок вместо узких техов — в растущей фазе широкий рост выгоднее концентрации.',
      pipeline:['Regime-specific факторный сигнал','Growth-Cycle квадрант + confidence','Сценарный анализ смены режима','Банковские прогнозы режима'],
      cands:[
        { t:'RSP',  name:'Invesco S&P 500 Equal Weight', why:'Равновзвешенный рынок снижает зависимость от мегатехов; в растущей экономике широкий рост выгоднее.', src:'Regime' },
      ] },
    { n:'03', cat:'Защита капитала', prio:'Низкий приоритет', tone:'hedge',
      title:'Защита капитала',
      lede:'Снижение хвостового риска при развороте техов — низковолатильный буфер.',
      pipeline:['Низкая Beta, дивидендный аристократ','Защитные сектора Healthcare/Gold','Positive P&L при recession сценарии','CDS + банковские отчёты по риску'],
      cands:[
        { t:'USMV', name:'iShares Min Vol USA', why:'Низковолатильные акции: β ~0,7 — смягчает падения, диверсифицирует от концентрации в техах.', src:'Quant Engine' },
      ] },
  ],

  // market regime
  regime: {
    name: 'Expansion',
    nameRu: 'Экспансия',
    confidence: 74,
    confirms: 3,
    growth: 0.14,
    cycle: 0.05,
    // position on quadrant in [-0.2..0.2]
    dot: { growth:0.14, cycle:0.05 },
    ragBacked: true,
    ragSignals: [
      { ok:true,  bank:'Goldman Sachs', text:'Базовый сценарий — продолжение экспансии: рост ВВП США выше тренда, мягкая посадка подтверждается опережающими индикаторами.' },
      { ok:true,  bank:'JPMorgan',      text:'Кредитные спреды остаются узкими, цикл риск-он; предпочтение циклическим и качественным акциям.' },
      { ok:false, bank:'Barclays',      text:'Предупреждение: оценки акций роста растянуты — тактически предпочитаем value над growth.' },
    ],
    drivers: [
      { name:'10Y−2Y Treasury spread', val:'+0,27 пп', trend:'▼ −0,19 за 1м', state:'актуально', tone:'pos' },
      { name:'US HY OAS (ICE BofA)',   val:'266 bp',   trend:'▼ −5,13 за 1м', state:'актуально', tone:'pos' },
      { name:'CBOE VIX',               val:'16,8',     trend:'▲ +1,81 за 1м', state:'актуально', tone:'pos' },
      { name:'10Y Breakeven Inflation',val:'2,23%',    trend:'▼ −0,17 за 1м', state:'актуально', tone:'pos' },
      { name:'US Unemployment Rate',   val:'4,30%',    trend:'▼ −0,09 за 3м', state:'устарело',  tone:'warn' },
      { name:'Real GDP growth (SAAR)', val:'1,60%',    trend:'▼ −3,15 за 3кв', state:'устарело',  tone:'warn' },
    ],
    confirm: 'Вывод движка о росте экономики подтверждается: кривая доходности положительная, кредитные спреды узкие, страх рынка низкий. Независимые сигналы согласны на ≥80%.',
    confirmBullets: [
      { ok:true,  t:'Кривая доходности 10Y−2Y +0,27 пп — положительная, без признаков рецессии' },
      { ok:true,  t:'Кредитный спред HY OAS 2,66% — узкий (<3,5%), режим риск-он' },
      { ok:true,  t:'VIX 16,78 — низкий страх рынка (<20), спокойствие' },
      { ok:true,  t:'Инфляционные ожидания 2,23% — заякорены у цели, без стагфляции' },
      { ok:false, t:'Факторные беты портфеля высокие (NVDA 2,3) — больше акций роста, чем советует Barclays' },
      { ok:true,  t:'Банковский консенсус [GS][JPM] подтверждает рост — циклические в фаворе' },
    ],
    regimeAI: 'Рынок: экономика растёт (уверенность 74%) — не перегрев и не спад. Goldman и JPMorgan в такой фазе советуют циклические и качественные акции, Barclays — недооценённые акции над акциями роста.',
  },

  // CoVe — chain-of-verification provenance.  Consolidated 2026-07-09
  // (24 → 16 rows): Quant-risk+Euler, the two SEC lines, the six FRED series,
  // and the two LLM-checkers are each merged into a single logical row.
  cove: [
    { st:'ok',   title:'Риск-метрики: Vol · CVaR · TE · IR · Max DD · TRC/MCTR (Euler)', meta:'Quant Engine MAC3 · Wilder RMA · EWMA hl=63 ⊕ Ledoit-Wolf 70/30 · Politis-Romano bootstrap CI · Euler-декомпозиция (MCTR = Σw/σ_p)' },
    { st:'warn', title:'Факторная независимость',        meta:'Σ=B·F·Bᵀ+D · κ=61,59 · max|corr|=0,8899 · близки к коллинеарности (FACTOR_ORTHOGONALIZE вкл.)' },
    { st:'ok',   title:'Цены и история активов',         meta:'Tradernet (Freedom Broker) · Daily CLOSE · окно 1825д · ATR via OHLC · as_of 2026-06-21' },
    { st:'ok',   title:'Валютный слой',                  meta:'Base = USD · RFR 4,50% · конверсия не требуется (все активы в USD)' },
    { st:'ok',   title:'Фундамент (SEC EDGAR): Z-scores · Altman-Z / Piotroski-F / Coverage', meta:'SEC EDGAR CompanyFacts · 10-K FY · sector-normalised MAD (Group B) ⊕ баланс/P&L' },
    { st:'warn', title:'CDS spreads (credit signal)',    meta:'FRED HY proxy + WGB sovereign · sanity 1–3000 bps · 9/10 loaded · 1 gated out' },
    { st:'ok',   title:'Action levels (Buy / Sell / Stop)', meta:'Quant Engine · ATR Wilder + SMA50/200 + RSI(14) + MACD(12,26,9) · 11 позиций' },
    { st:'ok',   title:'Black-Litterman target weights', meta:'Quant Engine · reverse-opt prior + score-views · τ=0,05 · 10 позиций reweighted' },
    { st:'ok',   title:'Регим-классификатор',            meta:'Quant Engine · Growth × Cycle · окно 60д · Expansion · confidence 74%' },
    { st:'warn', title:'Стресс-сценарии',                meta:'Quant Engine · parametric factor shocks · per-asset β · 7 сценариев · 2 proxy' },
    { st:'warn', title:'Макро-драйверы (FRED)',          meta:'FRED St. Louis Fed · yield curve · HY · VIX · breakeven · unemployment · GDP · 6 серий · требуют внимания: безработица, GDP' },
    { st:'fail', title:'Smart-Money (инсайдеры Form 4)', meta:'SEC EDGAR · Form 4 · SMART_MONEY_INSIDERS=0 (выкл. по умолчанию)' },
    { st:'ok',   title:'Bank RAG (выдержки)',            meta:'ChromaDB · GS / MS / JPM PDF reports · cosine retrieval (semantic 0,6 ⊕ recency 0,4)' },
    { st:'ok',   title:'ИИ-цитирование банк-аналитики',  meta:'CoVe · [RAG:файл] (проверено) vs. банк-консенсус из знаний модели' },
    { st:'ok',   title:'AI verdict · bullets',           meta:'Anthropic · Claude Opus 4.8 · advisory only · не является ИИР' },
    { st:'ok',   title:'LLM-чекеры: галлюцинации + вычисления', meta:'CoVe · held-filter + data-driven + фильтр противоречий ⊕ stress-cap + no-self-aggregation — настроены' },
  ],

  // data-quality chips (footer)
  quality: [
    'Рыночные данные · Tradernet · 501 день · daily CLOSE',
    'Серия доходностей · 100% покрытие · 500 дней',
    'Факторная модель · Ridge β · 10/10 факторов',
    'Euler-декомпозиция · Ledoit-Wolf 70/30',
    'CVaR Bootstrap · Politis-Romano · 2000 блоков · 95% CI',
    'RAG · ChromaDB · cosine ≥0,72 · ~69 отрывков',
    'AI-модель · Claude Opus 4.8',
    'Валюта · USD · RFR 4,50%',
  ],
};

window.DEEP = DEEP;
