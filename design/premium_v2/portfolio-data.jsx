/* Portfolio data — shape the BASE Risk Report into structured props */

const PORTFOLIO = {
  meta: {
    tier: 'BASE',
    id: '2026-05-14-U1042',
    engine: 'MAC3',
    session: 'A6C7B1F3',
    generated: '2026-05-14 09:24 UTC+5',
  },
  verdict: {
    headline: 'Портфель в зоне умеренного риска',
    sub: 'Держать можно — но почти весь риск собран в двух позициях.',
    riskIndex: 62,            // 0-100
    riskTrendDelta: +1,       // mock — index up 1 pt vs last month
    nav: 24000,
  },
  kpis: {
    cvar:   { label:'CVaR 95%',     value:-5.2, unit:'%', delta:-0.3, deltaText:'мягче', frame:'1 день · ≈ −$1 240' },
    sharpe: { label:'Sharpe Ratio', value:1.18,            delta:+0.06, deltaText:'к мес.', frame:'Sortino 1.60 · бенч 0.74' },
    dd:     { label:'Max Drawdown', value:-12.8,unit:'%',  delta:0,     deltaText:'без изм.', frame:'пик→дно · ≈ −$3 070' },
    vol:    { label:'Волатильность',value:14.8, unit:'%',  delta:+0.4,  deltaText:'выше рынка', frame:'рынок 11.2%' },
  },
  // KPI strip pills (echo reference "Interviews/Hired/Project time" rhythm)
  factorPills: [
    { label:'Tech share',     value:62, accent:'gold',  warn:true,  cap:80 },
    { label:'Hotspots',       value:44, accent:'dark',  warn:false, cap:60, suffix:'% риска', display:'44.2' }, // PLTR+CRWD
    { label:'Diversification',value:23, accent:'mute',  warn:false, cap:100, suffix:'% эффект', display:'23' },
    { label:'Beta',           value:118,accent:'mute',  warn:false, cap:200, display:'1.18', raw:true },
    { label:'Quality',        value:71, accent:'sage',  warn:false, cap:100, suffix:'/100',  display:'71' },
  ],
  // hero stats (right of welcome line)
  heroStats: [
    { label:'Позиции',  value:9,    icon:'briefcase' },
    { label:'YTD',      value:'+7.1%', icon:'trendUp' },
    { label:'NAV',      value:'$24K', icon:'wallet' },
  ],
  // top hotspot featured card
  topHotspot: {
    ticker:'PLTR',
    name:'Palantir Technologies',
    sector:'Технологии',
    weight:18.0,
    riskShare:24.4,
    pnlPct:32.1,
    pnlUsd:2140,
    signal:'BUY',
    note:'24% всего риска портфеля собрано здесь. Beta 1.81 · ATR 3.1%.',
  },
  // sector breakdown
  sectors: [
    { name:'Технологии',      pct:62, warn:true,  hue:'#1c1b1a' },
    { name:'Потреб. товары',  pct:16, warn:false, hue:'#a8a293' },
    { name:'Здравоохранение', pct:14, warn:false, hue:'#c9beac' },
    { name:'Энергетика',      pct: 6, warn:false, hue:'#e3d9c4' },
    { name:'Облигации',       pct: 2, warn:false, hue:'#efe8d4' },
  ],
  // risk decomposition (waterfall) — pre-diversified standalone risks + diversification + total
  riskDecomp: {
    standalone: [
      { t:'PLTR',  v:6.2 },
      { t:'CRWD',  v:5.0 },
      { t:'AAPL',  v:3.4 },
      { t:'JNJ',   v:2.0 },
      { t:'KO',    v:1.8 },
    ],
    diversification: -4.2,
    total: 14.2,
    sumStandalone: 18.4,
  },
  // holdings (all 9)
  holdings: [
    { t:'PLTR',  name:'Palantir Technologies',  cls:'Технологии',     w:18.0, beta:1.81, risk:24.4, pnlPct:32.1, pnlUsd:+2140, status:'HOTSPOT', signal:'BUY',         fund:{ roe:'28.4%', margin:'22.1%', debt:'0.18', growth:'+30.4%', atr:'3.1%' }, note:'Сильный фундаментал и рост выручки, но дневные колебания высокие — бумага усиливает риск портфеля.' },
    { t:'CRWD',  name:'CrowdStrike Holdings',   cls:'Технологии',     w:14.0, beta:1.55, risk:20.8, pnlPct:18.5, pnlUsd:+890,  status:'HOTSPOT', signal:'HOLD',        fund:{ roe:'19.7%', margin:'15.2%', debt:'0.22', growth:'+33.1%', atr:'2.8%' }, note:'Движется почти синхронно с PLTR — вместе вторая по величине концентрация риска.' },
    { t:'AAPL',  name:'Apple Inc.',             cls:'Технологии',     w:16.0, beta:1.21, risk:17.5, pnlPct:12.3, pnlUsd:+640,  status:'NORMAL',  signal:'BUY',         fund:{ roe:'147%',  margin:'30.1%', debt:'0.31', growth:'+2.0%',  atr:'1.6%' }, note:'Очень прибыльная и стабильная — высокая маржа, умеренные колебания.' },
    { t:'JNJ',   name:'Johnson & Johnson',      cls:'Здравоохранение',w:14.0, beta:0.74, risk:11.0, pnlPct: 4.2, pnlUsd:+310,  status:'NORMAL',  signal:'STRONG BUY',  fund:{ roe:'24.8%', margin:'25.3%', debt:'0.34', growth:'+3.4%',  atr:'0.9%' }, note:'Тихая, надёжная позиция: низкая бета, крепкий фундаментал — снижает общий риск.' },
    { t:'KO',    name:'The Coca-Cola Company',  cls:'Потреб. товары', w:16.0, beta:0.61, risk: 9.8, pnlPct:-1.8, pnlUsd:-190,  status:'NORMAL',  signal:'HOLD',        fund:{ roe:'41.2%', margin:'28.7%', debt:'0.46', growth:'+1.1%',  atr:'0.7%' }, note:'Самый «тихий» актив. Сейчас в небольшом минусе, но именно он гасит общий риск.' },
    { t:'MSFT',  name:'Microsoft Corp.',        cls:'Технологии',     w:10.0, beta:1.08, risk: 8.1, pnlPct: 9.4, pnlUsd:+420,  status:'NORMAL',  signal:'BUY',         fund:{ roe:'38.5%', margin:'44.6%', debt:'0.25', growth:'+15.7%', atr:'1.4%' }, note:'Высокая маржа и устойчивый рост при умеренной бете — качественная техпозиция.' },
    { t:'XOM',   name:'Exxon Mobil Corp.',      cls:'Энергетика',     w: 6.0, beta:0.92, risk: 4.0, pnlPct:-3.1, pnlUsd:-160,  status:'NORMAL',  signal:'TRIM',        fund:{ roe:'14.1%', margin:'12.8%', debt:'0.20', growth:'−6.2%',  atr:'1.9%' }, note:'Выручка снижается, позиция в небольшом минусе — движок предлагает сократить.' },
    { t:'NVDA',  name:'NVIDIA Corp.',           cls:'Технологии',     w: 4.0, beta:1.74, risk: 2.9, pnlPct:44.0, pnlUsd:+1210, status:'NORMAL',  signal:'HOLD',        fund:{ roe:'91.5%', margin:'54.1%', debt:'0.14', growth:'+122%',  atr:'3.6%' }, note:'Лучший рост в портфеле, доля небольшая — общий риск пока не раздувает.' },
    { t:'BND',   name:'Vanguard Total Bond',    cls:'Облигации',      w: 2.0, beta:0.12, risk: 1.5, pnlPct: 0.8, pnlUsd:+20,   status:'NORMAL',  signal:'HOLD',        fund:{ roe:'—',     margin:'—',     debt:'—',    growth:'—',      atr:'0.3%' }, note:'Облигационный фонд — почти нулевая бета, стабилизирует портфель.' },
  ],
  // performance vs S&P — 12 months cumulative
  performance: {
    months: ['Июн','Июл','Авг','Сен','Окт','Ноя','Дек','Янв','Фев','Мар','Апр','Май'],
    port:    [0, 1.4, 2.6, 3.9, 5.2, 4.7, 6.8, 8.5, 10.1, 11.4, 12.6, 14.2],
    spx:     [0, 0.8, 1.8, 2.6, 3.5, 3.1, 4.6, 5.7,  6.7,  7.5,  8.3,  9.1],
    vol:     { port:14.8, spx:11.2 },
    periods: [
      { label:'1 мес',  p:+2.0, s:+1.4, d:+0.6 },
      { label:'3 мес',  p:+5.5, s:+3.6, d:+1.9 },
      { label:'YTD',    p:+7.1, s:+4.4, d:+2.7 },
      { label:'6 мес',  p:+9.4, s:+6.0, d:+3.4 },
      { label:'12 мес', p:+14.2, s:+9.1, d:+5.1 },
    ],
  },
  // AI ideas
  ideas: [
    {
      n:'01', title:'Сократить долю PLTR с 18% до ~12%', cat:'Снижение риска', prio:'Высокий приоритет',
      lede:'Одна бумага даёт 24% риска. Позиция в плюсе +32% — есть что фиксировать без убытка.',
      pipeline:['PLTR даёт 24% риска, HOTSPOT','ждём роста волатильности в IT','Morgan Stanley: фиксировать прибыль'],
      tickers:[
        { t:'TXN',  why:'Стабильный производитель чипов, волатильность вдвое ниже PLTR.' },
        { t:'CSCO', why:'Зрелый тех с дивидендом и низкой бетой.' },
        { t:'ACN',  why:'IT-консалтинг с устойчивой выручкой, мягче в просадке.' },
      ],
      effect:['Вклад PLTR в риск: 24.4% → ~16%','Доля IT-сектора: 62% → ~56%'],
      sources:['Quant Engine','SEC EDGAR','MS_TechOutlook_2026'],
    },
    {
      n:'02', title:'Усилить защитные сектора — здравоохранение и товары', cat:'Диверсификация', prio:'Высокий приоритет',
      lede:'6 из 9 бумаг — технологии. JPMorgan называет healthcare и staples «островками стабильности».',
      pipeline:['6 из 9 бумаг в IT, бета 1.18','поздний цикл — спрос на защиту','JPMorgan: healthcare и staples устойчивы'],
      tickers:[
        { t:'PG',  why:'Потребтовары, очень низкая бета, стабильный денежный поток.' },
        { t:'MRK', why:'Крупная фарма, защитный сектор, недорогие мультипликаторы.' },
        { t:'PEP', why:'Staples с дивидендом, слабая связь с IT.' },
      ],
      effect:['Доля IT-сектора: 62% → ~50%','Бета портфеля: 1.18 → ~1.05'],
      sources:['Quant Engine','JPM_Strategy_Q2_2026'],
    },
    {
      n:'03', title:'Сократить или закрыть позицию XOM', cat:'Ребалансировка', prio:'Средний приоритет',
      lede:'Выручка Exxon −6% г/г, сигнал Trim. Позиция небольшая и в минусе — закрытие почти не затронет доходность.',
      pipeline:['выручка XOM −6% г/г, сигнал Trim','слабый цикл в энергетике','банки осторожны по ценам на нефть'],
      tickers:[
        { t:'NEE', why:'Коммунальщик и ВИЭ, стабильный спрос вместо цикличной нефти.' },
        { t:'DUK', why:'Регулируемая коммуналка, предсказуемый доход.' },
        { t:'SO',  why:'Высокий дивиденд, низкая бета.' },
      ],
      effect:['Высвобождает ~6% капитала','под защитный блок (идея 02)'],
      sources:['SEC EDGAR','Quant Engine'],
    },
    {
      n:'04', title:'Увеличить риск качественно — 3 бумаги вне портфеля', cat:'Увеличение риска', prio:'Средний приоритет',
      lede:'Защитный блок в норме — есть запас, чтобы добавить доходности через качество и новые сектора.',
      pipeline:['фильтр качества: ROE↑, долг↓','поздний цикл → качество и не-IT','сверка с Goldman Sachs и JPMorgan'],
      tickers:[
        { t:'GOOGL', why:'Качество и ИИ-экспозиция дешевле по мультипликаторам, чем PLTR.' },
        { t:'LLY',   why:'Структурный рост в здравоохранении, слабая связь с IT-ядром.' },
        { t:'V',     why:'Компаундер высокого качества, добавляет сектор финансов.' },
      ],
      effect:['Потенциал доходности: ↑','+финансы, +здравоохранение'],
      sources:['Factor Engine','Regime Model','GS_Q2_2026','JPM_Strategy_Q2_2026'],
    },
  ],
};

window.PORTFOLIO = PORTFOLIO;
