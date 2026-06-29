/* Overview section: hero + factor pills + asymmetric main grid */

// ── Factor pill (mini progress with chip label) — echoes the reference "Interviews/Hired" rhythm
const FactorPill = ({ label, value, display, cap=100, accent='gold', warn, suffix }) => {
  const pct = Math.min(100, (value / cap) * 100);
  const bgMap = {
    gold: 'bg-ink-900',  // dark pill, gold fill
    dark: 'bg-ink-900',
    sage: 'bg-ink-900',
    mute: 'bg-cream-200',
  };
  const fillMap = {
    gold: 'bg-gold-400',
    dark: 'bg-ink-900',
    sage: 'bg-sage-500',
    mute: 'bg-ink-700',
  };
  const textColor = accent === 'mute' ? 'text-ink-900' : 'text-white';
  const valTxt = display || `${value}${suffix||'%'}`;
  return (
    <div className="flex flex-col gap-2 min-w-[120px]">
      <div className="flex items-center gap-2">
        <span className="text-[12px] text-ink-500 font-medium tracking-tight">{label}</span>
        {warn && <span className="w-1.5 h-1.5 rounded-full bg-rust-500"/>}
      </div>
      <div className={`relative h-7 rounded-full overflow-hidden ${accent==='mute'?'bg-cream-200/70':'bg-ink-900/85'}`}>
        {/* fill */}
        <div className={`absolute inset-y-0 left-0 ${fillMap[accent]} rounded-full`}
             style={{ width: `${pct}%` }}/>
        {/* label */}
        <span className={`absolute inset-0 flex items-center pl-3 text-[11px] font-semibold num
                          ${accent==='mute' ? 'text-ink-900' : 'text-white mix-blend-difference'}`}>
          {valTxt}
        </span>
      </div>
    </div>
  );
};

// ── Hero stat (big number with thin label, e.g. "9 Позиции").  `small` renders a
// long textual value (e.g. the mandate profile «Умеренно-агрессивный») compactly
// so it doesn't blow up to the 5xl numeric size and break the hero row.
const HeroStat = ({ value, label, IconC, small }) => (
  <div className="flex flex-col items-center gap-1 px-3 sm:px-4">
    <IconC size={18} className="text-ink-500 mb-1" stroke={1.4}/>
    <div className={`font-light tracking-tight num text-ink-900 ${small
        ? 'text-base font-semibold leading-tight text-center max-w-[130px] pt-1.5'
        : 'text-5xl leading-none'}`}>{value}</div>
    <div className="text-[12px] text-ink-500 font-medium tracking-tight">{label}</div>
  </div>
);

// ── Top hotspot card (tall, replaces "profile photo" card from reference)
const TopHotspotCard = ({ h }) => (
  <div className="relative rounded-4xl overflow-hidden shadow-card-lg lift h-full min-h-[420px]"
       style={{ background: 'linear-gradient(155deg, #2a2825 0%, #1c1b1a 45%, #3a2f1e 100%)' }}>
    {/* subtle ticker watermark */}
    <div className="absolute -right-6 -top-6 text-[260px] font-black leading-none text-white/[0.04] select-none num">
      {h.ticker.slice(0,2)}
    </div>
    {/* gold glow */}
    <div className="absolute -bottom-24 -right-12 w-80 h-80 rounded-full"
         style={{ background:'radial-gradient(circle, rgba(245,208,78,0.55), transparent 60%)' }}/>
    {/* hotspot tag */}
    <div className="absolute top-5 left-5 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-gold-400 text-ink-900 text-[10px] font-bold tracking-wider uppercase">
      <Icons.Warning size={11} stroke={2}/> Hotspot
    </div>
    <div className="absolute top-5 right-5 text-white/40 text-[10px] font-mono tracking-widest">
      RISK · {h.riskShare}%
    </div>

    {/* big ticker glyph in center */}
    <div className="absolute inset-0 flex items-center justify-center pt-2">
      <div className="text-white num font-light tracking-tight" style={{ fontSize: 96, letterSpacing:'-0.04em' }}>
        {h.ticker}
      </div>
    </div>

    {/* bottom info block */}
    <div className="absolute left-5 right-5 bottom-5">
      <div className="rounded-3xl p-4 backdrop-blur-md" style={{ background:'rgba(255,255,255,0.08)', border:'1px solid rgba(255,255,255,0.08)' }}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-white text-[15px] font-semibold tracking-tight leading-tight">{h.name}</div>
            <div className="text-white/50 text-[11px] mt-0.5">{h.sector}</div>
          </div>
          <div className="px-3 py-1 rounded-full bg-gold-400/95 text-ink-900 text-[10px] font-bold tracking-wider">
            {h.signal}
          </div>
        </div>
        <div className="grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-white/10">
          <div>
            <div className="text-white/40 text-[10px] tracking-wider uppercase">Вес</div>
            <div className="text-white text-lg font-medium num">{h.weight}%</div>
          </div>
          <div>
            <div className="text-white/40 text-[10px] tracking-wider uppercase">P/L</div>
            <div className={`text-lg font-medium num ${h.pnlPct>=0?'text-sage-500':'text-rust-500'}`}>{h.pnlPct>=0?'+':'−'}{Math.abs(h.pnlPct)}%</div>
          </div>
          <div>
            <div className="text-white/40 text-[10px] tracking-wider uppercase">USD</div>
            <div className="text-white text-lg font-medium num">{h.pnlUsd>=0?'+':'−'}${Math.abs(h.pnlUsd)>=10000 ? (Math.abs(h.pnlUsd)/1000).toFixed(1)+'K' : Math.round(Math.abs(h.pnlUsd)).toLocaleString('ru-RU')}</div>
          </div>
        </div>
        <div className="mt-3 text-white/60 text-[11px] leading-snug">{h.note}</div>
      </div>
    </div>
  </div>
);

// ── Compact risk-index gauge for the hero (moved up from the main grid so the
// headline verdict and its risk score read together — user request «перемести
// Индекс Риска в это пространство»).  Horizontal layout fills the hero's right
// column without the tall empty card the gauge used to leave at the grid bottom.
const HeroRiskGauge = ({ value, delta, profile }) => (
  <div className="glass-strong rounded-4xl p-5 shadow-card lift flex items-center gap-5">
    <div className="flex-shrink-0">
      <RiskGauge value={value} size={148}/>
    </div>
    <div className="flex flex-col gap-2.5 min-w-0">
      <div>
        <div className="text-ink-500 text-[10px] font-medium tracking-widest uppercase">Индекс риска</div>
        <h3 className="text-lg font-semibold tracking-tight text-ink-900 leading-tight">Сводный 0–100</h3>
      </div>
      {(delta !== undefined && delta !== null) && (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-cream-50 border border-ink-900/5 text-[11px] text-ink-700 w-fit">
          <Icons.Pulse size={13} stroke={1.8}/>
          <span>За месяц {delta>0?'+':''}{delta} пт</span>
        </div>
      )}
      <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-ink-900 text-white text-[11px] font-medium hover:bg-ink-800 transition w-fit">
        <Icons.Refresh size={12} stroke={2}/> Пересчитать
      </button>
    </div>
  </div>
);

// ── Risk decomposition card
const RiskDecompCard = ({ data }) => (
  <div className="glass-strong rounded-4xl p-6 shadow-card lift h-full min-h-[420px] flex flex-col">
    <div className="flex items-start justify-between gap-3">
      <div>
        <div className="text-ink-500 text-[10px] font-medium tracking-widest uppercase">Декомпозиция</div>
        <h3 className="text-2xl font-semibold tracking-tight text-ink-900 leading-tight">Риск портфеля</h3>
        <div className="text-[11px] text-ink-400 mt-1">Вклад позиций в волатильность, %</div>
      </div>
      <div className="text-right flex-shrink-0">
        <div className="text-[10px] text-ink-400 tracking-wider uppercase">Итог</div>
        <div className="text-xl font-semibold num text-ink-900 leading-none mt-0.5">{data.total != null ? `${data.total}%` : '—'}</div>
      </div>
    </div>
    <div className="flex-1 flex items-center justify-center -mx-1 my-2">
      <Waterfall data={data}/>
    </div>
    <div className="flex items-center justify-center gap-5 text-[11px] text-ink-500 pt-3 border-t border-ink-900/5">
      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-ink-900"/> отдельно</span>
      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-rust-500"/> диверсификация</span>
      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-gold-400"/> итог</span>
    </div>
  </div>
);

// ── Sector mix card
const SectorMixCard = ({ sectors }) => {
  const overweight = (sectors || []).find(s => s.warn);
  return (
  <div className="glass-strong rounded-4xl p-6 shadow-card lift h-full flex flex-col min-h-[200px]">
    <div className="flex items-start justify-between gap-3 mb-1">
      <div>
        <div className="text-ink-500 text-[10px] font-medium tracking-widest uppercase">Структура</div>
        <h3 className="text-xl font-semibold tracking-tight text-ink-900 leading-tight">По секторам</h3>
      </div>
      {overweight && (
        <div className="px-2.5 py-1 rounded-full bg-rust-500/12 text-rust-600 text-[10px] font-semibold tracking-wider uppercase flex items-center gap-1 flex-shrink-0">
          <Icons.Warning size={11} stroke={2}/> Перевес
        </div>
      )}
    </div>
    <div className="mt-4">
      <SectorBar sectors={sectors}/>
    </div>
    <div className="grid grid-cols-2 gap-x-5 gap-y-2.5 mt-5">
      {sectors.map(s => (
        <div key={s.name} className="flex items-center gap-2.5">
          <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background:s.hue }}/>
          <div className="flex-1 min-w-0 text-[11px] text-ink-700 font-medium truncate">{s.name}</div>
          <div className="text-[12px] font-semibold num text-ink-900 tabular-nums">{s.pct}%</div>
        </div>
      ))}
    </div>
  </div>
  );
};

// ── Quick AI insight (dark pill row, echoes the reference "Onboarding Task" mini cards style)
const AIInsightCard = ({ verdict }) => (
  <div className="rounded-4xl p-6 shadow-dark lift h-full flex flex-col min-h-[200px]"
       style={{ background:'linear-gradient(160deg, #1c1b1a 0%, #2a2825 100%)' }}>
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2 text-gold-400 text-[10px] font-mono tracking-widest uppercase">
        <Icons.Sparkles size={13} stroke={1.8}/> AI · {window.PORTFOLIO.meta.aiModel}
      </div>
    </div>
    <div className="mt-4 text-white text-[15px] leading-relaxed font-light">
      Индекс <span className="text-gold-400 font-semibold num">{verdict.riskIndex}</span>
      {verdict.riskTier && verdict.riskTier!=='–' ? ` · ${verdict.riskTier}` : ''}. {verdict.sub}
    </div>
    <div className="mt-auto pt-4 flex items-center gap-2 flex-wrap">
      <span className="px-2.5 py-1 rounded-full bg-white/8 text-white/70 text-[10px] font-mono tracking-wider">RAG · банки</span>
      <span className="px-2.5 py-1 rounded-full bg-white/8 text-white/70 text-[10px] font-mono tracking-wider">SEC EDGAR</span>
      <span className="px-2.5 py-1 rounded-full bg-white/8 text-white/70 text-[10px] font-mono tracking-wider">MAC3</span>
    </div>
  </div>
);

// ── Hero — verdict + factor pills + right stats
const Hero = () => {
  const p = window.PORTFOLIO;
  const v = p.verdict;
  return (
    <section id="overview" className="rise" data-screen-label="01 Overview">
      <div className="flex items-start justify-between gap-x-10 gap-y-6 flex-wrap mb-8">
        <div className="flex-1 min-w-[280px] max-w-[820px]">
          <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-3">
            <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/>
            Portfolio Risk Report · Tier {p.meta.tier}
          </div>
          <h1 className="text-[clamp(32px,4.6vw,56px)] leading-[1.05] tracking-[-0.03em] font-light text-ink-900">
            {v.headline}<span className="text-ink-400">.</span>
          </h1>
          <p className="text-[17px] text-ink-500 mt-3 max-w-[640px] font-light">{v.sub}</p>
        </div>

        {/* Right column — hero stat triplet + risk-index gauge moved up here */}
        <div className="flex flex-col gap-4 w-full lg:w-auto lg:min-w-[360px] lg:max-w-[440px]">
          <div className="flex items-end justify-start lg:justify-end gap-1 divide-x divide-ink-900/10">
            {p.heroStats.map((s,i) => {
              const IconC = { briefcase: Icons.Briefcase, trendUp: Icons.TrendUp, wallet: Icons.Wallet }[s.icon] || Icons.Briefcase;
              return <HeroStat key={i} value={s.value} label={s.label} IconC={IconC} small={s.small}/>;
            })}
          </div>
          <HeroRiskGauge value={v.riskIndex} delta={v.riskTrendDelta} profile={p.meta.profile}/>
        </div>
      </div>

      {/* Factor pills strip (echoes reference small KPI bars) — a tidy 2-col grid
          on phones, an inline flowing row from sm up. */}
      <div className="grid grid-cols-2 sm:flex sm:flex-wrap items-end gap-x-8 gap-y-5 mb-10">
        {p.factorPills.map((f,i) => <FactorPill key={i} {...f}/>)}
      </div>

      {/* Asymmetric main grid — gauge moved to the hero, so three aligned cards:
          hotspot · risk decomposition (wider) · sector mix + AI insight stack. */}
      <div className="grid grid-cols-12 gap-5 items-stretch">
        <div className="col-span-12 sm:col-span-6 lg:col-span-3"><TopHotspotCard h={p.topHotspot}/></div>
        <div className="col-span-12 sm:col-span-6 lg:col-span-5"><RiskDecompCard data={p.riskDecomp}/></div>
        <div className="col-span-12 lg:col-span-4 grid grid-rows-[1fr_auto] gap-5">
          <SectorMixCard sectors={p.sectors}/>
          <AIInsightCard verdict={v}/>
        </div>
      </div>
    </section>
  );
};

Object.assign(window, { Hero });
