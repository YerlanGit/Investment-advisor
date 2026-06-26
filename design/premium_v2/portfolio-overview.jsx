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

// ── Hero stat (big number with thin label, e.g. "9 Позиции")
const HeroStat = ({ value, label, IconC }) => (
  <div className="flex flex-col items-center gap-1 px-4">
    <IconC size={18} className="text-ink-500 mb-1" stroke={1.4}/>
    <div className="text-5xl font-light tracking-tight num leading-none text-ink-900">{value}</div>
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
            <div className="text-gold-400 text-lg font-medium num">+{h.pnlPct}%</div>
          </div>
          <div>
            <div className="text-white/40 text-[10px] tracking-wider uppercase">USD</div>
            <div className="text-white text-lg font-medium num">+${(h.pnlUsd/1000).toFixed(2)}K</div>
          </div>
        </div>
        <div className="mt-3 text-white/60 text-[11px] leading-snug">{h.note}</div>
      </div>
    </div>
  </div>
);

// ── Risk gauge card
const RiskGaugeCard = ({ value, delta }) => (
  <div className="glass-strong rounded-4xl p-6 shadow-card lift h-full min-h-[420px] flex flex-col">
    <div className="flex items-center justify-between">
      <div>
        <div className="text-ink-500 text-[12px] font-medium">Индекс риска</div>
        <h3 className="text-2xl font-semibold tracking-tight text-ink-900 leading-tight">Сводный 0–100</h3>
      </div>
      <button className="w-9 h-9 rounded-full bg-ink-900/5 hover:bg-ink-900/10 flex items-center justify-center text-ink-700 transition" aria-label="open">
        <Icons.ArrowUR size={16}/>
      </button>
    </div>
    <div className="flex-1 flex items-center justify-center -mt-2">
      <RiskGauge value={value} size={240}/>
    </div>
    <div className="flex items-center justify-between gap-2">
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-cream-50 border border-ink-900/5 text-[11px] text-ink-700">
        <Icons.Pulse size={13} stroke={1.8}/>
        <span>За месяц {delta>0?'+':''}{delta} пт</span>
      </div>
      <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-ink-900 text-white text-[11px] font-medium hover:bg-ink-800 transition">
        <Icons.Refresh size={12} stroke={2}/> Пересчитать
      </button>
    </div>
  </div>
);

// ── Risk decomposition card
const RiskDecompCard = ({ data }) => (
  <div className="glass-strong rounded-4xl p-6 shadow-card lift h-full min-h-[420px] flex flex-col">
    <div className="flex items-start justify-between">
      <div>
        <div className="text-ink-500 text-[12px] font-medium">Декомпозиция</div>
        <h3 className="text-2xl font-semibold tracking-tight text-ink-900 leading-tight">Риск портфеля</h3>
      </div>
      <div className="flex items-center gap-2">
        <span className="px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5 text-[10px] font-mono tracking-wider text-ink-700">
          14.2%
        </span>
        <button className="w-9 h-9 rounded-full bg-ink-900/5 hover:bg-ink-900/10 flex items-center justify-center text-ink-700 transition">
          <Icons.ArrowUR size={16}/>
        </button>
      </div>
    </div>
    <div className="flex-1 flex items-center -mx-1">
      <Waterfall data={data}/>
    </div>
    <div className="flex items-center gap-4 text-[11px] text-ink-500 pt-2 border-t border-ink-900/5">
      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-ink-900"/> отдельно</span>
      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-rust-500"/> диверсификация</span>
      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-gold-400"/> итог</span>
    </div>
  </div>
);

// ── Sector mix card
const SectorMixCard = ({ sectors }) => (
  <div className="glass-strong rounded-4xl p-6 shadow-card lift h-full flex flex-col min-h-[200px]">
    <div className="flex items-start justify-between mb-1">
      <div>
        <div className="text-ink-500 text-[12px] font-medium">Структура</div>
        <h3 className="text-xl font-semibold tracking-tight text-ink-900 leading-tight">По секторам</h3>
      </div>
      <div className="px-2.5 py-1 rounded-full bg-rust-500/12 text-rust-600 text-[10px] font-semibold tracking-wider uppercase flex items-center gap-1">
        <Icons.Warning size={11} stroke={2}/> Перевес IT
      </div>
    </div>
    <div className="mt-3">
      <SectorBar sectors={sectors}/>
    </div>
    <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-4">
      {sectors.map(s => {
        const IconC = sectorIcon(s.name);
        return (
          <div key={s.name} className="flex items-center gap-2.5 group">
            <span className="w-6 h-6 rounded-lg flex items-center justify-center" style={{ background:s.hue+'25' }}>
              <IconC size={12} className="text-ink-700" stroke={1.6}/>
            </span>
            <div className="flex-1 min-w-0">
              <div className="text-[11px] text-ink-700 font-medium truncate">{s.name}</div>
            </div>
            <div className="text-[12px] font-semibold num text-ink-900">{s.pct}%</div>
          </div>
        );
      })}
    </div>
  </div>
);

// ── Quick AI insight (dark pill row, echoes the reference "Onboarding Task" mini cards style)
const AIInsightCard = ({ verdict }) => (
  <div className="rounded-4xl p-6 shadow-dark lift h-full flex flex-col min-h-[200px]"
       style={{ background:'linear-gradient(160deg, #1c1b1a 0%, #2a2825 100%)' }}>
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2 text-gold-400 text-[10px] font-mono tracking-widest uppercase">
        <Icons.Sparkles size={13} stroke={1.8}/> AI · HAIKU
      </div>
      <button className="w-8 h-8 rounded-full bg-white/5 hover:bg-white/10 flex items-center justify-center text-white/70 transition">
        <Icons.ArrowUR size={14}/>
      </button>
    </div>
    <div className="mt-4 text-white text-[15px] leading-relaxed font-light">
      Индекс <span className="text-gold-400 font-semibold num">62</span> — верх «умеренной» зоны.
      Портфель соответствует профилю, но почти весь риск собран в <span className="text-white font-semibold">2 бумагах</span> из 9.
    </div>
    <div className="mt-auto pt-4 flex items-center gap-2 flex-wrap">
      <span className="px-2.5 py-1 rounded-full bg-white/8 text-white/70 text-[10px] font-mono tracking-wider">RAG: GS_Q2_2026</span>
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
      <div className="flex items-start justify-between gap-8 flex-wrap mb-6">
        <div className="flex-1 min-w-[480px]">
          <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-3">
            <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/>
            Portfolio Risk Report · Tier {p.meta.tier}
          </div>
          <h1 className="text-[58px] leading-[1.02] tracking-[-0.03em] font-light text-ink-900 max-w-[860px]">
            {v.headline}<span className="text-ink-400">.</span>
          </h1>
          <p className="text-[18px] text-ink-500 mt-3 max-w-[640px] font-light">{v.sub}</p>
        </div>

        {/* big stat triplet — echoes reference 78/56/203 */}
        <div className="flex items-end gap-2 pt-4 divide-x divide-ink-900/10">
          {p.heroStats.map((s,i) => {
            const IconC = { briefcase: Icons.Briefcase, trendUp: Icons.TrendUp, wallet: Icons.Wallet }[s.icon];
            return <HeroStat key={i} value={s.value} label={s.label} IconC={IconC}/>;
          })}
        </div>
      </div>

      {/* Factor pills strip (echoes reference small KPI bars) */}
      <div className="flex items-end gap-6 flex-wrap mb-10">
        {p.factorPills.map((f,i) => <FactorPill key={i} {...f}/>)}
        {/* hatch "rest of metrics" bar — echoes the reference dashed bar */}
        <div className="flex-1 min-w-[160px] flex flex-col gap-2">
          <span className="text-[12px] text-ink-500 font-medium">Прочие метрики</span>
          <div className="h-7 rounded-full hatch border border-ink-900/8"/>
        </div>
      </div>

      {/* Asymmetric main grid */}
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 lg:col-span-3"><TopHotspotCard h={p.topHotspot}/></div>
        <div className="col-span-12 sm:col-span-6 lg:col-span-3"><RiskDecompCard data={p.riskDecomp}/></div>
        <div className="col-span-12 sm:col-span-6 lg:col-span-3"><RiskGaugeCard value={v.riskIndex} delta={v.riskTrendDelta}/></div>
        <div className="col-span-12 lg:col-span-3 grid grid-rows-2 gap-5">
          <SectorMixCard sectors={p.sectors}/>
          <AIInsightCard verdict={v}/>
        </div>
      </div>
    </section>
  );
};

Object.assign(window, { Hero });
