/* DEEP Overview — verdict, risk gauge, KPIs, mandate compliance, risk concentration */

const HeroStat = ({ value, label, IconC, small }) => (
  <div className="flex flex-col items-center gap-1 px-4">
    <IconC size={17} className="text-ink-500 mb-1" stroke={1.4}/>
    <div className={`font-light tracking-tight num leading-none text-ink-900 ${small?'text-2xl pt-2':'text-5xl'}`}>{value}</div>
    <div className="text-[12px] text-ink-500 font-medium tracking-tight">{label}</div>
  </div>
);

// Compact risk-index gauge for the HERO right column (moved up from the main
// grid — user request «перенеси Индекс риска в эту зону»).  Horizontal layout:
// gauge on the left, tier + forward return/Sharpe on the right.
const HeroGaugeCard = ({ v }) => (
  <div className="glass-strong rounded-4xl p-5 shadow-card lift flex items-center gap-5">
    <div className="flex-shrink-0"><RiskGauge value={v.riskIndex} size={150}/></div>
    <div className="flex flex-col gap-2.5 min-w-0">
      <div className="flex items-center gap-2 flex-wrap">
        <div>
          <div className="text-ink-500 text-[10px] font-medium tracking-widest uppercase">Индекс риска</div>
          <h3 className="text-base font-semibold tracking-tight text-ink-900 leading-tight">Сводный 0–100</h3>
        </div>
        {(v.riskTier && v.riskTier !== '–') &&
          <span className="px-2 py-0.5 rounded-full bg-gold-400/25 text-gold-700 text-[9px] font-mono font-bold tracking-wider uppercase">{v.riskTier}</span>}
      </div>
      {(v.expReturn && v.expReturn !== '–') && (
        <div className="flex items-center gap-4">
          <div>
            <div className="text-[9px] text-ink-500 font-medium uppercase tracking-wider">Ожид. дох. (год.)</div>
            <div className="num text-lg font-light text-sage-600 leading-none mt-0.5">{v.expReturn}</div>
          </div>
          {(v.expSharpe && v.expSharpe !== '–') && (
            <div>
              <div className="text-[9px] text-ink-500 font-medium uppercase tracking-wider">Фвд-Sharpe</div>
              <div className="num text-lg font-light text-ink-900 leading-none mt-0.5">{v.expSharpe}</div>
            </div>
          )}
        </div>
      )}
    </div>
  </div>
);

// Verdict / AI summary dark card with bullets
const VerdictCard = ({ v }) => (
  <div className="rounded-4xl p-7 shadow-dark lift flex flex-col h-full"
       style={{ background:'linear-gradient(160deg, #1c1b1a 0%, #2a2825 100%)' }}>
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2 text-gold-400 text-[10px] font-mono tracking-widest uppercase">
        <Icons.Sparkles size={13} stroke={1.8}/> AI вердикт · {window.DEEP.meta.aiModel}
      </div>
      <span className="text-white/30 text-[10px] font-mono tracking-widest">DEEP</span>
    </div>
    <p className="text-white text-[15px] leading-relaxed font-light mb-5">{v.summary}</p>
    <div className="mt-auto space-y-2.5">
      {v.bullets.map((b,i) => (
        <div key={i} className="flex items-start gap-3">
          {b.tag
            ? <span className="mt-[3px] px-1.5 py-0.5 rounded-md bg-gold-400/15 text-gold-400 text-[9px] font-mono font-bold tracking-wider uppercase flex-shrink-0 w-[58px] text-center">{b.tag}</span>
            : <span className="mt-[3px] w-1.5 h-1.5 rounded-full bg-gold-400/50 flex-shrink-0"/>}
          <p className="text-white/70 text-[11.5px] leading-snug font-light flex-1">{b.text}
            {b.src && <span className="text-white/35 font-mono"> [{b.src}]</span>}
          </p>
        </div>
      ))}
    </div>
  </div>
);

// Mandate compliance card
const MandateCard = ({ m }) => (
  <div className="glass-strong rounded-4xl p-6 shadow-card lift flex flex-col">
    <div className="flex items-start justify-between mb-1 gap-2">
      <div>
        <div className="text-ink-500 text-[12px] font-medium">Соответствие мандату</div>
        <h3 className="text-xl font-semibold tracking-tight text-ink-900 leading-tight">{m.profile}</h3>
      </div>
      {m.violations > 0
        ? <span className="px-2.5 py-1 rounded-full bg-rust-500/12 text-rust-600 text-[10px] font-semibold tracking-wider uppercase flex items-center gap-1 flex-shrink-0">
            <Icons.Warning size={11} stroke={2}/> {m.violations} нарушение
          </span>
        : <span className="px-2.5 py-1 rounded-full bg-sage-500/15 text-sage-600 text-[10px] font-semibold tracking-wider uppercase flex-shrink-0">соответствует</span>}
    </div>
    <div className="text-[10.5px] text-ink-400 font-mono mb-2">
      целевая волат. ≈{m.targetVol}% · отклонение от ориентира ≤{m.trackingCap}%
    </div>
    {/* Маржинальный долг — рендерится ТОЛЬКО на реально левереджёванном счёте
        (кэш < 0); на обычном портфеле упоминаний плеча нет (правило §−13). */}
    {m.leveraged && (
      <div className="flex items-center gap-2 rounded-2xl bg-rust-500/10 border border-rust-500/30 px-3 py-2 mb-3">
        <Icons.Warning size={13} className="text-rust-600 flex-shrink-0" stroke={2}/>
        <span className="text-[11px] text-rust-600 font-medium">
          Маржинальный долг {m.marginPct > 0 ? `≈${m.marginPct}% NAV` : 'обнаружен'} — позиции частично куплены в долг
        </span>
      </div>
    )}
    <div className="space-y-3.5 mt-auto">
      {m.rows.map((r,i) => {
        const tone = { ok:'text-ink-900', over:'text-rust-600', under:'text-gold-700' }[r.state];
        return (
          <div key={i}>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[12px] text-ink-700">{r.label}</span>
              <span className={`text-[12px] font-semibold num ${tone}`}>{r.value}%</span>
            </div>
            <MandateBar value={r.value} lo={r.lo} hi={r.hi} state={r.state}/>
            <div className="text-[9.5px] text-ink-400 font-mono mt-1">допустимо {r.lo}–{r.hi}%</div>
          </div>
        );
      })}
    </div>
  </div>
);

// KPI card — value + a prominent 12-month trend chart (user request «добавь
// графики к показателям»).  Draws the design <Sparkline> from the real numeric
// series; falls back to the server-rendered SVG if only that is present.
// Per-metric methodology note (hover) — makes the denominators explicit
// (Sprint-3 #8): Sharpe/Sortino use the STRUCTURAL factor volatility σ, not a
// realised sample vol.
const _KPI_METHOD = {
  sharpe: 'Знаменатель — структурная (факторная) волатильность σ = √(w′Σw), Σ = B·F·Bᵀ + D; числитель — геом. годовая доходность − валютно-сопоставленная RFR.',
  cvar:   'Средний убыток в худшие 5% дней (1-дневный горизонт), эмпирически + bootstrap-CI.',
  dd:     'Максимальная просадка пик→дно по реконструированной кривой капитала exp(Σ log-доходностей).',
};

const KpiCard = ({ k }) => {
  const border = { normal:'#5d7c5c', good:'#caa01a', watch:'#c47358' }[k.status];
  const hasPts = Array.isArray(k.pts) && k.pts.length >= 2;
  return (
    <div className="glass-strong rounded-4xl p-6 shadow-card lift flex flex-col"
         style={{ borderTop:`2px solid ${border}` }}>
      <div className="flex items-center justify-between mb-2">
        <div className="text-[10px] tracking-widest uppercase text-ink-500 font-mono cursor-help"
             title={_KPI_METHOD[k.key] || undefined}>{k.name}</div>
        <span className="text-[40px] leading-none font-light num text-ink-900 tracking-tight">{k.value}</span>
      </div>
      {/* trend chart — how the metric moved over the last 12 months */}
      <div className="rounded-2xl bg-cream-50/70 border border-ink-900/5 px-3 pt-2 pb-1.5 mb-3">
        <div className="flex items-center justify-between text-[8.5px] tracking-widest uppercase text-ink-400 font-mono mb-0.5">
          <span>Динамика · 12 мес</span><span>сейчас</span>
        </div>
        <div className="h-12">
          {hasPts
            ? <Sparkline points={k.pts} color={k.color} height={44} width={300} gradId={`spk-${k.key}`}/>
            : (k.svg
                ? <div className="w-full h-full" dangerouslySetInnerHTML={{ __html: k.svg }}/>
                : <div className="w-full h-full flex items-center justify-center text-[9px] text-ink-300 font-mono">нет истории</div>)}
        </div>
      </div>
      <div className="text-[10.5px] text-ink-400 font-mono leading-snug mb-4">{k.sub}</div>
      <div className="mt-auto flex items-start gap-2.5 rounded-2xl bg-cream-50 border border-ink-900/5 px-3.5 py-3">
        <Icons.Sparkles size={13} className="text-gold-600 mt-0.5 flex-shrink-0" stroke={1.8}/>
        <p className="text-[11.5px] text-ink-700 leading-snug font-light">{k.ai}</p>
      </div>
    </div>
  );
};

// Concentration table
const ConcentrationCard = ({ rows }) => (
  <div className="glass-strong rounded-4xl p-6 shadow-card lift flex flex-col">
    <div className="flex items-start justify-between mb-4">
      <div>
        <div className="text-ink-500 text-[12px] font-medium">Где сосредоточен риск</div>
        <h3 className="text-xl font-semibold tracking-tight text-ink-900 leading-tight">Топ-вкладчики</h3>
      </div>
      <span className="px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5 text-[10px] font-mono tracking-wider text-ink-700">TRC · Euler</span>
    </div>
    <div className="grid grid-cols-[1.4fr_1fr_1fr_1.6fr_auto] gap-3 px-1 pb-2 text-[9px] tracking-widest uppercase text-ink-400 font-mono border-b border-ink-900/8">
      <div>Тикер</div><div className="text-right">Вес</div><div className="text-right">Beta</div><div>Вклад в риск</div><div></div>
    </div>
    <div className="divide-y divide-ink-900/5 mt-1">
      {rows.map((r,i) => {
        const hot = r.status === 'HOTSPOT';
        return (
          <div key={i} className="grid grid-cols-[1.4fr_1fr_1fr_1.6fr_auto] gap-3 items-center px-1 py-2.5">
            <div className="text-[14px] font-bold num text-ink-900">{r.t}</div>
            <div className="text-[13px] num text-ink-700 text-right">{r.w.toFixed(1)}%</div>
            <div className="text-[13px] num text-ink-700 text-right">{r.beta.toFixed(2)}</div>
            <div className="flex items-center gap-2">
              <div className="flex-1"><MiniBar value={r.risk} max={36} color={hot?'#f5d04e':'#a8a293'}/></div>
              <span className={`text-[12px] font-semibold num ${hot?'text-gold-700':'text-ink-900'}`}>{r.risk.toFixed(1)}%</span>
            </div>
            <div className="flex justify-end">
              {hot
                ? <span className="px-2 py-0.5 rounded-full bg-gold-400 text-ink-900 text-[9px] font-bold tracking-wider uppercase">Hotspot</span>
                : <span className="px-2 py-0.5 rounded-full bg-cream-200/60 text-ink-500 text-[9px] font-semibold tracking-wider uppercase">Norm</span>}
            </div>
          </div>
        );
      })}
    </div>
  </div>
);

const MiniBar = ({ value, max=30, color='#1c1b1a', height=4 }) => (
  <div className="w-full bg-ink-900/8 rounded-full overflow-hidden" style={{ height }}>
    <div className="rounded-full" style={{ width:`${Math.max(0, Math.min(100,(value/max)*100))}%`, height:'100%', background:color }}/>
  </div>
);

// Waterfall card
const WaterfallCard = ({ data, ai }) => (
  <div className="glass-strong rounded-4xl p-6 shadow-card lift flex flex-col">
    <div className="flex items-start justify-between mb-2">
      <div>
        <div className="text-ink-500 text-[12px] font-medium">Декомпозиция</div>
        <h3 className="text-xl font-semibold tracking-tight text-ink-900 leading-tight">Риск портфеля</h3>
      </div>
      <span className="px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5 text-[10px] font-mono tracking-wider text-ink-700">{data.total.toFixed(1)}% год.</span>
    </div>
    <div className="flex-1 flex items-center -mx-1">
      <Waterfall data={data} height={200}/>
    </div>
    <div className="flex items-center gap-4 text-[11px] text-ink-500 pt-2 border-t border-ink-900/5">
      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-ink-900"/> отдельно</span>
      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-rust-500"/> диверсификация</span>
      <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-gold-400"/> итог</span>
    </div>
  </div>
);

const Overview = () => {
  const p = window.DEEP;
  const v = p.verdict;
  return (
    <section id="overview" className="rise" data-screen-label="01 Overview">
      <div className="flex items-start justify-between gap-x-10 gap-y-6 flex-wrap mb-8">
        <div className="flex-1 min-w-[280px] max-w-[760px]">
          <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-3">
            <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/>
            Portfolio Risk Report · Tier {p.meta.tier}
          </div>
          <h1 className="text-[clamp(32px,4.4vw,54px)] leading-[1.05] tracking-[-0.03em] font-light text-ink-900">
            {v.headline}<span className="text-ink-400">.</span>
          </h1>
          <p className="text-[17px] text-ink-500 mt-3 max-w-[620px] font-light">{v.sub}</p>
        </div>
        {/* Right column — hero stat triplet + risk-index gauge moved up here */}
        <div className="flex flex-col gap-4 w-full lg:w-auto lg:min-w-[380px] lg:max-w-[460px]">
          <div className="flex items-end justify-start lg:justify-end gap-1 divide-x divide-ink-900/10">
            {p.heroStats.map((s,i) => {
              const IconC = { briefcase: Icons.Briefcase, wallet: Icons.Wallet, shield: Icons.Shield }[s.icon] || Icons.Briefcase;
              return <HeroStat key={i} value={s.value} label={s.label} IconC={IconC} small={s.small}/>;
            })}
          </div>
          <HeroGaugeCard v={v}/>
        </div>
      </div>

      {/* main grid: verdict / mandate (gauge moved to hero) */}
      <div className="grid grid-cols-12 gap-5 mb-5 items-stretch">
        <div className="col-span-12 lg:col-span-7"><VerdictCard v={v}/></div>
        <div className="col-span-12 lg:col-span-5"><MandateCard m={p.mandate}/></div>
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-5">
        {p.kpis.map(k => <KpiCard key={k.key} k={k}/>)}
      </div>

      {/* concentration + waterfall */}
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 lg:col-span-7"><ConcentrationCard rows={p.concentration}/></div>
        <div className="col-span-12 lg:col-span-5"><WaterfallCard data={p.riskDecomp} ai={p.concAI}/></div>
      </div>

      {/* AI strip */}
      <div className="mt-5 rounded-3xl p-5 flex items-start gap-4"
           style={{ background:'linear-gradient(120deg, #fbf3d9 0%, #f6ebc0 100%)' }}>
        <div className="w-10 h-10 rounded-2xl bg-ink-900 text-gold-400 flex items-center justify-center flex-shrink-0">
          <Icons.Sparkles size={17} stroke={1.7}/>
        </div>
        <div>
          <div className="text-[10px] tracking-widest uppercase font-mono text-ink-700 mb-1">AI · комментарий к риску</div>
          <p className="text-[14px] text-ink-900 leading-relaxed font-light">{p.concAI}</p>
        </div>
      </div>
    </section>
  );
};

Object.assign(window, { Overview });
