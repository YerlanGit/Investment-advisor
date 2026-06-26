/* Performance section — chart vs S&P 500 + period stats */

const PerfSummaryCard = ({ label, value, sub, accent, IconC }) => (
  <div className={`rounded-3xl p-5 lift shadow-card flex flex-col gap-1
                   ${accent==='dark' ? 'bg-ink-900 text-white' :
                     accent==='gold' ? 'bg-gold-400 text-ink-900' :
                     'glass-strong text-ink-900'}`}>
    <div className="flex items-center justify-between">
      <span className={`text-[11px] tracking-widest uppercase font-mono ${accent==='dark'?'text-white/50':'text-ink-500'}`}>
        {label}
      </span>
      {IconC && <IconC size={16} className={accent==='dark'?'text-white/60':'text-ink-500'} stroke={1.6}/>}
    </div>
    <div className="text-[36px] leading-none font-light num tracking-tight mt-1">{value}</div>
    <div className={`text-[12px] ${accent==='dark'?'text-white/55':'text-ink-500'} mt-1`}>{sub}</div>
  </div>
);

const PeriodRow = ({ p, isMax }) => (
  <div className={`flex items-center gap-4 px-5 py-3 rounded-2xl transition-colors hover:bg-cream-50
                   ${isMax?'bg-cream-50':''}`}>
    <div className="w-16 text-[12px] font-medium text-ink-500">{p.label}</div>
    <div className="flex-1 grid grid-cols-3 gap-2">
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-gold-400"/>
        <span className="text-[14px] font-semibold num text-ink-900">+{p.p.toFixed(1)}%</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-ink-900"/>
        <span className="text-[14px] num text-ink-700">+{p.s.toFixed(1)}%</span>
      </div>
      <div className="flex items-center gap-2 justify-end">
        <span className="text-[12px] text-ink-500">Δ</span>
        <span className="text-[14px] font-semibold num text-sage-600">+{p.d.toFixed(1)} пп</span>
      </div>
    </div>
  </div>
);

const Performance = () => {
  const p = window.PORTFOLIO.performance;
  const [period, setPeriod] = React.useState('12 мес');
  const periods = ['1 мес','3 мес','YTD','6 мес','12 мес'];

  return (
    <section id="performance" className="rise" data-screen-label="03 Performance">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-6">
        <div>
          <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2">
            <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/> Performance · 12 месяцев
          </div>
          <h2 className="text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900">
            Рост против рынка<span className="text-ink-400">.</span>
          </h2>
          <p className="text-[15px] text-ink-500 mt-2 font-light">Накопленная доходность портфеля в сравнении с S&P 500.</p>
        </div>

        <div className="flex items-center gap-1 p-1 rounded-full bg-white/60 border border-ink-900/8 backdrop-blur-md">
          {periods.map(pr => (
            <button key={pr} onClick={()=>setPeriod(pr)}
              className={`px-3.5 py-1.5 rounded-full text-[12px] font-medium transition-colors
                          ${period===pr?'bg-ink-900 text-white':'text-ink-700 hover:bg-ink-900/5'}`}>
              {pr}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-5">
        {/* Chart card */}
        <div className="col-span-12 lg:col-span-8 glass-strong rounded-4xl shadow-card lift p-7">
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="text-ink-500 text-[12px] font-medium mb-1">Накопленная доходность</div>
              <div className="flex items-end gap-3">
                <span className="text-[48px] leading-none font-light num text-ink-900">+14.2<span className="text-[28px] text-ink-500">%</span></span>
                <span className="px-2.5 py-1 rounded-full bg-sage-500/15 text-sage-600 text-[11px] font-semibold mb-1.5 flex items-center gap-1">
                  <Icons.TrendUp size={11} stroke={2.2}/> +5.1 пп vs S&P
                </span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 text-[11px] text-ink-700">
                <span className="w-3 h-1 rounded-full bg-gold-400"/> Ваш портфель
              </div>
              <div className="flex items-center gap-1.5 text-[11px] text-ink-500">
                <span className="w-3 h-[2px] bg-ink-900" style={{ backgroundImage:'linear-gradient(to right, #1c1b1a 50%, transparent 50%)', backgroundSize:'4px 2px' }}/> S&P 500
              </div>
            </div>
          </div>
          <PerfChart months={p.months} port={p.port} spx={p.spx}/>
        </div>

        {/* Side cards */}
        <div className="col-span-12 lg:col-span-4 grid grid-cols-2 lg:grid-cols-1 gap-5">
          <PerfSummaryCard label="Доходность" value="+14.2%" sub="за 12 месяцев" accent="gold" IconC={Icons.TrendUp}/>
          <PerfSummaryCard label="Опережение" value="+5.1пп" sub="портфель быстрее рынка" accent="dark" IconC={Icons.Bolt}/>
          <div className="col-span-2 lg:col-span-1 grid grid-cols-2 gap-3">
            <PerfSummaryCard label="Волатильность" value="14.8%" sub="рынок 11.2%" accent="light"/>
            <PerfSummaryCard label="S&P 500" value="+9.1%" sub="за 12 мес" accent="light"/>
          </div>
        </div>

        {/* Period breakdown table */}
        <div className="col-span-12 glass-strong rounded-4xl shadow-card p-6">
          <div className="flex items-center justify-between mb-3 px-2">
            <div className="text-[13px] font-semibold text-ink-900">Разбивка по периодам</div>
            <div className="text-[11px] text-ink-500 font-mono">Портфель / S&P 500 / Опережение</div>
          </div>
          <div className="space-y-1">
            {p.periods.map(pr => <PeriodRow key={pr.label} p={pr} isMax={pr.label === '12 мес'}/>)}
          </div>
        </div>
      </div>
    </section>
  );
};

Object.assign(window, { Performance });
