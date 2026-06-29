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
  <div className={`flex items-center gap-3 sm:gap-4 px-3 sm:px-5 py-3 rounded-2xl transition-colors hover:bg-cream-50
                   ${isMax?'bg-cream-50':''}`}>
    <div className="w-12 sm:w-16 text-[12px] font-medium text-ink-500 flex-shrink-0">{p.label}</div>
    <div className="flex-1 grid grid-cols-3 gap-2 min-w-0">
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-gold-400 flex-shrink-0"/>
        <span className="text-[14px] font-semibold num text-ink-900 whitespace-nowrap">{p.p>=0?'+':'−'}{Math.abs(p.p).toFixed(1)}%</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-ink-900 flex-shrink-0"/>
        <span className="text-[14px] num text-ink-700 whitespace-nowrap">{p.s>=0?'+':'−'}{Math.abs(p.s).toFixed(1)}%</span>
      </div>
      <div className="flex items-center gap-1.5 justify-end">
        <span className="text-[12px] text-ink-500">Δ</span>
        <span className={`text-[14px] font-semibold num whitespace-nowrap ${p.d>=0?'text-sage-600':'text-rust-600'}`}>{p.d>=0?'+':'−'}{Math.abs(p.d).toFixed(1)} пп</span>
      </div>
    </div>
  </div>
);

// Horizon (in months) for a period label.  'YTD' → 'YTD'; unknown → null.
// Handles '1м'/'1 мес'/'1M'/'12М' etc.
const _perfMonths = (label) => {
  const t = String(label || '').toLowerCase();
  if (t.includes('ytd')) return 'YTD';
  const d = t.replace(/[^0-9]/g, '');
  return d ? parseInt(d, 10) : null;
};

// Order periods 1<3<6<12, YTD last (sources sometimes emit YTD between 3 and 6).
const _perfOrder = (a, b) => {
  const va = _perfMonths(a.label), vb = _perfMonths(b.label);
  const ka = va === 'YTD' ? 99 : (va || 50);
  const kb = vb === 'YTD' ? 99 : (vb || 50);
  return ka - kb;
};

const Performance = () => {
  const p = window.PORTFOLIO.performance;
  // Periods sorted into a logical horizon order for BOTH the selector and table.
  const periods = [...(p.periods || [])].sort(_perfOrder);
  const labels  = periods.map(x => x.label);
  // Default to the 12-month view (longest cumulative window), else the last.
  const defLabel = labels.find(l => _perfMonths(l) === 12) || labels[labels.length-1] || '';
  const [period, setPeriod] = React.useState(defLabel);
  const sel = periods.find(x => x.label === period) || {};

  const volPort = (p.summary || {}).volPort ?? (p.vol || {}).port;
  const s = { ret: sel.p ?? 0,
              exc: sel.d ?? ((sel.p ?? 0) - (sel.s ?? 0)),
              spx: sel.s ?? 0,
              volPort };
  const fmt  = (x) => `${x>=0?'+':'−'}${Math.abs(Number(x)||0).toFixed(1)}`;
  const beats = (s.exc || 0) >= 0;

  // Calendar months elapsed this year — positions the YTD window's x-axis.
  const ytdMonths = (() => {
    const g = String((window.PORTFOLIO.meta || {}).generated || '');
    const mm = g.match(/\.(\d{2})\.\d{4}/);          // dd.MM.yyyy → month
    return mm ? Math.max(1, parseInt(mm[1], 10)) : 6;
  })();

  // Build a REAL cumulative curve for the SELECTED window by exact nesting of the
  // period endpoints: cumulative over [window-start … −h] = (1+r_window)/(1+r_h)−1.
  // No interpolation — every point is algebra over the SAME figures shown in the
  // breakdown table, so the chart's endpoint == the headline == its table row.
  const series = React.useMemo(() => {
    const byMonth = {};
    periods.forEach(r => { const m = _perfMonths(r.label); if (m !== 'YTD' && m) byMonth[m] = r; });
    const selM = _perfMonths(period);
    const span = selM === 'YTD' ? ytdMonths : selM;
    const rM = (sel.p || 0) / 100, sM = (sel.s || 0) / 100;
    if (!span || !isFinite(span)) {
      return { xs: [0, 1], labels: ['старт', 'сейчас'], port: [0, sel.p || 0], spx: [0, sel.s || 0] };
    }
    const nested = [1, 3, 6, 12].filter(h => h < span && byMonth[h]).sort((a, b) => b - a);
    const pts = [{ off: span, port: 0, spx: 0 }];
    nested.forEach(h => {
      const rh = (byMonth[h].p || 0) / 100, sh = (byMonth[h].s || 0) / 100;
      pts.push({ off: h, port: ((1+rM)/(1+rh) - 1) * 100, spx: ((1+sM)/(1+sh) - 1) * 100 });
    });
    pts.push({ off: 0, port: sel.p || 0, spx: sel.s || 0 });
    return {
      xs:     pts.map(pt => (span - pt.off) / span),
      labels: pts.map((pt, i) => i === 0 ? 'старт' : (i === pts.length-1 ? 'сейчас' : `−${pt.off}М`)),
      port:   pts.map(pt => pt.port),
      spx:    pts.map(pt => pt.spx),
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [period]);

  return (
    <section id="performance" className="rise" data-screen-label="03 Performance">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-6">
        <div>
          <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2">
            <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/> Performance · {period}
          </div>
          <h2 className="text-[clamp(28px,3.4vw,40px)] leading-[1.05] tracking-[-0.02em] font-light text-ink-900">
            Рост против рынка<span className="text-ink-400">.</span>
          </h2>
          <p className="text-[15px] text-ink-500 mt-2 font-light">Накопленная доходность портфеля в сравнении с S&P 500.</p>
        </div>

        <div className="flex items-center gap-1 p-1 rounded-full bg-white/60 border border-ink-900/8 backdrop-blur-md flex-wrap">
          {labels.map(pr => (
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
          <div className="flex items-start justify-between gap-3 flex-wrap mb-4">
            <div>
              <div className="text-ink-500 text-[12px] font-medium mb-1">Доходность за период · {period}</div>
              <div className="flex items-end gap-3 flex-wrap">
                <span className="text-[48px] leading-none font-light num text-ink-900">{fmt(s.ret)}<span className="text-[28px] text-ink-500">%</span></span>
                <span className={`px-2.5 py-1 rounded-full text-[11px] font-semibold mb-1.5 flex items-center gap-1 ${beats?'bg-sage-500/15 text-sage-600':'bg-rust-500/15 text-rust-600'}`}>
                  <Icons.TrendUp size={11} stroke={2.2}/> {fmt(s.exc)} пп vs S&P
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
          <PerfChart labels={series.labels} port={series.port} spx={series.spx} xs={series.xs}/>
        </div>

        {/* Side cards */}
        <div className="col-span-12 lg:col-span-4 grid grid-cols-2 lg:grid-cols-1 gap-5">
          <PerfSummaryCard label="Доходность" value={`${fmt(s.ret)}%`} sub={`за ${period}`} accent="gold" IconC={Icons.TrendUp}/>
          <PerfSummaryCard label={beats?'Опережение':'Отставание'} value={`${fmt(s.exc)}пп`} sub={beats?'портфель быстрее рынка':'портфель медленнее рынка'} accent="dark" IconC={Icons.Bolt}/>
          <div className="col-span-2 lg:col-span-1 grid grid-cols-2 gap-3">
            <PerfSummaryCard label="Волатильность" value={volPort!=null ? `${volPort}%` : '—'} sub="год." accent="light"/>
            <PerfSummaryCard label="S&P 500" value={`${fmt(s.spx)}%`} sub={`за ${period}`} accent="light"/>
          </div>
        </div>

        {/* Period breakdown table */}
        <div className="col-span-12 glass-strong rounded-4xl shadow-card p-6">
          <div className="flex items-center justify-between mb-3 px-2">
            <div className="text-[13px] font-semibold text-ink-900">Разбивка по периодам</div>
            <div className="text-[11px] text-ink-500 font-mono">Портфель / S&P 500 / Опережение</div>
          </div>
          <div className="space-y-1">
            {periods.map(pr => <PeriodRow key={pr.label} p={pr} isMax={pr.label === period}/>)}
          </div>
        </div>
      </div>
    </section>
  );
};

Object.assign(window, { Performance });
