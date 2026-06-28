/* DEEP Holdings — 11 positions, expandable fundamentals + sector mix */

const SignalChip = ({ signal }) => {
  const styles = {
    'BUY':  'bg-sage-500/15 text-sage-600',
    'HOLD': 'bg-ink-900/6 text-ink-600',
    'TRIM': 'bg-gold-500/18 text-gold-700',
    'SELL': 'bg-rust-500 text-white',
  }[signal] || 'bg-ink-900/6 text-ink-600';
  return <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wider ${styles}`}>{signal}</span>;
};

const HBar = ({ value, max, color, height=4 }) => (
  <div className="w-full bg-ink-900/8 rounded-full overflow-hidden" style={{ height }}>
    <div className="rounded-full" style={{ width:`${Math.min(100,(value/max)*100)}%`, height:'100%', background:color }}/>
  </div>
);

const FundCell = ({ label, value, warn }) => (
  <div className="rounded-2xl bg-cream-50 border border-ink-900/5 px-3 py-2.5">
    <div className="text-[9.5px] uppercase tracking-wider text-ink-500 font-mono">{label}</div>
    <div className={`text-[14px] font-semibold num mt-0.5 ${warn?'text-rust-600':'text-ink-900'}`}>{value}</div>
  </div>
);

const HoldingRow = ({ h, open, onToggle }) => {
  const IconC = sectorIcon(h.cls);
  const pos = h.pnlPct >= 0;
  const hot = h.status === 'HOTSPOT';
  const atrWarn = parseFloat(h.fund.atr) > 3;
  return (
    <div className={`relative transition-colors ${open?'bg-cream-50/70':'hover:bg-cream-50/40'}`}>
      <button onClick={onToggle} className="w-full grid grid-cols-[36px_minmax(0,1.9fr)_minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.1fr)_84px_36px] items-center gap-3 px-5 py-3.5 text-left">
        <div className="w-8 h-8 rounded-xl bg-cream-100 border border-ink-900/5 flex items-center justify-center text-ink-700">
          <IconC size={15} stroke={1.6}/>
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-[15px] font-bold tracking-tight num text-ink-900 truncate">{h.short || h.t}</span>
            {hot && <span className="px-1.5 py-0.5 rounded-md bg-gold-400 text-ink-900 text-[8.5px] font-bold tracking-wider uppercase flex-shrink-0">Hotspot</span>}
          </div>
          <div className="text-[11.5px] text-ink-500 truncate mt-0.5">{h.name}</div>
        </div>
        <div className="text-[12px] text-ink-700">{h.cls}</div>
        <div>
          <div className="text-[13px] font-semibold num text-ink-900">{h.w.toFixed(1)}%</div>
          <div className="mt-1.5"><HBar value={h.w} max={17} color="#1c1b1a"/></div>
        </div>
        <div>
          <div className="text-[13px] font-semibold num text-ink-900">{h.risk.toFixed(1)}%</div>
          <div className="mt-1.5"><HBar value={h.risk} max={36} color={hot?'#f5d04e':'#a8a293'}/></div>
        </div>
        <div>
          <div className={`text-[14px] font-semibold num ${h.cash?'text-ink-400':pos?'text-sage-600':'text-rust-600'}`}>{pos?'+':''}{h.pnlPct.toFixed(1)}%</div>
          <div className={`text-[11px] num ${h.cash?'text-ink-300':pos?'text-sage-500':'text-rust-500'}`}>{pos?'+':'−'}${Math.abs(h.pnlUsd).toLocaleString('ru-RU')}</div>
        </div>
        <div className="flex justify-end"><SignalChip signal={h.signal}/></div>
        <div className={`w-8 h-8 rounded-full bg-ink-900/5 flex items-center justify-center text-ink-700 transition-transform ${open?'rotate-180':''}`}>
          <Icons.Chevron size={13}/>
        </div>
      </button>

      <div className="mob-detail overflow-hidden transition-[max-height,opacity] duration-500 ease-out"
           style={{ maxHeight: open?640:0, opacity: open?1:0 }}>
        <div className="px-5 pb-5 pt-1">
          <div className="rounded-3xl p-5 bg-white/70 border border-ink-900/5">
            <div className="flex items-start justify-between gap-4 mb-4">
              <div>
                <div className="text-[10.5px] tracking-widest uppercase text-ink-500 font-mono">Фундаментал · SEC EDGAR</div>
                <div className="text-[15px] text-ink-900 font-medium mt-0.5">{h.name}</div>
              </div>
              <span className="text-[10px] font-mono text-ink-400 tracking-wider px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5">{h.cls}</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
              <FundCell label="ROE"        value={h.fund.roe}/>
              <FundCell label="Маржа"      value={h.fund.margin}/>
              <FundCell label="Долг/А"     value={h.fund.debt}/>
              <FundCell label="Рост г/г"   value={h.fund.growth}/>
              <FundCell label="ATR"        value={h.fund.atr} warn={atrWarn}/>
              <FundCell label="Altman-Z"   value={h.fund.z}/>
            </div>
            <div className="mt-4 flex items-start gap-3 text-[13px] text-ink-700 leading-relaxed">
              <Icons.Sparkles size={14} className="text-gold-600 mt-1 flex-shrink-0" stroke={1.8}/>
              <p className="font-light">{h.note} <span className="text-ink-400 font-mono text-[11px]">[SEC EDGAR] [Quant Engine]</span></p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const SectorMix = ({ sectors, warns }) => {
  let acc = 0;
  return (
    <div className="glass-strong rounded-4xl p-6 shadow-card lift flex flex-col h-full">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="text-ink-500 text-[12px] font-medium">Структура</div>
          <h3 className="text-xl font-semibold tracking-tight text-ink-900 leading-tight">По секторам</h3>
        </div>
        <span className="px-2.5 py-1 rounded-full bg-rust-500/12 text-rust-600 text-[10px] font-semibold tracking-wider uppercase flex items-center gap-1">
          <Icons.Warning size={11} stroke={2}/> Перевес IT
        </span>
      </div>
      {/* stacked bar */}
      <svg viewBox="0 0 320 18" className="w-full h-3.5 mb-4" preserveAspectRatio="none">
        <rect x="0" y="0" width="320" height="18" rx="9" fill="#efe9d8"/>
        {sectors.map((s,i) => {
          const x = (acc/100)*320; const w = (s.pct/100)*320 - (i===sectors.length-1?0:1.5); acc += s.pct;
          return <rect key={s.name} x={x} y="0" width={Math.max(0,w)} height="18" fill={s.hue} rx={i===0?9:0} ry={i===0?9:0}/>;
        })}
      </svg>
      <div className="space-y-2 flex-1">
        {sectors.map(s => (
          <div key={s.name} className="flex items-center gap-2.5">
            <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background:s.hue }}/>
            <span className={`flex-1 text-[12px] ${s.warn?'text-gold-700 font-semibold':'text-ink-700'}`}>{s.name}</span>
            <span className={`text-[12px] font-semibold num ${s.warn?'text-gold-700':'text-ink-900'}`}>{s.pct}%</span>
          </div>
        ))}
      </div>
      <div className="mt-4 space-y-2">
        {warns.map((w,i) => (
          <div key={i} className="flex items-start gap-2 rounded-2xl bg-gold-400/12 border border-gold-400/35 px-3 py-2">
            <Icons.Warning size={12} className="text-gold-700 mt-0.5 flex-shrink-0" stroke={2}/>
            <p className="text-[10.5px] text-gold-700 leading-snug font-medium">{w}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

const Holdings = () => {
  const [openIdx, setOpenIdx] = React.useState(0);
  const [filter, setFilter] = React.useState('Все');
  const p = window.DEEP;
  const filters = ['Все','HOTSPOT','Акции США','Защитные','В минусе','SELL · TRIM'];
  const rows = p.holdings.filter(h => {
    if (filter==='Все') return true;
    if (filter==='HOTSPOT') return h.status==='HOTSPOT';
    if (filter==='Акции США') return h.cls==='Акции США';
    if (filter==='Защитные') return ['Облигации','Сырьё','Ден. средства'].includes(h.cls);
    if (filter==='В минусе') return h.pnlPct < 0;
    if (filter==='SELL · TRIM') return ['SELL','TRIM'].includes(h.signal);
    return true;
  });

  return (
    <section id="holdings" className="rise" data-screen-label="02 Holdings">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-6">
        <div>
          <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2">
            <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/> Holdings · {p.meta.positions} позиций
          </div>
          <h2 className="text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900">
            Что вы держите<span className="text-ink-400">.</span>
          </h2>
          <p className="text-[15px] text-ink-500 mt-2 font-light">Нажмите на строку, чтобы увидеть фундаментал бумаги.</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {filters.map(f => (
            <button key={f} onClick={()=>{setFilter(f); setOpenIdx(-1);}}
              className={`px-3.5 py-1.5 rounded-full text-[12px] font-medium transition-colors
                          ${filter===f?'bg-ink-900 text-white':'bg-white/60 text-ink-700 hover:bg-white border border-ink-900/8'}`}>
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-5 items-stretch">
        <div className="col-span-12 lg:col-span-8">
          <div className="glass-strong rounded-4xl shadow-card overflow-hidden">
            <div className="swipe-hint items-center gap-1 text-[10px] font-mono text-gold-700 bg-gold-400/15 rounded-full px-2.5 py-1 mb-2.5 w-max">↔ листайте таблицу</div>
    <div className="mob-scroll-x"><div>
            <div className="grid grid-cols-[36px_minmax(0,1.9fr)_minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.1fr)_84px_36px] items-center gap-3 px-5 py-3 border-b border-ink-900/6 text-[9.5px] tracking-widest uppercase text-ink-500 font-mono">
              <div></div><div>Тикер · Имя</div><div>Класс</div><div>Вес</div><div>Риск</div><div>P/L</div><div className="text-right">Сигнал</div><div></div>
            </div>
            <div className="divide-y divide-ink-900/5">
              {rows.map((h,i) => (
                <HoldingRow key={h.t} h={h} open={openIdx===i} onToggle={()=>setOpenIdx(openIdx===i?-1:i)}/>
              ))}
              {rows.length===0 && <div className="px-6 py-12 text-center text-ink-500 text-[14px]">Ничего не подходит под фильтр «{filter}».</div>}
            </div>
            </div></div>
          </div>
        </div>
        <div className="col-span-12 lg:col-span-4">
          <SectorMix sectors={p.sectors} warns={p.sectorWarn}/>
        </div>
      </div>

      <div className="mt-5 rounded-3xl p-5 glass-strong shadow-card flex items-start gap-4">
        <div className="w-10 h-10 rounded-2xl bg-ink-900 text-gold-400 flex items-center justify-center flex-shrink-0">
          <Icons.Sparkles size={17} stroke={1.7}/>
        </div>
        <div>
          <div className="text-[10px] tracking-widest uppercase font-mono text-ink-500 mb-1">AI · сводка по составу</div>
          <p className="text-[14px] text-ink-800 leading-relaxed font-light">{p.holdingsAI}</p>
        </div>
      </div>
    </section>
  );
};

Object.assign(window, { Holdings });
