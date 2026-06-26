/* DEEP Plan — Action Plan levels + Expected effect + AI Ideas */

const actionChipCls = {
  BUY:'bg-sage-500/15 text-sage-600', HOLD:'bg-ink-900/6 text-ink-600',
  TRIM:'bg-gold-500/18 text-gold-700', SELL:'bg-rust-500 text-white',
};

const ActionPlan = ({ rows }) => (
  <div className="glass-strong rounded-4xl p-7 shadow-card">
    <div className="flex items-start justify-between gap-4 flex-wrap mb-5">
      <div>
        <h3 className="text-2xl font-semibold tracking-tight text-ink-900">Action Plan — уровни Buy / Sell / Stop</h3>
        <p className="text-[12px] text-ink-500 font-mono mt-1">ATR (Wilder RMA) · SMA50/200 · RSI(14) · MACD(12,26,9) · без внешних таргет-прайсов</p>
      </div>
      <span className="text-[10px] font-mono text-ink-400 tracking-wider px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5">Quant Engine</span>
    </div>
    <div className="grid grid-cols-[minmax(0,1fr)_72px_minmax(0,1fr)_minmax(0,1.5fr)_minmax(0,1fr)_minmax(0,1.6fr)] gap-3 px-1 pb-2.5 text-[9.5px] tracking-widest uppercase text-ink-400 font-mono border-b border-ink-900/8">
      <div>Тикер</div><div>Действие</div><div className="text-right">Цена</div><div className="text-right">Sell target</div><div className="text-right">Stop</div><div>Причина</div>
    </div>
    <div className="divide-y divide-ink-900/5">
      {rows.map((r,i) => (
        <div key={i} className={`grid grid-cols-[minmax(0,1fr)_72px_minmax(0,1fr)_minmax(0,1.5fr)_minmax(0,1fr)_minmax(0,1.6fr)] gap-3 items-center px-1 py-3 ${r.defer?'opacity-65':''}`}>
          <div className="text-[13.5px] font-bold num text-ink-900 flex items-center gap-1.5">
            {r.t}{r.hot && <span className="text-[10px]" title="Hotspot TRC > 20%">🔥</span>}
          </div>
          <div><span className={`px-2 py-0.5 rounded-full text-[9px] font-bold tracking-wider ${actionChipCls[r.action]}`}>{r.action}</span></div>
          <div className="text-right text-[12px] num text-ink-700">{r.price.toFixed(2)}</div>
          <div className="text-right text-[12px] num text-sage-600">{r.target}</div>
          <div className="text-right text-[12px] num text-rust-600">{r.stop}</div>
          <div className="text-[11px] text-ink-500 leading-tight">
            Score {r.score>0?'+':''}{r.score.toFixed(1)}{r.hot && ' · Hotspot TRC>20%'}{r.defer && ' · отложено (turnover cap)'}
          </div>
        </div>
      ))}
    </div>
    <div className="mt-4 rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3.5 flex items-start gap-3">
      <Icons.Sparkles size={14} className="text-gold-600 mt-0.5 flex-shrink-0" stroke={1.8}/>
      <p className="text-[12.5px] text-ink-700 leading-relaxed font-light">{window.DEEP.actionAI}</p>
    </div>
  </div>
);

const EffectGrid = ({ rows, verdict }) => (
  <div className="glass-strong rounded-4xl p-7 shadow-card">
    <div className="flex items-start justify-between gap-4 flex-wrap mb-5">
      <div>
        <h3 className="text-2xl font-semibold tracking-tight text-ink-900">Ожидаемый эффект на риск</h3>
        <p className="text-[12px] text-ink-500 font-mono mt-1">оценка «до / после» при исполнении Action Plan · горизонт 1 квартал</p>
      </div>
      <span className="text-[10px] font-mono text-gold-700 tracking-wider px-2.5 py-1 rounded-full bg-gold-400/15">Δ по идеям: MSFT · ORCL · SLV · SPCX · AAPL · GLD · NVDA</span>
    </div>
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {rows.map((r,i) => {
        const tone = { pos:'text-sage-600', neg:'text-rust-600', neut:'text-ink-500' }[r.tone];
        return (
          <div key={i} className="rounded-2xl bg-white/70 border border-ink-900/6 px-4 py-3.5">
            <div className="text-[9.5px] uppercase tracking-wider text-ink-500 font-mono mb-2">{r.name}</div>
            <div className="flex items-baseline gap-1.5 num">
              <span className="text-[14px] text-ink-400">{r.before}</span>
              <Icons.ArrowR size={11} className="text-gold-600" stroke={2.4}/>
              <span className="text-[16px] font-semibold text-ink-900">{r.after}</span>
            </div>
            <div className={`text-[11px] num font-semibold mt-1.5 ${tone}`}>{r.delta}</div>
          </div>
        );
      })}
    </div>
    <div className="mt-4 flex items-start gap-3 rounded-2xl bg-gold-400/12 border border-gold-400/35 px-4 py-3">
      <Icons.Scale size={15} className="text-gold-700 mt-0.5 flex-shrink-0" stroke={1.8}/>
      <p className="text-[12px] text-ink-800 leading-relaxed font-light"><span className="font-semibold text-gold-700">Сводный вердикт по плану:</span> {verdict}</p>
    </div>
    <div className="mt-3 rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3.5 flex items-start gap-3">
      <Icons.Sparkles size={14} className="text-gold-600 mt-0.5 flex-shrink-0" stroke={1.8}/>
      <p className="text-[12.5px] text-ink-700 leading-relaxed font-light">{window.DEEP.effectAI}</p>
    </div>
  </div>
);

const ideaTone = {
  grow:      { border:'#5d7c5c', chip:'bg-sage-500/15 text-sage-600', icon:Icons.TrendUp },
  rebalance: { border:'#1c1b1a', chip:'bg-ink-900/8 text-ink-700',    icon:Icons.Refresh },
  rotation:  { border:'#caa01a', chip:'bg-gold-500/18 text-gold-700', icon:Icons.Compass },
  hedge:     { border:'#a8a293', chip:'bg-ink-900/6 text-ink-600',    icon:Icons.Shield },
};

const PipeNode = ({ n, label, last }) => (
  <>
    <div className="flex-1 rounded-xl p-2.5 bg-white/70 border border-ink-900/5 min-w-0">
      <div className="text-[8.5px] tracking-widest uppercase text-ink-400 font-mono mb-1">{label}</div>
      <div className="text-[10.5px] text-ink-800 font-medium leading-tight">{n}</div>
    </div>
    {!last && <div className="flex items-center justify-center w-5 flex-shrink-0"><Icons.ArrowR size={12} className="text-ink-300" stroke={2}/></div>}
  </>
);

const IdeaCard = ({ idea, open, onToggle, highlight }) => {
  const tone = ideaTone[idea.tone];
  const IconC = tone.icon;
  const stages = ['Factor','Regime','Stress','RAG'];
  return (
    <div className={`rounded-4xl shadow-card lift overflow-hidden ${highlight?'text-white':'glass-strong'}`}
         style={highlight ? { background:'linear-gradient(155deg, #2a2825 0%, #1c1b1a 100%)' } : { borderLeft:`2px solid ${tone.border}` }}>
      <div className="p-6">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className={`w-11 h-11 rounded-2xl flex items-center justify-center ${highlight?'bg-gold-400 text-ink-900':'bg-ink-900/5 text-ink-900'}`}>
              <IconC size={18} stroke={1.7}/>
            </div>
            <div>
              <div className={`text-[10px] tracking-widest uppercase font-mono mb-0.5 ${highlight?'text-white/50':'text-ink-400'}`}>Идея {idea.n} · {idea.cat}</div>
              <div className={`flex items-center gap-1.5 ${highlight?'text-gold-400':'text-ink-600'}`}>
                <span className="w-1.5 h-1.5 rounded-full bg-current"/>
                <span className="text-[11px] font-semibold tracking-wide">{idea.prio}</span>
              </div>
            </div>
          </div>
          <button onClick={onToggle}
            className={`w-9 h-9 rounded-full flex items-center justify-center transition ${highlight?'bg-white/10 text-white hover:bg-white/20':'bg-ink-900/5 text-ink-700 hover:bg-ink-900/10'} ${open?'rotate-180':''}`}>
            <Icons.Chevron size={15}/>
          </button>
        </div>

        <h3 className={`mt-5 text-[20px] leading-[1.15] tracking-tight font-medium ${highlight?'text-white':'text-ink-900'}`}>{idea.title}</h3>
        <p className={`mt-2.5 text-[13px] leading-relaxed font-light ${highlight?'text-white/70':'text-ink-500'}`}>{idea.lede}</p>

        <div className="mt-5">
          <div className={`text-[10px] tracking-widest uppercase font-mono mb-2 ${highlight?'text-white/40':'text-ink-400'}`}>
            {idea.cands.length} кандидат(а) — нет в портфеле
          </div>
          <div className="flex flex-wrap gap-2">
            {idea.cands.map(c => (
              <span key={c.t} className={`px-3 py-1.5 rounded-xl text-[12px] font-bold num ${highlight?'bg-white/8 text-white':'bg-cream-50 border border-ink-900/8 text-ink-900'}`}>{c.t}</span>
            ))}
          </div>
        </div>
      </div>

      <div className="overflow-hidden transition-[max-height,opacity] duration-500 ease-out" style={{ maxHeight: open?900:0, opacity: open?1:0 }}>
        <div className={`px-6 pb-6 pt-2 space-y-5 ${highlight?'border-t border-white/10':'border-t border-ink-900/6'}`}>
          <div className="pt-4">
            <div className={`text-[10px] tracking-widest uppercase font-mono mb-2.5 ${highlight?'text-white/40':'text-ink-400'}`}>Конвейер · Factor → Regime → Stress → RAG</div>
            <div className={`flex items-stretch ${highlight ? '[&_div.rounded-xl]:!bg-white/8 [&_div.rounded-xl]:!border-white/10 [&_.text-ink-800]:!text-white [&_.text-ink-400]:!text-white/40 [&_svg]:!text-white/40' : ''}`}>
              {idea.pipeline.map((s,i) => <PipeNode key={i} n={s} label={stages[i]} last={i===idea.pipeline.length-1}/>)}
            </div>
          </div>
          <div>
            <div className={`text-[10px] tracking-widest uppercase font-mono mb-2.5 ${highlight?'text-white/40':'text-ink-400'}`}>Почему именно эти бумаги</div>
            <div className="space-y-2.5">
              {idea.cands.map(c => (
                <div key={c.t} className={`rounded-2xl p-3.5 ${highlight?'bg-white/8':'bg-cream-50 border border-ink-900/5'}`}>
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-[13px] font-bold num ${highlight?'text-white':'text-ink-900'}`}>{c.t}</span>
                    <span className={`text-[11px] ${highlight?'text-white/50':'text-ink-500'}`}>{c.name}</span>
                    <span className={`ml-auto text-[9px] font-mono px-1.5 py-0.5 rounded ${highlight?'bg-white/10 text-white/60':'bg-ink-900/5 text-ink-600'}`}>{c.src}</span>
                  </div>
                  <p className={`text-[11.5px] leading-snug font-light ${highlight?'text-white/65':'text-ink-600'}`}>{c.why}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const Plan = () => {
  const p = window.DEEP;
  const [open, setOpen] = React.useState({ '01': true });
  const toggle = (n) => setOpen(o => ({ ...o, [n]: !o[n] }));
  return (
    <section id="plan" className="rise" data-screen-label="05 Action Plan">
      <div className="mb-6">
        <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2">
          <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/> Action Plan · Effect · AI Ideas · DEEP
        </div>
        <h2 className="text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900">
          От идей к конкретным уровням<span className="text-ink-400">.</span>
        </h2>
        <p className="text-[15px] text-ink-500 mt-2 font-light max-w-[680px]">
          Конкретные уровни Buy / Sell / Stop, оценка эффекта до/после и стратегические идеи с кандидатами.
        </p>
      </div>

      <div className="space-y-5">
        <ActionPlan rows={p.actionPlan}/>
        <EffectGrid rows={p.effect} verdict={p.effectVerdict}/>

        <div>
          <div className="rounded-4xl p-6 mb-5 relative overflow-hidden" style={{ background:'linear-gradient(120deg, #fbf3d9 0%, #f6ebc0 100%)' }}>
            <div className="absolute -right-6 top-1/2 -translate-y-1/2 w-40 h-40 rounded-full opacity-30" style={{ background:'radial-gradient(circle, #caa01a, transparent 65%)' }}/>
            <div className="relative flex items-start gap-4">
              <div className="w-11 h-11 rounded-2xl bg-ink-900 text-gold-400 flex items-center justify-center flex-shrink-0"><Icons.Sparkles size={18} stroke={1.7}/></div>
              <div>
                <div className="text-[10px] tracking-widest uppercase font-mono text-ink-700 mb-1">AI Ideas · 4 идеи · каждая прошла Factor → Regime → Stress → RAG</div>
                <p className="text-[14.5px] text-ink-900 leading-relaxed font-light">
                  Тикеры-кандидаты <span className="font-medium">не из вашего портфеля</span> — рассмотрите как замену или дополнение. Раскройте карточку, чтобы увидеть конвейер отбора и обоснование по каждому кандидату.
                </p>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {p.ideas.map((idea,i) => (
              <IdeaCard key={idea.n} idea={idea} open={!!open[idea.n]} onToggle={()=>toggle(idea.n)} highlight={i===0}/>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

Object.assign(window, { Plan });
