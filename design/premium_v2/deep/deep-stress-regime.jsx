/* DEEP Stress scenarios + Market regime */

const StressTable = ({ rows }) => (
  <div className="glass-strong rounded-4xl p-7 shadow-card">
    <div className="flex items-start justify-between gap-4 flex-wrap mb-5">
      <div>
        <h3 className="text-2xl font-semibold tracking-tight text-ink-900">Стресс-сценарии</h3>
        <p className="text-[12px] text-ink-500 font-mono mt-1">Параметрические шоки факторов (ΔPnL = w′·B·shock) · горизонт 1 квартал</p>
      </div>
      <span className="text-[10px] font-mono text-ink-400 tracking-wider px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5">7 сценариев · не прогноз</span>
    </div>
    <div className="swipe-hint items-center gap-1 text-[10px] font-mono text-gold-700 bg-gold-400/15 rounded-full px-2.5 py-1 mb-2.5 w-max">↔ листайте таблицу</div>
    <div className="mob-scroll-x"><div>
    <div className="grid grid-cols-[minmax(0,2.1fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.4fr)_minmax(0,0.9fr)_minmax(0,1fr)] gap-3 px-1 pb-2.5 text-[9.5px] tracking-widest uppercase text-ink-400 font-mono border-b border-ink-900/8">
      <div>Сценарий</div><div className="text-right">Δ Портфель</div><div className="text-right">Δ Стоимость</div><div>Магнитуда</div><div className="text-right">Drawdown</div><div className="text-right">Восстановл.</div>
    </div>
    <div className="divide-y divide-ink-900/5">
      {rows.map((r,i) => {
        const pos = r.pct >= 0;
        return (
          <div key={i} className="grid grid-cols-[minmax(0,2.1fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.4fr)_minmax(0,0.9fr)_minmax(0,1fr)] gap-3 items-center px-1 py-3">
            <div className="text-[12.5px] text-ink-800">{r.name}</div>
            <div className={`text-right text-[13px] num font-semibold ${pos?'text-sage-600':'text-rust-600'}`}>{pos?'+':''}{r.pct.toFixed(1)}%</div>
            <div className={`text-right text-[12px] num ${pos?'text-sage-500':'text-rust-500'}`}>{pos?'+':'−'}${Math.abs(r.usd).toLocaleString('ru-RU')}</div>
            <div><MagnitudeBar value={r.pct}/></div>
            <div className={`text-right text-[12px] num ${r.dd==null?'text-ink-300':'text-rust-600'}`}>{r.dd==null?'—':`${r.dd.toFixed(1)}%`}</div>
            <div className="text-right text-[11.5px] num text-ink-500">{r.rec}</div>
          </div>
        );
      })}
    </div>
    </div></div>
    <div className="mt-4 rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3.5 flex items-start gap-3">
      <Icons.Sparkles size={14} className="text-gold-600 mt-0.5 flex-shrink-0" stroke={1.8}/>
      <p className="text-[12.5px] text-ink-700 leading-relaxed font-light">{window.DEEP.stressAI}</p>
    </div>
  </div>
);

const RegimeBlock = ({ r }) => (
  <div className="grid grid-cols-12 gap-5">
    {/* quadrant */}
    <div className="col-span-12 lg:col-span-5">
      <div className="glass-strong rounded-4xl p-6 shadow-card lift h-full flex flex-col">
        <div className="flex items-start justify-between mb-2">
          <div>
            <div className="text-ink-500 text-[12px] font-medium">Growth × Cycle</div>
            <h3 className="text-xl font-semibold tracking-tight text-ink-900">Координаты режима</h3>
          </div>
          <span className="text-[10px] font-mono text-ink-400 tracking-wider px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5">60-day</span>
        </div>
        <div className="flex-1 flex items-center justify-center py-2">
          <RegimeQuadrant dot={r.dot} size={300}/>
        </div>
      </div>
    </div>
    {/* summary */}
    <div className="col-span-12 lg:col-span-7">
      <div className="glass-strong rounded-4xl p-6 shadow-card lift h-full flex flex-col">
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="text-[10px] tracking-widest uppercase text-ink-500 font-mono mb-1">Текущий режим</div>
            <div className="flex items-baseline gap-3">
              <span className="text-[32px] font-light tracking-tight text-ink-900">{r.name}</span>
              <span className="text-[14px] text-ink-500 font-light">· {r.nameRu}</span>
            </div>
            <div className="text-[11.5px] text-ink-500 font-mono mt-1">Уверенность модели <b className="text-gold-700">{r.confidence}%</b> · {r.confirms} подтверждающих сигнала</div>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-sage-500/12 text-sage-600 text-[11px] font-semibold">
            <Icons.Check size={13} stroke={2.2}/> Рост
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3">
            <div className="text-[10px] uppercase tracking-wider text-ink-500 font-mono">Growth-фактор</div>
            <div className="text-[20px] num font-semibold text-sage-600 mt-0.5">+{r.growth.toFixed(2)}</div>
            <div className="text-[10px] text-ink-400">здоровый рост</div>
          </div>
          <div className="rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3">
            <div className="text-[10px] uppercase tracking-wider text-ink-500 font-mono">Cycle-фактор</div>
            <div className="text-[20px] num font-semibold text-sage-600 mt-0.5">+{r.cycle.toFixed(2)}</div>
            <div className="text-[10px] text-ink-400">цикл. экспансия</div>
          </div>
        </div>

        {/* macro drivers */}
        <div className="text-[10px] tracking-widest uppercase text-ink-400 font-mono mb-2">Сигналы-драйверы · as_of 2026-06-22</div>
        <div className="space-y-1.5 flex-1">
          {r.drivers.map((d,i) => (
            <div key={i} className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-3 py-1.5 border-b border-ink-900/5 last:border-0">
              <span className="text-[11.5px] text-ink-700 truncate">{d.name}</span>
              <span className={`text-[11.5px] num font-semibold ${d.tone==='warn'?'text-gold-700':'text-sage-600'}`}>{d.val}</span>
              <span className="text-[9.5px] font-mono text-ink-400 whitespace-nowrap">{d.trend}</span>
              <span className={`text-[8.5px] font-mono font-bold tracking-wider px-1.5 py-0.5 rounded-full ${d.tone==='warn'?'bg-gold-400/18 text-gold-700':'bg-sage-500/12 text-sage-600'}`}>{d.state}</span>
            </div>
          ))}
        </div>
      </div>
    </div>

    {/* RAG signals + confirm */}
    <div className="col-span-12">
      <div className="rounded-4xl p-6 shadow-card"
           style={{ background:'linear-gradient(120deg, #f3f6f1 0%, #eef3ea 100%)' }}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 text-sage-600 text-[11px] font-semibold">
            <Icons.Check size={14} stroke={2.2}/> ИИ подтверждает режим
          </div>
          <span className="text-[10px] font-mono text-ink-400">{window.DEEP.meta.aiModel}</span>
        </div>
        <p className="text-[13.5px] text-ink-800 leading-relaxed font-light mb-4">{r.confirm}</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2">
          {r.confirmBullets.map((b,i) => (
            <div key={i} className="flex items-start gap-2.5">
              {b.ok
                ? <Icons.Check size={13} className="text-sage-600 mt-0.5 flex-shrink-0" stroke={2.4}/>
                : <Icons.Warning size={13} className="text-gold-700 mt-0.5 flex-shrink-0" stroke={2}/>}
              <span className="text-[11.5px] text-ink-700 leading-snug">{b.t}</span>
            </div>
          ))}
        </div>
        <div className="mt-4 pt-3 border-t border-ink-900/8 flex flex-wrap gap-2">
          {r.ragSignals.map((s,i) => (
            <span key={i} className="text-[10.5px] text-ink-600 bg-white/60 border border-ink-900/6 rounded-full px-3 py-1 font-mono">RAG · {s}</span>
          ))}
        </div>
      </div>
    </div>
  </div>
);

const StressRegime = () => {
  const p = window.DEEP;
  return (
    <section id="stress" className="rise" data-screen-label="04 Stress & Regime">
      <div className="mb-6">
        <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2">
          <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/> Stress test · Market regime · DEEP
        </div>
        <h2 className="text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900">
          Устойчивость и контекст<span className="text-ink-400">.</span>
        </h2>
        <p className="text-[15px] text-ink-500 mt-2 font-light max-w-[680px]">
          Как портфель ведёт себя при гипотетических шоках и в какой фазе рынка мы находимся.
        </p>
      </div>

      <div className="mb-5"><StressTable rows={p.stress}/></div>
      <RegimeBlock r={p.regime}/>

      <div className="mt-5 rounded-3xl p-5 glass-strong shadow-card flex items-start gap-4">
        <div className="w-10 h-10 rounded-2xl bg-ink-900 text-gold-400 flex items-center justify-center flex-shrink-0">
          <Icons.Sparkles size={17} stroke={1.7}/>
        </div>
        <div>
          <div className="text-[10px] tracking-widest uppercase font-mono text-ink-500 mb-1">AI · комментарий к режиму</div>
          <p className="text-[14px] text-ink-800 leading-relaxed font-light">{p.regime.regimeAI}</p>
        </div>
      </div>
    </section>
  );
};

Object.assign(window, { StressRegime });
