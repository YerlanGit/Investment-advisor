/* DEEP Factors — β-radar + 4-pillar scoring */

const FactorTable = ({ factors }) => (
  <table className="w-full">
    <thead>
      <tr className="text-[9.5px] tracking-widest uppercase text-ink-400 font-mono border-b border-ink-900/8">
        <th className="text-left font-medium py-2">Фактор</th>
        <th className="text-right font-medium py-2">β портф.</th>
        <th className="text-right font-medium py-2">Рынок</th>
        <th className="text-right font-medium py-2">Наклон Δ</th>
      </tr>
    </thead>
    <tbody>
      {factors.map((f,i) => {
        const d = f.port - f.mkt;
        const tone = d > 0 ? 'text-gold-700' : d < 0 ? 'text-sage-600' : 'text-ink-400';
        return (
          <tr key={i} className="border-b border-ink-900/5 last:border-0">
            <td className="text-left py-2 text-[12px] text-ink-800 font-medium">{f.name}</td>
            <td className="text-right py-2 text-[12px] num text-ink-900">{f.port.toFixed(2)}</td>
            <td className="text-right py-2 text-[12px] num text-ink-400">{f.mkt.toFixed(2)}</td>
            <td className={`text-right py-2 text-[12px] num font-semibold ${tone}`}>{d>0?'+':''}{d.toFixed(2)}</td>
          </tr>
        );
      })}
    </tbody>
  </table>
);

const pillarTone = (v) => v == null ? 'text-ink-300' : v > 0 ? 'text-sage-600' : v < 0 ? 'text-rust-600' : 'text-ink-400';

const ScoreCard = ({ s }) => {
  const totalTone = s.total > 0 ? 'text-sage-600' : s.total < 0 ? 'text-rust-600' : 'text-ink-500';
  const accent = { BUY:'#5d7c5c', HOLD:'#a8a293', TRIM:'#caa01a', SELL:'#c47358' }[s.action];
  const actionChip = {
    BUY:'bg-sage-500/15 text-sage-600', HOLD:'bg-ink-900/6 text-ink-600',
    TRIM:'bg-gold-500/18 text-gold-700', SELL:'bg-rust-500 text-white',
  }[s.action];
  const pillars = [['F',s.F],['V',s.V],['T',s.T],['C',s.C]];
  return (
    <div className="rounded-3xl bg-white/70 border border-ink-900/6 p-4 shadow-card lift" style={{ borderLeft:`2px solid ${accent}` }}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-[14px] font-bold num text-ink-900">{s.t}</span>
        <span className={`text-[12px] font-semibold num ${totalTone}`}>Итог {s.total>0?'+':''}{s.total.toFixed(1)}</span>
      </div>
      <div className="space-y-2">
        {pillars.map(([name,val]) => (
          <div key={name} className="grid grid-cols-[14px_1fr_30px] items-center gap-2.5">
            <span className="text-[10px] font-mono font-semibold text-ink-500">{name}</span>
            {val == null
              ? <div className="h-1.5 rounded-full bg-ink-900/4 flex items-center justify-center"><span className="text-[8px] text-ink-300 font-mono">н/п</span></div>
              : <ScorePillar value={val}/>}
            <span className={`text-[10.5px] num font-semibold text-right ${pillarTone(val)}`}>{val==null?'—':`${val>0?'+':''}${val.toFixed(1)}`}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 pt-2.5 border-t border-ink-900/6 flex items-center gap-2">
        <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold tracking-wider ${actionChip}`}>{s.action}</span>
        <span className="text-[10px] text-ink-500 leading-tight">{s.reason}</span>
      </div>
    </div>
  );
};

const Factors = () => {
  const p = window.DEEP;
  return (
    <section id="factors" className="rise" data-screen-label="03 Factors">
      <div className="mb-6">
        <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2">
          <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/> Factor decomposition · 4-Pillar · DEEP
        </div>
        <h2 className="text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900">
          Факторы и качество позиций<span className="text-ink-400">.</span>
        </h2>
        <p className="text-[15px] text-ink-500 mt-2 font-light max-w-[680px]">
          Скрытые концентрации по 10 факторам и оценка каждой бумаги по четырём столпам.
        </p>
      </div>

      {/* β-radar + table */}
      <div className="glass-strong rounded-4xl p-7 shadow-card mb-5">
        <div className="flex items-start justify-between gap-4 flex-wrap mb-4">
          <div>
            <h3 className="text-2xl font-semibold tracking-tight text-ink-900">Факторное разложение (β)</h3>
            <p className="text-[12px] text-ink-500 font-mono mt-1">Ridge β (α=0.001) · EWMA hl=63 ⊕ Ledoit-Wolf 70/30 · окно 60 дней · покрытие {p.factorCoverage}%</p>
          </div>
          <div className="flex items-center gap-4 text-[11px] text-ink-600">
            <span className="flex items-center gap-2"><span className="w-4 h-0 border-t-2 border-gold-600"/> Портфель</span>
            <span className="flex items-center gap-2"><span className="w-4 h-0 border-t-2 border-dashed border-ink-700"/> Рынок (S&P 500)</span>
          </div>
        </div>
        <div className="grid grid-cols-12 gap-7 items-center">
          <div className="col-span-12 lg:col-span-5 flex justify-center">
            <FactorRadar factors={p.factors} size={320}/>
          </div>
          <div className="col-span-12 lg:col-span-7">
            <FactorTable factors={p.factors}/>
            <p className="text-[11px] text-ink-500 leading-relaxed font-light mt-4">
              <span className="text-ink-800 font-medium">Что показывает радар:</span> насколько портфель завязан на глобальные факторы. Большая площадь — больше зависимости от рынка; совпадение направлений по нескольким факторам — скрытая общая ставка.
            </p>
          </div>
        </div>
        <div className="mt-5 rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3.5 flex items-start gap-3">
          <Icons.Sparkles size={14} className="text-gold-600 mt-0.5 flex-shrink-0" stroke={1.8}/>
          <p className="text-[12.5px] text-ink-700 leading-relaxed font-light">{p.factorAI}</p>
        </div>
      </div>

      {/* 4-pillar scoring */}
      <div className="glass-strong rounded-4xl p-7 shadow-card">
        <div className="flex items-start justify-between gap-4 flex-wrap mb-5">
          <div>
            <h3 className="text-2xl font-semibold tracking-tight text-ink-900">4-Pillar Scoring</h3>
            <p className="text-[12px] text-ink-500 mt-1 font-light">
              <b className="text-ink-800">F</b> Фундамент · <b className="text-ink-800">V</b> Оценка · <b className="text-ink-800">T</b> Техника · <b className="text-ink-800">C</b> Кредит — каждый столп −2…+2, итог ∈ [−6, +6]
            </p>
          </div>
          <span className="text-[10px] font-mono text-ink-400 tracking-wider px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5">Quant Engine + SEC EDGAR</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {p.scores.map(s => <ScoreCard key={s.t} s={s}/>)}
        </div>
        <div className="text-[10.5px] text-ink-400 font-mono mt-4">{p.scoresNote}</div>
        <div className="mt-4 rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3.5 flex items-start gap-3">
          <Icons.Sparkles size={14} className="text-gold-600 mt-0.5 flex-shrink-0" stroke={1.8}/>
          <p className="text-[12.5px] text-ink-700 leading-relaxed font-light">{p.scoresAI}</p>
        </div>
      </div>
    </section>
  );
};

Object.assign(window, { Factors });
