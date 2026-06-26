/* DEEP CoVe — chain-of-verification provenance appendix */

const CoveItem = ({ c }) => {
  const cfg = {
    ok:   { mark:'✓', cls:'text-sage-600', bg:'bg-sage-500/12' },
    warn: { mark:'!', cls:'text-gold-700', bg:'bg-gold-400/15' },
    fail: { mark:'✗', cls:'text-rust-600', bg:'bg-rust-500/12' },
  }[c.st];
  return (
    <div className="flex items-start gap-3 py-2.5 border-b border-ink-900/5">
      <span className={`w-5 h-5 rounded-md ${cfg.bg} ${cfg.cls} flex items-center justify-center text-[11px] font-bold flex-shrink-0 mt-0.5`}>{cfg.mark}</span>
      <div className="min-w-0">
        <div className="text-[12px] text-ink-900 font-medium leading-tight">{c.title}</div>
        <div className="text-[10px] text-ink-500 font-mono leading-snug mt-0.5">{c.meta}</div>
      </div>
    </div>
  );
};

const Cove = () => {
  const p = window.DEEP;
  const half = Math.ceil(p.cove.length/2);
  const cols = [p.cove.slice(0,half), p.cove.slice(half)];
  const counts = p.cove.reduce((a,c)=>{ a[c.st]++; return a; }, { ok:0, warn:0, fail:0 });
  return (
    <section id="cove" className="rise" data-screen-label="06 CoVe">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-6">
        <div>
          <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2">
            <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/> Chain-of-Verification · DEEP
          </div>
          <h2 className="text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900">
            Откуда данные<span className="text-ink-400">.</span>
          </h2>
          <p className="text-[15px] text-ink-500 mt-2 font-light max-w-[640px]">
            Каждый показатель прослеживается до первичного источника с методом расчёта и статусом QualityGate.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-sage-500/12 text-sage-600 text-[11px] font-semibold"><span className="font-bold">✓</span> {counts.ok} прошли</span>
          <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-gold-400/15 text-gold-700 text-[11px] font-semibold"><span className="font-bold">!</span> {counts.warn} частично</span>
          <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-rust-500/12 text-rust-600 text-[11px] font-semibold"><span className="font-bold">✗</span> {counts.fail} недоступно</span>
        </div>
      </div>

      <div className="glass-strong rounded-4xl p-7 shadow-card">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-10">
          {cols.map((col,ci) => (
            <div key={ci}>{col.map((c,i) => <CoveItem key={i} c={c}/>)}</div>
          ))}
        </div>
        <div className="mt-5 pt-4 border-t border-ink-900/8 text-[11px] text-ink-500 leading-relaxed font-light">
          <b className="text-sage-600">✓</b> — данные прошли QualityGate · <b className="text-gold-700">!</b> — частичное покрытие или fallback на устаревший кэш · <b className="text-rust-600">✗</b> — источник недоступен.
          Документ не является ИИР (индивидуальной инвестиционной рекомендацией) — только аналитический материал.
        </div>
      </div>
    </section>
  );
};

Object.assign(window, { Cove });
