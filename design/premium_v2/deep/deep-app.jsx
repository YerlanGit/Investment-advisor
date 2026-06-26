/* DEEP app shell — topbar, nav, sections, footer */

const NAV = [
  { id:'overview', label:'Обзор',   short:'01' },
  { id:'holdings', label:'Бумаги',  short:'02' },
  { id:'factors',  label:'Факторы', short:'03' },
  { id:'stress',   label:'Стресс · Режим', short:'04' },
  { id:'plan',     label:'План',    short:'05' },
  { id:'cove',     label:'CoVe',    short:'06' },
];

const TopBar = ({ active }) => {
  const meta = window.DEEP.meta;
  const onJump = (id) => {
    const el = document.getElementById(id);
    if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 96, behavior:'smooth' });
  };
  return (
    <header className="sticky top-0 z-40 px-6 pt-5 pb-3 backdrop-blur-md"
            style={{ background:'linear-gradient(to bottom, rgba(251,248,241,0.88) 0%, rgba(251,248,241,0.65) 70%, transparent 100%)' }}>
      <div className="max-w-[1480px] mx-auto flex items-center justify-between gap-4">
        <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/80 border border-ink-900/8 shadow-sm flex-shrink-0">
          <div className="w-5 h-5 rounded-md bg-ink-900 flex items-center justify-center">
            <div className="w-2 h-2 rounded-sm bg-gold-400"/>
          </div>
          <span className="font-bold tracking-tight text-[14px] text-ink-900">DEEP</span>
          <span className="px-1.5 py-0.5 rounded-md bg-gold-400/30 text-gold-700 text-[9px] font-mono font-bold tracking-wider">TIER</span>
        </div>

        <nav className="hidden lg:flex items-center gap-1 p-1 rounded-full bg-white/70 border border-ink-900/8 backdrop-blur-md">
          {NAV.map(n => (
            <button key={n.id} onClick={()=>onJump(n.id)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-full text-[12.5px] font-medium transition-colors
                          ${active===n.id?'bg-ink-900 text-white':'text-ink-700 hover:bg-ink-900/5'}`}>
              <span className={`text-[10px] font-mono opacity-60 ${active===n.id?'text-gold-400':''}`}>{n.short}</span>
              {n.label}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-2 flex-shrink-0">
          <button className="hidden sm:flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-white/70 border border-ink-900/8 text-[12px] text-ink-700 hover:bg-white transition">
            <Icons.Download size={13} stroke={1.8}/> <span className="hidden xl:inline">PDF</span>
          </button>
          <div className="w-9 h-9 rounded-full overflow-hidden border border-ink-900/8 flex-shrink-0"
               style={{ background:'linear-gradient(135deg, #f5d04e 0%, #caa01a 100%)' }}>
            <div className="w-full h-full flex items-center justify-center text-ink-900 text-[11px] font-bold">YК</div>
          </div>
        </div>
      </div>

      <div className="max-w-[1480px] mx-auto mt-3 flex items-center justify-between text-[10px] font-mono tracking-wider text-ink-400 uppercase gap-3 flex-wrap">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-sage-500 animate-pulse"/> Risk Engine · {meta.engine}</span>
          <span className="opacity-30">/</span>
          <span>ID · {meta.id}</span>
          <span className="opacity-30">/</span>
          <span>AI · {meta.aiModel}</span>
          <span className="opacity-30 hidden sm:inline">/</span>
          <span className="hidden sm:inline">Профиль · {meta.profile}</span>
        </div>
        <span>Generated {meta.generated}</span>
      </div>
    </header>
  );
};

const Footer = () => {
  const p = window.DEEP;
  return (
    <footer className="mt-16 mb-8">
      <div className="rounded-4xl p-7 glass-strong shadow-card">
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div className="max-w-[560px]">
            <div className="text-[10px] tracking-widest uppercase text-ink-400 font-mono mb-3">Контроль качества данных</div>
            <div className="flex flex-wrap gap-2">
              {p.quality.map((q,i) => (
                <span key={i} className="flex items-center gap-1.5 text-[10px] font-mono text-ink-600 bg-cream-50 border border-ink-900/6 rounded-full px-2.5 py-1">
                  <span className="text-sage-600 font-bold">✓</span> {q}
                </span>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <button className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-white border border-ink-900/8 text-ink-700 text-[12px] font-medium hover:bg-cream-50 transition">
              <Icons.Download size={13} stroke={1.8}/> Скачать PDF
            </button>
            <button className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-white border border-ink-900/8 text-ink-700 text-[12px] font-medium hover:bg-cream-50 transition">
              <Icons.Share size={13} stroke={1.8}/> Поделиться
            </button>
            <button className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-ink-900 text-white text-[12px] font-medium hover:bg-ink-800 transition">
              <Icons.Sparkles size={13} stroke={1.8}/> Новый расчёт
            </button>
          </div>
        </div>
        <div className="mt-6 pt-5 border-t border-ink-900/8 flex items-start gap-3">
          <Icons.Warning size={15} className="text-rust-500 mt-0.5 flex-shrink-0" stroke={1.8}/>
          <p className="text-[12px] text-ink-500 leading-relaxed font-light">
            <span className="text-ink-700 font-medium">Это аналитический материал, а не индивидуальная инвестиционная рекомендация.</span> Расчёты основаны на исторических данных и публичной отчётности; они не учитывают вашу налоговую ситуацию, горизонт и цели. Источники: Tradernet · SEC EDGAR · FRED · Quant Engine MAC3 · ChromaDB (GS / MS / JPM) · {p.meta.aiModel}.
          </p>
        </div>
      </div>
      <div className="text-center text-[10px] tracking-widest uppercase text-ink-400 font-mono mt-6">
        Portfolio Risk Report · DEEP Tier · {p.meta.id} · v2026.6
      </div>
    </footer>
  );
};

const Fab = () => (
  <div className="fixed bottom-6 right-6 z-50">
    <button title="Пересчитать риск" onClick={()=>window.scrollTo({top:0,behavior:'smooth'})}
            className="w-14 h-14 rounded-full bg-ink-900 text-gold-400 shadow-card-lg flex items-center justify-center hover:scale-105 transition">
      <Icons.Sparkles size={20} stroke={1.7}/>
    </button>
  </div>
);

const useActiveSection = () => {
  const [active, setActive] = React.useState('overview');
  React.useEffect(() => {
    const els = NAV.map(n => document.getElementById(n.id)).filter(Boolean);
    const obs = new IntersectionObserver((entries) => {
      const visible = entries.filter(e => e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);
      if (visible[0]) setActive(visible[0].target.id);
    }, { rootMargin:'-110px 0px -55% 0px', threshold:0 });
    els.forEach(el => obs.observe(el));
    return () => obs.disconnect();
  }, []);
  return active;
};

const App = () => {
  const active = useActiveSection();
  React.useEffect(() => { document.getElementById('root').classList.remove('preload'); }, []);
  return (
    <div className="min-h-screen">
      <TopBar active={active}/>
      <main className="max-w-[1480px] mx-auto px-6 pt-8 pb-12 space-y-20">
        <Overview/>
        <Holdings/>
        <Factors/>
        <StressRegime/>
        <Plan/>
        <Cove/>
        <Footer/>
      </main>
      <Fab/>
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
