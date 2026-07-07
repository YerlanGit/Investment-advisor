/* App shell — sticky topbar with section pills, mounted sections, footer */

const NAV = [
  { id:'overview',    label:'Обзор',     short:'01' },
  { id:'holdings',    label:'Бумаги',    short:'02' },
  { id:'performance', label:'Доходность',short:'03' },
  { id:'ideas',       label:'Идеи ИИ',   short:'04' },
];

const TopBar = ({ active, setActive }) => {
  const meta = window.PORTFOLIO.meta;
  const onJump = (id) => {
    setActive(id);
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  return (
    <header className="sticky top-0 z-40 px-6 pt-5 pb-3 backdrop-blur-md"
            style={{ background:'linear-gradient(to bottom, rgba(251,248,241,0.85) 0%, rgba(251,248,241,0.65) 70%, transparent 100%)' }}>
      <div className="max-w-[1480px] mx-auto flex items-center justify-between gap-4">
        {/* Logo + tier */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/80 border border-ink-900/8 shadow-sm">
            <div className="w-5 h-5 rounded-md bg-ink-900 flex items-center justify-center">
              <div className="w-2 h-2 rounded-sm bg-gold-400"/>
            </div>
            <span className="font-bold tracking-tight text-[14px] text-ink-900">BASE</span>
            <span className="px-1.5 py-0.5 rounded-md bg-gold-400/30 text-gold-700 text-[9px] font-mono font-bold tracking-wider">TIER</span>
          </div>
        </div>

        {/* Section nav pills */}
        <nav className="hidden md:flex items-center gap-1 p-1 rounded-full bg-white/70 border border-ink-900/8 backdrop-blur-md">
          {NAV.map(n => (
            <button key={n.id} onClick={()=>onJump(n.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-full text-[13px] font-medium transition-colors
                          ${active===n.id?'bg-ink-900 text-white':'text-ink-700 hover:bg-ink-900/5'}`}>
              <span className={`text-[10px] font-mono opacity-60 ${active===n.id?'text-gold-400':''}`}>{n.short}</span>
              {n.label}
            </button>
          ))}
        </nav>

        {/* Right tools */}
        <div className="flex items-center gap-2">
          <button className="hidden sm:flex items-center gap-2 pl-3 pr-4 py-2 rounded-full bg-white/70 border border-ink-900/8 text-[12px] text-ink-700 hover:bg-white transition">
            <Icons.Search size={14} stroke={1.8}/> <span className="hidden lg:inline">Найти бумагу…</span>
          </button>
          <button className="w-9 h-9 rounded-full bg-white/70 border border-ink-900/8 flex items-center justify-center text-ink-700 hover:bg-white transition relative">
            <Icons.Bell size={15} stroke={1.8}/>
            <span className="absolute top-2 right-2 w-1.5 h-1.5 rounded-full bg-rust-500"/>
          </button>
          <button className="w-9 h-9 rounded-full bg-white/70 border border-ink-900/8 flex items-center justify-center text-ink-700 hover:bg-white transition">
            <Icons.Settings size={15} stroke={1.7}/>
          </button>
          <div className="w-9 h-9 rounded-full overflow-hidden border border-ink-900/8 flex-shrink-0"
               style={{ background:'linear-gradient(135deg, #f5d04e 0%, #caa01a 100%)' }}>
            <div className="w-full h-full flex items-center justify-center text-ink-900 text-[11px] font-bold">YК</div>
          </div>
        </div>
      </div>

      {/* Meta strip */}
      <div className="max-w-[1480px] mx-auto mt-3 flex items-center justify-between text-[10px] font-mono tracking-wider text-ink-400 uppercase gap-3 flex-wrap">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-sage-500 animate-pulse"/> Risk Engine · {meta.engine}
          </span>
          <span className="opacity-30">/</span>
          <span>ID · {meta.id}</span>
          <span className="opacity-30">/</span>
          <span>Session · {meta.session}</span>
        </div>
        <span>Generated {meta.generated}</span>
      </div>
    </header>
  );
};

const Footer = () => {
  const p = window.PORTFOLIO;
  // Паритет с DEEP (RAG-наблюдаемость): банк-источники заявляем ТОЛЬКО когда
  // RAG реально консультировался (quality несёт «✓ … RAG …»).  Прежде футер
  // BASE статично печатал GS/MS/JPM — вводило в заблуждение при пустом RAG.
  const quality = p.quality || [];
  const ragOn = quality.some(q => typeof q === 'string' && q.includes('✓') && /rag/i.test(q));
  const sources = ['SEC EDGAR', 'Tradernet', 'FRED', 'Quant Engine MAC3']
    .concat(ragOn ? ['ChromaDB (GS / MS / JPM)'] : [])
    .concat([p.meta && p.meta.aiModel].filter(Boolean)).join(' · ');
  return (
  <footer className="mt-16 mb-8">
    <div className="rounded-4xl p-7 glass-strong shadow-card">
      <div className="flex items-start justify-between gap-6 flex-wrap">
        <div className="max-w-[560px]">
          <div className="text-[10px] tracking-widest uppercase text-ink-400 font-mono mb-3">Контроль качества данных</div>
          <div className="flex flex-wrap gap-2">
            {quality.map((q,i) => {
              // Каждая строка уже несёт СВОЙ статус-символ («✓ …»/«✗ …»/«⚠ …»/«— …»);
              // красим именно его — честно для любого состояния (в т.ч. RAG 3-state).
              const s = String(q).trim();
              const sym = s.charAt(0);
              const rest = s.slice(1).trim();
              const cls = sym === '✗' ? 'text-rust-600'
                        : sym === '⚠' ? 'text-gold-700'
                        : sym === '—' ? 'text-ink-400'
                        : 'text-sage-600';
              return (
                <span key={i} className="flex items-center gap-1.5 text-[10px] font-mono text-ink-600 bg-cream-50 border border-ink-900/6 rounded-full px-2.5 py-1">
                  <span className={`${cls} font-bold`}>{sym}</span> {rest}
                </span>
              );
            })}
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
          <span className="text-ink-700 font-medium">Это аналитический материал, а не индивидуальная инвестиционная рекомендация.</span> Расчёты основаны на исторических данных и публичной отчётности. Источники: {sources}.
        </p>
      </div>
    </div>
    <div className="text-center text-[10px] tracking-widest uppercase text-ink-400 font-mono mt-6">
      Portfolio Risk Report · BASE Tier · {p.meta.id} · v2026.7
    </div>
  </footer>
  );
};

// Floating action button (echoes reference pattern — bottom right luxury chip)
const Fab = () => (
  <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-2">
    <button title="Пересчитать риск"
            className="w-14 h-14 rounded-full bg-ink-900 text-gold-400 shadow-card-lg flex items-center justify-center hover:scale-105 transition">
      <Icons.Sparkles size={20} stroke={1.7}/>
    </button>
  </div>
);

// IntersectionObserver-driven active-section state
const useActiveSection = () => {
  const [active, setActive] = React.useState('overview');
  React.useEffect(() => {
    const ids = NAV.map(n => n.id);
    const els = ids.map(id => document.getElementById(id)).filter(Boolean);
    const obs = new IntersectionObserver((entries) => {
      // pick the entry closest to top among intersecting ones
      const visible = entries.filter(e => e.isIntersecting)
                              .sort((a,b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible[0]) setActive(visible[0].target.id);
    }, { rootMargin: '-110px 0px -55% 0px', threshold: 0 });
    els.forEach(el => obs.observe(el));
    return () => obs.disconnect();
  }, []);
  return [active, setActive];
};

const App = () => {
  const [active, setActive] = useActiveSection();

  // reveal after mount (prevents FOUC)
  React.useEffect(() => {
    document.getElementById('root').classList.remove('preload');
  }, []);

  return (
    <div className="min-h-screen">
      <TopBar active={active} setActive={setActive}/>
      <main className="max-w-[1480px] mx-auto px-6 pt-8 pb-12 space-y-20">
        <Hero/>
        <Holdings/>
        <Performance/>
        <Ideas/>
        <Footer/>
      </main>
      <Fab/>
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
