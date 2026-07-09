/* AI Ideas section — 4 actionable ideas with expandable details */

const catTone = (cat) => ({
  'Снижение риска':   { chip:'bg-rust-500/15 text-rust-600',  dot:'bg-rust-500',  icon: Icons.Shield },
  'Диверсификация':   { chip:'bg-sage-500/15 text-sage-600',  dot:'bg-sage-500',  icon: Icons.Layers },
  'Ребалансировка':   { chip:'bg-ink-900/8 text-ink-700',     dot:'bg-ink-900',   icon: Icons.Refresh },
  'Увеличение риска': { chip:'bg-gold-500/20 text-gold-700',  dot:'bg-gold-500',  icon: Icons.TrendUp },
}[cat] || { chip:'bg-ink-900/8 text-ink-700', dot:'bg-ink-900', icon: Icons.Sparkles });

const TickerCard = ({ ticker, why, dark }) => (
  <div className={`rounded-2xl p-3.5 border transition group cursor-pointer ${dark?'bg-white/8 border-white/10':'bg-cream-50 border-ink-900/5 hover:border-ink-900/15'}`}>
    <div className="flex items-center justify-between mb-1">
      <span className={`text-[14px] font-bold num tracking-tight ${dark?'text-white':'text-ink-900'}`}>{ticker}</span>
      <span className={`text-[10px] font-mono uppercase tracking-wider opacity-0 group-hover:opacity-100 transition ${dark?'text-white/40':'text-ink-400'}`}>view</span>
    </div>
    <p className={`text-[11.5px] leading-snug font-light ${dark?'text-white/70':'text-ink-500'}`}>{why}</p>
  </div>
);

const PipelineNode = ({ label, value, last }) => (
  <>
    <div className="flex-1 rounded-2xl p-3 bg-white/70 border border-ink-900/5">
      <div className="text-[9px] tracking-widest uppercase text-ink-400 font-mono mb-1">{label}</div>
      <div className="text-[12px] text-ink-900 font-medium leading-tight">{value}</div>
    </div>
    {!last && (
      <div className="flex items-center justify-center w-7 flex-shrink-0">
        <Icons.ArrowR size={14} className="text-ink-400" stroke={2}/>
      </div>
    )}
  </>
);

const IdeaCard = ({ idea, open, onToggle, isHighlight }) => {
  const tone = catTone(idea.cat);
  const IconC = tone.icon;
  return (
    <div className={`rounded-4xl shadow-card lift overflow-hidden
                     ${isHighlight
                        ? 'text-white'
                        : 'glass-strong'}`}
         style={isHighlight ? { background:'linear-gradient(155deg, #2a2825 0%, #1c1b1a 100%)' } : {}}>
      <div className="p-6">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className={`w-11 h-11 rounded-2xl flex items-center justify-center
                            ${isHighlight?'bg-gold-400 text-ink-900':'bg-ink-900/5 text-ink-900'}`}>
              <IconC size={18} stroke={1.7}/>
            </div>
            <div>
              <div className={`text-[10px] tracking-widest uppercase font-mono mb-0.5
                              ${isHighlight?'text-white/50':'text-ink-400'}`}>
                Идея {idea.n} · {idea.cat}
              </div>
              <div className={`flex items-center gap-1 ${isHighlight?'text-gold-400':'text-rust-600'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${tone.dot}`}/>
                <span className="text-[11px] font-semibold tracking-wide">{idea.prio}</span>
              </div>
            </div>
          </div>
          <button onClick={onToggle}
                  className={`w-9 h-9 rounded-full flex items-center justify-center transition
                              ${isHighlight
                                ? 'bg-white/10 text-white hover:bg-white/20'
                                : 'bg-ink-900/5 text-ink-700 hover:bg-ink-900/10'}
                              ${open?'rotate-180':''}`}>
            <Icons.Chevron size={15}/>
          </button>
        </div>

        <h3 className={`mt-5 text-[22px] leading-[1.15] tracking-tight font-medium
                        ${isHighlight?'text-white':'text-ink-900'}`}>
          {idea.title}
        </h3>
        <p className={`mt-3 text-[13.5px] leading-relaxed font-light
                        ${isHighlight?'text-white/70':'text-ink-500'}`}>
          {idea.lede}
        </p>

        {/* Ticker chips — always visible */}
        <div className="mt-5">
          <div className={`text-[10px] tracking-widest uppercase font-mono mb-2
                          ${isHighlight?'text-white/40':'text-ink-400'}`}>
            Идеи по замене — нет в портфеле
          </div>
          <div className="grid grid-cols-3 gap-2">
            {idea.tickers.map(t => (
              <div key={t.t}
                   className={`rounded-xl px-3 py-2 text-center transition
                              ${isHighlight
                                ? 'bg-white/8 hover:bg-white/15 text-white'
                                : 'bg-cream-50 border border-ink-900/5 hover:border-ink-900/15 text-ink-900'}`}>
                <div className="text-[13px] font-bold num">{t.t}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Expanded details */}
      <div className="overflow-hidden transition-[max-height,opacity] duration-500 ease-out"
           style={{ maxHeight: open?900:0, opacity: open?1:0 }}>
        <div className={`px-6 pb-6 pt-2 space-y-5
                         ${isHighlight?'border-t border-white/10':'border-t border-ink-900/6'}`}>
          {/* Pipeline */}
          <div className="pt-4">
            <div className={`text-[10px] tracking-widest uppercase font-mono mb-2.5
                            ${isHighlight?'text-white/40':'text-ink-400'}`}>
              Конвейер · Factor → Regime → RAG
            </div>
            <div className={`flex items-center ${isHighlight ? '[&_div.rounded-2xl]:!bg-white/8 [&_div.rounded-2xl]:!border-white/10 [&_.text-ink-900]:!text-white [&_.text-ink-400]:!text-white/40 [&_svg]:!text-white/40' : ''}`}>
              <PipelineNode label="Factor" value={idea.pipeline[0]}/>
              <PipelineNode label="Regime" value={idea.pipeline[1]}/>
              <PipelineNode label="RAG" value={idea.pipeline[2]} last/>
            </div>
          </div>

          {/* Tickers w/ reasons */}
          <div>
            <div className={`text-[10px] tracking-widest uppercase font-mono mb-2.5
                            ${isHighlight?'text-white/40':'text-ink-400'}`}>
              Почему именно эти бумаги
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
              {idea.tickers.map(t => <TickerCard key={t.t} ticker={t.t} why={t.why} dark={isHighlight}/>)}
            </div>
          </div>

          {/* Effect + sources — render ONLY when there's content (BASE ideas
              carry no effect/sources → was an empty «Ожидаемый эффект» box). */}
          {((idea.effect && idea.effect.length>0) || (idea.sources && idea.sources.length>0)) && (
          <div className="flex items-start justify-between gap-4 flex-wrap">
            {idea.effect && idea.effect.length>0 && (
            <div className={`rounded-2xl p-4 flex-1 min-w-[260px]
                            ${isHighlight?'bg-gold-400/10 border border-gold-400/30':'bg-gold-400/15 border border-gold-400/40'}`}>
              <div className={`text-[10px] tracking-widest uppercase font-mono mb-1.5
                              ${isHighlight?'text-gold-400':'text-gold-700'}`}>
                Ожидаемый эффект
              </div>
              <ul className="space-y-1">
                {idea.effect.map((e,i) => (
                  <li key={i} className={`text-[12.5px] font-medium leading-tight
                                          ${isHighlight?'text-white':'text-ink-900'}`}>
                    {e}
                  </li>
                ))}
              </ul>
            </div>)}
            {idea.sources && idea.sources.length>0 && (
            <div className="flex flex-wrap gap-1.5 items-start pt-1">
              {idea.sources.map(s => (
                <span key={s} className={`px-2.5 py-1 rounded-full text-[10px] font-mono tracking-wider
                                          ${isHighlight?'bg-white/8 text-white/60':'bg-ink-900/5 text-ink-700'}`}>
                  {s}
                </span>
              ))}
            </div>)}
          </div>)}
        </div>
      </div>
    </div>
  );
};

// «Применить идею» flow.  The report is a STATIC page — it CANNOT charge a
// token itself.  So the button opens a two-step modal (pick 1 of 4 ideas →
// confirm «Да/Нет»); «Да» deep-links to the Telegram bot, which runs the
// Scenario-tier analysis and charges the 1 token there (t.me/<bot>?start=scn_N).
const scenarioDeepLink = (bot, n) =>
  `https://t.me/${encodeURIComponent(String(bot || 'KEN_investment_bot').replace(/^@/, ''))}`
  + `?start=scn_${String(n).replace(/[^0-9A-Za-z_]/g, '')}`;

const ApplyIdeaModal = ({ ideas, botUsername, onClose }) => {
  const [sel, setSel] = React.useState(null);
  const go = () => {
    if (!sel) return;
    window.open(scenarioDeepLink(botUsername, sel.n), '_blank', 'noopener,noreferrer');
    onClose();
  };
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
         style={{ background:'rgba(28,27,26,0.55)', backdropFilter:'blur(4px)' }}
         onClick={onClose}>
      <div className="w-full max-w-lg rounded-4xl bg-cream-50 shadow-card overflow-hidden"
           onClick={(e)=>e.stopPropagation()}>
        <div className="px-6 pt-6 pb-4 border-b border-ink-900/8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono">
              <Icons.Sparkles size={13} stroke={1.8} className="text-gold-600"/> Применить идею
            </div>
            <button onClick={onClose} aria-label="Закрыть"
                    className="w-8 h-8 rounded-full bg-ink-900/5 text-ink-700 hover:bg-ink-900/10 flex items-center justify-center text-[13px]">✕</button>
          </div>
          <h3 className="text-[22px] font-light text-ink-900 mt-3 tracking-tight">
            {sel ? 'Подтвердите запуск' : 'Выберите идею для сценарного анализа'}
          </h3>
        </div>

        {!sel ? (
          <div className="p-4 space-y-2 max-h-[60vh] overflow-y-auto">
            {ideas.map(idea => (
              <button key={idea.n} onClick={()=>setSel(idea)}
                className="w-full text-left rounded-2xl px-4 py-3 bg-white/70 border border-ink-900/6 hover:border-ink-900/20 transition flex items-start gap-3">
                <span className="text-[11px] font-mono text-ink-400 mt-1">{idea.n}</span>
                <span className="min-w-0">
                  <span className="block text-[10px] tracking-widest uppercase text-ink-400 font-mono">{idea.cat}</span>
                  <span className="block text-[15px] text-ink-900 font-medium leading-tight mt-0.5">{idea.title}</span>
                </span>
              </button>
            ))}
          </div>
        ) : (
          <div className="p-6">
            <div className="rounded-2xl p-4 bg-gold-400/15 border border-gold-400/40">
              <div className="text-[10px] tracking-widest uppercase font-mono text-gold-700 mb-1.5">Сценарный анализ</div>
              <p className="text-[14px] text-ink-900 leading-relaxed">
                Сделать <span className="font-semibold">Scenario Analysis</span> для идеи «{sel.title}» — с вас спишется <span className="font-semibold">1 токен</span>.
              </p>
            </div>
            <p className="text-[12px] text-ink-500 mt-3 leading-relaxed font-light">
              Откроется бот RAMP в Telegram и запустит сценарный анализ вашего портфеля. Токен списывается только после готового отчёта.
            </p>
            <div className="flex items-center gap-3 mt-5">
              <button onClick={go}
                className="flex-1 px-4 py-2.5 rounded-full bg-ink-900 text-white text-[13px] font-semibold hover:bg-ink-800 transition">
                Да
              </button>
              <button onClick={()=>setSel(null)}
                className="flex-1 px-4 py-2.5 rounded-full bg-white/70 border border-ink-900/10 text-ink-700 text-[13px] font-semibold hover:bg-white transition">
                Нет
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const Ideas = () => {
  const ideas = window.PORTFOLIO.ideas;
  const [open, setOpen] = React.useState({ '01': true });
  const toggle = (n) => setOpen({ ...open, [n]: !open[n] });
  const [applyOpen, setApplyOpen] = React.useState(false);

  return (
    <section id="ideas" className="rise" data-screen-label="04 Ideas">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-6">
        <div>
          <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2">
            <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/> AI Ideas · 4 идеи
          </div>
          <h2 className="text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900">
            Идеи на основе данных<span className="text-ink-400">.</span>
          </h2>
          <p className="text-[15px] text-ink-500 mt-2 font-light max-w-[640px]">
            На основе портфеля, отчётности SEC и обзоров инвестбанков. Нажмите на карточку, чтобы раскрыть детали.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button className="flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-white/60 border border-ink-900/8 text-ink-700 text-[12px] font-medium hover:bg-white transition">
            <Icons.Download size={13} stroke={1.8}/> Экспорт PDF
          </button>
          <button onClick={()=>setApplyOpen(true)}
                  className="flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-ink-900 text-white text-[12px] font-medium hover:bg-ink-800 transition">
            <Icons.Sparkles size={13} stroke={1.8}/> Применить идею
          </button>
        </div>
      </div>

      {applyOpen && (
        <ApplyIdeaModal ideas={ideas}
                        botUsername={(window.PORTFOLIO.meta || {}).botUsername}
                        onClose={()=>setApplyOpen(false)}/>
      )}

      {/* AI summary banner */}
      <div className="rounded-4xl p-6 mb-6 relative overflow-hidden"
           style={{ background:'linear-gradient(120deg, #fbf3d9 0%, #f6ebc0 100%)' }}>
        <div className="absolute -right-6 top-1/2 -translate-y-1/2 w-40 h-40 rounded-full opacity-30"
             style={{ background:'radial-gradient(circle, #caa01a, transparent 65%)' }}/>
        <div className="relative flex items-start gap-4">
          <div className="w-11 h-11 rounded-2xl bg-ink-900 text-gold-400 flex items-center justify-center flex-shrink-0">
            <Icons.Sparkles size={18} stroke={1.7}/>
          </div>
          <div className="flex-1">
            <div className="text-[10px] tracking-widest uppercase font-mono text-ink-700 mb-1">AI · {window.PORTFOLIO.meta.aiModel} · сводка</div>
            <p className="text-[15px] text-ink-900 leading-relaxed font-light">
              {window.PORTFOLIO.verdict.headline} <span className="text-ink-600">Идеи ниже — как улучшить позиционирование портфеля.</span>
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {ideas.map((idea, i) => (
          <IdeaCard key={idea.n} idea={idea}
                    open={!!open[idea.n]} onToggle={() => toggle(idea.n)}
                    isHighlight={i === 0}/>
        ))}
      </div>

      {/* Disclaimer */}
      <div className="mt-8 rounded-3xl p-5 bg-white/40 border border-ink-900/6 flex items-start gap-3">
        <Icons.Warning size={16} className="text-rust-500 mt-0.5 flex-shrink-0" stroke={1.8}/>
        <p className="text-[12.5px] text-ink-500 leading-relaxed font-light">
          <span className="text-ink-700 font-medium">Это аналитические идеи, а не инвестиционная рекомендация.</span> Расчёты
          основаны на исторических данных и публичной отчётности; они не учитывают вашу налоговую ситуацию, горизонт и цели.
          Окончательное решение — за вами или вашим финансовым консультантом.
        </p>
      </div>
    </section>
  );
};

Object.assign(window, { Ideas });
