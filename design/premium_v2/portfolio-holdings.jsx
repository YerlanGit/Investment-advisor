/* Holdings section — interactive expandable rows + filter chips */

const StatusBadge = ({ status }) => status === 'HOTSPOT'
  ? <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-gold-400 text-ink-900 text-[10px] font-bold tracking-wider uppercase">
      <Icons.Warning size={10} stroke={2.2}/> Hotspot
    </span>
  : <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-cream-200/70 text-ink-700 text-[10px] font-semibold tracking-wider uppercase">
      Normal
    </span>;

const SignalChip = ({ signal }) => {
  const styles = {
    'BUY':         'bg-sage-500/15 text-sage-600',
    'STRONG BUY':  'bg-sage-500 text-white',
    'HOLD':        'bg-ink-900/5 text-ink-700',
    'TRIM':        'bg-rust-500/15 text-rust-600',
    'SELL':        'bg-rust-500 text-white',
  }[signal] || 'bg-ink-900/5 text-ink-700';
  return <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wider ${styles}`}>{signal}</span>;
};

// Mini horizontal bar showing weight or risk share
const MiniBar = ({ value, max=30, color='#1c1b1a', height=4 }) => {
  const pct = Math.max(0, Math.min(100, (value/max)*100));  // clamp: маржинальный кэш даёт отрицательный вес
  return (
    <div className="w-full bg-ink-900/8 rounded-full overflow-hidden" style={{height}}>
      <div className="rounded-full" style={{ width:`${pct}%`, height:'100%', background:color }}/>
    </div>
  );
};

const HoldingRow = ({ h, open, onToggle, idx }) => {
  const IconC = sectorIcon(h.cls);
  const pos = h.pnlPct >= 0;
  return (
    <div className={`relative transition-colors ${open?'bg-cream-50/70':'hover:bg-cream-50/40'}`}>
      <button onClick={onToggle} className="w-full grid grid-cols-[40px_minmax(0,2fr)_minmax(0,1.6fr)_repeat(4,minmax(0,1fr))_88px_40px] items-center gap-3 px-6 py-4 text-left">
        {/* sector icon */}
        <div className="w-9 h-9 rounded-2xl bg-cream-100 border border-ink-900/5 flex items-center justify-center text-ink-700">
          <IconC size={16} stroke={1.6}/>
        </div>
        {/* ticker + name */}
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-[16px] font-bold tracking-tight num text-ink-900">{h.t}</span>
            <StatusBadge status={h.status}/>
          </div>
          <div className="text-[12px] text-ink-500 truncate mt-0.5">{h.name}</div>
        </div>
        {/* sector */}
        <div className="text-[12px] text-ink-700">{h.cls}</div>
        {/* weight w/ bar */}
        <div>
          <div className="text-[13px] font-semibold num text-ink-900">{h.w.toFixed(1)}%</div>
          <div className="mt-1.5"><MiniBar value={h.w} max={22} color="#1c1b1a"/></div>
        </div>
        {/* risk w/ bar */}
        <div>
          <div className="text-[13px] font-semibold num text-ink-900">{h.risk.toFixed(1)}%</div>
          <div className="mt-1.5"><MiniBar value={h.risk} max={26} color={h.status==='HOTSPOT'?'#f5d04e':'#a8a293'}/></div>
        </div>
        {/* beta */}
        <div className="text-[13px] font-medium num text-ink-700">{h.beta.toFixed(2)}</div>
        {/* P/L */}
        <div>
          <div className={`text-[14px] font-semibold num ${pos?'text-sage-600':'text-rust-600'}`}>{pos?'+':''}{h.pnlPct.toFixed(1)}%</div>
          <div className={`text-[11px] num ${pos?'text-sage-500':'text-rust-500'}`}>{pos?'+':''}${Math.abs(h.pnlUsd).toLocaleString('ru-RU')}</div>
        </div>
        {/* signal */}
        <div className="flex justify-end"><SignalChip signal={h.signal}/></div>
        {/* chevron */}
        <div className={`w-9 h-9 rounded-full bg-ink-900/5 flex items-center justify-center text-ink-700 transition-transform ${open?'rotate-180':''}`}>
          <Icons.Chevron size={14}/>
        </div>
      </button>

      {/* Expanded panel */}
      <div className="mob-detail overflow-hidden transition-[max-height,opacity] duration-500 ease-out"
           style={{ maxHeight: open?720:0, opacity: open?1:0 }}>
        <div className="px-6 pb-6 pt-1">
          <div className="rounded-3xl p-5 bg-white/70 border border-ink-900/5">
            <div className="mb-4">
              <div className="text-[11px] tracking-widest uppercase text-ink-500 font-mono">Фундаментал · SEC EDGAR</div>
              <div className="text-[15px] text-ink-900 font-medium mt-0.5">{h.name}</div>
              {/* −37: валюта сделки. Стоимость позиции и вес считаются в базовой
                  валюте отчёта; цена покупки печатается в ней же, а исходная
                  сумма — в скобках, чтобы «12 000 ₸» не читалось как доллары. */}
              {h.buyPrice && h.buyPrice !== '—' && (
                <div className="text-[11px] text-ink-500 font-mono mt-1">
                  Цена покупки: <span className="text-ink-700">{h.buyPrice}</span>
                  {h.fxConverted && <span className="text-ink-400"> · стоимость пересчитана по курсу</span>}
                </div>
              )}
            </div>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
              <FundCell label="ROE"        value={h.fund.roe}/>
              <FundCell label="Маржа"      value={h.fund.margin}/>
              <FundCell label="Долг/А"     value={h.fund.debt}/>
              <FundCell label="Рост г/г"   value={h.fund.growth}/>
              <FundCell label="ATR"        value={h.fund.atr}/>
              <FundCell label="Beta"       value={h.beta.toFixed(2)}/>
            </div>
            {/* R-18: блок появляется ТОЛЬКО при наличии текста — иначе на каждой
                раскрытой карточке висела золотая ✨ с пустым абзацем.  Источник —
                тот же, что в DEEP: вывод по фундаменталу, иначе заметка позиции. */}
            {(h.fundNote || h.note) && (
              <div className="mt-4 flex items-start gap-3 text-[13px] text-ink-700 leading-relaxed">
                <Icons.Sparkles size={14} className="text-gold-600 mt-1 flex-shrink-0" stroke={1.8}/>
                <p className="font-light">
                  <span className="text-ink-500 font-medium">Вывод: </span>{h.fundNote || h.note}
                  <span className="text-ink-400 font-mono text-[11px]"> [SEC EDGAR]</span>
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const FundCell = ({ label, value }) => (
  <div className="rounded-2xl bg-cream-50 border border-ink-900/5 px-3 py-2.5">
    <div className="text-[10px] uppercase tracking-wider text-ink-500 font-mono">{label}</div>
    <div className="text-[14px] font-semibold num text-ink-900 mt-0.5">{value}</div>
  </div>
);

// Robust sector/class matchers for the holdings filters.  Production tags each
// holding with a GICS-style `sector` (English: "Technology"/"Semiconductors"/…)
// AND an asset `cls` (Russian: "Акции США"/"Облигации"/…).  The old filters
// compared `cls` against SECTOR names → never matched (bug: "Ничего не подходит
// под фильтр «Технологии»").  These matchers read `sector` first and fall back
// to `cls`, tolerant of English/Russian, so they work on real + sample data.
// Русское склонение «позиция» по числу — счётчик секции строится по данным,
// а не литералом, поэтому число может быть любым.
const _plural = (n) => {
  const a = Math.abs(Number(n) || 0), d10 = a % 10, d100 = a % 100;
  if (d10 === 1 && d100 !== 11) return 'позиция';
  if (d10 >= 2 && d10 <= 4 && (d100 < 12 || d100 > 14)) return 'позиции';
  return 'позиций';
};

const _blob = (h) => `${h.sector||''} ${h.cls||''}`.toLowerCase();
const _isTech = (h) => /tech|semicond|communicat|software|internet|технолог|полупровод|коммуникац|софт/.test(_blob(h));
const _isDefensive = (h) =>
  /health|staple|utilit|consumer defensive|bond|treasur|gold|silver|precious|здрав|потреб|коммунал|облигац|золот|серебр|драгмет/.test(_blob(h))
  || ['Облигации','Ден. средства','Сырьё'].includes(h.cls);

const Holdings = () => {
  const [openIdx, setOpenIdx] = React.useState(0);
  const [filter, setFilter] = React.useState('Все');
  const all = window.PORTFOLIO.holdings;
  // L-15 (2026-07-19): «HOTSPOT» — только когда такие позиции есть.
  const hasHot = all.some(h => h.status === 'HOTSPOT');
  const filters = ['Все', ...(hasHot ? ['HOTSPOT'] : []), 'Технологии','Защитные','Доходные','В минусе'];
  const rows = all.filter(h => {
    if (filter==='Все') return true;
    if (filter==='HOTSPOT') return h.status==='HOTSPOT';
    if (filter==='Технологии') return _isTech(h);
    if (filter==='Защитные') return _isDefensive(h);
    if (filter==='Доходные') return h.pnlPct >= 10;
    if (filter==='В минусе') return h.pnlPct < 0;
    return true;
  });

  return (
    <section id="holdings" className="rise" data-screen-label="02 Holdings">
      <div className="flex items-end justify-between gap-4 flex-wrap mb-6">
        <div>
          <div className="flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2">
            {/* Аудит 2026-08-12: было «Holdings · 9 позиций» ЛИТЕРАЛОМ макета —
                каждый отчёт печатал 9 независимо от книги (живой отчёт 12.08:
                17 позиций в шапке и «9 позиций» здесь же).  Тот же класс, что
                mock-дата R-2 и зашитый счётчик «4 идеи» (§−48).  Считаем по
                данным и склоняем слово; при фильтре показываем «N из M». */}
            <span className="w-1.5 h-1.5 rounded-full bg-gold-400"/> Holdings · {rows.length !== all.length ? `${rows.length} из ${all.length}` : all.length} {_plural(rows.length !== all.length ? all.length : rows.length)}
          </div>
          <h2 className="text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900">
            Что вы держите<span className="text-ink-400">.</span>
          </h2>
          <p className="text-[15px] text-ink-500 mt-2 font-light">Нажмите на строку, чтобы увидеть фундаментал бумаги.</p>
        </div>

        {/* Filter pills */}
        <div className="flex items-center gap-2 flex-wrap">
          {filters.map(f => (
            <button key={f} onClick={()=>setFilter(f)}
              className={`px-3.5 py-1.5 rounded-full text-[12px] font-medium transition-colors
                          ${filter===f?'bg-ink-900 text-white':'bg-white/60 text-ink-700 hover:bg-white border border-ink-900/8'}`}>
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-strong rounded-4xl shadow-card overflow-hidden">
        <div className="swipe-hint items-center gap-1 text-[10px] font-mono text-gold-700 bg-gold-400/15 rounded-full px-2.5 py-1 mb-2.5 w-max">↔ листайте таблицу</div>
    <div className="mob-scroll-x"><div>
        {/* Table header */}
        <div className="grid grid-cols-[40px_minmax(0,2fr)_minmax(0,1.6fr)_repeat(4,minmax(0,1fr))_88px_40px] items-center gap-3 px-6 py-3.5 border-b border-ink-900/6 text-[10px] tracking-widest uppercase text-ink-500 font-mono">
          <div></div>
          <div>Тикер · Имя</div>
          <div>Класс</div>
          <div>Вес</div>
          <div>Риск</div>
          <div>Beta</div>
          <div>P/L</div>
          <div className="text-right">Сигнал</div>
          <div></div>
        </div>
        <div className="divide-y divide-ink-900/5">
          {rows.map((h,i) => (
            <HoldingRow key={h.t} h={h} idx={i}
                        open={openIdx===i}
                        onToggle={() => setOpenIdx(openIdx===i?-1:i)}/>
          ))}
          {rows.length === 0 && (
            <div className="px-6 py-12 text-center text-ink-500 text-[14px]">Ничего не подходит под фильтр «{filter}».</div>
          )}
        </div>
        </div></div>
      </div>

      {/* R-18: секционная сводка ИИ по составу — паритет с DEEP.  BASE-промпт
          её запрашивает, экстрактор сохраняет; раньше терялась в маппере. */}
      {window.PORTFOLIO.holdingsAI && (
        <div className="mt-5 rounded-3xl p-5 glass-strong shadow-card flex items-start gap-4">
          <div className="w-10 h-10 rounded-2xl bg-ink-900 text-gold-400 flex items-center justify-center flex-shrink-0">
            <Icons.Sparkles size={17} stroke={1.7}/>
          </div>
          <div>
            <div className="text-[10px] tracking-widest uppercase font-mono text-ink-500 mb-1">AI · сводка по составу</div>
            <p className="text-[14px] text-ink-800 leading-relaxed font-light">{window.PORTFOLIO.holdingsAI}</p>
          </div>
        </div>
      )}
    </section>
  );
};

Object.assign(window, { Holdings });
