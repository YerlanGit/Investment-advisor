/* SVG chart primitives — all hand-drawn, no chart lib.
   Tabular numerics, soft strokes, gold + ink palette. */

// ── Risk gauge: dashed track 0→100 with filled arc to value, big number center
const RiskGauge = ({ value=62, size=240, label='Индекс риска' }) => {
  const r = size/2 - 18;
  const cx = size/2, cy = size/2;
  // 270° sweep from 135° to 45° (top open like reference time tracker)
  const start = 135 * Math.PI/180;
  const end   = (135 + 270 * (value/100)) * Math.PI/180;
  const endFull = (135 + 270) * Math.PI/180;
  const pt = (a) => `${cx + r*Math.cos(a)} ${cy + r*Math.sin(a)}`;
  const largeArc = (270*(value/100)) > 180 ? 1 : 0;
  const zone = value <= 33 ? 'низкий' : value <= 66 ? 'умеренный' : 'высокий';
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {/* Dashed full track */}
      <path d={`M ${pt(start)} A ${r} ${r} 0 1 1 ${pt(endFull)}`}
            fill="none" stroke="#1c1b1a" strokeOpacity=".25"
            strokeWidth="2" strokeDasharray="2 6" strokeLinecap="round"/>
      {/* Filled arc */}
      <path d={`M ${pt(start)} A ${r} ${r} 0 ${largeArc} 1 ${pt(end)}`}
            fill="none" stroke="#f5d04e" strokeWidth="14" strokeLinecap="round"/>
      {/* Inner subtle ring */}
      <circle cx={cx} cy={cy} r={r-22} fill="none" stroke="#1c1b1a" strokeOpacity=".06" strokeWidth="1"/>
      {/* Center value */}
      <text x={cx} y={cy-4} textAnchor="middle" className="num"
            fontFamily="Manrope" fontWeight="700" fontSize={size*0.28} fill="#1c1b1a">{value}</text>
      <text x={cx} y={cy+size*0.13} textAnchor="middle"
            fontFamily="Manrope" fontWeight="500" fontSize={size*0.062} fill="#6b6862"
            letterSpacing="0.08em">/ 100 · {zone.toUpperCase()}</text>
    </svg>
  );
};

// ── Waterfall: standalone bars + diversification (negative) + total
const Waterfall = ({ data, height=200 }) => {
  const { standalone, diversification, total, sumStandalone } = data;
  // «Прочие» bridges the shown top-N standalone bars up to the FULL sum so the
  // waterfall RECONCILES: ΣstandAlone + diversification = total.  Without it the
  // 4 shown bars (≈20) plus diversification landed at ≈7.7, not the 18.7 total —
  // the diversification step looked detached/wrong.
  const _shown = standalone.reduce((a,s)=>a+s.v, 0);
  const _other = Math.round((sumStandalone - _shown)*10)/10;
  const cols = [
    ...standalone.map(s => ({ t:s.t, v:s.v, kind:'pos' })),
    ...(_other > 0.5 ? [{ t:'Прочие', v:_other, kind:'pos' }] : []),
    { t:'Дивер-сификация', v:diversification, kind:'neg' },
    { t:'Итог', v:total, kind:'total' },
  ];
  // maxV = running peak of the standalone steps ⊕ total (fills the card; no gap).
  let _run = 0, _peak = 0;
  cols.forEach(c => { if (c.kind==='pos') { _run += c.v; _peak = Math.max(_peak, _run); } });
  const maxV = Math.max(_peak, total) * 1.06;
  const W = 520, H = height, padL = 32, padR = 12, padT = 18, padB = 36;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const barW = innerW / cols.length * 0.62;
  const gap  = innerW / cols.length;

  // running sum for waterfall floors
  let running = 0;
  const placed = cols.map((c, i) => {
    let y0, y1, color;
    if (c.kind === 'pos') {
      y0 = running;
      y1 = running + c.v;
      running = y1;
      color = '#1c1b1a';         // standalone bars — all «отдельно» (legend: black)
    } else if (c.kind === 'neg') {
      y0 = running;
      y1 = running + c.v;        // v is negative
      running = y1;
      color = '#c47358';         // «диверсификация» (legend: rust)
    } else {                     // total — floor at 0
      y0 = 0;
      y1 = c.v;
      color = '#f5d04e';         // «итог» (legend: gold)
    }
    const top = Math.max(y0, y1);
    const bot = Math.min(y0, y1);
    return {
      ...c,
      yTop: padT + (1 - top/maxV) * innerH,
      yBot: padT + (1 - bot/maxV) * innerH,
      x: padL + gap*i + (gap - barW)/2,
      color, isTotal: c.kind === 'total',
    };
  });
  const yAxis = padT + innerH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
      {/* baseline */}
      <line x1={padL} y1={yAxis} x2={W-padR} y2={yAxis} stroke="#1c1b1a" strokeOpacity=".12"/>
      {/* gridlines */}
      {[5,10,15].map(g => (
        <g key={g}>
          <line x1={padL} y1={padT+(1-g/maxV)*innerH} x2={W-padR} y2={padT+(1-g/maxV)*innerH}
                stroke="#1c1b1a" strokeOpacity=".06" strokeDasharray="2 4"/>
          <text x={padL-6} y={padT+(1-g/maxV)*innerH+3} textAnchor="end"
                fontSize="9" fontFamily="JetBrains Mono" fill="#9a958c">{g}%</text>
        </g>
      ))}
      {/* bars */}
      {placed.map((b, i) => (
        <g key={i}>
          <rect x={b.x} y={b.yTop} width={barW} height={Math.max(2, b.yBot-b.yTop)}
                rx="5" ry="5" fill={b.color}
                opacity={b.isTotal ? 1 : 0.92}/>
          {/* value label */}
          <text x={b.x + barW/2} y={b.yTop - 6} textAnchor="middle"
                fontSize="11" fontFamily="JetBrains Mono" fontWeight="500"
                fill={b.kind==='neg' ? '#a85a40' : '#1c1b1a'}>
            {b.v>0?'+':''}{b.v.toFixed(1)}
          </text>
          {/* x label */}
          <text x={b.x + barW/2} y={yAxis + 18} textAnchor="middle"
                fontSize="10" fontFamily="Manrope" fontWeight="500" fill="#6b6862">
            {b.t}
          </text>
        </g>
      ))}
    </svg>
  );
};

// ── Sector stacked bar (horizontal) + legend
const SectorBar = ({ sectors }) => {
  let acc = 0;
  return (
    <svg viewBox="0 0 320 18" className="w-full h-3.5" preserveAspectRatio="none">
      <rect x="0" y="0" width="320" height="18" rx="9" fill="#efe9d8"/>
      {sectors.map((s,i) => {
        const x = (acc / 100) * 320;
        const w = (s.pct / 100) * 320 - (i===sectors.length-1?0:1.5);
        acc += s.pct;
        return <rect key={s.name} x={x} y="0" width={w} height="18" fill={s.hue}
                     rx={i===0?9:0} ry={i===0?9:0}/>;
      })}
    </svg>
  );
};

// ── Performance line chart (port vs SPX)
// `port`/`spx` are cumulative % series for the SELECTED window; `labels` are the
// x-axis ticks; optional `xs` gives each point's 0..1 horizontal position (real
// elapsed-time spacing).  The domain always includes the 0% baseline and is
// sign-aware, so a losing window (all-negative) renders correctly with a visible
// zero line and signed axis labels.
const PerfChart = ({ labels=[], port=[], spx=[], xs=null, height=248 }) => {
  const n = Math.max(port.length, spx.length);
  const W = 720, H = height, padL = 42, padR = 60, padT = 26, padB = 34;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  // Sign-aware domain that always brackets 0 with a little headroom.
  const all = [...port, ...spx, 0];
  let vmin = Math.min(...all), vmax = Math.max(...all);
  const range = (vmax - vmin) || 1;
  vmax += range * 0.14;
  if (vmin < 0) vmin -= range * 0.10;
  const span = (vmax - vmin) || 1;

  const xfrac = (i) => (xs && xs.length === n) ? xs[i] : (n > 1 ? i/(n-1) : 0);
  const px = (i) => padL + xfrac(i) * innerW;
  const py = (v) => padT + (1 - (v - vmin)/span) * innerH;

  const line = (arr) => arr.map((v,i)=> `${i===0?'M':'L'} ${px(i).toFixed(1)} ${py(v).toFixed(1)}`).join(' ');
  const area = (arr) => `${line(arr)} L ${px(arr.length-1).toFixed(1)} ${py(0).toFixed(1)} L ${px(0).toFixed(1)} ${py(0).toFixed(1)} Z`;

  // Evenly-spaced y ticks across the real (signed) domain.  Tick labels gain a
  // decimal when the window is small (a ±2% range as integers duplicates to
  // "+2% +2% +1% +1%").  End-value labels always carry one decimal so they read
  // identically to the headline (+14.2%, not +14%).
  const NT = 4;
  const ticksY = Array.from({length: NT+1}, (_,k) => vmin + span * k/NT);
  const tickDec = (span / NT) < 1.5 ? 1 : 0;
  // Sign-aware %, guarding against a "−0%" when a tiny negative rounds to zero.
  const signedPct = (v, dec) => {
    const r = Math.abs(v).toFixed(dec);
    return `${(v < 0 && parseFloat(r) !== 0) ? '−' : '+'}${r}%`;
  };
  const fmtTick = (v) => signedPct(v, tickDec);
  const last = n - 1;
  const fmtPct = (v) => signedPct(v, 1);

  // Keep the port/spx end-labels from colliding when the two lines are close.
  const portY = py(port[last]), spxY = py(spx[last]);
  const collide = Math.abs(portY - spxY) < 16;
  const portLabelY = collide && portY >= spxY ? portY + 7 : portY - 9;
  const spxLabelY  = collide && portY >= spxY ? spxY  - 9 : spxY  + 14;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="gradPort" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%"  stopColor="#f5d04e" stopOpacity="0.45"/>
          <stop offset="100%" stopColor="#f5d04e" stopOpacity="0"/>
        </linearGradient>
        <linearGradient id="gradSpx" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%"  stopColor="#1c1b1a" stopOpacity="0.08"/>
          <stop offset="100%" stopColor="#1c1b1a" stopOpacity="0"/>
        </linearGradient>
      </defs>
      {/* gridlines + signed y labels */}
      {ticksY.map((t,i) => (
        <g key={i}>
          <line x1={padL} y1={py(t)} x2={W-padR} y2={py(t)}
                stroke="#1c1b1a" strokeOpacity=".06" strokeDasharray="2 4"/>
          <text x={padL-8} y={py(t)+3} textAnchor="end" fontSize="9"
                fontFamily="JetBrains Mono" fill="#9a958c">{fmtTick(t)}</text>
        </g>
      ))}
      {/* emphasised 0% baseline */}
      <line x1={padL} y1={py(0)} x2={W-padR} y2={py(0)} stroke="#1c1b1a" strokeOpacity=".18"/>
      {/* x labels */}
      {labels.map((m,i) => (
        <text key={i} x={px(i)} y={H-10}
              textAnchor={i===0?'start':(i===n-1?'end':'middle')} fontSize="10"
              fontFamily="Manrope" fill="#9a958c">{m}</text>
      ))}
      {/* SPX area+line */}
      <path d={area(spx)} fill="url(#gradSpx)"/>
      <path d={line(spx)} fill="none" stroke="#1c1b1a" strokeOpacity=".55" strokeWidth="1.5" strokeDasharray="4 4"/>
      {/* Port area+line */}
      <path d={area(port)} fill="url(#gradPort)"/>
      <path d={line(port)} fill="none" stroke="#caa01a" strokeWidth="2.4"/>
      {/* end points */}
      <circle cx={px(last)} cy={spxY}  r="4" fill="#fff"    stroke="#1c1b1a" strokeWidth="1.5"/>
      <circle cx={px(last)} cy={portY} r="5" fill="#f5d04e" stroke="#caa01a" strokeWidth="1.6"/>
      {/* end value labels — anchored to the RIGHT of the marker inside padR so they never clip */}
      <text x={px(last)+9} y={portLabelY} textAnchor="start"
            fontSize="12" fontFamily="JetBrains Mono" fontWeight="700" fill="#caa01a">{fmtPct(port[last])}</text>
      <text x={px(last)+9} y={spxLabelY} textAnchor="start"
            fontSize="11" fontFamily="JetBrains Mono" fontWeight="500" fill="#6b6862">{fmtPct(spx[last])}</text>
    </svg>
  );
};

// ── Mini sparkline (used in KPI cards)
const Sparkline = ({ points, color='#1c1b1a', height=36, width=110, gradId='spark' }) => {
  const padY = 4;
  const minV = Math.min(...points);
  const maxV = Math.max(...points);
  const span = (maxV - minV) || 1;
  const W = width, H = height;
  const px = (i) => (i/(points.length-1)) * W;
  const py = (v) => padY + (1 - (v-minV)/span) * (H - padY*2);
  const p = points.map((v,i)=>`${i===0?'M':'L'} ${px(i).toFixed(1)} ${py(v).toFixed(1)}`).join(' ');
  const area = `${p} L ${W} ${H} L 0 ${H} Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
      <defs>
        <linearGradient id={gradId} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%"  stopColor={color} stopOpacity="0.25"/>
          <stop offset="100%" stopColor={color} stopOpacity="0"/>
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gradId})`}/>
      <path d={p} fill="none" stroke={color} strokeWidth="1.6" strokeLinecap="round"/>
    </svg>
  );
};

// ── Animated counter (count-up on mount/visible)
const Counter = ({ value, decimals=0, prefix='', suffix='', duration=900, className='' }) => {
  const [v, setV] = React.useState(0);
  React.useEffect(() => {
    let raf, t0;
    const tick = (t) => {
      if (!t0) t0 = t;
      const k = Math.min(1, (t - t0) / duration);
      const e = 1 - Math.pow(1-k, 3); // easeOutCubic
      setV(value * e);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);
  const formatted = Math.abs(v) >= 1000
    ? v.toLocaleString('ru-RU', { maximumFractionDigits: decimals })
    : v.toFixed(decimals);
  return <span className={`num ${className}`}>{prefix}{formatted}{suffix}</span>;
};

Object.assign(window, { RiskGauge, Waterfall, SectorBar, PerfChart, Sparkline, Counter });
