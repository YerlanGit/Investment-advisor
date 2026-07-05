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
  // waterfall RECONCILES: ΣstandAlone + diversification = total (was detached).
  const _shown = standalone.reduce((a,s)=>a+s.v, 0);
  const _other = Math.round((sumStandalone - _shown)*10)/10;
  const cols = [
    ...standalone.map(s => ({ t:s.t, v:s.v, kind:'pos' })),
    ...(_other > 0.5 ? [{ t:'Прочие', v:_other, kind:'pos' }] : []),
    { t:'Дивер-сификация', v:diversification, kind:'neg' },
    { t:'Итог', v:total, kind:'total' },
  ];
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
const PerfChart = ({ months, port, spx, height=240 }) => {
  const W = 720, H = height, padL = 36, padR = 20, padT = 24, padB = 32;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const all = [...port, ...spx];
  const maxV = Math.max(...all) * 1.15;
  const minV = Math.min(0, Math.min(...all));
  const span = maxV - minV;
  const px = (i) => padL + (i/(months.length-1)) * innerW;
  const py = (v) => padT + (1 - (v - minV)/span) * innerH;

  const path = (arr) => arr.map((v,i)=> `${i===0?'M':'L'} ${px(i).toFixed(1)} ${py(v).toFixed(1)}`).join(' ');
  const area = (arr) => `${path(arr)} L ${px(arr.length-1)} ${py(0)} L ${px(0)} ${py(0)} Z`;

  const ticksY = [0, maxV/4, maxV/2, (3*maxV)/4, maxV].map(v => Math.round(v));
  const lastIdx = months.length - 1;

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
      {/* gridlines */}
      {ticksY.map((t,i) => (
        <g key={i}>
          <line x1={padL} y1={py(t)} x2={W-padR} y2={py(t)}
                stroke="#1c1b1a" strokeOpacity=".06" strokeDasharray="2 4"/>
          <text x={padL-8} y={py(t)+3} textAnchor="end" fontSize="9"
                fontFamily="JetBrains Mono" fill="#9a958c">+{t}%</text>
        </g>
      ))}
      {/* x labels */}
      {months.map((m,i) => (
        <text key={i} x={px(i)} y={H-10} textAnchor="middle" fontSize="10"
              fontFamily="Manrope" fill="#9a958c">{m}</text>
      ))}
      {/* SPX area+line */}
      <path d={area(spx)} fill="url(#gradSpx)"/>
      <path d={path(spx)} fill="none" stroke="#1c1b1a" strokeOpacity=".55" strokeWidth="1.5" strokeDasharray="4 4"/>
      {/* Port area+line */}
      <path d={area(port)} fill="url(#gradPort)"/>
      <path d={path(port)} fill="none" stroke="#caa01a" strokeWidth="2.2"/>
      {/* end points */}
      <circle cx={px(lastIdx)} cy={py(port[lastIdx])} r="5" fill="#f5d04e" stroke="#caa01a" strokeWidth="1.5"/>
      <circle cx={px(lastIdx)} cy={py(spx[lastIdx])}  r="4" fill="#fff"    stroke="#1c1b1a" strokeWidth="1.5"/>
      {/* end labels */}
      <g>
        <rect x={px(lastIdx)-4} y={py(port[lastIdx])-26} width="56" height="18" rx="9" fill="#1c1b1a"/>
        <text x={px(lastIdx)+24} y={py(port[lastIdx])-13} textAnchor="middle"
              fontSize="11" fontFamily="JetBrains Mono" fontWeight="500" fill="#f5d04e">+{port[lastIdx]}%</text>
        <rect x={px(lastIdx)-4} y={py(spx[lastIdx])+8} width="56" height="18" rx="9" fill="#fff" stroke="#1c1b1a" strokeOpacity=".15"/>
        <text x={px(lastIdx)+24} y={py(spx[lastIdx])+20} textAnchor="middle"
              fontSize="11" fontFamily="JetBrains Mono" fontWeight="500" fill="#3a3833">+{spx[lastIdx]}%</text>
      </g>
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

// ── Factor β-radar: portfolio polygon vs market reference (10 axes)
const FactorRadar = ({ factors, size=300 }) => {
  const cx = size/2, cy = size/2;
  const R = size/2 - 46;
  const VMIN = -0.5, VMAX = 1.2;
  const norm = (v) => Math.max(0, (v - VMIN) / (VMAX - VMIN));
  const N = factors.length;
  const ang = (i) => (-90 + i*360/N) * Math.PI/180;
  const pt = (i, rr) => [cx + rr*Math.cos(ang(i)), cy + rr*Math.sin(ang(i))];
  const poly = (key) => factors.map((f,i) => pt(i, norm(f[key])*R).map(n=>n.toFixed(1)).join(',')).join(' ');
  const rings = [-0.5, 0, 0.5, 1.0];
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-auto" style={{ maxWidth: size }}>
      {/* rings */}
      {rings.map((rv,k) => {
        const rr = norm(rv)*R;
        const p = factors.map((f,i)=>pt(i,rr).map(n=>n.toFixed(1)).join(',')).join(' ');
        return <polygon key={k} points={p} fill="none" stroke="#1c1b1a"
                        strokeOpacity={rv===0?0.22:0.08} strokeWidth={rv===0?1:0.7}
                        strokeDasharray={rv===0?'none':'2 4'}/>;
      })}
      {/* spokes + labels */}
      {factors.map((f,i) => {
        const [ex,ey] = pt(i, R);
        const [lx,ly] = pt(i, R+22);
        const anchor = Math.abs(lx-cx) < 8 ? 'middle' : (lx > cx ? 'start' : 'end');
        return (
          <g key={i}>
            <line x1={cx} y1={cy} x2={ex} y2={ey} stroke="#1c1b1a" strokeOpacity="0.07" strokeWidth="0.7"/>
            <text x={lx} y={ly+3} textAnchor={anchor} fontSize="9.5" fontFamily="JetBrains Mono"
                  fill="#6b6862">{f.name}</text>
          </g>
        );
      })}
      {/* market reference polygon (dashed) */}
      <polygon points={poly('mkt')} fill="#1c1b1a" fillOpacity="0.04"
               stroke="#1c1b1a" strokeOpacity="0.35" strokeWidth="1.2" strokeDasharray="4 4"/>
      {/* portfolio polygon (gold) */}
      <polygon points={poly('port')} fill="#f5d04e" fillOpacity="0.22"
               stroke="#caa01a" strokeWidth="2"/>
      {factors.map((f,i) => {
        const [px,py] = pt(i, norm(f.port)*R);
        return <circle key={i} cx={px} cy={py} r="2.6" fill="#caa01a"/>;
      })}
    </svg>
  );
};

// ── Regime quadrant: Growth × Cycle, dot at current coords
const RegimeQuadrant = ({ dot, size=300 }) => {
  const pad = 34;
  const inner = size - pad*2;
  const cx = pad + inner/2, cy = pad + inner/2;
  const SCALE = 0.22; // axis half-range
  const X = (v) => cx + (v/SCALE) * (inner/2);
  const Y = (v) => cy - (v/SCALE) * (inner/2);
  const dx = X(dot.cycle), dy = Y(dot.growth);
  // Audit 2026-07-05 (R-1): labels pinned to finance/regime.py quadrants —
  // X=cycle, Y=growth ⇒ top-left (growth+, cycle−) is SLOWDOWN and bottom-right
  // (growth−, cycle+) is RECOVERY.  The old layout had them SWAPPED (pre-fix
  // engine semantics), so a Slowdown reading would land the dot in a quadrant
  // labelled «RECOVERY» — visually contradicting the regime label.
  const quads = [
    { x:pad,        y:pad,        label:'SLOWDOWN',  fill:'#a8a293' },
    { x:cx,         y:pad,        label:'EXPANSION', fill:'#caa01a' },
    { x:pad,        y:cy,         label:'RECESSION', fill:'#c47358' },
    { x:cx,         y:cy,         label:'RECOVERY',  fill:'#5d7c5c' },
  ];
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-auto" style={{ maxWidth: size }}>
      {quads.map((q,i)=>(
        <g key={i}>
          <rect x={q.x} y={q.y} width={inner/2} height={inner/2} fill={q.fill}
                fillOpacity={q.label==='EXPANSION'?0.14:0.05}/>
          <text x={q.x+inner/4} y={q.y + (q.y<cy? 16 : inner/2-8)} textAnchor="middle"
                fontSize="9" fontFamily="JetBrains Mono" fontWeight="600"
                fill="#1c1b1a" fillOpacity="0.5" letterSpacing="0.05em">{q.label}</text>
        </g>
      ))}
      {/* axes */}
      <line x1={cx} y1={pad} x2={cx} y2={pad+inner} stroke="#1c1b1a" strokeOpacity="0.35" strokeWidth="0.8"/>
      <line x1={pad} y1={cy} x2={pad+inner} y2={cy} stroke="#1c1b1a" strokeOpacity="0.35" strokeWidth="0.8"/>
      {/* axis labels */}
      <text x={cx} y={pad-12} textAnchor="middle" fontSize="9.5" fontFamily="JetBrains Mono" fontWeight="600" fill="#6b6862">↑ Growth</text>
      <text x={pad+inner+2} y={cy-6} textAnchor="end" fontSize="9.5" fontFamily="JetBrains Mono" fontWeight="600" fill="#6b6862">Cycle →</text>
      {/* connector + dot */}
      <line x1={cx} y1={cy} x2={dx} y2={dy} stroke="#caa01a" strokeOpacity="0.45" strokeWidth="1.2" strokeDasharray="3 2"/>
      <circle cx={dx} cy={dy} r="13" fill="#f5d04e" fillOpacity="0.2"/>
      <circle cx={dx} cy={dy} r="7" fill="#f5d04e" stroke="#caa01a" strokeWidth="1.5"/>
      <text x={dx+14} y={dy-6} fontSize="9.5" fontFamily="JetBrains Mono" fontWeight="600" fill="#caa01a">сейчас</text>
      <text x={dx+14} y={dy+5} fontSize="8" fontFamily="JetBrains Mono" fill="#6b6862">
        {/* R-4: signed formatting — «+-0.05» is impossible now */}
        G {(dot.growth>=0?'+':'')+dot.growth.toFixed(2)} · C {(dot.cycle>=0?'+':'')+dot.cycle.toFixed(2)}
      </text>
    </svg>
  );
};

// ── Mandate compliance bar: allowed band [lo,hi] + tick at value
const MandateBar = ({ value, lo, hi, state }) => {
  const tone = { ok:'#5d7c5c', over:'#c47358', under:'#caa01a' }[state] || '#6b6862';
  return (
    <div className="relative h-1.5 rounded-full bg-ink-900/8 overflow-visible">
      <div className="absolute inset-y-0 rounded-full bg-sage-500/20"
           style={{ left:`${lo}%`, width:`${hi-lo}%` }}/>
      <div className="absolute top-1/2 -translate-y-1/2 w-[3px] h-3.5 rounded-full"
           style={{ left:`calc(${Math.min(100,Math.max(0,value))}% - 1.5px)`, background:tone }}/>
    </div>
  );
};

// ── Score pillar bar: diverging from center 0, value in [-2,2]
const ScorePillar = ({ value }) => {
  const v = Math.max(-2, Math.min(2, value));
  const half = Math.abs(v)/2 * 50;
  const pos = v >= 0;
  return (
    <div className="relative h-1.5 rounded-full bg-ink-900/6">
      <div className="absolute top-0 bottom-0 w-px bg-ink-900/25" style={{ left:'50%' }}/>
      <div className="absolute top-0 bottom-0 rounded-full"
           style={{ [pos?'left':'right']:'50%', width:`${half}%`, background: pos?'#5d7c5c':'#c47358' }}/>
    </div>
  );
};

// ── Magnitude bar for stress (diverging, normalised to ±20%)
const MagnitudeBar = ({ value }) => {
  const w = Math.min(50, Math.abs(value)/20 * 50);
  const pos = value >= 0;
  return (
    <div className="relative h-1.5 rounded-full bg-ink-900/6">
      <div className="absolute top-0 bottom-0 w-px bg-ink-900/25" style={{ left:'50%' }}/>
      <div className="absolute top-0 bottom-0 rounded-full"
           style={{ [pos?'left':'right']:'50%', width:`${w}%`, background: pos?'#5d7c5c':'#c47358' }}/>
    </div>
  );
};

Object.assign(window, { RiskGauge, Waterfall, SectorBar, PerfChart, Sparkline, Counter,
  FactorRadar, RegimeQuadrant, MandateBar, ScorePillar, MagnitudeBar });
