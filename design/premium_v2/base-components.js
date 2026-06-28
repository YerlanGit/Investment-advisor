/* Lucide-style thin-stroke icons as React components (1.5 stroke). */

const Icon = ({
  d,
  size = 18,
  stroke = 1.6,
  className = '',
  children,
  viewBox = '0 0 24 24',
  fill = 'none'
}) => /*#__PURE__*/React.createElement("svg", {
  xmlns: "http://www.w3.org/2000/svg",
  width: size,
  height: size,
  viewBox: viewBox,
  fill: fill,
  stroke: "currentColor",
  strokeWidth: stroke,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  className: className
}, d ? /*#__PURE__*/React.createElement("path", {
  d: d
}) : children);
const Icons = {
  Sparkles: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M19 14l.7 1.8L21.5 16.5l-1.8.7L19 19l-.7-1.8L16.5 16.5l1.8-.7z"
  })),
  TrendUp: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("polyline", {
    points: "3 17 9 11 13 15 21 7"
  }), /*#__PURE__*/React.createElement("polyline", {
    points: "14 7 21 7 21 14"
  })),
  TrendDown: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("polyline", {
    points: "3 7 9 13 13 9 21 17"
  }), /*#__PURE__*/React.createElement("polyline", {
    points: "14 17 21 17 21 10"
  })),
  Briefcase: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("rect", {
    x: "3",
    y: "7",
    width: "18",
    height: "13",
    rx: "2"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"
  })),
  Wallet: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M3 8a2 2 0 0 1 2-2h14v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M16 13h3"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M19 6V5a2 2 0 0 0-2-2H6"
  })),
  Bell: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M6 8a6 6 0 1 1 12 0c0 4 1.5 6 1.5 6h-15S6 12 6 8z"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M10 19a2 2 0 0 0 4 0"
  })),
  Search: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("circle", {
    cx: "11",
    cy: "11",
    r: "7"
  }), /*#__PURE__*/React.createElement("path", {
    d: "m20 20-3.5-3.5"
  })),
  Settings: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("circle", {
    cx: "12",
    cy: "12",
    r: "3"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h0a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v0a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"
  })),
  ArrowUR: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M7 17L17 7"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M8 7h9v9"
  })),
  ArrowR: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M5 12h14"
  }), /*#__PURE__*/React.createElement("path", {
    d: "m13 6 6 6-6 6"
  })),
  Plus: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M12 5v14"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M5 12h14"
  })),
  Minus: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M5 12h14"
  })),
  Chevron: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("polyline", {
    points: "6 9 12 15 18 9"
  })),
  Warning: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M12 3 22 20H2L12 3z"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M12 10v5"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "12",
    cy: "18",
    r: ".5",
    fill: "currentColor"
  })),
  Shield: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"
  })),
  Pulse: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M3 12h4l2-6 4 12 2-6h6"
  })),
  Layers: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M12 3 2 8l10 5 10-5z"
  }), /*#__PURE__*/React.createElement("path", {
    d: "m2 13 10 5 10-5"
  }), /*#__PURE__*/React.createElement("path", {
    d: "m2 18 10 5 10-5"
  })),
  Filter: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M3 5h18l-7 9v6l-4-2v-4z"
  })),
  Refresh: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("polyline", {
    points: "23 4 23 10 17 10"
  }), /*#__PURE__*/React.createElement("polyline", {
    points: "1 20 1 14 7 14"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M3.5 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.65 4.36A9 9 0 0 0 20.5 15"
  })),
  Dot: p => /*#__PURE__*/React.createElement(Icon, {
    ...p,
    viewBox: "0 0 8 8"
  }, /*#__PURE__*/React.createElement("circle", {
    cx: "4",
    cy: "4",
    r: "3",
    fill: "currentColor"
  })),
  Download: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M12 3v12"
  }), /*#__PURE__*/React.createElement("path", {
    d: "m7 10 5 5 5-5"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M5 21h14"
  })),
  Share: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("circle", {
    cx: "18",
    cy: "5",
    r: "3"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "6",
    cy: "12",
    r: "3"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "18",
    cy: "19",
    r: "3"
  }), /*#__PURE__*/React.createElement("path", {
    d: "m8.6 13.5 6.8 4M15.4 6.5l-6.8 4"
  })),
  User: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("circle", {
    cx: "12",
    cy: "8",
    r: "4"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M4 21a8 8 0 0 1 16 0"
  })),
  Globe: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("circle", {
    cx: "12",
    cy: "12",
    r: "9"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"
  })),
  Cpu: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("rect", {
    x: "6",
    y: "6",
    width: "12",
    height: "12",
    rx: "2"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "9",
    y: "9",
    width: "6",
    height: "6"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M12 2v4M12 18v4M2 12h4M18 12h4M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"
  })),
  Heart: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M12 21s-7-4.5-9.5-9A5.5 5.5 0 0 1 12 6a5.5 5.5 0 0 1 9.5 6c-2.5 4.5-9.5 9-9.5 9z"
  })),
  Coffee: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M4 8h12v6a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4z"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M16 10h2a3 3 0 0 1 0 6h-2"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M7 4v2M10 4v2M13 4v2"
  })),
  Bolt: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "m13 2-9 12h7l-1 8 9-12h-7z"
  })),
  Fuel: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("rect", {
    x: "3",
    y: "4",
    width: "11",
    height: "16",
    rx: "2"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M14 9h3a2 2 0 0 1 2 2v7a1 1 0 0 1-2 0v-3a1 1 0 0 0-1-1h-2"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M19 7l-2-2"
  })),
  Tag: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M12 2H4a2 2 0 0 0-2 2v8l10 10 10-10z"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "7",
    cy: "7",
    r: "1.5",
    fill: "currentColor"
  })),
  Spark: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M12 2v6M12 16v6M4 12H2M22 12h-2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5"
  }))
};

// Map sector name → icon
const sectorIcon = cls => ({
  'Технологии': Icons.Cpu,
  'Здравоохранение': Icons.Heart,
  'Потреб. товары': Icons.Coffee,
  'Энергетика': Icons.Fuel,
  'Облигации': Icons.Shield
})[cls] || Icons.Tag;
window.Icons = Icons;
window.sectorIcon = sectorIcon;
/* SVG chart primitives — all hand-drawn, no chart lib.
   Tabular numerics, soft strokes, gold + ink palette. */

// ── Risk gauge: dashed track 0→100 with filled arc to value, big number center
const RiskGauge = ({
  value = 62,
  size = 240,
  label = 'Индекс риска'
}) => {
  const r = size / 2 - 18;
  const cx = size / 2,
    cy = size / 2;
  // 270° sweep from 135° to 45° (top open like reference time tracker)
  const start = 135 * Math.PI / 180;
  const end = (135 + 270 * (value / 100)) * Math.PI / 180;
  const endFull = (135 + 270) * Math.PI / 180;
  const pt = a => `${cx + r * Math.cos(a)} ${cy + r * Math.sin(a)}`;
  const largeArc = 270 * (value / 100) > 180 ? 1 : 0;
  const zone = value <= 33 ? 'низкий' : value <= 66 ? 'умеренный' : 'высокий';
  return /*#__PURE__*/React.createElement("svg", {
    width: size,
    height: size,
    viewBox: `0 0 ${size} ${size}`
  }, /*#__PURE__*/React.createElement("path", {
    d: `M ${pt(start)} A ${r} ${r} 0 1 1 ${pt(endFull)}`,
    fill: "none",
    stroke: "#1c1b1a",
    strokeOpacity: ".25",
    strokeWidth: "2",
    strokeDasharray: "2 6",
    strokeLinecap: "round"
  }), /*#__PURE__*/React.createElement("path", {
    d: `M ${pt(start)} A ${r} ${r} 0 ${largeArc} 1 ${pt(end)}`,
    fill: "none",
    stroke: "#f5d04e",
    strokeWidth: "14",
    strokeLinecap: "round"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: cx,
    cy: cy,
    r: r - 22,
    fill: "none",
    stroke: "#1c1b1a",
    strokeOpacity: ".06",
    strokeWidth: "1"
  }), /*#__PURE__*/React.createElement("text", {
    x: cx,
    y: cy - 4,
    textAnchor: "middle",
    className: "num",
    fontFamily: "Manrope",
    fontWeight: "700",
    fontSize: size * 0.28,
    fill: "#1c1b1a"
  }, value), /*#__PURE__*/React.createElement("text", {
    x: cx,
    y: cy + size * 0.13,
    textAnchor: "middle",
    fontFamily: "Manrope",
    fontWeight: "500",
    fontSize: size * 0.062,
    fill: "#6b6862",
    letterSpacing: "0.08em"
  }, "/ 100 · ", zone.toUpperCase()));
};

// ── Waterfall: standalone bars + diversification (negative) + total
const Waterfall = ({
  data,
  height = 200
}) => {
  const {
    standalone,
    diversification,
    total,
    sumStandalone
  } = data;
  const cols = [...standalone.map(s => ({
    t: s.t,
    v: s.v,
    kind: 'pos'
  })), {
    t: 'Дивер-сификация',
    v: diversification,
    kind: 'neg'
  }, {
    t: 'Итог',
    v: total,
    kind: 'total'
  }];
  const maxV = Math.max(sumStandalone, total) * 1.15;
  const W = 520,
    H = height,
    padL = 32,
    padR = 12,
    padT = 18,
    padB = 36;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const barW = innerW / cols.length * 0.62;
  const gap = innerW / cols.length;

  // running sum for waterfall floors
  let running = 0;
  const placed = cols.map((c, i) => {
    let y0, y1, color;
    if (c.kind === 'pos') {
      y0 = running;
      y1 = running + c.v;
      running = y1;
      color = i === 0 ? '#1c1b1a' : i === 1 ? '#f5d04e' : '#3a3833';
    } else if (c.kind === 'neg') {
      y0 = running;
      y1 = running + c.v; // v is negative
      running = y1;
      color = '#c47358';
    } else {
      // total — floor at 0
      y0 = 0;
      y1 = c.v;
      color = '#1c1b1a';
    }
    const top = Math.max(y0, y1);
    const bot = Math.min(y0, y1);
    return {
      ...c,
      yTop: padT + (1 - top / maxV) * innerH,
      yBot: padT + (1 - bot / maxV) * innerH,
      x: padL + gap * i + (gap - barW) / 2,
      color,
      isTotal: c.kind === 'total'
    };
  });
  const yAxis = padT + innerH;
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: `0 0 ${W} ${H}`,
    className: "w-full h-auto",
    preserveAspectRatio: "xMidYMid meet"
  }, /*#__PURE__*/React.createElement("line", {
    x1: padL,
    y1: yAxis,
    x2: W - padR,
    y2: yAxis,
    stroke: "#1c1b1a",
    strokeOpacity: ".12"
  }), [5, 10, 15].map(g => /*#__PURE__*/React.createElement("g", {
    key: g
  }, /*#__PURE__*/React.createElement("line", {
    x1: padL,
    y1: padT + (1 - g / maxV) * innerH,
    x2: W - padR,
    y2: padT + (1 - g / maxV) * innerH,
    stroke: "#1c1b1a",
    strokeOpacity: ".06",
    strokeDasharray: "2 4"
  }), /*#__PURE__*/React.createElement("text", {
    x: padL - 6,
    y: padT + (1 - g / maxV) * innerH + 3,
    textAnchor: "end",
    fontSize: "9",
    fontFamily: "JetBrains Mono",
    fill: "#9a958c"
  }, g, "%"))), placed.map((b, i) => /*#__PURE__*/React.createElement("g", {
    key: i
  }, /*#__PURE__*/React.createElement("rect", {
    x: b.x,
    y: b.yTop,
    width: barW,
    height: Math.max(2, b.yBot - b.yTop),
    rx: "5",
    ry: "5",
    fill: b.color,
    opacity: b.isTotal ? 1 : 0.92
  }), /*#__PURE__*/React.createElement("text", {
    x: b.x + barW / 2,
    y: b.yTop - 6,
    textAnchor: "middle",
    fontSize: "11",
    fontFamily: "JetBrains Mono",
    fontWeight: "500",
    fill: b.kind === 'neg' ? '#a85a40' : '#1c1b1a'
  }, b.v > 0 ? '+' : '', b.v.toFixed(1)), /*#__PURE__*/React.createElement("text", {
    x: b.x + barW / 2,
    y: yAxis + 18,
    textAnchor: "middle",
    fontSize: "10",
    fontFamily: "Manrope",
    fontWeight: "500",
    fill: "#6b6862"
  }, b.t))));
};

// ── Sector stacked bar (horizontal) + legend
const SectorBar = ({
  sectors
}) => {
  let acc = 0;
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: "0 0 320 18",
    className: "w-full h-3.5",
    preserveAspectRatio: "none"
  }, /*#__PURE__*/React.createElement("rect", {
    x: "0",
    y: "0",
    width: "320",
    height: "18",
    rx: "9",
    fill: "#efe9d8"
  }), sectors.map((s, i) => {
    const x = acc / 100 * 320;
    const w = s.pct / 100 * 320 - (i === sectors.length - 1 ? 0 : 1.5);
    acc += s.pct;
    return /*#__PURE__*/React.createElement("rect", {
      key: s.name,
      x: x,
      y: "0",
      width: w,
      height: "18",
      fill: s.hue,
      rx: i === 0 ? 9 : 0,
      ry: i === 0 ? 9 : 0
    });
  }));
};

// ── Performance line chart (port vs SPX)
const PerfChart = ({
  months,
  port,
  spx,
  height = 240
}) => {
  const W = 720,
    H = height,
    padL = 36,
    padR = 20,
    padT = 24,
    padB = 32;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const all = [...port, ...spx];
  const maxV = Math.max(...all) * 1.15;
  const minV = Math.min(0, Math.min(...all));
  const span = maxV - minV;
  const px = i => padL + i / (months.length - 1) * innerW;
  const py = v => padT + (1 - (v - minV) / span) * innerH;
  const path = arr => arr.map((v, i) => `${i === 0 ? 'M' : 'L'} ${px(i).toFixed(1)} ${py(v).toFixed(1)}`).join(' ');
  const area = arr => `${path(arr)} L ${px(arr.length - 1)} ${py(0)} L ${px(0)} ${py(0)} Z`;
  const ticksY = [0, maxV / 4, maxV / 2, 3 * maxV / 4, maxV].map(v => Math.round(v));
  const lastIdx = months.length - 1;
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: `0 0 ${W} ${H}`,
    className: "w-full h-auto",
    preserveAspectRatio: "xMidYMid meet"
  }, /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("linearGradient", {
    id: "gradPort",
    x1: "0",
    x2: "0",
    y1: "0",
    y2: "1"
  }, /*#__PURE__*/React.createElement("stop", {
    offset: "0%",
    stopColor: "#f5d04e",
    stopOpacity: "0.45"
  }), /*#__PURE__*/React.createElement("stop", {
    offset: "100%",
    stopColor: "#f5d04e",
    stopOpacity: "0"
  })), /*#__PURE__*/React.createElement("linearGradient", {
    id: "gradSpx",
    x1: "0",
    x2: "0",
    y1: "0",
    y2: "1"
  }, /*#__PURE__*/React.createElement("stop", {
    offset: "0%",
    stopColor: "#1c1b1a",
    stopOpacity: "0.08"
  }), /*#__PURE__*/React.createElement("stop", {
    offset: "100%",
    stopColor: "#1c1b1a",
    stopOpacity: "0"
  }))), ticksY.map((t, i) => /*#__PURE__*/React.createElement("g", {
    key: i
  }, /*#__PURE__*/React.createElement("line", {
    x1: padL,
    y1: py(t),
    x2: W - padR,
    y2: py(t),
    stroke: "#1c1b1a",
    strokeOpacity: ".06",
    strokeDasharray: "2 4"
  }), /*#__PURE__*/React.createElement("text", {
    x: padL - 8,
    y: py(t) + 3,
    textAnchor: "end",
    fontSize: "9",
    fontFamily: "JetBrains Mono",
    fill: "#9a958c"
  }, "+", t, "%"))), months.map((m, i) => /*#__PURE__*/React.createElement("text", {
    key: i,
    x: px(i),
    y: H - 10,
    textAnchor: "middle",
    fontSize: "10",
    fontFamily: "Manrope",
    fill: "#9a958c"
  }, m)), /*#__PURE__*/React.createElement("path", {
    d: area(spx),
    fill: "url(#gradSpx)"
  }), /*#__PURE__*/React.createElement("path", {
    d: path(spx),
    fill: "none",
    stroke: "#1c1b1a",
    strokeOpacity: ".55",
    strokeWidth: "1.5",
    strokeDasharray: "4 4"
  }), /*#__PURE__*/React.createElement("path", {
    d: area(port),
    fill: "url(#gradPort)"
  }), /*#__PURE__*/React.createElement("path", {
    d: path(port),
    fill: "none",
    stroke: "#caa01a",
    strokeWidth: "2.2"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: px(lastIdx),
    cy: py(port[lastIdx]),
    r: "5",
    fill: "#f5d04e",
    stroke: "#caa01a",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: px(lastIdx),
    cy: py(spx[lastIdx]),
    r: "4",
    fill: "#fff",
    stroke: "#1c1b1a",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("g", null, /*#__PURE__*/React.createElement("rect", {
    x: px(lastIdx) - 4,
    y: py(port[lastIdx]) - 26,
    width: "56",
    height: "18",
    rx: "9",
    fill: "#1c1b1a"
  }), /*#__PURE__*/React.createElement("text", {
    x: px(lastIdx) + 24,
    y: py(port[lastIdx]) - 13,
    textAnchor: "middle",
    fontSize: "11",
    fontFamily: "JetBrains Mono",
    fontWeight: "500",
    fill: "#f5d04e"
  }, "+", port[lastIdx], "%"), /*#__PURE__*/React.createElement("rect", {
    x: px(lastIdx) - 4,
    y: py(spx[lastIdx]) + 8,
    width: "56",
    height: "18",
    rx: "9",
    fill: "#fff",
    stroke: "#1c1b1a",
    strokeOpacity: ".15"
  }), /*#__PURE__*/React.createElement("text", {
    x: px(lastIdx) + 24,
    y: py(spx[lastIdx]) + 20,
    textAnchor: "middle",
    fontSize: "11",
    fontFamily: "JetBrains Mono",
    fontWeight: "500",
    fill: "#3a3833"
  }, "+", spx[lastIdx], "%")));
};

// ── Mini sparkline (used in KPI cards)
const Sparkline = ({
  points,
  color = '#1c1b1a',
  height = 36,
  width = 110,
  gradId = 'spark'
}) => {
  const padY = 4;
  const minV = Math.min(...points);
  const maxV = Math.max(...points);
  const span = maxV - minV || 1;
  const W = width,
    H = height;
  const px = i => i / (points.length - 1) * W;
  const py = v => padY + (1 - (v - minV) / span) * (H - padY * 2);
  const p = points.map((v, i) => `${i === 0 ? 'M' : 'L'} ${px(i).toFixed(1)} ${py(v).toFixed(1)}`).join(' ');
  const area = `${p} L ${W} ${H} L 0 ${H} Z`;
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: `0 0 ${W} ${H}`,
    className: "w-full h-full"
  }, /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("linearGradient", {
    id: gradId,
    x1: "0",
    x2: "0",
    y1: "0",
    y2: "1"
  }, /*#__PURE__*/React.createElement("stop", {
    offset: "0%",
    stopColor: color,
    stopOpacity: "0.25"
  }), /*#__PURE__*/React.createElement("stop", {
    offset: "100%",
    stopColor: color,
    stopOpacity: "0"
  }))), /*#__PURE__*/React.createElement("path", {
    d: area,
    fill: `url(#${gradId})`
  }), /*#__PURE__*/React.createElement("path", {
    d: p,
    fill: "none",
    stroke: color,
    strokeWidth: "1.6",
    strokeLinecap: "round"
  }));
};

// ── Animated counter (count-up on mount/visible)
const Counter = ({
  value,
  decimals = 0,
  prefix = '',
  suffix = '',
  duration = 900,
  className = ''
}) => {
  const [v, setV] = React.useState(0);
  React.useEffect(() => {
    let raf, t0;
    const tick = t => {
      if (!t0) t0 = t;
      const k = Math.min(1, (t - t0) / duration);
      const e = 1 - Math.pow(1 - k, 3); // easeOutCubic
      setV(value * e);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);
  const formatted = Math.abs(v) >= 1000 ? v.toLocaleString('ru-RU', {
    maximumFractionDigits: decimals
  }) : v.toFixed(decimals);
  return /*#__PURE__*/React.createElement("span", {
    className: `num ${className}`
  }, prefix, formatted, suffix);
};
Object.assign(window, {
  RiskGauge,
  Waterfall,
  SectorBar,
  PerfChart,
  Sparkline,
  Counter
});
/* Overview section: hero + factor pills + asymmetric main grid */

// ── Factor pill (mini progress with chip label) — echoes the reference "Interviews/Hired" rhythm
const FactorPill = ({
  label,
  value,
  display,
  cap = 100,
  accent = 'gold',
  warn,
  suffix
}) => {
  const pct = Math.min(100, value / cap * 100);
  const bgMap = {
    gold: 'bg-ink-900',
    // dark pill, gold fill
    dark: 'bg-ink-900',
    sage: 'bg-ink-900',
    mute: 'bg-cream-200'
  };
  const fillMap = {
    gold: 'bg-gold-400',
    dark: 'bg-ink-900',
    sage: 'bg-sage-500',
    mute: 'bg-ink-700'
  };
  const textColor = accent === 'mute' ? 'text-ink-900' : 'text-white';
  const valTxt = display || `${value}${suffix || '%'}`;
  return /*#__PURE__*/React.createElement("div", {
    className: "flex flex-col gap-2 min-w-[120px]"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[12px] text-ink-500 font-medium tracking-tight"
  }, label), warn && /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-rust-500"
  })), /*#__PURE__*/React.createElement("div", {
    className: `relative h-7 rounded-full overflow-hidden ${accent === 'mute' ? 'bg-cream-200/70' : 'bg-ink-900/85'}`
  }, /*#__PURE__*/React.createElement("div", {
    className: `absolute inset-y-0 left-0 ${fillMap[accent]} rounded-full`,
    style: {
      width: `${pct}%`
    }
  }), /*#__PURE__*/React.createElement("span", {
    className: `absolute inset-0 flex items-center pl-3 text-[11px] font-semibold num
                          ${accent === 'mute' ? 'text-ink-900' : 'text-white mix-blend-difference'}`
  }, valTxt)));
};

// ── Hero stat (big number with thin label, e.g. "9 Позиции")
const HeroStat = ({
  value,
  label,
  IconC
}) => /*#__PURE__*/React.createElement("div", {
  className: "flex flex-col items-center gap-1 px-4"
}, /*#__PURE__*/React.createElement(IconC, {
  size: 18,
  className: "text-ink-500 mb-1",
  stroke: 1.4
}), /*#__PURE__*/React.createElement("div", {
  className: "text-5xl font-light tracking-tight num leading-none text-ink-900"
}, value), /*#__PURE__*/React.createElement("div", {
  className: "text-[12px] text-ink-500 font-medium tracking-tight"
}, label));

// ── Top hotspot card (tall, replaces "profile photo" card from reference)
const TopHotspotCard = ({
  h
}) => /*#__PURE__*/React.createElement("div", {
  className: "relative rounded-4xl overflow-hidden shadow-card-lg lift h-full min-h-[420px]",
  style: {
    background: 'linear-gradient(155deg, #2a2825 0%, #1c1b1a 45%, #3a2f1e 100%)'
  }
}, /*#__PURE__*/React.createElement("div", {
  className: "absolute -right-6 -top-6 text-[260px] font-black leading-none text-white/[0.04] select-none num"
}, h.ticker.slice(0, 2)), /*#__PURE__*/React.createElement("div", {
  className: "absolute -bottom-24 -right-12 w-80 h-80 rounded-full",
  style: {
    background: 'radial-gradient(circle, rgba(245,208,78,0.55), transparent 60%)'
  }
}), /*#__PURE__*/React.createElement("div", {
  className: "absolute top-5 left-5 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-gold-400 text-ink-900 text-[10px] font-bold tracking-wider uppercase"
}, /*#__PURE__*/React.createElement(Icons.Warning, {
  size: 11,
  stroke: 2
}), " Hotspot"), /*#__PURE__*/React.createElement("div", {
  className: "absolute top-5 right-5 text-white/40 text-[10px] font-mono tracking-widest"
}, "RISK · ", h.riskShare, "%"), /*#__PURE__*/React.createElement("div", {
  className: "absolute inset-0 flex items-center justify-center pt-2"
}, /*#__PURE__*/React.createElement("div", {
  className: "text-white num font-light tracking-tight",
  style: {
    fontSize: 96,
    letterSpacing: '-0.04em'
  }
}, h.ticker)), /*#__PURE__*/React.createElement("div", {
  className: "absolute left-5 right-5 bottom-5"
}, /*#__PURE__*/React.createElement("div", {
  className: "rounded-3xl p-4 backdrop-blur-md",
  style: {
    background: 'rgba(255,255,255,0.08)',
    border: '1px solid rgba(255,255,255,0.08)'
  }
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-start justify-between gap-3"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-white text-[15px] font-semibold tracking-tight leading-tight"
}, h.name), /*#__PURE__*/React.createElement("div", {
  className: "text-white/50 text-[11px] mt-0.5"
}, h.sector)), /*#__PURE__*/React.createElement("div", {
  className: "px-3 py-1 rounded-full bg-gold-400/95 text-ink-900 text-[10px] font-bold tracking-wider"
}, h.signal)), /*#__PURE__*/React.createElement("div", {
  className: "grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-white/10"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-white/40 text-[10px] tracking-wider uppercase"
}, "Вес"), /*#__PURE__*/React.createElement("div", {
  className: "text-white text-lg font-medium num"
}, h.weight, "%")), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-white/40 text-[10px] tracking-wider uppercase"
}, "P/L"), /*#__PURE__*/React.createElement("div", {
  className: "text-gold-400 text-lg font-medium num"
}, "+", h.pnlPct, "%")), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-white/40 text-[10px] tracking-wider uppercase"
}, "USD"), /*#__PURE__*/React.createElement("div", {
  className: "text-white text-lg font-medium num"
}, "+$", (h.pnlUsd / 1000).toFixed(2), "K"))), /*#__PURE__*/React.createElement("div", {
  className: "mt-3 text-white/60 text-[11px] leading-snug"
}, h.note))));

// ── Risk gauge card
const RiskGaugeCard = ({
  value,
  delta
}) => /*#__PURE__*/React.createElement("div", {
  className: "glass-strong rounded-4xl p-6 shadow-card lift h-full min-h-[420px] flex flex-col"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-center justify-between"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-ink-500 text-[12px] font-medium"
}, "Индекс риска"), /*#__PURE__*/React.createElement("h3", {
  className: "text-2xl font-semibold tracking-tight text-ink-900 leading-tight"
}, "Сводный 0–100")), /*#__PURE__*/React.createElement("button", {
  className: "w-9 h-9 rounded-full bg-ink-900/5 hover:bg-ink-900/10 flex items-center justify-center text-ink-700 transition",
  "aria-label": "open"
}, /*#__PURE__*/React.createElement(Icons.ArrowUR, {
  size: 16
}))), /*#__PURE__*/React.createElement("div", {
  className: "flex-1 flex items-center justify-center -mt-2"
}, /*#__PURE__*/React.createElement(RiskGauge, {
  value: value,
  size: 240
})), /*#__PURE__*/React.createElement("div", {
  className: "flex items-center justify-between gap-2"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-2 px-3 py-1.5 rounded-full bg-cream-50 border border-ink-900/5 text-[11px] text-ink-700"
}, /*#__PURE__*/React.createElement(Icons.Pulse, {
  size: 13,
  stroke: 1.8
}), /*#__PURE__*/React.createElement("span", null, "За месяц ", delta > 0 ? '+' : '', delta, " пт")), /*#__PURE__*/React.createElement("button", {
  className: "flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-ink-900 text-white text-[11px] font-medium hover:bg-ink-800 transition"
}, /*#__PURE__*/React.createElement(Icons.Refresh, {
  size: 12,
  stroke: 2
}), " Пересчитать")));

// ── Risk decomposition card
const RiskDecompCard = ({
  data
}) => /*#__PURE__*/React.createElement("div", {
  className: "glass-strong rounded-4xl p-6 shadow-card lift h-full min-h-[420px] flex flex-col"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-start justify-between"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-ink-500 text-[12px] font-medium"
}, "Декомпозиция"), /*#__PURE__*/React.createElement("h3", {
  className: "text-2xl font-semibold tracking-tight text-ink-900 leading-tight"
}, "Риск портфеля")), /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-2"
}, /*#__PURE__*/React.createElement("span", {
  className: "px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5 text-[10px] font-mono tracking-wider text-ink-700"
}, "14.2%"), /*#__PURE__*/React.createElement("button", {
  className: "w-9 h-9 rounded-full bg-ink-900/5 hover:bg-ink-900/10 flex items-center justify-center text-ink-700 transition"
}, /*#__PURE__*/React.createElement(Icons.ArrowUR, {
  size: 16
})))), /*#__PURE__*/React.createElement("div", {
  className: "flex-1 flex items-center -mx-1"
}, /*#__PURE__*/React.createElement(Waterfall, {
  data: data
})), /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-4 text-[11px] text-ink-500 pt-2 border-t border-ink-900/5"
}, /*#__PURE__*/React.createElement("span", {
  className: "flex items-center gap-1.5"
}, /*#__PURE__*/React.createElement("span", {
  className: "w-2 h-2 rounded-full bg-ink-900"
}), " отдельно"), /*#__PURE__*/React.createElement("span", {
  className: "flex items-center gap-1.5"
}, /*#__PURE__*/React.createElement("span", {
  className: "w-2 h-2 rounded-full bg-rust-500"
}), " диверсификация"), /*#__PURE__*/React.createElement("span", {
  className: "flex items-center gap-1.5"
}, /*#__PURE__*/React.createElement("span", {
  className: "w-2 h-2 rounded-full bg-gold-400"
}), " итог")));

// ── Sector mix card
const SectorMixCard = ({
  sectors
}) => /*#__PURE__*/React.createElement("div", {
  className: "glass-strong rounded-4xl p-6 shadow-card lift h-full flex flex-col min-h-[200px]"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-start justify-between mb-1"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-ink-500 text-[12px] font-medium"
}, "Структура"), /*#__PURE__*/React.createElement("h3", {
  className: "text-xl font-semibold tracking-tight text-ink-900 leading-tight"
}, "По секторам")), /*#__PURE__*/React.createElement("div", {
  className: "px-2.5 py-1 rounded-full bg-rust-500/12 text-rust-600 text-[10px] font-semibold tracking-wider uppercase flex items-center gap-1"
}, /*#__PURE__*/React.createElement(Icons.Warning, {
  size: 11,
  stroke: 2
}), " Перевес IT")), /*#__PURE__*/React.createElement("div", {
  className: "mt-3"
}, /*#__PURE__*/React.createElement(SectorBar, {
  sectors: sectors
})), /*#__PURE__*/React.createElement("div", {
  className: "grid grid-cols-2 gap-x-4 gap-y-2 mt-4"
}, sectors.map(s => {
  const IconC = sectorIcon(s.name);
  return /*#__PURE__*/React.createElement("div", {
    key: s.name,
    className: "flex items-center gap-2.5 group"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-6 h-6 rounded-lg flex items-center justify-center",
    style: {
      background: s.hue + '25'
    }
  }, /*#__PURE__*/React.createElement(IconC, {
    size: 12,
    className: "text-ink-700",
    stroke: 1.6
  })), /*#__PURE__*/React.createElement("div", {
    className: "flex-1 min-w-0"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[11px] text-ink-700 font-medium truncate"
  }, s.name)), /*#__PURE__*/React.createElement("div", {
    className: "text-[12px] font-semibold num text-ink-900"
  }, s.pct, "%"));
})));

// ── Quick AI insight (dark pill row, echoes the reference "Onboarding Task" mini cards style)
const AIInsightCard = ({
  verdict
}) => /*#__PURE__*/React.createElement("div", {
  className: "rounded-4xl p-6 shadow-dark lift h-full flex flex-col min-h-[200px]",
  style: {
    background: 'linear-gradient(160deg, #1c1b1a 0%, #2a2825 100%)'
  }
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-center justify-between"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-2 text-gold-400 text-[10px] font-mono tracking-widest uppercase"
}, /*#__PURE__*/React.createElement(Icons.Sparkles, {
  size: 13,
  stroke: 1.8
}), " AI · HAIKU"), /*#__PURE__*/React.createElement("button", {
  className: "w-8 h-8 rounded-full bg-white/5 hover:bg-white/10 flex items-center justify-center text-white/70 transition"
}, /*#__PURE__*/React.createElement(Icons.ArrowUR, {
  size: 14
}))), /*#__PURE__*/React.createElement("div", {
  className: "mt-4 text-white text-[15px] leading-relaxed font-light"
}, "Индекс ", /*#__PURE__*/React.createElement("span", {
  className: "text-gold-400 font-semibold num"
}, "62"), " — верх «умеренной» зоны. Портфель соответствует профилю, но почти весь риск собран в ", /*#__PURE__*/React.createElement("span", {
  className: "text-white font-semibold"
}, "2 бумагах"), " из 9."), /*#__PURE__*/React.createElement("div", {
  className: "mt-auto pt-4 flex items-center gap-2 flex-wrap"
}, /*#__PURE__*/React.createElement("span", {
  className: "px-2.5 py-1 rounded-full bg-white/8 text-white/70 text-[10px] font-mono tracking-wider"
}, "RAG: GS_Q2_2026"), /*#__PURE__*/React.createElement("span", {
  className: "px-2.5 py-1 rounded-full bg-white/8 text-white/70 text-[10px] font-mono tracking-wider"
}, "SEC EDGAR"), /*#__PURE__*/React.createElement("span", {
  className: "px-2.5 py-1 rounded-full bg-white/8 text-white/70 text-[10px] font-mono tracking-wider"
}, "MAC3")));

// ── Hero — verdict + factor pills + right stats
const Hero = () => {
  const p = window.PORTFOLIO;
  const v = p.verdict;
  return /*#__PURE__*/React.createElement("section", {
    id: "overview",
    className: "rise",
    "data-screen-label": "01 Overview"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-8 flex-wrap mb-6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex-1 min-w-[480px]"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-gold-400"
  }), "Portfolio Risk Report · Tier ", p.meta.tier), /*#__PURE__*/React.createElement("h1", {
    className: "text-[58px] leading-[1.02] tracking-[-0.03em] font-light text-ink-900 max-w-[860px]"
  }, v.headline, /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, ".")), /*#__PURE__*/React.createElement("p", {
    className: "text-[18px] text-ink-500 mt-3 max-w-[640px] font-light"
  }, v.sub)), /*#__PURE__*/React.createElement("div", {
    className: "flex items-end gap-2 pt-4 divide-x divide-ink-900/10"
  }, p.heroStats.map((s, i) => {
    const IconC = {
      briefcase: Icons.Briefcase,
      trendUp: Icons.TrendUp,
      wallet: Icons.Wallet
    }[s.icon];
    return /*#__PURE__*/React.createElement(HeroStat, {
      key: i,
      value: s.value,
      label: s.label,
      IconC: IconC
    });
  }))), /*#__PURE__*/React.createElement("div", {
    className: "flex items-end gap-6 flex-wrap mb-10"
  }, p.factorPills.map((f, i) => /*#__PURE__*/React.createElement(FactorPill, {
    key: i,
    ...f
  })), /*#__PURE__*/React.createElement("div", {
    className: "flex-1 min-w-[160px] flex flex-col gap-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[12px] text-ink-500 font-medium"
  }, "Прочие метрики"), /*#__PURE__*/React.createElement("div", {
    className: "h-7 rounded-full hatch border border-ink-900/8"
  }))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-12 gap-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-3"
  }, /*#__PURE__*/React.createElement(TopHotspotCard, {
    h: p.topHotspot
  })), /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 sm:col-span-6 lg:col-span-3"
  }, /*#__PURE__*/React.createElement(RiskDecompCard, {
    data: p.riskDecomp
  })), /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 sm:col-span-6 lg:col-span-3"
  }, /*#__PURE__*/React.createElement(RiskGaugeCard, {
    value: v.riskIndex,
    delta: v.riskTrendDelta
  })), /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-3 grid grid-rows-2 gap-5"
  }, /*#__PURE__*/React.createElement(SectorMixCard, {
    sectors: p.sectors
  }), /*#__PURE__*/React.createElement(AIInsightCard, {
    verdict: v
  }))));
};
Object.assign(window, {
  Hero
});
/* Holdings section — interactive expandable rows + filter chips */

const StatusBadge = ({
  status
}) => status === 'HOTSPOT' ? /*#__PURE__*/React.createElement("span", {
  className: "inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-gold-400 text-ink-900 text-[10px] font-bold tracking-wider uppercase"
}, /*#__PURE__*/React.createElement(Icons.Warning, {
  size: 10,
  stroke: 2.2
}), " Hotspot") : /*#__PURE__*/React.createElement("span", {
  className: "inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-cream-200/70 text-ink-700 text-[10px] font-semibold tracking-wider uppercase"
}, "Normal");
const SignalChip = ({
  signal
}) => {
  const styles = {
    'BUY': 'bg-sage-500/15 text-sage-600',
    'STRONG BUY': 'bg-sage-500 text-white',
    'HOLD': 'bg-ink-900/5 text-ink-700',
    'TRIM': 'bg-rust-500/15 text-rust-600',
    'SELL': 'bg-rust-500 text-white'
  }[signal] || 'bg-ink-900/5 text-ink-700';
  return /*#__PURE__*/React.createElement("span", {
    className: `inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wider ${styles}`
  }, signal);
};

// Mini horizontal bar showing weight or risk share
const MiniBar = ({
  value,
  max = 30,
  color = '#1c1b1a',
  height = 4
}) => {
  const pct = Math.min(100, value / max * 100);
  return /*#__PURE__*/React.createElement("div", {
    className: "w-full bg-ink-900/8 rounded-full overflow-hidden",
    style: {
      height
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "rounded-full",
    style: {
      width: `${pct}%`,
      height: '100%',
      background: color
    }
  }));
};
const HoldingRow = ({
  h,
  open,
  onToggle,
  idx
}) => {
  const IconC = sectorIcon(h.cls);
  const pos = h.pnlPct >= 0;
  return /*#__PURE__*/React.createElement("div", {
    className: `relative transition-colors ${open ? 'bg-cream-50/70' : 'hover:bg-cream-50/40'}`
  }, /*#__PURE__*/React.createElement("button", {
    onClick: onToggle,
    className: "w-full grid grid-cols-[40px_minmax(0,2fr)_minmax(0,1.6fr)_repeat(4,minmax(0,1fr))_88px_40px] items-center gap-3 px-6 py-4 text-left"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-9 h-9 rounded-2xl bg-cream-100 border border-ink-900/5 flex items-center justify-center text-ink-700"
  }, /*#__PURE__*/React.createElement(IconC, {
    size: 16,
    stroke: 1.6
  })), /*#__PURE__*/React.createElement("div", {
    className: "min-w-0"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[16px] font-bold tracking-tight num text-ink-900"
  }, h.t), /*#__PURE__*/React.createElement(StatusBadge, {
    status: h.status
  })), /*#__PURE__*/React.createElement("div", {
    className: "text-[12px] text-ink-500 truncate mt-0.5"
  }, h.name)), /*#__PURE__*/React.createElement("div", {
    className: "text-[12px] text-ink-700"
  }, h.cls), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[13px] font-semibold num text-ink-900"
  }, h.w.toFixed(1), "%"), /*#__PURE__*/React.createElement("div", {
    className: "mt-1.5"
  }, /*#__PURE__*/React.createElement(MiniBar, {
    value: h.w,
    max: 22,
    color: "#1c1b1a"
  }))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[13px] font-semibold num text-ink-900"
  }, h.risk.toFixed(1), "%"), /*#__PURE__*/React.createElement("div", {
    className: "mt-1.5"
  }, /*#__PURE__*/React.createElement(MiniBar, {
    value: h.risk,
    max: 26,
    color: h.status === 'HOTSPOT' ? '#f5d04e' : '#a8a293'
  }))), /*#__PURE__*/React.createElement("div", {
    className: "text-[13px] font-medium num text-ink-700"
  }, h.beta.toFixed(2)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: `text-[14px] font-semibold num ${pos ? 'text-sage-600' : 'text-rust-600'}`
  }, pos ? '+' : '', h.pnlPct.toFixed(1), "%"), /*#__PURE__*/React.createElement("div", {
    className: `text-[11px] num ${pos ? 'text-sage-500' : 'text-rust-500'}`
  }, pos ? '+' : '', "$", Math.abs(h.pnlUsd).toLocaleString('ru-RU'))), /*#__PURE__*/React.createElement("div", {
    className: "flex justify-end"
  }, /*#__PURE__*/React.createElement(SignalChip, {
    signal: h.signal
  })), /*#__PURE__*/React.createElement("div", {
    className: `w-9 h-9 rounded-full bg-ink-900/5 flex items-center justify-center text-ink-700 transition-transform ${open ? 'rotate-180' : ''}`
  }, /*#__PURE__*/React.createElement(Icons.Chevron, {
    size: 14
  }))), /*#__PURE__*/React.createElement("div", {
    className: "overflow-hidden transition-[max-height,opacity] duration-500 ease-out",
    style: {
      maxHeight: open ? 720 : 0,
      opacity: open ? 1 : 0
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "px-6 pb-6 pt-1"
  }, /*#__PURE__*/React.createElement("div", {
    className: "rounded-3xl p-5 bg-white/70 border border-ink-900/5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-4 mb-4"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[11px] tracking-widest uppercase text-ink-500 font-mono"
  }, "Фундаментал · SEC EDGAR"), /*#__PURE__*/React.createElement("div", {
    className: "text-[15px] text-ink-900 font-medium mt-0.5"
  }, h.name)), /*#__PURE__*/React.createElement("button", {
    className: "flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-ink-900 text-white text-[11px] font-medium hover:bg-ink-800 transition"
  }, "Открыть бумагу ", /*#__PURE__*/React.createElement(Icons.ArrowR, {
    size: 12,
    stroke: 2.2
  }))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-2 md:grid-cols-6 gap-3"
  }, /*#__PURE__*/React.createElement(FundCell, {
    label: "ROE",
    value: h.fund.roe
  }), /*#__PURE__*/React.createElement(FundCell, {
    label: "Маржа",
    value: h.fund.margin
  }), /*#__PURE__*/React.createElement(FundCell, {
    label: "Долг/А",
    value: h.fund.debt
  }), /*#__PURE__*/React.createElement(FundCell, {
    label: "Рост г/г",
    value: h.fund.growth
  }), /*#__PURE__*/React.createElement(FundCell, {
    label: "ATR",
    value: h.fund.atr
  }), /*#__PURE__*/React.createElement(FundCell, {
    label: "Beta",
    value: h.beta.toFixed(2)
  })), /*#__PURE__*/React.createElement("div", {
    className: "mt-4 flex items-start gap-3 text-[13px] text-ink-700 leading-relaxed"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 14,
    className: "text-gold-600 mt-1 flex-shrink-0",
    stroke: 1.8
  }), /*#__PURE__*/React.createElement("p", {
    className: "font-light"
  }, h.note))))));
};
const FundCell = ({
  label,
  value
}) => /*#__PURE__*/React.createElement("div", {
  className: "rounded-2xl bg-cream-50 border border-ink-900/5 px-3 py-2.5"
}, /*#__PURE__*/React.createElement("div", {
  className: "text-[10px] uppercase tracking-wider text-ink-500 font-mono"
}, label), /*#__PURE__*/React.createElement("div", {
  className: "text-[14px] font-semibold num text-ink-900 mt-0.5"
}, value));
const Holdings = () => {
  const [openIdx, setOpenIdx] = React.useState(0);
  const [filter, setFilter] = React.useState('Все');
  const all = window.PORTFOLIO.holdings;
  const filters = ['Все', 'HOTSPOT', 'Технологии', 'Защитные', 'Доходные', 'В минусе'];
  const rows = all.filter(h => {
    if (filter === 'Все') return true;
    if (filter === 'HOTSPOT') return h.status === 'HOTSPOT';
    if (filter === 'Технологии') return h.cls === 'Технологии';
    if (filter === 'Защитные') return ['Здравоохранение', 'Потреб. товары', 'Облигации'].includes(h.cls);
    if (filter === 'Доходные') return h.pnlPct >= 10;
    if (filter === 'В минусе') return h.pnlPct < 0;
    return true;
  });
  return /*#__PURE__*/React.createElement("section", {
    id: "holdings",
    className: "rise",
    "data-screen-label": "02 Holdings"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-end justify-between gap-4 flex-wrap mb-6"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-gold-400"
  }), " Holdings · 9 позиций"), /*#__PURE__*/React.createElement("h2", {
    className: "text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900"
  }, "Что вы держите", /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, ".")), /*#__PURE__*/React.createElement("p", {
    className: "text-[15px] text-ink-500 mt-2 font-light"
  }, "Нажмите на строку, чтобы увидеть фундаментал бумаги.")), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 flex-wrap"
  }, filters.map(f => /*#__PURE__*/React.createElement("button", {
    key: f,
    onClick: () => setFilter(f),
    className: `px-3.5 py-1.5 rounded-full text-[12px] font-medium transition-colors
                          ${filter === f ? 'bg-ink-900 text-white' : 'bg-white/60 text-ink-700 hover:bg-white border border-ink-900/8'}`
  }, f)))), /*#__PURE__*/React.createElement("div", {
    className: "glass-strong rounded-4xl shadow-card overflow-hidden"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mob-scroll-x"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-[40px_minmax(0,2fr)_minmax(0,1.6fr)_repeat(4,minmax(0,1fr))_88px_40px] items-center gap-3 px-6 py-3.5 border-b border-ink-900/6 text-[10px] tracking-widest uppercase text-ink-500 font-mono"
  }, /*#__PURE__*/React.createElement("div", null), /*#__PURE__*/React.createElement("div", null, "Тикер · Имя"), /*#__PURE__*/React.createElement("div", null, "Класс"), /*#__PURE__*/React.createElement("div", null, "Вес"), /*#__PURE__*/React.createElement("div", null, "Риск"), /*#__PURE__*/React.createElement("div", null, "Beta"), /*#__PURE__*/React.createElement("div", null, "P/L"), /*#__PURE__*/React.createElement("div", {
    className: "text-right"
  }, "Сигнал"), /*#__PURE__*/React.createElement("div", null)), /*#__PURE__*/React.createElement("div", {
    className: "divide-y divide-ink-900/5"
  }, rows.map((h, i) => /*#__PURE__*/React.createElement(HoldingRow, {
    key: h.t,
    h: h,
    idx: i,
    open: openIdx === i,
    onToggle: () => setOpenIdx(openIdx === i ? -1 : i)
  })), rows.length === 0 && /*#__PURE__*/React.createElement("div", {
    className: "px-6 py-12 text-center text-ink-500 text-[14px]"
  }, "Ничего не подходит под фильтр «", filter, "»."))))));
};
Object.assign(window, {
  Holdings
});
/* Performance section — chart vs S&P 500 + period stats */

const PerfSummaryCard = ({
  label,
  value,
  sub,
  accent,
  IconC
}) => /*#__PURE__*/React.createElement("div", {
  className: `rounded-3xl p-5 lift shadow-card flex flex-col gap-1
                   ${accent === 'dark' ? 'bg-ink-900 text-white' : accent === 'gold' ? 'bg-gold-400 text-ink-900' : 'glass-strong text-ink-900'}`
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-center justify-between"
}, /*#__PURE__*/React.createElement("span", {
  className: `text-[11px] tracking-widest uppercase font-mono ${accent === 'dark' ? 'text-white/50' : 'text-ink-500'}`
}, label), IconC && /*#__PURE__*/React.createElement(IconC, {
  size: 16,
  className: accent === 'dark' ? 'text-white/60' : 'text-ink-500',
  stroke: 1.6
})), /*#__PURE__*/React.createElement("div", {
  className: "text-[36px] leading-none font-light num tracking-tight mt-1"
}, value), /*#__PURE__*/React.createElement("div", {
  className: `text-[12px] ${accent === 'dark' ? 'text-white/55' : 'text-ink-500'} mt-1`
}, sub));
const PeriodRow = ({
  p,
  isMax
}) => /*#__PURE__*/React.createElement("div", {
  className: `flex items-center gap-4 px-5 py-3 rounded-2xl transition-colors hover:bg-cream-50
                   ${isMax ? 'bg-cream-50' : ''}`
}, /*#__PURE__*/React.createElement("div", {
  className: "w-16 text-[12px] font-medium text-ink-500"
}, p.label), /*#__PURE__*/React.createElement("div", {
  className: "flex-1 grid grid-cols-3 gap-2"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-2"
}, /*#__PURE__*/React.createElement("span", {
  className: "w-2 h-2 rounded-full bg-gold-400"
}), /*#__PURE__*/React.createElement("span", {
  className: "text-[14px] font-semibold num text-ink-900"
}, "+", p.p.toFixed(1), "%")), /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-2"
}, /*#__PURE__*/React.createElement("span", {
  className: "w-2 h-2 rounded-full bg-ink-900"
}), /*#__PURE__*/React.createElement("span", {
  className: "text-[14px] num text-ink-700"
}, "+", p.s.toFixed(1), "%")), /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-2 justify-end"
}, /*#__PURE__*/React.createElement("span", {
  className: "text-[12px] text-ink-500"
}, "Δ"), /*#__PURE__*/React.createElement("span", {
  className: "text-[14px] font-semibold num text-sage-600"
}, "+", p.d.toFixed(1), " пп"))));
const Performance = () => {
  const p = window.PORTFOLIO.performance;
  const [period, setPeriod] = React.useState('12 мес');
  const periods = ['1 мес', '3 мес', 'YTD', '6 мес', '12 мес'];
  return /*#__PURE__*/React.createElement("section", {
    id: "performance",
    className: "rise",
    "data-screen-label": "03 Performance"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-end justify-between gap-4 flex-wrap mb-6"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-gold-400"
  }), " Performance · 12 месяцев"), /*#__PURE__*/React.createElement("h2", {
    className: "text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900"
  }, "Рост против рынка", /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, ".")), /*#__PURE__*/React.createElement("p", {
    className: "text-[15px] text-ink-500 mt-2 font-light"
  }, "Накопленная доходность портфеля в сравнении с S&P 500.")), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-1 p-1 rounded-full bg-white/60 border border-ink-900/8 backdrop-blur-md"
  }, periods.map(pr => /*#__PURE__*/React.createElement("button", {
    key: pr,
    onClick: () => setPeriod(pr),
    className: `px-3.5 py-1.5 rounded-full text-[12px] font-medium transition-colors
                          ${period === pr ? 'bg-ink-900 text-white' : 'text-ink-700 hover:bg-ink-900/5'}`
  }, pr)))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-12 gap-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-8 glass-strong rounded-4xl shadow-card lift p-7"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between mb-4"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-ink-500 text-[12px] font-medium mb-1"
  }, "Накопленная доходность"), /*#__PURE__*/React.createElement("div", {
    className: "flex items-end gap-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[48px] leading-none font-light num text-ink-900"
  }, "+14.2", /*#__PURE__*/React.createElement("span", {
    className: "text-[28px] text-ink-500"
  }, "%")), /*#__PURE__*/React.createElement("span", {
    className: "px-2.5 py-1 rounded-full bg-sage-500/15 text-sage-600 text-[11px] font-semibold mb-1.5 flex items-center gap-1"
  }, /*#__PURE__*/React.createElement(Icons.TrendUp, {
    size: 11,
    stroke: 2.2
  }), " +5.1 пп vs S&P"))), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-1.5 text-[11px] text-ink-700"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-3 h-1 rounded-full bg-gold-400"
  }), " Ваш портфель"), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-1.5 text-[11px] text-ink-500"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-3 h-[2px] bg-ink-900",
    style: {
      backgroundImage: 'linear-gradient(to right, #1c1b1a 50%, transparent 50%)',
      backgroundSize: '4px 2px'
    }
  }), " S&P 500"))), /*#__PURE__*/React.createElement(PerfChart, {
    months: p.months,
    port: p.port,
    spx: p.spx
  })), /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-4 grid grid-cols-2 lg:grid-cols-1 gap-5"
  }, /*#__PURE__*/React.createElement(PerfSummaryCard, {
    label: "Доходность",
    value: "+14.2%",
    sub: "за 12 месяцев",
    accent: "gold",
    IconC: Icons.TrendUp
  }), /*#__PURE__*/React.createElement(PerfSummaryCard, {
    label: "Опережение",
    value: "+5.1пп",
    sub: "портфель быстрее рынка",
    accent: "dark",
    IconC: Icons.Bolt
  }), /*#__PURE__*/React.createElement("div", {
    className: "col-span-2 lg:col-span-1 grid grid-cols-2 gap-3"
  }, /*#__PURE__*/React.createElement(PerfSummaryCard, {
    label: "Волатильность",
    value: "14.8%",
    sub: "рынок 11.2%",
    accent: "light"
  }), /*#__PURE__*/React.createElement(PerfSummaryCard, {
    label: "S&P 500",
    value: "+9.1%",
    sub: "за 12 мес",
    accent: "light"
  }))), /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 glass-strong rounded-4xl shadow-card p-6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between mb-3 px-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[13px] font-semibold text-ink-900"
  }, "Разбивка по периодам"), /*#__PURE__*/React.createElement("div", {
    className: "text-[11px] text-ink-500 font-mono"
  }, "Портфель / S&P 500 / Опережение")), /*#__PURE__*/React.createElement("div", {
    className: "space-y-1"
  }, p.periods.map(pr => /*#__PURE__*/React.createElement(PeriodRow, {
    key: pr.label,
    p: pr,
    isMax: pr.label === '12 мес'
  }))))));
};
Object.assign(window, {
  Performance
});
/* AI Ideas section — 4 actionable ideas with expandable details */

const catTone = cat => ({
  'Снижение риска': {
    chip: 'bg-rust-500/15 text-rust-600',
    dot: 'bg-rust-500',
    icon: Icons.Shield
  },
  'Диверсификация': {
    chip: 'bg-sage-500/15 text-sage-600',
    dot: 'bg-sage-500',
    icon: Icons.Layers
  },
  'Ребалансировка': {
    chip: 'bg-ink-900/8 text-ink-700',
    dot: 'bg-ink-900',
    icon: Icons.Refresh
  },
  'Увеличение риска': {
    chip: 'bg-gold-500/20 text-gold-700',
    dot: 'bg-gold-500',
    icon: Icons.TrendUp
  }
})[cat] || {
  chip: 'bg-ink-900/8 text-ink-700',
  dot: 'bg-ink-900',
  icon: Icons.Sparkles
};
const TickerCard = ({
  ticker,
  why,
  dark
}) => /*#__PURE__*/React.createElement("div", {
  className: `rounded-2xl p-3.5 border transition group cursor-pointer ${dark ? 'bg-white/8 border-white/10' : 'bg-cream-50 border-ink-900/5 hover:border-ink-900/15'}`
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-center justify-between mb-1"
}, /*#__PURE__*/React.createElement("span", {
  className: `text-[14px] font-bold num tracking-tight ${dark ? 'text-white' : 'text-ink-900'}`
}, ticker), /*#__PURE__*/React.createElement("span", {
  className: `text-[10px] font-mono uppercase tracking-wider opacity-0 group-hover:opacity-100 transition ${dark ? 'text-white/40' : 'text-ink-400'}`
}, "view")), /*#__PURE__*/React.createElement("p", {
  className: `text-[11.5px] leading-snug font-light ${dark ? 'text-white/70' : 'text-ink-500'}`
}, why));
const PipelineNode = ({
  label,
  value,
  last
}) => /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
  className: "flex-1 rounded-2xl p-3 bg-white/70 border border-ink-900/5"
}, /*#__PURE__*/React.createElement("div", {
  className: "text-[9px] tracking-widest uppercase text-ink-400 font-mono mb-1"
}, label), /*#__PURE__*/React.createElement("div", {
  className: "text-[12px] text-ink-900 font-medium leading-tight"
}, value)), !last && /*#__PURE__*/React.createElement("div", {
  className: "flex items-center justify-center w-7 flex-shrink-0"
}, /*#__PURE__*/React.createElement(Icons.ArrowR, {
  size: 14,
  className: "text-ink-400",
  stroke: 2
})));
const IdeaCard = ({
  idea,
  open,
  onToggle,
  isHighlight
}) => {
  const tone = catTone(idea.cat);
  const IconC = tone.icon;
  return /*#__PURE__*/React.createElement("div", {
    className: `rounded-4xl shadow-card lift overflow-hidden
                     ${isHighlight ? 'text-white' : 'glass-strong'}`,
    style: isHighlight ? {
      background: 'linear-gradient(155deg, #2a2825 0%, #1c1b1a 100%)'
    } : {}
  }, /*#__PURE__*/React.createElement("div", {
    className: "p-6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: `w-11 h-11 rounded-2xl flex items-center justify-center
                            ${isHighlight ? 'bg-gold-400 text-ink-900' : 'bg-ink-900/5 text-ink-900'}`
  }, /*#__PURE__*/React.createElement(IconC, {
    size: 18,
    stroke: 1.7
  })), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: `text-[10px] tracking-widest uppercase font-mono mb-0.5
                              ${isHighlight ? 'text-white/50' : 'text-ink-400'}`
  }, "Идея ", idea.n, " · ", idea.cat), /*#__PURE__*/React.createElement("div", {
    className: `flex items-center gap-1 ${isHighlight ? 'text-gold-400' : 'text-rust-600'}`
  }, /*#__PURE__*/React.createElement("span", {
    className: `w-1.5 h-1.5 rounded-full ${tone.dot}`
  }), /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] font-semibold tracking-wide"
  }, idea.prio)))), /*#__PURE__*/React.createElement("button", {
    onClick: onToggle,
    className: `w-9 h-9 rounded-full flex items-center justify-center transition
                              ${isHighlight ? 'bg-white/10 text-white hover:bg-white/20' : 'bg-ink-900/5 text-ink-700 hover:bg-ink-900/10'}
                              ${open ? 'rotate-180' : ''}`
  }, /*#__PURE__*/React.createElement(Icons.Chevron, {
    size: 15
  }))), /*#__PURE__*/React.createElement("h3", {
    className: `mt-5 text-[22px] leading-[1.15] tracking-tight font-medium
                        ${isHighlight ? 'text-white' : 'text-ink-900'}`
  }, idea.title), /*#__PURE__*/React.createElement("p", {
    className: `mt-3 text-[13.5px] leading-relaxed font-light
                        ${isHighlight ? 'text-white/70' : 'text-ink-500'}`
  }, idea.lede), /*#__PURE__*/React.createElement("div", {
    className: "mt-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: `text-[10px] tracking-widest uppercase font-mono mb-2
                          ${isHighlight ? 'text-white/40' : 'text-ink-400'}`
  }, "Идеи по замене — нет в портфеле"), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-3 gap-2"
  }, idea.tickers.map(t => /*#__PURE__*/React.createElement("div", {
    key: t.t,
    className: `rounded-xl px-3 py-2 text-center transition
                              ${isHighlight ? 'bg-white/8 hover:bg-white/15 text-white' : 'bg-cream-50 border border-ink-900/5 hover:border-ink-900/15 text-ink-900'}`
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[13px] font-bold num"
  }, t.t)))))), /*#__PURE__*/React.createElement("div", {
    className: "overflow-hidden transition-[max-height,opacity] duration-500 ease-out",
    style: {
      maxHeight: open ? 900 : 0,
      opacity: open ? 1 : 0
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: `px-6 pb-6 pt-2 space-y-5
                         ${isHighlight ? 'border-t border-white/10' : 'border-t border-ink-900/6'}`
  }, /*#__PURE__*/React.createElement("div", {
    className: "pt-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: `text-[10px] tracking-widest uppercase font-mono mb-2.5
                            ${isHighlight ? 'text-white/40' : 'text-ink-400'}`
  }, "Конвейер · Factor → Regime → RAG"), /*#__PURE__*/React.createElement("div", {
    className: `flex items-center ${isHighlight ? '[&_div.rounded-2xl]:!bg-white/8 [&_div.rounded-2xl]:!border-white/10 [&_.text-ink-900]:!text-white [&_.text-ink-400]:!text-white/40 [&_svg]:!text-white/40' : ''}`
  }, /*#__PURE__*/React.createElement(PipelineNode, {
    label: "Factor",
    value: idea.pipeline[0]
  }), /*#__PURE__*/React.createElement(PipelineNode, {
    label: "Regime",
    value: idea.pipeline[1]
  }), /*#__PURE__*/React.createElement(PipelineNode, {
    label: "RAG",
    value: idea.pipeline[2],
    last: true
  }))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: `text-[10px] tracking-widest uppercase font-mono mb-2.5
                            ${isHighlight ? 'text-white/40' : 'text-ink-400'}`
  }, "Почему именно эти бумаги"), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 md:grid-cols-3 gap-2.5"
  }, idea.tickers.map(t => /*#__PURE__*/React.createElement(TickerCard, {
    key: t.t,
    ticker: t.t,
    why: t.why,
    dark: isHighlight
  })))), /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-4 flex-wrap"
  }, /*#__PURE__*/React.createElement("div", {
    className: `rounded-2xl p-4 flex-1 min-w-[260px]
                            ${isHighlight ? 'bg-gold-400/10 border border-gold-400/30' : 'bg-gold-400/15 border border-gold-400/40'}`
  }, /*#__PURE__*/React.createElement("div", {
    className: `text-[10px] tracking-widest uppercase font-mono mb-1.5
                              ${isHighlight ? 'text-gold-400' : 'text-gold-700'}`
  }, "Ожидаемый эффект"), /*#__PURE__*/React.createElement("ul", {
    className: "space-y-1"
  }, idea.effect.map((e, i) => /*#__PURE__*/React.createElement("li", {
    key: i,
    className: `text-[12.5px] font-medium leading-tight
                                          ${isHighlight ? 'text-white' : 'text-ink-900'}`
  }, e)))), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-1.5 items-start pt-1"
  }, idea.sources.map(s => /*#__PURE__*/React.createElement("span", {
    key: s,
    className: `px-2.5 py-1 rounded-full text-[10px] font-mono tracking-wider
                                          ${isHighlight ? 'bg-white/8 text-white/60' : 'bg-ink-900/5 text-ink-700'}`
  }, s)))))));
};
const Ideas = () => {
  const ideas = window.PORTFOLIO.ideas;
  const [open, setOpen] = React.useState({
    '01': true
  });
  const toggle = n => setOpen({
    ...open,
    [n]: !open[n]
  });
  return /*#__PURE__*/React.createElement("section", {
    id: "ideas",
    className: "rise",
    "data-screen-label": "04 Ideas"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-end justify-between gap-4 flex-wrap mb-6"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-gold-400"
  }), " AI Ideas · 4 идеи"), /*#__PURE__*/React.createElement("h2", {
    className: "text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900"
  }, "Идеи на основе данных", /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, ".")), /*#__PURE__*/React.createElement("p", {
    className: "text-[15px] text-ink-500 mt-2 font-light max-w-[640px]"
  }, "На основе портфеля, отчётности SEC и обзоров инвестбанков. Нажмите на карточку, чтобы раскрыть детали.")), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("button", {
    className: "flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-white/60 border border-ink-900/8 text-ink-700 text-[12px] font-medium hover:bg-white transition"
  }, /*#__PURE__*/React.createElement(Icons.Download, {
    size: 13,
    stroke: 1.8
  }), " Экспорт PDF"), /*#__PURE__*/React.createElement("button", {
    className: "flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-ink-900 text-white text-[12px] font-medium hover:bg-ink-800 transition"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 13,
    stroke: 1.8
  }), " Применить идею"))), /*#__PURE__*/React.createElement("div", {
    className: "rounded-4xl p-6 mb-6 relative overflow-hidden",
    style: {
      background: 'linear-gradient(120deg, #fbf3d9 0%, #f6ebc0 100%)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "absolute -right-6 top-1/2 -translate-y-1/2 w-40 h-40 rounded-full opacity-30",
    style: {
      background: 'radial-gradient(circle, #caa01a, transparent 65%)'
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "relative flex items-start gap-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-11 h-11 rounded-2xl bg-ink-900 text-gold-400 flex items-center justify-center flex-shrink-0"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 18,
    stroke: 1.7
  })), /*#__PURE__*/React.createElement("div", {
    className: "flex-1"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] tracking-widest uppercase font-mono text-ink-700 mb-1"
  }, "AI · ", window.PORTFOLIO.meta.aiModel, " · сводка"), /*#__PURE__*/React.createElement("p", {
    className: "text-[15px] text-ink-900 leading-relaxed font-light"
  }, "Главная мысль одной строкой: ", /*#__PURE__*/React.createElement("span", {
    className: "font-medium"
  }, "портфель зарабатывает"), ", но риск собран в одном углу — ", /*#__PURE__*/React.createElement("span", {
    className: "font-medium num"
  }, "62%"), " в технологиях. Идеи ниже — как сохранить рост и сделать портфель устойчивее к просадке.")))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 lg:grid-cols-2 gap-5"
  }, ideas.map((idea, i) => /*#__PURE__*/React.createElement(IdeaCard, {
    key: idea.n,
    idea: idea,
    open: !!open[idea.n],
    onToggle: () => toggle(idea.n),
    isHighlight: i === 0
  }))), /*#__PURE__*/React.createElement("div", {
    className: "mt-8 rounded-3xl p-5 bg-white/40 border border-ink-900/6 flex items-start gap-3"
  }, /*#__PURE__*/React.createElement(Icons.Warning, {
    size: 16,
    className: "text-rust-500 mt-0.5 flex-shrink-0",
    stroke: 1.8
  }), /*#__PURE__*/React.createElement("p", {
    className: "text-[12.5px] text-ink-500 leading-relaxed font-light"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-ink-700 font-medium"
  }, "Это аналитические идеи, а не инвестиционная рекомендация."), " Расчёты основаны на исторических данных и публичной отчётности; они не учитывают вашу налоговую ситуацию, горизонт и цели. Окончательное решение — за вами или вашим финансовым консультантом.")));
};
Object.assign(window, {
  Ideas
});
/* App shell — sticky topbar with section pills, mounted sections, footer */

const NAV = [{
  id: 'overview',
  label: 'Обзор',
  short: '01'
}, {
  id: 'holdings',
  label: 'Бумаги',
  short: '02'
}, {
  id: 'performance',
  label: 'Доходность',
  short: '03'
}, {
  id: 'ideas',
  label: 'Идеи ИИ',
  short: '04'
}];
const TopBar = ({
  active,
  setActive
}) => {
  const meta = window.PORTFOLIO.meta;
  const onJump = id => {
    setActive(id);
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
  };
  return /*#__PURE__*/React.createElement("header", {
    className: "sticky top-0 z-40 px-6 pt-5 pb-3 backdrop-blur-md",
    style: {
      background: 'linear-gradient(to bottom, rgba(251,248,241,0.85) 0%, rgba(251,248,241,0.65) 70%, transparent 100%)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "max-w-[1480px] mx-auto flex items-center justify-between gap-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 px-4 py-2 rounded-full bg-white/80 border border-ink-900/8 shadow-sm"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-5 h-5 rounded-md bg-ink-900 flex items-center justify-center"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-2 h-2 rounded-sm bg-gold-400"
  })), /*#__PURE__*/React.createElement("span", {
    className: "font-bold tracking-tight text-[14px] text-ink-900"
  }, "BASE"), /*#__PURE__*/React.createElement("span", {
    className: "px-1.5 py-0.5 rounded-md bg-gold-400/30 text-gold-700 text-[9px] font-mono font-bold tracking-wider"
  }, "TIER"))), /*#__PURE__*/React.createElement("nav", {
    className: "hidden md:flex items-center gap-1 p-1 rounded-full bg-white/70 border border-ink-900/8 backdrop-blur-md"
  }, NAV.map(n => /*#__PURE__*/React.createElement("button", {
    key: n.id,
    onClick: () => onJump(n.id),
    className: `flex items-center gap-2 px-4 py-2 rounded-full text-[13px] font-medium transition-colors
                          ${active === n.id ? 'bg-ink-900 text-white' : 'text-ink-700 hover:bg-ink-900/5'}`
  }, /*#__PURE__*/React.createElement("span", {
    className: `text-[10px] font-mono opacity-60 ${active === n.id ? 'text-gold-400' : ''}`
  }, n.short), n.label))), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("button", {
    className: "hidden sm:flex items-center gap-2 pl-3 pr-4 py-2 rounded-full bg-white/70 border border-ink-900/8 text-[12px] text-ink-700 hover:bg-white transition"
  }, /*#__PURE__*/React.createElement(Icons.Search, {
    size: 14,
    stroke: 1.8
  }), " ", /*#__PURE__*/React.createElement("span", {
    className: "hidden lg:inline"
  }, "Найти бумагу…")), /*#__PURE__*/React.createElement("button", {
    className: "w-9 h-9 rounded-full bg-white/70 border border-ink-900/8 flex items-center justify-center text-ink-700 hover:bg-white transition relative"
  }, /*#__PURE__*/React.createElement(Icons.Bell, {
    size: 15,
    stroke: 1.8
  }), /*#__PURE__*/React.createElement("span", {
    className: "absolute top-2 right-2 w-1.5 h-1.5 rounded-full bg-rust-500"
  })), /*#__PURE__*/React.createElement("button", {
    className: "w-9 h-9 rounded-full bg-white/70 border border-ink-900/8 flex items-center justify-center text-ink-700 hover:bg-white transition"
  }, /*#__PURE__*/React.createElement(Icons.Settings, {
    size: 15,
    stroke: 1.7
  })), /*#__PURE__*/React.createElement("div", {
    className: "w-9 h-9 rounded-full overflow-hidden border border-ink-900/8 flex-shrink-0",
    style: {
      background: 'linear-gradient(135deg, #f5d04e 0%, #caa01a 100%)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-full h-full flex items-center justify-center text-ink-900 text-[11px] font-bold"
  }, "YК")))), /*#__PURE__*/React.createElement("div", {
    className: "max-w-[1480px] mx-auto mt-3 flex items-center justify-between text-[10px] font-mono tracking-wider text-ink-400 uppercase gap-3 flex-wrap"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-3 flex-wrap"
  }, /*#__PURE__*/React.createElement("span", {
    className: "flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-sage-500 animate-pulse"
  }), " Risk Engine · ", meta.engine), /*#__PURE__*/React.createElement("span", {
    className: "opacity-30"
  }, "/"), /*#__PURE__*/React.createElement("span", null, "ID · ", meta.id), /*#__PURE__*/React.createElement("span", {
    className: "opacity-30"
  }, "/"), /*#__PURE__*/React.createElement("span", null, "Session · ", meta.session)), /*#__PURE__*/React.createElement("span", null, "Generated ", meta.generated)));
};
const Footer = () => /*#__PURE__*/React.createElement("footer", {
  className: "mt-16 mb-8"
}, /*#__PURE__*/React.createElement("div", {
  className: "rounded-4xl p-7 glass-strong shadow-card flex items-start justify-between gap-6 flex-wrap"
}, /*#__PURE__*/React.createElement("div", {
  className: "max-w-[520px]"
}, /*#__PURE__*/React.createElement("div", {
  className: "text-[10px] tracking-widest uppercase text-ink-400 font-mono mb-2"
}, "Источники данных"), /*#__PURE__*/React.createElement("p", {
  className: "text-[13px] text-ink-700 leading-relaxed font-light"
}, /*#__PURE__*/React.createElement("span", {
  className: "font-medium"
}, "SEC EDGAR"), " · публичная отчётность · ", /*#__PURE__*/React.createElement("span", {
  className: "font-medium"
}, "Quant Engine MAC3"), " · расчёт риска и сигналов ·", /*#__PURE__*/React.createElement("span", {
  className: "font-medium"
}, " Factor Engine"), " · ", /*#__PURE__*/React.createElement("span", {
  className: "font-medium"
}, "Regime Model"), " · ", /*#__PURE__*/React.createElement("span", {
  className: "font-medium"
}, "Goldman Sachs"), " Q2 2026 · ", /*#__PURE__*/React.createElement("span", {
  className: "font-medium"
}, "Morgan Stanley"), " Tech Outlook 2026 · ", /*#__PURE__*/React.createElement("span", {
  className: "font-medium"
}, "JPMorgan"), " Strategy Q2 2026.")), /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-2 flex-wrap"
}, /*#__PURE__*/React.createElement("button", {
  className: "flex items-center gap-1.5 px-4 py-2 rounded-full bg-white border border-ink-900/8 text-ink-700 text-[12px] font-medium hover:bg-cream-50 transition"
}, /*#__PURE__*/React.createElement(Icons.Download, {
  size: 13,
  stroke: 1.8
}), " Скачать PDF"), /*#__PURE__*/React.createElement("button", {
  className: "flex items-center gap-1.5 px-4 py-2 rounded-full bg-white border border-ink-900/8 text-ink-700 text-[12px] font-medium hover:bg-cream-50 transition"
}, /*#__PURE__*/React.createElement(Icons.Share, {
  size: 13,
  stroke: 1.8
}), " Поделиться"), /*#__PURE__*/React.createElement("button", {
  className: "flex items-center gap-1.5 px-4 py-2 rounded-full bg-ink-900 text-white text-[12px] font-medium hover:bg-ink-800 transition"
}, /*#__PURE__*/React.createElement(Icons.Sparkles, {
  size: 13,
  stroke: 1.8
}), " Новый расчёт"))), /*#__PURE__*/React.createElement("div", {
  className: "text-center text-[10px] tracking-widest uppercase text-ink-400 font-mono mt-6"
}, "Portfolio Risk Report · BASE Tier · ", window.PORTFOLIO.meta.id, " · v2026.5"));

// Floating action button (echoes reference pattern — bottom right luxury chip)
const Fab = () => /*#__PURE__*/React.createElement("div", {
  className: "fixed bottom-6 right-6 z-50 flex flex-col items-end gap-2"
}, /*#__PURE__*/React.createElement("button", {
  title: "Пересчитать риск",
  className: "w-14 h-14 rounded-full bg-ink-900 text-gold-400 shadow-card-lg flex items-center justify-center hover:scale-105 transition"
}, /*#__PURE__*/React.createElement(Icons.Sparkles, {
  size: 20,
  stroke: 1.7
})));

// IntersectionObserver-driven active-section state
const useActiveSection = () => {
  const [active, setActive] = React.useState('overview');
  React.useEffect(() => {
    const ids = NAV.map(n => n.id);
    const els = ids.map(id => document.getElementById(id)).filter(Boolean);
    const obs = new IntersectionObserver(entries => {
      // pick the entry closest to top among intersecting ones
      const visible = entries.filter(e => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible[0]) setActive(visible[0].target.id);
    }, {
      rootMargin: '-110px 0px -55% 0px',
      threshold: 0
    });
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
  return /*#__PURE__*/React.createElement("div", {
    className: "min-h-screen"
  }, /*#__PURE__*/React.createElement(TopBar, {
    active: active,
    setActive: setActive
  }), /*#__PURE__*/React.createElement("main", {
    className: "max-w-[1480px] mx-auto px-6 pt-8 pb-12 space-y-20"
  }, /*#__PURE__*/React.createElement(Hero, null), /*#__PURE__*/React.createElement(Holdings, null), /*#__PURE__*/React.createElement(Performance, null), /*#__PURE__*/React.createElement(Ideas, null), /*#__PURE__*/React.createElement(Footer, null)), /*#__PURE__*/React.createElement(Fab, null));
};
ReactDOM.createRoot(document.getElementById('root')).render(/*#__PURE__*/React.createElement(App, null));
