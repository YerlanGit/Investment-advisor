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
  })),
  Target: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("circle", {
    cx: "12",
    cy: "12",
    r: "9"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "12",
    cy: "12",
    r: "5"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "12",
    cy: "12",
    r: "1",
    fill: "currentColor"
  })),
  Check: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("polyline", {
    points: "20 6 9 17 4 12"
  })),
  X: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M18 6 6 18M6 6l12 12"
  })),
  Scale: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M12 3v18M7 7h10M5 7l-3 6h6zM19 7l-3 6h6zM7 21h10"
  })),
  Activity: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M22 12h-4l-3 9-6-18-3 9H2"
  })),
  Compass: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("circle", {
    cx: "12",
    cy: "12",
    r: "9"
  }), /*#__PURE__*/React.createElement("polygon", {
    points: "16 8 10 10 8 16 14 14",
    fill: "currentColor",
    fillOpacity: "0.15"
  })),
  Dollar: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M12 2v20M17 6.5C17 4.6 14.8 3.5 12 3.5S7 4.6 7 6.5 9.2 9.5 12 9.5s5 1.1 5 3-2.2 3-5 3-5-1.1-5-3"
  })),
  Grid: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("rect", {
    x: "3",
    y: "3",
    width: "7",
    height: "7",
    rx: "1"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "14",
    y: "3",
    width: "7",
    height: "7",
    rx: "1"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "3",
    y: "14",
    width: "7",
    height: "7",
    rx: "1"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "14",
    y: "14",
    width: "7",
    height: "7",
    rx: "1"
  })),
  Gem: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("path", {
    d: "M6 3h12l3 6-9 12L3 9z"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M3 9h18M9 3 6 9l6 12 6-12-3-6"
  })),
  Coin: p => /*#__PURE__*/React.createElement(Icon, p, /*#__PURE__*/React.createElement("ellipse", {
    cx: "12",
    cy: "6",
    rx: "8",
    ry: "3"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"
  }))
};

// Map asset class → icon
const sectorIcon = cls => ({
  'Акции США': Icons.Cpu,
  'Акции KZ': Icons.Globe,
  'Облигации': Icons.Shield,
  'Сырьё': Icons.Gem,
  'Ден. средства': Icons.Coin
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
  // «Прочие» bridges the shown top-N standalone bars up to the FULL sum so the
  // waterfall RECONCILES: ΣstandAlone + diversification = total (was detached).
  const _shown = standalone.reduce((a, s) => a + s.v, 0);
  const _other = Math.round((sumStandalone - _shown) * 10) / 10;
  const cols = [...standalone.map(s => ({
    t: s.t,
    v: s.v,
    kind: 'pos'
  })), ...(_other > 0.5 ? [{
    t: 'Прочие',
    v: _other,
    kind: 'pos'
  }] : []), {
    t: 'Дивер-сификация',
    v: diversification,
    kind: 'neg'
  }, {
    t: 'Итог',
    v: total,
    kind: 'total'
  }];
  let _run = 0,
    _peak = 0;
  cols.forEach(c => {
    if (c.kind === 'pos') {
      _run += c.v;
      _peak = Math.max(_peak, _run);
    }
  });
  const maxV = Math.max(_peak, total) * 1.06;
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
      color = '#1c1b1a'; // standalone bars — all «отдельно» (legend: black)
    } else if (c.kind === 'neg') {
      y0 = running;
      y1 = running + c.v; // v is negative
      running = y1;
      color = '#c47358'; // «диверсификация» (legend: rust)
    } else {
      // total — floor at 0
      y0 = 0;
      y1 = c.v;
      color = '#f5d04e'; // «итог» (legend: gold)
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

// ── Factor β-radar: portfolio polygon vs market reference (10 axes)
const FactorRadar = ({
  factors,
  size = 300
}) => {
  const cx = size / 2,
    cy = size / 2;
  const R = size / 2 - 46;
  const VMIN = -0.5,
    VMAX = 1.2;
  const norm = v => Math.max(0, (v - VMIN) / (VMAX - VMIN));
  const N = factors.length;
  const ang = i => (-90 + i * 360 / N) * Math.PI / 180;
  const pt = (i, rr) => [cx + rr * Math.cos(ang(i)), cy + rr * Math.sin(ang(i))];
  const poly = key => factors.map((f, i) => pt(i, norm(f[key]) * R).map(n => n.toFixed(1)).join(',')).join(' ');
  const rings = [-0.5, 0, 0.5, 1.0];
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: `0 0 ${size} ${size}`,
    className: "w-full h-auto",
    style: {
      maxWidth: size
    }
  }, rings.map((rv, k) => {
    const rr = norm(rv) * R;
    const p = factors.map((f, i) => pt(i, rr).map(n => n.toFixed(1)).join(',')).join(' ');
    return /*#__PURE__*/React.createElement("polygon", {
      key: k,
      points: p,
      fill: "none",
      stroke: "#1c1b1a",
      strokeOpacity: rv === 0 ? 0.22 : 0.08,
      strokeWidth: rv === 0 ? 1 : 0.7,
      strokeDasharray: rv === 0 ? 'none' : '2 4'
    });
  }), factors.map((f, i) => {
    const [ex, ey] = pt(i, R);
    const [lx, ly] = pt(i, R + 22);
    const anchor = Math.abs(lx - cx) < 8 ? 'middle' : lx > cx ? 'start' : 'end';
    return /*#__PURE__*/React.createElement("g", {
      key: i
    }, /*#__PURE__*/React.createElement("line", {
      x1: cx,
      y1: cy,
      x2: ex,
      y2: ey,
      stroke: "#1c1b1a",
      strokeOpacity: "0.07",
      strokeWidth: "0.7"
    }), /*#__PURE__*/React.createElement("text", {
      x: lx,
      y: ly + 3,
      textAnchor: anchor,
      fontSize: "9.5",
      fontFamily: "JetBrains Mono",
      fill: "#6b6862"
    }, f.name));
  }), /*#__PURE__*/React.createElement("polygon", {
    points: poly('mkt'),
    fill: "#1c1b1a",
    fillOpacity: "0.04",
    stroke: "#1c1b1a",
    strokeOpacity: "0.35",
    strokeWidth: "1.2",
    strokeDasharray: "4 4"
  }), /*#__PURE__*/React.createElement("polygon", {
    points: poly('port'),
    fill: "#f5d04e",
    fillOpacity: "0.22",
    stroke: "#caa01a",
    strokeWidth: "2"
  }), factors.map((f, i) => {
    const [px, py] = pt(i, norm(f.port) * R);
    return /*#__PURE__*/React.createElement("circle", {
      key: i,
      cx: px,
      cy: py,
      r: "2.6",
      fill: "#caa01a"
    });
  }));
};

// ── Regime quadrant: Growth × Cycle, dot at current coords
const RegimeQuadrant = ({
  dot,
  size = 300
}) => {
  const pad = 34;
  const inner = size - pad * 2;
  const cx = pad + inner / 2,
    cy = pad + inner / 2;
  const SCALE = 0.22; // axis half-range
  const X = v => cx + v / SCALE * (inner / 2);
  const Y = v => cy - v / SCALE * (inner / 2);
  const dx = X(dot.cycle),
    dy = Y(dot.growth);
  // Audit 2026-07-05 (R-1): labels pinned to finance/regime.py quadrants —
  // X=cycle, Y=growth ⇒ top-left (growth+, cycle−) is SLOWDOWN and bottom-right
  // (growth−, cycle+) is RECOVERY.  The old layout had them SWAPPED (pre-fix
  // engine semantics), so a Slowdown reading would land the dot in a quadrant
  // labelled «RECOVERY» — visually contradicting the regime label.
  const quads = [{
    x: pad,
    y: pad,
    label: 'SLOWDOWN',
    fill: '#a8a293'
  }, {
    x: cx,
    y: pad,
    label: 'EXPANSION',
    fill: '#caa01a'
  }, {
    x: pad,
    y: cy,
    label: 'RECESSION',
    fill: '#c47358'
  }, {
    x: cx,
    y: cy,
    label: 'RECOVERY',
    fill: '#5d7c5c'
  }];
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: `0 0 ${size} ${size}`,
    className: "w-full h-auto",
    style: {
      maxWidth: size
    }
  }, quads.map((q, i) => /*#__PURE__*/React.createElement("g", {
    key: i
  }, /*#__PURE__*/React.createElement("rect", {
    x: q.x,
    y: q.y,
    width: inner / 2,
    height: inner / 2,
    fill: q.fill,
    fillOpacity: q.label === 'EXPANSION' ? 0.14 : 0.05
  }), /*#__PURE__*/React.createElement("text", {
    x: q.x + inner / 4,
    y: q.y + (q.y < cy ? 16 : inner / 2 - 8),
    textAnchor: "middle",
    fontSize: "9",
    fontFamily: "JetBrains Mono",
    fontWeight: "600",
    fill: "#1c1b1a",
    fillOpacity: "0.5",
    letterSpacing: "0.05em"
  }, q.label))), /*#__PURE__*/React.createElement("line", {
    x1: cx,
    y1: pad,
    x2: cx,
    y2: pad + inner,
    stroke: "#1c1b1a",
    strokeOpacity: "0.35",
    strokeWidth: "0.8"
  }), /*#__PURE__*/React.createElement("line", {
    x1: pad,
    y1: cy,
    x2: pad + inner,
    y2: cy,
    stroke: "#1c1b1a",
    strokeOpacity: "0.35",
    strokeWidth: "0.8"
  }), /*#__PURE__*/React.createElement("text", {
    x: cx,
    y: pad - 12,
    textAnchor: "middle",
    fontSize: "9.5",
    fontFamily: "JetBrains Mono",
    fontWeight: "600",
    fill: "#6b6862"
  }, "↑ Growth"), /*#__PURE__*/React.createElement("text", {
    x: pad + inner + 2,
    y: cy - 6,
    textAnchor: "end",
    fontSize: "9.5",
    fontFamily: "JetBrains Mono",
    fontWeight: "600",
    fill: "#6b6862"
  }, "Cycle →"), /*#__PURE__*/React.createElement("line", {
    x1: cx,
    y1: cy,
    x2: dx,
    y2: dy,
    stroke: "#caa01a",
    strokeOpacity: "0.45",
    strokeWidth: "1.2",
    strokeDasharray: "3 2"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: dx,
    cy: dy,
    r: "13",
    fill: "#f5d04e",
    fillOpacity: "0.2"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: dx,
    cy: dy,
    r: "7",
    fill: "#f5d04e",
    stroke: "#caa01a",
    strokeWidth: "1.5"
  }), /*#__PURE__*/React.createElement("text", {
    x: dx + 14,
    y: dy - 6,
    fontSize: "9.5",
    fontFamily: "JetBrains Mono",
    fontWeight: "600",
    fill: "#caa01a"
  }, "сейчас"), /*#__PURE__*/React.createElement("text", {
    x: dx + 14,
    y: dy + 5,
    fontSize: "8",
    fontFamily: "JetBrains Mono",
    fill: "#6b6862"
  }, "G ", (dot.growth >= 0 ? '+' : '') + dot.growth.toFixed(2), " · C ", (dot.cycle >= 0 ? '+' : '') + dot.cycle.toFixed(2)));
};

// ── Mandate compliance bar: allowed band [lo,hi] + tick at value
const MandateBar = ({
  value,
  lo,
  hi,
  state
}) => {
  const tone = {
    ok: '#5d7c5c',
    over: '#c47358',
    under: '#caa01a'
  }[state] || '#6b6862';
  return /*#__PURE__*/React.createElement("div", {
    className: "relative h-1.5 rounded-full bg-ink-900/8 overflow-visible"
  }, /*#__PURE__*/React.createElement("div", {
    className: "absolute inset-y-0 rounded-full bg-sage-500/20",
    style: {
      left: `${lo}%`,
      width: `${hi - lo}%`
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "absolute top-1/2 -translate-y-1/2 w-[3px] h-3.5 rounded-full",
    style: {
      left: `calc(${Math.min(100, Math.max(0, value))}% - 1.5px)`,
      background: tone
    }
  }));
};

// ── Score pillar bar: diverging from center 0, value in [-2,2]
const ScorePillar = ({
  value
}) => {
  const v = Math.max(-2, Math.min(2, value));
  const half = Math.abs(v) / 2 * 50;
  const pos = v >= 0;
  return /*#__PURE__*/React.createElement("div", {
    className: "relative h-1.5 rounded-full bg-ink-900/6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "absolute top-0 bottom-0 w-px bg-ink-900/25",
    style: {
      left: '50%'
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "absolute top-0 bottom-0 rounded-full",
    style: {
      [pos ? 'left' : 'right']: '50%',
      width: `${half}%`,
      background: pos ? '#5d7c5c' : '#c47358'
    }
  }));
};

// ── Magnitude bar for stress (diverging, normalised to ±20%)
const MagnitudeBar = ({
  value
}) => {
  const w = Math.min(50, Math.abs(value) / 20 * 50);
  const pos = value >= 0;
  return /*#__PURE__*/React.createElement("div", {
    className: "relative h-1.5 rounded-full bg-ink-900/6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "absolute top-0 bottom-0 w-px bg-ink-900/25",
    style: {
      left: '50%'
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "absolute top-0 bottom-0 rounded-full",
    style: {
      [pos ? 'left' : 'right']: '50%',
      width: `${w}%`,
      background: pos ? '#5d7c5c' : '#c47358'
    }
  }));
};
Object.assign(window, {
  RiskGauge,
  Waterfall,
  SectorBar,
  PerfChart,
  Sparkline,
  Counter,
  FactorRadar,
  RegimeQuadrant,
  MandateBar,
  ScorePillar,
  MagnitudeBar
});
/* DEEP Overview — verdict, risk gauge, KPIs, mandate compliance, risk concentration */

const HeroStat = ({
  value,
  label,
  IconC,
  small
}) => /*#__PURE__*/React.createElement("div", {
  className: "flex flex-col items-center gap-1 px-4"
}, /*#__PURE__*/React.createElement(IconC, {
  size: 17,
  className: "text-ink-500 mb-1",
  stroke: 1.4
}), /*#__PURE__*/React.createElement("div", {
  className: `font-light tracking-tight num leading-none text-ink-900 ${small ? 'text-2xl pt-2' : 'text-5xl'}`
}, value), /*#__PURE__*/React.createElement("div", {
  className: "text-[12px] text-ink-500 font-medium tracking-tight"
}, label));

// Compact risk-index gauge for the HERO right column (moved up from the main
// grid — user request «перенеси Индекс риска в эту зону»).  Horizontal layout:
// gauge on the left, tier + forward return/Sharpe on the right.
const HeroGaugeCard = ({
  v
}) => /*#__PURE__*/React.createElement("div", {
  className: "glass-strong rounded-4xl p-5 shadow-card lift flex items-center gap-5"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex-shrink-0"
}, /*#__PURE__*/React.createElement(RiskGauge, {
  value: v.riskIndex,
  size: 150
})), /*#__PURE__*/React.createElement("div", {
  className: "flex flex-col gap-2.5 min-w-0"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-2 flex-wrap"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-ink-500 text-[10px] font-medium tracking-widest uppercase"
}, "Индекс риска"), /*#__PURE__*/React.createElement("h3", {
  className: "text-base font-semibold tracking-tight text-ink-900 leading-tight"
}, "Сводный 0–100")), v.riskTier && v.riskTier !== '–' && /*#__PURE__*/React.createElement("span", {
  className: "px-2 py-0.5 rounded-full bg-gold-400/25 text-gold-700 text-[9px] font-mono font-bold tracking-wider uppercase"
}, v.riskTier)), v.expReturn && v.expReturn !== '–' && /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-4"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-[9px] text-ink-500 font-medium uppercase tracking-wider"
}, "Ожид. дох. (год.)"), /*#__PURE__*/React.createElement("div", {
  className: "num text-lg font-light text-sage-600 leading-none mt-0.5"
}, v.expReturn)), v.expSharpe && v.expSharpe !== '–' && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-[9px] text-ink-500 font-medium uppercase tracking-wider"
}, "Фвд-Sharpe"), /*#__PURE__*/React.createElement("div", {
  className: "num text-lg font-light text-ink-900 leading-none mt-0.5"
}, v.expSharpe)))));

// Verdict / AI summary dark card with bullets
const VerdictCard = ({
  v
}) => /*#__PURE__*/React.createElement("div", {
  className: "rounded-4xl p-7 shadow-dark lift flex flex-col h-full",
  style: {
    background: 'linear-gradient(160deg, #1c1b1a 0%, #2a2825 100%)'
  }
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-center justify-between mb-4"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-2 text-gold-400 text-[10px] font-mono tracking-widest uppercase"
}, /*#__PURE__*/React.createElement(Icons.Sparkles, {
  size: 13,
  stroke: 1.8
}), " AI вердикт · ", window.DEEP.meta.aiModel), /*#__PURE__*/React.createElement("span", {
  className: "text-white/30 text-[10px] font-mono tracking-widest"
}, "DEEP")), /*#__PURE__*/React.createElement("p", {
  className: "text-white text-[15px] leading-relaxed font-light mb-5"
}, v.summary), /*#__PURE__*/React.createElement("div", {
  className: "mt-auto space-y-2.5"
}, v.bullets.map((b, i) => /*#__PURE__*/React.createElement("div", {
  key: i,
  className: "flex items-start gap-3"
}, b.tag ? /*#__PURE__*/React.createElement("span", {
  className: "mt-[3px] px-1.5 py-0.5 rounded-md bg-gold-400/15 text-gold-400 text-[9px] font-mono font-bold tracking-wider uppercase flex-shrink-0 w-[58px] text-center"
}, b.tag) : /*#__PURE__*/React.createElement("span", {
  className: "mt-[3px] w-1.5 h-1.5 rounded-full bg-gold-400/50 flex-shrink-0"
}), /*#__PURE__*/React.createElement("p", {
  className: "text-white/70 text-[11.5px] leading-snug font-light flex-1"
}, b.text, b.src && /*#__PURE__*/React.createElement("span", {
  className: "text-white/35 font-mono"
}, " [", b.src, "]"))))));

// Русская плюрализация: 1 нарушение · 2–4 нарушения · 5+ нарушений (§−14 C-6).
const _plural = (n, one, few, many) => {
  const n10 = n % 10,
    n100 = n % 100;
  if (n10 === 1 && n100 !== 11) return one;
  if (n10 >= 2 && n10 <= 4 && (n100 < 12 || n100 > 14)) return few;
  return many;
};

// Mandate compliance card
const MandateCard = ({
  m
}) => /*#__PURE__*/React.createElement("div", {
  className: "glass-strong rounded-4xl p-6 shadow-card lift flex flex-col"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-start justify-between mb-1 gap-2"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-ink-500 text-[12px] font-medium"
}, "Соответствие мандату"), /*#__PURE__*/React.createElement("h3", {
  className: "text-xl font-semibold tracking-tight text-ink-900 leading-tight"
}, m.profile)), m.violations > 0 ? /*#__PURE__*/React.createElement("span", {
  className: "px-2.5 py-1 rounded-full bg-rust-500/12 text-rust-600 text-[10px] font-semibold tracking-wider uppercase flex items-center gap-1 flex-shrink-0"
}, /*#__PURE__*/React.createElement(Icons.Warning, {
  size: 11,
  stroke: 2
}), " ", m.violations, " ", _plural(m.violations, 'нарушение', 'нарушения', 'нарушений')) : /*#__PURE__*/React.createElement("span", {
  className: "px-2.5 py-1 rounded-full bg-sage-500/15 text-sage-600 text-[10px] font-semibold tracking-wider uppercase flex-shrink-0"
}, "соответствует")), /*#__PURE__*/React.createElement("div", {
  className: "text-[10.5px] text-ink-400 font-mono mb-2"
}, "целевая волат. ≈", m.targetVol, "% · отклонение от ориентира ≤", m.trackingCap, "%"), m.leveraged && /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-2 rounded-2xl bg-rust-500/10 border border-rust-500/30 px-3 py-2 mb-3"
}, /*#__PURE__*/React.createElement(Icons.Warning, {
  size: 13,
  className: "text-rust-600 flex-shrink-0",
  stroke: 2
}), /*#__PURE__*/React.createElement("span", {
  className: "text-[11px] text-rust-600 font-medium"
}, "Маржинальный долг ", m.marginPct > 0 ? `≈${m.marginPct}% NAV` : 'обнаружен', " — позиции частично куплены в долг")), /*#__PURE__*/React.createElement("div", {
  className: "space-y-3.5 mt-auto"
}, m.rows.map((r, i) => {
  const tone = {
    ok: 'text-ink-900',
    over: 'text-rust-600',
    under: 'text-gold-700'
  }[r.state];
  return /*#__PURE__*/React.createElement("div", {
    key: i
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between mb-1.5"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[12px] text-ink-700"
  }, r.label), /*#__PURE__*/React.createElement("span", {
    className: `text-[12px] font-semibold num ${tone}`
  }, r.value, "%")), /*#__PURE__*/React.createElement(MandateBar, {
    value: r.value,
    lo: r.lo,
    hi: r.hi,
    state: r.state
  }), /*#__PURE__*/React.createElement("div", {
    className: "text-[9.5px] text-ink-400 font-mono mt-1"
  }, "допустимо ", r.lo, "–", r.hi, "%"));
})));

// KPI card — value + a prominent 12-month trend chart (user request «добавь
// графики к показателям»).  Draws the design <Sparkline> from the real numeric
// series; falls back to the server-rendered SVG if only that is present.
// Per-metric methodology note (hover) — makes the denominators explicit
// (Sprint-3 #8): Sharpe/Sortino use the STRUCTURAL factor volatility σ, not a
// realised sample vol.
const _KPI_METHOD = {
  sharpe: 'Знаменатель — структурная (факторная) волатильность σ = √(w′Σw), Σ = B·F·Bᵀ + D; числитель — геом. годовая доходность − валютно-сопоставленная RFR.',
  cvar: 'Средний убыток в худшие 5% дней (1-дневный горизонт), эмпирически + bootstrap-CI.',
  dd: 'Максимальная просадка пик→дно по реконструированной кривой капитала exp(Σ log-доходностей).'
};
const KpiCard = ({
  k
}) => {
  const border = {
    normal: '#5d7c5c',
    good: '#caa01a',
    watch: '#c47358'
  }[k.status];
  const hasPts = Array.isArray(k.pts) && k.pts.length >= 2;
  return /*#__PURE__*/React.createElement("div", {
    className: "glass-strong rounded-4xl p-6 shadow-card lift flex flex-col",
    style: {
      borderTop: `2px solid ${border}`
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between mb-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] tracking-widest uppercase text-ink-500 font-mono cursor-help",
    title: _KPI_METHOD[k.key] || undefined
  }, k.name), /*#__PURE__*/React.createElement("span", {
    className: "text-[40px] leading-none font-light num text-ink-900 tracking-tight"
  }, k.value)), /*#__PURE__*/React.createElement("div", {
    className: "rounded-2xl bg-cream-50/70 border border-ink-900/5 px-3 pt-2 pb-1.5 mb-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between text-[8.5px] tracking-widest uppercase text-ink-400 font-mono mb-0.5"
  }, /*#__PURE__*/React.createElement("span", null, "Динамика · 12 мес"), /*#__PURE__*/React.createElement("span", null, "сейчас")), /*#__PURE__*/React.createElement("div", {
    className: "h-12"
  }, hasPts ? /*#__PURE__*/React.createElement(Sparkline, {
    points: k.pts,
    color: k.color,
    height: 44,
    width: 300,
    gradId: `spk-${k.key}`
  }) : k.svg ? /*#__PURE__*/React.createElement("div", {
    className: "w-full h-full",
    dangerouslySetInnerHTML: {
      __html: k.svg
    }
  }) : /*#__PURE__*/React.createElement("div", {
    className: "w-full h-full flex items-center justify-center text-[9px] text-ink-300 font-mono"
  }, "нет истории"))), /*#__PURE__*/React.createElement("div", {
    className: "text-[10.5px] text-ink-400 font-mono leading-snug mb-4"
  }, k.sub), /*#__PURE__*/React.createElement("div", {
    className: "mt-auto flex items-start gap-2.5 rounded-2xl bg-cream-50 border border-ink-900/5 px-3.5 py-3"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 13,
    className: "text-gold-600 mt-0.5 flex-shrink-0",
    stroke: 1.8
  }), /*#__PURE__*/React.createElement("p", {
    className: "text-[11.5px] text-ink-700 leading-snug font-light"
  }, k.ai)));
};

// Concentration table
const ConcentrationCard = ({
  rows
}) => /*#__PURE__*/React.createElement("div", {
  className: "glass-strong rounded-4xl p-6 shadow-card lift flex flex-col"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-start justify-between mb-4"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-ink-500 text-[12px] font-medium"
}, "Где сосредоточен риск"), /*#__PURE__*/React.createElement("h3", {
  className: "text-xl font-semibold tracking-tight text-ink-900 leading-tight"
}, "Топ-вкладчики")), /*#__PURE__*/React.createElement("span", {
  className: "px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5 text-[10px] font-mono tracking-wider text-ink-700"
}, "TRC · Euler")), /*#__PURE__*/React.createElement("div", {
  className: "grid grid-cols-[1.4fr_1fr_1fr_1.6fr_auto] gap-3 px-1 pb-2 text-[9px] tracking-widest uppercase text-ink-400 font-mono border-b border-ink-900/8"
}, /*#__PURE__*/React.createElement("div", null, "Тикер"), /*#__PURE__*/React.createElement("div", {
  className: "text-right"
}, "Вес"), /*#__PURE__*/React.createElement("div", {
  className: "text-right"
}, "Beta"), /*#__PURE__*/React.createElement("div", null, "Вклад в риск"), /*#__PURE__*/React.createElement("div", null)), /*#__PURE__*/React.createElement("div", {
  className: "divide-y divide-ink-900/5 mt-1"
}, rows.map((r, i) => {
  const hot = r.status === 'HOTSPOT';
  return /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "grid grid-cols-[1.4fr_1fr_1fr_1.6fr_auto] gap-3 items-center px-1 py-2.5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[14px] font-bold num text-ink-900"
  }, r.t), /*#__PURE__*/React.createElement("div", {
    className: "text-[13px] num text-ink-700 text-right"
  }, r.w.toFixed(1), "%"), /*#__PURE__*/React.createElement("div", {
    className: "text-[13px] num text-ink-700 text-right"
  }, r.beta.toFixed(2)), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex-1"
  }, /*#__PURE__*/React.createElement(MiniBar, {
    value: r.risk,
    max: 36,
    color: hot ? '#f5d04e' : '#a8a293'
  })), /*#__PURE__*/React.createElement("span", {
    className: `text-[12px] font-semibold num ${hot ? 'text-gold-700' : 'text-ink-900'}`
  }, r.risk.toFixed(1), "%")), /*#__PURE__*/React.createElement("div", {
    className: "flex justify-end"
  }, hot ? /*#__PURE__*/React.createElement("span", {
    className: "px-2 py-0.5 rounded-full bg-gold-400 text-ink-900 text-[9px] font-bold tracking-wider uppercase"
  }, "Hotspot") : /*#__PURE__*/React.createElement("span", {
    className: "px-2 py-0.5 rounded-full bg-cream-200/60 text-ink-500 text-[9px] font-semibold tracking-wider uppercase"
  }, "Norm")));
})));
const MiniBar = ({
  value,
  max = 30,
  color = '#1c1b1a',
  height = 4
}) => /*#__PURE__*/React.createElement("div", {
  className: "w-full bg-ink-900/8 rounded-full overflow-hidden",
  style: {
    height
  }
}, /*#__PURE__*/React.createElement("div", {
  className: "rounded-full",
  style: {
    width: `${Math.max(0, Math.min(100, value / max * 100))}%`,
    height: '100%',
    background: color
  }
}));

// Waterfall card
const WaterfallCard = ({
  data,
  ai
}) => /*#__PURE__*/React.createElement("div", {
  className: "glass-strong rounded-4xl p-6 shadow-card lift flex flex-col"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-start justify-between mb-2"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "text-ink-500 text-[12px] font-medium"
}, "Декомпозиция"), /*#__PURE__*/React.createElement("h3", {
  className: "text-xl font-semibold tracking-tight text-ink-900 leading-tight"
}, "Риск портфеля")), /*#__PURE__*/React.createElement("span", {
  className: "px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5 text-[10px] font-mono tracking-wider text-ink-700"
}, data.total.toFixed(1), "% год.")), /*#__PURE__*/React.createElement("div", {
  className: "flex-1 flex items-center -mx-1"
}, /*#__PURE__*/React.createElement(Waterfall, {
  data: data,
  height: 200
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
const Overview = () => {
  const p = window.DEEP;
  const v = p.verdict;
  return /*#__PURE__*/React.createElement("section", {
    id: "overview",
    className: "rise",
    "data-screen-label": "01 Overview"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-x-10 gap-y-6 flex-wrap mb-8"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex-1 min-w-[280px] max-w-[760px]"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-gold-400"
  }), "Portfolio Risk Report · Tier ", p.meta.tier), /*#__PURE__*/React.createElement("h1", {
    className: "text-[clamp(32px,4.4vw,54px)] leading-[1.05] tracking-[-0.03em] font-light text-ink-900"
  }, v.headline, /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, ".")), /*#__PURE__*/React.createElement("p", {
    className: "text-[17px] text-ink-500 mt-3 max-w-[620px] font-light"
  }, v.sub)), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-col gap-4 w-full lg:w-auto lg:min-w-[380px] lg:max-w-[460px]"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-end justify-start lg:justify-end gap-1 divide-x divide-ink-900/10"
  }, p.heroStats.map((s, i) => {
    const IconC = {
      briefcase: Icons.Briefcase,
      wallet: Icons.Wallet,
      shield: Icons.Shield
    }[s.icon] || Icons.Briefcase;
    return /*#__PURE__*/React.createElement(HeroStat, {
      key: i,
      value: s.value,
      label: s.label,
      IconC: IconC,
      small: s.small
    });
  })), /*#__PURE__*/React.createElement(HeroGaugeCard, {
    v: v
  }))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-12 gap-5 mb-5 items-stretch"
  }, /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-7"
  }, /*#__PURE__*/React.createElement(VerdictCard, {
    v: v
  })), /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-5"
  }, /*#__PURE__*/React.createElement(MandateCard, {
    m: p.mandate
  }))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 md:grid-cols-3 gap-5 mb-5"
  }, p.kpis.map(k => /*#__PURE__*/React.createElement(KpiCard, {
    key: k.key,
    k: k
  }))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-12 gap-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-7"
  }, /*#__PURE__*/React.createElement(ConcentrationCard, {
    rows: p.concentration
  })), /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-5"
  }, /*#__PURE__*/React.createElement(WaterfallCard, {
    data: p.riskDecomp,
    ai: p.concAI
  }))), /*#__PURE__*/React.createElement("div", {
    className: "mt-5 rounded-3xl p-5 flex items-start gap-4",
    style: {
      background: 'linear-gradient(120deg, #fbf3d9 0%, #f6ebc0 100%)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-10 h-10 rounded-2xl bg-ink-900 text-gold-400 flex items-center justify-center flex-shrink-0"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 17,
    stroke: 1.7
  })), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] tracking-widest uppercase font-mono text-ink-700 mb-1"
  }, "AI · комментарий к риску"), /*#__PURE__*/React.createElement("p", {
    className: "text-[14px] text-ink-900 leading-relaxed font-light"
  }, p.concAI))));
};
Object.assign(window, {
  Overview
});
/* DEEP Holdings — 11 positions, expandable fundamentals + sector mix */

const SignalChip = ({
  signal
}) => {
  const styles = {
    'BUY': 'bg-sage-500/15 text-sage-600',
    'HOLD': 'bg-ink-900/6 text-ink-600',
    'TRIM': 'bg-gold-500/18 text-gold-700',
    'SELL': 'bg-rust-500 text-white'
  }[signal] || 'bg-ink-900/6 text-ink-600';
  return /*#__PURE__*/React.createElement("span", {
    className: `inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wider ${styles}`
  }, signal);
};
const HBar = ({
  value,
  max,
  color,
  height = 4
}) => /*#__PURE__*/React.createElement("div", {
  className: "w-full bg-ink-900/8 rounded-full overflow-hidden",
  style: {
    height
  }
}, /*#__PURE__*/React.createElement("div", {
  className: "rounded-full",
  style: {
    width: `${Math.max(0, Math.min(100, value / max * 100))}%`,
    height: '100%',
    background: color
  }
}));
const FundCell = ({
  label,
  value,
  warn,
  hint
}) => /*#__PURE__*/React.createElement("div", {
  className: "rounded-2xl bg-cream-50 border border-ink-900/5 px-3 py-2.5",
  title: hint || undefined
}, /*#__PURE__*/React.createElement("div", {
  className: "text-[9.5px] uppercase tracking-wider text-ink-500 font-mono leading-tight"
}, label), /*#__PURE__*/React.createElement("div", {
  className: `text-[14px] font-semibold num mt-0.5 ${warn ? 'text-rust-600' : 'text-ink-900'}`
}, value));
const HoldingRow = ({
  h,
  open,
  onToggle
}) => {
  const IconC = sectorIcon(h.cls);
  const pos = h.pnlPct >= 0;
  const hot = h.status === 'HOTSPOT';
  const atrWarn = parseFloat(h.fund.atr) > 3;
  return /*#__PURE__*/React.createElement("div", {
    className: `relative transition-colors ${open ? 'bg-cream-50/70' : 'hover:bg-cream-50/40'}`
  }, /*#__PURE__*/React.createElement("button", {
    onClick: onToggle,
    className: "w-full grid grid-cols-[36px_minmax(0,1.9fr)_minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.1fr)_84px_36px] items-center gap-3 px-5 py-3.5 text-left"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-8 h-8 rounded-xl bg-cream-100 border border-ink-900/5 flex items-center justify-center text-ink-700"
  }, /*#__PURE__*/React.createElement(IconC, {
    size: 15,
    stroke: 1.6
  })), /*#__PURE__*/React.createElement("div", {
    className: "min-w-0"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[15px] font-bold tracking-tight num text-ink-900 truncate"
  }, h.short || h.t), hot && /*#__PURE__*/React.createElement("span", {
    className: "px-1.5 py-0.5 rounded-md bg-gold-400 text-ink-900 text-[8.5px] font-bold tracking-wider uppercase flex-shrink-0"
  }, "Hotspot")), /*#__PURE__*/React.createElement("div", {
    className: "text-[11.5px] text-ink-500 truncate mt-0.5"
  }, h.name)), /*#__PURE__*/React.createElement("div", {
    className: "text-[12px] text-ink-700"
  }, h.cls), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[13px] font-semibold num text-ink-900"
  }, h.w.toFixed(1), "%"), /*#__PURE__*/React.createElement("div", {
    className: "mt-1.5"
  }, /*#__PURE__*/React.createElement(HBar, {
    value: h.w,
    max: 17,
    color: "#1c1b1a"
  }))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[13px] font-semibold num text-ink-900"
  }, h.risk.toFixed(1), "%"), /*#__PURE__*/React.createElement("div", {
    className: "mt-1.5"
  }, /*#__PURE__*/React.createElement(HBar, {
    value: h.risk,
    max: 36,
    color: hot ? '#f5d04e' : '#a8a293'
  }))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: `text-[14px] font-semibold num ${h.cash ? 'text-ink-400' : pos ? 'text-sage-600' : 'text-rust-600'}`
  }, pos ? '+' : '', h.pnlPct.toFixed(1), "%"), /*#__PURE__*/React.createElement("div", {
    className: `text-[11px] num ${h.cash ? 'text-ink-300' : pos ? 'text-sage-500' : 'text-rust-500'}`
  }, pos ? '+' : '−', "$", Math.abs(h.pnlUsd).toLocaleString('ru-RU'))), /*#__PURE__*/React.createElement("div", {
    className: "flex justify-end"
  }, /*#__PURE__*/React.createElement(SignalChip, {
    signal: h.signal
  })), /*#__PURE__*/React.createElement("div", {
    className: `w-8 h-8 rounded-full bg-ink-900/5 flex items-center justify-center text-ink-700 transition-transform ${open ? 'rotate-180' : ''}`
  }, /*#__PURE__*/React.createElement(Icons.Chevron, {
    size: 13
  }))), /*#__PURE__*/React.createElement("div", {
    className: "mob-detail overflow-hidden transition-[max-height,opacity] duration-500 ease-out",
    style: {
      maxHeight: open ? 640 : 0,
      opacity: open ? 1 : 0
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "px-5 pb-5 pt-1"
  }, /*#__PURE__*/React.createElement("div", {
    className: "rounded-3xl p-5 bg-white/70 border border-ink-900/5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-4 mb-4"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[10.5px] tracking-widest uppercase text-ink-500 font-mono"
  }, "Фундаментал · SEC EDGAR"), /*#__PURE__*/React.createElement("div", {
    className: "text-[15px] text-ink-900 font-medium mt-0.5"
  }, h.name)), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-mono text-ink-400 tracking-wider px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5"
  }, h.cls)), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3"
  }, /*#__PURE__*/React.createElement(FundCell, {
    label: "ROE",
    value: h.fund.roe,
    hint: "Рентабельность собственного капитала (чистая прибыль ÷ капитал)"
  }), /*#__PURE__*/React.createElement(FundCell, {
    label: "Опер. маржа",
    value: h.fund.margin,
    hint: "Операционная маржа — операционная прибыль ÷ выручка"
  }), /*#__PURE__*/React.createElement(FundCell, {
    label: "Долг / Активы",
    value: h.fund.debt,
    hint: "Долговая нагрузка: суммарный долг ÷ активы. 0% = долг не найден в отчётности или его нет"
  }), /*#__PURE__*/React.createElement(FundCell, {
    label: "Рост выручки",
    value: h.fund.growth,
    hint: "Рост выручки год к году"
  }), /*#__PURE__*/React.createElement(FundCell, {
    label: "ATR · день",
    value: h.fund.atr,
    warn: atrWarn,
    hint: "Average True Range — средний дневной диапазон цены, % (волатильность бумаги)"
  }), /*#__PURE__*/React.createElement(FundCell, {
    label: "Altman-Z",
    value: h.fund.z,
    hint: "Z-score Альтмана (риск банкротства): > 3 — безопасно, 1.8–3 — серая зона, < 1.8 — риск"
  })), (h.fundNote || h.note) && /*#__PURE__*/React.createElement("div", {
    className: "mt-4 flex items-start gap-3 text-[13px] text-ink-700 leading-relaxed"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 14,
    className: "text-gold-600 mt-1 flex-shrink-0",
    stroke: 1.8
  }), /*#__PURE__*/React.createElement("p", {
    className: "font-light"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-ink-500 font-medium"
  }, "Вывод: "), h.fundNote || h.note, " ", /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400 font-mono text-[11px]"
  }, "[SEC EDGAR] [Quant Engine]")))))));
};
const SectorMix = ({
  sectors,
  warns
}) => {
  let acc = 0;
  // Бейдж «Перевес» — только когда есть реальный warn-сектор или предупреждение
  // движка (был хардкод «Перевес IT», рисовался всегда — template-literal).
  const hasOverweight = (sectors || []).some(s => s.warn) || (warns || []).some(w => w && w !== '–');
  return /*#__PURE__*/React.createElement("div", {
    className: "glass-strong rounded-4xl p-6 shadow-card lift flex flex-col h-full"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between mb-4 gap-2"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-ink-500 text-[12px] font-medium"
  }, "Структура"), /*#__PURE__*/React.createElement("h3", {
    className: "text-xl font-semibold tracking-tight text-ink-900 leading-tight"
  }, "По секторам")), hasOverweight && /*#__PURE__*/React.createElement("span", {
    className: "px-2.5 py-1 rounded-full bg-rust-500/12 text-rust-600 text-[10px] font-semibold tracking-wider uppercase flex items-center gap-1 flex-shrink-0"
  }, /*#__PURE__*/React.createElement(Icons.Warning, {
    size: 11,
    stroke: 2
  }), " Перевес")), /*#__PURE__*/React.createElement("svg", {
    viewBox: "0 0 320 18",
    className: "w-full h-3.5 mb-4",
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
      width: Math.max(0, w),
      height: "18",
      fill: s.hue,
      rx: i === 0 ? 9 : 0,
      ry: i === 0 ? 9 : 0
    });
  })), /*#__PURE__*/React.createElement("div", {
    className: "space-y-2 flex-1"
  }, sectors.map(s => /*#__PURE__*/React.createElement("div", {
    key: s.name,
    className: "flex items-center gap-2.5"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-2.5 h-2.5 rounded-sm flex-shrink-0",
    style: {
      background: s.hue
    }
  }), /*#__PURE__*/React.createElement("span", {
    className: `flex-1 text-[12px] ${s.warn ? 'text-gold-700 font-semibold' : 'text-ink-700'}`
  }, s.name), /*#__PURE__*/React.createElement("span", {
    className: `text-[12px] font-semibold num ${s.warn ? 'text-gold-700' : 'text-ink-900'}`
  }, s.pct, "%")))), /*#__PURE__*/React.createElement("div", {
    className: "mt-4 space-y-2"
  }, (warns || []).filter(w => w && w !== '–').map((w, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "flex items-start gap-2 rounded-2xl bg-gold-400/12 border border-gold-400/35 px-3 py-2"
  }, /*#__PURE__*/React.createElement(Icons.Warning, {
    size: 12,
    className: "text-gold-700 mt-0.5 flex-shrink-0",
    stroke: 2
  }), /*#__PURE__*/React.createElement("p", {
    className: "text-[10.5px] text-gold-700 leading-snug font-medium"
  }, w)))));
};
const Holdings = () => {
  const [openIdx, setOpenIdx] = React.useState(0);
  const [filter, setFilter] = React.useState('Все');
  const p = window.DEEP;
  // L-15 (2026-07-19): показываем чип «HOTSPOT» только когда hotspot-позиции
  // реально есть (TRC > 20% ⇒ euler_extreme) — пустой фильтр с «Ничего не
  // подходит» сбивает читателя с толку на здоровой книге.
  const hasHot = p.holdings.some(h => h.status === 'HOTSPOT');
  const filters = ['Все', ...(hasHot ? ['HOTSPOT'] : []), 'Акции США', 'Защитные', 'В минусе', 'SELL · TRIM'];
  const rows = p.holdings.filter(h => {
    if (filter === 'Все') return true;
    if (filter === 'HOTSPOT') return h.status === 'HOTSPOT';
    if (filter === 'Акции США') return h.cls === 'Акции США';
    if (filter === 'Защитные') return ['Облигации', 'Сырьё', 'Ден. средства'].includes(h.cls);
    if (filter === 'В минусе') return h.pnlPct < 0;
    if (filter === 'SELL · TRIM') return ['SELL', 'TRIM'].includes(h.signal);
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
  }), " Holdings · ", p.meta.positions, " позиций"), /*#__PURE__*/React.createElement("h2", {
    className: "text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900"
  }, "Что вы держите", /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, ".")), /*#__PURE__*/React.createElement("p", {
    className: "text-[15px] text-ink-500 mt-2 font-light"
  }, "Нажмите на строку, чтобы увидеть фундаментал бумаги.")), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 flex-wrap"
  }, filters.map(f => /*#__PURE__*/React.createElement("button", {
    key: f,
    onClick: () => {
      setFilter(f);
      setOpenIdx(-1);
    },
    className: `px-3.5 py-1.5 rounded-full text-[12px] font-medium transition-colors
                          ${filter === f ? 'bg-ink-900 text-white' : 'bg-white/60 text-ink-700 hover:bg-white border border-ink-900/8'}`
  }, f)))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-12 gap-5 items-stretch"
  }, /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-8"
  }, /*#__PURE__*/React.createElement("div", {
    className: "glass-strong rounded-4xl shadow-card overflow-hidden"
  }, /*#__PURE__*/React.createElement("div", {
    className: "swipe-hint items-center gap-1 text-[10px] font-mono text-gold-700 bg-gold-400/15 rounded-full px-2.5 py-1 mb-2.5 w-max"
  }, "↔ листайте таблицу"), /*#__PURE__*/React.createElement("div", {
    className: "mob-scroll-x"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-[36px_minmax(0,1.9fr)_minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.1fr)_84px_36px] items-center gap-3 px-5 py-3 border-b border-ink-900/6 text-[9.5px] tracking-widest uppercase text-ink-500 font-mono"
  }, /*#__PURE__*/React.createElement("div", null), /*#__PURE__*/React.createElement("div", null, "Тикер · Имя"), /*#__PURE__*/React.createElement("div", null, "Класс"), /*#__PURE__*/React.createElement("div", null, "Вес"), /*#__PURE__*/React.createElement("div", null, "Риск"), /*#__PURE__*/React.createElement("div", null, "P/L"), /*#__PURE__*/React.createElement("div", {
    className: "text-right"
  }, "Сигнал"), /*#__PURE__*/React.createElement("div", null)), /*#__PURE__*/React.createElement("div", {
    className: "divide-y divide-ink-900/5"
  }, rows.map((h, i) => /*#__PURE__*/React.createElement(HoldingRow, {
    key: h.t,
    h: h,
    open: openIdx === i,
    onToggle: () => setOpenIdx(openIdx === i ? -1 : i)
  })), rows.length === 0 && /*#__PURE__*/React.createElement("div", {
    className: "px-6 py-12 text-center text-ink-500 text-[14px]"
  }, "Ничего не подходит под фильтр «", filter, "».")))))), /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-4"
  }, /*#__PURE__*/React.createElement(SectorMix, {
    sectors: p.sectors,
    warns: p.sectorWarn
  }))), /*#__PURE__*/React.createElement("div", {
    className: "mt-5 rounded-3xl p-5 glass-strong shadow-card flex items-start gap-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-10 h-10 rounded-2xl bg-ink-900 text-gold-400 flex items-center justify-center flex-shrink-0"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 17,
    stroke: 1.7
  })), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] tracking-widest uppercase font-mono text-ink-500 mb-1"
  }, "AI · сводка по составу"), /*#__PURE__*/React.createElement("p", {
    className: "text-[14px] text-ink-800 leading-relaxed font-light"
  }, p.holdingsAI))));
};
Object.assign(window, {
  Holdings
});
/* DEEP Factors — β-radar + 4-pillar scoring */

const FactorTable = ({
  factors,
  benchmarkName
}) => /*#__PURE__*/React.createElement("table", {
  className: "w-full"
}, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", {
  className: "text-[9.5px] tracking-widest uppercase text-ink-400 font-mono border-b border-ink-900/8"
}, /*#__PURE__*/React.createElement("th", {
  className: "text-left font-medium py-2"
}, "Фактор"), /*#__PURE__*/React.createElement("th", {
  className: "text-right font-medium py-2"
}, "β портф."), /*#__PURE__*/React.createElement("th", {
  className: "text-right font-medium py-2",
  title: `${benchmarkName} — факторные беты вашего бенчмарка (его тоже разложили по этим осям)`
}, benchmarkName), /*#__PURE__*/React.createElement("th", {
  className: "text-right font-medium py-2",
  title: "Активный наклон портфеля относительно бенчмарка"
}, "Наклон Δ"))), /*#__PURE__*/React.createElement("tbody", null, factors.map((f, i) => {
  const d = f.port - f.mkt;
  const tone = d > 0 ? 'text-gold-700' : d < 0 ? 'text-sage-600' : 'text-ink-400';
  return /*#__PURE__*/React.createElement("tr", {
    key: i,
    className: "border-b border-ink-900/5 last:border-0"
  }, /*#__PURE__*/React.createElement("td", {
    className: "text-left py-2 text-[12px] text-ink-800 font-medium"
  }, f.name), /*#__PURE__*/React.createElement("td", {
    className: "text-right py-2 text-[12px] num text-ink-900"
  }, f.port.toFixed(2)), /*#__PURE__*/React.createElement("td", {
    className: "text-right py-2 text-[12px] num text-ink-400"
  }, f.mkt.toFixed(2)), /*#__PURE__*/React.createElement("td", {
    className: `text-right py-2 text-[12px] num font-semibold ${tone}`
  }, d > 0 ? '+' : '', d.toFixed(2)));
})));

// Источники риска — факторная декомпозиция дисперсии (additive layer,
// finance/factor_decomposition → payload.factor_variance → factorVariance).
// Отвечает «откуда берётся риск ПО ИСТОЧНИКАМ» (Euler по факторам), дополняя
// TRC-разложение по активам.  Отрицательная доля = фактор-хедж.  Блок целиком
// скрыт, когда движок пропустил декомпозицию (factorVariance == null).
// Один ряд-«метр»: источник риска, дорожка-бар, доля дисперсии.  Верстается
// адаптивно — на мобильном лейбл+% в строке, бар во всю ширину под ними;
// на sm+ единый выровненный ряд «лейбл · бар · %».  Драйверы и чип «хедж»
// живут в подписи под баром с выравниванием по колонке лейбла.
const VarianceRow = ({
  r,
  maxAbs
}) => {
  const neg = r.pct < 0;
  const w = Math.max(Math.min(Math.abs(r.pct) / maxAbs * 100, 100), 1.5);
  const fill = neg ? '#7a9a78' : '#caa01a';
  const pctCls = neg ? 'text-sage-600' : 'text-ink-900';
  return /*#__PURE__*/React.createElement("div", {
    className: "py-2.5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex flex-col sm:flex-row sm:items-center gap-1.5 sm:gap-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-baseline justify-between gap-2 sm:block sm:w-[35%] sm:shrink-0 min-w-0"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[11.5px] text-ink-800 font-medium leading-tight truncate"
  }, r.source), /*#__PURE__*/React.createElement("span", {
    className: `sm:hidden text-[12px] num font-semibold tabular-nums shrink-0 ${pctCls}`
  }, r.pct, "%")), /*#__PURE__*/React.createElement("div", {
    className: "flex-1 min-w-0 h-2 rounded-full bg-ink-900/5 relative overflow-hidden"
  }, /*#__PURE__*/React.createElement("div", {
    className: "absolute inset-y-0 left-0 rounded-full",
    style: {
      width: `${w}%`,
      background: fill
    }
  })), /*#__PURE__*/React.createElement("span", {
    className: `hidden sm:block sm:w-[52px] text-right text-[12px] num font-semibold tabular-nums shrink-0 ${pctCls}`
  }, r.pct, "%")), (r.drivers || neg) && /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-1.5 mt-1 sm:pl-[calc(35%+0.75rem)] min-w-0"
  }, neg && /*#__PURE__*/React.createElement("span", {
    className: "text-[8.5px] font-mono uppercase tracking-wider text-sage-600 bg-sage-500/15 rounded px-1 py-px shrink-0"
  }, "хедж"), r.drivers && /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-mono text-ink-400 truncate"
  }, r.drivers)));
};
const FactorVariance = ({
  fv
}) => {
  if (!fv || !fv.rows || !fv.rows.length) return null;
  const maxAbs = Math.max(...fv.rows.map(r => Math.abs(r.pct)), 1);
  return /*#__PURE__*/React.createElement("div", {
    className: "mt-5 rounded-3xl bg-cream-50 border border-ink-900/5 p-4 sm:p-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-3 flex-wrap mb-3"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[13px] font-semibold text-ink-900 leading-tight"
  }, "Откуда берётся риск"), /*#__PURE__*/React.createElement("div", {
    className: "text-[10.5px] text-ink-500 font-light mt-0.5"
  }, "декомпозиция дисперсии по источникам · σ² = bᵀFb + wᵀDw")), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-1.5 shrink-0"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-mono rounded-full bg-gold-500/20 text-gold-700 px-2 py-0.5"
  }, "систематика ", fv.systematic, "%"), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-mono rounded-full bg-sage-500/15 text-sage-600 px-2 py-0.5"
  }, "специфика ", fv.idio, "%"))), /*#__PURE__*/React.createElement("div", {
    className: "divide-y divide-ink-900/5"
  }, fv.rows.map((r, i) => /*#__PURE__*/React.createElement(VarianceRow, {
    key: i,
    r: r,
    maxAbs: maxAbs
  }))), fv.twins && fv.twins.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "mt-3.5 pt-3 border-t border-ink-900/8"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[11px] text-ink-800 font-medium mb-2"
  }, "Факторные двойники ", /*#__PURE__*/React.createElement("span", {
    className: "font-light text-ink-500"
  }, "— одна ставка куплена дважды (systematic corr ≥ 0.90)")), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-1.5"
  }, fv.twins.map((t, i) => /*#__PURE__*/React.createElement("span", {
    key: i,
    className: "inline-flex items-center gap-1.5 text-[10px] font-mono rounded-full bg-white/70 border border-ink-900/8 px-2.5 py-1"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-ink-800 font-semibold whitespace-nowrap"
  }, t.pair), /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, "corr ", t.corr), /*#__PURE__*/React.createElement("span", {
    className: "text-ink-300"
  }, "·"), /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, "вес ", t.w, "%"))))));
};

// 4-Pillar legend — spells out the F·V·T·C abbreviations AND the concrete
// metrics each pillar scores (user request «раскрыть какие показатели и
// аббревиатуры»).  Inputs mirror src/finance/scoring.py.
const PILLARS = [{
  k: 'F',
  name: 'Fundamentals',
  ru: 'Фундамент',
  metrics: 'ROE · опер. маржа · рост выручки · долг/активы · FCF-маржа',
  color: '#5d7c5c'
}, {
  k: 'V',
  name: 'Valuation',
  ru: 'Оценка',
  metrics: 'P/E и P/S относительно сектора и собственной истории',
  color: '#caa01a'
}, {
  k: 'T',
  name: 'Technical',
  ru: 'Техника',
  metrics: 'тренд SMA50/200 · RSI(14) · MACD(12,26,9)',
  color: '#9a7a10'
},
// Credit is intentionally ASYMMETRIC (−2…+1): weak credit can drag the score
// hard, but strong credit only earns a modest upside — it never rescues a
// poor F/V/T on its own.
{
  k: 'C',
  name: 'Credit',
  ru: 'Кредит',
  metrics: 'Altman-Z · покрытие процентов · долг (диапазон −2…+1, асимметрично)',
  color: '#c47358'
}];
const PillarLegend = () => /*#__PURE__*/React.createElement("div", {
  className: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-5"
}, PILLARS.map(pl => /*#__PURE__*/React.createElement("div", {
  key: pl.k,
  className: "rounded-2xl bg-cream-50 border border-ink-900/5 p-3.5"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-center gap-2.5 mb-1.5"
}, /*#__PURE__*/React.createElement("span", {
  className: "w-7 h-7 rounded-lg flex items-center justify-center text-[12px] font-bold num text-white flex-shrink-0",
  style: {
    background: pl.color
  }
}, pl.k), /*#__PURE__*/React.createElement("div", {
  className: "min-w-0"
}, /*#__PURE__*/React.createElement("div", {
  className: "text-[12.5px] font-semibold text-ink-900 leading-none truncate"
}, pl.name), /*#__PURE__*/React.createElement("div", {
  className: "text-[10px] text-ink-500 leading-none mt-1"
}, pl.ru))), /*#__PURE__*/React.createElement("div", {
  className: "text-[10.5px] text-ink-600 leading-snug"
}, pl.metrics))));
const pillarTone = v => v == null ? 'text-ink-300' : v > 0 ? 'text-sage-600' : v < 0 ? 'text-rust-600' : 'text-ink-400';
const ScoreCard = ({
  s
}) => {
  const totalTone = s.total > 0 ? 'text-sage-600' : s.total < 0 ? 'text-rust-600' : 'text-ink-500';
  const accent = {
    BUY: '#5d7c5c',
    HOLD: '#a8a293',
    TRIM: '#caa01a',
    SELL: '#c47358'
  }[s.action];
  const actionChip = {
    BUY: 'bg-sage-500/15 text-sage-600',
    HOLD: 'bg-ink-900/6 text-ink-600',
    TRIM: 'bg-gold-500/18 text-gold-700',
    SELL: 'bg-rust-500 text-white'
  }[s.action];
  const pillars = [['F', s.F], ['V', s.V], ['T', s.T], ['C', s.C]];
  return /*#__PURE__*/React.createElement("div", {
    className: "rounded-3xl bg-white/70 border border-ink-900/6 p-4 shadow-card lift",
    style: {
      borderLeft: `2px solid ${accent}`
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between mb-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[14px] font-bold num text-ink-900"
  }, s.t), /*#__PURE__*/React.createElement("span", {
    className: `text-[12px] font-semibold num ${totalTone}`
  }, "Итог ", s.total > 0 ? '+' : '', s.total.toFixed(1))), /*#__PURE__*/React.createElement("div", {
    className: "space-y-2"
  }, pillars.map(([name, val]) => /*#__PURE__*/React.createElement("div", {
    key: name,
    className: "grid grid-cols-[14px_1fr_30px] items-center gap-2.5"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-mono font-semibold text-ink-500"
  }, name), val == null ? /*#__PURE__*/React.createElement("div", {
    className: "h-1.5 rounded-full bg-ink-900/4 flex items-center justify-center"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[8px] text-ink-300 font-mono"
  }, "н/п")) : /*#__PURE__*/React.createElement(ScorePillar, {
    value: val
  }), /*#__PURE__*/React.createElement("span", {
    className: `text-[10.5px] num font-semibold text-right ${pillarTone(val)}`
  }, val == null ? '—' : `${val > 0 ? '+' : ''}${val.toFixed(1)}`)))), /*#__PURE__*/React.createElement("div", {
    className: "mt-3 pt-2.5 border-t border-ink-900/6 flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: `px-2 py-0.5 rounded-full text-[9px] font-bold tracking-wider ${actionChip}`
  }, s.action), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] text-ink-500 leading-tight"
  }, s.reason)));
};
const Factors = () => {
  const p = window.DEEP;
  // B1 (2026-07-17): динамический бенчмарк секции — столбец, легенда и плашка
  // следуют за мандатным бенчмарком клиента (payload.benchmarkName); фолбэк
  // «S&P 500» сохраняет прежний вид для легаси-данных и S&P-константы.
  const benchName = p.benchmarkName || 'S&P 500';
  return /*#__PURE__*/React.createElement("section", {
    id: "factors",
    className: "rise",
    "data-screen-label": "03 Factors"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mb-6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-gold-400"
  }), " Factor decomposition · 4-Pillar · DEEP"), /*#__PURE__*/React.createElement("h2", {
    className: "text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900"
  }, "Факторы и качество позиций", /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, ".")), /*#__PURE__*/React.createElement("p", {
    className: "text-[15px] text-ink-500 mt-2 font-light max-w-[680px]"
  }, "Скрытые концентрации по 10 факторам и оценка каждой бумаги по четырём столпам.")), /*#__PURE__*/React.createElement("div", {
    className: "glass-strong rounded-4xl p-7 shadow-card mb-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-4 flex-wrap mb-4"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h3", {
    className: "text-2xl font-semibold tracking-tight text-ink-900"
  }, "Факторное разложение (β)"), /*#__PURE__*/React.createElement("p", {
    className: "text-[12px] text-ink-500 font-mono mt-1"
  }, "Ridge β (α=0.001) · EWMA hl=63 ⊕ Ledoit-Wolf 70/30 · окно 60 дней · покрытие ", p.factorCoverage, "%")), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-4 text-[11px] text-ink-600"
  }, /*#__PURE__*/React.createElement("span", {
    className: "flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-4 h-0 border-t-2 border-gold-600"
  }), " Портфель"), /*#__PURE__*/React.createElement("span", {
    className: "flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-4 h-0 border-t-2 border-dashed border-ink-700"
  }), " ", `Бенчмарк (${benchName})`))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-12 gap-7 items-center"
  }, /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-5 flex justify-center"
  }, /*#__PURE__*/React.createElement(FactorRadar, {
    factors: p.factors,
    size: 320
  })), /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-7"
  }, /*#__PURE__*/React.createElement(FactorTable, {
    factors: p.factors,
    benchmarkName: benchName
  }), /*#__PURE__*/React.createElement("p", {
    className: "text-[11px] text-ink-500 leading-relaxed font-light mt-4"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-ink-800 font-medium"
  }, "Что показывает радар:"), " насколько портфель завязан на глобальные факторы. Большая площадь — больше зависимости от рынка; совпадение направлений по нескольким факторам — скрытая общая ставка."), /*#__PURE__*/React.createElement("p", {
    className: "text-[11px] text-ink-500 leading-relaxed font-light mt-2 rounded-2xl bg-cream-50 border border-ink-900/5 px-3.5 py-2.5"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-ink-800 font-medium"
  }, `Столбец «${benchName}»`), " — факторные беты самого бенчмарка (его тоже разложили по этим осям). ", /*#__PURE__*/React.createElement("span", {
    className: "font-medium"
  }, "Наклон Δ = β портфеля − β бенчмарка"), " — ваш активный перекос относительно бенчмарка: именно он и есть полезный сигнал. Если ваш бенчмарк — широкий рынок (S&P 500), у него β_Market ≈ 1, а стили ≈ 0; у стилевых и секторных бенчмарков (напр. Nasdaq 100) появляются собственные наклоны — это нормально."))), /*#__PURE__*/React.createElement(FactorVariance, {
    fv: p.factorVariance
  }), /*#__PURE__*/React.createElement("div", {
    className: "mt-5 rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3.5 flex items-start gap-3"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 14,
    className: "text-gold-600 mt-0.5 flex-shrink-0",
    stroke: 1.8
  }), /*#__PURE__*/React.createElement("p", {
    className: "text-[12.5px] text-ink-700 leading-relaxed font-light"
  }, p.factorAI))), /*#__PURE__*/React.createElement("div", {
    className: "glass-strong rounded-4xl p-7 shadow-card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-4 flex-wrap mb-5"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h3", {
    className: "text-2xl font-semibold tracking-tight text-ink-900"
  }, "4-Pillar Scoring"), /*#__PURE__*/React.createElement("p", {
    className: "text-[12px] text-ink-500 mt-1 font-light"
  }, "Каждая бумага оценивается по четырём столпам — ", /*#__PURE__*/React.createElement("b", {
    className: "text-ink-800"
  }, "F"), "undamentals · ", /*#__PURE__*/React.createElement("b", {
    className: "text-ink-800"
  }, "V"), "aluation · ", /*#__PURE__*/React.createElement("b", {
    className: "text-ink-800"
  }, "T"), "echnical · ", /*#__PURE__*/React.createElement("b", {
    className: "text-ink-800"
  }, "C"), "redit — каждый −2…+2, итог ∈ [−6, +6].")), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-mono text-ink-400 tracking-wider px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5"
  }, "Quant Engine + SEC EDGAR")), /*#__PURE__*/React.createElement(PillarLegend, null), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
  }, p.scores.map(s => /*#__PURE__*/React.createElement(ScoreCard, {
    key: s.t,
    s: s
  }))), p.scoresNote && /*#__PURE__*/React.createElement("div", {
    className: "text-[10.5px] text-ink-400 font-mono mt-4"
  }, p.scoresNote), /*#__PURE__*/React.createElement("div", {
    className: "mt-4 rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3.5 flex items-start gap-3"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 14,
    className: "text-gold-600 mt-0.5 flex-shrink-0",
    stroke: 1.8
  }), /*#__PURE__*/React.createElement("p", {
    className: "text-[12.5px] text-ink-700 leading-relaxed font-light"
  }, p.scoresAI))));
};
Object.assign(window, {
  Factors
});
/* DEEP Stress scenarios + Market regime */

const StressTable = ({
  rows
}) => /*#__PURE__*/React.createElement("div", {
  className: "glass-strong rounded-4xl p-7 shadow-card"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-start justify-between gap-4 flex-wrap mb-5"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h3", {
  className: "text-2xl font-semibold tracking-tight text-ink-900"
}, "Стресс-сценарии"), /*#__PURE__*/React.createElement("p", {
  className: "text-[12px] text-ink-500 font-mono mt-1"
}, "Параметрические шоки факторов (ΔPnL = w′·B·shock) · горизонт 1 квартал")), /*#__PURE__*/React.createElement("span", {
  className: "text-[10px] font-mono text-ink-400 tracking-wider px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5"
}, "7 сценариев · не прогноз")), /*#__PURE__*/React.createElement("div", {
  className: "swipe-hint items-center gap-1 text-[10px] font-mono text-gold-700 bg-gold-400/15 rounded-full px-2.5 py-1 mb-2.5 w-max"
}, "↔ листайте таблицу"), /*#__PURE__*/React.createElement("div", {
  className: "mob-scroll-x"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "grid grid-cols-[minmax(0,2.1fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.4fr)_minmax(0,0.9fr)_minmax(0,1fr)] gap-3 px-1 pb-2.5 text-[9.5px] tracking-widest uppercase text-ink-400 font-mono border-b border-ink-900/8"
}, /*#__PURE__*/React.createElement("div", null, "Сценарий"), /*#__PURE__*/React.createElement("div", {
  className: "text-right"
}, "Δ Портфель"), /*#__PURE__*/React.createElement("div", {
  className: "text-right"
}, "Δ Стоимость"), /*#__PURE__*/React.createElement("div", null, "Магнитуда"), /*#__PURE__*/React.createElement("div", {
  className: "text-right"
}, "Drawdown"), /*#__PURE__*/React.createElement("div", {
  className: "text-right"
}, "Восстановл.")), /*#__PURE__*/React.createElement("div", {
  className: "divide-y divide-ink-900/5"
}, rows.map((r, i) => {
  const pos = r.pct >= 0;
  return /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "grid grid-cols-[minmax(0,2.1fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.4fr)_minmax(0,0.9fr)_minmax(0,1fr)] gap-3 items-center px-1 py-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[12.5px] text-ink-800"
  }, r.name), /*#__PURE__*/React.createElement("div", {
    className: `text-right text-[13px] num font-semibold ${pos ? 'text-sage-600' : 'text-rust-600'}`
  }, pos ? '+' : '', r.pct.toFixed(1), "%"), /*#__PURE__*/React.createElement("div", {
    className: `text-right text-[12px] num ${pos ? 'text-sage-500' : 'text-rust-500'}`
  }, pos ? '+' : '−', "$", Math.abs(r.usd).toLocaleString('ru-RU')), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement(MagnitudeBar, {
    value: r.pct
  })), /*#__PURE__*/React.createElement("div", {
    className: `text-right text-[12px] num ${r.dd == null ? 'text-ink-300' : 'text-rust-600'}`
  }, r.dd == null ? '—' : `${r.dd.toFixed(1)}%`), /*#__PURE__*/React.createElement("div", {
    className: "text-right text-[11.5px] num text-ink-500"
  }, r.rec));
})))), /*#__PURE__*/React.createElement("div", {
  className: "mt-4 rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3.5 flex items-start gap-3"
}, /*#__PURE__*/React.createElement(Icons.Sparkles, {
  size: 14,
  className: "text-gold-600 mt-0.5 flex-shrink-0",
  stroke: 1.8
}), /*#__PURE__*/React.createElement("p", {
  className: "text-[12.5px] text-ink-700 leading-relaxed font-light"
}, window.DEEP.stressAI)));

/* Audit 2026-07-05 (R-2/R-3/R-4/R-6/R-8): the regime block used to hardcode a
   design-mock as_of date, a green «Рост» chip, «+»-prefixed sage values with
   «здоровый рост» captions regardless of the data, a green «ИИ подтверждает»
   header even on stance=diverges, and «RAG ·» chips that showed ETF momentum.
   Everything below is data-driven now. */
const _phaseChip = name => ({
  Expansion: {
    t: 'Рост',
    cls: 'bg-sage-500/12 text-sage-600'
  },
  Recovery: {
    t: 'Восстановление',
    cls: 'bg-sage-500/12 text-sage-600'
  },
  Slowdown: {
    t: 'Замедление',
    cls: 'bg-gold-400/18 text-gold-700'
  },
  Recession: {
    t: 'Спад',
    cls: 'bg-rust-500/12 text-rust-600'
  }
})[name] || {
  t: name,
  cls: 'bg-cream-50 text-ink-600'
};
const _axisCard = (label, v, posCaption, negCaption) => /*#__PURE__*/React.createElement("div", {
  className: "rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3"
}, /*#__PURE__*/React.createElement("div", {
  className: "text-[10px] uppercase tracking-wider text-ink-500 font-mono"
}, label), /*#__PURE__*/React.createElement("div", {
  className: `text-[20px] num font-semibold mt-0.5 ${v >= 0 ? 'text-sage-600' : 'text-rust-600'}`
}, (v >= 0 ? '+' : '') + v.toFixed(2)), /*#__PURE__*/React.createElement("div", {
  className: "text-[10px] text-ink-400"
}, v >= 0 ? posCaption : negCaption));
const _stanceHeader = stance => ({
  confirms: {
    t: 'ИИ подтверждает режим',
    cls: 'text-sage-600',
    warn: false
  },
  partial: {
    t: 'Частичное подтверждение ИИ',
    cls: 'text-gold-700',
    warn: true
  },
  diverges: {
    t: 'ИИ расходится с моделью',
    cls: 'text-rust-600',
    warn: true
  }
})[stance] || {
  t: 'ИИ-сверка режима',
  cls: 'text-ink-600',
  warn: false
};
const RegimeBlock = ({
  r
}) => {
  const phase = _phaseChip(r.name);
  const head = _stanceHeader(r.confirmStance);
  return /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-12 gap-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "glass-strong rounded-4xl p-6 shadow-card lift h-full flex flex-col"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between mb-2"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-ink-500 text-[12px] font-medium"
  }, "Growth × Cycle"), /*#__PURE__*/React.createElement("h3", {
    className: "text-xl font-semibold tracking-tight text-ink-900"
  }, "Координаты режима")), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-mono text-ink-400 tracking-wider px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5"
  }, "60-day")), /*#__PURE__*/React.createElement("div", {
    className: "flex-1 flex items-center justify-center py-2"
  }, /*#__PURE__*/React.createElement(RegimeQuadrant, {
    dot: r.dot,
    size: 300
  })))), /*#__PURE__*/React.createElement("div", {
    className: "col-span-12 lg:col-span-7"
  }, /*#__PURE__*/React.createElement("div", {
    className: "glass-strong rounded-4xl p-6 shadow-card lift h-full flex flex-col"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between mb-4"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] tracking-widest uppercase text-ink-500 font-mono mb-1"
  }, "Текущий режим"), /*#__PURE__*/React.createElement("div", {
    className: "flex items-baseline gap-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[32px] font-light tracking-tight text-ink-900"
  }, r.name), r.nameRu && r.nameRu !== r.name && /*#__PURE__*/React.createElement("span", {
    className: "text-[14px] text-ink-500 font-light"
  }, "· ", r.nameRu)), /*#__PURE__*/React.createElement("div", {
    className: "text-[11.5px] text-ink-500 font-mono mt-1"
  }, "Уверенность модели ", /*#__PURE__*/React.createElement("b", {
    className: "text-gold-700"
  }, r.confidence, "%"), " · ", r.confirms, " подтверждающих сигнала")), /*#__PURE__*/React.createElement("div", {
    className: `flex items-center gap-2 px-3 py-1.5 rounded-full text-[11px] font-semibold ${phase.cls}`
  }, /*#__PURE__*/React.createElement(Icons.Check, {
    size: 13,
    stroke: 2.2
  }), " ", phase.t)), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-2 gap-3 mb-4"
  }, _axisCard('Growth-фактор', r.growth, 'здоровый рост', 'рост под давлением'), _axisCard('Cycle-фактор', r.cycle, 'цикл. экспансия', 'цикл. сжатие')), /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] tracking-widest uppercase text-ink-400 font-mono mb-2"
  }, "Сигналы-драйверы", r.driversAsOf ? ` · as_of ${r.driversAsOf}` : ''), /*#__PURE__*/React.createElement("div", {
    className: "space-y-1.5 flex-1"
  }, r.drivers.map((d, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "grid grid-cols-[1fr_auto_auto_auto] items-center gap-3 py-1.5 border-b border-ink-900/5 last:border-0"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[11.5px] text-ink-700 truncate"
  }, d.name), /*#__PURE__*/React.createElement("span", {
    className: `text-[11.5px] num font-semibold ${d.tone === 'warn' ? 'text-gold-700' : 'text-sage-600'}`
  }, d.val), /*#__PURE__*/React.createElement("span", {
    className: "text-[9.5px] font-mono text-ink-400 whitespace-nowrap"
  }, d.trend), /*#__PURE__*/React.createElement("span", {
    className: `text-[8.5px] font-mono font-bold tracking-wider px-1.5 py-0.5 rounded-full ${d.tone === 'warn' ? 'bg-gold-400/18 text-gold-700' : 'bg-sage-500/12 text-sage-600'}`
  }, d.state)))))), /*#__PURE__*/React.createElement("div", {
    className: "col-span-12"
  }, /*#__PURE__*/React.createElement("div", {
    className: "rounded-4xl p-6 shadow-card",
    style: {
      background: 'linear-gradient(120deg, #f3f6f1 0%, #eef3ea 100%)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between mb-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: `flex items-center gap-2 text-[11px] font-semibold ${head.cls}`
  }, head.warn ? /*#__PURE__*/React.createElement(Icons.Warning, {
    size: 14,
    stroke: 2.2
  }) : /*#__PURE__*/React.createElement(Icons.Check, {
    size: 14,
    stroke: 2.2
  }), " ", head.t), /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-mono text-ink-400"
  }, window.DEEP.meta.aiModel)), /*#__PURE__*/React.createElement("p", {
    className: "text-[13.5px] text-ink-800 leading-relaxed font-light mb-4"
  }, r.confirm), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2"
  }, r.confirmBullets.map((b, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "flex items-start gap-2.5"
  }, b.ok ? /*#__PURE__*/React.createElement(Icons.Check, {
    size: 13,
    className: "text-sage-600 mt-0.5 flex-shrink-0",
    stroke: 2.4
  }) : /*#__PURE__*/React.createElement(Icons.Warning, {
    size: 13,
    className: "text-gold-700 mt-0.5 flex-shrink-0",
    stroke: 2
  }), /*#__PURE__*/React.createElement("span", {
    className: "text-[11.5px] text-ink-700 leading-snug"
  }, b.t)))), r.consistency && r.consistency.note ? /*#__PURE__*/React.createElement("div", {
    className: `mt-3 text-[11px] leading-snug ${r.consistency.status === 'aligned' ? 'text-sage-600' : 'text-gold-700'}`
  }, r.consistency.status === 'aligned' ? '✓ ' : '⚠ ', r.consistency.note) : null, /*#__PURE__*/React.createElement("div", {
    className: "mt-4 pt-3 border-t border-ink-900/8"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] tracking-widest uppercase font-mono text-ink-400 mb-2"
  }, r.ragBacked ? 'Из банковской аналитики · RAG' : 'Моментум-объяснители'), /*#__PURE__*/React.createElement("div", {
    className: "space-y-1.5"
  }, r.ragSignals.map((s, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "flex items-start gap-2"
  }, s.ok ? /*#__PURE__*/React.createElement(Icons.Check, {
    size: 12,
    className: "text-sage-600 mt-0.5 flex-shrink-0",
    stroke: 2.4
  }) : /*#__PURE__*/React.createElement(Icons.Warning, {
    size: 12,
    className: "text-gold-700 mt-0.5 flex-shrink-0",
    stroke: 2
  }), s.bank ? /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-mono font-bold tracking-wider text-ink-900 bg-ink-900/6 rounded px-1.5 py-0.5 flex-shrink-0"
  }, s.bank) : /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-mono font-semibold tracking-wider text-ink-400 flex-shrink-0"
  }, r.ragBacked ? 'RAG' : 'Моментум'), /*#__PURE__*/React.createElement("span", {
    className: "text-[11.5px] text-ink-700 leading-snug font-light"
  }, s.text))))))));
};
const StressRegime = () => {
  const p = window.DEEP;
  return /*#__PURE__*/React.createElement("section", {
    id: "stress",
    className: "rise",
    "data-screen-label": "04 Stress & Regime"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mb-6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-gold-400"
  }), " Stress test · Market regime · DEEP"), /*#__PURE__*/React.createElement("h2", {
    className: "text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900"
  }, "Устойчивость и контекст", /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, ".")), /*#__PURE__*/React.createElement("p", {
    className: "text-[15px] text-ink-500 mt-2 font-light max-w-[680px]"
  }, "Как портфель ведёт себя при гипотетических шоках и в какой фазе рынка мы находимся.")), /*#__PURE__*/React.createElement("div", {
    className: "mb-5"
  }, /*#__PURE__*/React.createElement(StressTable, {
    rows: p.stress
  })), /*#__PURE__*/React.createElement(RegimeBlock, {
    r: p.regime
  }), /*#__PURE__*/React.createElement("div", {
    className: "mt-5 rounded-3xl p-5 glass-strong shadow-card flex items-start gap-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-10 h-10 rounded-2xl bg-ink-900 text-gold-400 flex items-center justify-center flex-shrink-0"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 17,
    stroke: 1.7
  })), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] tracking-widest uppercase font-mono text-ink-500 mb-1"
  }, "AI · комментарий к режиму"), /*#__PURE__*/React.createElement("p", {
    className: "text-[14px] text-ink-800 leading-relaxed font-light"
  }, p.regime.regimeAI))));
};
Object.assign(window, {
  StressRegime
});
/* DEEP Plan — Action Plan levels + Expected effect + AI Ideas */

const actionChipCls = {
  BUY: 'bg-sage-500/15 text-sage-600',
  HOLD: 'bg-ink-900/6 text-ink-600',
  TRIM: 'bg-gold-500/18 text-gold-700',
  SELL: 'bg-rust-500 text-white'
};

// «Применить идею» → Scenario deep-link (same behaviour as the BASE report).
// Static page cannot charge a token; «Да» hands off to the Telegram bot, which
// runs the Scenario-tier analysis and charges 1 token there.
const scenarioDeepLink = (bot, n) => `https://t.me/${encodeURIComponent(String(bot || 'KEN_investment_bot').replace(/^@/, ''))}` + `?start=scn_${String(n).replace(/[^0-9A-Za-z_]/g, '')}`;
const ApplyIdeaModal = ({
  ideas,
  botUsername,
  onClose
}) => {
  const [sel, setSel] = React.useState(null);
  const go = () => {
    if (!sel) return;
    window.open(scenarioDeepLink(botUsername, sel.n), '_blank', 'noopener,noreferrer');
    onClose();
  };
  return /*#__PURE__*/React.createElement("div", {
    className: "fixed inset-0 z-50 flex items-center justify-center p-4",
    style: {
      background: 'rgba(28,27,26,0.55)',
      backdropFilter: 'blur(4px)'
    },
    onClick: onClose
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-full max-w-lg rounded-4xl bg-cream-50 shadow-card overflow-hidden",
    onClick: e => e.stopPropagation()
  }, /*#__PURE__*/React.createElement("div", {
    className: "px-6 pt-6 pb-4 border-b border-ink-900/8"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center justify-between"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 13,
    stroke: 1.8,
    className: "text-gold-600"
  }), " Применить идею"), /*#__PURE__*/React.createElement("button", {
    onClick: onClose,
    "aria-label": "Закрыть",
    className: "w-8 h-8 rounded-full bg-ink-900/5 text-ink-700 hover:bg-ink-900/10 flex items-center justify-center text-[13px]"
  }, "✕")), /*#__PURE__*/React.createElement("h3", {
    className: "text-[22px] font-light text-ink-900 mt-3 tracking-tight"
  }, sel ? 'Подтвердите запуск' : 'Выберите идею для сценарного анализа')), !sel ? /*#__PURE__*/React.createElement("div", {
    className: "p-4 space-y-2 max-h-[60vh] overflow-y-auto"
  }, ideas.map(idea => /*#__PURE__*/React.createElement("button", {
    key: idea.n,
    onClick: () => setSel(idea),
    className: "w-full text-left rounded-2xl px-4 py-3 bg-white/70 border border-ink-900/6 hover:border-ink-900/20 transition flex items-start gap-3"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] font-mono text-ink-400 mt-1"
  }, idea.n), /*#__PURE__*/React.createElement("span", {
    className: "min-w-0"
  }, /*#__PURE__*/React.createElement("span", {
    className: "block text-[10px] tracking-widest uppercase text-ink-400 font-mono"
  }, idea.cat), /*#__PURE__*/React.createElement("span", {
    className: "block text-[15px] text-ink-900 font-medium leading-tight mt-0.5"
  }, idea.title))))) : /*#__PURE__*/React.createElement("div", {
    className: "p-6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "rounded-2xl p-4 bg-gold-400/15 border border-gold-400/40"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] tracking-widest uppercase font-mono text-gold-700 mb-1.5"
  }, "Сценарный анализ"), /*#__PURE__*/React.createElement("p", {
    className: "text-[14px] text-ink-900 leading-relaxed"
  }, "Сделать ", /*#__PURE__*/React.createElement("span", {
    className: "font-semibold"
  }, "Scenario Analysis"), " для идеи «", sel.title, "» — с вас спишется ", /*#__PURE__*/React.createElement("span", {
    className: "font-semibold"
  }, "1 токен"), ".")), /*#__PURE__*/React.createElement("p", {
    className: "text-[12px] text-ink-500 mt-3 leading-relaxed font-light"
  }, "Откроется бот RAMP в Telegram и запустит сценарный анализ вашего портфеля. Токен списывается только после готового отчёта."), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-3 mt-5"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: go,
    className: "flex-1 px-4 py-2.5 rounded-full bg-ink-900 text-white text-[13px] font-semibold hover:bg-ink-800 transition"
  }, "Да"), /*#__PURE__*/React.createElement("button", {
    onClick: () => setSel(null),
    className: "flex-1 px-4 py-2.5 rounded-full bg-white/70 border border-ink-900/10 text-ink-700 text-[13px] font-semibold hover:bg-white transition"
  }, "Нет")))));
};
const ActionPlan = ({
  rows
}) => /*#__PURE__*/React.createElement("div", {
  className: "glass-strong rounded-4xl p-7 shadow-card"
}, /*#__PURE__*/React.createElement("div", {
  className: "flex items-start justify-between gap-4 flex-wrap mb-5"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h3", {
  className: "text-2xl font-semibold tracking-tight text-ink-900"
}, "Action Plan — уровни Buy / Sell / Stop"), /*#__PURE__*/React.createElement("p", {
  className: "text-[12px] text-ink-500 font-mono mt-1"
}, "ATR (Wilder RMA) · SMA50/200 · RSI(14) · MACD(12,26,9) · без внешних таргет-прайсов")), /*#__PURE__*/React.createElement("span", {
  className: "text-[10px] font-mono text-ink-400 tracking-wider px-2.5 py-1 rounded-full bg-cream-50 border border-ink-900/5"
}, "Quant Engine")), /*#__PURE__*/React.createElement("div", {
  className: "swipe-hint items-center gap-1 text-[10px] font-mono text-gold-700 bg-gold-400/15 rounded-full px-2.5 py-1 mb-2.5 w-max"
}, "↔ листайте таблицу"), /*#__PURE__*/React.createElement("div", {
  className: "mob-scroll-x"
}, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
  className: "grid grid-cols-[minmax(0,1fr)_72px_minmax(0,1.25fr)_minmax(0,0.9fr)_minmax(0,1.3fr)_minmax(0,0.9fr)_minmax(0,1.45fr)] gap-3 px-1 pb-2.5 text-[9.5px] tracking-widest uppercase text-ink-400 font-mono border-b border-ink-900/8"
}, /*#__PURE__*/React.createElement("div", null, "Тикер"), /*#__PURE__*/React.createElement("div", null, "Действие"), /*#__PURE__*/React.createElement("div", {
  className: "text-right"
}, "Количество"), /*#__PURE__*/React.createElement("div", {
  className: "text-right"
}, "Цена"), /*#__PURE__*/React.createElement("div", {
  className: "text-right"
}, "Sell target"), /*#__PURE__*/React.createElement("div", {
  className: "text-right"
}, "Stop"), /*#__PURE__*/React.createElement("div", null, "Причина")), /*#__PURE__*/React.createElement("div", {
  className: "divide-y divide-ink-900/5"
}, rows.map((r, i) => {
  const hasQty = r.qty != null && r.qty !== 0;
  const sell = r.qty != null && r.qty < 0 || r.dw != null && r.dw < 0;
  const qtyTone = !hasQty ? 'text-ink-400' : sell ? 'text-rust-600' : 'text-sage-600';
  // §−14 A-2: qty/dw идут из Black-Litterman-оптимизатора, а чип действия —
  // из 4-Pillar-скоринга; направления МОГУТ расходиться (честные данные двух
  // движков).  При противоречии красим Δw нейтрально и помечаем источник BL,
  // чтобы зелёный «+0.3 пп» не читался как ошибка под TRIM-чипом.
  const dwPos = r.dw != null && r.dw > 0;
  const contradicts = ['SELL', 'TRIM'].includes(r.action) && dwPos || ['BUY', 'STRONG'].includes(r.action) && r.dw != null && r.dw < 0;
  const dwTone = contradicts ? 'text-ink-400' : sell ? 'text-rust-500' : 'text-sage-500';
  return /*#__PURE__*/React.createElement("div", {
    key: i,
    className: `grid grid-cols-[minmax(0,1fr)_72px_minmax(0,1.25fr)_minmax(0,0.9fr)_minmax(0,1.3fr)_minmax(0,0.9fr)_minmax(0,1.45fr)] gap-3 items-center px-1 py-3 ${r.defer ? 'opacity-65' : ''}`
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[13.5px] font-bold num text-ink-900 flex items-center gap-1.5"
  }, r.t, r.hot && /*#__PURE__*/React.createElement("span", {
    className: "text-[10px]",
    title: "Hotspot TRC > 20%"
  }, "🔥")), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: `px-2 py-0.5 rounded-full text-[9px] font-bold tracking-wider ${actionChipCls[r.action]}`
  }, r.action)), /*#__PURE__*/React.createElement("div", {
    className: "text-right leading-tight"
  }, /*#__PURE__*/React.createElement("div", {
    className: `text-[12.5px] num font-semibold ${qtyTone}`
  }, hasQty ? `${r.qty > 0 ? '+' : '−'}${Math.abs(r.qty).toLocaleString('ru-RU')} шт` : '—'), r.dw != null && Math.abs(r.dw) >= 0.05 && /*#__PURE__*/React.createElement("div", {
    className: `text-[10px] num ${dwTone}`,
    title: "Δ веса по Black-Litterman-оптимизатору (может расходиться с сигналом 4-Pillar)"
  }, "BL ", r.dw > 0 ? '+' : '−', Math.abs(r.dw).toFixed(1), " пп")), /*#__PURE__*/React.createElement("div", {
    className: "text-right text-[12px] num text-ink-700"
  }, r.price.toFixed(2)), /*#__PURE__*/React.createElement("div", {
    className: "text-right text-[12px] num text-sage-600"
  }, r.target), /*#__PURE__*/React.createElement("div", {
    className: "text-right text-[12px] num text-rust-600"
  }, r.stop), /*#__PURE__*/React.createElement("div", {
    className: "text-[11px] text-ink-500 leading-tight"
  }, "Score ", r.score > 0 ? '+' : '', r.score.toFixed(1), r.hot && ' · Hotspot TRC>20%', r.defer && ' · отложено (turnover cap)'));
})))), /*#__PURE__*/React.createElement("div", {
  className: "mt-4 rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3.5 flex items-start gap-3"
}, /*#__PURE__*/React.createElement(Icons.Sparkles, {
  size: 14,
  className: "text-gold-600 mt-0.5 flex-shrink-0",
  stroke: 1.8
}), /*#__PURE__*/React.createElement("p", {
  className: "text-[12.5px] text-ink-700 leading-relaxed font-light"
}, window.DEEP.actionAI)));
const EffectGrid = ({
  rows,
  verdict,
  scope,
  scoped,
  actions
}) => {
  // 2026-07-18: split the traded positions into explicit SELL / BUY lists so
  // the panel says WHAT is sold and WHAT is bought — the before/after metric
  // cards below are the RESULT of exactly these trades.  Falls back to the
  // flat «Меняем позиции» chip when per-side actions aren't available.
  const acts = Array.isArray(actions) ? actions : [];
  const sells = acts.filter(a => a.key === 'sell' || (a.side || '').toLowerCase().startsWith('прод'));
  const buys = acts.filter(a => a.key === 'buy' || (a.side || '').toLowerCase().startsWith('куп'));
  const fmtDw = dw => dw != null && Math.abs(dw) >= 0.05 ? ` ${dw > 0 ? '+' : '−'}${Math.abs(dw).toFixed(1)}пп` : '';
  const nameList = arr => arr.map(a => `${a.t}${fmtDw(a.dw)}`).join(', ');
  return /*#__PURE__*/React.createElement("div", {
    className: "glass-strong rounded-4xl p-7 shadow-card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-4 flex-wrap mb-4"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h3", {
    className: "text-2xl font-semibold tracking-tight text-ink-900"
  }, "Ожидаемый эффект на риск"), /*#__PURE__*/React.createElement("p", {
    className: "text-[12px] text-ink-500 font-mono mt-1"
  }, "оценка «до / после» при исполнении Action Plan · горизонт 1 квартал")), scoped && scope && scope.length > 0 && acts.length === 0 && /*#__PURE__*/React.createElement("span", {
    className: "text-[10px] font-mono text-gold-700 tracking-wider px-2.5 py-1 rounded-full bg-gold-400/15",
    title: "Эти позиции меняет Action Plan — по ним и посчитан эффект «до/после»"
  }, "Меняем позиции: ", scope.join(' · '))), acts.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "flex flex-col sm:flex-row gap-2.5 mb-5"
  }, sells.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "flex-1 rounded-2xl bg-rust-500/8 border border-rust-500/20 px-4 py-2.5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[9.5px] uppercase tracking-wider text-rust-600 font-mono mb-1 flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement(Icons.ArrowR, {
    size: 11,
    stroke: 2.4,
    className: "rotate-90"
  }), " Продать / сократить"), /*#__PURE__*/React.createElement("div", {
    className: "text-[12.5px] num text-ink-800 leading-snug"
  }, nameList(sells))), buys.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "flex-1 rounded-2xl bg-sage-500/10 border border-sage-500/25 px-4 py-2.5"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[9.5px] uppercase tracking-wider text-sage-600 font-mono mb-1 flex items-center gap-1.5"
  }, /*#__PURE__*/React.createElement(Icons.ArrowR, {
    size: 11,
    stroke: 2.4,
    className: "-rotate-90"
  }), " Купить / нарастить"), /*#__PURE__*/React.createElement("div", {
    className: "text-[12.5px] num text-ink-800 leading-snug"
  }, nameList(buys)))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-2 md:grid-cols-4 gap-3"
  }, rows.map((r, i) => {
    const tone = {
      pos: 'text-sage-600',
      neg: 'text-rust-600',
      neut: 'text-ink-500'
    }[r.tone];
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      className: "rounded-2xl bg-white/70 border border-ink-900/6 px-4 py-3.5"
    }, /*#__PURE__*/React.createElement("div", {
      className: "text-[9.5px] uppercase tracking-wider text-ink-500 font-mono mb-2"
    }, r.name), /*#__PURE__*/React.createElement("div", {
      className: "flex items-baseline gap-1.5 num"
    }, /*#__PURE__*/React.createElement("span", {
      className: "text-[14px] text-ink-400"
    }, r.before), /*#__PURE__*/React.createElement(Icons.ArrowR, {
      size: 11,
      className: "text-gold-600",
      stroke: 2.4
    }), /*#__PURE__*/React.createElement("span", {
      className: "text-[16px] font-semibold text-ink-900"
    }, r.after)), /*#__PURE__*/React.createElement("div", {
      className: `text-[11px] num font-semibold mt-1.5 ${tone}`
    }, r.delta));
  })), /*#__PURE__*/React.createElement("div", {
    className: "mt-4 flex items-start gap-3 rounded-2xl bg-gold-400/12 border border-gold-400/35 px-4 py-3"
  }, /*#__PURE__*/React.createElement(Icons.Scale, {
    size: 15,
    className: "text-gold-700 mt-0.5 flex-shrink-0",
    stroke: 1.8
  }), /*#__PURE__*/React.createElement("p", {
    className: "text-[12px] text-ink-800 leading-relaxed font-light"
  }, /*#__PURE__*/React.createElement("span", {
    className: "font-semibold text-gold-700"
  }, "Сводный вердикт по плану:"), " ", verdict)), /*#__PURE__*/React.createElement("div", {
    className: "mt-3 rounded-2xl bg-cream-50 border border-ink-900/5 px-4 py-3.5 flex items-start gap-3"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 14,
    className: "text-gold-600 mt-0.5 flex-shrink-0",
    stroke: 1.8
  }), /*#__PURE__*/React.createElement("p", {
    className: "text-[12.5px] text-ink-700 leading-relaxed font-light"
  }, window.DEEP.effectAI)));
};
const ideaTone = {
  grow: {
    border: '#5d7c5c',
    chip: 'bg-sage-500/15 text-sage-600',
    icon: Icons.TrendUp
  },
  rebalance: {
    border: '#1c1b1a',
    chip: 'bg-ink-900/8 text-ink-700',
    icon: Icons.Refresh
  },
  rotation: {
    border: '#caa01a',
    chip: 'bg-gold-500/18 text-gold-700',
    icon: Icons.Compass
  },
  hedge: {
    border: '#a8a293',
    chip: 'bg-ink-900/6 text-ink-600',
    icon: Icons.Shield
  }
};
const PipeNode = ({
  n,
  label,
  last
}) => /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
  className: "flex-1 rounded-xl p-2.5 bg-white/70 border border-ink-900/5 min-w-0"
}, /*#__PURE__*/React.createElement("div", {
  className: "text-[8.5px] tracking-widest uppercase text-ink-400 font-mono mb-1"
}, label), /*#__PURE__*/React.createElement("div", {
  className: "text-[10.5px] text-ink-800 font-medium leading-tight"
}, n)), !last && /*#__PURE__*/React.createElement("div", {
  className: "hidden sm:flex items-center justify-center w-5 flex-shrink-0"
}, /*#__PURE__*/React.createElement(Icons.ArrowR, {
  size: 12,
  className: "text-ink-300",
  stroke: 2
})));
const IdeaCard = ({
  idea,
  open,
  onToggle,
  highlight
}) => {
  const tone = ideaTone[idea.tone];
  const IconC = tone.icon;
  const stages = ['Factor', 'Regime', 'Stress', 'RAG'];
  return /*#__PURE__*/React.createElement("div", {
    className: `rounded-4xl shadow-card lift overflow-hidden ${highlight ? 'text-white' : 'glass-strong'}`,
    style: highlight ? {
      background: 'linear-gradient(155deg, #2a2825 0%, #1c1b1a 100%)'
    } : {
      borderLeft: `2px solid ${tone.border}`
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "p-6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: `w-11 h-11 rounded-2xl flex items-center justify-center ${highlight ? 'bg-gold-400 text-ink-900' : 'bg-ink-900/5 text-ink-900'}`
  }, /*#__PURE__*/React.createElement(IconC, {
    size: 18,
    stroke: 1.7
  })), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: `text-[10px] tracking-widest uppercase font-mono mb-0.5 ${highlight ? 'text-white/50' : 'text-ink-400'}`
  }, "Идея ", idea.n, " · ", idea.cat), /*#__PURE__*/React.createElement("div", {
    className: `flex items-center gap-1.5 ${highlight ? 'text-gold-400' : 'text-ink-600'}`
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-current"
  }), /*#__PURE__*/React.createElement("span", {
    className: "text-[11px] font-semibold tracking-wide"
  }, idea.prio)))), /*#__PURE__*/React.createElement("button", {
    onClick: onToggle,
    className: `w-9 h-9 rounded-full flex items-center justify-center transition ${highlight ? 'bg-white/10 text-white hover:bg-white/20' : 'bg-ink-900/5 text-ink-700 hover:bg-ink-900/10'} ${open ? 'rotate-180' : ''}`
  }, /*#__PURE__*/React.createElement(Icons.Chevron, {
    size: 15
  }))), /*#__PURE__*/React.createElement("h3", {
    className: `mt-5 text-[20px] leading-[1.15] tracking-tight font-medium ${highlight ? 'text-white' : 'text-ink-900'}`
  }, idea.title), /*#__PURE__*/React.createElement("p", {
    className: `mt-2.5 text-[13px] leading-relaxed font-light ${highlight ? 'text-white/70' : 'text-ink-500'}`
  }, idea.lede), /*#__PURE__*/React.createElement("div", {
    className: "mt-5"
  }, /*#__PURE__*/React.createElement("div", {
    className: `text-[10px] tracking-widest uppercase font-mono mb-2 ${highlight ? 'text-white/40' : 'text-ink-400'}`
  }, idea.cands.length, " кандидат(а) — нет в портфеле"), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-2"
  }, idea.cands.map(c => /*#__PURE__*/React.createElement("span", {
    key: c.t,
    className: `px-3 py-1.5 rounded-xl text-[12px] font-bold num ${highlight ? 'bg-white/8 text-white' : 'bg-cream-50 border border-ink-900/8 text-ink-900'}`
  }, c.t))))), /*#__PURE__*/React.createElement("div", {
    className: "overflow-hidden transition-[max-height,opacity] duration-500 ease-out",
    style: {
      maxHeight: open ? 900 : 0,
      opacity: open ? 1 : 0
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: `px-6 pb-6 pt-2 space-y-5 ${highlight ? 'border-t border-white/10' : 'border-t border-ink-900/6'}`
  }, /*#__PURE__*/React.createElement("div", {
    className: "pt-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: `text-[10px] tracking-widest uppercase font-mono mb-2.5 ${highlight ? 'text-white/40' : 'text-ink-400'}`
  }, "Конвейер · Factor → Regime → Stress → RAG"), /*#__PURE__*/React.createElement("div", {
    className: `flex flex-col sm:flex-row sm:items-stretch gap-1.5 sm:gap-0 ${highlight ? '[&_div.rounded-xl]:!bg-white/8 [&_div.rounded-xl]:!border-white/10 [&_.text-ink-800]:!text-white [&_.text-ink-400]:!text-white/40 [&_svg]:!text-white/40' : ''}`
  }, idea.pipeline.map((s, i) => /*#__PURE__*/React.createElement(PipeNode, {
    key: i,
    n: s,
    label: stages[i],
    last: i === idea.pipeline.length - 1
  })))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: `text-[10px] tracking-widest uppercase font-mono mb-2.5 ${highlight ? 'text-white/40' : 'text-ink-400'}`
  }, "Почему именно эти бумаги"), /*#__PURE__*/React.createElement("div", {
    className: "space-y-2.5"
  }, idea.cands.map(c => /*#__PURE__*/React.createElement("div", {
    key: c.t,
    className: `rounded-2xl p-3.5 ${highlight ? 'bg-white/8' : 'bg-cream-50 border border-ink-900/5'}`
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 mb-1"
  }, /*#__PURE__*/React.createElement("span", {
    className: `text-[13px] font-bold num ${highlight ? 'text-white' : 'text-ink-900'}`
  }, c.t), /*#__PURE__*/React.createElement("span", {
    className: `text-[11px] ${highlight ? 'text-white/50' : 'text-ink-500'}`
  }, c.name), /*#__PURE__*/React.createElement("span", {
    className: `ml-auto text-[9px] font-mono px-1.5 py-0.5 rounded ${highlight ? 'bg-white/10 text-white/60' : 'bg-ink-900/5 text-ink-600'}`
  }, c.src)), /*#__PURE__*/React.createElement("p", {
    className: `text-[11.5px] leading-snug font-light ${highlight ? 'text-white/65' : 'text-ink-600'}`
  }, c.why))))))));
};
const Plan = () => {
  const p = window.DEEP;
  const [open, setOpen] = React.useState({
    '01': true
  });
  const toggle = n => setOpen(o => ({
    ...o,
    [n]: !o[n]
  }));
  const [applyOpen, setApplyOpen] = React.useState(false);
  return /*#__PURE__*/React.createElement("section", {
    id: "plan",
    className: "rise",
    "data-screen-label": "05 Action Plan"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mb-6"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-gold-400"
  }), " Action Plan · Effect · AI Ideas · DEEP"), /*#__PURE__*/React.createElement("h2", {
    className: "text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900"
  }, "От идей к конкретным уровням", /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, ".")), /*#__PURE__*/React.createElement("p", {
    className: "text-[15px] text-ink-500 mt-2 font-light max-w-[680px]"
  }, "Конкретные уровни Buy / Sell / Stop, оценка эффекта до/после и стратегические идеи с кандидатами.")), applyOpen && /*#__PURE__*/React.createElement(ApplyIdeaModal, {
    ideas: p.ideas,
    botUsername: (p.meta || {}).botUsername,
    onClose: () => setApplyOpen(false)
  }), /*#__PURE__*/React.createElement("div", {
    className: "space-y-5"
  }, /*#__PURE__*/React.createElement(ActionPlan, {
    rows: p.actionPlan
  }), /*#__PURE__*/React.createElement(EffectGrid, {
    rows: p.effect,
    verdict: p.effectVerdict,
    scope: p.effectScope,
    scoped: p.effectScoped,
    actions: p.effectActions
  }), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "rounded-4xl p-6 mb-5 relative overflow-hidden",
    style: {
      background: 'linear-gradient(120deg, #fbf3d9 0%, #f6ebc0 100%)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "absolute -right-6 top-1/2 -translate-y-1/2 w-40 h-40 rounded-full opacity-30",
    style: {
      background: 'radial-gradient(circle, #caa01a, transparent 65%)'
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "relative flex items-start justify-between gap-4 flex-wrap"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start gap-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-11 h-11 rounded-2xl bg-ink-900 text-gold-400 flex items-center justify-center flex-shrink-0"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 18,
    stroke: 1.7
  })), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] tracking-widest uppercase font-mono text-ink-700 mb-1"
  }, "AI Ideas · 4 идеи · каждая прошла Factor → Regime → Stress → RAG"), /*#__PURE__*/React.createElement("p", {
    className: "text-[14.5px] text-ink-900 leading-relaxed font-light"
  }, "Тикеры-кандидаты ", /*#__PURE__*/React.createElement("span", {
    className: "font-medium"
  }, "не из вашего портфеля"), " — рассмотрите как замену или дополнение. Раскройте карточку, чтобы увидеть конвейер отбора и обоснование по каждому кандидату."))), /*#__PURE__*/React.createElement("button", {
    onClick: () => setApplyOpen(true),
    className: "flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-ink-900 text-white text-[12px] font-medium hover:bg-ink-800 transition flex-shrink-0"
  }, /*#__PURE__*/React.createElement(Icons.Sparkles, {
    size: 13,
    stroke: 1.8
  }), " Применить идею"))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 lg:grid-cols-2 gap-5"
  }, p.ideas.map((idea, i) => /*#__PURE__*/React.createElement(IdeaCard, {
    key: idea.n,
    idea: idea,
    open: !!open[idea.n],
    onToggle: () => toggle(idea.n),
    highlight: i === 0
  }))))));
};
Object.assign(window, {
  Plan
});
/* DEEP CoVe — chain-of-verification provenance appendix */

const CoveItem = ({
  c
}) => {
  const cfg = {
    ok: {
      mark: '✓',
      cls: 'text-sage-600',
      bg: 'bg-sage-500/12'
    },
    warn: {
      mark: '!',
      cls: 'text-gold-700',
      bg: 'bg-gold-400/15'
    },
    fail: {
      mark: '✗',
      cls: 'text-rust-600',
      bg: 'bg-rust-500/12'
    }
  }[c.st];
  return /*#__PURE__*/React.createElement("div", {
    className: "flex items-start gap-3 py-2.5 border-b border-ink-900/5"
  }, /*#__PURE__*/React.createElement("span", {
    className: `w-5 h-5 rounded-md ${cfg.bg} ${cfg.cls} flex items-center justify-center text-[11px] font-bold flex-shrink-0 mt-0.5`
  }, cfg.mark), /*#__PURE__*/React.createElement("div", {
    className: "min-w-0"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[12px] text-ink-900 font-medium leading-tight"
  }, c.title), /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] text-ink-500 font-mono leading-snug mt-0.5"
  }, c.meta)));
};
const Cove = () => {
  const p = window.DEEP;
  const half = Math.ceil(p.cove.length / 2);
  const cols = [p.cove.slice(0, half), p.cove.slice(half)];
  const counts = p.cove.reduce((a, c) => {
    a[c.st]++;
    return a;
  }, {
    ok: 0,
    warn: 0,
    fail: 0
  });
  return /*#__PURE__*/React.createElement("section", {
    id: "cove",
    className: "rise",
    "data-screen-label": "06 CoVe"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-end justify-between gap-4 flex-wrap mb-6"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 text-[11px] tracking-widest uppercase text-ink-500 font-mono mb-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "w-1.5 h-1.5 rounded-full bg-gold-400"
  }), " Chain-of-Verification · DEEP"), /*#__PURE__*/React.createElement("h2", {
    className: "text-[40px] leading-[1.05] tracking-[-0.02em] font-light text-ink-900"
  }, "Откуда данные", /*#__PURE__*/React.createElement("span", {
    className: "text-ink-400"
  }, ".")), /*#__PURE__*/React.createElement("p", {
    className: "text-[15px] text-ink-500 mt-2 font-light max-w-[640px]"
  }, "Каждый показатель прослеживается до первичного источника с методом расчёта и статусом QualityGate.")), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2"
  }, /*#__PURE__*/React.createElement("span", {
    className: "flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-sage-500/12 text-sage-600 text-[11px] font-semibold"
  }, /*#__PURE__*/React.createElement("span", {
    className: "font-bold"
  }, "✓"), " ", counts.ok, " прошли"), /*#__PURE__*/React.createElement("span", {
    className: "flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-gold-400/15 text-gold-700 text-[11px] font-semibold"
  }, /*#__PURE__*/React.createElement("span", {
    className: "font-bold"
  }, "!"), " ", counts.warn, " частично"), /*#__PURE__*/React.createElement("span", {
    className: "flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-rust-500/12 text-rust-600 text-[11px] font-semibold"
  }, /*#__PURE__*/React.createElement("span", {
    className: "font-bold"
  }, "✗"), " ", counts.fail, " недоступно"))), /*#__PURE__*/React.createElement("div", {
    className: "glass-strong rounded-4xl p-7 shadow-card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 lg:grid-cols-2 gap-x-10"
  }, cols.map((col, ci) => /*#__PURE__*/React.createElement("div", {
    key: ci
  }, col.map((c, i) => /*#__PURE__*/React.createElement(CoveItem, {
    key: i,
    c: c
  }))))), /*#__PURE__*/React.createElement("div", {
    className: "mt-5 pt-4 border-t border-ink-900/8 text-[11px] text-ink-500 leading-relaxed font-light"
  }, /*#__PURE__*/React.createElement("b", {
    className: "text-sage-600"
  }, "✓"), " — данные прошли QualityGate · ", /*#__PURE__*/React.createElement("b", {
    className: "text-gold-700"
  }, "!"), " — частичное покрытие или fallback на устаревший кэш · ", /*#__PURE__*/React.createElement("b", {
    className: "text-rust-600"
  }, "✗"), " — источник недоступен. Документ не является ИИР (индивидуальной инвестиционной рекомендацией) — только аналитический материал.")));
};
Object.assign(window, {
  Cove
});
/* DEEP app shell — topbar, nav, sections, footer */

const NAV = [{
  id: 'overview',
  label: 'Обзор',
  short: '01'
}, {
  id: 'holdings',
  label: 'Бумаги',
  short: '02'
}, {
  id: 'factors',
  label: 'Факторы',
  short: '03'
}, {
  id: 'stress',
  label: 'Стресс · Режим',
  short: '04'
}, {
  id: 'plan',
  label: 'План',
  short: '05'
}, {
  id: 'cove',
  label: 'CoVe',
  short: '06'
}];
const TopBar = ({
  active
}) => {
  const meta = window.DEEP.meta;
  const onJump = id => {
    const el = document.getElementById(id);
    if (el) window.scrollTo({
      top: el.getBoundingClientRect().top + window.scrollY - 96,
      behavior: 'smooth'
    });
  };
  return /*#__PURE__*/React.createElement("header", {
    className: "sticky top-0 z-40 px-6 pt-5 pb-3 backdrop-blur-md",
    style: {
      background: 'linear-gradient(to bottom, rgba(251,248,241,0.88) 0%, rgba(251,248,241,0.65) 70%, transparent 100%)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "max-w-[1480px] mx-auto flex items-center justify-between gap-4"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 px-4 py-2 rounded-full bg-white/80 border border-ink-900/8 shadow-sm flex-shrink-0"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-5 h-5 rounded-md bg-ink-900 flex items-center justify-center"
  }, /*#__PURE__*/React.createElement("div", {
    className: "w-2 h-2 rounded-sm bg-gold-400"
  })), /*#__PURE__*/React.createElement("span", {
    className: "font-bold tracking-tight text-[14px] text-ink-900"
  }, "DEEP"), /*#__PURE__*/React.createElement("span", {
    className: "px-1.5 py-0.5 rounded-md bg-gold-400/30 text-gold-700 text-[9px] font-mono font-bold tracking-wider"
  }, "TIER")), /*#__PURE__*/React.createElement("nav", {
    className: "hidden lg:flex items-center gap-1 p-1 rounded-full bg-white/70 border border-ink-900/8 backdrop-blur-md"
  }, NAV.map(n => /*#__PURE__*/React.createElement("button", {
    key: n.id,
    onClick: () => onJump(n.id),
    className: `flex items-center gap-2 px-3.5 py-2 rounded-full text-[12.5px] font-medium transition-colors
                          ${active === n.id ? 'bg-ink-900 text-white' : 'text-ink-700 hover:bg-ink-900/5'}`
  }, /*#__PURE__*/React.createElement("span", {
    className: `text-[10px] font-mono opacity-60 ${active === n.id ? 'text-gold-400' : ''}`
  }, n.short), n.label))), /*#__PURE__*/React.createElement("div", {
    className: "flex items-center gap-2 flex-shrink-0"
  }, /*#__PURE__*/React.createElement("button", {
    className: "hidden sm:flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-white/70 border border-ink-900/8 text-[12px] text-ink-700 hover:bg-white transition"
  }, /*#__PURE__*/React.createElement(Icons.Download, {
    size: 13,
    stroke: 1.8
  }), " ", /*#__PURE__*/React.createElement("span", {
    className: "hidden xl:inline"
  }, "PDF")), /*#__PURE__*/React.createElement("div", {
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
  }, "/"), /*#__PURE__*/React.createElement("span", null, "AI · ", meta.aiModel), /*#__PURE__*/React.createElement("span", {
    className: "opacity-30 hidden sm:inline"
  }, "/"), /*#__PURE__*/React.createElement("span", {
    className: "hidden sm:inline"
  }, "Профиль · ", meta.profile)), /*#__PURE__*/React.createElement("span", null, "Generated ", meta.generated)));
};
const Footer = () => {
  const p = window.DEEP;
  // §−14 C-6: ChromaDB (банковский RAG) в списке источников — только когда RAG
  // реально консультировался (quality несёт «✓ … RAG …»); при выключенном/пустом
  // RAG источник не заявляется (футер был статичным — вводил в заблуждение).
  const ragOn = (p.quality || []).some(q => typeof q === 'string' && q.includes('✓') && /rag/i.test(q));
  const sources = ['Tradernet', 'SEC EDGAR', 'FRED', 'Quant Engine MAC3'].concat(ragOn ? ['ChromaDB (GS / MS / JPM)'] : []).concat([p.meta.aiModel]).join(' · ');
  return /*#__PURE__*/React.createElement("footer", {
    className: "mt-16 mb-8"
  }, /*#__PURE__*/React.createElement("div", {
    className: "rounded-4xl p-7 glass-strong shadow-card"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex items-start justify-between gap-6 flex-wrap"
  }, /*#__PURE__*/React.createElement("div", {
    className: "max-w-[560px]"
  }, /*#__PURE__*/React.createElement("div", {
    className: "text-[10px] tracking-widest uppercase text-ink-400 font-mono mb-3"
  }, "Контроль качества данных"), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-2"
  }, p.quality.map((q, i) => {
    // Audit 2026-07-04: each quality string already carries its OWN
    // status symbol («✓ …» / «✗ …» / «⚠ …» / «— …»).  The footer used
    // to HARDCODE a green ✓ in front, so an «✗ RAG» pill rendered as
    // «✓ ✗ RAG» (double symbol, green on a failure).  Colour the
    // string's own leading symbol instead — honest for every state.
    const s = String(q).trim();
    const sym = s.charAt(0);
    const rest = s.slice(1).trim();
    const cls = sym === '✗' ? 'text-rust-600' : sym === '⚠' ? 'text-gold-700' : sym === '—' ? 'text-ink-400' : 'text-sage-600';
    return /*#__PURE__*/React.createElement("span", {
      key: i,
      className: "flex items-center gap-1.5 text-[10px] font-mono text-ink-600 bg-cream-50 border border-ink-900/6 rounded-full px-2.5 py-1"
    }, /*#__PURE__*/React.createElement("span", {
      className: `${cls} font-bold`
    }, sym), " ", rest);
  }))), /*#__PURE__*/React.createElement("div", {
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
    className: "mt-6 pt-5 border-t border-ink-900/8 flex items-start gap-3"
  }, /*#__PURE__*/React.createElement(Icons.Warning, {
    size: 15,
    className: "text-rust-500 mt-0.5 flex-shrink-0",
    stroke: 1.8
  }), /*#__PURE__*/React.createElement("p", {
    className: "text-[12px] text-ink-500 leading-relaxed font-light"
  }, /*#__PURE__*/React.createElement("span", {
    className: "text-ink-700 font-medium"
  }, "Это аналитический материал, а не индивидуальная инвестиционная рекомендация."), " Расчёты основаны на исторических данных и публичной отчётности; они не учитывают вашу налоговую ситуацию, горизонт и цели. Источники: ", sources, "."))), /*#__PURE__*/React.createElement("div", {
    className: "text-center text-[10px] tracking-widest uppercase text-ink-400 font-mono mt-6"
  }, "Portfolio Risk Report · DEEP Tier · ", p.meta.id, " · v2026.6"));
};
const Fab = () => /*#__PURE__*/React.createElement("div", {
  className: "fixed bottom-6 right-6 z-50"
}, /*#__PURE__*/React.createElement("button", {
  title: "Пересчитать риск",
  onClick: () => window.scrollTo({
    top: 0,
    behavior: 'smooth'
  }),
  className: "w-14 h-14 rounded-full bg-ink-900 text-gold-400 shadow-card-lg flex items-center justify-center hover:scale-105 transition"
}, /*#__PURE__*/React.createElement(Icons.Sparkles, {
  size: 20,
  stroke: 1.7
})));
const useActiveSection = () => {
  const [active, setActive] = React.useState('overview');
  React.useEffect(() => {
    const els = NAV.map(n => document.getElementById(n.id)).filter(Boolean);
    const obs = new IntersectionObserver(entries => {
      const visible = entries.filter(e => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible[0]) setActive(visible[0].target.id);
    }, {
      rootMargin: '-110px 0px -55% 0px',
      threshold: 0
    });
    els.forEach(el => obs.observe(el));
    return () => obs.disconnect();
  }, []);
  return active;
};
const App = () => {
  const active = useActiveSection();
  React.useEffect(() => {
    document.getElementById('root').classList.remove('preload');
  }, []);
  return /*#__PURE__*/React.createElement("div", {
    className: "min-h-screen"
  }, /*#__PURE__*/React.createElement(TopBar, {
    active: active
  }), /*#__PURE__*/React.createElement("main", {
    className: "max-w-[1480px] mx-auto px-6 pt-8 pb-12 space-y-20"
  }, /*#__PURE__*/React.createElement(Overview, null), /*#__PURE__*/React.createElement(Holdings, null), /*#__PURE__*/React.createElement(Factors, null), /*#__PURE__*/React.createElement(StressRegime, null), /*#__PURE__*/React.createElement(Plan, null), /*#__PURE__*/React.createElement(Cove, null), /*#__PURE__*/React.createElement(Footer, null)), /*#__PURE__*/React.createElement(Fab, null));
};
ReactDOM.createRoot(document.getElementById('root')).render(/*#__PURE__*/React.createElement(App, null));
