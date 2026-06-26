/* Lucide-style thin-stroke icons as React components (1.5 stroke). */

const Icon = ({ d, size=18, stroke=1.6, className='', children, viewBox='0 0 24 24', fill='none' }) =>
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox={viewBox} fill={fill}
       stroke="currentColor" strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round" className={className}>
    {d ? <path d={d}/> : children}
  </svg>;

const Icons = {
  Sparkles:  (p) => <Icon {...p}><path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/><path d="M19 14l.7 1.8L21.5 16.5l-1.8.7L19 19l-.7-1.8L16.5 16.5l1.8-.7z"/></Icon>,
  TrendUp:   (p) => <Icon {...p}><polyline points="3 17 9 11 13 15 21 7"/><polyline points="14 7 21 7 21 14"/></Icon>,
  TrendDown: (p) => <Icon {...p}><polyline points="3 7 9 13 13 9 21 17"/><polyline points="14 17 21 17 21 10"/></Icon>,
  Briefcase: (p) => <Icon {...p}><rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></Icon>,
  Wallet:    (p) => <Icon {...p}><path d="M3 8a2 2 0 0 1 2-2h14v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M16 13h3"/><path d="M19 6V5a2 2 0 0 0-2-2H6"/></Icon>,
  Bell:      (p) => <Icon {...p}><path d="M6 8a6 6 0 1 1 12 0c0 4 1.5 6 1.5 6h-15S6 12 6 8z"/><path d="M10 19a2 2 0 0 0 4 0"/></Icon>,
  Search:    (p) => <Icon {...p}><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></Icon>,
  Settings:  (p) => <Icon {...p}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h0a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v0a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></Icon>,
  ArrowUR:   (p) => <Icon {...p}><path d="M7 17L17 7"/><path d="M8 7h9v9"/></Icon>,
  ArrowR:    (p) => <Icon {...p}><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></Icon>,
  Plus:      (p) => <Icon {...p}><path d="M12 5v14"/><path d="M5 12h14"/></Icon>,
  Minus:     (p) => <Icon {...p}><path d="M5 12h14"/></Icon>,
  Chevron:   (p) => <Icon {...p}><polyline points="6 9 12 15 18 9"/></Icon>,
  Warning:   (p) => <Icon {...p}><path d="M12 3 22 20H2L12 3z"/><path d="M12 10v5"/><circle cx="12" cy="18" r=".5" fill="currentColor"/></Icon>,
  Shield:    (p) => <Icon {...p}><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/></Icon>,
  Pulse:     (p) => <Icon {...p}><path d="M3 12h4l2-6 4 12 2-6h6"/></Icon>,
  Layers:    (p) => <Icon {...p}><path d="M12 3 2 8l10 5 10-5z"/><path d="m2 13 10 5 10-5"/><path d="m2 18 10 5 10-5"/></Icon>,
  Filter:    (p) => <Icon {...p}><path d="M3 5h18l-7 9v6l-4-2v-4z"/></Icon>,
  Refresh:   (p) => <Icon {...p}><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.5 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.65 4.36A9 9 0 0 0 20.5 15"/></Icon>,
  Dot:       (p) => <Icon {...p} viewBox="0 0 8 8"><circle cx="4" cy="4" r="3" fill="currentColor"/></Icon>,
  Download:  (p) => <Icon {...p}><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></Icon>,
  Share:     (p) => <Icon {...p}><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8.6 13.5 6.8 4M15.4 6.5l-6.8 4"/></Icon>,
  User:      (p) => <Icon {...p}><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></Icon>,
  Globe:     (p) => <Icon {...p}><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></Icon>,
  Cpu:       (p) => <Icon {...p}><rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/></Icon>,
  Heart:     (p) => <Icon {...p}><path d="M12 21s-7-4.5-9.5-9A5.5 5.5 0 0 1 12 6a5.5 5.5 0 0 1 9.5 6c-2.5 4.5-9.5 9-9.5 9z"/></Icon>,
  Coffee:    (p) => <Icon {...p}><path d="M4 8h12v6a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4z"/><path d="M16 10h2a3 3 0 0 1 0 6h-2"/><path d="M7 4v2M10 4v2M13 4v2"/></Icon>,
  Bolt:      (p) => <Icon {...p}><path d="m13 2-9 12h7l-1 8 9-12h-7z"/></Icon>,
  Fuel:      (p) => <Icon {...p}><rect x="3" y="4" width="11" height="16" rx="2"/><path d="M14 9h3a2 2 0 0 1 2 2v7a1 1 0 0 1-2 0v-3a1 1 0 0 0-1-1h-2"/><path d="M19 7l-2-2"/></Icon>,
  Tag:       (p) => <Icon {...p}><path d="M12 2H4a2 2 0 0 0-2 2v8l10 10 10-10z"/><circle cx="7" cy="7" r="1.5" fill="currentColor"/></Icon>,
  Spark:     (p) => <Icon {...p}><path d="M12 2v6M12 16v6M4 12H2M22 12h-2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5"/></Icon>,
};

// Map sector name → icon
const sectorIcon = (cls) => ({
  'Технологии':       Icons.Cpu,
  'Здравоохранение':  Icons.Heart,
  'Потреб. товары':   Icons.Coffee,
  'Энергетика':       Icons.Fuel,
  'Облигации':        Icons.Shield,
}[cls] || Icons.Tag);

window.Icons = Icons;
window.sectorIcon = sectorIcon;
