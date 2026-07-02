"""
Smoke-render mock fixtures for the report templates (§−14 C-7).

Extracted from html_renderer.py, which had grown ~450 lines of data constants
inside the RENDERER module.  Shapes mirror what pdf_payload / the engine
helpers emit — same field names and types — so the same template bindings work
in production.  Consumed ONLY by html_renderer._mock_payload (CLI smoke) and
the test fixtures; production rendering never touches this module.
"""
from __future__ import annotations

# ── Mock fixtures (smoke render only — production uses real engine output) ──
# All shapes mirror what build_payload / engine helpers emit, so the same
# Jinja bindings work without modification in production.

_MOCK_STRESS_SCENARIOS = [
    {"name": "Equity DM −20%",  "port_pct": -0.158, "port_dollar":  -7900,
     "max_dd_pct": -0.220, "recovery_months": 6, "tag": "Tech sell-off"},
    {"name": "Equity EM −25%",  "port_pct": -0.075, "port_dollar":  -3750,
     "max_dd_pct": -0.110, "recovery_months": 4, "tag": "EM rout"},
    {"name": "Rates +100 bp",   "port_pct": -0.038, "port_dollar":  -1900,
     "max_dd_pct": -0.062, "recovery_months": 3, "tag": "Duration shock"},
    {"name": "HY Credit +200bp","port_pct": -0.028, "port_dollar":  -1400,
     "max_dd_pct": -0.044, "recovery_months": 3, "tag": "Credit widening"},
    {"name": "USD +10% (proxy)","port_pct":  0.012, "port_dollar":    600,
     "max_dd_pct": -0.018, "recovery_months": 2, "tag": "DXY rally"},
    {"name": "Oil −30%",        "port_pct": -0.014, "port_dollar":   -700,
     "max_dd_pct": -0.025, "recovery_months": 2, "tag": "Energy drawdown"},
    {"name": "CPI +1pp (proxy)","port_pct": -0.022, "port_dollar":  -1100,
     "max_dd_pct": -0.038, "recovery_months": 3, "tag": "Inflation surprise"},
]

_MOCK_EXPECTED_EFFECT = {
    "risk_index":      {"before": 62,    "after": 54,    "delta_pp": -8,    "favourable": True},
    "cvar_95":         {"before": -0.052,"after": -0.041,"delta_pp": 1.1,   "favourable": True},
    "sharpe":          {"before": 1.18,  "after": 1.32,  "delta_pp": 0.14,  "favourable": True},
    "max_drawdown":    {"before": -0.128,"after": -0.104,"delta_pp": 2.4,   "favourable": True},
    "vol":             {"before": 0.148, "after": 0.126, "delta_pp": -2.2,  "favourable": True},
    "max_erc_pct":     {"before": 0.244, "after": 0.168, "delta_pp": -7.6,  "favourable": True},
    "expected_return": {"before": 0.142, "after": 0.126, "delta_pp": -1.6,  "favourable": False},
    "it_share":        {"before": 0.62,  "after": 0.50,  "delta_pp": -12.0, "favourable": True},
}

_MOCK_MACRO_DRIVERS = {
    "as_of":  "2026-05-14",
    "regime": "Expansion (late)",
    "series": [
        {"id":"T10Y2Y","name":"Yield curve 10Y−2Y","value":"+0.18 pp","as_of":"2026-05-14","status":"ok",
         "comment":"положительный наклон — рецессия не сигналит"},
        {"id":"BAMLH0A0HYM2","name":"HY OAS (ICE BofA)","value":"312 bp","as_of":"2026-05-14","status":"ok",
         "comment":"спреды узкие — рынок не закладывает кредитный риск"},
        {"id":"VIXCLS","name":"CBOE VIX","value":"14.2","as_of":"2026-05-14","status":"ok",
         "comment":"низкая ожидаемая волатильность — комфортная зона"},
        {"id":"T10YIE","name":"10Y Breakeven Inflation","value":"2.34%","as_of":"2026-05-14","status":"ok",
         "comment":"инфляционные ожидания у цели ФРС 2%"},
    ],
}

_MOCK_REGIME = {
    "label":      "Expansion (late)",
    "confidence": 72,
    "growth":     0.08,
    "cycle":      0.04,
    "explainers": [
        "SPY обгоняет IEF на +4.1% за 60 дней (рост vs облигации)",
        "Discretionary > Staples: +2.8% за 60д (цикличные on)",
        "EEM 60д +1.2% (EM risk-on)",
    ],
}

_MOCK_ACTION_PLAN = [
    {"ticker":"AAPL","action":"Sell 25%","reason":"HOTSPOT 24% риска; фиксируем часть прибыли (+25%)",
     "price":"$197.40","buy_zone":"$182–188","sell_target":"$215","stop_loss":"$175"},
    {"ticker":"KSPI","action":"Reduce 50%","reason":"TRC 22% при просадке −16.7% — концентрация выше порога",
     "price":"₸123.50","buy_zone":"₸115–119","sell_target":"₸140","stop_loss":"₸108"},
    {"ticker":"BND","action":"Hold","reason":"Hedge против rate shock; держим вес 20%",
     "price":"$72.18","buy_zone":"$71–73","sell_target":"$76","stop_loss":"$69"},
]

_MOCK_HOTSPOTS = [
    {"ticker":"AAPL","trc_pct":18.4,"reason":"TRC 18.4% при весе 50% — высокая концентрация"},
    {"ticker":"KSPI","trc_pct":22.1,"reason":"TRC 22% — KSPI EM hotspot"},
]

# Risk waterfall — engine shape from pdf_payload._build_risk_waterfall.
# Sample numbers chosen so sum_standalone − total = positive diversification.
_MOCK_RISK_WATERFALL = {
    "contributions": [
        {"ticker": "AAPL", "weight_pct": 50.0, "standalone_vol_pct": 24.5,
         "standalone_pp": 12.25, "standalone_share_pct": 67.3},
        {"ticker": "KSPI", "weight_pct": 30.0, "standalone_vol_pct": 18.0,
         "standalone_pp":  5.40, "standalone_share_pct": 29.7},
        {"ticker": "BND",  "weight_pct": 20.0, "standalone_vol_pct":  2.7,
         "standalone_pp":  0.54, "standalone_share_pct":  3.0},
    ],
    "sum_standalone_pp":     18.19,
    "total_vol_pp":          14.20,
    "diversification_pp":     3.99,
    "diversification_ratio":  0.219,
    "method": ("σᵢ = √Σᵢᵢ (annualised) · standalone = wᵢ·σᵢ · "
                "diversified = √(w'Σw) · benefit = Σstandalone − diversified"),
}

# 4-pillar scoring — engine shape from pdf_payload (asset_scores).
_MOCK_SCORE_BREAKDOWN = [
    {"ticker":"AAPL","fundamentals":"+2.0","valuations": "0.0","technicals":"+1.0","credit":"+1.0",
     "total":"+4.0","action":"Buy","action_color":"pos"},
    {"ticker":"KSPI","fundamentals":"+1.0","valuations":"+0.5","technicals":"-1.0","credit": "0.0",
     "total":"+0.5","action":"Hold","action_color":"neut"},
    {"ticker":"BND", "fundamentals": "0.0","valuations": "0.0","technicals": "0.0","credit":"+1.0",
     "total":"+1.0","action":"Hold","action_color":"neut"},
]

# Scenarios (benchmark comparison) — payload shape from pdf_payload (scenarios).
_MOCK_SCENARIOS = [
    {"name": "S&P 500", "excess": "+5.1%", "te": "8.4%", "ir": "0.61",
     "beating": True,  "color": "pos", "pnl": "+9.1%"},
]

# Multi-period returns — payload shape from results.period_returns_table.
# Engine builds this from benchmark_comparison.periods list.  Per-period
# shape: {period_label, portfolio_return, benchmark_return, excess}.
_MOCK_PERIOD_RETURNS = {
    "S&P 500": {
        "periods": [
            {"label":"1М",  "portfolio":"+2.0%", "benchmark":"+1.4%", "excess":"+0.6 пп"},
            {"label":"3М",  "portfolio":"+5.5%", "benchmark":"+3.6%", "excess":"+1.9 пп"},
            {"label":"6М",  "portfolio":"+9.4%", "benchmark":"+6.0%", "excess":"+3.4 пп"},
            {"label":"YTD", "portfolio":"+7.1%", "benchmark":"+4.4%", "excess":"+2.7 пп"},
            {"label":"12М", "portfolio":"+14.2%","benchmark":"+9.1%", "excess":"+5.1 пп"},
        ],
        "window_start": "2025-05-15",
        "window_end":   "2026-05-14",
    },
}

# AI stock picks — Haiku/Sonnet output shape.  DEEP-specific schema:
#   • 4 ideas across 4 buckets (risk_reduction · diversification · growth · hedge)
#   • 4-stage pipeline (FACTOR → REGIME → STRESS → RAG) — adds STRESS vs BASE's 3-stage
#   • candidates carry mini-info (beta, sector) — DEEP enrichment
#   • expected_effect lists 3 metrics — DEEP enrichment (BASE shows 2)
#   • sources include a stress reference chip — DEEP enrichment
# Engine populates via AdvisorBot.generate_stock_picks(tier='deep').
_MOCK_AI_STOCK_PICKS = {
    "risk_reduction": [
        {"idea_num":"01", "category":"Снижение риска", "priority":"high",
         "title":"Сократить долю AAPL с 50% до ~30%",
         "rationale":"AAPL помечен HOTSPOT — 18% всего риска портфеля при доходности "
                     "+25% (есть что фиксировать без убытка).  В late-cycle режиме при "
                     "коррекции IT перевес 50% становится критичным.",
         "candidates":[
            {"ticker":"TXN",  "name":"Texas Instruments", "scenario":"стабильный чип-производитель, vol/2 vs AAPL",
             "beta":0.85, "sector":"Technology"},
            {"ticker":"CSCO", "name":"Cisco",             "scenario":"зрелый тех с дивидендом и низкой бетой",
             "beta":0.72, "sector":"Technology"},
            {"ticker":"ACN",  "name":"Accenture",         "scenario":"IT-консалтинг с устойчивой выручкой",
             "beta":1.05, "sector":"Technology"},
         ],
         "pipeline":[
            ("FACTOR", "AAPL даёт 18% риска, HOTSPOT (TRC > 20% порога)"),
            ("REGIME", "late-cycle Expansion, рост vol ожидается в IT"),
            ("STRESS", "под Equity DM −20%: −15.8% vs −9.5% после rebalance"),
            ("RAG",    "Morgan Stanley: фиксировать прибыль в перегретых техах"),
         ],
         "expected_effect":[
            ("Вклад AAPL в риск", "18.4%", "~12%"),
            ("Доля IT-сектора",   "50%",   "~38%"),
            ("Sharpe Ratio",      "1.18",  "1.24"),
         ],
         "sources":["Quant Engine", "SEC EDGAR", "Stress: Equity DM −20%", "RAG: MS_TechOutlook_2026"],
        },
    ],
    "diversification": [
        {"idea_num":"02", "category":"Диверсификация", "priority":"medium",
         "title":"Добавить healthcare 10–15%",
         "rationale":"Текущая структура: 50% IT + 30% KSPI EM = 80% в двух высоко-β "
                     "секторах.  Healthcare исторически некоррелирован с IT, "
                     "стабилизирует портфель в late-cycle transitions.",
         "candidates":[
            {"ticker":"JNJ", "name":"Johnson & Johnson",  "scenario":"healthcare hedge, β 0.74",
             "beta":0.74, "sector":"Healthcare"},
            {"ticker":"UNH", "name":"UnitedHealth",       "scenario":"managed care, defensive growth",
             "beta":0.68, "sector":"Healthcare"},
            {"ticker":"PFE", "name":"Pfizer",             "scenario":"large-cap pharma с дивидендом",
             "beta":0.65, "sector":"Healthcare"},
         ],
         "pipeline":[
            ("FACTOR", "Concentration HHI = 3520 → 'concentrated' band"),
            ("REGIME", "late-cycle: защитные сектора начинают opt"),
            ("STRESS", "под Equity DM −20%: текущ. −15.8%, с healthcare −11.2%"),
            ("RAG",    "JPMorgan: 'islands of stability' в healthcare/staples"),
         ],
         "expected_effect":[
            ("Concentration HHI",     "3520",  "~2300"),
            ("Beta к рынку",          "1.18",  "~0.98"),
            ("Max Drawdown (модель)", "−12.8%","~−10.2%"),
         ],
         "sources":["Quant Engine", "Stress: Equity DM −20%", "RAG: JPM_Strategy_Q2_2026"],
        },
    ],
    "growth": [
        {"idea_num":"03", "category":"Рост качества", "priority":"medium",
         "title":"Усилить позицию в MSFT при откатах",
         "rationale":"MSFT даёт качественный экспозиш в IT без HOTSPOT: вес 8%, "
                     "TRC только 9%.  4-Pillar Total +3.5 (sky-high). "
                     "Buy zone $402-408 (SMA50 − 1·ATR).",
         "candidates":[
            {"ticker":"MSFT", "name":"Microsoft",        "scenario":"core add — quality at scale",
             "beta":0.98, "sector":"Technology"},
            {"ticker":"GOOGL","name":"Alphabet",         "scenario":"diversified ad+cloud экспозиш",
             "beta":1.05, "sector":"Communication Services"},
            {"ticker":"V",    "name":"Visa",             "scenario":"payments — высокий ROE, низкий β",
             "beta":0.92, "sector":"Financial Services"},
         ],
         "pipeline":[
            ("FACTOR", "MSFT score F+1 V+1 T+1 C+0.5 (Total +3.5)"),
            ("REGIME", "late-cycle: high-quality holds value лучше"),
            ("STRESS", "под все 7 сценариев: −7% медиана (лучше портфеля)"),
            ("RAG",    "Goldman: maintain quality bias in tech"),
         ],
         "expected_effect":[
            ("MSFT weight",           "8%",   "12%"),
            ("Quality score (port)",  "+1.8", "+2.4"),
            ("Sharpe Ratio",          "1.18", "1.21"),
         ],
         "sources":["Quant Engine", "SEC EDGAR", "Stress: 7-scenario median", "RAG: GS_Q2_2026"],
        },
    ],
    "hedge": [
        {"idea_num":"04", "category":"Хедж rate-shock", "priority":"low",
         "title":"Купить TLT/IEF на 5% при росте 10Y > 5%",
         "rationale":"Портфель β-Rates = −0.30 (умеренная rate-sensitivity).  При "
                     "hawkish surprise Fed +50bp портфель просядет на −4.2%.  TLT "
                     "(20Y bonds) на 5% сглаживает duration shock.",
         "candidates":[
            {"ticker":"TLT", "name":"iShares 20Y T-Bond ETF", "scenario":"duration hedge, β-Rates +0.85",
             "beta":-0.35, "sector":"Fixed Income"},
            {"ticker":"IEF", "name":"iShares 7-10Y T-Bond",   "scenario":"мягче duration vs TLT",
             "beta":-0.20, "sector":"Fixed Income"},
            {"ticker":"AGG", "name":"iShares Core US Bond",   "scenario":"broad-market bond exposure",
             "beta":-0.15, "sector":"Fixed Income"},
         ],
         "pipeline":[
            ("FACTOR", "β-Rates портфеля = −0.30 (rate-sensitive)"),
            ("REGIME", "late-cycle: Fed surprises возможны"),
            ("STRESS", "под Rates +100bp: −3.8% портфеля, с TLT 5% → −2.6%"),
            ("RAG",    "Bank consensus: 10Y range 4.2-5.0% Q3 2026"),
         ],
         "expected_effect":[
            ("β-Rates",          "−0.30", "−0.18"),
            ("Stress Rates+100bp","−3.8%", "−2.6%"),
            ("Vol (год.)",        "14.8%", "13.9%"),
         ],
         "sources":["Quant Engine", "FRED: T10Y2Y", "Stress: Rates +100bp", "RAG: Bank consensus"],
        },
    ],
}
