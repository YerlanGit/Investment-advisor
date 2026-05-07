"""
RiskProfileManager — pure scoring and mandate logic.
No I/O, no aiogram, no DB. Safe to unit-test in isolation.

Questionnaire: 6 questions (Q1–Q6), each scored 1–3 pts.
Score range: 6 (most conservative) → 18 (most aggressive).

Profiles (CFA / MiFID II-aligned):
  6–8   → Консервативный       (Conservative)
  9–12  → Умеренный            (Moderate)
  13–15 → Умеренно-агрессивный (Moderately Aggressive)
  16–18 → Агрессивный          (Aggressive)
"""

import copy

ASSET_KEYS: list[str] = [
    "Bonds", "Stocks_US", "GlobalETFs", "Commodities", "Crypto", "Stocks_KZ",
]

ASSET_DISPLAY: dict[str, str] = {
    "Bonds":       "Облигации",
    "Stocks_US":   "Акции США",
    "GlobalETFs":  "Global ETFs",
    "Commodities": "Сырьё / Commodities",
    "Crypto":      "Крипто",
    "Stocks_KZ":   "Акции KZ / KASE",
}

# ── Benchmark catalogue ──────────────────────────────────────────────────────
# Users can select any of these as their mandate benchmark.
# Factor ETFs are included so that the data is always pre-fetched
# (avoids a second download pass).
# Key = Tradernet ticker, Value = human-readable display name.
BENCHMARK_LIST: dict[str, str] = {
    # Broad market indices
    "SPY.US":  "S&P 500",
    "QQQ.US":  "Nasdaq 100",
    "URTH.US": "MSCI World",
    "IWM.US":  "Russell 2000",
    # Bond indices
    "AGG.US":  "Bloomberg US Aggregate Bond",
    # Emerging Markets
    "EEM.US":  "MSCI Emerging Markets",
    "EMB.US":  "JP Morgan EM Bond",
    # Factor ETFs (style premia)
    "MTUM.US": "Momentum Factor (MTUM)",
    "VLUE.US": "Value Factor (VLUE)",
    "QUAL.US": "Quality Factor (QUAL)",
    "DBC.US":  "Commodities (DBC)",
    "IEF.US":  "US 7-10Y Treasury (IEF)",
}

# Maps profile name → default Tradernet ETF benchmark.
# Users can override via the benchmark selection step.
PROFILE_BENCH_TICKER: dict[str, str] = {
    "Консервативный":       "AGG.US",
    "Умеренный":            "SPY.US",
    "Умеренно-агрессивный": "SPY.US",
    "Агрессивный":          "QQQ.US",
}

# Keys are (score_lo, score_hi) inclusive ranges.
_PROFILE_MAP: dict[tuple[int, int], dict] = {
    (6, 8): {
        "name":       "Консервативный",
        "target_vol": 0.05,
        "target_te":  0.02,
        "bench":      "20/80",
        "bench_name": "20% MSCI World / 80% Bloomberg Global Aggregate",
        "limits": {
            "Bonds":       [70, 100],
            "Stocks_US":   [0,  20],
            "GlobalETFs":  [0,  15],
            "Commodities": [0,   5],
            "Crypto":      [0,   0],
            "Stocks_KZ":   [0,   5],
        },
    },
    (9, 12): {
        "name":       "Умеренный",
        "target_vol": 0.10,
        "target_te":  0.04,
        "bench":      "60/40",
        "bench_name": "60% MSCI World / 40% Bloomberg Global Aggregate",
        "limits": {
            "Bonds":       [30,  50],
            "Stocks_US":   [20,  40],
            "GlobalETFs":  [10,  30],
            "Commodities": [0,   10],
            "Crypto":      [0,    3],
            "Stocks_KZ":   [0,   10],
        },
    },
    (13, 15): {
        "name":       "Умеренно-агрессивный",
        "target_vol": 0.14,
        "target_te":  0.06,
        "bench":      "80/20",
        "bench_name": "80% MSCI World / 20% Bloomberg Global Aggregate",
        "limits": {
            "Bonds":       [10,  30],
            "Stocks_US":   [30,  60],
            "GlobalETFs":  [10,  40],
            "Commodities": [0,   15],
            "Crypto":      [0,    5],
            "Stocks_KZ":   [0,   15],
        },
    },
    (16, 18): {
        "name":       "Агрессивный",
        "target_vol": 0.20,
        "target_te":  0.08,
        "bench":      "100/0",
        "bench_name": "100% MSCI All Country World Index (ACWI)",
        "limits": {
            "Bonds":       [0,   10],
            "Stocks_US":   [40, 100],
            "GlobalETFs":  [10,  50],
            "Commodities": [0,   15],
            "Crypto":      [0,   10],
            "Stocks_KZ":   [0,   25],
        },
    },
}


class RiskProfileManager:
    """Static-method-only class. All methods are pure functions."""

    @staticmethod
    def score_to_profile(score: int) -> dict:
        """
        Map a raw score (6–18) to a profile dict.

        Returns a deep copy — mutating it does not affect the module constant.
        Raises ValueError for out-of-range scores.
        """
        if not (6 <= score <= 18):
            raise ValueError(f"Score {score} is out of range [6, 18].")
        for (lo, hi), entry in _PROFILE_MAP.items():
            if lo <= score <= hi:
                return copy.deepcopy(entry)
        raise ValueError(f"Score {score} matched no profile band.")  # unreachable

    @staticmethod
    def apply_universe(profile: dict, selected_assets: list[str]) -> dict:
        """
        Return a limits dict where unselected asset classes are forced to [0, 0].
        """
        selected = set(selected_assets)
        limits   = copy.deepcopy(profile["limits"])
        for key in ASSET_KEYS:
            if key not in selected:
                limits[key] = [0, 0]
        return limits

    @staticmethod
    def build_mandate_summary(
        profile: dict,
        limits_dict: dict,
        benchmark_ticker: str | None = None,
    ) -> str:
        """
        Build a Goldman Sachs-style mandate summary for Telegram (Markdown).
        Sounds like a recommendation, not a dry form.
        """
        vol_pct    = int(profile["target_vol"] * 100)
        te_pct     = int(profile["target_te"]  * 100)
        name       = profile["name"]

        # Use user-selected benchmark or profile default
        if benchmark_ticker and benchmark_ticker in BENCHMARK_LIST:
            bench_display = BENCHMARK_LIST[benchmark_ticker]
        else:
            bench_display = profile.get("bench_name", profile.get("bench", "—"))

        lines = [
            "🏛 *RAMP подобрал для вас оптимальную стратегию*\n",
            f"На основе вашего профиля риска RAMP рекомендует стратегию *{name}*.\n",
            "📌 *Мы рекомендуем:*",
            f"  ›  Бенчмарк: *{bench_display}*",
            f"  ›  Целевая волатильность портфеля: *{vol_pct}% годовых*",
            f"  ›  Допустимый риск (Tracking Error): *{te_pct}%* относительно бенчмарка\n",
            "📊 *Лимиты по классам активов:*",
        ]

        for key in ASSET_KEYS:
            if key not in limits_dict:
                continue
            lo, hi  = limits_dict[key]
            display = ASSET_DISPLAY[key]
            if lo == 0 and hi == 0:
                lines.append(f"  •  {display}: ❌ не включено в стратегию")
            else:
                lines.append(f"  •  {display}: {lo}–{hi}%")

        lines.append(
            "\n_Данный мандат является персональной инвестиционной стратегией, "
            "разработанной RAMP на основе ваших ответов. Не является ИИИ._\n"
            "\nУтвердите мандат, чтобы начать работу с RAMP:"
        )
        return "\n".join(lines)
