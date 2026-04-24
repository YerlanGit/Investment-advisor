"""
RiskProfileManager — pure scoring and mandate logic.
No I/O, no aiogram, no DB. Safe to unit-test in isolation.
"""

import copy

ASSET_KEYS: list[str] = ["Bonds", "GlobalETFs", "Commodities", "Crypto", "EM_KASE"]

ASSET_DISPLAY: dict[str, str] = {
    "Bonds":       "Облигации",
    "GlobalETFs":  "Global ETFs",
    "Commodities": "Сырьё / Commodities",
    "Crypto":      "Крипто",
    "EM_KASE":     "EM / KASE",
}

# Maps profile name → yfinance ticker used as the TE benchmark proxy.
# Conservative: AGG (Bloomberg US Agg, bond-heavy blended proxy)
# Moderate:     ^GSPC (S&P 500, balanced proxy)
# Aggressive:   ^NDX  (Nasdaq 100, growth proxy)
PROFILE_BENCH_TICKER: dict[str, str] = {
    "Консервативный": "AGG",
    "Умеренный":      "^GSPC",
    "Агрессивный":    "^NDX",
}

# Keys are (score_lo, score_hi) inclusive ranges.
_PROFILE_MAP: dict[tuple[int, int], dict] = {
    (4, 6): {
        "name":       "Консервативный",
        "target_vol": 0.05,
        "target_te":  0.02,
        "bench":      "20/80",
        "bench_name": "20% MSCI World / 80% Bloomberg Global Aggregate",
        "limits": {
            "Bonds":       [70, 100],
            "GlobalETFs":  [0,   30],
            "Commodities": [0,    5],
            "Crypto":      [0,    0],
            "EM_KASE":     [0,    5],
        },
    },
    (7, 9): {
        "name":       "Умеренный",
        "target_vol": 0.10,
        "target_te":  0.04,
        "bench":      "60/40",
        "bench_name": "60% MSCI World / 40% Bloomberg Global Aggregate",
        "limits": {
            "Bonds":       [30,  50],
            "GlobalETFs":  [40,  60],
            "Commodities": [0,   10],
            "Crypto":      [0,    3],
            "EM_KASE":     [0,   10],
        },
    },
    (10, 12): {
        "name":       "Агрессивный",
        "target_vol": 0.18,
        "target_te":  0.08,
        "bench":      "100/0",
        "bench_name": "100% MSCI All Country World Index (ACWI)",
        "limits": {
            "Bonds":       [0,   10],
            "GlobalETFs":  [50, 100],
            "Commodities": [0,   15],
            "Crypto":      [0,   10],
            "EM_KASE":     [0,   25],
        },
    },
}


class RiskProfileManager:
    """Static-method-only class. All methods are pure functions."""

    @staticmethod
    def score_to_profile(score: int) -> dict:
        """
        Map a raw score (4–12) to a profile dict.

        Returns a deep copy — mutating it does not affect the module constant.
        Raises ValueError for out-of-range scores.
        """
        if not (4 <= score <= 12):
            raise ValueError(f"Score {score} is out of range [4, 12].")
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
    def build_mandate_summary(profile: dict, limits_dict: dict) -> str:
        """
        Build a Goldman Sachs-style mandate summary for Telegram (Markdown).
        Sounds like a recommendation, not a dry form.
        """
        vol_pct    = int(profile["target_vol"] * 100)
        te_pct     = int(profile["target_te"]  * 100)
        name       = profile["name"]
        bench_name = profile.get("bench_name", profile.get("bench", "—"))

        lines = [
            "🏛 *RAMP подобрал для вас оптимальную стратегию*\n",
            f"На основе вашего профиля риска RAMP рекомендует стратегию *{name}*.\n",
            "📌 *Мы рекомендуем:*",
            f"  ›  Бенчмарк: *{bench_name}*",
            f"  ›  Целевая волатильность портфеля: *{vol_pct}% годовых*",
            f"  ›  Допустимый риск (Tracking Error): *{te_pct}%* относительно бенчмарка\n",
            "📊 *Лимиты по классам активов:*",
        ]

        for key in ASSET_KEYS:
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
