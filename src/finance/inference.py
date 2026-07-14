"""
Статистический inference для риск-метрик — доверительные интервалы малых окон.

P-5 аудита методологии (docs/audit/risk-methodology-audit.md, A3/D4/M-1/M-2):
точечные σ и ρ, оценённые на коротких окнах (молодые листинги: SPCX ~20
торговых дней), подавались без меры неопределённости — ложная точность.
Модуль даёт две классические конструкции:

  • χ²-ДИ волатильности:  σ ∈ [ s·√((T−1)/χ²_{1−α/2}), s·√((T−1)/χ²_{α/2}) ],
    df = T−1, T — ТОРГОВЫЕ дни.  Эталоны: T=20 → множитель [0.76, 1.46],
    T=31 → [0.80, 1.34].

  • Fisher z-ДИ корреляции:  z = arctanh(ρ̂), SE = 1/√(n−3),
    ДИ = tanh(z ± z_crit·SE).  Эталон: ρ̂=0.30, n=20 → [−0.16, 0.66]
    (пересекает 0 — знак корреляции на таком окне статистически не определён).

Без scipy (его нет в requirements): χ²-квантили — приближение
Wilson–Hilferty (кубическая нормальная аппроксимация, точность <0.5% на
df≥10), обратный нормальный — stdlib `statistics.NormalDist.inv_cdf`.
Модуль dependency-light (stdlib + math) — тот же паттерн, что
`period_returns` / `leveraged`: импортируем без sklearn.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Optional

_NORM = NormalDist()


def chi2_quantile(p: float, df: int) -> Optional[float]:
    """χ²-квантиль порядка p при df степенях свободы (Wilson–Hilferty).

    χ²_p ≈ df·(1 − 2/(9·df) + z_p·√(2/(9·df)))³ — стандартная кубическая
    аппроксимация; на df=19 даёт 32.855 против точных 32.852 (0.01%),
    на хвосте 0.025 — ошибка <0.2%.  None при некорректных входах.
    """
    if not (0.0 < p < 1.0) or df < 1:
        return None
    z = _NORM.inv_cdf(p)
    a = 2.0 / (9.0 * df)
    val = df * (1.0 - a + z * math.sqrt(a)) ** 3
    return max(val, 0.0)


def sigma_ci_multiplier(n_obs: int, confidence: float = 0.95
                        ) -> Optional[tuple[float, float]]:
    """Множители (lo, hi) на точечную выборочную σ для (1−α)-ДИ по χ².

    σ_true ∈ [s·lo, s·hi] с уровнем `confidence`; df = n_obs − 1.
    T=20 → ≈(0.76, 1.46); T=31 → ≈(0.80, 1.34); широкое окно → → (1, 1).
    None при n_obs < 2 (ДИ не определён).
    """
    if n_obs < 2 or not (0.0 < confidence < 1.0):
        return None
    df = n_obs - 1
    alpha = 1.0 - confidence
    chi_hi = chi2_quantile(1.0 - alpha / 2.0, df)   # верхний квантиль → lo-мультипликатор
    chi_lo = chi2_quantile(alpha / 2.0, df)         # нижний → hi-мультипликатор
    if not chi_hi or not chi_lo:
        return None
    return (math.sqrt(df / chi_hi), math.sqrt(df / chi_lo))


def fisher_rho_ci(rho: float, n_obs: int, confidence: float = 0.95
                  ) -> Optional[tuple[float, float]]:
    """(1−α)-ДИ выборочной корреляции ρ̂ через Fisher z-преобразование.

    ρ̂=0.30: n=20 → ≈(−0.16, 0.66), n=31 → ≈(−0.06, 0.59).
    None при n_obs < 4 (SE не определён) или |ρ̂| ≥ 1 (вырождение).
    """
    if n_obs < 4 or not (0.0 < confidence < 1.0):
        return None
    r = float(rho)
    if not math.isfinite(r) or abs(r) >= 1.0:
        return None
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n_obs - 3)
    z_crit = _NORM.inv_cdf(0.5 + confidence / 2.0)
    return (math.tanh(z - z_crit * se), math.tanh(z + z_crit * se))


__all__ = ["chi2_quantile", "sigma_ci_multiplier", "fisher_rho_ci"]
