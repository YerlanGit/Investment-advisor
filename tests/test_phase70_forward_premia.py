"""`§−134` — H-1: форвард бумаги больше не считает рыночную премию дважды.

Ортогонализация (BLOCK 3.5) очищает стилевые факторы от рынка: бета бумаги к
`Momentum` — это бета к РЫНОЧНО-НЕЙТРАЛЬНОМУ моментуму, а рыночная часть
стиля уже сидит в `Beta_Market`. Прогноз же умножал эти беты на СЫРЫЕ средние
фондов-факторов (MTUM вместе с его рыночной частью), то есть премия рынка
входила второй раз: завышение = Σ βᵢ,стиль · β̂(стиль→рынок) · μ_рынка.
Вдобавок очищенный ряд хранил СЫРОЕ среднее (`resid + mean(child)`).

Сверка — с ИСТИНОЙ: ряды собраны так, что у стиля НЕТ собственной премии
(MTUM = 0.8·рынок + шум со средним 0). Истинная ожидаемая доходность бумаги
«рынок + 0.5·чистый моментум» — ровно премия рынка.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_TESTS = Path(__file__).resolve().parent
SRC = _TESTS.parent / "src"
for p in (str(SRC), str(_TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

N = 1500
MU_MKT = 0.0005          # дневной лог-дрейф рынка ≈ 13% годовых


def _world(seed: int = 5):
    """Факторные фонды и бумага: у моментума нет своей премии."""
    from finance.investment_logic import MAC3RiskEngine
    eng = MAC3RiskEngine()
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=N, freq="B")
    mkt = MU_MKT + rng.normal(0.0, 0.010, N)
    rets = {"Market": mkt}
    def _zero_mean(sd: float) -> np.ndarray:
        x = rng.normal(0.0, sd, N)
        return x - x.mean()                             # нулевой дрейф В ВЫБОРКЕ
    for name in eng.factor_tickers:
        if name == "Market":
            continue
        rets[name] = _zero_mean(0.006)
    pure_mom = _zero_mean(0.006)
    rets["Momentum"] = 0.8 * mkt + pure_mom             # сырое среднее = 0.8·μ_mkt
    frame = pd.DataFrame(index=idx)
    for name, etf in eng.factor_tickers.items():
        frame[etf] = 100.0 * np.exp(np.cumsum(rets[name]))
    asset = mkt + 0.5 * pure_mom + _zero_mean(0.004)
    frame["AST.US"] = 100.0 * np.exp(np.cumsum(asset))
    return eng, frame


class HedgedFactorSeriesTest(unittest.TestCase):

    def test_child_keeps_only_its_market_neutral_premium(self) -> None:
        """Очищенный ряд = ребёнок − Σβ̂·родитель; его среднее — интерсепт."""
        from finance.engine.risk_engine import orthogonalize_factors_hierarchical
        rng = np.random.default_rng(1)
        mkt = 0.001 + rng.normal(0, 0.01, 800)
        mom = 0.0002 + 0.8 * mkt + rng.normal(0, 0.005, 800)
        f = pd.DataFrame({"Market": mkt, "Momentum": mom})
        out, betas = orthogonalize_factors_hierarchical(f, return_betas=True)
        b = betas["Momentum"]["Market"]
        np.testing.assert_allclose(out["Momentum"].values, mom - b * mkt, atol=1e-12)
        # Среднее — рыночно-нейтральная премия (≈ 0.0002), а не сырое (≈ 0.001).
        self.assertAlmostEqual(out["Momentum"].mean(), mom.mean() - b * mkt.mean(),
                               places=12)

    def test_betas_do_not_move(self) -> None:
        """Сдвиг среднего регрессора не меняет беты (Ridge с интерсептом)."""
        eng, frame = _world()
        w = {"AST.US": 1.0}
        _, exp, _ = eng.calculate_structural_risk(frame, list(w), w)
        self.assertAlmostEqual(float(exp.loc["AST.US", "Beta_Market"]), 1.0, delta=0.05)
        self.assertAlmostEqual(float(exp.loc["AST.US", "Beta_Momentum"]), 0.5, delta=0.05)


class ForwardCountsTheMarketOnceTest(unittest.TestCase):

    def test_premia_are_hedged(self) -> None:
        from finance.engine.portfolio_manager import factor_premia_daily
        eng, frame = _world()
        w = {"AST.US": 1.0}
        eng.calculate_structural_risk(frame, list(w), w)
        mu = factor_premia_daily(frame, eng.factor_tickers, eng._last_ortho_betas)
        raw = np.log(frame / frame.shift(1)).dropna().mean()
        b = eng._last_ortho_betas["Momentum"]["Market"]
        self.assertAlmostEqual(mu["Market"], raw["SPY.US"], places=12)
        self.assertAlmostEqual(mu["Momentum"], raw["MTUM.US"] - b * raw["SPY.US"],
                               places=12)
        self.assertLess(abs(mu["Momentum"]), 0.3 * abs(raw["MTUM.US"]))

    def test_asset_forward_is_the_market_premium(self) -> None:
        """β·μ бумаги = премия рынка (истина), а не рынок + 0.4·рынок."""
        from finance.engine.portfolio_manager import factor_premia_daily
        eng, frame = _world()
        w = {"AST.US": 1.0}
        _, exp, _ = eng.calculate_structural_risk(frame, list(w), w)
        mu = factor_premia_daily(frame, eng.factor_tickers, eng._last_ortho_betas)
        betas = {c[len("Beta_"):]: float(exp.loc["AST.US", c])
                 for c in exp.columns if c.startswith("Beta_")}
        fwd = sum(betas[k] * mu.get(k, 0.0) for k in betas)
        truth = np.log(frame["SPY.US"] / frame["SPY.US"].shift(1)).dropna().mean()
        self.assertAlmostEqual(fwd, truth, delta=0.03 * truth)
        old = sum(betas[k] * np.log(frame[eng.factor_tickers[k]]
                                    / frame[eng.factor_tickers[k]].shift(1)).dropna().mean()
                  for k in betas)
        self.assertGreater(old, 1.3 * truth)            # прежнее двойное счётное

    def test_without_orthogonalization_raw_means_stay(self) -> None:
        from finance.engine.portfolio_manager import factor_premia_daily
        eng, frame = _world()
        mu = factor_premia_daily(frame, eng.factor_tickers, {})
        raw = np.log(frame / frame.shift(1)).dropna().mean()
        self.assertAlmostEqual(mu["Momentum"], raw["MTUM.US"], places=12)


if __name__ == "__main__":
    unittest.main()
