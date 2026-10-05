"""`§−131` — H-2: ставочные шоки стресс-теста — в ЦЕНОВОМ пространстве IEF.

Фактор `Rates` движка — цена фонда IEF (UST 7–10 лет), а каталог описывал
ставочные сценарии изменением ДОХОДНОСТИ и клал б.п. в шок как есть:
«Fed +50 bps» → `Rates: +0.005`. В итоге при росте ставок облигации РОСЛИ, при
снижении — падали, а масштаб был занижен ~в 7 раз (дюрация IEF ≈ 7). Книга с
TLT в сценарии повышения ставок получала по облигациям прибыль.

Сверка — с ИСТИНОЙ: точная переоценка купонной облигации с дюрацией IEF и
движок, подогнанный на ряде, где «TLT» движется как 2.3 × IEF.
"""

from __future__ import annotations

import math
import re
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


def _bond_price(y: float, *, coupon: float = 0.04, years: float = 8.5,
                freq: int = 2) -> float:
    """Точная цена купонной облигации (номинал 1) при доходности `y`."""
    n = int(round(years * freq))
    c = coupon / freq
    r = y / freq
    return sum(c / (1 + r) ** k for k in range(1, n + 1)) + 1 / (1 + r) ** n


class PriceShockIsTheBondsRepriceTest(unittest.TestCase):

    def test_matches_exact_reprice_of_an_ief_like_bond(self) -> None:
        """Облигация 8.5 лет / купон 4% при 4% — дюрация ≈ 7.1, как у IEF.
        Приближение второго порядка совпадает с точной переоценкой на всех
        сдвигах каталога — значит, и единицы выпуклости переведены верно
        (ошибка в ×100 дала бы промах в пункты на +150 б.п.)."""
        from finance.stress import rates_price_shock
        p0 = _bond_price(0.04)
        for dy_bp in (-50, -30, +50, +150):
            with self.subTest(dy_bp=dy_bp):
                exact = _bond_price(0.04 + dy_bp / 10_000) / p0 - 1.0
                self.assertAlmostEqual(rates_price_shock(dy_bp), exact, delta=0.002)

    def test_sign_and_scale(self) -> None:
        from finance.stress import rates_price_shock
        self.assertLess(rates_price_shock(+50), -0.03)      # ≈ −3.5%, не +0.5%
        self.assertGreater(rates_price_shock(-50), +0.03)
        self.assertAlmostEqual(rates_price_shock(+150), -0.0995, places=3)
        # Выпуклость: рост цены при −Δy больше, чем падение при +Δy.
        self.assertGreater(rates_price_shock(-50), -rates_price_shock(+50))
        self.assertEqual(rates_price_shock(0), 0.0)


class CatalogSpeaksInYieldsTest(unittest.TestCase):

    def test_every_rates_shock_comes_from_a_yield_move(self) -> None:
        """Шок `Rates` — только из `rates_dy_bp`, литерал запрещён."""
        from finance.stress import DEFAULT_SCENARIOS, rates_price_shock
        n = 0
        for sc in DEFAULT_SCENARIOS:
            if "Rates" not in sc.shocks:
                self.assertIsNone(sc.rates_dy_bp, sc.name)
                continue
            n += 1
            with self.subTest(scenario=sc.name):
                self.assertIsNotNone(sc.rates_dy_bp)
                self.assertEqual(sc.shocks["Rates"], rates_price_shock(sc.rates_dy_bp))
                # Рост доходности → цена фонда падает, и наоборот.
                self.assertEqual(math.copysign(1, sc.shocks["Rates"]),
                                 -math.copysign(1, sc.rates_dy_bp))
        self.assertEqual(n, 4)

    def test_hikes_hurt_bonds_cuts_help(self) -> None:
        from finance.stress import DEFAULT_SCENARIOS
        by = {s.name: s for s in DEFAULT_SCENARIOS}
        self.assertLess(by["Fed +50 bps surprise"].shocks["Rates"], 0)
        self.assertLess(by["CPI shock (+1 пп surprise)"].shocks["Rates"], 0)
        self.assertGreater(by["Fed cut surprise (−50 bps)"].shocks["Rates"], 0)
        # Бегство в качество при кредитном шоке — UST растут.
        self.assertGreater(by["Credit blow-out (+200 bps HY)"].shocks["Rates"], 0)

    def test_notes_agree_with_the_shock(self) -> None:
        """Заметка «IEF ±x%» обязана совпадать с шоком — прежде «IEF −2%»
        стояло рядом с шоком +2%, а «IEF −0.5%» — рядом с +0.5%."""
        from finance.stress import DEFAULT_SCENARIOS
        for sc in DEFAULT_SCENARIOS:
            if "Rates" not in sc.shocks:
                continue
            with self.subTest(scenario=sc.name):
                m = re.search(r"IEF ([+\-−]\d+\.\d)%", sc.note)
                self.assertIsNotNone(m, sc.note)
                printed = float(m.group(1).replace("−", "-"))
                self.assertAlmostEqual(printed, round(sc.shocks["Rates"] * 100, 1))


class EngineStressOfABondBookTest(unittest.TestCase):
    """Конец в конец: движок подогнан на ряде, где «TLT» = 2.3 × IEF."""

    @classmethod
    def setUpClass(cls) -> None:
        from finance.investment_logic import MAC3RiskEngine
        from finance.stress import run_stress_scenarios
        n = 500
        idx = pd.date_range("2024-01-02", periods=n, freq="B")
        eng = MAC3RiskEngine()
        frame = pd.DataFrame(index=idx)
        rets = {}
        for i, (name, etf) in enumerate(eng.factor_tickers.items()):
            r = np.random.default_rng(300 + i).normal(0.0002, 0.008, n)
            rets[name] = r
            frame[etf] = 100.0 * np.exp(np.cumsum(r))
        noise = np.random.default_rng(1).normal(0.0, 0.0015, n)
        frame["TLT.US"] = 100.0 * np.exp(np.cumsum(2.3 * rets["Rates"] + noise))
        stock = 1.1 * rets["Market"] + np.random.default_rng(2).normal(0, 0.01, n)
        frame["STK.US"] = 100.0 * np.exp(np.cumsum(stock))
        w = {"TLT.US": 0.5, "STK.US": 0.5}
        _, exp, metrics = eng.calculate_structural_risk(frame, list(w), w)
        perf = exp.copy()
        perf.index.name = "Ticker"
        perf = perf.reset_index()
        perf["Current_Value"] = perf["Ticker"].map(w) * 100_000.0
        cls.beta_rates = float(perf.set_index("Ticker").loc["TLT.US", "Beta_Rates"])
        rows = run_stress_scenarios(perf, 100_000.0, metrics,
                                    ortho_betas=getattr(eng, "_last_ortho_betas", {}))
        cls.tlt = {r["name"]: next(a for a in r["by_asset"] if a["ticker"] == "TLT.US")
                   for r in rows}

    def test_beta_was_recovered(self) -> None:
        self.assertAlmostEqual(self.beta_rates, 2.3, delta=0.1)

    def test_rate_hikes_lose_money_on_bonds(self) -> None:
        from finance.stress import rates_price_shock
        for name, dy in (("Fed +50 bps surprise", +50),
                         ("CPI shock (+1 пп surprise)", +150)):
            with self.subTest(scenario=name):
                got = self.tlt[name]["asset_delta_raw"] / 100.0
                self.assertLess(got, 0.0)
                # ≈ β·ΔP_IEF; рыночная нога у «TLT» ≈ 0.
                self.assertAlmostEqual(got, self.beta_rates * rates_price_shock(dy),
                                       delta=0.01)

    def test_cuts_and_flight_to_quality_make_money(self) -> None:
        self.assertGreater(self.tlt["Fed cut surprise (−50 bps)"]["asset_delta_pct"], 0)
        self.assertGreater(self.tlt["Credit blow-out (+200 bps HY)"]["asset_delta_pct"], 0)


if __name__ == "__main__":
    unittest.main()
