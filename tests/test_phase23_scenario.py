"""
Phase 23 — Scenario Analysis engine (Панели A/B, docs/ROADMAP_SCENARIO_TIER.md v2).

Hermetic: синтетические цены, ни сети, ни LLM.  Пины: ковариационная vol
(строже gross), RFR-Sharpe, funding decision-tree, MCTR-sizing капы,
look-ahead guard Панели B, юридические дисклеймеры Фазы 3.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finance import scenario_engine as se  # noqa: E402


def _prices(n=1600, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    z = rng.normal(0, 0.01, n)                       # общий фактор
    def path(mu, vol, load):
        eps = rng.normal(0, vol, n)
        return 100 * np.cumprod(1 + mu + load * z + eps)
    return pd.DataFrame({
        "WIN.US":  path(0.0009, 0.003, 0.8),         # сильный актив
        "LOSE.US": path(-0.0004, 0.012, 0.8),        # слабое звено (Sharpe<0, ret<0)
        "DUP.US":  path(0.0009, 0.0005, 0.8),        # почти двойник WIN (corr≈0.93)
        "HEDGE.US": path(0.0003, 0.007, -0.5),       # отрицательная корреляция
        "YOUNG.US": pd.Series(path(0.0008, 0.01, 0.5)).where(
            pd.Series(range(n)) > n - 200).values,   # молодой актив (<252 точек)
    }, index=idx)


class CovVolStricterThanGross(unittest.TestCase):
    def test_cov_vol_below_gross_and_math_selfcheck(self):
        p = _prices()
        w = {"WIN.US": 0.4, "HEDGE.US": 0.6}
        cov_vol = se.portfolio_vol_cov(p, w)
        m = se.five_metrics(p, w)
        self.assertIsNotNone(cov_vol)
        # Диверсификация: истинная σ_p СТРОГО ниже gross Σwσ (hedge отрицательно
        # коррелирован) — ровно то, чем тир строже фреймворка.
        self.assertLess(cov_vol, m["vol_gross_ref"] * 0.97)
        # Self-check замкнутой формулы на 2 активах: w²σ²+…+2w₁w₂ρσ₁σ₂.
        r = p[["WIN.US", "HEDGE.US"]].pct_change(fill_method=None).dropna()
        s1, s2 = r.std(ddof=1) * np.sqrt(252)
        rho = r.corr().iloc[0, 1]
        manual = np.sqrt(0.16 * s1**2 + 0.36 * s2**2 + 2 * 0.4 * 0.6 * rho * s1 * s2)
        self.assertAlmostEqual(cov_vol, manual, places=3)

    def test_young_asset_masked_not_crashing(self):
        p = _prices()
        w = {"WIN.US": 0.5, "YOUNG.US": 0.5}
        v = se.portfolio_vol_cov(p, w)          # YOUNG < min_periods → исключён
        self.assertIsNotNone(v)                 # матрица НЕ упала
        self.assertGreater(v, 0)

    def test_sharpe_is_rfr_adjusted(self):
        p = _prices()
        m = se.five_metrics(p, {"WIN.US": 1.0})
        self.assertAlmostEqual(
            m["sharpe_rfr"], (m["ann_return"] - m["rfr"]) / m["vol_cov"], places=9)


class FundingDecisionTree(unittest.TestCase):
    def test_weak_link_flagged_and_ranked_worst_first(self):
        p = _prices()
        cands = se.funding_candidates(
            p, {"WIN.US": 0.3, "LOSE.US": 0.3, "DUP.US": 0.4})
        tickers = [c["ticker"] for c in cands]
        self.assertIn("LOSE.US", tickers)
        lose = next(c for c in cands if c["ticker"] == "LOSE.US")
        self.assertTrue(any("Sharpe" in f for f in lose["flags"]))
        self.assertTrue(any("< 0" in f for f in lose["flags"]))
        self.assertEqual(tickers[0], "LOSE.US")   # худший — первым

    def test_duplicated_risk_flag(self):
        p = _prices()
        cands = se.funding_candidates(p, {"WIN.US": 0.5, "DUP.US": 0.5})
        dup_flags = [f for c in cands for f in c["flags"] if "дублирующийся" in f]
        self.assertTrue(dup_flags)                # corr(WIN,DUP) ≥ 0.90


class SizingCaps(unittest.TestCase):
    def test_high_vol_capped_at_15pct(self):
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2021-01-01", periods=800)
        p = pd.DataFrame({
            "CALM.US": 100 * np.cumprod(1 + rng.normal(4e-4, 0.004, 800)),
            "WILD.US": 100 * np.cumprod(1 + rng.normal(1e-3, 0.035, 800)),  # ~55% ann
        }, index=idx)
        w, why = se.size_position(p, {"CALM.US": 0.8}, "WILD.US", 0.30)
        self.assertLessEqual(w, se.HIGH_VOL_WEIGHT_CAP + 1e-9)
        self.assertIn("35%", why)

    def test_within_caps_untouched(self):
        p = _prices()
        w, why = se.size_position(p, {"WIN.US": 0.6, "HEDGE.US": 0.3},
                                  "HEDGE.US", 0.10)
        self.assertAlmostEqual(w, 0.10, places=6)


class PanelBLookAhead(unittest.TestCase):
    def test_signal_never_sees_future(self):
        p = _prices(n=900)
        seen: list[pd.Timestamp] = []
        def spy_guard(visible: pd.DataFrame) -> bool:
            seen.append(visible.index.max())
            # сигнал «цена выше своей 200-дневной» — только прошлое
            s = visible["WIN.US"].dropna()
            return bool(s.iloc[-1] > s.tail(200).mean())
        res = se.walk_forward(p, "WIN.US", spy_guard, step_days=63)
        self.assertGreater(res.n_signals, 0)
        # look-ahead guard: максимум видимой даты строго возрастает и каждый
        # вызов не видит последнюю котировку фрейма (горизонты зарезервированы).
        self.assertTrue(all(a <= b for a, b in zip(seen, seen[1:])))
        self.assertTrue(all(t < p.index.max() for t in seen))
        summ = res.summary()
        self.assertIn("63d", summ["horizons"])
        # Фаза 3: дисклеймеры зашиты в результат Панели B.
        self.assertEqual(tuple(summ["disclaimers"]), se.DISCLAIMERS)
        self.assertTrue(any("Survivorship" in d or "выжившего" in d
                            for d in summ["disclaimers"]))
        self.assertTrue(any("ИИР" in d for d in summ["disclaimers"]))


class RegimeSurvival(unittest.TestCase):
    def test_seven_shocks_group_into_three_regimes(self):
        rows = [{"name": "Tech sell-off (как Q2 2022)", "port_pct": -11.7},
                {"name": "Geopolitical risk-off",       "port_pct": -7.7},
                {"name": "Credit blow-out (+200 bps HY)", "port_pct": -4.5},
                {"name": "CPI shock (+1 пп surprise)",  "port_pct": -3.3},
                {"name": "Fed +50 bps surprise",        "port_pct": -2.0},
                {"name": "USD +5% rally",               "port_pct": -0.5},
                {"name": "Fed cut surprise (−50 bps)",  "port_pct": 3.1}]
        out = {r["regime"]: r for r in se.regime_survival(rows)}
        self.assertEqual(set(out), {"risk_on", "rate_shock", "risk_off"})
        self.assertGreater(out["risk_on"]["avg_pct"], 0)      # cut → плюс
        self.assertLess(out["risk_off"]["avg_pct"], 0)
        self.assertEqual(sum(r["n_shocks"] for r in out.values()), 7)

    def test_delta_metrics_none_safe(self):
        d = se.delta_metrics({"sharpe_rfr": 0.9, "beta": None},
                             {"sharpe_rfr": 1.28, "beta": 0.7})
        self.assertAlmostEqual(d["sharpe_rfr"], 0.38, places=9)
        self.assertIsNone(d["beta"])


class DoNoHarmIsolation(unittest.TestCase):
    def test_core_engine_does_not_import_scenario(self):
        """Do-No-Harm: базовый пайплайн не знает о сценарном модуле."""
        for mod in ("finance/investment_logic.py", "finance/scoring_orchestrator.py"):
            src = (SRC / mod).read_text(encoding="utf-8")
            self.assertNotIn("scenario_engine", src, mod)

    def test_scenario_lookback_is_separate_env(self):
        self.assertEqual(se.SCENARIO_LOOKBACK_DAYS, 1825)   # default 5 лет
        # окно ОСНОВНОГО отчёта не трогаем — отдельная env-ручка:
        from finance.investment_logic import MAC3RiskEngine
        import inspect
        self.assertIn("HISTORY_LOOKBACK_DAYS",
                      inspect.getsource(MAC3RiskEngine.get_market_data))


if __name__ == "__main__":
    unittest.main()
