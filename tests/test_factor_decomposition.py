# -*- coding: utf-8 -*-
"""
Factor-variance decomposition + marginal-overlap tests (additive layer).

Covers finance/factor_decomposition end-to-end:
  1. Математика: тождество wᵀΣw = bᵀFb + wᵀDw, Euler-доли суммируются в 100%,
     диагональный F даёт точные аналитические доли, знак хеджа сохраняется.
  2. Marginal overlap: «факторные двойники» (identical B rows → corr≈1),
     пороги веса, unique-risk (0% для чисто систематического актива,
     100% для чисто идиосинкратического).
  3. Движок: portfolio_metrics["factor_decomposition"] появляется на
     synthetic-прогоне, ранние return-пути НЕ оставляют устаревшую
     декомпозицию с прошлого вызова.
  4. Payload: _build_factor_variance формирует display-rows + twins,
     None когда движок пропустил декомпозицию.
  5. Prompt: _summarise_for_prompt несёт factor_decomposition-блок;
     детерминированный fallback-комментарий строится по 4-шаговой схеме.

Network-free: только синтетика.
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

from finance.factor_decomposition import (   # noqa: E402
    IDIO_LABEL,
    build_factor_decomposition,
    driven_by,
    marginal_overlap,
    variance_decomposition,
)


# ── 1. Variance-decomposition math ───────────────────────────────────────────

class VarianceDecompositionMathTest(unittest.TestCase):
    """Тождества и точные аналитические случаи."""

    @staticmethod
    def _random_psd(k: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(k, k))
        return A @ A.T / k + np.eye(k) * 1e-4

    def test_identity_and_shares_sum_to_100(self) -> None:
        """wᵀΣw = bᵀFb + wᵀDw точно; Σ(factor shares) + idio = 100%."""
        rng = np.random.default_rng(7)
        n, k = 6, 4
        B = rng.normal(0.5, 0.4, size=(n, k))
        F = self._random_psd(k, 3) * 1e-4          # daily-variance scale
        D = np.abs(rng.normal(0, 1e-4, size=n))
        w = np.array([0.25, 0.20, 0.15, 0.15, 0.10, 0.05])
        names = ["Market", "Value", "Rates", "Commodities"]

        out = variance_decomposition(B, F, D, w, names)
        self.assertTrue(out, "decomposition must be defined")

        # Тождество — алгебраически точное, зазор только float-шум.
        self.assertLess(out["identity_gap_pct"], 1e-6)

        # Euler-доли по факторам суммируются в систематическую часть...
        f_sum = sum(r["share_pct"] for r in out["factor_shares"])
        self.assertAlmostEqual(f_sum, out["systematic_pct"], delta=0.15)
        # ...а группы + идио — в 100% (с учётом округления 1 dp на строку).
        g_sum = sum(r["share_pct"] for r in out["group_shares"])
        self.assertAlmostEqual(g_sum, 100.0, delta=0.35)
        self.assertAlmostEqual(out["systematic_pct"] + out["idio_pct"],
                               100.0, delta=0.15)

    def test_diagonal_F_gives_exact_analytic_shares(self) -> None:
        """При диагональном F доля фактора = b_f²·F_ff / σ² точно."""
        B = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        F = np.diag([4e-4, 1e-4])
        D = np.array([0.0, 0.0])
        w = np.array([0.5, 0.5])
        out = variance_decomposition(B, F, D, w, ["Market", "Rates"])

        # b = (0.5, 0.5); c_Market = 0.25·4e-4 = 1e-4; c_Rates = 0.25·1e-4.
        total = 0.25 * 4e-4 + 0.25 * 1e-4
        self.assertAlmostEqual(out["factor_shares"][0]["share_pct"],
                               round(1e-4 / total * 100, 1))
        self.assertAlmostEqual(out["factor_shares"][1]["share_pct"],
                               round(0.25e-4 / total * 100, 1))
        self.assertAlmostEqual(out["idio_pct"], 0.0)

    def test_hedge_factor_keeps_negative_share(self) -> None:
        """Отрицательная Euler-доля (фактор-хедж) сохраняет знак, сумма = 100%."""
        # b = (1, -0.5), сильная положительная ковариация Market↔Rates →
        # вклад Rates отрицателен: c_2 = b_2(Fb)_2 = -0.5·(ρσ² − 0.5σ²) < 0
        # при ρ > 0.5.
        B = np.array([[1.0, 0.0],
                      [0.0, -0.5]])
        s2 = 1e-4
        rho = 0.8
        F = np.array([[s2, rho * s2],
                      [rho * s2, s2]])
        D = np.array([0.0, 0.0])
        w = np.array([1.0, 1.0])
        out = variance_decomposition(B, F, D, w, ["Market", "Rates"])
        rates_share = out["factor_shares"][1]["share_pct"]
        self.assertLess(rates_share, 0.0)
        g_sum = sum(r["share_pct"] for r in out["group_shares"])
        self.assertAlmostEqual(g_sum, 100.0, delta=0.35)

    def test_idio_row_present_and_orphan_factors_grouped(self) -> None:
        """Неизвестная ось попадает в «Прочие факторы», идио-строка всегда есть."""
        B = np.ones((2, 2))
        F = np.eye(2) * 1e-4
        D = np.array([1e-4, 1e-4])
        w = np.array([0.5, 0.5])
        out = variance_decomposition(B, F, D, w, ["Market", "SomethingNew"])
        labels = [g["source"] for g in out["group_shares"]]
        self.assertIn("Прочие факторы", labels)
        self.assertIn(IDIO_LABEL, labels)

    def test_degenerate_inputs_return_empty(self) -> None:
        """Нулевая дисперсия и несогласованные формы → {} (graceful)."""
        # Нулевая дисперсия.
        out = variance_decomposition(np.zeros((2, 2)), np.zeros((2, 2)),
                                     np.zeros(2), np.array([0.5, 0.5]),
                                     ["Market", "Rates"])
        self.assertEqual(out, {})
        # Несогласованные формы.
        out = variance_decomposition(np.ones((2, 3)), np.eye(2), np.ones(2),
                                     np.array([0.5, 0.5]), ["A", "B"])
        self.assertEqual(out, {})


# ── driven_by attribution ────────────────────────────────────────────────────

class DrivenByTest(unittest.TestCase):
    def test_contributions_sum_to_portfolio_beta_and_order(self) -> None:
        B = np.array([[2.0, 0.1],
                      [0.5, 0.1],
                      [0.1, 1.0]])
        w = np.array([0.5, 0.3, 0.2])
        names_f = ["Market", "Rates"]
        names_a = ["NVDA", "KO", "TLT"]
        out = driven_by(B, w, names_f, names_a, top_n=3, min_abs=0.0)

        mkt = out["Market"]
        # Топ по |вкладу|: NVDA (1.0), KO (0.15), TLT (0.02).
        self.assertEqual(mkt[0]["ticker"], "NVDA")
        self.assertAlmostEqual(mkt[0]["contribution"], 1.0, places=3)
        # Сумма вкладов = b_Market = (Bᵀw)_Market.
        self.assertAlmostEqual(sum(r["contribution"] for r in mkt),
                               float((B.T @ w)[0]), places=3)


# ── 2. Marginal overlap (двойники + unique risk) ─────────────────────────────

class MarginalOverlapTest(unittest.TestCase):
    def test_identical_factor_rows_detected_as_twins(self) -> None:
        """Два актива с одинаковой строкой B → систематическая corr ≈ 1."""
        B = np.array([[1.0, 0.5],
                      [1.0, 0.5],
                      [0.0, 1.0]])
        F = np.eye(2) * 1e-4
        D = np.array([1e-5, 1e-5, 1e-5])
        w = np.array([0.4, 0.4, 0.2])
        out = marginal_overlap(B, F, D, w, ["AAPL", "MSFT", "TLT"])
        twins = out["twins"]
        self.assertEqual(len(twins), 1)
        self.assertEqual(set(twins[0]["pair"]), {"AAPL", "MSFT"})
        self.assertGreaterEqual(twins[0]["systematic_corr"], 0.999)
        self.assertAlmostEqual(twins[0]["combined_weight_pct"], 80.0, places=1)

    def test_small_weights_excluded_from_twins(self) -> None:
        """Ноги легче TWIN_MIN_WEIGHT_PCT не образуют пару."""
        B = np.array([[1.0], [1.0]])
        F = np.eye(1) * 1e-4
        D = np.array([0.0, 0.0])
        w = np.array([0.01, 0.5])          # 1% < порога 2%
        out = marginal_overlap(B, F, D, w, ["A", "B"])
        self.assertEqual(out["twins"], [])

    def test_unique_risk_bounds(self) -> None:
        """Чисто систематический актив → 0%; чисто идиосинкратический → 100%."""
        B = np.array([[1.0],
                      [0.0]])
        F = np.eye(1) * 1e-4
        D = np.array([0.0, 5e-4])
        w = np.array([0.5, 0.5])
        out = marginal_overlap(B, F, D, w, ["SPY_CLONE", "BIOTECH"])
        by_t = {u["ticker"]: u["unique_risk_pct"] for u in out["unique_risk"]}
        self.assertAlmostEqual(by_t["SPY_CLONE"], 0.0)
        self.assertAlmostEqual(by_t["BIOTECH"], 100.0)


# ── Orchestrator ─────────────────────────────────────────────────────────────

class BuildFactorDecompositionTest(unittest.TestCase):
    def test_full_bundle_keys(self) -> None:
        rng = np.random.default_rng(11)
        B = rng.normal(0.6, 0.3, size=(4, 3))
        A = rng.normal(size=(3, 3))
        F = (A @ A.T / 3 + np.eye(3) * 1e-4) * 1e-4
        D = np.abs(rng.normal(0, 1e-4, size=4))
        w = np.array([0.4, 0.3, 0.2, 0.1])
        out = build_factor_decomposition(
            B, F, D, w, ["Market", "Value", "Rates"], ["A", "B", "C", "E"])
        for key in ("portfolio_betas", "betas_covered", "factor_shares",
                    "group_shares", "systematic_pct", "idio_pct",
                    "identity_gap_pct", "driven_by", "twins", "unique_risk"):
            self.assertIn(key, out)

    def test_empty_on_degenerate(self) -> None:
        out = build_factor_decomposition(
            np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1),
            np.array([0.0]), ["Market"], ["A"])
        self.assertEqual(out, {})


# ── 3. Engine integration (additive hook) ────────────────────────────────────

class EngineHookTest(unittest.TestCase):
    @staticmethod
    def _make_engine():
        from finance.investment_logic import MAC3RiskEngine
        return MAC3RiskEngine(reporting_currency="USD")

    @classmethod
    def _synthetic_frame(cls, engine, n_days=300, seed=21):
        idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
        df: dict[str, np.ndarray] = {}
        for i, tkr in enumerate(engine.factor_tickers.values()):
            r = np.random.default_rng(seed + i + 1).normal(0.0004, 0.01, n_days)
            df[tkr] = 100.0 * np.exp(np.cumsum(r))
        for j, tkr in enumerate(("AAPL.US", "MSFT.US")):
            r = np.random.default_rng(seed + 100 + j).normal(0.0008, 0.014, n_days)
            df[tkr] = 100.0 * np.exp(np.cumsum(r))
        return pd.DataFrame(df, index=idx)

    def test_metrics_carry_decomposition(self) -> None:
        engine = self._make_engine()
        data = self._synthetic_frame(engine)
        _, _, metrics = engine.calculate_structural_risk(
            data, ["AAPL.US", "MSFT.US"], {"AAPL.US": 0.5, "MSFT.US": 0.3})
        fd = metrics.get("factor_decomposition")
        self.assertTrue(fd, "hook must populate factor_decomposition")
        self.assertLess(fd["identity_gap_pct"], 1e-4)
        g_sum = sum(g["share_pct"] for g in fd["group_shares"])
        self.assertAlmostEqual(g_sum, 100.0, delta=0.6)
        # Разбавление кэшем (Σw=0.8): raw-беты ≠ covered-беты.
        self.assertNotEqual(fd["portfolio_betas"], fd["betas_covered"])

    def test_decomposition_total_reconstructs_engine_volatility(self) -> None:
        """
        Сквозная сверка: полная дисперсия декомпозиции (daily) после
        аннуализации ×252 должна давать ровно Total_Volatility_Ann движка —
        обе величины считаются от одной и той же Σ = B·F·Bᵀ + D.
        """
        engine = self._make_engine()
        data = self._synthetic_frame(engine)
        _, _, metrics = engine.calculate_structural_risk(
            data, ["AAPL.US", "MSFT.US"], {"AAPL.US": 0.6, "MSFT.US": 0.4})
        fd = metrics["factor_decomposition"]
        vol_from_fd = float(np.sqrt(fd["total_variance_daily"] * 252))
        self.assertAlmostEqual(vol_from_fd, metrics["Total_Volatility_Ann"],
                               places=10)

    def test_early_return_resets_stale_decomposition(self) -> None:
        """Ранний return (пустые данные) не должен оставлять прошлый результат."""
        engine = self._make_engine()
        data = self._synthetic_frame(engine)
        engine.calculate_structural_risk(
            data, ["AAPL.US"], {"AAPL.US": 1.0})
        self.assertTrue(engine._last_factor_decomposition)
        # Второй вызов с пустым фреймом идёт по early-return-пути.
        engine.calculate_structural_risk(
            pd.DataFrame(), ["AAPL.US"], {"AAPL.US": 1.0})
        self.assertEqual(engine._last_factor_decomposition, {})


# ── 4. Payload builder ───────────────────────────────────────────────────────

class PayloadBuilderTest(unittest.TestCase):
    @staticmethod
    def _fake_results() -> dict:
        return {"portfolio_metrics": {"factor_decomposition": {
            "group_shares": [
                {"source": "Рыночная бета", "share_pct": 62.0, "factors": ["Market"]},
                {"source": "Стилевые наклоны", "share_pct": 20.0,
                 "factors": ["Momentum", "Value"]},
                {"source": IDIO_LABEL, "share_pct": 18.0, "factors": []},
            ],
            "driven_by": {
                "Market": [{"ticker": "NVDA", "contribution": 0.31},
                           {"ticker": "MSFT", "contribution": 0.18}],
                "Momentum": [{"ticker": "NVDA", "contribution": 0.05}],
            },
            "systematic_pct": 82.0, "idio_pct": 18.0,
            "twins": [{"pair": ["NVDA", "MSFT"], "systematic_corr": 0.93,
                       "combined_weight_pct": 32.4}],
        }}}

    def test_rows_sorted_with_drivers_and_twins(self) -> None:
        from pdf_payload import _build_factor_variance
        panel = _build_factor_variance(self._fake_results())
        self.assertIsNotNone(panel)
        self.assertEqual(panel["rows"][0]["source"], "Рыночная бета")
        self.assertIn("NVDA", panel["rows"][0]["drivers"])
        self.assertEqual(panel["twins"][0]["pair_label"], "NVDA ↔ MSFT")
        self.assertAlmostEqual(panel["idio_pct"], 18.0)

    def test_none_when_engine_skipped(self) -> None:
        from pdf_payload import _build_factor_variance
        self.assertIsNone(_build_factor_variance({"portfolio_metrics": {}}))
        self.assertIsNone(_build_factor_variance({}))

    def test_premium_mapper_carries_factor_variance(self) -> None:
        """premium_payload прокидывает factor_variance → factorVariance."""
        from pdf_payload import _build_factor_variance
        import premium_payload as pp
        panel = _build_factor_variance(self._fake_results())
        payload = {"factor_variance": panel}
        fv = pp._g(payload, "factor_variance", default=None)
        self.assertTrue(isinstance(fv, dict) and fv.get("rows"))


# ── 5. Prompt + fallback commentary ──────────────────────────────────────────

class PromptWiringTest(unittest.TestCase):
    @staticmethod
    def _fake_results() -> dict:
        return {"portfolio_metrics": {"factor_decomposition": {
            "group_shares": [
                {"source": "Рыночная бета", "share_pct": 62.0, "factors": ["Market"]},
                {"source": IDIO_LABEL, "share_pct": 38.0, "factors": []},
            ],
            "betas_covered": {"Market": 1.05, "Value": -0.26},
            "driven_by": {"Market": [{"ticker": "NVDA", "contribution": 0.31}]},
            "idio_pct": 38.0, "systematic_pct": 62.0,
            "twins": [{"pair": ["NVDA", "MSFT"], "systematic_corr": 0.93,
                       "combined_weight_pct": 32.4}],
            "unique_risk": [
                {"ticker": "GLD", "weight_pct": 5.6, "unique_risk_pct": 88.0},
                {"ticker": "NVDA", "weight_pct": 14.8, "unique_risk_pct": 12.0},
                {"ticker": "DUST", "weight_pct": 0.5, "unique_risk_pct": 99.0},
            ],
        }}}

    def test_summary_contains_factor_block(self) -> None:
        from ai_narrative import _summarise_for_prompt
        summary = _summarise_for_prompt(self._fake_results())
        fd = summary["factor_decomposition"]
        self.assertEqual(fd["var_shares"]["Рыночная бета"], 62.0)
        self.assertEqual(fd["betas"]["Value"], -0.26)
        self.assertEqual(fd["twins"][0]["pair"], ["NVDA", "MSFT"])
        # most_unique: сортировка по unique_risk_pct, вес ≥ 2% (DUST отсеян).
        self.assertEqual(fd["most_unique"][0]["ticker"], "GLD")
        self.assertNotIn("DUST", [u["ticker"] for u in fd["most_unique"]])

    def test_summary_block_empty_when_engine_skipped(self) -> None:
        from ai_narrative import _summarise_for_prompt
        summary = _summarise_for_prompt({"portfolio_metrics": {}})
        self.assertEqual(summary["factor_decomposition"], {})

    def test_fallback_comment_follows_recipe(self) -> None:
        from ai_narrative import _fallback_factor_comment
        txt = _fallback_factor_comment(self._fake_results())
        self.assertIn("62.0% дисперсии", txt)
        self.assertIn("Рыночная бета", txt)
        self.assertIn("NVDA+MSFT", txt)          # twins
        self.assertIn("специфика", txt)           # idio > 15%
        self.assertIn("[Quant Engine]", txt)

    def test_fallback_comment_empty_without_decomposition(self) -> None:
        from ai_narrative import _fallback_factor_comment
        self.assertEqual(_fallback_factor_comment({}), "")

    def test_deep_prompt_mentions_factor_decomposition(self) -> None:
        """DEEP-промпт инструктирует модель работать от factor_decomposition."""
        from ai_narrative import _summarise_for_prompt, _user_prompt
        results = self._fake_results()
        results["total_value"] = 10_000.0
        summary = _summarise_for_prompt(results)
        prompt = _user_prompt(summary, tier="deep")
        self.assertIn("factor_decomposition", prompt)
        self.assertIn("ИСТОЧНИК РИСКА", prompt)
        self.assertIn("СКРЫТАЯ КОНЦЕНТРАЦИЯ", prompt)
        # Данные декомпозиции реально дошли до текста промпта.
        self.assertIn("Рыночная бета", prompt)


if __name__ == "__main__":
    unittest.main()
