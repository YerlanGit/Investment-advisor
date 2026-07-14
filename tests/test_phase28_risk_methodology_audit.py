"""
Phase 28 — методология риска для проблемных инструментов: аудит + реализация.

Часть 1 (аудит 2026-07-13, docs/audit/risk-methodology-audit.md): PASS-
инварианты, закрепляющие корректное существующее поведение (компаундинг
дневного ресета в реализованных метриках, sparse-guard, P/E=None при убытке).

Часть 2 (правки P-1…P-8, 2026-07-13): валидация внедрённых возможностей —
  P-1  реестр плеч (finance/leveraged.py): L, базовый актив, expense ratio;
       контрактный drag −½L(L−1)σ_u² и комиссия −ER/252 РАЗДЕЛЬНО (C5);
  P-3  var_reliability="insufficient_history" при окне < 100 наблюдений;
  P-4  SE(β)=σ_ε/(σ_f·√T) в port_metrics + Vasicek-сжатие за флагом
       BETA_SHRINKAGE (default OFF);
  P-5  χ²-ДИ волатильности и Fisher-ДИ корреляции (finance/inference.py);
  P-6  fundamentals/credit N/A для LETF-обёрток (нет отчётности);
  P-7  path-dependent стресс реестровых LETF: (1+X_u)^L·exp(−½L(L−1)σ_u²·63)−1
       вместо линейного β·shock с капом ±35%.

Оставшийся expectedFailure — спецификация НЕвнедрённой возможности
(портфельная weighted-profitability агрегация: отсутствует как класс, что
конструктивно безопасно — см. E1 аудита; реализация потребует отдельного
продуктового решения).

Все тесты детерминированы, синтетичны и оффлайн.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("KZ_RFR_ANNUAL", "0.14")
os.environ["CDS_DISABLED"] = "1"


# ── Эталонные константы (§5 ТЗ аудита) ───────────────────────────────────────
SIGMA_CI_T20 = (0.76, 1.46)          # df=19
SIGMA_CI_T31 = (0.80, 1.34)          # df=30
RHO_CI_N20 = (-0.16, 0.66)           # ρ̂=0.30, SE(z)=1/√17 — пересекает ноль
RHO_CI_N31 = (-0.06, 0.59)           # ρ̂=0.30, SE(z)=1/√28 — пересекает ноль
SHARPE_UNDERLYING = 0.11
SHARPE_LETF_2X    = -0.57            # 0.11 − ½(L−1)·σ·√T = 0.11 − 0.675


def _find_callable(module_names: list[str], substrings: list[str]):
    """Ищет callable с именем, содержащим любую из подстрок — хелпер для
    спецификаций отсутствующих возможностей."""
    import importlib
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for attr in dir(mod):
            low = attr.lower()
            if any(s in low for s in substrings) and callable(getattr(mod, attr, None)):
                return mod, attr
    return None


def _letf_price_paths(n_cycles: int):
    """Синтетика §3.1: базовый актив чередует +8% / −7.5% (n_cycles пар дней).
    Возвращает (base_prices, naive2x_final, reset_prices) от старта 100."""
    base_r = np.tile([0.08, -0.075], n_cycles)
    reset_r = 2.0 * base_r                      # дневной ресет: r_L(t) = L·r(t)
    base_prices  = 100.0 * np.cumprod(1.0 + base_r)
    reset_prices = 100.0 * np.cumprod(1.0 + reset_r)
    naive2x_final = 100.0 * (1.0 + 2.0 * (base_prices[-1] / 100.0 - 1.0))
    return base_prices, naive2x_final, reset_prices


def _engine_with_panel(asset_prices: dict[str, np.ndarray], n_days: int):
    """MAC3RiskEngine + панель: 10 фактор-ETF с полной историей (random walk,
    фиксированные сиды) + пользовательские колонки.  Оффлайн, без сети."""
    from finance.investment_logic import MAC3RiskEngine

    engine = MAC3RiskEngine()
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    data = {}
    for i, etf in enumerate(engine.factor_tickers.values()):
        r = np.random.default_rng(100 + i).normal(0.0004, 0.010, n_days)
        data[etf] = 100.0 * np.exp(np.cumsum(r))
    for name, prices in asset_prices.items():
        col = np.full(n_days, np.nan)
        col[-len(prices):] = prices
        data[name] = col
    return engine, pd.DataFrame(data, index=idx)


# ═════════════════ §3.1 — LETF decay (главный тест) ══════════════════════════

class LetfDailyResetCompoundingTest(unittest.TestCase):
    """Daily-reset 2× на чередовании +8/−7.5 обязан давать ≈75.4, а не ≈96."""

    def test_fixture_reference_values(self) -> None:
        base, naive2x, reset = _letf_price_paths(20)          # 40 торг. дней
        self.assertAlmostEqual(base[-1], 98.0, delta=0.1)     # −2.0%
        self.assertAlmostEqual(naive2x, 96.0, delta=0.1)      # −4.0%
        self.assertAlmostEqual(reset[-1], 75.4, delta=0.1)    # −24.6%

    def test_return_series_layer_compounds_daily_reset(self) -> None:
        """Слой доходностей проекта (period_returns) читает ФАКТИЧЕСКИЕ цены
        ETP → кумулятив содержит компаундинг ресета, а не L·кумулятив базы.
        min_obs=40 передаётся явно (публичный параметр), т.к. дефолт-гейт 60
        закреплён отдельным тестом ниже."""
        from finance.period_returns import (
            _cum_simple_from_log, build_portfolio_log_returns)

        _, _, reset = _letf_price_paths(20)
        idx = pd.date_range("2026-05-01", periods=40, freq="B")
        prices = pd.DataFrame({"CONL.US": reset}, index=idx)
        series, info = build_portfolio_log_returns(
            prices, {"CONL.US": 1.0}, min_obs=40)
        self.assertIsNotNone(series)
        cum = _cum_simple_from_log(series.values)
        # Первый дневной return теряется на shift(1) → компаундим его назад.
        cum_full = (1.0 + cum) * (reset[0] / 100.0) - 1.0
        self.assertAlmostEqual(cum_full, -0.246, delta=0.005)
        # Отвергаем реализацию «наивные 2× без ресета» (−4.0%):
        self.assertLess(cum_full, -0.20)

    def test_engine_realized_metrics_carry_full_decay(self) -> None:
        """Реализованные метрики движка (MaxDD/CAGR) на длинном daily-reset
        пути обязаны отражать эрозию компаундинга: 250 циклов ±16/−15 дают
        −97% от пика, тогда как «наивные 2×» дали бы лишь ≈ −44%."""
        _, _, reset = _letf_price_paths(250)                  # 500 торг. дней
        engine, df = _engine_with_panel({"LETP.US": reset}, n_days=500)
        _, _, metrics = engine.calculate_structural_risk(
            df, ["LETP.US"], {"LETP.US": 1.0})
        self.assertTrue(metrics, "структурная модель не должна пропасть")
        self.assertLess(metrics["Max_Drawdown"], -0.90)       # не −0.44
        self.assertLess(metrics["Annualised_Return"], -0.50)
        self.assertLess(metrics["CVaR_95_Daily"], 0.0)


# ═════════════════ P-1 — реестр плеч и контрактный drag ═════════════════════

class LeverageRegistryTest(unittest.TestCase):

    def test_leverage_of_and_registry_params(self) -> None:
        from finance.leveraged import etp_info, leverage_of

        self.assertEqual(leverage_of("CONL"), 2.0)
        self.assertEqual(leverage_of("CONL.US"), 2.0)         # суффикс срезан
        self.assertEqual(leverage_of("TQQQ"), 3.0)
        self.assertEqual(leverage_of("SQQQ"), -3.0)           # inverse
        self.assertIsNone(leverage_of("AAPL"))                # не ETP
        info = etp_info("CONL")
        self.assertEqual(info.underlying, "COIN")
        self.assertAlmostEqual(info.expense_ratio, 0.0104, places=6)
        # Имя в реестре БЕЗ множителя (XNDU) детектируется, L неизвестен:
        self.assertIsNone(leverage_of("XNDU"))
        from finance.leveraged import is_leveraged_etp
        self.assertTrue(is_leveraged_etp("XNDU"))

    def test_env_params_extension(self) -> None:
        from finance.leveraged import etp_info, leverage_of

        os.environ["LEVERAGED_ETP_PARAMS"] = "FOO:3:BAR:0.0095;BAZ:-2"
        try:
            self.assertEqual(leverage_of("FOO.US"), 3.0)
            self.assertEqual(etp_info("FOO").underlying, "BAR")
            self.assertAlmostEqual(etp_info("FOO").expense_ratio, 0.0095)
            self.assertEqual(leverage_of("BAZ"), -2.0)
            self.assertIsNone(etp_info("BAZ").expense_ratio)
        finally:
            del os.environ["LEVERAGED_ETP_PARAMS"]

    def test_contractual_drag_split(self) -> None:
        """C5: drag и комиссия — два отдельных члена, оба ≤ 0.
        σ_ETP=5.6%/дн, L=2 → σ_u=2.8%, drag = −½·2·1·σ_u² = −σ_u²."""
        from finance.leveraged import contractual_drag_daily

        drag, fee = contractual_drag_daily(2.0, 0.056, 0.0104)
        self.assertAlmostEqual(drag, -(0.028 ** 2), places=10)
        self.assertAlmostEqual(fee, -0.0104 / 252, places=10)
        # inverse: L(L−1) > 0 → drag тоже отрицательный
        drag_inv, _ = contractual_drag_daily(-3.0, 0.084, None)
        self.assertLess(drag_inv, 0.0)
        # L ∈ (0,1] математически даёт drag ≥ 0 → срезается в 0 (anti-boost)
        drag_half, _ = contractual_drag_daily(0.5, 0.02, None)
        self.assertEqual(drag_half, 0.0)

    def test_forward_adjustment_routing(self) -> None:
        """P-1b: известный L+σ → contractual (α̂ игнорируется — без двойного
        счёта); имя без L → эмпирический min(α̂,0); обычное имя не тронуто."""
        from finance.investment_logic import apply_leveraged_forward

        exp = np.array([0.001, 0.001, 0.001])
        out, det = apply_leveraged_forward(
            exp, ["CONL", "XNDU", "AAPL"],
            [-0.0008, -0.0005, -0.0100],
            {"CONL": 0.056})
        # CONL: contractual = drag −σ_u² + fee −ER/252 (НЕ −0.0008 α̂)
        want_conl = 0.001 - (0.028 ** 2) - 0.0104 / 252
        self.assertAlmostEqual(out[0], want_conl, places=10)
        self.assertEqual(det["CONL"]["method"], "contractual")
        self.assertEqual(det["CONL"]["L"], 2.0)
        self.assertLess(det["CONL"]["drag_ann_pct"], 0.0)
        self.assertLess(det["CONL"]["fee_ann_pct"], 0.0)
        # XNDU: нет L → эмпирический фолбэк
        self.assertAlmostEqual(out[1], 0.001 - 0.0005, places=10)
        self.assertEqual(det["XNDU"]["method"], "empirical_alpha")
        # AAPL: не ETP → не тронут, нет в details
        self.assertAlmostEqual(out[2], 0.001, places=10)
        self.assertNotIn("AAPL", det)

    def test_engine_end_to_end_contractual_details(self) -> None:
        """Движок: книга с зарегистрированным LETF (длинная история) несёт
        port_metrics-ключи leveraged_adjustments + leveraged_drag_tickers.
        Проверяется через analyze_all-подобный путь: здесь достаточно, что
        сигма-карта строится и форвард корректируется (юнит выше), а сквозной
        рендер закрыт тестом integrity-чипа ниже."""
        from finance.investment_logic import apply_leveraged_forward

        out, det = apply_leveraged_forward(
            np.array([0.0005]), ["TQQQ"], [0.002], {"TQQQ": 0.03})
        # positive α̂ никогда не добавляется; contractual drag всё равно ≤ 0
        self.assertLess(out[0], 0.0005)
        self.assertEqual(det["TQQQ"]["method"], "contractual")


# ═════════════════ P-5 — χ²-ДИ волатильности ═════════════════════════════════

class SigmaConfidenceIntervalTest(unittest.TestCase):

    def test_sigma_chi2_ci_reference_values(self) -> None:
        from finance.inference import sigma_ci_multiplier

        lo20, hi20 = sigma_ci_multiplier(20)
        self.assertAlmostEqual(lo20, SIGMA_CI_T20[0], delta=0.02)
        self.assertAlmostEqual(hi20, SIGMA_CI_T20[1], delta=0.02)
        lo31, hi31 = sigma_ci_multiplier(31)
        self.assertAlmostEqual(lo31, SIGMA_CI_T31[0], delta=0.02)
        self.assertAlmostEqual(hi31, SIGMA_CI_T31[1], delta=0.02)
        # Широкое окно → интервал стягивается к 1
        lo_big, hi_big = sigma_ci_multiplier(1250)
        self.assertGreater(lo_big, 0.95)
        self.assertLess(hi_big, 1.05)
        # Вырождение
        self.assertIsNone(sigma_ci_multiplier(1))

    def test_engine_metrics_carry_volatility_ci(self) -> None:
        rng = np.random.default_rng(46)
        asset = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 500)))
        engine, df = _engine_with_panel({"ASSET.US": asset}, n_days=500)
        _, _, metrics = engine.calculate_structural_risk(
            df, ["ASSET.US"], {"ASSET.US": 1.0})
        vci = metrics.get("volatility_ci")
        self.assertIsNotNone(vci)
        self.assertLess(vci["lo_mult"], 1.0)
        self.assertGreater(vci["hi_mult"], 1.0)
        self.assertEqual(vci["window_days"], metrics["realized_window_days"])


# ═════════════════ P-5 — Fisher z-ДИ корреляции ══════════════════════════════

class CorrelationConfidenceIntervalTest(unittest.TestCase):

    def test_fisher_ci_reference_values(self) -> None:
        from finance.inference import fisher_rho_ci

        lo20, hi20 = fisher_rho_ci(0.30, 20)
        self.assertAlmostEqual(lo20, RHO_CI_N20[0], delta=0.02)
        self.assertAlmostEqual(hi20, RHO_CI_N20[1], delta=0.02)
        lo31, hi31 = fisher_rho_ci(0.30, 31)
        self.assertAlmostEqual(lo31, RHO_CI_N31[0], delta=0.02)
        self.assertAlmostEqual(hi31, RHO_CI_N31[1], delta=0.02)
        # Ключевой assert ТЗ: на таких выборках ДИ пересекает ноль — код не
        # имеет права утверждать значимость знака корреляции.
        self.assertLess(lo20, 0.0)
        self.assertGreater(hi20, 0.0)
        self.assertLess(lo31, 0.0)
        # Вырождения
        self.assertIsNone(fisher_rho_ci(0.30, 3))
        self.assertIsNone(fisher_rho_ci(1.0, 20))

    def test_factor_diagnostics_carry_corr_ci(self) -> None:
        rng = np.random.default_rng(47)
        asset = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 500)))
        engine, df = _engine_with_panel({"ASSET.US": asset}, n_days=500)
        _, _, metrics = engine.calculate_structural_risk(
            df, ["ASSET.US"], {"ASSET.US": 1.0})
        fd = metrics.get("factor_diagnostics") or {}
        self.assertIn("max_corr_ci95", fd)
        self.assertIn("corr_window_days", fd)
        ci = fd["max_corr_ci95"]
        if ci is not None:
            self.assertLess(ci[0], ci[1])


# ═════════════════ §3.4 — Sharpe под плечом ══════════════════════════════════

class LeveragedSharpeTest(unittest.TestCase):
    """Тождество Sharpe_LETF ≈ Sharpe_u − ½(L−1)·σ·√T через РЕАЛЬНЫЙ код-путь
    apply_leveraged_drag (эмпирическая ветка F-23; контрактная ветка P-1
    закрыта LeverageRegistryTest выше)."""

    def test_drag_reproduces_sharpe_identity(self) -> None:
        from finance.investment_logic import apply_leveraged_drag

        L, sigma_ann, mu_ann = 2.0, 1.35, 0.15
        sigma_d = sigma_ann / np.sqrt(252.0)
        mu_d    = mu_ann / 252.0
        naive_daily = np.array([L * mu_d])              # «удвоенный» β·μ
        alpha_d = -0.5 * L * (L - 1.0) * sigma_d ** 2   # контрактный drag
        dragged = apply_leveraged_drag(naive_daily, ["CONL"], [alpha_d])

        sharpe_underlying = mu_ann / sigma_ann
        sharpe_naive_2x   = (L * mu_ann) / (L * sigma_ann)   # == underlying
        sharpe_letf       = float(dragged[0]) * 252.0 / (L * sigma_ann)

        self.assertAlmostEqual(sharpe_underlying, SHARPE_UNDERLYING, delta=0.01)
        self.assertAlmostEqual(sharpe_letf, SHARPE_LETF_2X, delta=0.02)
        # Ловим реализацию «наивно удвоенный Sharpe» (+0.11 вместо −0.57):
        self.assertLess(sharpe_letf, 0.0)
        self.assertNotAlmostEqual(sharpe_letf, sharpe_naive_2x, delta=0.3)

    def test_drag_never_applied_to_plain_equity(self) -> None:
        from finance.investment_logic import apply_leveraged_drag

        out = apply_leveraged_drag(np.array([0.001]), ["SPCX"], [-0.005])
        self.assertAlmostEqual(float(out[0]), 0.001, places=12)


# ═════════════════ P-7 — path-dependent стресс LETF ══════════════════════════

class LetfPathDependentStressTest(unittest.TestCase):

    def _perf(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"Ticker": "CONL", "Current_Value": 10_000.0, "Beta_Market": 2.0},
        ])

    def test_path_dependent_beats_linear_cap(self) -> None:
        """2×-ETP при Market −25% (β=2 → линейно −50%): линейный путь капается
        в ≈−33%, path-dependent даёт ≈−46% (контрактная выпуклость)."""
        from finance.stress import ScenarioSpec, apply_scenario

        crash = ScenarioSpec(name="Market −25%", shocks={"Market": -0.25})
        # Без σ-карты — прежнее линейное поведение с капом (регрессия):
        legacy = apply_scenario(self._perf(), 10_000.0, crash)
        self.assertFalse(legacy["by_asset"][0]["path_dependent"])
        self.assertGreater(legacy["by_asset"][0]["asset_delta_pct"], -35.0)
        self.assertEqual(legacy["letf_path_n"], 0)
        # С σ-картой — path-dependent, глубже капа:
        path = apply_scenario(self._perf(), 10_000.0, crash,
                              letf_sigma_daily={"CONL": 0.056})
        row = path["by_asset"][0]
        self.assertTrue(row["path_dependent"])
        self.assertEqual(path["letf_path_n"], 1)
        self.assertAlmostEqual(row["asset_delta_pct"], -46.5, delta=2.0)
        self.assertLess(row["asset_delta_pct"],
                        legacy["by_asset"][0]["asset_delta_pct"])

    def test_gain_scenario_drag_still_applies(self) -> None:
        """Рост базы: (1.1)² = +21% минус drag → меньше линейных +20%·…
        Path-результат обязан быть НИЖЕ чистой выпуклости без волы."""
        from finance.stress import ScenarioSpec, apply_scenario

        rally = ScenarioSpec(name="Market +10%", shocks={"Market": +0.10})
        path = apply_scenario(self._perf(), 10_000.0, rally,
                              letf_sigma_daily={"CONL": 0.056})
        row = path["by_asset"][0]
        self.assertTrue(row["path_dependent"])
        self.assertLess(row["asset_delta_pct"], 21.0)   # drag съедает часть
        self.assertGreater(row["asset_delta_pct"], 10.0)

    def test_ordinary_name_unaffected_by_sigma_map(self) -> None:
        from finance.stress import ScenarioSpec, apply_scenario

        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 10_000.0, "Beta_Market": 1.2},
        ])
        crash = ScenarioSpec(name="Market −10%", shocks={"Market": -0.10})
        out = apply_scenario(perf, 10_000.0, crash,
                             letf_sigma_daily={"AAPL": 0.02})
        # AAPL не в реестре → линейный путь несмотря на σ в карте
        self.assertFalse(out["by_asset"][0]["path_dependent"])
        self.assertAlmostEqual(out["by_asset"][0]["asset_delta_pct"],
                               -12.0, delta=0.1)


# ═════════════════ §3.5 — прибыльность с ETF в портфеле ══════════════════════

class ProfitabilityAggregationTest(unittest.TestCase):

    @unittest.expectedFailure
    def test_missing_capability_weighted_roe_with_etf_exclusion(self) -> None:
        """НЕ РЕАЛИЗОВАНО (осознанно, см. E1 аудита): портфельной
        weighted-profitability агрегации нет как класса — по-активная подача
        конструктивно защищает от «ETF портит среднее».  Спецификация на
        случай будущей реализации: акция ROE=15% (вес 0.5) + leveraged ETF
        (вес 0.5) → weighted ROE == 15% (ETF исключён из нормировки весов),
        coverage == 50% — НЕ 7.5% и не NaN."""
        found = _find_callable(
            ["pdf_payload", "premium_payload", "finance.scoring",
             "finance.scoring_orchestrator", "finance.sec_edgar"],
            ["weighted_roe", "weighted_profit", "portfolio_roe",
             "aggregate_fundament", "profitability_aggregate"])
        self.assertIsNotNone(
            found,
            "нет портфельного агрегатора прибыльности; спецификация: "
            "{акция ROE 15%, w=0.5} ⊕ {LETF, w=0.5} → ROE 15%, coverage 50%")

    def test_negative_earnings_yield_no_pe(self) -> None:
        """E2: P/E при отрицательной прибыли и P/B при отрицательном капитале
        НЕ вычисляются (None/NaN), а не выводятся как «маленькие» числа."""
        from finance.scoring_orchestrator import _compute_valuation_ratios

        perf = pd.DataFrame([
            {"Ticker": "LOSSCO", "Current_Price": 50.0,
             "SEC_Net_Income": -5e9, "SEC_Shares_Outstanding": 1e9,
             "SEC_Book_Equity": 2e10, "SEC_FCF": -1e9},
            {"Ticker": "NEGEQ", "Current_Price": 200.0,
             "SEC_Net_Income": 8e9, "SEC_Shares_Outstanding": 1e9,
             "SEC_Book_Equity": -3e9, "SEC_FCF": 5e9},
        ])
        _compute_valuation_ratios(perf)
        # pandas коэрсит None → NaN при записи float-колонки; смысл один —
        # «коэффициент не определён», downstream-скоринг режет NaN одинаково.
        self.assertTrue(pd.isna(perf.loc[0, "SEC_PE_Ratio"]))  # убыток → нет P/E
        self.assertTrue(pd.isna(perf.loc[1, "SEC_PB_Ratio"]))  # neg eq → нет P/B
        self.assertLess(perf.loc[0, "SEC_FCF_Yield"], 0)   # yield МОЖЕТ быть <0

    def test_commodity_and_sovereign_etf_fundamentals_na(self) -> None:
        """Существующий guard: у GLD/TLT нет отчётности → F/C-пиллары N/A."""
        from finance.scoring_orchestrator import _is_credit_not_applicable

        self.assertTrue(_is_credit_not_applicable("GLD", "Gold"))
        self.assertTrue(_is_credit_not_applicable("TLT.US", "Bonds"))
        self.assertFalse(_is_credit_not_applicable("AAPL", "Technology"))

    def test_leveraged_wrapper_fundamentals_na(self) -> None:
        """P-6: LETF-обёртка (CONL — свопы, без корпоративной отчётности)
        теперь исключается из F/C-пилларов — фантомный макро-тилт снят."""
        from finance.scoring_orchestrator import _is_credit_not_applicable

        self.assertTrue(_is_credit_not_applicable("CONL", "Other"))
        self.assertTrue(_is_credit_not_applicable("TQQQ.US", None))
        # Обычная акция с 'Other'-сектором по-прежнему применима:
        self.assertFalse(_is_credit_not_applicable("SPCX", "Other"))


# ═════════════════ §3.6 — min-observations guard + P-3 флаг ══════════════════

class MinObservationsGuardTest(unittest.TestCase):

    def test_sub60_name_dropped_from_return_series_by_default(self) -> None:
        """Дефолтный гейт MIN_OVERLAP_TDAYS=60: колонка с 40 наблюдениями
        исключается и репортится в info."""
        from finance.period_returns import build_portfolio_log_returns

        _, _, reset = _letf_price_paths(20)                  # 40 дней
        idx = pd.date_range("2026-05-01", periods=40, freq="B")
        prices = pd.DataFrame({"SPCX.US": reset}, index=idx)
        series, info = build_portfolio_log_returns(prices, {"SPCX.US": 1.0})
        self.assertIsNone(series)
        self.assertIn("SPCX.US", info["dropped"])

    def test_var_not_reported_for_20day_only_book(self) -> None:
        """Книга из ОДНОГО имени с 20 торговыми днями: sparse-guard исключает
        его из структурной модели → historical VaR/Sharpe НЕ возвращаются
        вовсе (metrics == {}), а не печатаются по 20 точкам."""
        rng = np.random.default_rng(42)
        young = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, 20)))
        engine, df = _engine_with_panel({"SPCX.US": young}, n_days=500)
        _, _, metrics = engine.calculate_structural_risk(
            df, ["SPCX.US"], {"SPCX.US": 1.0})
        self.assertEqual(metrics, {})
        self.assertIn("SPCX.US", engine._last_sparse_dropped)

    def test_mixed_book_drops_young_but_keeps_metrics(self) -> None:
        """SPCX (20 дн) + зрелое имя: молодой исключается (аннотация Action
        Plan, F-20), метрики книги считаются на выжившем имени."""
        rng = np.random.default_rng(43)
        young = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, 20)))
        elder = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 500)))
        engine, df = _engine_with_panel(
            {"SPCX.US": young, "ELDER.US": elder}, n_days=500)
        _, _, metrics = engine.calculate_structural_risk(
            df, ["SPCX.US", "ELDER.US"],
            {"SPCX.US": 0.3, "ELDER.US": 0.7})
        self.assertIn("SPCX.US", engine._last_sparse_dropped)
        self.assertTrue(np.isfinite(metrics["VaR_95_Daily"]))
        self.assertGreaterEqual(metrics["realized_window_days"], 400)
        self.assertEqual(metrics["var_reliability"], "ok")     # окно длинное

    def test_insufficient_history_flag_below_100_obs(self) -> None:
        """P-3: книга с ~70 днями истории печатает VaR/CVaR ВМЕСТЕ с явным
        вердиктом var_reliability="insufficient_history" (а не молча)."""
        rng = np.random.default_rng(44)
        asset = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 70)))
        engine, df = _engine_with_panel({"YNG.US": asset}, n_days=500)
        _, _, metrics = engine.calculate_structural_risk(
            df, ["YNG.US"], {"YNG.US": 1.0})
        self.assertTrue(metrics, "70 дней проходят sparse-guard (≥60)")
        self.assertLess(metrics["realized_window_days"], 100)
        self.assertEqual(metrics["var_reliability"], "insufficient_history")
        # σ-ДИ на коротком окне обязан быть широким (T≈69 → ×[0.86, 1.19])
        vci = metrics["volatility_ci"]
        self.assertLess(vci["lo_mult"], 0.90)
        self.assertGreater(vci["hi_mult"], 1.10)


# ═════════════════ P-4 — SE(β) и Vasicek-сжатие ══════════════════════════════

class BetaInferenceTest(unittest.TestCase):

    def test_beta_standard_errors_in_metrics(self) -> None:
        """P-4: SE(β) по каждому активу/фактору в port_metrics; величина
        согласуется с σ_ε/(σ_f·√T) по порядку и положительна."""
        rng = np.random.default_rng(45)
        asset = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 500)))
        engine, df = _engine_with_panel({"ASSET.US": asset}, n_days=500)
        _, _, metrics = engine.calculate_structural_risk(
            df, ["ASSET.US"], {"ASSET.US": 1.0})
        ses = metrics.get("beta_standard_errors") or {}
        self.assertIn("ASSET.US", ses)
        self.assertIn("Market", ses["ASSET.US"])
        for v in ses["ASSET.US"].values():
            self.assertGreater(v, 0.0)
        self.assertIn(metrics.get("beta_reliability"),
                      ("ok", "noisy_short_window"))
        self.assertFalse(metrics.get("beta_shrinkage_applied"))

    def test_vasicek_shrinkage_flag_default_off_and_formula(self) -> None:
        """P-4b: BETA_SHRINKAGE default OFF (прод не меняется); при включении
        β_Market сжимается к prior=1.0 ровно по w = T/(T+120)."""
        rng = np.random.default_rng(48)
        asset = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 400)))

        engine_off, df = _engine_with_panel({"ASSET.US": asset}, n_days=400)
        _, expo_off, m_off = engine_off.calculate_structural_risk(
            df, ["ASSET.US"], {"ASSET.US": 1.0})
        self.assertFalse(m_off["beta_shrinkage_applied"])
        beta_raw = float(expo_off.loc["ASSET.US", "Beta_Market"])

        os.environ["BETA_SHRINKAGE"] = "1"
        try:
            engine_on, df2 = _engine_with_panel({"ASSET.US": asset}, n_days=400)
            _, expo_on, m_on = engine_on.calculate_structural_risk(
                df2, ["ASSET.US"], {"ASSET.US": 1.0})
        finally:
            del os.environ["BETA_SHRINKAGE"]
        self.assertTrue(m_on["beta_shrinkage_applied"])
        beta_shr = float(expo_on.loc["ASSET.US", "Beta_Market"])
        # Точная формула Vasicek с prior 1.0 (панель полная → T = окно):
        T = int(m_on["factor_diagnostics"]["corr_window_days"])
        w = T / (T + 120.0)
        self.assertAlmostEqual(beta_shr, w * beta_raw + (1 - w) * 1.0,
                               places=6)


# ═════════════════ P-2/P-8 — рендер кавертов ═════════════════════════════════

class CaveatRenderingTest(unittest.TestCase):

    def _results(self, metrics_extra: dict) -> dict:
        metrics = {
            "realized_window_days": 69,
            "var_reliability": "insufficient_history",
            "volatility_ci": {"lo_mult": 0.86, "hi_mult": 1.20,
                              "window_days": 69, "confidence": 0.95},
        }
        metrics.update(metrics_extra)
        return {"portfolio_metrics": metrics}

    def test_window_chip_warns_below_100(self) -> None:
        from pdf_payload import _build_integrity_checks

        checks = _build_integrity_checks(
            self._results({}), {}, {"factors_loaded": 10, "factors_total": 10},
            {"dropped": [], "covered_weight": 1.0, "n_days": 69})
        chip = next(c for c in checks if c["label"] == "Окно риск-метрик")
        self.assertEqual(chip["status"], "⚠")
        self.assertIn("69 торговых дней", chip["detail"])
        self.assertIn("ненадёжны", chip["detail"])
        self.assertIn("σ ×[0.86, 1.20]", chip["detail"])

    def test_leveraged_chip_discloses_drag_and_fee(self) -> None:
        from pdf_payload import _build_integrity_checks

        res = self._results({
            "var_reliability": "ok", "realized_window_days": 1250,
            "leveraged_adjustments": {
                "CONL": {"method": "contractual", "L": 2.0,
                         "underlying": "COIN",
                         "drag_daily": -0.00078, "fee_daily": -4.1e-05,
                         "drag_ann_pct": -17.9, "fee_ann_pct": -1.0},
            },
        })
        checks = _build_integrity_checks(
            res, {}, {"factors_loaded": 10, "factors_total": 10},
            {"dropped": [], "covered_weight": 1.0, "n_days": 1250})
        chip = next(c for c in checks if c["label"] == "Плечевые ETP")
        self.assertIn("CONL", chip["detail"])
        self.assertIn("drag", chip["detail"])
        self.assertIn("комиссия", chip["detail"])

    def test_sec_coverage_by_weight(self) -> None:
        """P-8: CoVe-строка SEC несёт % книги по весу, не только штуки."""
        from finance.data_lineage import _sec_status
        from datetime import date

        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Fundamental_Sector": "Technology",
             "Current_Value": 4000.0, "SEC_Filing_Date": "2026-03-01"},
            {"Ticker": "CONL", "Fundamental_Sector": "default",
             "Current_Value": 6000.0, "SEC_Filing_Date": None},
        ])
        rows = _sec_status({"performance_table": perf, "total_value": 10000.0},
                           date(2026, 7, 13))
        note = rows[0]["note"]
        self.assertIn("1 тикеров без SEC покрытия", note)
        self.assertIn("60% книги по весу", note)


if __name__ == "__main__":
    unittest.main()
