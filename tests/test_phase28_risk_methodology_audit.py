"""
Phase 28 — верификационный аудит методологии риска/прибыльности для
проблемных инструментов (см. docs/audit/risk-methodology-audit.md):

  • молодые листинги с короткой историей (эталон: SPCX, ~20 торговых дней);
  • leveraged/inverse ETP с ежедневным ресетом (эталон: CONL, 2× COIN).

Тесты двух видов:

  PASS-инварианты — закрепляют СУЩЕСТВУЮЩЕЕ корректное поведение
  (компаундинг дневного ресета в реализованных метриках, sparse-guard,
  min(α̂,0)-drag, P/E=None при убытке и т.д.), чтобы регрессия его не сломала.

  expectedFailure-спецификации — фиксируют ОТСУТСТВУЮЩИЕ возможности
  (χ²-ДИ волатильности, Fisher-ДИ корреляции, SE(β), Blume/Vasicek-сжатие,
  реестр плеч L, weighted-profitability c исключением ETF, флаг
  insufficient_history на VaR).  Когда возможность появится, тест начнёт
  падать как «unexpected success» — сигнал снять декоратор и включить
  валидацию по эталонным константам ниже.

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
# χ²-ДИ множителя на σ:  [√((T−1)/χ²₀.₉₇₅), √((T−1)/χ²₀.₀₂₅)]
SIGMA_CI_T20 = (0.76, 1.46)          # df=19
SIGMA_CI_T31 = (0.80, 1.34)          # df=30
# Fisher z-ДИ корреляции ρ̂=0.30:
RHO_CI_N20 = (-0.16, 0.66)           # SE(z)=1/√17 — пересекает ноль
RHO_CI_N31 = (-0.06, 0.59)           # SE(z)=1/√28 — пересекает ноль
# Sharpe под плечом (L=2, σ=1.35, μ=0.15, T=1, RFR=0):
SHARPE_UNDERLYING = 0.11
SHARPE_LETF_2X    = -0.57            # 0.11 − ½(L−1)·σ·√T = 0.11 − 0.675


def _find_callable(module_names: list[str], substrings: list[str]):
    """Ищет в перечисленных модулях callable, чьё имя содержит любую из
    подстрок (case-insensitive).  Возвращает (module, name) или None —
    хелпер для спецификаций отсутствующих возможностей."""
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

    @unittest.expectedFailure
    def test_missing_capability_leverage_registry_with_multiplier(self) -> None:
        """НЕ РЕАЛИЗОВАНО (roadmap docs/METHODOLOGY_SPARSE_AND_LEVERAGED.md §5
        п.3): реестр плеч (тикер → L, базовый актив) и явный контрактный drag
        −½·L(L−1)·σ_u² вместо эмпирического min(α̂,0).  Сейчас
        _LEVERAGED_ETP_BASES (investment_logic.py:113) хранит только ИМЕНА —
        множитель L из него извлечь нельзя."""
        found = _find_callable(
            ["finance.investment_logic", "finance.asset_taxonomy"],
            ["multiplier", "leverage_of", "etp_leverage", "leverage_factor"])
        self.assertIsNotNone(
            found,
            "нет API «тикер → множитель плеча L»; спецификация: "
            "leverage_of('CONL') == 2.0, drag = −½·L(L−1)·σ_u² в форварде")


# ═════════════════ §3.2 — χ²-ДИ волатильности ════════════════════════════════

class SigmaConfidenceIntervalTest(unittest.TestCase):

    @unittest.expectedFailure
    def test_missing_capability_sigma_chi2_ci(self) -> None:
        """НЕ РЕАЛИЗОВАНО: доверительный интервал точечной σ.  Единственный
        ДИ в системе — bootstrap-ДИ CVaR (investment_logic.py:446).
        Спецификация: σ-мультипликатор [√((T−1)/χ²₀.₉₇₅), √((T−1)/χ²₀.₀₂₅)];
        T=20 → [0.76, 1.46] (SIGMA_CI_T20), T=31 → [0.80, 1.34]
        (SIGMA_CI_T31).  При появлении функции — валидировать по константам."""
        found = _find_callable(
            ["finance.investment_logic", "finance.scoring",
             "finance.period_returns", "finance.scenario_engine"],
            ["sigma_ci", "vol_ci", "volatility_ci", "vol_confidence",
             "sigma_confidence", "chi2"])
        self.assertIsNotNone(
            found,
            f"нет χ²-ДИ волатильности; эталоны: T=20 → {SIGMA_CI_T20}, "
            f"T=31 → {SIGMA_CI_T31}")


# ═════════════════ §3.3 — Fisher z-ДИ корреляции ═════════════════════════════

class CorrelationConfidenceIntervalTest(unittest.TestCase):

    @unittest.expectedFailure
    def test_missing_capability_fisher_z_ci(self) -> None:
        """НЕ РЕАЛИЗОВАНО: ДИ корреляции (Fisher z).  Σ строится на окне
        пересечения (investment_logic.py:803) / pairwise-cov
        (scenario_engine.py:121) без ДИ — на n=20 знак ρ̂=0.30 статистически
        не определён (ДИ [−0.16, 0.66] пересекает 0), и код не имеет права
        утверждать значимость такой корреляции."""
        found = _find_callable(
            ["finance.investment_logic", "finance.scenario_engine",
             "finance.scoring"],
            ["fisher", "corr_ci", "correlation_ci", "rho_ci"])
        self.assertIsNotNone(
            found,
            f"нет Fisher-ДИ корреляции; эталоны: ρ̂=0.30, n=20 → {RHO_CI_N20} "
            f"(пересекает 0), n=31 → {RHO_CI_N31} (пересекает 0)")


# ═════════════════ §3.4 — Sharpe под плечом ══════════════════════════════════

class LeveragedSharpeTest(unittest.TestCase):
    """Тождество Sharpe_LETF ≈ Sharpe_u − ½(L−1)·σ·√T через РЕАЛЬНЫЙ код-путь
    apply_leveraged_drag (F-23): форвард = β·μ + min(α̂,0), где α̂ несёт
    контрактный drag −½L(L−1)σ_d²."""

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


# ═════════════════ §3.5 — прибыльность с ETF в портфеле ══════════════════════

class ProfitabilityAggregationTest(unittest.TestCase):

    @unittest.expectedFailure
    def test_missing_capability_weighted_roe_with_etf_exclusion(self) -> None:
        """НЕ РЕАЛИЗОВАНО: портфельная weighted-profitability агрегация
        отсутствует как класс (фундаментал подаётся только по-активно:
        pdf_payload.py:1137-1153).  Спецификация: акция ROE=15% (вес 0.5) +
        leveraged ETF (вес 0.5) → weighted ROE == 15% (ETF исключён из
        нормировки весов), coverage == 50% — НЕ 7.5% и не NaN."""
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
        НЕ вычисляются (None), а не выводятся как «маленькие» числа."""
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
        """Существующий guard: у GLD/TLT нет отчётности → F/C-пиллары N/A
        (scoring_orchestrator.py:56-64, 571)."""
        from finance.scoring_orchestrator import _is_credit_not_applicable

        self.assertTrue(_is_credit_not_applicable("GLD", "Gold"))
        self.assertTrue(_is_credit_not_applicable("TLT.US", "Bonds"))
        self.assertFalse(_is_credit_not_applicable("AAPL", "Technology"))

    @unittest.expectedFailure
    def test_missing_capability_leveraged_wrapper_fundamentals_na(self) -> None:
        """НЕ РЕАЛИЗОВАНО: leveraged single-stock ETP (CONL — обёртка на
        свопах, без корпоративной отчётности) не входит в guard
        _CREDIT_NA_TICKER_PREFIXES / _CREDIT_NA_SECTORS → fundamentals пиллар
        считается «применимым» и деградирует к макро-тилту (сектор 'Other'),
        что читается как фантомный фундаментальный вердикт."""
        from finance.scoring_orchestrator import _is_credit_not_applicable

        self.assertTrue(
            _is_credit_not_applicable("CONL", "Other"),
            "ETP-обёртка должна получать fundamentals_applicable=False")


# ═════════════════ §3.6 — min-observations guard ═════════════════════════════

class MinObservationsGuardTest(unittest.TestCase):

    def test_sub60_name_dropped_from_return_series_by_default(self) -> None:
        """Дефолтный гейт MIN_OVERLAP_TDAYS=60 (period_returns.py:150):
        колонка с 40 наблюдениями исключается и репортится в info."""
        from finance.period_returns import build_portfolio_log_returns

        _, _, reset = _letf_price_paths(20)                  # 40 дней
        idx = pd.date_range("2026-05-01", periods=40, freq="B")
        prices = pd.DataFrame({"SPCX.US": reset}, index=idx)
        series, info = build_portfolio_log_returns(prices, {"SPCX.US": 1.0})
        self.assertIsNone(series)
        self.assertIn("SPCX.US", info["dropped"])

    def test_var_not_reported_for_20day_only_book(self) -> None:
        """Книга из ОДНОГО имени с 20 торговыми днями: sparse-guard
        (investment_logic.py:786-800) исключает его из структурной модели →
        historical VaR/Sharpe НЕ возвращаются вовсе (metrics == {}), а не
        печатаются по 20 точкам."""
        rng = np.random.default_rng(42)
        young = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, 20)))
        engine, df = _engine_with_panel({"SPCX.US": young}, n_days=500)
        _, _, metrics = engine.calculate_structural_risk(
            df, ["SPCX.US"], {"SPCX.US": 1.0})
        self.assertEqual(metrics, {})
        self.assertIn("SPCX.US", engine._last_sparse_dropped)

    def test_mixed_book_drops_young_but_keeps_metrics(self) -> None:
        """SPCX (20 дн) + зрелое имя: молодой исключается (с фиксацией в
        _last_sparse_dropped → аннотация Action Plan, F-20), метрики книги
        считаются на выжившем имени."""
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

    @unittest.expectedFailure
    def test_missing_capability_insufficient_history_flag(self) -> None:
        """НЕ РЕАЛИЗОВАНО: явный флаг надёжности VaR.  Книга с ~70 днями
        истории проходит все гейты (min_obs=30, bootstrap=60) и печатает
        VaR/CVaR как валидные числа; telemetry realized_window_days есть в
        payload, но булевого вердикта «insufficient_history» (окно < ~100
        наблюдений для 95/99%-квантилей) нет ни в metrics, ни в отчёте."""
        rng = np.random.default_rng(44)
        asset = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 70)))
        engine, df = _engine_with_panel({"YNG.US": asset}, n_days=500)
        _, _, metrics = engine.calculate_structural_risk(
            df, ["YNG.US"], {"YNG.US": 1.0})
        self.assertTrue(metrics, "70 дней проходят sparse-guard (≥60)")
        flag_keys = [k for k in metrics
                     if "insufficient" in str(k).lower()
                     or "reliab" in str(k).lower()]
        self.assertTrue(
            flag_keys,
            "VaR по 69 наблюдениям выводится без флага insufficient_history "
            f"(есть только telemetry realized_window_days="
            f"{metrics.get('realized_window_days')})")


# ═════════════════ Beta: SE и сжатие (спецификации) ══════════════════════════

class BetaInferenceTest(unittest.TestCase):

    @unittest.expectedFailure
    def test_missing_capability_beta_standard_error(self) -> None:
        """НЕ РЕАЛИЗОВАНО: SE(β) = σ_ε/(σ_m·√T).  Ridge-регрессия
        (investment_logic.py:883) не выдаёт стандартных ошибок бет; в
        exposures-отчёте нет колонок Beta_SE_* и нет предупреждения при
        большом SE на коротком окне."""
        rng = np.random.default_rng(45)
        asset = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 500)))
        engine, df = _engine_with_panel({"ASSET.US": asset}, n_days=500)
        _, exposures, _ = engine.calculate_structural_risk(
            df, ["ASSET.US"], {"ASSET.US": 1.0})
        se_cols = [c for c in exposures.columns
                   if "se" == str(c).lower().split("_")[-1]
                   or "std_err" in str(c).lower()
                   or "stderr" in str(c).lower()]
        self.assertTrue(
            se_cols,
            "нет SE(β); спецификация: SE(β)=σ_ε/(σ_m·√T), warning при "
            "SE > |β|/2 на коротких окнах")

    @unittest.expectedFailure
    def test_missing_capability_beta_shrinkage(self) -> None:
        """НЕ РЕАЛИЗОВАНО (roadmap docs/METHODOLOGY_SPARSE_AND_LEVERAGED.md
        §2.2/§5 п.2): Blume (β_adj = ⅔β + ⅓) / Vasicek-сжатие бет к прайору
        для окон 60–250 дней.  Ridge alpha=0.001 сжимает к НУЛЮ (не к 1.0) и
        пренебрежимо мало; Ledoit-Wolf сжимает ковариацию ФАКТОРОВ, не беты
        активов."""
        found = _find_callable(
            ["finance.investment_logic", "finance.scoring"],
            ["blume", "vasicek", "shrink_beta", "beta_shrink",
             "adjust_beta", "beta_adjust"])
        self.assertIsNotNone(
            found,
            "нет сжатия бет; спецификация: β_adj = ⅔·β_raw + ⅓·1.0 (Blume) "
            "или Vasicek w=n/(n+120), для ETP prior = L·β_базы")


if __name__ == "__main__":
    unittest.main()
