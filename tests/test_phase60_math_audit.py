"""`§−121` — математический аудит: три дефекта, у каждого гейт против ИСТИНЫ.

1. Доходность портфеля: логарифм СУММЫ, а не сумма логарифмов.
2. Маржинальный кредит (отрицательный кэш) не бесплатен — ни в реализованной
   панели, ни в форварде.
3. Стоп Action Plan лежит НИЖЕ зоны входа, а не выше неё.
4. Black-Litterman отдаёт ИЗБЫТОЧНУЮ доходность — панель «Эффект» вычитала rf
   второй раз (RFR mismatch) и печатала ложную «премию ниже нуля».

── 1 ──

Прежняя свёртка реализованной панели — `Σ wᵢ·rᵢ` по ЛОГ-доходностям бумаг.
Это средневзвешенный темп роста бумаг, а не темп роста книги: по неравенству
Йенсена он всегда ниже на «доходность диверсификации» ≈ ½(Σ wᵢσᵢ² − σₚ²) в
год. Ошибка систематическая и всегда пессимистичная: годовая доходность и
числитель Sharpe занижены, просадка и хвост завышены. На демо-книге —
9.47% против 13.36% годовых.

Тесты сверяют НЕ с формулой (тавтология), а с ИСТИНОЙ: книгу с постоянными
весами ведут по деньгам — капитал ×(1 + Σ wᵢ·Rᵢ) каждый день — и сравнивают
её фактический рост с тем, что печатает движок. Прежняя формула этот гейт не
проходит (`OldFormulaFailsTest` — мутация, доказывающая, что гейт не пуст).
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from finance.period_returns import aggregate_log_returns, build_portfolio_log_returns


def _dispersed_prices(n_days: int = 756, seed: int = 11) -> pd.DataFrame:
    """Три бумаги с высокой и разной волатильностью — доходность
    диверсификации здесь крупная, и прежняя формула промахивается заметно."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    vols = np.array([0.45, 0.30, 0.60]) / np.sqrt(252)
    common = rng.normal(0.0, 0.006, n_days)
    cols = {}
    for i, (name, v) in enumerate(zip(["AAA", "BBB", "CCC"], vols)):
        r = 0.0004 + 0.5 * common + rng.normal(0.0, v, n_days)
        cols[name] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=idx)


def _wealth_path_log_returns(prices: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    """ИСТИНА: книга с постоянными весами, ребаланс ежедневно, кэш (1 − Σw)
    лежит под 0%. Считается ДЕНЬГАМИ, без единого логарифма в свёртке."""
    simple = (prices / prices.shift(1) - 1.0).dropna().values
    wealth = [1.0]
    for day in simple:
        invested = wealth[-1] * w                     # доли капитала в бумагах
        cash = wealth[-1] * (1.0 - w.sum())
        wealth.append(float((invested * (1.0 + day)).sum() + cash))
    wealth = np.asarray(wealth)
    return np.log(wealth[1:] / wealth[:-1])


class AggregateMatchesWealthPathTest(unittest.TestCase):

    def test_fully_invested_book(self) -> None:
        px = _dispersed_prices()
        w = np.array([0.5, 0.3, 0.2])
        logs = np.log(px / px.shift(1)).dropna().values
        np.testing.assert_allclose(aggregate_log_returns(logs, w),
                                   _wealth_path_log_returns(px, w),
                                   rtol=0, atol=1e-12)

    def test_cash_dilutes_at_zero_return(self) -> None:
        px = _dispersed_prices()
        w = np.array([0.3, 0.2, 0.1])                 # 40% кэша
        logs = np.log(px / px.shift(1)).dropna().values
        np.testing.assert_allclose(aggregate_log_returns(logs, w),
                                   _wealth_path_log_returns(px, w),
                                   rtol=0, atol=1e-12)

    def test_composite_builder_matches_on_full_panel(self) -> None:
        """`build_portfolio_log_returns` renormalises to the invested sleeve;
        on a full panel that is the wealth path of the renormalised weights."""
        px = _dispersed_prices()
        raw = {"AAA": 0.3, "BBB": 0.2, "CCC": 0.1}
        series, info = build_portfolio_log_returns(px, raw)
        w = np.array([0.5, 1 / 3, 1 / 6])
        np.testing.assert_allclose(series.values,
                                   _wealth_path_log_returns(px, w),
                                   rtol=0, atol=1e-12)
        self.assertAlmostEqual(info["covered_weight"], 0.6)

    def test_catastrophic_leveraged_day_stays_finite(self) -> None:
        """Σw > 1 и обвал сильнее 1/Σw: ln(≤0) не должен отравить ряд."""
        logs = np.array([[np.log(0.4)], [0.01]])      # −60% день, затем +1%
        out = aggregate_log_returns(logs, np.array([2.0]))  # плечо 2×
        self.assertTrue(np.all(np.isfinite(out)))


class EngineHeadlineMatchesWealthPathTest(unittest.TestCase):
    """Сквозной замер: годовая доходность обложки == рост книги в деньгах."""

    def _engine_data(self):
        from finance.investment_logic import MAC3RiskEngine
        engine = MAC3RiskEngine()
        px = _dispersed_prices()
        data = {}
        for i, etf in enumerate(engine.factor_tickers.values()):
            r = np.random.default_rng(300 + i).normal(0.0003, 0.009, len(px))
            data[etf] = 100.0 * np.exp(np.cumsum(r))
        frame = pd.DataFrame(data, index=px.index)
        for c in px.columns:
            frame[f"{c}.US"] = px[c].values
        return engine, frame

    def test_annualised_return_is_the_books_growth(self) -> None:
        engine, frame = self._engine_data()
        weights = {"AAA.US": 0.4, "BBB.US": 0.25, "CCC.US": 0.15}   # 20% кэша
        _, _, metrics = engine.calculate_structural_risk(
            frame, list(weights), weights)
        px = frame[list(weights)]
        truth = _wealth_path_log_returns(px, np.array(list(weights.values())))
        ann_truth = float(np.exp(truth.mean() * engine.trading_days) - 1.0)
        self.assertAlmostEqual(metrics["Annualised_Return"], ann_truth, places=10)
        eq = np.exp(np.cumsum(truth))
        mdd_truth = float((eq / np.maximum.accumulate(eq) - 1.0).min())
        self.assertAlmostEqual(metrics["Max_Drawdown"], mdd_truth, places=10)

    def test_effect_panel_replay_shares_the_cover_basis(self) -> None:
        """Панель «До/после» и обложка обязаны считать ОДНО и то же: иначе
        «до» разъедется с KPI-полосой на доходность диверсификации."""
        from finance.simulate import _realised_expected_return, _sample_metrics
        px = _dispersed_prices()
        w = np.array([0.4, 0.25, 0.15])
        logs = np.log(px / px.shift(1)).dropna().values
        truth = _wealth_path_log_returns(px, w)
        ann_truth = float(np.exp(truth.mean() * 252) - 1.0)
        self.assertAlmostEqual(_realised_expected_return(logs, w), ann_truth,
                               places=10)
        eq = np.exp(np.cumsum(truth))
        mdd_truth = float((eq / np.maximum.accumulate(eq) - 1.0).min())
        self.assertAlmostEqual(_sample_metrics(logs, w, 0.04)["max_drawdown"],
                               mdd_truth, places=10)


class OldFormulaFailsTest(unittest.TestCase):
    """Мутация: прежняя свёртка `Σ wᵢ·rᵢ` против той же истины. Гейт
    обязан её ловить — иначе он проверяет формулу, а не деньги."""

    def test_sum_of_logs_misses_diversification_return(self) -> None:
        px = _dispersed_prices()
        w = np.array([0.5, 0.3, 0.2])
        logs = np.log(px / px.shift(1)).dropna().values
        truth = _wealth_path_log_returns(px, w)
        old = logs @ w
        ann = lambda s: float(np.exp(s.mean() * 252) - 1.0)
        gap_pp = (ann(truth) - ann(old)) * 100
        # Прежняя формула ВСЕГДА ниже (Йенсен), и заметно: > 1 пп в год.
        self.assertTrue(np.all(old <= truth + 1e-15))
        self.assertGreater(gap_pp, 1.0)
        # Разрыв ТЕМПА РОСТА (лог-единицы, до компаундинга) — это и есть
        # ½(Σwσ² − σₚ²): сверка с теорией второго порядка, до 0.1 пп.
        growth_gap_pp = (truth.mean() - old.mean()) * 252 * 100
        cov = np.cov(logs, rowvar=False) * 252
        theory_pp = 0.5 * (w @ np.diag(cov) - w @ cov @ w) * 100
        self.assertLess(abs(growth_gap_pp - theory_pp), 0.1)


# ═══════════════ Маржинальный кредит не бесплатен (`§−121`) ═══════════════

class MarginFinancingTest(unittest.TestCase):

    def test_positive_cash_is_untouched(self) -> None:
        from finance.period_returns import apply_margin_financing
        s = np.array([0.01, -0.02, 0.003])
        np.testing.assert_array_equal(apply_margin_financing(s, 0.15, 1e-4), s)
        np.testing.assert_array_equal(apply_margin_financing(s, -0.0005, 1e-4), s)

    def test_engine_realized_panel_pays_margin_interest(self) -> None:
        """ИСТИНА ведётся деньгами: бумаги на 120% капитала, кредит 20% растёт
        по rf каждый день. Обложка обязана совпасть с этим ростом."""
        from finance.investment_logic import MAC3RiskEngine
        engine = MAC3RiskEngine()
        px = _dispersed_prices()
        frame = pd.DataFrame(index=px.index)
        for i, etf in enumerate(engine.factor_tickers.values()):
            r = np.random.default_rng(500 + i).normal(0.0003, 0.009, len(px))
            frame[etf] = 100.0 * np.exp(np.cumsum(r))
        for c in px.columns:
            frame[f"{c}.US"] = px[c].values
        weights = {"AAA.US": 0.6, "BBB.US": 0.4, "CCC.US": 0.2, "USD": -0.2}
        risky = [t for t in weights if t != "USD"]
        _, _, metrics = engine.calculate_structural_risk(frame, risky, weights)

        rf_d = engine.current_rfr_daily
        simple = (px / px.shift(1) - 1.0).dropna().values
        w = np.array([0.6, 0.4, 0.2])
        wealth = [1.0]
        for day in simple:
            V = wealth[-1]
            wealth.append(float((V * w * (1.0 + day)).sum()
                                + V * (-0.2) * (1.0 + rf_d)))
        wealth = np.asarray(wealth)
        truth = np.log(wealth[1:] / wealth[:-1])
        ann_truth = float(np.exp(truth.mean() * engine.trading_days) - 1.0)
        self.assertAlmostEqual(metrics["Annualised_Return"], ann_truth, places=10)

        # Мутация: тот же прогон без процента по кредиту даёт доходность выше
        # примерно на 0.2·rf — и гейт это различает.
        free = aggregate_log_returns(np.log1p(simple), w)
        ann_free = float(np.exp(free.mean() * engine.trading_days) - 1.0)
        self.assertGreater(ann_free - metrics["Annualised_Return"],
                           0.5 * 0.2 * engine.current_rfr_annual)

    def test_forward_charges_the_loan(self) -> None:
        """Форвард `E[r_port]` сверяется построчно с performance_table:
        Σ w·E[r] по бумагам + кредит × rf. Сценарий golden «leveraged_fx» —
        реальная книга с кэшем −12.4%."""
        import sys
        from pathlib import Path
        tests_dir = str(Path(__file__).resolve().parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        import golden_support as gs

        res = gs.run_analyze_all("leveraged_fx")
        lev = res["leverage_metrics"]
        self.assertTrue(lev["is_leveraged"])
        pm = res["portfolio_metrics"]
        rf = pm["risk_free_rate_annual"]
        perf = res["performance_table"]
        er = perf["Expected_Return"].astype(float)
        w = perf["Weight_Pct"].astype(float) / 100.0
        cash = float(lev["cash_weight"])
        risky_er = float((w * er).sum())            # у строк кэша E[r] = 0
        self.assertAlmostEqual(pm["Expected_Return_Annual"],
                               risky_er + cash * rf, places=6)  # cash округлён до 1e-6



# ═══════════════ Стоп ниже входа (`§−121`) ═══════════════

class BuyStopBelowEntryTest(unittest.TestCase):

    def test_uptrend_stop_sits_below_the_entry_zone(self) -> None:
        """Живой случай из эталона: цена выше SMA50 + ATR (обычный тренд).
        Прежний стоп `price − 2·ATR` = 223.89 лежал ВЫШЕ всей зоны 220.7–222.1."""
        from finance.action_plan import ATR_STOP_MULT_BUY, compute_levels
        lv = compute_levels(action="Buy", price=226.69, atr_abs=1.40,
                            sma50=222.11, sma100=215.0, sma200=205.0,
                            high_52w=240.0, rsi=55.0)
        lo, hi = lv["buy_zone"]
        self.assertLess(hi, 226.69)                          # вход — откат
        self.assertAlmostEqual(lv["stop_loss"], lo - ATR_STOP_MULT_BUY * 1.40)
        self.assertLess(lv["stop_loss"], lo)

    def test_without_zone_the_legacy_market_rule_holds(self) -> None:
        """Нет SMA50 → нет зоны → вход по рынку → прежнее правило бит-в-бит."""
        from finance.action_plan import compute_levels
        lv = compute_levels(action="Buy", price=110.0, atr_abs=2.0,
                            sma50=None, sma100=104.0, sma200=100.0,
                            high_52w=130.0, rsi=50.0)
        self.assertIsNone(lv["buy_zone"])
        self.assertEqual(lv["stop_loss"], max(110.0 - 4.0, 100.0))

    def test_stop_is_monotone_in_the_mandate(self) -> None:
        """Консервативный стоп НЕ шире агрессивного — на любой геометрии."""
        from finance.action_plan import compute_levels
        rng = np.random.default_rng(12101)
        for _ in range(2000):
            price = float(rng.uniform(10, 500))
            kw = dict(action="Buy", price=price,
                      atr_abs=float(price * rng.uniform(0.005, 0.06)),
                      sma50=float(price * rng.uniform(0.85, 1.10)),
                      sma100=None,
                      sma200=float(price * rng.uniform(0.75, 1.20)),
                      high_52w=None, rsi=float(rng.uniform(10, 90)))
            c = compute_levels(mandate_scale=0.75, **kw)["stop_loss"]
            m = compute_levels(mandate_scale=1.00, **kw)["stop_loss"]
            a = compute_levels(mandate_scale=1.25, **kw)["stop_loss"]
            self.assertGreaterEqual(c, m)
            self.assertGreaterEqual(m, a)

    def test_property_grid(self) -> None:
        """Для ЛЮБОЙ покупки с зоной: стоп < нижней границы входа < цели."""
        from finance.action_plan import compute_levels
        rng = np.random.default_rng(121)
        for _ in range(3000):
            price = float(rng.uniform(10, 500))
            atr = float(price * rng.uniform(0.005, 0.06))
            sma50 = float(price * rng.uniform(0.85, 1.10))
            sma200 = float(price * rng.uniform(0.80, 1.20))
            rsi = float(rng.uniform(10, 90))
            for scale in (0.75, 1.0, 1.25):
                lv = compute_levels(action="Buy", price=price, atr_abs=atr,
                                    sma50=sma50, sma100=None, sma200=sma200,
                                    high_52w=None, rsi=rsi,
                                    macd_below_zero=bool(rng.integers(0, 2)),
                                    mandate_scale=scale)
                lo, hi = lv["buy_zone"]
                self.assertLess(lv["stop_loss"], lo)
                self.assertLess(lv["stop_loss"], price)
                self.assertLessEqual(lo, price)          # зона досягаема
                self.assertGreater(lv["take_target"], price)

    def test_protective_stop_is_below_market_for_every_action(self) -> None:
        """Hold/Trim/Sell: стоп лонга выше рынка сработал бы немедленно."""
        from finance.action_plan import compute_levels
        rng = np.random.default_rng(1210)
        for _ in range(3000):
            price = float(rng.uniform(10, 500))
            kw = dict(price=price,
                      atr_abs=float(price * rng.uniform(0.005, 0.06)),
                      sma50=float(price * rng.uniform(0.85, 1.15)),
                      sma100=float(price * rng.uniform(0.80, 1.20)),
                      sma200=float(price * rng.uniform(0.75, 1.25)),
                      high_52w=float(price * rng.uniform(1.0, 1.5)),
                      rsi=float(rng.uniform(10, 90)))
            for action in ("Strong Buy", "Buy", "Hold", "Trim", "Sell"):
                lv = compute_levels(action=action, **kw)
                self.assertLess(lv["stop_loss"], price, (action, kw))

    def test_sma_anchor_above_market_without_atr_gives_no_stop(self) -> None:
        """Без ATR и с SMA выше рынка честного стопа нет — не выдумываем."""
        from finance.action_plan import compute_levels
        lv = compute_levels(action="Hold", price=100.0, atr_abs=None,
                            sma50=None, sma100=105.0, sma200=None,
                            high_52w=None, rsi=None)
        self.assertIsNone(lv["stop_loss"])



# ═══════════════ BL-μ — избыточная доходность (`§−121`) ═══════════════

class BlExcessReturnTest(unittest.TestCase):

    def test_fully_invested_book_adds_rf_once(self) -> None:
        from finance.simulate import _bl_total_return
        self.assertAlmostEqual(
            _bl_total_return(0.03, {"AAA": 0.6, "BBB": 0.4}, 0.045), 0.075)

    def test_free_cash_earns_nothing(self) -> None:
        from finance.simulate import _bl_total_return
        # 80% вложено, 20% свободного кэша под 0%: rf только на вложенное.
        self.assertAlmostEqual(
            _bl_total_return(0.02, {"AAA": 0.8, "USD": 0.2}, 0.05),
            0.02 + 0.8 * 0.05)

    def test_margin_loan_is_charged_rf(self) -> None:
        from finance.simulate import _bl_total_return
        # 120% в бумагах на 20% кредита: rf·1.2 − rf·0.2 = rf·1.0.
        self.assertAlmostEqual(
            _bl_total_return(0.02, {"AAA": 1.2, "USD": -0.2}, 0.05), 0.07)

    def test_none_stays_none(self) -> None:
        from finance.simulate import _bl_total_return
        self.assertIsNone(_bl_total_return(None, {"AAA": 1.0}, 0.05))

    def test_golden_books_no_longer_claim_a_negative_premium(self) -> None:
        """Все три эталонные книги имели ПОЛОЖИТЕЛЬНУЮ премию BL, и все три
        печатали «ожидаемая доходность ниже безрисковой» — rf вычитался дважды."""
        import sys
        from pathlib import Path
        tests_dir = str(Path(__file__).resolve().parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        import golden_support as gs
        for sc in gs.SCENARIOS:
            with self.subTest(scenario=sc):
                ee = gs.run_analyze_all(sc)["expected_effect"]
                rf = 0.045
                self.assertGreater(ee["metrics"]["expected_return"]["before"], rf)
                self.assertTrue(ee["sharpe_delta_meaningful"])
                self.assertEqual(ee["sharpe_note"], "")


if __name__ == "__main__":
    unittest.main()
