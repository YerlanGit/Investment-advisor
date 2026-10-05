"""`§−128` — D-5/Q-3: рядом с индексом риска ФАКТ за 12 мес, а не прогноз.

Решения владельца (05.10): на обложке — текущая годовая доходность (12 мес),
прогнозная доходность из отчёта уходит; в «Эффекте» строки доходности нет.

Сверка — НЕ с формулой движка (тавтология), а с ИСТИНОЙ: книгу с
постоянными весами ведут ДЕНЬГАМИ (капитал ×(1 + Σ wᵢ·Rᵢ) каждый день, кэш под
0%, маржа под rf) и берут её рост за последние 252 торговых дня.

H-3: таблица периодов, TE и IR считались по ДРУГОМУ ряду — перенормированному
на 100% инвестированной книги (кэш не разбавлял, плечо не увеличивало).
Эталон `base` давал 12М +36.4% в таблице против +30.7% по ряду обложки.
Строка 12М таблицы и число обложки обязаны быть одним числом.
"""

from __future__ import annotations

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


def _prices(n_days: int = 600, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    cols = {}
    for i, (name, v) in enumerate(zip(["AAA", "BBB", "CCC"], (0.35, 0.25, 0.5))):
        r = 0.0005 + rng.normal(0.0, v / np.sqrt(252), n_days)
        cols[name] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=idx)


def _engine_frame(px: pd.DataFrame):
    from finance.investment_logic import MAC3RiskEngine
    engine = MAC3RiskEngine()
    frame = pd.DataFrame(index=px.index)
    for i, etf in enumerate(engine.factor_tickers.values()):
        r = np.random.default_rng(700 + i).normal(0.0003, 0.009, len(px))
        frame[etf] = 100.0 * np.exp(np.cumsum(r))
    for c in px.columns:
        frame[f"{c}.US"] = px[c].values
    return engine, frame


def _wealth_12m(px: pd.DataFrame, w: np.ndarray, *, cash_w: float,
                rf_daily: float = 0.0) -> float:
    """ИСТИНА: рост капитала за последние 252 дня, посчитанный деньгами."""
    simple = (px / px.shift(1) - 1.0).dropna().values[-252:]
    wealth = 1.0
    for day in simple:
        wealth = float((wealth * w * (1.0 + day)).sum()
                       + wealth * cash_w * (1.0 + (rf_daily if cash_w < 0 else 0.0)))
    return wealth - 1.0


class Return12mIsTheBooksGrowthTest(unittest.TestCase):

    def test_cash_dilutes(self) -> None:
        px = _prices()
        engine, frame = _engine_frame(px)
        weights = {"AAA.US": 0.4, "BBB.US": 0.25, "CCC.US": 0.15}   # 20% кэша
        _, _, m = engine.calculate_structural_risk(frame, list(weights), weights)
        truth = _wealth_12m(px, np.array([0.4, 0.25, 0.15]), cash_w=0.20)
        self.assertAlmostEqual(m["Return_12M"], truth, places=10)

    def test_margin_is_charged(self) -> None:
        px = _prices(seed=5)
        engine, frame = _engine_frame(px)
        weights = {"AAA.US": 0.6, "BBB.US": 0.4, "CCC.US": 0.2, "USD": -0.2}
        risky = [t for t in weights if t != "USD"]
        _, _, m = engine.calculate_structural_risk(frame, risky, weights)
        truth = _wealth_12m(px, np.array([0.6, 0.4, 0.2]), cash_w=-0.2,
                            rf_daily=engine.current_rfr_daily)
        self.assertAlmostEqual(m["Return_12M"], truth, places=10)

    def test_short_history_prints_nothing(self) -> None:
        """Короче торгового года годовую цифру не печатаем — None, а не 0."""
        px = _prices(n_days=200)
        engine, frame = _engine_frame(px)
        weights = {"AAA.US": 0.5, "BBB.US": 0.5}
        _, _, m = engine.calculate_structural_risk(frame, list(weights), weights)
        self.assertIsNone(m["Return_12M"])


class BookSeriesIsNeverStaleTest(unittest.TestCase):
    """Стадия бенчмарков читает ряд книги с движка — ранний выход риск-модели
    не вправе оставить ей ряд ПРОШЛОГО вызова."""

    def test_early_return_clears_the_series(self) -> None:
        px = _prices()
        engine, frame = _engine_frame(px)
        weights = {"AAA.US": 0.5, "BBB.US": 0.5}
        engine.calculate_structural_risk(frame, list(weights), weights)
        self.assertIsNotNone(engine._last_port_log_returns)
        engine.calculate_structural_risk(frame, ["NOPE.US"], {"NOPE.US": 1.0})
        self.assertIsNone(engine._last_port_log_returns)


class OneSeriesForCoverAndTableTest(unittest.TestCase):
    """H-3: 12М таблицы периодов = число обложки, на ВСЕХ эталонных книгах."""

    def test_golden_books_agree(self) -> None:
        import golden_support as gs
        for sc in gs.SCENARIOS:
            with self.subTest(scenario=sc):
                r = gs.run_analyze_all(sc)
                prt = r["period_returns_table"]
                first = next(iter(prt))
                row = next(p for p in prt[first]["periods"] if p["period"] == "12m")
                self.assertAlmostEqual(row["port_pct"],
                                       r["portfolio_metrics"]["Return_12M"],
                                       places=10)

    def test_te_basis_is_the_cover_return(self) -> None:
        """Годовая доходность в сравнении с рынком — та же, что на обложке."""
        import golden_support as gs
        r = gs.run_analyze_all("base")
        ann = r["portfolio_metrics"]["Annualised_Return"]
        for bm, row in r["benchmark_comparison"].items():
            with self.subTest(benchmark=bm):
                if row.get("Portfolio_Ann_Return") is None:
                    continue
                self.assertAlmostEqual(row["Portfolio_Ann_Return"], ann, places=6)


class CoverShowsFactNotForecastTest(unittest.TestCase):

    def test_payload_carries_12m_not_forward(self) -> None:
        from pdf_payload import build_payload
        pl = build_payload({"portfolio_metrics": {
            "Return_12M": 0.1234, "Expected_Return_Annual": 0.124,
            "Expected_Sharpe": 0.66}}, "deep", {})
        self.assertEqual(pl["return_12m"], "+12.3%")
        self.assertAlmostEqual(pl["return_12m_num"], 12.34)
        for gone in ("expected_return_annual", "expected_sharpe",
                     "expected_return_pct_num", "expected_effect_uses_bl"):
            self.assertNotIn(gone, pl)

    def test_negative_and_missing(self) -> None:
        from pdf_payload import build_payload
        neg = build_payload({"portfolio_metrics": {"Return_12M": -0.034}}, "deep", {})
        self.assertEqual(neg["return_12m"], "-3.4%")
        none = build_payload({"portfolio_metrics": {"Return_12M": None}}, "deep", {})
        self.assertEqual(none["return_12m"], "—")
        self.assertIsNone(none["return_12m_num"])

    def test_design_data_both_tiers(self) -> None:
        from premium_payload import build_design_data
        for tier in ("deep", "base"):
            with self.subTest(tier=tier):
                v = build_design_data({"return_12m": "+12.3%"}, tier)["verdict"]
                self.assertEqual(v["return12m"], "+12.3%")
                self.assertNotIn("expReturn", v)
                self.assertNotIn("expSharpe", v)

    def test_shipped_bundles_render_the_fact(self) -> None:
        """Пинится ПОСТАВЛЯЕМЫЙ артефакт, а не исходник `.jsx` (`CLAUDE.md`)."""
        assets = SRC / "premium_assets"
        for name in ("deep-components.js", "base-components.js"):
            with self.subTest(bundle=name):
                js = (assets / name).read_text(encoding="utf-8")
                self.assertIn("return12m", js)
                self.assertIn("Доходность за 12 мес", js)
                self.assertNotIn("Фвд-Sharpe", js)
                self.assertNotIn("Ожид. дох. (год.)", js)
                self.assertNotIn("Прогноз модели", js)


class EffectHasNoReturnRowTest(unittest.TestCase):
    """Q-3: строка «Ожид. доходность» из «Эффекта» убрана во всех рендерах."""

    def test_payload_drops_the_row(self) -> None:
        from pdf_payload import _build_expected_effect
        ee = _build_expected_effect({"metrics": {
            "sharpe": {"before": 0.5, "after": 0.6, "delta": 0.1, "improved": True},
            "expected_return": {"before": 0.05, "after": 0.04, "delta": -0.01,
                                "delta_pp": -1.0, "improved": False}}})
        self.assertIn("sharpe", ee)
        self.assertNotIn("expected_return", ee)

    def test_premium_cards(self) -> None:
        from premium_payload import build_design_data
        d = build_design_data({"expected_effect": {
            "expected_return": {"before": 0.05, "after": 0.04}}}, "deep")
        self.assertFalse([e for e in d["effect"] if "оходность" in e["name"]])

    def test_jinja_fallback(self) -> None:
        tpl = (SRC / "templates" / "report_deep_v3.html").read_text(encoding="utf-8")
        self.assertNotIn("_ef_card('expected_return'", tpl)
        self.assertNotIn("data.expected_return_annual", tpl)


if __name__ == "__main__":
    unittest.main()
