"""`§−130` — H-8: Sharpe и Sortino на базисе 12 месяцев.

Решение владельца 05.10: «Sharpe тогда стоит перевести на тот же 12-месячный
базис» (что и «Доходность за 12 мес» на обложке, `§−128`).

Было (F-4): числитель — среднегодовая доходность за ВСЁ окно (до 5 лет),
знаменатель — структурная σ (0.7·EWMA(hl=63) + 0.3·Ledoit-Wolf, вес к последним
~3 мес). Два горизонта в одной дроби: Sharpe нельзя было сверить ни с одним
числом страницы, рядом стояли спарклайн по 60 дням и «Эффект».

Сверка — с ИСТИНОЙ: капитал книги ведётся ДЕНЬГАМИ за последние 252 дня (кэш 0%,
маржа под rf), из его дневных доходностей считаются доходность, σ и нижнее
отклонение, а не формула движка.
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
ROOT = _TESTS.parent
for p in (str(SRC), str(_TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_phase65_return_12m import _engine_frame, _prices   # noqa: E402


def _truth(px: pd.DataFrame, w: np.ndarray, *, cash_w: float, rf_daily: float,
           rf_annual: float) -> dict:
    """Капитал деньгами за 252 дня → R_12M, σ, нижнее отклонение, Sharpe, Sortino."""
    simple = (px / px.shift(1) - 1.0).dropna().values[-252:]
    wealth, logs = 1.0, []
    for day in simple:
        nxt = float((wealth * w * (1.0 + day)).sum()
                    + wealth * cash_w * (1.0 + (rf_daily if cash_w < 0 else 0.0)))
        logs.append(math.log(nxt / wealth))
        wealth = nxt
    logs = np.array(logs)
    r12 = wealth - 1.0
    vol = float(np.std(logs, ddof=1) * math.sqrt(252))
    down = float(np.sqrt(np.mean(np.minimum(logs - rf_daily, 0.0) ** 2)) * math.sqrt(252))
    return {"r12": r12, "vol": vol, "down": down,
            "sharpe": (r12 - rf_annual) / vol, "sortino": (r12 - rf_annual) / down}


class SharpeIsTheLastTwelveMonthsTest(unittest.TestCase):

    def _check(self, px, weights, w, cash_w) -> None:
        engine, frame = _engine_frame(px)
        risky = [t for t in weights if t != "USD"]
        _, _, m = engine.calculate_structural_risk(frame, risky, weights)
        t = _truth(px, w, cash_w=cash_w, rf_daily=engine.current_rfr_daily,
                   rf_annual=engine.current_rfr_annual)
        self.assertAlmostEqual(m["Return_12M"], t["r12"], places=10)
        self.assertAlmostEqual(m["Volatility_12M"], t["vol"], places=10)
        self.assertAlmostEqual(m["Sharpe_Ratio"], t["sharpe"], places=9)
        self.assertAlmostEqual(m["Sortino_Ratio"], t["sortino"], places=9)

    def test_cash_book(self) -> None:
        self._check(_prices(), {"AAA.US": 0.4, "BBB.US": 0.25, "CCC.US": 0.15},
                    np.array([0.4, 0.25, 0.15]), 0.20)

    def test_margin_book(self) -> None:
        self._check(_prices(seed=5),
                    {"AAA.US": 0.6, "BBB.US": 0.4, "CCC.US": 0.2, "USD": -0.2},
                    np.array([0.6, 0.4, 0.2]), -0.2)

    def test_older_history_does_not_move_it(self) -> None:
        """Базис 12 мес: история ДО последних 252 дней Sharpe не меняет.

        Старый базис брал среднюю за всё окно и структурную σ — переписанное
        прошлое двигало обе части дроби.
        """
        px = _prices(n_days=600, seed=8)
        alt = px.copy()
        cut = len(px) - 253                       # последние 253 цены — общие
        rng = np.random.default_rng(99)
        shock = np.exp(np.cumsum(rng.normal(0.0, 0.03, (cut, px.shape[1])), axis=0))
        alt.iloc[:cut] = px.iloc[:cut].values * shock / shock[-1]
        weights = {"AAA.US": 0.5, "BBB.US": 0.3, "CCC.US": 0.2}
        e1, f1 = _engine_frame(px)
        e2, f2 = _engine_frame(alt)
        _, _, m1 = e1.calculate_structural_risk(f1, list(weights), weights)
        _, _, m2 = e2.calculate_structural_risk(f2, list(weights), weights)
        self.assertNotAlmostEqual(m1["Annualised_Return"], m2["Annualised_Return"], places=4)
        self.assertAlmostEqual(m1["Sharpe_Ratio"], m2["Sharpe_Ratio"], places=9)
        self.assertAlmostEqual(m1["Sortino_Ratio"], m2["Sortino_Ratio"], places=9)

    def test_short_history_prints_nothing(self) -> None:
        """Короче торгового года — «—», как у `Return_12M` (не годовая с обрывка)."""
        px = _prices(n_days=200)
        engine, frame = _engine_frame(px)
        weights = {"AAA.US": 0.5, "BBB.US": 0.5}
        _, _, m = engine.calculate_structural_risk(frame, list(weights), weights)
        self.assertIsNone(m["Volatility_12M"])
        self.assertTrue(math.isnan(m["Sharpe_Ratio"]))
        self.assertTrue(math.isnan(m["Sortino_Ratio"]))


class OnePageOneSharpeTest(unittest.TestCase):

    def test_golden_books_are_self_consistent(self) -> None:
        """Sharpe = (12М обложки − rf) / σ 12 мес на всех четырёх эталонах."""
        import golden_support as gs
        for sc in gs.SCENARIOS:
            with self.subTest(scenario=sc):
                m = gs.run_analyze_all(sc)["portfolio_metrics"]
                rf = m["risk_free_rate_annual"]
                self.assertAlmostEqual(
                    m["Sharpe_Ratio"], (m["Return_12M"] - rf) / m["Volatility_12M"],
                    places=10)

    def test_effect_panel_starts_from_the_card(self) -> None:
        """«Эффект»: Sharpe «до» = число карточки (якорь `simulate`)."""
        import golden_support as gs
        r = gs.run_analyze_all("base")
        row = (r.get("expected_effect") or {}).get("metrics", {}).get("sharpe") or {}
        if row.get("before") is None:
            self.skipTest("на эталоне нет строки Sharpe в «Эффекте»")
        self.assertAlmostEqual(row["before"], r["portfolio_metrics"]["Sharpe_Ratio"],
                               places=10)

    def test_basis_note_names_the_window(self) -> None:
        engine, frame = _engine_frame(_prices())
        w = {"AAA.US": 0.5, "BBB.US": 0.5}
        _, _, m = engine.calculate_structural_risk(frame, list(w), w)
        note = m["sharpe_basis_note"]
        self.assertIn("12 мес", note)
        self.assertNotIn("EWMA", note)


class SparklineSharpeIsExcessReturnTest(unittest.TestCase):
    """`§−133`: срез спарклайна — (среднее − rf)/σ·√252, как число карточки.

    Прежде rf не вычиталась: вся линия стояла выше на rf/σ (≈ +0.37 при 4.5% и
    σ 12%) и не сопоставлялась с числом над ней.
    """

    def test_points_subtract_the_daily_rate(self) -> None:
        from finance.portfolio_series import compute_kpi_trend_series
        rng = np.random.default_rng(11)
        lr = pd.Series(rng.normal(0.0004, 0.008, 400))
        rf_d = (1.045) ** (1 / 252) - 1
        got = compute_kpi_trend_series({"port_log_returns": lr,
                                        "portfolio_metrics": {"risk_free_rate_daily": rf_d}})
        tail = lr.iloc[-252:].reset_index(drop=True)
        step = len(tail) // 12
        expect = []
        for end in [step * (i + 1) - 1 for i in range(12)]:
            win = tail.iloc[max(0, end - 60):end + 1]
            if len(win) < 30:
                continue
            expect.append((win.mean() - rf_d) / win.std() * math.sqrt(252))
        self.assertEqual(len(expect), len(got["sharpe_pts"]))
        for a, b in zip(expect, got["sharpe_pts"]):
            self.assertAlmostEqual(a, b, places=12)

    def test_without_a_rate_nothing_is_invented(self) -> None:
        from finance.portfolio_series import compute_kpi_trend_series
        lr = pd.Series(np.random.default_rng(3).normal(0.0004, 0.008, 300))
        a = compute_kpi_trend_series({"port_log_returns": lr})
        b = compute_kpi_trend_series({"port_log_returns": lr,
                                      "portfolio_metrics": {"risk_free_rate_daily": 0.0}})
        self.assertEqual(a["sharpe_pts"], b["sharpe_pts"])


class ReportNamesTheBasisTest(unittest.TestCase):

    def test_integrity_row_prints_the_sigma(self) -> None:
        from pdf_payload import _build_integrity_checks
        checks = _build_integrity_checks(
            results={"portfolio_metrics": {"sharpe_basis_note": "x",
                                           "Volatility_12M": 0.138}},
            ai_summary={}, data_quality={}, return_series_coverage={})
        row = next(c for c in checks if c["label"] == "Базис Sharpe/Sortino")
        self.assertIn("12 мес", row["detail"])
        self.assertIn("13.8%", row["detail"])
        self.assertNotIn("EWMA", row["detail"])

    def test_premium_card_and_jinja(self) -> None:
        from premium_payload import build_design_data
        for tier in ("deep", "base"):
            with self.subTest(tier=tier):
                d = build_design_data({"sharpe": "0.84", "sortino": "1.10"}, tier)
                card = next(k for k in d["kpis"] if k["key"] == "sharpe")
                self.assertIn("12 мес", card["sub"])
        for name in ("report_deep_v3.html", "report_basic_v3.html"):
            with self.subTest(template=name):
                tpl = (SRC / "templates" / name).read_text(encoding="utf-8")
                self.assertIn("за 12 мес · Sortino", tpl)

    def test_shipped_bundles_caption_each_card_by_its_own_window(self) -> None:
        """Подпись «по полному окну истории» стояла под ВСЕМИ спарклайнами —
        неверно для Sharpe (12 мес) и для волатильности (структурная σ)."""
        assets = SRC / "premium_assets"
        for name in ("deep-components.js", "base-components.js"):
            with self.subTest(bundle=name):
                js = (assets / name).read_text(encoding="utf-8")
                self.assertIn("число выше — за последние 12 мес", js)
                self.assertIn("число выше — структурная σ", js)
                self.assertNotIn("числитель — геом. годовая доходность", js)

    def test_prompt_tells_the_model_the_window(self) -> None:
        txt = (ROOT / "SYSTEM_PROMPT.md").read_text(encoding="utf-8")
        self.assertIn("Sharpe_Ratio`, `Sortino_Ratio` — за последние 12 мес", txt)


if __name__ == "__main__":
    unittest.main()
