# -*- coding: utf-8 -*-
"""
Scenario-tier report + wiring tests.

Проверяет:
  1. finance/scenario_report.build_scenario_payload на синтетическом results —
     доступность, Euler-MCTR тождество (Σ% = 100), нормализацию единиц режима
     (доля→процент), funding-флаги, walk-forward Панели B.
  2. graceful-путь: пустые данные → available=False (шаблон не падает).
  3. html_renderer маршрутизирует tier='scenario' на свой Jinja-шаблон в обход
     Premium, и рендер проходит.
  4. Токен-тариф: base 1 · scenario 1 · deep 2.

Network-free: только синтетика; tg_bot импортируется с dummy-токеном.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finance.scenario_report import build_scenario_payload   # noqa: E402


def _synthetic_results(n=1300, seed=3) -> dict:
    idx = pd.date_range("2021-01-01", periods=n, freq="B")

    def gbm(mu, sd, s):
        r = np.random.default_rng(s).normal(mu, sd, n)
        return 100.0 * np.exp(np.cumsum(r))

    prices = pd.DataFrame({
        "AAPL.US": gbm(0.0006, 0.018, seed + 1),
        "MSFT.US": gbm(0.0005, 0.017, seed + 2),
        "NVDA.US": gbm(0.0011, 0.030, seed + 3),
        "GLD.US":  gbm(0.0002, 0.009, seed + 4),
        "SPY.US":  gbm(0.0004, 0.011, seed + 5),
    }, index=idx)
    perf = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "NVDA", "GLD"],
        "Current_Value": [3000, 4000, 2500, 1000],
        "Beta_Market": [1.1, 1.0, 1.6, 0.0],
        "Euler_Risk_Contribution_Pct": [22.0, 25.0, 45.0, 8.0],
    })
    return {
        "history_result": SimpleNamespace(data=prices),
        "performance_table": perf,
        "total_value": 10500.0,
        "stress_scenarios": [
            {"name": "Fed cut", "port_pct": 0.031},
            {"name": "Tech sell-off", "port_pct": -0.117},
            {"name": "Fed +50", "port_pct": -0.02},
            {"name": "CPI shock", "port_pct": -0.033},
            {"name": "Geopolitical", "port_pct": -0.077},
        ],
    }


class ScenarioPayloadTest(unittest.TestCase):
    def test_available_and_core_shape(self):
        s = build_scenario_payload(_synthetic_results())["scenario"]
        self.assertTrue(s["available"])
        self.assertEqual(s["window_days"], 1825)
        for k in ("metrics", "mctr_rows", "regime_survival", "funding",
                  "backtest", "disclaimers"):
            self.assertIn(k, s)

    def test_mctr_euler_identity_sums_to_100(self):
        s = build_scenario_payload(_synthetic_results())["scenario"]
        total = sum(r["pct_ctr"] for r in s["mctr_rows"])
        self.assertAlmostEqual(total, 100.0, places=4)
        # DISPLAY-тикеры проставлены, не resolved.
        self.assertTrue(all(not r["display"].endswith(".US") for r in s["mctr_rows"]))

    def test_regime_survival_units_are_percent(self):
        """Движок кладёт port_pct долей; builder нормализует к % ДО
        regime_survival (порог −10.0 в процентах)."""
        s = build_scenario_payload(_synthetic_results())["scenario"]
        by = {r["regime"]: r for r in s["regime_survival"]}
        self.assertEqual(set(by), {"risk_on", "rate_shock", "risk_off"})
        # risk_on = среднее по {Fed cut +3.1%} → +3.1, НЕ +0.031
        self.assertGreater(by["risk_on"]["avg_pct"], 1.0)
        self.assertLess(by["risk_off"]["avg_pct"], 0.0)
        # величины в разумном %-диапазоне (не доли)
        self.assertLess(abs(by["risk_off"]["avg_pct"]), 100.0)
        self.assertGreater(abs(by["risk_off"]["avg_pct"]), 1.0)

    def test_funding_flags_present(self):
        s = build_scenario_payload(_synthetic_results())["scenario"]
        # хотя бы один слабый (синтетика содержит недо-Sharpe имена)
        self.assertTrue(s["funding"])
        self.assertTrue(all("flags" in f and f["flags"] for f in s["funding"]))

    def test_backtest_panel_b(self):
        s = build_scenario_payload(_synthetic_results())["scenario"]
        bt = s["backtest"]
        self.assertIn("ticker", bt)
        self.assertIn("summary", bt)
        self.assertIn("n_signals", bt["summary"])
        # walk_forward несёт дисклеймеры Фазы 3
        self.assertTrue(bt["summary"]["disclaimers"])

    def test_missing_data_graceful(self):
        self.assertFalse(build_scenario_payload({})["scenario"]["available"])
        self.assertFalse(
            build_scenario_payload({"history_result": SimpleNamespace(data=pd.DataFrame()),
                                    "performance_table": pd.DataFrame(),
                                    "total_value": 0.0})["scenario"]["available"])


class ScenarioRenderRoutingTest(unittest.TestCase):
    def test_select_template_scenario(self):
        import html_renderer
        self.assertEqual(html_renderer._select_template("scenario"),
                         "report_scenario_v3.html")

    def test_scenario_renders_via_jinja_even_with_premium_on(self):
        os.environ["PREMIUM_REPORT_ENABLED"] = "true"
        import html_renderer
        html = html_renderer.render_report_html(None, 7, "Сценарный анализ",
                                                tier="scenario")
        # Jinja-шаблон, НЕ premium React bundle
        self.assertIn("Панель A", html)
        self.assertIn("Панель B", html)
        self.assertIn("Euler-MCTR", html)
        self.assertNotIn("window.DEEP", html)

    def test_real_payload_renders(self):
        import html_renderer
        payload = build_scenario_payload(_synthetic_results())
        html = html_renderer.render_report_html(payload, 7, "Сценарный анализ",
                                                tier="scenario")
        self.assertIn("Сценарный анализ", html)
        self.assertIn("look-ahead guard", html)


class TierTariffTest(unittest.TestCase):
    def test_token_tariff(self):
        os.environ.setdefault("RAMP_BOT_TOKEN", "dummy:token")
        import tg_bot
        self.assertEqual(tg_bot.TIER_COST["base"], 1)
        self.assertEqual(tg_bot.TIER_COST["scenario"], 1)
        self.assertEqual(tg_bot.TIER_COST["deep"], 2)
        # сценарная кнопка присутствует в клавиатуре выбора
        cbs = [b.callback_data for row in tg_bot.kb_analysis_choice().inline_keyboard
               for b in row]
        self.assertIn("analysis:scenario", cbs)


if __name__ == "__main__":
    unittest.main()
