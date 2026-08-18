"""P2-раунд: сопровождаемость и косметика отчёта (A-6, A-7, C-6, R-9…R-16).

Две находки доски проверены по коду и оказались **несуществующими** — они
закрыты как ложные, а не «исправлены»:

  R-14 «тона Effect не соответствуют знаку» — метрики величины названы
       `*_magnitude` и НЕ входят в `_RISK_METRICS_LOWER_IS_BETTER`, поэтому
       CVaR −3.3% → −2.7% даёт improved=True. Все 8 строк живого отчёта 02.08
       окрашены верно.
  R-16 «confidence без единицы измерения» — JSX печатает «{r.confidence}%».

Остальное исправлено; здесь — регрессия на каждое.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ──────────────────────────────────────────────────────────────────────────
# A-7 · мусор в env не роняет старт контейнера
# ──────────────────────────────────────────────────────────────────────────
class EnvParsingTest(unittest.TestCase):

    def test_env_int_falls_back_on_garbage(self):
        from env_config import env_int
        with patch.dict(os.environ, {"X_TTL": "48h"}):
            self.assertEqual(env_int("X_TTL", 48), 48)

    def test_env_int_reads_a_valid_value(self):
        from env_config import env_int
        with patch.dict(os.environ, {"X_TTL": "72"}):
            self.assertEqual(env_int("X_TTL", 48), 72)

    def test_env_int_clamps(self):
        """Синтаксически валидное, но абсурдное значение тоже опасно."""
        from env_config import env_int
        with patch.dict(os.environ, {"X_TTL": "0"}):
            self.assertEqual(env_int("X_TTL", 48, lo=1, hi=168), 1)
        with patch.dict(os.environ, {"X_TTL": "100000"}):
            self.assertEqual(env_int("X_TTL", 48, lo=1, hi=168), 168)

    def test_empty_string_is_treated_as_unset(self):
        from env_config import env_int
        with patch.dict(os.environ, {"X_TTL": "   "}):
            self.assertEqual(env_int("X_TTL", 48), 48)

    def test_env_float(self):
        from env_config import env_float
        with patch.dict(os.environ, {"X_R": "abc"}):
            self.assertEqual(env_float("X_R", 0.045), 0.045)
        with patch.dict(os.environ, {"X_R": "0.07"}):
            self.assertAlmostEqual(env_float("X_R", 0.045), 0.07)

    def test_no_bare_int_env_parse_left_at_module_level(self):
        """🔴 Суть A-7: `int(os.getenv(...))` на уровне модуля падает при ИМПОРТЕ.

        Ошибка возникает раньше логирования, поэтому контейнер не стартует без
        внятной причины. Внутри функций та же конструкция допустима — там она
        обёрнута в try/except.
        """
        import ast
        offenders = []
        for path in Path("src").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            # Только ПРИСВАИВАНИЯ верхнего уровня: именно они исполняются при
            # импорте. Внутрь def/class не спускаемся — там та же конструкция
            # законна (исполняется при вызове и обёрнута в try/except).
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if value is None:
                    continue
                for sub in ast.walk(value):
                    if not isinstance(sub, ast.Call):
                        continue
                    if getattr(sub.func, "id", None) not in ("int", "float"):
                        continue
                    inner = sub.args[0] if sub.args else None
                    src = ast.unparse(inner) if inner is not None else ""
                    if "os.getenv" in src or "os.environ" in src:
                        offenders.append(f"{path}:{sub.lineno}")
        self.assertEqual(offenders, [], f"разбор env на уровне модуля: {offenders}")

    def test_migrated_call_sites_still_work(self):
        from services.report_storage import DEFAULT_TTL_HOURS
        from finance.scenario_engine import SCENARIO_LOOKBACK_DAYS
        self.assertEqual(DEFAULT_TTL_HOURS, 48)
        self.assertEqual(SCENARIO_LOOKBACK_DAYS, 1825)


# ──────────────────────────────────────────────────────────────────────────
# A-6 · честная диагностика при отрицательном капитале
# ──────────────────────────────────────────────────────────────────────────
class NegativeEquityDiagnosticsTest(unittest.TestCase):
    """Три РАЗНЫХ состояния печатали одно сообщение про «проверьте брокера»."""

    def _run(self, current_prices, quantities):
        from finance.investment_logic import UniversalPortfolioManager
        mgr = UniversalPortfolioManager(price_source="demo")
        idx = pd.bdate_range(end="2026-07-31", periods=400)
        cols = list(dict.fromkeys(list(mgr.engine.factor_tickers.values())
                                  + ["AAPL.US", "MSFT.US"]))
        frame = pd.DataFrame({c: np.linspace(100, 120, 400) for c in cols}, index=idx)
        port = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "USD"],
            "Quantity": quantities,
            "Purchase_Price": [190.0, 380.0, 1.0],
            "Broker_Current_Price": current_prices,
        })
        hr = type("HR", (), {"data": frame, "loaded": list(frame.columns),
                             "failed": {}, "retried": []})()
        with patch.object(mgr.engine, "get_market_data", return_value=(frame, hr)):
            return mgr.analyze_all(port)

    def test_negative_equity_names_the_real_cause(self):
        from finance.broker_api import RealPortfolioRequired
        with self.assertRaises(RealPortfolioRequired) as ctx:
            self._run([100.0, 100.0, 1.0], [1, 1, -10_000.0])
        msg = str(ctx.exception)
        self.assertIn("Заёмные средства превышают", msg)
        self.assertNotIn("Проверьте подключение", msg)

    def test_negative_equity_says_it_is_not_a_connection_failure(self):
        """Пользователь шёл чинить подключение вместо того, чтобы гасить плечо."""
        from finance.broker_api import RealPortfolioRequired
        with self.assertRaises(RealPortfolioRequired) as ctx:
            self._run([100.0, 100.0, 1.0], [1, 1, -10_000.0])
        self.assertIn("НЕ сбой подключения", str(ctx.exception))

    def test_empty_account_keeps_the_old_message(self):
        """Регресс: у пустого счёта диагноз не изменился."""
        from finance.broker_api import RealPortfolioRequired
        with self.assertRaises(RealPortfolioRequired) as ctx:
            self._run([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])   # нулевые количества
        self.assertIn("Проверьте подключение к брокеру", str(ctx.exception))

    def test_all_three_branches_exist(self):
        src = Path("src/finance/engine/portfolio_manager.py").read_text(encoding="utf-8")
        self.assertIn("total_portfolio_value < 0", src)
        self.assertIn("total_portfolio_value == 0", src)
        self.assertIn("not np.isfinite(total_portfolio_value)", src)


# ──────────────────────────────────────────────────────────────────────────
# C-6 · вырожденная ковариация: СООБЩАТЬ, а не чинить повторно
# ──────────────────────────────────────────────────────────────────────────
class DegenerateCovarianceTest(unittest.TestCase):

    def _repair(self, mat):
        from finance.investment_logic import _nearest_psd
        _nearest_psd.last_repair = None
        _nearest_psd(np.asarray(mat, dtype=float))
        return _nearest_psd.last_repair

    def test_healthy_matrix_reports_nothing(self):
        self.assertIsNone(self._repair(np.eye(3)))

    def test_degenerate_matrix_is_recorded(self):
        rep = self._repair([[1.0, 1.0000001, 0.0],
                            [1.0000001, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
        self.assertIsNotNone(rep)
        self.assertLess(rep["min_eigenvalue"], 0)
        self.assertGreaterEqual(rep["n_clipped"], 1)

    def test_repair_math_is_unchanged(self):
        """C-6 только НАБЛЮДАЕТ — сама починка обязана остаться прежней."""
        from finance.investment_logic import _nearest_psd
        M = np.array([[1.0, 1.0000001, 0.0],
                      [1.0000001, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        out = _nearest_psd(M)
        w = np.linalg.eigvalsh(out)
        self.assertTrue(np.all(w >= -1e-10), w)

    def test_checker_levels_follow_the_profile(self):
        from finance import data_checks as dc
        rep = self._repair([[1.0, 1.0000001, 0.0],
                            [1.0000001, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
        self.assertFalse(dc.check_covariance_health(
            rep, profile=dc.CheckProfile.LEGACY).blocked)
        self.assertTrue(dc.check_covariance_health(
            rep, profile=dc.CheckProfile.STRICT).blocked)
        self.assertFalse(dc.check_covariance_health(
            rep, profile=dc.CheckProfile.DEMO).blocked)

    def test_float_noise_is_not_a_finding(self):
        """Клипование на уровне 1e-15 — рутина округления, а не вырождение."""
        from finance import data_checks as dc
        noise = {"min_eigenvalue": -1e-15, "n_clipped": 1, "size": 5}
        self.assertEqual(
            dc.check_covariance_health(noise, profile=dc.CheckProfile.STRICT).findings,
            [])

    def test_none_input_is_safe(self):
        from finance import data_checks as dc
        for bad in (None, {}, {"min_eigenvalue": None}, "x"):
            self.assertEqual(
                dc.check_covariance_health(bad, profile=dc.CheckProfile.STRICT).findings,
                [])

    def test_engine_exposes_the_marker(self):
        from finance.investment_logic import MAC3RiskEngine
        self.assertIsNone(MAC3RiskEngine(price_source="demo")._last_psd_repair)

    def test_real_run_does_not_trip_c6(self):
        from finance.demo_portfolio import build_demo_portfolio
        from finance.investment_logic import UniversalPortfolioManager
        mgr = UniversalPortfolioManager(price_source="demo")
        res = mgr.analyze_all(build_demo_portfolio())
        self.assertNotIn("C-6", {f["id"] for f in res["data_quality"]["findings"]})


# ──────────────────────────────────────────────────────────────────────────
# R-13 · деньги в тексте ИИ == деньги на карточке
# ──────────────────────────────────────────────────────────────────────────
class MoneyForPromptTest(unittest.TestCase):

    METRICS = {"CVaR_95_Daily": -0.03312, "VaR_95_Daily": -0.0245,
               "Max_Drawdown": -0.412}

    def test_matches_the_kpi_card_convention(self):
        """Карточка считает от ПОКАЗАННОГО процента — текст обязан так же."""
        from ai_narrative import _money_for_prompt
        money = _money_for_prompt(self.METRICS, 14160)
        self.assertEqual(money["cvar_95"], "$467")      # 3.3% × 14160, не 3.312%
        self.assertEqual(money["max_dd"], "$5,834")

    def test_identical_to_pdf_payload_dollar(self):
        """Одна конвенция на два слоя — иначе снова разойдутся."""
        from ai_narrative import _money_for_prompt
        total = 14160.0
        for frac in (-0.03312, -0.0245, -0.412, 0.0781):
            pct_disp = round(frac * 100.0, 1)
            expect = f"${abs(pct_disp / 100.0 * total):,.0f}"
            got = _money_for_prompt({"CVaR_95_Daily": frac}, total)["cvar_95"]
            self.assertEqual(got, expect, frac)

    def test_zero_nav_yields_nothing(self):
        from ai_narrative import _money_for_prompt
        self.assertEqual(_money_for_prompt(self.METRICS, 0), {})
        self.assertEqual(_money_for_prompt(self.METRICS, None), {})

    def test_prompt_forbids_recomputing_money(self):
        src = Path("src/ai_narrative.py").read_text(encoding="utf-8")
        self.assertIn("`money`", src)
        self.assertIn("Не умножай процент", src)

    def test_summary_carries_the_block(self):
        from ai_narrative import _summarise_for_prompt
        out = _summarise_for_prompt({
            "portfolio_metrics": self.METRICS, "total_value": 14160,
            "performance_table": pd.DataFrame({"Ticker": ["AAPL"]}),
        })
        self.assertIn("money", out)
        self.assertEqual(out["money"]["cvar_95"], "$467")


# ──────────────────────────────────────────────────────────────────────────
# R-9 / R-10 / R-12 / R-15 · подписи в отчёте
# ──────────────────────────────────────────────────────────────────────────
class ReportLabelsTest(unittest.TestCase):

    def _bundle(self, name="deep-components.js"):
        return (Path("src/premium_assets") / name).read_text(encoding="utf-8")

    def test_r9_sparkline_window_is_named(self):
        """Число — по полному окну, точки — по скользящему 60-дневному."""
        self.assertIn("скользящему окну 60 дней", self._bundle())

    def test_r10_beta_base_is_named(self):
        """Частная бета факторной модели ≠ средневзвешенная бета позиций."""
        js = self._bundle()
        self.assertIn("Ridge", js)
        self.assertIn("средневзвешенной бете позиций", js)

    def test_r12_mandate_sum_explained_only_when_levered(self):
        js = self._bundle()
        self.assertIn("сумма превышает 100%", js)
        self.assertIn("m.leveraged", js)

    def test_r15_negative_variance_footnote(self):
        js = self._bundle()
        self.assertIn("Отрицательная доля — не ошибка", js)

    def test_jsx_sources_match_the_bundles(self):
        for rel, needle in (
            # Карточка KPI переехала в общий файл обоих тиров (`§−97`):
            # раньше её вёрстка существовала в двух копиях, и BASE потерял
            # график динамики вместе с заметкой ИИ.
            ("design/premium_v2/shared-kpi.jsx", "скользящему окну 60 дней"),
            ("design/premium_v2/deep/deep-factors.jsx", "Отрицательная доля — не ошибка"),
        ):
            p = Path(rel)
            if not p.exists():
                self.skipTest("design/ отсутствует (Docker deploy-gate)")
            self.assertIn(needle, p.read_text(encoding="utf-8"), rel)


# ──────────────────────────────────────────────────────────────────────────
# R-14 / R-16 · ложные находки доски — фиксируем, что дефекта НЕТ
# ──────────────────────────────────────────────────────────────────────────
class StaleBoardFindingsTest(unittest.TestCase):

    def test_r14_effect_tones_match_the_sign(self):
        """Метрики величины названы `*_magnitude` и не входят в LOWER_IS_BETTER."""
        from finance.simulate import _delta_row
        self.assertIs(_delta_row(-0.033, -0.027, "cvar_95_magnitude")["improved"], True)
        self.assertIs(_delta_row(-0.412, -0.387, "max_drawdown_magnitude")["improved"], True)
        self.assertIs(_delta_row(0.233, 0.192, "volatility_ann")["improved"], True)
        self.assertIs(_delta_row(27.2, 33.2, "max_trc")["improved"], False)

    def test_r14_worsening_still_reads_as_worse(self):
        from finance.simulate import _delta_row
        self.assertIs(_delta_row(-0.027, -0.033, "cvar_95_magnitude")["improved"], False)
        self.assertIs(_delta_row(0.192, 0.233, "volatility_ann")["improved"], False)

    def test_r16_confidence_carries_its_unit(self):
        js = (Path("src/premium_assets") / "deep-components.js").read_text(encoding="utf-8")
        self.assertIn("Уверенность модели", js)
        self.assertIn("confidence", js)


if __name__ == "__main__":
    unittest.main()
