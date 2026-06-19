"""
BLOCK 1–4 audit/refactor (2026-06-16) — focused unit tests.

Covers the surgical changes made for the Lead-Quant-Architect audit:

  BLOCK 1.2  Model routing            → BASE=Sonnet 4.6, DEEP=Opus 4.8 (current
                                         IDs honouring the intent; 2024 IDs
                                         rejected as a regression).
  BLOCK 1.2  Temperature guard        → Opus omits `temperature`, Sonnet keeps it.
  BLOCK 1.1  Idea freshness directive → period-stamped recency clause present.
  BLOCK 2.3  High-priority effect     → simulate only the non-deferred actions.
  BLOCK 3.4  Macro enrichment         → UNRATE + GDP in catalog; gated regime
                                         overlay (OFF by default, ON tilts axes).
  BLOCK 4.6  Multicollinearity        → factor diagnostic helper math.
  BLOCK 4.7  4-Pillar hardening       → NaN/Inf cannot poison the scores.
  BLOCK 3.5  Smart-Money foundation   → gated, deterministic scorer + CoVe row.
  BLOCK 4.8  CoVe rows                → LLM-checker + smart-money + factor rows.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — LLM routing + idea freshness
# ─────────────────────────────────────────────────────────────────────────────
class ModelRoutingTest(unittest.TestCase):
    def test_default_routing_is_current_generation(self):
        """BASE→Sonnet 4.6, DEEP→Opus 4.8 (NOT the 2024 -3-* IDs)."""
        import importlib
        # Ensure no env override leaks in from the host.
        for k in ("ANTHROPIC_MODEL_BASE", "ANTHROPIC_MODEL_DEEP"):
            os.environ.pop(k, None)
        import ai_narrative
        importlib.reload(ai_narrative)
        self.assertEqual(ai_narrative.MODEL_BASE, "claude-sonnet-4-6")
        self.assertEqual(ai_narrative.MODEL_DEEP, "claude-opus-4-8")
        # The outdated 2024 IDs must NOT be the routing target.
        self.assertNotIn("claude-3", ai_narrative.MODEL_BASE)
        self.assertNotIn("claude-3", ai_narrative.MODEL_DEEP)

    def test_temperature_guard_matches_routing(self):
        """DEEP (Opus) omits temperature; BASE (Sonnet) keeps it."""
        from ai_narrative import _model_supports_temperature
        self.assertFalse(_model_supports_temperature("claude-opus-4-8"))
        self.assertTrue(_model_supports_temperature("claude-sonnet-4-6"))

    def test_env_override_still_wins(self):
        """Routing stays env-overridable (e.g. restore the cheap BASE tier)."""
        import importlib
        os.environ["ANTHROPIC_MODEL_BASE"] = "claude-haiku-4-5-20251001"
        try:
            import ai_narrative
            importlib.reload(ai_narrative)
            self.assertEqual(ai_narrative.MODEL_BASE, "claude-haiku-4-5-20251001")
        finally:
            os.environ.pop("ANTHROPIC_MODEL_BASE", None)
            importlib.reload(ai_narrative)

    def test_idea_freshness_directive_present(self):
        """ideas_rule carries a period-stamped recency clause (BLOCK 1.1)."""
        from datetime import date
        from ai_narrative import _user_prompt
        prompt = _user_prompt({"regime": {"regime": "Expansion"}}, tier="deep")
        self.assertIn("СВЕЖЕСТЬ ИДЕЙ", prompt)
        self.assertIn(f"{date.today():%Y-%m}", prompt)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2.3 — high-priority Expected Effect
# ─────────────────────────────────────────────────────────────────────────────
class HighPriorityTargetTest(unittest.TestCase):
    def test_only_actionable_rows_move_weights(self):
        from finance.simulate import high_priority_target_weights
        cur = {"AAPL": 0.40, "MSFT": 0.30, "BND": 0.30}
        rows = [
            {"ticker": "AAPL", "action": "Trim", "delta_w_pp": -10.0},
            {"ticker": "MSFT", "action": "Hold", "delta_w_pp": 0.0},
            # deferred row demoted to Hold with delta 0 by build_action_plan:
            {"ticker": "BND",  "action": "Hold", "delta_w_pp": 0.0,
             "reason": "deferred (turnover cap)"},
        ]
        target, hp = high_priority_target_weights(cur, rows, bl_records=None)
        self.assertEqual(hp, ["AAPL"])
        self.assertAlmostEqual(target["AAPL"], 0.30, places=6)   # 0.40 − 0.10
        self.assertAlmostEqual(target["MSFT"], 0.30, places=6)   # unchanged
        self.assertAlmostEqual(target["BND"],  0.30, places=6)   # unchanged

    def test_falls_back_to_bl_when_no_actionable_rows(self):
        from finance.simulate import high_priority_target_weights
        cur = {"AAPL": 0.50, "MSFT": 0.50}
        rows = [{"ticker": "AAPL", "action": "Hold", "delta_w_pp": 0.0}]
        bl = [{"ticker": "AAPL", "target_w": 0.40},
              {"ticker": "MSFT", "target_w": 0.60}]
        target, hp = high_priority_target_weights(cur, rows, bl_records=bl)
        self.assertEqual(hp, [])                                 # no high-priority
        self.assertAlmostEqual(target["AAPL"], 0.40, places=6)   # BL fallback
        self.assertAlmostEqual(target["MSFT"], 0.60, places=6)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3.4 — macro enrichment + gated regime overlay
# ─────────────────────────────────────────────────────────────────────────────
class MacroRegimeOverlayTest(unittest.TestCase):
    def _ramp_prices(self):
        import numpy as np
        import pandas as pd
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        # Mild uptrend in equities, flat bonds → small positive growth signal.
        spy = pd.Series(100 * (1 + np.linspace(0, 0.08, 200)), index=idx)
        ief = pd.Series(100 * (1 + np.linspace(0, 0.01, 200)), index=idx)
        return pd.DataFrame({"SPY.US": spy, "IEF.US": ief})

    def test_overlay_off_by_default_is_unchanged(self):
        from finance.regime import RegimeClassifier
        os.environ.pop("REGIME_MACRO_OVERLAY", None)
        prices = self._ramp_prices()
        macro = {"unemployment": {"value": 3.5, "status": "ok"},
                 "gdp_growth":   {"value": 4.0, "status": "ok"}}
        a = RegimeClassifier().classify(prices)                  # no macro
        b = RegimeClassifier().classify(prices, macro=macro)     # macro, gate OFF
        self.assertIsNotNone(a)
        self.assertAlmostEqual(a.growth_score, b.growth_score, places=9)
        self.assertNotIn("macro_gdp_growth_nudge", b.signals)

    def test_overlay_on_tilts_axes(self):
        from finance.regime import RegimeClassifier
        os.environ["REGIME_MACRO_OVERLAY"] = "1"
        try:
            prices = self._ramp_prices()
            base = RegimeClassifier().classify(prices)
            hot  = {"unemployment": {"value": 3.0, "status": "ok"},   # tight
                    "gdp_growth":   {"value": 5.0, "status": "ok"}}   # hot
            tilted = RegimeClassifier().classify(prices, macro=hot)
            self.assertIn("macro_gdp_growth_nudge", tilted.signals)
            self.assertIn("macro_unemployment_nudge", tilted.signals)
            # Hot macro pushes the cycle axis up vs no overlay.
            self.assertGreater(tilted.cycle_score, base.cycle_score)
        finally:
            os.environ.pop("REGIME_MACRO_OVERLAY", None)

    def test_overlay_reads_trend_not_just_level(self):
        """Rising unemployment tilts cycle DOWN vs a flat series at same level."""
        from finance.regime import RegimeClassifier
        os.environ["REGIME_MACRO_OVERLAY"] = "1"
        try:
            prices = self._ramp_prices()
            rising = {"unemployment": {"value": 3.5, "status": "ok",
                       "history_30d": [{"value": v} for v in [3.0, 3.1, 3.2, 3.5]]}}
            flat   = {"unemployment": {"value": 3.5, "status": "ok",
                       "history_30d": [{"value": v} for v in [3.5, 3.5, 3.5, 3.5]]}}
            r = RegimeClassifier().classify(prices, macro=rising)
            f = RegimeClassifier().classify(prices, macro=flat)
            self.assertIn("macro_unemployment_trend", r.signals)
            self.assertGreater(r.signals["macro_unemployment_trend"], 0)   # rising
            self.assertLess(r.cycle_score, f.cycle_score)                  # cools cycle
        finally:
            os.environ.pop("REGIME_MACRO_OVERLAY", None)

    def test_overlay_ignores_missing_status_series(self):
        from finance.regime import RegimeClassifier
        os.environ["REGIME_MACRO_OVERLAY"] = "1"
        try:
            prices = self._ramp_prices()
            base = RegimeClassifier().classify(prices)
            missing = {"unemployment": {"value": None, "status": "missing"},
                       "gdp_growth":   {"value": None, "status": "missing"}}
            same = RegimeClassifier().classify(prices, macro=missing)
            self.assertAlmostEqual(same.cycle_score, base.cycle_score, places=9)
        finally:
            os.environ.pop("REGIME_MACRO_OVERLAY", None)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4.6 — multicollinearity diagnostic
# ─────────────────────────────────────────────────────────────────────────────
class FactorDiagnosticTest(unittest.TestCase):
    def test_collinear_factors_flagged(self):
        """Near-duplicate factors → high κ + max|corr|≈1 → near_collinear."""
        import numpy as np
        # Two almost-identical factors.
        F = np.array([[1.0, 0.99], [0.99, 1.0]])
        d = np.sqrt(np.diag(F))
        corr = F / np.outer(d, d)
        max_off = float(np.max(np.abs(corr - np.eye(2))))
        cond = float(np.linalg.cond(corr))
        self.assertGreater(max_off, 0.95)
        self.assertTrue(max_off > 0.95 or cond > 30.0)

    def test_orthogonal_factors_clean(self):
        import numpy as np
        F = np.diag([1.0, 2.0, 0.5])
        d = np.sqrt(np.diag(F))
        corr = F / np.outer(d, d)
        max_off = float(np.max(np.abs(corr - np.eye(3))))
        self.assertLess(max_off, 1e-9)
        self.assertLess(float(np.linalg.cond(corr)), 30.0)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4.7 — 4-Pillar NaN/Inf hardening
# ─────────────────────────────────────────────────────────────────────────────
class PillarHardeningTest(unittest.TestCase):
    def test_composite_risk_score_survives_nan_inf(self):
        from finance.scoring import composite_risk_score
        # NaN vol, Inf cvar, NaN concentration must not crash or return NaN.
        s = composite_risk_score(float("nan"), float("inf"), float("nan"),
                                 mandate="MODERATE")
        self.assertIsInstance(s, int)
        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 100)

    def test_non_finite_inputs_neutralized_not_saturating(self):
        from finance.scoring import composite_risk_score
        # A non-finite input is DATA CORRUPTION, not "infinite risk": _finite
        # neutralises it to 0 so a garbage feed can't flash a false max-risk
        # alarm.  Inf cvar must score identically to cvar=0 (both neutral).
        s_inf  = composite_risk_score(0.30, float("inf"), 10.0, mandate="MODERATE")
        s_zero = composite_risk_score(0.30, 0.0,          10.0, mandate="MODERATE")
        self.assertEqual(s_inf, s_zero)

    def test_fundamentals_score_nan_macro_alignment(self):
        from finance.scoring import fundamentals_score
        import math
        out = fundamentals_score(roe_z=2.0, op_margin_z=None,
                                 debt_to_assets_z=None, revenue_growth_z=None,
                                 macro_alignment=float("nan"))
        self.assertFalse(math.isnan(out))
        self.assertEqual(out, 1.0)        # ROE +1, NaN macro coerced to 0


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3.5 — Smart-Money foundation
# ─────────────────────────────────────────────────────────────────────────────
class SmartMoneyTest(unittest.TestCase):
    def test_disabled_by_default(self):
        from finance.smart_money import build_insider_signals
        os.environ.pop("SMART_MONEY_INSIDERS", None)
        out = build_insider_signals(["AAPL", "MSFT"])
        self.assertEqual(out["AAPL"]["status"], "disabled")
        self.assertEqual(out["MSFT"]["status"], "disabled")

    def test_cluster_buy_scores_positive(self):
        from finance.smart_money import score_insider_flow
        score, cluster = score_insider_flow(
            net_flow_usd=5_000_000, distinct_buyers=4, sell_count=0,
            market_cap_usd=1_000_000_000, role_weight=1.5)
        self.assertTrue(cluster)
        self.assertGreater(score, 0)

    def test_insider_distribution_scores_negative(self):
        from finance.smart_money import score_insider_flow
        score, cluster = score_insider_flow(
            net_flow_usd=-3_000_000, distinct_buyers=0, sell_count=3,
            market_cap_usd=500_000_000, role_weight=1.0)
        self.assertFalse(cluster)
        self.assertLess(score, 0)

    def test_enabled_with_injected_fetcher(self):
        from finance.smart_money import build_insider_signals
        os.environ["SMART_MONEY_INSIDERS"] = "1"
        try:
            def fake_fetch(t):
                return {"net_flow_usd": 2_000_000, "distinct_buyers": 3,
                        "buy_count": 3, "sell_count": 0, "top_role": "CEO",
                        "as_of": "2026-06-10"}
            out = build_insider_signals(["AAPL"], fetch=fake_fetch,
                                        market_caps={"AAPL": 1_000_000_000})
            self.assertEqual(out["AAPL"]["status"], "ok")
            self.assertTrue(out["AAPL"]["cluster_flag"])
        finally:
            os.environ.pop("SMART_MONEY_INSIDERS", None)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4.8 — CoVe rows
# ─────────────────────────────────────────────────────────────────────────────
class CoVeRowsTest(unittest.TestCase):
    def test_llm_checker_rows_present(self):
        from finance.data_lineage import build_lineage
        rows = build_lineage({}, ai_summary={"verdict": "x", "bullets": ["y"]})
        names = [r["name"] for r in rows]
        self.assertIn("LLM-чекер: контроль галлюцинаций", names)
        self.assertIn("LLM-чекер: проверка вычислений", names)

    def test_smart_money_row_present_and_gated(self):
        from finance.data_lineage import build_lineage
        rows = build_lineage({}, ai_summary={})
        sm = [r for r in rows if "Smart-Money" in r["name"]]
        self.assertEqual(len(sm), 1)
        self.assertEqual(sm[0]["status"], "missing")

    def test_factor_diagnostic_row_reflects_metrics(self):
        from finance.data_lineage import build_lineage
        results = {"portfolio_metrics": {"factor_diagnostics": {
            "n_factors": 5, "max_abs_corr": 0.97,
            "condition_number": 42.0, "near_collinear": True}}}
        rows = build_lineage(results, ai_summary={})
        fd = [r for r in rows if "мультиколлинеарность" in r["name"]]
        self.assertEqual(len(fd), 1)
        self.assertEqual(fd[0]["status"], "warn")
        self.assertIn("0.97", fd[0]["note"])

    def test_macro_missing_row_no_longer_mentions_pmi(self):
        from finance.data_lineage import build_lineage
        rows = build_lineage({}, ai_summary={})
        macro = [r for r in rows if r["name"] == "Макро-драйверы (FRED)"]
        self.assertEqual(len(macro), 1)
        self.assertNotIn("PMI", macro[0]["method"])
        self.assertIn("unemployment", macro[0]["method"])


if __name__ == "__main__":
    unittest.main()
