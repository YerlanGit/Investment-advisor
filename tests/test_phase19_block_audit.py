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
        target, hp, acts = high_priority_target_weights(cur, rows, bl_records=None)
        self.assertEqual(hp, ["AAPL"])
        self.assertAlmostEqual(target["AAPL"], 0.30, places=6)   # 0.40 − 0.10
        self.assertAlmostEqual(target["MSFT"], 0.30, places=6)   # unchanged
        self.assertAlmostEqual(target["BND"],  0.30, places=6)   # unchanged
        # User #5 — the idea breakdown spells out the direction.
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts[0]["ticker"], "AAPL")
        self.assertEqual(acts[0]["side"], "sell")
        self.assertEqual(acts[0]["delta_pp"], -10.0)

    def test_falls_back_to_bl_when_no_actionable_rows(self):
        from finance.simulate import high_priority_target_weights
        cur = {"AAPL": 0.50, "MSFT": 0.50}
        rows = [{"ticker": "AAPL", "action": "Hold", "delta_w_pp": 0.0}]
        bl = [{"ticker": "AAPL", "target_w": 0.40},
              {"ticker": "MSFT", "target_w": 0.60}]
        target, hp, acts = high_priority_target_weights(cur, rows, bl_records=bl)
        self.assertEqual(hp, [])                                 # no high-priority
        self.assertEqual(acts, [])                               # no idea breakdown
        self.assertAlmostEqual(target["AAPL"], 0.40, places=6)   # BL fallback
        self.assertAlmostEqual(target["MSFT"], 0.60, places=6)

    def test_sells_only_idea_reinvests_into_held_bl_buys(self):
        # User #1 (06-25): a sells-only high-priority idea must SHOW the buy side
        # by reinvesting proceeds into held BL-buy targets (so metrics reflect it).
        from finance.simulate import high_priority_target_weights
        cur = {"NVDA": 0.40, "ORCL": 0.30, "GOOGL": 0.20, "BND": 0.10}
        rows = [{"ticker": "NVDA", "action": "Trim", "delta_w_pp": -10.0},
                {"ticker": "ORCL", "action": "Sell", "delta_w_pp": -10.0}]
        bl = [{"ticker": "GOOGL", "delta_w_pp": 8.0, "action": "Buy"},   # held → reinvest
              {"ticker": "XYZ",   "delta_w_pp": 5.0, "action": "Buy"}]   # not held → skip
        target, hp, acts = high_priority_target_weights(cur, rows, bl_records=bl)
        sides = {a["ticker"]: a["side"] for a in acts}
        self.assertEqual(sides.get("NVDA"), "sell")
        self.assertEqual(sides.get("GOOGL"), "buy")     # buy side now shown
        self.assertNotIn("XYZ", sides)                  # new name not simulated
        # 20pp freed, GOOGL capped at +12pp; remainder (8pp) stays cash.
        self.assertAlmostEqual(target["GOOGL"], 0.20 + 0.12, places=4)


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

    def test_overlay_explicit_off_is_unchanged(self):
        # BLOCK 2 (2026-06-26): the overlay is now DEFAULT-ON, so the
        # unchanged-classifier guarantee lives behind the explicit escape
        # hatch REGIME_MACRO_OVERLAY=0.
        from finance.regime import RegimeClassifier
        os.environ["REGIME_MACRO_OVERLAY"] = "0"
        try:
            prices = self._ramp_prices()
            macro = {"unemployment": {"value": 3.5, "status": "ok"},
                     "gdp_growth":   {"value": 4.0, "status": "ok"}}
            a = RegimeClassifier().classify(prices)                  # no macro
            b = RegimeClassifier().classify(prices, macro=macro)     # macro, gate OFF
            self.assertIsNotNone(a)
            self.assertAlmostEqual(a.growth_score, b.growth_score, places=9)
            self.assertNotIn("macro_gdp_growth_nudge", b.signals)
        finally:
            os.environ.pop("REGIME_MACRO_OVERLAY", None)

    def test_overlay_on_by_default_consults_macro(self):
        # Unset env → overlay active → macro dynamics fold into the axes.
        from finance.regime import RegimeClassifier
        os.environ.pop("REGIME_MACRO_OVERLAY", None)
        prices = self._ramp_prices()
        hot = {"unemployment": {"value": 3.0, "status": "ok"},
               "gdp_growth":   {"value": 5.0, "status": "ok"}}
        reading = RegimeClassifier().classify(prices, macro=hot)
        self.assertIsNotNone(reading)
        self.assertIn("macro_gdp_growth_nudge", reading.signals)
        self.assertIn("macro_unemployment_nudge", reading.signals)

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


# ─────────────────────────────────────────────────────────────────────────────
# F3 (UI) — macro rate-of-change chip in the DEEP Regime panel
# ─────────────────────────────────────────────────────────────────────────────
class MacroTrendChipTest(unittest.TestCase):
    def test_rising_monthly_series_up(self):
        from pdf_payload import _macro_series_trend
        row = {"series_id": "UNRATE", "unit": "%", "publish_cadence": "monthly",
               "history_30d": [{"value": v} for v in [3.8, 3.9, 4.0, 4.1]]}
        t = _macro_series_trend(row)
        self.assertEqual(t["dir"], "▲")
        self.assertGreater(t["delta"], 0)
        self.assertIn("за 3м", t["label"])

    def test_falling_daily_series_down(self):
        from pdf_payload import _macro_series_trend
        row = {"series_id": "VIXCLS", "unit": "index", "publish_cadence": "daily",
               "history_30d": [{"value": v} for v in [20, 18, 16, 15] * 6]}
        t = _macro_series_trend(row)
        self.assertEqual(t["dir"], "▼")
        self.assertIn("за 1м", t["label"])

    def test_hy_oas_delta_in_bp(self):
        from pdf_payload import _macro_series_trend
        # FRED reports HY OAS in % (3.10→3.40); the chip must show bp like value.
        row = {"series_id": "BAMLH0A0HYM2", "unit": "pp", "publish_cadence": "daily",
               "history_30d": [{"value": v} for v in [3.10, 3.20, 3.30, 3.40] * 6]}
        t = _macro_series_trend(row)
        self.assertIn("bp", t["label"])
        self.assertEqual(t["dir"], "▲")

    def test_insufficient_history_returns_none(self):
        from pdf_payload import _macro_series_trend
        self.assertIsNone(_macro_series_trend({"history_30d": [{"value": 1.0}]}))

    def test_panel_exposes_trend_label(self):
        from pdf_payload import _build_macro_drivers_panel
        raw = {"unemployment": {"series_id": "UNRATE", "value": 4.1, "unit": "%",
                                "status": "ok", "as_of": "2026-06-01",
                                "publish_cadence": "monthly",
                                "history_30d": [{"value": v} for v in [3.8, 3.9, 4.0, 4.1]]}}
        panel = _build_macro_drivers_panel(raw)
        self.assertEqual(len(panel["series"]), 1)
        self.assertTrue(panel["series"][0]["trend_label"])
        self.assertEqual(panel["series"][0]["trend_dir"], "▲")


# ─────────────────────────────────────────────────────────────────────────────
# B3.5 — hierarchical factor orthogonalization (κ reduction, names preserved)
# ─────────────────────────────────────────────────────────────────────────────
class FactorOrthogonalizationTest(unittest.TestCase):
    def _collinear_frame(self):
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        n = 250
        mkt = rng.normal(0, 0.01, n)
        return pd.DataFrame({
            "Market":      mkt,
            "Rates":       rng.normal(0, 0.005, n),
            "Commodities": rng.normal(0, 0.008, n),
            "Momentum":    mkt * 1.1 + rng.normal(0, 0.004, n),
            "Value":       mkt * 0.9 + rng.normal(0, 0.004, n),
            "Quality":     mkt * 1.0 + rng.normal(0, 0.004, n),
            "Size":        mkt * 1.2 + rng.normal(0, 0.004, n),
            "Volatility":  mkt * 0.7 + rng.normal(0, 0.004, n),
            "EM_Equity":   mkt * 1.3 + rng.normal(0, 0.004, n),
            "EM_Bond":     rng.normal(0, 0.005, n),
        })

    @staticmethod
    def _kappa(df):
        import numpy as np
        F = np.cov(df.values.T)
        d = np.sqrt(np.clip(np.diag(F), 1e-18, None))
        return float(np.linalg.cond(F / np.outer(d, d)))

    def test_orthogonalization_reduces_condition_number(self):
        from finance.investment_logic import orthogonalize_factors_hierarchical
        df = self._collinear_frame()
        k_raw = self._kappa(df)
        ortho = orthogonalize_factors_hierarchical(df)
        k_ortho = self._kappa(ortho)
        self.assertGreater(k_raw, 30.0)            # raw factors ARE collinear
        self.assertLess(k_ortho, k_raw)            # orthogonalization helps
        self.assertEqual(list(ortho.columns), list(df.columns))   # names kept

    def test_core_factors_pass_through_unchanged(self):
        import numpy as np, pandas as pd
        from finance.investment_logic import orthogonalize_factors_hierarchical
        n = 50
        df = pd.DataFrame({"Market": np.linspace(0, 1, n),
                           "Rates": np.linspace(1, 0, n),
                           "Commodities": np.linspace(0, 2, n),
                           "Momentum": np.linspace(0, 1, n)})
        out = orthogonalize_factors_hierarchical(df)
        np.testing.assert_allclose(out["Market"].values, df["Market"].values)
        np.testing.assert_allclose(out["Rates"].values, df["Rates"].values)

    def test_gate_default_on(self):
        # BLOCK 3 (2026-06-26): orthogonalization is now DEFAULT-ON so production
        # κ drops below the collinearity threshold.  Unset → enabled; only an
        # explicit off-switch restores the legacy collinear decomposition.
        import os
        from finance.investment_logic import factor_orthogonalize_enabled
        os.environ.pop("FACTOR_ORTHOGONALIZE", None)
        self.assertTrue(factor_orthogonalize_enabled())          # default ON
        try:
            os.environ["FACTOR_ORTHOGONALIZE"] = "0"
            self.assertFalse(factor_orthogonalize_enabled())     # escape hatch
            os.environ["FACTOR_ORTHOGONALIZE"] = "1"
            self.assertTrue(factor_orthogonalize_enabled())
        finally:
            os.environ.pop("FACTOR_ORTHOGONALIZE", None)


# ─────────────────────────────────────────────────────────────────────────────
# B1.1 — idea rotation / freshness (daily anchor + expanded ban)
# ─────────────────────────────────────────────────────────────────────────────
class IdeaRotationTest(unittest.TestCase):
    def test_daily_anchor_and_rotation_angle(self):
        from datetime import date
        from ai_narrative import _user_prompt
        p = _user_prompt({"regime": {"regime": "Expansion"}}, tier="base")
        self.assertIn(f"{date.today():%Y-%m-%d}", p)   # DAILY, not %Y-%m
        self.assertIn("УГОЛ РОТАЦИИ", p)

    def test_ban_list_expanded_to_default_bluechips(self):
        from ai_narrative import _user_prompt
        p = _user_prompt({"regime": {"regime": "Expansion"}}, tier="base")
        for name in ("UNH", "AVGO", "GS", "MA"):
            self.assertIn(name, p)

    def test_temperature_band_raised(self):
        import importlib, ai_narrative
        importlib.reload(ai_narrative)
        self.assertGreaterEqual(ai_narrative.NARRATIVE_TEMPERATURE, 0.5)
        self.assertLessEqual(ai_narrative.NARRATIVE_TEMPERATURE, 0.85)


# ─────────────────────────────────────────────────────────────────────────────
# B2.4 — Smart Money panel states
# ─────────────────────────────────────────────────────────────────────────────
class SmartMoneyPanelTest(unittest.TestCase):
    def test_disabled_state_is_renderable(self):
        import os
        from finance.smart_money import build_insider_signals
        from pdf_payload import _build_smart_money
        os.environ.pop("SMART_MONEY_INSIDERS", None)
        panel = _build_smart_money(build_insider_signals(["AAPL", "MSFT"]))
        self.assertEqual(panel["status"], "disabled")
        self.assertFalse(panel["enabled"])
        self.assertTrue(panel["headline"])
        self.assertIn("SMART_MONEY_INSIDERS", panel.get("hint", ""))

    def test_active_state_lists_rows(self):
        from pdf_payload import _build_smart_money
        raw = {"AAPL": {"status": "ok", "ticker": "AAPL", "net_flow_usd": 1e6,
                        "buy_count": 3, "sell_count": 0, "cluster_flag": True,
                        "score": 1.5}}
        panel = _build_smart_money(raw)
        self.assertEqual(panel["status"], "active")
        self.assertTrue(panel["enabled"])
        self.assertEqual(len(panel["rows"]), 1)
        self.assertTrue(panel["rows"][0]["cluster"])


# ─────────────────────────────────────────────────────────────────────────────
# B2.3 — trend over ≥3 changes (multi-point slope, robust to a single spike)
# ─────────────────────────────────────────────────────────────────────────────
class SeriesTrendThreeChangesTest(unittest.TestCase):
    def test_needs_four_points_three_changes(self):
        from finance.regime import series_trend
        total, _s, _n = series_trend([1.0, 2.0, 3.0], 3)      # 3 pts = 2 changes
        self.assertIsNone(total)
        total, _s, n = series_trend([1.0, 2.0, 3.0, 4.0], 3)  # 4 pts = 3 changes
        self.assertIsNotNone(total)
        self.assertGreater(total, 0)

    def test_slope_robust_to_single_last_spike(self):
        from finance.regime import series_trend
        # steady rise, last print dips — OLS over ≥3 changes still reads "up",
        # whereas a single latest−prior delta would have flipped to "down".
        total, _s, _n = series_trend([1.0, 2.0, 3.0, 4.0, 3.8], 4)
        self.assertGreater(total, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 06-23 audit fixes: Smart Money idea card, CoVe neutral status, MaxDD $, etc.
# ─────────────────────────────────────────────────────────────────────────────
class Audit0623Test(unittest.TestCase):
    def test_idea_card_is_smart_money_not_regime(self):
        from pdf_payload import _build_ai_ideas
        picks = {"smart_money": {"label": "Smart Money", "desc": "x",
                                 "picks": [{"ticker": "BRK.B", "name": "B",
                                            "why": "institutional", "type": "Stock"}]}}
        out = _build_ai_ideas(picks, tier="deep")
        self.assertTrue(out.get("rotation"))
        self.assertEqual(out["rotation"][0]["category"], "Smart Money")

    def test_smart_money_cove_status_disabled_not_missing(self):
        # Gated-off Smart Money must be 'disabled' (neutral –), NOT 'error'/red ✗.
        import os
        from finance.smart_money import build_insider_signals, insider_lineage_row
        os.environ.pop("SMART_MONEY_INSIDERS", None)
        row = insider_lineage_row(build_insider_signals(["AAPL", "MSFT"]))
        self.assertEqual(row["status"], "disabled")

    def test_expected_effect_carries_idea_actions(self):
        from pdf_payload import _build_expected_effect
        raw = {"metrics": {"risk_index": {"before": 49, "after": 51,
                                          "delta": 2, "improved": False}},
               "high_priority_tickers": ["NVDA"],
               "high_priority_actions": [{"ticker": "NVDA", "action": "Trim",
                                          "side": "sell", "delta_pp": -10.0}]}
        out = _build_expected_effect(raw)
        self.assertIn("high_priority_actions", out)
        self.assertEqual(out["high_priority_actions"][0]["side"], "Продать")
        self.assertEqual(out["high_priority_actions"][0]["side_key"], "sell")

    def test_expected_effect_empty_stays_empty(self):
        from pdf_payload import _build_expected_effect
        self.assertEqual(_build_expected_effect(None), {})
        self.assertEqual(_build_expected_effect({"metrics": {}}), {})


# ─────────────────────────────────────────────────────────────────────────────
# Premium V2 data mapper (engine payload → strict design contract)
# ─────────────────────────────────────────────────────────────────────────────
class PremiumMapperTest(unittest.TestCase):
    def test_deep_contract_is_exactly_29_keys(self):
        from premium_payload import build_design_data
        d = build_design_data({}, "deep")          # empty payload → no KeyError
        self.assertEqual(len(d), 29)

    def test_base_contract_is_exactly_11_keys(self):
        from premium_payload import build_design_data
        b = build_design_data({}, "base")
        self.assertEqual(len(b), 11)

    def test_none_payload_defensive_dash(self):
        from premium_payload import build_design_data
        d = build_design_data(None, "deep")        # None → never raises
        self.assertEqual(d["verdict"]["headline"], "–")   # missing text → neutral '–'
        self.assertEqual(d["meta"]["aiModel"], "–")

    def test_chart_numeric_fields_stay_numeric(self):
        # Defensive '–' must NOT leak into fields the React charts do math on.
        from premium_payload import build_design_data
        d = build_design_data({}, "deep")
        self.assertIsInstance(d["verdict"]["riskIndex"], (int, float))
        # holdings/actionPlan numeric subfields default to 0, not '–'
        b = build_design_data({"assets": [{"ticker": "X"}]}, "base")
        self.assertIsInstance(b["holdings"][0]["w"], (int, float))

    def test_cove_status_collapses_to_3_state_keeping_disabled_neutral(self):
        from premium_payload import _cove_st
        self.assertEqual(_cove_st("ok"), "ok")
        self.assertEqual(_cove_st("error"), "fail")
        self.assertEqual(_cove_st("warn"), "warn")
        self.assertEqual(_cove_st("disabled"), "warn")   # off ≠ failed (not 'fail')
        self.assertEqual(_cove_st("missing"), "warn")

    def test_idea_tone_only_valid_design_keys(self):
        from premium_payload import build_design_data
        d = build_design_data({"ai_ideas": {"risk_reduction": [{"category": "X"}],
                                            "diversification": [{"category": "Y"}]}}, "deep")
        for idea in d["ideas"]:
            self.assertIn(idea["tone"], {"grow", "rebalance", "rotation", "hedge"})

    def test_feature_flag_default_now_premium(self):
        # Sprint-1 #1: Premium V2 is the PRODUCTION DEFAULT — an unset flag routes
        # to the React shell, not v3.
        import os, importlib, html_renderer
        os.environ.pop("PREMIUM_REPORT_ENABLED", None)
        importlib.reload(html_renderer)
        try:
            self.assertTrue(html_renderer.PREMIUM_REPORT_ENABLED)
            html = html_renderer.render_report_html(html_renderer._mock_payload("deep"),
                                                    user_id="x", tier="deep")
            self.assertIn('id="root"', html)         # Premium V2 React shell
        finally:
            importlib.reload(html_renderer)

    def test_feature_flag_false_routes_to_v3_fallback(self):
        # v3 is RETAINED as the explicit fallback: forcing the flag off must still
        # render the classic Jinja pipeline (resilience net).
        import os, importlib, html_renderer
        os.environ["PREMIUM_REPORT_ENABLED"] = "false"
        importlib.reload(html_renderer)
        try:
            self.assertFalse(html_renderer.PREMIUM_REPORT_ENABLED)
            html = html_renderer.render_report_html(html_renderer._mock_payload("deep"),
                                                    user_id="x", tier="deep")
            self.assertNotIn('id="root"', html)      # v3 Jinja, not the React shell
        finally:
            os.environ.pop("PREMIUM_REPORT_ENABLED", None)
            importlib.reload(html_renderer)

    def test_feature_flag_on_routes_to_premium(self):
        # The flip side of the routing contract: with the flag ON the report is
        # the Premium V2 React shell, NOT v3.  Proves the assets are present and
        # the premium path renders without throwing (else it would fall back).
        import os, importlib, html_renderer
        os.environ["PREMIUM_REPORT_ENABLED"] = "true"
        importlib.reload(html_renderer)
        try:
            self.assertTrue(html_renderer.PREMIUM_REPORT_ENABLED)
            for tier in ("deep", "base"):
                html = html_renderer.render_report_html(
                    html_renderer._mock_payload(tier), user_id="x", tier=tier)
                self.assertIn('id="root"', html)         # React shell
                self.assertIn('createElement', html)     # compiled components present
                self.assertNotIn('class="sheet"', html)  # NOT the v3 fallback
        finally:
            os.environ.pop("PREMIUM_REPORT_ENABLED", None)
            importlib.reload(html_renderer)


class PremiumMapperAuditTest(unittest.TestCase):
    """2026-06-27 premium-report audit: lost-parameter + formatting fixes."""

    def test_mandate_reads_real_engine_keys(self):
        # The mapper used to read target_vol / tracking_cap / value / state — none
        # of which the engine emits — so the whole panel was '–' / 0.0.
        from premium_payload import build_design_data
        p = {"risk_mandate_label": "Умеренно-агрессивный",
             "mandate_compliance": {"target_vol_pct": 14.0, "target_te_pct": 5.0,
                "breaches": 1,
                "rows": [{"label": "Акции США", "actual": 68.1, "lo": 30.0, "hi": 60.0, "status": "over"}]}}
        m = build_design_data(p, "deep")["mandate"]
        self.assertEqual(m["targetVol"], 14.0)
        self.assertEqual(m["trackingCap"], 5.0)
        self.assertEqual(m["violations"], 1)
        self.assertEqual(m["rows"][0]["value"], 68.1)     # was 0.0 (wrong key)
        self.assertEqual(m["rows"][0]["state"], "over")   # was '–' (wrong key)

    def test_effect_cards_are_formatted_not_raw(self):
        # %-metrics stored as fractions must render as "18.7%", not "0.18692…".
        from premium_payload import build_design_data
        p = {"expected_effect": {
            "risk_index": {"before": 49, "after": 45, "delta_pp": -4.0, "favourable": True},
            "vol":        {"before": 0.18692, "after": 0.16094, "delta_pp": -2.6, "favourable": True},
            "sharpe":     {"before": 0.3419, "after": 0.4489, "delta_pp": 0.107, "favourable": True}}}
        eff = {e["name"]: e for e in build_design_data(p, "deep")["effect"]}
        self.assertEqual(eff["Волатильность"]["before"], "18.7%")
        self.assertEqual(eff["Sharpe"]["before"], "0.34")
        self.assertEqual(eff["Индекс риска"]["before"], "49")
        self.assertNotIn("0.18", eff["Волатильность"]["before"])

    def test_idea_pipeline_no_tuple_leak(self):
        # A (stage, detail) tuple must render as the detail string, never as a
        # Python repr «('RAG', 'Фундаментальный анализ')».
        from premium_payload import build_design_data
        p = {"ai_ideas": {"rotation": [{"title": "T", "rationale": "R",
              "pipeline": [("FACTOR", "4-Pillar"), ("RAG", "Фундаментальный анализ")],
              "candidates": [{"ticker": "MSCI", "name": "MSCI", "why": "w"}]}]}}
        pipe = build_design_data(p, "base")["ideas"][0]["pipeline"]
        self.assertEqual(pipe, ["4-Pillar", "Фундаментальный анализ"])
        self.assertFalse(any("(" in s and "'" in s for s in pipe))

    def test_generated_timestamp_flows_to_premium_meta(self):
        from premium_payload import build_design_data
        d = build_design_data({}, "deep", generated_at="27.06.2026 11:45 UTC+5")
        self.assertEqual(d["meta"]["generated"], "27.06.2026 11:45 UTC+5")
        self.assertNotEqual(d["meta"]["generated"], "–")

    def test_holdings_join_sec_fundamentals(self):
        # SEC EDGAR fundamentals live in a SEPARATE fundamental_layer[] keyed by
        # ticker — NOT on assets[].  The mapper used to read a non-existent
        # assets[].fundamentals key ⇒ every holding's grid was empty.
        from premium_payload import build_design_data
        p = {"assets": [{"ticker": "AAPL", "atr_pct": "1.60%", "weight_pct_num": 16.0},
                        {"ticker": "GLD",  "atr_pct": "0.44%", "weight_pct_num": 8.0}],
             "fundamental_layer": [{"ticker": "AAPL", "roe": "147.0%", "op_m": "30.1%",
                                    "dta": "31.0%", "rev_g": "+2.0%", "altman_z": "8.21"}]}
        for tier in ("deep", "base"):
            h = {x["t"]: x["fund"] for x in build_design_data(p, tier)["holdings"]}
            self.assertEqual(h["AAPL"]["roe"], "147.0%")     # was {} (wrong key)
            self.assertEqual(h["AAPL"]["margin"], "30.1%")   # op_m → margin
            self.assertEqual(h["AAPL"]["z"], "8.21")         # altman_z → z
            self.assertEqual(h["AAPL"]["atr"], "1.60%")      # atr from the asset row
            self.assertEqual(h["GLD"]["roe"], "н/д")         # ETF: no SEC coverage
            self.assertEqual(h["GLD"]["atr"], "0.44%")

    def test_regime_drivers_read_series_list(self):
        # macro_drivers is the ADAPTED panel {"series":[...]}, NOT a raw
        # {key:{value}} dict — the old .items() walk always yielded [].
        from premium_payload import build_design_data
        p = {"macro_drivers": {"series": [
                {"name": "Кривая 10Y−2Y", "value": "+0.18 pp", "status": "ok", "trend_label": "▲"},
                {"name": "VIX", "value": "14.2", "status": "stale", "trend_label": "▬"},
                {"name": "Безработица", "value": "—", "status": "missing"}]},  # dropped
             "regime": {"label": "Expansion", "confidence": 0.4}}
        drv = build_design_data(p, "deep")["regime"]["drivers"]
        self.assertEqual(len(drv), 2)                        # was 0; missing dropped
        self.assertEqual(drv[0]["name"], "Кривая 10Y−2Y")
        self.assertEqual(drv[0]["val"], "+0.18 pp")
        self.assertEqual(drv[0]["tone"], "pos")             # ok → sage
        self.assertEqual(drv[1]["tone"], "warn")            # stale → gold

    def test_sector_hues_are_distinct(self):
        # Every sector used to get the same #1c1b1a → the stacked bar + legend
        # rendered as one indistinguishable black block.
        from premium_payload import build_design_data
        p = {"sectors": [{"name": "Technology", "weight_pct": 49},
                         {"name": "Semiconductors", "weight_pct": 15},
                         {"name": "Bonds", "weight_pct": 9},
                         {"name": "Gold", "weight_pct": 6}]}
        hues = [s["hue"] for s in build_design_data(p, "base")["sectors"]]
        self.assertEqual(len(set(hues)), 4)                  # all distinct
        self.assertNotIn("#1c1b1a", hues)                    # not the old single black

    def test_performance_summary_is_real_not_template(self):
        # The component hardcoded +14.2% / +5.1пп vs S&P; on an underperforming
        # book that CONTRADICTED reality.  The mapper now carries the true 12-month
        # figures and derives the period excess (d = p − s).
        from premium_payload import build_design_data
        p = {"period_returns_table": {"S&P 500": {"periods": [
                {"label": "1М",  "portfolio": "-10.8%", "benchmark": "-1.9%"},
                {"label": "12М", "portfolio": "14.3%",  "benchmark": "21.2%"}]}},
             "volatility": 0.187}
        perf = build_design_data(p, "base")["performance"]
        self.assertEqual(perf["summary"]["ret"], 14.3)
        self.assertEqual(perf["summary"]["spx"], 21.2)
        self.assertEqual(perf["summary"]["exc"], -6.9)       # underperformance, was +5.1 template
        by = {r["label"]: r for r in perf["periods"]}
        self.assertEqual(by["1М"]["d"], round(-10.8 - (-1.9), 1))   # excess derived

    def test_regime_signals_parsed_to_objects(self):
        # The DEEP component renders each confirm bullet as {ok, t}; the mapper
        # passed raw strings → b.t undefined → six icon-only rows with NO text.
        from premium_payload import build_design_data
        p = {"regime_confirmation": {"stance": "partial", "signals": [
                "✓ Безработица снижается (темп −0,09)",
                "⚠ ВВП замедляется (темп −2,7) — против фазы роста"]}}
        cb = build_design_data(p, "deep")["regime"]["confirmBullets"]
        self.assertEqual(cb[0], {"ok": True,  "t": "Безработица снижается (темп −0,09)"})
        self.assertEqual(cb[1], {"ok": False, "t": "ВВП замедляется (темп −2,7) — против фазы роста"})

    def test_sector_warnings_extract_text_not_dict_repr(self):
        # sector_warnings are dicts; the mapper str()'d the whole dict, leaking
        # «{'sector': 'Technology', ...}» into the UI.  Extract the text field.
        from premium_payload import build_design_data
        p = {"sector_warnings": [
                {"sector": "Technology", "weight_pct": 48.8, "cap_pct": 40.0,
                 "overage_pp": 8.8, "text": "Technology: 49% портфеля — превышен лимит 40%"}]}
        sw = build_design_data(p, "deep")["sectorWarn"]
        self.assertEqual(sw[0], "Technology: 49% портфеля — превышен лимит 40%")
        self.assertNotIn("{", sw[0])

    def test_action_plan_joins_score_and_hotspot(self):
        # score + hotspot are NOT on action_plan rows — they live in
        # score_breakdown (total) and assets (hotspot).  The mapper read
        # non-existent score_total/hotspot keys ⇒ every row showed 0 / false.
        from premium_payload import build_design_data
        p = {"assets": [{"ticker": "NVDA", "hotspot": True, "euler_extreme": True},
                        {"ticker": "AAPL", "hotspot": False}],
             "score_breakdown": [{"ticker": "NVDA", "total": "+1.5"},
                                 {"ticker": "AAPL", "total": "-0.8"}],
             "action_plan": [{"ticker": "NVDA", "action": "Trim", "price": "875.30"},
                             {"ticker": "AAPL", "action": "Hold", "price": "230.10"}]}
        plan = {x["t"]: x for x in build_design_data(p, "deep")["actionPlan"]}
        self.assertEqual(plan["NVDA"]["score"], 1.5)     # was 0
        self.assertTrue(plan["NVDA"]["hot"])             # was False
        self.assertEqual(plan["AAPL"]["score"], -0.8)
        self.assertFalse(plan["AAPL"]["hot"])

    def test_base_top_hotspot_derived_from_max_risk_asset(self):
        # `hotspots` in the payload is a list of STRINGS — the old mapper indexed
        # it as a dict ⇒ the featured card was all '–'/0.  Derive from assets.
        from premium_payload import build_design_data
        p = {"assets": [
                {"ticker": "NVDA", "asset_class": "Технологии", "weight_pct_num": 15.0,
                 "euler_risk_pct": 32.2, "action": "Trim", "pnl_pct": "+24.1%", "pnl_abs": "+1200"},
                {"ticker": "AAPL", "asset_class": "Технологии", "weight_pct_num": 11.0,
                 "euler_risk_pct": 8.8, "action": "Hold"},
                {"ticker": "USD", "is_cash": True, "euler_risk_pct": 0.0}],
             "score_breakdown": [{"ticker": "NVDA", "action": "Trim"}]}
        th = build_design_data(p, "base")["topHotspot"]
        self.assertEqual(th["ticker"], "NVDA")           # was '–'
        self.assertEqual(th["riskShare"], 32.2)          # was 0
        self.assertEqual(th["weight"], 15.0)
        self.assertEqual(th["signal"], "TRIM")
        self.assertNotEqual(th["note"], "–")

    def test_leverage_phrasing_hidden_when_not_leveraged(self):
        # Rule: no leverage/debt term in the report when cash balance ≥ 0.
        from finance.data_lineage import build_lineage
        ai = {"verdict": "x", "bullets": ["y"]}
        rows = build_lineage({"leverage_metrics": {"is_leveraged": False}}, ai)
        blob = " ".join(str(r.get("method", "")) + str(r.get("note", "")) for r in rows)
        self.assertNotIn("плеч", blob)
        self.assertNotIn("leverage", blob)
        # …and present when the book IS margin-funded.
        rows_lev = build_lineage({"leverage_metrics": {"is_leveraged": True}}, ai)
        blob_lev = " ".join(str(r.get("method", "")) for r in rows_lev)
        self.assertIn("плеч", blob_lev)


class DeployConfigPinsPremiumFlagTest(unittest.TestCase):
    """ROOT-CAUSE regression guard (2026-06-27).

    `gcloud run deploy --set-env-vars` REPLACES the entire env-var set, so a
    PREMIUM_REPORT_ENABLED value set by hand in the console was wiped on the
    next CI deploy and production silently reverted to the v3 report.  The flag
    MUST be pinned in cloudbuild.yaml's --set-env-vars so it survives redeploys.
    """

    def test_cloudbuild_pins_premium_flag_true(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        cb_path = root / "cloudbuild.yaml"
        # The CI deploy-gate runs `unittest discover` INSIDE the built image,
        # whose Dockerfile copies only src/ + SYSTEM_PROMPT.md + tests/ — NOT the
        # deploy config.  So this repo-level invariant is unverifiable there:
        # skip (don't ERROR — that would fail the very build this guard exists to
        # protect).  It still runs against a repo checkout (local dev + the
        # pre-merge `python -m pytest tests/` gate in python-ci.yml).
        if not cb_path.exists():
            self.skipTest("cloudbuild.yaml not shipped in the deploy image (repo-only guard)")
        cb = cb_path.read_text(encoding="utf-8")
        # The env-var must live on a --set-env-vars line (so it persists), not
        # merely appear somewhere in a comment.
        set_env_lines = [ln for ln in cb.splitlines()
                         if "--set-env-vars" in ln and "PREMIUM_REPORT_ENABLED" in ln]
        self.assertTrue(set_env_lines,
                        "cloudbuild.yaml --set-env-vars must pin PREMIUM_REPORT_ENABLED "
                        "(else every deploy wipes the manually-set flag → v3 fallback).")
        self.assertIn("PREMIUM_REPORT_ENABLED=true", set_env_lines[0])


if __name__ == "__main__":
    unittest.main()
