"""
Phase 3 (Phase-15 file) — H1 billing / H3 port_log_returns / H4 risk mandate
/ M1 stress validator / M2 asset taxonomy.
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


# ── H4: risk-mandate composite score ────────────────────────────────────────

class RiskMandateTest(unittest.TestCase):

    def test_weights_sum_to_one(self) -> None:
        from finance.scoring import _RISK_MANDATE_MATRIX
        for name, m in _RISK_MANDATE_MATRIX.items():
            self.assertAlmostEqual(m["w_cvar"] + m["w_vol"] + m["w_erc"], 1.0,
                                   places=9, msg=name)

    def test_conservative_higher_than_aggressive(self) -> None:
        from finance.scoring import composite_risk_score as c
        a = c(0.20, -0.04, 20.0, mandate="CONSERVATIVE")
        b = c(0.20, -0.04, 20.0, mandate="AGGRESSIVE")
        self.assertGreater(a, b)

    def test_unknown_mandate_falls_back_moderate(self) -> None:
        from finance.scoring import composite_risk_score as c
        self.assertEqual(c(0.2, -0.04, 20.0, mandate="WAT"),
                         c(0.2, -0.04, 20.0, mandate="MODERATE"))

    def test_normalize_mandate_mapping(self) -> None:
        from finance.scoring import normalize_risk_mandate as nm
        self.assertEqual(nm("Консервативный"), "CONSERVATIVE")
        self.assertEqual(nm("Умеренный"), "MODERATE")
        self.assertEqual(nm("Умеренно-агрессивный"), "MODERATE")
        self.assertEqual(nm("Агрессивный"), "AGGRESSIVE")
        self.assertEqual(nm(7), "CONSERVATIVE")
        self.assertEqual(nm(17), "AGGRESSIVE")
        self.assertEqual(nm(None), "MODERATE")

    def test_engine_threads_mandate(self) -> None:
        from finance.investment_logic import MAC3RiskEngine
        eng = MAC3RiskEngine(risk_mandate="Агрессивный")
        self.assertEqual(eng.risk_mandate, "AGGRESSIVE")


# ── M1: stress-comment post-validator ───────────────────────────────────────

class StressValidatorTest(unittest.TestCase):

    def test_appends_note_when_missing(self) -> None:
        from ai_narrative import validate_stress_comment
        out = validate_stress_comment("Портфель просядет на 25% в стрессе.")
        self.assertIn("ограничител", out.lower())

    def test_keeps_text_when_cap_mentioned(self) -> None:
        from ai_narrative import validate_stress_comment
        txt = "AVGO −31% после ограничителя."
        self.assertEqual(validate_stress_comment(txt), txt)

    def test_empty_passthrough(self) -> None:
        from ai_narrative import validate_stress_comment
        self.assertEqual(validate_stress_comment(""), "")


# ── M2: asset taxonomy (Freedom metadata SSoT) ──────────────────────────────

class AssetTaxonomyTest(unittest.TestCase):

    def test_broker_type_authoritative(self) -> None:
        from finance.asset_taxonomy import from_freedom_metadata, AssetClass
        # t=2 → bond even if ticker looks like equity
        self.assertEqual(from_freedom_metadata(ticker="XYZ.US", t_field=2),
                         AssetClass.FIXED_INCOME)
        self.assertEqual(from_freedom_metadata(ticker="AAPL.US", t_field=1),
                         AssetClass.EQUITY)

    def test_ticker_heuristic_fallback(self) -> None:
        from finance.asset_taxonomy import from_freedom_metadata, AssetClass
        self.assertEqual(from_freedom_metadata(ticker="GLD"), AssetClass.COMMODITY)
        self.assertEqual(from_freedom_metadata(ticker="TLT"), AssetClass.FIXED_INCOME)
        self.assertEqual(from_freedom_metadata(ticker="FFSPC6.1028.AIX"),
                         AssetClass.STRUCTURED)
        self.assertEqual(from_freedom_metadata(ticker="USD"), AssetClass.CASH)
        self.assertEqual(from_freedom_metadata(ticker="BTC-USD"), AssetClass.CRYPTO)

    def test_display_label_cash_matches_template_vocab(self) -> None:
        from finance.asset_taxonomy import classify_display_from_freedom
        # Template filters on the exact string "Ден. средства".
        self.assertEqual(classify_display_from_freedom(ticker="USD"),
                         "Ден. средства")


# ── H3: port_log_returns exposed; bot consumes (no recompute) ───────────────

class PortLogReturnsTest(unittest.TestCase):

    def test_engine_exposes_series(self) -> None:
        from finance.investment_logic import MAC3RiskEngine
        eng = MAC3RiskEngine()
        idx = pd.date_range("2024-01-01", periods=300, freq="B")
        prices = {}
        for i, tkr in enumerate(eng.factor_tickers.values()):
            r = np.random.default_rng(i + 1).normal(0.0004, 0.01, len(idx))
            prices[tkr] = 100.0 * np.exp(np.cumsum(r))
        prices["AAPL.US"] = 100.0 * np.exp(np.cumsum(
            np.random.default_rng(99).normal(0.0006, 0.012, len(idx))))
        df = pd.DataFrame(prices, index=idx)
        eng.calculate_structural_risk(df, ["AAPL.US"], {"AAPL.US": 1.0})
        series = eng._last_port_log_returns
        self.assertIsNotNone(series)
        self.assertEqual(len(series), len(df) - 1)   # one lost to log-diff
        self.assertTrue(np.isfinite(series.values).all())

    def test_tg_bot_has_no_weighted_log_recompute(self) -> None:
        """H3 + Sprint-1 #3: the bot must NOT recompute the weighted portfolio
        log-returns.  The cap-weighted series math now lives in the finance layer
        (finance.portfolio_series), which consumes the engine's precomputed
        results["port_log_returns"]; the bot only delegates to it."""
        src_root = Path(__file__).resolve().parent.parent / "src"
        bot = (src_root / "tg_bot.py").read_text()
        self.assertNotIn("@ _np.array(weights)", bot)
        # Bot delegates the series math to the finance core (SoC).
        self.assertIn("compute_equity_curve_series", bot)
        # The engine-series consumption lives in the finance layer now.
        fin = (src_root / "finance" / "portfolio_series.py").read_text()
        self.assertIn('results.get("port_log_returns")', fin)


if __name__ == "__main__":
    unittest.main()
