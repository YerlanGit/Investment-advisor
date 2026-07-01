"""
Recommendation follow-ups (2026-07-01) — regression tests.

  • (а) 4-Pillar #3: bounded fundamental-momentum bonus in the F-pillar
        (margins improving YoY) — additive, None → byte-identical.
  • (б) Regime macro overlay: LEVEL ⊕ rate-of-change nudges from GDP /
        unemployment, bounded, env-gated, multi-point trend (≥3 changes).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FPillarMomentumTest(unittest.TestCase):
    """(а) — the momentum bonus is bounded, signed, and opt-in."""

    def _base(self, **kw):
        from finance.scoring import fundamentals_score
        args = dict(roe_z=0.0, op_margin_z=0.0, debt_to_assets_z=0.0,
                    revenue_growth_z=0.0, fcf_margin_z=0.0, macro_alignment=0.0)
        args.update(kw)
        return fundamentals_score(**args)

    def test_none_momentum_is_byte_identical(self):
        # Default (no momentum) must match omitting the argument entirely.
        self.assertEqual(self._base(), self._base(momentum=None))

    def test_positive_momentum_raises_score(self):
        base = self._base()
        self.assertAlmostEqual(self._base(momentum=1.0), base + 0.5, places=6)
        self.assertAlmostEqual(self._base(momentum=-1.0), base - 0.5, places=6)

    def test_momentum_is_bounded_half_point(self):
        # Even an out-of-range momentum can move F by at most ±0.5.
        self.assertAlmostEqual(self._base(momentum=9.0), self._base() + 0.5, places=6)
        self.assertAlmostEqual(self._base(momentum=-9.0), self._base() - 0.5, places=6)

    def test_nan_momentum_does_not_poison(self):
        self.assertEqual(self._base(momentum=float("nan")), self._base())

    def test_total_still_clipped(self):
        # Strong everything + momentum must not exceed the +2 pillar cap.
        s = self._base(roe_z=2, op_margin_z=2, revenue_growth_z=2,
                       fcf_margin_z=2, momentum=1.0)
        self.assertLessEqual(s, 2.0)


class RegimeMacroOverlayTest(unittest.TestCase):
    """(б) — the FRED overlay nudges the axes with level ⊕ rate-of-change."""

    def _pack(self, gdp_hist, unemp_hist):
        return {
            "gdp_growth":  {"value": gdp_hist[-1], "status": "ok",
                            "history_30d": [{"value": v} for v in gdp_hist]},
            "unemployment": {"value": unemp_hist[-1], "status": "ok",
                             "history_30d": [{"value": v} for v in unemp_hist]},
        }

    def test_series_trend_needs_three_changes(self):
        from finance.regime import series_trend
        # <4 points → no trend (needs ≥3 consecutive changes).
        self.assertIsNone(series_trend([1.0, 2.0], lag=3)[0])
        # A sustained rise → positive total change.
        total, slope, n = series_trend([1.0, 2.0, 3.0, 4.0], lag=3)
        self.assertIsNotNone(total)
        self.assertGreater(total, 0.0)

    def test_bullish_macro_tilts_growth_and_cycle_up(self):
        from finance.regime import _macro_nudges, _MACRO_MAX_NUDGE
        # Above-trend accelerating GDP + below-neutral falling unemployment.
        g, c, diag = _macro_nudges(self._pack(
            gdp_hist=[2.0, 2.5, 3.0, 3.5], unemp_hist=[4.2, 4.0, 3.9, 3.8]))
        self.assertIsNotNone(g); self.assertIsNotNone(c)
        self.assertGreater(g, 0.0)          # strong growth → +growth axis
        self.assertGreater(c, 0.0)          # tight labour → +cycle axis
        self.assertLessEqual(abs(g), _MACRO_MAX_NUDGE + 1e-9)   # bounded
        self.assertLessEqual(abs(c), _MACRO_MAX_NUDGE + 1e-9)
        self.assertIn("macro_gdp_trend", diag)

    def test_rising_unemployment_pulls_cycle_down(self):
        from finance.regime import _macro_nudges
        # Same level, but a RISING unemployment trend must lower the cycle nudge
        # vs a falling one (rate-of-change matters, not just the spot value).
        _, c_falling, _ = _macro_nudges(self._pack([2.0]*4, [4.3, 4.1, 4.0, 3.9]))
        _, c_rising,  _ = _macro_nudges(self._pack([2.0]*4, [3.5, 3.7, 3.8, 3.9]))
        self.assertGreater(c_falling, c_rising)

    def test_overlay_env_gate(self):
        from finance.regime import _macro_overlay_enabled
        prev = os.environ.get("REGIME_MACRO_OVERLAY")
        try:
            os.environ["REGIME_MACRO_OVERLAY"] = "0"
            self.assertFalse(_macro_overlay_enabled())
            os.environ["REGIME_MACRO_OVERLAY"] = "on"
            self.assertTrue(_macro_overlay_enabled())
            os.environ.pop("REGIME_MACRO_OVERLAY", None)
            self.assertTrue(_macro_overlay_enabled())    # DEFAULT-ON
        finally:
            if prev is None:
                os.environ.pop("REGIME_MACRO_OVERLAY", None)
            else:
                os.environ["REGIME_MACRO_OVERLAY"] = prev


class RagChunkingTest(unittest.TestCase):
    """(в) — section-aware chunking, bank + ticker extraction."""

    def test_section_aware_size_bounded_chunks(self):
        from agent.rag_engine import FinancialRAG
        md = ("# Outlook\n" + "A" * 50 + "\n"                 # too short → dropped
              "## Equities\n" + ("word " * 600) + "\n"        # long → sub-split
              "## Rates\n" + ("bond " * 60))                  # normal
        chunks = FinancialRAG._chunk_markdown(md, max_chars=1200, overlap=150)
        self.assertTrue(chunks)
        headings = {h for h, _ in chunks}
        self.assertIn("Equities", headings)
        self.assertIn("Rates", headings)
        # the long Equities section must have produced >1 bounded piece
        eq_pieces = [c for h, c in chunks if h == "Equities"]
        self.assertGreater(len(eq_pieces), 1)
        self.assertTrue(all(len(c) <= 1200 for _, c in chunks))

    def test_bank_detection(self):
        from agent.rag_engine import FinancialRAG
        self.assertEqual(FinancialRAG._extract_bank("goldman_Q3_2026.pdf", ""), "Goldman Sachs")
        self.assertEqual(FinancialRAG._extract_bank("x.pdf", "J.P. Morgan Research"), "JPMorgan")
        self.assertEqual(FinancialRAG._extract_bank("note.pdf", "no issuer here"), "Unknown")

    def test_ticker_extraction_wrapped_and_filtered(self):
        from agent.rag_engine import FinancialRAG
        tk = FinancialRAG._extract_tickers("We upgrade $NVDA and AAPL; the FED and GDP are noise.")
        self.assertIn(",NVDA,", tk)
        self.assertIn(",AAPL,", tk)
        self.assertNotIn(",FED,", tk)   # stop-word
        self.assertNotIn(",GDP,", tk)


if __name__ == "__main__":
    unittest.main()
