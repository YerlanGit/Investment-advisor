"""
Phase 26 regression tests — post-release fixes for the broken 2026-07-10 live
report (deep.html generated right after the Sprint-1 merge):

  F-14  Forward Expected_Return is β·μ ONLY (Ridge alpha excluded), per-asset
        clamp [−50%, +100%], and the portfolio-level forward panel
        (Expected_Return_Annual / Expected_Sharpe) is GATED on a regression
        window of ≥ 252 trading days.  Root cause of the live cover showing
        «ожид. дох. 218.4% · Sharpe 11.29»: after F-6 removed the bfill
        look-ahead the common window honestly shrank to the youngest listing,
        and a leveraged ETF's short bull streak annualised geometrically.

  F-15  Portfolio log-return series uses per-day MASKED renormalised weights
        (composite-backfill convention) instead of row-level dropna — one
        young listing no longer truncates five years of portfolio history,
        so the 12М/6М period rows stay computable.

  F-16  premium_payload._map_performance SKIPS data-less period rows
        (portfolio_num=None / «—») instead of coercing them to «+0.0%».

  F-17  tg_bot._clean_rag_excerpt strips leading bank-letterhead remnants
        («Morgan Putting…»), re-joins PDF hyphenation («mod- estly»), cuts
        chart-axis number runs («12% 10% 8% 6% 4% 2%»), and rejects
        non-prose (mostly-numeric) chunks.

All tests are deterministic, synthetic and offline.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("KZ_RFR_ANNUAL", "0.14")
os.environ["CDS_DISABLED"] = "1"


# ═══════════════════════ F-15: masked composite series ══════════════════════

class MaskedCompositeSeriesTest(unittest.TestCase):
    """build_portfolio_log_returns must span the UNION of constituents'
    histories (per-day renormalised weights), not their intersection."""

    N_DAYS = 500
    N_YOUNG = 120

    def _panel(self) -> pd.DataFrame:
        idx = pd.date_range("2024-01-02", periods=self.N_DAYS, freq="B")
        rng = np.random.default_rng(11)
        old = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.010, self.N_DAYS)))
        young = np.full(self.N_DAYS, np.nan)
        young[-self.N_YOUNG:] = 50.0 * np.exp(
            np.cumsum(rng.normal(0.0010, 0.020, self.N_YOUNG)))
        return pd.DataFrame({"OLD.US": old, "YOUNG.US": young}, index=idx)

    def test_series_spans_full_history_not_intersection(self) -> None:
        from finance.period_returns import build_portfolio_log_returns

        port_log, info = build_portfolio_log_returns(
            self._panel(), {"OLD.US": 0.6, "YOUNG.US": 0.4})
        self.assertIsNotNone(port_log)
        # Old row-dropna behaviour capped the series at N_YOUNG−1 = 119 days.
        self.assertGreaterEqual(len(port_log), self.N_DAYS - 20,
                                "series must cover the elder asset's history")
        self.assertEqual(sorted(info["kept"]), ["OLD.US", "YOUNG.US"])
        self.assertAlmostEqual(info["covered_weight"], 1.0, places=6)

    def test_pre_listing_days_use_renormalised_elder_weight(self) -> None:
        from finance.period_returns import build_portfolio_log_returns

        panel = self._panel()
        port_log, _ = build_portfolio_log_returns(
            panel, {"OLD.US": 0.6, "YOUNG.US": 0.4})
        old_log = np.log(panel["OLD.US"] / panel["OLD.US"].shift(1))
        # Before the young listing exists, coverage = 0.6 ≥ 0.5 → the day is
        # kept and the composite must equal the elder's OWN return exactly
        # (0.6·r / 0.6 = r) — renormalisation, not weight-dilution.
        probe = panel.index[10]
        self.assertIn(probe, port_log.index)
        self.assertAlmostEqual(float(port_log.loc[probe]),
                               float(old_log.loc[probe]), places=12)

    def test_low_coverage_day_is_masked(self) -> None:
        from finance.period_returns import build_portfolio_log_returns

        panel = self._panel()
        # Punch a hole in the DOMINANT name (w=0.9) inside the overlap era:
        # on the gap days only the 0.1-weight name trades → coverage 0.1 < 0.5.
        hole = panel.index[-30]
        panel.loc[hole, "OLD.US"] = np.nan
        port_log, _ = build_portfolio_log_returns(
            panel, {"OLD.US": 0.9, "YOUNG.US": 0.1})
        self.assertIsNotNone(port_log)
        # A NaN price kills the log-return on the hole day AND the next one.
        self.assertNotIn(hole, port_log.index)
        self.assertNotIn(panel.index[-29], port_log.index)

    def test_12m_period_row_stays_computable(self) -> None:
        """The live bug: 12М/6М rows nulled because the series was truncated
        to the young listing.  With the masked series they must compute."""
        from finance.period_returns import (build_portfolio_log_returns,
                                            compute_period_returns_table)

        panel = self._panel()
        port_log, _ = build_portfolio_log_returns(
            panel, {"OLD.US": 0.6, "YOUNG.US": 0.4})
        bm_log = np.log(panel["OLD.US"] / panel["OLD.US"].shift(1)).dropna()
        table = compute_period_returns_table(port_log, {"S&P 500": bm_log})
        rows = {r["period"]: r for r in table["S&P 500"]["periods"]}
        for period in ("6m", "12m"):
            self.assertIsNotNone(rows[period]["port_pct"],
                                 f"{period} must be computable on 500 days")
            self.assertIsNotNone(rows[period]["bm_pct"])


# ═══════════════ F-14: forward expected-return gate + clamp ═════════════════

class ForwardReturnGateTest(unittest.TestCase):
    """analyze_all end-to-end on synthetic data: the annualised forward panel
    must be ABSENT on a sub-year regression window and sane on a full one."""

    N_DAYS = 500

    def _run(self, asset_days: int, drift: float, seed: int = 3) -> dict:
        from finance.investment_logic import UniversalPortfolioManager

        upm = UniversalPortfolioManager()
        idx = pd.date_range("2023-06-01", periods=self.N_DAYS, freq="B")
        cols = list(dict.fromkeys(
            list(upm.engine.factor_tickers.values()) + upm.engine.BENCHMARK_EXTRA))
        prices = {}
        for i, c in enumerate(cols):
            r = np.random.default_rng(100 + i).normal(0.0004, 0.010, self.N_DAYS)
            prices[c] = 100.0 * np.exp(np.cumsum(r))
        # The tested asset: idiosyncratic (independent noise) with `drift`,
        # listed only for the last `asset_days` — mirrors the live CONL case.
        asset = np.full(self.N_DAYS, np.nan)
        asset[-asset_days:] = 20.0 * np.exp(np.cumsum(
            np.random.default_rng(seed).normal(drift, 0.020, asset_days)))
        prices["HOT.US"] = asset
        df_prices = pd.DataFrame(prices, index=idx)

        class _FakeHistory:
            data = df_prices
            ohlc_data: dict = {}
            loaded = list(df_prices.columns)
            failed: list = []
            retried: list = []

        upm.engine.get_market_data = (        # type: ignore[assignment]
            lambda tickers, period_days=1825: (df_prices, _FakeHistory()))
        portfolio = pd.DataFrame([{
            "Ticker": "HOT.US", "Quantity": 10, "Purchase_Price": 20.0,
        }])
        with mock.patch("finance.sec_edgar.batch_fundamental_scan",
                        return_value=pd.DataFrame()), \
             mock.patch("services.macro_data.MacroFeed.get_regime_drivers",
                        return_value={}):
            return upm.analyze_all(source=portfolio)

    def test_sub_year_window_gates_forward_panel(self) -> None:
        report = self._run(asset_days=120, drift=0.015)   # hot young listing
        pm = report["portfolio_metrics"]
        window = int(pm.get("expected_return_window_days", 0))
        self.assertGreater(window, 0, "window telemetry must be recorded")
        self.assertLess(window, 252, "common window must shrink to the listing")
        # The live-bug surface: NO annualised forward claim off a short window.
        self.assertNotIn("Expected_Return_Annual", pm)
        self.assertNotIn("Expected_Sharpe", pm)
        # Per-asset forward column stays, but clamped to the publishable band.
        perf = report["performance_table"]
        if "Expected_Return" in perf.columns:
            er = pd.to_numeric(perf["Expected_Return"], errors="coerce").dropna()
            if not er.empty:
                self.assertLessEqual(float(er.max()), 1.0 + 1e-9)
                self.assertGreaterEqual(float(er.min()), -0.5 - 1e-9)

    def test_full_window_emits_sane_panel_without_alpha(self) -> None:
        # ~0.003/day realised drift ≈ +113%/yr, but INDEPENDENT of the factors
        # → β·μ (alpha excluded) must stay far below the realised trajectory.
        report = self._run(asset_days=self.N_DAYS, drift=0.003)
        pm = report["portfolio_metrics"]
        self.assertGreaterEqual(int(pm.get("expected_return_window_days", 0)), 252)
        self.assertIn("Expected_Return_Annual", pm)
        er_ann = float(pm["Expected_Return_Annual"])
        self.assertGreaterEqual(er_ann, -0.5)
        self.assertLessEqual(er_ann, 1.0 + 0.15)   # + cash·rfr headroom
        # Alpha exclusion: with near-zero factor loadings the forward figure
        # cannot inherit the +113% idiosyncratic drift (the old alpha-in
        # formula produced ≥ +100% here — the «218.4%» failure mode).
        perf = report["performance_table"].set_index("Ticker")
        asset_er = float(perf.loc["HOT.US", "Expected_Return"])
        self.assertLess(asset_er, 0.5,
                        "forward E[r] must not absorb idiosyncratic drift")
        # The realised drift stays visible through the specific-alpha channel.
        self.assertIn("Alpha_Specific", perf.columns)


# ═══════════════ F-16: data-less period rows are skipped ════════════════════

class PerformanceMapperTest(unittest.TestCase):
    """premium_payload._map_performance must not fabricate «+0.0%» rows."""

    @staticmethod
    def _payload(rows: list) -> dict:
        return {"period_returns_table": {"S&P 500": {"periods": rows}},
                "volatility_num": 12.3}

    def test_none_rows_skipped(self) -> None:
        from premium_payload import _map_performance

        rows = [
            {"label": "1 мес", "portfolio": "+2.1%", "portfolio_num": 2.1,
             "benchmark": "+1.0%", "benchmark_num": 1.0,
             "excess": "+1.1 пп", "excess_num": 1.1},
            {"label": "3 мес", "portfolio": "+4.0%", "portfolio_num": 4.0,
             "benchmark": "+3.0%", "benchmark_num": 3.0,
             "excess": "+1.0 пп", "excess_num": 1.0},
            {"label": "6 мес", "portfolio": "—", "portfolio_num": None,
             "benchmark": "—", "benchmark_num": None,
             "excess": "—", "excess_num": None},
            {"label": "12 мес", "portfolio": "—", "portfolio_num": None,
             "benchmark": "—", "benchmark_num": None,
             "excess": "—", "excess_num": None},
        ]
        out = _map_performance(self._payload(rows))
        labels = [p["label"] for p in out["periods"]]
        self.assertEqual(labels, ["1 мес", "3 мес"])
        # No phantom zeros anywhere in the kept rows.
        self.assertTrue(all(p["p"] != 0.0 for p in out["periods"]))

    def test_summary_falls_back_to_longest_available(self) -> None:
        from premium_payload import _map_performance

        rows = [
            {"label": "1 мес", "portfolio": "+2.1%", "portfolio_num": 2.1,
             "benchmark": "+1.0%", "benchmark_num": 1.0,
             "excess": "+1.1 пп", "excess_num": 1.1},
            {"label": "12 мес", "portfolio": "—", "portfolio_num": None,
             "benchmark": "—", "benchmark_num": None,
             "excess": "—", "excess_num": None},
        ]
        out = _map_performance(self._payload(rows))
        # 12М is gone → the summary must read the longest REAL window (1 мес),
        # not a fabricated 0.0.
        self.assertEqual(out["summary"]["ret"], 2.1)
        self.assertEqual(out["summary"]["spx"], 1.0)

    def test_legacy_string_rows_survive(self) -> None:
        from premium_payload import _map_performance

        rows = [{"label": "3 мес", "portfolio": "+3.0%",
                 "benchmark": "+2.0%", "excess": "+1.0 пп"}]
        out = _map_performance(self._payload(rows))
        self.assertEqual(len(out["periods"]), 1)
        self.assertAlmostEqual(out["periods"][0]["p"], 3.0, places=6)


# ═══════════════ F-17: RAG excerpt cleaning ═════════════════════════════════

class RagExcerptCleanTest(unittest.TestCase):
    """_clean_rag_excerpt on the EXACT defect strings from the live report."""

    def _clean(self, text: str, **kw) -> str:
        os.environ.setdefault("RAMP_BOT_TOKEN", "dummy:token")
        try:
            from tg_bot import _clean_rag_excerpt
        except BaseException as exc:   # pyo3 PanicException is a BaseException
            self.skipTest(f"tg_bot import failed in this env: {exc!r}")
        return _clean_rag_excerpt(text, **kw)

    def test_bank_letterhead_remnant_and_hyphenation(self) -> None:
        raw = ("Morgan Putting these pieces together, we continue to forecast "
               "mod- estly higher yields over the balance of 2026 despite the "
               "recent rally.")
        out = self._clean(raw)
        self.assertTrue(out.startswith("Putting"),
                        f"letterhead remnant must be stripped, got: {out!r}")
        self.assertIn("modestly", out)
        self.assertNotIn("mod- estly", out)

    def test_chart_axis_run_is_cut(self) -> None:
        raw = ("We expect headline inflation to decline through 2026. "
               "Headline Inflation % change y/y 12% 10% 8% 6% 4% 2% "
               "Central Bank rates GS Policy Rate Forecasts")
        out = self._clean(raw)
        self.assertIn("decline through 2026", out)
        self.assertNotIn("12%", out)
        self.assertNotIn("Central Bank rates", out,
                         "legend soup after the axis run must be dropped")

    def test_pure_axis_soup_rejected(self) -> None:
        self.assertEqual(self._clean("12% 10% 8% 6% 4% 2% 0% -2%"), "")

    def test_mostly_numeric_table_rejected(self) -> None:
        self.assertEqual(
            self._clean("Q1 2.3 4.5 Q2 3.1 5.2 Q3 1.8 2.2 revenue"), "")

    def test_normal_prose_with_bank_subject_untouched(self) -> None:
        # «Morgan Stanley expects…» — the bank is the grammatical SUBJECT
        # (lowercase verb follows), not a letterhead remnant → keep it.
        raw = "Morgan Stanley expects the S&P 500 to reach 6,500 by mid-2027."
        out = self._clean(raw)
        self.assertTrue(out.startswith("Morgan Stanley expects"), repr(out))


if __name__ == "__main__":
    unittest.main()
