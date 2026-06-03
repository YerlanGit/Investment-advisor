"""
Phase 6 — FRED-backed FX feed (USD↔KZT via DEXKZUS).

Network-free: every test passes a mock `http_get` callable so no real
FRED call is ever made.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _MockResp:
    """Minimal stand-in for requests.Response."""
    def __init__(self, status: int, body: dict, headers: dict | None = None):
        self.status_code = int(status)
        self._body = body
        self.headers = headers or {}

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")


def _fred_payload(rows: list[tuple[str, float]]) -> dict:
    """Build a fake FRED 'observations' body."""
    return {"observations": [{"date": d, "value": str(v)} for d, v in rows]}


# ── Default factory ────────────────────────────────────────────────────────

class DefaultFxProviderFactoryTest(unittest.TestCase):

    def setUp(self) -> None:
        self._snapshot = os.environ.get("FRED_API_KEY")
        os.environ.pop("FRED_API_KEY", None)

    def tearDown(self) -> None:
        if self._snapshot is None:
            os.environ.pop("FRED_API_KEY", None)
        else:
            os.environ["FRED_API_KEY"] = self._snapshot

    def test_no_api_key_returns_none(self) -> None:
        from services.fx_feed import default_fx_provider
        self.assertIsNone(default_fx_provider())

    def test_with_api_key_returns_callable(self) -> None:
        os.environ["FRED_API_KEY"] = "dummy-key"
        from services.fx_feed import default_fx_provider, FredFxProvider
        prov = default_fx_provider()
        self.assertIsInstance(prov, FredFxProvider)


# ── Pair routing / inversion ───────────────────────────────────────────────

class FxProviderPairRoutingTest(unittest.TestCase):

    def _provider(self, rows, *, fail_status: int | None = None,
                  tmp_cache: Path | None = None):
        from services.fx_feed import FredFxProvider
        if fail_status is not None:
            def fake_get(url, params=None, timeout=None):
                return _MockResp(fail_status, {})
        else:
            def fake_get(url, params=None, timeout=None):
                return _MockResp(200, _fred_payload(rows))
        return FredFxProvider(api_key="k", days=30,
                              cache_dir=tmp_cache,
                              http_get=fake_get,
                              sleep=lambda _: None)

    def test_usd_to_kzt_passes_through(self) -> None:
        rows = [("2026-01-01", 500.0),
                ("2026-01-02", 510.0),
                ("2026-01-03", 520.0)]
        with tempfile.TemporaryDirectory() as tmp:
            prov = self._provider(rows, tmp_cache=Path(tmp))
            s = prov("USD", "KZT")
        self.assertIsNotNone(s)
        self.assertEqual(len(s), 3)
        self.assertEqual(s.iloc[-1], 520.0)
        # Sorted chronologically.
        self.assertTrue(s.index.is_monotonic_increasing)

    def test_kzt_to_usd_inverts(self) -> None:
        rows = [("2026-01-01", 500.0),
                ("2026-01-02", 500.0),
                ("2026-01-03", 520.0)]
        with tempfile.TemporaryDirectory() as tmp:
            prov = self._provider(rows, tmp_cache=Path(tmp))
            s = prov("KZT", "USD")
        # 1/500 = 0.002; 1/520 ≈ 0.001923.
        self.assertAlmostEqual(s.iloc[0], 1.0 / 500.0, places=10)
        self.assertAlmostEqual(s.iloc[-1], 1.0 / 520.0, places=10)

    def test_same_currency_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prov = self._provider([], tmp_cache=Path(tmp))
            self.assertIsNone(prov("USD", "USD"))

    def test_unsupported_pair_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prov = self._provider([], tmp_cache=Path(tmp))
            self.assertIsNone(prov("EUR", "JPY"))

    def test_memoised_per_pair(self) -> None:
        rows = [("2026-01-01", 500.0)]
        call_count = [0]
        def fake_get(url, params=None, timeout=None):
            call_count[0] += 1
            return _MockResp(200, _fred_payload(rows))
        from services.fx_feed import FredFxProvider
        with tempfile.TemporaryDirectory() as tmp:
            prov = FredFxProvider(api_key="k", days=30,
                                  cache_dir=Path(tmp),
                                  http_get=fake_get, sleep=lambda _: None)
            s1 = prov("USD", "KZT")
            s2 = prov("USD", "KZT")
        # Second call must hit the in-memory memo, not FRED again.
        self.assertEqual(call_count[0], 1)
        self.assertIs(s1, s2)


# ── Retry / cache resilience ───────────────────────────────────────────────

class FxProviderRetryAndCacheTest(unittest.TestCase):

    def test_retries_on_429_then_succeeds(self) -> None:
        rows = [("2026-01-01", 500.0)]
        attempts = [0]
        def fake_get(url, params=None, timeout=None):
            attempts[0] += 1
            if attempts[0] < 2:
                return _MockResp(429, {})
            return _MockResp(200, _fred_payload(rows))
        from services.fx_feed import FredFxProvider
        with tempfile.TemporaryDirectory() as tmp:
            prov = FredFxProvider(api_key="k", days=30,
                                  cache_dir=Path(tmp),
                                  http_get=fake_get, sleep=lambda _: None)
            s = prov("USD", "KZT")
        self.assertIsNotNone(s)
        self.assertEqual(attempts[0], 2)

    def test_falls_back_to_disk_cache_on_failure(self) -> None:
        """When FRED is unreachable, a previously cached payload is reused."""
        from services.fx_feed import FredFxProvider
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            # Seed disk cache by hand
            (tmp_p / "DEXKZUS.json").write_text(json.dumps([
                {"date": "2025-12-31", "value": 499.5},
                {"date": "2026-01-02", "value": 510.0},
            ]))
            def fake_get(url, params=None, timeout=None):
                return _MockResp(500, {})
            prov = FredFxProvider(api_key="k", days=30,
                                  cache_dir=tmp_p,
                                  http_get=fake_get, sleep=lambda _: None)
            s = prov("USD", "KZT")
        self.assertIsNotNone(s)
        self.assertEqual(len(s), 2)
        self.assertEqual(s.iloc[-1], 510.0)

    def test_no_cache_and_failure_returns_none(self) -> None:
        from services.fx_feed import FredFxProvider
        def fake_get(url, params=None, timeout=None):
            return _MockResp(500, {})
        with tempfile.TemporaryDirectory() as tmp:
            prov = FredFxProvider(api_key="k", days=30,
                                  cache_dir=Path(tmp),
                                  http_get=fake_get, sleep=lambda _: None)
            self.assertIsNone(prov("USD", "KZT"))


# ── End-to-end: provider feeds into convert_price_matrix ───────────────────

class FxProviderIntegrationTest(unittest.TestCase):
    """Provider plugs into convert_price_matrix without adapter glue."""

    def test_provider_drives_kzt_reporting_conversion(self) -> None:
        from services.fx_feed import FredFxProvider
        from finance.currency import ReportingCurrency, convert_price_matrix
        rows = [(d.strftime("%Y-%m-%d"), 500.0 + i)
                for i, d in enumerate(pd.date_range("2026-01-01", periods=5, freq="D"))]
        def fake_get(url, params=None, timeout=None):
            return _MockResp(200, _fred_payload(rows))
        with tempfile.TemporaryDirectory() as tmp:
            prov = FredFxProvider(api_key="k", days=30,
                                  cache_dir=Path(tmp),
                                  http_get=fake_get, sleep=lambda _: None)
            idx = pd.date_range("2026-01-01", periods=5, freq="D")
            prices = pd.DataFrame({"AAPL.US": [100.0, 101.0, 102.0, 103.0, 104.0]},
                                  index=idx)
            result = convert_price_matrix(
                prices, {"AAPL.US": "USD"},
                reporting=ReportingCurrency.KZT,
                fx_provider=prov, lag_one_day=True,
            )
        self.assertFalse(result.no_op)
        self.assertEqual(len(result.fx_records), 1)
        self.assertEqual(result.fx_records[0].pair, "USDKZT")
        # T-1 lag → day 0 backfills to first known fx=500;
        # day 1 → fx[0]=500; ... day 4 → fx[3]=503.
        np.testing.assert_allclose(
            result.prices_base["AAPL.US"].values,
            np.array([100*500, 101*500, 102*501, 103*502, 104*503]),
        )


if __name__ == "__main__":
    unittest.main()
