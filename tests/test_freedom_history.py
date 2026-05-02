"""Unit tests for freedom_portfolio.history — split detection + parsing."""

from __future__ import annotations

import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from freedom_portfolio.history import (  # noqa: E402
    Candle,
    KNOWN_SPLITS,
    _apply_known_splits,
    _candles_to_series,
    _detect_and_adjust_splits,
    _parse_hloc_response,
    get_candles,
)


# ── Parser ────────────────────────────────────────────────────────────────────


def test_parse_hloc_tradernet_kz_nested_shape():
    """
    tradernet.kz docs shape (per quotes-get-hloc): xSeries + yPrices nested
    UNDER hloc[ticker].  Different from the SDK 2-D shape.
    """
    raw = {"hloc": {"FRHC.US": {
        "xSeries": [1700000000, 1700086400],
        "yPrices": [
            {"o": 100, "h": 101, "l": 99, "c": 100.5, "v": 1000},
            {"o": 100.5, "h": 102, "l": 100, "c": 101.0, "v": 1500},
        ],
    }}}
    candles = _parse_hloc_response(raw, "FRHC.US")
    assert len(candles) == 2
    assert candles[0].c == 100.5 and candles[0].v == 1000
    assert candles[1].c == 101.0


def test_parse_hloc_official_sdk_shape():
    """
    Canonical Tradernet response per official Python SDK
    (tradernet/symbols/tradernet_symbol.py:97-110): parallel arrays keyed
    by symbol, ``hloc`` is 2-D where each row is [high, low, open, close].
    """
    raw = {"result": {
        "hloc":    {"AAPL.US": [[101, 99, 100, 100.5], [102, 100, 100.5, 101.0]]},
        "xSeries": {"AAPL.US": [1700000000, 1700086400]},
        "vl":      {"AAPL.US": [1000, 1500]},
    }}
    candles = _parse_hloc_response(raw, "AAPL.US")
    assert len(candles) == 2
    assert candles[0].h == 101 and candles[0].l == 99 and candles[0].o == 100 and candles[0].c == 100.5
    assert candles[0].v == 1000
    assert candles[1].c == 101.0


def test_parse_hloc_handles_millisecond_timestamps():
    """Some Tradernet endpoints return timestamps in ms; we auto-detect."""
    raw = {
        "hloc":    {"AAPL.US": [[101, 99, 100, 100.5]]},
        "xSeries": {"AAPL.US": [1700000000000]},  # ms
    }
    candles = _parse_hloc_response(raw, "AAPL.US")
    assert len(candles) == 1
    assert candles[0].t == 1700000000  # auto-converted to seconds


def test_parse_hloc_shape_flat_list():
    raw = {"hloc": [{"t": 1700000000, "c": 100.0}]}
    candles = _parse_hloc_response(raw, "AAPL.US")
    assert len(candles) == 1
    assert candles[0].c == 100.0


def test_parse_hloc_unknown_shape_returns_empty():
    raw = {"foo": "bar"}
    assert _parse_hloc_response(raw, "AAPL.US") == []


# ── Candles → Series ─────────────────────────────────────────────────────────


def test_candles_to_series_dedups_and_sorts():
    candles = [
        Candle(t=int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()), c=101.0),
        Candle(t=int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()), c=100.0),
        Candle(t=int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()), c=100.5),  # dup
    ]
    s = _candles_to_series(candles)
    assert len(s) == 2
    assert s.is_monotonic_increasing
    assert s.iloc[0] == 100.5  # last-keep on dup


def test_candles_to_series_drops_zero_close():
    candles = [Candle(t=1700000000, c=0.0), Candle(t=1700086400, c=100.0)]
    s = _candles_to_series(candles)
    assert len(s) == 1


# ── Known-splits adjustment ──────────────────────────────────────────────────


def test_known_splits_table_contains_nvda():
    """Sanity: NVDA 2024-06-10 10-for-1 is in the table."""
    assert "NVDA.US" in KNOWN_SPLITS
    assert any(d == date(2024, 6, 10) for d, _ in KNOWN_SPLITS["NVDA.US"])


def test_apply_known_splits_back_adjusts_pre_split_prices():
    """Pre-split prices should be divided by the split ratio."""
    idx = pd.date_range("2024-06-05", "2024-06-15", freq="D")
    # Simulate raw data: prices around 1200 before split, 120 after (10:1 split on 2024-06-10).
    prices = [1200, 1210, 1205, 1208, 1210, 121, 122, 121.5, 122.3, 123, 122.8]
    s = pd.Series(prices, index=idx, dtype=float)
    out = _apply_known_splits("NVDA.US", s)
    pre_split  = out.loc[out.index <  pd.Timestamp("2024-06-10")]
    post_split = out.loc[out.index >= pd.Timestamp("2024-06-10")]
    assert pre_split.max()  < 200, "pre-split prices must be back-adjusted (÷10)"
    assert post_split.min() > 100, "post-split prices unchanged"


def test_apply_known_splits_noop_on_already_adjusted():
    """If prices have no step at the split date, do nothing (server already adjusted)."""
    idx = pd.date_range("2024-06-05", "2024-06-15", freq="D")
    # Already-adjusted data: smooth around 120 throughout.
    prices = [120, 121, 120.5, 121, 120.8, 121, 122, 121.5, 122.3, 123, 122.8]
    s = pd.Series(prices, index=idx, dtype=float)
    out = _apply_known_splits("NVDA.US", s)
    pd.testing.assert_series_equal(out, s)


def test_apply_known_splits_skip_unknown_ticker():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    s = pd.Series([100, 101, 102, 103, 104], index=idx, dtype=float)
    out = _apply_known_splits("UNKNOWN.US", s)
    pd.testing.assert_series_equal(out, s)


# ── Heuristic split detector ─────────────────────────────────────────────────


def test_detect_and_adjust_splits_catches_unknown_split():
    """A 5-for-1 split on an unknown ticker should still be detected and back-adjusted."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    # 500 → 100 jump on day 6 (5-for-1 split equivalent).
    prices = [500, 502, 498, 501, 500, 100, 100.5, 101, 100.5, 102]
    s = pd.Series(prices, index=idx, dtype=float)
    out = _detect_and_adjust_splits("ZZZ.US", s)
    assert out.iloc[0] < 200, "pre-split prices should be back-adjusted"
    assert out.iloc[-1] > 50,  "post-split prices unchanged"


def test_detect_and_adjust_splits_ignores_news_driven_30pct_drop():
    """A non-integer-ratio big drop (e.g. 30% earnings shock) should NOT be adjusted."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    # 100 → 73 (a -27% drop, which is below threshold of -33%, so heuristic ignores).
    prices = [100, 101, 99, 100, 101, 73, 72, 73, 74, 73.5]
    s = pd.Series(prices, index=idx, dtype=float)
    out = _detect_and_adjust_splits("ZZZ.US", s)
    pd.testing.assert_series_equal(out, s, check_freq=False)


def test_detect_and_adjust_splits_idempotent_on_smooth_series():
    """Already-adjusted (smooth) series stays unchanged."""
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    prices = [100 + i * 0.5 for i in range(20)]
    s = pd.Series(prices, index=idx, dtype=float)
    out = _detect_and_adjust_splits("ZZZ.US", s)
    pd.testing.assert_series_equal(out, s, check_freq=False)


# ── End-to-end via mocked TradernetClient ────────────────────────────────────


def _mock_tradernet_client(response: dict) -> MagicMock:
    """Mock TradernetClient with attributes needed by the KZ fallback logic."""
    from freedom_portfolio.client import TradernetClient

    client = MagicMock()
    client._post_v2_signed = MagicMock(return_value=response)
    client.base_url = "https://tradernet.com/api/"
    client.public_key = "test_key"
    client.secret_key = "test_secret"
    client.timeout = 30
    client._session = MagicMock()
    # Bind the real _with_cf_retry so the Cloudflare retry wrapper works.
    client._with_cf_retry = lambda fn, *a, **kw: TradernetClient._with_cf_retry(client, fn, *a, **kw)
    client._CF_MAX_RETRIES = TradernetClient._CF_MAX_RETRIES
    client._CF_BACKOFF_BASE = TradernetClient._CF_BACKOFF_BASE
    return client


def test_get_candles_returns_series_from_v2_signed(tmp_path, monkeypatch):
    """End-to-end: v2 signed transport returns SDK-format candles."""
    from freedom_portfolio import history as hist_mod
    monkeypatch.setattr(hist_mod, "_CACHE_DIR", tmp_path)

    t1 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    t2 = int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp())
    t3 = int(datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp())
    response = {"hloc":    {"AAPL.US": [[101, 99, 100, 100.0], [102, 100, 100.5, 101.0], [103, 101, 101.5, 102.0]]},
                "xSeries": {"AAPL.US": [t1, t2, t3]}}
    client = _mock_tradernet_client(response)
    s = get_candles(client, "AAPL.US", days=10)
    assert len(s) == 3
    assert s.iloc[-1] == 102.0
    assert s.iloc[0]  == 100.0


def test_get_candles_uses_cache_on_second_call(tmp_path, monkeypatch):
    """Second call within TTL should not hit the network."""
    from freedom_portfolio import history as hist_mod
    monkeypatch.setattr(hist_mod, "_CACHE_DIR", tmp_path)

    response = {"hloc": {"AAPL.US": [[101, 99, 100, 100.0]]},
                "xSeries": {"AAPL.US": [1700000000]}}
    client = _mock_tradernet_client(response)

    get_candles(client, "AAPL.US", days=10)
    first_call_count = client._post_v2_signed.call_count

    get_candles(client, "AAPL.US", days=10)
    assert client._post_v2_signed.call_count == first_call_count, "second call must hit cache"


def test_fetch_hloc_sends_correct_date_format(tmp_path, monkeypatch):
    """Dates must be sent as DD.MM.YYYY HH:MM, not ISO format."""
    from freedom_portfolio import history as hist_mod
    from freedom_portfolio.history import _fetch_hloc
    monkeypatch.setattr(hist_mod, "_CACHE_DIR", tmp_path)

    response = {"hloc": [{"t": 1700000000, "c": 100.0}]}
    client = _mock_tradernet_client(response)
    _fetch_hloc(client, "AAPL.US", days=30, timeframe="D")

    # Check the params passed to _post_v2_signed
    call_args = client._post_v2_signed.call_args
    params = call_args[0][1]  # second positional arg

    # Verify date format is DD.MM.YYYY HH:MM
    assert "date_from" in params
    assert "date_to" in params
    import re
    date_pattern = re.compile(r"^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}$")
    assert date_pattern.match(params["date_from"]), \
        f"date_from must be DD.MM.YYYY HH:MM, got: {params['date_from']}"
    assert date_pattern.match(params["date_to"]), \
        f"date_to must be DD.MM.YYYY HH:MM, got: {params['date_to']}"

    # Verify userId is present (even if None)
    assert "userId" in params

