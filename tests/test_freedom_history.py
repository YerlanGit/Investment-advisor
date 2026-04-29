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


def test_parse_hloc_shape_dict_keyed_by_ticker():
    """MasyaSmv-style: {"hloc": {"AAPL.US": [{t,o,h,l,c,v}, ...]}}."""
    raw = {"hloc": {"AAPL.US": [
        {"t": 1700000000, "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 1000},
        {"t": 1700086400, "o": 100.5, "h": 102, "l": 100, "c": 101.0, "v": 1500},
    ]}}
    candles = _parse_hloc_response(raw, "AAPL.US")
    assert len(candles) == 2
    assert candles[0].c == 100.5
    assert candles[1].c == 101.0


def test_parse_hloc_shape_flat_list():
    raw = {"hloc": [{"t": 1700000000, "c": 100.0}]}
    candles = _parse_hloc_response(raw, "AAPL.US")
    assert len(candles) == 1
    assert candles[0].c == 100.0


def test_parse_hloc_shape_xseries_yseries():
    raw = {
        "xSeries": [1700000000, 1700086400],
        "ySeries": [{"c": [100.0, 101.0], "o": [99, 100], "h": [101, 102], "l": [98, 99]}],
    }
    candles = _parse_hloc_response(raw, "AAPL.US")
    assert len(candles) == 2
    assert candles[1].c == 101.0


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
    client = MagicMock()
    client._post_v2_signed = MagicMock(return_value=response)
    return client


def test_get_candles_returns_clean_series_from_modern_response(tmp_path, monkeypatch):
    """End-to-end: raw response → parsed → cached → returned as Series."""
    # Redirect cache to a tmp dir to avoid polluting /tmp.
    from freedom_portfolio import history as hist_mod
    monkeypatch.setattr(hist_mod, "_CACHE_DIR", tmp_path)

    response = {"result": {"hloc": {"AAPL.US": [
        {"t": int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()), "c": 100.0},
        {"t": int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()), "c": 101.0},
        {"t": int(datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp()), "c": 102.0},
    ]}}}
    client = _mock_tradernet_client(response)
    s = get_candles(client, "AAPL.US", days=10)
    assert len(s) == 3
    assert s.iloc[-1] == 102.0
    assert s.iloc[0]  == 100.0


def test_get_candles_uses_cache_on_second_call(tmp_path, monkeypatch):
    """Second call within TTL should not hit the network."""
    from freedom_portfolio import history as hist_mod
    monkeypatch.setattr(hist_mod, "_CACHE_DIR", tmp_path)

    response = {"hloc": [{"t": 1700000000, "c": 100.0}]}
    client = _mock_tradernet_client(response)

    get_candles(client, "AAPL.US", days=10)
    first_call_count = client._post_v2_signed.call_count

    get_candles(client, "AAPL.US", days=10)
    assert client._post_v2_signed.call_count == first_call_count, "second call must hit cache"
