"""Unit tests for freedom_portfolio.client — no live network access."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from freedom_portfolio.client import (  # noqa: E402
    AuthenticationError,
    BrokerAPIError,
    InvalidSignatureError,
    TradernetClient,
)


def _mock_session(response_json: dict, *, status: int = 200, text: str | None = None):
    """Build a mocked requests.Session whose .post() returns *response_json*."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = status
    response.json.return_value = response_json
    response.text = text if text is not None else json.dumps(response_json)
    response.raise_for_status.return_value = None
    if status >= 400:
        from requests import HTTPError
        response.raise_for_status.side_effect = HTTPError(f"HTTP {status}")
    session.post.return_value = response
    return session


def test_invalid_signature_maps_to_typed_exception():
    session = _mock_session({"code": 4, "errMsg": "Invalid signature"})
    client = TradernetClient("pub", "sec", session=session)
    with pytest.raises(InvalidSignatureError):
        client.get_portfolio()


def test_authentication_error_maps_to_typed_exception():
    session = _mock_session({"code": 12, "errMsg": "Invalid credentials"})
    client = TradernetClient("pub", "sec", session=session)
    with pytest.raises(AuthenticationError):
        client.get_portfolio()


def test_generic_error_maps_to_broker_api_error():
    # Code 99 is not in the typed-exception map → falls through to base class.
    # The client retries getPortfolio after getPositionJson fails, so the second
    # call also returns the same error, which is the one finally raised.
    session = _mock_session({"code": 99, "errMsg": "Unknown error"})
    client = TradernetClient("pub", "sec", session=session)
    with pytest.raises(BrokerAPIError):
        client.get_portfolio()


def test_signed_path_used_when_secret_present():
    session = _mock_session({"result": {"ps": {"key": "X", "acc": [], "pos": []}}})
    client = TradernetClient("pub", "sec", session=session)
    client.get_portfolio()

    # Both signed and unsigned paths send `data={"q": <json>}` form-encoded —
    # the difference is whether apiKey/nonce/sig appear at the top level of the
    # JSON (signed) or only inside params (unsigned).
    call_kwargs = session.post.call_args.kwargs
    assert "data" in call_kwargs
    assert "json" not in call_kwargs

    payload = json.loads(call_kwargs["data"]["q"])
    assert payload["apiKey"] == "pub"
    assert payload["cmd"]    == "getPositionJson"
    assert "sig" in payload
    assert "nonce" in payload


def test_unsigned_path_used_when_secret_missing():
    session = _mock_session({"ps": {"key": "X", "acc": [], "pos": []}})
    client = TradernetClient("pub", "", session=session)
    client.get_portfolio()

    call_kwargs = session.post.call_args.kwargs
    # Unsigned path uses form-encoded `data={"q": ...}` — no `json=`.
    assert "data" in call_kwargs
    assert "json" not in call_kwargs
    payload = json.loads(call_kwargs["data"]["q"])
    assert payload["params"]["apiKey"] == "pub"


def test_base_url_defaults_to_tradernet_com():
    client = TradernetClient("pub", "sec")
    assert client.base_url.startswith("https://tradernet.com/api/")


def test_portfolio_response_parsed_into_pydantic_model():
    session = _mock_session({
        "result": {
            "ps": {
                "key": "%U",
                "acc": [{"s": 100, "curr": "USD", "currval": 1.0}],
                "pos": [{"i": "AAPL.US", "q": 1, "s": 175, "mkt_price": 175, "open_bal": 150}],
            }
        }
    })
    client = TradernetClient("pub", "sec", session=session)
    portfolio = client.get_portfolio()
    assert portfolio.pos[0].i == "AAPL.US"
    assert portfolio.acc[0].s == 100
