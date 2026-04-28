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


def test_v2_signed_path_used_when_secret_present():
    """
    Phase 1 — v2 signed.  The client must hit /v2/cmd/{cmd}, send a
    form-urlencoded body (no `q` wrapper), and set BOTH X-NtApi-Sig and
    X-NtApi-PublicKey headers.
    """
    session = _mock_session({"result": {"ps": {"key": "X", "acc": [], "pos": []}}})
    client = TradernetClient("pub", "sec", session=session)
    client.get_portfolio()

    call_args   = session.post.call_args
    call_url    = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
    call_kwargs = call_args.kwargs

    # URL targets /v2/cmd/{cmd}, not the bare /api/.
    assert call_url.endswith("/v2/cmd/getPositionJson"), call_url

    # Body is form-urlencoded plain string (not a dict-with-`q`).
    assert isinstance(call_kwargs["data"], str)
    assert call_kwargs["data"].startswith("apiKey=") or "apiKey=" in call_kwargs["data"]

    # Both auth headers must be set.
    headers = call_kwargs["headers"]
    assert headers["X-NtApi-Sig"]
    assert headers["X-NtApi-PublicKey"] == "pub"
    assert headers["Content-Type"] == "application/x-www-form-urlencoded"


def test_v1_signed_path_attempted_after_v2_rejection():
    """
    When v2 returns code=12, the client falls back to legacy v1 md5 signing.
    """
    session = MagicMock()
    response_auth_error = MagicMock()
    response_auth_error.status_code = 200
    response_auth_error.raise_for_status.return_value = None
    response_auth_error.json.return_value = {"code": 12, "errMsg": "Invalid credentials"}
    response_auth_error.text = '{"code":12,"errMsg":"Invalid credentials"}'

    response_ok = MagicMock()
    response_ok.status_code = 200
    response_ok.raise_for_status.return_value = None
    response_ok.json.return_value = {"ps": {"key": "X", "acc": [], "pos": []}}
    response_ok.text = '{"ps":{}}'

    call_count = 0

    def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # v2 attempt(s) → auth error; v1 md5 attempt → success
        return response_auth_error if call_count == 1 else response_ok

    session.post.side_effect = _side_effect
    client = TradernetClient("pub", "sec", session=session)
    client.get_portfolio()

    # First call is v2 (URL contains /v2/cmd/)
    first_url = session.post.call_args_list[0].args[0]
    assert "/v2/cmd/" in first_url

    # Second call is v1 (legacy /api/ with q= form field)
    second_call = session.post.call_args_list[1]
    second_url  = second_call.args[0] if second_call.args else second_call.kwargs.get("url", "")
    assert "/v2/cmd/" not in second_url
    assert "data" in second_call.kwargs
    assert "q" in second_call.kwargs["data"]


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


def test_signed_falls_back_to_unsigned_on_auth_error():
    """
    When signed auth returns code=12 (wrong secret), the client must retry
    via unsigned before raising.  If unsigned succeeds, the portfolio is returned.
    """
    success_payload = {"ps": {"key": "X", "acc": [], "pos": [{"i": "TSLA.US", "q": 2, "s": 400, "mkt_price": 200, "open_bal": 350}]}}

    call_count = 0

    session = MagicMock()
    response_auth_error = MagicMock()
    response_auth_error.status_code = 200
    response_auth_error.raise_for_status.return_value = None
    response_auth_error.json.return_value = {"code": 12, "errMsg": "Invalid credentials"}
    response_auth_error.text = '{"code":12,"errMsg":"Invalid credentials"}'

    response_ok = MagicMock()
    response_ok.status_code = 200
    response_ok.raise_for_status.return_value = None
    response_ok.json.return_value = success_payload
    response_ok.text = json.dumps(success_payload)

    def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Calls 1+2: v2 signed (errors), call 3: v1 md5 signed (errors),
        # call 4: unsigned (succeeds).  v2 and v1 each break their cmd loop
        # on the first auth rejection, so each contributes one call.
        return response_auth_error if call_count <= 2 else response_ok

    session.post.side_effect = _side_effect

    client = TradernetClient("pub", "sec", session=session)
    portfolio = client.get_portfolio()

    assert portfolio.pos[0].i == "TSLA.US"

    # The full fallback chain: v2 signed → v1 md5 signed → unsigned (success).
    urls = [c.args[0] for c in session.post.call_args_list]
    assert "/v2/cmd/" in urls[0]                       # Phase 1 — v2 signed
    assert "/v2/cmd/" not in urls[1]                   # Phase 2 — v1 md5 signed
    assert "/v2/cmd/" not in urls[-1]                  # Phase 3 — unsigned

    # The succeeding call must be the unsigned q= form-encoded payload.
    final_payload = json.loads(session.post.call_args_list[-1].kwargs["data"]["q"])
    assert "sig" not in final_payload
    assert final_payload["params"]["apiKey"] == "pub"
