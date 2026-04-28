"""
Tradernet request signing.

Two algorithms exist on the same host:

  Legacy /api/   — md5( sorted("k=v")_concat_no_separator + secret_key ).
                   Designed for uid-based authentication where ``uid`` is the
                   user's account number.  apiKey-based credentials are NOT
                   accepted on this path — the server interprets the ``apiKey``
                   field as a uid and returns ``code=12 Invalid credentials``.
                   Kept here for backwards compatibility and tests.

  Modern /api/v2/cmd/{cmd} — HMAC-SHA256(secret, sorted("k=v") joined by "&").
                   Designed for apiKey-based authentication.  Requires both
                   ``X-NtApi-Sig`` and ``X-NtApi-PublicKey`` headers.
                   This is the path that retail Freedom Broker keys (32-char
                   public + 40-char secret) authenticate against.
"""

from __future__ import annotations

import hashlib
import hmac
import time


def _serialize(data: dict) -> str:
    """
    Serialize *data* into the canonical sorted "key=value..." string used by
    Tradernet for signing.  Nested dicts are serialized recursively in place
    of their value — the outer pair becomes ``parent=<inner_serialized>``.
    """
    parts: list[str] = []
    for key in sorted(data.keys()):
        value = data[key]
        if isinstance(value, dict):
            rendered = _serialize(value)
        elif value is None:
            rendered = ""
        elif isinstance(value, bool):
            # Tradernet PHP backend serializes booleans as 1/0, not True/False
            rendered = "1" if value else "0"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return "".join(parts)


def build_signature(params: dict, secret_key: str) -> str:
    """
    Compute the Tradernet ``sig`` value for *params* using *secret_key*.

    *params* MUST NOT contain the ``sig`` field itself.
    Returns the hex-encoded md5 digest (32 lowercase chars).
    """
    payload = _serialize(params) + secret_key
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def build_request(
    cmd: str,
    params: dict,
    public_key: str,
    secret_key: str,
    *,
    nonce: str | None = None,
) -> dict:
    """
    Assemble a signed request body for Tradernet REST POST.

    Adds ``apiKey``, ``cmd``, ``nonce`` (Unix ms as string), ``params``, then ``sig``.
    Returns a flat dict ready to be JSON-encoded as the POST body.

    The ``nonce`` argument is exposed so callers (and tests) can pin it; in
    production it is generated automatically from ``time.time()``.
    """
    if nonce is None:
        nonce = str(int(time.time() * 1000))

    body: dict = {
        "apiKey": public_key,
        "cmd":    cmd,
        "nonce":  nonce,
        "params": params or {},
    }
    body["sig"] = build_signature(body, secret_key)
    return body


# ── Modern /api/v2/cmd/ HMAC-SHA256 signing ──────────────────────────────────


def _serialize_v2(data: dict) -> str:
    """
    Serialize *data* into the v2 sign-input format: sorted "key=value" pairs
    joined by "&".  Nested dicts are serialized recursively so nested values
    become the inner serialized string (matching kofeinstyle/tradernet-sdk).
    """
    parts: list[str] = []
    for key in sorted(data.keys()):
        value = data[key]
        if isinstance(value, dict):
            rendered = _serialize_v2(value)
        elif value is None:
            rendered = ""
        elif isinstance(value, bool):
            rendered = "1" if value else "0"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return "&".join(parts)


def build_v2_signature(params: dict, secret_key: str) -> str:
    """
    Compute the v2 ``X-NtApi-Sig`` value: hex-encoded HMAC-SHA256 of the
    "&"-joined sorted "key=value" pairs.
    """
    pre = _serialize_v2(params)
    return hmac.new(
        secret_key.encode("utf-8"),
        pre.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def build_v2_request(
    cmd: str,
    params: dict,
    public_key: str,
    secret_key: str,
    *,
    nonce: int | None = None,
) -> tuple[dict, str]:
    """
    Build a v2 request payload + HMAC-SHA256 signature.

    ``nonce`` follows the JS SDK convention ``Math.floor(Date.now() * 10000)``
    which equals ``int(time.time() * 1e7)`` — a 100-ns-resolution monotonic
    counter that the server uses for replay protection.

    Returns
    -------
    (payload, sig)
        The dict {apiKey, cmd, nonce, params} and the corresponding signature.
        The caller submits the payload form-urlencoded with ``X-NtApi-Sig`` and
        ``X-NtApi-PublicKey`` headers.
    """
    if nonce is None:
        nonce = int(time.time() * 1e7)

    payload: dict = {
        "apiKey": public_key,
        "cmd":    cmd,
        "nonce":  nonce,
        "params": params or {},
    }
    sig = build_v2_signature(payload, secret_key)
    return payload, sig
