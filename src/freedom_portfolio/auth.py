"""
Tradernet request signing (md5+secret).

Algorithm (per official Tradernet docs, https://github.com/Tradernet/tn.api):
    1. Sort top-level keys alphabetically (ascending).
    2. Render each pair as "key=value", recursing into nested dicts.
    3. Concatenate pairs WITHOUT separator.
    4. Append the raw API secret string at the very end.
    5. md5(utf8 bytes of the full string) — hex digest is the signature.

Note: this is NOT HMAC. The "HMAC-MD5" wording sometimes used in the wild is
a misnomer — there is no HMAC keying schedule, just plain md5 of (data + secret).
"""

from __future__ import annotations

import hashlib
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
