"""
Report storage — uploads rendered HTML reports to Google Cloud Storage
and returns time-limited signed URLs for delivery via Telegram.

Why GCS?
  • google-cloud-storage is already in requirements.txt (used by the
    chroma_db boot sync), so no new dependency.
  • Signed URLs let us mail a single link that expires after N days
    without exposing the bucket publicly.
  • Static HTML is served straight from GCS — no compute on read.

Design constraints
──────────────────
  • Graceful degradation: when REPORT_BUCKET_NAME is unset (local dev),
    upload_report() returns a `file://` URL pointing to the locally
    written report.  The bot can still link to it via TG file-share for
    debugging, and unit tests don't need real GCS credentials.
  • Atomic-ish upload: HTML is uploaded with a content-disposition of
    `inline; filename=...`, so opening the URL in the user's browser
    renders the report directly (no download prompt).
  • TTL configurable via REPORT_URL_TTL_HOURS (default 168h = 7 days).
  • Object names are user/date keyed so re-running the same report on
    the same day overwrites cleanly: `r/<user_id>/<YYYY-MM-DD>/<tier>.html`.

Environment variables
─────────────────────
  REPORT_BUCKET_NAME      GCS bucket (e.g. "ramp-bot-reports").  When
                          empty, falls back to file:// mode.
  REPORT_URL_TTL_HOURS    Signed-URL lifetime in hours (default 168).
  GOOGLE_APPLICATION_CREDENTIALS  Standard GCP auth env var (Cloud Run
                          uses the service-account binding automatically).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


# ── Public config ────────────────────────────────────────────────────────────
BUCKET_NAME       = os.getenv("REPORT_BUCKET_NAME", "").strip()
# PII protection: reports contain holdings + P&L.  Default signed-URL life
# lowered 168h → 48h (still long enough for a user to revisit a link the
# bot sent, short enough to bound exposure if a link leaks).  Override via
# REPORT_URL_TTL_HOURS but the default is now privacy-conservative.
DEFAULT_TTL_HOURS = int(os.getenv("REPORT_URL_TTL_HOURS", "48"))    # 2 days
CONTENT_TYPE      = "text/html; charset=utf-8"
# Never let a private financial report rest in any shared/CDN/browser cache.
CACHE_CONTROL     = "private, no-store, max-age=0"


def _object_path(user_id: int | str, tier: str, today: Optional[str] = None) -> str:
    """Compose the GCS object key (also used as the URL path suffix)."""
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"r/{user_id}/{today}/{tier}.html"


def _signed_url(blob, credentials, ttl_hours: int) -> str:
    """
    Produce a v4 signed GET URL for ``blob``.

    Two signing paths:
      • Service-account key files carry a private key and can sign the
        URL locally with no network call.
      • Cloud Run / GCE supply token-only credentials
        (compute_engine.Credentials) that have NO private key.  Signing
        those locally raises "you need a private key to sign
        credentials" — so we sign via the IAM Credentials signBlob API
        by handing generate_signed_url the service-account email and a
        fresh access token.  This requires the runtime service account
        to hold roles/iam.serviceAccountTokenCreator on itself.
    """
    import google.auth.transport.requests   # type: ignore[import-not-found]

    expiration = timedelta(hours=ttl_hours)
    try:
        return blob.generate_signed_url(
            version    = "v4",
            expiration = expiration,
            method     = "GET",
        )
    except Exception as exc:
        # Expected on Cloud Run: compute_engine credentials have no private
        # key, so the v4 signer always falls back to IAM signBlob.  Keep at
        # DEBUG to avoid steady-state INFO noise on every report delivery.
        logger.debug("Local URL signing unavailable (%s); using IAM signBlob", exc)
        auth_request = google.auth.transport.requests.Request()
        credentials.refresh(auth_request)
        return blob.generate_signed_url(
            version               = "v4",
            expiration            = expiration,
            method                = "GET",
            service_account_email = credentials.service_account_email,
            access_token          = credentials.token,
        )


def _upload_to_gcs(local_path: Path,
                    bucket_name: str,
                    object_name: str,
                    ttl_hours:   int) -> str:
    """
    Push the file to GCS and return a v4 signed URL.

    The google-cloud-storage import is lazy: when the bucket isn't
    configured the caller falls back without ever touching the SDK,
    which means local-only test environments don't need GCP creds.
    """
    import google.auth                      # type: ignore[import-not-found]
    from google.cloud import storage        # type: ignore[import-not-found]

    credentials, _project = google.auth.default()
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(object_name)

    blob.cache_control       = CACHE_CONTROL
    blob.content_disposition = f'inline; filename="{Path(object_name).name}"'

    blob.upload_from_filename(str(local_path), content_type=CONTENT_TYPE)
    logger.info("Uploaded %s → gs://%s/%s (%.1f KB)",
                 local_path.name, bucket_name, object_name,
                 local_path.stat().st_size / 1024)

    return _signed_url(blob, credentials, ttl_hours)


def upload_report(local_html_path: str | Path,
                   user_id: int | str,
                   tier:    str,
                   *,
                   bucket_name: Optional[str] = None,
                   ttl_hours:   Optional[int] = None,
                   ) -> str:
    """
    Upload a rendered HTML report and return a shareable URL.

    Args:
        local_html_path : path to the file produced by
                           html_renderer.write_report_html().
        user_id         : Telegram user ID (used as the URL prefix).
        tier            : 'base' | 'deep'.
        bucket_name     : Override REPORT_BUCKET_NAME (for tests).
        ttl_hours       : Override REPORT_URL_TTL_HOURS (for tests).

    Returns:
        URL string.  When BUCKET_NAME is set → signed HTTPS URL valid
        for ttl_hours.  When not set → `file://<absolute path>`.
    """
    path = Path(local_html_path)
    if not path.exists():
        raise FileNotFoundError(f"Report file missing: {path}")

    bucket = (bucket_name if bucket_name is not None else BUCKET_NAME).strip()
    ttl    = ttl_hours if ttl_hours is not None else DEFAULT_TTL_HOURS

    if not bucket:
        # Local-dev fallback — return a file:// URL so the test/dev
        # workflow doesn't break.  Cloud Run logs will warn on every call.
        logger.warning("REPORT_BUCKET_NAME not configured — returning file:// URL")
        return f"file://{path.resolve()}"

    object_name = _object_path(user_id, tier)
    try:
        return _upload_to_gcs(path, bucket, object_name, ttl)
    except Exception as exc:
        # Don't crash the report flow if GCS is down — log + return file://
        # so the bot at least tells the user the report was generated.
        logger.error("GCS upload failed (%s); falling back to file://", exc)
        return f"file://{path.resolve()}"


__all__ = [
    "BUCKET_NAME",
    "DEFAULT_TTL_HOURS",
    "upload_report",
]
