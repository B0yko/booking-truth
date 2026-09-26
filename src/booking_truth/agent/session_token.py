"""Widget session tokens: HMAC-SHA256 over ``session_id|normalised email`` with ``BT_SESSION_SECRET``.

The widget receives a token with its first reply and sends it back on every request. A token for another
session or another email does not verify, so one widget session cannot read or change another's bookings.
"""

from __future__ import annotations

import base64
import hashlib
import hmac

from booking_truth.store import normalize_email

TOKEN_PREFIX = "wst1."  # noqa: S105 - a format marker, not a secret


def sign_session(secret: str, session_id: str, email: str) -> str:
    message = f"{session_id}|{normalize_email(email)}".encode()
    mac = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
    return TOKEN_PREFIX + base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")


def verify_session(secret: str, token: str | None, session_id: str, email: str) -> bool:
    if not token:
        return False
    return hmac.compare_digest(token, sign_session(secret, session_id, email))


def token_hash(token: str) -> str:
    """What the sessions table stores: never the token itself."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
