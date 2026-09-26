"""Redaction for everything the harness writes: email addresses, home-directory paths and URL hosts.

Every string of a JSON value (dictionary keys included) is rewritten: the lead's address becomes
``[lead_email]``, any other address becomes ``[email]``, and a home directory (``Users`` or ``home`` under the
root, or ``Users`` on a Windows drive, followed by the account name) becomes ``~``. Percent-encoded addresses
(``lead%40example.com`` in a logged URL) are caught too.

:func:`mask_hosts` rewrites the host of every URL: a loopback host becomes ``localhost``, a public API host
the harness knows is kept, and any other host (a machine name, a LAN address) becomes ``[host]``; user
credentials in a URL are dropped. :func:`shorten_paths` reduces absolute file paths in an error text to a
path inside the package (``booking_truth/harness/runner.py``) or a file name.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

LEAD_PLACEHOLDER = "[lead_email]"
EMAIL_PLACEHOLDER = "[email]"

_LOCAL = r"[A-Za-z0-9_+\-](?:[A-Za-z0-9._+\-]*[A-Za-z0-9_+\-])?"
_DOMAIN = r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}"
#: A plain address. The local part is kept to the common characters so that ``key=value@host`` style
#: neighbours are not swallowed.
#: Where an address may start: not inside a word, and not inside a ``%XX`` escape, but right after one
#: (``bt_lead_email%3Dlead%40example.com``).
_START = r"(?<!%)(?<!%[0-9A-Fa-f])(?:(?<=%[0-9A-Fa-f]{2})|(?<![A-Za-z0-9._+\-]))"
EMAIL = re.compile(rf"{_START}{_LOCAL}@{_DOMAIN}(?![A-Za-z0-9\-])")
#: An address whose ``@`` is percent-encoded, as in a logged query string.
ENCODED_EMAIL = re.compile(rf"{_START}{_LOCAL}%40{_DOMAIN}(?![A-Za-z0-9\-])", re.IGNORECASE)
HOME_PATH = re.compile(
    r"(?<![\w.\-])(?:/Users|/home)/[^/\\\s'\"<>:]+"
    r"|[A-Za-z]:(?:\\\\|\\|/)(?:Users|Documents and Settings)(?:\\\\|\\|/)[^/\\\s'\"<>:]+",
)


HOST_PLACEHOLDER = "[host]"
#: Public API hosts that name a vendor, not a machine; they are kept as they are.
PUBLIC_HOSTS: tuple[str, ...] = (
    "cal.com",
    "googleapis.com",
    "google.com",
    "hubapi.com",
    "hubspot.com",
    "openrouter.ai",
    "example.com",
)
URL_HOST = re.compile(
    r"(?P<scheme>\b[A-Za-z][A-Za-z0-9+.\-]*://)(?P<userinfo>[^/\s@'\"<>]*@)?"
    r"(?P<host>\[[0-9A-Fa-f:.]+\]|[^/\s:?#'\"<>\[\]]+)"
)
#: An absolute file path under a system or home root, with at least one directory after the root.
ABS_PATH = re.compile(
    r"(?<![\w.:/\-~])/(?:Users|home|private|tmp|var|opt|usr|app|root|srv|mnt|Volumes|Library|nix|workspace)"
    r"(?:/[^/\s'\"<>:,()]+)+"
    r"|[A-Za-z]:(?:\\\\|\\)(?:[^\\\s'\"<>:,()]+(?:\\\\|\\))+[^\\\s'\"<>:,()]+"
)


def _public_host(host: str) -> str:
    """How a URL host appears in outputs."""
    bare = host.strip("[]").lower()
    if bare == "localhost" or bare.endswith(".localhost"):
        return "localhost"
    try:
        address = ipaddress.ip_address(bare)
    except ValueError:
        address = None
    if address is not None:
        return "localhost" if address.is_loopback or address.is_unspecified else HOST_PLACEHOLDER
    if any(bare == known or bare.endswith("." + known) for known in PUBLIC_HOSTS):
        return host
    return HOST_PLACEHOLDER


def mask_url_hosts(text: str) -> str:
    """Rewrite the host of every URL in ``text`` (see the module docstring)."""
    return URL_HOST.sub(lambda m: f"{m['scheme']}{_public_host(m['host'])}", text)


def mask_hosts(value: Any) -> Any:
    """A copy of a JSON value with :func:`mask_url_hosts` applied to every string, keys included."""
    if isinstance(value, str):
        return mask_url_hosts(value)
    if isinstance(value, dict):
        return {mask_hosts(k): mask_hosts(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [mask_hosts(item) for item in value]
    return value


def short_path(path: str) -> str:
    """An absolute file path as a path inside an installed package (after ``site-packages``), inside
    ``booking_truth`` or ``tests``, or else its file name."""
    normal = path.replace("\\", "/")
    for marker in ("/site-packages/", "/dist-packages/"):
        if marker in normal:
            return normal.rsplit(marker, 1)[1]
    for package in ("booking_truth", "tests"):
        marker = f"/{package}/"
        if marker in normal:
            return package + "/" + normal.rsplit(marker, 1)[1]
    return normal.rsplit("/", 1)[-1]


def shorten_paths(text: str) -> str:
    """Replace every absolute file path in an error text with :func:`short_path`, then strip what is left
    of home directories."""
    return strip_home_paths(ABS_PATH.sub(lambda m: short_path(m[0]), text))


def scrub_text(text: str) -> str:
    """:func:`mask_url_hosts` and :func:`shorten_paths` on one string."""
    return shorten_paths(mask_url_hosts(text))


def scrub(value: Any) -> Any:
    """A copy of a JSON value with :func:`scrub_text` applied to every string, keys included. Run outputs
    (traces, the manifest) go through it after :func:`redact`, so they carry no machine names, LAN
    addresses or absolute paths."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {scrub(k): scrub(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [scrub(item) for item in value]
    return value


def find_absolute_paths(value: Any) -> list[str]:
    """Every absolute file path (under a system or home root) in any string of a JSON value."""
    return [m[0] for text in _strings(value) for m in ABS_PATH.finditer(text)]


def _lead_forms(lead_email: str) -> list[str]:
    lead = lead_email.strip()
    forms = {lead, lead.replace("@", "%40"), quote(lead, safe=""), quote(lead, safe="@")}
    return sorted((f for f in forms if f), key=len, reverse=True)


def strip_home_paths(text: str) -> str:
    """Replace every home-directory prefix with ``~``."""
    return HOME_PATH.sub("~", text)


def redact_text(text: str, lead_email: str | None) -> str:
    """Redact one string: the lead's address, then any other address, then home paths."""
    if lead_email:
        for form in _lead_forms(lead_email):
            text = re.sub(re.escape(form), LEAD_PLACEHOLDER, text, flags=re.IGNORECASE)
    text = EMAIL.sub(EMAIL_PLACEHOLDER, text)
    text = ENCODED_EMAIL.sub(EMAIL_PLACEHOLDER, text)
    return strip_home_paths(text)


def redact(value: Any, lead_email: str | None) -> Any:
    """A copy of a JSON value with every string redacted, keys included."""
    if isinstance(value, str):
        return redact_text(value, lead_email)
    if isinstance(value, dict):
        return {redact(k, lead_email): redact(v, lead_email) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(item, lead_email) for item in value]
    return value


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _strings(item)


def find_emails(value: Any) -> list[str]:
    """Every email address (plain or percent-encoded) in any string of a JSON value, keys included."""
    found: list[str] = []
    for text in _strings(value):
        found += [m[0] for m in EMAIL.finditer(text)]
        found += [m[0] for m in ENCODED_EMAIL.finditer(text)]
    return found


def find_home_paths(value: Any) -> list[str]:
    """Every home-directory path prefix in any string of a JSON value."""
    return [m[0] for text in _strings(value) for m in HOME_PATH.finditer(text)]
