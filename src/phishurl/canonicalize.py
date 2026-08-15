"""Reduce any URL-ish string to a canonical host name.

This is the first step of every path through the project, and it is the step
that makes the whole evaluation honest. Two measured properties of the source
data force it:

* In PhiUSIIL, **every** legitimate URL is ``https://www.<domain>`` — 0.00 % have
  a path, a query or an ``@`` — while phishing rows were dumped raw from feeds.
  A model reading the scheme or the path separates the two collection pipelines,
  not phishing from safety.
* Tranco ships bare domains while PhishTank ships deep URLs, so any full-URL
  comparison lets "has a path" masquerade as a detector.

Canonicalising to the host neutralises both.

Parsing is done with plain string operations rather than :func:`urllib.parse.urlsplit`
or the browser's ``URL``. Those two disagree — notably ``URL`` punycodes IDN hosts
and ``urlsplit`` does not — and this function has a twin in ``docs/app.js`` that
must agree bit for bit. Identical string surgery on both sides is the cheapest
way to guarantee that. Case folding is restricted to ASCII for the same reason.
"""

from __future__ import annotations

import re

_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
_ASCII_UPPER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")

MAX_INPUT_LENGTH = 2048


def canonical_host(raw: str) -> str:
    """Return the canonical host of ``raw``, or ``""`` if there is none.

    Strips, in order: surrounding whitespace, scheme, protocol-relative prefix,
    path/query/fragment, userinfo, port, brackets around IPv6 literals, trailing
    dots and a leading ``www.``. Never raises: unparseable input yields ``""``
    so callers can return a 400 rather than a stack trace.
    """
    if not isinstance(raw, str):
        return ""

    host = raw.strip()
    if not host:
        return ""

    host = _SCHEME_RE.sub("", host, count=1)
    if host.startswith("//"):
        host = host[2:]

    # Authority ends at the first path, query or fragment delimiter.
    for delimiter in ("/", "?", "#"):
        index = host.find(delimiter)
        if index != -1:
            host = host[:index]

    # Userinfo: `https://paypal.com@evil.tk` is a real attack, and the host is
    # the part after the LAST `@`.
    if "@" in host:
        host = host.rsplit("@", 1)[1]

    if host.startswith("["):
        end = host.find("]")
        host = host[1:end] if end != -1 else host[1:]
    elif ":" in host:
        host = host.split(":", 1)[0]

    host = host.translate(_ASCII_UPPER)
    host = host.rstrip(".")

    if host.startswith("www."):
        host = host[4:]

    return host


_IPV4_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")


def is_ipv4_literal(host: str) -> bool:
    match = _IPV4_RE.match(host)
    if not match:
        return False
    return all(0 <= int(part) <= 255 for part in match.groups())


def is_ipv6_literal(host: str) -> bool:
    """Loose IPv6 check: hex groups and colons only, with at least one colon.

    Deliberately permissive — the feature only needs to know "this is an address,
    not a name", and a strict RFC 4291 parser would be a large thing to mirror
    in JavaScript for no gain.
    """
    if ":" not in host:
        return False
    return all(character in "0123456789abcdef:." for character in host)


def is_ip_literal(host: str) -> bool:
    return is_ipv4_literal(host) or is_ipv6_literal(host)
