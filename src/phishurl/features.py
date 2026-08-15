"""Host-name feature extraction — the single source of truth for the feature vector.

Every feature here reads the canonical host and nothing else. That constraint is
not stylistic: PhiUSIIL's legitimate rows are uniformly ``https://www.<domain>``
with no path, so any feature touching the scheme, the ``www.`` prefix or the path
would separate the two collection pipelines instead of separating phishing from
safety. ``tests/test_no_host_leakage.py`` enforces it mechanically.

The second constraint is that ``docs/app.js`` contains a line-for-line port of this
module, so the browser demo and the Python service agree. Features that would need
a heavy Python dependency were rejected on those grounds alone; what is left is
character counting, a frozen Public Suffix List and two frozen word lists.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from phishurl.canonicalize import is_ip_literal
from phishurl.psl import is_platform_hosted, split_host

DATA = Path(__file__).resolve().parents[2] / "data"

VOWELS = frozenset("aeiou")
DIGITS = frozenset("0123456789")
LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz")

# Words that appear in host names built to look like a login page. Frozen, small,
# and deliberately label-independent — this is domain knowledge, not a fitted prior.
SUSPICIOUS_KEYWORDS = (
    "login", "signin", "secure", "verify", "verification", "account", "update",
    "confirm", "webscr", "banking", "recover", "unlock", "suspend", "billing",
    "invoice", "payment", "wallet", "support", "helpdesk", "auth",
)

# The 25 most frequent public suffixes in the training data, frozen into the model
# artifact. One-hot rather than target-encoded on purpose: a TLD phishing prior
# fitted on the data would smuggle the label back into the features.
TOP_SUFFIXES = (
    "com", "org", "net", "co.uk", "de", "ru", "io", "app", "co", "info",
    "com.br", "it", "fr", "nl", "au", "ca", "jp", "top", "xyz", "site",
    "dev", "link", "online", "edu", "es",
)


@lru_cache(maxsize=1)
def load_brand_tokens() -> frozenset[str]:
    """Brand names frozen from the Tranco top 1000.

    Label-independent by construction: Tranco ranks by popularity and knows
    nothing about phishing, so a brand list drawn from it cannot leak the target.
    """
    path = DATA / "brand_tokens.txt"
    return frozenset(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def shannon_entropy(text: str) -> float:
    """Shannon entropy in bits over the characters of ``text``."""
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for character in text:
        counts[character] = counts.get(character, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _max_run(text: str, members: frozenset[str]) -> int:
    best = 0
    current = 0
    for character in text:
        if character in members:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def _count(text: str, members: frozenset[str]) -> int:
    return sum(1 for character in text if character in members)


FEATURE_NAMES: tuple[str, ...] = (
    # shape
    "host_length", "n_labels", "n_subdomain_labels", "registrable_length",
    "subdomain_length", "suffix_length", "max_label_length", "mean_label_length",
    # composition
    "entropy_host", "entropy_registrable", "digit_count", "digit_ratio",
    "hyphen_count", "hyphen_ratio", "vowel_ratio", "max_consecutive_digits",
    "max_consecutive_consonants", "digits_in_registrable",
    # structure
    "is_ip_literal", "has_punycode", "is_multi_level_suffix", "is_platform_hosted",
    # lexical
    "has_suspicious_keyword", "brand_token_present", "brand_in_subdomain_only",
) + tuple(f"suffix_is_{suffix.replace('.', '_')}" for suffix in TOP_SUFFIXES) + ("suffix_is_other",)


def extract(host: str) -> list[float]:
    """Return the feature vector for a canonical ``host``.

    Total, by design: an empty or malformed host yields a well-defined vector of
    zeros rather than an exception, because this runs behind a public endpoint.
    """
    if not host:
        return [0.0] * len(FEATURE_NAMES)

    subdomain, registrable, suffix = split_host(host)
    labels = host.split(".")
    label_lengths = [len(label) for label in labels]

    letters = _count(host, LETTERS)
    vowels = _count(host, VOWELS)
    digits = _count(host, DIGITS)
    hyphens = host.count("-")
    consonants = frozenset(LETTERS - VOWELS)

    registrable_name = registrable[: len(registrable) - len(suffix) - 1] if registrable else ""
    brands = load_brand_tokens()
    brand_in_subdomain = any(brand in subdomain for brand in brands) if subdomain else False
    brand_in_registrable = registrable_name in brands

    values: list[float] = [
        # shape
        float(len(host)),
        float(len(labels)),
        float(len(subdomain.split(".")) if subdomain else 0),
        float(len(registrable)),
        float(len(subdomain)),
        float(len(suffix)),
        float(max(label_lengths)),
        float(sum(label_lengths) / len(label_lengths)),
        # composition
        shannon_entropy(host),
        shannon_entropy(registrable),
        float(digits),
        digits / len(host),
        float(hyphens),
        hyphens / len(host),
        (vowels / letters) if letters else 0.0,
        float(_max_run(host, DIGITS)),
        float(_max_run(host, consonants)),
        float(_count(registrable, DIGITS)),
        # structure
        float(is_ip_literal(host)),
        float("xn--" in host),
        float("." in suffix),
        float(is_platform_hosted(host)),
        # lexical
        float(any(keyword in host for keyword in SUSPICIOUS_KEYWORDS)),
        float(brand_in_subdomain or brand_in_registrable),
        # The paypal.com.evil.tk shape: a brand appears, but not where it would be
        # if the site actually belonged to that brand.
        float(brand_in_subdomain and not brand_in_registrable),
    ]

    values.extend(1.0 if suffix == candidate else 0.0 for candidate in TOP_SUFFIXES)
    values.append(0.0 if suffix in TOP_SUFFIXES else 1.0)

    return values


def extract_many(hosts: list[str]) -> list[list[float]]:
    return [extract(host) for host in hosts]
