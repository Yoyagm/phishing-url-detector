"""Feature extraction: hand-computed expectations and total behaviour."""

from __future__ import annotations

import math

import pytest

from phishurl.features import (
    FEATURE_NAMES,
    SUSPICIOUS_KEYWORDS,
    TOP_SUFFIXES,
    extract,
    shannon_entropy,
)


def feature(host: str, name: str) -> float:
    return extract(host)[FEATURE_NAMES.index(name)]


def test_names_are_unique() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0.0),
        ("aaaa", 0.0),
        ("ab", 1.0),
        ("abcd", 2.0),
    ],
)
def test_shannon_entropy(text: str, expected: float) -> None:
    assert shannon_entropy(text) == pytest.approx(expected)


def test_shape_features_on_a_known_host() -> None:
    host = "a.b.example.co.uk"
    assert feature(host, "host_length") == len(host)
    assert feature(host, "n_labels") == 5
    assert feature(host, "n_subdomain_labels") == 2
    assert feature(host, "registrable_length") == len("example.co.uk")
    assert feature(host, "subdomain_length") == len("a.b")
    assert feature(host, "suffix_length") == len("co.uk")
    assert feature(host, "max_label_length") == 7  # "example"
    assert feature(host, "is_multi_level_suffix") == 1.0


def test_composition_features() -> None:
    host = "ab-12.com"
    assert feature(host, "digit_count") == 2
    assert feature(host, "digit_ratio") == pytest.approx(2 / len(host))
    assert feature(host, "hyphen_count") == 1
    assert feature(host, "max_consecutive_digits") == 2
    # letters are a,b,c,o,m -> vowels a,o
    assert feature(host, "vowel_ratio") == pytest.approx(2 / 5)


def test_max_consecutive_consonants() -> None:
    assert feature("bcdfg.com", "max_consecutive_consonants") == 5
    assert feature("aeiou.com", "max_consecutive_consonants") == 1  # the 'c' of com


def test_structure_flags() -> None:
    assert feature("192.168.1.1", "is_ip_literal") == 1.0
    assert feature("example.com", "is_ip_literal") == 0.0
    assert feature("xn--80ak6aa92e.com", "has_punycode") == 1.0
    assert feature("myapp.vercel.app", "is_platform_hosted") == 1.0
    assert feature("example.com", "is_platform_hosted") == 0.0


@pytest.mark.parametrize("keyword", SUSPICIOUS_KEYWORDS[:5])
def test_suspicious_keyword_fires(keyword: str) -> None:
    assert feature(f"{keyword}-portal.tk", "has_suspicious_keyword") == 1.0


def test_suspicious_keyword_quiet_on_a_plain_host() -> None:
    assert feature("example.com", "has_suspicious_keyword") == 0.0


def test_brand_in_subdomain_only_is_the_impersonation_signal() -> None:
    """`paypal.com.evil.tk` carries a brand, but not where it would be if real."""
    assert feature("paypal.com.evil.tk", "brand_token_present") == 1.0
    assert feature("paypal.com.evil.tk", "brand_in_subdomain_only") == 1.0

    # The genuine article: brand sits in the registrable domain, so the
    # impersonation flag must stay silent.
    assert feature("paypal.com", "brand_token_present") == 1.0
    assert feature("paypal.com", "brand_in_subdomain_only") == 0.0

    # And an unrelated host trips neither.
    assert feature("some-shop.example", "brand_token_present") == 0.0
    assert feature("some-shop.example", "brand_in_subdomain_only") == 0.0


def test_suffix_one_hot_is_exclusive() -> None:
    vector = extract("example.com")
    flags = [vector[FEATURE_NAMES.index(f"suffix_is_{s.replace('.', '_')}")] for s in TOP_SUFFIXES]
    other = vector[FEATURE_NAMES.index("suffix_is_other")]
    assert sum(flags) + other == 1.0
    assert vector[FEATURE_NAMES.index("suffix_is_com")] == 1.0


def test_unknown_suffix_falls_into_other() -> None:
    vector = extract("example.museum")
    assert vector[FEATURE_NAMES.index("suffix_is_other")] == 1.0


def test_empty_and_degenerate_hosts_never_raise() -> None:
    for host in ["", ".", "..", "-", "a", "::1", "a" * 5000]:
        vector = extract(host)
        assert len(vector) == len(FEATURE_NAMES)
        assert all(math.isfinite(value) for value in vector)
