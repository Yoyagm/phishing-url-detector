"""Canonicalisation and Public Suffix List behaviour.

Every case in the spec's edge-case table is pinned here. The PSL implementation
is additionally cross-checked against ``tldextract`` on a real sample, so a
mistake in the hand-written lookup shows up as a test failure rather than as a
quietly wrong registrable domain in the training labels.
"""

from __future__ import annotations

import pytest

from phishurl.canonicalize import canonical_host, is_ip_literal, is_ipv4_literal
from phishurl.psl import is_platform_hosted, public_suffix_labels, split_host


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Scheme handling
        ("https://www.example.com", "example.com"),
        ("http://example.com", "example.com"),
        ("example.com", "example.com"),
        ("//example.com/path", "example.com"),
        ("ftp://example.com", "example.com"),
        # Path, query, fragment
        ("https://example.com/a/b/c", "example.com"),
        ("https://example.com?q=1", "example.com"),
        ("https://example.com#frag", "example.com"),
        ("example.com/x?y#z", "example.com"),
        # Userinfo — the host is what follows the LAST '@'
        ("https://paypal.com@evil.tk", "evil.tk"),
        ("https://user:pass@evil.tk/login", "evil.tk"),
        ("https://a@b@evil.tk", "evil.tk"),
        # Port
        ("https://example.com:8443/x", "example.com"),
        ("example.com:80", "example.com"),
        # IPv6 literal
        ("https://[::1]/", "::1"),
        ("http://[2001:db8::7]:8080/a", "2001:db8::7"),
        # Case, trailing dot, www
        ("HTTPS://WWW.EXAMPLE.COM", "example.com"),
        ("https://example.com.", "example.com"),
        ("https://www.example.com./x", "example.com"),
        ("WWW.Example.COM", "example.com"),
        # www stripped only as a leading label
        ("https://wwwexample.com", "wwwexample.com"),
        ("https://a.www.example.com", "a.www.example.com"),
        # Degenerate input never raises
        ("", ""),
        ("   ", ""),
        ("https://", ""),
        ("not a url", "not a url"),
        ("@", ""),
    ],
)
def test_canonical_host(raw: str, expected: str) -> None:
    assert canonical_host(raw) == expected


def test_canonical_host_rejects_non_strings() -> None:
    assert canonical_host(None) == ""  # type: ignore[arg-type]
    assert canonical_host(12345) == ""  # type: ignore[arg-type]


def test_very_long_host_is_returned_not_raised() -> None:
    """Length limits belong to the API layer; the parser must stay total."""
    long_host = "a" * 8000 + ".com"
    assert canonical_host(f"https://{long_host}/x") == long_host


@pytest.mark.parametrize(
    ("host", "expected_suffix_labels"),
    [
        ("example.com", 1),
        ("bbc.co.uk", 2),
        ("example.com.co", 2),
        ("foo.bar.example.com", 1),
        # Wildcard rule '*.ck' makes any <x>.ck a public suffix ...
        ("foo.ck", 2),
        # ... except for the '!www.ck' exception rule.
        ("www.ck", 1),
        ("city.kawasaki.jp", 2),
        ("foo.kawasaki.jp", 3),
        # No rule matches -> implicit '*' rule.
        ("localhost", 1),
    ],
)
def test_public_suffix_labels(host: str, expected_suffix_labels: int) -> None:
    assert public_suffix_labels(host) == expected_suffix_labels


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("example.com", ("", "example.com", "com")),
        ("www.bbc.co.uk", ("www", "bbc.co.uk", "co.uk")),
        ("a.b.example.com", ("a.b", "example.com", "com")),
        ("paypal.com.evil.tk", ("paypal.com", "evil.tk", "tk")),
        # A host that IS a public suffix has no registrable domain.
        ("co.uk", ("", "", "co.uk")),
    ],
)
def test_split_host(host: str, expected: tuple[str, str, str]) -> None:
    assert split_host(host) == expected


def test_split_host_of_empty_string() -> None:
    assert split_host("") == ("", "", "")


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("192.168.1.1", True),
        ("255.255.255.255", True),
        ("256.1.1.1", False),
        ("1.2.3", False),
        ("example.com", False),
    ],
)
def test_is_ipv4_literal(host: str, expected: bool) -> None:
    assert is_ipv4_literal(host) is expected


def test_is_ip_literal_covers_v6() -> None:
    assert is_ip_literal("::1") is True
    assert is_ip_literal("2001:db8::7") is True
    assert is_ip_literal("example.com") is False


def test_psl_agrees_with_tldextract_on_a_real_sample() -> None:
    """Cross-check the hand-written lookup against an independent implementation.

    ``tldextract`` is an oracle here, not a dependency of the pipeline: if the
    two ever disagree on a real host, that is a bug worth failing the build for.
    """
    tldextract = pytest.importorskip("tldextract")
    extractor = tldextract.TLDExtract(suffix_list_urls=())

    sample = [
        "example.com",
        "www.bbc.co.uk",
        "shop.example.com.co",
        "a.b.c.example.org",
        "foo.kawasaki.jp",
        "city.kawasaki.jp",
        "test.blogspot.com",
        "sub.github.io",
        "example.museum",
        "deep.sub.domain.example.co.jp",
    ]

    for host in sample:
        _, registrable, suffix = split_host(host)
        reference = extractor(host)
        assert suffix == reference.suffix, f"suffix mismatch for {host}"
        assert registrable == reference.top_domain_under_public_suffix, (
            f"eTLD+1 mismatch for {host}"
        )


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("user.github.io", True),
        ("myapp.vercel.app", True),
        ("blog.blogspot.com", True),
        ("example.com", False),
        ("bbc.co.uk", False),
        ("", False),
    ],
)
def test_is_platform_hosted(host: str, expected: bool) -> None:
    """Provider-hosted sites are a feature, never a grouping key.

    Grouping by them would let the model be rewarded for memorising a hosting
    provider; see the module docstring of ``phishurl.psl``.
    """
    assert is_platform_hosted(host) is expected


def test_grouping_uses_icann_so_one_platform_is_one_group() -> None:
    _, group_a, _ = split_host("evil.vercel.app")
    _, group_b, _ = split_host("nice.vercel.app")
    assert group_a == group_b == "vercel.app"

    # Under the private ruleset they would have been separate groups, which is
    # exactly what the split must avoid.
    assert split_host("evil.vercel.app", "all")[1] == "evil.vercel.app"
