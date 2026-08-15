"""The guard that keeps the whole study honest.

PhiUSIIL's legitimate rows are uniformly ``https://www.<domain>`` with no path,
no query and no ``@`` — a collection artefact, not a property of safe sites. A
feature that reads any of those parts would separate the two collection pipelines
and score ~99 % while learning nothing about phishing.

So: mutating a URL *anywhere outside the host* must leave the feature vector
bit-identical. If someone later adds a feature that peeks at the path, this file
fails and says why.
"""

from __future__ import annotations

import pytest

from phishurl.canonicalize import canonical_host
from phishurl.features import FEATURE_NAMES, extract

# Mutations that must be invisible to the model. Each entry rebuilds the same host
# with a different scheme, prefix, port, path or query.
MUTATIONS = [
    "https://{host}",
    "http://{host}",
    "https://www.{host}",
    "http://www.{host}/",
    "https://{host}/login/verify.php",
    "https://{host}/?next=/account&id=99",
    "https://{host}:8443/a/b/c?d=e#f",
    "https://user:pass@{host}/secure",
    "//{host}/path",
    "{host}",
    "{host}/deep/path/that/should/not/matter?q=1",
]

HOSTS = [
    "example.com",
    "bbc.co.uk",
    "paypal.com.evil.tk",
    "login-secure-update.xyz",
    "a1b2c3.vercel.app",
    "xn--80ak6aa92e.com",
    "192.168.1.1",
    "deep.sub.domain.example.co.jp",
]


@pytest.mark.parametrize("host", HOSTS)
def test_features_ignore_everything_outside_the_host(host: str) -> None:
    baseline = extract(canonical_host(f"https://{host}"))

    for template in MUTATIONS:
        mutated_url = template.format(host=host)
        mutated = extract(canonical_host(mutated_url))
        assert mutated == baseline, (
            f"Feature vector changed when only the non-host part changed.\n"
            f"  host     : {host}\n"
            f"  mutation : {mutated_url}\n"
            f"  differing: "
            + ", ".join(
                name
                for name, before, after in zip(FEATURE_NAMES, baseline, mutated)
                if before != after
            )
        )


def test_the_guard_would_actually_catch_a_leak() -> None:
    """A negative control: prove the assertion above can fail.

    A guard that cannot fail is worthless, so this simulates the mistake it is
    meant to catch — a feature reading the raw URL instead of the host — and
    checks that the two vectors do diverge.
    """

    def leaky_extract(raw_url: str) -> list[float]:
        host_features = extract(canonical_host(raw_url))
        # The leak: "does the URL have a path?", which is 0% for every legitimate
        # row in PhiUSIIL and 28% of phishing rows.
        rest = raw_url.split("://", 1)[-1]
        has_path = 1.0 if "/" in rest and rest.split("/", 1)[1] else 0.0
        return [*host_features, has_path]

    clean = leaky_extract("https://example.com")
    with_path = leaky_extract("https://example.com/login/verify.php")
    assert clean != with_path, "the negative control failed to detect a planted leak"


def test_feature_vector_length_matches_names() -> None:
    assert len(extract("example.com")) == len(FEATURE_NAMES)
    assert len(extract("")) == len(FEATURE_NAMES)
