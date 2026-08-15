"""Public Suffix List lookup.

Implemented from scratch, deliberately, against a frozen snapshot of the list
(``data/public_suffix_list.dat``). Two reasons:

1. ``tldextract`` fetches the list at run time, so two machines — or the Python
   trainer and the JavaScript demo — can silently disagree about what the
   registrable domain of a host is. A committed snapshot removes that.
2. Every function here has a line-for-line twin in ``docs/app.js``. Keeping the
   algorithm small and dependency-free is what makes that mirror auditable.

``tldextract`` is still used, but only as a test oracle (see
``tests/test_canonicalize.py``).

**The ICANN/private split matters here more than in most projects.** The list has
two sections: ICANN suffixes (``com``, ``co.uk``) and private ones registered by
hosting providers (``blogspot.com``, ``vercel.app``, ``github.io``). Phishing is
routinely hosted on those platforms — the OpenPhish sample captured for this
project contains ``*.vercel.app`` URLs — so the choice changes the experiment:

* **Grouping for the train/test split uses ICANN suffixes only.** That puts every
  ``*.vercel.app`` host into a single group, so the splitter is forced to send
  them all to one side. Otherwise the model could memorise "vercel.app means
  phishing" from training rows and be rewarded for it at test time.
* **The private section becomes a feature instead** (:func:`is_platform_hosted`),
  which is where that signal belongs: informative, and unable to inflate the score.

Reference: https://github.com/publicsuffix/list/wiki/Format
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

DEFAULT_PSL_PATH = Path(__file__).resolve().parents[2] / "data" / "public_suffix_list.dat"

_PRIVATE_MARKER = "// ===BEGIN PRIVATE DOMAINS==="


@lru_cache(maxsize=4)
def load_rules(path: str | Path = DEFAULT_PSL_PATH) -> tuple[frozenset[str], frozenset[str]]:
    """Parse the PSL into ``(icann_rules, all_rules)``.

    Exception rules keep their leading ``!`` and wildcard rules keep their ``*``
    label, so one set lookup distinguishes all three rule kinds.
    """
    icann: set[str] = set()
    everything: set[str] = set()
    in_private = False

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(_PRIVATE_MARKER):
                in_private = True
                continue
            if not line or line.startswith("//"):
                continue
            everything.add(line)
            if not in_private:
                icann.add(line)

    if not in_private:  # pragma: no cover - guards against a malformed snapshot
        raise ValueError(f"No private-domain marker found in {path}; snapshot looks truncated")

    return frozenset(icann), frozenset(everything)


def _rules_for(section: str) -> frozenset[str]:
    icann, everything = load_rules()
    if section == "icann":
        return icann
    if section == "all":
        return everything
    raise ValueError(f"section must be 'icann' or 'all', got {section!r}")


def public_suffix_labels(host: str, section: str = "icann") -> int:
    """Number of trailing labels of ``host`` forming its public suffix.

    Follows the PSL algorithm: an exception rule wins outright, otherwise the
    rule matching the most labels prevails, and a host matching no rule at all
    falls back to the implicit ``*`` rule of one label.
    """
    if not host:
        return 0

    rules = _rules_for(section)
    labels = host.split(".")
    n = len(labels)
    best = 0

    for i in range(n):
        candidate = ".".join(labels[i:])
        # `i` ascends, so the first exception we meet is the longest match.
        if "!" + candidate in rules:
            return n - i - 1
        if candidate in rules:
            best = max(best, n - i)
        wildcard = ".".join(["*"] + labels[i + 1 :])
        if wildcard in rules:
            best = max(best, n - i)

    return best if best else 1


def split_host(host: str, section: str = "icann") -> tuple[str, str, str]:
    """Split ``host`` into ``(subdomain, registrable_domain, public_suffix)``.

    A host that is itself a public suffix (``co.uk``) has no registrable domain;
    it comes back as the suffix with the other two parts empty. Callers must
    handle that — features degrade rather than raise.
    """
    if not host:
        return "", "", ""

    labels = host.split(".")
    n = len(labels)
    n_suffix = public_suffix_labels(host, section)

    if n_suffix >= n:
        return "", "", host

    suffix = ".".join(labels[n - n_suffix :])
    registrable = ".".join(labels[n - n_suffix - 1 :])
    subdomain = ".".join(labels[: n - n_suffix - 1])
    return subdomain, registrable, suffix


def is_platform_hosted(host: str) -> bool:
    """True when ``host`` sits under a provider-registered suffix.

    ``user.github.io`` and ``app.vercel.app`` are True; ``example.com`` is False.
    Detected as a longer suffix under the private ruleset than under ICANN's.
    """
    if not host:
        return False
    return public_suffix_labels(host, "all") > public_suffix_labels(host, "icann")
