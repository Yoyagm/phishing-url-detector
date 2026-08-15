"""Loading and assembly of every dataset the project uses.

Two conventions are established here and relied on everywhere else:

* **``y = 1`` means phishing.** PhiUSIIL ships the opposite (``label = 1`` is
  legitimate), which is an easy way to invert an entire study by accident, so the
  flip happens once, here, and is asserted in the tests.
* **Rows are host names, not URLs.** Everything passes through
  :func:`phishurl.canonicalize.canonical_host` on the way in. The reasons are
  measured and documented in ``scripts/leak_audit.py``.

The out-of-distribution assembly also removes the contamination between sources
that would otherwise flatter the results: hosts already seen in training, and the
hosts that appear on both a benign and a phishing list.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from phishurl.canonicalize import canonical_host
from phishurl.psl import split_host

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
EXTERNAL = RAW / "external"
SNAPSHOT = "2026-08-14"

# Upper bound on PhiUSIIL collection: the date on the UCI archive. PhishTank rows
# submitted after it are genuinely unseen by the model. 85.2% of the dump qualifies.
PHIUSIIL_CUTOFF = "2024-03-04"


def _phiusiil_path() -> Path:
    """Return the CSV, extracting it from the UCI zip on first use."""
    csv_path = RAW / "PhiUSIIL_Phishing_URL_Dataset.csv"
    if csv_path.exists():
        return csv_path

    zip_path = RAW / "PhiUSIIL_Phishing_URL_Dataset.csv.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Neither {csv_path.name} nor {zip_path.name} found in {RAW}. "
            "Run: uv run python scripts/fetch_data.py"
        )
    with zipfile.ZipFile(zip_path) as archive:
        member = next(n for n in archive.namelist() if n.endswith(".csv"))
        csv_path.write_bytes(archive.read(member))
    return csv_path


def load_phiusiil() -> pd.DataFrame:
    """Load PhiUSIIL with the label flipped so ``y = 1`` is phishing.

    Read with pandas rather than a naive split: the file carries a UTF-8 BOM and
    its ``Title`` column contains mojibake with embedded commas and quotes.
    """
    frame = pd.read_csv(_phiusiil_path(), encoding="utf-8-sig", low_memory=False)
    frame["y"] = 1 - frame["label"].astype(int)
    return frame


def add_host_columns(frame: pd.DataFrame, url_column: str = "URL") -> pd.DataFrame:
    """Attach canonical host, registrable domain (ICANN grouping) and suffix."""
    frame = frame.copy()
    frame["host"] = frame[url_column].map(canonical_host)
    split = frame["host"].map(split_host)
    frame["registrable_domain"] = [item[1] for item in split]
    frame["public_suffix"] = [item[2] for item in split]
    return frame


def build_training_frame() -> pd.DataFrame:
    """PhiUSIIL reduced to unique hosts, ready for a grouped split.

    Hosts appearing with both labels are dropped rather than arbitrated: keeping
    either copy would teach the model a contradiction.
    """
    frame = add_host_columns(load_phiusiil())
    frame = frame[frame["host"].str.len() > 0]

    per_host = frame.groupby("host", as_index=False).agg(
        y=("y", "max"),
        y_min=("y", "min"),
        registrable_domain=("registrable_domain", "first"),
        public_suffix=("public_suffix", "first"),
    )
    conflicted = per_host["y"] != per_host["y_min"]
    per_host = per_host[~conflicted].drop(columns="y_min")

    # A host that is itself a public suffix has no registrable domain; group it
    # by itself so the splitter still has a key.
    empty = per_host["registrable_domain"] == ""
    per_host.loc[empty, "registrable_domain"] = per_host.loc[empty, "host"]

    per_host.attrs["n_conflicting_hosts"] = int(conflicted.sum())
    return per_host.reset_index(drop=True)


def load_tranco(limit: int | None = None) -> pd.DataFrame:
    """Tranco top-1M as canonical hosts.

    The archive ships CRLF line endings; not stripping them makes every later set
    intersection silently return zero.
    """
    path = EXTERNAL / f"tranco-{SNAPSHOT}.csv.zip"
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.endswith(".csv"))
        with archive.open(member) as handle:
            frame = pd.read_csv(
                io.TextIOWrapper(handle, encoding="utf-8"),
                names=["rank", "domain"],
                nrows=limit,
            )
    frame["domain"] = frame["domain"].str.strip()
    frame["host"] = frame["domain"].map(canonical_host)
    return frame[frame["host"].str.len() > 0]


def load_phishtank() -> pd.DataFrame:
    """PhishTank verified dump, with submission time parsed."""
    path = EXTERNAL / f"phishtank-{SNAPSHOT}.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        frame = pd.read_csv(handle, low_memory=False)
    frame["host"] = frame["url"].map(canonical_host)
    frame["submission_time"] = frame["submission_time"].astype(str)
    return frame[frame["host"].str.len() > 0]


def load_openphish() -> pd.DataFrame:
    """OpenPhish live feed — a small, fully independent phishing probe."""
    path = EXTERNAL / f"openphish-{SNAPSHOT}.txt"
    urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame({"url": urls})
    frame["host"] = frame["url"].map(canonical_host)
    return frame[frame["host"].str.len() > 0]


@dataclass
class OODSets:
    """Out-of-distribution evaluation sets, plus the audit trail of what was removed."""

    frame: pd.DataFrame
    openphish: pd.DataFrame
    stats: dict[str, int]


def build_ood_frame(train_hosts: set[str], benign_limit: int = 100_000) -> OODSets:
    """Assemble the OOD/temporal evaluation set.

    Three removals, each of which would otherwise inflate the score:

    1. Benign hosts already seen in training (30.4% of Tranco's overlap with
       PhiUSIIL) — otherwise a third of the "unseen" benign set is memorised.
    2. Phishing hosts submitted on or before :data:`PHIUSIIL_CUTOFF`, which are
       not temporally out of sample.
    3. Hosts appearing on both a benign and a phishing list — compromised
       legitimate sites. Real label noise; excluded and counted, not hidden.
    """
    tranco = load_tranco()
    phishtank = load_phishtank()

    tranco_hosts = set(tranco["host"])
    recent = phishtank[phishtank["submission_time"] > PHIUSIIL_CUTOFF]
    phish_hosts = set(recent["host"])

    conflicting = phish_hosts & tranco_hosts
    seen_benign = tranco_hosts & train_hosts
    seen_phish = phish_hosts & train_hosts

    # Deduplicate explicitly: canonicalisation strips "www.", so two distinct Tranco
    # rows can collapse onto one host. Counting it twice would quietly weight part
    # of the benign set double.
    seen: set[str] = set()
    benign_pool: list[str] = []
    for host in tranco["host"]:
        if host in train_hosts or host in conflicting or host in seen:
            continue
        seen.add(host)
        benign_pool.append(host)

    benign_hosts = benign_pool[:benign_limit]
    final_phish = sorted(phish_hosts - conflicting - train_hosts - set(benign_hosts))

    frame = pd.DataFrame(
        {"host": list(benign_hosts) + final_phish,
         "y": [0] * len(benign_hosts) + [1] * len(final_phish)}
    )
    if frame["host"].duplicated().any():  # pragma: no cover - guarded above
        raise ValueError("duplicate hosts reached the OOD set; the metrics would be weighted wrong")
    split = frame["host"].map(split_host)
    frame["registrable_domain"] = [item[1] or host for item, host in zip(split, frame["host"])]

    openphish = load_openphish()
    openphish = openphish[~openphish["host"].isin(train_hosts)].copy()
    openphish["y"] = 1

    stats = {
        "tranco_hosts": len(tranco_hosts),
        "phishtank_hosts_total": phishtank["host"].nunique(),
        "phishtank_hosts_after_cutoff": len(phish_hosts),
        "removed_benign_seen_in_training": len(seen_benign),
        "removed_phishing_seen_in_training": len(seen_phish),
        "removed_conflicting_both_lists": len(conflicting),
        "final_benign": len(benign_hosts),
        "final_phishing": len(final_phish),
        "openphish_hosts": int(openphish["host"].nunique()),
    }
    return OODSets(frame=frame, openphish=openphish, stats=stats)
