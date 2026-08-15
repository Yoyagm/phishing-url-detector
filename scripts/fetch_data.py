"""Download every raw input the pipeline needs.

Nothing under ``data/raw/`` is committed. The four sources carry different and
in some cases restrictive redistribution terms — OpenPhish's community feed in
particular — so the repository ships the *recipe* rather than the data. This
also keeps the clone small and makes the provenance of every byte explicit.

Idempotent: files already present are left alone unless ``--force`` is passed.

Usage::

    uv run python scripts/fetch_data.py
    uv run python scripts/fetch_data.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXTERNAL = RAW / "external"

# The snapshot date is part of the filename on purpose: the phishing feeds are
# live and rotate within hours, so a result is only reproducible against a
# dated capture. 2026-08-14 is the capture this project's numbers come from.
SNAPSHOT = "2026-08-14"

SOURCES: list[tuple[Path, str, str]] = [
    (
        RAW / "PhiUSIIL_Phishing_URL_Dataset.csv.zip",
        "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip",
        "PhiUSIIL (UCI id 967) — 235,795 labelled URLs, the training source",
    ),
    (
        EXTERNAL / f"openphish-{SNAPSHOT}.txt",
        "https://openphish.com/feed.txt",
        "OpenPhish community feed — ~500 live phishing URLs, fully independent probe",
    ),
    (
        EXTERNAL / f"phishtank-{SNAPSHOT}.csv.gz",
        "https://data.phishtank.com/data/online-valid.csv.gz",
        "PhishTank verified dump — carries submission_time, enables the temporal holdout",
    ),
    (
        EXTERNAL / f"tranco-{SNAPSHOT}.csv.zip",
        "https://tranco-list.eu/top-1m.csv.zip",
        "Tranco top 1M — benign proxy for out-of-distribution evaluation",
    ),
    (
        ROOT / "data" / "public_suffix_list.dat",
        "https://publicsuffix.org/list/public_suffix_list.dat",
        "Public Suffix List — frozen snapshot shared by the Python and JS code",
    ),
]


def fetch(destination: Path, url: str, description: str, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        size_mb = destination.stat().st_size / 1e6
        print(f"  present  {destination.name}  ({size_mb:.1f} MB)  — {description}")
        return

    print(f"  fetching {destination.name}  — {description}")
    try:
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
            payload = response.read()
    except Exception as error:  # noqa: BLE001
        print(f"  FAILED   {destination.name}: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    destination.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()[:16]
    print(f"           {len(payload)/1e6:.1f} MB  sha256:{digest}…")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    print(f"Fetching raw inputs into {RAW.relative_to(ROOT)}/  (today: {date.today().isoformat()})")
    for destination, url, description in SOURCES:
        fetch(destination, url, description, args.force)

    print(
        "\nNote: the phishing feeds are live. Re-running on a later date yields a "
        f"different capture than the {SNAPSHOT} one the published metrics were computed on."
    )


if __name__ == "__main__":
    main()
