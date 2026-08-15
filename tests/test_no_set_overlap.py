"""Guards on the composition of the evaluation sets.

An out-of-distribution set is only out of distribution if the model has not seen
it. Three ways that quietly stops being true, all checked here:

* benign hosts that were already in training (measured: 42,005 of Tranco's hosts
  overlap PhiUSIIL, so ~42 % of an unfiltered 100k benign sample would be memorised);
* phishing submitted before the training data was collected, which is not a
  temporal test at all;
* hosts sitting on both a benign and a phishing list — compromised legitimate
  sites, whose label is genuinely ambiguous.

These tests read the real data and take a few seconds. That is the point: the
invariant is worth more than the speed.
"""

from __future__ import annotations

import pytest

from phishurl.datasets import PHIUSIIL_CUTOFF, build_ood_frame, load_phishtank

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def sets():
    train_hosts = {"example.com", "google.com", "github.com"}
    return build_ood_frame(train_hosts, benign_limit=5_000), train_hosts


def test_no_training_host_reaches_the_ood_set(sets) -> None:
    ood, train_hosts = sets
    assert not set(ood.frame["host"]) & train_hosts
    assert not set(ood.openphish["host"]) & train_hosts


def test_no_host_carries_both_labels(sets) -> None:
    ood, _ = sets
    benign = set(ood.frame.loc[ood.frame["y"] == 0, "host"])
    phishing = set(ood.frame.loc[ood.frame["y"] == 1, "host"])
    assert not benign & phishing, "a host is labelled both benign and phishing"


def test_ood_phishing_is_strictly_after_the_training_cutoff() -> None:
    """Every phishing host in the OOD set must post-date the training data."""
    phishtank = load_phishtank()
    recent = phishtank[phishtank["submission_time"] > PHIUSIIL_CUTOFF]

    assert len(recent) > 10_000, "temporal holdout unexpectedly small"
    assert recent["submission_time"].min() > PHIUSIIL_CUTOFF
    # Sanity: the cutoff actually removes a meaningful slice, so it is doing work.
    assert len(recent) < len(phishtank)


def test_both_classes_are_present_and_non_trivial(sets) -> None:
    ood, _ = sets
    counts = ood.frame["y"].value_counts()
    assert counts.get(0, 0) > 1_000
    assert counts.get(1, 0) > 1_000


def test_removal_counts_are_reported(sets) -> None:
    """The audit trail must be non-empty, or the filters silently did nothing."""
    ood, _ = sets
    assert ood.stats["removed_conflicting_both_lists"] > 0
    assert ood.stats["phishtank_hosts_after_cutoff"] < ood.stats["phishtank_hosts_total"]
