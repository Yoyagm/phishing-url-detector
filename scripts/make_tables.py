"""Generate the report's tables from the actual run outputs.

Every number in ``paper/report.pdf`` comes through here, so a table cannot drift
away from the run that produced it: change the pipeline, re-run, and the paper
changes with it. Nothing in the LaTeX source is typed by hand.

Also recomputes the Python↔JavaScript parity deltas by invoking the real browser
engine under Node, so the parity table in the paper reports measured values rather
than the tolerance the tests happen to allow.

Run::

    uv run python scripts/make_tables.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np

from phishurl.datasets import ROOT

warnings.filterwarnings("ignore")

REPORTS = ROOT / "reports"
GENERATED = ROOT / "paper" / "generated"


def thousands(text: str) -> str:
    """Group digits without touching LaTeX's ``\\,`` or ``\\%`` commands.

    A blunt ``str.replace(",", "{,}")`` turns ``\\,\\%`` into ``\\{,}\\%`` and the
    document stops compiling, so the substitution is anchored between digits.
    """
    return re.sub(r"(?<=\d),(?=\d)", "{,}", text)


def percent(value: float) -> str:
    return f"{100 * value:.2f}\\,\\%"


def write(name: str, body: str) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / f"{name}.tex").write_text(body.strip() + "\n")
    print(f"  paper/generated/{name}.tex")


def table_leak_profile(audit: dict) -> None:
    profile = audit["leak_2_collection_artefact"]["profile"]
    legitimate, phishing = profile["legitimate"], profile["phishing"]
    rows = [
        ("Scheme is HTTPS", "starts\\_https_pct", "starts_https_pct"),
        ("Host starts with \\texttt{www.}", "", "host_starts_www_pct"),
        ("Has a path", "", "has_path_pct"),
        ("Has a query string", "", "has_query_pct"),
        ("Contains \\texttt{@}", "", "has_at_pct"),
    ]
    lines = [
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Trait & Legitimate & Phishing \\", r"\midrule",
    ]
    for label, _, key in rows:
        lines.append(f"{label} & {legitimate[key]:.2f}\\,\\% & {phishing[key]:.2f}\\,\\% \\\\")
    lines.append(
        f"Mean URL length & {legitimate['mean_url_length']:.1f} & "
        f"{phishing['mean_url_length']:.1f} \\\\"
    )
    lines += [r"\midrule",
              f"Rows & {int(legitimate['n']):,} & {int(phishing['n']):,} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write("table_leak_profile", thousands("\n".join(lines)))


def table_leak_models(audit: dict) -> None:
    accuracy = audit["lightgbm_random_split_accuracy"]
    rows = [
        ("All 50 precomputed features", accuracy["all_precomputed_features"]),
        ("Minus \\texttt{URLSimilarityIndex}", accuracy["without_url_similarity_index"]),
        ("Minus page-content columns (21 features)", accuracy["without_content_columns"]),
        ("Hand-rolled lexical features on the raw URL", accuracy["naive_raw_url_lexical"]),
    ]
    lines = [r"\begin{tabular}{lr}", r"\toprule",
             r"Feature set & Accuracy \\", r"\midrule"]
    for label, value in rows:
        lines.append(f"{label} & {value:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_leak_models", "\n".join(lines))


def table_results(metrics: dict) -> None:
    internal, ood = metrics["internal"], metrics["ood"]
    rows = [
        ("Hosts evaluated", f"{internal['n']:,}", f"{ood['n']:,}"),
        ("Of which phishing", f"{internal['n_phishing']:,}", f"{ood['n_phishing']:,}"),
        ("ROC-AUC", f"{internal['roc_auc']:.4f}", f"{ood['roc_auc']:.4f}"),
        ("PR-AUC", f"{internal['pr_auc']:.4f}", f"{ood['pr_auc']:.4f}"),
        ("Recall @ FPR 1\\,\\%", percent(internal["recall_at_fpr_1pct"]),
         percent(ood["recall_at_fpr_1pct"])),
        ("Recall @ FPR 0.1\\,\\%", percent(internal["recall_at_fpr_01pct"]),
         percent(ood["recall_at_fpr_01pct"])),
        ("Precision at operating point", percent(internal["precision_at_operating_point"]),
         percent(ood["precision_at_operating_point"])),
        ("$F_1$ at operating point", f"{internal['f1_at_operating_point']:.4f}",
         f"{ood['f1_at_operating_point']:.4f}"),
        ("False positives",
         f"{internal['confusion']['fp']:,} / {internal['confusion']['fp'] + internal['confusion']['tn']:,}",
         f"{ood['confusion']['fp']:,} / {ood['confusion']['fp'] + ood['confusion']['tn']:,}"),
    ]
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"& Internal hold-out & OOD + temporal \\", r"\midrule"]
    for label, left, right in rows:
        lines.append(f"{label} & {left} & {right} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_results", thousands("\n".join(lines)))


def table_ood_composition(metrics: dict) -> None:
    stats = metrics["ood_composition"]
    rows = [
        ("Tranco hosts considered", stats["tranco_hosts"]),
        ("PhishTank hosts (all)", stats["phishtank_hosts_total"]),
        ("PhishTank hosts after the cutoff", stats["phishtank_hosts_after_cutoff"]),
        ("Removed: benign already seen in training", stats["removed_benign_seen_in_training"]),
        ("Removed: phishing already seen in training", stats["removed_phishing_seen_in_training"]),
        ("Removed: host on both a benign and a phishing list",
         stats["removed_conflicting_both_lists"]),
        ("Final benign", stats["final_benign"]),
        ("Final phishing", stats["final_phishing"]),
        ("OpenPhish probe hosts", stats["openphish_hosts"]),
    ]
    lines = [r"\begin{tabular}{lr}", r"\toprule", r"& Hosts \\", r"\midrule"]
    for label, value in rows:
        lines.append(f"{label} & {value:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table_ood_composition", thousands("\n".join(lines)))


def table_parity() -> dict:
    """Re-measure Python vs JavaScript agreement using the real engine."""
    import lightgbm as lgb
    import shap

    from phishurl.canonicalize import canonical_host
    from phishurl.features import extract

    node = shutil.which("node") or str(Path.home() / ".local/node/bin/node")
    corpus = ROOT / "tests" / "parity_corpus.txt"

    with tempfile.TemporaryDirectory() as workspace:
        output = Path(workspace) / "js.json"
        subprocess.run(
            [node, "tests/parity_runner.mjs", "tests/parity_corpus.txt", str(output)],
            cwd=ROOT, check=True, capture_output=True,
        )
        javascript = json.loads(output.read_text())

    booster = lgb.Booster(model_file=str(ROOT / "models" / "model.txt"))
    hosts = [entry["host"] for entry in javascript["results"]]
    matrix = np.asarray([extract(host) for host in hosts])

    js_features = np.asarray([entry["features"] for entry in javascript["results"]])
    js_raw = np.asarray([entry["raw_score"] for entry in javascript["results"]])
    js_shap = np.asarray([entry["shap"] for entry in javascript["results"]])

    python_raw = booster.predict(matrix, raw_score=True)
    python_shap = np.asarray(shap.TreeExplainer(booster).shap_values(matrix))
    if python_shap.ndim == 3:
        python_shap = python_shap[:, :, -1]

    host_mismatches = sum(
        canonical_host(entry["url"]) != entry["host"] for entry in javascript["results"]
    )
    additivity = np.abs(
        javascript["base_score"] + js_shap.sum(axis=1) - js_raw
    ).max()

    measured = {
        "n_urls": len(hosts),
        "host_mismatches": int(host_mismatches),
        "feature_values": int(matrix.size),
        "feature_max_delta": float(np.abs(matrix - js_features).max()),
        "raw_max_delta": float(np.abs(python_raw - js_raw).max()),
        "shap_values": int(python_shap.size),
        "shap_max_delta": float(np.abs(python_shap - js_shap).max()),
        "additivity_max_delta": float(additivity),
    }
    (REPORTS / "parity.json").write_text(json.dumps(measured, indent=2))

    def science(value: float) -> str:
        return "0" if value == 0 else f"{value:.1e}".replace("e-", r"\times 10^{-") + "}"

    lines = [
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Compared & Count & Max.\ abs.\ difference \\", r"\midrule",
        f"Canonical hosts & {measured['n_urls']:,} & "
        f"{'exact' if not host_mismatches else host_mismatches} \\\\",
        f"Feature values & {measured['feature_values']:,} & "
        f"${science(measured['feature_max_delta'])}$ \\\\",
        f"Raw scores & {measured['n_urls']:,} & ${science(measured['raw_max_delta'])}$ \\\\",
        f"SHAP attributions & {measured['shap_values']:,} & "
        f"${science(measured['shap_max_delta'])}$ \\\\",
        f"SHAP additivity in JS & {measured['n_urls']:,} & "
        f"${science(measured['additivity_max_delta'])}$ \\\\",
        r"\bottomrule", r"\end{tabular}",
    ]
    write("table_parity", thousands("\n".join(lines)))
    return measured


def macros(metrics: dict, audit: dict, parity: dict) -> None:
    """Numbers referenced inline in the prose, so those cannot drift either."""
    ood, internal = metrics["ood"], metrics["internal"]
    definitions = {
        "LeakOneAcc": f"{audit['leak_1_url_similarity_index']['single_rule']['accuracy']:.4f}",
        "LeakTwoAcc": f"{audit['leak_2_collection_artefact']['has_path_rule']['accuracy']:.4f}",
        "NaiveAcc": f"{audit['lightgbm_random_split_accuracy']['naive_raw_url_lexical']:.4f}",
        "FullAcc": f"{audit['lightgbm_random_split_accuracy']['all_precomputed_features']:.4f}",
        "OodAuc": f"{ood['roc_auc']:.4f}",
        "OodRecall": f"{100 * ood['recall_at_fpr_1pct']:.1f}",
        "InternalRecall": f"{100 * internal['recall_at_fpr_1pct']:.1f}",
        "OodThreshold": f"{ood['threshold_at_fpr_1pct']:.4f}",
        "OpenphishRate": f"{100 * metrics['openphish_detection_rate_at_ood_threshold']:.1f}",
        "TrainHosts": thousands(f"{metrics['training']['unique_hosts']:,}"),
        "TrainGroups": thousands(f"{metrics['training']['registrable_domains']:,}"),
        "NumFeatures": str(metrics["training"]["n_features"]),
        "NumTrees": str(metrics["training"]["num_boost_round"]),
        "ParityUrls": str(parity["n_urls"]),
        "ParityShapN": thousands(f"{parity['shap_values']:,}"),
    }
    body = "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in definitions.items())
    write("macros", body)


def main() -> None:
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    audit = json.loads((REPORTS / "leak_audit.json").read_text())

    print("Generating tables from real run outputs:")
    table_leak_profile(audit)
    table_leak_models(audit)
    table_results(metrics)
    table_ood_composition(metrics)
    parity = table_parity()
    macros(metrics, audit, parity)
    print("\nEvery figure and table in the report now traces to reports/*.json.")


if __name__ == "__main__":
    main()
