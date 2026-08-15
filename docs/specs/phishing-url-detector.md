# Explainable Phishing URL Detection — Design Specification

> Written in English because this file ships in the public repository.
> Status: **delivered**. Written 2026-08-14, completed the same day.
> All milestones closed; see "Outcome" at the end for what changed against this design.

## Problem

Published results on public phishing-URL benchmarks routinely report 99–100 % accuracy.
On PhiUSIIL (UCI id 967, 235 795 URLs), the most widely cited recent dataset, that number
is an artifact of the data, not a property of any model. Two independent leaks were measured
on the raw file before any modelling (see `scripts/leak_audit.py`, which reproduces both):

1. **`URLSimilarityIndex` is a label proxy.** It equals exactly `100.000` for **134 850 / 134 850**
   legitimate rows (100 %) and for only 786 / 100 945 phishing rows. The single rule
   `URLSimilarityIndex >= 99.9 → legitimate` scores **99.67 % accuracy** on its own.
2. **The legitimate class is a collection artifact.** Every legitimate URL is of the form
   `https://www.<domain>` — 100 % HTTPS, 100 % `www.`, **0.00 %** with a path, **0.00 %** with a
   query, **0.00 %** containing `@`. Phishing rows were dumped raw from feeds (48.7 % HTTPS,
   28.4 % with a path). This leak **survives discarding all 54 precomputed features and
   training from the raw URL string**, which is the obvious rescue plan. A model trained that
   way does not learn phishing; it learns to tell two collection pipelines apart.

The project's deliverable is therefore not "a classifier with a high score". It is an honest,
explainable, deployed host-name classifier, plus the measurement of what honest performance
actually looks like once both leaks are removed and the model is tested against phishing
that appeared **two years after** its training data.

## Scope

### In

- Reproducible audit quantifying both leaks, with committed figures and numbers.
- Canonical **host-only** feature extraction, defined once in Python and mirrored in JavaScript.
- LightGBM classifier trained with a grouped split by registrable domain (eTLD+1).
- Two-regime evaluation: internal grouped hold-out, and **out-of-distribution + temporal**
  evaluation against feeds captured 2026-08-14.
- SHAP explainability: global importance and per-prediction local attribution.
- FastAPI service serving the model, containerised and tested.
- Public static demo on GitHub Pages: scoring **and** exact TreeSHAP computed in the browser,
  with no backend, validated against the Python reference by parity tests.
- English README as landing page; LaTeX technical report compiled to PDF, every table and
  figure generated from a real run.
- Published public repository and a live demo URL.

### Not in

- **HTML/content features** (`LineOfCode`, `HasTitle`, `NoOfImage`, …). They require fetching the
  page, which is impossible when scoring a URL *before* the user visits it, and phishing pages
  are usually dead within days. All 27 such columns are discarded.
- **Network-derived signals**: DNS, WHOIS, TLS certificates, IP reputation, redirect resolution.
  Out of scope by design — this is a lexical detector and its limits are stated as such.
- **ISCX-URL2016** — behind a manual UNB registration form, not automatable.
- Retraining pipelines, model registry, MLOps, databases, authentication.
- Any claim that the artifact replaces a blocklist or a mail gateway.

## Assumptions

| # | Assumption | If wrong |
|---|---|---|
| A1 | `2024-03-04` (the UCI file's date) is a safe upper bound for PhiUSIIL collection; PhishTank rows after it are genuinely unseen. **85.2 % (61 201 rows) qualify**, 34 712 of them from 2026. | Cutoff moves later; holdout shrinks but stays large. |
| A2 | Host-level signal is sufficient for a useful classifier. Supported by measured TLD divergence (`.app` 6.31 % of phishing vs 0 % of legitimate; also `.top`, `.site`, `.link`, `.xyz`). | If AUC collapses, the honest finding is reported as the result. |
| A3 | Tranco top-1M is an acceptable benign proxy after removing the 40 962 hosts already in training and the 321 that also appear in PhishTank. | Benign label noise; documented and bounded. |
| A4 | A frozen Public Suffix List snapshot keeps Python and JS in agreement. | Parity tests fail loudly rather than silently. |
| A5 | Model JSON stays small enough for a static page. **Measured: 3.35 MB** for 300 trees × 31 leaves before pruning. | Reduce trees or prune harder. |

## Design

### Data flow

```
PhiUSIIL ─┐
Tranco   ─┼─→ ingest ─→ canonicalize to host ─→ dedup by eTLD+1 ─→ features (host only)
PhishTank─┤                    │                      │
OpenPhish─┘                    │                      ├─→ grouped split ─→ LightGBM
                               │                      └─→ OOD/temporal sets
                               ▼                                  │
                        leak_audit.py                              ▼
                     (reproduces both leaks)              evaluate + SHAP
                               │                                  │
                               └──────────→ figures ←─────────────┤
                                                                  ▼
                                                  export_model.py → model.json
                                                       ├─→ FastAPI service (real shap)
                                                       └─→ docs/ static demo (JS scorer
                                                            + JS TreeSHAP), parity-tested
```

### Canonicalization — the mandatory first step

Every URL from every source passes through one function before anything else:

```
canonicalize(url) -> host          # lowercase, strip scheme, userinfo, port,
                                   # path, query, fragment, trailing dot, leading "www."
```

This is what neutralises leak 2 **and** the out-of-distribution trap: Tranco ships bare
domains while PhishTank ships deep URLs, so any full-URL comparison would let the model
separate the classes with the trivial rule "has a path → phishing" and report a beautiful,
meaningless AUC — precisely the sin this project exists to expose.

`tests/test_no_host_leakage.py` enforces it: for a corpus of URLs, mutating **only** the
scheme, path, query, fragment, port or `www.` prefix must leave the feature vector
bit-identical. Any feature that peeks outside the host makes the suite fail.

### Public Suffix List

`tldextract` fetches the PSL at runtime, which would let Python and JavaScript drift apart.
Instead a snapshot is committed to `data/public_suffix_list.dat` and a ~40-line lookup
(wildcards `*.ck`, exceptions `!www.ck`, multi-level suffixes `co.uk`, `com.co`) is
implemented in **both** languages. `tldextract` is kept as a test oracle only.

### Features (host only, ~25)

All are pure string operations over the host plus two frozen lexicons, chosen so the
JavaScript mirror is a faithful line-for-line port. Anything needing a heavy Python
dependency was rejected for exactly that reason.

*Shape*: `host_length`, `n_labels`, `n_subdomain_labels`, `registrable_domain_length`,
`subdomain_length`, `tld_length`, `max_label_length`, `mean_label_length`.
*Composition*: `shannon_entropy_host`, `shannon_entropy_registrable`, `digit_count`,
`digit_ratio`, `hyphen_count`, `hyphen_ratio`, `vowel_ratio`, `max_consecutive_digits`,
`max_consecutive_consonants`, `digits_in_registrable`.
*Structure*: `is_ip_literal` (v4/v6), `has_punycode` (`xn--`), `is_multi_level_suffix`.
*TLD*: one-hot over the 40 most frequent TLDs plus `tld_other`. **Deliberately not target-encoded** —
a TLD phishing prior fitted on the full data would reintroduce leakage through the back door.
*Lexical priors*: `has_suspicious_keyword` (frozen list: login, secure, verify, account, update,
confirm, signin, webscr, banking…), `brand_token_present`, and `brand_in_subdomain_not_registrable`
— the `paypal.com.evil.tk` pattern, the highest-value single feature for T1566.002.
The brand vocabulary is frozen from Tranco's top 500 registrable names: label-independent,
so it cannot leak. **PhishTank's `target` column is deliberately not used as a feature source** —
it is derived from the evaluation set and would leak.

### Splitting and evaluation

- **Grouping**: `GroupShuffleSplit` on eTLD+1. URLs sharing a registrable domain never straddle
  the split, otherwise the model memorises domains instead of learning patterns.
- **Internal**: grouped hold-out from PhiUSIIL canonical hosts.
- **OOD / temporal**: benign = Tranco hosts *not* seen in training (the 40 962 overlap removed);
  phishing = PhishTank hosts submitted after 2024-03-04, minus the 321 also in Tranco and the
  25 also in PhiUSIIL's legitimate set — those are compromised legitimate sites hosting phishing,
  real-world label noise, excluded and documented. OpenPhish (300 live URLs) is a second,
  fully independent phishing probe.
- **Primary metric**: **recall at FPR ≤ 1 %**. A false positive blocks a legitimate site, which is
  operationally the expensive error. Also reported: ROC-AUC, PR-AUC, recall@FPR 0.1 %, precision,
  F1, confusion matrix at the chosen threshold.
- **Baselines to beat, all reported side by side**: (i) the `URLSimilarityIndex` leak rule,
  (ii) the "has a path" rule, (iii) TLD-only logistic regression, (iv) the model.
- **Performance is expected to drop out of distribution. That drop is the headline result and
  gets reported, not buried.**

### Model artifact contract (`docs/model.json`)

Pruned from `booster.dump_model()`, keeping only what path-dependent TreeSHAP needs —
verified present in the real dump: `split_feature`, `threshold`, `decision_type`,
`default_left`, `internal_value`, `internal_count`, `left_child`, `right_child`,
`leaf_value`, `leaf_count`.

```json
{ "format_version": 1, "created_utc": "...", "model_sha256": "...",
  "feature_names": [...], "feature_spec_sha256": "...",
  "base_score": <expected_value>, "objective": "binary_sigmoid", "trees": [...] }
```

`feature_spec_sha256` hashes the feature definition; the demo refuses to score if the JS
extractor's hash disagrees, making train/serve skew a loud failure instead of a silent one.

### Browser inference

`docs/app.js` implements three things ported from Python: the PSL lookup, the feature
extractor, and path-dependent TreeSHAP (Lundberg 2018 — `EXTEND` / `UNWIND` / `UNWOUND_SUM`).
Verified against the real stack: `shap.TreeExplainer` defaults to `tree_path_dependent`
with `model_output="raw"`, and additivity `expected_value + Σ shap == raw_score` holds to
**5·10⁻¹⁶**. The parity tolerance is therefore set at **1e-12**, not the 1e-6 originally proposed.

If the JS TreeSHAP port cannot be made to hold 1e-12, the fallback is Saabas path
contributions, shipped **labelled as such** rather than mislabelled as SHAP.

### FastAPI contract

| Method | Path | Behaviour |
|---|---|---|
| `POST` | `/v1/score` | `{"url": str}` → verdict, probability, threshold, model version, per-feature SHAP |
| `POST` | `/v1/score/batch` | ≤ 100 URLs |
| `GET` | `/v1/model` | metadata, feature names, hashes |
| `GET` | `/healthz` | liveness |

`422` invalid body · `400` unparseable URL · `413` payload over limit · URL capped at 2048 chars.

### Repository layout

```
README.md · requirements.txt · Makefile · .gitignore · Dockerfile
data/            raw/ (gitignored, fetched by scripts) · public_suffix_list.dat
src/phishurl/    canonicalize.py features.py datasets.py train.py evaluate.py
                 explain.py export_model.py api.py
scripts/         fetch_data.py leak_audit.py make_figures.py
docs/            index.html app.js model.json  (GitHub Pages root) · specs/
notebooks/       01_eda_and_leakage.ipynb
paper/           report.tex figures/ report.pdf
tests/           test_canonicalize.py test_features.py test_no_host_leakage.py
                 test_parity_features.py test_parity_shap.py test_api.py
```

### Rejected alternatives

- **Gradio / HF Spaces** — requires a PRO subscription since 2026 (402 Payment Required).
- **Streamlit Community Cloud** — sleeps; a recruiter would meet a wake-up screen.
- **ONNX + onnxruntime-web** — multi-MB WASM runtime and no in-browser SHAP; a pruned JSON plus
  a hand-written scorer is smaller, faster and fully auditable.
- **Target-encoded TLD** — reintroduces leakage.
- **Training on raw URLs** — defeated by leak 2.

## Edge cases

| Input | Behaviour |
|---|---|
| No scheme (`example.com/x`) | Parsed as host `example.com`; documented. |
| Userinfo (`https://paypal.com@evil.tk`) | Host is `evil.tk`. Explicit test — this is an attack, not a curiosity. |
| Punycode / IDN (`xn--80ak6aa92e.com`) | Kept encoded; `has_punycode` fires. |
| IPv6 literal (`https://[::1]/`) | `is_ip_literal` fires; label features degrade gracefully. |
| 8000-character host | Rejected at API (`413`); demo truncates and warns. |
| Empty string / not a URL | `400`, never an exception. |
| Multi-level suffix (`bbc.co.uk`) | PSL resolves eTLD+1 correctly; regression-tested. |
| Shortener (`bit.ly/abc`) | Scored on `bit.ly` only. **The demo says explicitly that a lexical detector cannot resolve shorteners** — a known, stated blind spot. |
| Trailing dot (`example.com.`) | Stripped during canonicalization. |
| Mixed case / Unicode | Lowercased; non-ASCII normalised, tested. |

## Test plan

| Level | File | Asserts |
|---|---|---|
| Unit | `test_canonicalize.py` | Every edge case above. |
| Unit | `test_features.py` | Known vectors for hand-computed hosts. |
| **Guard** | `test_no_host_leakage.py` | Mutating scheme/path/query/port/`www.` leaves the vector bit-identical. |
| **Guard** | `test_no_set_overlap.py` | Zero host overlap between train and OOD sets. |
| Integration | `test_parity_features.py` | Python vs JS feature vectors over ≥ 500 URLs, exact. |
| Integration | `test_parity_shap.py` | Python vs JS raw score and SHAP vector, \|Δ\| < 1e-12. |
| Integration | `test_api.py` | Every endpoint, every error code. |
| Regression | `test_leak_audit.py` | The two leak numbers still reproduce. |
| E2E | browser check | Demo page scores a known URL and renders its explanation. |

## Milestones

- [x] **H1 — Data ingestion + canonicalization + frozen PSL.**
      `make data && uv run pytest tests/test_canonicalize.py` — all edge cases green.
- [x] **H2 — Leak audit, both leaks reproduced with numbers and figures.**
      `uv run python scripts/leak_audit.py` prints both tables, writes figures;
      `pytest tests/test_leak_audit.py` asserts the rule reaches ≥ 0.99 accuracy and that
      legitimate-rows-with-path == 0.
- [x] **H3 — Host-only feature extractor + leakage guard.**
      `uv run pytest tests/test_features.py tests/test_no_host_leakage.py`.
- [x] **H4 — Training with grouped split + internal evaluation.**
      `make train` writes `models/model.txt` and `reports/metrics_internal.json`
      containing ROC-AUC, PR-AUC, recall@FPR1 %.
- [x] **H5 — OOD + temporal evaluation with dedup and conflict handling.**
      `make eval-ood` writes `reports/metrics_ood.json`; `pytest tests/test_no_set_overlap.py`.
- [x] **H6 — Model export + JS scorer + JS TreeSHAP + parity.**
      `uv run pytest tests/test_parity_features.py tests/test_parity_shap.py` — ≥ 500 URLs, Δ < 1e-12.
- [x] **H7 — Static demo page in `docs/`.**
      Loads with no network calls, scores a known URL, renders SHAP; screenshot captured.
- [x] **H8 — FastAPI service + Docker.**
      `uv run pytest tests/test_api.py`; `docker build` succeeds and a `curl` to `/v1/score` returns a verdict.
- [x] **H9 — README + LaTeX PDF from real runs + publish repo and Pages.**
      `make paper` produces `paper/report.pdf`; the live Pages URL returns HTTP 200.

## Misuse note

This is a defensive, assistive classifier with a measured error rate, not an authority.
It scores host names only: it cannot resolve shorteners, cannot see page content, and will
miss phishing hosted on compromised legitimate domains — a case shown to exist in the data
(321 PhishTank hosts sit in Tranco's top 1M). The README and the report state this plainly.


---

## Outcome

All nine milestones closed on 2026-08-14. `uv run pytest` reports **113 passed**.
Live demo: <https://yoyagm.github.io/phishing-url-detector/>

### Where the build departed from this design

| Design said | What was built | Why |
|---|---|---|
| Parity tolerance 1e-6 | **1e-12**, and the measured worst case is 8.9e-16 | `shap`'s own additivity residual on this stack measures 5e-16, so 1e-6 was three orders of magnitude too loose to be a real check. |
| ~25 features | **51** | The TLD one-hot expands to 26 columns; the other 25 are as specified. |
| PSL used as one rule set | **ICANN and private sections split** | Grouping under the private section would make `evil.vercel.app` and `nice.vercel.app` separate groups, letting the model be rewarded for memorising a hosting provider. Grouping uses ICANN; the private section became the `is_platform_hosted` feature instead. |
| `explain.py` for SHAP wrappers | Written, and produces global importance | Local explanation lives in `api.py` and `engine.js` where it is used. |
| — | `?url=` and `?theme=` on the demo | Shareable scan links, and the mechanism that captures the report figures reproducibly. |

### Assumptions that turned out to be wrong or imprecise

- **A5 (model size).** Estimated 3.35 MB from an unpruned dump; the pruned artifact is **0.86 MB**.
- **Leak-1 count.** The pre-design estimate of 815 phishing rows at index ≥ 99.9 came from a naive
  `awk` split that mis-parsed quoted fields. pandas gives **786**; accuracy 0.9967, not 0.9965.
- **A2 (host-level signal).** Held: ROC-AUC 0.9372 out of distribution. The design allowed for the
  possibility that it would not.

### Assumptions that held

A1 (temporal cutoff — 61,201 PhishTank rows qualify), A3 (Tranco as benign proxy after removing the
42,005 overlapping hosts), A4 (frozen PSL keeps the two languages in agreement — verified exactly).

### Not delivered

- **Licensing verification of the four data sources was never completed** (the research agent that
  was to do it did not run). Mitigated by choosing the conservative option regardless: no raw data is
  committed, and `scripts/fetch_data.py` fetches everything at build time.
- Multi-model agent delegation, which the environment did not support in this session.
