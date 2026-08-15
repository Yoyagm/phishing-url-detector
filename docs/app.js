/**
 * Demo page wiring: load the artifacts, score on submit, render the explanation.
 *
 * All the arithmetic lives in engine.js — the same file the parity tests run under
 * Node. This file only handles the DOM.
 */

import {
  canonicalHost,
  extract,
  loadPsl,
  predictProbability,
  rawScore,
  treeShap,
} from "./engine.js";

const element = (id) => document.getElementById(id);
const ui = {
  url: element("url"), scan: element("scan"), error: element("error"),
  result: element("result"), chip: element("chip"), prob: element("prob"),
  host: element("host"), shap: element("shap"), features: element("features"),
};

let model = null;
let vocab = null;
let metrics = null;

/** Threshold chosen on the out-of-distribution set at a 1 % false-positive rate. */
let threshold = 0.5;

// The 26 one-hot suffix columns are summed into one row: a user wants "the TLD
// helped" not twenty-six near-zero bars. Summing SHAP values is exact, so the
// aggregate is still a true decomposition.
const isSuffixFeature = (name) => name.startsWith("suffix_is_");

const PRETTY = {
  host_length: "Host length", n_labels: "Number of labels",
  n_subdomain_labels: "Subdomain depth", registrable_length: "Registrable domain length",
  subdomain_length: "Subdomain length", suffix_length: "Suffix length",
  max_label_length: "Longest label", mean_label_length: "Mean label length",
  entropy_host: "Character entropy (host)", entropy_registrable: "Character entropy (domain)",
  digit_count: "Digits", digit_ratio: "Digit ratio", hyphen_count: "Hyphens",
  hyphen_ratio: "Hyphen ratio", vowel_ratio: "Vowel ratio",
  max_consecutive_digits: "Longest digit run", max_consecutive_consonants: "Longest consonant run",
  digits_in_registrable: "Digits in domain", is_ip_literal: "Host is a raw IP address",
  has_punycode: "Punycode (xn--)", is_multi_level_suffix: "Multi-level suffix",
  is_platform_hosted: "Hosted on a platform subdomain",
  has_suspicious_keyword: "Contains a login/verify keyword",
  brand_token_present: "Contains a known brand",
  brand_in_subdomain_only: "Brand in subdomain, not in the domain",
};
const pretty = (name) => PRETTY[name] || name;

function formatValue(value) {
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(2);
}

function render(host, probability, phi, features) {
  const phishing = probability >= threshold;
  ui.chip.className = "chip " + (phishing ? "phish" : "legit");
  ui.chip.textContent = phishing ? "⚠ Likely phishing" : "✓ No phishing signal";
  ui.prob.textContent = (probability * 100).toFixed(1) + "%";
  ui.prob.style.color = phishing ? "var(--red)" : "var(--good)";
  ui.host.textContent = `Scored host: ${host}`;

  // Aggregate the suffix one-hots, then keep the strongest contributions.
  const rows = [];
  let suffixTotal = 0;
  model.feature_names.forEach((name, index) => {
    if (isSuffixFeature(name)) {
      suffixTotal += phi[index];
    } else {
      rows.push({ label: pretty(name), value: formatValue(features[index]), shap: phi[index] });
    }
  });
  rows.push({ label: "Top-level domain", value: host.split(".").pop(), shap: suffixTotal });

  rows.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
  const top = rows.slice(0, 8);
  const scale = Math.max(...top.map((row) => Math.abs(row.shap)), 1e-9);

  ui.shap.innerHTML = top
    .map((row) => {
      const width = (Math.abs(row.shap) / scale) * 50;
      const bar = `<div class="bar ${row.shap >= 0 ? "pos" : "neg"}" style="width:${width}%"></div>`;
      return `<tr>
        <td class="name"><b>${row.label}</b></td>
        <td class="val">${row.value}</td>
        <td><div class="track"><div class="zero"></div>${bar}</div></td>
        <td class="shap">${row.shap >= 0 ? "+" : "−"}${Math.abs(row.shap).toFixed(3)}</td>
      </tr>`;
    })
    .join("");

  ui.features.innerHTML = model.feature_names
    .map((name, index) => `<tr><td class="name">${pretty(name)}</td>
      <td class="val">${formatValue(features[index])}</td></tr>`)
    .join("");

  ui.result.classList.remove("hidden");
}

function scan() {
  const raw = ui.url.value;
  ui.error.classList.add("hidden");

  if (raw.length > 2048) {
    ui.error.textContent = "Input is longer than 2048 characters.";
    ui.error.classList.remove("hidden");
    ui.result.classList.add("hidden");
    return;
  }

  const host = canonicalHost(raw);
  if (!host) {
    ui.error.textContent = "No host name could be read from that input.";
    ui.error.classList.remove("hidden");
    ui.result.classList.add("hidden");
    return;
  }

  const features = extract(host, vocab);
  render(host, predictProbability(model, features), treeShap(model, features), features);
}

async function boot() {
  try {
    const [modelJson, pslJson, metricsJson] = await Promise.all([
      fetch("./model.json").then((response) => response.json()),
      fetch("./psl.json").then((response) => response.json()),
      fetch("./metrics.json").then((response) => response.json()).catch(() => null),
    ]);

    model = modelJson;
    loadPsl(pslJson);
    vocab = {
      top_suffixes: model.vocab.top_suffixes,
      suspicious_keywords: model.vocab.suspicious_keywords,
      brand_tokens: new Set(model.vocab.brand_tokens),
    };

    metrics = metricsJson;
    if (metrics) {
      threshold = metrics.ood.threshold_at_fpr_1pct;
      const recall = (metrics.ood.recall_at_fpr_1pct * 100).toFixed(0) + "%";
      element("b-auc").textContent = metrics.ood.roc_auc.toFixed(3);
      element("b-recall").textContent = recall;
      element("n-recall").textContent = recall;
    }
    element("b-trees").textContent = model.trees.length;
    element("b-feat").textContent = model.feature_names.length;

    ui.scan.disabled = false;
    ui.scan.textContent = "Scan";

    // Shareable links: /?url=… scores on load, so a finding can be sent to someone
    // as a link rather than as a screenshot. ?theme=light|dark pins the palette,
    // which is also how the figures in the report are captured.
    const query = new URLSearchParams(location.search);
    const theme = query.get("theme");
    if (theme === "light" || theme === "dark") {
      document.documentElement.dataset.theme = theme;
    }

    const requested = query.get("url");
    if (requested) {
      ui.url.value = requested;
      scan();
    }
  } catch (error) {
    ui.error.textContent = "Could not load the model artifacts: " + error.message;
    ui.error.classList.remove("hidden");
  }
}

ui.scan.addEventListener("click", scan);
ui.url.addEventListener("keydown", (event) => {
  if (event.key === "Enter") scan();
});
document.querySelectorAll(".examples button").forEach((button) => {
  button.addEventListener("click", () => {
    ui.url.value = button.dataset.url;
    scan();
  });
});

boot();

// Sanity check in the console: the artifact must describe the features this file
// builds. A mismatch means the model and the extractor drifted apart.
window.addEventListener("load", () => {
  if (model) {
    console.log(
      `[phishurl] model ${model.model_sha256.slice(0, 12)} · ` +
        `feature spec ${model.feature_spec_sha256.slice(0, 12)} · ` +
        `${model.trees.length} trees`,
    );
  }
});
