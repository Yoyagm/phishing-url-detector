/**
 * Browser-side scoring engine — a line-for-line port of the Python package.
 *
 * The demo has no backend: this file canonicalises the host, extracts the same 51
 * features, walks the same 300 trees and computes the same path-dependent TreeSHAP
 * attributions the Python service would. `tests/test_parity_*.py` runs both sides
 * over a corpus of URLs and fails if any feature or any attribution diverges by
 * more than 1e-12, so "line-for-line" is a checked claim rather than a hopeful one.
 *
 * Ports of, in order:
 *   phishurl/psl.py           -> loadPsl, publicSuffixLabels, splitHost, isPlatformHosted
 *   phishurl/canonicalize.py  -> canonicalHost, isIpLiteral
 *   phishurl/features.py      -> shannonEntropy, extract
 *   LightGBM predict          -> rawScore
 *   Lundberg et al. 2018      -> treeShap
 */

// ---------------------------------------------------------------- Public Suffix List

let PSL = null;

export function loadPsl(payload) {
  const icann = new Set(payload.rules.slice(0, payload.icann_count));
  const all = new Set(payload.rules);
  PSL = { icann, all };
  return PSL;
}

function rulesFor(section) {
  if (!PSL) throw new Error("PSL not loaded — call loadPsl() first");
  return section === "all" ? PSL.all : PSL.icann;
}

export function publicSuffixLabels(host, section = "icann") {
  if (!host) return 0;
  const rules = rulesFor(section);
  const labels = host.split(".");
  const n = labels.length;
  let best = 0;

  for (let i = 0; i < n; i++) {
    const candidate = labels.slice(i).join(".");
    // `i` ascends, so the first exception we meet is the longest match.
    if (rules.has("!" + candidate)) return n - i - 1;
    if (rules.has(candidate)) best = Math.max(best, n - i);
    const wildcard = ["*"].concat(labels.slice(i + 1)).join(".");
    if (rules.has(wildcard)) best = Math.max(best, n - i);
  }
  return best ? best : 1;
}

export function splitHost(host, section = "icann") {
  if (!host) return ["", "", ""];
  const labels = host.split(".");
  const n = labels.length;
  const nSuffix = publicSuffixLabels(host, section);
  if (nSuffix >= n) return ["", "", host];
  return [
    labels.slice(0, n - nSuffix - 1).join("."),
    labels.slice(n - nSuffix - 1).join("."),
    labels.slice(n - nSuffix).join("."),
  ];
}

export function isPlatformHosted(host) {
  if (!host) return false;
  return publicSuffixLabels(host, "all") > publicSuffixLabels(host, "icann");
}

// ---------------------------------------------------------------- canonicalisation

const SCHEME_RE = /^[A-Za-z][A-Za-z0-9+.\-]*:\/\//;

/**
 * Plain string surgery rather than `new URL()`. The browser's parser punycodes
 * IDN hosts and Python's `urlsplit` does not, so using either one would build a
 * disagreement into the two implementations. Case folding is ASCII-only for the
 * same reason.
 */
export function canonicalHost(raw) {
  if (typeof raw !== "string") return "";
  let host = raw.trim();
  if (!host) return "";

  host = host.replace(SCHEME_RE, "");
  if (host.startsWith("//")) host = host.slice(2);

  for (const delimiter of ["/", "?", "#"]) {
    const index = host.indexOf(delimiter);
    if (index !== -1) host = host.slice(0, index);
  }

  // `https://paypal.com@evil.tk` is a real attack: the host follows the LAST '@'.
  const at = host.lastIndexOf("@");
  if (at !== -1) host = host.slice(at + 1);

  if (host.startsWith("[")) {
    const end = host.indexOf("]");
    host = end !== -1 ? host.slice(1, end) : host.slice(1);
  } else {
    const colon = host.indexOf(":");
    if (colon !== -1) host = host.slice(0, colon);
  }

  host = host.replace(/[A-Z]/g, (character) =>
    String.fromCharCode(character.charCodeAt(0) + 32),
  );
  host = host.replace(/\.+$/, "");
  if (host.startsWith("www.")) host = host.slice(4);
  return host;
}

const IPV4_RE = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/;

export function isIpLiteral(host) {
  const match = IPV4_RE.exec(host);
  if (match) {
    return match.slice(1).every((part) => Number(part) >= 0 && Number(part) <= 255);
  }
  if (!host.includes(":")) return false;
  return [...host].every((character) => "0123456789abcdef:.".includes(character));
}

// ---------------------------------------------------------------- features

const VOWELS = new Set("aeiou");
const DIGITS = new Set("0123456789");
const LETTERS = new Set("abcdefghijklmnopqrstuvwxyz");
const CONSONANTS = new Set([...LETTERS].filter((c) => !VOWELS.has(c)));

export function shannonEntropy(text) {
  if (!text) return 0;
  const counts = new Map();
  for (const character of text) counts.set(character, (counts.get(character) || 0) + 1);
  const total = text.length;
  let entropy = 0;
  for (const count of counts.values()) {
    const probability = count / total;
    entropy -= probability * Math.log2(probability);
  }
  return entropy;
}

function maxRun(text, members) {
  let best = 0;
  let current = 0;
  for (const character of text) {
    if (members.has(character)) {
      current += 1;
      if (current > best) best = current;
    } else {
      current = 0;
    }
  }
  return best;
}

function countMembers(text, members) {
  let total = 0;
  for (const character of text) if (members.has(character)) total += 1;
  return total;
}

function countChar(text, target) {
  let total = 0;
  for (const character of text) if (character === target) total += 1;
  return total;
}

/** Extract the feature vector. `vocab` comes from model.json, so both sides share it. */
export function extract(host, vocab) {
  const width = 25 + vocab.top_suffixes.length + 1;
  if (!host) return new Array(width).fill(0);

  const [subdomain, registrable, suffix] = splitHost(host);
  const labels = host.split(".");
  const labelLengths = labels.map((label) => label.length);

  const letters = countMembers(host, LETTERS);
  const vowels = countMembers(host, VOWELS);
  const digits = countMembers(host, DIGITS);
  const hyphens = countChar(host, "-");

  const registrableName = registrable
    ? registrable.slice(0, registrable.length - suffix.length - 1)
    : "";
  const brands = vocab.brand_tokens instanceof Set
    ? vocab.brand_tokens
    : new Set(vocab.brand_tokens);
  const brandInSubdomain = subdomain
    ? [...brands].some((brand) => subdomain.includes(brand))
    : false;
  const brandInRegistrable = brands.has(registrableName);

  const values = [
    // shape
    host.length,
    labels.length,
    subdomain ? subdomain.split(".").length : 0,
    registrable.length,
    subdomain.length,
    suffix.length,
    Math.max(...labelLengths),
    labelLengths.reduce((a, b) => a + b, 0) / labelLengths.length,
    // composition
    shannonEntropy(host),
    shannonEntropy(registrable),
    digits,
    digits / host.length,
    hyphens,
    hyphens / host.length,
    letters ? vowels / letters : 0,
    maxRun(host, DIGITS),
    maxRun(host, CONSONANTS),
    countMembers(registrable, DIGITS),
    // structure
    isIpLiteral(host) ? 1 : 0,
    host.includes("xn--") ? 1 : 0,
    suffix.includes(".") ? 1 : 0,
    isPlatformHosted(host) ? 1 : 0,
    // lexical
    vocab.suspicious_keywords.some((keyword) => host.includes(keyword)) ? 1 : 0,
    brandInSubdomain || brandInRegistrable ? 1 : 0,
    brandInSubdomain && !brandInRegistrable ? 1 : 0,
  ];

  for (const candidate of vocab.top_suffixes) values.push(suffix === candidate ? 1 : 0);
  values.push(vocab.top_suffixes.includes(suffix) ? 0 : 1);
  return values;
}

// ---------------------------------------------------------------- scoring

function leafFor(tree, x) {
  let node = 0;
  while (tree.feature[node] >= 0) {
    const value = x[tree.feature[node]];
    const goLeft = Number.isNaN(value)
      ? tree.default_left[node] === 1
      : value <= tree.threshold[node];
    node = goLeft ? tree.left[node] : tree.right[node];
  }
  return node;
}

/** Sum of leaf values across the ensemble — the log-odds, matching `raw_score=True`. */
export function rawScore(model, x) {
  let total = 0;
  for (const tree of model.trees) total += tree.value[leafFor(tree, x)];
  return total;
}

export function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

export function predictProbability(model, x) {
  return sigmoid(rawScore(model, x));
}

// ---------------------------------------------------------------- TreeSHAP

/*
 * Path-dependent TreeSHAP, Lundberg, Erion & Lee (2018), "Consistent
 * Individualized Feature Attribution for Tree Ensembles", Algorithm 2.
 *
 * The decision path carries, per split seen so far, the fraction of training
 * samples that follow it when the feature is treated as absent (`zeroFraction`)
 * versus present (`oneFraction`), plus a weight. Extending the path adds a split;
 * unwinding removes one, which is needed both when a feature is split on twice
 * and when totalling a leaf's contribution.
 */

function extendPath(path, uniqueDepth, zeroFraction, oneFraction, featureIndex) {
  path[uniqueDepth] = {
    featureIndex,
    zeroFraction,
    oneFraction,
    pweight: uniqueDepth === 0 ? 1 : 0,
  };
  for (let i = uniqueDepth - 1; i >= 0; i--) {
    path[i + 1].pweight += (oneFraction * path[i].pweight * (i + 1)) / (uniqueDepth + 1);
    path[i].pweight = (zeroFraction * path[i].pweight * (uniqueDepth - i)) / (uniqueDepth + 1);
  }
}

function unwindPath(path, uniqueDepth, pathIndex) {
  const oneFraction = path[pathIndex].oneFraction;
  const zeroFraction = path[pathIndex].zeroFraction;
  let nextOnePortion = path[uniqueDepth].pweight;

  for (let i = uniqueDepth - 1; i >= 0; i--) {
    if (oneFraction !== 0) {
      const tmp = path[i].pweight;
      path[i].pweight = (nextOnePortion * (uniqueDepth + 1)) / ((i + 1) * oneFraction);
      nextOnePortion =
        tmp - (path[i].pweight * zeroFraction * (uniqueDepth - i)) / (uniqueDepth + 1);
    } else {
      path[i].pweight = (path[i].pweight * (uniqueDepth + 1)) / (zeroFraction * (uniqueDepth - i));
    }
  }
  for (let i = pathIndex; i < uniqueDepth; i++) {
    path[i].featureIndex = path[i + 1].featureIndex;
    path[i].zeroFraction = path[i + 1].zeroFraction;
    path[i].oneFraction = path[i + 1].oneFraction;
  }
}

function unwoundPathSum(path, uniqueDepth, pathIndex) {
  const oneFraction = path[pathIndex].oneFraction;
  const zeroFraction = path[pathIndex].zeroFraction;
  let nextOnePortion = path[uniqueDepth].pweight;
  let total = 0;

  if (oneFraction !== 0) {
    for (let i = uniqueDepth - 1; i >= 0; i--) {
      const tmp = (nextOnePortion * (uniqueDepth + 1)) / ((i + 1) * oneFraction);
      total += tmp;
      nextOnePortion =
        path[i].pweight - (tmp * zeroFraction * (uniqueDepth - i)) / (uniqueDepth + 1);
    }
  } else {
    for (let i = uniqueDepth - 1; i >= 0; i--) {
      total += path[i].pweight / ((zeroFraction * (uniqueDepth - i)) / (uniqueDepth + 1));
    }
  }
  return total;
}

function recurse(tree, x, phi, node, uniqueDepth, parentPath, pZero, pOne, pFeature) {
  const path = new Array(uniqueDepth + 1);
  for (let i = 0; i < uniqueDepth; i++) path[i] = { ...parentPath[i] };
  extendPath(path, uniqueDepth, pZero, pOne, pFeature);

  if (tree.feature[node] < 0) {
    for (let i = 1; i <= uniqueDepth; i++) {
      const weight = unwoundPathSum(path, uniqueDepth, i);
      phi[path[i].featureIndex] +=
        weight * (path[i].oneFraction - path[i].zeroFraction) * tree.value[node];
    }
    return;
  }

  const feature = tree.feature[node];
  const value = x[feature];
  const goLeft = Number.isNaN(value)
    ? tree.default_left[node] === 1
    : value <= tree.threshold[node];
  const hot = goLeft ? tree.left[node] : tree.right[node];
  const cold = goLeft ? tree.right[node] : tree.left[node];

  const cover = tree.count[node];
  const hotZero = tree.count[hot] / cover;
  const coldZero = tree.count[cold] / cover;

  let incomingZero = 1;
  let incomingOne = 1;
  let pathIndex = 0;
  for (; pathIndex <= uniqueDepth; pathIndex++) {
    if (path[pathIndex].featureIndex === feature) break;
  }

  let depth = uniqueDepth;
  if (pathIndex <= uniqueDepth) {
    // Already split on this feature higher up: undo it before re-adding.
    incomingZero = path[pathIndex].zeroFraction;
    incomingOne = path[pathIndex].oneFraction;
    unwindPath(path, uniqueDepth, pathIndex);
    depth = uniqueDepth - 1;
  }

  recurse(tree, x, phi, hot, depth + 1, path, hotZero * incomingZero, incomingOne, feature);
  recurse(tree, x, phi, cold, depth + 1, path, coldZero * incomingZero, 0, feature);
}

/**
 * Per-feature SHAP values in log-odds space.
 *
 * `model.base_score + sum(treeShap(...))` equals `rawScore(...)`; that identity is
 * asserted in the parity tests, which is what makes the bars on the demo page an
 * exact decomposition rather than a plausible-looking one.
 */
export function treeShap(model, x) {
  const phi = new Array(model.feature_names.length).fill(0);
  for (const tree of model.trees) {
    recurse(tree, x, phi, 0, 0, [], 1, 1, -1);
  }
  return phi;
}
