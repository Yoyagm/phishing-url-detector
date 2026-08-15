/**
 * Run the browser engine over the parity corpus and emit its results as JSON.
 *
 * Invoked by `tests/test_parity_*.py`, which runs the same corpus through the
 * Python package and compares the two. Node is used only as a headless host for
 * the exact file the demo page loads — there is no second implementation here.
 *
 * Usage: node tests/parity_runner.mjs <corpus.txt> <out.json>
 */

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  canonicalHost,
  extract,
  loadPsl,
  rawScore,
  predictProbability,
  treeShap,
} from "../docs/engine.js";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");

const [, , corpusPath, outputPath] = process.argv;
if (!corpusPath || !outputPath) {
  console.error("usage: node tests/parity_runner.mjs <corpus.txt> <out.json>");
  process.exit(2);
}

const model = JSON.parse(readFileSync(resolve(root, "docs/model.json"), "utf8"));
loadPsl(JSON.parse(readFileSync(resolve(root, "docs/psl.json"), "utf8")));

const vocab = {
  top_suffixes: model.vocab.top_suffixes,
  suspicious_keywords: model.vocab.suspicious_keywords,
  brand_tokens: new Set(model.vocab.brand_tokens),
};

// Keep empty lines: "" is a corpus entry in its own right, and the engine must
// return a defined vector for it rather than throwing.
const corpus = readFileSync(resolve(root, corpusPath), "utf8").split("\n").slice(0, -1);

const results = corpus.map((url) => {
  const host = canonicalHost(url);
  const features = extract(host, vocab);
  return {
    url,
    host,
    features,
    raw_score: rawScore(model, features),
    probability: predictProbability(model, features),
    shap: treeShap(model, features),
  };
});

writeFileSync(
  resolve(root, outputPath),
  JSON.stringify({
    feature_spec_sha256: model.feature_spec_sha256,
    base_score: model.base_score,
    n: results.length,
    results,
  }),
);

console.log(`scored ${results.length} URLs with ${model.trees.length} trees`);
