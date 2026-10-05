# Decision Model Benchmark

Reproducible evaluation of **candidate coverage, rejection policy transfer and missing-answer sources** in typed decision models.

This benchmark evaluates **Laya 0.3.21**, **Jev 1.13.0**, **Qwen2.5-7B-Instruct**, and a fixed-ontology **BM25 scope gate**. Evaluation endpoints use public dataset reference labels.

## Install and check

Python 3.10 or newer. Offline analysis needs no GPU, API key or dataset download.

```bash
git clone https://github.com/luckykevvv/Decision_Model_Benchmark.git
cd Decision_Model_Benchmark
python -m pip install -e .
decision-benchmark verify
decision-benchmark list
decision-benchmark reproduce --profile quick --output-dir runs/quick --bootstrap-replicates 20
```

`quick` scores the recorded AG News/Jev core slice and checks its point estimates against the release. It is an integration check; published classification intervals use 2,000 bootstrap draws.

## Reproduce the completed experiments

```bash
decision-benchmark reproduce --profile core --output-dir runs/core --bootstrap-replicates 2000
decision-benchmark reproduce --profile controlled --output-dir runs/controlled --bootstrap-replicates 2000
decision-benchmark reproduce --profile intent --output-dir runs/intent
# Or run every confirmation stage:
decision-benchmark reproduce --profile all --output-dir runs/all --bootstrap-replicates 2000
```

Each command writes derived inputs, fresh metrics and a `reproduction_checks.json` file. Existing output directories are refused. Add `--joint-bootstrap` to recompute supplementary source-threshold-refit intervals; primary policy-transfer intervals hold the fitted threshold fixed. Use `--model laya`, `jev`, `qwen` or `bm25` to select a released model in an applicable profile.

| Profile | Text/query support | Responses per neural model | Purpose |
|---|---:|---:|---|
| `core` | 1,456 classification texts | 39,168 | Paired reference-label coverage; local and cross-task policies |
| `controlled` | 120 reused classification texts | 28,920 | Matched names, membership, order and NONE wording |
| `intent` | 750 CLINC150 queries | 6,900 | Artificial omission, natural retrieval misses and public OOS |

The three neural models provide **224,964 recorded responses**, plus **6,900 BM25 decisions**. Variants and the controlled subset do not add independent texts. The released reference metrics retain original 2,000-draw intervals, failures and all tested source/target transfers.

## Rebuild confirmation figures and tables

```bash
python -m pip install -e ".[plots]"
decision-benchmark displays --output-dir runs/displays
```

This reads the bundled complete reference metrics and writes four figures as PDF/SVG/PNG and six metric tables. A `--metrics-dir` may point to a directory with `core-{model}/metrics.json`, `controls-{model}.json` and `intent-{model}.json` in the same complete 2,000-draw layout. Existing output directories are refused.

## Prepare texts and run a model

Frozen requests contain ordered option definitions, not dataset text. Recorded-output scoring uses explicit text indices. New inference reconstructs texts from the pinned original sources and verifies both source-file and per-text hashes.

```bash
python -m pip install -e ".[data]"
decision-benchmark prepare --profile core --with-text --output-dir runs/core-inputs
```

Choose one adapter:

```bash
# Laya: supports CPU or CUDA; the first run downloads the pinned checkpoint.
python -m pip install -e ".[laya]"
decision-benchmark run --model laya --device cpu --inputs runs/core-inputs --output-dir runs/laya

# Jev: set TYPESAFE_API_KEY in your process environment, then choose a cost budget.
decision-benchmark run --model jev --inputs runs/core-inputs --output-dir runs/jev --budget-usd 1.00 --preflight

# Qwen: pinned BF16 CUDA baseline, greedy exact-key output.
python -m pip install -e ".[qwen]"
decision-benchmark download-model --model qwen
decision-benchmark run --model qwen --inputs runs/core-inputs --output-dir runs/qwen
```

`--preflight` is a partial adapter check; continue with the same output directory without that flag to complete the frozen matrix. Inference resumes only after checking existing inputs/model settings and predictions. Jev calls consume the user-selected budget; frozen reproduction makes no calls. Laya retains native temperature handling; Qwen is greedy BF16 with a 2,048-token input cap and a 32-token output cap. Its parser accepts one offered key or an exact JSON string key. Invalid outputs remain failures.

For the corpus reference:

```bash
decision-benchmark prepare --profile intent --with-text --output-dir runs/intent-inputs
decision-benchmark run --model bm25 --inputs runs/intent-inputs --output-dir runs/bm25
```

Score a completed run:

```bash
decision-benchmark score --profile core --run-dir runs/laya --output-dir runs/laya-scores --bootstrap-replicates 2000
```

## Protocol and outputs

Covered inputs partition into correct choice, wrong choice, rejection and failure. Missing inputs partition into rejection, acceptance and failure. Probability models support negative-max, normalized entropy and raw NONE-probability scores. Discrete models receive no invented scores.

Thresholds maximize calibration detection subject to empirical false rejection <=5%, with strict `score > threshold`. Cross-task transfer uses 80 fixed calibration pairs per task. Target-local recalibration consumes its own 80 labels and is reported separately. Variants are averaged within texts before bootstrap aggregation. Primary intervals hold thresholds fixed; optional supplements refit source thresholds.

Artificial omission substitutes a reference label at a matched slot/count. Natural retrieval menus are not repaired with gold labels. Global OOS uses its own population. CLINC150 retrieval uses maximum document BM25 scores per intent, k1=1.2 and b=0.75 over 14,700 training utterances after selected-query exclusion.

### Request/response schema

A run contains `cases.jsonl`, `requests.jsonl`, `manifest.json` and `predictions.jsonl`. Every request preserves `request_id`, `case_id`, task/split, ordered `criteria`, instruction, candidate count and experimental-condition fields. Reference labels are used for scoring, not supplied as model hints.

Success records contain an offered `choice`, `status: ok` and either an offered-key probability map or `probabilities: null` for discrete models. Failures use `interface_failure` or `truncation_failure`, null choice/probabilities and an error description. Duplicate IDs, changed inputs and incomplete collections are rejected. Jev's two-decimal maps retain declared rounding tolerance and native choice/argmax disagreements.

Offline scoring uses explicit `text-index-v1` cases with raw/normalized text hashes. New inference requires reconstructed text. `prepare --with-text` verifies pinned source files, source labels, per-text identities, original field order/line endings and complete original case-file bytes. Add `--offline` to require a verified existing cache.

To add a model, use `systematic.confirmation_runtime.prepare_run` with a new pinned model descriptor, emit and validate one prediction per frozen request, then invoke `score`. Freeze a separate calibration subset plan before evaluating custom task transfers.

### Sources

Exact dataset source URLs/revisions/checksums are in `decision_benchmark/resources/release.json`. Dataset references: [AG News/DBpedia](https://proceedings.neurips.cc/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html), [Emotion](https://aclanthology.org/D18-1404/), [TREC](https://aclanthology.org/C02-1150/), [CLINC150](https://aclanthology.org/D19-1131/).

Model references: [Laya 0.3.21](https://pypi.org/project/laya/0.3.21/), [Jev API](https://docs.typesafe.ai/api), [Qwen2.5](https://arxiv.org/abs/2412.15115v2). Laya and Qwen checkpoints are pinned in their manifests; native model weights are downloaded separately.

TREC is supplemental because its retained test set has no ABBR examples. Natural CLINC150 misses number 31/11/5 at candidate counts 2/5/10; report those denominators. Empirical calibration constraints are not population risk guarantees. Qwen's recorded interface failures remain in native decision denominators. BM25 consumes a labeled corpus and fixed intent identifiers; it is a resource-conditioned reference.

## Tests and citation

```bash
python -m unittest discover -s systematic -p "test_*.py"
python -m unittest discover -s tests -p "test_*.py"
```

Use [`CITATION.cff`](CITATION.cff) to cite the software. Benchmark code uses the MIT license. Original datasets and model checkpoints retain their source terms; their texts and weights are not bundled here.

The 39 tests pass on Linux and Windows. Full 2,000-draw primary recomputation rescored all 231,864 recorded decisions: 114,388 deterministic numeric checks and 43,156 interval values matched. All three input profiles reconstruct byte-identically. Supplementary source-threshold-refit intervals remain separately released.
