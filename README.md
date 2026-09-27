# Amazon Business Entity Resolution

Solution pipeline for the **Amazon ML Challenge 2026 — Business Entity Resolution Challenge**:
for every Source 1 (S1) business record, find all records in Source 2 (S2) and Source 3 (S3) that
describe the same real-world business, optimizing per-S1 macro F0.5.

---

## Overview

The system is a blocking + learned-matcher pipeline:

```
Challenge TSVs
  → normalization (Unicode-safe, multilingual)
  → integer-ID entity caches (Parquet)
  → candidate blocking (4 retrieval channels, streamed in chunks)
  → pairwise feature extraction (35 model features)
  → XGBoost pairwise matcher
  → S1-level contextual second-stage decision model
  → matching_results.tsv + candidate_pairs.tsv
  → challenge submission validator
```

Results measured on a 20,000-S1 training sample (validation = 4,000 held-out S1; see
[Current Status](#current-status)):

| Stage | Result |
|---|---|
| Candidate recall (final union), S2 / S3 | 97.35% / 97.04% |
| Base XGBoost matcher (`xgb_skeleton`), validation macro F0.5 | 0.9181 |
| + S1-level contextual decision layer, validation macro F0.5 | **0.9254** |

The production inference pipeline for the full test set is implemented, tested and restartable, and
has been **run on the complete test dataset** (1,732,544 S1); the resulting `matching_results.tsv`
passed the official submission validator (see [Current Status](#current-status)). No leaderboard
score is known yet.

---

## Challenge

Facts from the challenge's `student_resource/README.md` and from measurements on the provided data.

- Each source file has `entity_id`, `business_name`, `business_address`, `country`. The ID prefix
  (`S1-`, `S2-`, `S3-`) gives the source. **S1 is the deduplicated reference**; an S1 entity may
  match zero, one or many S2/S3 records.
- Countries: training covers **US** and **India**; the test set additionally contains **France**.

| File | Records | | File | Records |
|---|---|---|---|---|
| `train_source1.tsv` | 2,206,821 | | `test_source1.tsv` | 1,732,544 (India 809,986 · US 663,106 · France 259,452) |
| `train_source2.tsv` | 5,034,616 | | `test_source2.tsv` | 4,887,273 |
| `train_source3.tsv` | 5,285,603 | | `test_source3.tsv` | 5,082,316 |

- `train_ground_truth.tsv`: one row per S1 with comma-separated matched S2/S3 IDs (empty = no match).
  7,638,365 true links (S2 3,693,619 · S3 3,944,746); 123,247 S1 (5.58%) have no match; about 26%
  of S2/S3 records are never linked to any S1 (distractors); no S2/S3 record is linked to more than
  one S1.
- **Metric:** F0.5 per S1, macro-averaged over **all** S1 (an S1 with no true match scores 1.0 for an
  empty prediction and 0.0 otherwise). F0.5 weights precision over recall, so false merges are costly.
- **Why blocking:** all S1×(S2+S3) training pairs would be ≈ 2.2M × 10.3M ≈ 2.3 × 10¹³ comparisons;
  blocking reduces this to a few hundred candidates per S1.

---

## Architecture

| Stage | Module(s) | Purpose |
|---|---|---|
| Data processing | `src/data/convert.py`, `src/data/loader.py` | TSV → Parquet; safe string-typed loading |
| Normalization | `src/preprocessing/normalize.py`, `transliterate.py` | Unicode-safe name/address/country normalization; Latin transliteration |
| Entity caches | `src/blocking/entity_cache.py` | normalized, integer-ID Parquet per source and dataset (+ training ground truth) |
| Candidate blocking | `src/blocking/channels.py`, `address_keys.py`, `candidate_pipeline.py` | 4 retrieval channels per (target source, country); streamed S1 chunks |
| Candidate validation | `src/blocking/validate_candidates.py` | duplicates, country, provenance, completeness, determinism |
| Pairwise features | `src/features/pairwise.py` (+ `text_`, `address_`, `retrieval_features.py`) | 40 feature columns per candidate; labels on training data only |
| Base matcher | `src/models/xgb_matcher.py` | XGBoost pairwise classifier (training + evaluation) |
| Decision layer | `src/decision/oof.py`, `src/decision/s1_context.py` | out-of-fold base scores + S1-context second stage |
| Inference | `src/inference/*.py` | restartable test pipeline: scoring, decisions, output files |

---

## Repository Structure

```text
solution/
├── README.md
├── requirements.txt            pinned dependencies
├── configs/paths.yaml          not read by any code (paths are resolved relative to the repo)
├── experiments/experiment_log.csv
├── src/
│   ├── data/                   loader.py, convert.py
│   ├── preprocessing/          normalize.py, transliterate.py
│   ├── blocking/
│   │   ├── entity_cache.py  address_keys.py  channels.py
│   │   ├── candidate_pipeline.py            production candidate generation
│   │   ├── validate_candidates.py           candidate validation
│   │   ├── benchmark_*.py  diagnose_union.py  profile_true_matches.py
│   │   │   inspect_zero_name_matches.py     research / benchmarks
│   │   └── exact_name.py  token_index.py  char_tfidf.py  test_exact_name.py
│   │                                        early prototypes (superseded)
│   ├── features/               text_features.py, address_features.py, retrieval_features.py,
│   │                           pairwise.py (production); audit_features.py (diagnostic)
│   ├── evaluation/             f05.py, split.py, labels.py (used in training/evaluation);
│   │                           error_analysis.py, name_miss_analysis.py,
│   │                           fp_skeleton_analysis.py (diagnostic)
│   ├── models/                 xgb_matcher.py
│   ├── decision/               oof.py, s1_context.py
│   └── inference/              run_pipeline.py, predict_base.py, decide.py, write_outputs.py
└── tests/                      unittest suite
```

Everything under `outputs/` (Parquet caches, candidates, features, trained models, experiment and
production runs) is **git-ignored and must be regenerated** — see
[Reproducing the model artifacts](#reproducing-the-model-artifacts).

**Production vs. research code.** The production path is: `data` → `preprocessing` →
`entity_cache` → `address_keys` + `channels` + `candidate_pipeline` → `validate_candidates` →
`features/{text,address,retrieval}_features` + `pairwise` → `models/xgb_matcher` (training) →
`decision/{oof,s1_context}` (second-stage training; `context_features`/`design` reused at inference)
→ `inference/*`. The `benchmark_*`, `diagnose_union`, `profile_*`, `inspect_*`, `audit_features` and
`evaluation/*_analysis.py` modules are research and diagnostic tools that informed design decisions;
they are not part of the production run. `channels.py` also contains a `TransliteratedNameChannel`
that was benchmarked and **not** adopted.

---

## Methodology

### Data Processing

- `src/data/loader.py`: reads TSVs with `sep="\t"`, all columns as strings, `keep_default_na=False`
  (empty stays empty); validates headers.
- `src/data/convert.py`: `train_*`/`test_*` TSV → zstd Parquet in `outputs/parquet/` (atomic writes;
  existing files skipped).
- `src/blocking/entity_cache.py`: normalizes every record once and writes
  `outputs/normalized/{train,test}_source{1,2,3}.parquet` (`id` int32, `country`, `norm_name`,
  `norm_address`) and `train_truth.parquet`. All IDs in every file are integers < 2³¹, so the
  integer representation is lossless.

### Normalization

- NFKC + lowercase; Unicode letters, digits and combining marks are kept in every script
  (Devanagari, Tamil, Kannada, … stay intact); punctuation/symbols become spaces.
- Business names: Latin-script legal prefixes/suffixes removed (`inc`, `llc`, `ltd`, `pvt`,
  `private`, `limited`, `gmbh`, …).
- `transliterate_text`: `unidecode`-based Latin transliteration, used for transliterated name features.

### Candidate Generation

Blocking runs per target source (S2, S3) and country partition. Settings are the
`PipelineConfig` defaults in `src/blocking/candidate_pipeline.py`:

| Channel | Method | Setting |
|---|---|---|
| `exact_name` | all targets with the identical normalized name (sorted hash array + `searchsorted`) | all matches |
| `name_char` | character-trigram TF-IDF (`char_wb` (3,3), hashed to 2²² features, no vocabulary dict); trigrams in > 30,000 target names are left out of the sparse index; top 1,000 by pruned score are **reranked by full cosine** | rerank pool 1,000, final K = **100** |
| `address_char` | same machinery on normalized addresses | rerank pool 1,000, final K = **25** |
| `house_city` | exact match on the extracted `house number | city` key | blocks ≤ **1,000** targets |

Channel outputs are unioned and deduplicated per (S1, target); every candidate keeps its channel
bitmask, per-channel ranks and scores. S1 are processed in chunks (sorted by id); each chunk is one
Parquet part, so the candidate set is never held in memory.

**Country-aware behavior** (`src/blocking/address_keys.py`): city and house+city keys need a state
table and a city lexicon, which exist for US and India only. For any other country (France in the
test set) the house number is still extracted (the rule is country-independent) but city, postal
code and house+city are empty — the `house_city` channel contributes no candidates and
`city_match`/`house_city_match` are NaN. The exact-name, name and address channels work for every
country. City lexicons are learned from training S1 and reused for test (`--lexicon-dir`).

Measured candidate recall (training ground truth; 20k random training S1, complete S2/S3 populations):

| Configuration | S2 recall | S3 recall | Scope |
|---|---|---|---|
| Exact normalized name | 41.49% | 40.11% | all 2.2M training S1 |
| Name char, rerank 1000 → top 100 | 71.35% | 70.10% | 20k-S1 sample |
| Exact ∪ name char ∪ address char (K=25) | 96.61% | 96.02% | 20k-S1 sample |
| **+ house+city (final union)** | **97.35%** | **97.04%** | 20k-S1 sample, ~173 candidates/S1 |

`validate_candidates` checks: no duplicate pairs; S1/targets in the part's country; channel
provenance; exact-name and house+city correctness and completeness; rank/score sanity; per-S1
counts; streaming iteration; and (with `--compare`) identical content across runs.

### Pairwise Features

`src/features/pairwise.py` streams candidate parts and writes one feature part per candidate part.
Transliteration, skeletons, address keys and name frequencies are computed once per entity.

| Family | Features |
|---|---|
| Name | exact, RapidFuzz ratio / partial / token-sort / token-set, token Jaccard, transliterated ratio and token Jaccard |
| Transliteration skeleton | `translit_skeleton_ratio`: transliterated name with Latin and transliterated legal suffixes removed (`praaivett`, `limittedd`, …), phonetic digraphs merged, vowels dropped, repeated letters collapsed; NaN for skeletons shorter than 2 letters |
| Name rarity | `target_name_frequency` (targets sharing the normalized name within its source and country; computed from the entity population, no labels), `target_name_rarity` = 1 / frequency |
| Address | ratio, partial, token-sort, token-set, token Jaccard |
| Address numbers | number-set Jaccard, ordered number-sequence similarity, numeric-token count difference, secondary-number match (all digit runs, not just the first) |
| Structured | house-number match, city match, house+city match, country match |
| Retrieval | channel hits, number of channels, name/address channel rank and score, best ranks, candidates per S1 |
| Missingness | S1/target name and address missing flags |

Similarities are NaN when either side is empty, so "unknown" is never treated as "dissimilar".
Labels (`src/evaluation/labels.py`) are built only on training data, per target source (S2
candidates are never labeled from S3 truth); on test data `label` and `split` are written as −1.

### XGBoost Matcher

`src/models/xgb_matcher.py` (experiment `xgb_skeleton`):

- **35 input features** — the 40 feature columns minus `country_match`, `s1_name_missing`,
  `s1_address_missing` (constant) and `name_exact`, `house_city_match` (duplicates).
- S1-level split (`src/evaluation/split.py`): 16,000 train / 4,000 validation S1, seed 42; early
  stopping on a further 10% S1 hold-out carved from the train split.
- `hist` trees, learning rate 0.1, depth 6, min child weight 5, subsample / colsample 0.8,
  up to 1,000 trees with early stopping (50 rounds, AUC-PR), `scale_pos_weight` = negatives /
  positives of the fitted rows (≈ 103), CPU only, seed 42.
- Evaluated with the challenge metric (per-S1 macro F0.5 over all validation S1, including
  blocking misses); threshold swept on validation.

Feature additions were accepted one at a time on the same split (validation macro F0.5):

| Experiment | Features | Macro F0.5 | Threshold |
|---|---|---|---|
| `xgb_baseline` | 28 | 0.8857 | 0.994 |
| `xgb_numeric` (+ address numbers) | 32 | 0.9081 | 0.994 |
| `xgb_rarity` (+ name rarity) | 34 | 0.9135 | 0.993 |
| `xgb_skeleton` (+ transliteration skeleton) | 35 | 0.9181 | 0.991 |

### S1-Level Decision Layer

A second model re-decides candidates using the context of **all candidates of the same S1**
(`src/decision/s1_context.py`; used by `src/inference/decide.py`):

- **Context features** (no labels): S1 candidate count; counts above score levels; candidate rank
  and percentile; S1 top and second score; gap to the top score; high-score counts per target source;
  whether the top candidate is S3; the candidate's difference to the S1's top candidate in address
  token-set, name ratio, number-set Jaccard, skeleton ratio and name frequency; plus the candidate's
  own name frequency, missing-address flag and address-contradiction flag. The production model uses
  22 such inputs.
- **Training without leakage** (`src/decision/oof.py`): 5-fold S1-grouped **out-of-fold** base scores
  on the training S1 (same base configuration per fold). The second-stage model type (logistic
  regression vs. small XGBoost) and its threshold are selected on train OOF; validation is scored once.
- **Production model:** XGBoost, 300 trees, depth 3, learning rate 0.05. Only candidates with base
  score ≥ **0.5** are re-decided; a candidate is matched when the second-stage probability ≥ **0.65**.
- Validation: macro F0.5 0.9181 → **0.9254**; false positives 416 → 369; false negatives 1,516 → 1,385;
  no-match S1 receiving a prediction 34 → 18.

### Inference Pipeline

`src/inference/run_pipeline.py` runs, in order:

1. **candidates** — `candidate_pipeline` on the test caches, reusing the training lexicons
2. **validate** — `validate_candidates` (read-only checks)
3. **features** — `pairwise` (no labels)
4. **scores** — `predict_base`: `xgb_skeleton` probability per candidate
5. **decisions** — `decide`: S1 context + second stage
6. **submission** — `write_outputs`: the two challenge TSVs

S1 context needs every candidate of an S1 from both S2 and S3. Each country's S1 list is chunked
identically for both targets, so candidate part *k* of S2 and of S3 cover the same S1; `decide`
processes each (country, *k*) pair as one group and refuses to run unless both report the same S1
range and count and groups do not overlap. Context is never computed on a partial group.

**Restartable:** every part is written atomically (temporary file + rename) followed by a JSON
sidecar; re-running the same command skips finished parts, and a candidate partition whose parts are
all done is not refitted. Resuming with a different configuration is refused.

---

## Setup

Requirements: **Python 3.12**, [uv](https://docs.astral.sh/uv/), the challenge dataset. No GPU or
CUDA is used. The repository has no `pyproject.toml`/`uv.lock`; dependencies are pinned in
`requirements.txt` (so use `uv pip install -r requirements.txt`, not `uv sync`).

Resources (measured on 16 cores / 23 GB RAM): candidate generation peaks at ~5 GB RAM with 8 worker
processes; ≥ 16 GB RAM recommended.

After setup, a fresh clone still has no model artifacts (`outputs/` is git-ignored) — see
[Reproducing the model artifacts](#reproducing-the-model-artifacts) before running inference.

### Linux

```bash
# uv (skip if installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone NEXT TO the challenge's student_resource/ directory (see Dataset Layout)
cd <challenge-workspace>
git clone https://github.com/Arjun13-git/amazon-entity-resolution.git solution
cd solution

uv venv --python 3.12 .venv        # uv downloads Python 3.12 if needed
source .venv/bin/activate
uv pip install -r requirements.txt

python -c "import xgboost, rapidfuzz, pyarrow; print('ok', xgboost.__version__)"
```

Run every command below from the repository root with the environment active.

### Windows

> **Windows support**
>
> - **Native Windows is suitable for:** repository setup, the unit tests, and submission validation.
> - **Run under WSL2 (Ubuntu), following the Linux guide:** entity-cache building, full candidate
>   generation and production inference (`run_pipeline`). Reasons:
>   - `src/blocking/entity_cache.py` and `src/blocking/channels.py` use fork-based multiprocessing
>     (`multiprocessing.get_context("fork")`), which does not exist on Windows;
>   - `requirements.txt` pins `nvidia-nccl-cu13`, a Linux-only wheel (excluded in step 7 below).

PowerShell (Windows 10/11):

```powershell
# 1. Git
winget install --id Git.Git -e

# 2-3. uv, then Python 3.12 through uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.12

# 4-5. Clone next to student_resource\ and enter it
cd <challenge-workspace>
git clone https://github.com/Arjun13-git/amazon-entity-resolution.git solution
cd solution

# 6. Virtual environment
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 7. Dependencies (without the Linux-only nvidia-nccl-cu13 wheel)
Get-Content requirements.txt | Where-Object { $_ -notmatch '^nvidia-nccl' } |
    Set-Content "$env:TEMP\requirements-windows.txt"
uv pip install -r "$env:TEMP\requirements-windows.txt"

# 8. Place the dataset (see Dataset Layout)

# 9. Verify
python -c "import xgboost, rapidfuzz, pyarrow; print('ok', xgboost.__version__)"

# 10. Tests
python -m unittest discover -s tests -v
```

11–12. Production inference: use WSL2 (`wsl --install -d Ubuntu`, then the Linux guide; keep the
repository and dataset inside the WSL file system for speed). The validator runs natively:

```powershell
cd ..\student_resource
python utils\validate_submission.py `
    --matching ..\solution\outputs\production\test_v1\submission\output\matching_results.tsv `
    --candidate ..\solution\outputs\production\test_v1\submission\output\candidate_pairs.tsv `
    --test-dir dataset\test
```

---

## Dataset Layout

The dataset is **not** in this repository and must not be committed. The code locates it at
`../student_resource/dataset` relative to the repository root:

```text
<challenge-workspace>/
├── student_resource/
│   ├── dataset/
│   │   ├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
│   │   └── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
│   └── utils/validate_submission.py
└── solution/        ← this repository (any folder name works, as long as it sits here)
```

---

## Running Tests

```bash
python -m unittest discover -s tests -v
```

61 tests (standard-library `unittest`; no pytest needed) covering numeric address features, name
rarity, transliteration skeletons, false-positive analysis helpers, S1-context features and rules,
country support, exact S1 grouping, output writing, and a tiny end-to-end production run with
resume. The end-to-end test uses the trained model artifacts under `outputs/` and is **skipped**
when they are absent (e.g. on a fresh clone); it relies on `fork`, so it is not expected to pass on
native Windows.

---

## Running Inference

### Reproducing the model artifacts

> **A fresh clone does not contain `outputs/`.** Because `outputs/` is git-ignored, the repository
> ships **no** entity caches, trained base-model artifacts (`outputs/experiments/xgb_skeleton/`),
> second-stage artifacts (`outputs/experiments/s1_context/`) or generated city lexicons
> (`outputs/candidates/sample20k_a/lexicon_*.txt`). Reproduce them with the commands below
> **before** running full test inference; `run_pipeline` uses these paths by default.

These are the module invocations used during development (training data only; roughly an hour in
total on the development machine, the OOF step being the longest at ~12 min):

```bash
python -m src.data.convert --datasets train test
python -m src.blocking.entity_cache --datasets train test

# Development candidate set (also provides the city lexicons used for test)
python -m src.blocking.candidate_pipeline --out outputs/candidates/sample20k_a \
    --s1-sample 20000 --seed 42 --chunk-size 5000
python -m src.blocking.validate_candidates outputs/candidates/sample20k_a --recall

# Features + base matcher
python -m src.features.pairwise outputs/candidates/sample20k_a --out outputs/features/sample20k_skeleton
python -m src.models.xgb_matcher outputs/features/sample20k_skeleton --name xgb_skeleton

# Second stage: out-of-fold base scores, then the S1-context model
python -m src.decision.oof outputs/experiments/xgb_skeleton outputs/features/sample20k_skeleton \
    --out outputs/experiments/s1_context
python -m src.decision.s1_context outputs/experiments/xgb_skeleton outputs/features/sample20k_skeleton \
    --oof outputs/experiments/s1_context --compare outputs/experiments/xgb_skeleton
```

`--compare` names a reference experiment whose validation predictions are added to the comparison
table (the row is labelled "xgb_rarity (reference)" in the output; development used `xgb_rarity`,
and any experiment with validation predictions for the same candidates works).

### Full test inference

```bash
python -m src.inference.run_pipeline \
  --dataset test \
  --out outputs/production/test_v1 \
  --chunk-size 10000
```

Defaults: `--base-experiment outputs/experiments/xgb_skeleton`,
`--second-stage outputs/experiments/s1_context`, `--lexicon-dir outputs/candidates/sample20k_a`,
`--workers 8`, `--seed 42`. `--stages` runs a subset (e.g. skip `validate`); `--s1-sample N` runs a
small smoke test. If the run is interrupted, **re-run the same command** to resume.

Measured on the completed full test run (`outputs/production/test_v1`, chunk size 10,000, 8 workers,
16 cores / 23 GB RAM): 617,773,353 candidate rows; ≈ 8.4 h for all six stages (candidates ≈ 5.0 h,
validate ≈ 4 min, features ≈ 2.7 h, scores ≈ 14 min, decisions ≈ 10 min, submission ≈ 12 min);
candidate-stage peak RSS ≈ 5.6 GB; ≈ 46 GB on disk for the run directory.

Run directory:

```text
outputs/production/test_v1/
├── run_manifest.json
├── candidates/   features/   scores/   decisions/
└── submission/output/{matching_results.tsv, candidate_pairs.tsv}
```

All of it is git-ignored.

---

## Submission Outputs

| File | Content |
|---|---|
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` — one row per test S1; comma-separated S2/S3 IDs of accepted matches, empty when none (the leaderboard file) |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` — one row per test S1; every candidate the matcher scored (the exact model input set), empty when blocking found none |

Every S1 appears exactly once; matched IDs are always a subset of the S1's candidates.

## Submission Validation

Run from `student_resource/` (the challenge's stdlib-only validator).

**Validation performed for the full test run** — `matching_results.tsv` only, without `--candidate`
and without `--check-ids`:

```bash
python3 utils/validate_submission.py \
  --matching ../solution/outputs/production/test_v1/submission/output/matching_results.tsv \
  --test-dir dataset/test
```

`candidate_pairs.tsv` was checked separately with streaming structural checks (results in
[Current Status](#current-status)).

**General / full validation command** (matching results and candidate pairs together):

```bash
python3 utils/validate_submission.py \
  --matching ../solution/outputs/production/test_v1/submission/output/matching_results.tsv \
  --candidate ../solution/outputs/production/test_v1/submission/output/candidate_pairs.tsv \
  --test-dir dataset/test
```

> **Warning:** validating the full ≈ 8 GB `candidate_pairs.tsv` this way can require substantial
> resources; on the development machine it caused excessive resource usage and froze the desktop
> session, so it was not completed for the full test run.

`--check-ids` additionally verifies that every ID exists in the test S2/S3 files (uses several GB of RAM).

---

## Reproducibility

- `run_manifest.json` (production): `created`; `candidate_config` (chunk size, seed, workers, dataset,
  lexicon directory, blocking settings); SHA-256 of the base model, its feature list, the second-stage
  model and metadata, and each lexicon; SHA-256 of the input TSVs (unless `--skip-input-hash`); per
  invocation: git HEAD, dirty flag, SHA-256 of the `src/blocking` and `src/preprocessing` sources,
  Python version, platform and package versions; per-stage finish times.
- Stage manifests: candidate config, schema, lexicon origin and hashes, per-part counts; scoring and
  decision manifests record the model hashes, second-stage feature list, floor and threshold.
- Seeds: S1 sample and split seed 42; XGBoost `random_state` 42; OOF folds seeded.
- Candidate generation is deterministic (repeated runs produce identical parts).
- The full test run's `run_manifest.json` records one invocation of all six stages at git HEAD
  `a23f0ba` with a clean working tree, plus SHA-256 hashes of the three test TSVs.

---

## Current Status

Implemented and verified:

- The end-to-end pipeline (data → candidates → features → matcher → decision → output files) is
  implemented for both training and test data.
- It has been validated on development data (the 20k-S1 training sample) and on a small test-data
  smoke run, and has been run on the complete test dataset (below).
- On the development data, the production inference path reproduces the frozen experiments exactly
  (identical features, base scores and second-stage probabilities; validation macro F0.5 0.9254).
- A 600-S1 smoke run on **test** data (including France) of the candidate, validation and feature
  stages completed without errors during development.

Full test inference (`outputs/production/test_v1`, completed):

- Dataset `test`: 1,732,544 S1 processed; candidate rows S2 307,282,922 + S3 310,490,431 =
  617,773,353; the candidate-validation stage passed.
- Final outputs in `outputs/production/test_v1/submission/output/`:
  - `matching_results.tsv` — 1,732,544 data rows + header (95,490,011 bytes, ≈ 95 MB);
    1,620,027 S1 with at least one match, 112,517 with an empty match list; 5,667,444 matched
    entity IDs in total.
  - `candidate_pairs.tsv` — 1,732,544 data rows + header (7,984,780,287 bytes, ≈ 8.0 GB / 7.4 GiB).

Validation of the final files:

- **`matching_results.tsv` passed the official validator** (`utils/validate_submission.py`, run
  without `--candidate`): "PASS — no blocking issues found. Safe to submit."
- **`candidate_pairs.tsv` was not passed through the official validator** — doing so caused excessive
  system resource usage and froze the desktop session. It passed independent streaming structural
  checks instead (`BAD_FIELD_ROWS: 0`, `EMPTY_S1_ROWS: 0`); the same streaming check on
  `matching_results.tsv` reported `BAD_FIELD_ROWS: 0`.
- `--check-ids` was not run.

Not yet done:

- The submission has only been prepared and validated locally; no upload or leaderboard result is
  recorded in this repository.

## Limitations

- **France / unsupported countries:** no city or house+city keys (no validated address rules), so
  one blocking channel and two structured features are unavailable for them.
- Candidate recall is ~97% on the training sample; missed links (mostly Indian cross-script names and
  renamed businesses at weak addresses) cannot be recovered by the matcher.
- The base model and second stage are trained on a 20k-S1 training sample.
- Full-scale runtime, memory and disk figures come from a single run on the development machine.
- Entity-cache building and candidate generation require `fork`: run them on Linux (or WSL2), not native Windows.
- `candidate_pairs.tsv` for the full test set is large (≈ 8.0 GB); the official validator could not
  process it on the development machine, so it was checked with streaming structural checks instead.

## License

The repository does not currently include a license file. Third-party dependencies (e.g. XGBoost,
Apache-2.0) keep their own licenses.
