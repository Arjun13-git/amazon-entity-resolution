# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains the solution pipeline for the **Amazon ML Challenge 2026
Business Entity Resolution Challenge**: for every Source 1 (S1) business record,
find all records in Source 2 (S2) and Source 3 (S3) that refer to the same
real-world business, optimizing the challenge's per-S1 macro F0.5 metric.

The pipeline is being built stage by stage. Everything up to and including
pairwise feature extraction, labeling and the feature audit is implemented and
validated on a 20,000-S1 training sample. **No matching model has been trained yet.**

---

## 1. Current pipeline status

| Stage | Status |
| --- | --- |
| Data loading (TSV loaders, Parquet conversion) | ✅ Implemented |
| Multilingual text normalization | ✅ Implemented |
| Transliteration utility | ✅ Implemented |
| Entity caching (normalized, integer-ID Parquet cache) | ✅ Implemented |
| Candidate generation / blocking (streaming) | ✅ Implemented |
| Candidate validation | ✅ Implemented |
| Candidate recall benchmarking | ✅ Implemented |
| Pairwise feature extraction | ✅ Implemented |
| Ground-truth candidate labeling | ✅ Implemented |
| Feature auditing | ✅ Implemented |
| S1-level train/validation split | ✅ Implemented |
| Macro F0.5 metric utility | ✅ Implemented (`src/evaluation/f05.py`) |
| XGBoost matcher | ❌ Not implemented |
| Decision / threshold optimization | ❌ Not implemented |
| Final set-level matching logic | ❌ Not implemented |
| Optional global conflict resolution | ❌ Not implemented |
| Full-scale test inference | ❌ Not implemented |
| Final submission packaging (`matching_results.tsv`, `candidate_pairs.tsv`) | ❌ Not implemented |

### Known limitations of the current code

- **Training data only.** `src/data/convert.py` and `src/blocking/entity_cache.py`
  only process the `train_*` files. Nothing reads the test files yet.
- **France is not supported yet.** The test set contains a third country, `France`,
  that does not appear in training (see [Challenge overview](#2-challenge-overview)).
  `src/blocking/address_keys.py` only has state tables for the US and India, so
  `extract_keys` (used by the house-number + city channel and the structured
  features) raises a `KeyError` for any other country. This must be generalized
  before test inference.
- The candidate pipeline has only been run on the 20,000-S1 benchmark sample, not
  on the full 2.2M training S1 set.

---

## 2. Challenge overview

Facts below come from the challenge's `student_resource/README.md` and from
measurements on the provided training data.

### Sources

Each source file has the columns `entity_id`, `business_name`, `business_address`,
`country`. The ID prefix (`S1-`, `S2-`, `S3-`) identifies the source.

- **S1** is the deduplicated reference source.
- **S2** and **S3** are independent, noisy sources. Names may contain abbreviations,
  legal suffixes, typos and transliterations. Addresses may be partial, reordered,
  landmark-based or missing components.
- An S1 entity may match **zero, one or many** S2/S3 records.

| File | Records |
| --- | --- |
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `test_source1.tsv` | 1,732,544 |
| `test_source2.tsv` | 4,887,273 |
| `test_source3.tsv` | 5,082,316 |

Training data covers `US` and `India`. The **test set also contains `France`**
(259,452 test S1 records), which does not appear in training. The challenge asks
that `country` be treated as an open set of labels.

### Ground truth

`train_ground_truth.tsv` has one row per S1 entity:

| Column | Meaning |
| --- | --- |
| `source1_entity_id` | S1 entity ID |
| `matched_entity_ids` | Comma-separated S2/S3 IDs; **empty when the S1 entity has no match** |

Measured on the training data:

- **7,638,365 true links**: 3,693,619 to S2 and 3,944,746 to S3.
- **123,247 unmatched S1 entities (5.58%)**, which have an empty `matched_entity_ids`.
- Matched S1 entities have 3.67 links on average (maximum 11).
- **Distractor targets.** 1,340,997 S2 records (26.6%) and 1,340,857 S3 records
  (25.4%) are not linked to any S1 entity. No S2/S3 record is linked to more than
  one S1 entity.

### Why all-pairs matching is infeasible

Comparing every training S1 record with every S2 and S3 record would mean about
2.2M × 10.3M ≈ 2.3 × 10¹³ pairs. Even within a single country (US: 1.32M S1 against
about 6.2M S2+S3 records) it is about 8 × 10¹² pairs. Candidate generation (blocking)
therefore has to reduce this to a few hundred candidates per S1 while keeping
almost all true matches.

### Evaluation metric

F0.5 is computed **per S1 entity**, then macro-averaged over **all** S1 entities:

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- F0.5 weights precision more than recall, so false merges cost more than missed links.
- An S1 entity with no true match scores **1.0** for an empty prediction and **0.0**
  for any non-empty prediction. Correctly predicting "no match" matters, and false
  merges on unmatched S1 entities are penalized.
- Candidate generation sets the upper bound on recall: a true link that is not a
  candidate can never be predicted.

`src/evaluation/f05.py` implements `fbeta`, `set_f05` (per-S1 F0.5) and `macro_f05`
with these rules.

---

## 3. Repository structure

```text
solution/
├── README.md
├── requirements.txt           # pinned dependencies
├── .gitignore
├── configs/
│   └── paths.yaml             # not read by any code (see note below)
├── experiments/
│   └── experiment_log.csv     # header only; no experiments logged yet
└── src/
    ├── data/
    │   ├── loader.py          # safe TSV / ground-truth loaders (+ small inspection CLI)
    │   └── convert.py         # train TSV -> Parquet conversion
    ├── preprocessing/
    │   ├── normalize.py       # Unicode-safe name / address / country normalization
    │   └── transliterate.py   # Latin-script transliteration (unidecode)
    ├── blocking/
    │   ├── entity_cache.py        # normalized integer-ID cache + ground-truth cache
    │   ├── channels.py            # retrieval channels (exact, char n-gram, address, house+city)
    │   ├── address_keys.py        # house number / postal / city key extraction
    │   ├── candidate_pipeline.py  # FINAL streaming candidate generation
    │   ├── validate_candidates.py # candidate-set validation
    │   ├── benchmark_candidates.py        # per-channel recall benchmark (+ memory estimates)
    │   ├── diagnose_union.py              # exact ∪ char overlap / miss diagnosis, rerank variants
    │   ├── benchmark_address_union.py     # address channel benchmark
    │   ├── benchmark_translit_union.py    # transliterated-name channel benchmark (not adopted)
    │   ├── benchmark_structured_union.py  # structured address-key benchmark
    │   ├── profile_true_matches.py        # similarity profile of true S1->S2/S3 links
    │   ├── inspect_zero_name_matches.py   # inspection of zero-name-similarity true links
    │   └── exact_name.py, token_index.py, char_tfidf.py,
    │       benchmark_exact_name.py, benchmark_token_blocking.py,
    │       benchmark_char_tfidf.py, test_exact_name.py
    │                                      # early dict-based prototypes (superseded)
    ├── features/
    │   ├── text_features.py       # name similarity features
    │   ├── address_features.py    # address similarity + structured-key features
    │   ├── retrieval_features.py  # candidate-provenance features
    │   ├── pairwise.py            # streaming feature extraction + labeling driver
    │   └── audit_features.py      # feature / label audit
    ├── evaluation/
    │   ├── f05.py             # per-S1 and macro F0.5
    │   ├── split.py           # S1-level train/validation split
    │   └── labels.py          # candidate labeling against ground truth
    ├── decision/              # empty package (placeholder)
    └── inference/             # empty package (placeholder)
```

Notes:

- `src/decision/` and `src/inference/` only contain an empty `__init__.py`.
- There is no `src/models/` directory in the repository.
- `configs/paths.yaml` is not read by any code. Paths are resolved relative to the
  repository (see [Dataset setup](#5-dataset-setup)).
- Generated data goes to `outputs/`, which git ignores.

### Blocking channels (final design)

Candidate generation runs **per target source (S2, S3) and per country partition**,
and combines four channels (`src/blocking/channels.py`):

| Channel | What it does | Final setting |
| --- | --- | --- |
| `exact_name` | All targets whose normalized business name equals the S1 name. Uses sorted hash arrays + `searchsorted`, not a Python dict. | all matches |
| `name_char` | Character-trigram TF-IDF over normalized names. Uses hashed features (no vocabulary dict) with IDF computed from the targets. Trigrams appearing in more than 30,000 target names are left out of the sparse index. The top 1,000 targets by pruned score are **reranked by full cosine similarity** (all trigrams). | rerank pool = 1000, final K = **100** |
| `address_char` | The same machinery over normalized addresses. | rerank pool = 1000, final K = **25** |
| `house_city` | Exact match on an extracted `house number | city` address key, kept only when the key's block holds at most 1,000 targets. | block cap = **1000** |

The channels' candidates are **unioned and deduplicated** on the integer pair
(S1 ID, target ID). The output keeps which channels retrieved each pair, plus
each channel's rank and score. This candidate set is the input to the (future)
matcher.

Channels that were benchmarked and **not** adopted:

- **Transliterated-name retrieval** (`TransliteratedNameChannel`): about +0.2–0.3 recall points.
- **Postal-code blocking**: postal codes are almost never present in the data.

---

## 4. Environment / setup

### Requirements

- Linux (development and all measurements were done on Linux x86-64).
- **Python 3.12.**
- [uv](https://docs.astral.sh/uv/) for the virtual environment. The repository has
  **no `pyproject.toml` or `uv.lock`**; dependencies are pinned in `requirements.txt`.
- **No GPU or CUDA is required** for anything implemented so far. `requirements.txt`
  includes `nvidia-nccl-cu13` only because it is pinned as a dependency of the
  `xgboost` Linux wheel.

### Fresh clone

The code finds the dataset at `../student_resource/dataset` relative to the
repository root, so clone the repository **next to** the challenge's
`student_resource/` directory:

```bash
cd <challenge-workspace>          # the directory that contains student_resource/
git clone https://github.com/Arjun13-git/amazon-entity-resolution.git solution
cd solution

uv venv --python 3.12 .venv       # uv downloads Python 3.12 if it is not installed
source .venv/bin/activate
uv pip install -r requirements.txt
```

Run every command in this README from the repository root with the environment
activated. All entry points are Python modules (`python -m src....`).

### Resource considerations

These are measured on the development machine (16 CPU cores, 23 GB RAM):

- The streaming candidate pipeline peaked at about **5.1 GB RSS** in the parent
  process on the 20,000-S1 sample. It holds fitted country-partition indexes of up
  to 3.2M targets.
- The char-channel worker processes (default 8, forked) share the index
  copy-on-write. The largest worker reported about 3.1 GB RSS, which includes
  shared pages.
- **16 GB of RAM or more is recommended.** Lower `--workers` to reduce memory.
- The CPU is the bottleneck: the char channels use one process per worker.
- Disk use for the 20k-sample run: normalized cache about 770 MB, candidates about
  60 MB, features about 155 MB.

---

## 5. Dataset setup

The challenge dataset is **not** part of this repository and **must not be
committed**. Obtain it from the challenge separately and place it like this:

```text
<challenge-workspace>/
├── student_resource/
│   └── dataset/
│       ├── train/
│       │   ├── train_source1.tsv
│       │   ├── train_source2.tsv
│       │   ├── train_source3.tsv
│       │   └── train_ground_truth.tsv
│       └── test/
│           ├── test_source1.tsv
│           ├── test_source2.tsv
│           └── test_source3.tsv
└── solution/                     # this repository
```

All files are **tab-separated** and must be read with `sep="\t"`. Addresses and ID
lists contain commas. The repository loaders read every column as a string with
`keep_default_na=False`, so empty values stay empty strings rather than NaN.

---

## 6. Data preprocessing

### TSV loading — `src/data/loader.py`

- `load_entities(path, usecols=None, nrows=None)` validates the column names and
  reads all values as strings.
- `load_ground_truth(path, nrows=None)` validates the ground-truth header.

A small inspection CLI is included:

```bash
python -m src.data.loader ../student_resource/dataset/train/train_source1.tsv --nrows 1000
python -m src.data.loader ../student_resource/dataset/train/train_ground_truth.tsv --ground-truth
```

### Parquet conversion — `src/data/convert.py`

This converts the three **training** source TSVs to zstd-compressed Parquet in
`outputs/parquet/`. Existing files are skipped.

```bash
python -m src.data.convert
```

### Normalization — `src/preprocessing/normalize.py`

- **Unicode-safe.** NFKC normalization and lowercasing. Unicode letters, digits and
  combining marks are kept in every script; punctuation and symbols become spaces;
  whitespace is collapsed. Names in Devanagari, Telugu, Kannada, Tamil, Bengali and
  other scripts therefore stay intact instead of being stripped to ASCII. Stripping
  to ASCII would erase them (about 23% of Indian S2 names are non-Latin).
- **Business names.** Common legal suffixes and prefixes (`inc`, `llc`, `ltd`,
  `pvt`, `private`, `limited`, `gmbh`, `sarl`, …) are removed from the start and end
  of the token sequence. The list is Latin-script only, so non-Latin suffixes such
  as `प्राइवेट लिमिटेड` are not removed.
- **Addresses and countries.** The same text normalization is applied, without
  suffix removal.

### Transliteration — `src/preprocessing/transliterate.py`

`transliterate_text` maps text to Latin script with `unidecode`. It is used for the
transliterated-name features. As a retrieval channel it was benchmarked and not
adopted: phonetic transliteration of Indic-script names (for example
`फ्यूचर लोटस` → `phyuucr lotts`) rarely matches the English spelling.

### Entity cache — `src/blocking/entity_cache.py`

This reads `outputs/parquet/` and writes normalized, integer-ID caches to
`outputs/normalized/`:

- `train_source{1,2,3}.parquet` with the columns `id` (int32), `country`, `norm_name`
  and `norm_address`
- `train_truth.parquet` with `s1_id`, `target_source` (2/3) and `target_id`

Every later stage reads one country partition at a time from this cache.

```bash
python -m src.blocking.entity_cache            # --workers 8 (default), --force to rebuild
```

---

## 7. Candidate generation

### Architecture — `src/blocking/candidate_pipeline.py`

For each target source (S2, S3) and each country partition:

1. Load the target partition and fit all four channels once.
2. Process S1 in chunks (`--chunk-size`, default 50,000).
3. Run every channel on the chunk, union the results on integer (S1, target) keys,
   and deduplicate them.
4. Write one Parquet part per chunk:
   `outputs/candidates/<run>/target=S2/country=us/part-00000.parquet`.

The full candidate set is never held in memory, only one chunk at a time.
`open_candidates(<run dir>)` returns a lazy `pyarrow` dataset over all parts.

Each candidate row has these columns:

| Column | Meaning |
| --- | --- |
| `s1_id`, `target_id` | integer IDs (the `S1-`/`S2-`/`S3-` prefix removed) |
| `target_source` | 2 = S2, 3 = S3 |
| `country` | normalized country of the partition |
| `candidate_source_mask` | bitmask: `exact_name`=1, `name_char`=2, `address_char`=4, `house_city`=8 |
| `exact_name_rank` | position within the exact-name block (−1 if not retrieved) |
| `name_char_rank`, `name_char_score` | name channel rank (−1) and full-cosine score (NaN) |
| `address_char_rank`, `address_char_score` | address channel rank (−1) and full-cosine score (NaN) |
| `house_city_hit` | retrieved by the house-number + city channel |

Each run directory also contains:

- `manifest.json`: config, schema, per-part counts and bytes, fit times, peak
  memory, and code version (git HEAD, dirty flag, SHA-256 of the blocking and
  preprocessing sources)
- the learned city lexicons (`lexicon_{country}.txt`)
- for sampled runs, `s1_ids.parquet`

These are enough to regenerate the candidate set exactly.

### Measured recall

All figures are **candidate recall against training ground truth**. The final
design was selected on a **random sample of 20,000 training S1 entities (seed 42)**
evaluated against the **complete** S2/S3 training populations. These are
blocking-stage results, **not** test or leaderboard performance.

| Configuration | S2 recall | S3 recall | Scope |
| --- | --- | --- | --- |
| Exact normalized name | 41.49% | 40.11% | all 2.2M training S1 |
| Exact normalized name | 41.56% | 39.76% | 20k-S1 sample |
| Name char, pruned top-100 (no rerank) | 65.88% | 62.54% | 20k-S1 sample |
| Name char, rerank 1000 → top-100 | 71.35% | 70.10% | 20k-S1 sample |
| Exact ∪ name-char rerank | 73.75% | 73.03% | 20k-S1 sample |
| … ∪ address-char rerank K=25 | 96.61% | 96.02% | 20k-S1 sample |
| **… ∪ house-number + city (block ≤ 1000) — final** | **97.35%** | **97.04%** | 20k-S1 sample |

For the final union on the 20k sample:

| | S2 | S3 |
| --- | --- | --- |
| Candidate pairs | 3,433,276 | 3,493,526 |
| Candidates per S1: mean / median / max | 171.7 / 125 / 1,113 | 174.7 / 125 / 1,133 |

Every sampled S1 entity has at least one candidate.

The full 2.2M-S1 candidate set has **not** been generated yet.

---

## 8. Candidate validation

`src/blocking/validate_candidates.py` streams a candidate run one part at a time
and checks:

1. **No duplicates.** Within each part, (s1_id, target_id) is strictly increasing,
   and no S1 entity appears in more than one part.
2. **Country consistency.** Every S1 and target ID exists in the part's country
   partition, and the `country` and `target_source` columns match the part.
3. **Channel provenance.** Mask bits agree with the rank, score and hit columns,
   and no row has an empty or unknown mask.
4. **Exact-name correctness and completeness.** Every `exact_name` pair has equal,
   non-empty normalized names, and every same-name target is present.
5. **House-city correctness and completeness.** Every `house_city` pair shares a
   non-empty key, only blocks of at most 1,000 targets are used, and those blocks
   are complete.
6. **Rank and score checks.** Char-channel ranks are 0…n−1 per S1 and below K;
   scores are in [0, 1] and never increase with rank.
7. **Per-S1 counts.** Mean, median, p99 and maximum candidates per S1.
8. **Streaming iteration.** All parts are read through `open_candidates` in
   batches, and the row count must equal the manifest total.
9. **Determinism** (`--compare <other run>`). The configs are equal, the part
   files are the same, and the content is identical (NaN-aware comparison).

`--recall` also reports recall per target and per channel against ground truth.

On the 20k sample, two independent pipeline runs with the same seed and config
produced **identical candidate content**, and all checks passed. The validator was
also confirmed to catch an injected duplicate row, a flipped provenance bit and a
dropped `house_city` row.

---

## 9. Pairwise feature extraction

`src/features/pairwise.py` reads a candidate run one part at a time and writes one
feature part per candidate part: `outputs/features/<run>/target=…/country=…/part-*.parquet`.

- Transliteration and address keys are computed once per unique entity in a part.
- Features are float32 (flags are int8) and IDs are int32.
- The candidate set is never modified.

| Group | Features |
| --- | --- |
| Name (`text_features.py`) | `name_exact`, `name_ratio`, `name_partial_ratio`, `name_token_sort_ratio`, `name_token_set_ratio`, `name_token_jaccard`, `translit_name_ratio`, `translit_name_token_jaccard` |
| Address (`address_features.py`) | `address_ratio`, `address_partial_ratio`, `address_token_sort_ratio`, `address_token_set_ratio`, `address_token_jaccard` |
| Structured (`address_features.py`) | `house_number_match`, `city_match`, `house_city_match`, `country_match` |
| Retrieval (`retrieval_features.py`) | `exact_name_hit`, `name_char_hit`, `address_char_hit`, `house_city_hit`, `n_channels`, `name_char_rank`, `name_char_score`, `address_char_rank`, `address_char_score`, `best_name_rank`, `best_address_rank`, `s1_candidate_count` |
| Missingness | `s1_name_missing`, `target_name_missing`, `s1_address_missing`, `target_address_missing` |

String similarities use RapidFuzz (`cpdist`, multi-threaded element-wise) and are
scaled to [0, 1].

**Missing-value handling:**

- Name and address similarities, including the transliterated ones and
  `name_exact`, are **NaN when either side's text is empty**. Missing text is
  never treated as "dissimilar".
- Structured matches (house number, city, house + city) are 1 or 0, and NaN
  when either side has no extracted key.
- Retrieval ranks and scores are NaN when that channel did not retrieve the pair.
- `best_name_rank` is 0 for exact-name hits, otherwise the name-char rank.
- `best_address_rank` is 0 for house-city hits, otherwise the address-char rank.
- The four `*_missing` flags record empty normalized name or address explicitly.

Each row also has `s1_id`, `target_id`, `target_source`, `country`, `split`
(0 = train, 1 = validation) and `label`.

---

## 10. Candidate labeling

`src/evaluation/labels.py`:

- **Labels from ground truth.** `label = 1` if and only if the candidate's
  target is in that S1 entity's `matched_entity_ids`, otherwise 0.
- **S2 and S3 are kept separate.** `TruthIndex` is built for one target source
  and raises an error if it is given candidates from the other source. An S2
  candidate can never be labeled from S3 truth, or the reverse.
- **Empty `matched_entity_ids`** means the S1 entity has no true target. It has no
  truth rows, so all of its candidates are negatives.
- **Independent check.** `raw_truth_strings` rebuilds the links as `"S1-…|S2-…"`
  strings directly from the raw TSV, without the integer cache. The audit uses it
  to re-verify every label.

**S1-level split.** `src/evaluation/split.py` (`make_s1_validation_split`) assigns
each S1 entity entirely to train or validation. All candidates of an S1 entity stay
in one split, which prevents leakage between candidate rows. `pairwise.py` uses a
validation fraction of 0.2 and seed 42 by default, and saves the assignment to
`split.parquet`.

---

## 11. Feature audit results (20k-S1 sample)

From `python -m src.features.audit_features` on the 20k-sample feature set. These
are **sample / development observations**, not final model or test results.

| Metric | Value |
| --- | --- |
| Candidate pairs | 6,926,802 |
| Positive pairs | 66,996 (S2 32,535, S3 34,461) |
| Negative pairs | 6,859,806 |
| Positive rate | 0.967% |
| Positives per S1 | S2 mean 1.63 (max 5), S3 mean 1.72 (max 6) |
| Negatives per S1 | S2 mean 170, S3 mean 173 (median 124) |
| Train / validation S1 | 16,000 / 4,000 (zero overlap) |
| Train / validation positives | 53,632 / 13,364 |
| Label disagreements vs raw ground truth | 0 |
| Candidate recall for the sampled S1 (S2+S3) | 97.19% (66,996 / 68,931 true links) |

Observations:

- **High name similarity alone produces many false positives.** 783,863 negatives
  (11.3% of all candidates) have a name ratio of at least 0.9. Only about 4% of
  candidates with *identical* names are true matches; the rest are mostly generic
  names in different cities.
- **Address similarity is more discriminative.** 44% (S2) and 55% (S3) of
  candidates with an address ratio of at least 0.9 are positive. Address token
  Jaccard averages 0.64 for positives and 0.10 for negatives. The remaining hard
  negatives are mostly neighbouring units or other businesses in the same building.
- **House-city-only candidates are numerous and almost all negative.** 1.70M rows
  (24.5% of candidates) contain 611 positives, a 0.036% positive rate.
- **Constant or duplicate features.**
  - `country_match`, `s1_name_missing` and `s1_address_missing` are constant.
  - `name_exact` is identical to `exact_name_hit`.
  - `house_city_match` is almost identical to `house_city_hit` (r = 0.991).
- **F0.5 makes precision important.** With about 1 positive per 100 candidates,
  and many high-name-similarity negatives, the matcher and decision layer must
  favour precision.

---

## 12. Reproducing the current benchmark

Run these from the repository root with the environment activated. Runtimes are
from logs on the development machine (16 cores, 8 workers).

```bash
# 1. Train TSV -> Parquet (outputs/parquet/)
python -m src.data.convert

# 2. Normalized integer-ID cache + truth cache (outputs/normalized/)   ~45 s
python -m src.blocking.entity_cache

# 3. Candidate generation on the 20k-S1 sample                          ~11 min
python -m src.blocking.candidate_pipeline \
    --out outputs/candidates/sample20k_a --s1-sample 20000 --seed 42 --chunk-size 5000

# 4. Candidate validation + recall                                      ~2.5 min
python -m src.blocking.validate_candidates outputs/candidates/sample20k_a --recall

#    Optional determinism check: generate a second run and compare
python -m src.blocking.candidate_pipeline \
    --out outputs/candidates/sample20k_b --s1-sample 20000 --seed 42 --chunk-size 5000
python -m src.blocking.validate_candidates outputs/candidates/sample20k_a \
    --recall --compare outputs/candidates/sample20k_b

# 5. Pairwise features + labels + S1 split (outputs/features/sample20k/) ~100 s
python -m src.features.pairwise outputs/candidates/sample20k_a --out outputs/features/sample20k

# 6. Feature / label audit                                              ~15 s
python -m src.features.audit_features outputs/features/sample20k
```

`candidate_pipeline` and `pairwise` refuse to write into an existing output
directory. Delete the directory or pick a new `--out`.

The research benchmarks behind the channel choices can also be rerun. Unless noted,
each one evaluates the 20k sample against the complete target populations. Logged
runtimes range from about 1 minute (exact name) to about 25 minutes
(`benchmark_address_union`):

```bash
python -m src.blocking.profile_true_matches --sample 50000 --seed 42
python -m src.blocking.benchmark_candidates --channels exact            # all 2.2M S1, ~1 min
python -m src.blocking.benchmark_candidates --channels char --s1-sample 20000 --estimate-only
python -m src.blocking.diagnose_union --rerank-pool 1000 --renormalize-pruned
python -m src.blocking.benchmark_address_union
python -m src.blocking.benchmark_translit_union
python -m src.blocking.benchmark_structured_union
```

---

## 13. Reproducibility

- **Seeds.**
  - The 20k S1 sample uses `numpy.random.default_rng(42)` (`--s1-sample 20000 --seed 42`).
  - The S1 train/validation split uses validation fraction 0.2 and seed 42.
  - The audit's negative subsample uses seed 0.
- **Determinism.** Two independent candidate runs with the same config produced
  identical content (checked with `validate_candidates --compare`).
- **Version tracking.**
  - Each candidate `manifest.json` records the git HEAD, a dirty-tree flag and a
    SHA-256 hash of the `src/blocking` and `src/preprocessing` sources.
  - The feature manifest copies this information and records the split settings
    and feature columns.
- **S1-level split** as described in [Candidate labeling](#10-candidate-labeling).
- **Generated outputs are ignored by git.** `outputs/`, `logs/`, `cache/`, `models/`,
  `artifacts/` and `/data/` are in `.gitignore`.
- **Do not commit the dataset or generated Parquet artifacts** (caches, candidates,
  features). They are large and reproducible from the commands above.

---

## 14. Current results summary

| Stage | Result | Scope |
| --- | --- | --- |
| Exact-name blocking | S2 41.49%, S3 40.11% candidate recall | all 2.2M training S1, full S2/S3 |
| Final candidate union | S2 97.35%, S3 97.04% candidate recall; about 173 candidates per S1 | 20k training S1 sample (seed 42), full S2/S3 |
| Candidate validation | all checks passed; repeated runs identical | 20k-sample candidate run |
| Feature extraction | 6,926,802 labeled pairs × 33 features, 155 MB Parquet, about 100 s | 20k-sample candidate run |
| Label verification | 0 disagreements with raw ground truth | all 6.9M sample candidate pairs |
| Candidate recall (S2+S3 combined) | 97.19% (66,996 / 68,931 true links) | 20k training S1 sample |

None of these numbers is a matching F0.5 or a test/leaderboard result. No matcher
has been trained yet.

---

## 15. Next step

**Next stage:** train and validate the first XGBoost pairwise matcher using the
generated feature dataset, with the S1-level split and per-S1 macro F0.5.

Model training has **not** been implemented yet. The decision/threshold layer,
set-level matching, full-scale and test inference (including support for the
unseen `France` country) and submission packaging also remain to be built.
