# Amazon Entity Resolution

Machine learning pipeline for the **Amazon ML Challenge 2026 – Business Entity Resolution Challenge**.

## Objective

Resolve entities from source datasets to their corresponding entities across target datasets while optimizing the challenge's **per-entity macro F0.5** metric.

## Pipeline

```text
Raw Data
    ↓
Normalization
    ↓
Candidate Generation / Blocking
    ↓
Pairwise Feature Engineering
    ↓
ML Matching
    ↓
F0.5-Tuned Decision Layer
    ↓
Validation
    ↓
Submission
```

## Core Approach

- Robust multilingual text normalization
- High-recall candidate generation and blocking
- Name and address similarity features
- Gradient-boosted tree based entity matching
- F0.5-oriented decision optimization
- Set-level match selection
- Optional global assignment for conflict resolution
- Optional deep-learning components for difficult multilingual cases

## Environment

- Python 3.12
- uv
- pandas
- NumPy
- SciPy
- scikit-learn
- PyArrow
- RapidFuzz
- XGBoost

## Project Structure

```text
solution/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── blocking/
│   ├── features/
│   ├── models/
│   ├── decision/
│   ├── evaluation/
│   └── inference/
├── configs/
├── experiments/
├── outputs/
├── logs/
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The original challenge dataset is kept outside this repository and treated as **read-only**.

Large datasets, generated outputs, models, caches, and logs are excluded from version control.

## Development Strategy

The system is developed incrementally:

1. Infrastructure and data pipeline
2. Baseline matching
3. Normalization
4. Candidate generation / blocking
5. Pairwise feature engineering
6. ML matcher
7. F0.5 decision optimization
8. Conflict resolution
9. Optional hard-case modeling
10. Full inference and submission validation

## Evaluation

Experiments are evaluated using the challenge's **per-S1 macro F0.5** metric, with candidate-generation recall tracked separately.

Each experiment records its configuration, candidate statistics, model settings, precision, recall, F0.5, and runtime.

## Status

**Phase 1 — Infrastructure setup**
