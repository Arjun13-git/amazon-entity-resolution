"""
Baseline XGBoost pairwise matcher.

Trains an ``XGBClassifier`` on precomputed pairwise features (from
``src.features.pairwise``) and evaluates it with the challenge objective:
per-S1 macro F0.5 over the validation S1 entities, sweeping the
probability threshold.

Data handling:
- Features are read from the feature Parquet parts; nothing is recomputed.
- The S1-level split stored with the features is used as is. Early
  stopping uses a further S1-level hold-out carved from the TRAIN split,
  so the validation split is only used for threshold selection and
  reporting.
- Class imbalance: ``scale_pos_weight = negatives / positives`` of the
  rows the model is fitted on.

Evaluation:
- An S1's ground truth is all of its S2 and S3 links, including links the
  candidate generator missed, so validation F0.5 includes blocking losses.
- Validation S1 entities with no true match count (1.0 for an empty
  prediction, 0.0 otherwise), as in the challenge metric.

Artifacts (under ``--out``, default ``outputs/experiments/<name>``):
    model.json                   XGBoost model
    features.json                final feature list (+ dropped features)
    config.json                  model / experiment configuration
    metadata.json                git, data, counts, timings, threshold
    threshold_sweep.csv          metrics for every threshold
    validation_predictions.parquet
    diagnostics.json

    python -m src.models.xgb_matcher outputs/features/sample20k --name xgb_baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from src.evaluation.f05 import f05_from_counts, macro_f05
from src.evaluation.labels import TruthIndex, raw_truth_strings
from src.evaluation.split import make_s1_validation_split


ROOT = Path(__file__).resolve().parents[2]

# Constant on the development data.
CONSTANT_FEATURES = ["country_match", "s1_name_missing", "s1_address_missing"]

# (dropped, kept): exact / near-exact duplicates.
DUPLICATE_FEATURES = [
    ("name_exact", "exact_name_hit"),
    ("house_city_match", "house_city_hit"),
]

ID_COLUMNS = ["s1_id", "target_id", "target_source", "split", "label"]

TARGET_PREFIX = {2: "S2", 3: "S3"}


@dataclass
class MatcherConfig:
    seed: int = 42
    early_stopping_fraction: float = 0.10
    # 0.05..0.95, then finer steps near 1: with scale_pos_weight ~100 the
    # scores are pushed towards 1, so the F0.5 optimum lies above 0.95.
    thresholds: list[float] = field(
        default_factory=lambda: [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]
        + [0.96, 0.97, 0.98, 0.99]
        + [round(t, 3) for t in np.arange(0.991, 0.9995, 0.001)]
        + [0.9995, 0.9999]
        + [0.99991, 0.99993, 0.99995, 0.99997, 0.99999]
    )
    xgb_params: dict = field(
        default_factory=lambda: {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "device": "cpu",
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "max_bin": 512,
            "early_stopping_rounds": 100,
            "n_jobs": os.cpu_count(),
        }
    )


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


def select_features(all_features: list[str]) -> tuple[list[str], dict[str, str]]:
    """Final feature list and the reason each dropped feature was removed."""

    dropped = {name: "constant" for name in CONSTANT_FEATURES}
    dropped.update({drop: f"duplicate of {keep}" for drop, keep in DUPLICATE_FEATURES})

    missing = [name for name in dropped if name not in all_features]
    if missing:
        raise ValueError(f"Expected features not found in the dataset: {missing}")

    return [name for name in all_features if name not in dropped], dropped


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    s1_id: np.ndarray
    target_id: np.ndarray
    target_source: np.ndarray


def load_feature_splits(feature_dir: Path, features: list[str]) -> dict[int, SplitData]:
    """Read feature parts into float32 matrices, grouped by the stored split."""

    manifest = json.loads((feature_dir / "manifest.json").read_text())
    parts: dict[int, list[dict[str, np.ndarray]]] = {0: [], 1: []}

    for meta in manifest["parts"]:
        table = pq.read_table(feature_dir / meta["file"], columns=ID_COLUMNS + features)
        split = table["split"].to_numpy()

        X = np.empty((table.num_rows, len(features)), dtype=np.float32)
        for j, name in enumerate(features):
            X[:, j] = table[name].to_numpy(zero_copy_only=False)

        for flag in (0, 1):
            m = split == flag
            parts[flag].append(
                {
                    "X": X[m],
                    "y": table["label"].to_numpy()[m],
                    "s1_id": table["s1_id"].to_numpy()[m],
                    "target_id": table["target_id"].to_numpy()[m],
                    "target_source": table["target_source"].to_numpy()[m],
                }
            )

        del table, X

    return {
        flag: SplitData(
            **{key: np.concatenate([p[key] for p in chunks]) for key in chunks[0]}
        )
        for flag, chunks in parts.items()
    }


def true_link_counts(s1_ids: np.ndarray) -> np.ndarray:
    """Number of true S2+S3 links for each id in sorted ``s1_ids``."""

    counts = np.zeros(len(s1_ids), dtype=np.int64)

    for target in TARGET_PREFIX.values():
        linked_s1 = (TruthIndex(target, s1_ids).keys >> 32).astype(np.int64)
        counts += np.bincount(np.searchsorted(s1_ids, linked_s1), minlength=len(s1_ids))

    return counts


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def s1_metrics(
    s1_index: np.ndarray,
    label: np.ndarray,
    selected: np.ndarray,
    n_true: np.ndarray,
) -> dict:
    """Per-S1 counts and macro metrics for one set of selected candidates."""

    n = len(n_true)
    n_pred = np.bincount(s1_index[selected], minlength=n)
    tp = np.bincount(s1_index[selected & (label == 1)], minlength=n)

    f05 = f05_from_counts(tp, n_pred, n_true)
    has_pred = n_pred > 0
    has_true = n_true > 0

    return {
        "macro_f05": float(f05.mean()),
        # Precision is averaged over S1 with >=1 prediction, recall over
        # S1 with >=1 true link (each is undefined otherwise).
        "macro_precision": float((tp[has_pred] / n_pred[has_pred]).mean()) if has_pred.any() else np.nan,
        "macro_recall": float((tp[has_true] / n_true[has_true]).mean()) if has_true.any() else np.nan,
        "micro_precision": float(tp.sum() / max(n_pred.sum(), 1)),
        "micro_recall": float(tp.sum() / max(n_true.sum(), 1)),
        "s1_with_prediction": int(has_pred.sum()),
        "avg_predicted_per_s1": float(n_pred.mean()),
        "f05_singletons": float(f05[~has_true].mean()) if (~has_true).any() else np.nan,
        "f05_matched": float(f05[has_true].mean()) if has_true.any() else np.nan,
        "_n_pred": n_pred,
        "_f05": f05,
    }


def threshold_sweep(
    s1_index: np.ndarray,
    label: np.ndarray,
    score: np.ndarray,
    n_true: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:

    rows = []
    for t in thresholds:
        m = s1_metrics(s1_index, label, score >= t, n_true)
        rows.append({"threshold": t, **{k: v for k, v in m.items() if not k.startswith("_")}})

    return pd.DataFrame(rows)


def check_with_reference_metric(
    val: SplitData,
    score: np.ndarray,
    threshold: float,
    val_s1: np.ndarray,
) -> float:
    """
    Recompute macro F0.5 at ``threshold`` with the string-set reference
    implementation (``macro_f05``) and truth rebuilt from the raw TSV.
    """

    ground_truth: dict[str, list[str]] = {f"S1-{i}": [] for i in val_s1}
    for link in raw_truth_strings(val_s1):
        s1, target = link.split("|")
        ground_truth[s1].append(target)

    predictions: dict[str, list[str]] = {}
    sel = score >= threshold
    for s1, src, tid in zip(val.s1_id[sel], val.target_source[sel], val.target_id[sel]):
        predictions.setdefault(f"S1-{s1}", []).append(f"{TARGET_PREFIX[int(src)]}-{tid}")

    return macro_f05(predictions, ground_truth)


def diagnostics(
    val: SplitData,
    score: np.ndarray,
    threshold: float,
    s1_index: np.ndarray,
    n_true: np.ndarray,
    features: list[str],
    n_examples: int = 10,
) -> dict:

    pos = score[val.y == 1]
    neg = score[val.y == 0]
    q = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    m = s1_metrics(s1_index, val.y, score >= threshold, n_true)
    n_pred = m["_n_pred"]

    def examples(mask: np.ndarray, ascending: bool) -> list[dict]:
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(score[idx])]
        if not ascending:
            idx = idx[::-1]
        out = []
        for i in idx[:n_examples]:
            row = {
                "s1_id": int(val.s1_id[i]),
                "target_id": f"{TARGET_PREFIX[int(val.target_source[i])]}-{val.target_id[i]}",
                "score": float(score[i]),
            }
            for name in ("name_ratio", "translit_name_ratio", "address_ratio", "house_number_match", "n_channels"):
                if name in features:
                    v = float(val.X[i, features.index(name)])
                    row[name] = None if np.isnan(v) else round(v, 3)
            out.append(row)
        return out

    low_true = (val.y == 1) & (score < 0.10)
    high_false = (val.y == 0) & (score >= threshold)

    return {
        "threshold": threshold,
        "score_quantiles": {
            "positives": dict(zip(map(str, q), np.quantile(pos, q).round(4).tolist())),
            "negatives": dict(zip(map(str, q), np.quantile(neg, q).round(4).tolist())),
        },
        "positives": int(len(pos)),
        "positives_below_threshold": int((pos < threshold).sum()),
        "positives_below_0.10": int(low_true.sum()),
        "negatives": int(len(neg)),
        "negatives_at_or_above_threshold": int((neg >= threshold).sum()),
        "negatives_at_or_above_0.90": int((neg >= 0.90).sum()),
        "s1_total": int(len(n_true)),
        "s1_with_zero_predictions": int((n_pred == 0).sum()),
        "s1_zero_predictions_no_true_match": int(((n_pred == 0) & (n_true == 0)).sum()),
        "s1_zero_predictions_with_true_match": int(((n_pred == 0) & (n_true > 0)).sum()),
        "s1_with_multiple_predictions": int((n_pred > 1).sum()),
        "max_predictions_single_s1": int(n_pred.max()),
        "s1_without_true_match": int((n_true == 0).sum()),
        "s1_without_true_match_but_predicted": int(((n_true == 0) & (n_pred > 0)).sum()),
        "examples_true_low_score": examples(low_true, ascending=True),
        "examples_false_high_score": examples(high_false, ascending=False),
    }


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------


def code_version() -> dict:

    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, cwd=ROOT
            ).stdout.strip()
        except Exception:
            return None

    digest = hashlib.sha256()
    for folder in ("blocking", "preprocessing", "features", "evaluation", "models"):
        for path in sorted((ROOT / "src" / folder).glob("*.py")):
            digest.update(path.read_bytes())

    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": digest.hexdigest(),
    }


def run(feature_dir: Path, out_dir: Path, cfg: MatcherConfig) -> dict:

    out_dir.mkdir(parents=True, exist_ok=False)
    timings = {}

    feature_manifest = json.loads((feature_dir / "manifest.json").read_text())
    features, dropped = select_features(feature_manifest["feature_columns"])

    start = time.perf_counter()
    splits = load_feature_splits(feature_dir, features)
    train, val = splits[0], splits[1]
    timings["load_seconds"] = round(time.perf_counter() - start, 1)

    # Early-stopping hold-out: S1-level, carved from the TRAIN split only.
    train_s1 = np.unique(train.s1_id)
    _, es_ids = make_s1_validation_split(
        pd.Series(train_s1),
        validation_fraction=cfg.early_stopping_fraction,
        random_state=cfg.seed,
    )
    is_es = np.isin(train.s1_id, es_ids.to_numpy())
    fit_idx, es_idx = np.flatnonzero(~is_es), np.flatnonzero(is_es)

    n_pos = int(train.y[fit_idx].sum())
    n_neg = int(len(fit_idx) - n_pos)
    scale_pos_weight = n_neg / n_pos

    params = {**cfg.xgb_params, "random_state": cfg.seed, "scale_pos_weight": scale_pos_weight}
    model = xgb.XGBClassifier(**params)

    print(
        f"[train] fit rows {len(fit_idx):,} (pos {n_pos:,}), early-stop rows {len(es_idx):,}, "
        f"validation rows {len(val.y):,}; scale_pos_weight {scale_pos_weight:.2f}; "
        f"{len(features)} features",
        flush=True,
    )

    start = time.perf_counter()
    model.fit(
        train.X[fit_idx],
        train.y[fit_idx],
        eval_set=[(train.X[es_idx], train.y[es_idx])],
        verbose=100,
    )
    timings["train_seconds"] = round(time.perf_counter() - start, 1)

    start = time.perf_counter()
    score = model.predict_proba(val.X)[:, 1].astype(np.float32)
    timings["predict_seconds"] = round(time.perf_counter() - start, 1)

    # Evaluate over EVERY validation S1 in the split file, with truth
    # counts covering all S2+S3 links (including blocking misses).
    split_table = pq.read_table(feature_dir / "split.parquet").to_pandas()
    val_s1 = np.sort(split_table.loc[split_table["split"] == 1, "s1_id"].to_numpy())
    n_true = true_link_counts(val_s1)
    s1_index = np.searchsorted(val_s1, val.s1_id)
    if not (val_s1[s1_index] == val.s1_id).all():
        raise ValueError("validation candidates reference S1 outside the validation split")

    sweep = threshold_sweep(s1_index, val.y, score, n_true, cfg.thresholds)
    best = sweep.loc[sweep["macro_f05"].idxmax()]
    threshold = float(best["threshold"])
    if threshold == max(cfg.thresholds):
        print(f"WARNING: selected threshold {threshold} is the top of the grid", flush=True)

    oracle = s1_metrics(s1_index, val.y, val.y == 1, n_true)

    reference = check_with_reference_metric(val, score, threshold, val_s1)
    if not np.isclose(reference, best["macro_f05"], atol=1e-9):
        raise AssertionError(
            f"macro F0.5 mismatch: vectorized {best['macro_f05']} vs reference {reference}"
        )

    diag = diagnostics(val, score, threshold, s1_index, n_true, features)
    diag["oracle_macro_f05_on_candidates"] = oracle["macro_f05"]
    diag["feature_importance_gain"] = dict(
        sorted(
            model.get_booster().get_score(importance_type="gain").items(),
            key=lambda kv: -kv[1],
        )
    )
    # Booster names features f0..fN when trained on numpy arrays.
    diag["feature_importance_gain"] = {
        features[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k: round(v, 2)
        for k, v in diag["feature_importance_gain"].items()
    }

    # --- Artifacts ------------------------------------------------------
    model.save_model(out_dir / "model.json")

    (out_dir / "features.json").write_text(
        json.dumps({"features": features, "dropped": dropped}, indent=2)
    )
    (out_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "xgb_params_resolved": params}, indent=2)
    )
    sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)
    pq.write_table(
        pa.table(
            {
                "s1_id": val.s1_id,
                "target_id": val.target_id,
                "target_source": val.target_source,
                "label": val.y,
                "score": score,
                "predicted": (score >= threshold).astype(np.int8),
            }
        ),
        out_dir / "validation_predictions.parquet",
        compression="zstd",
    )
    (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))

    metadata = {
        "code": code_version(),
        "feature_dir": str(feature_dir),
        "feature_manifest_split": feature_manifest["split"],
        "candidate_code": feature_manifest.get("candidate_code"),
        "seed": cfg.seed,
        "rows": {
            "train_total": int(len(train.y)),
            "train_fit": int(len(fit_idx)),
            "train_early_stopping": int(len(es_idx)),
            "validation": int(len(val.y)),
        },
        "class_counts": {
            "train_fit": {"positive": n_pos, "negative": n_neg},
            "train_early_stopping": {
                "positive": int(train.y[es_idx].sum()),
                "negative": int(len(es_idx) - train.y[es_idx].sum()),
            },
            "validation": {
                "positive": int(val.y.sum()),
                "negative": int(len(val.y) - val.y.sum()),
            },
        },
        "s1_counts": {
            "train": int(len(train_s1)),
            "train_early_stopping": int(len(es_ids)),
            "validation": int(len(val_s1)),
            "validation_true_links": int(n_true.sum()),
        },
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": int(model.best_iteration),
        "selected_threshold": threshold,
        "validation_at_threshold": {k: float(v) for k, v in best.items()},
        "reference_macro_f05_check": reference,
        "oracle_macro_f05_on_candidates": oracle["macro_f05"],
        "timings": timings,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {"metadata": metadata, "sweep": sweep, "diagnostics": diag}


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("--name", default="xgb_baseline")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Experiment directory (default: outputs/experiments/<name>). Must not exist.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    cfg = MatcherConfig(seed=args.seed)
    cfg.xgb_params["n_jobs"] = args.n_jobs

    out_dir = args.out or ROOT / "outputs" / "experiments" / args.name
    result = run(args.feature_dir, out_dir, cfg)

    meta, sweep, diag = result["metadata"], result["sweep"], result["diagnostics"]

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("\n=== THRESHOLD SWEEP (validation S1, macro F0.5) ===")
        print(sweep.to_string(index=False))

    print(
        f"\nSelected threshold {meta['selected_threshold']:g}: "
        f"macro F0.5 {meta['validation_at_threshold']['macro_f05']:.4f} "
        f"(reference check {meta['reference_macro_f05_check']:.4f}; "
        f"oracle on candidates {meta['oracle_macro_f05_on_candidates']:.4f})"
    )
    print(f"Timings: {meta['timings']}; best iteration {meta['best_iteration']}")
    print(f"Artifacts: {out_dir}")
    print("\n=== DIAGNOSTICS ===")
    print(json.dumps({k: v for k, v in diag.items() if not k.startswith("examples")}, indent=2))
    for key in ("examples_true_low_score", "examples_false_high_score"):
        print(f"\n{key}:")
        for row in diag[key]:
            print("  ", row)


if __name__ == "__main__":
    main()
