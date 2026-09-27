"""
Out-of-fold base-matcher predictions for the TRAIN split.

A second decision layer must be fitted on base scores the base model has
not seen during training. This trains K copies of the base XGBoost
matcher (same features, hyperparameters and seed as a given experiment),
each on the train S1 entities outside one fold, with the same S1-level
early-stopping hold-out scheme, and predicts the held-out fold.

The frozen experiment's own model is not touched; validation rows are
never used.

    python -m src.decision.oof outputs/experiments/xgb_skeleton \\
        outputs/features/sample20k_skeleton --out outputs/experiments/s1_context
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from src.evaluation.split import make_s1_validation_split
from src.models.xgb_matcher import load_feature_splits


def s1_folds(s1_ids: np.ndarray, k: int, seed: int) -> dict[int, int]:
    """Deterministic S1 -> fold assignment."""

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.unique(s1_ids))

    return {int(s1): i % k for i, s1 in enumerate(shuffled)}


def oof_predictions(experiment: Path, feature_dir: Path, k: int = 5) -> tuple[pd.DataFrame, dict]:

    features = json.loads((experiment / "features.json").read_text())["features"]
    config = json.loads((experiment / "config.json").read_text())
    params = dict(config["xgb_params_resolved"])
    seed = config["seed"]
    es_fraction = config["early_stopping_fraction"]

    train = load_feature_splits(feature_dir, features)[0]

    fold_of = s1_folds(train.s1_id, k, seed)
    row_fold = np.fromiter((fold_of[int(s)] for s in train.s1_id), np.int8, len(train.s1_id))

    score = np.full(len(train.y), np.nan, dtype=np.float32)
    info = {"folds": []}

    for fold in range(k):
        start = time.perf_counter()
        held = row_fold == fold
        rest_s1 = np.unique(train.s1_id[~held])

        _, es_ids = make_s1_validation_split(
            pd.Series(rest_s1), validation_fraction=es_fraction, random_state=seed
        )
        is_es = np.isin(train.s1_id, es_ids.to_numpy()) & ~held
        fit = ~held & ~is_es

        pos = int(train.y[fit].sum())
        fold_params = {**params, "scale_pos_weight": (fit.sum() - pos) / pos}

        model = xgb.XGBClassifier(**fold_params)
        model.fit(train.X[fit], train.y[fit], eval_set=[(train.X[is_es], train.y[is_es])], verbose=False)
        score[held] = model.predict_proba(train.X[held])[:, 1]

        info["folds"].append(
            {
                "fold": fold,
                "fit_rows": int(fit.sum()),
                "early_stop_rows": int(is_es.sum()),
                "held_rows": int(held.sum()),
                "best_iteration": int(model.best_iteration),
                "scale_pos_weight": fold_params["scale_pos_weight"],
                "seconds": round(time.perf_counter() - start, 1),
            }
        )
        print(f"[oof] fold {fold}: {info['folds'][-1]}", flush=True)

    df = pd.DataFrame(
        {
            "s1_id": train.s1_id,
            "target_id": train.target_id,
            "target_source": train.target_source,
            "label": train.y,
            "fold": row_fold,
            "score": score,
        }
    )
    info.update({"k": k, "seed": seed, "experiment": str(experiment), "features": features})

    return df, info


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "oof_train_predictions.parquet"
    if target.exists():
        raise SystemExit(f"{target} exists; refusing to overwrite")

    df, info = oof_predictions(args.experiment, args.feature_dir, args.k)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), target, compression="zstd")
    (args.out / "oof_metadata.json").write_text(json.dumps(info, indent=2))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
