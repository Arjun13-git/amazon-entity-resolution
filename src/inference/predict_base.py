"""
Base matcher inference: score every feature part with a frozen
``xgb_matcher`` experiment (model.json + features.json).

One score part per feature part (same relative path), written atomically
with a JSON sidecar; finished parts are skipped on resume.

    python -m src.inference.predict_base outputs/production/test_v1/features \\
        --experiment outputs/experiments/xgb_skeleton --out outputs/production/test_v1/scores
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from src.blocking.candidate_pipeline import atomic_write_json, atomic_write_parquet


KEY = ["s1_id", "target_id", "target_source"]


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base_model(experiment: Path) -> tuple[xgb.XGBClassifier, list[str]]:

    features = json.loads((experiment / "features.json").read_text())["features"]
    model = xgb.XGBClassifier()
    model.load_model(experiment / "model.json")
    return model, features


def feature_matrix(table: pa.Table, features: list[str]) -> np.ndarray:
    """float32 matrix in the model's feature order (as in training)."""

    X = np.empty((table.num_rows, len(features)), dtype=np.float32)
    for j, name in enumerate(features):
        X[:, j] = table[name].to_numpy(zero_copy_only=False)
    return X


def run(feature_dir: Path, experiment: Path, out_dir: Path) -> dict:

    started = time.perf_counter()
    manifest_in = json.loads((feature_dir / "manifest.json").read_text())
    model, features = load_base_model(experiment)

    missing = [f for f in features if f not in manifest_in["feature_columns"]]
    if missing:
        raise ValueError(f"feature dir lacks model features: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for meta in manifest_in["parts"]:
        path = out_dir / meta["file"]
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            records.append(json.loads(sidecar.read_text()))
            continue

        t0 = time.perf_counter()
        table = pq.read_table(feature_dir / meta["file"], columns=KEY + features)
        score = model.predict_proba(feature_matrix(table, features))[:, 1].astype(np.float32)

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(
            pa.table({**{k: table[k] for k in KEY}, "score": score}), path
        )
        record = {
            "file": meta["file"], "target": meta["target"], "country": meta["country"],
            "part": meta.get("part"), "rows": table.num_rows,
            "seconds": round(time.perf_counter() - t0, 2),
        }
        atomic_write_json(sidecar, record)
        records.append(record)
        print(f"  {meta['file']}: {table.num_rows:,} rows scored", flush=True)

    manifest = {
        "feature_dir": str(feature_dir),
        "experiment": str(experiment),
        "model_sha256": sha256_file(experiment / "model.json"),
        "features": features,
        "parts": records,
        "rows": sum(r["rows"] for r in records),
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    m = run(args.feature_dir, args.experiment, args.out)
    print(f"Scored {m['rows']:,} rows in {m['runtime_seconds']:,.0f}s")


if __name__ == "__main__":
    main()
