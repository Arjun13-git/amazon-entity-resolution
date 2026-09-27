"""
Second-stage S1-context decision on base scores.

Exactness of S1 context at scale
--------------------------------
S1 context (top/second score, counts, ranks, gaps, S2/S3 counts) must see
ALL candidates of an S1, across both target sources. The candidate
pipeline chunks each country's S1 partition (sorted by id) identically for
every target, so candidate part k of S2 and part k of S3 cover exactly the
same S1 set and together contain every candidate of those S1.

This stage therefore processes one *decision group* = (country, part k)
at a time, after verifying from the candidate manifest that
- every target has part k for that country,
- all of them report the same S1 range and S1 count,
- the S1 ranges of different groups of a country do not overlap.
Any violation raises; context is never computed on a partial group.

Inputs per group: base score parts (``predict_base``) and the matching
feature parts (for the few evidence columns the second stage uses).
Labels are never read.

    python -m src.inference.decide outputs/production/test_v1 \\
        --second-stage outputs/experiments/s1_context
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

from src.blocking.candidate_pipeline import atomic_write_json, atomic_write_parquet
from src.decision.s1_context import ROW_FEATURES, context_features, design
from src.inference.predict_base import sha256_file


KEY = ["s1_id", "target_id", "target_source"]


def decision_groups(candidate_manifest: dict) -> list[dict]:
    """
    (country, part) groups with the per-target part files, validated so
    that each group holds complete S1s (see module docstring).
    """

    targets = candidate_manifest["config"]["targets"]
    by_key: dict[tuple[str, int], dict[str, dict]] = {}

    for p in candidate_manifest["parts"]:
        part = p.get("part")
        if part is None:
            part = int(Path(p["file"]).stem.split("-")[1])
        by_key.setdefault((p["country"], part), {})[p["target"]] = p

    groups = []
    for (country, part), per_target in sorted(by_key.items()):
        if set(per_target) != set(targets):
            raise ValueError(f"group {country}/{part} missing targets: {set(targets) - set(per_target)}")

        spans = {(m["s1_id_min"], m["s1_id_max"], m["s1_rows"]) for m in per_target.values()}
        if len(spans) != 1:
            raise ValueError(f"group {country}/{part}: targets cover different S1 sets: {spans}")

        s1_min, s1_max, s1_rows = spans.pop()
        groups.append({
            "country": country, "part": part, "s1_id_min": s1_min, "s1_id_max": s1_max,
            "s1_rows": s1_rows, "files": {t: per_target[t]["file"] for t in targets},
        })

    for country in {g["country"] for g in groups}:
        spans = sorted((g["s1_id_min"], g["s1_id_max"]) for g in groups if g["country"] == country)
        for (_, prev_max), (next_min, _) in zip(spans, spans[1:]):
            if next_min <= prev_max:
                raise ValueError(f"{country}: S1 ranges of different groups overlap")

    return groups


class SecondStage:
    """Frozen second-stage model + its decision parameters."""

    def __init__(self, directory: Path):
        meta = json.loads((directory / "metadata.json").read_text())
        kind = meta["second_stage_chosen"]
        if kind != "tree":
            raise ValueError(f"production path expects the tree second stage, got {kind}")

        self.features = meta["second_stage_features"]
        self.floor = float(meta["second_stage_floor"])
        self.threshold = float(meta["second_stage"][kind]["threshold"])
        self.model = xgb.XGBClassifier()
        self.model.load_model(directory / "second_stage_model.json")
        self.model_sha256 = sha256_file(directory / "second_stage_model.json")

    def decide(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(probability, predicted) for complete S1 groups in ``df``."""

        ctx = context_features(df)
        X = design(df, ctx)[self.features]
        prob = np.zeros(len(df), dtype=np.float64)
        eligible = (df["score"] >= self.floor).to_numpy()
        if eligible.any():
            prob[eligible] = self.model.predict_proba(X[eligible].to_numpy(np.float32))[:, 1]
        return prob, prob >= self.threshold


def load_group(group: dict, score_dir: Path, feature_dir: Path) -> pd.DataFrame:

    frames = []
    for file in group["files"].values():
        scores = pq.read_table(score_dir / file).to_pandas()
        feats = pq.read_table(feature_dir / file, columns=KEY + ROW_FEATURES).to_pandas()
        if len(scores) != len(feats) or not (scores[KEY].to_numpy() == feats[KEY].to_numpy()).all():
            raise ValueError(f"{file}: score and feature rows are not aligned")
        frames.append(pd.concat([scores, feats[ROW_FEATURES]], axis=1))

    df = pd.concat(frames, ignore_index=True)
    s1 = df["s1_id"].to_numpy()
    if len(df) and (s1.min() < group["s1_id_min"] or s1.max() > group["s1_id_max"]):
        raise ValueError(f"group {group['country']}/{group['part']}: S1 outside the declared range")
    return df


def run(run_dir: Path, second_stage_dir: Path) -> dict:

    started = time.perf_counter()
    cand_manifest = json.loads((run_dir / "candidates" / "manifest.json").read_text())
    groups = decision_groups(cand_manifest)
    stage = SecondStage(second_stage_dir)

    out_dir = run_dir / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for g in groups:
        path = out_dir / f"country={g['country']}" / f"group-{g['part']:05d}.parquet"
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            records.append(json.loads(sidecar.read_text()))
            continue

        t0 = time.perf_counter()
        df = load_group(g, run_dir / "scores", run_dir / "features")
        prob, predicted = stage.decide(df)

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(pa.table({
            **{k: df[k].to_numpy() for k in KEY},
            "score": df["score"].to_numpy(np.float32),
            "second_stage_prob": prob.astype(np.float32),
            "predicted": predicted.astype(np.int8),
        }), path)
        record = {**{k: g[k] for k in ("country", "part", "s1_id_min", "s1_id_max", "s1_rows")},
                  "file": str(path.relative_to(out_dir)), "candidates": len(df),
                  "predicted": int(predicted.sum()), "seconds": round(time.perf_counter() - t0, 2)}
        atomic_write_json(sidecar, record)
        records.append(record)
        print(f"  {record['file']}: {len(df):,} candidates -> {record['predicted']:,} predicted", flush=True)

    manifest = {
        "second_stage_dir": str(second_stage_dir), "second_stage_model_sha256": stage.model_sha256,
        "second_stage_features": stage.features, "floor": stage.floor, "threshold": stage.threshold,
        "groups": records, "candidates": sum(r["candidates"] for r in records),
        "predicted": sum(r["predicted"] for r in records),
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Production run directory (candidates/, features/, scores/).")
    parser.add_argument("--second-stage", type=Path, required=True)
    args = parser.parse_args()

    m = run(args.run_dir, args.second_stage)
    print(f"{m['predicted']:,} matches from {m['candidates']:,} candidates in {m['runtime_seconds']:,.0f}s")


if __name__ == "__main__":
    main()
