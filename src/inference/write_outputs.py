"""
Write the challenge output files from decision groups:

    <run>/submission/output/matching_results.tsv   source1_entity_id, matched_entity_ids
    <run>/submission/output/candidate_pairs.tsv    source1_entity_id, candidate_entity_ids

Every S1 of the dataset gets exactly one row (empty list when nothing is
matched / no candidate). The S1 set of each group is recovered from the
cached S1 partition and the group's recorded id range, and checked against
the recorded S1 count. Rows are ordered by (country, group, s1_id); ids in
a list by (source, id). Fragments are written per group (restartable) and
concatenated atomically at the end.

    python -m src.inference.write_outputs outputs/production/test_v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.blocking.candidate_pipeline import atomic_write_json
from src.blocking.entity_cache import load_partition

HEADERS = {
    "matching_results.tsv": "source1_entity_id\tmatched_entity_ids\n",
    "candidate_pairs.tsv": "source1_entity_id\tcandidate_entity_ids\n",
}


def id_lists(df: pd.DataFrame, s1_ids: np.ndarray) -> list[str]:
    """For each id in ``s1_ids``: comma-joined "S<src>-<id>" of its rows in ``df``."""

    if df.empty:
        return [""] * len(s1_ids)

    df = df.sort_values(["s1_id", "target_source", "target_id"], kind="stable")
    labels = "S" + df["target_source"].astype(str) + "-" + df["target_id"].astype(str)
    joined = labels.groupby(df["s1_id"].to_numpy(), sort=False).agg(",".join)

    return [joined.get(s, "") for s in s1_ids]


def group_s1_ids(s1_partition_ids: np.ndarray, group: dict, sample: np.ndarray | None) -> np.ndarray:

    ids = s1_partition_ids
    if sample is not None:
        ids = ids[np.isin(ids, sample)]
    ids = ids[(ids >= group["s1_id_min"]) & (ids <= group["s1_id_max"])]

    if len(ids) != group["s1_rows"]:
        raise ValueError(f"group {group['country']}/{group['part']}: {len(ids)} S1 found, manifest says {group['s1_rows']}")
    return ids


def run(run_dir: Path) -> dict:

    started = time.perf_counter()
    cand_cfg = json.loads((run_dir / "candidates" / "config.json").read_text())
    dataset = cand_cfg["dataset"]
    decisions = json.loads((run_dir / "decisions" / "manifest.json").read_text())

    sample_path = run_dir / "candidates" / "s1_ids.parquet"
    sample = pq.read_table(sample_path)["s1_id"].to_numpy() if sample_path.exists() else None

    frag_dir = run_dir / "submission" / "fragments"
    frag_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted(decisions["groups"], key=lambda g: (g["country"], g["part"]))

    s1_by_country: dict[str, np.ndarray] = {}
    totals = {"s1_rows": 0, "matched_s1": 0, "matched_ids": 0, "candidate_ids": 0}

    for g in groups:
        stem = f"{g['country']}-{g['part']:05d}"
        sidecar = frag_dir / f"{stem}.json"
        if sidecar.exists():
            rec = json.loads(sidecar.read_text())
        else:
            if g["country"] not in s1_by_country:
                s1_by_country = {g["country"]: load_partition("source1", g["country"], [], dataset)["id"].to_numpy()}
            s1_ids = group_s1_ids(s1_by_country[g["country"]], g, sample)

            df = pq.read_table(run_dir / "decisions" / g["file"]).to_pandas()
            matched = id_lists(df[df["predicted"] == 1], s1_ids)
            candidates = id_lists(df, s1_ids)

            for name, lists in (("matching_results.tsv", matched), ("candidate_pairs.tsv", candidates)):
                tmp = frag_dir / f"{stem}.{name}.tmp"
                with open(tmp, "w") as fh:
                    fh.writelines(f"S1-{s}\t{ids}\n" for s, ids in zip(s1_ids, lists))
                tmp.replace(frag_dir / f"{stem}.{name}")

            rec = {"s1_rows": len(s1_ids), "matched_s1": sum(bool(x) for x in matched),
                   "matched_ids": int((df["predicted"] == 1).sum()), "candidate_ids": len(df)}
            atomic_write_json(sidecar, rec)

        for k in totals:
            totals[k] += rec[k]

    out_dir = run_dir / "submission" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, header in HEADERS.items():
        tmp = out_dir / f"{name}.tmp"
        with open(tmp, "w") as out:
            out.write(header)
            for g in groups:
                with open(frag_dir / f"{g['country']}-{g['part']:05d}.{name}") as fh:
                    for line in fh:
                        out.write(line)
        tmp.replace(out_dir / name)

    manifest = {**totals, "dataset": dataset, "runtime_seconds": round(time.perf_counter() - started, 1),
                "files": {n: str(out_dir / n) for n in HEADERS}}
    atomic_write_json(run_dir / "submission" / "manifest.json", manifest)
    return manifest


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    m = run(args.run_dir)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
