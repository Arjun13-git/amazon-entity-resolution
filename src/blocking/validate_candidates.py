"""
Validate a streaming candidate-generation output directory.

Every check streams one part file at a time; only per-partition lookup
arrays (ids, name/key hashes) are held in memory.

Checks:
  1. no duplicate (s1_id, target_id) pairs (within and across parts)
  2. every candidate's S1 and target belong to the part's country
  3. channel provenance:
       - mask bits agree with rank/score/hit columns
       - exact_name pairs have equal normalized names, and every
         same-name target is present
       - house_city pairs share a non-empty key with block <= cap, and
         every such block is present in full
       - char ranks are 0..n-1 per S1, within K, scores non-increasing
  4. candidate counts per S1
  5. parts iterate/merge lazily via pyarrow.dataset
  6. (--compare) identical content to another run
  plus (--recall) true-link recall against train ground truth.

    python -m src.blocking.validate_candidates outputs/candidates/sample20k --recall
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.blocking.address_keys import extract_keys
from src.blocking.benchmark_candidates import TARGET_FILES, locate
from src.blocking.candidate_pipeline import SOURCE_BITS, TARGET_CODES, open_candidates
from src.blocking.channels import hash_strings
from src.blocking.entity_cache import load_partition, load_truth


class PartitionLookup:
    """Sorted ids with name hashes and house|city key hashes."""

    def __init__(self, source: str, country: str, lexicon: frozenset[str], dataset: str = "train"):
        df = load_partition(source, country, ["norm_name", "norm_address"], dataset)

        self.ids = df["id"].to_numpy()
        names = df["norm_name"].fillna("")
        self.name_hash = hash_strings(names)
        self.name_empty = (names == "").to_numpy()

        keys = extract_keys(df["norm_address"], country, lexicon)["house_city"]
        self.key_hash = hash_strings(keys)
        self.key_empty = (keys == "").to_numpy()

    def rows(self, ids: np.ndarray) -> np.ndarray:
        pos = locate(self.ids, ids)
        if (pos < 0).any():
            raise AssertionError(f"{(pos < 0).sum()} ids not in this country partition")
        return pos


def block_sizes(sorted_hash: np.ndarray, query_hash: np.ndarray) -> np.ndarray:

    return (
        np.searchsorted(sorted_hash, query_hash, "right")
        - np.searchsorted(sorted_hash, query_hash, "left")
    )


def check_part(table, meta, cfg, s1: PartitionLookup, tg: PartitionLookup, tg_sorted):
    """Checks 1-4 for one part. Returns per-S1 counts and error list."""

    errors = []
    n = table.num_rows

    s1_id = table["s1_id"].to_numpy()
    t_id = table["target_id"].to_numpy()
    mask = table["candidate_source_mask"].to_numpy()

    # 1. Sorted strictly increasing composite key => no duplicates.
    composite = (s1_id.astype(np.int64) << 32) | t_id.astype(np.int64)
    if n and not (np.diff(composite) > 0).all():
        errors.append("duplicate or unsorted (s1_id, target_id) pairs")

    # 2. Country / target partition.
    if set(table["country"].unique().to_pylist()) - {meta["country"]}:
        errors.append("country column differs from partition")
    if set(table["target_source"].unique().to_pylist()) - {TARGET_CODES[meta["target"]]}:
        errors.append("target_source differs from partition")

    try:
        s1_rows = s1.rows(s1_id)
        t_rows = tg.rows(t_id)
    except AssertionError as exc:
        errors.append(f"country partition: {exc}")
        return None, errors

    # 3. Provenance: mask vs columns.
    bit = {name: (mask & b) > 0 for name, b in SOURCE_BITS.items()}

    if (mask == 0).any():
        errors.append("rows with empty source mask")
    if (mask & ~np.uint8(sum(SOURCE_BITS.values()))).any():
        errors.append("unknown mask bits")

    for name, col in (
        ("exact_name", "exact_name_rank"),
        ("name_char", "name_char_rank"),
        ("address_char", "address_char_rank"),
    ):
        present = table[col].to_numpy() >= 0
        if (present != bit[name]).any():
            errors.append(f"{col} disagrees with mask")

    for name, col in (("name_char", "name_char_score"), ("address_char", "address_char_score")):
        present = ~np.isnan(table[col].to_numpy())
        if (present != bit[name]).any():
            errors.append(f"{col} disagrees with mask")

    if (table["house_city_hit"].to_numpy() != bit["house_city"]).any():
        errors.append("house_city_hit disagrees with mask")

    # Exact name: equal non-empty names, and complete blocks.
    ex = bit["exact_name"]
    if (s1.name_hash[s1_rows[ex]] != tg.name_hash[t_rows[ex]]).any() or s1.name_empty[s1_rows[ex]].any():
        errors.append("exact_name pair with different/empty names")

    uniq_s1, first = np.unique(s1_id, return_index=True)
    u_rows = s1_rows[first]

    exact_counts = np.bincount(np.searchsorted(uniq_s1, s1_id[ex]), minlength=len(uniq_s1))
    expected = np.where(
        s1.name_empty[u_rows], 0, block_sizes(tg_sorted["name"], s1.name_hash[u_rows])
    )
    if (exact_counts != expected).any():
        errors.append(
            f"exact_name incomplete for {(exact_counts != expected).sum()} S1"
        )

    # House|city: equal non-empty keys, block <= cap, complete blocks.
    hc = bit["house_city"]
    if (s1.key_hash[s1_rows[hc]] != tg.key_hash[t_rows[hc]]).any() or s1.key_empty[s1_rows[hc]].any():
        errors.append("house_city pair with different/empty keys")

    hc_counts = np.bincount(np.searchsorted(uniq_s1, s1_id[hc]), minlength=len(uniq_s1))
    sizes = np.where(
        s1.key_empty[u_rows], 0, block_sizes(tg_sorted["key"], s1.key_hash[u_rows])
    )
    expected = np.where(sizes <= cfg["house_city_max_block"], sizes, 0)
    if (hc_counts != expected).any():
        errors.append(f"house_city incomplete for {(hc_counts != expected).sum()} S1")

    # Char channels: ranks 0..n-1 within K, scores non-increasing by rank.
    for name, k in (("name_char", cfg["name_k"]), ("address_char", cfg["address_k"])):
        sel = bit[name]
        ranks = table[f"{name}_rank"].to_numpy()[sel].astype(np.int64)
        scores = table[f"{name}_score"].to_numpy()[sel]
        owners = s1_id[sel]

        if len(ranks) and (ranks.max() >= k or ranks.min() < 0):
            errors.append(f"{name} rank outside [0, {k})")

        order = np.lexsort((ranks, owners))
        r, o, s = ranks[order], owners[order], scores[order]
        starts = np.r_[True, o[1:] != o[:-1]]
        group_start = np.maximum.accumulate(np.where(starts, np.arange(len(o)), 0))
        if len(r) and (r != np.arange(len(r)) - group_start).any():
            errors.append(f"{name} ranks not contiguous per S1")
        if len(s) > 1 and (np.diff(s)[~starts[1:]] > 1e-6).any():
            errors.append(f"{name} scores increase with rank")
        if len(s) and ((s < -1e-6) | (s > 1 + 1e-5)).any():
            errors.append(f"{name} score outside [0, 1]")

    # 4. Counts per S1 (including S1 rows with no candidates).
    counts = np.bincount(np.searchsorted(uniq_s1, s1_id), minlength=len(uniq_s1))

    return (uniq_s1, counts), errors


def tables_equal(a, b) -> bool:
    """Column-wise equality where NaN == NaN (Table.equals treats NaN as unequal)."""

    if a.schema != b.schema or a.num_rows != b.num_rows:
        return False

    for name in a.column_names:
        x = a[name].to_numpy(zero_copy_only=False)
        y = b[name].to_numpy(zero_copy_only=False)
        if x.dtype.kind == "f":
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False

    return True


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--recall", action="store_true")
    parser.add_argument("--compare", type=Path, default=None)
    args = parser.parse_args()

    manifest = json.loads((args.out_dir / "manifest.json").read_text())
    cfg = manifest["config"]

    lexicons = {
        country: frozenset((args.out_dir / info["file"]).read_text().split())
        for country, info in manifest["lexicons"].items()
    }

    s1_path = args.out_dir / "s1_ids.parquet"
    sample_ids = pq.read_table(s1_path)["s1_id"].to_numpy() if s1_path.exists() else None

    all_errors = []
    by_target = defaultdict(lambda: {"s1": [], "counts": [], "rows": 0})

    parts = sorted(manifest["parts"], key=lambda p: (p["target"], p["country"], p["file"]))
    current = None

    for meta in parts:
        key = (meta["target"], meta["country"])
        if key != current:
            current = key
            target, country = key
            dataset = cfg.get("dataset", "train")
            s1 = PartitionLookup("source1", country, lexicons[country], dataset)
            tg = PartitionLookup(TARGET_FILES[target], country, lexicons[country], dataset)
            tg_sorted = {
                "name": np.sort(tg.name_hash[~tg.name_empty]),
                "key": np.sort(tg.key_hash[~tg.key_empty]),
            }
            print(f"[{target} | {country}] lookups built", flush=True)

        table = pq.read_table(args.out_dir / meta["file"])

        if table.num_rows != meta["candidates"]:
            all_errors.append(f"{meta['file']}: row count differs from manifest")

        result, errors = check_part(table, meta, cfg, s1, tg, tg_sorted)
        all_errors += [f"{meta['file']}: {e}" for e in errors]

        if result is not None:
            uniq, counts = result
            agg = by_target[meta["target"]]
            agg["s1"].append(uniq)
            agg["counts"].append(counts)
            agg["rows"] += table.num_rows

        del table

    print("\n=== 1-4. PART CHECKS ===")

    for target, agg in by_target.items():
        s1_all = np.concatenate(agg["s1"])
        if len(np.unique(s1_all)) != len(s1_all):
            all_errors.append(f"{target}: an S1 id appears in more than one part")

        counts = np.concatenate(agg["counts"])

        expected_s1 = len(sample_ids) if sample_ids is not None else None
        zero = (expected_s1 - len(s1_all)) if expected_s1 is not None else 0
        full = np.r_[counts, np.zeros(zero, int)]

        print(
            f"  {target}: {agg['rows']:,} candidates, S1 with candidates "
            f"{len(s1_all):,}" + (f" / {expected_s1:,}" if expected_s1 else "")
            + f"; per S1 mean {full.mean():.1f}, median {np.median(full):.0f}, "
            f"p99 {np.percentile(full, 99):.0f}, max {full.max():,}"
        )

    # 5. Lazy iteration over the whole directory.
    dataset = open_candidates(args.out_dir)
    streamed = 0
    batches = 0
    for batch in dataset.to_batches(columns=["s1_id", "target_id"], batch_size=1_000_000):
        streamed += batch.num_rows
        batches += 1
    print(
        f"\n=== 5. STREAMING ITERATION === {streamed:,} rows in {batches:,} batches "
        f"(manifest total {manifest['total_candidates']:,})"
    )
    if streamed != manifest["total_candidates"]:
        all_errors.append("streamed row count differs from manifest total")

    # 6. Determinism.
    if args.compare:
        other = json.loads((args.compare / "manifest.json").read_text())
        same_cfg = other["config"] == cfg
        mismatched = [
            meta["file"]
            for meta in parts
            if not tables_equal(
                pq.read_table(args.out_dir / meta["file"]),
                pq.read_table(args.compare / meta["file"]),
            )
        ]
        same_files = {p["file"] for p in other["parts"]} == {p["file"] for p in parts}
        print(
            f"\n=== 6. DETERMINISM vs {args.compare} === config equal: {same_cfg}; "
            f"same part files: {same_files}; content mismatches: {len(mismatched)}"
        )
        if not (same_cfg and same_files) or mismatched:
            all_errors.append(f"not deterministic: {mismatched[:5]}")

    # Recall against ground truth.
    if args.recall and cfg.get("dataset", "train") != "train":
        print("\n(recall skipped: no ground truth for this dataset)")
    elif args.recall:
        print("\n=== RECALL (train ground truth) ===")
        for target in cfg["targets"]:
            truth = load_truth(target)
            if sample_ids is not None:
                truth = truth[np.isin(truth["s1_id"].to_numpy(), sample_ids)]
            truth_keys = np.sort(
                (truth["s1_id"].to_numpy().astype(np.int64) << 32)
                | truth["target_id"].to_numpy().astype(np.int64)
            )

            found = np.zeros(len(truth_keys), bool)
            per_channel = defaultdict(lambda: np.zeros(len(truth_keys), bool))

            for meta in parts:
                if meta["target"] != target:
                    continue
                table = pq.read_table(
                    args.out_dir / meta["file"],
                    columns=["s1_id", "target_id", "candidate_source_mask"],
                )
                keys = (table["s1_id"].to_numpy().astype(np.int64) << 32) | table[
                    "target_id"
                ].to_numpy().astype(np.int64)
                mask = table["candidate_source_mask"].to_numpy()
                found |= np.isin(truth_keys, keys)
                for name, b in SOURCE_BITS.items():
                    per_channel[name] |= np.isin(truth_keys, keys[(mask & b) > 0])

            print(
                f"  {target}: {found.sum():,} / {len(truth_keys):,} = {found.mean():.4%}  ("
                + ", ".join(f"{n} {v.mean():.2%}" for n, v in per_channel.items())
                + ")"
            )

    print("\n=== RESULT ===")
    if all_errors:
        print(f"FAILED: {len(all_errors)} problem(s)")
        for e in all_errors[:50]:
            print("  -", e)
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
