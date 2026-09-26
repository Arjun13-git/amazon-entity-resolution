"""
Audit a pairwise feature directory before any model training.

Streams feature parts; exact statistics (counts, means, missing rates)
are accumulated part by part. Quantiles and correlations use all
positives plus a fixed-seed sample of negatives, so memory stays bounded.
Labels are re-verified against link strings derived directly from the
raw ground-truth TSV.

    python -m src.features.audit_features outputs/features/sample20k
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.blocking.entity_cache import cache_path
from src.evaluation.labels import raw_truth_strings


ID_COLUMNS = ["s1_id", "target_id", "target_source", "country", "split", "label"]
SOURCE_FILES = {1: "source1", 2: "source2", 3: "source3"}


def fetch_text(source: str, ids: np.ndarray) -> pd.DataFrame:
    """norm_name / norm_address for a few ids (pushed-down filter)."""

    table = ds.dataset(cache_path(source)).to_table(
        columns=["id", "norm_name", "norm_address"],
        filter=ds.field("id").isin(ids.tolist()),
    )

    return table.to_pandas().set_index("id")


def print_examples(title: str, rows: pd.DataFrame, columns: list[str]) -> None:

    print(f"\n--- {title} ({len(rows)} shown) ---")

    if rows.empty:
        return

    s1 = fetch_text("source1", rows["s1_id"].unique())
    targets = {
        code: fetch_text(SOURCE_FILES[code], rows.loc[rows["target_source"] == code, "target_id"].unique())
        for code in rows["target_source"].unique()
    }

    for _, r in rows.iterrows():
        t = targets[r["target_source"]].loc[r["target_id"]]
        a = s1.loc[r["s1_id"]]
        stats = " ".join(
            f"{c}={r[c]:.2f}" if isinstance(r[c], float) else f"{c}={r[c]}" for c in columns
        )
        print(
            f"  [S{r['target_source']} {r['country']} label={r['label']}] {stats}\n"
            f"     S1 : {a['norm_name']!r} | {a['norm_address']!r}\n"
            f"     S{r['target_source']} : {t['norm_name']!r} | {t['norm_address']!r}"
        )


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("--negative-sample", type=float, default=0.10)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads((args.feature_dir / "manifest.json").read_text())
    cand_dir = Path(manifest["candidate_dir"])
    features = manifest["feature_columns"]

    s1_path = cand_dir / "s1_ids.parquet"
    s1_ids = pq.read_table(s1_path)["s1_id"].to_numpy() if s1_path.exists() else None
    truth_strings = raw_truth_strings(s1_ids)

    rng = np.random.default_rng(args.seed)

    totals = defaultdict(int)
    sums = {c: np.zeros(2) for c in features}
    present = {c: np.zeros(2) for c in features}
    per_s1 = []            # (s1_id, target_source, positives, negatives, split)
    samples = []
    pools = defaultdict(list)
    label_mismatch = 0

    for meta in manifest["parts"]:
        table = pq.read_table(args.feature_dir / meta["file"])
        df = table.to_pandas()

        label = df["label"].to_numpy()
        totals["rows"] += len(df)
        totals["pos"] += int(label.sum())

        # 12. Independent label check against raw TSV link strings.
        src = df["target_source"].to_numpy()
        keys = [
            f"S1-{a}|S{s}-{b}"
            for a, s, b in zip(df["s1_id"].to_numpy(), src, df["target_id"].to_numpy())
        ]
        raw_label = np.fromiter((k in truth_strings for k in keys), bool, len(keys))
        label_mismatch += int((raw_label != (label == 1)).sum())
        totals["raw_pos"] += int(raw_label.sum())

        # Exact per-class sums / non-missing counts.
        for c in features:
            v = df[c].to_numpy(dtype=np.float64)
            ok = ~np.isnan(v)
            for cls in (0, 1):
                m = ok & (label == cls)
                sums[c][cls] += v[m].sum()
                present[c][cls] += m.sum()

        g = df.groupby("s1_id", sort=False).agg(
            positives=("label", "sum"),
            candidates=("label", "size"),
            split=("split", "first"),
            n_split=("split", "nunique"),
        )
        g["target_source"] = meta["target"]
        per_s1.append(g.reset_index())

        keep = (label == 1) | (rng.random(len(df)) < args.negative_sample)
        samples.append(df[keep])

        # Candidate pools for the example inspection: exact counts, plus
        # a small per-part random sample of rows to show.
        pool_masks = {
            "tp": label == 1,
            "hard_name": (label == 0) & (df["name_ratio"] >= 0.9).to_numpy(),
            "hard_address": (label == 0) & (df["address_ratio"] >= 0.9).to_numpy(),
            "house_city_only": ((df["house_city_hit"] == 1) & (df["n_channels"] == 1)).to_numpy(),
        }
        for key, m in pool_masks.items():
            totals[f"{key}_n"] += int(m.sum())
            totals[f"{key}_pos"] += int(label[m].sum())
            pool = df[m]
            pools[key].append(pool.sample(min(len(pool), 50), random_state=args.seed))

        del table, df

    pos, rows = totals["pos"], totals["rows"]
    neg = rows - pos

    s1 = pd.concat(per_s1, ignore_index=True)
    s1["negatives"] = s1["candidates"] - s1["positives"]

    print("=== 1-6. CANDIDATE / LABEL COUNTS ===")
    print(f"  candidate pairs : {rows:,}")
    print(f"  positive pairs  : {pos:,}")
    print(f"  negative pairs  : {neg:,}")
    print(f"  positive rate   : {pos / rows:.4%}")
    for target, g in s1.groupby("target_source"):
        p, n = g["positives"], g["negatives"]
        print(
            f"  {target}: S1 {len(g):,}; positives/S1 mean {p.mean():.2f} median {p.median():.0f} "
            f"max {p.max()}; S1 with 0 positives {(p == 0).mean():.1%}; "
            f"negatives/S1 mean {n.mean():.1f} median {n.median():.0f} max {n.max():,}"
        )

    print("\n=== 10-11. SPLIT ===")
    if (s1["n_split"] > 1).any():
        print("  ERROR: an S1 has candidates in both splits")
    both = s1.groupby("s1_id")["split"].nunique()
    print(f"  S1 in more than one split across targets: {(both > 1).sum()}")
    uniq = s1.drop_duplicates("s1_id")
    for flag, name in ((0, "train"), (1, "validation")):
        part = s1[s1["split"] == flag]
        print(
            f"  {name:<10}: unique S1 {uniq[uniq['split'] == flag].shape[0]:,}; "
            f"rows {part['candidates'].sum():,}; positives {part['positives'].sum():,} "
            f"(S2 {part.loc[part['target_source'] == 'S2', 'positives'].sum():,}, "
            f"S3 {part.loc[part['target_source'] == 'S3', 'positives'].sum():,}); "
            f"positive rate {part['positives'].sum() / part['candidates'].sum():.3%}"
        )

    print("\n=== 12. LABEL VERIFICATION (raw ground-truth TSV strings) ===")
    print(f"  rows whose label disagrees with raw truth: {label_mismatch:,}")
    print(f"  positives by raw truth: {totals['raw_pos']:,} vs labels: {pos:,}")
    retrievable = sum(1 for k in truth_strings)
    print(f"  raw truth links for these S1: {retrievable:,} -> candidate recall {pos / retrievable:.4%}")

    sample = pd.concat(samples, ignore_index=True)
    sp, sn = sample[sample["label"] == 1], sample[sample["label"] == 0]

    print(
        f"\n=== 7-8. FEATURES: exact means & missing rates; quantiles on "
        f"{len(sp):,} positives + {len(sn):,} sampled negatives ==="
    )
    rows_out = []
    for c in features:
        rows_out.append(
            {
                "feature": c,
                "mean pos": sums[c][1] / max(present[c][1], 1),
                "mean neg": sums[c][0] / max(present[c][0], 1),
                "p10/p50/p90 pos": "/".join(f"{q:.2f}" for q in sp[c].quantile([0.1, 0.5, 0.9])),
                "p10/p50/p90 neg": "/".join(f"{q:.2f}" for q in sn[c].quantile([0.1, 0.5, 0.9])),
                "missing pos": 1 - present[c][1] / pos,
                "missing neg": 1 - present[c][0] / neg,
            }
        )
    with pd.option_context("display.width", 250, "display.max_rows", None, "display.float_format", "{:,.3f}".format):
        print(pd.DataFrame(rows_out).to_string(index=False))

    print("\n=== 9. CORRELATIONS (sample; pairwise-complete Pearson) ===")
    corr = sample[features + ["label"]].astype(np.float32).corr()
    with_label = corr["label"].drop("label").sort_values(key=np.abs, ascending=False)
    print("  with label (negatives subsampled, so magnitudes are inflated vs population):")
    print("   " + "\n   ".join(f"{k:<26} {v:+.3f}" for k, v in with_label.items()))
    upper = corr.drop(index="label", columns="label").where(np.triu(np.ones((len(features),) * 2, bool), 1))
    high = upper.stack().loc[lambda s: s.abs() >= 0.9].sort_values(key=np.abs, ascending=False)
    print("  feature pairs with |r| >= 0.9:")
    print("   " + "\n   ".join(f"{a} ~ {b}: {v:+.3f}" for (a, b), v in high.items()))
    constant = [c for c in features if sample[c].nunique(dropna=True) <= 1]
    print(f"  constant features: {constant}")

    print("\n=== HARD-NEGATIVE / CHANNEL POOLS (exact counts over all rows) ===")
    cols = ["name_ratio", "translit_name_ratio", "address_ratio", "house_city_match", "n_channels", "name_char_rank", "address_char_rank"]
    for key, title in (
        ("tp", "True positives"),
        ("hard_name", "Hard negatives: name_ratio >= 0.9"),
        ("hard_address", "Hard negatives: address_ratio >= 0.9"),
        ("house_city_only", "Retrieved only by house_city"),
    ):
        n, p = totals[f"{key}_n"], totals[f"{key}_pos"]
        print(f"\n  {title}: {n:,} rows ({n / rows:.2%} of candidates), positive rate {p / max(n, 1):.3%}")
        pool = pd.concat(pools[key], ignore_index=True)
        print_examples(title, pool.sample(min(args.examples, len(pool)), random_state=args.seed), cols)

if __name__ == "__main__":
    main()
