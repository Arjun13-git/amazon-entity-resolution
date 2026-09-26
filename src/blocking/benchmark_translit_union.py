"""
Transliterated-name channel benchmark and its incremental recall over
exact ∪ name-char-rerank ∪ address-char-rerank.

Same protocol as the other blocking diagnostics: a random S1 sample
(default 20,000, seed 42) against the COMPLETE S2/S3 country partitions.

    python -m src.blocking.benchmark_translit_union
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.blocking.benchmark_address_union import timed_keys
from src.blocking.benchmark_candidates import TARGET_FILES, locate, sample_s1_ids
from src.blocking.channels import (
    AddressChannel,
    CharNgramChannel,
    ExactNameChannel,
    TransliteratedNameChannel,
)
from src.blocking.diagnose_union import PERCENTILES, pair_similarities
from src.blocking.entity_cache import list_countries, load_partition, load_truth
from src.blocking.inspect_zero_name_matches import dominant_script


BASE = "exact ∪ name-rerank ∪ address-rerank"


def run_target(target, args, s1_ids):

    truth = load_truth(target)
    truth = truth[np.isin(truth["s1_id"].to_numpy(), s1_ids)]

    k_values = sorted(args.translit_k)
    k_max = max(k_values)

    agg: dict[str, dict] = {}
    link_frames = []

    def add(label, keys, n_q, stride, truth_keys, runtime):
        a = agg.setdefault(
            label, {"pairs": 0, "counts": [], "hits": 0, "runtime_s": 0.0}
        )
        a["pairs"] += len(keys)
        a["counts"].append(np.bincount(keys // stride, minlength=n_q))
        a["hits"] += int(np.isin(truth_keys, keys).sum())
        a["runtime_s"] += runtime

    for country in list_countries("source1"):
        cols = ["norm_name", "norm_address"]

        queries = load_partition("source1", country, cols)
        queries = queries[
            np.isin(queries["id"].to_numpy(), s1_ids)
        ].reset_index(drop=True)
        targets = load_partition(TARGET_FILES[target], country, cols)

        n_q = len(queries)
        stride = np.int64(len(targets) + 1)

        q_pos = locate(queries["id"].to_numpy(), truth["s1_id"].to_numpy())
        t_pos = locate(targets["id"].to_numpy(), truth["target_id"].to_numpy())
        keep = q_pos >= 0
        truth_q, truth_t = q_pos[keep], t_pos[keep]
        truth_keys = truth_q.astype(np.int64) * stride + truth_t

        print(
            f"\n[{target} | {country}] queries={n_q:,} "
            f"targets={len(targets):,} true links={len(truth_q):,}"
        )

        common = dict(
            max_df=args.max_df, workers=args.workers, chunk_work=args.chunk_work
        )

        exact_keys, _, exact_time = timed_keys(
            ExactNameChannel(), targets, queries, stride
        )

        name_keys, name_rank, name_time = timed_keys(
            CharNgramChannel(k=100, rerank_pool=args.rerank_pool, **common),
            targets,
            queries,
            stride,
        )
        name_keys = name_keys[name_rank < 100]

        addr_keys, addr_rank, addr_time = timed_keys(
            AddressChannel(k=args.address_k, rerank_pool=args.rerank_pool, **common),
            targets,
            queries,
            stride,
        )
        addr_keys = addr_keys[addr_rank < args.address_k]

        tr_keys, tr_rank, tr_time = timed_keys(
            TransliteratedNameChannel(k=k_max, rerank_pool=args.rerank_pool, **common),
            targets,
            queries,
            stride,
        )

        print(
            f"  exact {exact_time:.1f}s, name rerank {name_time:.1f}s, "
            f"address rerank {addr_time:.1f}s, translit rerank {tr_time:.1f}s"
        )

        base_keys = np.union1d(np.union1d(exact_keys, name_keys), addr_keys)
        base_time = exact_time + name_time + addr_time

        add("exact", exact_keys, n_q, stride, truth_keys, exact_time)
        add(f"name rerank{args.rerank_pool} K=100", name_keys, n_q, stride, truth_keys, name_time)
        add(
            f"address rerank{args.rerank_pool} K={args.address_k}",
            addr_keys, n_q, stride, truth_keys, addr_time,
        )
        add(BASE, base_keys, n_q, stride, truth_keys, base_time)

        # Per-link record for the incremental breakdown.
        valid = truth_t >= 0
        q_names = queries["norm_name"].fillna("").to_numpy()[truth_q]
        t_names = np.full(len(truth_t), "", dtype=object)
        t_names[valid] = targets["norm_name"].fillna("").to_numpy()[truth_t[valid]]

        q_script = pd.Series(q_names).map(dominant_script).to_numpy()
        t_script = pd.Series(t_names).map(dominant_script).to_numpy()

        links = pd.DataFrame(
            {
                "country": country,
                "q": truth_q,
                "t": truth_t,
                "cross_script": (q_script != t_script) & (q_script != "") & (t_script != ""),
                "script_pair": [
                    f"{a or '-'}→{b or '-'}" for a, b in zip(q_script, t_script)
                ],
                "in_base": np.isin(truth_keys, base_keys),
            }
        )

        for k in k_values:
            tk = tr_keys[tr_rank < k]

            add(f"translit rerank{args.rerank_pool} K={k}", tk, n_q, stride, truth_keys, tr_time)
            add(
                f"{BASE} ∪ translit K={k}",
                np.union1d(base_keys, tk),
                n_q, stride, truth_keys, base_time + tr_time,
            )

            links[f"in_tr{k}"] = np.isin(truth_keys, tk)

        # Pair similarities for links still missed after the largest K.
        still = ~links["in_base"] & ~links[f"in_tr{k_max}"] & valid
        sims = pd.DataFrame(index=links.index)

        if still.any():
            idx = np.flatnonzero(still.to_numpy())
            q_rows = queries.iloc[truth_q[idx]].reset_index(drop=True)
            t_rows = targets.iloc[truth_t[idx]].reset_index(drop=True)
            part = pair_similarities(q_rows, t_rows)
            part.index = idx
            part["s1_name"] = q_rows["norm_name"].to_numpy()
            part["target_name"] = t_rows["norm_name"].to_numpy()
            part["s1_address"] = q_rows["norm_address"].to_numpy()
            part["target_address"] = t_rows["norm_address"].to_numpy()
            sims = sims.join(part)

        link_frames.append(pd.concat([links, sims], axis=1))

        del targets

    total = len(truth)

    rows = []
    for label, a in agg.items():
        counts = np.concatenate(a["counts"])
        rows.append(
            {
                "target": target,
                "variant": label,
                "recall": a["hits"] / max(total, 1),
                "pairs": a["pairs"],
                "avg/S1": counts.mean(),
                "median/S1": float(np.median(counts)),
                "max/S1": int(counts.max()),
                "S1 coverage": (counts > 0).mean(),
                "runtime_s": a["runtime_s"],
            }
        )

    return pd.DataFrame(rows), pd.concat(link_frames, ignore_index=True), total


def incremental_table(links: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """Translit-only recoveries over the base union, by segment."""

    segments = {
        "all": np.ones(len(links), bool),
        "india": links["country"] == "india",
        "us": links["country"] == "us",
        "cross-script": links["cross_script"],
        "same-script": ~links["cross_script"],
        "india cross-script": (links["country"] == "india") & links["cross_script"],
    }

    rows = []
    for name, mask in segments.items():
        seg = links[mask]
        missed_by_base = ~seg["in_base"]
        row = {
            "segment": name,
            "true links": len(seg),
            "base recall": seg["in_base"].mean() if len(seg) else np.nan,
            "base missed": int(missed_by_base.sum()),
        }
        for k in k_values:
            gain = (seg[f"in_tr{k}"] & missed_by_base).sum()
            row[f"gain K={k}"] = int(gain)
        k = max(k_values)
        row[f"recall ∪ K={k}"] = (
            (seg["in_base"] | seg[f"in_tr{k}"]).mean() if len(seg) else np.nan
        )
        row[f"still missed K={k}"] = int((~seg["in_base"] & ~seg[f"in_tr{k}"]).sum())
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--targets", nargs="+", default=["S2", "S3"])
    parser.add_argument("--s1-sample", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--translit-k", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--address-k", type=int, default=25)
    parser.add_argument("--rerank-pool", type=int, default=1000)
    parser.add_argument("--max-df", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-work", type=int, default=20_000_000)
    parser.add_argument("--examples", type=int, default=15)
    parser.add_argument(
        "--links-out",
        default=None,
        help="Optional parquet path prefix for per-link results.",
    )
    args = parser.parse_args()

    s1_ids = sample_s1_ids(args.s1_sample, args.seed)
    k_values = sorted(args.translit_k)
    k_max = max(k_values)

    fmt = (
        "display.width", 250,
        "display.max_columns", None,
        "display.max_colwidth", 50,
        "display.float_format", "{:,.4f}".format,
    )

    for target in args.targets:
        table, links, total = run_target(target, args, s1_ids)

        if args.links_out:
            links.to_parquet(f"{args.links_out}_{target}.parquet")

        with pd.option_context(*fmt):
            print(f"\n=== S1 → {target}: CANDIDATE SETS ===")
            print(table.to_string(index=False))

            print(f"\n=== S1 → {target}: INCREMENTAL RECALL OF TRANSLITERATION over {BASE} ===")
            print(incremental_table(links, k_values).to_string(index=False))

            cross_missed = links[links["cross_script"] & ~links["in_base"]]
            if len(cross_missed):
                print("\n  cross-script base misses by script pair (top 8):")
                by = cross_missed.groupby("script_pair")[f"in_tr{k_max}"].agg(["size", "sum"])
                by["recovered %"] = by["sum"] / by["size"]
                print(by.sort_values("size", ascending=False).head(8).to_string())

            still = links[~links["in_base"] & ~links[f"in_tr{k_max}"] & (links["t"] >= 0)]
            print(
                f"\n--- Still missed after ∪ translit K={k_max}: "
                f"{len(still):,} links ({len(still) / max(total, 1):.2%}) ---"
            )
            if len(still):
                print(
                    "  by country: "
                    + ", ".join(f"{c} {n:,}" for c, n in still["country"].value_counts().items())
                    + f"; cross-script {still['cross_script'].mean():.1%}"
                )
                print(
                    still[
                        ["name_ratio", "translit_name_ratio", "name_jaccard", "address_ratio", "address_jaccard"]
                    ].describe(percentiles=PERCENTILES).T.to_string()
                )
                best = np.maximum(still["name_ratio"], still["translit_name_ratio"])
                print(
                    "  best name ratio ≥0.75: "
                    f"{(best >= 0.75).mean():.1%}, <0.50: {(best < 0.5).mean():.1%}; "
                    f"address ratio ≥0.75: {(still['address_ratio'] >= 0.75).mean():.1%}, "
                    f"<0.50: {(still['address_ratio'] < 0.5).mean():.1%}; "
                    "both name<0.5 and address<0.5: "
                    f"{((best < 0.5) & (still['address_ratio'] < 0.5)).mean():.1%}"
                )

                print(f"\n  {args.examples} random examples:")
                sample = still.sample(min(args.examples, len(still)), random_state=0)
                for _, r in sample.iterrows():
                    print(
                        f"   [{r['country']}, {r['script_pair']}] "
                        f"name {r['name_ratio']:.2f}/{r['translit_name_ratio']:.2f} "
                        f"addr {r['address_ratio']:.2f}\n"
                        f"      S1: {r['s1_name']!r} | {r['s1_address']!r}\n"
                        f"      {target}: {r['target_name']!r} | {r['target_address']!r}"
                    )


if __name__ == "__main__":
    main()
