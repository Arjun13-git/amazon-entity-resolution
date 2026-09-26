"""
Address channel benchmark and its incremental recall over
exact-name ∪ char-name-rerank.

Same protocol as ``diagnose_union``: a random S1 sample (default 20,000,
seed 42) against the COMPLETE S2/S3 country partitions.

    python -m src.blocking.benchmark_address_union
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.blocking.benchmark_candidates import TARGET_FILES, locate, sample_s1_ids
from src.blocking.channels import AddressChannel, CharNgramChannel, ExactNameChannel
from src.blocking.diagnose_union import PERCENTILES, collect_keys, pair_similarities
from src.blocking.entity_cache import list_countries, load_partition, load_truth


def address_modes(args) -> list[tuple[str, dict]]:

    modes = [("address", {})]

    if args.address_rerank_pool:
        modes.append(
            (
                f"address rerank{args.address_rerank_pool}",
                {"rerank_pool": args.address_rerank_pool},
            )
        )

    return modes


def timed_keys(channel, targets, queries, stride):

    start = time.perf_counter()
    channel.fit(targets)
    keys, rank, _ = collect_keys(channel, queries, stride)

    return keys, rank, time.perf_counter() - start


def run_target(target, args, s1_ids):

    truth = load_truth(target)
    truth = truth[np.isin(truth["s1_id"].to_numpy(), s1_ids)]

    k_values = sorted(args.address_k)
    k_max = max(k_values)

    agg: dict[str, dict] = {}
    incremental = []
    missed_frames = []

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

        channel_kwargs = dict(
            max_df=args.max_df, workers=args.workers, chunk_work=args.chunk_work
        )

        exact_keys, _, exact_time = timed_keys(
            ExactNameChannel(), targets, queries, stride
        )

        char_keys, char_rank, char_time = timed_keys(
            CharNgramChannel(k=100, rerank_pool=args.name_rerank_pool, **channel_kwargs),
            targets,
            queries,
            stride,
        )
        char_keys = char_keys[char_rank < 100]

        print(f"  exact {exact_time:.1f}s, char rerank {char_time:.1f}s")

        base_keys = np.union1d(exact_keys, char_keys)
        base_time = exact_time + char_time

        add("exact", exact_keys, n_q, stride, truth_keys, exact_time)
        add(f"char rerank{args.name_rerank_pool} K=100", char_keys, n_q, stride, truth_keys, char_time)
        add("exact ∪ char-rerank", base_keys, n_q, stride, truth_keys, base_time)

        in_base = np.isin(truth_keys, base_keys)

        best_union = None

        for label, options in address_modes(args):
            addr_keys, addr_rank, addr_time = timed_keys(
                AddressChannel(k=k_max, **options, **channel_kwargs),
                targets,
                queries,
                stride,
            )
            print(f"  {label} {addr_time:.1f}s")

            for k in k_values:
                ak = addr_keys[addr_rank < k]
                union = np.union1d(base_keys, ak)

                add(f"{label} K={k}", ak, n_q, stride, truth_keys, addr_time)
                add(
                    f"exact ∪ char-rerank ∪ {label} K={k}",
                    union,
                    n_q,
                    stride,
                    truth_keys,
                    base_time + addr_time,
                )

                in_addr = np.isin(truth_keys, ak)
                incremental.append(
                    {
                        "target": target,
                        "country": country,
                        "address": f"{label} K={k}",
                        "true links": len(truth_keys),
                        "base hits": int(in_base.sum()),
                        "address hits": int(in_addr.sum()),
                        "address-only gain": int((in_addr & ~in_base).sum()),
                        "still missed": int((~in_addr & ~in_base).sum()),
                    }
                )

                best_union = (label, k, in_addr)

        # Links missed by the last (largest-K, last-mode) full union.
        label, k, in_addr = best_union
        missed = ~in_base & ~in_addr & (truth_t >= 0)

        if missed.any():
            q_rows = queries.iloc[truth_q[missed]].reset_index(drop=True)
            t_rows = targets.iloc[truth_t[missed]].reset_index(drop=True)
            sims = pair_similarities(q_rows, t_rows)
            sims["country"] = country
            missed_frames.append(sims)

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

    missed = (
        pd.concat(missed_frames, ignore_index=True) if missed_frames else pd.DataFrame()
    )

    return pd.DataFrame(rows), pd.DataFrame(incremental), missed, total, best_union[:2]


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--targets", nargs="+", default=["S2", "S3"])
    parser.add_argument("--s1-sample", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--address-k", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--max-df", type=int, default=30_000)
    parser.add_argument("--name-rerank-pool", type=int, default=1000)
    parser.add_argument(
        "--address-rerank-pool",
        type=int,
        default=1000,
        help="Also run the address channel with full-cosine rerank (0 = off).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-work", type=int, default=20_000_000)
    args = parser.parse_args()

    s1_ids = sample_s1_ids(args.s1_sample, args.seed)

    fmt = (
        "display.width", 250,
        "display.max_columns", None,
        "display.max_colwidth", 60,
        "display.float_format", "{:,.4f}".format,
    )

    for target in args.targets:
        table, inc, missed, total, (label, k) = run_target(target, args, s1_ids)

        with pd.option_context(*fmt):
            print(f"\n=== S1 → {target}: CANDIDATE SETS ===")
            print(table.to_string(index=False))

            print(f"\n=== S1 → {target}: INCREMENTAL RECALL OF ADDRESS ===")
            by = inc.groupby("address", sort=False)[
                ["true links", "base hits", "address hits", "address-only gain", "still missed"]
            ].sum()
            by["gain % of truth"] = by["address-only gain"] / total
            by["still missed %"] = by["still missed"] / total
            print(by.to_string())

            print("\n  by country:")
            inc["gain %"] = inc["address-only gain"] / inc["true links"]
            inc["missed %"] = inc["still missed"] / inc["true links"]
            print(
                inc[["country", "address", "true links", "address-only gain", "gain %", "still missed", "missed %"]]
                .to_string(index=False)
            )

            if len(missed):
                print(
                    f"\n--- Still missed by exact ∪ char-rerank ∪ {label} K={k}: "
                    f"{len(missed):,} links ---"
                )
                print(
                    missed.drop(columns="country")
                    .describe(percentiles=PERCENTILES)
                    .T.to_string()
                )
                best = np.maximum(missed["name_ratio"], missed["translit_name_ratio"])
                print(
                    "  name (best of original/translit) ratio: "
                    f"≥0.75 {(best >= 0.75).mean():.1%}, "
                    f"<0.50 {(best < 0.5).mean():.1%}; "
                    "translit − original > 0.3 (cross-script-like): "
                    f"{((missed['translit_name_ratio'] - missed['name_ratio']) > 0.3).mean():.1%}; "
                    f"address ratio ≥0.75: {(missed['address_ratio'] >= 0.75).mean():.1%}"
                )


if __name__ == "__main__":
    main()
