"""
Structured-address exact blocking and its incremental recall over
exact ∪ name-char-rerank ∪ address-char-rerank.

Same protocol as the other blocking diagnostics: a random S1 sample
(default 20,000, seed 42) against the COMPLETE S2/S3 country partitions.

Keys (see ``address_keys``): house, postal, city, house_city. Each key's
block sizes and recall are computed from sorted-hash ``searchsorted``
ranges without materializing pairs; candidates are materialized only for
S1 records whose block holds at most ``cap`` targets.

    python -m src.blocking.benchmark_structured_union
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.blocking.address_keys import extract_keys, learn_city_lexicon
from src.blocking.benchmark_address_union import timed_keys
from src.blocking.benchmark_candidates import TARGET_FILES, locate, sample_s1_ids
from src.blocking.channels import (
    AddressChannel,
    CharNgramChannel,
    ExactNameChannel,
    hash_strings,
)
from src.blocking.entity_cache import list_countries, load_partition, load_truth
from src.blocking.profile_true_matches import fuzzy_ratio


KEYS = ["house", "postal", "city", "house_city"]
BASE = "existing union"


def block_ranges(target_keys: pd.Series, query_keys: pd.Series):
    """Per-query [left, right) range into targets sorted by key hash."""

    t_hash = hash_strings(target_keys)
    t_valid = (target_keys != "").to_numpy()

    order = np.argsort(t_hash[t_valid], kind="stable")
    sorted_hash = t_hash[t_valid][order]
    sorted_rows = np.flatnonzero(t_valid)[order].astype(np.int32)

    q_hash = hash_strings(query_keys)
    left = np.searchsorted(sorted_hash, q_hash, "left")
    right = np.searchsorted(sorted_hash, q_hash, "right")

    empty = (query_keys == "").to_numpy()
    right[empty] = left[empty]

    return left, right, sorted_rows


def materialize(left, right, sorted_rows, keep, stride):
    """Candidate keys (q * stride + t) for the queries where ``keep``."""

    counts = np.where(keep, right - left, 0)
    q = np.repeat(np.arange(len(counts)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    t = sorted_rows[np.repeat(left, counts) + offsets]

    return q.astype(np.int64) * stride + t


def run_target(target, args, s1_ids, lexicons):

    truth = load_truth(target)
    truth = truth[np.isin(truth["s1_id"].to_numpy(), s1_ids)]

    caps = sorted(args.caps)

    agg: dict[str, dict] = {}
    key_stats: dict[str, dict] = {}
    link_frames = []

    def add(label, n_pairs, counts, hits, runtime):
        a = agg.setdefault(label, {"pairs": 0, "counts": [], "hits": 0, "runtime_s": 0.0})
        a["pairs"] += int(n_pairs)
        a["counts"].append(counts)
        a["hits"] += int(hits)
        a["runtime_s"] += runtime

    for country in list_countries("source1"):
        cols = ["norm_name", "norm_address"]

        queries = load_partition("source1", country, cols)
        queries = queries[np.isin(queries["id"].to_numpy(), s1_ids)].reset_index(drop=True)
        targets = load_partition(TARGET_FILES[target], country, cols)

        n_q = len(queries)
        stride = np.int64(len(targets) + 1)

        q_pos = locate(queries["id"].to_numpy(), truth["s1_id"].to_numpy())
        t_pos = locate(targets["id"].to_numpy(), truth["target_id"].to_numpy())
        keep = q_pos >= 0
        truth_q, truth_t = q_pos[keep], t_pos[keep]
        truth_keys = truth_q.astype(np.int64) * stride + truth_t
        valid = truth_t >= 0

        print(
            f"\n[{target} | {country}] queries={n_q:,} "
            f"targets={len(targets):,} true links={len(truth_q):,}"
        )

        # --- Existing union (unchanged channels) -------------------------
        common = dict(max_df=args.max_df, workers=args.workers, chunk_work=args.chunk_work)

        exact_keys, _, exact_time = timed_keys(ExactNameChannel(), targets, queries, stride)
        name_keys, name_rank, name_time = timed_keys(
            CharNgramChannel(k=100, rerank_pool=args.rerank_pool, **common),
            targets, queries, stride,
        )
        addr_keys, addr_rank, addr_time = timed_keys(
            AddressChannel(k=args.address_k, rerank_pool=args.rerank_pool, **common),
            targets, queries, stride,
        )

        base_keys = np.union1d(
            np.union1d(exact_keys, name_keys[name_rank < 100]),
            addr_keys[addr_rank < args.address_k],
        )
        base_time = exact_time + name_time + addr_time
        base_counts = np.bincount(base_keys // stride, minlength=n_q)
        in_base = np.isin(truth_keys, base_keys)

        add(BASE, len(base_keys), base_counts, in_base.sum(), base_time)

        # --- Structured keys ------------------------------------------------
        start = time.perf_counter()
        qk = extract_keys(queries["norm_address"], country, lexicons[country])
        tk = extract_keys(targets["norm_address"], country, lexicons[country])
        extract_time = time.perf_counter() - start

        print(
            f"  existing union {base_time:.1f}s; key extraction {extract_time:.1f}s "
            f"({len(targets):,} targets)"
        )

        base_q = (base_keys // stride).astype(np.int64)
        base_t = (base_keys % stride).astype(np.int64)

        links = pd.DataFrame({"country": country, "in_base": in_base, "valid": valid})

        capped_union = {cap: [base_keys] for cap in caps}

        for key in KEYS:
            start = time.perf_counter()
            left, right, sorted_rows = block_ranges(tk[key], qk[key])
            sizes = right - left

            q_key = qk[key].to_numpy()
            t_key = tk[key].to_numpy()

            # Truth link retrieved (uncapped) iff both keys equal and non-empty.
            t_key_truth = np.where(valid, t_key[np.where(valid, truth_t, 0)], "")
            hit = (q_key[truth_q] == t_key_truth) & (q_key[truth_q] != "")
            key_time = time.perf_counter() - start

            # Existing-union pairs that the key also produces (overlap).
            overlap = (q_key[base_q] == t_key[base_t]) & (q_key[base_q] != "")
            overlap_per_q = np.bincount(base_q[overlap], minlength=n_q)

            s = key_stats.setdefault(key, {"has_key": 0, "sizes": [], "runtime_s": 0.0})
            s["has_key"] += int((q_key != "").sum())
            s["sizes"].append(sizes)
            s["runtime_s"] += key_time + extract_time / len(KEYS)

            add(f"{key} (uncapped)", sizes.sum(), sizes, hit.sum(), key_time)
            add(
                f"{BASE} ∪ {key} (uncapped)",
                len(base_keys) + sizes.sum() - overlap.sum(),
                base_counts + sizes - overlap_per_q,
                (in_base | hit).sum(),
                base_time + key_time,
            )
            links[f"{key}"] = hit

            for cap in caps:
                start = time.perf_counter()
                small = sizes <= cap
                ck = materialize(left, right, sorted_rows, small, stride)
                union = np.union1d(base_keys, ck)
                cap_time = time.perf_counter() - start

                c_hit = hit & small[truth_q]
                add(
                    f"{BASE} ∪ {key} (block ≤ {cap})",
                    len(union),
                    np.bincount(union // stride, minlength=n_q),
                    (in_base | c_hit).sum(),
                    base_time + key_time + cap_time,
                )
                links[f"{key}@{cap}"] = c_hit
                capped_union[cap].append(ck)

        for cap in caps:
            union = np.unique(np.concatenate(capped_union[cap]))
            all_hit = np.zeros(len(truth_q), bool)
            for key in KEYS:
                all_hit |= links[f"{key}@{cap}"].to_numpy()
            add(
                f"{BASE} ∪ all keys (block ≤ {cap})",
                len(union),
                np.bincount(union // stride, minlength=n_q),
                (in_base | all_hit).sum(),
                base_time + extract_time,
            )
            links[f"all@{cap}"] = all_hit

        # Name / address ratios for base misses.
        miss = np.flatnonzero(~in_base & valid)
        name_r = np.full(len(truth_q), np.nan)
        addr_r = np.full(len(truth_q), np.nan)
        qn = queries["norm_name"].fillna("").to_numpy()
        qa = queries["norm_address"].fillna("").to_numpy()
        tn = targets["norm_name"].fillna("").to_numpy()
        ta = targets["norm_address"].fillna("").to_numpy()
        for i in miss:
            name_r[i] = fuzzy_ratio(qn[truth_q[i]], tn[truth_t[i]])
            addr_r[i] = fuzzy_ratio(qa[truth_q[i]], ta[truth_t[i]])
        links["name_ratio"] = name_r
        links["address_ratio"] = addr_r

        link_frames.append(links)

        del targets

    return agg, key_stats, pd.concat(link_frames, ignore_index=True), len(truth)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--targets", nargs="+", default=["S2", "S3"])
    parser.add_argument("--s1-sample", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--caps", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--address-k", type=int, default=25)
    parser.add_argument("--rerank-pool", type=int, default=1000)
    parser.add_argument("--max-df", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-work", type=int, default=20_000_000)
    parser.add_argument("--links-out", default=None)
    args = parser.parse_args()

    s1_ids = sample_s1_ids(args.s1_sample, args.seed)

    lexicons = {}
    for country in list_countries("source1"):
        start = time.perf_counter()
        s1 = load_partition("source1", country, ["norm_address"])
        lexicons[country] = learn_city_lexicon(s1["norm_address"], country)
        print(
            f"[lexicon] {country}: {len(lexicons[country]):,} cities "
            f"from {len(s1):,} S1 addresses ({time.perf_counter() - start:.0f}s)"
        )
        del s1

    fmt = (
        "display.width", 250,
        "display.max_columns", None,
        "display.float_format", "{:,.4f}".format,
    )

    for target in args.targets:
        agg, key_stats, links, total = run_target(target, args, s1_ids, lexicons)

        if args.links_out:
            links.to_parquet(f"{args.links_out}_{target}.parquet")

        n_s1 = len(np.concatenate(agg[BASE]["counts"]))

        with pd.option_context(*fmt):
            print(f"\n=== S1 → {target}: STRUCTURED KEYS (uncapped exact blocks) ===")
            rows = []
            for key, s in key_stats.items():
                sizes = np.concatenate(s["sizes"])
                has = sizes > 0
                rows.append(
                    {
                        "key": key,
                        "S1 with key": s["has_key"],
                        "S1 with ≥1 cand": int(has.sum()),
                        "pairs": int(sizes.sum()),
                        "mean": sizes.mean(),
                        "p50": np.median(sizes),
                        "p90": np.percentile(sizes, 90),
                        "p99": np.percentile(sizes, 99),
                        "max": int(sizes.max()),
                        "≤100": (has & (sizes <= 100)).sum() / max(has.sum(), 1),
                        "≤1000": (has & (sizes <= 1000)).sum() / max(has.sum(), 1),
                        "recall": links[key].sum() / total,
                        "runtime_s": s["runtime_s"],
                    }
                )
            print(pd.DataFrame(rows).to_string(index=False))

            print(f"\n=== S1 → {target}: UNIONS ===")
            rows = []
            base_hits = agg[BASE]["hits"]
            for label, a in agg.items():
                if not label.startswith(BASE):
                    continue
                counts = np.concatenate(a["counts"])
                rows.append(
                    {
                        "variant": label,
                        "recall": a["hits"] / total,
                        "incremental links": a["hits"] - base_hits,
                        "pairs": a["pairs"],
                        "avg/S1": counts.mean(),
                        "median/S1": float(np.median(counts)),
                        "max/S1": int(counts.max()),
                        "S1 coverage": (counts > 0).mean(),
                        "runtime_s": a["runtime_s"],
                    }
                )
            print(pd.DataFrame(rows).to_string(index=False))

            missed = links[~links["in_base"] & links["valid"]]
            hard = missed[(missed["name_ratio"] < 0.5) & (missed["address_ratio"] < 0.5)]
            print(
                f"\n=== S1 → {target}: BASE MISSES ({len(missed):,}); "
                f"name<0.5 AND address<0.5: {len(hard):,} ==="
            )
            cols = [c for c in links.columns if c in KEYS or "@" in c]
            rec = pd.DataFrame(
                {
                    "all base misses": missed[cols].sum(),
                    "hard (both<0.5)": hard[cols].sum(),
                    "hard %": hard[cols].mean() if len(hard) else np.nan,
                    "india hard": hard[hard["country"] == "india"][cols].sum(),
                    "us hard": hard[hard["country"] == "us"][cols].sum(),
                }
            )
            print("  recovered by:")
            print(rec.to_string())
            print(
                "  hard misses by country: "
                + ", ".join(f"{c} {n:,}" for c, n in hard["country"].value_counts().items())
            )


if __name__ == "__main__":
    main()
