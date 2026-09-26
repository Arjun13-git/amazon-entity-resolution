"""
Union/overlap diagnosis of exact-name and character n-gram candidates.

Uses the same random S1 sample as ``benchmark_candidates`` (same
``--s1-sample`` / ``--seed``) against the COMPLETE S2/S3 partitions.

For character K=100 each true link falls in one of:
    A  exact-name only
    B  character top-K (split into B-only and D = both)
    C  neither

Missed links (C) are then explained using the true target's exact rank
among all targets, its retrieval score vs. the query's K-th score, and
its full (un-pruned) cosine, to separate:
    - crowded out: similar name, but >= K targets scored higher
    - index pruning: similar by full cosine, but the df-capped index
      barely sees it
    - dissimilar name: the retriever had nothing to find

    python -m src.blocking.diagnose_union --s1-sample 20000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.blocking.benchmark_candidates import (
    TARGET_FILES,
    locate,
    sample_s1_ids,
)
from src.blocking.channels import CharNgramChannel, ExactNameChannel
from src.blocking.entity_cache import list_countries, load_partition, load_truth
from src.blocking.profile_true_matches import fuzzy_ratio, token_jaccard
from src.preprocessing.transliterate import transliterate_text


PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# Name-similarity thresholds used to label missed links.
SIMILAR_RATIO = 0.75
DISSIMILAR_RATIO = 0.50


def collect_keys(channel, queries, stride, k=None):
    """All candidate keys (q * stride + t) of a channel, with ranks."""

    keys, ranks, scores = [], [], []

    for batch in channel.retrieve(queries):
        keys.append(batch.q_idx.astype(np.int64) * stride + batch.t_idx)
        ranks.append(batch.rank)
        scores.append(batch.score)

    return np.concatenate(keys), np.concatenate(ranks), np.concatenate(scores)


def summarize(label, keys, n_queries, stride, truth_keys, runtime):

    counts = np.bincount(keys // stride, minlength=n_queries)
    hit = np.isin(truth_keys, keys)

    return {
        "setting": label,
        "pairs": int(len(keys)),
        "counts": counts,
        "hit": hit,
        "runtime_s": runtime,
    }


def true_ranks(channel, queries, q_rows, t_rows, chunk=200):
    """
    For each (query, true target) pair: the target's retrieval score and
    how many targets score strictly higher / at least as high.
    """

    matrix = channel._vectorize(queries["norm_name"].iloc[np.unique(q_rows)])
    unique_q = np.unique(q_rows)
    row_of = np.searchsorted(unique_q, q_rows)

    score = np.zeros(len(q_rows), np.float32)
    above = np.zeros(len(q_rows), np.int64)
    at_or_above = np.zeros(len(q_rows), np.int64)

    for start in range(0, len(unique_q), chunk):
        end = min(start + chunk, len(unique_q))
        product = (matrix[start:end] @ channel.index_t).tocsr()

        for i in np.flatnonzero((row_of >= start) & (row_of < end)):
            r = row_of[i] - start
            lo, hi = product.indptr[r], product.indptr[r + 1]
            cols = product.indices[lo:hi]
            vals = product.data[lo:hi]

            match = vals[cols == t_rows[i]]
            s = match[0] if len(match) else 0.0

            score[i] = s
            above[i] = (vals > s).sum() if s > 0 else len(vals)
            at_or_above[i] = (vals >= s).sum() if s > 0 else len(vals)

    return score, above, at_or_above


def full_cosine(channel, left: pd.Series, right: pd.Series) -> np.ndarray:
    """Cosine of un-pruned TF-IDF vectors (all n-grams)."""

    a = channel._vectorize(left)
    b = channel._vectorize(right)

    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def pair_similarities(q: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:

    qn = q["norm_name"].fillna("").to_numpy()
    tn = t["norm_name"].fillna("").to_numpy()
    qa = q["norm_address"].fillna("").to_numpy()
    ta = t["norm_address"].fillna("").to_numpy()

    return pd.DataFrame(
        {
            "name_ratio": [fuzzy_ratio(a, b) for a, b in zip(qn, tn)],
            "translit_name_ratio": [
                fuzzy_ratio(transliterate_text(a), transliterate_text(b))
                for a, b in zip(qn, tn)
            ],
            "name_jaccard": [token_jaccard(a, b) for a, b in zip(qn, tn)],
            "address_ratio": [fuzzy_ratio(a, b) for a, b in zip(qa, ta)],
            "address_jaccard": [token_jaccard(a, b) for a, b in zip(qa, ta)],
        }
    )


def run_target(target, args, s1_ids):

    truth = load_truth(target)
    truth = truth[np.isin(truth["s1_id"].to_numpy(), s1_ids)]

    k_values = sorted(args.k_values)
    k_max = max(k_values)

    per_setting: dict[str, dict] = {}
    missed_frames = []
    category_frames = []
    n_queries_total = 0

    for country in list_countries("source1"):
        cols = ["norm_name", "norm_address"]

        queries = load_partition("source1", country, cols)
        queries = queries[
            np.isin(queries["id"].to_numpy(), s1_ids)
        ].reset_index(drop=True)

        targets = load_partition(TARGET_FILES[target], country, cols)

        stride = np.int64(len(targets) + 1)
        n_q = len(queries)
        n_queries_total += n_q

        q_pos = locate(queries["id"].to_numpy(), truth["s1_id"].to_numpy())
        t_pos = locate(targets["id"].to_numpy(), truth["target_id"].to_numpy())
        keep = q_pos >= 0
        truth_q, truth_t = q_pos[keep], t_pos[keep]
        truth_keys = truth_q.astype(np.int64) * stride + truth_t

        print(
            f"\n[{target} | {country}] queries={n_q:,} "
            f"targets={len(targets):,} true links={len(truth_q):,}"
        )

        # --- Exact name ---------------------------------------------------
        start = time.perf_counter()
        exact = ExactNameChannel()
        exact.fit(targets)
        exact_keys, _, _ = collect_keys(exact, queries, stride)
        exact_time = time.perf_counter() - start

        # --- Character top-K ---------------------------------------------
        start = time.perf_counter()
        char = CharNgramChannel(
            max_df=args.max_df,
            k=k_max,
            workers=args.workers,
            chunk_work=args.chunk_work,
        )
        char.fit(targets)
        char_keys, char_rank, char_score = collect_keys(char, queries, stride)
        char_time = time.perf_counter() - start

        print(
            f"  exact {exact_time:.1f}s, char {char_time:.1f}s "
            f"(index {char.index_nbytes() / 2**20:,.0f} MB)"
        )

        # --- Alternative character variants ----------------------------
        variant_out = {}

        for label, options in variants(args):
            start = time.perf_counter()
            channel = CharNgramChannel(
                max_df=args.max_df,
                k=k_max,
                workers=args.workers,
                chunk_work=args.chunk_work,
                **options,
            )
            channel.fit(targets)
            v_keys, v_rank, v_score = collect_keys(channel, queries, stride)
            v_time = time.perf_counter() - start

            print(
                f"  {label} {v_time:.1f}s "
                f"(index {channel.index_nbytes() / 2**20:,.0f} MB)"
            )

            variant_out[label] = (channel, v_keys, v_rank, v_score, v_time)

        results = [
            summarize("exact", exact_keys, n_q, stride, truth_keys, exact_time)
        ]

        for k in k_values:
            ck = char_keys[char_rank < k]
            results.append(
                summarize(f"char K={k}", ck, n_q, stride, truth_keys, char_time)
            )
            results.append(
                summarize(
                    f"exact ∪ char K={k}",
                    np.union1d(exact_keys, ck),
                    n_q,
                    stride,
                    truth_keys,
                    exact_time + char_time,
                )
            )

        for label, (_, v_keys, v_rank, _, v_time) in variant_out.items():
            for k in k_values:
                ck = v_keys[v_rank < k]
                name = f"{label} K={k}"
                results.append(
                    summarize(name, ck, n_q, stride, truth_keys, v_time)
                )
                results.append(
                    summarize(
                        f"exact ∪ {name}",
                        np.union1d(exact_keys, ck),
                        n_q,
                        stride,
                        truth_keys,
                        exact_time + v_time,
                    )
                )

        for r in results:
            agg = per_setting.setdefault(
                r["setting"],
                {"pairs": 0, "counts": [], "hits": 0, "runtime_s": 0.0},
            )
            agg["pairs"] += r["pairs"]
            agg["counts"].append(r["counts"])
            agg["hits"] += int(r["hit"].sum())
            agg["runtime_s"] += r["runtime_s"]

        # --- Overlap categories at K = k_max -----------------------------
        in_exact = np.isin(truth_keys, exact_keys)
        in_char = np.isin(truth_keys, char_keys)

        category = np.select(
            [in_exact & in_char, in_exact, in_char],
            ["D both", "A exact only", "B char only"],
            default="C neither",
        )

        # Query's K-th best score (0 when it has < K candidates).
        kth = np.zeros(n_q, np.float32)
        last = char_rank == k_max - 1
        kth[(char_keys[last] // stride).astype(np.int64)] = char_score[last]

        # Number of targets sharing the query's exact name.
        same_name = np.bincount(exact_keys // stride, minlength=n_q)

        frame = pd.DataFrame(
            {
                "category": category,
                "q": truth_q,
                "t": truth_t,
                "same_name_targets": same_name[truth_q],
                "kth_score": kth[truth_q],
            }
        )

        for label, (_, v_keys, _, _, _) in variant_out.items():
            in_v = np.isin(truth_keys, v_keys)
            frame[f"in_{label}"] = in_v
            frame[f"cat_{label}"] = np.select(
                [in_exact & in_v, in_exact, in_v],
                ["D both", "A exact only", "B char only"],
                default="C neither",
            )

        category_frames.append(frame.drop(columns=["q", "t"]))

        # --- Explain misses (A and C: not in char top-K) -----------------
        missed = frame[~in_char & (truth_t >= 0)].copy()

        if len(missed):
            score, above, at_or_above = true_ranks(
                char, queries, missed["q"].to_numpy(), missed["t"].to_numpy()
            )
            missed["true_score"] = score
            missed["targets_above"] = above
            missed["targets_at_or_above"] = at_or_above

            q_rows = queries.iloc[missed["q"].to_numpy()].reset_index(drop=True)
            t_rows = targets.iloc[missed["t"].to_numpy()].reset_index(drop=True)

            missed["full_cosine"] = full_cosine(
                char, q_rows["norm_name"], t_rows["norm_name"]
            )

            # Each variant's own retrieval score for the true pair, and the
            # query's K-th score under that variant.
            for label, (channel, v_keys, v_rank, v_score, _) in variant_out.items():
                if channel.rerank_pool:
                    continue
                missed[f"score_{label}"] = np.asarray(
                    channel.retrieval_vectors(q_rows["norm_name"])
                    .multiply(channel.retrieval_vectors(t_rows["norm_name"]))
                    .sum(axis=1)
                ).ravel()
                v_kth = np.zeros(n_q, np.float32)
                last = v_rank == k_max - 1
                v_kth[(v_keys[last] // stride).astype(np.int64)] = v_score[last]
                missed[f"kth_{label}"] = v_kth[missed["q"].to_numpy()]

            sims = pair_similarities(q_rows, t_rows)
            missed = pd.concat(
                [missed.reset_index(drop=True), sims], axis=1
            )
            missed_frames.append(missed)

        del targets, char, exact, variant_out

    # --- Report --------------------------------------------------------
    total_truth = len(truth)

    rows = []
    for label, agg in per_setting.items():
        counts = np.concatenate(agg["counts"])
        rows.append(
            {
                "target": target,
                "setting": label,
                "recall": agg["hits"] / max(total_truth, 1),
                "pairs": agg["pairs"],
                "avg/S1": counts.mean(),
                "median/S1": float(np.median(counts)),
                "max/S1": int(counts.max()),
                "S1 coverage": (counts > 0).mean(),
                "runtime_s": agg["runtime_s"],
            }
        )

    categories = pd.concat(category_frames)
    missed = (
        pd.concat(missed_frames, ignore_index=True)
        if missed_frames
        else pd.DataFrame()
    )

    return pd.DataFrame(rows), categories, missed, total_truth


def classify_miss(missed: pd.DataFrame, k: int) -> pd.Series:
    """Why a true link is outside the character top-K."""

    similar = missed["name_ratio"] >= SIMILAR_RATIO
    dissimilar = missed["name_ratio"] < DISSIMILAR_RATIO

    pruned = (
        (missed["full_cosine"] >= 0.5)
        & (missed["true_score"] < 0.5 * missed["full_cosine"])
    )

    return pd.Series(
        np.select(
            [
                missed["true_score"] <= 0,
                pruned,
                similar,
                dissimilar,
            ],
            [
                "not reachable (no shared indexed n-gram)",
                "index pruning (full cosine high, indexed score low)",
                "crowded out (name ratio ≥ 0.75)",
                "dissimilar name (name ratio < 0.50)",
            ],
            default="partly similar (0.50 ≤ ratio < 0.75)",
        ),
        index=missed.index,
    )


def print_report(target, table, categories, missed, total_truth, k):

    fmt = {
        "display.width": 250,
        "display.max_columns": None,
        "display.float_format": "{:,.4f}".format,
    }

    with pd.option_context(*[x for kv in fmt.items() for x in kv]):
        print(f"\n=== S1 → {target}: CANDIDATE SETS ===")
        print(table.to_string(index=False))

        print(f"\n=== S1 → {target}: OVERLAP AT CHARACTER K={k} ===")
        vc = categories["category"].value_counts().sort_index()
        for name, count in vc.items():
            print(f"  {name:<14} {count:>8,}  {count / total_truth:7.2%}")

        c = missed[missed["category"] == "C neither"]
        a = missed[missed["category"] == "A exact only"]

        print(f"\n--- C (missed by both): {len(c):,} links ---")
        print(
            c[
                [
                    "name_ratio",
                    "translit_name_ratio",
                    "name_jaccard",
                    "address_ratio",
                    "address_jaccard",
                    "full_cosine",
                    "true_score",
                    "kth_score",
                ]
            ].describe(percentiles=PERCENTILES).T.to_string()
        )

        print(f"\n--- A (exact only, crowded out of top-{k}): {len(a):,} links ---")
        if len(a):
            print(
                "  same-name targets per query: "
                f"median {a['same_name_targets'].median():,.0f}, "
                f"≥ {k}: {(a['same_name_targets'] >= k).mean():.1%}; "
                "true score == K-th score (tie): "
                f"{np.isclose(a['true_score'], a['kth_score']).mean():.1%}"
            )

        print(f"\n--- Why links are outside character top-{k} (A + C) ---")
        for name, frame in (("A", a), ("C", c)):
            if not len(frame):
                continue
            reasons = classify_miss(frame, k).value_counts()
            print(f"  {name}:")
            for reason, count in reasons.items():
                print(f"    {reason:<55} {count:>7,}  {count / len(frame):6.1%}")

        if len(c):
            rank = c["targets_above"] + 1
            print(f"\n  C: true target's rank in the full character ranking")
            for lo, hi in ((k + 1, 200), (201, 500), (501, 1000), (1001, None)):
                sel = rank >= lo if hi is None else rank.between(lo, hi)
                label = f"{lo}+" if hi is None else f"{lo}-{hi}"
                print(f"    rank {label:<10} {sel.mean():6.1%}")

            print(
                "  C: name ratio buckets: "
                f"≥0.90 {(c['name_ratio'] >= 0.9).mean():.1%}, "
                f"≥0.75 {(c['name_ratio'] >= 0.75).mean():.1%}, "
                f"<0.50 {(c['name_ratio'] < 0.5).mean():.1%}, "
                f"==0 {(c['name_ratio'] == 0).mean():.1%}"
            )


def variants(args) -> list[tuple[str, dict]]:
    """Character-channel variants to compare against the baseline."""

    out = []

    if args.rerank_pool:
        out.append((f"char rerank{args.rerank_pool}", {"rerank_pool": args.rerank_pool}))

    if args.renormalize_pruned:
        out.append(("char renorm", {"renormalize_pruned": True}))

    return out


def print_variant_report(label, target, categories, missed, total_truth, k):

    print(f"\n=== S1 → {target}: {label.upper()} vs baseline char top-{k} ===")

    print("  Overlap with exact name:")
    vc = categories[f"cat_{label}"].value_counts().sort_index()
    for name, count in vc.items():
        print(f"    {name:<14} {count:>8,}  {count / total_truth:7.2%}")

    base = categories["category"].isin(["B char only", "D both"])
    new = categories[f"in_{label}"]
    print(
        f"  Baseline char top-{k} hits: {base.sum():,}; {label} hits: "
        f"{new.sum():,}; gained {(new & ~base).sum():,}, "
        f"lost {(base & ~new).sum():,}"
    )

    if not len(missed):
        return

    missed = missed.copy()
    missed["reason"] = classify_miss(missed, k)
    hit = f"in_{label}"

    print(f"  Recovery of baseline misses (outside baseline char top-{k}):")
    for group in ("A exact only", "C neither"):
        frame = missed[missed["category"] == group]
        print(f"    {group} ({len(frame):,}):")
        for reason, sub in frame.groupby("reason"):
            print(
                f"      {reason:<55} {sub[hit].sum():>6,} / "
                f"{len(sub):>6,} recovered ({sub[hit].mean():6.1%})"
            )

    identical = missed[missed["full_cosine"] >= 0.999]
    pruning = missed["reason"].str.startswith("index pruning")

    print(
        f"  Character-for-character identical names among baseline misses: "
        f"{len(identical):,}; recovered {identical[hit].sum():,} "
        f"({identical[hit].mean():.1%})"
    )
    print(
        "  Pruning-distortion misses (A + C): "
        f"{pruning.sum():,}; recovered {missed.loc[pruning, hit].sum():,} "
        f"({missed.loc[pruning, hit].mean():.1%})"
    )

    score_col = f"score_{label}"
    if score_col in missed and len(identical):
        old = identical["true_score"]
        new_score = identical[score_col]
        kth = identical[f"kth_{label}"]
        print(
            "  Identical-name pairs, retrieval score of true target: "
            f"baseline median {old.median():.3f} (≥0.999: {(old >= 0.999).mean():.1%}); "
            f"{label} median {new_score.median():.3f} "
            f"(≥0.999: {(new_score >= 0.999).mean():.1%}, "
            f"zero: {(new_score <= 0).mean():.1%})"
        )
        still = identical[~identical[hit]]
        if len(still):
            print(
                f"  Identical-name pairs still missed: {len(still):,}; "
                f"score ≥0.999 {(still[score_col] >= 0.999).mean():.1%}, "
                f"tied with K-th score {np.isclose(still[score_col], still[f'kth_{label}']).mean():.1%}, "
                f"same-name targets ≥{k}: {(still['same_name_targets'] >= k).mean():.1%}, "
                f"zero score (all n-grams pruned) {(still[score_col] <= 0).mean():.1%}"
            )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--targets", nargs="+", default=["S2", "S3"])
    parser.add_argument("--s1-sample", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--max-df", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-work", type=int, default=20_000_000)
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=None,
        help="Also run pruned-top-N + full-cosine rerank (e.g. 1000).",
    )
    parser.add_argument(
        "--renormalize-pruned",
        action="store_true",
        help="Also run the re-normalized pruned-space cosine variant.",
    )
    parser.add_argument(
        "--missed-out",
        default=None,
        help="Optional parquet path prefix to save missed-link details.",
    )
    args = parser.parse_args()

    s1_ids = sample_s1_ids(args.s1_sample, args.seed)

    for target in args.targets:
        table, categories, missed, total = run_target(target, args, s1_ids)

        print_report(target, table, categories, missed, total, max(args.k_values))
        for label, _ in variants(args):
            print_variant_report(
                label, target, categories, missed, total, max(args.k_values)
            )

        if args.missed_out:
            missed.to_parquet(f"{args.missed_out}_{target}.parquet")


if __name__ == "__main__":
    main()
