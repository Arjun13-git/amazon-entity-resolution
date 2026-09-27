"""
Benchmark candidate-generation channels against TRAIN ground truth.

Targets are always the COMPLETE train_source2 / train_source3 country
partitions. S1 may be sampled (``--s1-sample``); per-link recall on a
random S1 sample is an unbiased estimate of full recall, because every
true link of a sampled S1 is evaluated against every target.

Examples:

    # Memory/work estimate only (builds the index, no retrieval).
    python -m src.blocking.benchmark_candidates --channels char \\
        --s1-sample 20000 --estimate-only

    python -m src.blocking.benchmark_candidates --channels exact
    python -m src.blocking.benchmark_candidates --channels char \\
        --s1-sample 20000 --k-values 10 20 50 100

Requires the cache: ``python -m src.blocking.entity_cache``.
"""

from __future__ import annotations

import argparse
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.blocking.channels import (
    AddressChannel,
    CandidateChannel,
    CharNgramChannel,
    ExactNameChannel,
)
from src.blocking.entity_cache import (
    cache_path,
    list_countries,
    load_partition,
    load_truth,
)


TEXT_COLUMNS = ["norm_name", "norm_address"]

TARGET_FILES = {
    "S2": "source2",
    "S3": "source3",
}


def peak_rss_mb() -> tuple[float, float]:
    """Peak RSS of this process and of its largest finished child (MB)."""

    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024

    return own, child


@dataclass
class SettingStats:
    """Accumulated metrics for one channel setting (e.g. one K)."""

    label: str
    counts: list[np.ndarray] = field(default_factory=list)
    retrieved: int = 0
    s1_with_hit: int = 0


def build_channel(name: str, args: argparse.Namespace) -> CandidateChannel:

    if name == "exact":
        return ExactNameChannel()

    if name in ("char", "address"):
        cls = AddressChannel if name == "address" else CharNgramChannel
        return cls(
            ngram_range=tuple(args.ngram),
            max_df=args.max_df,
            k=max(args.k_values),
            workers=args.workers,
            chunk_work=args.chunk_work,
        )

    raise ValueError(f"Unknown channel: {name}")


def settings_for(name: str, args: argparse.Namespace) -> list[int | None]:
    """Rank cut-offs to evaluate; None = keep every candidate."""

    return sorted(args.k_values) if name in ("char", "address") else [None]


def sample_s1_ids(sample: int | None, seed: int, dataset: str = "train") -> np.ndarray | None:

    if not sample:
        return None

    ids = pq.read_table(
        cache_path("source1", dataset),
        columns=["id"],
    )["id"].to_numpy()

    rng = np.random.default_rng(seed)

    return np.sort(rng.choice(ids, size=min(sample, len(ids)), replace=False))


def locate(sorted_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Positions of ``ids`` in ``sorted_ids``; -1 where absent."""

    pos = np.searchsorted(sorted_ids, ids)
    pos = np.minimum(pos, len(sorted_ids) - 1)

    found = sorted_ids[pos] == ids if len(sorted_ids) else np.zeros(len(ids), bool)

    return np.where(found, pos, -1)


def run_channel(
    channel_name: str,
    target: str,
    args: argparse.Namespace,
    s1_ids: np.ndarray | None,
) -> list[dict]:

    truth = load_truth(target)

    if s1_ids is not None:
        truth = truth[np.isin(truth["s1_id"].to_numpy(), s1_ids)]

    total_truth = len(truth)
    n_matched_s1 = truth["s1_id"].nunique()

    settings = settings_for(channel_name, args)
    stats = {
        k: SettingStats(label="all" if k is None else f"K={k}")
        for k in settings
    }

    n_queries = 0
    runtime = 0.0

    writer = None

    for country in list_countries("source1"):
        queries = load_partition("source1", country, TEXT_COLUMNS)

        if s1_ids is not None:
            queries = queries[
                np.isin(queries["id"].to_numpy(), s1_ids)
            ].reset_index(drop=True)

        targets = load_partition(TARGET_FILES[target], country, TEXT_COLUMNS)

        query_ids = queries["id"].to_numpy()
        target_ids = targets["id"].to_numpy()

        n_queries += len(queries)

        # Truth rows for this partition's queries, as positional indices.
        q_pos = locate(query_ids, truth["s1_id"].to_numpy())
        t_pos = locate(target_ids, truth["target_id"].to_numpy())

        mask = q_pos >= 0
        order = np.argsort(q_pos[mask], kind="stable")
        truth_q = q_pos[mask][order]
        truth_t = t_pos[mask][order]

        stride = np.int64(len(targets) + 1)
        truth_keys = truth_q.astype(np.int64) * stride + truth_t

        print(
            f"\n[{channel_name} → {target} | {country}] "
            f"queries={len(queries):,} targets={len(targets):,} "
            f"true links={len(truth_q):,} "
            f"(target outside partition: {(truth_t < 0).sum():,})"
        )

        channel = build_channel(channel_name, args)

        start = time.perf_counter()
        channel.fit(targets)
        fit_time = time.perf_counter() - start

        del targets

        print(f"  fit: {fit_time:.1f}s")

        if isinstance(channel, CharNgramChannel):
            print(
                f"  index: {channel.index_nbytes() / 2**20:,.0f} MB, "
                f"n-grams dropped (df>{args.max_df:,}): "
                f"{(channel.df > args.max_df).sum():,}"
            )

        if args.estimate_only:
            if isinstance(channel, CharNgramChannel):
                matrix = channel._vectorize(queries[channel.column])
                bounds = channel._chunks(matrix)
                work = channel.last_work
                print(
                    f"  est. postings/query: mean {work.mean():,.0f} "
                    f"p99 {np.percentile(work, 99):,.0f}; "
                    f"total {work.sum():.3g}; chunks {len(bounds):,}; "
                    f"worst-case product per worker "
                    f"~{args.chunk_work * 8 / 2**20:,.0f} MB"
                )
                print(
                    "  queries with no indexed n-gram: "
                    f"{(work == 0).mean():.2%}"
                )
            runtime += fit_time
            continue

        country_counts = {k: np.zeros(len(queries), np.int32) for k in settings}

        start = time.perf_counter()

        for batch in channel.retrieve(queries):
            lo = np.searchsorted(truth_q, batch.q_start, "left")
            hi = np.searchsorted(truth_q, batch.q_end, "left")
            batch_truth = truth_keys[lo:hi]
            batch_truth_q = truth_q[lo:hi]

            cand_keys = batch.q_idx.astype(np.int64) * stride + batch.t_idx

            for k in settings:
                keep = slice(None) if k is None else batch.rank < k
                keys = cand_keys[keep]

                country_counts[k][batch.q_start:batch.q_end] = np.bincount(
                    batch.q_idx[keep] - batch.q_start,
                    minlength=batch.q_end - batch.q_start,
                )

                hit = np.isin(batch_truth, keys)
                stats[k].retrieved += int(hit.sum())
                stats[k].s1_with_hit += len(np.unique(batch_truth_q[hit]))

            if args.output_dir:
                if writer is None:
                    writer = open_writer(args.output_dir, channel, target)

                writer.write_table(
                    pa.table(
                        {
                            "s1_id": query_ids[batch.q_idx],
                            "target_id": target_ids[batch.t_idx],
                            "score": batch.score,
                            "rank": batch.rank,
                        }
                    )
                )

        retrieve_time = time.perf_counter() - start
        runtime += fit_time + retrieve_time

        for k in settings:
            stats[k].counts.append(country_counts[k])

        own, child = peak_rss_mb()
        print(
            f"  retrieve: {retrieve_time:.1f}s; "
            f"peak RSS parent {own:,.0f} MB, largest worker {child:,.0f} MB"
        )

    if writer is not None:
        writer.close()

    if args.estimate_only:
        return []

    rows = []

    for k in settings:
        counts = np.concatenate(stats[k].counts)

        rows.append(
            {
                "channel": channel_name,
                "target": target,
                "setting": stats[k].label,
                "s1": n_queries,
                "pairs": int(counts.sum()),
                "avg/S1": counts.mean(),
                "median/S1": float(np.median(counts)),
                "max/S1": int(counts.max()),
                "S1 ≥1 cand": (counts > 0).mean(),
                "true links": total_truth,
                "retrieved": stats[k].retrieved,
                "recall": stats[k].retrieved / max(total_truth, 1),
                "matched S1 ≥1 hit": stats[k].s1_with_hit / max(n_matched_s1, 1),
                "runtime_s": runtime,
            }
        )

    return rows


def open_writer(
    output_dir: Path,
    channel: CandidateChannel,
    target: str,
) -> pq.ParquetWriter:

    output_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("s1_id", pa.int32()),
            ("target_id", pa.int32()),
            ("score", pa.float32()),
            ("rank", pa.int16()),
        ]
    )

    return pq.ParquetWriter(
        output_dir / f"{channel.name}_{target}.parquet",
        schema,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--channels", nargs="+", default=["exact"], choices=["exact", "char", "address"]
    )
    parser.add_argument(
        "--targets", nargs="+", default=["S2", "S3"], choices=list(TARGET_FILES)
    )
    parser.add_argument(
        "--s1-sample",
        type=int,
        default=None,
        help="Random S1 entities to query (default: all S1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[10, 20, 50, 100]
    )
    parser.add_argument("--ngram", type=int, nargs=2, default=[3, 3])
    parser.add_argument("--max-df", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-work", type=int, default=20_000_000)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Also write candidates (s1_id, target_id, score, rank) as parquet.",
    )
    args = parser.parse_args()

    s1_ids = sample_s1_ids(args.s1_sample, args.seed)

    rows = []

    for channel_name in args.channels:
        for target in args.targets:
            rows.extend(run_channel(channel_name, target, args, s1_ids))

    if not rows:
        return

    report = pd.DataFrame(rows)

    print("\n=== CANDIDATE GENERATION BENCHMARK ===")

    with pd.option_context(
        "display.width", 250,
        "display.max_columns", None,
        "display.float_format", "{:,.4f}".format,
    ):
        print(report.to_string(index=False))

    own, child = peak_rss_mb()
    print(f"\nPeak RSS: parent {own:,.0f} MB, largest worker {child:,.0f} MB")


if __name__ == "__main__":
    main()
