"""
Streaming candidate generation (final blocking design).

Channels, per target source (S2, S3) and country partition:

    exact_name     exact normalized business name (all matches)
    name_char      name char-trigram TF-IDF, pruned top-1000 -> full-cosine top-100
    address_char   address char-trigram TF-IDF, pruned top-1000 -> full-cosine top-25
    house_city     exact house-number + city key, blocks <= 1000 targets

Channels are fitted once per (target, country); S1 is then processed in
chunks. Each chunk's candidates are merged on integer (s1, target) keys,
deduplicated, and written as one Parquet file:

    <out>/target=S2/country=us/part-00000.parquet

Nothing larger than one chunk's candidates is ever held in memory.
``manifest.json`` records the config, code version, per-part counts and
peak memory; the city lexicons and (for sampled runs) the S1 id list are
saved next to it so the candidate set can be regenerated exactly.

    python -m src.blocking.candidate_pipeline --out outputs/candidates/sample20k \\
        --s1-sample 20000 --seed 42 --chunk-size 5000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.blocking.address_keys import learn_city_lexicon
from src.blocking.benchmark_candidates import TARGET_FILES, sample_s1_ids
from src.blocking.channels import (
    AddressChannel,
    CharNgramChannel,
    ExactNameChannel,
    HouseCityChannel,
)
from src.blocking.entity_cache import list_countries, load_partition


# Bit per channel in candidate_source_mask.
SOURCE_BITS = {
    "exact_name": 1,
    "name_char": 2,
    "address_char": 4,
    "house_city": 8,
}

TARGET_CODES = {"S2": 2, "S3": 3}

# Rank columns use -1 and score columns NaN when the channel did not
# produce the pair.
SCHEMA = pa.schema(
    [
        ("s1_id", pa.int32()),
        ("target_id", pa.int32()),
        ("target_source", pa.int8()),
        ("country", pa.string()),
        ("candidate_source_mask", pa.uint8()),
        ("exact_name_rank", pa.int16()),
        ("name_char_rank", pa.int16()),
        ("name_char_score", pa.float32()),
        ("address_char_rank", pa.int16()),
        ("address_char_score", pa.float32()),
        ("house_city_hit", pa.bool_()),
    ]
)


@dataclass
class PipelineConfig:
    targets: list[str] = field(default_factory=lambda: ["S2", "S3"])
    s1_sample: int | None = None
    seed: int = 42
    chunk_size: int = 50_000
    max_df: int = 30_000
    rerank_pool: int = 1000
    name_k: int = 100
    address_k: int = 25
    house_city_max_block: int = 1000
    lexicon_min_count: int = 25
    workers: int = 8
    chunk_work: int = 20_000_000


def build_channels(cfg: PipelineConfig, country: str, lexicon: frozenset[str]):

    common = dict(
        max_df=cfg.max_df,
        rerank_pool=cfg.rerank_pool,
        workers=cfg.workers,
        chunk_work=cfg.chunk_work,
    )

    return {
        "exact_name": ExactNameChannel(),
        "name_char": CharNgramChannel(k=cfg.name_k, **common),
        "address_char": AddressChannel(k=cfg.address_k, **common),
        "house_city": HouseCityChannel(
            country=country,
            lexicon=lexicon,
            max_block=cfg.house_city_max_block,
        ),
    }


def channel_candidates(channel, queries: pd.DataFrame, stride: np.int64):
    """(keys, rank, score) of one channel for one S1 chunk."""

    keys, ranks, scores = [], [], []

    for batch in channel.retrieve(queries):
        keys.append(batch.q_idx.astype(np.int64) * stride + batch.t_idx)
        ranks.append(batch.rank)
        scores.append(batch.score)

    if not keys:
        return (
            np.empty(0, np.int64),
            np.empty(0, np.int16),
            np.empty(0, np.float32),
        )

    return np.concatenate(keys), np.concatenate(ranks), np.concatenate(scores)


def merge_chunk(per_channel: dict, stride: np.int64) -> dict[str, np.ndarray]:
    """Deduplicate (s1, target) keys across channels, keeping provenance."""

    all_keys = np.unique(np.concatenate([keys for keys, _, _ in per_channel.values()]))
    n = len(all_keys)

    out = {
        "key": all_keys,
        "mask": np.zeros(n, np.uint8),
        "exact_name_rank": np.full(n, -1, np.int16),
        "name_char_rank": np.full(n, -1, np.int16),
        "name_char_score": np.full(n, np.nan, np.float32),
        "address_char_rank": np.full(n, -1, np.int16),
        "address_char_score": np.full(n, np.nan, np.float32),
    }

    for name, (keys, rank, score) in per_channel.items():
        if len(np.unique(keys)) != len(keys):
            raise AssertionError(f"{name} produced duplicate pairs within a chunk")

        pos = np.searchsorted(all_keys, keys)
        out["mask"][pos] |= SOURCE_BITS[name]

        if name == "exact_name":
            out["exact_name_rank"][pos] = rank
        elif name == "name_char":
            out["name_char_rank"][pos] = rank
            out["name_char_score"][pos] = score
        elif name == "address_char":
            out["address_char_rank"][pos] = rank
            out["address_char_score"][pos] = score

    return out


def code_version() -> dict:

    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return None

    files = sorted(Path(__file__).parent.glob("*.py")) + sorted(
        (Path(__file__).parents[1] / "preprocessing").glob("*.py")
    )
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.read_bytes())

    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": digest.hexdigest(),
    }


def peak_rss_mb() -> dict:

    return {
        "parent": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "largest_child": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
    }


def run(cfg: PipelineConfig, out_dir: Path) -> dict:

    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    s1_ids = sample_s1_ids(cfg.s1_sample, cfg.seed)

    if s1_ids is not None:
        pq.write_table(pa.table({"s1_id": s1_ids.astype(np.int32)}), out_dir / "s1_ids.parquet")

    manifest = {
        "config": asdict(cfg),
        "schema": SCHEMA.to_string(),
        "source_bits": SOURCE_BITS,
        "code": code_version(),
        "lexicons": {},
        "parts": [],
    }

    countries = list_countries("source1")

    # City lexicons are learned from the FULL S1 partition, so they do
    # not depend on the sample or the chunking.
    lexicons = {}
    for country in countries:
        s1_all = load_partition("source1", country, ["norm_address"])
        lexicons[country] = learn_city_lexicon(
            s1_all["norm_address"], country, min_count=cfg.lexicon_min_count
        )
        path = out_dir / f"lexicon_{country}.txt"
        path.write_text("\n".join(sorted(lexicons[country])) + "\n")
        manifest["lexicons"][country] = {
            "file": path.name,
            "size": len(lexicons[country]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        del s1_all

    for target in cfg.targets:
        for country in countries:
            queries_all = load_partition("source1", country, ["norm_name", "norm_address"])
            if s1_ids is not None:
                queries_all = queries_all[
                    np.isin(queries_all["id"].to_numpy(), s1_ids)
                ].reset_index(drop=True)

            targets = load_partition(
                TARGET_FILES[target], country, ["norm_name", "norm_address"]
            )
            target_ids = targets["id"].to_numpy()
            stride = np.int64(len(targets) + 1)

            fit_start = time.perf_counter()
            channels = build_channels(cfg, country, lexicons[country])
            for channel in channels.values():
                channel.fit(targets)
            fit_time = time.perf_counter() - fit_start

            del targets

            print(
                f"[{target} | {country}] S1={len(queries_all):,} "
                f"targets={len(target_ids):,} fit {fit_time:.1f}s",
                flush=True,
            )

            part_dir = out_dir / f"target={target}" / f"country={country}"
            part_dir.mkdir(parents=True)

            for part, start in enumerate(range(0, len(queries_all), cfg.chunk_size)):
                chunk_start = time.perf_counter()
                queries = queries_all.iloc[start:start + cfg.chunk_size].reset_index(drop=True)

                per_channel = {
                    name: channel_candidates(channel, queries, stride)
                    for name, channel in channels.items()
                }
                merged = merge_chunk(per_channel, stride)

                q = (merged["key"] // stride).astype(np.int64)
                t = (merged["key"] % stride).astype(np.int64)
                n = len(q)

                table = pa.table(
                    {
                        "s1_id": queries["id"].to_numpy()[q],
                        "target_id": target_ids[t],
                        "target_source": np.full(n, TARGET_CODES[target], np.int8),
                        "country": pa.array([country] * n, pa.string())
                        if n else pa.array([], pa.string()),
                        "candidate_source_mask": merged["mask"],
                        "exact_name_rank": merged["exact_name_rank"],
                        "name_char_rank": merged["name_char_rank"],
                        "name_char_score": merged["name_char_score"],
                        "address_char_rank": merged["address_char_rank"],
                        "address_char_score": merged["address_char_score"],
                        "house_city_hit": (merged["mask"] & SOURCE_BITS["house_city"]) > 0,
                    },
                    schema=SCHEMA,
                )

                path = part_dir / f"part-{part:05d}.parquet"
                pq.write_table(table, path, compression="zstd")

                elapsed = time.perf_counter() - chunk_start
                manifest["parts"].append(
                    {
                        "file": str(path.relative_to(out_dir)),
                        "target": target,
                        "country": country,
                        "s1_rows": len(queries),
                        "s1_id_min": int(queries["id"].min()),
                        "s1_id_max": int(queries["id"].max()),
                        "candidates": n,
                        "per_channel": {
                            name: int(len(keys)) for name, (keys, _, _) in per_channel.items()
                        },
                        "bytes": path.stat().st_size,
                        "seconds": round(elapsed, 2),
                    }
                )

                print(
                    f"  part {part:05d}: S1 {len(queries):,} -> {n:,} candidates "
                    f"({n / max(len(queries), 1):.1f}/S1) {elapsed:.1f}s",
                    flush=True,
                )

                del per_channel, merged, table

            manifest.setdefault("fit_seconds", {})[f"{target}/{country}"] = round(fit_time, 1)

            del channels, queries_all

    manifest["runtime_seconds"] = round(time.perf_counter() - started, 1)
    manifest["peak_rss_mb"] = peak_rss_mb()
    manifest["total_candidates"] = sum(p["candidates"] for p in manifest["parts"])
    manifest["total_bytes"] = sum(p["bytes"] for p in manifest["parts"])

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return manifest


def open_candidates(out_dir: Path) -> "ds.Dataset":
    """
    Lazy dataset over all candidate parts listed in the manifest.

    Iterate with ``.to_batches(...)`` or filter with e.g.
    ``ds.field("target_source") == 2``; nothing is loaded until read.
    """

    import pyarrow.dataset as ds

    manifest = json.loads((Path(out_dir) / "manifest.json").read_text())
    files = [str(Path(out_dir) / part["file"]) for part in manifest["parts"]]

    return ds.dataset(files, format="parquet", schema=SCHEMA)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=["S2", "S3"])
    parser.add_argument("--s1-sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = PipelineConfig(
        targets=args.targets,
        s1_sample=args.s1_sample,
        seed=args.seed,
        chunk_size=args.chunk_size,
        workers=args.workers,
    )

    manifest = run(cfg, args.out)

    print(
        f"\nDone: {manifest['total_candidates']:,} candidates, "
        f"{manifest['total_bytes'] / 2**20:,.1f} MB, "
        f"{manifest['runtime_seconds']:,.0f}s, peak RSS {manifest['peak_rss_mb']}"
    )


if __name__ == "__main__":
    main()
