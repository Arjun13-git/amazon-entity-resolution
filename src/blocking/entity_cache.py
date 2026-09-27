"""
Compact, normalized on-disk copies of the entity sources.

Normalizing ~12.5M rows takes minutes, so it is done once and cached as
parquet with integer IDs (``S2-681193310`` -> ``681193310``). Candidate
generation then reads one country partition at a time with a pushed-down
filter, never the whole source.

Build the cache:

    python -m src.blocking.entity_cache
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.loader import dataset_dir, load_ground_truth
from src.preprocessing.normalize import (
    normalize_address,
    normalize_business_name,
    normalize_country,
)


ROOT = Path(__file__).resolve().parents[2]

PARQUET_DIR = ROOT / "outputs" / "parquet"
CACHE_DIR = ROOT / "outputs" / "normalized"

GROUND_TRUTH = dataset_dir() / "train" / "train_ground_truth.tsv"

SOURCES = ("source1", "source2", "source3")

DATASETS = ("train", "test")

# Target source code used in the truth cache.
TARGET_CODES = {"S2": 2, "S3": 3}

BATCH_ROWS = 250_000

SCHEMA = pa.schema(
    [
        ("id", pa.int32()),
        ("country", pa.string()),
        ("norm_name", pa.string()),
        ("norm_address", pa.string()),
    ]
)


def strip_prefix(ids: pd.Series) -> np.ndarray:
    """``S2-681193310`` -> 681193310 (all IDs are < 2**31)."""

    return ids.str[3:].astype("int32").to_numpy()


def _normalize_batch(batch: pa.RecordBatch) -> pa.Table:

    df = batch.to_pandas()

    out = pd.DataFrame(
        {
            "id": strip_prefix(df["entity_id"]),
            "country": df["country"].map(normalize_country),
            "norm_name": df["business_name"].map(normalize_business_name),
            "norm_address": df["business_address"].map(normalize_address),
        }
    )

    return pa.Table.from_pandas(out, schema=SCHEMA, preserve_index=False)


def cache_path(source: str, dataset: str = "train") -> Path:

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    return CACHE_DIR / f"{dataset}_{source}.parquet"


def build_source_cache(source: str, workers: int, dataset: str = "train") -> None:

    src = pq.ParquetFile(PARQUET_DIR / f"{dataset}_{source}.parquet")
    dst = cache_path(source, dataset)
    tmp = dst.with_suffix(".tmp")

    with (
        mp.get_context("fork").Pool(workers) as pool,
        pq.ParquetWriter(tmp, SCHEMA) as writer,
    ):
        for table in pool.imap(
            _normalize_batch,
            src.iter_batches(batch_size=BATCH_ROWS),
        ):
            writer.write_table(table)

    tmp.rename(dst)

    print(f"[cache] {dst.name}: {src.metadata.num_rows:,} rows")


def truth_path() -> Path:

    return CACHE_DIR / "train_truth.parquet"


def build_truth_cache() -> None:
    """Ground truth as (s1_id, target_source, target_id) integer rows."""

    gt = load_ground_truth(GROUND_TRUTH)

    links = (
        gt.assign(target=gt["matched_entity_ids"].str.split(","))
        .explode("target")[["source1_entity_id", "target"]]
    )

    links["target"] = links["target"].str.strip()
    links = links[links["target"] != ""]

    prefix = links["target"].str[:2]
    unknown = ~prefix.isin(list(TARGET_CODES))

    if unknown.any():
        raise ValueError(
            f"Unknown target prefixes: {prefix[unknown].unique()[:10]}"
        )

    table = pa.table(
        {
            "s1_id": strip_prefix(links["source1_entity_id"]),
            "target_source": prefix.map(TARGET_CODES).astype("int8").to_numpy(),
            "target_id": strip_prefix(links["target"]),
        }
    )

    pq.write_table(table, truth_path())

    print(f"[cache] {truth_path().name}: {table.num_rows:,} links")


def load_partition(
    source: str,
    country: str,
    columns: list[str],
    dataset: str = "train",
) -> pd.DataFrame:
    """Load one country's rows of a cached source, sorted by id."""

    table = pq.read_table(
        cache_path(source, dataset),
        columns=["id", *columns],
        filters=[("country", "=", country)],
    )

    return (
        table.to_pandas()
        .sort_values("id", kind="stable")
        .reset_index(drop=True)
    )


def list_countries(source: str, dataset: str = "train") -> list[str]:

    column = pq.read_table(cache_path(source, dataset), columns=["country"])["country"]

    return sorted(column.unique().to_pylist())


def load_truth(target_source: str) -> pd.DataFrame:
    """True (s1_id, target_id) links for one target source (``S2``/``S3``)."""

    return pq.read_table(
        truth_path(),
        columns=["s1_id", "target_id"],
        filters=[("target_source", "=", TARGET_CODES[target_source])],
    ).to_pandas()


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["train"], choices=DATASETS)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for source in SOURCES:
            if args.force or not cache_path(source, dataset).exists():
                build_source_cache(source, args.workers, dataset)

    # Ground truth exists for the training data only.
    if "train" in args.datasets and (args.force or not truth_path().exists()):
        build_truth_cache()


if __name__ == "__main__":
    main()
