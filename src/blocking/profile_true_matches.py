from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from rapidfuzz.fuzz import ratio

from src.data.loader import load_ground_truth
from src.preprocessing.normalize import (
    normalize_address,
    normalize_business_name,
    normalize_country,
)
from src.preprocessing.transliterate import (
    transliterate_text,
)


ROOT = Path(__file__).resolve().parents[2]

PARQUET_DIR = ROOT / "outputs" / "parquet"

GROUND_TRUTH = (
    ROOT.parent
    / "student_resource"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

SAMPLE_LINKS = 100_000
RANDOM_STATE = 42

ENTITY_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

# Target ID prefix -> parquet file holding those entities.
TARGET_SOURCES = {
    "S2": "train_source2.parquet",
    "S3": "train_source3.parquet",
}

PERCENTILES = [
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
]


def load_true_links() -> tuple[pd.DataFrame, int]:
    """
    Explode ground truth into individual S1 -> target links.

    Returns the non-empty links and the number of S1 records
    with no matched entities.
    """

    gt = load_ground_truth(GROUND_TRUTH)

    empty = gt["matched_entity_ids"].str.strip().eq("")
    n_unmatched = int(empty.sum())

    links = gt[~empty].copy()

    links["target_id"] = (
        links["matched_entity_ids"]
        .str.split(",")
    )

    links = links.explode(
        "target_id",
        ignore_index=True,
    )

    links["target_id"] = links["target_id"].str.strip()

    links = links[links["target_id"] != ""]

    links["target_source"] = links["target_id"].str[:2]

    return (
        links[
            [
                "source1_entity_id",
                "target_id",
                "target_source",
            ]
        ].reset_index(drop=True),
        n_unmatched,
    )


def load_rows(
    parquet_path: Path,
    entity_ids: list[str],
) -> pd.DataFrame:
    """Load only the requested entity rows via a pushed-down filter."""

    dataset = ds.dataset(parquet_path)

    return dataset.to_table(
        columns=ENTITY_COLUMNS,
        filter=ds.field("entity_id").isin(entity_ids),
    ).to_pandas()


def normalize_entities(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add normalized name, address, and country columns."""

    df = df.copy()

    df["norm_name"] = df[
        "business_name"
    ].map(
        normalize_business_name
    )

    df["norm_address"] = df[
        "business_address"
    ].map(
        normalize_address
    )

    df["norm_country"] = df[
        "country"
    ].map(
        normalize_country
    )

    return df


def prefixed_entities(
    parquet_path: Path,
    entity_ids: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Load, normalize, and column-prefix the requested entities."""

    df = normalize_entities(
        load_rows(
            parquet_path,
            entity_ids,
        )
    )

    return df[
        [
            "entity_id",
            "norm_name",
            "norm_address",
            "norm_country",
        ]
    ].add_prefix(f"{prefix}_")


def safe_text(value: object) -> str:
    """Convert missing/non-string values to an empty string."""

    if pd.isna(value):
        return ""

    return str(value)


def token_jaccard(
    left: object,
    right: object,
) -> float:
    """Compute token Jaccard similarity safely."""

    left = safe_text(left)
    right = safe_text(right)

    left_tokens = set(left.split())
    right_tokens = set(right.split())

    if not left_tokens and not right_tokens:
        return 1.0

    if not left_tokens or not right_tokens:
        return 0.0

    return len(
        left_tokens & right_tokens
    ) / len(
        left_tokens | right_tokens
    )


def fuzzy_ratio(
    left: object,
    right: object,
) -> float:
    """Compute RapidFuzz ratio safely."""

    left = safe_text(left)
    right = safe_text(right)

    if not left or not right:
        return 0.0

    return ratio(
        left,
        right,
    ) / 100.0


def pairwise(
    func,
    left: pd.Series,
    right: pd.Series,
) -> list[float]:

    return [
        func(
            a,
            b,
        )
        for a, b in zip(
            left,
            right,
        )
    ]


def compute_signals(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Add all name/address/country signals to joined pairs."""

    for field in (
        "name",
        "address",
        "country",
    ):
        pairs[f"{field}_exact"] = (
            pairs[f"s1_norm_{field}"].fillna("")
            == pairs[f"tgt_norm_{field}"].fillna("")
        )

    for field in (
        "name",
        "address",
    ):
        pairs[f"{field}_ratio"] = pairwise(
            fuzzy_ratio,
            pairs[f"s1_norm_{field}"],
            pairs[f"tgt_norm_{field}"],
        )

        pairs[f"{field}_jaccard"] = pairwise(
            token_jaccard,
            pairs[f"s1_norm_{field}"],
            pairs[f"tgt_norm_{field}"],
        )

    for side in (
        "s1",
        "tgt",
    ):
        pairs[f"{side}_translit_name"] = (
            pairs[f"{side}_norm_name"]
            .fillna("")
            .map(transliterate_text)
        )

    pairs["translit_name_ratio"] = pairwise(
        fuzzy_ratio,
        pairs["s1_translit_name"],
        pairs["tgt_translit_name"],
    )

    pairs["translit_name_jaccard"] = pairwise(
        token_jaccard,
        pairs["s1_translit_name"],
        pairs["tgt_translit_name"],
    )

    return pairs


def describe(
    pairs: pd.DataFrame,
    column: str,
    label: str,
) -> None:

    print(f"\n{label}:")

    print(
        pairs[column].describe(
            percentiles=PERCENTILES,
        )
    )


def report_signals(
    pairs: pd.DataFrame,
    title: str,
) -> None:

    print(f"\n=== {title} ===")

    print(
        f"Pairs: {len(pairs):,}"
    )

    for field in (
        "name",
        "address",
        "country",
    ):
        print(
            f"{field.capitalize()} exact: "
            f"{pairs[f'{field}_exact'].mean():.2%}"
        )

    describe(pairs, "name_ratio", "Name ratio")
    describe(pairs, "address_ratio", "Address ratio")
    describe(pairs, "name_jaccard", "Name token Jaccard")
    describe(pairs, "address_jaccard", "Address token Jaccard")
    describe(
        pairs,
        "translit_name_ratio",
        "Transliterated name ratio",
    )
    describe(
        pairs,
        "translit_name_jaccard",
        "Transliterated name token Jaccard",
    )

    print("\nOriginal vs transliterated name ratio:")

    for label, threshold, op in (
        ("> 0", 0.0, "gt"),
        (">= 0.75", 0.75, "ge"),
        (">= 0.90", 0.90, "ge"),
    ):
        original = getattr(pairs["name_ratio"], op)(threshold).mean()
        translit = getattr(pairs["translit_name_ratio"], op)(threshold).mean()

        print(
            f"  ratio {label:<8} "
            f"original {original:7.2%}   "
            f"transliterated {translit:7.2%}"
        )


def profile_population(
    links: pd.DataFrame,
    source: str,
    sample: int,
    seed: int,
) -> pd.DataFrame:
    """Sample one target population, join its rows, compute signals."""

    population = links[
        links["target_source"] == source
    ]

    sampled = (
        population
        .sample(
            n=min(
                sample,
                len(population),
            ),
            random_state=seed,
        )
        .reset_index(drop=True)
    )

    print(
        f"\n[{source}] sampled links: {len(sampled):,} "
        f"of {len(population):,}"
    )

    s1 = prefixed_entities(
        PARQUET_DIR / "train_source1.parquet",
        sampled["source1_entity_id"].unique().tolist(),
        "s1",
    )

    target = prefixed_entities(
        PARQUET_DIR / TARGET_SOURCES[source],
        sampled["target_id"].unique().tolist(),
        "tgt",
    )

    pairs = sampled.merge(
        s1,
        left_on="source1_entity_id",
        right_on="s1_entity_id",
        how="left",
    ).merge(
        target,
        left_on="target_id",
        right_on="tgt_entity_id",
        how="left",
    )

    print(
        f"[{source}] unjoined S1 rows: "
        f"{pairs['s1_entity_id'].isna().sum():,}; "
        f"unjoined {source} rows: "
        f"{pairs['tgt_entity_id'].isna().sum():,}"
    )

    return compute_signals(pairs)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Profile similarity signals on true S1 -> S2/S3 links."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=SAMPLE_LINKS,
        help="Links sampled per target population (S2 and S3 each).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
    )
    args = parser.parse_args()

    print("Loading ground truth...")

    links, n_unmatched = load_true_links()

    counts = links["target_source"].value_counts()

    print("\n=== GROUND TRUTH COMPOSITION ===")

    print(
        f"S2 links: {counts.get('S2', 0):,}"
    )
    print(
        f"S3 links: {counts.get('S3', 0):,}"
    )
    print(
        f"empty/unmatched S1 records: {n_unmatched:,}"
    )

    unknown = counts.drop(
        labels=list(TARGET_SOURCES),
        errors="ignore",
    )

    if not unknown.empty:
        print(
            f"Unknown target prefixes (ignored): {unknown.to_dict()}"
        )

    results = {}

    for source in TARGET_SOURCES:
        results[source] = profile_population(
            links,
            source,
            args.sample,
            args.seed,
        )

    del links

    for source, pairs in results.items():
        report_signals(
            pairs,
            f"S1 → {source} TRUE MATCH SIGNALS",
        )

    report_signals(
        pd.concat(
            results.values(),
            ignore_index=True,
        ),
        "COMBINED TRUE MATCH SIGNALS",
    )


if __name__ == "__main__":
    main()
