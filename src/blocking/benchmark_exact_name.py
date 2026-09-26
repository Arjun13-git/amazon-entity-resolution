from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_country,
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


def normalize_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized country and business-name columns."""
    df = df.copy()

    df["norm_country"] = df["country"].map(
        normalize_country
    )

    df["norm_name"] = df["business_name"].map(
        normalize_business_name
    )

    return df


def build_target_index(path: Path) -> pd.DataFrame:
    """Load and normalize one target source."""

    df = pd.read_parquet(path)

    df = normalize_entities(df)

    return df[
        [
            "entity_id",
            "norm_country",
            "norm_name",
        ]
    ]


def evaluate_source(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    target_name: str,
) -> None:
    """Evaluate exact normalized-name + country blocking."""

    print(f"\n=== Evaluating {target_name} ===")

    # Generate candidates.
    candidates = source1.merge(
        target,
        on=["norm_country", "norm_name"],
        how="inner",
        suffixes=("_s1", "_target"),
    )

    candidate_pairs = len(candidates)

    print(
        f"Candidate pairs: {candidate_pairs:,}"
    )

    # Build retrieved candidate sets.
    retrieved: dict[str, set[str]] = {}

    for row in candidates[
        ["entity_id_s1", "entity_id_target"]
    ].itertuples(index=False):
        s1_id, target_id = row

        retrieved.setdefault(
            s1_id,
            set(),
        ).add(target_id)

    # Count ground-truth links belonging to this target.
    total_true_links = 0
    retrieved_true_links = 0

    evaluated_s1 = 0
    s1_with_candidates = 0

    for s1_id, actual_targets in ground_truth.items():

        target_actual = {
            entity_id
            for entity_id in actual_targets
            if entity_id in target_entity_ids
        }

        if not target_actual:
            continue

        evaluated_s1 += 1

        total_true_links += len(target_actual)

        predicted = retrieved.get(
            s1_id,
            set(),
        )

        if predicted:
            s1_with_candidates += 1

        retrieved_true_links += len(
            predicted & target_actual
        )

    candidate_recall = (
        retrieved_true_links / total_true_links
        if total_true_links
        else 0.0
    )

    coverage = (
        s1_with_candidates / evaluated_s1
        if evaluated_s1
        else 0.0
    )

    print(
        f"Ground-truth links: {total_true_links:,}"
    )

    print(
        f"Retrieved true links: {retrieved_true_links:,}"
    )

    print(
        f"Candidate recall: {candidate_recall:.4%}"
    )

    print(
        f"S1 with >=1 candidate: "
        f"{s1_with_candidates:,} / {evaluated_s1:,} "
        f"({coverage:.4%})"
    )

    print(
        "Average candidates / S1: "
        f"{candidate_pairs / len(source1):.4f}"
    )


def load_ground_truth(
    path: Path,
) -> dict[str, set[str]]:
    """Load ground truth into an S1 -> target-ID set mapping."""

    df = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    ground_truth: dict[str, set[str]] = {}

    for row in df.itertuples(index=False):

        matched_ids = {
            entity_id.strip()
            for entity_id in row.matched_entity_ids.split(",")
            if entity_id.strip()
        }

        ground_truth[row.source1_entity_id] = matched_ids

    return ground_truth


def main() -> None:
    print("Loading training S1...")

    source1 = pd.read_parquet(
        PARQUET_DIR / "train_source1.parquet"
    )

    source1 = normalize_entities(source1)

    print(
        f"S1 rows: {len(source1):,}"
    )

    print("\nLoading ground truth...")

    ground_truth = load_ground_truth(
        GROUND_TRUTH
    )

    print(
        f"Ground-truth S1 rows: "
        f"{len(ground_truth):,}"
    )

    s2 = build_target_index(
        PARQUET_DIR / "train_source2.parquet"
    )

    s3 = build_target_index(
        PARQUET_DIR / "train_source3.parquet"
    )

    global target_entity_ids

    target_entity_ids = set(
        s2["entity_id"]
    )

    evaluate_source(
        source1,
        s2,
        ground_truth,
        "S2",
    )

    target_entity_ids = set(
        s3["entity_id"]
    )

    evaluate_source(
        source1,
        s3,
        ground_truth,
        "S3",
    )


if __name__ == "__main__":
    main()