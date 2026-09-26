from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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

# Keep this benchmark deliberately small.
S1_SAMPLE = 100_000
TARGET_SAMPLE = 500_000

TOP_K_VALUES = [10, 20, 50, 100]

RANDOM_STATE = 42


def normalize_entities(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add normalized country and business-name columns."""

    df = df.copy()

    df["norm_country"] = df["country"].map(
        normalize_country
    )

    df["norm_name"] = df["business_name"].map(
        normalize_business_name
    )

    return df


def load_ground_truth(
    path: Path,
) -> dict[str, set[str]]:
    """Load S1 -> matched target IDs."""

    df = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    ground_truth: dict[str, set[str]] = {}

    for row in df.itertuples(index=False):
        ground_truth[row.source1_entity_id] = {
            entity_id.strip()
            for entity_id in row.matched_entity_ids.split(",")
            if entity_id.strip()
        }

    return ground_truth


def evaluate_recall(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    target_ids: set[str],
) -> tuple[int, int, float]:
    """
    Evaluate recall over ground-truth links belonging
    to the sampled target population.
    """

    total_true_links = 0
    retrieved_true_links = 0

    for s1_id, actual in ground_truth.items():

        # Only evaluate links whose target exists
        # in the sampled target set.
        sampled_actual = actual & target_ids

        if not sampled_actual:
            continue

        total_true_links += len(sampled_actual)

        predicted = set(
            predictions.get(s1_id, [])
        )

        retrieved_true_links += len(
            predicted & sampled_actual
        )

    recall = (
        retrieved_true_links / total_true_links
        if total_true_links
        else 0.0
    )

    return (
        total_true_links,
        retrieved_true_links,
        recall,
    )


def benchmark_country(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    top_k: int,
) -> None:
    """
    Run country-specific character TF-IDF retrieval.
    """

    predictions: dict[str, list[str]] = {}

    target_ids = set(
        target["entity_id"]
    )

    print(
        f"\n--- Top-{top_k} ---"
    )

    for country, s1_group in source1.groupby(
        "norm_country",
        sort=False,
    ):
        target_group = target[
            target["norm_country"] == country
        ]

        if target_group.empty:
            for entity_id in s1_group["entity_id"]:
                predictions[entity_id] = []
            continue

        # Remove empty names.
        s1_group = s1_group[
            s1_group["norm_name"].str.len() > 0
        ]

        target_group = target_group[
            target_group["norm_name"].str.len() > 0
        ]

        if target_group.empty or s1_group.empty:
            continue

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
            dtype="float32",
        )

        target_matrix = vectorizer.fit_transform(
            target_group["norm_name"]
        )

        n_neighbors = min(
            top_k,
            len(target_group),
        )

        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )

        nn.fit(target_matrix)

        query_matrix = vectorizer.transform(
            s1_group["norm_name"]
        )

        _, indices = nn.kneighbors(
            query_matrix,
            return_distance=True,
        )

        target_entity_ids = (
            target_group["entity_id"]
            .to_numpy()
        )

        source_entity_ids = (
            s1_group["entity_id"]
            .to_numpy()
        )

        for row_idx, source_id in enumerate(
            source_entity_ids
        ):
            predictions[source_id] = [
                target_entity_ids[index]
                for index in indices[row_idx]
            ]

        print(
            f"country={country!r} "
            f"S1={len(s1_group):,} "
            f"target={len(target_group):,}"
        )

    (
        total_true,
        retrieved_true,
        recall,
    ) = evaluate_recall(
        predictions,
        ground_truth,
        target_ids,
    )

    total_candidates = sum(
        len(values)
        for values in predictions.values()
    )

    print(
        f"True links in sampled target: "
        f"{total_true:,}"
    )

    print(
        f"Retrieved true links: "
        f"{retrieved_true:,}"
    )

    print(
        f"Recall@{top_k}: "
        f"{recall:.4%}"
    )

    print(
        f"Candidate pairs: "
        f"{total_candidates:,}"
    )

    print(
        f"Average candidates/S1: "
        f"{total_candidates / len(source1):.2f}"
    )


def main() -> None:
    print("Loading S1...")

    source1 = pd.read_parquet(
        PARQUET_DIR / "train_source1.parquet"
    )

    source1 = normalize_entities(source1)

    source1 = (
        source1
        .sample(
            n=min(S1_SAMPLE, len(source1)),
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    print(
        f"S1 sample: {len(source1):,}"
    )

    print("\nLoading S2...")

    target = pd.read_parquet(
        PARQUET_DIR / "train_source2.parquet"
    )

    target = normalize_entities(target)

    target = (
        target
        .sample(
            n=min(TARGET_SAMPLE, len(target)),
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    print(
        f"S2 sample: {len(target):,}"
    )

    print("\nLoading ground truth...")

    ground_truth = load_ground_truth(
        GROUND_TRUTH
    )

    print(
        f"Ground-truth S1 rows: "
        f"{len(ground_truth):,}"
    )

    for top_k in TOP_K_VALUES:
        benchmark_country(
            source1,
            target,
            ground_truth,
            top_k,
        )


if __name__ == "__main__":
    main()