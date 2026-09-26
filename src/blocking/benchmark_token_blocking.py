from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.blocking.token_index import (
    build_token_index,
    retrieve_token_candidates,
)
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

S1_SAMPLE = 100_000
TARGET_SAMPLE = 500_000

RANDOM_STATE = 42


def normalize_entities(
    df: pd.DataFrame,
) -> pd.DataFrame:

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

    df = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    return {
        row.source1_entity_id: {
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        }
        for row in df.itertuples(index=False)
    }


def evaluate(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    target_ids: set[str],
) -> None:

    total_true = 0
    retrieved_true = 0

    candidate_pairs = 0

    for s1_id, actual in ground_truth.items():

        actual = actual & target_ids

        if not actual:
            continue

        total_true += len(actual)

        predicted = set(
            predictions.get(s1_id, [])
        )

        candidate_pairs += len(predicted)

        retrieved_true += len(
            predicted & actual
        )

    recall = (
        retrieved_true / total_true
        if total_true
        else 0.0
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
        f"Candidate recall: "
        f"{recall:.4%}"
    )

    print(
        f"Candidate pairs: "
        f"{candidate_pairs:,}"
    )

    print(
        f"Average candidates/S1: "
        f"{candidate_pairs / S1_SAMPLE:.2f}"
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
            n=S1_SAMPLE,
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
            n=TARGET_SAMPLE,
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

    target_ids = set(
        target["entity_id"]
    )

    print("\nBuilding token index...")

    token_index = build_token_index(
        target,
        column="norm_name",
        max_frequency=5000,
    )

    print(
        f"Indexed tokens: "
        f"{len(token_index):,}"
    )

    for max_tokens in [1, 2, 3, 5]:

        print(
            f"\n=== Top {max_tokens} rare tokens ==="
        )

        predictions = retrieve_token_candidates(
            source1,
            token_index,
            max_tokens=max_tokens,
        )

        evaluate(
            predictions,
            ground_truth,
            target_ids,
        )


if __name__ == "__main__":
    main()