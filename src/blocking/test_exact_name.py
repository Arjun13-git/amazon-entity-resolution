from pathlib import Path

from src.data.loader import load_entities
from src.blocking.exact_name import (
    build_exact_name_index,
    retrieve_exact_name_candidates,
)


ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = (
    ROOT.parent
    / "student_resource"
    / "dataset"
    / "train"
)


def main() -> None:
    source1 = load_entities(
        TRAIN_DIR / "train_source1.tsv",
        nrows=10_000,
    )

    source2 = load_entities(
        TRAIN_DIR / "train_source2.tsv",
        nrows=10_000,
    )

    index = build_exact_name_index(source2)

    predictions = retrieve_exact_name_candidates(
        source1,
        index,
    )

    total_candidates = sum(
        len(candidates)
        for candidates in predictions.values()
    )

    matched_s1 = sum(
        bool(candidates)
        for candidates in predictions.values()
    )

    print(f"S1 records: {len(source1):,}")
    print(f"S2 records indexed: {len(source2):,}")
    print(f"S1 with >=1 candidate: {matched_s1:,}")
    print(f"Total candidates: {total_candidates:,}")
    print(
        "Average candidates/S1: "
        f"{total_candidates / len(source1):.4f}"
    )


if __name__ == "__main__":
    main()