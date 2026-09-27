from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


# Environment variable that overrides the challenge dataset location.
DATASET_ENV = "ER_DATASET_DIR"


def dataset_dir() -> Path:
    """
    Challenge dataset directory (the one containing ``train/`` and ``test/``).

    ``$ER_DATASET_DIR`` if set; otherwise ``<code root>/../student_resource/dataset``,
    i.e. the dataset next to the repository / ``business_entity_resolution`` folder.
    """

    override = os.environ.get(DATASET_ENV)
    if override:
        return Path(override).expanduser().resolve()

    return Path(__file__).resolve().parents[2].parent / "student_resource" / "dataset"


EXPECTED_ENTITY_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

EXPECTED_GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


def load_entities(
    path: str | Path,
    *,
    usecols: Iterable[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load an entity TSV file.

    Parameters
    ----------
    path:
        Path to the TSV file.

    usecols:
        Optional subset of columns to load.

    nrows:
        Optional number of rows to read. Useful during development
        to avoid loading multi-hundred-MB files unnecessarily.

    Returns
    -------
    pd.DataFrame
        Entity records with string-valued columns.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if nrows is not None and nrows <= 0:
        raise ValueError("nrows must be positive or None")

    columns = (
        list(usecols)
        if usecols is not None
        else EXPECTED_ENTITY_COLUMNS
    )

    unknown_columns = [
        column
        for column in columns
        if column not in EXPECTED_ENTITY_COLUMNS
    ]

    if unknown_columns:
        raise ValueError(
            f"Unknown entity columns requested: {unknown_columns}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=columns,
        dtype="string",
        keep_default_na=False,
        nrows=nrows,
    )

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing expected columns in {path.name}: {missing}"
        )

    return df


def load_ground_truth(
    path: str | Path,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load the training ground-truth mapping.

    The matched_entity_ids column contains comma-separated
    S2/S3 entity IDs.

    Parameters
    ----------
    path:
        Path to train_ground_truth.tsv.

    nrows:
        Optional number of rows to read.

    Returns
    -------
    pd.DataFrame
        Ground-truth mappings.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {path}"
        )

    if nrows is not None and nrows <= 0:
        raise ValueError("nrows must be positive or None")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        nrows=nrows,
    )

    if list(df.columns) != EXPECTED_GROUND_TRUTH_COLUMNS:
        raise ValueError(
            f"Unexpected ground-truth columns in {path.name}: "
            f"{list(df.columns)}"
        )

    return df


def summarize_entities(
    df: pd.DataFrame,
    name: str,
) -> None:
    """
    Print a compact summary of an entity dataframe.
    """
    print(f"\n=== {name} ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    for column in [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]:
        if column in df.columns:
            empty = (
                df[column]
                .str.strip()
                .eq("")
                .sum()
            )

            print(f"{column}: {empty:,} empty")


def summarize_ground_truth(
    df: pd.DataFrame,
    name: str,
) -> None:
    """
    Print a compact summary of the ground-truth dataframe.
    """
    print(f"\n=== {name} ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    if "matched_entity_ids" in df.columns:
        match_counts = (
            df["matched_entity_ids"]
            .str.split(",")
            .str.len()
        )

        print("\nMatches per S1:")
        print(match_counts.describe())


def main() -> None:
    """
    Command-line entry point.

    Usage examples
    --------------
    python -m src.data.loader path/to/file.tsv

    python -m src.data.loader path/to/file.tsv --nrows 1000
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect an Amazon Entity Resolution TSV file."
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to the TSV file.",
    )

    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Number of rows to load.",
    )

    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="Interpret the input file as train_ground_truth.tsv.",
    )

    args = parser.parse_args()

    if args.ground_truth:
        df = load_ground_truth(
            args.path,
            nrows=args.nrows,
        )
        summarize_ground_truth(
            df,
            args.path.name,
        )
    else:
        df = load_entities(
            args.path,
            nrows=args.nrows,
        )
        summarize_entities(
            df,
            args.path.name,
        )


if __name__ == "__main__":
    main()