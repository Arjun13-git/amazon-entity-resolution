from __future__ import annotations

import pandas as pd


def make_s1_validation_split(
    source1_ids: pd.Series,
    *,
    validation_fraction: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Create a deterministic S1-level train/validation split.

    Each S1 entity is assigned entirely to either the training
    or validation partition.

    Parameters
    ----------
    source1_ids:
        Series containing unique S1 entity IDs.

    validation_fraction:
        Fraction of S1 entities assigned to validation.

    random_state:
        Random seed for reproducibility.

    Returns
    -------
    train_ids, validation_ids
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    ids = (
        source1_ids
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if ids.empty:
        raise ValueError("source1_ids is empty")

    shuffled = ids.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    validation_size = max(
        1,
        int(len(shuffled) * validation_fraction),
    )

    validation_ids = shuffled.iloc[:validation_size]
    train_ids = shuffled.iloc[validation_size:]

    return train_ids, validation_ids