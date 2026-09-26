from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_country,
)


def build_exact_name_index(
    df: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
    """
    Build an exact normalized-name index.

    Key:
        (country, normalized_business_name)

    Value:
        list of entity IDs sharing that key.
    """
    index: dict[tuple[str, str], list[str]] = defaultdict(list)

    for entity_id, name, country in zip(
        df["entity_id"],
        df["business_name"],
        df["country"],
    ):
        normalized_name = normalize_business_name(name)
        normalized_country = normalize_country(country)

        if not normalized_name:
            continue

        index[
            (normalized_country, normalized_name)
        ].append(entity_id)

    return dict(index)


def retrieve_exact_name_candidates(
    source1: pd.DataFrame,
    target_index: dict[tuple[str, str], list[str]],
) -> dict[str, list[str]]:
    """
    Retrieve candidates for each S1 entity using
    exact normalized business name + country.
    """
    predictions: dict[str, list[str]] = {}

    for entity_id, name, country in zip(
        source1["entity_id"],
        source1["business_name"],
        source1["country"],
    ):
        normalized_name = normalize_business_name(name)
        normalized_country = normalize_country(country)

        if not normalized_name:
            predictions[entity_id] = []
            continue

        predictions[entity_id] = target_index.get(
            (normalized_country, normalized_name),
            [],
        )

    return predictions