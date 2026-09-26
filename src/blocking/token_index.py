from __future__ import annotations

from collections import defaultdict

import pandas as pd


def tokenize(text: str) -> list[str]:
    """Simple normalized whitespace tokenization."""
    if not text:
        return []

    return [
        token
        for token in text.split()
        if token
    ]


def build_token_index(
    target: pd.DataFrame,
    *,
    column: str = "norm_name",
    max_frequency: int = 5000,
) -> dict[str, list[str]]:
    """
    Build an inverted token index.

    Tokens occurring in more than max_frequency target
    records are ignored because they are too common.
    """

    token_to_entities: dict[str, list[str]] = defaultdict(list)

    for entity_id, text in zip(
        target["entity_id"],
        target[column],
    ):
        tokens = set(tokenize(text))

        for token in tokens:
            token_to_entities[token].append(
                entity_id
            )

    return {
        token: entity_ids
        for token, entity_ids
        in token_to_entities.items()
        if len(entity_ids) <= max_frequency
    }


def retrieve_token_candidates(
    source1: pd.DataFrame,
    token_index: dict[str, list[str]],
    *,
    max_tokens: int = 3,
) -> dict[str, list[str]]:
    """
    Retrieve candidates using the rarest available
    name tokens.
    """

    predictions: dict[str, list[str]] = {}

    for entity_id, text in zip(
        source1["entity_id"],
        source1["norm_name"],
    ):
        tokens = set(tokenize(text))

        available = [
            token
            for token in tokens
            if token in token_index
        ]

        # Prefer rare tokens.
        available.sort(
            key=lambda token: len(
                token_index[token]
            )
        )

        selected = available[:max_tokens]

        candidates = set()

        for token in selected:
            candidates.update(
                token_index[token]
            )

        predictions[entity_id] = list(
            candidates
        )

    return predictions