"""
Address string-similarity and structured-key features.
"""

from __future__ import annotations

import re

import numpy as np
from rapidfuzz.distance import Indel

from src.features.text_features import exact_match, string_similarities


NUMBER_RUN = re.compile(r"\d+")

NUMERIC_FEATURES = (
    "address_number_set_jaccard",
    "address_number_sequence_similarity",
    "address_numeric_token_count_diff",
    "secondary_number_match",
)


def address_features(s1_address: np.ndarray, t_address: np.ndarray) -> dict[str, np.ndarray]:

    return string_similarities(s1_address, t_address, "address")


def structured_features(
    s1_keys: dict[str, np.ndarray],
    t_keys: dict[str, np.ndarray],
    country_match: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Exact agreement of extracted house number, city and house|city keys
    (1 / 0, NaN when either side has no key), plus country agreement.
    """

    return {
        "house_number_match": exact_match(s1_keys["house"], t_keys["house"]),
        "city_match": exact_match(s1_keys["city"], t_keys["city"]),
        "house_city_match": exact_match(s1_keys["house_city"], t_keys["house_city"]),
        "country_match": country_match.astype(np.float32),
    }


def extract_numbers(address: str | None) -> tuple[str, ...]:
    """
    All numeric components of a normalized address, in order.

    Each maximal digit run is one component, with leading zeros removed
    ("0645" -> "645", "000" -> "0"), so "64/1016a" -> ("64", "1016") and
    "flat 010" matches "flat 10".
    """

    if not address:
        return ()

    return tuple(run.lstrip("0") or "0" for run in NUMBER_RUN.findall(address))


def numeric_features(
    s1_numbers: np.ndarray,
    t_numbers: np.ndarray,
    s1_address_empty: np.ndarray,
    t_address_empty: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Agreement of ALL numeric address components (``house_number_match``
    only compares the first one).

    Inputs are aligned object arrays of ``extract_numbers`` tuples, computed
    once per entity by the caller.

    - address_number_set_jaccard: Jaccard of the number sets; NaN if either
      address has no numbers.
    - address_number_sequence_similarity: normalized Indel similarity of the
      ordered number sequences (1 - edits / total length); NaN if either
      has no numbers.
    - address_numeric_token_count_diff: |#numbers(S1) - #numbers(target)|;
      NaN only if either address is empty.
    - secondary_number_match: 1 if the numbers after the first one are
      identical (same order), 0 otherwise; NaN unless both addresses have
      at least two numbers.
    """

    n = len(s1_numbers)
    jaccard = np.full(n, np.nan, dtype=np.float32)
    sequence = np.full(n, np.nan, dtype=np.float32)
    count_diff = np.full(n, np.nan, dtype=np.float32)
    secondary = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        a = s1_numbers[i]
        b = t_numbers[i]

        if not (s1_address_empty[i] or t_address_empty[i]):
            count_diff[i] = abs(len(a) - len(b))

        if not a or not b:
            continue

        sa, sb = set(a), set(b)
        jaccard[i] = len(sa & sb) / len(sa | sb)
        sequence[i] = Indel.normalized_similarity(a, b)

        if len(a) > 1 and len(b) > 1:
            secondary[i] = float(a[1:] == b[1:])

    return {
        "address_number_set_jaccard": jaccard,
        "address_number_sequence_similarity": sequence,
        "address_numeric_token_count_diff": count_diff,
        "secondary_number_match": secondary,
    }
