"""
Address string-similarity and structured-key features.
"""

from __future__ import annotations

import numpy as np

from src.features.text_features import exact_match, string_similarities


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
