"""
Pairwise string-similarity features.

All scores are float32 in [0, 1]. When either side is empty the pair gets
NaN (missingness is reported separately), so "unknown" is never confused
with "dissimilar".
"""

from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist


FUZZ_SCORERS = {
    "ratio": fuzz.ratio,
    "partial_ratio": fuzz.partial_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
}


def token_jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Jaccard similarity of whitespace token sets."""

    out = np.empty(len(left), dtype=np.float32)

    for i, (a, b) in enumerate(zip(left, right)):
        ta = set(a.split())
        tb = set(b.split())
        union = len(ta | tb)
        out[i] = len(ta & tb) / union if union else np.nan

    return out


def string_similarities(
    left: np.ndarray,
    right: np.ndarray,
    prefix: str,
    *,
    scorers: tuple[str, ...] = tuple(FUZZ_SCORERS),
    jaccard: bool = True,
    workers: int = -1,
) -> dict[str, np.ndarray]:
    """
    Element-wise similarities between two aligned object arrays of
    normalized strings. Returns ``{f"{prefix}_{name}": float32 array}``.
    """

    missing = (left == "") | (right == "")
    features = {}

    for name in scorers:
        scores = cpdist(
            left,
            right,
            scorer=FUZZ_SCORERS[name],
            dtype=np.float32,
            workers=workers,
        ) / np.float32(100)
        scores[missing] = np.nan
        features[f"{prefix}_{name}"] = scores.astype(np.float32)

    if jaccard:
        scores = token_jaccard(left, right)
        scores[missing] = np.nan
        features[f"{prefix}_token_jaccard"] = scores

    return features


def exact_match(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """1 if equal and non-empty, 0 if different, NaN if either is empty."""

    out = (left == right).astype(np.float32)
    out[(left == "") | (right == "")] = np.nan

    return out


def name_features(
    s1_name: np.ndarray,
    t_name: np.ndarray,
    s1_translit: np.ndarray,
    t_translit: np.ndarray,
) -> dict[str, np.ndarray]:

    features = {"name_exact": exact_match(s1_name, t_name)}
    features.update(string_similarities(s1_name, t_name, "name"))

    translit = string_similarities(
        s1_translit,
        t_translit,
        "translit_name",
        scorers=("ratio",),
    )
    features.update(translit)

    return features
