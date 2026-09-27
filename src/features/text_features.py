"""
Pairwise string-similarity features.

All scores are float32 in [0, 1]. When either side is empty the pair gets
NaN (missingness is reported separately), so "unknown" is never confused
with "dissimilar".
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from src.preprocessing.transliterate import transliterate_text


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


def name_frequency(names: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """
    For each entity, how many entities share its normalized name.

    ``names`` must cover the whole population (e.g. one target source's
    country partition), not just the candidates. When ``groups`` is given
    (e.g. "S2|us"), names are only counted within the same group.
    Empty names get NaN: they are not one shared business name.

    Returns float32 (NaN for empty names).
    """

    names = pd.Series(names, dtype=object).fillna("")
    empty = (names == "").to_numpy()

    if groups is not None:
        names = pd.Series(groups, dtype=object).astype(str) + "\x00" + names

    codes, _ = pd.factorize(names.to_numpy())
    counts = np.bincount(codes)

    frequency = counts[codes].astype(np.float32)
    frequency[empty] = np.nan

    return frequency


def name_rarity_features(frequency: np.ndarray) -> dict[str, np.ndarray]:
    """
    Target-name frequency features from a per-pair ``name_frequency`` array.

    ``target_name_is_unique`` is deliberately not emitted: it equals
    ``target_name_frequency == 1``, which a tree model already expresses
    with a single split.
    """

    frequency = frequency.astype(np.float32)

    return {
        "target_name_frequency": frequency,
        "target_name_rarity": (np.float32(1) / frequency).astype(np.float32),
    }


# ----------------------------------------------------------------------
# Transliterated consonant skeleton (cross-script name matching)
# ----------------------------------------------------------------------

# Legal-suffix words as they appear in Latin names and in unidecode output
# of Indian scripts: private / praaivett / praiveett / piraiveett /
# praaibhett, limited / limittedd / limittett / limirrrrdd, pvt, ltd.
SKELETON_SUFFIX = re.compile(
    r"^(?:p+i*r+a*i+(?:v|bh?)[a-z]*|l+i+m+i+[tr]+[a-z]*|pvt|ltd)$"
)

# Abbreviated "pvt ltd" (प्रा लि -> "praa li"): "li" is only dropped right
# after one of these.
SKELETON_SHORT_PRIVATE = re.compile(r"^p+r+a+$")

SKELETON_DIGRAPHS = (
    ("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"),
    ("dh", "d"), ("sh", "s"), ("ch", "c"), ("w", "v"),
)

SKELETON_VOWELS = re.compile(r"[aeiouy]")
SKELETON_REPEATS = re.compile(r"(.)\1+")

# Skeletons with fewer letters than this carry no reliable evidence.
SKELETON_MIN_LETTERS = 2


def skeleton_from_translit(translit_name: str) -> str:
    """
    Consonant skeleton of an already transliterated (lowercase Latin) name.

    Legal-suffix words are dropped, common phonetic digraphs merged,
    vowels removed and repeated letters collapsed, per token; token order
    and boundaries are kept. Returns "" when fewer than
    ``SKELETON_MIN_LETTERS`` letters remain.
    """

    tokens = (translit_name or "").lower().split()
    kept = []

    for i, token in enumerate(tokens):
        if SKELETON_SUFFIX.match(token) or SKELETON_SHORT_PRIVATE.match(token):
            continue
        if token == "li" and i > 0 and SKELETON_SHORT_PRIVATE.match(tokens[i - 1]):
            continue

        for digraph, replacement in SKELETON_DIGRAPHS:
            token = token.replace(digraph, replacement)
        token = SKELETON_REPEATS.sub(r"\1", SKELETON_VOWELS.sub("", token))

        if token:
            kept.append(token)

    skeleton = " ".join(kept)

    if sum(ch.isalpha() for ch in skeleton) < SKELETON_MIN_LETTERS:
        return ""

    return skeleton


def name_skeleton(norm_name: str) -> str:
    """Skeleton of a normalized name: transliterate, then skeletonize."""

    return skeleton_from_translit(transliterate_text(norm_name))


def skeleton_features(s1_skeleton: np.ndarray, t_skeleton: np.ndarray) -> dict[str, np.ndarray]:
    """``translit_skeleton_ratio``: fuzz.ratio of the skeletons in [0, 1]; NaN if either is empty."""

    scores = cpdist(
        s1_skeleton, t_skeleton, scorer=fuzz.ratio, dtype=np.float32, workers=-1
    ) / np.float32(100)
    scores[(s1_skeleton == "") | (t_skeleton == "")] = np.nan

    return {"translit_skeleton_ratio": scores.astype(np.float32)}
