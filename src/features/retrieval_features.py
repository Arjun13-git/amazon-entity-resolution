"""
Features derived from candidate-generation provenance.

Ranks and scores are float32 with NaN where the channel did not
retrieve the pair (candidate files store -1 / NaN).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from src.blocking.candidate_pipeline import SOURCE_BITS


def _rank(values: np.ndarray) -> np.ndarray:

    out = values.astype(np.float32)
    out[values < 0] = np.nan

    return out


def retrieval_features(candidates: pa.Table) -> dict[str, np.ndarray]:

    mask = candidates["candidate_source_mask"].to_numpy()

    hits = {
        f"{name}_hit": ((mask & bit) > 0).astype(np.int8)
        for name, bit in SOURCE_BITS.items()
    }

    name_rank = _rank(candidates["name_char_rank"].to_numpy())
    address_rank = _rank(candidates["address_char_rank"].to_numpy())
    translit_rank = _rank(candidates["translit_name_rank"].to_numpy())

    # Best rank across a field's channels; exact / block hits count as rank 0.
    best_name = np.where(
        (hits["exact_name_hit"] == 1) | (hits["name_token_sort_hit"] == 1),
        0.0,
        np.where(hits["translit_name_hit"] == 1, np.minimum(name_rank, translit_rank), name_rank),
    ).astype(np.float32)
    best_address = np.where(
        (hits["house_city_hit"] == 1) | (hits["postal_code_hit"] == 1),
        0.0,
        address_rank,
    ).astype(np.float32)

    # Candidates per S1 (every S1's candidates live in one part).
    s1 = candidates["s1_id"].to_numpy()
    _, inverse, counts = np.unique(s1, return_inverse=True, return_counts=True)

    return {
        **hits,
        "n_channels": sum(hits.values()).astype(np.int8),
        "name_char_rank": name_rank,
        "name_char_score": candidates["name_char_score"].to_numpy().astype(np.float32),
        "address_char_rank": address_rank,
        "address_char_score": candidates["address_char_score"].to_numpy().astype(np.float32),
        "translit_name_rank": translit_rank,
        "translit_name_score": candidates["translit_name_score"].to_numpy().astype(np.float32),
        "best_name_rank": best_name,
        "best_address_rank": best_address,
        "s1_candidate_count": counts[inverse].astype(np.int32),
    }
