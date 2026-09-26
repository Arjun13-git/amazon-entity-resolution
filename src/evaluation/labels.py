"""
Candidate labeling against train ground truth.

Truth is kept per target source: an S2 candidate is only ever looked up
in the S2 truth table and an S3 candidate only in the S3 table. S1
records with empty ``matched_entity_ids`` have no truth rows, so all of
their candidates are negatives.
"""

from __future__ import annotations

import numpy as np

from src.blocking.entity_cache import GROUND_TRUTH, TARGET_CODES, load_truth
from src.data.loader import load_ground_truth


def pair_keys(s1_id: np.ndarray, target_id: np.ndarray) -> np.ndarray:

    return (s1_id.astype(np.int64) << 32) | target_id.astype(np.int64)


class TruthIndex:
    """Sorted (s1_id, target_id) keys for one target source."""

    def __init__(self, target_source: str, s1_ids: np.ndarray | None = None):
        if target_source not in TARGET_CODES:
            raise ValueError(f"Unknown target source: {target_source}")

        truth = load_truth(target_source)

        if s1_ids is not None:
            truth = truth[np.isin(truth["s1_id"].to_numpy(), s1_ids)]

        self.target_source = target_source
        self.code = TARGET_CODES[target_source]
        self.keys = np.sort(
            pair_keys(truth["s1_id"].to_numpy(), truth["target_id"].to_numpy())
        )

    def label(
        self,
        s1_id: np.ndarray,
        target_id: np.ndarray,
        target_source: np.ndarray,
    ) -> np.ndarray:

        if (target_source != self.code).any():
            raise ValueError(
                f"{self.target_source} truth applied to candidates of another source"
            )

        return np.isin(pair_keys(s1_id, target_id), self.keys).astype(np.int8)


def raw_truth_strings(s1_ids: np.ndarray | None = None) -> set[str]:
    """
    Independent re-derivation of truth links from the raw TSV, as
    "S1-<id>|S2-<id>" strings (no integer cache involved). Used to verify
    labels.
    """

    gt = load_ground_truth(GROUND_TRUTH)

    if s1_ids is not None:
        wanted = {f"S1-{i}" for i in s1_ids}
        gt = gt[gt["source1_entity_id"].isin(wanted)]

    links = set()

    for s1, matched in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        for target in matched.split(","):
            target = target.strip()
            if target:
                links.add(f"{s1}|{target}")

    return links
