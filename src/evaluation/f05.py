from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def fbeta(
    precision: float,
    recall: float,
    beta: float = 0.5,
) -> float:
    """
    Compute F-beta from precision and recall.
    """
    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_squared = beta ** 2

    denominator = (
        beta_squared * precision + recall
    )

    if denominator == 0.0:
        return 0.0

    return (
        (1.0 + beta_squared)
        * precision
        * recall
        / denominator
    )


def set_f05(
    predicted: Iterable[str],
    actual: Iterable[str],
) -> float:
    """
    Compute F0.5 for one S1 entity using set predictions.
    """
    predicted_set = set(predicted)
    actual_set = set(actual)

    if not predicted_set and not actual_set:
        return 1.0

    if not predicted_set:
        return 0.0

    true_positive = len(
        predicted_set & actual_set
    )

    precision = (
        true_positive / len(predicted_set)
    )

    recall = (
        true_positive / len(actual_set)
        if actual_set
        else 0.0
    )

    return fbeta(
        precision,
        recall,
        beta=0.5,
    )


def macro_f05(
    predictions: dict[str, Iterable[str]],
    ground_truth: dict[str, Iterable[str]],
) -> float:
    """
    Compute macro F0.5 across S1 entities.
    """
    if not ground_truth:
        raise ValueError("ground_truth is empty")

    scores = []

    for source1_id, actual in ground_truth.items():
        predicted = predictions.get(
            source1_id,
            [],
        )

        scores.append(
            set_f05(
                predicted,
                actual,
            )
        )

    return sum(scores) / len(scores)

def f05_from_counts(
    true_positive: np.ndarray,
    n_predicted: np.ndarray,
    n_actual: np.ndarray,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Vectorized per-S1 F-beta from match counts, with the same rules as
    ``set_f05``:

    - no predictions and no true matches  -> 1.0
    - no predictions but true matches      -> 0.0
    - predictions but no true matches      -> 0.0
    - otherwise ``fbeta(precision, recall)``
    """
    tp = np.asarray(true_positive, dtype=np.float64)
    pred = np.asarray(n_predicted, dtype=np.float64)
    actual = np.asarray(n_actual, dtype=np.float64)

    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)

    b2 = beta ** 2
    denominator = b2 * precision + recall
    score = np.divide(
        (1.0 + b2) * precision * recall,
        denominator,
        out=np.zeros_like(tp),
        where=denominator > 0,
    )

    return np.where((pred == 0) & (actual == 0), 1.0, score)
