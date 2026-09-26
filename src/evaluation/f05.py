from __future__ import annotations

from collections.abc import Iterable


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