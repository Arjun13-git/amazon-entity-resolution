from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class CharacterTfidfBlocker:
    """
    Character n-gram TF-IDF candidate generator.

    Targets are indexed separately by country.
    """

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (2, 5),
        max_features: int = 300_000,
        top_k: int = 20,
        min_df: int = 2,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.top_k = top_k
        self.min_df = min_df

        self.vectorizers: dict[str, TfidfVectorizer] = {}
        self.matrices = {}
        self.neighbors = {}
        self.entity_ids: dict[str, np.ndarray] = {}

    def fit(
        self,
        target: pd.DataFrame,
    ) -> None:
        """
        Build country-specific TF-IDF indexes.

        Required columns:
            entity_id
            norm_name
            norm_country
        """

        for country, group in target.groupby(
            "norm_country",
            sort=False,
        ):
            texts = group["norm_name"].fillna("").tolist()

            if not any(texts):
                continue

            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_features=self.max_features,
                sublinear_tf=True,
            )

            matrix = vectorizer.fit_transform(texts)

            nn = NearestNeighbors(
                n_neighbors=min(
                    self.top_k,
                    len(group),
                ),
                metric="cosine",
                algorithm="brute",
                n_jobs=-1,
            )

            nn.fit(matrix)

            self.vectorizers[country] = vectorizer
            self.matrices[country] = matrix
            self.neighbors[country] = nn
            self.entity_ids[country] = (
                group["entity_id"]
                .to_numpy()
            )

            print(
                f"[INDEX] country={country!r} "
                f"targets={len(group):,} "
                f"features={matrix.shape[1]:,}"
            )

    def retrieve(
        self,
        source1: pd.DataFrame,
    ) -> dict[str, list[str]]:
        """
        Retrieve top-k target candidates for each S1 entity.
        """

        predictions: dict[str, list[str]] = {}

        for country, group in source1.groupby(
            "norm_country",
            sort=False,
        ):
            if country not in self.vectorizers:
                for entity_id in group["entity_id"]:
                    predictions[entity_id] = []
                continue

            vectorizer = self.vectorizers[country]
            nn = self.neighbors[country]
            target_ids = self.entity_ids[country]

            query_matrix = vectorizer.transform(
                group["norm_name"].fillna("")
            )

            distances, indices = nn.kneighbors(
                query_matrix,
                return_distance=True,
            )

            source_ids = group["entity_id"].to_numpy()

            for row_idx, source_id in enumerate(
                source_ids
            ):
                predictions[source_id] = [
                    target_ids[index]
                    for index in indices[row_idx]
                    if distances[row_idx][
                        list(indices[row_idx]).index(index)
                    ] < 1.0
                ]

        return predictions