"""
Unit tests for the pure helpers of the false-positive analysis.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.fp_skeleton_analysis import categorize, rank_auc, s1_score_summary


def row(**overrides) -> dict:
    base = {
        "cross_script": False, "name_ratio": 0.3, "translit_name_ratio": 0.3,
        "translit_skeleton_ratio": 0.3, "address_token_set_ratio": 0.5,
        "address_number_set_jaccard": np.nan, "target_name_frequency": 1.0,
        "house_city_match": np.nan,
    }
    base.update(overrides)
    return base


class S1ScoreSummaryTest(unittest.TestCase):

    def test_top_second_margin_and_counts(self):
        out = s1_score_summary(
            np.array([1, 1, 1, 2]),
            np.array([0.2, 0.995, 0.9, 0.7]),
            np.array([0, 1, 0, 0]),
        )

        self.assertEqual(out.loc[1, "candidate_count"], 3)
        self.assertAlmostEqual(out.loc[1, "top_score"], 0.995)
        self.assertAlmostEqual(out.loc[1, "second_score"], 0.9)
        self.assertAlmostEqual(out.loc[1, "score_margin"], 0.095)
        self.assertEqual(out.loc[1, "n_predicted"], 1)

    def test_single_candidate_has_zero_second_score(self):
        out = s1_score_summary(np.array([7]), np.array([0.8]), np.array([0]))

        self.assertEqual(out.loc[7, "second_score"], 0.0)
        self.assertAlmostEqual(out.loc[7, "score_margin"], 0.8)


class CategorizeTest(unittest.TestCase):

    def categories(self, *rows) -> list[str]:
        return categorize(pd.DataFrame(list(rows))).tolist()

    def test_rules(self):
        got = self.categories(
            row(cross_script=True, name_ratio=0.05, translit_skeleton_ratio=0.9),
            row(translit_skeleton_ratio=0.9, name_ratio=0.5, translit_name_ratio=0.5),
            row(name_ratio=1.0, address_token_set_ratio=0.95, address_number_set_jaccard=1.0),
            row(address_token_set_ratio=0.95, address_number_set_jaccard=0.5),
            row(name_ratio=1.0, address_token_set_ratio=0.4, target_name_frequency=40),
            row(name_ratio=1.0, address_token_set_ratio=np.nan, target_name_frequency=1),
            row(address_token_set_ratio=0.95),
            row(),
        )

        self.assertEqual(got, [
            "transliteration collision (cross-script)",
            "skeleton collision (Latin, spelling differs)",
            "near-duplicate record: name & address ≥0.9, numbers agree",
            "address near-duplicate, numbers conflict",
            "same/near name, weak or missing address — common name (freq > 5)",
            "same/near name, weak or missing address — rare name (freq ≤ 5)",
            "same/near address, different name",
            "other",
        ])


class RankAucTest(unittest.TestCase):

    def test_perfect_random_and_ties(self):
        self.assertEqual(rank_auc([3, 4], [1, 2]), 1.0)
        self.assertEqual(rank_auc([1, 2], [3, 4]), 0.0)
        self.assertEqual(rank_auc([1, 1], [1, 1]), 0.5)
        self.assertTrue(np.isnan(rank_auc([], [1])))


if __name__ == "__main__":
    unittest.main()
