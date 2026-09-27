"""
Unit tests for the S1-context decision helpers.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.decision.s1_context import context_features, decide_single_stricter, lr_matrix, design


def frame() -> pd.DataFrame:
    # S1 1: three candidates (S2, S3, S3); S1 2: a single candidate.
    return pd.DataFrame({
        "s1_id": [1, 1, 1, 2],
        "target_id": [10, 11, 12, 20],
        "target_source": [2, 3, 3, 2],
        "score": [0.995, 0.999, 0.50, 0.992],
        "label": [1, 1, 0, 0],
        "address_token_set_ratio": [0.9, 1.0, 0.4, 0.8],
        "name_ratio": [0.8, 1.0, 0.3, 0.9],
        "address_number_set_jaccard": [1.0, 1.0, np.nan, 0.5],
        "translit_skeleton_ratio": [0.9, 1.0, 0.2, 0.9],
        "target_name_frequency": [1.0, 3.0, 50.0, 1.0],
        "house_number_match": [1.0, 1.0, 0.0, 0.0],
        "target_address_missing": [0, 0, 0, 0],
    }, index=[5, 6, 7, 8])


class ContextFeaturesTest(unittest.TestCase):

    def test_counts_ranks_and_scores(self):
        ctx = context_features(frame())

        self.assertEqual(ctx.loc[5, "s1_candidate_count"], 3)
        self.assertEqual(ctx.loc[5, "s1_count_ge_0.99"], 2)
        self.assertEqual(ctx.loc[6, "candidate_score_rank"], 1)
        self.assertEqual(ctx.loc[5, "candidate_score_rank"], 2)
        self.assertEqual(ctx.loc[6, "is_top"], 1)
        self.assertAlmostEqual(ctx.loc[5, "s1_top_score"], 0.999)
        self.assertAlmostEqual(ctx.loc[5, "s1_second_score"], 0.995)
        self.assertAlmostEqual(ctx.loc[6, "s1_score_margin"], 0.004)
        self.assertAlmostEqual(ctx.loc[7, "candidate_score_minus_top"], 0.50 - 0.999)

    def test_single_candidate_s1(self):
        ctx = context_features(frame())

        self.assertEqual(ctx.loc[8, "s1_candidate_count"], 1)
        self.assertEqual(ctx.loc[8, "s1_second_score"], 0.0)
        self.assertEqual(ctx.loc[8, "other_candidates_high_count"], 1 - 1)  # 0.992 >= 0.99 is itself

    def test_source_context(self):
        ctx = context_features(frame())

        self.assertEqual(ctx.loc[5, "s1_high_s2_count"], 1)
        self.assertEqual(ctx.loc[5, "s1_high_s3_count"], 1)
        self.assertEqual(ctx.loc[5, "other_source_high_count"], 1)   # S2 row sees one high S3
        self.assertEqual(ctx.loc[5, "top_is_s3"], 1)

    def test_diff_to_top_and_no_label_use(self):
        f = frame()
        ctx = context_features(f)
        self.assertAlmostEqual(ctx.loc[5, "diff_top_address_token_set_ratio"], 0.9 - 1.0)
        self.assertEqual(ctx.loc[6, "diff_top_name_ratio"], 0.0)

        f2 = f.assign(label=1 - f["label"])
        pd.testing.assert_frame_equal(ctx, context_features(f2))

    def test_row_alignment_is_preserved(self):
        f = frame().sample(frac=1.0, random_state=3)
        pd.testing.assert_frame_equal(context_features(f).sort_index(), context_features(frame()).sort_index())


class DecisionRuleTest(unittest.TestCase):

    def test_single_stricter(self):
        f = frame()
        base = decide_single_stricter(f, 0.99, 0.99)
        strict = decide_single_stricter(f, 0.99, 0.999)

        np.testing.assert_array_equal(base, [True, True, False, True])
        # S1 2 has only one candidate >= 0.99 and it is below 0.999 -> dropped;
        # S1 1 has two -> unaffected.
        np.testing.assert_array_equal(strict, [True, True, False, False])

    def test_contradiction_only(self):
        f = frame().assign(house_number_match=[1.0, 1.0, 0.0, 1.0], address_number_set_jaccard=[1.0, 1.0, np.nan, 1.0])
        np.testing.assert_array_equal(decide_single_stricter(f, 0.99, 0.999, contradiction_only=True), [True, True, False, True])


class DesignMatrixTest(unittest.TestCase):

    def test_logistic_matrix_width_is_data_independent(self):
        f = frame()
        X = design(f, context_features(f))
        X_no_nan = X.fillna(0.5)

        self.assertEqual(lr_matrix(X).shape, lr_matrix(X_no_nan).shape)
        self.assertFalse(np.isnan(lr_matrix(X)).any())


if __name__ == "__main__":
    unittest.main()
