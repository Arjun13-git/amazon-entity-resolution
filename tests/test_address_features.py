"""
Unit tests for the numeric address features.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.features.address_features import (
    NUMERIC_FEATURES,
    extract_numbers,
    numeric_features,
)


def features_for(left: str, right: str) -> dict[str, float]:
    """Numeric features for one address pair, as plain floats."""

    s1 = np.empty(1, dtype=object)
    t = np.empty(1, dtype=object)
    s1[0] = extract_numbers(left)
    t[0] = extract_numbers(right)

    out = numeric_features(
        s1,
        t,
        np.array([not left]),
        np.array([not right]),
    )

    return {name: float(values[0]) for name, values in out.items()}


class ExtractNumbersTest(unittest.TestCase):

    def test_multiple_components_in_order(self):
        self.assertEqual(
            extract_numbers("survey no 5 2 to 5 23 nac campus"),
            ("5", "2", "5", "23"),
        )

    def test_digit_runs_inside_tokens_and_leading_zeros(self):
        self.assertEqual(extract_numbers("64 1016a flat 010 b2c"), ("64", "1016", "10", "2"))
        self.assertEqual(extract_numbers("unit 000"), ("0",))

    def test_no_numbers_and_empty(self):
        self.assertEqual(extract_numbers("main street springfield"), ())
        self.assertEqual(extract_numbers(""), ())
        self.assertEqual(extract_numbers(None), ())


class NumericFeaturesTest(unittest.TestCase):

    def test_returns_all_features_as_float32(self):
        s1 = np.empty(2, dtype=object)
        t = np.empty(2, dtype=object)
        s1[:] = [("1",), ()]
        t[:] = [("1",), ("2",)]
        out = numeric_features(s1, t, np.zeros(2, bool), np.zeros(2, bool))

        self.assertEqual(tuple(out), NUMERIC_FEATURES)
        for values in out.values():
            self.assertEqual(values.dtype, np.float32)
            self.assertEqual(len(values), 2)

    def test_exact_same_numeric_sequence(self):
        f = features_for("15000 15200 plank road baker la", "15000 15200 plank rd baker la")

        self.assertEqual(f["address_number_set_jaccard"], 1.0)
        self.assertEqual(f["address_number_sequence_similarity"], 1.0)
        self.assertEqual(f["address_numeric_token_count_diff"], 0.0)
        self.assertEqual(f["secondary_number_match"], 1.0)

    def test_same_first_number_different_second(self):
        f = features_for("15000 15200 plank road", "15000 15209 plank rd")

        self.assertAlmostEqual(f["address_number_set_jaccard"], 1 / 3, places=6)
        self.assertAlmostEqual(f["address_number_sequence_similarity"], 0.5, places=6)
        self.assertEqual(f["address_numeric_token_count_diff"], 0.0)
        self.assertEqual(f["secondary_number_match"], 0.0)

    def test_different_first_number(self):
        f = features_for("130 26 springfield boulevard", "131 26 springfield boulevard")

        self.assertAlmostEqual(f["address_number_set_jaccard"], 1 / 3, places=6)
        self.assertAlmostEqual(f["address_number_sequence_similarity"], 0.5, places=6)
        self.assertEqual(f["secondary_number_match"], 1.0)

    def test_multiple_components_near_duplicate(self):
        # Diagnostics example: "5 15 to 5 23" vs "5 2 to 5 23".
        f = features_for("survey no 5 2 to 5 23", "survey no 5 15 to 5 23")

        self.assertAlmostEqual(f["address_number_set_jaccard"], 2 / 4, places=6)
        self.assertAlmostEqual(f["address_number_sequence_similarity"], 0.75, places=6)
        self.assertEqual(f["address_numeric_token_count_diff"], 0.0)
        self.assertEqual(f["secondary_number_match"], 0.0)

    def test_extra_numeric_component(self):
        f = features_for("130 26 springfield gardens", "130 26 springfield gardens 11413")

        self.assertAlmostEqual(f["address_number_set_jaccard"], 2 / 3, places=6)
        self.assertEqual(f["address_numeric_token_count_diff"], 1.0)
        self.assertEqual(f["secondary_number_match"], 0.0)

    def test_one_address_without_numbers(self):
        f = features_for("main street springfield", "12 main street springfield")

        self.assertTrue(math.isnan(f["address_number_set_jaccard"]))
        self.assertTrue(math.isnan(f["address_number_sequence_similarity"]))
        self.assertTrue(math.isnan(f["secondary_number_match"]))
        self.assertEqual(f["address_numeric_token_count_diff"], 1.0)

    def test_secondary_unavailable_on_one_side(self):
        f = features_for("12 main street", "12 34 main street")

        self.assertTrue(math.isnan(f["secondary_number_match"]))
        self.assertAlmostEqual(f["address_number_set_jaccard"], 0.5, places=6)

    def test_empty_or_malformed_addresses(self):
        for left, right in (("", "12 main street"), ("12 main street", ""), ("", "")):
            f = features_for(left, right)
            for name in NUMERIC_FEATURES:
                self.assertTrue(math.isnan(f[name]), (left, right, name))

        # Punctuation-only / garbage text is non-empty but has no numbers.
        f = features_for("   ", "## -- ##")
        self.assertEqual(f["address_numeric_token_count_diff"], 0.0)
        self.assertTrue(math.isnan(f["address_number_set_jaccard"]))

    def test_symmetric(self):
        a, b = "5 2 to 5 23 hyderabad", "5 15 to 5 23 hyderabad 500081"
        fa, fb = features_for(a, b), features_for(b, a)

        for name in NUMERIC_FEATURES:
            self.assertEqual(fa[name], fb[name], name)


if __name__ == "__main__":
    unittest.main()
