"""
Unit tests for target-name frequency / rarity features.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.features.text_features import name_frequency, name_rarity_features


class NameFrequencyTest(unittest.TestCase):

    def test_unique_name(self):
        freq = name_frequency(np.array(["acme traders", "blue tech", "zenith"], dtype=object))

        np.testing.assert_array_equal(freq, [1, 1, 1])
        self.assertEqual(freq.dtype, np.float32)

    def test_name_appearing_twice(self):
        freq = name_frequency(np.array(["fko agents", "blue tech", "fko agents"], dtype=object))

        np.testing.assert_array_equal(freq, [2, 1, 2])

    def test_name_appearing_many_times(self):
        names = np.array(["pediatric care"] * 25 + ["osprey"], dtype=object)
        freq = name_frequency(names)

        self.assertTrue((freq[:25] == 25).all())
        self.assertEqual(freq[25], 1)

    def test_empty_name_is_missing_not_a_group(self):
        names = np.array(["", "acme", "", None, "acme"], dtype=object)
        freq = name_frequency(names)

        self.assertTrue(math.isnan(freq[0]))
        self.assertTrue(math.isnan(freq[2]))
        self.assertTrue(math.isnan(freq[3]))
        np.testing.assert_array_equal(freq[[1, 4]], [2, 2])

    def test_country_and_source_separation(self):
        names = np.array(["blue tech", "blue tech", "blue tech", "blue tech"], dtype=object)
        groups = np.array(["S2|us", "S2|us", "S2|india", "S3|us"], dtype=object)
        freq = name_frequency(names, groups)

        np.testing.assert_array_equal(freq, [2, 2, 1, 1])

    def test_group_separator_cannot_merge_names(self):
        # "a" + "b c" must not collide with "a b" + "c".
        names = np.array(["b c", "c"], dtype=object)
        groups = np.array(["a", "a b"], dtype=object)

        np.testing.assert_array_equal(name_frequency(names, groups), [1, 1])


class NameRarityFeaturesTest(unittest.TestCase):

    def test_rarity_is_inverse_frequency_and_nan_for_missing(self):
        out = name_rarity_features(np.array([1, 2, 5, np.nan], dtype=np.float32))

        self.assertEqual(set(out), {"target_name_frequency", "target_name_rarity"})
        np.testing.assert_allclose(out["target_name_rarity"][:3], [1.0, 0.5, 0.2], rtol=1e-6)
        self.assertTrue(math.isnan(out["target_name_rarity"][3]))
        self.assertTrue(math.isnan(out["target_name_frequency"][3]))
        for values in out.values():
            self.assertEqual(values.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
