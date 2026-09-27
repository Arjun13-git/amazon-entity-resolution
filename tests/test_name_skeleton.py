"""
Unit tests for the transliterated consonant skeleton feature.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.features.text_features import (
    name_skeleton,
    skeleton_features,
    skeleton_from_translit,
)


def skeleton_ratio(left: str, right: str) -> float:
    s1 = np.array([name_skeleton(left)], dtype=object)
    t = np.array([name_skeleton(right)], dtype=object)
    return float(skeleton_features(s1, t)["translit_skeleton_ratio"][0])


class SkeletonTest(unittest.TestCase):

    def test_latin_name(self):
        self.assertEqual(name_skeleton("eastern media"), "strn md")
        self.assertEqual(name_skeleton("green constructions"), "grn cnstrctns")

    def test_devanagari_matches_latin_spelling(self):
        target = "ईस्टर्न मीडिया प्राइवेट लिमिटेड"
        self.assertEqual(name_skeleton(target), "strn md")
        self.assertEqual(skeleton_ratio("eastern media", target), 1.0)

    def test_devanagari_abbreviated_private_limited(self):
        # प्रा लि -> "praa li"; "li" is only dropped after "pra".
        self.assertEqual(name_skeleton("बॉम्बे केयर प्रा लि"), "bnmb kr")
        self.assertEqual(skeleton_from_translit("li ning sports"), "l ng sprts")

    def test_malayalam(self):
        target = "സൂര്യ ഇംപെക്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്"
        self.assertEqual(name_skeleton(target), "sr npks")
        self.assertGreater(skeleton_ratio("surya impex", target), 0.6)

    def test_tamil(self):
        target = "கிரீன் கன்ஸ்ட்ரக்ஷன்ஸ் பிரைவேட் லிமிடெட்"
        self.assertEqual(name_skeleton(target), "krn knstrksns")
        self.assertGreater(skeleton_ratio("green constructions", target), 0.6)

    def test_kannada(self):
        target = "ಹರಿ ಫೈನಾನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"
        self.assertEqual(name_skeleton(target), "hr fns")
        self.assertGreater(skeleton_ratio("hari finance", target), 0.8)

    def test_legal_suffix_removal(self):
        for suffix in (
            "private limited", "pvt ltd", "praaivett limittedd", "praiveett limittedd",
            "piraiveett limittett", "praaibhett limittedd", "praivrrrr limirrrrdd",
        ):
            self.assertEqual(skeleton_from_translit(f"hari finance {suffix}"), "hr fnc", suffix)

    def test_doubled_letters_collapse(self):
        self.assertEqual(skeleton_from_translit("goldd phaainyaans"), "gld fns")
        self.assertEqual(skeleton_from_translit("kiriinnn"), "krn")

    def test_vowels_removed(self):
        self.assertEqual(skeleton_from_translit("aeiouy bcd"), "bcd")

    def test_digraph_normalization(self):
        self.assertEqual(skeleton_from_translit("phal bhavan khan ghat thana dhan shah chand wala"),
                         "fl bvn kn gt tn dn sh cnd vl")  # "shah" -> "sah" -> "sh"

    def test_token_order_and_boundaries_kept(self):
        self.assertEqual(skeleton_from_translit("media eastern"), "md strn")

    def test_empty_and_short_skeletons_are_missing(self):
        for name in ("", "a", "al", "aai", "private limited", "pvt ltd"):
            self.assertEqual(skeleton_from_translit(name), "", name)
        self.assertTrue(math.isnan(skeleton_ratio("al", "al")))
        self.assertTrue(math.isnan(skeleton_ratio("", "eastern media")))

    def test_feature_dtype_and_range(self):
        out = skeleton_features(
            np.array(["strn md", "", "grn"], dtype=object),
            np.array(["strn md", "strn md", "blu"], dtype=object),
        )["translit_skeleton_ratio"]

        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out[0], 1.0)
        self.assertTrue(math.isnan(out[1]))
        self.assertTrue(0.0 <= out[2] < 1.0)


if __name__ == "__main__":
    unittest.main()
