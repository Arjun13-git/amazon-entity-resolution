"""
Country generalization of structured address keys.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import pandas as pd

from src.blocking.address_keys import (
    extract_keys,
    is_supported,
    learn_city_lexicon,
    state_phrases,
)


class UnsupportedCountryTest(unittest.TestCase):

    def test_france_is_unsupported_and_does_not_raise(self):
        self.assertFalse(is_supported("france"))
        self.assertEqual(state_phrases("france"), [])
        self.assertEqual(learn_city_lexicon(pd.Series(["25 mail pablo picasso nantes pays de la loire"] * 50), "france"), frozenset())

    def test_france_keys(self):
        keys = extract_keys(
            pd.Series(["25 mail pablo picasso nantes pays de la loire", "av willy brandt lille", "", None]),
            "france",
            frozenset(),
        )

        self.assertEqual(keys["house"].tolist(), ["25", "", "", ""])
        self.assertEqual(set(keys["city"]), {""})
        self.assertEqual(set(keys["postal"]), {""})
        self.assertEqual(set(keys["house_city"]), {""})

    def test_unsupported_country_ignores_any_lexicon(self):
        # Even a lexicon containing the city must not create keys for an
        # unsupported country.
        keys = extract_keys(pd.Series(["40 rue bonne nouvelle tourcoing 59200"]), "france", frozenset({"tourcoing"}))

        self.assertEqual(keys.loc[0, "city"], "")
        self.assertEqual(keys.loc[0, "house_city"], "")
        self.assertEqual(keys.loc[0, "postal"], "")

    def test_any_new_country_behaves_like_france(self):
        keys = extract_keys(pd.Series(["12 some street somewhere"]), "atlantis", frozenset({"somewhere"}))
        self.assertEqual(keys.loc[0].tolist(), ["12", "", "", ""])


class SupportedCountriesUnchangedTest(unittest.TestCase):

    def test_us(self):
        keys = extract_keys(
            pd.Series(["3501 main street seattle wa 98101", "lubbock tx 2021 15th street", "8506 lakemont drive dallas texas"]),
            "us",
            frozenset({"seattle", "lubbock", "dallas"}),
        )
        self.assertEqual(keys["house"].tolist(), ["3501", "2021", "8506"])
        self.assertEqual(keys["postal"].tolist(), ["98101", "", ""])
        self.assertEqual(keys["city"].tolist(), ["seattle", "lubbock", "dallas"])
        self.assertEqual(keys["house_city"].tolist(), ["3501|seattle", "2021|lubbock", "8506|dallas"])

    def test_india(self):
        keys = extract_keys(
            pd.Series(["c 34 jagdamba nagar jaipur rajasthan 302001", "h no 5 13 bangalore ka"]),
            "india",
            frozenset({"jaipur", "bangalore"}),
        )
        self.assertEqual(keys["house"].tolist(), ["34", "5"])
        self.assertEqual(keys["postal"].tolist(), ["302001", ""])
        self.assertEqual(keys["city"].tolist(), ["jaipur", "bangalore"])
        self.assertEqual(keys["house_city"].tolist(), ["34|jaipur", "5|bangalore"])

    def test_us_lexicon_learning_still_works(self):
        addresses = pd.Series(["1 main st springfield il"] * 30 + ["2 oak rd portland or"] * 30)
        self.assertEqual(learn_city_lexicon(addresses, "us"), frozenset({"springfield", "portland"}))


if __name__ == "__main__":
    unittest.main()
