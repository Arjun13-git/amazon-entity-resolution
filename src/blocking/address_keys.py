"""
Structured keys extracted from ``norm_address`` for exact blocking.

Normalized addresses have no delimiters and no fixed token order, so keys
are extracted by token rules rather than by position:

- house number: first token containing a digit, reduced to its digits
  without leading zeros ("0645" == "645", "010" == "10")
- postal code: last 5-digit (US) / 6-digit (India) token other than the
  house-number token
- city: rightmost token found in a city lexicon, after removing state
  names/codes. The lexicon is learned from S1: tokens that appear right
  before a state name, kept when frequent. It holds a few thousand
  strings per country.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd


US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut",
    "de": "delaware", "dc": "district of columbia", "fl": "florida",
    "ga": "georgia", "hi": "hawaii", "id": "idaho", "il": "illinois",
    "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky",
    "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming",
}

INDIA_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam",
    "br": "bihar", "cg": "chhattisgarh", "ct": "chhattisgarh",
    "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachal pradesh", "jh": "jharkhand", "ka": "karnataka",
    "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra",
    "mn": "manipur", "ml": "meghalaya", "mz": "mizoram",
    "nl": "nagaland", "od": "odisha", "or": "orissa", "pb": "punjab",
    "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu",
    "tg": "telangana", "ts": "telangana", "tr": "tripura",
    "up": "uttar pradesh", "uk": "uttarakhand", "ut": "uttarakhand",
    "wb": "west bengal", "dl": "delhi", "jk": "jammu and kashmir",
    "ch": "chandigarh", "py": "puducherry", "la": "ladakh",
    "dn": "dadra and nagar haveli", "ld": "lakshadweep",
    "an": "andaman and nicobar islands",
}

STATES = {"us": US_STATES, "india": INDIA_STATES}

# Frequent tokens before a state name that are not cities.
GENERIC = {
    "road", "street", "st", "nagar", "colony", "sector", "city", "town",
    "township", "county", "district", "distt", "dist", "village", "cdp",
    "east", "west", "north", "south", "new", "the", "and", "of", "no",
    "floor", "block", "near", "opp", "area", "main", "cross", "lane",
    "avenue", "ave", "rd", "dr", "drive", "unit", "suite", "apt", "null",
}

DIGITS = re.compile(r"\d")


def state_phrases(country: str) -> list[tuple[str, ...]]:
    """All state names and codes as token tuples, longest first."""

    table = STATES[country]
    phrases = {(code,) for code in table}
    phrases |= {tuple(name.split()) for name in table.values()}
    phrases.add(("jammu", "kashmir"))

    return sorted(phrases, key=len, reverse=True)


def strip_states(tokens: list[str], phrases: list[tuple[str, ...]]) -> tuple[list[str], list[int]]:
    """Remove state phrases; also return the index each one started at."""

    by_len = _phrase_sets(tuple(phrases))
    lengths = sorted(by_len, reverse=True)

    out, starts = [], []
    i = 0

    while i < len(tokens):
        for n in lengths:
            if tuple(tokens[i:i + n]) in by_len[n]:
                starts.append(len(out))
                i += n
                break
        else:
            out.append(tokens[i])
            i += 1

    return out, starts


_PHRASE_CACHE: dict[tuple, dict[int, frozenset]] = {}


def _phrase_sets(phrases: tuple) -> dict[int, frozenset]:
    """State phrases grouped by token length, for set lookups."""

    if phrases not in _PHRASE_CACHE:
        grouped: dict[int, set] = {}
        for phrase in phrases:
            grouped.setdefault(len(phrase), set()).add(phrase)
        _PHRASE_CACHE[phrases] = {n: frozenset(v) for n, v in grouped.items()}

    return _PHRASE_CACHE[phrases]


def learn_city_lexicon(
    addresses: pd.Series,
    country: str,
    min_count: int = 25,
) -> frozenset[str]:
    """Tokens that frequently precede a *full* state name in ``addresses``."""

    full_names = [
        tuple(name.split())
        for name in set(STATES[country].values())
    ]
    # US S1 writes states as codes; India S1 writes full names.
    phrases = (
        [(code,) for code in STATES[country]]
        if country == "us"
        else sorted(full_names, key=len, reverse=True)
    )

    counts: Counter[str] = Counter()

    for address in addresses:
        tokens = address.split()
        kept, starts = strip_states(tokens, phrases)

        for s in starts:
            if s > 0:
                prev = kept[s - 1]
                if not DIGITS.search(prev) and len(prev) >= 3 and prev not in GENERIC:
                    counts[prev] += 1

    return frozenset(t for t, c in counts.items() if c >= min_count)


def house_number(tokens: list[str]) -> tuple[str, int]:

    for i, token in enumerate(tokens):
        if DIGITS.search(token):
            digits = re.sub(r"\D", "", token).lstrip("0")
            if digits:
                return digits, i
            return "", i

    return "", -1


def extract_keys(
    addresses: pd.Series,
    country: str,
    lexicon: frozenset[str],
) -> pd.DataFrame:
    """Per-address structured keys ('' = not extracted)."""

    phrases = state_phrases(country)
    postal_len = 5 if country == "us" else 6

    house, postal, city = [], [], []

    for address in addresses.fillna(""):
        tokens = address.split()
        h, h_idx = house_number(tokens)

        p = ""
        for i in range(len(tokens) - 1, -1, -1):
            tok = tokens[i]
            if i != h_idx and len(tok) == postal_len and tok.isdigit():
                p = tok
                break

        kept, _ = strip_states(tokens, phrases)
        c = ""
        for tok in reversed(kept):
            if tok in lexicon:
                c = tok
                break

        house.append(h)
        postal.append(p)
        city.append(c)

    df = pd.DataFrame({"house": house, "postal": postal, "city": city})
    both = (df["house"] != "") & (df["city"] != "")
    df["house_city"] = np.where(both, df["house"] + "|" + df["city"], "")

    return df
