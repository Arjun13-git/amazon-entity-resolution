from __future__ import annotations

import re
import unicodedata

from rapidfuzz.fuzz import ratio


BUSINESS_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "llc",
    "ltd",
    "limited",
    "plc",
    "llp",
    "pvt",
    "private",
    "sarl",
    "sasu",
    "srl",
    "gmbh",
    "ag",
    "sa",
    "sas",
    "eurl",
}


def normalize_unicode(text: str) -> str:
    """
    Apply Unicode compatibility normalization while preserving
    multilingual scripts and combining marks.
    """
    if not text:
        return ""

    return unicodedata.normalize("NFKC", text)


def normalize_text(text: str) -> str:
    """
    General-purpose multilingual text normalization.

    Operations:
    - Unicode NFKC normalization
    - lowercase
    - punctuation/symbol normalization
    - whitespace normalization

    Unicode letters, numbers, and combining marks are preserved.
    """
    if not text:
        return ""

    text = normalize_unicode(text)
    text = text.lower()

    normalized_chars = []

    for char in text:
        category = unicodedata.category(char)

        # Keep letters, numbers, combining marks, and whitespace.
        if (
            category.startswith("L")
            or category.startswith("N")
            or category.startswith("M")
            or char.isspace()
        ):
            normalized_chars.append(char)
        else:
            # Convert punctuation/symbols to spaces.
            normalized_chars.append(" ")

    text = "".join(normalized_chars)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_business_name(
    text: str,
    *,
    remove_suffixes: bool = True,
) -> str:
    """
    Normalize a business name.

    Business suffixes are removed conservatively from the
    beginning and end of the normalized token sequence.
    """
    text = normalize_text(text)

    if not text:
        return ""

    if not remove_suffixes:
        return text

    tokens = text.split()

    # Remove common legal/business prefixes.
    while tokens and tokens[0] in BUSINESS_SUFFIXES:
        tokens.pop(0)

    # Remove common legal/business suffixes.
    while tokens and tokens[-1] in BUSINESS_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def normalize_address(text: str) -> str:
    """
    Normalize a business address without removing
    business/legal terms.
    """
    return normalize_text(text)


def normalize_country(text: str) -> str:
    """
    Normalize a country label.
    """
    return normalize_text(text)


def name_similarity(
    left: str,
    right: str,
) -> float:
    """
    Return normalized fuzzy similarity between two names.

    Returns
    -------
    float
        Similarity in the range [0, 1].
    """
    left_norm = normalize_business_name(left)
    right_norm = normalize_business_name(right)

    if not left_norm or not right_norm:
        return 0.0

    return ratio(
        left_norm,
        right_norm,
    ) / 100.0