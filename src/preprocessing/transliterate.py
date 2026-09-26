from __future__ import annotations

from unidecode import unidecode


def transliterate_text(text: str) -> str:
    """
    Convert multilingual Unicode text to a Latin-script representation.

    The original normalized text should always be retained separately.
    """

    if not text:
        return ""

    return unidecode(text).lower().strip()