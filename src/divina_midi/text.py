"""Text normalization and tokenization for Dante's Italian."""

from __future__ import annotations

import re
import unicodedata

# Typographic apostrophes/quotes and the metrical diaeresis (patrïa, vïaggio)
# are spelling variants that would otherwise split one word into several.
_TRANSLATE = str.maketrans({
    "’": "'", "‘": "'", "«": '"', "»": '"', "“": '"', "”": '"',
    "ï": "i", "ü": "u", "Ï": "I", "Ü": "U",
})
_WORD = re.compile(r"[^\W\d_]+")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_TRANSLATE)


def tokenize(text: str) -> list[str]:
    """Lowercase words of a verse; elisions are split (``ch'i'`` -> ``ch``, ``i``)."""
    return _WORD.findall(normalize(text).lower())
