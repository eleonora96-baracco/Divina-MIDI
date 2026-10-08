"""Text normalization and tokenization for Dante's Italian."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

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


def words(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per word: its verse position, index in the verse, the word and its length."""
    rows = [
        (verse.cantica, verse.canto, verse.tercet, verse.line, i, word, len(word))
        for verse in corpus.itertuples(index=False)
        for i, word in enumerate(tokenize(verse.text))
    ]
    return pd.DataFrame(rows, columns=["cantica", "canto", "tercet", "line", "word_index", "word", "length"])
