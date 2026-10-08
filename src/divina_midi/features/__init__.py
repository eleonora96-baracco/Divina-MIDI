"""Feature extractors: turn the corpus into numeric tables.

Every extractor returns a DataFrame with the corpus position columns
(``cantica, canto, tercet, line``, plus ``word_index, word`` at word level)
followed by numeric feature columns that the mapping can use.
"""

from __future__ import annotations

import pandas as pd

POSITION_COLUMNS = ["cantica", "canto", "tercet", "line"]


def mean_by_verse(words: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Average word-level features over each verse."""
    return words.groupby(POSITION_COLUMNS, sort=False, as_index=False)[columns].mean()
