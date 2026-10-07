"""Feature extractors: turn the corpus into numeric tables.

Every extractor returns a DataFrame with the corpus position columns
(``cantica, canto, tercet, line``, plus ``word_index, word`` at word level)
followed by numeric feature columns that the mapping can use.
"""

POSITION_COLUMNS = ["cantica", "canto", "tercet", "line"]
