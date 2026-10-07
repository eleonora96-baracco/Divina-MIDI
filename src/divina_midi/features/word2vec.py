"""Word2Vec features: vectors trained on the Commedia itself.

The dimensions of a Word2Vec vector have no meaning of their own, so music made
from them reflects word co-occurrence patterns rather than anything a listener
can name. This is the baseline the other extractors are compared against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from ..text import tokenize
from . import POSITION_COLUMNS


def train(corpus: pd.DataFrame, vector_size: int = 8, window: int = 5,
          epochs: int = 5, seed: int = 42) -> Word2Vec:
    """Train on the whole corpus, one verse per sentence.

    A single worker thread makes training deterministic for a given seed.
    """
    sentences = [tokenize(text) for text in corpus["text"]]
    return Word2Vec(sentences, vector_size=vector_size, window=window, min_count=1,
                    epochs=epochs, seed=seed, workers=1)


def _dim_columns(model: Word2Vec) -> list[str]:
    return [f"dim_{i}" for i in range(model.wv.vector_size)]


def word_features(corpus: pd.DataFrame, model: Word2Vec) -> pd.DataFrame:
    """One row per word: position, the word, its length and its vector."""
    rows, vectors = [], []
    for verse in corpus.itertuples(index=False):
        for i, word in enumerate(tokenize(verse.text)):
            rows.append((verse.cantica, verse.canto, verse.tercet, verse.line, i, word, len(word)))
            vectors.append(model.wv[word])
    df = pd.DataFrame(rows, columns=POSITION_COLUMNS + ["word_index", "word", "length"])
    dims = pd.DataFrame(np.array(vectors), columns=_dim_columns(model))
    return pd.concat([df, dims], axis=1)


def verse_features(corpus: pd.DataFrame, model: Word2Vec) -> pd.DataFrame:
    """One row per verse: mean of its word vectors and mean word length."""
    words = word_features(corpus, model)
    numeric = ["length"] + _dim_columns(model)
    return words.groupby(POSITION_COLUMNS, sort=False, as_index=False)[numeric].mean()
