"""Contextual embeddings: each word gets a vector computed inside its verse.

Unlike Word2Vec, where a word always has the same vector, a pretrained
multilingual sentence model reads the whole verse, so the same word can sound
different in different contexts, and verses with similar meaning get similar
vectors. The vectors are projected on their main axes with a single PCA fitted
on the whole poem, so that an axis means the same thing in every canto. Those
axes still have no name, which is the limit the sentiment level addresses.

Requires the ``sentiment`` extra (PyTorch and Transformers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..text import word_spans, words
from . import mean_by_verse

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def pool_words(hidden: np.ndarray, offsets: np.ndarray, spans: list[tuple[int, int]]) -> list[np.ndarray]:
    """Average the token vectors overlapping each word's character span.

    ``offsets`` holds each token's (start, end) characters; special and padding
    tokens have empty spans and are ignored. A word no token overlaps (it should
    not happen) falls back to the mean of the whole text.
    """
    token_start, token_end = offsets[:, 0], offsets[:, 1]
    is_text = token_end > token_start
    pooled = []
    for word_start, word_end in spans:
        overlap = is_text & (token_start < word_end) & (token_end > word_start)
        pooled.append(hidden[overlap if overlap.any() else is_text].mean(axis=0))
    return pooled


def word_vectors(texts: list[str], model: str = MODEL, batch_size: int = 64) -> np.ndarray:
    """One contextual vector per word of each text, stacked in reading order.

    Words are exactly those of :func:`divina_midi.text.tokenize`; a word split
    into several sub-tokens gets the mean of their vectors.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    encoder = AutoModel.from_pretrained(model).eval()

    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            normalized, spans = zip(*(word_spans(text) for text in texts[start:start + batch_size]))
            batch = tokenizer(list(normalized), padding=True, truncation=True, max_length=128,
                              return_offsets_mapping=True, return_tensors="pt")
            offsets = batch.pop("offset_mapping").numpy()
            hidden = encoder(**batch).last_hidden_state.numpy()
            for verse_hidden, verse_offsets, verse_spans in zip(hidden, offsets, spans):
                vectors.extend(pool_words(verse_hidden, verse_offsets, verse_spans))
    return np.array(vectors, dtype=np.float32)


def raw_word_embeddings(corpus: pd.DataFrame, model: str = MODEL,
                        cache_dir: Optional[Path] = Path("data") / "embeddings") -> np.ndarray:
    """Word vectors for the whole corpus, cached since computing them takes minutes."""
    cache = Path(cache_dir) / f"{model.replace('/', '__')}.npy" if cache_dir else None
    n_words = len(words(corpus))
    if cache is not None and cache.exists():
        vectors = np.load(cache).astype(np.float32)
        if len(vectors) == n_words:
            return vectors

    vectors = word_vectors(corpus["text"].tolist(), model)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors.astype(np.float16))  # halves the size; precision is plenty
    return vectors


def pca(vectors: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Project on the main axes of variation; returns projections and explained variance ratios.

    Each axis is oriented so that its largest loading is positive, which makes
    the result independent of the arbitrary sign of eigenvectors.
    """
    centered = vectors - vectors.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    order = np.argsort(eigenvalues)[::-1][:n_components]
    axes = eigenvectors[:, order]
    axes *= np.sign(axes[np.abs(axes).argmax(axis=0), np.arange(axes.shape[1])])
    return centered @ axes, eigenvalues[order] / eigenvalues.sum()


def _pc_columns(n_components: int) -> list[str]:
    return [f"pc_{i}" for i in range(n_components)]


def word_features(corpus: pd.DataFrame, n_components: int = 8, model: str = MODEL,
                  cache_dir: Optional[Path] = Path("data") / "embeddings") -> pd.DataFrame:
    """One row per word: position, the word, its length and its main components.

    Fit on the whole poem, then select cantos from the result, so that the axes
    are shared by every canto. The explained variance ratios are stored in
    ``.attrs["explained_variance"]``.
    """
    projected, explained = pca(raw_word_embeddings(corpus, model, cache_dir), n_components)
    table = pd.concat([words(corpus), pd.DataFrame(projected, columns=_pc_columns(n_components))],
                      axis=1)
    table.attrs["explained_variance"] = explained.tolist()  # pandas attrs can't hold arrays
    return table


def verse_features(corpus: pd.DataFrame, n_components: int = 8, model: str = MODEL,
                   cache_dir: Optional[Path] = Path("data") / "embeddings") -> pd.DataFrame:
    """One row per verse: mean word length and mean of its words' components."""
    table = word_features(corpus, n_components, model, cache_dir)
    return mean_by_verse(table, ["length"] + _pc_columns(n_components))
