import numpy as np
import pandas as pd

from divina_midi.features import embeddings
from divina_midi.text import tokenize, word_spans


def test_word_spans_match_tokenize():
    verse = "dirò de l’altre cose ch’i’ v’ ho scorte."
    text, spans = word_spans(verse)
    assert [text[start:end].lower() for start, end in spans] == tokenize(verse)


def test_pool_words_averages_subtokens_and_skips_special_tokens():
    # "dirò de": <s> ▁di rò ▁de </s> <pad>
    offsets = np.array([[0, 0], [0, 2], [2, 4], [5, 7], [0, 0], [0, 0]])
    hidden = np.array([[100.0], [1.0], [3.0], [10.0], [100.0], [100.0]])
    pooled = embeddings.pool_words(hidden, offsets, [(0, 4), (5, 7)])
    assert [p.item() for p in pooled] == [2.0, 10.0]


def test_pca_orders_axes_by_variance_and_fixes_sign():
    rng = np.random.default_rng(0)
    vectors = np.column_stack([rng.normal(0, 1, 500), rng.normal(0, 10, 500), rng.normal(0, 3, 500)])
    projected, explained = embeddings.pca(vectors, 2)
    assert projected.shape == (500, 2)
    assert explained[0] > explained[1] > 0.05
    # The main axis is the second input column, oriented positively.
    assert np.corrcoef(projected[:, 0], vectors[:, 1])[0, 1] > 0.99
    assert np.allclose(embeddings.pca(-vectors, 2)[0], -projected)


def test_word_features_use_cached_vectors(tmp_path):
    corpus = pd.DataFrame({"cantica": "Inferno", "canto": 1, "tercet": [1, 1], "line": [1, 2],
                           "text": ["Nel mezzo del cammin", "di nostra vita"]})
    rng = np.random.default_rng(1)
    np.save(tmp_path / f"{embeddings.MODEL.replace('/', '__')}.npy", rng.normal(size=(7, 16)))

    words = embeddings.word_features(corpus, n_components=3, cache_dir=tmp_path)
    assert words["word"].tolist() == ["nel", "mezzo", "del", "cammin", "di", "nostra", "vita"]
    assert [f"pc_{i}" for i in range(3)] == words.columns[-3:].tolist()
    assert len(words.attrs["explained_variance"]) == 3

    verses = embeddings.verse_features(corpus, n_components=3, cache_dir=tmp_path)
    assert np.isclose(verses.loc[1, "pc_0"], words["pc_0"][4:].mean())
