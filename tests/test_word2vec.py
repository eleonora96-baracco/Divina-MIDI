import numpy as np
import pandas as pd

from divina_midi.features import word2vec

CORPUS = pd.DataFrame({
    "cantica": "Inferno", "canto": 1, "tercet": [1, 1, 1],
    "line": [1, 2, 3],
    "text": [
        "Nel mezzo del cammin di nostra vita",
        "mi ritrovai per una selva oscura,",
        "ché la diritta via era smarrita.",
    ],
})


def test_training_is_reproducible_with_a_seed():
    a = word2vec.train(CORPUS, seed=1)
    b = word2vec.train(CORPUS, seed=1)
    np.testing.assert_array_equal(a.wv["selva"], b.wv["selva"])


def test_word_features_have_one_row_per_word():
    model = word2vec.train(CORPUS, vector_size=4)
    words = word2vec.word_features(CORPUS, model)
    assert len(words) == 7 + 6 + 6
    assert words.loc[0, ["word", "length", "word_index"]].tolist() == ["nel", 3, 0]
    assert [f"dim_{i}" for i in range(4)] == words.columns[-4:].tolist()


def test_verse_features_average_word_vectors():
    model = word2vec.train(CORPUS, vector_size=4)
    words = word2vec.word_features(CORPUS, model)
    verses = word2vec.verse_features(CORPUS, model)
    assert verses["line"].tolist() == [1, 2, 3]
    expected = words[words["line"] == 2]["dim_0"].mean()
    assert np.isclose(verses.loc[1, "dim_0"], expected)
