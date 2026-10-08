import pandas as pd

from divina_midi.features import sentiment


def test_tercets_join_verses_and_keep_first_line():
    corpus = pd.DataFrame({
        "cantica": "Inferno", "canto": 1, "tercet": [1, 1, 1, 2],
        "line": [1, 2, 3, 4],
        "text": ["Nel mezzo", "mi ritrovai", "ché la diritta via", "Ahi quanto a dir"],
    })
    table = sentiment.tercets(corpus)
    assert table["line"].tolist() == [1, 4]
    assert table["text"].tolist() == ["Nel mezzo mi ritrovai ché la diritta via", "Ahi quanto a dir"]


def test_tercets_normalize_apostrophes():
    corpus = pd.DataFrame({"cantica": "Inferno", "canto": 1, "tercet": [1], "line": [1],
                           "text": ["Tant’è amara"]})
    assert sentiment.tercets(corpus)["text"].item() == "Tant'è amara"
