from divina_midi.text import normalize, tokenize


def test_normalize_unifies_apostrophes_quotes_and_diaeresis():
    assert normalize("ch’i’ «patrïa»") == "ch'i' \"patria\""


def test_tokenize_lowercases_and_splits_elisions():
    assert tokenize("Tant’è amara che poco è più morte;") == [
        "tant", "è", "amara", "che", "poco", "è", "più", "morte",
    ]
    assert tokenize('"Miserere di me", gridai a lui,') == ["miserere", "di", "me", "gridai", "a", "lui"]
