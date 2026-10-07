import pytest

from divina_midi import corpus

# Trimmed from the rendered Wikisource page of Inferno I: verse numbers live in
# span.numeroriga, stanzas are separated by a double <br/>, and quotation spans
# can wrap several stanzas.
CANTO_HTML = """
<div class="mw-parser-output">
<p><i>Incomincia la Comedia di Dante Alleghieri di Fiorenza...</i></p>
<div class="poem"><p><span> </span><br/>
<span class="Citazione">Nel mezzo del cammin di nostra vita<br/>
mi ritrovai per una selva oscura,<br/>
ché la diritta via era smarrita. <style>.mw-parser-output .numeroriga{float:right}</style><span class="numeroriga">3</span><br/>
<br/>
Ahi quanto a dir qual era è cosa dura<br/>
esta selva selvaggia e aspra e forte<br/>
che nel pensier rinova la paura! <link/><span class="numeroriga">6</span></span><br/>
<br/>
Allor si mosse, e io li tenni dietro.
</p></div>
</div>
"""


def test_to_roman():
    assert [corpus.to_roman(n) for n in (1, 4, 9, 14, 19, 24, 34)] == [
        "I", "IV", "IX", "XIV", "XIX", "XXIV", "XXXIV",
    ]


def test_parse_canto_groups_stanzas_and_drops_verse_numbers():
    stanzas = corpus.parse_canto(CANTO_HTML)
    assert stanzas == [
        [
            "Nel mezzo del cammin di nostra vita",
            "mi ritrovai per una selva oscura,",
            "ché la diritta via era smarrita.",
        ],
        [
            "Ahi quanto a dir qual era è cosa dura",
            "esta selva selvaggia e aspra e forte",
            "che nel pensier rinova la paura!",
        ],
        ["Allor si mosse, e io li tenni dietro."],
    ]


def test_parse_canto_without_poem_raises():
    with pytest.raises(ValueError):
        corpus.parse_canto("<div>nothing here</div>")


def test_canto_rows_numbers_lines_and_tercets():
    rows = corpus.canto_rows("Inferno", 1, corpus.parse_canto(CANTO_HTML))
    assert [(r["tercet"], r["line"]) for r in rows] == [
        (1, 1), (1, 2), (1, 3), (2, 4), (2, 5), (2, 6), (3, 7),
    ]


def test_select_is_case_insensitive():
    df = corpus.pd.DataFrame(
        corpus.canto_rows("Inferno", 1, [["a"]]) + corpus.canto_rows("Paradiso", 1, [["b"]])
    )
    assert corpus.select(df, cantica="inferno")["text"].tolist() == ["a"]
    assert len(corpus.select(df, canto=1)) == 2


@pytest.fixture(scope="module")
def full_corpus():
    if not corpus.DEFAULT_PATH.exists():
        pytest.skip("corpus not downloaded")
    return corpus.load_corpus(download=False)


@pytest.mark.corpus
def test_full_corpus_shape(full_corpus):
    assert len(full_corpus) == 14233
    assert full_corpus.groupby(["cantica", "canto"]).ngroups == 100


@pytest.mark.corpus
def test_every_canto_is_in_terza_rima(full_corpus):
    """Each canto is n tercets plus one closing verse: 3n + 1 lines."""
    for (cantica, canto), verses in full_corpus.groupby(["cantica", "canto"]):
        sizes = verses.groupby("tercet").size().tolist()
        assert sizes[:-1] == [3] * (len(sizes) - 1), f"{cantica} {canto}"
        assert sizes[-1] == 1, f"{cantica} {canto}"
