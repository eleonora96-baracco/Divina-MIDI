# divina-midi

Turn the cantos of Dante's *Divina Commedia* into MIDI music.

The project compares approaches of increasing sophistication for translating the text into notes:

1. **Word2Vec**: vectors trained on the text, mapped to pitches. The result is artistic but arbitrary.
2. **Contextual embeddings** (multilingual BERT): similar verses sound similar.
3. **Sentiment / emotion**: dimensions with a meaning, mapped onto musical conventions (mode, tempo, dynamics).

> 🚧 Work in progress: only the corpus download is available so far.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev]"
```

## Usage

Download the text from Wikisource (100 cantos, about one minute):

```bash
divina-midi download
```

This creates `data/commedia.csv`, with one verse per row:

| cantica | canto | tercet | line | text |
|---|---|---|---|---|
| Inferno | 1 | 1 | 1 | Nel mezzo del cammin di nostra vita |

`tercet` is the stanza number within the canto (the closing single verse of each canto counts as its own stanza) and `line` is the verse number within the canto.

From Python:

```python
from divina_midi.corpus import load_corpus, select

df = load_corpus()
inferno_1 = select(df, cantica="inferno", canto=1)
```

## Tests

```bash
pytest
```

Tests marked `corpus` check the full text (14,233 verses, every canto in terza rima) and are skipped until the corpus has been downloaded.

## Text source

The text is downloaded from [Italian Wikisource](https://it.wikisource.org/wiki/Divina_Commedia) (licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)) and is not included in this repository.
