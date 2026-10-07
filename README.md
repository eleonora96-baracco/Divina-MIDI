# divina-midi

Turn the cantos of Dante's *Divina Commedia* into MIDI music.

The project compares approaches of increasing sophistication for translating the text into notes:

1. **Word2Vec**: vectors trained on the text, mapped to pitches. The result is artistic but arbitrary.
2. **Contextual embeddings** (multilingual BERT): similar verses sound similar.
3. **Sentiment / emotion**: dimensions with a meaning, mapped onto musical conventions (mode, tempo, dynamics).

> 🚧 Work in progress: the Word2Vec level is available; the other two are coming.

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

### Rendering a canto

```bash
divina-midi render --cantica inferno --canto 1
```

This writes `output/inferno_01_word2vec_word.mid`. Useful options:

| Option | Values | Default |
|---|---|---|
| `--unit` | `word` (one note per word) or `verse` (one note per verse) | `word` |
| `--scale` | `c_major`, `a_minor`, `c_pentatonic`, `chromatic` | `c_major` |
| `--tempo` | beats per minute | `90` |
| `--seed` | random seed for Word2Vec training | `42` |

### How text becomes music

1. **Features**: each word (or verse) becomes a row of numbers. With Word2Vec these are the dimensions of a vector trained on the whole poem, plus the word length.
2. **Mapping**: one feature drives the pitch (within the chosen scale), one the velocity, one the duration. Each feature's range is learned from the whole poem, ignoring outliers, so the same value gives the same note in every canto.
3. **Terza rima**: independently of the features, the last note of each tercet is accented and lengthened, and short rests separate verses and tercets.

## Tests

```bash
pytest
```

Tests marked `corpus` check the full text (14,233 verses, every canto in terza rima) and are skipped until the corpus has been downloaded.

## Text source

The text is downloaded from [Italian Wikisource](https://it.wikisource.org/wiki/Divina_Commedia) (licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)) and is not included in this repository.
