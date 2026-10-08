# divina-midi

Turn the cantos of Dante's *Divina Commedia* into MIDI music.

The project compares approaches of increasing sophistication for translating the text into notes:

1. **Word2Vec**: vectors trained on the text, mapped to pitches. The result is artistic but arbitrary.
2. **Contextual embeddings** (a multilingual sentence model): the music follows the structure of the language, but its axes still have no name.
3. **Sentiment / emotion**: dimensions with a meaning, mapped onto musical conventions (mode, tempo, dynamics).

## Installation

Requires Python 3.9+ (developed on 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev]"
```

The embeddings and sentiment levels need PyTorch and Hugging Face Transformers (about 1.5 GB with the models), so they are an optional extra; the notebooks need Jupyter and Matplotlib:

```bash
pip install -e ".[dev,sentiment,notebooks]"
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
divina-midi render --cantica inferno --canto 1                       # level 1: Word2Vec
divina-midi render --cantica inferno --canto 1 --method embeddings   # level 2: contextual embeddings
divina-midi render --cantica inferno --canto 1 --method sentiment    # level 3: sentiment
```

Files are written to `output/`, e.g. `output/inferno_01_embeddings_word.mid`. The first run of levels 2 and 3 processes the whole poem on a CPU (about 2 minutes for embeddings, 8 for sentiment) and caches the results in `data/`; later runs take seconds. Options:

| Option | Values | Default |
|---|---|---|
| `--method` | `word2vec`, `embeddings` or `sentiment` | `word2vec` |
| `--tempo` | beats per minute | `90` |
| `--unit` | levels 1-2: `word` (one note per word) or `verse` (one note per verse) | `word` |
| `--scale` | levels 1-2: `c_major`, `a_minor`, `c_pentatonic`, `chromatic` | `c_major` |
| `--seed` | level 1: random seed for Word2Vec training | `42` |

### Listening

A MIDI file holds instructions, not sound: how it sounds depends on the player, and the synthesizer built into Windows is notably harsh. To get audio that sounds the same everywhere, render the files to WAV with a SoundFont:

```bash
pip install -e ".[audio]"
divina-midi audio                 # every .mid file in output/
divina-midi audio output/inferno_01_sentiment.mid
```

The first run downloads [GeneralUser GS](https://github.com/mrbumpy409/GeneralUser-GS) (32 MB, free to use) into `data/soundfonts/`; use `--soundfont` for another one.

## How text becomes music

All three levels share the same principles: every feature's range is learned from the **whole poem**, ignoring outliers, so the same value gives the same music in every canto; and the **terza rima** is always audible, with rests between verses and tercets and an accent on the note that closes each tercet.

### Level 1: Word2Vec

Each word becomes a vector trained on the whole poem. One dimension drives the pitch (within the chosen scale), another the velocity, and the word length the duration. The vector dimensions have no meaning, so the music follows word co-occurrence patterns: the three cantiche end up with almost the same distribution of notes.

### Level 2: contextual embeddings

Each word gets a vector from [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), computed inside its verse: unlike Word2Vec, the same word can sound different in different contexts. A single PCA fitted on the whole poem reduces the vectors to their main axes; the first drives the pitch, the second the velocity.

The axes capture the structure of the language rather than its emotions. The first separates function words (*e, che, ma, di*) from content words (*fondo, petto, terra*); the second separates Dante speaking about himself (*io, mi, dissi, fui*) from descriptions of the world (*luce, sole, mondo, ombra*). The three cantiche differ a little (the Paradiso is somewhat louder), but nothing a listener could name.

### Level 3: sentiment

Each tercet is scored by two Italian classifiers, [feel-it sentiment](https://huggingface.co/MilaNLProc/feel-it-italian-sentiment) and [feel-it emotion](https://huggingface.co/MilaNLProc/feel-it-italian-emotion). Their scores are smoothed over neighbouring tercets and mapped onto musical conventions:

| From the text (per tercet) | To the music |
|---|---|
| **valence** (positive vs negative) | **mode**, from darkest to brightest: Phrygian, Aeolian (minor), Dorian, Mixolydian, Ionian (major), Lydian |
| **arousal** (anger, fear or joy vs sadness) | **speed and loudness** |
| **cantica** | **register**: Inferno low, Purgatorio middle, Paradiso high |
| **tercet structure** | verses end on the fifth, the third, then the root: open, open, closed |

A piano plays the melody (one note per word) and strings hold one chord per tercet, so major and minor can be heard. Over the whole poem, 53% of the Inferno's tercets are in Phrygian, the darkest mode, against 20% of the Paradiso's; Ionian and Lydian together go from 8% to 36%.

[`notebooks/01_sentiment_feasibility.ipynb`](notebooks/01_sentiment_feasibility.ipynb) checks whether these models make sense on Dante's language, and where they fail.

## Tests

```bash
pytest
```

Tests marked `corpus` check the full text (14,233 verses, every canto in terza rima) and are skipped until the corpus has been downloaded.

## Text source

The text is downloaded from [Italian Wikisource](https://it.wikisource.org/wiki/Divina_Commedia) (licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)) and is not included in this repository.
