# divina-midi

Turn the cantos of Dante's *Divina Commedia* into MIDI music.

The project compares approaches of increasing sophistication for translating the text into notes:

1. **Word2Vec**: vectors trained on the text, mapped to pitches. The result is artistic but arbitrary.
2. **Contextual embeddings** (multilingual BERT): similar verses sound similar.
3. **Sentiment / emotion**: dimensions with a meaning, mapped onto musical conventions (mode, tempo, dynamics).

> 🚧 Work in progress: the Word2Vec and sentiment levels are available; contextual embeddings are coming.

## Installation

Requires Python 3.9+ (developed on 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev]"
```

The sentiment level needs PyTorch and Hugging Face Transformers (about 1 GB with the models), so it is an optional extra; the notebooks need Jupyter and Matplotlib:

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
divina-midi render --cantica inferno --canto 1                      # Word2Vec level
divina-midi render --cantica inferno --canto 1 --method sentiment   # sentiment level
```

This writes `output/inferno_01_word2vec_word.mid` or `output/inferno_01_sentiment.mid`. The first sentiment run scores the whole poem (about 8 minutes on a CPU) and caches the results in `data/sentiment/`. Options:

| Option | Values | Default |
|---|---|---|
| `--method` | `word2vec` or `sentiment` | `word2vec` |
| `--tempo` | beats per minute | `90` |
| `--unit` | Word2Vec only: `word` (one note per word) or `verse` (one note per verse) | `word` |
| `--scale` | Word2Vec only: `c_major`, `a_minor`, `c_pentatonic`, `chromatic` | `c_major` |
| `--seed` | Word2Vec only: random seed for training | `42` |

## How text becomes music

Both levels share the same principles: every feature's range is learned from the **whole poem**, ignoring outliers, so the same value gives the same music in every canto; and the **terza rima** is always audible, with rests between verses and tercets and an accent on the note that closes each tercet.

### Level 1: Word2Vec

Each word becomes a vector trained on the whole poem. One dimension drives the pitch (within the chosen scale), another the velocity, and the word length the duration. The vector dimensions have no meaning, so the music follows word co-occurrence patterns: the three cantiche end up with almost the same distribution of notes.

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
