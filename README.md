# Divina MIDI

[![tests](https://github.com/eleonora96-baracco/Divina-MIDI/actions/workflows/tests.yml/badge.svg)](https://github.com/eleonora96-baracco/Divina-MIDI/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Turn the cantos of Dante's *Divina Commedia* into music.

**🎧 [Listen to the demo](https://eleonora96-baracco.github.io/Divina-MIDI/)**: the first canto of each cantica, rendered three ways.

The project compares three approaches of increasing sophistication for translating the text into notes:

| Level | What drives the music | What you hear |
|---|---|---|
| 1. **Word2Vec** | vectors learned from the poem itself | artistic but arbitrary: the three cantiche sound almost the same |
| 2. **Contextual embeddings** | a multilingual language model reading each verse | the structure of the language, with axes that still have no name |
| 3. **Sentiment** | Italian classifiers scoring each tercet's emotion | the arc from Hell to Paradise, through mode, register, speed and dynamics |

Every word of the poem becomes a note, so a canto lasts 10 to 13 minutes: about as long as reading it aloud.

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

Rendering to audio (WAV or MP3) is a third extra, `audio`.

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

The first run downloads [GeneralUser GS](https://github.com/mrbumpy409/GeneralUser-GS) (32 MB, free to use) into `data/soundfonts/`; use `--soundfont` for another one. Add `--format mp3` for compressed files and `--seconds 60` for an excerpt.

### Rebuilding the demo page

The [demo page](docs/index.html) is served by GitHub Pages from `docs/`. To regenerate its audio excerpts and chart data (needs all the extras):

```bash
divina-midi demo
```

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

## Project structure

```
src/divina_midi/
├── corpus.py          # download and parse the text from Wikisource
├── text.py            # normalization and tokenization
├── features/          # one extractor per level
│   ├── word2vec.py
│   ├── embeddings.py
│   └── sentiment.py
├── mapping.py         # features -> notes (Mapping for levels 1-2, AffectMapping for level 3)
├── midi.py            # notes -> MIDI file
├── audio.py           # MIDI -> WAV / MP3 with a SoundFont
└── cli.py             # the divina-midi command
notebooks/             # analyses (sentiment feasibility)
docs/                  # demo page
tests/
```

The project grew out of a set of exploratory notebooks; this package rewrites them into a reproducible pipeline.

## Tests

```bash
pytest
```

Tests that need the downloaded corpus, the models or the SoundFont are skipped when those are missing; GitHub Actions runs the rest on Python 3.9 and 3.12 at every push.

## Credits

- **Text**: [Italian Wikisource](https://it.wikisource.org/wiki/Divina_Commedia), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). It is downloaded by the code and not included in this repository.
- **Models**: [feel-it](https://huggingface.co/MilaNLProc/feel-it-italian-sentiment) by MilaNLProc (sentiment and emotion), [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) by Sentence Transformers.
- **Sound**: [GeneralUser GS](https://github.com/mrbumpy409/GeneralUser-GS) SoundFont by S. Christian Collins.

## License

The code is released under the [MIT License](LICENSE).
