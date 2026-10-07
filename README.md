# divina-midi

Trasformare i canti della *Divina Commedia* in musica MIDI.

Il progetto confronta approcci di complessità crescente per tradurre il testo in note:

1. **Word2Vec**: vettori addestrati sul testo, mappati su altezze. Il risultato è artistico ma arbitrario.
2. **Embedding contestuali** (BERT multilingue): versi simili suonano in modo simile.
3. **Sentiment / emozione**: dimensioni con un significato, mappate su convenzioni musicali (modo, tempo, dinamica).

> 🚧 Work in progress: per ora è disponibile solo il download del corpus.

## Installazione

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
```

## Uso

Scaricare il testo da Wikisource (100 canti, ~1 minuto):

```bash
divina-midi download
```

Il risultato è `data/commedia.csv`, con un verso per riga:

| cantica | canto | tercet | line | text |
|---|---|---|---|---|
| Inferno | 1 | 1 | 1 | Nel mezzo del cammin di nostra vita |

In Python:

```python
from divina_midi.corpus import load_corpus, select

df = load_corpus()
inferno_1 = select(df, cantica="inferno", canto=1)
```

## Test

```bash
pytest
```

## Fonte del testo

Il testo è scaricato da [Wikisource](https://it.wikisource.org/wiki/Divina_Commedia) (licenza [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)) e non è incluso nel repository.
