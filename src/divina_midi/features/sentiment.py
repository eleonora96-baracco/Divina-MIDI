"""Sentiment and emotion features from pretrained Italian classifiers.

Scores are computed per tercet: a single verse is often too short to carry a
clear emotion, and the tercet is also the natural musical phrase. The raw
logits are kept rather than probabilities, because these classifiers tend to
be saturated (probabilities of 0 or 1) on Dante's language while the logits
still rank tercets from more to less negative.

Requires the ``sentiment`` extra: ``pip install -e ".[sentiment]"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..text import normalize

MODELS = {
    "feel-it-sentiment": "MilaNLProc/feel-it-italian-sentiment",
    "feel-it-emotion": "MilaNLProc/feel-it-italian-emotion",
    "xlm-roberta-sentiment": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
}
TERCET_COLUMNS = ["cantica", "canto", "tercet"]


def tercets(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per tercet, with its verses joined into a single text."""
    return (corpus.groupby(TERCET_COLUMNS, sort=False)
            .agg(line=("line", "first"), text=("text", lambda verses: " ".join(map(normalize, verses))))
            .reset_index())


def logits(texts: list[str], model: str, batch_size: int = 32) -> pd.DataFrame:
    """Raw classifier logits, one column per label."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = MODELS.get(model, model)
    tokenizer = AutoTokenizer.from_pretrained(name)
    classifier = AutoModelForSequenceClassification.from_pretrained(name).eval()

    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=128, return_tensors="pt")
            batches.append(classifier(**batch).logits.numpy())
    labels = [classifier.config.id2label[i].lower() for i in range(classifier.config.num_labels)]
    return pd.DataFrame(np.concatenate(batches), columns=labels)


def tercet_features(corpus: pd.DataFrame, model: str = "feel-it-sentiment",
                    cache_dir: Optional[Path] = Path("data") / "sentiment") -> pd.DataFrame:
    """Tercets with the model's logits, cached as CSV since scoring takes minutes."""
    cache = Path(cache_dir) / f"{model.replace('/', '__')}.csv" if cache_dir else None
    if cache is not None and cache.exists():
        return pd.read_csv(cache, encoding="utf-8")

    table = tercets(corpus)
    scores = logits(table["text"].tolist(), model)
    if {"positive", "negative"} <= set(scores.columns):
        scores["valence"] = scores["positive"] - scores["negative"]
    result = pd.concat([table, scores], axis=1)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache, index=False, encoding="utf-8")
    return result


def affect(corpus: pd.DataFrame,
           cache_dir: Optional[Path] = Path("data") / "sentiment") -> pd.DataFrame:
    """Tercets with two affect dimensions, from the two feel-it models.

    - ``valence``: positive vs negative (sentiment model).
    - ``arousal``: energy, taken as the strongest of anger / fear / joy against
      sadness, the one low-energy emotion of the emotion model.
    """
    sent = tercet_features(corpus, "feel-it-sentiment", cache_dir)
    emo = tercet_features(corpus, "feel-it-emotion", cache_dir)
    high = emo[["anger", "fear", "joy"]].to_numpy()
    emo = emo[TERCET_COLUMNS].assign(arousal=np.logaddexp.reduce(high, axis=1) - emo["sadness"])
    return sent[TERCET_COLUMNS + ["line", "text", "valence"]].merge(emo, on=TERCET_COLUMNS)
