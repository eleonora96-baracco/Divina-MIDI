"""Map feature tables onto notes.

A :class:`Mapping` reads one row of features per note, in corpus order, and
decides pitch, velocity and duration from configurable feature columns. The
range of each feature is learned from the data (``fit``), ideally on the whole
corpus, so that the same value gives the same note in every canto.

The tercet structure is rendered independently of the features: the last note
of each tercet is accented and lengthened, and rests separate verses and
tercets, so the 3+3+3 rhythm of terza rima is audible whatever the method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Intervals in semitones from the root, and the root as a MIDI pitch.
SCALES = {
    "c_major": (60, [0, 2, 4, 5, 7, 9, 11]),
    "a_minor": (57, [0, 2, 3, 5, 7, 8, 10]),
    "c_pentatonic": (60, [0, 2, 4, 7, 9]),
    "chromatic": (60, list(range(12))),
}


def scale_pitches(scale: str, octaves: int = 2) -> list[int]:
    """MIDI pitches of ``scale`` over ``octaves`` octaves, closing on the root."""
    root, intervals = SCALES[scale]
    pitches = [root + 12 * octave + i for octave in range(octaves) for i in intervals]
    return pitches + [root + 12 * octaves]


@dataclass
class Note:
    start: float  # beats
    pitch: int
    velocity: int
    duration: float  # beats


@dataclass
class Mapping:
    pitch_feature: str = "dim_0"
    scale: str = "c_major"
    octaves: int = 2
    velocity_feature: Optional[str] = "dim_1"
    velocity_range: tuple[int, int] = (50, 110)
    duration_feature: Optional[str] = "length"
    duration_range: tuple[float, float] = (0.5, 2.0)
    tercet_accent: bool = True
    verse_rest: float = 0.5
    tercet_rest: float = 1.0
    percentiles: tuple[float, float] = (2, 98)

    def __post_init__(self):
        self.bounds: dict[str, tuple[float, float]] = {}

    def _features(self) -> list[str]:
        names = [self.pitch_feature, self.velocity_feature, self.duration_feature]
        return [name for name in names if name is not None]

    def fit(self, features: pd.DataFrame) -> "Mapping":
        """Learn each feature's range, ignoring outliers beyond the percentiles."""
        for name in self._features():
            low, high = np.percentile(features[name], self.percentiles)
            self.bounds[name] = (float(low), float(high))
        return self

    def scaled(self, features: pd.DataFrame, name: str) -> np.ndarray:
        """Feature values rescaled to [0, 1] using the fitted range."""
        if name not in self.bounds:
            raise RuntimeError(f"feature {name!r} not fitted; call fit() first")
        low, high = self.bounds[name]
        if high == low:
            return np.full(len(features), 0.5)
        return np.clip((features[name].to_numpy() - low) / (high - low), 0.0, 1.0)

    def render(self, features: pd.DataFrame) -> list[Note]:
        """Turn feature rows (in order) into a monophonic melody."""
        n = len(features)
        pitches = scale_pitches(self.scale, self.octaves)
        pitch_idx = np.minimum((self.scaled(features, self.pitch_feature) * len(pitches)).astype(int),
                               len(pitches) - 1)

        if self.velocity_feature:
            low, high = self.velocity_range
            velocities = low + self.scaled(features, self.velocity_feature) * (high - low)
        else:
            velocities = np.full(n, 90.0)

        if self.duration_feature:
            low, high = self.duration_range
            durations = low + self.scaled(features, self.duration_feature) * (high - low)
            durations = np.maximum(np.round(durations * 4) / 4, 0.25)  # sixteenth-note grid
        else:
            durations = np.ones(n)

        # A row closes a verse/tercet when the next row belongs to a different one.
        line = features[["cantica", "canto", "line"]].astype(str).agg("|".join, axis=1).to_numpy()
        tercet = features[["cantica", "canto", "tercet"]].astype(str).agg("|".join, axis=1).to_numpy()
        verse_end = np.append(line[1:] != line[:-1], True)
        tercet_end = np.append(tercet[1:] != tercet[:-1], True)

        notes, t = [], 0.0
        for i in range(n):
            velocity, duration = velocities[i], float(durations[i])
            if self.tercet_accent and tercet_end[i]:
                velocity, duration = velocity + 20, duration * 2
            notes.append(Note(t, pitches[pitch_idx[i]], int(min(round(velocity), 127)), duration))
            t += duration
            if tercet_end[i]:
                t += self.tercet_rest
            elif verse_end[i]:
                t += self.verse_rest
        return notes
