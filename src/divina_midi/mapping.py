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

import math
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


# Diatonic modes from darkest to brightest: each one raises a single degree of
# the previous one, so moving along the list is a gradual change of colour.
MODES = {
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],  # natural minor
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "ionian": [0, 2, 4, 5, 7, 9, 11],  # major
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}
# Root pitch of each cantica: the journey climbs from low to high (D3, A3, E4).
REGISTERS = {"Inferno": 50, "Purgatorio": 57, "Paradiso": 64}
# Scale degree each verse of a tercet ends on: fifth, third, then home (open,
# open, closed), so the terza rima is heard as harmony.
CADENCE = (4, 2, 0)
TERCET_KEY = ["cantica", "canto", "tercet"]


def mode_pitch(root: int, mode: str, degree: int) -> int:
    """MIDI pitch of a scale degree (0 = root; may be negative or above 6)."""
    octave, step = divmod(degree, 7)
    return root + 12 * octave + MODES[mode][step]


@dataclass
class AffectMapping:
    """Map tercet affect (valence, arousal) onto mode, register, speed and dynamics.

    Unlike :class:`Mapping`, every musical choice here has a name: valence picks
    the mode (darker to brighter), arousal sets speed and loudness, the cantica
    sets the register. The melody contour itself only follows the verse
    structure, so what changes between tercets is what the text expresses.
    """

    smoothing: int = 3  # tercets in the moving average
    velocity_range: tuple[int, int] = (55, 105)
    beat_range: tuple[float, float] = (1.0, 0.5)  # beats per word at low / high arousal
    chord_velocity: int = 45
    contour_peak: int = 4  # highest scale degree of the melodic arch
    percentiles: tuple[float, float] = (5, 95)

    def __post_init__(self):
        self.bounds: dict[str, tuple[float, float]] = {}

    def smooth(self, affect: pd.DataFrame) -> pd.DataFrame:
        """Moving average over neighbouring tercets, never across cantos."""
        out = affect.copy()
        by_canto = affect.groupby(["cantica", "canto"], sort=False)
        for name in ("valence", "arousal"):
            out[name] = by_canto[name].transform(
                lambda v: v.rolling(self.smoothing, center=True, min_periods=1).mean())
        return out

    def fit(self, affect: pd.DataFrame) -> "AffectMapping":
        """Learn the range of smoothed valence and arousal, ideally on the whole poem."""
        smoothed = self.smooth(affect)
        for name in ("valence", "arousal"):
            low, high = np.percentile(smoothed[name], self.percentiles)
            self.bounds[name] = (float(low), float(high))
        return self

    def _scaled(self, values: pd.Series, name: str) -> np.ndarray:
        low, high = self.bounds[name]
        if high == low:
            return np.full(len(values), 0.5)
        return np.clip((values.to_numpy() - low) / (high - low), 0.0, 1.0)

    def plan(self, affect: pd.DataFrame) -> pd.DataFrame:
        """The musical decisions for each tercet: mode, root, velocity, beat."""
        if not self.bounds:
            raise RuntimeError("call fit() first")
        plan = self.smooth(affect)[TERCET_KEY + ["valence", "arousal"]]
        valence = self._scaled(plan["valence"], "valence")
        arousal = self._scaled(plan["arousal"], "arousal")

        names = list(MODES)
        plan = plan.assign(
            mode=[names[min(int(v * len(names)), len(names) - 1)] for v in valence],
            root=plan["cantica"].map(REGISTERS),
            velocity=np.round(self.velocity_range[0]
                              + arousal * (self.velocity_range[1] - self.velocity_range[0])).astype(int),
            beat=np.round((self.beat_range[0] + arousal * (self.beat_range[1] - self.beat_range[0])) * 4) / 4,
        )
        return plan

    def render(self, words: pd.DataFrame, affect: pd.DataFrame) -> tuple[list[Note], list[Note]]:
        """Melody (one note per word) and harmony (one sustained chord per tercet)."""
        plan = self.plan(affect).set_index(TERCET_KEY)
        melody, chords, t = [], [], 0.0
        for key, tercet_words in words.groupby(TERCET_KEY, sort=False):
            p = plan.loc[key].to_dict()  # by name: Series.mode is a method
            start = t
            verses = [verse for _, verse in tercet_words.groupby("line", sort=False)]
            for v, verse in enumerate(verses):
                # The single verse closing a canto goes straight home.
                end_degree = 0 if len(verses) == 1 else CADENCE[v % 3]
                closes_tercet = v == len(verses) - 1
                n = len(verse)
                for k, word in enumerate(verse.itertuples(index=False)):
                    last = k == n - 1
                    if last:
                        degree, duration = end_degree, p["beat"] * 2
                    else:
                        # An arch over the verse, nudged by word length for variety.
                        arch = round(self.contour_peak * math.sin(math.pi * (k + 1) / (n + 1)))
                        degree = arch + word.length % 3 - 1
                        duration = p["beat"] * (1.5 if word.length >= 7 else 1)
                    velocity = p["velocity"] + (15 if last and closes_tercet else 0)
                    melody.append(Note(t, mode_pitch(p["root"], p["mode"], degree), int(min(velocity, 127)),
                                       float(duration)))
                    t += duration
                t += p["beat"] / 2  # breath between verses
            chords += [Note(start, mode_pitch(p["root"] - 12, p["mode"], d), self.chord_velocity, t - start)
                       for d in (0, 2, 4)]
            t += p["beat"]  # longer breath between tercets
        return melody, chords
