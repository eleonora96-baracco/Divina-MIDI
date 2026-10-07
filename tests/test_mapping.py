import pandas as pd
import pytest

from divina_midi.mapping import Mapping, scale_pitches
from divina_midi.midi import write_midi


def make_features():
    """Two tercets of word rows: tercet 1 has verses 1-3, tercet 2 has verse 4."""
    lines = [1, 1, 2, 2, 3, 4, 4]
    tercets = [1, 1, 1, 1, 1, 2, 2]
    return pd.DataFrame({
        "cantica": "Inferno", "canto": 1, "tercet": tercets, "line": lines,
        "dim_0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "dim_1": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        "length": [1, 1, 1, 1, 1, 1, 1],
    })


def test_scale_pitches():
    assert scale_pitches("c_major", octaves=1) == [60, 62, 64, 65, 67, 69, 71, 72]
    assert scale_pitches("a_minor", octaves=1)[:3] == [57, 59, 60]


def test_pitch_follows_feature_within_scale():
    features = make_features()
    mapping = Mapping(percentiles=(0, 100), tercet_accent=False).fit(features)
    pitches = [note.pitch for note in mapping.render(features)]
    assert pitches == sorted(pitches)
    assert pitches[0] == 60 and pitches[-1] == 84
    assert set(pitches) <= set(scale_pitches("c_major"))


def test_outliers_are_clipped_to_the_fitted_range():
    features = make_features()
    mapping = Mapping(percentiles=(0, 100)).fit(features.iloc[1:6])
    scaled = mapping.scaled(features, "dim_0")
    assert scaled[0] == 0.0 and scaled[-1] == 1.0


def test_render_requires_fit():
    with pytest.raises(RuntimeError):
        Mapping().render(make_features())


def test_tercet_ends_are_accented_and_followed_by_rests():
    features = make_features()
    mapping = Mapping(velocity_feature=None, duration_feature=None, verse_rest=0.5,
                      tercet_rest=1.0).fit(features)
    notes = mapping.render(features)

    # Last word of each tercet (rows 4 and 6) is louder and twice as long.
    assert [n.duration for n in notes] == [1, 1, 1, 1, 2, 1, 2]
    assert [n.velocity for n in notes] == [90, 90, 90, 90, 110, 90, 110]
    # Rests: 0.5 beat after verses 1 and 2, 1 beat after the tercet.
    assert [n.start for n in notes] == [0, 1, 2.5, 3.5, 5, 8, 9]


def test_write_midi_roundtrip(tmp_path):
    mido = pytest.importorskip("mido")
    features = make_features()
    notes = Mapping().fit(features).render(features)
    path = write_midi(notes, tmp_path / "out.mid")

    on = [m for m in mido.MidiFile(path).tracks[0] if m.type == "note_on"]
    assert [m.note for m in on] == [n.pitch for n in notes]
    assert [m.velocity for m in on] == [n.velocity for n in notes]
