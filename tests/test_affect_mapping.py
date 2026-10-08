import pandas as pd
import pytest

from divina_midi.mapping import MODES, REGISTERS, AffectMapping, mode_pitch
from divina_midi.midi import PIANO, STRINGS, write_tracks
from divina_midi.text import words


def make_canto(cantica="Inferno", valences=(-5.0, 0.0, 5.0)):
    """A canto of len(valences) - 1 tercets plus the closing verse, with given valences."""
    rows, line = [], 0
    for tercet in range(1, len(valences) + 1):
        for _ in range(1 if tercet == len(valences) else 3):
            line += 1
            rows.append({"cantica": cantica, "canto": 1, "tercet": tercet, "line": line,
                         "text": "nel mezzo del cammin"})
    corpus = pd.DataFrame(rows)
    affect = pd.DataFrame({"cantica": cantica, "canto": 1, "tercet": range(1, len(valences) + 1),
                           "valence": valences, "arousal": [0.0, 1.0, 2.0][:len(valences)]})
    return corpus, affect


def test_modes_go_from_dark_to_bright():
    # Each mode raises exactly one degree of the previous one.
    steps = list(MODES.values())
    for darker, brighter in zip(steps, steps[1:]):
        diffs = [b - a for a, b in zip(darker, brighter)]
        assert sorted(diffs) == [0] * 6 + [1]


def test_mode_pitch_wraps_octaves():
    assert mode_pitch(60, "ionian", 0) == 60
    assert mode_pitch(60, "ionian", 7) == 72
    assert mode_pitch(60, "aeolian", 2) == 63
    assert mode_pitch(60, "ionian", -1) == 59


def test_smoothing_stays_within_a_canto():
    _, affect = make_canto()
    other = affect.assign(canto=2, valence=100.0)
    smoothed = AffectMapping(smoothing=3).smooth(pd.concat([affect, other], ignore_index=True))
    assert smoothed["valence"].tolist()[:3] == [-2.5, 0.0, 2.5]


def test_plan_maps_valence_to_mode_and_cantica_to_register():
    _, affect = make_canto(cantica="Paradiso")
    mapping = AffectMapping(smoothing=1, percentiles=(0, 100)).fit(affect)
    plan = mapping.plan(affect)
    assert plan["mode"].tolist() == ["phrygian", "mixolydian", "lydian"]
    assert set(plan["root"]) == {REGISTERS["Paradiso"]}
    # More arousal: louder and faster.
    assert plan["velocity"].is_monotonic_increasing
    assert plan["beat"].is_monotonic_decreasing


def test_render_requires_fit():
    corpus, affect = make_canto()
    with pytest.raises(RuntimeError):
        AffectMapping().render(words(corpus), affect)


def test_tercet_cadence_and_one_chord_per_tercet():
    corpus, affect = make_canto(valences=(0.0, 0.0))  # one tercet + closing verse, same mode
    mapping = AffectMapping(smoothing=1).fit(affect)
    melody, chords = mapping.render(words(corpus), affect)
    plan = mapping.plan(affect).iloc[0]
    root, mode = int(plan["root"]), plan["mode"]

    # 4 words per verse; each verse ends on the fifth, the third, then the root.
    verse_ends = [melody[i].pitch for i in (3, 7, 11, 15)]
    assert verse_ends == [mode_pitch(root, mode, d) for d in (4, 2, 0, 0)]
    # The note closing the tercet is accented.
    assert melody[11].velocity == melody[10].velocity + 15

    assert len(chords) == 2 * 3
    assert [c.pitch for c in chords[:3]] == [mode_pitch(root - 12, mode, d) for d in (0, 2, 4)]
    # The first chord lasts until the tercet's last note has finished, then a rest follows.
    tercet_end = melody[11].start + melody[11].duration
    assert chords[0].start == 0 and chords[0].start + chords[0].duration >= tercet_end
    assert chords[3].start > tercet_end


def test_write_tracks_has_one_track_per_part(tmp_path):
    mido = pytest.importorskip("mido")
    corpus, affect = make_canto()
    melody, chords = AffectMapping().fit(affect).render(words(corpus), affect)
    midi = mido.MidiFile(write_tracks([(melody, PIANO), (chords, STRINGS)], tmp_path / "x.mid"))

    assert len(midi.tracks) == 3  # tempo + melody + chords
    programs = [m.program for t in midi.tracks for m in t if m.type == "program_change"]
    assert programs == [PIANO, STRINGS]
    assert sum(m.type == "note_on" for m in midi.tracks[2]) == len(chords)
