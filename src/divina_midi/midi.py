"""Write notes to a standard MIDI file."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import mido

from .mapping import Note

TICKS_PER_BEAT = 480
PIANO, STRINGS = 0, 48  # General MIDI programs


def _track(notes: Iterable[Note], program: int, channel: int) -> mido.MidiTrack:
    events = []  # (tick, order, message): note_off sorts before note_on at the same tick
    for note in notes:
        start = round(note.start * TICKS_PER_BEAT)
        end = start + max(round(note.duration * TICKS_PER_BEAT), 1)
        events.append((start, 1, mido.Message("note_on", channel=channel, note=note.pitch,
                                               velocity=note.velocity)))
        events.append((end, 0, mido.Message("note_off", channel=channel, note=note.pitch, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1]))

    track = mido.MidiTrack()
    track.append(mido.Message("program_change", channel=channel, program=program))
    previous = 0
    for tick, _, message in events:
        track.append(message.copy(time=tick - previous))
        previous = tick
    return track


def write_tracks(tracks: Sequence[tuple[Iterable[Note], int]], path: Path,
                 tempo_bpm: float = 90) -> Path:
    """Save several ``(notes, program)`` parts, each on its own track and channel."""
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    tempo = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm))])
    midi.tracks.append(tempo)
    for channel, (notes, program) in enumerate(tracks):
        midi.tracks.append(_track(notes, program, channel))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
    return path


def write_midi(notes: Iterable[Note], path: Path, tempo_bpm: float = 90,
               program: int = PIANO) -> Path:
    """Save a single part (``program`` 0 = piano)."""
    return write_tracks([(notes, program)], path, tempo_bpm)
