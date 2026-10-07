"""Write notes to a standard MIDI file."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import mido

from .mapping import Note

TICKS_PER_BEAT = 480


def write_midi(notes: Iterable[Note], path: Path, tempo_bpm: float = 90,
               program: int = 0) -> Path:
    """Save ``notes`` as a single-track MIDI file (``program`` 0 = piano)."""
    events = []  # (tick, order, message): note_off sorts before note_on at the same tick
    for note in notes:
        start = round(note.start * TICKS_PER_BEAT)
        end = start + max(round(note.duration * TICKS_PER_BEAT), 1)
        events.append((start, 1, mido.Message("note_on", note=note.pitch, velocity=note.velocity)))
        events.append((end, 0, mido.Message("note_off", note=note.pitch, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1]))

    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm)))
    track.append(mido.Message("program_change", program=program))
    previous = 0
    for tick, _, message in events:
        track.append(message.copy(time=tick - previous))
        previous = tick

    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    midi.tracks.append(track)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
    return path
