"""Render MIDI files to audio with a SoundFont, so they sound the same everywhere.

A MIDI file only holds instructions; how it sounds depends on the synthesizer
that plays it (the one built into Windows is notoriously harsh). Rendering with
a fixed SoundFont gives a WAV file that sounds the same on any device and can
be shared or embedded in a web page.

Requires the ``audio`` extra: ``pip install -e ".[audio]"``.
"""

from __future__ import annotations

import wave
from pathlib import Path

import mido
import numpy as np
import requests

# GeneralUser GS by S. Christian Collins: free, including for music made with it.
SOUNDFONT_URL = "https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2"
DEFAULT_SOUNDFONT = Path("data") / "soundfonts" / "GeneralUser-GS.sf2"
SAMPLE_RATE = 44100
GAIN_DB = 12.0  # the SoundFont is quiet at default gain; renders peak around 0.85
PEAK = 0.95  # never louder than this (full scale = 1)


def ensure_soundfont(path: Path = DEFAULT_SOUNDFONT) -> Path:
    """Download the default SoundFont (32 MB) on first use."""
    path = Path(path)
    if not path.exists():
        if path != DEFAULT_SOUNDFONT:
            raise FileNotFoundError(path)
        print(f"Downloading SoundFont to {path} (32 MB)...")
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(SOUNDFONT_URL, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def render_audio(midi_path: Path, soundfont: Path = DEFAULT_SOUNDFONT,
                 sample_rate: int = SAMPLE_RATE, tail: float = 3.0) -> np.ndarray:
    """Synthesize a MIDI file; returns stereo float samples of shape (n, 2).

    Events are replayed in time order and audio is generated between them, so
    tempo and timing are exactly those of the file. ``tail`` seconds are added
    at the end to let the last notes ring.
    """
    import tinysoundfont

    synth = tinysoundfont.Synth(gain=GAIN_DB, samplerate=sample_rate)
    font = synth.sfload(str(soundfont))

    chunks, pending = [], 0.0

    def advance(seconds: float) -> None:
        nonlocal pending
        pending += seconds
        samples = int(pending * sample_rate)
        if samples > 0:
            # generate() returns a view into a reused buffer: copy it.
            chunks.append(np.frombuffer(synth.generate(samples), dtype=np.float32).copy())
            pending -= samples / sample_rate

    for message in mido.MidiFile(midi_path):  # message.time = seconds since the previous one
        advance(message.time)
        if message.type == "program_change":
            synth.program_select(message.channel, font, 0, message.program)
        elif message.type == "note_on" and message.velocity > 0:
            synth.noteon(message.channel, message.note, message.velocity)
        elif message.type in ("note_on", "note_off"):
            synth.noteoff(message.channel, message.note)
    advance(tail)

    audio = np.concatenate(chunks).reshape(-1, 2)
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > PEAK:  # only turn down, so relative loudness between files is kept
        audio *= PEAK / peak
    return audio


def write_wav(audio: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Save stereo float samples as a 16-bit WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())
    return path
