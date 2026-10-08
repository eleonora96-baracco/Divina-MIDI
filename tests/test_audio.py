import wave

import numpy as np
import pytest

from divina_midi import audio
from divina_midi.mapping import Note
from divina_midi.midi import PIANO, STRINGS, write_tracks


def test_write_wav_roundtrip(tmp_path):
    samples = np.array([[0.0, 0.5], [-0.5, 1.5]], dtype=np.float32)  # 1.5 gets clipped
    path = audio.write_wav(samples, tmp_path / "x.wav", sample_rate=8000)
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (2, 2, 8000)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert pcm.tolist() == [0, 16383, -16383, 32767]


@pytest.mark.skipif(not audio.DEFAULT_SOUNDFONT.exists(), reason="SoundFont not downloaded")
def test_render_audio_follows_the_midi_timing(tmp_path):
    pytest.importorskip("tinysoundfont")
    melody = [Note(0, 60, 100, 1), Note(2, 64, 100, 1)]  # a rest between the notes
    chords = [Note(0, 48, 60, 3)]
    midi = write_tracks([(melody, PIANO), (chords, STRINGS)], tmp_path / "x.mid", tempo_bpm=60)

    samples = audio.render_audio(midi, sample_rate=8000, tail=1.0)
    assert samples.shape == (4 * 8000, 2)  # 3 beats at 60 bpm + 1 second of tail
    assert 0.01 < np.abs(samples).max() <= audio.PEAK
