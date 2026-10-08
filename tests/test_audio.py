import wave

import numpy as np
import pytest

from divina_midi import audio
from divina_midi.mapping import Note
from divina_midi.midi import PIANO, STRINGS, write_tracks


def test_write_wav_roundtrip(tmp_path):
    pytest.importorskip("soundfile")
    samples = np.array([[0.0, 0.5], [-0.5, 1.5]], dtype=np.float32)  # 1.5 gets clipped
    path = audio.write_audio(samples, tmp_path / "x.wav", sample_rate=8000)
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (2, 2, 8000)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert pcm.tolist() == [0, 16384, -16384, 32767]


def test_excerpt_trims_and_fades_out():
    samples = np.ones((10 * 100, 2), dtype=np.float32)
    clip = audio.excerpt(samples, seconds=4, fade=2, sample_rate=100)
    assert clip.shape == (400, 2)
    assert clip[0, 0] == 1 and clip[199, 0] == 1  # untouched before the fade
    assert clip[-1, 0] == 0 and 0.4 < clip[300, 0] < 0.6
    assert samples[399, 0] == 1  # the input is not modified


@pytest.mark.skipif(not audio.DEFAULT_SOUNDFONT.exists(), reason="SoundFont not downloaded")
def test_render_audio_follows_the_midi_timing(tmp_path):
    pytest.importorskip("tinysoundfont")
    melody = [Note(0, 60, 100, 1), Note(2, 64, 100, 1)]  # a rest between the notes
    chords = [Note(0, 48, 60, 3)]
    midi = write_tracks([(melody, PIANO), (chords, STRINGS)], tmp_path / "x.mid", tempo_bpm=60)

    samples = audio.render_audio(midi, sample_rate=8000, tail=1.0)
    assert samples.shape == (4 * 8000, 2)  # 3 beats at 60 bpm + 1 second of tail
    assert 0.01 < np.abs(samples).max() <= audio.PEAK
