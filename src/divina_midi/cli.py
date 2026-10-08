"""Command line interface: ``divina-midi <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import corpus
from .mapping import MODES, SCALES, AffectMapping, Mapping
from .midi import PIANO, STRINGS, write_midi, write_tracks
from .text import words

METHODS = ["word2vec", "embeddings", "sentiment"]


def cmd_download(args: argparse.Namespace) -> None:
    df = corpus.build_corpus(delay=args.delay)
    path = corpus.save_corpus(df, args.out)
    print(f"Saved {len(df)} verses from {df.groupby(['cantica', 'canto']).ngroups} canti to {path}")


def default_midi_path(cantica: str, canto: int, method: str, unit: str = "word") -> Path:
    suffix = method if method == "sentiment" else f"{method}_{unit}"
    return Path("output") / f"{cantica.lower()}_{canto:02d}_{suffix}.mid"


def render_canto(text: pd.DataFrame, cantica: str, canto: int, method: str, unit: str = "word",
                 scale: str = "c_major", tempo: float = 90, seed: int = 42,
                 out: Path | None = None) -> Path:
    """Render one canto with one of the three methods and return the MIDI path."""
    if corpus.select(text, cantica=cantica, canto=canto).empty:
        raise SystemExit(f"No verses for {cantica} {canto}")
    out = out or default_midi_path(cantica, canto, method, unit)
    if method == "sentiment":
        render_sentiment(text, cantica, canto, tempo, out)
    else:
        render_vectors(text, cantica, canto, method, unit, scale, tempo, seed, out)
    return out


def render_sentiment(text, cantica: str, canto: int, tempo: float, out: Path) -> None:
    from .features import sentiment

    # First run scores the whole poem (several minutes); later runs read the cache.
    affect = sentiment.affect(text)
    mapping = AffectMapping().fit(affect)
    selected = corpus.select(affect, cantica=cantica, canto=canto)
    melody, chords = mapping.render(words(corpus.select(text, cantica=cantica, canto=canto)), selected)

    write_tracks([(melody, PIANO), (chords, STRINGS)], out, tempo_bpm=tempo)
    modes = mapping.plan(selected)["mode"].value_counts()
    print(f"Saved {len(melody)} notes and {len(chords) // 3} chords to {out}")
    print("Modes used:", ", ".join(f"{mode} {count}" for mode, count in modes.items()))


def render_vectors(text, cantica: str, canto: int, method: str, unit: str, scale: str,
                   tempo: float, seed: int, out: Path) -> None:
    """Word2Vec or contextual embeddings: two vector features drive pitch and velocity."""
    if method == "word2vec":
        from .features import word2vec

        model = word2vec.train(text, seed=seed)
        extract = word2vec.word_features if unit == "word" else word2vec.verse_features
        features = extract(text, model)
        pitch, velocity = "dim_0", "dim_1"
    else:
        from .features import embeddings

        # First run embeds the whole poem (a few minutes); later runs read the cache.
        extract = embeddings.word_features if unit == "word" else embeddings.verse_features
        features = extract(text)
        pitch, velocity = "pc_0", "pc_1"

    # Fit the mapping on the whole poem so every canto shares the same scale.
    mapping = Mapping(pitch_feature=pitch, velocity_feature=velocity, scale=scale).fit(features)
    notes = mapping.render(corpus.select(features, cantica=cantica, canto=canto))
    write_midi(notes, out, tempo_bpm=tempo)
    print(f"Saved {len(notes)} notes to {out}")


def cmd_render(args: argparse.Namespace) -> None:
    render_canto(corpus.load_corpus(args.corpus), args.cantica, args.canto, args.method, args.unit,
                 args.scale, args.tempo, args.seed, args.out)


def cmd_audio(args: argparse.Namespace) -> None:
    from . import audio

    paths = []
    for path in args.paths:
        paths += sorted(path.glob("*.mid")) if path.is_dir() else [path]
    if not paths:
        raise SystemExit("No .mid files found")

    soundfont = audio.ensure_soundfont(args.soundfont)
    for path in paths:
        samples = audio.render_audio(path, soundfont)
        if args.seconds:
            samples = audio.excerpt(samples, args.seconds)
        print(f"Saved {audio.write_audio(samples, path.with_suffix('.' + args.format))}")


def arc_data(text: pd.DataFrame) -> dict:
    """Per-canto valence and per-cantica mode shares, for the demo page charts."""
    from .features import sentiment

    affect = sentiment.affect(text)
    by_canto = affect.groupby(["cantica", "canto"], sort=False)["valence"].mean().reset_index()
    plan = AffectMapping().fit(affect).plan(affect)
    shares = pd.crosstab(plan["cantica"], plan["mode"], normalize="index")
    return {
        "valence": [{"cantica": row.cantica, "canto": int(row.canto), "valence": round(row.valence, 2)}
                    for row in by_canto.itertuples()],
        "modes": list(MODES),
        "modeShares": {cantica: [round(float(shares.loc[cantica].get(mode, 0)), 3) for mode in MODES]
                       for cantica in corpus.CANTICHE},
    }


def cmd_demo(args: argparse.Namespace) -> None:
    """Render the demo: the first canto of each cantica with each method, as MP3 excerpts."""
    from . import audio

    text = corpus.load_corpus(args.corpus)
    soundfont = audio.ensure_soundfont()
    for cantica in corpus.CANTICHE:
        for method in METHODS:
            midi = render_canto(text, cantica, 1, method)
            clip = audio.excerpt(audio.render_audio(midi, soundfont), args.seconds)
            print(f"Saved {audio.write_audio(clip, args.out / 'audio' / midi.with_suffix('.mp3').name)}")

    data = args.out / "data.js"
    data.write_text("window.DEMO_DATA = " + json.dumps(arc_data(text)) + ";\n", encoding="utf-8")
    print(f"Saved {data}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="divina-midi")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download the text from Wikisource")
    download.add_argument("--out", type=Path, default=corpus.DEFAULT_PATH)
    download.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    download.set_defaults(func=cmd_download)

    render = sub.add_parser("render", help="turn a canto into a MIDI file")
    render.add_argument("--cantica", required=True, choices=[c.lower() for c in corpus.CANTICHE],
                        type=str.lower)
    render.add_argument("--canto", required=True, type=int)
    render.add_argument("--method", default="word2vec", choices=METHODS,
                        help="embeddings and sentiment need the 'sentiment' extra")
    render.add_argument("--unit", default="word", choices=["word", "verse"],
                        help="word2vec/embeddings: one note per word or per verse")
    render.add_argument("--scale", default="c_major", choices=sorted(SCALES),
                        help="word2vec/embeddings (with sentiment the text picks the mode)")
    render.add_argument("--tempo", type=float, default=90, help="beats per minute")
    render.add_argument("--seed", type=int, default=42)
    render.add_argument("--corpus", type=Path, default=corpus.DEFAULT_PATH)
    render.add_argument("--out", type=Path, help="output .mid path (default: output/...)")
    render.set_defaults(func=cmd_render)

    to_audio = sub.add_parser("audio", help="render MIDI files to audio with a SoundFont")
    to_audio.add_argument("paths", nargs="*", type=Path, default=[Path("output")],
                          help=".mid files or folders (default: output/)")
    to_audio.add_argument("--format", default="wav", choices=["wav", "mp3"])
    to_audio.add_argument("--seconds", type=float, help="keep only the first N seconds, fading out")
    to_audio.add_argument("--soundfont", type=Path, default=Path("data") / "soundfonts" / "GeneralUser-GS.sf2",
                          help="SoundFont file (the default one is downloaded on first use)")
    to_audio.set_defaults(func=cmd_audio)

    demo = sub.add_parser("demo", help="build the audio excerpts and chart data of the demo page")
    demo.add_argument("--out", type=Path, default=Path("docs"))
    demo.add_argument("--seconds", type=float, default=60, help="length of each excerpt")
    demo.add_argument("--corpus", type=Path, default=corpus.DEFAULT_PATH)
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
