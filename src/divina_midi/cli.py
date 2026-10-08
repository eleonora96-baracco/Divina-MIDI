"""Command line interface: ``divina-midi <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import corpus
from .mapping import SCALES, AffectMapping, Mapping
from .midi import PIANO, STRINGS, write_midi, write_tracks
from .text import words


def cmd_download(args: argparse.Namespace) -> None:
    df = corpus.build_corpus(delay=args.delay)
    path = corpus.save_corpus(df, args.out)
    print(f"Saved {len(df)} verses from {df.groupby(['cantica', 'canto']).ngroups} canti to {path}")


def cmd_render(args: argparse.Namespace) -> None:
    text = corpus.load_corpus(args.corpus)
    if corpus.select(text, cantica=args.cantica, canto=args.canto).empty:
        raise SystemExit(f"No verses for {args.cantica} {args.canto}")
    if args.method == "sentiment":
        render_sentiment(text, args)
    else:
        render_vectors(text, args)


def render_sentiment(text, args: argparse.Namespace) -> None:
    from .features import sentiment

    # First run scores the whole poem (several minutes); later runs read the cache.
    affect = sentiment.affect(text)
    mapping = AffectMapping().fit(affect)
    selected = corpus.select(affect, cantica=args.cantica, canto=args.canto)
    melody, chords = mapping.render(words(corpus.select(text, cantica=args.cantica, canto=args.canto)),
                                    selected)

    out = args.out or Path("output") / f"{args.cantica}_{args.canto:02d}_sentiment.mid"
    write_tracks([(melody, PIANO), (chords, STRINGS)], out, tempo_bpm=args.tempo)
    modes = mapping.plan(selected)["mode"].value_counts()
    print(f"Saved {len(melody)} notes and {len(chords) // 3} chords to {out}")
    print("Modes used:", ", ".join(f"{mode} {count}" for mode, count in modes.items()))


def render_vectors(text, args: argparse.Namespace) -> None:
    """Word2Vec or contextual embeddings: two vector features drive pitch and velocity."""
    if args.method == "word2vec":
        from .features import word2vec

        model = word2vec.train(text, seed=args.seed)
        extract = word2vec.word_features if args.unit == "word" else word2vec.verse_features
        features = extract(text, model)
        pitch, velocity = "dim_0", "dim_1"
    else:
        from .features import embeddings

        # First run embeds the whole poem (a few minutes); later runs read the cache.
        extract = embeddings.word_features if args.unit == "word" else embeddings.verse_features
        features = extract(text)
        pitch, velocity = "pc_0", "pc_1"

    # Fit the mapping on the whole poem so every canto shares the same scale.
    mapping = Mapping(pitch_feature=pitch, velocity_feature=velocity, scale=args.scale).fit(features)
    notes = mapping.render(corpus.select(features, cantica=args.cantica, canto=args.canto))
    out = args.out or Path("output") / f"{args.cantica.lower()}_{args.canto:02d}_{args.method}_{args.unit}.mid"
    write_midi(notes, out, tempo_bpm=args.tempo)
    print(f"Saved {len(notes)} notes to {out}")


def cmd_audio(args: argparse.Namespace) -> None:
    from . import audio

    paths = []
    for path in args.paths:
        paths += sorted(path.glob("*.mid")) if path.is_dir() else [path]
    if not paths:
        raise SystemExit("No .mid files found")

    soundfont = audio.ensure_soundfont(args.soundfont)
    for path in paths:
        wav = audio.write_wav(audio.render_audio(path, soundfont), path.with_suffix(".wav"))
        print(f"Saved {wav}")


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
    render.add_argument("--method", default="word2vec", choices=["word2vec", "embeddings", "sentiment"],
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

    to_audio = sub.add_parser("audio", help="render MIDI files to WAV with a SoundFont")
    to_audio.add_argument("paths", nargs="*", type=Path, default=[Path("output")],
                          help=".mid files or folders (default: output/)")
    to_audio.add_argument("--soundfont", type=Path, default=Path("data") / "soundfonts" / "GeneralUser-GS.sf2",
                          help="SoundFont file (the default one is downloaded on first use)")
    to_audio.set_defaults(func=cmd_audio)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
