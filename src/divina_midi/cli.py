"""Command line interface: ``divina-midi <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import corpus
from .mapping import SCALES, Mapping
from .midi import write_midi


def cmd_download(args: argparse.Namespace) -> None:
    df = corpus.build_corpus(delay=args.delay)
    path = corpus.save_corpus(df, args.out)
    print(f"Saved {len(df)} verses from {df.groupby(['cantica', 'canto']).ngroups} canti to {path}")


def cmd_render(args: argparse.Namespace) -> None:
    from .features import word2vec

    text = corpus.load_corpus(args.corpus)
    model = word2vec.train(text, seed=args.seed)
    extract = word2vec.word_features if args.unit == "word" else word2vec.verse_features

    # Fit the mapping on the whole poem so every canto shares the same scale.
    features = extract(text, model)
    mapping = Mapping(scale=args.scale).fit(features)
    selected = corpus.select(features, cantica=args.cantica, canto=args.canto)
    if selected.empty:
        raise SystemExit(f"No verses for {args.cantica} {args.canto}")

    notes = mapping.render(selected)
    out = args.out or Path("output") / f"{args.cantica.lower()}_{args.canto:02d}_{args.method}_{args.unit}.mid"
    write_midi(notes, out, tempo_bpm=args.tempo)
    print(f"Saved {len(notes)} notes to {out}")


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
    render.add_argument("--method", default="word2vec", choices=["word2vec"])
    render.add_argument("--unit", default="word", choices=["word", "verse"],
                        help="one note per word or per verse")
    render.add_argument("--scale", default="c_major", choices=sorted(SCALES))
    render.add_argument("--tempo", type=float, default=90, help="beats per minute")
    render.add_argument("--seed", type=int, default=42)
    render.add_argument("--corpus", type=Path, default=corpus.DEFAULT_PATH)
    render.add_argument("--out", type=Path, help="output .mid path (default: output/...)")
    render.set_defaults(func=cmd_render)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
