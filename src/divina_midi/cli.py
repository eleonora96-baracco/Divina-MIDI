"""Command line interface: ``divina-midi <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import corpus


def cmd_download(args: argparse.Namespace) -> None:
    df = corpus.build_corpus(delay=args.delay)
    path = corpus.save_corpus(df, args.out)
    print(f"Saved {len(df)} verses from {df.groupby(['cantica', 'canto']).ngroups} canti to {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="divina-midi")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download the text from Wikisource")
    download.add_argument("--out", type=Path, default=corpus.DEFAULT_PATH)
    download.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    download.set_defaults(func=cmd_download)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
