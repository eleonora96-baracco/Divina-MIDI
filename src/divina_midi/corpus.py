"""Download the Divina Commedia from Italian Wikisource and load it as a table.

The text is fetched once through the MediaWiki API and saved as a CSV with one
row per verse:

    cantica | canto | tercet | line | text

where ``tercet`` is the stanza number within the canto (the closing single
verse of each canto counts as its own stanza) and ``line`` is the verse number
within the canto, as printed in the edition.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

API_URL = "https://it.wikisource.org/w/api.php"
SOURCE_URL = "https://it.wikisource.org/wiki/Divina_Commedia"
USER_AGENT = "divina-midi/0.1 (https://github.com/eleonora96-baracco/divina-midi)"

CANTICHE = {"Inferno": 34, "Purgatorio": 33, "Paradiso": 33}
COLUMNS = ["cantica", "canto", "tercet", "line", "text"]
DEFAULT_PATH = Path("data") / "commedia.csv"

_BR = "\x00BR\x00"
_ROMAN = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def to_roman(n: int) -> str:
    """Roman numeral for 1 <= n < 40, enough for canto numbers."""
    out = ""
    for value, numeral in _ROMAN:
        while n >= value:
            out += numeral
            n -= value
    return out


def page_title(cantica: str, canto: int) -> str:
    return f"Divina Commedia/{cantica}/Canto {to_roman(canto)}"


def fetch_canto_html(cantica: str, canto: int, session: requests.Session) -> str:
    """Return the rendered HTML of one canto page."""
    params = {
        "action": "parse",
        "page": page_title(cantica, canto),
        "prop": "text",
        "format": "json",
        "formatversion": 2,
        "redirects": 1,
    }
    response = session.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"{page_title(cantica, canto)}: {data['error'].get('info')}")
    return data["parse"]["text"]


def parse_canto(html: str) -> list[list[str]]:
    """Extract the verses of a canto page, grouped into stanzas.

    Verse numbers, inline styles and markup are dropped; stanzas are the blocks
    separated by blank lines in the poem.
    """
    soup = BeautifulSoup(html, "html.parser")
    poem = soup.find("div", class_="poem")
    if poem is None:
        raise ValueError("no <div class='poem'> in page")

    for tag in poem.find_all(["style", "link"]):
        tag.decompose()
    for tag in poem.find_all("span", class_="numeroriga"):
        tag.decompose()
    # Newlines in the HTML source are just whitespace; only <br/> breaks a line.
    for br in poem.find_all("br"):
        br.replace_with(_BR)
    text = poem.get_text().replace("\n", " ").replace(_BR, "\n").replace("\xa0", " ")
    stanzas = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            stanzas.append(lines)
    return stanzas


def canto_rows(cantica: str, canto: int, stanzas: list[list[str]]) -> list[dict]:
    rows = []
    line = 0
    for tercet, stanza in enumerate(stanzas, start=1):
        for text in stanza:
            line += 1
            rows.append(
                {"cantica": cantica, "canto": canto, "tercet": tercet, "line": line, "text": text}
            )
    return rows


def build_corpus(
    delay: float = 0.5,
    progress: Optional[Callable[[str], None]] = print,
) -> pd.DataFrame:
    """Download all 100 canti and return them as a DataFrame."""
    rows = []
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        for cantica, n_canti in CANTICHE.items():
            for canto in range(1, n_canti + 1):
                if progress:
                    progress(f"{cantica} {to_roman(canto)}")
                stanzas = parse_canto(fetch_canto_html(cantica, canto, session))
                rows.extend(canto_rows(cantica, canto, stanzas))
                time.sleep(delay)
    return pd.DataFrame(rows, columns=COLUMNS)


def save_corpus(df: pd.DataFrame, path: Path = DEFAULT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def load_corpus(path: Path = DEFAULT_PATH, download: bool = True) -> pd.DataFrame:
    """Load the corpus from ``path``, downloading it first if it is missing."""
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found; run `divina-midi download`")
        save_corpus(build_corpus(), path)
    return pd.read_csv(path, encoding="utf-8")


def select(
    df: pd.DataFrame, cantica: Optional[str] = None, canto: Optional[int] = None
) -> pd.DataFrame:
    """Filter the corpus by cantica (case-insensitive) and/or canto number."""
    mask = pd.Series(True, index=df.index)
    if cantica is not None:
        mask &= df["cantica"].str.lower() == cantica.lower()
    if canto is not None:
        mask &= df["canto"] == canto
    return df[mask]
