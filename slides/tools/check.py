"""Checks of the HTML deck.

1. Build: every formula of the fragments in slides/src/ must be in the
   built index.html, unchanged, and every link to a backup slide must have
   a target.
2. Browser: headless Chrome loads index.html?check and reports the slides
   whose content overflows and the formulas MathJax could not read.

Usage: python slides/tools/check.py [--no-browser]
Exit status 1 if anything is missing, overflows or fails to typeset.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
MATH = r"\\begin\{(align\*|gather\*)\}.*?\\end\{\1\}|\\\[.*?\\\]|(?<!\\)\$.*?(?<!\\)\$"


def formulas(text: str) -> Counter:
    # The outline slides are generated from the section markers.
    text = re.sub(
        r'<!--@.*?-->|<section[^>]*class="outline".*?</section>',
        "",
        text,
        flags=re.S,
    )
    return Counter(
        re.sub(r"\s+", "", m.group(0))
        for m in re.finditer(MATH, html.unescape(text), re.S)
    )


def build() -> int:
    fragments = "".join(
        p.read_text(encoding="utf-8")
        for p in sorted((HERE / "src").glob("*.html"))
    )
    page = (
        (HERE / "index.html")
        .read_text(encoding="utf-8")
        .split('<pre id="check-report"')[0]
    )
    bad = 0
    a, b = formulas(fragments), formulas(page)
    for item, n in ((a - b) + (b - a)).items():
        print(
            f"index.html math differs from the fragments ({n}x): {item[:90]}"
        )
        bad += 1
    ids = set(re.findall(r'<section[^>]*\bid="([^"]+)"', page))
    for target in sorted(set(re.findall(r'href="#/([^"]+)"', page)) - ids):
        print(f"link to a missing slide: {target}")
        bad += 1
    slides = len(re.findall(r"<section\b", page))
    notes = len(
        re.findall(
            r"<li>",
            "".join(
                re.findall(r'<aside class="notes">.*?</aside>', page, re.S)
            ),
        )
    )
    statements = len(
        re.findall(
            r'<div class="block (?:definition|lemma|proposition|theorem)"',
            page,
        )
    )
    print(
        f"build: {slides} slides, {statements} statements, {sum(a.values())} formulas, {notes} note lines"
        f" -> {'ok' if not bad else f'{bad} problems'}"
    )
    return bad


def browser() -> int:
    url = (HERE / "index.html").as_uri() + "?check"
    with tempfile.TemporaryDirectory() as profile:
        run = subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--virtual-time-budget=30000",
                f"--user-data-dir={profile}",
                "--dump-dom",
                url,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
    m = re.search(r'<pre id="check-report"[^>]*>(.*?)</pre>', run.stdout, re.S)
    if not m or not m.group(1).strip():
        print("browser: the deck did not finish loading")
        return 1
    rep = json.loads(html.unescape(m.group(1)))
    for o in rep["overflow"]:
        print(
            f"overflow: slide {o['index']} ({o['title']}): {o['dy']}px down, {o['dx']}px right"
        )
    print(
        f"browser: {rep['slides']} slides, {rep['live']} interactive figures, "
        f"{len(rep['overflow'])} overflowing, {rep['mathErrors']} math errors"
    )
    return len(rep["overflow"]) + rep["mathErrors"]


def main() -> int:
    bad = build()
    if "--no-browser" not in sys.argv:
        bad += browser()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
