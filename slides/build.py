"""Build slides/index.html from the fragments in slides/src/.

Standard library only. The fragments hold one <section> per slide, in the
split of tex/slides/. This script adds the title slide, the outline and the
outline divider that opens parts 2 to 5, numbers the main slides, and
writes one self-contained page that loads everything from this folder, so
it opens from file:// with no network.

It stops with an error if a link points to a slide that does not exist or
an image file is missing.

Usage: python slides/build.py
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"

MAIN = [
    "00-opening",
    "01-framework",
    "02-binary",
    "03-multiclass-geometry",
    "04-partition-optimization",
    "05-voronoi",
    "06-multiclass-experiments",
    "07-sensitivity",
    "08-conclusion",
    "99-references",
]
BACKUP = ["90-backup", "91-proofs"]

TITLE_SLIDE = """<section class="title-slide">
<img class="title-bg" src="figures/static/bg-ime.svg" alt="">
<div class="title-box">
<h1>Argmax is not enough</h1>
<p class="subtitle">Empirical threshold and partition optimization<br>for binary and multiclass classification</p>
<p class="author">Carlo Domenico Longo de Lemos</p>
<p class="date">São Paulo, October 23, 2026</p>
<div class="institute">
<p>Graduate Program in Probability and Statistics, IME-USP</p>
<div class="people">
<strong>Advisor:</strong><span>Prof. Dr. Renato Vicente</span>
<strong>Committee:</strong><span>Prof. Dr. Renato Vicente (IME-USP, chair)<br>Prof. Dr. Rafael Izbicki (UFSCar)<br>Prof. Dr. Tiago Mendonça dos Santos (Insper)</span>
</div>
</div>
<p class="logos"><img class="logo-ime" src="figures/static/ime-logo.svg" alt="IME"><img class="logo-usp" src="figures/static/usp-logo.svg" alt="USP"></p>
</div>
<img class="logo-cc" src="figures/static/cc-by.svg" alt="CC BY">
</section>"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Argmax is not enough</title>
<link rel="stylesheet" href="vendor/reveal/reset.css">
<link rel="stylesheet" href="vendor/reveal/reveal.css">
<link rel="stylesheet" href="theme/ime.css">
</head>
<body>
<div class="reveal">
<div class="slides">
%SLIDES%
</div>
</div>
<pre id="check-report" hidden></pre>
<script src="theme/mathjax-config.js"></script>
<script src="vendor/mathjax/tex-svg-full.js"></script>
<script src="vendor/reveal/reveal.js"></script>
<script src="vendor/reveal/notes/notes.js"></script>
%SCRIPTS%
<script src="theme/deck.js"></script>
</body>
</html>
"""


def read_parts(names: list[str]) -> str:
    return "\n\n".join(
        (SRC / f"{n}.html").read_text(encoding="utf-8") for n in names
    )


def outline(
    toc: list[tuple[str, list[str]]],
    current: int | None,
    sub: int | None = None,
) -> str:
    """The roadmap in two columns: parts 1 and 3 on the left, 2 and 4 on the right."""

    def part(k: int) -> str:
        title, subs = toc[k]
        cls = (
            ""
            if current is None
            else (" current" if k == current else " faded")
        )
        mark = ' class="sub-current"'
        items = "".join(
            f"<li{mark if k == current and j == sub else ''}>{s}</li>"
            for j, s in enumerate(subs)
        )
        inner = f"<ul>{items}</ul>" if subs else ""
        return f'<li class="part{cls}"><span class="num">{k + 1}</span>{title}{inner}</li>'

    left = "".join(part(k) for k in range(0, len(toc), 2))
    right = "".join(part(k) for k in range(1, len(toc), 2))
    return (
        f'<section class="outline">\n<h2>Roadmap</h2>\n<div class="body">\n'
        f'<div class="columns mid"><ol class="col toc">{left}</ol><ol class="col toc">{right}</ol></div>\n'
        f"</div>\n</section>"
    )


def main() -> int:
    main_src = read_parts(MAIN)
    toc: list[tuple[str, list[str]]] = []
    for kind, title in re.findall(
        r"<!--@(section|subsection) (.*?)-->", main_src
    ):
        if kind == "section":
            toc.append((title, []))
        else:
            toc[-1][1].append(title)

    # Each part after the first opens with the roadmap, its own entry highlighted,
    # and so does every later subsection, with the subsection in bold.
    part = -1
    sub = -1

    def open_part(m: re.Match) -> str:
        nonlocal part, sub
        if m.group(1) == "section":
            part += 1
            sub = -1
            return f"<!--part {part}-->" + (outline(toc, part) if part else "")
        sub += 1
        # The first subsection of a part follows its own roadmap (or, for part 1, the overview).
        return outline(toc, part, sub) if sub > 0 else ""

    main_src = re.sub(
        r"<!--@(section|subsection) .*?-->\n*", open_part, main_src
    )
    # The motivation and the contributions come before the outline.
    opening = (SRC / "00-motivation.html").read_text(encoding="utf-8")
    slides = "\n\n".join([TITLE_SLIDE, opening, outline(toc, None), main_src])

    # Number the main slides and tag each with its part, for the header map.
    total = len(re.findall(r"<section\b", slides))
    count, part = 0, -1

    def tag(m: re.Match) -> str:
        nonlocal count, part
        if m.group(0).startswith("<!--part"):
            part = int(m.group(1))
            return ""
        count += 1
        name = (
            html.escape(re.sub(r"\$", "", toc[part][0]), quote=True)
            if part >= 0
            else ""
        )
        return f'<section data-n="{count}" data-total="{total}" data-part-name="{name}"{m.group(2)}>'

    slides = re.sub(r"<!--part (\d+)-->|<section\b([^>]*)>", tag, slides)
    backup = re.sub(r"<section\b", "<section data-backup", read_parts(BACKUP))
    slides = slides + "\n\n" + backup

    # A formula and the punctuation that follows it stay on one line.
    slides = re.sub(
        r"(\(?\$[^$<]+\$)([,.;:)]*)",
        lambda m: (
            m.group(0)
            if m.group(0).startswith("$") and not m.group(2)
            else f'<span class="nb">{m.group(1)}{m.group(2)}</span>'
        ),
        slides,
    )

    errors = []
    ids = set(re.findall(r'<section[^>]*\bid="([^"]+)"', slides))
    for target in sorted(set(re.findall(r'href="#/([^"]+)"', slides)) - ids):
        errors.append(f"link to a missing slide: {target}")
    for src in sorted(set(re.findall(r'src="([^"]+)"', slides))):
        if not (HERE / src).exists():
            errors.append(f"missing file: {src}")
    if errors:
        print("\n".join(errors))
        return 1

    scripts = [p for p in sorted((HERE / "figures" / "data").glob("*.js"))]
    scripts += [HERE / "figures" / "js" / "common.js"]
    scripts += [
        p
        for p in sorted((HERE / "figures" / "js").glob("*.js"))
        if p.name != "common.js"
    ]
    tags = "\n".join(
        f'<script src="{p.relative_to(HERE).as_posix()}"></script>'
        for p in scripts
        if p.exists()
    )

    page = PAGE.replace("%SLIDES%", slides).replace("%SCRIPTS%", tags)
    (HERE / "index.html").write_text(page, encoding="utf-8", newline="\n")
    n_all = len(re.findall(r"<section\b", slides))
    print(f"index.html: {total} main slides, {n_all - total} backup slides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
