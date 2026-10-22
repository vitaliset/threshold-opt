"""Screenshots of the HTML deck with headless Chrome, and contact sheets.

Usage: python slides/tools/shots.py [--query static] [--only 3,17] [--out DIR] [--pdf]

Writes one PNG per slide to slides/.build/shots/ (or DIR) and contact
sheets of twelve slides each, to be read by eye. With --pdf the screenshots
are also bound into slides/.build/slides-html.pdf, one page per slide: the
fallback copy of the deck, showing each figure in its default state.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def shoot(index: int, out: Path, query: str, size: str) -> Path:
    png = out / f"s{index:02d}.png"
    url = (
        (HERE / "index.html").as_uri()
        + (f"?{query}" if query else "")
        + f"#/{index}"
    )
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={size}",
            "--virtual-time-budget=15000",
            f"--user-data-dir={out / 'profiles' / ('p%02d' % index)}",
            f"--screenshot={png}",
            url,
        ],
        capture_output=True,
        timeout=180,
    )
    return png


def sheets(pngs: list[Path], out: Path, per: int = 12, cols: int = 3) -> None:
    from PIL import Image

    for k in range(0, len(pngs), per):
        group = [Image.open(p) for p in pngs[k : k + per] if p.exists()]
        if not group:
            continue
        w, h = 640, 360
        rows = -(-len(group) // cols)
        sheet = Image.new("RGB", (w * cols, h * rows), "white")
        for i, im in enumerate(group):
            sheet.paste(
                im.resize((w, h), Image.LANCZOS),
                ((i % cols) * w, (i // cols) * h),
            )
        sheet.save(out / f"sheet{k // per:02d}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(HERE / ".build" / "shots"))
    ap.add_argument("--size", default="1600,900")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    n = len(
        re.findall(
            r"<section\b", (HERE / "index.html").read_text(encoding="utf-8")
        )
    )
    todo = (
        [int(x) for x in args.only.split(",")] if args.only else list(range(n))
    )
    with ThreadPoolExecutor(6) as pool:
        pngs = list(
            pool.map(lambda i: shoot(i, out, args.query, args.size), todo)
        )
    if not args.only:
        sheets(pngs, out)
    if args.pdf:
        from PIL import Image

        pages = [Image.open(p).convert("RGB") for p in pngs]
        target = HERE / ".build" / "slides-html.pdf"
        pages[0].save(
            target, save_all=True, append_images=pages[1:], resolution=120.0
        )
        print(f"{len(pages)} pages in {target}")
    print(f"{len(pngs)} screenshots in {out}")


if __name__ == "__main__":
    main()
