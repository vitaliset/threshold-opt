"""Find repeated phrases in the dissertation LaTeX prose.

Strips LaTeX (math, citations, refs, structural commands, environments like
``tikzpicture`` and ``tabular``) and reports n-grams that appear more than a
threshold number of times. Useful for spotting AI-style repetition where the
same multi-word phrase recurs across chapters.

Usage:
    uv run python tex/scripts/find_repeated_phrases.py
    uv run python tex/scripts/find_repeated_phrases.py \\
        --files tex/content/06-voronoi-tessellations.tex
    uv run python tex/scripts/find_repeated_phrases.py \\
        --min-n 4 --max-n 8 --min-count 3 --show-locations
    uv run python tex/scripts/find_repeated_phrases.py \\
        --rank score --top 60
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

TEX_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOB = "content/*.tex"


# ----- LaTeX stripping ----------------------------------------------------

# Math environments: drop body entirely.
_MATH_ENVS = (
    "equation",
    "align",
    "gather",
    "multline",
    "displaymath",
    "eqnarray",
    "array",
    "split",
    "cases",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "smallmatrix",
)

# Non-prose environments: drop body entirely.
_NON_PROSE_ENVS = (
    "tikzpicture",
    "tabular",
    "tabularx",
    "longtable",
    "verbatim",
    "lstlisting",
    "minted",
)

# Commands whose single brace-arg is prose to keep.
_TEXT_BEARING_CMDS = (
    "textbf",
    "emph",
    "textit",
    "textsc",
    "textrm",
    "textsf",
    "texttt",
    "text",
    "mathrm",
    "caption",
    "paragraph",
    "subparagraph",
    "section",
    "subsection",
    "subsubsection",
    "chapter",
    "section*",
    "subsection*",
    "href",
    "url",
    "footnote",
)

# Commands to delete together with all of their {...} and [...] arguments
# (so file paths, lengths, refs, etc. do not leak into tokenization).
_DROP_WITH_ARGS = (
    "input",
    "include",
    "includegraphics",
    "resizebox",
    "scalebox",
    "rotatebox",
    "raisebox",
    "setlength",
    "addtolength",
    "vspace",
    "hspace",
    "label",
    "ref",
    "eqref",
    "cref",
    "Cref",
    "autoref",
    "pageref",
    "nameref",
    "cite",
    "citep",
    "citet",
    "citeauthor",
    "citeyear",
    "nocite",
    "bibliography",
    "bibliographystyle",
    "usepackage",
    "documentclass",
    "newcommand",
    "renewcommand",
    "providecommand",
    "DeclareMathOperator",
    "newenvironment",
    "AtBeginDocument",
    "newtheorem",
    "geometry",
    "hypersetup",
    "tikzset",
)

_DROP_CMDS_RE = re.compile(
    rf"\\(?:{'|'.join(_DROP_WITH_ARGS)})\*?(?![a-zA-Z])"
)


def _drop_command_with_args(text: str) -> str:
    """Remove ``\\cmd[...]{...}{...}...`` for each cmd in _DROP_WITH_ARGS,
    consuming all balanced bracket/brace groups that follow the command.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        m = _DROP_CMDS_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        i = m.end()
        # Skip a run of {...} and [...] groups (and inter-group whitespace).
        while i < n:
            while i < n and text[i] in " \t":
                i += 1
            if i >= n:
                break
            if text[i] == "{":
                depth, i = 1, i + 1
                while i < n and depth:
                    if text[i] == "\\" and i + 1 < n:
                        i += 2
                        continue
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                    i += 1
            elif text[i] == "[":
                depth, i = 1, i + 1
                while i < n and depth:
                    if text[i] == "[":
                        depth += 1
                    elif text[i] == "]":
                        depth -= 1
                    i += 1
            else:
                break
        out.append(" ")
    return "".join(out)


def strip_latex(text: str) -> str:
    # Line comments (% ... EOL), respecting \%.
    text = re.sub(r"(?<!\\)%.*", " ", text)

    # Math / non-prose environments.
    for env in _MATH_ENVS + _NON_PROSE_ENVS:
        text = re.sub(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}",
            " ",
            text,
            flags=re.DOTALL,
        )

    # Display math \[...\] and $$...$$.
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)

    # Inline math: $...$ and \(...\). $ is not escaped inside math, but
    # \$ in prose is a literal dollar; handle that by tolerating \$.
    text = re.sub(r"(?<!\\)\$(?:\\.|[^$\\])*\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.DOTALL)

    # Drop commands with their bracket/brace arguments (file paths, refs, ...).
    text = _drop_command_with_args(text)

    # Replace \cmd{X} (text-bearing) with X. Iterate to unwind nesting.
    _text_cmd_alt = "|".join(re.escape(c) for c in _TEXT_BEARING_CMDS)
    cmd_re = re.compile(rf"\\(?:{_text_cmd_alt})\*?\{{([^{{}}]*)\}}")
    for _ in range(5):
        new = cmd_re.sub(r" \1 ", text)
        if new == text:
            break
        text = new

    # Strip remaining \command tokens (keep any brace contents).
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)
    text = re.sub(r"\\[^a-zA-Z\s]", " ", text)

    # Drop leftover structural characters.
    text = re.sub(r"[{}~^_]", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


# ----- n-gram counting ----------------------------------------------------


def ngrams(tokens: list[str], n: int):
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _find_occurrences(phrase: str, text: str, max_hits: int) -> list[int]:
    pattern = re.compile(
        r"\W+".join(re.escape(t) for t in phrase.split()),
        flags=re.IGNORECASE,
    )
    out = []
    for m in pattern.finditer(text):
        out.append(_line_of(text, m.start()))
        if len(out) >= max_hits:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find repeated multi-word phrases in the dissertation."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="LaTeX files to scan (default: tex/content/*.tex).",
    )
    parser.add_argument(
        "--min-n", type=int, default=5, help="Shortest n-gram (default 5)."
    )
    parser.add_argument(
        "--max-n", type=int, default=8, help="Longest n-gram (default 8)."
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Minimum occurrences (default 3).",
    )
    parser.add_argument(
        "--top", type=int, default=40, help="Top phrases per n (default 40)."
    )
    parser.add_argument(
        "--rank",
        choices=("count", "score"),
        default="count",
        help="Sort by raw count, or by count*length (default count).",
    )
    parser.add_argument(
        "--show-locations",
        action="store_true",
        help="Print file:line for each occurrence.",
    )
    parser.add_argument(
        "--max-locations",
        type=int,
        default=6,
        help="Max occurrences per phrase with --show-locations (default 6).",
    )
    parser.add_argument(
        "--stripped-out",
        type=Path,
        default=None,
        help="Dump the stripped prose of each file to this directory (debug).",
    )
    args = parser.parse_args()

    if args.files:
        paths = [Path(p) for p in args.files]
    else:
        paths = sorted(TEX_ROOT.glob(DEFAULT_GLOB))

    per_file_tokens: dict[Path, list[str]] = {}
    original_text: dict[Path, str] = {}

    for path in paths:
        raw = path.read_text(encoding="utf-8")
        original_text[path] = raw
        prose = strip_latex(raw)
        per_file_tokens[path] = tokenize(prose)
        if args.stripped_out is not None:
            args.stripped_out.mkdir(parents=True, exist_ok=True)
            (args.stripped_out / (path.stem + ".txt")).write_text(
                prose, encoding="utf-8"
            )

    total_tokens = sum(len(t) for t in per_file_tokens.values())
    print(f"Scanned {len(paths)} files, {total_tokens:,} tokens (post-strip).")

    for n in range(args.min_n, args.max_n + 1):
        counter: Counter[tuple[str, ...]] = Counter()
        by_file: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(
            Counter
        )
        for path, toks in per_file_tokens.items():
            for ng in ngrams(toks, n):
                counter[ng] += 1
                by_file[ng][path.name] += 1

        repeated = [
            (ng, c) for ng, c in counter.items() if c >= args.min_count
        ]
        if args.rank == "score":
            repeated.sort(key=lambda kv: (-(kv[1] * n), -kv[1], kv[0]))
        else:
            repeated.sort(key=lambda kv: (-kv[1], kv[0]))

        # Suppress n-grams that are fully contained in a longer reported phrase
        # of the same count: their information already came up.
        kept = []
        seen_longer: set[tuple[str, ...]] = set()
        for ng, c in repeated[: args.top * 4]:
            phrase = " ".join(ng)
            if any(phrase in " ".join(s) for s in seen_longer):
                continue
            kept.append((ng, c))
            seen_longer.add(ng)
            if len(kept) >= args.top:
                break

        print(
            f"\n=== {n}-grams with count >= {args.min_count}"
            f" (top {len(kept)}) ==="
        )
        if not kept:
            print("  (none)")
            continue

        for ng, c in kept:
            phrase = " ".join(ng)
            files_summary = ", ".join(
                f"{f} x{k}" for f, k in by_file[ng].most_common()
            )
            print(f"  [{c:3d}] {phrase}")
            print(f"        in: {files_summary}")
            if args.show_locations:
                for path in paths:
                    if path.name not in by_file[ng]:
                        continue
                    hits = _find_occurrences(
                        phrase, original_text[path], args.max_locations
                    )
                    if hits:
                        loc = ", ".join(f"{path.name}:{i}" for i in hits)
                        print(f"        @ {loc}")


if __name__ == "__main__":
    main()
