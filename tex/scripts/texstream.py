"""Flatten the dissertation's \\input tree into a single reading-order stream.

Shared by the tex audit scripts:
  - find_forward_refs.py   (\\ref targets defined later in the stream)
  - ../../lean/scripts/render_table.py  (theorem environments -> Lean status)

The stream is the sequence of source lines you would read if every \\input
were expanded in place, with LaTeX comments stripped. Each line keeps its
originating file and line number so callers can report real source locations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TEX_ROOT = Path(__file__).resolve().parent.parent
MASTER = TEX_ROOT / "carlo_master.tex"

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(ref|eqref)\{([^}]+)\}")


def strip_comment(line: str) -> str:
    """Remove the trailing LaTeX comment, honoring backslash escapes."""
    i = 0
    while i < len(line):
        if line[i] == "%":
            backslashes = 0
            j = i - 1
            while j >= 0 and line[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                return line[:i]
        i += 1
    return line


def resolve_input(arg: str, tex_root: Path = TEX_ROOT) -> Path | None:
    """Resolve an \\input argument to a file under tex_root, if it exists."""
    arg = arg.strip()
    if arg.startswith("./"):
        arg = arg[2:]
    candidates = [tex_root / arg, tex_root / f"{arg}.tex"]
    for c in candidates:
        if c.is_file():
            return c
    return None


@dataclass
class StreamLine:
    file: Path
    lineno: int
    text: str


def flatten(
    path: Path,
    visited: set[Path],
    stream: list[StreamLine],
    tex_root: Path = TEX_ROOT,
) -> None:
    """Append path's comment-stripped lines to stream, expanding \\input."""
    resolved = path.resolve()
    if resolved in visited:
        return
    visited.add(resolved)
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        raw = path.read_text(encoding="latin-1").splitlines()
    for lineno, raw_line in enumerate(raw, start=1):
        text = strip_comment(raw_line)
        m = INPUT_RE.search(text)
        if m:
            child = resolve_input(m.group(1), tex_root)
            if child is not None:
                flatten(child, visited, stream, tex_root)
            else:
                # Unresolved \input - keep the line so a stray \ref or \label
                # outside the missing file still gets recorded.
                stream.append(StreamLine(path, lineno, text))
            continue
        stream.append(StreamLine(path, lineno, text))


def read_stream(
    master: Path = MASTER, tex_root: Path = TEX_ROOT
) -> list[StreamLine]:
    """Flatten master into a reading-order stream of source lines."""
    stream: list[StreamLine] = []
    flatten(master, set(), stream, tex_root)
    return stream


def rel(path: Path, tex_root: Path = TEX_ROOT) -> str:
    """Repo-relative, forward-slashed path for stable report output."""
    try:
        return str(path.relative_to(tex_root.parent)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
