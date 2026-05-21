"""Audit \\ref / \\eqref forward references in the dissertation.

Walks tex/carlo_master.tex, recursively flattens every \\input into a single
reading-order stream, then reports every citation site whose target \\label is
defined later in that stream.

Outputs:
  - stdout: one-line summary + grouped listing by citing file.
  - JSON:   tex/scripts/forward_refs_report.json (machine-readable detail).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

TEX_ROOT = Path(__file__).resolve().parent.parent
MASTER = TEX_ROOT / "carlo_master.tex"
JSON_OUT = Path(__file__).resolve().parent / "forward_refs_report.json"

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


def resolve_input(arg: str) -> Path | None:
    arg = arg.strip()
    if arg.startswith("./"):
        arg = arg[2:]
    candidates = [TEX_ROOT / arg, TEX_ROOT / f"{arg}.tex"]
    for c in candidates:
        if c.is_file():
            return c
    return None


@dataclass
class StreamLine:
    file: Path
    lineno: int
    text: str


def flatten(path: Path, visited: set[Path], stream: list[StreamLine]) -> None:
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
            child = resolve_input(m.group(1))
            if child is not None:
                flatten(child, visited, stream)
            else:
                # Unresolved \input - keep the line so a stray \ref or \label
                # outside the missing file still gets recorded.
                stream.append(StreamLine(path, lineno, text))
            continue
        stream.append(StreamLine(path, lineno, text))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(TEX_ROOT.parent)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> int:
    stream: list[StreamLine] = []
    flatten(MASTER, set(), stream)

    labels: dict[str, dict] = {}
    duplicates: list[dict] = []
    citations: list[dict] = []

    for idx, item in enumerate(stream):
        for m in LABEL_RE.finditer(item.text):
            name = m.group(1)
            entry = {
                "label": name,
                "file": rel(item.file),
                "line": item.lineno,
                "reading_index": idx,
            }
            if name in labels:
                duplicates.append(entry)
            else:
                labels[name] = entry
        for m in REF_RE.finditer(item.text):
            citations.append(
                {
                    "command": m.group(1),
                    "label": m.group(2),
                    "file": rel(item.file),
                    "line": item.lineno,
                    "reading_index": idx,
                }
            )

    forward: list[dict] = []
    undefined: list[dict] = []
    for c in citations:
        target = labels.get(c["label"])
        if target is None:
            undefined.append(c)
            continue
        if c["reading_index"] < target["reading_index"]:
            forward.append(
                {
                    "citation": {
                        k: c[k] for k in ("file", "line", "command", "label")
                    },
                    "definition": {
                        k: target[k] for k in ("file", "line", "label")
                    },
                }
            )

    report = {
        "summary": {
            "stream_lines": len(stream),
            "labels": len(labels),
            "citations": len(citations),
            "forward_refs": len(forward),
            "undefined": len(undefined),
            "duplicate_labels": len(duplicates),
        },
        "forward_refs": forward,
        "undefined": undefined,
        "duplicate_labels": duplicates,
    }

    JSON_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"forward refs: {len(forward)} | undefined: {len(undefined)} | "
        f"duplicate labels: {len(duplicates)}"
    )

    if forward:
        print()
        print("Forward references (citation -> later \\label):")
        by_file: dict[str, list[dict]] = {}
        for f in forward:
            by_file.setdefault(f["citation"]["file"], []).append(f)
        for file in sorted(by_file):
            print(f"\n  {file}")
            for f in sorted(
                by_file[file], key=lambda x: x["citation"]["line"]
            ):
                c = f["citation"]
                d = f["definition"]
                print(
                    f"    {c['line']:>5}  \\{c['command']}{{{c['label']}}}  "
                    f"->  {d['file']}:{d['line']}"
                )

    if undefined:
        print()
        print("Undefined references:")
        for c in sorted(undefined, key=lambda x: (x["file"], x["line"])):
            print(
                f"  {c['file']}:{c['line']}  \\{c['command']}{{{c['label']}}}"
            )

    if duplicates:
        print()
        print("Duplicate \\label definitions (first definition wins):")
        for d in duplicates:
            print(f"  {d['file']}:{d['line']}  \\label{{{d['label']}}}")

    print(f"\nFull report: {rel(JSON_OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
