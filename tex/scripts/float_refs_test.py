"""Every figure and table of the dissertation is labelled and referenced."""

from __future__ import annotations

import re
from pathlib import Path

from texstream import StreamLine, read_stream, rel

FLOAT_RE = re.compile(
    r"\\begin\{(figure\*?|table\*?|longtable)\}(.*?)\\end\{\1\}", re.S
)
REF_RE = re.compile(r"\\(?:[cC]ref|[aA]utoref|ref|pageref)\*?\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")

# Floats that are deliberately not cited from the text.
EXEMPT = {
    "tab:notation",  # the Notation chapter consists of this table alone
}


def float_problems(stream: list[StreamLine]) -> list[str]:
    """One message per float that has no label or is never referenced."""
    text = "\n".join(line.text for line in stream)
    starts, pos = [], 0
    for line in stream:
        starts.append(pos)
        pos += len(line.text) + 1

    def where(offset: int) -> str:
        i = max(k for k, s in enumerate(starts) if s <= offset)
        return f"{rel(stream[i].file)}:{stream[i].lineno}"

    referenced = {
        label.strip()
        for match in REF_RE.finditer(text)
        for label in match.group(1).split(",")
    }
    problems = []
    for match in FLOAT_RE.finditer(text):
        labels = LABEL_RE.findall(match.group(2))
        if not labels:
            problems.append(
                f"{where(match.start())}: {match.group(1)} has no \\label"
            )
        elif not (set(labels) & (referenced | EXEMPT)):
            names = ", ".join(labels)
            problems.append(
                f"{where(match.start())}: {names} is never referenced"
            )
    return problems


def _stream(*lines: str) -> list[StreamLine]:
    return [StreamLine(Path("ch.tex"), i, t) for i, t in enumerate(lines, 1)]


def test_flags_missing_labels_and_unreferenced_floats():
    stream = _stream(
        r"See Figure~\ref{fig:a} and Tables~\cref{tab:b,tab:c}.",
        r"\begin{figure}\caption{A}\label{fig:a}\end{figure}",
        r"\begin{table}\caption{B}\label{tab:b}\end{table}",
        r"\begin{table}\caption{No label}\end{table}",
        r"\begin{figure}\caption{D}\label{fig:d}\end{figure}",
        r"\begin{longtable}{l}\caption{N}\label{tab:notation}\\\end{longtable}",
    )
    assert float_problems(stream) == [
        "ch.tex:4: table has no \\label",
        "ch.tex:5: fig:d is never referenced",
    ]


def test_every_figure_and_table_is_referenced():
    problems = float_problems(read_stream())
    assert not problems, "\n".join(problems)
