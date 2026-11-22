"""Typographic conventions of the dissertation that new text could break.

- Run-in paragraph headings (``\\paragraph{...}``) end with a period.
- Panel labels in captions, such as ``(a)~``, are bold: ``\\textbf{(a)}~``.
- Every definition environment emphasizes the term it defines (``\\emph``).
- Common terms are lowercase: the capitalized forms in LOWERCASE_TERMS do
  not appear outside figures.
"""

from __future__ import annotations

import bisect
import re
from pathlib import Path

from texstream import StreamLine, read_stream, rel

PARAGRAPH_RE = re.compile(r"\\paragraph\*?\{((?:[^{}]|\{[^{}]*\})*)\}")
CAPTION_RE = re.compile(r"\\caption(?:\[[^\]]*\])?\{")
PANEL_RE = re.compile(r"\(([a-h])\)~")
DEFINITION_RE = re.compile(
    r"\\begin\{definition\}(?:\[[^\]]*\])?(.*?)\\end\{definition\}", re.S
)
TIKZ_RE = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.S)
# Capitalized forms of common terms that the text writes in lowercase
# (proper names such as Bayes, Platt and Dirichlet keep their capital).
LOWERCASE_TERMS = (
    "Bayes Optimal",
    "Bayes Risk",
    "Conditional Risk",
    "Expected Risk",
    "Cost Matrix",
    "Dominant Label Constraint",
    "Probability Ratio Thresholding",
    "Parallel Shift",
    "Temperature Scaling",
    "Platt Scaling",
    "Isotonic Regression",
    "Dirichlet Calibration",
    "Expected Calibration Error",
    "Logarithmic Loss",
    "Brier Score",
)


def style_problems(stream: list[StreamLine]) -> list[str]:
    """One message per heading or panel label that breaks the conventions."""
    text = "\n".join(line.text for line in stream)
    starts, pos = [], 0
    for line in stream:
        starts.append(pos)
        pos += len(line.text) + 1

    def where(offset: int) -> str:
        line = stream[bisect.bisect_right(starts, offset) - 1]
        return f"{rel(line.file)}:{line.lineno}"

    problems = []
    for match in PARAGRAPH_RE.finditer(text):
        heading = " ".join(match.group(1).split())
        if not heading.endswith((".", "?", "!")):
            problems.append(
                f"{where(match.start())}: paragraph heading '{heading}' "
                "does not end with a period"
            )
    for match in CAPTION_RE.finditer(text):
        depth, end = 1, match.end()
        while depth and end < len(text):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            end += 1
        body = text[match.end() : end - 1]
        for panel in PANEL_RE.finditer(body):
            if not body[: panel.start()].endswith("\\textbf{"):
                problems.append(
                    f"{where(match.end() + panel.start())}: panel label "
                    f"({panel.group(1)}) in a caption is not bold"
                )
    for match in DEFINITION_RE.finditer(text):
        if "\\emph{" not in match.group(1):
            problems.append(
                f"{where(match.start())}: definition does not emphasize "
                "the term it defines"
            )
    prose = TIKZ_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    for term in LOWERCASE_TERMS:
        for match in re.finditer(rf"\b{re.escape(term)}\b", prose):
            problems.append(
                f"{where(match.start())}: write '{term}' in lowercase "
                "(proper names keep their capital)"
            )
    return problems


def _stream(*lines: str) -> list[StreamLine]:
    return [StreamLine(Path("ch.tex"), i, t) for i, t in enumerate(lines, 1)]


def test_flags_capitalized_common_terms_outside_figures():
    stream = _stream(
        r"The Bayes Optimal rule and the Bayes-optimal rule.",
        r"\begin{tikzpicture}\node {Cost Matrix};\end{tikzpicture}",
        r"Temperature scaling and Platt scaling.",
    )
    assert style_problems(stream) == [
        "ch.tex:1: write 'Bayes Optimal' in lowercase (proper names keep "
        "their capital)"
    ]


def test_flags_definitions_without_an_emphasized_term():
    stream = _stream(
        r"\begin{definition}[A]A set is \emph{open} if it is.\end{definition}",
        r"\begin{definition}[B]",
        r"A set is closed if its complement is open.",
        r"\end{definition}",
    )
    assert style_problems(stream) == [
        "ch.tex:2: definition does not emphasize the term it defines"
    ]


def test_flags_headings_without_period_and_plain_panel_labels():
    stream = _stream(
        r"\paragraph{Contributions} The main contributions are:",
        r"\paragraph{Search budget.} Every strategy uses $T = 301$.",
        r"\caption{Two panels. \textbf{(a)}~First; \textbf{(b)}~second.}",
        r"\caption{Regions: \textbf{(a)}~Default,",
        r"(b)~Voronoi.}",
    )
    assert style_problems(stream) == [
        "ch.tex:1: paragraph heading 'Contributions' does not end with "
        "a period",
        "ch.tex:5: panel label (b) in a caption is not bold",
    ]


def test_dissertation_follows_the_typographic_conventions():
    problems = style_problems(read_stream())
    assert not problems, "\n".join(problems)
