"""Typographic conventions of the dissertation that new text could break.

- Run-in paragraph headings (``\\paragraph{...}``) end with a period.
- Panel labels in captions, such as ``(a)~``, are bold: ``\\textbf{(a)}~``.
- Every definition environment emphasizes the term it defines (``\\emph``).
- Common terms are lowercase: the capitalized forms in LOWERCASE_TERMS do
  not appear outside figures.
- Displayed equations are part of the sentence: no colon right before one,
  a period at its end when the sentence ends there (new paragraph, end of a
  statement or proof, or a capitalized word next), and a comma before a
  following "where".
- The text speaks in the first person plural ("we", "nós" in the resumo).
  The singular ("I", "my", "me", "eu", "meu") appears only in the personal
  statement of SINGULAR_FILES; the reader is never addressed as "you", and
  the author never refers to themself as "the author".
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
COLON_DISPLAY_RE = re.compile(
    r":\s*(?:\\label\{[^}]*\}\s*)?"
    r"(?:\\begin\{(?:equation|align|gather|multline)\*?\}|\\\[|\$\$)"
)
DISPLAY_RE = re.compile(
    r"\\begin\{(equation|align|gather|multline)(\*?)\}(.*?)\\end\{\1\2\}"
    r"|\\\[(.*?)\\\]",
    re.S,
)
# Text after a display that ends the sentence: a blank line, the end of a
# statement, proof or list, or a new structural command.
SENTENCE_END_RE = re.compile(
    r"[ \t]*\n[ \t]*\n|\s*\\(?:end\{(?:definition|theorem|lemma|proposition"
    r"|corollary|example|assumption|remark|proof|enumerate|itemize)\}"
    r"|item\b|(?:sub)*section\b|paragraph\b|begin\{(?:figure|table)\})"
)
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
# The appendix on how AI was used is a personal statement, in the singular.
SINGULAR_FILES = ("appendix-f-ai-disclosure.tex",)
SINGULAR_RE = re.compile(
    r"(?<![\\A-Za-zÀ-ÿ])(I|[Mm]y|[Mm]e|[Mm]ine|[Mm]yself"
    r"|[Ee]u|[Mm]eus?|[Mm]inhas?|[Mm]im)(?![A-Za-zÀ-ÿ])"
)
# Never allowed: addressing the reader, or the author in the third person.
OTHER_PERSON_RE = re.compile(
    r"(?<![\\A-Za-zÀ-ÿ])([Yy]ou|[Yy]our|[Yy]ours|[Yy]ourself|[Yy]ourselves"
    r"|[Vv]ocês?|[Tt]he author|[Tt]his author|[Tt]he present author"
    r"|[Tt]he writer|[Oo] autor|[Aa] autora)(?![A-Za-zÀ-ÿ])"
)
MATH_RE = re.compile(
    r"\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]"
    r"|\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}",
    re.S,
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
    for match in COLON_DISPLAY_RE.finditer(text):
        problems.append(
            f"{where(match.start())}: colon right before a displayed equation"
        )
    for match in DISPLAY_RE.finditer(text):
        math = match.group(3) if match.group(3) is not None else match.group(4)
        math = re.sub(r"\\label\{[^}]*\}|\\nonumber|\\notag", "", math)
        math = re.sub(r"(\\\\|\\[,;!]|\\q?quad|\s)+$", "", math)
        # The final mark may sit inside \text{...}, before closing braces or
        # before the end of an inner environment such as cases or split.
        last = re.sub(r"(\\end\{[a-z]+\*?\}|\\\\|[\s}])+$", "", math)[-1:]
        after = text[match.end() : match.end() + 200]
        word = re.match(r"\s*([A-Za-z]+)", after)
        ends = SENTENCE_END_RE.match(after) is not None or (
            word is not None and word.group(1)[0].isupper()
        )
        if ends and last != ".":
            problems.append(
                f"{where(match.start())}: displayed equation ends a sentence "
                "without a period"
            )
        elif not ends and word and word.group(1) == "where" and last != ",":
            problems.append(
                f"{where(match.start())}: displayed equation needs a comma "
                "before 'where'"
            )
        elif not ends and last == ".":
            problems.append(
                f"{where(match.start())}: displayed equation ends with a "
                "period but the sentence continues"
            )
    no_math = MATH_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    no_math = TIKZ_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), no_math)
    for match in SINGULAR_RE.finditer(no_math):
        line = stream[bisect.bisect_right(starts, match.start()) - 1]
        if line.file.name not in SINGULAR_FILES:
            problems.append(
                f"{where(match.start())}: first person singular "
                f"'{match.group(1)}'; the text uses 'we'"
            )
    for match in OTHER_PERSON_RE.finditer(no_math):
        problems.append(
            f"{where(match.start())}: '{match.group(1)}'; the text uses 'we'"
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


def test_flags_first_person_singular_outside_the_personal_statement():
    stream = [
        StreamLine(Path("ch.tex"), 1, r"We derive the rule; I checked it."),
        StreamLine(
            Path("ch.tex"), 2, r"Let $\mathbb{I}$ and $\mathbf{I}$ be."
        ),
        StreamLine(Path("appendix-f-ai-disclosure.tex"), 1, r"My reading."),
        StreamLine(Path("ch.tex"), 3, r"As you can see, the author agrees."),
        StreamLine(Path("ch.tex"), 4, r"Nesta dissertação eu mostro o meu."),
        StreamLine(
            Path("ch.tex"), 5, r"The authors of [3] agree with Youden."
        ),
    ]
    assert style_problems(stream) == [
        "ch.tex:1: first person singular 'I'; the text uses 'we'",
        "ch.tex:4: first person singular 'eu'; the text uses 'we'",
        "ch.tex:4: first person singular 'meu'; the text uses 'we'",
        "ch.tex:3: 'you'; the text uses 'we'",
        "ch.tex:3: 'the author'; the text uses 'we'",
    ]


def test_flags_colons_before_displayed_equations():
    stream = _stream(
        r"The risk is:",
        r"\begin{equation}R = 1\end{equation}",
        r"and the bound reads, with probability $1-\nu$,",
        r"\[ R \le 2 \]",
    )
    assert style_problems(stream) == [
        "ch.tex:1: colon right before a displayed equation"
    ]


def test_flags_displayed_equations_punctuated_unlike_text():
    stream = _stream(
        r"The risk is",
        r"\begin{equation}R = 1\end{equation}",
        r"The next sentence starts here, and the loss is",
        r"\[ L = 0, \]",
        r"where $L$ is the loss. The cases read",
        r"\[ f = \begin{cases} 1 & \text{otherwise}. \end{cases} \]",
        r"",
        r"Then the bound is \[ B = 2. \] for all $m$, and also \[ C = 3 \]",
        r"where $C$ is constant.",
    )
    assert style_problems(stream) == [
        "ch.tex:2: displayed equation ends a sentence without a period",
        "ch.tex:8: displayed equation ends with a period but the sentence "
        "continues",
        "ch.tex:8: displayed equation needs a comma before 'where'",
    ]


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
    stream = read_stream()
    # Start at the abstracts: the title pages and the catalog record before
    # them are template material (the record numbers its entries I, II).
    first = next(
        i for i, line in enumerate(stream) if line.file.name == "abstract.tex"
    )
    problems = style_problems(stream[first:])
    assert not problems, "\n".join(problems)
