"""Every acronym is spelled out where it first appears.

Reading starts at the introduction (the abstracts and the notation table
are summaries) and follows the \\input tree, captions and tables included.
The first occurrence of an acronym must come with its expansion, either as
``expansion (ACR)`` or as ``ACR (expansion)``. Product names, units and
common computing terms are listed in EXEMPT.
"""

from __future__ import annotations

import bisect
import re
from pathlib import Path

from texstream import StreamLine, read_stream, rel

EXEMPT = {
    # Roman numerals.
    "II", "III", "IV",
    # Metric and model names used as names: F-scores are defined by their
    # formulas, LGBM and XGB are library names.
    "F1", "F2", "LGBM", "XGB",
    # Hardware, software, units and currencies (Appendix D).
    "CPU", "GPU", "RAM", "GB", "LTS", "NVIDIA", "HBM", "CUDA", "VM", "BLAS",
    "USD", "BRL", "H200",
    # Institutions, formats and common computing terms.
    "IME", "USP", "PDF", "AI", "ID", "API", "IDE",
}  # fmt: skip
ACRONYM_RE = re.compile(
    r"(?<![A-Za-z0-9\\])([A-Z][A-Z0-9]*[A-Z0-9])s?(?![A-Za-z0-9])"
)
MASK_RES = [
    re.compile(p, re.S)
    for p in (
        r"\$\$.*?\$\$",
        r"\$[^$]*\$",
        r"\\\[.*?\\\]",
        r"\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}",
        r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
        r"\\(?:label|ref|eqref|cref|Cref|autoref|pageref|cite[pt]?"
        r"|citeauthor|citeyear|texttt|url|href|input|includegraphics)"
        r"\*?(?:\[[^\]]*\])*\{(?:[^{}]|\{[^{}]*\})*\}",
    )
]


def acronym_problems(stream: list[StreamLine]) -> list[str]:
    """One message per acronym whose first occurrence is not defined."""
    text = "\n".join(line.text for line in stream)
    for mask in MASK_RES:
        text = mask.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    starts, pos = [], 0
    for line in stream:
        starts.append(pos)
        pos += len(line.text) + 1

    def where(offset: int) -> str:
        line = stream[bisect.bisect_right(starts, offset) - 1]
        return f"{rel(line.file)}:{line.lineno}"

    seen: set[str] = set()
    problems = []
    for match in ACRONYM_RE.finditer(text):
        acronym = match.group(1)
        if acronym in EXEMPT or acronym in seen:
            continue
        seen.add(acronym)
        before = text[max(0, match.start() - 1) : match.start()]
        after = text[match.end() : match.end() + 30]
        in_parens = before == "(" and after.startswith(")")
        followed_by_expansion = (
            re.match(r"\s*\([A-Za-z][a-z]", after) is not None
        )
        if in_parens or followed_by_expansion:
            continue
        problems.append(
            f"{where(match.start())}: {acronym} is used before it is "
            "spelled out"
        )
    return problems


def _stream(*lines: str) -> list[StreamLine]:
    return [StreamLine(Path("ch.tex"), i, t) for i, t in enumerate(lines, 1)]


def test_flags_acronyms_used_before_their_expansion():
    stream = _stream(
        r"We report the standard error (SE) and the ROC curve.",
        r"The SE shrinks; the ROC (receiver operating characteristic) too.",
        r"The DLC (dominant label constraint) holds on the GPU.",
        r"Actor-critic methods (PPO, SAC) and CIs ($\pm$ SE) are lists.",
        r"Math such as $\mathrm{AUC}$ and \texttt{CSV\_{}X} is ignored.",
    )
    assert acronym_problems(stream) == [
        "ch.tex:1: ROC is used before it is spelled out",
        "ch.tex:4: PPO is used before it is spelled out",
        "ch.tex:4: SAC is used before it is spelled out",
        "ch.tex:4: CI is used before it is spelled out",
    ]


def test_every_acronym_is_spelled_out_at_first_use():
    stream = read_stream()
    first = next(
        i
        for i, line in enumerate(stream)
        if "00-introduction" in str(line.file)
    )
    problems = acronym_problems(stream[first:])
    assert not problems, "\n".join(problems)
