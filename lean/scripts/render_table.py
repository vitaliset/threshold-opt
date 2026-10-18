"""Render the Lean formalization status table for Appendix E.

Joins two sources:
  - the theorem-like environments declared in tex/content/*.tex, discovered by
    flattening the \\input tree in reading order (tex/scripts/texstream.py);
  - lean/audit.json, emitted by `lake exe audit`, which records the axiom
    closure of every declaration tagged @[thesis "<label>"].

Writes tex/tables/formalization.tex. Every statement in the dissertation gets a
row, including the ones with no Lean counterpart, so the table cannot overstate
what is machine-checked.

Usage:
    uv run python lean/scripts/render_table.py            # write the table
    uv run python lean/scripts/render_table.py --check    # fail if stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tex" / "scripts"))

from texstream import LABEL_RE, StreamLine, read_stream  # noqa: E402

AUDIT_JSON = REPO_ROOT / "lean" / "audit.json"
TABLE_OUT = REPO_ROOT / "tex" / "tables" / "formalization.tex"

GENERATED_BY = "lean/scripts/render_table.py"

# Theorem-like environments declared in tex/carlo_master.tex.
THEOREM_ENVS = (
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "example",
    "remark",
    "assumption",
)

ENV_DISPLAY = {e: e.capitalize() for e in THEOREM_ENVS}

BEGIN_RE = re.compile(
    r"\\begin\{(" + "|".join(THEOREM_ENVS) + r")\}"
    r"(?:\[(?P<name>(?:[^\[\]]|\[[^\]]*\])*)\])?"
    r"(?:\s*\\label\{(?P<label>[^}]+)\})?"
)
CHAPTER_RE = re.compile(r"\\chapter\*?\{([^}]*)\}")

# Lean's own axioms. A declaration whose closure is a subset of these is
# checked from Mathlib alone.
CORE_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})

SORRY_AXIOM = "sorryAx"

# Results imported from the literature and never reproved in the dissertation.
# Each is an `axiom` in ThresholdOpt.Literature. The key is the superscript
# used in the Status column; the legend lives in appendix-e-formalization.tex.
# Unknown literature axioms are a hard error, so this cannot drift silently.
#
# None remains. The last one, Mohri et al., Foundations of Machine Learning,
# Corollary 3.19 (key "M"), is proved in ThresholdOpt/UniformConvergence.lean;
# `ThresholdOpt.Literature.uniform_convergence_vc` is now a theorem.
LITERATURE_KEYS: dict[str, str] = {}

# Theorem-like environments that assert nothing a proof could establish:
# commentary on how to read other results. They get their own status so that
# "Not formalized" means "could be formalized and is not". Tagging a Lean
# declaration with one of these labels is an error.
NO_FORMAL_CONTENT = frozenset({"rem:voronoi_bound_reading"})


@dataclass(frozen=True)
class Statement:
    """A theorem-like environment as it appears in the dissertation."""

    env: str
    label: str | None
    name: str | None
    chapter: str
    file: str
    lineno: int


def find_statements(stream: list[StreamLine]) -> list[Statement]:
    """Collect theorem-like environments from a flattened tex stream."""
    statements: list[Statement] = []
    chapter = ""
    for item in stream:
        chap = CHAPTER_RE.search(item.text)
        if chap:
            chapter = chap.group(1).strip()
        for m in BEGIN_RE.finditer(item.text):
            statements.append(
                Statement(
                    env=m.group(1),
                    label=m.group("label"),
                    name=(m.group("name") or "").strip() or None,
                    chapter=chapter,
                    file=item.file.name,
                    lineno=item.lineno,
                )
            )
    return statements


def all_labels(stream: list[StreamLine]) -> set[str]:
    """Every \\label in the dissertation, of any kind."""
    return {
        m.group(1) for item in stream for m in LABEL_RE.finditer(item.text)
    }


def load_audit(path: Path = AUDIT_JSON) -> dict:
    """Read lean/audit.json, or an empty audit when Lean has not run yet."""
    if not path.is_file():
        return {"toolchain": None, "mathlib_rev": None, "declarations": []}
    return json.loads(path.read_text(encoding="utf-8"))


def classify(record: dict | None) -> tuple[str, list[str]]:
    """Map an audit record to a (status, literature keys) pair.

    Raises ValueError on an axiom that is neither one of Lean's own nor a
    registered literature axiom, so an untracked assumption cannot slip into
    the table as "Verified".

    Tagged definitions get their own status. Nothing is proved about a
    definition, so its axiom closure carries no information and reporting it
    alongside the theorems would inflate the verified count.
    """
    if record is None:
        return "none", []
    axioms = set(record.get("axioms", []))
    if SORRY_AXIOM in axioms or record.get("sorry"):
        return "sorry", []
    extra = sorted(axioms - CORE_AXIOMS)
    unknown = [a for a in extra if a not in LITERATURE_KEYS]
    if unknown:
        raise ValueError(
            f"{record.get('decl')} depends on unregistered axiom(s) "
            f"{unknown}; add them to LITERATURE_KEYS in {GENERATED_BY} "
            f"and to the legend in appendix-e-formalization.tex"
        )
    keys = [LITERATURE_KEYS[a] for a in extra]
    if record.get("definition"):
        # A definition has no proof, so its axiom closure is the core three no
        # matter what. Counting that as "Verified" would report transcription
        # as verification.
        return "definition", keys
    if record.get("partial"):
        return "partial", keys
    return ("cited" if extra else "verified"), keys


def classify_statement(
    label: str, record: dict | None
) -> tuple[str, list[str]]:
    """Classify a dissertation statement, given its audit record if any.

    A statement listed in NO_FORMAL_CONTENT is reported as such. Raises
    ValueError if a Lean declaration is tagged with one of those labels, since
    the two claims contradict each other.
    """
    if label in NO_FORMAL_CONTENT:
        if record is not None:
            raise ValueError(
                f"{record.get('decl')} is tagged with {label}, which "
                f"{GENERATED_BY} lists as having no formal content"
            )
        return "nocontent", []
    return classify(record)


def tt(text: str) -> str:
    """Typeset a Lean identifier in a monospace cell."""
    return "\\texttt{" + text.replace("_", "\\_") + "}"


def tt_decl(text: str) -> str:
    """Typeset a qualified Lean name that may break after each module dot.

    The longest declaration names are wider than half the text block, so the
    declaration column allows a line break after a namespace prefix.
    """
    return tt(text).replace(".", ".\\allowbreak{}")


def sup(keys: list[str]) -> str:
    return "$^{\\text{" + ",".join(keys) + "}}$" if keys else ""


def status_cell(status: str, keys: list[str]) -> str:
    if status == "definition":
        return "Formalized" + sup(keys)
    if status == "verified":
        return "Verified"
    if status == "cited":
        return "Verified" + sup(keys)
    if status == "partial":
        return "Verified in part" + sup(keys)
    if status == "sorry":
        return "\\textbf{Incomplete}"
    if status == "nocontent":
        return "No formal content"
    return "Not formalized"


def render(statements: list[Statement], audit: dict) -> str:
    """Build the longtable body for tex/tables/formalization.tex."""
    by_label = {
        d["label"]: d for d in audit.get("declarations", []) if d.get("label")
    }

    lines = [
        f"% Auto-generated by {GENERATED_BY} --- do not edit.",
        "{\\small",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\begin{longtable}{@{}l "
        ">{\\raggedright\\arraybackslash}p{0.58\\textwidth} l@{}}",
        "\\caption{Machine-checked status of every theorem-like statement in "
        "this dissertation.}\\label{tab:formalization}\\\\",
        "\\toprule",
        "\\textbf{Statement} & \\textbf{Lean declaration} & "
        "\\textbf{Status} \\\\",
        "\\midrule",
        "\\endfirsthead",
        "",
        "\\multicolumn{3}{@{}l}{\\textit{Table~\\ref{tab:formalization} "
        "(continued)}}\\\\",
        "\\toprule",
        "\\textbf{Statement} & \\textbf{Lean declaration} & "
        "\\textbf{Status} \\\\",
        "\\midrule",
        "\\endhead",
        "",
        "\\bottomrule",
        "\\endlastfoot",
        "",
    ]

    current_chapter = None
    for st in statements:
        if st.label is None:
            continue
        if st.chapter != current_chapter:
            if current_chapter is not None:
                lines.append("\\midrule")
            lines.append(
                "\\multicolumn{3}{@{}l}{\\textit{"
                + st.chapter.replace("&", "\\&")
                + "}}\\\\"
            )
            lines.append("\\midrule")
            current_chapter = st.chapter

        record = by_label.get(st.label)
        status, keys = classify_statement(st.label, record)
        decl = record["decl"] if record else None
        decl_cell = (
            tt_decl(decl.removeprefix("ThresholdOpt.")) if decl else "\\ "
        )
        lines.append(
            f"{ENV_DISPLAY[st.env]}~\\ref{{{st.label}}} & "
            f"{decl_cell} & {status_cell(status, keys)} \\\\"
        )

    # Some Lean theorems formalize a displayed equation rather than a
    # theorem-like environment. Listing them keeps the table a complete
    # picture of what was checked.
    theorem_labels = {s.label for s in statements if s.label}
    extra = sorted(set(by_label) - theorem_labels)
    if extra:
        if current_chapter is not None:
            lines.append("\\midrule")
        lines.append(
            "\\multicolumn{3}{@{}l}{\\textit{Displayed equations}}\\\\"
        )
        lines.append("\\midrule")
        for label in extra:
            record = by_label[label]
            status, keys = classify(record)
            lines.append(
                f"Equation~\\eqref{{{label}}} & "
                f"{tt_decl(record['decl'].removeprefix('ThresholdOpt.'))} & "
                f"{status_cell(status, keys)} \\\\"
            )

    lines.append("\\end{longtable}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def summarize(statements: list[Statement], audit: dict) -> dict[str, int]:
    by_label = {
        d["label"]: d for d in audit.get("declarations", []) if d.get("label")
    }
    counts = {
        "verified": 0,
        "cited": 0,
        "definition": 0,
        "partial": 0,
        "sorry": 0,
        "none": 0,
        "nocontent": 0,
        "unlabeled": 0,
    }
    for st in statements:
        if st.label is None:
            counts["unlabeled"] += 1
            continue
        status, _ = classify_statement(st.label, by_label.get(st.label))
        counts[status] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if the committed table is out of date",
    )
    args = parser.parse_args(argv)

    stream = read_stream()
    statements = find_statements(stream)
    audit = load_audit()
    table = render(statements, audit)

    counts = summarize(statements, audit)
    print(
        f"statements: {sum(counts.values())} | "
        f"verified: {counts['verified']} | "
        f"cited: {counts['cited']} | "
        f"definitions: {counts['definition']} | "
        f"partial: {counts['partial']} | "
        f"not formalized: {counts['none']} | "
        f"no formal content: {counts['nocontent']} | "
        f"unlabeled: {counts['unlabeled']}"
    )
    if counts["sorry"]:
        print(f"ERROR: {counts['sorry']} declaration(s) still use sorry")
        return 1

    # A @[thesis] tag naming a label that does not exist in the dissertation
    # would produce a row whose \ref renders as "??" in the PDF.
    known = all_labels(stream)
    dangling = sorted(
        d["label"]
        for d in audit.get("declarations", [])
        if d.get("label") and d["label"] not in known
    )
    if dangling:
        print("\nERROR: @[thesis] labels with no \\label in the tex sources:")
        for label in dangling:
            print(f"  {label}")
        return 1

    unlabeled = [s for s in statements if s.label is None]
    if unlabeled:
        print("\nEnvironments with no \\label (omitted from the table):")
        for s in unlabeled:
            name = f" [{s.name}]" if s.name else ""
            print(f"  {s.file}:{s.lineno}  \\begin{{{s.env}}}{name}")

    if args.check:
        current = (
            TABLE_OUT.read_text(encoding="utf-8")
            if TABLE_OUT.is_file()
            else ""
        )
        if current != table:
            print(f"\nERROR: {TABLE_OUT.name} is stale; run make lean-audit")
            return 1
        print(f"\n{TABLE_OUT.name} is up to date")
        return 0

    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text(table, encoding="utf-8")
    print(f"\nwrote {TABLE_OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
