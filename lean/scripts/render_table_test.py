"""Unit tests for lean/scripts/render_table.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from render_table import (
    Statement,
    classify,
    find_statements,
    render,
    status_cell,
    summarize,
    tt,
    tt_decl,
)
from texstream import StreamLine


def stream(*lines: str) -> list[StreamLine]:
    """Build a flattened stream from raw source lines."""
    return [
        StreamLine(Path("ch.tex"), i, text)
        for i, text in enumerate(lines, start=1)
    ]


def audit(*records: dict) -> dict:
    return {
        "toolchain": "x",
        "mathlib_rev": "y",
        "declarations": list(records),
    }


def record(
    label: str,
    decl: str,
    axioms: list[str],
    sorry: bool = False,
    partial: bool = False,
    definition: bool = False,
):
    return {
        "label": label,
        "decl": decl,
        "module": "ThresholdOpt.Test",
        "axioms": axioms,
        "sorry": sorry,
        "partial": partial,
        "definition": definition,
    }


CORE = ["propext", "Classical.choice", "Quot.sound"]

# No literature axiom is declared any more (the last one, Mohri et al.
# Corollary 3.19, is proved). The citation machinery is exercised with a
# hypothetical axiom registered for the duration of a test.
LIT = "ThresholdOpt.Literature.example_axiom"


@pytest.fixture
def lit_axiom(monkeypatch):
    import render_table

    monkeypatch.setitem(render_table.LITERATURE_KEYS, LIT, "M")
    return LIT


class TestFindStatements:
    """Discovery of theorem-like environments in the flattened tex stream."""

    def test_captures_env_name_and_label(self):
        st = find_statements(
            stream(r"\begin{lemma}[Optimality]\label{lem:opt}")
        )
        assert len(st) == 1
        assert st[0].env == "lemma"
        assert st[0].name == "Optimality"
        assert st[0].label == "lem:opt"

    def test_label_is_none_when_absent(self):
        st = find_statements(stream(r"\begin{remark}[Bound]"))
        assert st[0].label is None
        assert st[0].name == "Bound"

    def test_optional_name_may_be_absent(self):
        st = find_statements(stream(r"\begin{remark}\label{rem:x}"))
        assert st[0].name is None
        assert st[0].label == "rem:x"

    def test_ignores_equation_labels_inside_the_body(self):
        # The label on the \begin line is the statement's; a later equation
        # label inside the body must not be picked up instead.
        st = find_statements(
            stream(
                r"\begin{definition}[Calibration]\label{def:cal}",
                r"\begin{equation}\label{eq:cal}",
                r"\end{definition}",
            )
        )
        assert len(st) == 1
        assert st[0].label == "def:cal"

    def test_tracks_the_enclosing_chapter(self):
        st = find_statements(
            stream(
                r"\chapter{Theoretical framework}\label{sec:theory}",
                r"\begin{theorem}\label{thm:a}",
                r"\chapter{Voronoi tessellations}",
                r"\begin{lemma}\label{lem:b}",
            )
        )
        assert [s.chapter for s in st] == [
            "Theoretical framework",
            "Voronoi tessellations",
        ]

    def test_ignores_non_theorem_environments(self):
        st = find_statements(
            stream(r"\begin{proof}", r"\begin{table}", r"\begin{align}")
        )
        assert st == []

    def test_preserves_document_order(self):
        st = find_statements(
            stream(
                r"\begin{definition}\label{def:a}",
                r"\begin{theorem}\label{thm:b}",
                r"\begin{corollary}\label{cor:c}",
            )
        )
        assert [s.label for s in st] == ["def:a", "thm:b", "cor:c"]


class TestClassify:
    """Mapping an audit record to a status."""

    def test_missing_record_is_not_formalized(self):
        assert classify(None) == ("none", [])

    def test_core_axioms_only_is_verified(self):
        status, keys = classify(record("l", "D", CORE))
        assert status == "verified"
        assert keys == []

    def test_empty_axiom_closure_is_verified(self):
        assert classify(record("l", "D", []))[0] == "verified"

    def test_literature_axiom_is_cited(self, lit_axiom):
        status, keys = classify(
            record(
                "l",
                "D",
                [*CORE, LIT],
            )
        )
        assert status == "cited"
        assert keys == ["M"]

    def test_multiple_literature_axioms_are_all_reported(
        self, monkeypatch, lit_axiom
    ):
        import render_table

        monkeypatch.setitem(
            render_table.LITERATURE_KEYS, "ThresholdOpt.Literature.other", "Z"
        )
        status, keys = classify(
            record(
                "l",
                "D",
                [
                    *CORE,
                    "ThresholdOpt.Literature.other",
                    LIT,
                ],
            )
        )
        assert status == "cited"
        assert sorted(keys) == ["M", "Z"]

    def test_sorry_axiom_is_flagged(self):
        assert classify(record("l", "D", [*CORE, "sorryAx"]))[0] == "sorry"

    def test_sorry_flag_is_flagged_even_without_the_axiom(self):
        assert classify(record("l", "D", CORE, sorry=True))[0] == "sorry"

    def test_unregistered_axiom_is_a_hard_error(self):
        # An untracked assumption must never be reported as "Verified".
        with pytest.raises(ValueError, match="unregistered axiom"):
            classify(record("l", "ThresholdOpt.D", [*CORE, "myCheat"]))

    def test_partial_flag_downgrades_verified(self):
        status, _ = classify(record("l", "D", CORE, partial=True))
        assert status == "partial"

    def test_partial_keeps_its_literature_keys(self, lit_axiom):
        # A statement can be both weaker than the text and dependent on a
        # cited result; the superscript must survive the downgrade.
        status, keys = classify(
            record(
                "l",
                "D",
                [*CORE, LIT],
                partial=True,
            )
        )
        assert status == "partial"
        assert keys == ["M"]

    def test_sorry_beats_partial(self):
        status, _ = classify(
            record("l", "D", [*CORE, "sorryAx"], partial=True)
        )
        assert status == "sorry"

    def test_definition_is_not_verified(self):
        # A definition's axiom closure is the core three whatever it says, so
        # reporting it as "Verified" would count transcription as proof.
        status, _ = classify(record("l", "D", CORE, definition=True))
        assert status == "definition"

    def test_definition_keeps_its_literature_keys(self, lit_axiom):
        status, keys = classify(
            record(
                "l",
                "D",
                [*CORE, LIT],
                definition=True,
            )
        )
        assert status == "definition"
        assert keys == ["M"]

    def test_sorry_beats_definition(self):
        status, _ = classify(
            record("l", "D", [*CORE, "sorryAx"], definition=True)
        )
        assert status == "sorry"


class TestStatusCell:
    """Rendering of the Status column."""

    def test_verified(self):
        assert status_cell("verified", []) == "Verified"

    def test_cited_carries_the_axiom_keys(self):
        assert "AB" in status_cell("cited", ["AB"])

    def test_not_formalized_is_spelled_out(self):
        assert status_cell("none", []) == "Not formalized"

    def test_no_formal_content_is_not_reported_as_a_gap(self):
        assert status_cell("nocontent", []) != status_cell("none", [])
        assert "Verified" not in status_cell("nocontent", [])

    def test_sorry_is_visually_loud(self):
        assert "textbf" in status_cell("sorry", [])

    def test_partial_is_distinguishable_from_verified(self):
        assert status_cell("partial", []) != status_cell("verified", [])
        assert "part" in status_cell("partial", [])

    def test_partial_carries_axiom_keys(self):
        assert "DS" in status_cell("partial", ["DS"])

    def test_definition_is_distinguishable_from_verified(self):
        assert status_cell("definition", []) != status_cell("verified", [])

    def test_definition_does_not_claim_verification(self):
        assert "Verified" not in status_cell("definition", [])


class TestTt:
    """Monospace cell escaping."""

    def test_escapes_underscores(self):
        assert tt("a_b_c") == "\\texttt{a\\_b\\_c}"

    def test_leaves_dots_alone(self):
        assert tt("A.b") == "\\texttt{A.b}"


class TestRender:
    """The generated longtable."""

    def test_includes_a_row_per_labeled_statement(self):
        out = render(
            find_statements(
                stream(
                    r"\chapter{Ch}",
                    r"\begin{lemma}\label{lem:a}",
                    r"\begin{theorem}\label{thm:b}",
                )
            ),
            audit(),
        )
        assert "Lemma~\\ref{lem:a}" in out
        assert "Theorem~\\ref{thm:b}" in out

    def test_omits_unlabeled_statements(self):
        out = render(find_statements(stream(r"\begin{remark}")), audit())
        # The only \ref left is the longtable's own continuation header.
        assert "Remark~\\ref{" not in out
        assert "Not formalized" not in out

    def test_strips_the_package_prefix_from_declarations(self):
        out = render(
            find_statements(stream(r"\chapter{C}", r"\begin{lemma}\label{l}")),
            audit(record("l", "ThresholdOpt.Binary.elkan", CORE)),
        )
        assert "\\texttt{Binary.\\allowbreak{}elkan}" in out
        assert "ThresholdOpt.Binary" not in out

    def test_declaration_names_may_break_after_a_module_dot(self):
        assert tt_decl("A.b_c") == "\\texttt{A.\\allowbreak{}b\\_c}"

    def test_unmatched_statement_renders_as_not_formalized(self):
        out = render(
            find_statements(stream(r"\chapter{C}", r"\begin{lemma}\label{l}")),
            audit(),
        )
        assert "Not formalized" in out

    def test_emits_a_group_header_per_chapter(self):
        out = render(
            find_statements(
                stream(
                    r"\chapter{First}",
                    r"\begin{lemma}\label{a}",
                    r"\chapter{Second}",
                    r"\begin{lemma}\label{b}",
                )
            ),
            audit(),
        )
        assert "\\textit{First}" in out
        assert "\\textit{Second}" in out

    def test_is_a_well_formed_longtable(self):
        out = render(
            find_statements(stream(r"\chapter{C}", r"\begin{lemma}\label{a}")),
            audit(),
        )
        assert out.count("\\begin{longtable}") == 1
        assert out.count("\\end{longtable}") == 1
        assert "\\label{tab:formalization}" in out
        assert out.endswith("\n")

    def test_header_marks_the_file_as_generated(self):
        out = render([], audit())
        assert out.startswith("% Auto-generated by")


class TestSummarize:
    """Counts printed by the CLI."""

    def test_counts_each_status(self, lit_axiom):
        statements = find_statements(
            stream(
                r"\chapter{C}",
                r"\begin{lemma}\label{a}",
                r"\begin{lemma}\label{b}",
                r"\begin{lemma}\label{c}",
                r"\begin{definition}\label{d}",
                r"\begin{remark}",
            )
        )
        counts = summarize(
            statements,
            audit(
                record("a", "D.a", CORE),
                record(
                    "b",
                    "D.b",
                    [*CORE, LIT],
                ),
                record("d", "D.d", CORE, definition=True),
            ),
        )
        assert counts == {
            "verified": 1,
            "cited": 1,
            "definition": 1,
            "partial": 0,
            "sorry": 0,
            "none": 1,
            "nocontent": 0,
            "unlabeled": 1,
        }


class TestAgainstTheDissertation:
    """Guards on the real tex sources, not a fixture."""

    def test_every_statement_is_labeled(self):
        # An unlabeled environment cannot be \ref'd, so it would silently drop
        # out of the appendix table.
        from render_table import read_stream

        unlabeled = [
            s for s in find_statements(read_stream()) if s.label is None
        ]
        assert unlabeled == [], "add \\label to: " + ", ".join(
            f"{s.file}:{s.lineno} ({s.env})" for s in unlabeled
        )

    def test_labels_are_unique(self):
        from render_table import read_stream

        labels = [s.label for s in find_statements(read_stream()) if s.label]
        assert len(labels) == len(set(labels))


class TestLiteratureKeys:
    """The registry must track the axioms actually declared in Lean."""

    def _declared_axioms(self) -> set[str]:
        import re

        from render_table import REPO_ROOT

        src = (
            REPO_ROOT / "lean" / "ThresholdOpt" / "Literature.lean"
        ).read_text(encoding="utf-8")
        # Drop doc comments first: they discuss axioms in prose, and the word
        # "axiom" at the start of a wrapped line would otherwise match.
        src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
        return {
            f"ThresholdOpt.Literature.{m}"
            for m in re.findall(r"^axiom\s+(\w+)", src, re.MULTILINE)
        }

    def test_every_registered_key_is_a_real_axiom(self):
        from render_table import LITERATURE_KEYS

        declared = self._declared_axioms()
        stale = sorted(set(LITERATURE_KEYS) - declared)
        assert stale == [], (
            f"LITERATURE_KEYS names non-existent axioms: {stale}"
        )

    def test_every_declared_axiom_is_registered(self):
        # An unregistered axiom makes classify() raise, which would break the
        # table build rather than silently mislabel; catch it here instead.
        from render_table import LITERATURE_KEYS

        missing = sorted(self._declared_axioms() - set(LITERATURE_KEYS))
        assert missing == [], f"axioms missing from LITERATURE_KEYS: {missing}"

    def test_keys_are_unique(self):
        from render_table import LITERATURE_KEYS

        keys = list(LITERATURE_KEYS.values())
        assert len(keys) == len(set(keys))


class TestStatementDataclass:
    def test_is_hashable(self):
        s = Statement("lemma", "l", None, "C", "f.tex", 1)
        assert {s, s} == {s}
