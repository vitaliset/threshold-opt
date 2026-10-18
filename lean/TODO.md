# Lean formalization: open items

Current state: 40 theorem-like environments (30 verified, 10 formalized
definitions or assumptions) and nine displayed equations, for 49 tagged
declarations (39 proved, 10 definitions). No partial tag, `sorry`, or
literature axiom remains.

## Mathematics

- [ ] Extend the probabilistic scaling bound at `K-1` to the entire
      real-parameter class. Rational weights alone do not preserve every tie;
      the current approximation adds a score offset and has capacity `K`.
      The deterministic dimension bound `K-1` is proved. No tagged statement
      depends on this extension.
- The sharper order `d_Nat log K` in the multiclass bound is not pursued; the
  text states its bounds up to logarithmic factors.

## Infrastructure

- [ ] **Verify the CI path filters.** `.github/workflows/lean.yml` filters on
      `lean/**`. Push a Lean-only change and a Python-only change and confirm
      the right workflows trigger.
- [ ] **Consider `lean4checker` or `nanoda` in CI.** `lean-action` exposes both.
      They re-check the built environment with an independent kernel, guarding
      against elaborator bugs and against `Environment` tampering, which
      strengthens what Appendix E can claim.
- [ ] **Narrow the imports.** `Capacity`, `Examples`, `GaussianKL`, `Learning`
      and `NatarajanLemma` import all of `Mathlib.Tactic`, which costs minutes
      on a cold build. Importing only the tactics used would cut CI time.
- [ ] **A `lake test` driver**, so `lean-action`'s `test` input can be enabled
      instead of running the audit as a separate step.
- [ ] **Refresh source pointers in comments.** Some comments cite file and line
      positions of an earlier version of the text (`Basic`, `Complexity`,
      `Multiclass`, `Partition`, `Sensitivity`, `Voronoi`); a few say the text
      requires distinct prototypes, while the text allows coincident ones.
- [ ] **`make test-provenance` and `lean/`.** The opt-in provenance test treats
      every path outside `tex/` as code, so Lean commits count as code
      changes. Exclude `lean/` from its code pathspec.

## Invariants worth preserving

- The audit gate catches `sorry`, duplicate labels and definitions masquerading
  as proofs; `render_table.py` catches unregistered axioms and dangling tags.
  **Nothing checks that a Lean statement says what the dissertation statement
  says.** Reading tags against the text has found tags on assumptions that were
  not formalized and on statements far weaker than the text (one proof was
  `congrArg`), so every new tag needs that reading.
- **An axiom is a statement too, and a wrong one is unrecoverable.** Imported
  results once lost the probability quantifier of their source, which made them
  refutable: instantiate `risk` above the bound and `empRisk` at zero and you
  have `False`. Before adding an axiom, try to refute it by instantiating its
  hypotheses as degenerately as they allow.
- When a Lean statement is weaker than the text, tag it `@[thesisPartial]` and
  add a line to Appendix E naming the missing part.
- After any change, run `make lean-audit` and commit the regenerated
  `lean/audit.json` and `tex/tables/formalization.tex` with the proof.
