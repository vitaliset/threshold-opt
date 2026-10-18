import Lean

/-!
# The `@[thesis]` attribute

Links a Lean declaration to the LaTeX label of the dissertation statement it
formalizes:

```lean
@[thesis "lem:threshold_optimality"]
theorem bayes_eq_threshold : ... := ...
```

`lake exe audit` reads this table back out of the environment, collects each
tagged declaration's axiom closure, and emits `lean/audit.json`. Recording the
link in the environment rather than in a comment is what lets the appendix
table be generated rather than maintained by hand.
-/

namespace ThresholdOpt

open Lean

/-- `@[thesis "lem:some_label"]` -/
syntax (name := thesis) "thesis" str : attr

/-- `@[thesisPartial "lem:some_label"]`, for a Lean statement that is strictly
weaker than the dissertation statement it corresponds to: one part of a
multi-part proposition, or the pointwise core of an integrated claim. The
appendix table reports these separately so that a partial formalization is
never presented as a complete one. -/
syntax (name := thesisPartial) "thesisPartial" str : attr

/-- Maps a tagged declaration to the LaTeX label it formalizes. -/
initialize thesisAttr : ParametricAttribute String ←
  registerParametricAttribute {
    name := `thesis
    descr := "LaTeX label of the dissertation statement this formalizes"
    getParam := fun _ stx => do
      match stx with
      | `(attr| thesis $s:str) => return s.getString
      | _ => throwError "expected: @[thesis \"lem:some_label\"]"
  }

/-- As `thesisAttr`, for partial formalizations. -/
initialize thesisPartialAttr : ParametricAttribute String ←
  registerParametricAttribute {
    name := `thesisPartial
    descr := "LaTeX label this partially formalizes"
    getParam := fun _ stx => do
      match stx with
      | `(attr| thesisPartial $s:str) => return s.getString
      | _ => throwError "expected: @[thesisPartial \"lem:some_label\"]"
  }

/-- All tagged declarations in the current environment, as
`(declaration, label, isPartial)`.

Filters by `getParam?` rather than reading the extension's internal state, so
this does not depend on how the attribute stores its table. -/
def thesisEntries (env : Environment) : Array (Name × String × Bool) :=
  env.constants.fold (init := #[]) fun acc declName _ =>
    match thesisAttr.getParam? env declName with
    | some label => acc.push (declName, label, false)
    | none =>
      match thesisPartialAttr.getParam? env declName with
      | some label => acc.push (declName, label, true)
      | none => acc

end ThresholdOpt
