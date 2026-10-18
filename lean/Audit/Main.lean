import ThresholdOpt.Attr

/-!
# `lake exe audit`

Emits `lean/audit.json`: for every `@[thesis "<label>"]`-tagged declaration,
the LaTeX label it formalizes and the full axiom closure of its proof.

Imports only `ThresholdOpt.Attr` (which depends on `Lean` alone). The full
development is loaded at *runtime* via `importModules`, reading the `.olean`
files that `lake build` produced. Importing `ThresholdOpt` here instead would
force Lake to compile all of Mathlib to native code to link the executable,
which takes hours; reading the oleans costs seconds.

The axiom closure is the point. `lake build` only *warns* on `sorry`, so a
green build is not evidence that anything was proved. This tool fails hard
when a tagged declaration depends on `sorryAx`, and it records which
declarations lean on the results imported from the literature
(`ThresholdOpt.Literature`) so the appendix table can separate
"checked from Mathlib alone" from "checked modulo a cited result".

Usage:
  lake exe audit           write audit.json (fails on sorry / duplicate labels)
  lake exe audit --check   verify only, write nothing
-/

open Lean

namespace ThresholdOpt.Audit

/-- Read the pinned Mathlib revision out of `lake-manifest.json`. -/
def mathlibRev : IO (Option String) := do
  let path : System.FilePath := "lake-manifest.json"
  if !(← path.pathExists) then return none
  let raw ← IO.FS.readFile path
  let some json := (Json.parse raw).toOption | return none
  let some pkgs := (json.getObjValAs? (Array Json) "packages").toOption
    | return none
  for p in pkgs do
    if (p.getObjValAs? String "name").toOption == some "mathlib" then
      return (p.getObjValAs? String "rev").toOption
  return none

def toolchain : IO (Option String) := do
  let path : System.FilePath := "lean-toolchain"
  if !(← path.pathExists) then return none
  return some (← IO.FS.readFile path).trimAscii.toString

structure Entry where
  label : String
  decl : Name
  module : Name
  axioms : Array Name
  hasSorry : Bool
  isPartial : Bool
  /-- Whether the tagged constant is a definition rather than a proof. A
  definition has no axiom closure worth reporting: it is transcribed, not
  verified, and the appendix table must not count it as the latter. -/
  isDefinition : Bool

def Entry.toJson (e : Entry) : Json :=
  Json.mkObj [
    ("label", Json.str e.label),
    ("decl", Json.str e.decl.toString),
    ("module", Json.str e.module.toString),
    ("axioms", Json.arr (e.axioms.map fun a => Json.str a.toString)),
    ("literature_axioms", Json.arr
      (e.axioms.filter (·.getPrefix == `ThresholdOpt.Literature)
        |>.map fun a => Json.str a.toString)),
    ("sorry", Json.bool e.hasSorry),
    ("partial", Json.bool e.isPartial),
    ("definition", Json.bool e.isDefinition)
  ]

/-- Collect every tagged declaration together with its axiom closure.

Uses `Lean.collectAxioms`, the same traversal that backs `#print axioms`. A
hand-rolled walk over `Environment.find?` is *not* equivalent: theorem values
live in the kernel environment (`env.checked`), and reading the elaboration
environment silently yields types only, which under-reports the axioms. -/
def gather : CoreM (Array Entry) := do
  let env ← getEnv
  let mut out : Array Entry := #[]
  for (decl, label, isPartial) in thesisEntries env do
    let axs ← Lean.collectAxioms decl
    let sorted := axs.qsort (fun a b => a.toString < b.toString)
    out := out.push {
      label := label
      decl := decl
      module :=
        match env.getModuleIdxFor? decl with
        | some idx =>
          (env.allImportedModuleNames)[idx.toNat]?.getD Name.anonymous
        | none => Name.anonymous
      axioms := sorted
      hasSorry := sorted.contains ``sorryAx
      isPartial := isPartial
      isDefinition :=
        match env.find? decl with
        | some (.thmInfo _) => false
        | _ => true }
  return out.qsort (fun a b => a.label < b.label)

def render (tc : Option String) (rev : Option String) (es : Array Entry) :
    String :=
  let obj := Json.mkObj [
    ("toolchain", match tc with | some s => Json.str s | none => Json.null),
    ("mathlib_rev", match rev with | some s => Json.str s | none => Json.null),
    ("declarations", Json.arr (es.map Entry.toJson))
  ]
  obj.pretty ++ "\n"

/-- Labels claimed by more than one declaration. Two Lean theorems mapping to
one LaTeX statement would make the table ambiguous. -/
def duplicateLabels (es : Array Entry) : Array String := Id.run do
  let mut seen : Std.HashMap String Nat := {}
  for e in es do
    seen := seen.insert e.label (seen.getD e.label 0 + 1)
  let mut dups := #[]
  for e in es do
    if seen.getD e.label 0 > 1 && !dups.contains e.label then
      dups := dups.push e.label
  return dups

end ThresholdOpt.Audit

open ThresholdOpt.Audit in
def main (args : List String) : IO UInt32 := do
  let checkOnly := args.contains "--check"
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := `ThresholdOpt }] {} (trustLevel := 1024)
  let (entries, _) ← (gather).toIO
    { fileName := "<audit>", fileMap := default } { env := env }
  let out := render (← toolchain) (← mathlibRev) entries

  let isCore (a : Name) : Bool :=
    a == ``propext || a == ``Classical.choice || a == ``Quot.sound
  let nDefs := entries.filter (·.isDefinition)
  let nCore := entries.filter (fun e => !e.isDefinition && e.axioms.all isCore)
  let nCited := entries.filter (fun e =>
    e.axioms.any (·.getPrefix == `ThresholdOpt.Literature))
  let sorried := entries.filter (·.hasSorry)
  let dups := duplicateLabels entries

  IO.println s!"tagged: {entries.size} | verified: {nCore.size} | \
    cited: {nCited.size} | definitions: {nDefs.size} | sorry: {sorried.size}"

  let mut failed := false
  for e in sorried do
    IO.eprintln s!"ERROR: {e.decl} ({e.label}) depends on sorryAx"
    failed := true
  for d in dups do
    IO.eprintln s!"ERROR: label {d} is claimed by more than one declaration"
    failed := true
  if failed then return 1

  let path : System.FilePath := "audit.json"
  if checkOnly then
    let current ← if ← path.pathExists then IO.FS.readFile path else pure ""
    if current != out then
      IO.eprintln "ERROR: audit.json is stale; run `lake exe audit`"
      return 1
    IO.println "audit.json is up to date"
  else
    IO.FS.writeFile path out
    IO.println s!"wrote lean/{path}"
  return 0
