# Lean formalization

Machine-checked proofs for the mathematical results of the dissertation, in
Lean 4 with [Mathlib](https://github.com/leanprover-community/mathlib4). Each
formalized statement is tagged with the LaTeX label of the environment it
formalizes in [`tex/content/`](../tex/content/), and `lake exe audit` reads
those tags back out to produce a per-theorem record of what each proof
depends on. Appendix E of the dissertation describes the development and
prints the generated status table.

The audit distinguishes five statuses:

| Status | Meaning |
| --- | --- |
| Verified | a Lean theorem proves the statement of the text, or a stronger statement that implies it, and its proof closure uses only `propext`, `Classical.choice`, and `Quot.sound` |
| Formalized | a definition or assumption is transcribed and used, without treating it as a proved result |
| Verified in part | the Lean statement is weaker than the text; no current row has this status |
| Not formalized | a mathematical statement has no Lean counterpart; no current row has this status |
| No formal content | interpretive commentary with no further assertion to prove; no current row has this status |

Current state: 40 theorem-like environments, of which 30 are verified and 10
are formalized definitions or assumptions, plus nine displayed equations, for
49 tagged declarations (39 proved, 10 definitions). No `sorry`, partial tag,
or literature axiom remains: the uniform-convergence theorem the text cites
is itself proved.

The Lean code was generated with AI assistants. The kernel check guarantees
that each tagged theorem is proved; whether each Lean statement says what the
printed statement says rests on an AI-assisted comparison of each tagged
statement with the text.

## Quick start

```bash
make lean-setup    # install elan + download Mathlib's prebuilt cache (GBs, once)
make lean-build    # type-check everything
make lean-audit    # regenerate audit.json + tex/tables/formalization.tex
```

Without `make`:

```bash
cd lean
lake exe cache get       # prebuilt Mathlib artifacts; building from source takes hours
lake build               # type-check
lake exe audit           # write audit.json
lake exe audit --check   # verify only; this is the CI gate
```

> `lake exe cache get` and `lake update` both write to `~/.cache/mathlib`. Do
> not run them concurrently: on Windows the two processes collide renaming the
> same `.part` files and both fail with `error 32`.

## Tagging a statement

```lean
@[thesis "prop:calibrate_then_decide"]        -- matches the text
@[thesisPartial "prop:excess_risk_bound"]     -- weaker than the text
```

The label must match a `\label{...}` in the dissertation. Three checks keep the
mapping honest, and each fails the build rather than degrading quietly:

- `lake exe audit` fails if any tagged declaration depends on `sorryAx`. This is
  the real gate: `lake build` only *warns* on `sorry`.
- `lake exe audit` fails if two declarations claim the same label.
- [`scripts/render_table.py`](scripts/render_table.py) fails if a proof depends
  on an axiom not registered in its `LITERATURE_KEYS` table, and if a tag names
  a label that does not exist in the tex sources.

What no tool checks is whether a Lean statement *says* what the dissertation
statement says. That is why `@[thesisPartial]` exists, and why new tags should
be read side by side with the text before being added.

The same applies to `axiom`s, and there it is worse: nothing downstream of a
false axiom means anything, and dropping a hypothesis or a quantifier to avoid
developing the objects it ranges over usually produces one. An imported result
must be stated with the quantifier structure of its source even when that costs
new scaffolding, as the sample measure in
[`Learning.lean`](ThresholdOpt/Learning.lean) does for the uniform-convergence
bound. When the cited statement cannot be written faithfully at all, leave it
out and say why rather than weakening it into the list.

## Conventions

Two conventions are fixed in [`ThresholdOpt/Basic.lean`](ThresholdOpt/Basic.lean)
because the dissertation states each of them in two places:

- `C i j` is the cost of predicting `j` when the truth is `i`, so the cost
  *vector* `c_i` is the `i`-th **column** and the conditional risk contracts
  over the row index.
- `lexArgmin` returns the least index attaining the minimum, matching the
  "smallest index" tie-break of `eq:bayes_optimal`.

The class count is `Fin K` with `[NeZero K]`, so statements read with the same
`K` as the text. The simplex is Mathlib's `stdSimplex ℝ (Fin K)`, not a
redefinition.

## Structure

Chapter and appendix numbers refer to the dissertation.

```
lakefile.toml           Mathlib pinned to a release tag by git rev
lean-toolchain          exact Lean version (committed)
lake-manifest.json      resolved dependency revisions (committed)
audit.json              generated: per-theorem axiom provenance (committed)
ThresholdOpt.lean       root module
ThresholdOpt/
  Attr.lean             the @[thesis] attribute
  Basic.lean            Ch. 2: cost matrices, simplex, conditional risk, lexArgmin
  Risk.lean             Ch. 2: pointwise excess-risk bound
  ConditionalLawExists.lean  Ch. 2: existence of the conditional class probabilities
  Binary.lean           Ch. 3: reasonableness, Elkan threshold, calibrate-then-decide
  Multiclass.lean       Ch. 5: admissibility, element-wise reasonableness, closed Bayes cover
  BoundaryMinimizer.lean Ch. 5: every nonempty optimality region meets the boundary
  Examples.lean         Ch. 5: the two cost-matrix witnesses
  Partition.lean        Ch. 6: parallel shift, linear scaling, the equivalences
  Voronoi.lean          Ch. 7: Voronoi rule, DLC, H_PS in H_Vor
  Witness.lean          Ch. 7: the strict risk-separation witness
  SequentialSampler.lean  Ch. 7: the sequential d² sampler
  Complexity.lean       Ch. 7, App. C: Natarajan-dimension bounds, H_Vor = H_Full
  Capacity.lean         VC / pseudo / Natarajan dimensions, the two capacity
                        theorems of App. C, calibrator families
  NatarajanLemma.lean   Natarajan's lemma, from the Natarajan dimension of the
                        rules to the VC dimension of the loss level sets
  Learning.lean         risk, empirical risk, the law of the sample, the rate
  McDiarmid.lean        bounded-difference concentration on product measures
  Massart.lean          finite-class Rademacher averages
  Sauer.lean            growth-function bounds
  Symmetrization.lean   countable-class Rademacher generalization bound
  UniformConvergence.lean  Mohri et al. Corollary 3.19, proved
  Literature.lean       binary and multiclass cost-sensitive bounds
  PointwiseApprox.lean  extend bounds through pointwise approximation
  SeparableBounds.lean  full binary proposition with its displayed rate
  DecisionClassBounds.lean  real-parameter decision-class extensions
  VoronoiSimplex.lean   strict simplex restriction and geometric witnesses
  CapacityLowerBounds.lean  exact 0/1 scaling dimension and quadratic lower
                        bounds for Full and Voronoi (not stated in the text)
  Mahalanobis.lean      fixed quadratic metrics and a discrete-metric
                        counterexample (not stated in the text)
  Sensitivity.lean      Ch. 9: tangent space and the affine cost functional
  AffineSensitivity.lean  Ch. 9: smoothness of the cost functional off the boundary
  Diffusion.lean        Ch. 9: the uniform sphere, dimensional dilution
  IntegratedDiffusion.lean  Ch. 9: the integrated dilution
  SphereSymmetry.lean   Ch. 9: second moments of the uniform sphere
  GaussianKL.lean       App. B: KL between Gaussians, the chain rule, the budgets
  GaussianKLGeneral.lean  App. B: the Gaussian formula for general covariances
Audit/Main.lean         lake exe audit
scripts/
  render_table.py       audit.json + tex -> tex/tables/formalization.tex
  render_table_test.py
```

## Scope

- The binary generalization bound (Proposition 2.6.1) assumes a countable
  class of measurable rules that approximates each rule pointwise;
  measurability of the final event alone is not sufficient.
- `DecisionClassBounds` covers all real-parameter parallel shifts at `K-1`,
  and full matrices, unconstrained and simplex prototypes at `K(K-1)`, which
  implies the `K²` bounds stated in the text. Subclass monotonicity also
  covers DLC rules at that bound.
- For all real-parameter scaling rules the proved probabilistic bound uses
  `K`; the deterministic Natarajan bound `K-1` is proved. The text uses only
  the deterministic bound for this class.
- `VoronoiSimplex` proves strict inclusion in full matrices for `K ≥ 2`, a
  nondegenerate band witness for `K ≥ 3`, and a three-class parallel-shift
  rule outside the simplex-prototype class, even away from boundaries.
- `CapacityLowerBounds` and `Mahalanobis` prove results that the
  dissertation does not state; no tag points at them.

Formal verification does not validate the experiment software or the
empirical claims.

## Development

```bash
make lean-build    # type-check (lake build)
make lean-audit    # regenerate the appendix table
uv run pytest lean/scripts    # tests for the table renderer
make lint          # ruff + ruff-format + chktex (no hook touches .lean)
```

Open items are listed in [`TODO.md`](TODO.md).
