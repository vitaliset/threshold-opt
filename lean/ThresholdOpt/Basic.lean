import Mathlib.Analysis.Convex.StdSimplex
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Data.ENat.Lattice
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import ThresholdOpt.Attr

/-!
# The cost-sensitive decision setting

Formalizes the objects of Chapter 1 (`sec:theory`): the label space, the cost
matrix, the probability simplex, the conditional risk, and the tie-broken
`argmin` that defines the Bayes-optimal decision rule.

## Conventions

Two conventions are fixed here once, because the dissertation states each of
them in two places and the two statements must agree for the downstream results
to typecheck.

* **Index order.** `C i j` is the cost of predicting `j` when the truth is `i`,
  matching `eq:expected_risk` (`C_{y, h(x)}`, truth first). The cost *vector*
  `c_i` of the text is therefore the `i`-th **column**, so the conditional risk
  of action `i` contracts over the row index.
* **Ties.** `lexArgmin` returns the least index attaining the minimum, matching
  the "smallest index" rule of `eq:bayes_optimal`.

The class count is kept as `Fin K` with `[NeZero K]` rather than `Fin (K+1)` so
that statements read with the same `K` as the text.
-/

namespace ThresholdOpt

open Finset

variable {K : ℕ}

/-! ### Cost matrices and conditional risk -/

/-- A cost matrix. `C i j` is the penalty for predicting `j` when the truth is
`i`. -/
abbrev CostMatrix (K : ℕ) := Matrix (Fin K) (Fin K) ℝ

/-- The conditional risk of action `i` at the probability vector `p`, written
`𝓛_i(p) = c_i^T p` in the text (`eq:conditional_risk`). The contraction is over
the *true* label, the row index. -/
def condRisk (C : CostMatrix K) (p : Fin K → ℝ) (i : Fin K) : ℝ :=
  ∑ j, C j i * p j

/-- A cost matrix is nonnegative and bounded by `Cmax`, as assumed throughout
Chapter 1. -/
structure Bounded (C : CostMatrix K) (Cmax : ℝ) : Prop where
  nonneg : ∀ i j, 0 ≤ C i j
  le_max : ∀ i j, C i j ≤ Cmax

/-! ### The probability simplex

Mathlib's `stdSimplex ℝ (Fin K)` is `{p | (∀ i, 0 ≤ p i) ∧ ∑ i, p i = 1}`,
which is exactly `Δ^{K-1}` of the text. It is reused rather than redefined so
that `convex_stdSimplex` and the rest of Mathlib apply. -/

theorem sum_eq_one_of_mem_stdSimplex {p : Fin K → ℝ}
    (hp : p ∈ stdSimplex ℝ (Fin K)) : ∑ i, p i = 1 := hp.2

theorem nonneg_of_mem_stdSimplex {p : Fin K → ℝ}
    (hp : p ∈ stdSimplex ℝ (Fin K)) (i : Fin K) : 0 ≤ p i := hp.1 i

/-- The vertex `e_k` of the simplex. -/
noncomputable def vertex (k : Fin K) : Fin K → ℝ := fun i => if i = k then 1 else 0

@[simp] theorem vertex_apply_self (k : Fin K) : vertex k k = 1 := by
  simp [vertex]

@[simp] theorem vertex_apply_of_ne {i k : Fin K} (h : i ≠ k) : vertex k i = 0 := by
  simp [vertex, h]

theorem vertex_mem_stdSimplex (k : Fin K) : vertex k ∈ stdSimplex ℝ (Fin K) := by
  constructor
  · intro i; by_cases h : i = k <;> simp [vertex, h]
  · simp [vertex]

/-- Evaluating the conditional risk at a vertex reads off a single matrix
entry: `𝓛_i(e_j) = C j i`. This is the computation behind
`prop:elementwise_admissible`. -/
@[simp] theorem condRisk_vertex (C : CostMatrix K) (j i : Fin K) :
    condRisk C (vertex j) i = C j i := by
  simp [condRisk, vertex, Finset.sum_ite_eq' Finset.univ j (fun j => C j i)]

/-! ### Tie-broken argmin

`lexArgmin f` is the least index attaining `min f`. Every decision rule in the
dissertation is built from it, and the four partition-equivalence propositions
of Chapter 5 are invariance statements about it. -/

section Argmin

variable [NeZero K]

open scoped Classical in
/-- The set of indices attaining the minimum of `f`. -/
noncomputable def argminSet (f : Fin K → ℝ) : Finset (Fin K) :=
  {i | ∀ j, f i ≤ f j}

open scoped Classical in
theorem argminSet_nonempty (f : Fin K → ℝ) : (argminSet f).Nonempty := by
  have : (Finset.univ : Finset (Fin K)).Nonempty :=
    Finset.univ_nonempty (α := Fin K)
  obtain ⟨i, -, hi⟩ := Finset.exists_min_image Finset.univ f this
  exact ⟨i, by simp [argminSet, fun j => hi j (Finset.mem_univ j)]⟩

open scoped Classical in
/-- The Bayes tie-breaking rule: the least index attaining the minimum
(`eq:bayes_optimal`). -/
noncomputable def lexArgmin (f : Fin K → ℝ) : Fin K :=
  (argminSet f).min' (argminSet_nonempty f)

open scoped Classical in
theorem lexArgmin_mem (f : Fin K → ℝ) : lexArgmin f ∈ argminSet f :=
  Finset.min'_mem _ _

/-- `lexArgmin f` attains the minimum. -/
theorem lexArgmin_le (f : Fin K → ℝ) (j : Fin K) : f (lexArgmin f) ≤ f j := by
  classical
  have := lexArgmin_mem f
  simp only [argminSet, Finset.mem_filter, Finset.mem_univ, true_and] at this
  exact this j

/-- `lexArgmin f` is the *least* minimizer. -/
theorem lexArgmin_le_index {f : Fin K → ℝ} {i : Fin K} (hi : ∀ j, f i ≤ f j) :
    lexArgmin f ≤ i := by
  classical
  exact Finset.min'_le _ _ (by simp [argminSet, hi])

/-- A strict minimizer is *the* `lexArgmin`. This is the workhorse for the
witness computations: the vertex arguments of `prop:elementwise_admissible` and
every numeric check in `Examples`. -/
theorem lexArgmin_eq_of_strict {f : Fin K → ℝ} {i : Fin K}
    (h : ∀ j, j ≠ i → f i < f j) : lexArgmin f = i := by
  classical
  have hmin : ∀ j, f i ≤ f j := by
    intro j
    rcases eq_or_ne j i with rfl | hj
    · exact le_rfl
    · exact (h j hj).le
  refine le_antisymm (lexArgmin_le_index hmin) ?_
  by_contra hcon
  have hlt : lexArgmin f < i := not_le.1 hcon
  have hne : lexArgmin f ≠ i := ne_of_lt hlt
  exact absurd (lexArgmin_le f i) (not_le.2 (h _ hne))

/-- Adding one constant to every score leaves the decision unchanged. This is
the additive gauge freedom of Section 5 (`05:118`). -/
@[simp] theorem lexArgmin_add_const (f : Fin K → ℝ) (c : ℝ) :
    lexArgmin (fun k => f k + c) = lexArgmin f := by
  classical
  have hset : argminSet (fun k => f k + c) = argminSet f := by
    ext i
    simp only [argminSet, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨fun h j => by simpa using (add_le_add_iff_right c).1 (h j),
           fun h j => by simpa using add_le_add_right (h j) c⟩
  simp [lexArgmin, hset]

/-- Scaling every score by a positive constant leaves the decision unchanged.
This is the multiplicative gauge freedom of Section 5 (`05:104`). -/
@[simp] theorem lexArgmin_const_mul {c : ℝ} (hc : 0 < c) (f : Fin K → ℝ) :
    lexArgmin (fun k => c * f k) = lexArgmin f := by
  classical
  have hset : argminSet (fun k => c * f k) = argminSet f := by
    ext i
    simp only [argminSet, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨fun h j => le_of_mul_le_mul_left (h j) hc,
           fun h j => mul_le_mul_of_nonneg_left (h j) hc.le⟩
  simp [lexArgmin, hset]

/-- Pointwise-equal score vectors induce the same decision. -/
theorem lexArgmin_congr {f g : Fin K → ℝ} (h : ∀ k, f k = g k) :
    lexArgmin f = lexArgmin g := by
  simp only [funext h]

/-- The tie-broken `argmax`, used for the `argmax`-form rules of Chapter 5. -/
noncomputable def lexArgmax (f : Fin K → ℝ) : Fin K := lexArgmin (fun k => -f k)

theorem lexArgmax_ge (f : Fin K → ℝ) (j : Fin K) : f j ≤ f (lexArgmax f) :=
  neg_le_neg_iff.1 (lexArgmin_le (fun k => -f k) j)

theorem lexArgmax_eq_of_strict {f : Fin K → ℝ} {i : Fin K}
    (h : ∀ j, j ≠ i → f j < f i) : lexArgmax f = i :=
  lexArgmin_eq_of_strict (by intro j hj; simpa using h j hj)

end Argmin

/-! ### Decision rules -/

section Decision

variable [NeZero K]

/-- The Bayes-optimal decision rule on the simplex,
`δ*(p) = argmin_i (C^T p)_i` (`eq:bayes_optimal_delta`). -/
noncomputable def bayesRule (C : CostMatrix K) (p : Fin K → ℝ) : Fin K :=
  lexArgmin (condRisk C p)

/-- The Bayes decision region of class `i` (`01:87`). -/
def bayesRegion (C : CostMatrix K) (i : Fin K) : Set (Fin K → ℝ) :=
  {p ∈ stdSimplex ℝ (Fin K) | ∀ j, condRisk C p i ≤ condRisk C p j}

theorem bayesRule_mem_region (C : CostMatrix K) {p : Fin K → ℝ}
    (hp : p ∈ stdSimplex ℝ (Fin K)) : p ∈ bayesRegion C (bayesRule C p) :=
  ⟨hp, fun j => lexArgmin_le _ j⟩

end Decision

end ThresholdOpt
