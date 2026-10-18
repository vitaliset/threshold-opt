import ThresholdOpt.Basic

/-!
# Multiclass classification (Chapter 4)

Admissibility (`def:admissibility`), element-wise reasonableness
(`def:elementwise`), the strict implication between them
(`prop:elementwise_admissible`), and the rigid polyhedral partition of the
simplex induced by a cost matrix (`thm:rigid_partition`).
-/

namespace ThresholdOpt.Multiclass

open ThresholdOpt

variable {K : ℕ}

/-- **Admissible cost matrix** (`def:admissibility`).

Every class is the unique risk-minimizer at some point of the simplex, so no
decision region is empty. -/
def Admissible (C : CostMatrix K) : Prop :=
  ∀ i, ∃ p ∈ stdSimplex ℝ (Fin K), ∀ j, j ≠ i → condRisk C p i < condRisk C p j

/-- **Element-wise reasonable cost matrix** (`def:elementwise`).

For every true class `j`, a correct prediction costs strictly less than any
misprediction: `C_{jj} < C_{ji}` for `i ≠ j`. -/
def ElementwiseReasonable (C : CostMatrix K) : Prop :=
  ∀ j i, i ≠ j → C j j < C j i

attribute [thesis "def:admissibility"] Admissible
attribute [thesis "def:elementwise"] ElementwiseReasonable

/-- **Element-wise reasonableness implies admissibility**
(`prop:elementwise_admissible`, part 1).

The witness for class `j` is the vertex `e_j`, where the risk of action `i` is
exactly the matrix entry `C j i`.

The statement tagged with the label, with both parts of the proposition, is
`Examples.elementwise_admissible`. -/
theorem admissible_of_elementwiseReasonable {C : CostMatrix K}
    (h : ElementwiseReasonable C) : Admissible C := by
  intro i
  refine ⟨vertex i, vertex_mem_stdSimplex i, ?_⟩
  intro j hj
  simpa [condRisk_vertex] using h i j hj

/-- At a vertex the Bayes rule of an element-wise reasonable matrix returns the
vertex's own class. This is the "undersmoothing" observation of `04:145`: when
the estimate concentrates on one class, the plug-in rule reduces to the
cost-blind `argmax`. -/
theorem bayesRule_vertex_eq [NeZero K] {C : CostMatrix K}
    (h : ElementwiseReasonable C) (i : Fin K) :
    bayesRule C (vertex i) = i := by
  refine lexArgmin_eq_of_strict ?_
  intro j hj
  simpa [condRisk_vertex] using h i j hj

/-! ### The rigid partition -/

/-- The risk difference between two actions is linear in the probability
vector. This is what makes every decision boundary a hyperplane. -/
theorem isLinearMap_condRisk_sub (C : CostMatrix K) (i j : Fin K) :
    IsLinearMap ℝ (fun p : Fin K → ℝ => condRisk C p i - condRisk C p j) := by
  constructor
  · intro p q
    simp only [condRisk, Pi.add_apply, mul_add, Finset.sum_add_distrib]
    ring
  · intro c p
    simp only [condRisk, Pi.smul_apply, smul_eq_mul]
    rw [show (∑ x, C x i * (c * p x)) = c * ∑ x, C x i * p x by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun x _ => by ring,
        show (∑ x, C x j * (c * p x)) = c * ∑ x, C x j * p x by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun x _ => by ring]
    ring

/-- The separating hyperplane between actions `i` and `j`, with normal
`c_i - c_j` (`thm:rigid_partition`). -/
def boundary (C : CostMatrix K) (i j : Fin K) : Set (Fin K → ℝ) :=
  {p | condRisk C p i - condRisk C p j = 0}

/-- The functional defining the `i`-versus-`j` boundary is the inner product
with `c_i - c_j`. This is the sense in which the normal of the boundary is
fixed by the cost matrix alone: it does not depend on `p`, and there is no free
parameter anywhere in it. -/
theorem condRisk_sub_eq_inner (C : CostMatrix K) (i j : Fin K)
    (p : Fin K → ℝ) :
    condRisk C p i - condRisk C p j = ∑ k, (C k i - C k j) * p k := by
  simp only [condRisk, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun k _ => by ring

/-- The decision region of `i` is the simplex cut by the half-spaces
`𝓛_i ≤ 𝓛_j`. This is the polyhedral part of `thm:rigid_partition`. -/
theorem bayesRegion_eq_inter (C : CostMatrix K) (i : Fin K) :
    bayesRegion C i
      = stdSimplex ℝ (Fin K)
        ∩ ⋂ j, {p | condRisk C p i - condRisk C p j ≤ 0} := by
  ext p
  constructor
  · rintro ⟨hp, hmin⟩
    exact ⟨hp, Set.mem_iInter.2 fun j => sub_nonpos.2 (hmin j)⟩
  · rintro ⟨hp, hint⟩
    exact ⟨hp, fun j => sub_nonpos.1 (Set.mem_iInter.1 hint j)⟩

/-- Each Bayes decision region is an intersection of half-spaces with the
simplex, hence convex. -/
theorem convex_bayesRegion (C : CostMatrix K) (i : Fin K) :
    Convex ℝ (bayesRegion C i) := by
  rw [bayesRegion_eq_inter]
  refine (convex_stdSimplex ℝ (Fin K)).inter (convex_iInter fun j => ?_)
  exact convex_halfSpace_le (isLinearMap_condRisk_sub C i j) 0

/-- **Rigid polyhedral partition** (`thm:rigid_partition`).

Four claims, matching the theorem as stated in the text:

1. each decision region is the simplex cut by half-spaces, so the partition is
   polyhedral;
2. each bounding functional is the inner product with `c_i - c_j`, so the
   normal of every boundary is determined by the cost matrix with no free
   parameter, which is the "rigid" part;
3. each region is convex;
4. the regions cover the simplex, the `argmin` rule placing every point in one
   of them. -/
@[thesis "thm:rigid_partition"]
theorem rigid_partition [NeZero K] (C : CostMatrix K) :
    (∀ i, bayesRegion C i
      = stdSimplex ℝ (Fin K)
        ∩ ⋂ j, {p | condRisk C p i - condRisk C p j ≤ 0}) ∧
    (∀ i j p, condRisk C p i - condRisk C p j
      = ∑ k, (C k i - C k j) * p k) ∧
    (∀ i, Convex ℝ (bayesRegion C i)) ∧
    (∀ p ∈ stdSimplex ℝ (Fin K), p ∈ bayesRegion C (bayesRule C p)) :=
  ⟨bayesRegion_eq_inter C, condRisk_sub_eq_inner C, convex_bayesRegion C,
    fun _ hp => bayesRule_mem_region C hp⟩

end ThresholdOpt.Multiclass
