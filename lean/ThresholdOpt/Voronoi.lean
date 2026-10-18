import ThresholdOpt.Partition

/-!
# Voronoi tessellations (Chapter 6)

The Voronoi decision rule (`def:voronoi_rule`), the identity that makes the
cost-blind `argmax` a nearest-vertex rule, the Dominant Label Constraint
(`def:dlc`), the containment `H_PS ⊆ H_Vor` (`prop:ps_in_vor`), and the strict
risk separation of `thm:rotational_dominance`.

Distances are squared Euclidean throughout. `argmin` over squared distance and
over distance agree because `Real.sqrt` is monotone on nonnegatives, so nothing
is lost by avoiding the square root.
-/

namespace ThresholdOpt.Voronoi

open ThresholdOpt ThresholdOpt.Partition

variable {K : ℕ}

/-- Squared Euclidean distance on `Fin K → ℝ`. -/
def dsq (p q : Fin K → ℝ) : ℝ := ∑ i, (p i - q i) ^ 2

theorem dsq_nonneg (p q : Fin K → ℝ) : 0 ≤ dsq p q :=
  Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- A prototype configuration: one anchor per class. -/
abbrev Prototypes (K : ℕ) := Fin K → (Fin K → ℝ)

/-- **The Voronoi decision rule** (`def:voronoi_rule`): assign `p` to the class
of its nearest prototype, ties to the smallest index. -/
@[thesis "def:voronoi_rule"]
noncomputable def voronoiRule [NeZero K] (μ : Prototypes K) (p : Fin K → ℝ) :
    Fin K :=
  lexArgmin fun k => dsq p (μ k)

/-! ### The cost-blind `argmax` is a nearest-vertex rule

`06:8`. Expanding `‖p - e_i‖²` leaves `-2 p_i` as the only `i`-dependent term,
so minimizing distance to a vertex maximizes that coordinate. -/

/-- `d²(p, e_k) = ‖p‖² - 2 p_k + 1` (`eq:dsq_identity`). -/
@[thesis "eq:dsq_identity"]
theorem dsq_vertex (p : Fin K → ℝ) (k : Fin K) :
    dsq p (vertex k) = (∑ i, p i ^ 2) - 2 * p k + 1 := by
  have hterm : ∀ i : Fin K,
      (p i - vertex k i) ^ 2
        = p i ^ 2 + (if i = k then 1 - 2 * p k else 0) := by
    intro i
    by_cases h : i = k
    · subst h; simp [vertex]; ring
    · simp [vertex, h]
  simp only [dsq]
  rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_add_distrib,
    Finset.sum_ite_eq' Finset.univ k fun _ => 1 - 2 * p k]
  simp
  ring

/-- **The `argmax` rule is the nearest-vertex Voronoi rule** (`06:8`).

Placing the prototypes at the simplex vertices recovers the cost-blind
`argmax`, which is the sense in which the Voronoi class contains the default
rule. -/
theorem voronoi_vertex_eq_argmax [NeZero K] (p : Fin K → ℝ) :
    voronoiRule (fun k => vertex k) p = lexArgmax p := by
  have h : (fun k => dsq p (vertex k))
      = fun k => 2 * (fun j => -p j) k + ((∑ i, p i ^ 2) + 1) := by
    funext k
    rw [dsq_vertex]
    ring
  simp only [voronoiRule, lexArgmax, h, lexArgmin_add_const,
    lexArgmin_const_mul (by norm_num : (0:ℝ) < 2)]

/-! ### The Dominant Label Constraint -/

/-- **Dominant label constraint** (`def:dlc`, `eq:dlc`): each vertex is strictly
closest to its own prototype. -/
@[thesis "def:dlc"]
def DLC (μ : Prototypes K) : Prop :=
  ∀ k j, j ≠ k → dsq (vertex k) (μ k) < dsq (vertex k) (μ j)

/-- Under the DLC the rule labels each vertex by its own class. -/
theorem voronoiRule_vertex_of_DLC [NeZero K] {μ : Prototypes K} (h : DLC μ)
    (k : Fin K) :
    voronoiRule μ (vertex k) = k :=
  lexArgmin_eq_of_strict fun j hj => h k j hj

/-- **Shrink to vertex** (`06:129`).

If every prototype is placed within `λ√2` of its own vertex with `λ < 1/2`,
the DLC holds on every draw rather than merely with high probability. The proof
is the reverse triangle inequality: any other vertex is at distance at least
`(1-λ)√2`, and `λ < 1 - λ`.

Nonnegativity of `λ` is not needed: `λ² < (1-λ)²` reduces to `λ < 1/2`
outright. -/
theorem DLC_of_shrink {μ : Prototypes K} {lam : ℝ}
    (hhalf : lam < 1 / 2)
    (hclose : ∀ k, dsq (vertex k) (μ k) ≤ lam ^ 2 * 2)
    (hfar : ∀ k j, j ≠ k → (1 - lam) ^ 2 * 2 ≤ dsq (vertex k) (μ j)) :
    DLC μ := by
  intro k j hj
  have h1 : lam ^ 2 * 2 < (1 - lam) ^ 2 * 2 := by nlinarith
  exact lt_of_le_of_lt (hclose k) (lt_of_lt_of_le h1 (hfar k j hj))

/-! ### `H_PS ⊆ H_Vor`

`prop:ps_in_vor`. Placing the prototype for class `k` at `-½ c_k + β_k 1`
turns the squared distance into the anchor risk plus a per-class constant. -/

/-- The per-class constant produced by completing the square,
`G_k = K β_k² - (2 + c_k·1) β_k + ¼‖c_k‖²`. -/
noncomputable def gauge (C : CostMatrix K) (β : Fin K → ℝ) (k : Fin K) : ℝ :=
  (K : ℝ) * β k ^ 2 - (2 + ∑ i, C i k) * β k + (1 / 4) * ∑ i, C i k ^ 2

/-- The prototype configuration `μ_k = -½ c_k + β_k 1` of `prop:ps_in_vor`. -/
noncomputable def shiftedPrototypes (C : CostMatrix K) (β : Fin K → ℝ) :
    Prototypes K :=
  fun k i => -(1 / 2) * C i k + β k

/-- The completing-the-square identity behind `prop:ps_in_vor`: on the simplex
the squared distance to `μ_k` is the conditional risk of action `k`, plus a
term depending only on `k` through `gauge`, plus a term independent of `k`. -/
theorem dsq_shiftedPrototypes {C : CostMatrix K} {β : Fin K → ℝ}
    {p : Fin K → ℝ} (hp : p ∈ stdSimplex ℝ (Fin K)) (k : Fin K) :
    dsq p (shiftedPrototypes C β k)
      = (∑ i, p i ^ 2) + condRisk C p k + gauge C β k := by
  have hsum : ∑ i, p i = 1 := hp.2
  simp only [dsq, shiftedPrototypes, condRisk, gauge]
  have expand : ∀ i : Fin K,
      (p i - (-(1 / 2) * C i k + β k)) ^ 2
        = p i ^ 2 + C i k * p i - 2 * β k * p i
          + ((1 / 4) * C i k ^ 2 - β k * C i k + β k ^ 2) := by
    intro i; ring
  rw [Finset.sum_congr rfl fun i _ => expand i]
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib]
  rw [← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum]
  simp only [hsum, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  ring

/-- The conditional core of `prop:ps_in_vor`: if `β` solves
`gauge C β k = v k` for every `k`, the Voronoi rule with prototypes
`-½ c_k + β_k 1` agrees with the parallel-shift rule with offsets `v` at every
point of the simplex. -/
theorem voronoiRule_shiftedPrototypes [NeZero K] {C : CostMatrix K}
    {β v : Fin K → ℝ} (hβ : ∀ k, gauge C β k = v k) {p : Fin K → ℝ}
    (hp : p ∈ stdSimplex ℝ (Fin K)) :
    voronoiRule (shiftedPrototypes C β) p = psRule C v p := by
  have h : (fun k => dsq p (shiftedPrototypes C β k))
      = fun k => psScore C v p k + (∑ i, p i ^ 2) := by
    funext k
    rw [dsq_shiftedPrototypes hp k, hβ k, psScore]
    ring
  simp only [voronoiRule, psRule, h, lexArgmin_add_const]

/-! ### Solving the gauge quadratic

What makes `prop:ps_in_vor` a containment rather than a conditional statement
is that the quadratic `gauge C β k = v k + c` can always be solved once the
common constant `c` is taken large enough: `gauge` is a quadratic in `β k` with
leading coefficient `K > 0`, so raising `c` raises the discriminant without
bound. The shift by a single `c` is invisible to the rule
(`psRule_add_const`). -/

/-- A quadratic with positive leading coefficient and nonnegative discriminant
has a real root. -/
theorem exists_quadratic_root {a b e : ℝ} (ha : 0 < a)
    (hD : 0 ≤ b ^ 2 - 4 * a * e) :
    ∃ x : ℝ, a * x ^ 2 - b * x + e = 0 := by
  have ha' : a ≠ 0 := ne_of_gt ha
  have hs : Real.sqrt (b ^ 2 - 4 * a * e) ^ 2 = b ^ 2 - 4 * a * e :=
    Real.sq_sqrt hD
  -- Completing the square, generically in the root of the discriminant.
  have key : ∀ s : ℝ,
      a * ((b + s) / (2 * a)) ^ 2 - b * ((b + s) / (2 * a)) + e
        = (s ^ 2 - (b ^ 2 - 4 * a * e)) / (4 * a) := by
    intro s
    field_simp
    ring
  exact ⟨_, by rw [key, hs, sub_self, zero_div]⟩

/-- **The gauge system is solvable.** For any cost matrix and any offsets `v`
there are `β` and a single constant `c` with `gauge C β k = v k + c` for every
`k`. This is the discriminant condition of `prop:ps_in_vor`, discharged by
choosing `c` above the finitely many per-class thresholds. -/
theorem exists_gauge_solution [NeZero K] (C : CostMatrix K) (v : Fin K → ℝ) :
    ∃ (β : Fin K → ℝ) (c : ℝ), ∀ k, gauge C β k = v k + c := by
  have hK : (0 : ℝ) < K := by
    exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne K)
  obtain ⟨c, hc⟩ := Finite.exists_le (fun k : Fin K =>
    ((K : ℝ) * (∑ i, C i k ^ 2) - (2 + ∑ i, C i k) ^ 2) / (4 * K) - v k)
  have hroot : ∀ k : Fin K, ∃ x : ℝ,
      (K : ℝ) * x ^ 2 - (2 + ∑ i, C i k) * x
        + ((1 / 4) * ∑ i, C i k ^ 2 - (v k + c)) = 0 := by
    intro k
    refine exists_quadratic_root hK ?_
    have hck := hc k
    rw [sub_le_iff_le_add, div_le_iff₀ (by linarith : (0:ℝ) < 4 * K)] at hck
    nlinarith [hck]
  choose β hβ using hroot
  refine ⟨β, c, fun k => ?_⟩
  have := hβ k
  simp only [gauge]
  linarith

/-- **`H_PS ⊆ H_Vor`** (`prop:ps_in_vor`).

For any cost matrix and any parallel-shift rule there exist prototypes whose
Voronoi rule induces the same partition of the simplex. The prototypes are
`μ_k = -½ c_k + β_k 1` with `β` any solution of the gauge quadratic, which
`exists_gauge_solution` supplies.

This is the construction of the proposition, rule by rule. The statement
tagged with the label is `Complexity.HPS_ssubset_HVor`, which adds what this
one leaves out: the prototypes can be taken distinct, as `06:14` requires, and
the inclusion of classes is proper. -/
theorem ps_in_vor [NeZero K] (C : CostMatrix K) (v : Fin K → ℝ) :
    ∃ μ : Prototypes K, ∀ p ∈ stdSimplex ℝ (Fin K),
      voronoiRule μ p = psRule C v p := by
  obtain ⟨β, c, hβ⟩ := exists_gauge_solution C v
  refine ⟨shiftedPrototypes C β, fun p hp => ?_⟩
  rw [voronoiRule_shiftedPrototypes hβ hp]
  exact psRule_add_const C v c p

/-! ### `H_Vor = H_Full` on the simplex

The completing-the-square identity of `prop:ps_in_vor` uses nothing about the
cost matrix: it turns the squared distance to *any* prototype into a linear
score plus a term common to all classes, and conversely realizes *any* matrix
of linear scores by prototypes. On the simplex the unconstrained Voronoi rules
and the full-matrix rules of `eq:hfull_def` are therefore the same functions,
with the same smallest-index tie-break, since the two score vectors differ by a
constant at every point. The dissertation does not state this; it treats
`H_Vor` as intermediate between `H_PS` and `H_Full`. -/

/-- The partition matrix of a prototype configuration: column `k` is
`-2 μ_k + ‖μ_k‖² 1`. -/
noncomputable def vorMatrix (μ : Prototypes K) : CostMatrix K :=
  Matrix.of fun i k => -2 * μ k i + ∑ j, μ k j ^ 2

/-- On the simplex the squared distance to `μ_k` is the linear score of column
`k` of `vorMatrix μ`, plus `‖p‖²`. -/
theorem dsq_eq_condRisk_vorMatrix (μ : Prototypes K) {p : Fin K → ℝ}
    (hp : p ∈ stdSimplex ℝ (Fin K)) (k : Fin K) :
    dsq p (μ k) = condRisk (vorMatrix μ) p k + ∑ i, p i ^ 2 := by
  have hsum : ∑ i, p i = 1 := hp.2
  have expand : ∀ i : Fin K,
      (p i - μ k i) ^ 2 = -2 * μ k i * p i + μ k i ^ 2 + p i ^ 2 := by
    intro i; ring
  have hterm : ∀ i : Fin K,
      vorMatrix μ i k * p i
        = -2 * μ k i * p i + (∑ j, μ k j ^ 2) * p i := by
    intro i; simp only [vorMatrix, Matrix.of_apply]; ring
  simp only [dsq, condRisk]
  rw [Finset.sum_congr rfl fun i _ => expand i,
    Finset.sum_congr rfl fun i _ => hterm i]
  simp only [Finset.sum_add_distrib]
  rw [← Finset.mul_sum, hsum, mul_one]

/-- **`H_Vor ⊆ H_Full`, rule by rule.** Every Voronoi rule, with prototypes
anywhere in `ℝ^K` and not necessarily distinct, agrees at every point of the
simplex with the full-matrix rule of `vorMatrix μ`. -/
theorem voronoiRule_eq_fullRule [NeZero K] (μ : Prototypes K) {p : Fin K → ℝ}
    (hp : p ∈ stdSimplex ℝ (Fin K)) :
    voronoiRule μ p = lexArgmin (condRisk (vorMatrix μ) p) := by
  have h : (fun k => dsq p (μ k))
      = fun k => condRisk (vorMatrix μ) p k + ∑ i, p i ^ 2 := by
    funext k; exact dsq_eq_condRisk_vorMatrix μ hp k
  simp only [voronoiRule, h, lexArgmin_add_const]

/-- **`H_Full ⊆ H_Vor`, rule by rule.** For any real matrix `A` there are
prototypes whose Voronoi rule agrees with `argmin_k a_k^T p` at every point of
the simplex. This is `ps_in_vor` with zero offsets: its proof never uses that
the matrix is a cost matrix. -/
theorem full_in_vor [NeZero K] (A : CostMatrix K) :
    ∃ μ : Prototypes K, ∀ p ∈ stdSimplex ℝ (Fin K),
      voronoiRule μ p = lexArgmin (condRisk A p) := by
  obtain ⟨μ, hμ⟩ := ps_in_vor A 0
  refine ⟨μ, fun p hp => ?_⟩
  rw [hμ p hp]
  exact lexArgmin_congr fun k => by simp [psScore]

/-! ### Distinct prototypes

`06:14` asks for `K` *distinct* prototypes. The construction above can return
coincident ones when `A` has repeated columns. The requirement costs nothing:
a column equal to an earlier one never wins under the smallest-index
tie-break, so its score can be raised by a constant without changing the rule,
and prototypes built from pairwise distinct columns are pairwise distinct. -/

/-- Raising scores by nonnegative amounts that vanish at the selected index
leaves the selection unchanged. -/
theorem lexArgmin_add_of_zero [NeZero K] {s r : Fin K → ℝ} (hr : ∀ k, 0 ≤ r k)
    (h0 : r (lexArgmin s) = 0) :
    lexArgmin (fun k => s k + r k) = lexArgmin s := by
  refine le_antisymm (lexArgmin_le_index fun j => ?_)
    (lexArgmin_le_index fun j => ?_)
  · have := lexArgmin_le s j
    simp only [h0, add_zero]
    linarith [hr j]
  · have h1 := lexArgmin_le (fun k => s k + r k) (lexArgmin s)
    simp only [h0, add_zero] at h1
    have h2 := lexArgmin_le s j
    linarith [hr (lexArgmin fun k => s k + r k)]

/-- Prototypes built from pairwise distinct columns with one common gauge
constant are pairwise distinct. -/
theorem shiftedPrototypes_injective {C : CostMatrix K} {β : Fin K → ℝ} {c : ℝ}
    (hβ : ∀ k, gauge C β k = c)
    (hcol : ∀ k j, (∀ i, C i k = C i j) → k = j) :
    Function.Injective (shiftedPrototypes C β) := by
  intro k j hkj
  refine hcol k j fun i => ?_
  have h1 := dsq_shiftedPrototypes (C := C) (β := β) (vertex_mem_stdSimplex i) k
  have h2 := dsq_shiftedPrototypes (C := C) (β := β) (vertex_mem_stdSimplex i) j
  rw [hkj, h2, hβ k, hβ j, condRisk_vertex, condRisk_vertex] at h1
  linarith

open scoped Classical in
/-- Any matrix can be replaced by one with pairwise distinct columns that
induces the same rule on the simplex. -/
theorem exists_distinct_columns [NeZero K] (A : CostMatrix K) :
    ∃ A' : CostMatrix K, (∀ k j, (∀ i, A' i k = A' i j) → k = j) ∧
      ∀ p ∈ stdSimplex ℝ (Fin K),
        lexArgmin (condRisk A' p) = lexArgmin (condRisk A p) := by
  obtain ⟨M, hM⟩ :=
    Finite.exists_le (fun q : Fin K × Fin K => |A 0 q.1 - A 0 q.2|)
  -- `dup k`: column `k` repeats an earlier column, so class `k` never wins.
  let dup : Fin K → Prop := fun k => ∃ j, j < k ∧ ∀ i, A i j = A i k
  let r : Fin K → ℝ := fun k => if dup k then (M + 1) * ((k : ℕ) + 1) else 0
  have hM0 : 0 ≤ M := le_trans (abs_nonneg _) (hM (0, 0))
  have hr : ∀ k, 0 ≤ r k := by
    intro k
    simp only [r]
    split_ifs
    · positivity
    · exact le_rfl
  have hrdup : ∀ k, dup k → M + 1 ≤ r k := by
    intro k hk
    simp only [r, if_pos hk]
    have : (0 : ℝ) ≤ ((k : ℕ) : ℝ) := Nat.cast_nonneg _
    nlinarith
  refine ⟨Matrix.of fun i k => A i k + r k, ?_, ?_⟩
  · intro k j hcol
    simp only [Matrix.of_apply] at hcol
    by_contra hne
    have h0 := hcol 0
    have habs1 := abs_le.1 (hM (k, j))
    have habs2 := abs_le.1 (hM (j, k))
    simp only at habs1 habs2
    by_cases hk : dup k <;> by_cases hj : dup j
    · -- both raised: the two raises differ by at least `M + 1`
      simp only [r, if_pos hk, if_pos hj] at h0
      rcases lt_or_gt_of_ne hne with hlt | hlt
      · have : ((k : ℕ) : ℝ) + 1 ≤ ((j : ℕ) : ℝ) := by exact_mod_cast hlt
        nlinarith
      · have : ((j : ℕ) : ℝ) + 1 ≤ ((k : ℕ) : ℝ) := by exact_mod_cast hlt
        nlinarith
    · have h1 := hrdup k hk
      simp only [r, if_neg hj, add_zero] at h0
      linarith [habs2.1, habs2.2]
    · have h1 := hrdup j hj
      simp only [r, if_neg hk, add_zero] at h0
      linarith [habs1.1, habs1.2]
    · -- neither raised: the columns of `A` coincide, so the later one is `dup`
      have hA : ∀ i, A i k = A i j := by
        intro i
        have := hcol i
        simp only [r, if_neg hk, if_neg hj, add_zero] at this
        exact this
      rcases lt_or_gt_of_ne hne with hlt | hlt
      · exact hj ⟨k, hlt, hA⟩
      · exact hk ⟨j, hlt, fun i => (hA i).symm⟩
  · intro p hp
    have hsum : ∑ i, p i = 1 := hp.2
    have hscore : condRisk (Matrix.of fun i k => A i k + r k) p
        = fun k => condRisk A p k + r k := by
      funext k
      have hterm : ∀ i : Fin K,
          (Matrix.of fun i k => A i k + r k : CostMatrix K) i k * p i
            = A i k * p i + r k * p i := by
        intro i; simp only [Matrix.of_apply]; ring
      simp only [condRisk]
      rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_add_distrib,
        ← Finset.mul_sum, hsum, mul_one]
    rw [hscore]
    refine lexArgmin_add_of_zero hr ?_
    by_contra hne
    have hdup : dup (lexArgmin (condRisk A p)) := by
      by_contra hnd
      exact hne (by simp only [r, if_neg hnd])
    obtain ⟨j, hlt, hA⟩ := hdup
    have hmin : ∀ l, condRisk A p j ≤ condRisk A p l := by
      intro l
      have : condRisk A p j = condRisk A p (lexArgmin (condRisk A p)) := by
        simp only [condRisk, hA]
      rw [this]
      exact lexArgmin_le _ l
    exact absurd (lexArgmin_le_index hmin) (not_le.2 hlt)

/-- **`H_Full ⊆ H_Vor` with `K` distinct prototypes.** The requirement of
`06:14` that the prototypes be distinct does not shrink the class: every
full-matrix rule is the Voronoi rule of pairwise distinct prototypes at every
point of the simplex, ties included. -/
theorem full_in_vor_injective [NeZero K] (A : CostMatrix K) :
    ∃ μ : Prototypes K, Function.Injective μ ∧ ∀ p ∈ stdSimplex ℝ (Fin K),
      voronoiRule μ p = lexArgmin (condRisk A p) := by
  obtain ⟨A', hcol, hrule⟩ := exists_distinct_columns A
  obtain ⟨β, c, hβ⟩ := exists_gauge_solution A' 0
  have hβ' : ∀ k, gauge A' β k = c := fun k => by simpa using hβ k
  refine ⟨shiftedPrototypes A' β, shiftedPrototypes_injective hβ' hcol,
    fun p hp => ?_⟩
  rw [voronoiRule_shiftedPrototypes (v := fun _ => c) hβ' hp, ← hrule p hp]
  have h : psScore A' (fun _ => c) p = fun k => condRisk A' p k + c := by
    funext k; rfl
  simp only [psRule, h, lexArgmin_add_const]

end ThresholdOpt.Voronoi
