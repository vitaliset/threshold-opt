import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Tactic
import ThresholdOpt.Multiclass

/-!
# The two cost-matrix witnesses (Chapter 4)

Machine-checked replacements for the hand verifications of Appendix A:
`ex:inadmissible`, a matrix whose inadmissibility is invisible in its entries,
and `ex:hedge`, an admissible matrix that is not element-wise reasonable.

Together they witness that the inclusion of `prop:elementwise_admissible` is
strict.

Each matrix's nine entries are recorded as `simp` lemmas. The `!![...]`
notation does not reduce on its own at literal `Fin 3` indices, and stating the
entries explicitly also puts the numbers of Appendix A in the source where they
can be read against the text.
-/

namespace ThresholdOpt.Examples

open ThresholdOpt ThresholdOpt.Multiclass

/-- Case analysis on `Fin 3` that substitutes numerals rather than
`⟨n, _⟩` terms, so the entry `simp` lemmas below apply directly. -/
theorem fin3_cases (j : Fin 3) : j = 0 ∨ j = 1 ∨ j = 2 := by
  revert j
  decide

/-! ### An inadmissible matrix with no element-wise tell (`ex:inadmissible`) -/

/-- `C = [[0, 6, 10], [4, 6, 4], [10, 6, 0]]`. Class 1 is not dominated
element-wise by any single alternative, yet it is optimal nowhere. -/
noncomputable def Cinad : CostMatrix 3 :=
  Matrix.of ![![0, 6, 10], ![4, 6, 4], ![10, 6, 0]]

@[simp] theorem Cinad_00 : Cinad 0 0 = 0 := rfl
@[simp] theorem Cinad_01 : Cinad 0 1 = 6 := rfl
@[simp] theorem Cinad_02 : Cinad 0 2 = 10 := rfl
@[simp] theorem Cinad_10 : Cinad 1 0 = 4 := rfl
@[simp] theorem Cinad_11 : Cinad 1 1 = 6 := rfl
@[simp] theorem Cinad_12 : Cinad 1 2 = 4 := rfl
@[simp] theorem Cinad_20 : Cinad 2 0 = 10 := rfl
@[simp] theorem Cinad_21 : Cinad 2 1 = 6 := rfl
@[simp] theorem Cinad_22 : Cinad 2 2 = 0 := rfl

/-- The risk of action 1 is the constant `6` on the whole simplex: its cost
column is `(6, 6, 6)`. -/
theorem condRisk_one_inad {p : Fin 3 → ℝ}
    (hp : p ∈ stdSimplex ℝ (Fin 3)) : condRisk Cinad p 1 = 6 := by
  have hsum : ∑ i, p i = 1 := hp.2
  rw [Fin.sum_univ_three] at hsum
  simp only [condRisk, Fin.sum_univ_three, Cinad_01, Cinad_11, Cinad_21]
  linarith

/-- The lower envelope of actions 0 and 2 never exceeds `5`, because their
average is `5 p_0 + 4 p_1 + 5 p_2 ≤ 5`. -/
theorem min_lower_envelope {p : Fin 3 → ℝ} (hp : p ∈ stdSimplex ℝ (Fin 3)) :
    min (condRisk Cinad p 0) (condRisk Cinad p 2) ≤ 5 := by
  have hsum : ∑ i, p i = 1 := hp.2
  rw [Fin.sum_univ_three] at hsum
  have h0 := hp.1 0
  have h1 := hp.1 1
  have h2 := hp.1 2
  have hexp0 : condRisk Cinad p 0 = 4 * p 1 + 10 * p 2 := by
    simp only [condRisk, Fin.sum_univ_three, Cinad_00, Cinad_10, Cinad_20]
    ring
  have hexp2 : condRisk Cinad p 2 = 10 * p 0 + 4 * p 1 := by
    simp only [condRisk, Fin.sum_univ_three, Cinad_02, Cinad_12, Cinad_22]
    ring
  rcases le_total (condRisk Cinad p 0) (condRisk Cinad p 2) with h | h
  · rw [min_eq_left h]; rw [hexp0] at *; linarith
  · rw [min_eq_right h]; rw [hexp2] at *; linarith

/-- **`ex:inadmissible` is inadmissible.**

Action 1 costs `6` everywhere while the better of actions 0 and 2 costs at most
`5`, so class 1's decision region is empty. -/
@[thesis "ex:inadmissible"]
theorem not_admissible_inad : ¬ Admissible Cinad := by
  intro h
  obtain ⟨p, hp, hlt⟩ := h 1
  have h6 := condRisk_one_inad hp
  have henv := min_lower_envelope hp
  have hlt0 : condRisk Cinad p 1 < condRisk Cinad p 0 := hlt 0 (by decide)
  have hlt2 : condRisk Cinad p 1 < condRisk Cinad p 2 := hlt 2 (by decide)
  rw [h6] at hlt0 hlt2
  rcases min_le_iff.1 henv with h | h <;> linarith

/-- The inadmissibility is genuinely invisible in the entries: class 1 beats
class 0 when the truth is 2, and beats class 2 when the truth is 0. -/
theorem inad_not_entrywise_dominated :
    Cinad 2 1 < Cinad 2 0 ∧ Cinad 0 1 < Cinad 0 2 := by
  constructor <;> norm_num

/-! ### An admissible matrix that is not element-wise reasonable
(`ex:hedge`) -/

/-- `C = [[0, 10, 2], [10, 0, 2], [3, 3, 5]]`. Classes 0 and 1 are a costly
confusable pair; class 2 is a cheap hedge that is optimal at no vertex. -/
noncomputable def Chedge : CostMatrix 3 :=
  Matrix.of ![![0, 10, 2], ![10, 0, 2], ![3, 3, 5]]

@[simp] theorem Chedge_00 : Chedge 0 0 = 0 := rfl
@[simp] theorem Chedge_01 : Chedge 0 1 = 10 := rfl
@[simp] theorem Chedge_02 : Chedge 0 2 = 2 := rfl
@[simp] theorem Chedge_10 : Chedge 1 0 = 10 := rfl
@[simp] theorem Chedge_11 : Chedge 1 1 = 0 := rfl
@[simp] theorem Chedge_12 : Chedge 1 2 = 2 := rfl
@[simp] theorem Chedge_20 : Chedge 2 0 = 3 := rfl
@[simp] theorem Chedge_21 : Chedge 2 1 = 3 := rfl
@[simp] theorem Chedge_22 : Chedge 2 2 = 5 := rfl

/-- The witness for class 2: at `(1/2, 1/2, 0)` the hedge costs `2` against the
`5` shared by the confusable pair. The point is the midpoint of the edge from
`e_0` to `e_1`, on the boundary of the simplex. -/
noncomputable def hedgeWitness : Fin 3 → ℝ := ![1 / 2, 1 / 2, 0]

@[simp] theorem hedgeWitness_0 : hedgeWitness 0 = 1 / 2 := rfl
@[simp] theorem hedgeWitness_1 : hedgeWitness 1 = 1 / 2 := rfl
@[simp] theorem hedgeWitness_2 : hedgeWitness 2 = 0 := rfl

theorem hedgeWitness_mem : hedgeWitness ∈ stdSimplex ℝ (Fin 3) := by
  constructor
  · intro i
    rcases fin3_cases i with rfl | rfl | rfl <;> norm_num
  · rw [Fin.sum_univ_three]; norm_num

/-- **`ex:hedge` is admissible.**

Classes 0 and 1 are uniquely optimal at their own vertices, and class 2 is
uniquely optimal at `(1/2, 1/2, 0)`. -/
theorem admissible_hedge : Admissible Chedge := by
  have h2 : condRisk Chedge hedgeWitness 2 = 2 := by
    simp only [condRisk, Fin.sum_univ_three, Chedge_02, Chedge_12, Chedge_22]
    norm_num
  have h0 : condRisk Chedge hedgeWitness 0 = 5 := by
    simp only [condRisk, Fin.sum_univ_three, Chedge_00, Chedge_10, Chedge_20]
    norm_num
  have h1 : condRisk Chedge hedgeWitness 1 = 5 := by
    simp only [condRisk, Fin.sum_univ_three, Chedge_01, Chedge_11, Chedge_21]
    norm_num
  intro i
  rcases fin3_cases i with rfl | rfl | rfl
  · refine ⟨vertex 0, vertex_mem_stdSimplex 0, ?_⟩
    intro j hj
    rw [condRisk_vertex, condRisk_vertex]
    rcases fin3_cases j with rfl | rfl | rfl
    · exact absurd rfl hj
    · norm_num
    · norm_num
  · refine ⟨vertex 1, vertex_mem_stdSimplex 1, ?_⟩
    intro j hj
    rw [condRisk_vertex, condRisk_vertex]
    rcases fin3_cases j with rfl | rfl | rfl
    · norm_num
    · exact absurd rfl hj
    · norm_num
  · refine ⟨hedgeWitness, hedgeWitness_mem, ?_⟩
    intro j hj
    rcases fin3_cases j with rfl | rfl | rfl
    · rw [h2, h0]; norm_num
    · rw [h2, h1]; norm_num
    · exact absurd rfl hj

/-- **`ex:hedge` is not element-wise reasonable.**

At the true class 2 the correct prediction costs `C_{22} = 5` while predicting
0 or 1 costs only `3`. -/
theorem not_elementwiseReasonable_hedge : ¬ ElementwiseReasonable Chedge := by
  intro h
  have := h 2 0 (by decide)
  rw [Chedge_22, Chedge_20] at this
  norm_num at this

/-- The hedge class is not a minimizer at any vertex. -/
theorem hedge_not_optimal_at_vertex (j : Fin 3) : ∃ k, Chedge j k < Chedge j 2 := by
  rcases fin3_cases j with rfl | rfl | rfl
  · exact ⟨0, by norm_num⟩
  · exact ⟨1, by norm_num⟩
  · exact ⟨0, by norm_num⟩

/-- The hedge class is the strict minimizer at `(1/2, 1/2, 0)`. -/
theorem hedge_strict_at_witness (j : Fin 3) (hj : j ≠ 2) :
    condRisk Chedge hedgeWitness 2 < condRisk Chedge hedgeWitness j := by
  rcases fin3_cases j with rfl | rfl | rfl
  · simp only [condRisk, Fin.sum_univ_three, Chedge_00, Chedge_10, Chedge_20,
      Chedge_02, Chedge_12, Chedge_22, hedgeWitness_0, hedgeWitness_1, hedgeWitness_2]
    norm_num
  · simp only [condRisk, Fin.sum_univ_three, Chedge_01, Chedge_11, Chedge_21,
      Chedge_02, Chedge_12, Chedge_22, hedgeWitness_0, hedgeWitness_1, hedgeWitness_2]
    norm_num
  · exact absurd rfl hj

/-- **`ex:hedge`**: the matrix is admissible and not element-wise reasonable,
its hedge class minimizes risk at no vertex, and is the strict minimizer at
`(1/2, 1/2, 0)`. For the corresponding general statement for arbitrary cost
matrices, see `Multiclass.minimizer_exists_on_boundary`. -/
@[thesis "ex:hedge"]
theorem elementwise_strictly_stronger :
    Admissible Chedge ∧ ¬ ElementwiseReasonable Chedge
      ∧ (∀ j, ∃ k, Chedge j k < Chedge j 2)
      ∧ hedgeWitness 2 = 0
      ∧ ∀ j, j ≠ 2 → condRisk Chedge hedgeWitness 2 < condRisk Chedge hedgeWitness j :=
  ⟨admissible_hedge, not_elementwiseReasonable_hedge, hedge_not_optimal_at_vertex,
    rfl, hedge_strict_at_witness⟩

/-! ### Strictness for every `K ≥ 2` (`prop:elementwise_admissible`, part 2)

Part (2) of the proposition makes three claims, each proved below.

* The inclusion is strict for every `K ≥ 2`. The witness is the `0/1` loss
  with the labels of two actions exchanged: it is admissible, since each class
  is the unique minimizer at the vertex of the class it was exchanged with,
  and it is not element-wise reasonable.
* At `K = 2` there is nothing else: an admissible `2 × 2` matrix is
  element-wise reasonable or becomes so after exchanging the two actions, and
  in both cases each class is the strict minimizer at a vertex.
* For every `K ≥ 3` there is an admissible matrix with a class that is not a
  minimizer at any vertex, not even a tied one. The witness extends `ex:hedge`
  by classes that are strictly optimal at their own vertices, as the proof in
  Appendix A indicates.

The text describes such a class as "risk-optimal solely in the interior of
`Δ^{K-1}` and at no vertex". Only "at no vertex" is true: in every witness
here the class is the strict minimizer at a boundary point, the midpoint of an
edge, and this is proved too. See `elementwise_strictly_stronger`. -/

section Strictness

variable {K : ℕ}

/-- The `0/1` loss with the labels of actions `a` and `b` exchanged. -/
noncomputable def swappedZeroOne (a b : Fin K) : CostMatrix K :=
  Matrix.of fun i j => if i = Equiv.swap a b j then (0 : ℝ) else 1

theorem swappedZeroOne_nonneg (a b : Fin K) (i j : Fin K) :
    0 ≤ swappedZeroOne a b i j := by
  simp only [swappedZeroOne, Matrix.of_apply]
  split_ifs <;> norm_num

theorem admissible_swappedZeroOne (a b : Fin K) :
    Admissible (swappedZeroOne a b) := by
  intro j
  refine ⟨vertex (Equiv.swap a b j), vertex_mem_stdSimplex _, fun k hk => ?_⟩
  rw [condRisk_vertex, condRisk_vertex]
  have hne : Equiv.swap a b j ≠ Equiv.swap a b k :=
    fun h => hk ((Equiv.swap a b).injective h).symm
  simp [swappedZeroOne, hne]

theorem not_elementwiseReasonable_swappedZeroOne {a b : Fin K} (hab : a ≠ b) :
    ¬ ElementwiseReasonable (swappedZeroOne a b) := by
  intro h
  have := h a b hab.symm
  simp [swappedZeroOne, hab] at this
  linarith

/-- **The inclusion is strict for every `K ≥ 2`.** -/
theorem exists_admissible_not_elementwise (hK : 2 ≤ K) :
    ∃ C : CostMatrix K, (∀ i j, 0 ≤ C i j) ∧ Admissible C
      ∧ ¬ ElementwiseReasonable C := by
  have hne : (⟨0, by omega⟩ : Fin K) ≠ ⟨1, by omega⟩ := by
    simp [Fin.ext_iff]
  exact ⟨swappedZeroOne ⟨0, by omega⟩ ⟨1, by omega⟩, swappedZeroOne_nonneg _ _,
    admissible_swappedZeroOne _ _, not_elementwiseReasonable_swappedZeroOne hne⟩

/-! #### `K = 2` -/

/-- An admissible `2 × 2` matrix has the entry pattern of an element-wise
reasonable matrix, or that pattern with the two actions exchanged. This is the
affine sign argument of Appendix A: the risk difference of the two actions is
affine on the segment and takes both signs, so its values at the two endpoints
have strict opposite signs. -/
theorem admissible_two_cases {C : CostMatrix 2} (h : Admissible C) :
    (C 0 0 < C 0 1 ∧ C 1 1 < C 1 0) ∨ (C 0 1 < C 0 0 ∧ C 1 0 < C 1 1) := by
  obtain ⟨p, hp, hp0⟩ := h 0
  obtain ⟨q, hq, hq1⟩ := h 1
  have h0 := hp0 1 (by decide)
  have h1 := hq1 0 (by decide)
  simp only [condRisk, Fin.sum_univ_two] at h0 h1
  have p0 := hp.1 0
  have p1 := hp.1 1
  have q0 := hq.1 0
  have q1 := hq.1 1
  rcases lt_or_ge (C 0 0) (C 0 1) with a | a
  · rcases lt_or_ge (C 1 1) (C 1 0) with b | b
    · exact Or.inl ⟨a, b⟩
    · nlinarith [mul_nonneg q0 (sub_nonneg.2 a.le), mul_nonneg q1 (sub_nonneg.2 b)]
  · rcases lt_or_ge (C 1 0) (C 1 1) with b | b
    · rcases lt_or_ge (C 0 1) (C 0 0) with c | c
      · exact Or.inr ⟨c, b⟩
      · nlinarith [mul_nonneg q0 (sub_nonneg.2 c), mul_nonneg q1 (sub_nonneg.2 b.le)]
    · nlinarith [mul_nonneg p0 (sub_nonneg.2 a), mul_nonneg p1 (sub_nonneg.2 b)]

/-- **At `K = 2` the only admissible matrices outside the element-wise class
are obtained by relabeling the two actions.** -/
theorem elementwise_or_swapped_of_admissible_two {C : CostMatrix 2}
    (h : Admissible C) :
    ElementwiseReasonable C
      ∨ ElementwiseReasonable (C.submatrix id (Equiv.swap (0 : Fin 2) 1)) := by
  rcases admissible_two_cases h with ⟨a, b⟩ | ⟨a, b⟩
  · left
    refine Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨fun h => absurd rfl h,
      fun _ => a⟩, Fin.forall_fin_two.2 ⟨fun _ => b, fun h => absurd rfl h⟩⟩
  · right
    refine Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨fun h => absurd rfl h,
      fun _ => ?_⟩, Fin.forall_fin_two.2 ⟨fun _ => ?_, fun h => absurd rfl h⟩⟩
    · simpa using a
    · simpa using b

/-- **At `K = 2` every class is optimal at a vertex**: each class of an
admissible matrix is the strict minimizer at some vertex. -/
theorem strict_at_vertex_of_admissible_two {C : CostMatrix 2}
    (h : Admissible C) (i : Fin 2) : ∃ j, ∀ k, k ≠ i → C j i < C j k := by
  rcases admissible_two_cases h with ⟨a, b⟩ | ⟨a, b⟩
  · refine ⟨i, ?_⟩
    revert i
    exact Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨fun h => absurd rfl h,
      fun _ => a⟩, Fin.forall_fin_two.2 ⟨fun _ => b, fun h => absurd rfl h⟩⟩
  · revert i
    refine Fin.forall_fin_two.2 ⟨⟨1, ?_⟩, ⟨0, ?_⟩⟩
    · exact Fin.forall_fin_two.2 ⟨fun h => absurd rfl h, fun _ => b⟩
    · exact Fin.forall_fin_two.2 ⟨fun _ => a, fun h => absurd rfl h⟩

/-! #### `K ≥ 3`: the hedge, extended -/

/-- `ex:hedge` extended to `K` classes: the first three rows and columns are
`Chedge`, and every further class costs `0` when correct and `10` otherwise
(`3` when the truth is the hedge class). -/
noncomputable def hedgeK (K : ℕ) : CostMatrix K :=
  Matrix.of fun i j =>
    if (j : ℕ) = 2 then (if (i : ℕ) = 2 then 5 else if (i : ℕ) ≤ 1 then 2 else 10)
    else if (i : ℕ) = (j : ℕ) then 0
    else if (i : ℕ) = 2 then 3 else 10

theorem hedgeK_nonneg (i j : Fin K) : 0 ≤ hedgeK K i j := by
  simp only [hedgeK, Matrix.of_apply]
  split_ifs <;> norm_num

/-- The extension restricts to `Chedge` on the first three classes. -/
theorem hedgeK_three : hedgeK 3 = Chedge := by
  ext i j
  rcases fin3_cases i with rfl | rfl | rfl <;>
    rcases fin3_cases j with rfl | rfl | rfl <;>
    simp [hedgeK]

theorem hedgeK_diag {i : Fin K} (hi : (i : ℕ) ≠ 2) : hedgeK K i i = 0 := by
  simp [hedgeK, hi]

theorem two_le_hedgeK_of_ne {i j : Fin K} (hi : (i : ℕ) ≠ 2) (hij : j ≠ i) :
    2 ≤ hedgeK K i j := by
  have hv : (i : ℕ) ≠ (j : ℕ) := fun h => hij (Fin.ext h).symm
  simp only [hedgeK, Matrix.of_apply, if_neg hv, if_neg hi]
  split_ifs <;> norm_num

/-- The midpoint of the vertices of classes `0` and `1`. -/
noncomputable def hedgeMid (a b : Fin K) : Fin K → ℝ :=
  fun l => (vertex a l + vertex b l) / 2

theorem hedgeMid_mem (a b : Fin K) : hedgeMid a b ∈ stdSimplex ℝ (Fin K) := by
  have ha := vertex_mem_stdSimplex a
  have hb := vertex_mem_stdSimplex b
  constructor
  · intro l
    have := ha.1 l
    have := hb.1 l
    simp only [hedgeMid]
    linarith
  · simp only [hedgeMid]
    rw [← Finset.sum_div, Finset.sum_add_distrib, ha.2, hb.2]
    norm_num

theorem condRisk_hedgeMid (C : CostMatrix K) (a b k : Fin K) :
    condRisk C (hedgeMid a b) k = (C a k + C b k) / 2 := by
  rw [← condRisk_vertex C a k, ← condRisk_vertex C b k]
  simp only [condRisk, hedgeMid]
  rw [← Finset.sum_add_distrib, Finset.sum_div]
  exact Finset.sum_congr rfl fun l _ => by ring

theorem admissible_hedgeK (hK : 3 ≤ K) : Admissible (hedgeK K) := by
  intro i
  by_cases hi : (i : ℕ) = 2
  · -- The hedge class wins strictly at the midpoint of `e_0` and `e_1`.
    refine ⟨hedgeMid ⟨0, by omega⟩ ⟨1, by omega⟩, hedgeMid_mem _ _,
      fun j hj => ?_⟩
    have hj2 : (j : ℕ) ≠ 2 := fun h => hj (Fin.ext (h.trans hi.symm))
    rw [condRisk_hedgeMid, condRisk_hedgeMid]
    have e0 : hedgeK K ⟨0, by omega⟩ i = 2 := by simp [hedgeK, hi]
    have e1 : hedgeK K ⟨1, by omega⟩ i = 2 := by simp [hedgeK, hi]
    have hsum : 10 ≤ hedgeK K ⟨0, by omega⟩ j + hedgeK K ⟨1, by omega⟩ j := by
      simp only [hedgeK, Matrix.of_apply, if_neg hj2]
      split_ifs <;> first | omega | contradiction | norm_num
    rw [e0, e1]
    linarith
  · -- Every other class wins strictly at its own vertex.
    refine ⟨vertex i, vertex_mem_stdSimplex i, fun j hj => ?_⟩
    rw [condRisk_vertex, condRisk_vertex, hedgeK_diag hi]
    linarith [two_le_hedgeK_of_ne hi hj]

/-- The hedge class is not a minimizer at any vertex: at each vertex some other
class is strictly cheaper. -/
theorem hedgeK_not_optimal_at_vertex (hK : 3 ≤ K) (j : Fin K) :
    ∃ k, hedgeK K j k < hedgeK K j ⟨2, by omega⟩ := by
  by_cases hj : (j : ℕ) = 2
  · refine ⟨⟨0, by omega⟩, ?_⟩
    simp [hedgeK, hj]
    norm_num
  · refine ⟨j, ?_⟩
    rw [hedgeK_diag hj]
    simp only [hedgeK, Matrix.of_apply, if_neg hj]
    split_ifs <;> norm_num

/-- The hedge class of `hedgeK` is the strict minimizer at the midpoint of the
edge from `e_0` to `e_1`. -/
theorem hedgeK_strict_at_mid (hK : 3 ≤ K) (j : Fin K) (hj : j ≠ ⟨2, by omega⟩) :
    condRisk (hedgeK K) (hedgeMid ⟨0, by omega⟩ ⟨1, by omega⟩) ⟨2, by omega⟩
      < condRisk (hedgeK K) (hedgeMid ⟨0, by omega⟩ ⟨1, by omega⟩) j := by
  have hj2 : (j : ℕ) ≠ 2 := fun h => hj (Fin.ext h)
  rw [condRisk_hedgeMid, condRisk_hedgeMid]
  have e0 : hedgeK K ⟨0, by omega⟩ ⟨2, by omega⟩ = 2 := by simp [hedgeK]
  have e1 : hedgeK K ⟨1, by omega⟩ ⟨2, by omega⟩ = 2 := by simp [hedgeK]
  have hsum : 10 ≤ hedgeK K ⟨0, by omega⟩ j + hedgeK K ⟨1, by omega⟩ j := by
    simp only [hedgeK, Matrix.of_apply, if_neg hj2]
    split_ifs <;> first | omega | contradiction | norm_num
  rw [e0, e1]
  linarith

/-- That midpoint is on the boundary of the simplex: its hedge coordinate is
zero. -/
theorem hedgeMid_boundary (hK : 3 ≤ K) :
    hedgeMid (⟨0, by omega⟩ : Fin K) ⟨1, by omega⟩ ⟨2, by omega⟩ = 0 := by
  simp [hedgeMid, vertex, Fin.ext_iff]

/-- **For every `K ≥ 3` a class can be optimal at no vertex.** There is a
nonnegative admissible matrix with a class that is not a minimizer at any
vertex. The same class is the strict minimizer at a boundary point of the
simplex, so it is not optimal "solely in the interior". -/
theorem exists_admissible_no_vertex (hK : 3 ≤ K) :
    ∃ C : CostMatrix K, (∀ i j, 0 ≤ C i j) ∧ Admissible C
      ∧ ∃ i, (∀ j, ∃ k, C j k < C j i)
        ∧ ∃ p ∈ stdSimplex ℝ (Fin K), p i = 0
          ∧ ∀ j, j ≠ i → condRisk C p i < condRisk C p j :=
  ⟨hedgeK K, hedgeK_nonneg, admissible_hedgeK hK, ⟨2, by omega⟩,
    hedgeK_not_optimal_at_vertex hK, hedgeMid ⟨0, by omega⟩ ⟨1, by omega⟩,
    hedgeMid_mem _ _, hedgeMid_boundary hK, hedgeK_strict_at_mid hK⟩

end Strictness

/-- **Element-wise reasonableness and admissibility**
(`prop:elementwise_admissible`). The implication holds for every real matrix.
Strictness is witnessed by a nonnegative matrix for every `K ≥ 2`. For two
classes admissibility implies element-wise reasonableness after possibly
exchanging the actions, and every action is strictly optimal at a vertex. For
`K ≥ 3` an admissible matrix can have an action optimal at no vertex and
strictly optimal at a boundary point. The general boundary-minimizer fact is
`Multiclass.minimizer_exists_on_boundary`. -/
@[thesis "prop:elementwise_admissible"]
theorem elementwise_admissible :
    (∀ (K : ℕ) (C : CostMatrix K), ElementwiseReasonable C → Admissible C) ∧
    (∀ K : ℕ, 2 ≤ K → ∃ C : CostMatrix K, (∀ i j, 0 ≤ C i j) ∧ Admissible C
      ∧ ¬ ElementwiseReasonable C) ∧
    (∀ C : CostMatrix 2, Admissible C →
      (ElementwiseReasonable C
        ∨ ElementwiseReasonable (C.submatrix id (Equiv.swap (0 : Fin 2) 1)))
      ∧ ∀ i, ∃ j, ∀ k, k ≠ i → C j i < C j k) ∧
    (∀ K : ℕ, 3 ≤ K → ∃ C : CostMatrix K, (∀ i j, 0 ≤ C i j) ∧ Admissible C
      ∧ ∃ i, (∀ j, ∃ k, C j k < C j i)
        ∧ ∃ p ∈ stdSimplex ℝ (Fin K), p i = 0
          ∧ ∀ j, j ≠ i → condRisk C p i < condRisk C p j) :=
  ⟨fun _ _ h => admissible_of_elementwiseReasonable h,
    fun _ hK => exists_admissible_not_elementwise hK,
    fun _ h => ⟨elementwise_or_swapped_of_admissible_two h,
      strict_at_vertex_of_admissible_two h⟩,
    fun _ hK => exists_admissible_no_vertex hK⟩

end ThresholdOpt.Examples
