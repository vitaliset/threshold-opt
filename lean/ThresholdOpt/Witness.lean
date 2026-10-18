import ThresholdOpt.Voronoi

/-!
# The vertical-stack witness (Chapter 6)

`thm:rotational_dominance`: the Voronoi class attains strictly lower population
risk than the parallel-shift class on an explicit `K = 3` configuration under
`0/1` loss. This is the witness construction of Appendix A, checked.

The distribution puts mass `1/3` on each of three points of the simplex, with
the estimator the identity, so the risk of a rule is one third the number of
the three points it misclassifies.

* No parallel-shift rule labels both `pA` as class 1 and `pB` as class 2. All
  three points lie on the altitude `p₁ = p₂`, where the class-1-versus-class-2
  parallel-shift score difference collapses to `v₁ - v₂` and so cannot depend
  on the point. Hence every parallel-shift rule errs somewhere and has risk at
  least `1/3`.
* The stated prototypes label all three points correctly, so the Voronoi rule
  attains risk `0`, and they satisfy the Dominant Label Constraint, so the
  witness lies in the subclass `H_Vor^DLC` that the dissertation actually fits.

Coordinates are written as fractions rather than decimals so that `norm_num`
closes the arithmetic exactly.
-/

namespace ThresholdOpt.Witness

open ThresholdOpt ThresholdOpt.Partition ThresholdOpt.Voronoi

/-- Case analysis on `Fin 3` producing numerals. -/
theorem fin3_cases (j : Fin 3) : j = 0 ∨ j = 1 ∨ j = 2 := by
  revert j
  decide

/-! ### The three points -/

/-- `p_C = (0.8, 0.1, 0.1)`, true label 0. -/
noncomputable def pC : Fin 3 → ℝ := ![4 / 5, 1 / 10, 1 / 10]
/-- `p_A = (0.4, 0.3, 0.3)`, true label 1. -/
noncomputable def pA : Fin 3 → ℝ := ![2 / 5, 3 / 10, 3 / 10]
/-- `p_B = (0, 0.5, 0.5)`, true label 2. -/
noncomputable def pB : Fin 3 → ℝ := ![0, 1 / 2, 1 / 2]

@[simp] theorem pC_0 : pC 0 = 4 / 5 := rfl
@[simp] theorem pC_1 : pC 1 = 1 / 10 := rfl
@[simp] theorem pC_2 : pC 2 = 1 / 10 := rfl
@[simp] theorem pA_0 : pA 0 = 2 / 5 := rfl
@[simp] theorem pA_1 : pA 1 = 3 / 10 := rfl
@[simp] theorem pA_2 : pA 2 = 3 / 10 := rfl
@[simp] theorem pB_0 : pB 0 = 0 := rfl
@[simp] theorem pB_1 : pB 1 = 1 / 2 := rfl
@[simp] theorem pB_2 : pB 2 = 1 / 2 := rfl

theorem pC_mem : pC ∈ stdSimplex ℝ (Fin 3) := by
  constructor
  · intro i; rcases fin3_cases i with rfl | rfl | rfl <;> norm_num
  · rw [Fin.sum_univ_three]; norm_num

theorem pA_mem : pA ∈ stdSimplex ℝ (Fin 3) := by
  constructor
  · intro i; rcases fin3_cases i with rfl | rfl | rfl <;> norm_num
  · rw [Fin.sum_univ_three]; norm_num

theorem pB_mem : pB ∈ stdSimplex ℝ (Fin 3) := by
  constructor
  · intro i; rcases fin3_cases i with rfl | rfl | rfl <;> norm_num
  · rw [Fin.sum_univ_three]; norm_num

/-! ### The parallel-shift class cannot separate the stack -/

/-- **No parallel-shift rule labels `pA` as 1 and `pB` as 2.**

Both points sit on the altitude `p₁ = p₂`, where the `0/1` parallel-shift
scores of classes 1 and 2 differ by `v₁ - v₂` regardless of the point.
Labelling `pA` as 1 forces `v₁ ≤ v₂`, labelling `pB` as 2 forces `v₂ ≤ v₁`,
and then the two scores tie at `pB`, where the smallest-index tie-break returns
1 rather than 2. -/
theorem ps_cannot_separate (v : Fin 3 → ℝ) :
    ¬ (psRule (zeroOne 3) v pA = 1 ∧ psRule (zeroOne 3) v pB = 2) := by
  rintro ⟨hA, hB⟩
  have hsA : ∀ k, psScore (zeroOne 3) v pA k = 1 - pA k + v k := fun k => by
    rw [psScore, condRisk_zeroOne pA_mem k]
  have hsB : ∀ k, psScore (zeroOne 3) v pB k = 1 - pB k + v k := fun k => by
    rw [psScore, condRisk_zeroOne pB_mem k]
  have hA' : lexArgmin (psScore (zeroOne 3) v pA) = 1 := hA
  have hB' : lexArgmin (psScore (zeroOne 3) v pB) = 2 := hB
  have hminA : ∀ j, psScore (zeroOne 3) v pA 1 ≤ psScore (zeroOne 3) v pA j := by
    intro j
    have h := lexArgmin_le (psScore (zeroOne 3) v pA) j
    rwa [hA'] at h
  have hminB : ∀ j, psScore (zeroOne 3) v pB 2 ≤ psScore (zeroOne 3) v pB j := by
    intro j
    have h := lexArgmin_le (psScore (zeroOne 3) v pB) j
    rwa [hB'] at h
  -- pA is labelled 1, so score 1 is minimal there; on the altitude the
  -- p-dependent parts of scores 1 and 2 agree, leaving v 1 ≤ v 2.
  have h12 : v 1 ≤ v 2 := by
    have h := hminA 2
    rw [hsA 1, hsA 2] at h
    simp only [pA_1, pA_2] at h
    linarith
  -- pB is labelled 2, so score 2 is minimal there, giving v 2 ≤ v 1.
  have h21 : v 2 ≤ v 1 := by
    have h := hminB 1
    rw [hsB 2, hsB 1] at h
    simp only [pB_1, pB_2] at h
    linarith
  have hv : v 1 = v 2 := le_antisymm h12 h21
  -- Score 2 is minimal at pB, and score 1 equals it, so score 1 is minimal
  -- too. The least minimizing index is therefore at most 1.
  have hmin1 : ∀ j, psScore (zeroOne 3) v pB 1 ≤ psScore (zeroOne 3) v pB j := by
    intro j
    have h12eq : psScore (zeroOne 3) v pB 1 = psScore (zeroOne 3) v pB 2 := by
      rw [hsB 1, hsB 2, hv]
      simp
    rw [h12eq]
    exact hminB j
  have hle : lexArgmin (psScore (zeroOne 3) v pB) ≤ 1 := lexArgmin_le_index hmin1
  rw [show lexArgmin (psScore (zeroOne 3) v pB) = psRule (zeroOne 3) v pB from
    rfl, hB] at hle
  exact absurd hle (by decide)

/-! ### The Voronoi witness -/

/-- `μ₀ = e₀`. -/
noncomputable def m0 : Fin 3 → ℝ := ![1, 0, 0]
/-- `μ₁ = (0.4, 0.5, 0.1)`. -/
noncomputable def m1 : Fin 3 → ℝ := ![2 / 5, 1 / 2, 1 / 10]
/-- `μ₂ = (0.1, 0.4, 0.5)`. -/
noncomputable def m2 : Fin 3 → ℝ := ![1 / 10, 2 / 5, 1 / 2]

/-- The prototype configuration of the witness. -/
noncomputable def mu : Prototypes 3 := ![m0, m1, m2]

@[simp] theorem mu_0 : mu 0 = m0 := rfl
@[simp] theorem mu_1 : mu 1 = m1 := rfl
@[simp] theorem mu_2 : mu 2 = m2 := rfl

@[simp] theorem m0_0 : m0 0 = 1 := rfl
@[simp] theorem m0_1 : m0 1 = 0 := rfl
@[simp] theorem m0_2 : m0 2 = 0 := rfl
@[simp] theorem m1_0 : m1 0 = 2 / 5 := rfl
@[simp] theorem m1_1 : m1 1 = 1 / 2 := rfl
@[simp] theorem m1_2 : m1 2 = 1 / 10 := rfl
@[simp] theorem m2_0 : m2 0 = 1 / 10 := rfl
@[simp] theorem m2_1 : m2 1 = 2 / 5 := rfl
@[simp] theorem m2_2 : m2 2 = 1 / 2 := rfl

theorem dsq_three (p q : Fin 3 → ℝ) :
    dsq p q = (p 0 - q 0) ^ 2 + (p 1 - q 1) ^ 2 + (p 2 - q 2) ^ 2 := by
  simp [dsq, Fin.sum_univ_three]

/-- Vertex coordinates at literal `Fin 3` indices. Stated explicitly because
the generic `vertex` lemmas leave an `if i = k` that the numeric tactics do
not discharge on their own here. -/
@[simp] theorem e0_0 : vertex (0 : Fin 3) 0 = 1 := vertex_apply_self 0
@[simp] theorem e0_1 : vertex (0 : Fin 3) 1 = 0 := by
  rw [vertex_apply_of_ne (by decide)]
@[simp] theorem e0_2 : vertex (0 : Fin 3) 2 = 0 := by
  rw [vertex_apply_of_ne (by decide)]
@[simp] theorem e1_0 : vertex (1 : Fin 3) 0 = 0 := by
  rw [vertex_apply_of_ne (by decide)]
@[simp] theorem e1_1 : vertex (1 : Fin 3) 1 = 1 := vertex_apply_self 1
@[simp] theorem e1_2 : vertex (1 : Fin 3) 2 = 0 := by
  rw [vertex_apply_of_ne (by decide)]
@[simp] theorem e2_0 : vertex (2 : Fin 3) 0 = 0 := by
  rw [vertex_apply_of_ne (by decide)]
@[simp] theorem e2_1 : vertex (2 : Fin 3) 1 = 0 := by
  rw [vertex_apply_of_ne (by decide)]
@[simp] theorem e2_2 : vertex (2 : Fin 3) 2 = 1 := vertex_apply_self 2

/-- The witness satisfies the Dominant Label Constraint, so it lies in the
subclass `H_Vor^DLC` that Chapter 6 actually fits. -/
theorem mu_DLC : DLC mu := by
  intro k j hj
  rcases fin3_cases k with rfl | rfl | rfl <;>
    rcases fin3_cases j with rfl | rfl | rfl <;>
      first
        | exact absurd rfl hj
        | (rw [dsq_three, dsq_three]
           simp only [mu_0, mu_1, mu_2, m0_0, m0_1, m0_2, m1_0, m1_1,
             m1_2, m2_0, m2_1, m2_2, e0_0, e0_1, e0_2, e1_0, e1_1, e1_2,
             e2_0, e2_1, e2_2]
           norm_num)

/-- The Voronoi rule labels all three points correctly. -/
theorem voronoi_correct :
    voronoiRule mu pC = 0 ∧ voronoiRule mu pA = 1 ∧ voronoiRule mu pB = 2 := by
  refine ⟨?_, ?_, ?_⟩ <;>
    (refine lexArgmin_eq_of_strict ?_
     intro j hj
     rcases fin3_cases j with rfl | rfl | rfl <;>
       first
         | exact absurd rfl hj
         | (rw [dsq_three, dsq_three]
            simp only [mu_0, mu_1, mu_2, m0_0, m0_1, m0_2, m1_0, m1_1, m1_2,
              m2_0, m2_1, m2_2, pC_0, pC_1, pC_2, pA_0, pA_1, pA_2, pB_0,
              pB_1, pB_2]
            norm_num))

/-! ### The population risk on the vertical stack

The distribution of the witness: mass `1/3` on each of the three points, with
the estimator the identity, so the risk of a rule under `0/1` loss is one third
of the number of points it mislabels. -/

/-- The three points of the configuration. -/
noncomputable def points : Fin 3 → (Fin 3 → ℝ) := ![pC, pA, pB]

/-- Their true labels. -/
def labels : Fin 3 → Fin 3 := ![0, 1, 2]

@[simp] theorem points_0 : points 0 = pC := rfl
@[simp] theorem points_1 : points 1 = pA := rfl
@[simp] theorem points_2 : points 2 = pB := rfl
@[simp] theorem labels_0 : labels 0 = 0 := rfl
@[simp] theorem labels_1 : labels 1 = 1 := rfl
@[simp] theorem labels_2 : labels 2 = 2 := rfl

/-- The population risk `R(δ)` of a decision rule on this configuration. -/
noncomputable def risk (δ : (Fin 3 → ℝ) → Fin 3) : ℝ :=
  (∑ i, if δ (points i) = labels i then (0 : ℝ) else 1) / 3

theorem risk_nonneg (δ : (Fin 3 → ℝ) → Fin 3) : 0 ≤ risk δ := by
  refine div_nonneg (Finset.sum_nonneg fun i _ => ?_) (by norm_num)
  split <;> norm_num

/-- The Voronoi witness attains zero risk: it labels all three points
correctly. -/
theorem risk_voronoi_mu : risk (voronoiRule mu) = 0 := by
  simp only [risk, Fin.sum_univ_three, points_0, points_1, points_2,
    labels_0, labels_1, labels_2, voronoi_correct.1, voronoi_correct.2.1,
    voronoi_correct.2.2]
  norm_num

/-- Every parallel-shift rule has risk at least `1/3`: it cannot label both
`pA` and `pB` correctly, so it mislabels at least one of the three points. -/
theorem one_third_le_risk_ps (v : Fin 3 → ℝ) :
    1 / 3 ≤ risk (psRule (zeroOne 3) v) := by
  -- Restated at `points`/`labels` so that every term below is syntactically
  -- the one the definition of `risk` produces.
  have hsep : ¬ (psRule (zeroOne 3) v (points 1) = labels 1
      ∧ psRule (zeroOne 3) v (points 2) = labels 2) := ps_cannot_separate v
  unfold risk
  rw [Fin.sum_univ_three]
  set a := if psRule (zeroOne 3) v (points 0) = labels 0 then (0 : ℝ) else 1
    with ha
  set b := if psRule (zeroOne 3) v (points 1) = labels 1 then (0 : ℝ) else 1
    with hb
  set c := if psRule (zeroOne 3) v (points 2) = labels 2 then (0 : ℝ) else 1
    with hc
  have ha0 : 0 ≤ a := by rw [ha]; split <;> norm_num
  have hbc : 1 ≤ b + c := by
    by_cases hA : psRule (zeroOne 3) v (points 1) = labels 1
    · rw [hb, hc, if_pos hA, if_neg (fun hB => hsep ⟨hA, hB⟩)]
      norm_num
    · have hc0 : 0 ≤ c := by rw [hc]; split <;> norm_num
      rw [hb, if_neg hA]
      linarith
  linarith

/-- The witness prototypes are probability vectors, as `def:dlc` requires. -/
theorem mu_mem (k : Fin 3) : mu k ∈ stdSimplex ℝ (Fin 3) := by
  rcases fin3_cases k with rfl | rfl | rfl
  · rw [mu_0]
    constructor
    · intro i; rcases fin3_cases i with rfl | rfl | rfl <;> norm_num
    · rw [Fin.sum_univ_three]; norm_num
  · rw [mu_1]
    constructor
    · intro i; rcases fin3_cases i with rfl | rfl | rfl <;> norm_num
    · rw [Fin.sum_univ_three]; norm_num
  · rw [mu_2]
    constructor
    · intro i; rcases fin3_cases i with rfl | rfl | rfl <;> norm_num
    · rw [Fin.sum_univ_three]; norm_num

/-! ### The two attainable risk sets -/

/-- Risks attainable on this configuration by the rules of `H_Vor^DLC`:
Voronoi rules whose prototypes lie in the simplex and satisfy the Dominant
Label Constraint (`def:dlc`). -/
def vorRisks : Set ℝ :=
  {r | ∃ μ : Prototypes 3, (∀ k, μ k ∈ stdSimplex ℝ (Fin 3)) ∧ DLC μ
    ∧ r = risk (voronoiRule μ)}

/-- Risks attainable on this configuration by parallel-shift rules. -/
def psRisks : Set ℝ := {r | ∃ v : Fin 3 → ℝ, r = risk (psRule (zeroOne 3) v)}

theorem zero_mem_vorRisks : (0 : ℝ) ∈ vorRisks :=
  ⟨mu, mu_mem, mu_DLC, risk_voronoi_mu.symm⟩

theorem bddBelow_vorRisks : BddBelow vorRisks := by
  refine ⟨0, ?_⟩
  rintro r ⟨μ, -, -, rfl⟩
  exact risk_nonneg _

theorem psRisks_nonempty : psRisks.Nonempty := ⟨_, ⟨0, rfl⟩⟩

theorem one_third_le_sInf_psRisks : 1 / 3 ≤ sInf psRisks := by
  refine le_csInf psRisks_nonempty ?_
  rintro r ⟨v, rfl⟩
  exact one_third_le_risk_ps v

/-- **Strict risk separation** (`thm:rotational_dominance`).

There is a distribution and a cost matrix, here the three-point vertical stack
under `0/1` loss, on which

`inf_{δ ∈ H_Vor^DLC} R(δ) < inf_{δ ∈ H_PS} R(δ)`.

The left side is `0`, attained by the DLC-compliant simplex prototypes `mu`;
the right side is at least `1/3`, because no parallel-shift rule labels both
`pA` and `pB` correctly. -/
@[thesis "thm:rotational_dominance"]
theorem rotational_dominance : sInf vorRisks < sInf psRisks := by
  have hvor : sInf vorRisks ≤ 0 :=
    csInf_le bddBelow_vorRisks zero_mem_vorRisks
  have hps : (1 : ℝ) / 3 ≤ sInf psRisks := one_third_le_sInf_psRisks
  linarith

/-- The ingredients of the separation, stated separately because Appendix A
verifies them one at a time: the parallel-shift obstruction, the dominant
label constraint for the witness, and the three correct labels. -/
theorem rotational_dominance_witness :
    (∀ v : Fin 3 → ℝ,
      ¬ (psRule (zeroOne 3) v pA = 1 ∧ psRule (zeroOne 3) v pB = 2)) ∧
    DLC mu ∧ voronoiRule mu pC = 0 ∧ voronoiRule mu pA = 1
      ∧ voronoiRule mu pB = 2 :=
  ⟨ps_cannot_separate, mu_DLC, voronoi_correct.1, voronoi_correct.2.1,
    voronoi_correct.2.2⟩

/-- The witness configuration is a genuine element of the Voronoi class
satisfying the DLC, which is what makes the separation above a statement about
`H_Vor^DLC` rather than about one handpicked function. -/
theorem witness_mem_HVor :
    ∃ μ : Prototypes 3, DLC μ ∧ (fun p => voronoiRule μ p) = voronoiRule mu :=
  ⟨mu, mu_DLC, rfl⟩

end ThresholdOpt.Witness
