import ThresholdOpt.Complexity

/-!
# Fixed quadratic metrics for unrestricted prototype rules

A fixed invertible real linear transform `T` induces squared distance
`sum_i (T (p - mu))_i^2`. The unrestricted prototype class for this distance
is exactly the full-matrix class on the probability simplex. All equalities
include the fixed smallest-index rule on ties. No result here asserts that
simplex or DLC restrictions are preserved by a change of metric.
-/

namespace ThresholdOpt.Mahalanobis

open ThresholdOpt ThresholdOpt.Capacity ThresholdOpt.Partition
open ThresholdOpt.Voronoi ThresholdOpt.Complexity

variable {K : ℕ}

abbrev Vec (K : ℕ) := Fin K → ℝ

/-- Squared distance after a fixed invertible linear transformation. -/
def metricDsq (T : Vec K ≃ₗ[ℝ] Vec K) (p q : Vec K) : ℝ := dsq (T p) (T q)

noncomputable def metricRule [NeZero K] (T : Vec K ≃ₗ[ℝ] Vec K)
    (μ : Prototypes K) (p : Vec K) : Fin K :=
  lexArgmin fun k => metricDsq T p (μ k)

def HMetric [NeZero K] (T : Vec K ≃ₗ[ℝ] Vec K) : Set (Simplex K → Fin K) :=
  {δ | ∃ μ : Prototypes K, δ = fun p => metricRule T μ p.val}

/-- Evaluation of a linear map in the coordinate basis. -/
theorem linear_eval (L : Vec K →ₗ[ℝ] Vec K) (p : Vec K) (j : Fin K) :
    L p j = ∑ i, p i * L (vertex i) j := by
  have hp : p = ∑ i, p i • vertex i := by
    funext r
    simp [Finset.sum_apply, vertex, Pi.smul_apply, smul_eq_mul, mul_ite]
  calc L p j = L (∑ i, p i • vertex i) j := by rw [← hp]
    _ = ∑ i, p i * L (vertex i) j := by
      simp [map_sum, map_smul, Finset.sum_apply]

/-- The matrix expressing the original scores in transformed coordinates. -/
noncomputable def pullMatrix (A : CostMatrix K) (T : Vec K ≃ₗ[ℝ] Vec K) : CostMatrix K :=
  Matrix.of fun i k => condRisk A (T.symm (vertex i)) k

/-- Pulling back coefficients preserves their linear score exactly. -/
theorem pullMatrix_risk (A : CostMatrix K) (T : Vec K ≃ₗ[ℝ] Vec K)
    (x : Vec K) (k : Fin K) :
    condRisk (pullMatrix A T) x k = condRisk A (T.symm x) k := by
  simp only [pullMatrix, Matrix.of_apply, condRisk, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro r _
  have he : T.symm x r = ∑ i, x i * T.symm (vertex i) r :=
    linear_eval T.symm.toLinearMap x r
  rw [he, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- The transformed simplex lies in `u · x = 1`, where `u` is this vector. -/
noncomputable def normal (T : Vec K ≃ₗ[ℝ] Vec K) : Vec K :=
  fun i => ∑ r, T.symm (vertex i) r

theorem normal_pairing (T : Vec K ≃ₗ[ℝ] Vec K) (x : Vec K) :
    (∑ i, normal T i * x i) = ∑ r, T.symm x r := by
  simp only [normal, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro r _
  have he : T.symm x r = ∑ i, x i * T.symm (vertex i) r :=
    linear_eval T.symm.toLinearMap x r
  rw [he]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem normal_pairing_simplex (T : Vec K ≃ₗ[ℝ] Vec K)
    {p : Vec K} (hp : p ∈ stdSimplex ℝ (Fin K)) :
    (∑ i, normal T i * T p i) = 1 := by
  rw [normal_pairing, T.symm_apply_apply]
  exact hp.2

theorem normal_sq_pos [NeZero K] (T : Vec K ≃ₗ[ℝ] Vec K) :
    0 < ∑ i, normal T i ^ 2 := by
  have hn : ∃ i, normal T i ≠ 0 := by
    by_contra hall
    push Not at hall
    have h := normal_pairing_simplex T (vertex_mem_stdSimplex (0 : Fin K))
    simp [hall] at h
  obtain ⟨i, hi⟩ := hn
  exact lt_of_lt_of_le (sq_pos_of_ne_zero hi)
    (Finset.single_le_sum (fun j _ => sq_nonneg (normal T j)) (Finset.mem_univ i))

/-- Completing the square on an arbitrary nonzero affine hyperplane. -/
noncomputable def gauge (B : CostMatrix K) (u : Vec K) (β : Fin K → ℝ) (k : Fin K) : ℝ :=
  (∑ i, u i ^ 2) * β k ^ 2 - (2 + ∑ i, B i k * u i) * β k
    + (1 / 4) * ∑ i, B i k ^ 2

noncomputable def shifted (B : CostMatrix K) (u : Vec K) (β : Fin K → ℝ) : Prototypes K :=
  fun k i => -(1 / 2) * B i k + β k * u i

theorem dsq_shifted {B : CostMatrix K} {u x : Vec K} {β : Fin K → ℝ}
    (hx : (∑ i, u i * x i) = 1) (k : Fin K) :
    dsq x (shifted B u β k) = (∑ i, x i ^ 2) + condRisk B x k + gauge B u β k := by
  have hexpand : ∀ i : Fin K,
      (x i - (-(1 / 2) * B i k + β k * u i)) ^ 2 =
      x i ^ 2 + B i k * x i - (2 * β k) * (u i * x i)
        + (1 / 4) * B i k ^ 2 - β k * (B i k * u i) + β k ^ 2 * u i ^ 2 := by
    intro i
    ring
  simp only [dsq, shifted, condRisk, gauge]
  rw [Finset.sum_congr rfl fun i _ => hexpand i]
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib]
  rw [← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, hx]
  ring

theorem exists_gauge_solution [NeZero K] (B : CostMatrix K) (u : Vec K)
    (hu : 0 < ∑ i, u i ^ 2) :
    ∃ (β : Fin K → ℝ) (c : ℝ), ∀ k, gauge B u β k = c := by
  obtain ⟨c, hc⟩ := Finite.exists_le (fun k : Fin K => (1 / 4 : ℝ) * ∑ i, B i k ^ 2)
  have hroot : ∀ k : Fin K, ∃ x : ℝ,
      (∑ i, u i ^ 2) * x ^ 2 - (2 + ∑ i, B i k * u i) * x
        + ((1 / 4) * ∑ i, B i k ^ 2 - c) = 0 := by
    intro k
    apply Voronoi.exists_quadratic_root hu
    have hn : 0 ≤ (∑ i, u i ^ 2) * (c - (1 / 4) * ∑ i, B i k ^ 2) :=
      mul_nonneg hu.le (sub_nonneg.2 (hc k))
    nlinarith [sq_nonneg (2 + ∑ i, B i k * u i)]
  choose β hβ using hroot
  exact ⟨β, c, fun k => by simpa only [gauge] using (eq_of_sub_eq_zero (by linarith [hβ k]))⟩

/-- Every linear score matrix is realized by unrestricted transformed prototypes. -/
theorem full_in_metric [NeZero K] (T : Vec K ≃ₗ[ℝ] Vec K) (A : CostMatrix K) :
    ∃ μ : Prototypes K, ∀ p ∈ stdSimplex ℝ (Fin K),
      metricRule T μ p = lexArgmin (condRisk A p) := by
  let B := pullMatrix A T
  let u := normal T
  obtain ⟨β, c, hβ⟩ := exists_gauge_solution B u (normal_sq_pos T)
  let μ : Prototypes K := fun k => T.symm (shifted B u β k)
  refine ⟨μ, fun p hp => ?_⟩
  have hscore : ∀ k,
      metricDsq T p (μ k) = condRisk A p k + ((∑ i, T p i ^ 2) + c) := by
    intro k
    simp only [metricDsq, μ, T.apply_symm_apply]
    rw [dsq_shifted (normal_pairing_simplex T hp), hβ k, pullMatrix_risk,
      T.symm_apply_apply]
    ring
  simp only [metricRule]
  rw [show (fun k => metricDsq T p (μ k)) =
    (fun k => condRisk A p k + ((∑ i, T p i ^ 2) + c)) from funext hscore]
  exact lexArgmin_add_const _ _

/-- Unrestricted transformed prototypes induce affine scores on the original input. -/
theorem metric_subset_full [NeZero K] (T : Vec K ≃ₗ[ℝ] Vec K) :
    HMetric T ⊆ HFull := by
  rintro δ ⟨μ, rfl⟩
  let C : CostMatrix K := Matrix.of fun i k => -2 * T (μ k) i
  let D := pullMatrix C T.symm
  let v : Fin K → ℝ := fun k => ∑ i, T (μ k) i ^ 2
  apply HPS_subset_HFull D
  refine ⟨v, funext fun p => ?_⟩
  have hscore : ∀ k, metricDsq T p.val (μ k) =
      psScore D v p.val k + ∑ i, T p.val i ^ 2 := by
    intro k
    have hd := pullMatrix_risk C T.symm p.val k
    simp only [LinearEquiv.symm_symm] at hd
    rw [psScore, hd]
    simp only [metricDsq, dsq, condRisk, C, Matrix.of_apply, v]
    rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  simp only [metricRule, psRule]
  rw [show (fun k => metricDsq T p.val (μ k)) =
      (fun k => psScore D v p.val k + ∑ i, T p.val i ^ 2) from funext hscore]
  exact lexArgmin_add_const _ _

/-- Fixed invertible linear changes of squared Euclidean distance preserve
exactly the unrestricted full-matrix decision class on the simplex. -/
theorem HMetric_eq_HFull [NeZero K] (T : Vec K ≃ₗ[ℝ] Vec K) :
    HMetric T = HFull := by
  apply Set.Subset.antisymm (metric_subset_full T)
  rintro δ ⟨A, rfl⟩
  obtain ⟨μ, hμ⟩ := full_in_metric T A
  exact ⟨μ, funext fun p => (hμ p.val p.property).symm⟩

end ThresholdOpt.Mahalanobis
namespace ThresholdOpt.Mahalanobis

open ThresholdOpt ThresholdOpt.Voronoi ThresholdOpt.Complexity

/-- The discrete metric is a valid metric on the same vector space. -/
noncomputable def discreteDist {K : ℕ} (p q : Vec K) : ℝ :=
  if p = q then 0 else 1

theorem discreteDist_nonneg {K : ℕ} (p q : Vec K) : 0 ≤ discreteDist p q := by
  classical
  unfold discreteDist
  split_ifs <;> norm_num

theorem discreteDist_eq_zero_iff {K : ℕ} (p q : Vec K) :
    discreteDist p q = 0 ↔ p = q := by
  classical
  by_cases h : p = q <;> simp [discreteDist, h]

theorem discreteDist_symm {K : ℕ} (p q : Vec K) :
    discreteDist p q = discreteDist q p := by
  classical
  simp [discreteDist, eq_comm]

theorem discreteDist_triangle {K : ℕ} (p q r : Vec K) :
    discreteDist p r ≤ discreteDist p q + discreteDist q r := by
  classical
  by_cases hpq : p = q
  · subst q
    simp [discreteDist]
  · by_cases hqr : q = r
    · subst r
      simp [discreteDist]
    · simp only [discreteDist, if_neg hpq, if_neg hqr]
      split_ifs <;> norm_num

noncomputable def discreteRule {K : ℕ} [NeZero K] (μ : Prototypes K) (p : Vec K) :
    Fin K := lexArgmin fun k => discreteDist p (μ k)

/-- The midpoint is interior to the binary simplex. -/
noncomputable def midpoint : Simplex 2 := ⟨fun _ => 1 / 2, by
  constructor
  · intro i
    norm_num
  · simp⟩

noncomputable def discretePrototypes : Prototypes 2 := ![vertex 0, midpoint.val]

theorem midpoint_ne_vertex (i : Fin 2) : midpoint.val ≠ vertex i := by
  intro h
  have he := congrFun h i
  norm_num [midpoint] at he

open scoped Classical in
/-- With these prototypes the discrete rule chooses class one exactly at
an interior point, and the least-index tie rule chooses zero everywhere else. -/
theorem discreteRule_is_spike (p : Simplex 2) :
    discreteRule discretePrototypes p.val = if p = midpoint then 1 else 0 := by
  classical
  by_cases hp : p = midpoint
  · subst p
    rw [if_pos rfl]
    apply lexArgmin_eq_of_strict
    intro j hj
    fin_cases j
    · norm_num [discretePrototypes, discreteDist, midpoint_ne_vertex]
    · exact (hj rfl).elim
  · rw [if_neg hp]
    have hpv : p.val ≠ midpoint.val := fun h => hp (Subtype.ext h)
    apply le_antisymm
    · apply lexArgmin_le_index
      intro j
      fin_cases j
      · exact le_rfl
      · change discreteDist p.val (vertex 0) ≤ discreteDist p.val midpoint.val
        simp only [discreteDist, if_neg hpv]
        split_ifs <;> norm_num
    · exact Fin.zero_le _

/-- A binary linear-score rule cannot isolate a simplex-interior point.
Consequently an arbitrary replacement of the metric need not preserve HFull. -/
theorem discreteRule_not_mem_HFull :
    (fun p : Simplex 2 => discreteRule discretePrototypes p.val) ∉ HFull := by
  classical
  rintro ⟨A, hA⟩
  have hm : lexArgmin (condRisk A midpoint.val) = 1 := by
    have he := (congrFun hA midpoint).symm
    simpa [discreteRule_is_spike] using he
  have hv : ∀ j : Fin 2, lexArgmin (condRisk A (vertex j)) = 0 := by
    intro j
    have hj : vtx j ≠ midpoint := by
      intro h
      exact midpoint_ne_vertex j (congrArg Subtype.val h).symm
    have he := (congrFun hA (vtx j)).symm
    have hval := discreteRule_is_spike (vtx j)
    rw [if_neg hj] at hval
    exact he.trans hval
  have h0 := lexArgmin_le (condRisk A (vertex (0 : Fin 2))) 1
  have h1 := lexArgmin_le (condRisk A (vertex (1 : Fin 2))) 1
  rw [hv, condRisk_vertex, condRisk_vertex] at h0 h1
  have hmin : ∀ j : Fin 2, condRisk A midpoint.val 0 ≤ condRisk A midpoint.val j := by
    intro j
    fin_cases j
    · exact le_rfl
    · simp only [condRisk, midpoint, Fin.sum_univ_two]
      change A 0 0 * (1 / 2) + A 1 0 * (1 / 2) ≤
        A 0 1 * (1 / 2) + A 1 1 * (1 / 2)
      linarith
  have hleast := lexArgmin_le_index hmin
  rw [hm] at hleast
  norm_num at hleast


open scoped Classical in
/-- The discrete metric is valid, its two-prototype rule selects one exactly
at the midpoint, and that rule is not a full-matrix rule. -/
theorem discrete_metric_counterexample :
    (∀ p q : Vec 2, 0 ≤ discreteDist p q) ∧
    (∀ p q : Vec 2, discreteDist p q = 0 ↔ p = q) ∧
    (∀ p q : Vec 2, discreteDist p q = discreteDist q p) ∧
    (∀ p q r : Vec 2, discreteDist p r ≤ discreteDist p q + discreteDist q r) ∧
    (∀ p : Simplex 2,
      discreteRule discretePrototypes p.val = if p = midpoint then 1 else 0) ∧
    (fun p : Simplex 2 => discreteRule discretePrototypes p.val) ∉ HFull := by
  classical
  exact ⟨discreteDist_nonneg, discreteDist_eq_zero_iff, discreteDist_symm,
    discreteDist_triangle, discreteRule_is_spike, discreteRule_not_mem_HFull⟩

end ThresholdOpt.Mahalanobis