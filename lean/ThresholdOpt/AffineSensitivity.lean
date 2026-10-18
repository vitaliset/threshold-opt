import ThresholdOpt.Diffusion
import ThresholdOpt.Risk

/-!
# Local smoothness of the actual cost functional
The centred cost column is a vector in the Euclidean tangent space. For a
locally stable action, the relative derivative is its inner product and the
Taylor remainder is exactly zero. This connects the cost functional in
Chapter 8 to the full LocalSmoothness definition.
-/

namespace ThresholdOpt.Diffusion
open ThresholdOpt
open scoped BigOperators RealInnerProductSpace Classical
variable {K : ℕ}

/-- The centred cost gradient really preserves total mass. -/
theorem gradient_sum_zero [NeZero K] (C : CostMatrix K) (k : Fin K) :
    ∑ i, Sensitivity.gradient C k i = 0 := by
  have hK : (K : ℝ) ≠ 0 := by exact_mod_cast (NeZero.ne K)
  simp only [Sensitivity.gradient, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Sensitivity.colMean]
  field_simp
  ring

noncomputable def centeredGradient [NeZero K] (C : CostMatrix K) (k : Fin K) :
    Tangent K :=
  ⟨WithLp.toLp 2 (Sensitivity.gradient C k), by
    change ∑ i, Sensitivity.gradient C k i = 0
    exact gradient_sum_zero C k⟩

theorem centeredGradient_inner [NeZero K] (C : CostMatrix K) (k : Fin K)
    (v : Tangent K) :
    ⟪centeredGradient C k, v⟫ = ∑ i, Sensitivity.gradient C k i * v.val i := by
  change ⟪WithLp.toLp 2 (Sensitivity.gradient C k), v.val⟫ =
    ∑ i, Sensitivity.gradient C k i * v.val i
  simp [PiLp.inner_apply, mul_comm]

/-- Conditional cost of a fixed action, on an ambient extension. -/
noncomputable def actionCost (C : CostMatrix K) (k : Fin K)
    (p : EuclideanSpace ℝ (Fin K)) : ℝ := condRisk C p.ofLp k

theorem actionCost_add_tangent [NeZero K] (C : CostMatrix K) (k : Fin K)
    (p : EuclideanSpace ℝ (Fin K)) (v : Tangent K) :
    actionCost C k (p + v.val) = actionCost C k p + ⟪centeredGradient C k, v⟫ := by
  rw [centeredGradient_inner]
  have h := Sensitivity.condRisk_add_tangent C k p.ofLp
    (ε := v.val.ofLp) (show v.val.ofLp ∈ Sensitivity.tangent K from v.property)
  exact h

theorem actionCost_hasFDerivAt [NeZero K] (C : CostMatrix K) (k : Fin K)
    (p : EuclideanSpace ℝ (Fin K)) :
    HasFDerivAt (fun v : Tangent K => actionCost C k (p + v.val))
      (innerSL ℝ (centeredGradient C k)) 0 := by
  have heq : (fun v : Tangent K => actionCost C k (p + v.val)) =
      (fun v : Tangent K => actionCost C k p + (innerSL ℝ (centeredGradient C k)) v) := by
    funext v
    exact actionCost_add_tangent C k p v
  rw [heq]
  exact ((innerSL ℝ (centeredGradient C k)).hasFDerivAt).const_add _

/-- The actual per-sample rule cost phi_delta, with an ambient extension. -/
noncomputable def ruleCost (C : CostMatrix K)
    (δ : EuclideanSpace ℝ (Fin K) → Fin K) (p : EuclideanSpace ℝ (Fin K)) : ℝ :=
  actionCost C (δ p) p

theorem measurable_ruleCost (C : CostMatrix K)
    {δ : EuclideanSpace ℝ (Fin K) → Fin K} (hδ : Measurable δ) :
    Measurable (ruleCost C δ) := by
  exact Risk.measurable_condRisk_action C
    (MeasurableEquiv.toLp 2 (Fin K → ℝ)).symm.measurable hδ

/-- Local action stability is the precise boundary-exclusion premise.
The ball-validity premise only states that the chosen chart stays in the
simplex. Under these premises the cost rule satisfies every smoothness field,
with the centred gradient and any positive M, because the remainder vanishes. -/
theorem localSmoothness_ruleCost [NeZero K] (C : CostMatrix K)
    (δ : EuclideanSpace ℝ (Fin K) → Fin K) (p : EuclideanSpace ℝ (Fin K))
    (hp : ∀ i, 0 < p i) (hmass : ∑ i, p i = 1) {M ρ : ℝ}
    (hM : 0 < M) (hρ : 0 < ρ)
    (hball : ∀ v : Tangent K, ‖v‖ ≤ ρ → ∀ i, 0 ≤ (p + v.val) i)
    (hstable : ∀ v : Tangent K, ‖v‖ ≤ ρ → δ (p + v.val) = δ p) :
    LocalSmoothness (ruleCost C δ) p (centeredGradient C (δ p)) M ρ := by
  refine ⟨hp, hmass, hM, hρ, hball, ?_, ?_⟩
  · apply (actionCost_hasFDerivAt C (δ p) p).congr_of_eventuallyEq
    filter_upwards [Metric.ball_mem_nhds (0 : Tangent K) hρ] with v hv
    have hnorm : ‖v‖ ≤ ρ := by
      have : ‖v‖ < ρ := by simpa using hv
      exact this.le
    simp only [ruleCost, hstable v hnorm]
  · intro v hv
    simp only [ruleCost, hstable v hv, actionCost_add_tangent]
    simp only [add_sub_cancel_left, sub_self, abs_zero]
    exact mul_nonneg hM.le (sq_nonneg _)


/-- Every relative-interior simplex point admits a positive closed tangent
ball that stays in the simplex. This removes an extra domain-ball premise
from applications of local smoothness. -/
theorem exists_simplex_tangent_ball (p : EuclideanSpace ℝ (Fin K))
    (hp : ∀ i, 0 < p i) (hmass : ∑ i, p i = 1) :
    ∃ ρ : ℝ, 0 < ρ ∧ ∀ v : Tangent K, ‖v‖ ≤ ρ →
      (p + v.val).ofLp ∈ stdSimplex ℝ (Fin K) := by
  let U : Set (Tangent K) := {v | ∀ i, 0 < (p + v.val) i}
  have hU : IsOpen U := by
    simp only [U, Set.ofPred_forall]
    exact isOpen_iInter_of_finite fun i =>
      isOpen_lt continuous_const ((PiLp.continuous_apply 2 (fun _ : Fin K => ℝ) i).comp
        (continuous_const.add continuous_subtype_val))
  have h0 : (0 : Tangent K) ∈ U := by simpa [U] using hp
  obtain ⟨r, hr, hsub⟩ := Metric.isOpen_iff.mp hU 0 h0
  refine ⟨r / 2, half_pos hr, ?_⟩
  intro v hv
  have hvr : v ∈ Metric.ball (0 : Tangent K) r := by
    have hlt : ‖v‖ < r := lt_of_le_of_lt hv (half_lt_self hr)
    simpa using hlt
  have hpos : ∀ i, 0 < (p + v.val) i := hsub hvr
  refine ⟨fun i => (hpos i).le, ?_⟩
  change (∑ i, (p i + v.val i)) = 1
  rw [Finset.sum_add_distrib, hmass]
  have hzero : ∑ i, v.val i = 0 := v.property
  rw [hzero, add_zero]

/-- The Chapter 8 verification of smoothness for an interior point away from
the rule's decision boundary. Local action stability is the boundary-exclusion
premise; the simplex-valid radius, derivative and zero remainder are derived.
M=1 is positive as required by the assumption, although the remainder is zero. -/
theorem smoothness_of_locally_stable [NeZero K] (C : CostMatrix K)
    (δ : EuclideanSpace ℝ (Fin K) → Fin K) (p : EuclideanSpace ℝ (Fin K))
    (hp : ∀ i, 0 < p i) (hmass : ∑ i, p i = 1)
    (hstable : ∃ R : ℝ, 0 < R ∧
      ∀ v : Tangent K, ‖v‖ ≤ R → δ (p + v.val) = δ p) :
    ∃ ρ : ℝ, 0 < ρ ∧
      LocalSmoothness (ruleCost C δ) p (centeredGradient C (δ p)) 1 ρ := by
  obtain ⟨R, hR, hstableR⟩ := hstable
  obtain ⟨S, hS, hballS⟩ := exists_simplex_tangent_ball p hp hmass
  have hρ : 0 < min R S := lt_min hR hS
  refine ⟨min R S, hρ, ?_⟩
  apply localSmoothness_ruleCost C δ p hp hmass zero_lt_one hρ
  · intro v hv
    exact (hballS v (hv.trans (min_le_right _ _))).1
  · intro v hv
    exact hstableR v (hv.trans (min_le_left _ _))
end ThresholdOpt.Diffusion