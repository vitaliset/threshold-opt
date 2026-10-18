import ThresholdOpt.Diffusion

/-!
# Integrated dilution
A common smoothness radius is necessary in addition to the text's pointwise
assumptions. With that correction, square-integrability of M and the gradient
controls the joint nonlinear remainder. Product sampling represents the same
uniform conditional sphere law for every X.
-/

namespace ThresholdOpt.Diffusion
open ThresholdOpt MeasureTheory
open scoped RealInnerProductSpace BigOperators Classical
variable {K : ℕ}
set_option maxHeartbeats 800000

/-- Corrected integrated theorem. Partial relative to the text because rho is
a common positive radius, whereas a.e. point-dependent radii are insufficient.
The expectation is over the actual joint product probability law; all inner,
outer and joint loss integrability is proved, not assumed. -/
theorem integrated_dilution {X : Type*} [MeasurableSpace X]
    {P : Measure X} [IsProbabilityMeasure P] (hK : 2 ≤ K)
    (b : OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K)) (hm : SphereMoments b)
    {φ : EuclideanSpace ℝ (Fin K) → ℝ} (hφ : Measurable φ)
    {p : X → EuclideanSpace ℝ (Fin K)} (hp : Measurable p)
    {g : X → Tangent K} {M : X → ℝ} {ρ r : ℝ}
    (hs : ∀ᵐ x ∂P, LocalSmoothness φ (p x) (g x) (M x) ρ)
    (hg2 : Integrable (fun x => ‖g x‖ ^ 2) P)
    (hM2 : Integrable (fun x => M x ^ 2) P)
    (hr : 0 < r) (hρ : r ≤ ρ) :
    |(∫ z : X × UnitSphere K,
        (φ (p z.1 + (noise r z.2).val) - φ (p z.1)) ^ 2 ∂P.prod (unitSphereLaw K)) -
      r ^ 2 / (K - 1 : ℕ) * (∫ x, ‖g x‖ ^ 2 ∂P)| ≤
      ((∫ x, M x ^ 2 ∂P) + (∫ x, ‖g x‖ ^ 2 ∂P)) * r ^ 3 +
        (∫ x, M x ^ 2 ∂P) * r ^ 4 := by
  haveI := unitSphereLaw_probability hK
  let F : X × UnitSphere K → ℝ := fun z =>
    (φ (p z.1 + (noise r z.2).val) - φ (p z.1)) ^ 2
  let A : X → ℝ := fun x => ∫ u, F (x, u) ∂unitSphereLaw K
  let c : ℝ := r ^ 2 / (K - 1 : ℕ)
  let E : X → ℝ := fun x => (M x ^ 2 + ‖g x‖ ^ 2) * r ^ 3 + M x ^ 2 * r ^ 4
  have hF : Measurable F := by
    have hnoise : Measurable (fun u : UnitSphere K => (noise r u).val) :=
      measurable_subtype_coe.comp (measurable_noise r)
    exact ((hφ.comp ((hp.comp measurable_fst).add
      (hnoise.comp measurable_snd))).sub (hφ.comp (hp.comp measurable_fst))).pow_const 2
  have hAmeas : AEStronglyMeasurable A P :=
    hF.stronglyMeasurable.integral_prod_right'.aestronglyMeasurable
  have hE : Integrable E P :=
    ((hM2.add hg2).mul_const (r ^ 3)).add (hM2.mul_const (r ^ 4))
  have hbound : ∀ᵐ x ∂P, ‖A x - c * ‖g x‖ ^ 2‖ ≤ E x := by
    filter_upwards [hs] with x hx
    have h := dimensional_dilution_on_unit hK b hm hφ hx hr.le hρ
    have hMg : 2 * ‖g x‖ * M x ≤ M x ^ 2 + ‖g x‖ ^ 2 := by
      nlinarith [sq_nonneg (M x - ‖g x‖)]
    change |A x - c * ‖g x‖ ^ 2| ≤ E x
    exact h.trans (by
      change 2 * ‖g x‖ * M x * r ^ 3 + M x ^ 2 * r ^ 4 ≤
        (M x ^ 2 + ‖g x‖ ^ 2) * r ^ 3 + M x ^ 2 * r ^ 4
      exact add_le_add (mul_le_mul_of_nonneg_right hMg (show 0 ≤ r ^ 3 by positivity))
        (le_refl (M x ^ 2 * r ^ 4)))
  have hR : Integrable (fun x => A x - c * ‖g x‖ ^ 2) P :=
    hE.mono' (hAmeas.sub (hg2.aestronglyMeasurable.const_mul c)) hbound
  have hA : Integrable A P := by
    have h := hR.add (hg2.const_mul c)
    change Integrable (fun x => (A x - c * ‖g x‖ ^ 2) + c * ‖g x‖ ^ 2) P at h
    simpa only [sub_add_cancel] using h
  have hprod : Integrable F (P.prod (unitSphereLaw K)) := by
    apply (integrable_prod_iff hF.aestronglyMeasurable).mpr
    constructor
    · filter_upwards [hs] with x hx
      change Integrable (fun u : UnitSphere K =>
        (φ (p x + (noise r u).val) - φ (p x)) ^ 2) (unitSphereLaw K)
      exact integrable_change_square hK hφ hx hr.le hρ
    · simpa only [F, Real.norm_eq_abs, abs_sq] using hA
  have hEeq : (∫ x, E x ∂P) =
      ((∫ x, M x ^ 2 ∂P) + (∫ x, ‖g x‖ ^ 2 ∂P)) * r ^ 3 +
        (∫ x, M x ^ 2 ∂P) * r ^ 4 := by
    change (∫ x, (M x ^ 2 + ‖g x‖ ^ 2) * r ^ 3 + M x ^ 2 * r ^ 4 ∂P) = _
    have hsum := hM2.add hg2
    change Integrable (fun x => M x ^ 2 + ‖g x‖ ^ 2) P at hsum
    rw [integral_add (hsum.mul_const (r ^ 3)) (hM2.mul_const (r ^ 4)),
      integral_mul_const, integral_mul_const, integral_add hM2 hg2]
  change |(∫ z, F z ∂P.prod (unitSphereLaw K)) - c * (∫ x, ‖g x‖ ^ 2 ∂P)| ≤ _
  rw [integral_prod F hprod]
  change |(∫ x, A x ∂P) - c * (∫ x, ‖g x‖ ^ 2 ∂P)| ≤ _
  rw [← integral_const_mul, ← integral_sub hA (hg2.const_mul c)]
  have hnorm := norm_integral_le_of_norm_le hE hbound
  simpa only [Real.norm_eq_abs, hEeq] using hnorm

/-- A single explicit constant works for every sufficiently small positive r. -/
theorem integrated_dilution_bigO_bound {X : Type*} [MeasurableSpace X]
    {P : Measure X} [IsProbabilityMeasure P] (hK : 2 ≤ K)
    (b : OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K)) (hm : SphereMoments b)
    {φ : EuclideanSpace ℝ (Fin K) → ℝ} (hφ : Measurable φ)
    {p : X → EuclideanSpace ℝ (Fin K)} (hp : Measurable p)
    {g : X → Tangent K} {M : X → ℝ} {ρ : ℝ}
    (hs : ∀ᵐ x ∂P, LocalSmoothness φ (p x) (g x) (M x) ρ)
    (hg2 : Integrable (fun x => ‖g x‖ ^ 2) P)
    (hM2 : Integrable (fun x => M x ^ 2) P) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ r : ℝ, 0 < r → r ≤ min ρ 1 →
      |(∫ z : X × UnitSphere K,
          (φ (p z.1 + (noise r z.2).val) - φ (p z.1)) ^ 2 ∂P.prod (unitSphereLaw K)) -
        r ^ 2 / (K - 1 : ℕ) * (∫ x, ‖g x‖ ^ 2 ∂P)| ≤ C * r ^ 3 := by
  have hMpos : 0 ≤ ∫ x, M x ^ 2 ∂P := integral_nonneg fun _ => sq_nonneg _
  have hGpos : 0 ≤ ∫ x, ‖g x‖ ^ 2 ∂P := integral_nonneg fun _ => sq_nonneg _
  refine ⟨2 * (∫ x, M x ^ 2 ∂P) + (∫ x, ‖g x‖ ^ 2 ∂P), by positivity, ?_⟩
  intro r hr hsmall
  have hr1 := hsmall.trans (min_le_right _ _)
  have hr43 : r ^ 4 ≤ r ^ 3 := by
    calc
      r ^ 4 = r ^ 3 * r := by ring
      _ ≤ r ^ 3 * 1 := mul_le_mul_of_nonneg_left hr1 (by positivity)
      _ = r ^ 3 := mul_one _
  calc
    _ ≤ ((∫ x, M x ^ 2 ∂P) + (∫ x, ‖g x‖ ^ 2 ∂P)) * r ^ 3 +
        (∫ x, M x ^ 2 ∂P) * r ^ 4 :=
      integrated_dilution hK b hm hφ hp hs hg2 hM2 hr (hsmall.trans (min_le_left _ _))
    _ ≤ ((∫ x, M x ^ 2 ∂P) + (∫ x, ‖g x‖ ^ 2 ∂P)) * r ^ 3 +
        (∫ x, M x ^ 2 ∂P) * r ^ 3 := by gcongr
    _ = _ := by ring

end ThresholdOpt.Diffusion