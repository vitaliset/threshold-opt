import ThresholdOpt.Sensitivity
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Normed.Lp.MeasurableSpace
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.MeasureTheory.Constructions.HaarToSphere
import Mathlib.MeasureTheory.Measure.Haar.Basic
import Mathlib.Tactic.GCongr

/-!
# Dimensional dilution with explicit uniform-sphere background
The tangent norm is Euclidean. Coordinate second moments of the normalized
uniform sphere are explicit standard background; finite-sum projection algebra
and Taylor remainder estimates are proved here.
-/

namespace ThresholdOpt.Diffusion
open ThresholdOpt MeasureTheory
open scoped BigOperators RealInnerProductSpace Classical
variable {K : ℕ}

/-- Total-mass derivative on the Euclidean ambient space. -/
def massDerivative (K : ℕ) : EuclideanSpace ℝ (Fin K) →ₗ[ℝ] ℝ where
  toFun v := ∑ i, v i
  map_add' a b := by simp [Finset.sum_add_distrib]
  map_smul' c a := by simp [← Finset.mul_sum]

/-- The sum-zero subspace, with the inherited Euclidean inner product. -/
abbrev Tangent (K : ℕ) := LinearMap.ker (massDerivative K)

@[simp] theorem mem_tangent (v : EuclideanSpace ℝ (Fin K)) :
    v ∈ Tangent K ↔ ∑ i, v i = 0 := Iff.rfl

theorem tangent_finrank [NeZero K] : Module.finrank ℝ (Tangent K) = K - 1 := by
  have hsurj : Function.Surjective (massDerivative K) := by
    intro r
    refine ⟨EuclideanSpace.single (0 : Fin K) r, ?_⟩
    simp [massDerivative]
  have h := (massDerivative K).finrank_range_add_finrank_ker
  rw [LinearMap.range_eq_top.mpr hsurj, finrank_top, Module.finrank_self,
    finrank_euclideanSpace_fin] at h
  change 1 + Module.finrank ℝ (Tangent K) = K at h
  omega

/-- An actual basis with K-1 coordinates; rank-nullity proved the dimension. -/
noncomputable def tangentBasis [NeZero K] :
    OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K) :=
  (stdOrthonormalBasis ℝ (Tangent K)).reindex (finCongr tangent_finrank)

abbrev UnitSphere (K : ℕ) := Metric.sphere (0 : Tangent K) 1

/-- Normalize Haar-induced Euclidean surface measure on the unit sphere. -/
noncomputable def unitSphereLaw (K : ℕ) : Measure (UnitSphere K) :=
  let S := (Measure.addHaar : Measure (Tangent K)).toSphere
  (S Set.univ)⁻¹ • S

theorem unitSphereLaw_probability (hK : 2 ≤ K) :
    IsProbabilityMeasure (unitSphereLaw K) := by
  have hKn : K ≠ 0 := by omega
  haveI : NeZero K := ⟨hKn⟩
  let S := (Measure.addHaar : Measure (Tangent K)).toSphere
  have hS : S ≠ 0 := by
    intro hzero
    have hz := (Measure.toSphere_eq_zero_iff_finrank
      (Measure.addHaar : Measure (Tangent K))).mp hzero
    rw [tangent_finrank] at hz
    omega
  haveI : NeZero S := ⟨hS⟩
  change IsProbabilityMeasure ((S Set.univ)⁻¹ • S)
  infer_instance

/-- A radius-r direction generated from the actual uniform unit sphere. -/
def noise (r : ℝ) (u : UnitSphere K) : Tangent K := r • u.val

theorem noise_norm {r : ℝ} (hr : 0 ≤ r) (u : UnitSphere K) :
    ‖noise r u‖ = r := by
  have hu : ‖u.val‖ = 1 := by simpa using u.property
  simp [noise, norm_smul, Real.norm_eq_abs, abs_of_nonneg hr, hu]

noncomputable def uniformSphere (K : ℕ) (r : ℝ) : Measure (Tangent K) :=
  (unitSphereLaw K).map (noise r)

/-- The ambient extension is only used on the tangent ball within the simplex.
The derivative is that of the relative chart at zero. -/
@[thesis "ass:smoothness"]
def LocalSmoothness (φ : EuclideanSpace ℝ (Fin K) → ℝ)
    (p : EuclideanSpace ℝ (Fin K)) (g : Tangent K) (M ρ : ℝ) : Prop :=
  (∀ i, 0 < p i) ∧ (∑ i, p i = 1) ∧ 0 < M ∧ 0 < ρ ∧
    (∀ v : Tangent K, ‖v‖ ≤ ρ → (∀ i, 0 ≤ (p + v.val) i)) ∧
    HasFDerivAt (fun v : Tangent K => φ (p + v.val)) (innerSL ℝ g) 0 ∧
    ∀ v : Tangent K, ‖v‖ ≤ ρ →
      |φ (p + v.val) - φ p - ⟪g, v⟫| ≤ M * ‖v‖ ^ 2

/-- Assumption of positive radius within the smoothness ball and exactly the
uniform Euclidean sphere law, rather than a moment condition alone. -/
@[thesis "ass:diffuse"]
def Diffuse (P : Measure (Tangent K)) (r ρ : ℝ) : Prop :=
  0 < r ∧ r ≤ ρ ∧ P = uniformSphere K r

/-- Standard background for the actual uniform sphere: orthonormal-coordinate
products are integrable and have covariance identity divided by the dimension.
This does not assume the projection or nonlinear conclusion of the lemma. -/
structure SphereMoments (b : OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K)) : Prop where
  integrable : ∀ i j, Integrable (fun u : UnitSphere K => ⟪b i, u.val⟫ * ⟪b j, u.val⟫)
    (unitSphereLaw K)
  value : ∀ i j, (∫ u : UnitSphere K, ⟪b i, u.val⟫ * ⟪b j, u.val⟫ ∂unitSphereLaw K) =
    if i = j then (1 : ℝ) / (K - 1 : ℕ) else 0

theorem inner_square_expansion
    (b : OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K)) (g v : Tangent K) :
    ⟪g, v⟫ ^ 2 = ∑ i, ∑ j,
      (⟪g, b i⟫ * ⟪g, b j⟫) * (⟪b i, v⟫ * ⟪b j, v⟫) := by
  rw [← b.sum_inner_mul_inner g v, pow_two, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem integrable_inner_square
    (b : OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K))
    (hm : SphereMoments b) (g : Tangent K) :
    Integrable (fun u : UnitSphere K => ⟪g, u.val⟫ ^ 2) (unitSphereLaw K) := by
  simp_rw [inner_square_expansion b g]
  exact integrable_finsetSum _ fun i _ =>
    integrable_finsetSum _ fun j _ => (hm.integrable i j).const_mul _

/-- The exact leading projection identity, obtained from finite sums and
Parseval; the only assumed integration facts are individual coordinate moments. -/
theorem projection_second_moment
    (b : OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K))
    (hm : SphereMoments b) (g : Tangent K) (r : ℝ) :
    (∫ u : UnitSphere K, ⟪g, noise r u⟫ ^ 2 ∂unitSphereLaw K) =
      r ^ 2 / (K - 1 : ℕ) * ‖g‖ ^ 2 := by
  have hsum : (∫ u : UnitSphere K, ⟪g, u.val⟫ ^ 2 ∂unitSphereLaw K) =
      (1 : ℝ) / (K - 1 : ℕ) * ‖g‖ ^ 2 := by
    simp_rw [inner_square_expansion b g]
    rw [integral_finsetSum _ (fun i _ =>
      integrable_finsetSum _ fun j _ => (hm.integrable i j).const_mul _)]
    have hi : ∀ i, (∫ u : UnitSphere K, ∑ j,
        (⟪g, b i⟫ * ⟪g, b j⟫) * (⟪b i, u.val⟫ * ⟪b j, u.val⟫) ∂unitSphereLaw K) =
        ∑ j, (⟪g, b i⟫ * ⟪g, b j⟫) *
          (if i = j then (1 : ℝ) / (K - 1 : ℕ) else 0) := by
      intro i
      rw [integral_finsetSum _ (fun j _ => (hm.integrable i j).const_mul _)]
      simp_rw [integral_const_mul, hm.value]
    simp_rw [hi]
    simp only [mul_ite, mul_zero, Finset.sum_ite_eq, Finset.mem_univ, if_true]
    simp_rw [← pow_two]
    rw [← Finset.sum_mul, b.sum_sq_inner_left]
    ring
  have hscale : ∀ u : UnitSphere K, ⟪g, noise r u⟫ = r * ⟪g, u.val⟫ := by
    intro u
    exact inner_smul_right g u.val r
  simp_rw [hscale, mul_pow]
  rw [integral_const_mul, hsum]
  ring
/-- Elementary quantitative Taylor-error estimate. -/
theorem square_error_le {a b B R : ℝ} (hB : 0 ≤ B) (_hR : 0 ≤ R)
    (hb : |b| ≤ B) (he : |a - b| ≤ R) :
    |a ^ 2 - b ^ 2| ≤ 2 * B * R + R ^ 2 := by
  calc
    |a ^ 2 - b ^ 2| = |(a - b) ^ 2 + 2 * b * (a - b)| := by congr 1; ring
    _ ≤ |(a - b) ^ 2| + |2 * b * (a - b)| := abs_add_le _ _
    _ = |a - b| ^ 2 + 2 * |b| * |a - b| := by
      simp [abs_mul]
    _ ≤ R ^ 2 + 2 * B * R := by gcongr
    _ = 2 * B * R + R ^ 2 := by ring


theorem noise_inner_abs {r : ℝ} (hr : 0 ≤ r) (g : Tangent K) (u : UnitSphere K) :
    |⟪g, noise r u⟫| ≤ ‖g‖ * r := by
  calc
    |⟪g, noise r u⟫| = ‖⟪g, noise r u⟫‖ := Real.norm_eq_abs _ |>.symm
    _ ≤ ‖g‖ * ‖noise r u‖ := norm_inner_le_norm _ _
    _ = ‖g‖ * r := by rw [noise_norm hr]

theorem local_remainder_bound {φ : EuclideanSpace ℝ (Fin K) → ℝ}
    {p : EuclideanSpace ℝ (Fin K)} {g : Tangent K} {M ρ r : ℝ}
    (hs : LocalSmoothness φ p g M ρ) (hr : 0 ≤ r) (hρ : r ≤ ρ) (u : UnitSphere K) :
    |φ (p + (noise r u).val) - φ p - ⟪g, noise r u⟫| ≤ M * r ^ 2 := by
  simpa only [noise_norm hr] using
    hs.2.2.2.2.2.2 (noise r u) (by rw [noise_norm hr]; exact hρ)

theorem local_change_bound {φ : EuclideanSpace ℝ (Fin K) → ℝ}
    {p : EuclideanSpace ℝ (Fin K)} {g : Tangent K} {M ρ r : ℝ}
    (hs : LocalSmoothness φ p g M ρ) (hr : 0 ≤ r) (hρ : r ≤ ρ) (u : UnitSphere K) :
    |φ (p + (noise r u).val) - φ p| ≤ ‖g‖ * r + M * r ^ 2 := by
  have he := local_remainder_bound hs hr hρ u
  have hb := noise_inner_abs hr g u
  have ht := abs_add_le ⟪g, noise r u⟫
    (φ (p + (noise r u).val) - φ p - ⟪g, noise r u⟫)
  have hid : ⟪g, noise r u⟫ +
      (φ (p + (noise r u).val) - φ p - ⟪g, noise r u⟫) =
      φ (p + (noise r u).val) - φ p := by ring
  rw [hid] at ht
  linarith

theorem measurable_noise (r : ℝ) : Measurable (noise r : UnitSphere K → Tangent K) := by
  change Measurable (fun u : UnitSphere K => r • u.val)
  exact (continuous_subtype_val.const_smul r).measurable

theorem measurable_change_square {φ : EuclideanSpace ℝ (Fin K) → ℝ}
    (hφ : Measurable φ) (p : EuclideanSpace ℝ (Fin K)) :
    Measurable (fun v : Tangent K => (φ (p + v.val) - φ p) ^ 2) :=
  ((hφ.comp (continuous_const.add continuous_subtype_val).measurable).sub
    measurable_const).pow_const 2

theorem integrable_change_square (hK : 2 ≤ K)
    {φ : EuclideanSpace ℝ (Fin K) → ℝ} (hφ : Measurable φ)
    {p : EuclideanSpace ℝ (Fin K)} {g : Tangent K} {M ρ r : ℝ}
    (hs : LocalSmoothness φ p g M ρ) (hr : 0 ≤ r) (hρ : r ≤ ρ) :
    Integrable (fun u : UnitSphere K =>
      (φ (p + (noise r u).val) - φ p) ^ 2) (unitSphereLaw K) := by
  haveI := unitSphereLaw_probability hK
  refine ⟨((measurable_change_square hφ p).comp (measurable_noise r)).aestronglyMeasurable, ?_⟩
  apply HasFiniteIntegral.of_bounded (C := (‖g‖ * r + M * r ^ 2) ^ 2)
  apply Filter.Eventually.of_forall
  intro u
  rw [Real.norm_eq_abs, abs_sq, ← sq_abs (φ (p + (noise r u).val) - φ p)]
  have hM : 0 ≤ M := hs.2.2.1.le
  have hb := local_change_bound hs hr hρ u
  gcongr

/-- Quantitative form of the pointwise lemma, before pushing the radius-r
noise to the tangent space. Its bound is O(r³) with fixed M and gradient. -/
theorem dimensional_dilution_on_unit (hK : 2 ≤ K)
    (b : OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K)) (hm : SphereMoments b)
    {φ : EuclideanSpace ℝ (Fin K) → ℝ} (hφ : Measurable φ)
    {p : EuclideanSpace ℝ (Fin K)} {g : Tangent K} {M ρ r : ℝ}
    (hs : LocalSmoothness φ p g M ρ) (hr : 0 ≤ r) (hρ : r ≤ ρ) :
    |(∫ u : UnitSphere K, (φ (p + (noise r u).val) - φ p) ^ 2 ∂unitSphereLaw K) -
      r ^ 2 / (K - 1 : ℕ) * ‖g‖ ^ 2| ≤
      2 * ‖g‖ * M * r ^ 3 + M ^ 2 * r ^ 4 := by
  haveI := unitSphereLaw_probability hK
  have hM : 0 ≤ M := hs.2.2.1.le
  have hL : Integrable (fun u : UnitSphere K => ⟪g, noise r u⟫ ^ 2) (unitSphereLaw K) := by
    have hscale : ∀ u : UnitSphere K, ⟪g, noise r u⟫ = r * ⟪g, u.val⟫ :=
      fun u => inner_smul_right g u.val r
    simp_rw [hscale, mul_pow]
    exact (integrable_inner_square b hm g).const_mul (r ^ 2)
  rw [← projection_second_moment b hm g r,
    ← integral_sub (integrable_change_square hK hφ hs hr hρ) hL]
  have hbound : ∀ u : UnitSphere K,
      |(φ (p + (noise r u).val) - φ p) ^ 2 - ⟪g, noise r u⟫ ^ 2| ≤
        2 * ‖g‖ * M * r ^ 3 + M ^ 2 * r ^ 4 := by
    intro u
    have h := square_error_le (B := ‖g‖ * r) (R := M * r ^ 2)
      (by positivity) (by positivity) (noise_inner_abs hr g u)
      (local_remainder_bound hs hr hρ u)
    convert h using 1; ring
  have hnorm := norm_integral_le_of_norm_le_const
    (μ := unitSphereLaw K)
    (f := fun u : UnitSphere K =>
      (φ (p + (noise r u).val) - φ p) ^ 2 - ⟪g, noise r u⟫ ^ 2)
    (C := 2 * ‖g‖ * M * r ^ 3 + M ^ 2 * r ^ 4)
    (Filter.Eventually.of_forall fun u => by simpa only [Real.norm_eq_abs] using hbound u)
  simpa only [Real.norm_eq_abs, probReal_univ, mul_one] using hnorm

/-- Full nonlinear dimensional dilution from individual sphere moments.
The explicit bound proves the stated leading coefficient plus O(r³);
a scalar measurability hypothesis supplies the implicit expectation background. -/
theorem dimensional_dilution (hK : 2 ≤ K)
    (b : OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K)) (hm : SphereMoments b)
    {φ : EuclideanSpace ℝ (Fin K) → ℝ} (hφ : Measurable φ)
    {p : EuclideanSpace ℝ (Fin K)} {g : Tangent K} {M ρ r : ℝ}
    (hs : LocalSmoothness φ p g M ρ)
    {P : Measure (Tangent K)} (hd : Diffuse P r ρ) :
    |(∫ v, (φ (p + v.val) - φ p) ^ 2 ∂P) -
      r ^ 2 / (K - 1 : ℕ) * ‖g‖ ^ 2| ≤
      2 * ‖g‖ * M * r ^ 3 + M ^ 2 * r ^ 4 := by
  rw [hd.2.2, uniformSphere,
    integral_map (measurable_noise r).aemeasurable
      (measurable_change_square hφ p).aestronglyMeasurable]
  exact dimensional_dilution_on_unit hK b hm hφ hs hd.1.le hd.2.1

/-- Explicit constants and a punctured neighbourhood spelling out O(r³). -/
theorem dimensional_dilution_bigO_bound (hK : 2 ≤ K)
    (b : OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K)) (hm : SphereMoments b)
    {φ : EuclideanSpace ℝ (Fin K) → ℝ} (hφ : Measurable φ)
    {p : EuclideanSpace ℝ (Fin K)} {g : Tangent K} {M ρ : ℝ}
    (hs : LocalSmoothness φ p g M ρ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ r : ℝ, 0 < r → r ≤ min ρ 1 →
      |(∫ v, (φ (p + v.val) - φ p) ^ 2 ∂uniformSphere K r) -
        r ^ 2 / (K - 1 : ℕ) * ‖g‖ ^ 2| ≤ C * r ^ 3 := by
  have hM : 0 ≤ M := hs.2.2.1.le
  refine ⟨2 * ‖g‖ * M + M ^ 2, by positivity, ?_⟩
  intro r hr hsmall
  have hρ := hsmall.trans (min_le_left _ _)
  have hr1 := hsmall.trans (min_le_right _ _)
  have hr43 : r ^ 4 ≤ r ^ 3 := by
    calc
      r ^ 4 = r ^ 3 * r := by ring
      _ ≤ r ^ 3 * 1 := mul_le_mul_of_nonneg_left hr1 (by positivity)
      _ = r ^ 3 := mul_one _
  calc
    _ ≤ 2 * ‖g‖ * M * r ^ 3 + M ^ 2 * r ^ 4 :=
      dimensional_dilution hK b hm hφ hs ⟨hr, hρ, rfl⟩
    _ ≤ 2 * ‖g‖ * M * r ^ 3 + M ^ 2 * r ^ 3 := by gcongr
    _ = (2 * ‖g‖ * M + M ^ 2) * r ^ 3 := by ring
end ThresholdOpt.Diffusion