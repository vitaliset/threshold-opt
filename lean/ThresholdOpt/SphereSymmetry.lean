import Mathlib.Analysis.InnerProductSpace.Projection.Reflection
import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import ThresholdOpt.IntegratedDiffusion

/-!
# Second moments of the uniform sphere (Chapter 8)

`Diffusion.SphereMoments` is the one hypothesis of `lem:dimensional_dilution`
as formalized in [`Diffusion.lean`](Diffusion.lean): on the unit sphere of the
tangent space, with its normalized surface measure, the coordinates `Z_k` in an
orthonormal basis satisfy `E[Z_k Z_l] = 0` for `k ≠ l` and
`E[Z_k²] = 1 / (K - 1)`. The dissertation derives this in the proof of the
lemma, from the invariance of the uniform distribution under sign flips and
coordinate permutations. This file proves it, and with it the lemma and its
integrated form hold with no hypothesis on the sphere.

The argument is the one in the text.

* A linear isometry preserves Haar measure, because it preserves the unit ball.
  Hence it preserves the surface measure of the unit sphere.
* The reflection that sends `b_k` to `-b_k` fixes the other basis vectors, so
  `E[Z_k Z_l] = -E[Z_k Z_l]`.
* The reflection that exchanges `b_k` and `b_l` gives `E[Z_k²] = E[Z_l²]`.
* `∑_k Z_k² = 1` on the sphere, so the common value is `1 / (K - 1)`.
-/

namespace ThresholdOpt.SphereSymmetry

open MeasureTheory Metric Set ThresholdOpt.Diffusion
open scoped RealInnerProductSpace Pointwise

/-! ### Linear isometries preserve Haar measure and the surface measure -/

section Isometry

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]

/-- A linear isometry preserves every additive Haar measure: it scales it by a
constant, and it maps the unit ball onto itself. -/
theorem addHaar_preimage_isometry (μ : Measure E) [μ.IsAddHaarMeasure]
    (f : E ≃ₗᵢ[ℝ] E) (s : Set E) : μ (f ⁻¹' s) = μ s := by
  have h := Measure.addHaar_preimage_continuousLinearEquiv μ f.toContinuousLinearEquiv
  have hball : (f.toContinuousLinearEquiv : E → E) ⁻¹' ball 0 1 = ball 0 1 := by
    ext x
    simp
  have h1 := h (ball 0 1)
  rw [hball] at h1
  have hpos : μ (ball (0 : E) 1) ≠ 0 :=
    (isOpen_ball.measure_pos μ (nonempty_ball.2 one_pos)).ne'
  have hfin : μ (ball (0 : E) 1) ≠ ⊤ := measure_ball_lt_top.ne
  have hc : ENNReal.ofReal
      |LinearMap.det (f.toContinuousLinearEquiv.symm : E →ₗ[ℝ] E)| = 1 := by
    have h2 : ENNReal.ofReal
        |LinearMap.det (f.toContinuousLinearEquiv.symm : E →ₗ[ℝ] E)| * μ (ball 0 1)
          = 1 * μ (ball 0 1) := by rw [one_mul]; exact h1.symm
    exact (ENNReal.mul_left_inj hpos hfin).1 h2
  have h3 := h s
  rw [hc, one_mul] at h3
  exact h3

/-- A linear isometry, restricted to the unit sphere. -/
def sphereMap (f : E ≃ₗᵢ[ℝ] E) (u : sphere (0 : E) 1) : sphere (0 : E) 1 :=
  ⟨f u, by
    have hu : ‖(u : E)‖ = 1 := mem_sphere_zero_iff_norm.1 u.property
    exact mem_sphere_zero_iff_norm.2 (by rw [f.norm_map, hu])⟩

omit [FiniteDimensional ℝ E] in
theorem measurable_sphereMap (f : E ≃ₗᵢ[ℝ] E) : Measurable (sphereMap f) :=
  ((f.continuous.comp continuous_subtype_val).subtype_mk _).measurable

/-- The surface measure of the unit sphere is invariant under linear
isometries. -/
theorem toSphere_map_sphereMap (μ : Measure E) [μ.IsAddHaarMeasure] (f : E ≃ₗᵢ[ℝ] E) :
    (μ.toSphere).map (sphereMap f) = μ.toSphere := by
  refine Measure.ext fun s hs => ?_
  rw [Measure.map_apply (measurable_sphereMap f) hs,
    Measure.toSphere_apply' μ (measurable_sphereMap f hs), Measure.toSphere_apply' μ hs]
  congr 1
  have himg : ((↑) : sphere (0 : E) 1 → E) '' (sphereMap f ⁻¹' s)
      = f ⁻¹' (((↑) : sphere (0 : E) 1 → E) '' s) := by
    ext x
    constructor
    · rintro ⟨u, hu, rfl⟩
      exact ⟨sphereMap f u, hu, rfl⟩
    · rintro ⟨w, hw, hwx⟩
      have hw1 : ‖(w : E)‖ = 1 := mem_sphere_zero_iff_norm.1 w.property
      have hx : ‖x‖ = 1 := by rw [← f.norm_map x, ← hwx, hw1]
      refine ⟨⟨x, mem_sphere_zero_iff_norm.2 hx⟩, ?_, rfl⟩
      have : sphereMap f ⟨x, mem_sphere_zero_iff_norm.2 hx⟩ = w := Subtype.ext hwx.symm
      rw [Set.mem_preimage, this]
      exact hw
  have hsmul : ∀ A : Set E, Ioo (0 : ℝ) 1 • (f ⁻¹' A) = f ⁻¹' (Ioo (0 : ℝ) 1 • A) := by
    intro A
    ext y
    simp only [Set.mem_smul, Set.mem_preimage]
    constructor
    · rintro ⟨t, ht, a, ha, rfl⟩
      exact ⟨t, ht, f a, ha, (map_smul f t a).symm⟩
    · rintro ⟨t, ht, a, ha, hta⟩
      refine ⟨t, ht, f.symm a, by simpa using ha, ?_⟩
      apply f.injective
      rw [map_smul, LinearIsometryEquiv.apply_symm_apply, hta]
  rw [himg, hsmul, addHaar_preimage_isometry μ f]

omit [MeasurableSpace E] [BorelSpace E] in
/-- A reflection is self-adjoint. -/
theorem inner_reflection (V : Submodule ℝ E) (x y : E) :
    ⟪x, V.reflection y⟫ = ⟪V.reflection x, y⟫ := by
  have h := (V.reflection).inner_map_map x (V.reflection y)
  rw [Submodule.reflection_reflection] at h
  exact h.symm

end Isometry

/-! ### The moments -/

variable {K : ℕ}

/-- The normalized surface measure of the unit sphere of the tangent space is
invariant under linear isometries. -/
theorem unitSphereLaw_map (f : Tangent K ≃ₗᵢ[ℝ] Tangent K) :
    (unitSphereLaw K).map (sphereMap f) = unitSphereLaw K := by
  unfold unitSphereLaw
  simp only
  rw [Measure.map_smul, toSphere_map_sphereMap]

/-- Integrals over the uniform sphere are invariant under linear isometries. -/
theorem integral_comp_sphereMap (f : Tangent K ≃ₗᵢ[ℝ] Tangent K)
    {g : UnitSphere K → ℝ} (hg : Continuous g) :
    ∫ u, g (sphereMap f u) ∂unitSphereLaw K = ∫ u, g u ∂unitSphereLaw K := by
  have h := integral_map (μ := unitSphereLaw K) (measurable_sphereMap f).aemeasurable
    (f := g) hg.aestronglyMeasurable
  rw [unitSphereLaw_map] at h
  exact h.symm

theorem continuous_coord (v : Tangent K) :
    Continuous fun u : UnitSphere K => ⟪v, u.val⟫ :=
  continuous_const.inner continuous_subtype_val

theorem abs_coord_le (b : OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K)) (i : Fin (K - 1))
    (u : UnitSphere K) : |⟪b i, u.val⟫| ≤ 1 := by
  have hu : ‖u.val‖ = 1 := mem_sphere_zero_iff_norm.1 u.property
  calc |⟪b i, u.val⟫| ≤ ‖b i‖ * ‖u.val‖ := abs_real_inner_le_norm _ _
    _ = 1 := by rw [b.orthonormal.1 i, hu, mul_one]

/-- **The second moments of the uniform sphere.** For `K ≥ 2` and any
orthonormal basis of the tangent space, the coordinates of a uniform point of
the unit sphere have `E[Z_k Z_l] = 0` for `k ≠ l` and `E[Z_k²] = 1 / (K - 1)`.
This is the hypothesis `SphereMoments` of `Diffusion.dimensional_dilution`. -/
theorem sphereMoments (hK : 2 ≤ K) (b : OrthonormalBasis (Fin (K - 1)) ℝ (Tangent K)) :
    SphereMoments b := by
  have := unitSphereLaw_probability hK
  have hint : ∀ i j, Integrable
      (fun u : UnitSphere K => ⟪b i, u.val⟫ * ⟪b j, u.val⟫) (unitSphereLaw K) := by
    intro i j
    refine (integrable_const (1 : ℝ)).mono'
      ((continuous_coord (b i)).mul (continuous_coord (b j))).aestronglyMeasurable
      (Filter.Eventually.of_forall fun u => ?_)
    rw [Real.norm_eq_abs, abs_mul]
    calc |⟪b i, u.val⟫| * |⟪b j, u.val⟫| ≤ 1 * 1 :=
          mul_le_mul (abs_coord_le b i u) (abs_coord_le b j u) (abs_nonneg _) zero_le_one
      _ = 1 := one_mul 1
  -- Cross moments vanish: reflect `b i` to `-b i`.
  have hcross : ∀ i j, i ≠ j →
      ∫ u : UnitSphere K, ⟪b i, u.val⟫ * ⟪b j, u.val⟫ ∂unitSphereLaw K = 0 := by
    intro i j hij
    let R := (ℝ ∙ b i)ᗮ.reflection
    have hRi : R (b i) = -b i := Submodule.reflection_orthogonalComplement_singleton_eq_neg (b i)
    have hRj : R (b j) = b j := by
      refine Submodule.reflection_mem_subspace_eq_self ?_
      rw [Submodule.mem_orthogonal_singleton_iff_inner_right]
      exact b.orthonormal.2 hij
    have h : ∫ u : UnitSphere K,
          ⟪b i, (sphereMap R u).val⟫ * ⟪b j, (sphereMap R u).val⟫ ∂unitSphereLaw K
        = ∫ u : UnitSphere K, ⟪b i, u.val⟫ * ⟪b j, u.val⟫ ∂unitSphereLaw K :=
      integral_comp_sphereMap R (g := fun u => ⟪b i, u.val⟫ * ⟪b j, u.val⟫)
        ((continuous_coord (b i)).mul (continuous_coord (b j)))
    have hpt : ∀ u : UnitSphere K,
        ⟪b i, (sphereMap R u).val⟫ * ⟪b j, (sphereMap R u).val⟫
          = -(⟪b i, u.val⟫ * ⟪b j, u.val⟫) := by
      intro u
      show ⟪b i, R u.val⟫ * ⟪b j, R u.val⟫ = _
      rw [inner_reflection, inner_reflection, hRi, hRj, inner_neg_left]
      ring
    rw [integral_congr_ae (Filter.Eventually.of_forall hpt), integral_neg] at h
    linarith
  -- Diagonal moments are equal: exchange `b i` and `b j`.
  have hdiag : ∀ i j,
      ∫ u : UnitSphere K, ⟪b i, u.val⟫ * ⟪b i, u.val⟫ ∂unitSphereLaw K
        = ∫ u : UnitSphere K, ⟪b j, u.val⟫ * ⟪b j, u.val⟫ ∂unitSphereLaw K := by
    intro i j
    let T := (ℝ ∙ (b j - b i))ᗮ.reflection
    have hT : T (b j) = b i :=
      Submodule.reflection_sub (by rw [b.orthonormal.1 j, b.orthonormal.1 i])
    have h : ∫ u : UnitSphere K,
          ⟪b j, (sphereMap T u).val⟫ * ⟪b j, (sphereMap T u).val⟫ ∂unitSphereLaw K
        = ∫ u : UnitSphere K, ⟪b j, u.val⟫ * ⟪b j, u.val⟫ ∂unitSphereLaw K :=
      integral_comp_sphereMap T (g := fun u => ⟪b j, u.val⟫ * ⟪b j, u.val⟫)
        ((continuous_coord (b j)).mul (continuous_coord (b j)))
    have hpt : ∀ u : UnitSphere K,
        ⟪b j, (sphereMap T u).val⟫ * ⟪b j, (sphereMap T u).val⟫
          = ⟪b i, u.val⟫ * ⟪b i, u.val⟫ := by
      intro u
      show ⟪b j, T u.val⟫ * ⟪b j, T u.val⟫ = _
      rw [inner_reflection, hT]
    rw [integral_congr_ae (Filter.Eventually.of_forall hpt)] at h
    exact h
  -- The diagonal moments sum to one.
  have hsum : ∑ i, ∫ u : UnitSphere K, ⟪b i, u.val⟫ * ⟪b i, u.val⟫ ∂unitSphereLaw K = 1 := by
    rw [← integral_finsetSum _ fun i _ => hint i i]
    have hone : ∀ u : UnitSphere K, ∑ i, ⟪b i, u.val⟫ * ⟪b i, u.val⟫ = 1 := by
      intro u
      have hu : ‖u.val‖ = 1 := mem_sphere_zero_iff_norm.1 u.property
      have := b.sum_sq_inner_right u.val
      simp only [pow_two] at this
      rw [this, hu, mul_one]
    simp only [hone]
    simp
  refine ⟨hint, fun i j => ?_⟩
  by_cases hij : i = j
  · subst hij
    rw [if_pos rfl]
    have hall : ∑ _k : Fin (K - 1),
        ∫ u : UnitSphere K, ⟪b i, u.val⟫ * ⟪b i, u.val⟫ ∂unitSphereLaw K = 1 := by
      rw [← hsum]
      exact Finset.sum_congr rfl fun k _ => hdiag i k
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hall
    have hK1 : ((K - 1 : ℕ) : ℝ) ≠ 0 := by
      have : 0 < K - 1 := by omega
      exact_mod_cast this.ne'
    rw [eq_div_iff hK1, mul_comm]
    exact hall
  · rw [if_neg hij]
    exact hcross i j hij

/-! ### The dilution statements without the hypothesis -/

/-- **Dimensional dilution** (`lem:dimensional_dilution`), with nothing
assumed about the sphere: under local smoothness at `p` and a perturbation
uniform on the radius-`r` sphere of the tangent space,

`|E[(φ(p + ε) − φ(p))²] − r²/(K−1) ‖g‖²| ≤ 2‖g‖M r³ + M² r⁴`,

which is the lemma's `r²/(K−1) ‖g‖² + O(r³)` with an explicit constant. The
one hypothesis the text does not state is that `φ` is measurable, which is
what makes the expectation an integral. The label was on
`Diffusion.dimensional_dilution`, the same statement with the sphere moments
as a hypothesis; it moved here once they were proved. -/
@[thesis "lem:dimensional_dilution"]
theorem dimensional_dilution (hK : 2 ≤ K)
    {φ : EuclideanSpace ℝ (Fin K) → ℝ} (hφ : Measurable φ)
    {p : EuclideanSpace ℝ (Fin K)} {g : Tangent K} {M ρ r : ℝ}
    (hs : LocalSmoothness φ p g M ρ)
    {P : Measure (Tangent K)} (hd : Diffuse P r ρ) :
    |(∫ v, (φ (p + v.val) - φ p) ^ 2 ∂P) -
      r ^ 2 / (K - 1 : ℕ) * ‖g‖ ^ 2| ≤
      2 * ‖g‖ * M * r ^ 3 + M ^ 2 * r ^ 4 := by
  have : NeZero K := ⟨by omega⟩
  exact Diffusion.dimensional_dilution hK tangentBasis (sphereMoments hK tangentBasis)
    hφ hs hd

/-- **Integrated dilution** (`cor:integrated_dilution`).
For `K ≥ 2`, a measurable functional and posterior curve, a common positive
smoothness radius `ρ`, square-integrable gradients and remainder constants,
and `0 < r ≤ ρ`, the uniform tangent-sphere perturbation has the integrated
second-moment expansion with the explicit remainder displayed below. Sphere
moments are proved; there is no additional sphere-moment hypothesis. -/
@[thesis "cor:integrated_dilution"]
theorem integrated_dilution (hK : 2 ≤ K) {X : Type*} [MeasurableSpace X]
    {P : Measure X} [IsProbabilityMeasure P]
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
  have : NeZero K := ⟨by omega⟩
  exact Diffusion.integrated_dilution hK tangentBasis (sphereMoments hK tangentBasis)
    hφ hp hs hg2 hM2 hr hρ

end ThresholdOpt.SphereSymmetry
