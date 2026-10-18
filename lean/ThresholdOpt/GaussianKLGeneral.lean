import ThresholdOpt.GaussianKL

/-!
# The Gaussian formula for general covariance matrices (Appendix B)

`eq:kl_gaussian_general` in full: for Mathlib's multivariate Gaussian measures
on Euclidean space, with any means and any positive definite covariance
matrices,

`KL(N(μ₀, Σ₀) ‖ N(μ₁, Σ₁)) = ½ [tr(Σ₁⁻¹ Σ₀) − d + log(det Σ₁ / det Σ₀)
  + (μ₁ − μ₀)ᵀ Σ₁⁻¹ (μ₁ − μ₀)]`.

Nothing is assumed. The proof reduces to the one-dimensional formula of
[`GaussianKL.lean`](GaussianKL.lean).

* The image of a multivariate Gaussian under an affine map `x ↦ b + M x` is the
  multivariate Gaussian with mean `b + M μ` and covariance `M Σ Mᵀ`
  (`multivariateGaussian_map_affine`, by characteristic functions).
* Two positive definite matrices are simultaneously diagonalizable by
  congruence: there is `T` with `T Σ₁ Tᵀ = 1` and `T Σ₀ Tᵀ` diagonal with
  positive entries (`exists_simultaneous_diagonalization`, from the spectral
  theorem applied twice).
* The affine map `x ↦ T (x − μ₁)` has an affine inverse, so it preserves the
  divergence. It sends the two measures to a product of real Gaussians and to
  the standard Gaussian, whose divergence is a sum of one-dimensional terms.
* The sum is rewritten with `Tᵀ T = Σ₁⁻¹`: the diagonal entries sum to
  `tr(Σ₁⁻¹ Σ₀)`, their product is `det Σ₀ / det Σ₁`, and the squared norm of
  the transformed mean difference is the quadratic form.
-/

namespace ThresholdOpt.GaussianKL

open MeasureTheory ProbabilityTheory InformationTheory Real WithLp Matrix
open scoped ENNReal NNReal RealInnerProductSpace

variable {d : ℕ}

/-- Moving a matrix across the Euclidean inner product. -/
theorem inner_toEuclideanCLM_left (M : Matrix (Fin d) (Fin d) ℝ)
    (x t : EuclideanSpace ℝ (Fin d)) :
    ⟪toEuclideanCLM (𝕜 := ℝ) M x, t⟫ = ⟪x, toEuclideanCLM (𝕜 := ℝ) Mᵀ t⟫ := by
  rw [real_inner_comm, inner_toEuclideanCLM, inner_toEuclideanCLM, Matrix.mulVec_transpose,
    Matrix.dotProduct_mulVec, dotProduct_comm]

/-- The image of a multivariate Gaussian under an affine map is the
multivariate Gaussian with the transformed mean and covariance. -/
theorem multivariateGaussian_map_affine (μ b : EuclideanSpace ℝ (Fin d))
    {S : Matrix (Fin d) (Fin d) ℝ} (hS : S.PosSemidef) (M : Matrix (Fin d) (Fin d) ℝ) :
    (multivariateGaussian μ S).map (fun x => b + toEuclideanCLM (𝕜 := ℝ) M x)
      = multivariateGaussian (b + toEuclideanCLM (𝕜 := ℝ) M μ) (M * S * Mᵀ) := by
  have hS' : (M * S * Mᵀ).PosSemidef := by
    have := hS.mul_mul_conjTranspose_same M
    rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at this
  apply Measure.ext_of_charFun
  ext t
  have hmap : (multivariateGaussian μ S).map (fun x => b + toEuclideanCLM (𝕜 := ℝ) M x)
      = ((multivariateGaussian μ S).map (toEuclideanCLM (𝕜 := ℝ) M)).map (fun y => b + y) := by
    rw [Measure.map_map (by fun_prop) (by fun_prop)]
    rfl
  have hlin : charFun ((multivariateGaussian μ S).map (toEuclideanCLM (𝕜 := ℝ) M)) t
      = charFun (multivariateGaussian μ S) (toEuclideanCLM (𝕜 := ℝ) Mᵀ t) := by
    rw [charFun_apply, charFun_apply, integral_map (by fun_prop) (by fun_prop)]
    refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
    simp only [inner_toEuclideanCLM_left]
  rw [hmap, charFun_map_const_add, hlin, charFun_multivariateGaussian hS,
    charFun_multivariateGaussian hS', ← Complex.exp_add]
  congr 1
  have h1 : ⟪t, b + toEuclideanCLM (𝕜 := ℝ) M μ⟫
      = ⟪toEuclideanCLM (𝕜 := ℝ) Mᵀ t, μ⟫ + ⟪b, t⟫ := by
    have h := inner_toEuclideanCLM_left M μ t
    rw [inner_add_right, real_inner_comm b t, real_inner_comm (toEuclideanCLM (𝕜 := ℝ) M μ) t,
      h, real_inner_comm μ]
    ring
  have h2 : (toEuclideanCLM (𝕜 := ℝ) Mᵀ t).ofLp ⬝ᵥ S *ᵥ (toEuclideanCLM (𝕜 := ℝ) Mᵀ t).ofLp
      = t.ofLp ⬝ᵥ (M * S * Mᵀ) *ᵥ t.ofLp := by
    rw [ofLp_toEuclideanCLM, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec,
      Matrix.dotProduct_mulVec t.ofLp M, ← Matrix.mulVec_transpose]
  rw [h2, h1]
  push_cast
  ring

/-- Independent real Gaussians with arbitrary variances form the multivariate
Gaussian with diagonal covariance. -/
theorem pi_gaussianReal_map_toLp (m : Fin d → ℝ) (v : Fin d → ℝ≥0) :
    (Measure.pi fun i => gaussianReal (m i) (v i)).map (toLp 2)
      = multivariateGaussian (toLp 2 m) (Matrix.diagonal fun i => ((v i : ℝ≥0) : ℝ)) := by
  have hS : (Matrix.diagonal fun i => ((v i : ℝ≥0) : ℝ)).PosSemidef :=
    Matrix.posSemidef_diagonal_iff.2 fun i => (v i).coe_nonneg
  apply Measure.ext_of_charFun (E := EuclideanSpace ℝ (Fin d))
  ext t
  rw [charFun_pi, charFun_multivariateGaussian hS]
  simp_rw [charFun_gaussianReal, ← Complex.exp_sum]
  congr 1
  simp [Matrix.mulVec_diagonal, dotProduct, PiLp.inner_apply, Finset.sum_sub_distrib,
    Finset.sum_mul, Finset.sum_div]
  congr 1 <;> exact Finset.sum_congr rfl fun i _ => by ring

/-- The divergence is unchanged by a measurable map that has a measurable left
inverse. -/
theorem klDiv_map_of_leftInverse {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (μ ν : Measure α) [IsFiniteMeasure μ] [IsFiniteMeasure ν] {f : α → β} {g : β → α}
    (hf : Measurable f) (hg : Measurable g) (hgf : ∀ x, g (f x) = x) :
    klDiv (μ.map f) (ν.map f) = klDiv μ ν := by
  refine le_antisymm (klDiv_map_le μ ν hf) ?_
  have h := klDiv_map_le (μ.map f) (ν.map f) hg
  have hid : g ∘ f = id := funext hgf
  rwa [Measure.map_map hg hf, Measure.map_map hg hf, hid, Measure.map_id, Measure.map_id] at h

/-- The closed form of the divergence between two real Gaussians is
nonnegative. -/
theorem gaussianKL_term_nonneg (m₀ m₁ : ℝ) {v₀ v₁ : ℝ≥0} (h₀ : v₀ ≠ 0) (h₁ : v₁ ≠ 0) :
    0 ≤ 1 / 2 * ((v₀ : ℝ) / v₁ - 1 + Real.log (v₁ / v₀) + (m₁ - m₀) ^ 2 / v₁) := by
  by_contra hneg
  have h := klDiv_gaussianReal m₀ m₁ h₀ h₁
  rw [ENNReal.ofReal_of_nonpos (not_le.1 hneg).le] at h
  obtain ⟨hm, hv⟩ := (gaussianReal_ext_iff).1 ((klDiv_eq_zero_iff).1 h)
  apply hneg
  rw [hm, hv]
  simp [div_self (NNReal.coe_ne_zero.2 h₁)]

/-- Divergence between two products of real Gaussians, with coordinatewise
means and variances. -/
theorem klDiv_pi_gaussianReal (m₀ m₁ : Fin d → ℝ) (v₀ v₁ : Fin d → ℝ≥0)
    (h₀ : ∀ i, v₀ i ≠ 0) (h₁ : ∀ i, v₁ i ≠ 0) :
    klDiv (Measure.pi fun i => gaussianReal (m₀ i) (v₀ i))
        (Measure.pi fun i => gaussianReal (m₁ i) (v₁ i))
      = ENNReal.ofReal (∑ i, 1 / 2 * (((v₀ i : ℝ≥0) : ℝ) / v₁ i - 1
          + Real.log (v₁ i / v₀ i) + (m₁ i - m₀ i) ^ 2 / v₁ i)) := by
  rw [klDiv_pi, ENNReal.ofReal_sum_of_nonneg fun i _ =>
    gaussianKL_term_nonneg (m₀ i) (m₁ i) (h₀ i) (h₁ i)]
  exact Finset.sum_congr rfl fun i _ => klDiv_gaussianReal (m₀ i) (m₁ i) (h₀ i) (h₁ i)

/-- A real symmetric matrix is `U Λ Uᵀ` with `Uᵀ U = U Uᵀ = 1` and `Λ` the
diagonal matrix of its eigenvalues. -/
theorem exists_orthogonal_diagonalization {A : Matrix (Fin d) (Fin d) ℝ} (hA : A.IsHermitian) :
    ∃ (U : Matrix (Fin d) (Fin d) ℝ) (ev : Fin d → ℝ), Uᵀ * U = 1 ∧ U * Uᵀ = 1
      ∧ A = U * Matrix.diagonal ev * Uᵀ ∧ ev = hA.eigenvalues := by
  refine ⟨(hA.eigenvectorUnitary : Matrix (Fin d) (Fin d) ℝ), hA.eigenvalues, ?_, ?_, ?_, rfl⟩
  · have := Unitary.coe_star_mul_self hA.eigenvectorUnitary
    rwa [Matrix.star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial] at this
  · have := Unitary.coe_mul_star_self hA.eigenvectorUnitary
    rwa [Unitary.coe_star, Matrix.star_eq_conjTranspose,
      Matrix.conjTranspose_eq_transpose_of_trivial] at this
  · have := hA.spectral_theorem
    simpa [Unitary.conjStarAlgAut_apply, Matrix.star_eq_conjTranspose,
      Matrix.conjTranspose_eq_transpose_of_trivial] using this

/-- **Simultaneous diagonalization.** For two positive definite matrices there
is a matrix `T` with `T S₁ Tᵀ = 1` and `T S₀ Tᵀ` diagonal with positive
entries. -/
theorem exists_simultaneous_diagonalization {S₀ S₁ : Matrix (Fin d) (Fin d) ℝ}
    (h₀ : S₀.PosDef) (h₁ : S₁.PosDef) :
    ∃ (T : Matrix (Fin d) (Fin d) ℝ) (dv : Fin d → ℝ), (∀ i, 0 < dv i)
      ∧ T * S₁ * Tᵀ = 1 ∧ T * S₀ * Tᵀ = Matrix.diagonal dv := by
  -- Whiten `S₁`.
  obtain ⟨U, lam, hUtU, -, hS₁, hlam_eq⟩ := exists_orthogonal_diagonalization h₁.1
  have hlam : ∀ i, 0 < lam i := fun i => by rw [hlam_eq]; exact h₁.eigenvalues_pos i
  let a : Fin d → ℝ := fun i => (Real.sqrt (lam i))⁻¹
  let T₁ : Matrix (Fin d) (Fin d) ℝ := Matrix.diagonal a * Uᵀ
  have hT₁t : T₁ᵀ = U * Matrix.diagonal a := by
    simp only [T₁, Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.diagonal_transpose]
  have hUtU' : ∀ X : Matrix (Fin d) (Fin d) ℝ, Uᵀ * (U * X) = X := fun X => by
    rw [← Matrix.mul_assoc, hUtU, Matrix.one_mul]
  have hwhite : T₁ * S₁ * T₁ᵀ = 1 := by
    rw [hT₁t, hS₁]
    simp only [T₁, Matrix.mul_assoc, hUtU']
    rw [Matrix.diagonal_mul_diagonal, Matrix.diagonal_mul_diagonal, ← Matrix.diagonal_one]
    congr 1
    funext i
    have hs : 0 < Real.sqrt (lam i) := Real.sqrt_pos.2 (hlam i)
    have hsq : Real.sqrt (lam i) * Real.sqrt (lam i) = lam i := Real.mul_self_sqrt (hlam i).le
    simp only [a]
    field_simp
    linarith
  -- Diagonalize the whitened `S₀`.
  have hS' : (T₁ * S₀ * T₁ᵀ).PosSemidef := by
    have := h₀.posSemidef.mul_mul_conjTranspose_same T₁
    rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at this
  obtain ⟨V, dv, hVtV, -, hD, hdv_eq⟩ := exists_orthogonal_diagonalization hS'.1
  have hdv0 : ∀ i, 0 ≤ dv i := fun i => by rw [hdv_eq]; exact hS'.eigenvalues_nonneg i
  have hVtV' : ∀ X : Matrix (Fin d) (Fin d) ℝ, Vᵀ * (V * X) = X := fun X => by
    rw [← Matrix.mul_assoc, hVtV, Matrix.one_mul]
  have hTS₁ : Vᵀ * T₁ * S₁ * (Vᵀ * T₁)ᵀ = 1 := by
    calc Vᵀ * T₁ * S₁ * (Vᵀ * T₁)ᵀ = Vᵀ * (T₁ * S₁ * T₁ᵀ) * V := by
          simp only [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
      _ = 1 := by rw [hwhite, Matrix.mul_one, hVtV]
  have hTS₀ : Vᵀ * T₁ * S₀ * (Vᵀ * T₁)ᵀ = Matrix.diagonal dv := by
    calc Vᵀ * T₁ * S₀ * (Vᵀ * T₁)ᵀ = Vᵀ * (T₁ * S₀ * T₁ᵀ) * V := by
          simp only [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
      _ = Matrix.diagonal dv := by
          rw [hD]
          simp only [Matrix.mul_assoc, hVtV', hVtV, Matrix.mul_one]
  refine ⟨Vᵀ * T₁, dv, fun i => ?_, hTS₁, hTS₀⟩
  -- The diagonal entries are nonnegative with nonzero product.
  have hdetT : (Vᵀ * T₁).det ≠ 0 := by
    intro hzero
    have := congrArg Matrix.det hTS₁
    rw [Matrix.det_mul, Matrix.det_mul, hzero] at this
    simp at this
  have hprod : ∏ j, dv j = (Vᵀ * T₁).det * S₀.det * (Vᵀ * T₁).det := by
    have := congrArg Matrix.det hTS₀
    rw [Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose, Matrix.det_diagonal] at this
    exact this.symm
  have hne : ∏ j, dv j ≠ 0 := by
    rw [hprod]
    exact mul_ne_zero (mul_ne_zero hdetT h₀.det_pos.ne') hdetT
  exact lt_of_le_of_ne (hdv0 i) (Ne.symm (Finset.prod_ne_zero_iff.1 hne i (Finset.mem_univ i)))

/-- **The Gaussian formula** (`eq:kl_gaussian_general`), for Mathlib's
multivariate Gaussian measures and arbitrary positive definite covariance
matrices:

`KL(N(μ₀, Σ₀) ‖ N(μ₁, Σ₁)) = ½ [tr(Σ₁⁻¹ Σ₀) − d + log(det Σ₁ / det Σ₀)
  + (μ₁ − μ₀)ᵀ Σ₁⁻¹ (μ₁ − μ₀)]`.

The text states the formula as known and does not prove it. Positive
definiteness of both matrices is the hypothesis under which the formula is
meaningful (`Σ₁⁻¹` and `log det`). -/
@[thesis "eq:kl_gaussian_general"]
theorem klDiv_multivariateGaussian (μ₀ μ₁ : EuclideanSpace ℝ (Fin d))
    {S₀ S₁ : Matrix (Fin d) (Fin d) ℝ} (h₀ : S₀.PosDef) (h₁ : S₁.PosDef) :
    klDiv (multivariateGaussian μ₀ S₀) (multivariateGaussian μ₁ S₁)
      = ENNReal.ofReal (1 / 2 * (trace (S₁⁻¹ * S₀) - d + Real.log (S₁.det / S₀.det)
          + (ofLp μ₁ - ofLp μ₀) ⬝ᵥ (S₁⁻¹ *ᵥ (ofLp μ₁ - ofLp μ₀)))) := by
  obtain ⟨T, dv, hdv, hT₁, hT₀⟩ := exists_simultaneous_diagonalization h₀ h₁
  -- `T` is invertible.
  have hdet₁ : T.det * S₁.det * T.det = 1 := by
    have := congrArg Matrix.det hT₁
    rwa [Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
  have hdetT : T.det ≠ 0 := by
    intro hzero
    rw [hzero] at hdet₁
    simp at hdet₁
  have hRT : T⁻¹ * T = 1 := Matrix.nonsing_inv_mul T (isUnit_iff_ne_zero.2 hdetT)
  -- The affine change of variables and its left inverse.
  let e : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d) :=
    fun x => -(toEuclideanCLM (𝕜 := ℝ) T μ₁) + toEuclideanCLM (𝕜 := ℝ) T x
  let g : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d) :=
    fun y => μ₁ + toEuclideanCLM (𝕜 := ℝ) T⁻¹ y
  have hRTapp : ∀ z : EuclideanSpace ℝ (Fin d),
      toEuclideanCLM (𝕜 := ℝ) T⁻¹ (toEuclideanCLM (𝕜 := ℝ) T z) = z := by
    intro z
    apply (WithLp.ofLp_injective 2)
    rw [ofLp_toEuclideanCLM, ofLp_toEuclideanCLM, Matrix.mulVec_mulVec, hRT, Matrix.one_mulVec]
  have hge : ∀ x, g (e x) = x := by
    intro x
    simp only [g, e, map_add, map_neg, hRTapp]
    abel
  have he : Measurable e := by fun_prop
  have hg : Measurable g := by fun_prop
  -- The two images.
  have hN₁ : (multivariateGaussian μ₁ S₁).map e = multivariateGaussian 0 1 := by
    have := multivariateGaussian_map_affine μ₁ (-(toEuclideanCLM (𝕜 := ℝ) T μ₁))
      h₁.posSemidef T
    rw [this, hT₁, neg_add_cancel]
  have hN₀ : (multivariateGaussian μ₀ S₀).map e
      = multivariateGaussian (-(toEuclideanCLM (𝕜 := ℝ) T μ₁) + toEuclideanCLM (𝕜 := ℝ) T μ₀)
          (Matrix.diagonal dv) := by
    have := multivariateGaussian_map_affine μ₀ (-(toEuclideanCLM (𝕜 := ℝ) T μ₁))
      h₀.posSemidef T
    rw [this, hT₀]
  -- Both images are products of real Gaussians.
  let mvec : Fin d → ℝ := T *ᵥ (ofLp μ₀ - ofLp μ₁)
  let dn : Fin d → ℝ≥0 := fun i => ⟨dv i, (hdv i).le⟩
  have hmvec : -(toEuclideanCLM (𝕜 := ℝ) T μ₁) + toEuclideanCLM (𝕜 := ℝ) T μ₀ = toLp 2 mvec := by
    apply (WithLp.ofLp_injective 2)
    simp only [mvec, WithLp.ofLp_add, WithLp.ofLp_neg, ofLp_toEuclideanCLM, Matrix.mulVec_sub]
    abel
  have hD₀ := pi_gaussianReal_map_toLp mvec dn
  have hD₁ := pi_gaussianReal_map_toLp (d := d) (fun _ => (0 : ℝ)) (fun _ => (1 : ℝ≥0))
  have hdiag : (Matrix.diagonal fun i => ((dn i : ℝ≥0) : ℝ)) = Matrix.diagonal dv := rfl
  have hone : (Matrix.diagonal fun _ : Fin d => (((1 : ℝ≥0) : ℝ≥0) : ℝ))
      = (1 : Matrix (Fin d) (Fin d) ℝ) := by
    simp
  have hzero : toLp 2 (fun _ : Fin d => (0 : ℝ)) = (0 : EuclideanSpace ℝ (Fin d)) := rfl
  rw [hdiag] at hD₀
  rw [hone, hzero] at hD₁
  rw [← klDiv_map_of_leftInverse _ _ he hg hge, hN₀, hN₁, hmvec, ← hD₀, ← hD₁]
  have hkl := klDiv_map_equiv (MeasurableEquiv.toLp 2 (Fin d → ℝ))
    (Measure.pi fun i => gaussianReal (mvec i) (dn i))
    (Measure.pi fun _ : Fin d => gaussianReal 0 1)
  rw [show ((MeasurableEquiv.toLp 2 (Fin d → ℝ)) : (Fin d → ℝ) → EuclideanSpace ℝ (Fin d))
      = toLp 2 from rfl] at hkl
  rw [hkl, klDiv_pi_gaussianReal mvec (fun _ => 0) dn (fun _ => 1)
    (fun i => NNReal.coe_ne_zero.1 (show ((dn i : ℝ≥0) : ℝ) ≠ 0 from (hdv i).ne'))
    (fun _ => one_ne_zero)]
  congr 1
  -- What remains is linear algebra: trace, determinant and quadratic form.
  have hinv : S₁⁻¹ = Tᵀ * T := by
    have h1 : T * (S₁ * Tᵀ) = 1 := by rw [← Matrix.mul_assoc]; exact hT₁
    have h3 : S₁ * Tᵀ = T⁻¹ := by
      calc S₁ * Tᵀ = T⁻¹ * (T * (S₁ * Tᵀ)) := by rw [← Matrix.mul_assoc, hRT, Matrix.one_mul]
        _ = T⁻¹ := by rw [h1, Matrix.mul_one]
    have h2 : S₁ * Tᵀ * T = 1 := by rw [h3, hRT]
    exact Matrix.inv_eq_right_inv (by rw [← Matrix.mul_assoc]; exact h2)
  have htr : trace (S₁⁻¹ * S₀) = ∑ i, dv i := by
    rw [hinv, ← Matrix.trace_mul_cycle T S₀ Tᵀ, hT₀, Matrix.trace_diagonal]
  have hprod : ∏ i, dv i = S₀.det / S₁.det := by
    have h := congrArg Matrix.det hT₀
    rw [Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose, Matrix.det_diagonal] at h
    rw [eq_div_iff h₁.det_pos.ne', ← h]
    linear_combination S₀.det * hdet₁
  have hlog : ∑ i, Real.log (1 / dv i) = Real.log (S₁.det / S₀.det) := by
    simp only [one_div, Real.log_inv, Finset.sum_neg_distrib]
    rw [← Real.log_prod fun i _ => (hdv i).ne', hprod,
      Real.log_div h₀.det_pos.ne' h₁.det_pos.ne', Real.log_div h₁.det_pos.ne' h₀.det_pos.ne']
    ring
  have hquad : (ofLp μ₁ - ofLp μ₀) ⬝ᵥ (S₁⁻¹ *ᵥ (ofLp μ₁ - ofLp μ₀)) = ∑ i, mvec i ^ 2 := by
    have hw : ofLp μ₁ - ofLp μ₀ = -(ofLp μ₀ - ofLp μ₁) := by abel
    rw [hw, Matrix.mulVec_neg, neg_dotProduct, dotProduct_neg, neg_neg, hinv,
      ← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, Matrix.vecMul_transpose]
    simp only [dotProduct, mvec, pow_two]
  have hLHS : ∀ i, 1 / 2 * (((dn i : ℝ≥0) : ℝ) / ((1 : ℝ≥0) : ℝ) - 1
        + Real.log (((1 : ℝ≥0) : ℝ) / ((dn i : ℝ≥0) : ℝ)) + (0 - mvec i) ^ 2 / ((1 : ℝ≥0) : ℝ))
      = 1 / 2 * (dv i - 1 + Real.log (1 / dv i) + mvec i ^ 2) := by
    intro i
    have hdn : ((dn i : ℝ≥0) : ℝ) = dv i := rfl
    rw [hdn, NNReal.coe_one, div_one, div_one, zero_sub, neg_sq]
  rw [Finset.sum_congr rfl fun i _ => hLHS i, ← Finset.mul_sum, Finset.sum_add_distrib,
    Finset.sum_add_distrib, Finset.sum_sub_distrib, htr, hlog, hquad]
  simp

end ThresholdOpt.GaussianKL
