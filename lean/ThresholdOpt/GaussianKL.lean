import Mathlib.InformationTheory.KullbackLeibler.ChainRule
import Mathlib.InformationTheory.KullbackLeibler.DataProcessing
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Moments.Variance
import Mathlib.Tactic
import ThresholdOpt.Attr

/-!
# K-invariance of the misspecification budget (Appendix B)

The Kullback-Leibler computations of Appendix B, for actual probability
measures and Mathlib's `InformationTheory.klDiv`.

Nothing is assumed. In particular the Gaussian formula `eq:kl_gaussian_general`
is not taken as a hypothesis: the appendix only ever applies it to covariance
matrices that are multiples of the identity, and for those it is proved here
from the definition of the divergence. The formula for general covariance
matrices is proved in [`GaussianKLGeneral.lean`](GaussianKLGeneral.lean).

* `klDiv_gaussianReal`: the divergence between two real Gaussians.
* `klDiv_prod`, `klDiv_pi`: the divergence is additive over independent
  coordinates. With the previous item this gives `klDiv_isoGaussian`, the
  divergence between `N(m₀, v₀ I)` and `N(m₁, v₁ I)` in any dimension.
* `klDiv_jointLaw`: for two joint laws of a label and a feature vector that
  share the label marginal, the divergence is the prior-weighted average of the
  per-class divergences. This is the chain rule in the form the appendix uses.

The isotropic Gaussian `N(m, v I)` on `ℝ^d` is the product measure
`isoGaussian m v` of `d` real Gaussians. `isoGaussian_map_toLp` identifies it
with Mathlib's `multivariateGaussian m (v • 1)`.
-/

namespace ThresholdOpt.GaussianKL

open MeasureTheory ProbabilityTheory InformationTheory Real
open scoped ENNReal NNReal

/-! ### Two real Gaussians -/

/-- The logarithm of the Gaussian density. -/
theorem log_gaussianPDFReal (m : ℝ) {v : ℝ≥0} (hv : v ≠ 0) (x : ℝ) :
    Real.log (gaussianPDFReal m v x)
      = -(Real.log (2 * π * v)) / 2 - (x - m) ^ 2 / (2 * v) := by
  have hv' : (0 : ℝ) < v := NNReal.coe_pos.2 (pos_iff_ne_zero.2 hv)
  have hpos : 0 < 2 * π * (v : ℝ) := by positivity
  rw [gaussianPDFReal_def]
  simp only
  rw [Real.log_mul (inv_ne_zero (Real.sqrt_ne_zero'.2 hpos)) (Real.exp_ne_zero _),
    Real.log_inv, Real.log_sqrt hpos.le, Real.log_exp]
  ring

/-- The log-likelihood ratio of `N(m₀, v₀)` with respect to `N(m₁, v₁)`. -/
noncomputable def llrGauss (m₀ m₁ : ℝ) (v₀ v₁ : ℝ≥0) (x : ℝ) : ℝ :=
  1 / 2 * Real.log (v₁ / v₀) - (x - m₀) ^ 2 / (2 * v₀) + (x - m₁) ^ 2 / (2 * v₁)

/-- One nondegenerate real Gaussian has a density with respect to another: the
ratio of the two Lebesgue densities. -/
theorem gaussianReal_eq_withDensity (m₀ m₁ : ℝ) {v₀ v₁ : ℝ≥0} (h₀ : v₀ ≠ 0)
    (h₁ : v₁ ≠ 0) :
    gaussianReal m₀ v₀ = (gaussianReal m₁ v₁).withDensity
      (fun x => gaussianPDF m₀ v₀ x / gaussianPDF m₁ v₁ x) := by
  have hmeas : Measurable fun x => gaussianPDF m₀ v₀ x / gaussianPDF m₁ v₁ x :=
    (measurable_gaussianPDF m₀ v₀).div (measurable_gaussianPDF m₁ v₁)
  rw [gaussianReal_of_var_ne_zero m₀ h₀, gaussianReal_of_var_ne_zero m₁ h₁,
    ← withDensity_mul _ (measurable_gaussianPDF m₁ v₁) hmeas]
  congr 1
  funext x
  simp only [Pi.mul_apply]
  rw [ENNReal.mul_div_cancel (gaussianPDF_pos m₁ h₁ x).ne' gaussianPDF_ne_top]

theorem gaussianReal_ac (m₀ m₁ : ℝ) {v₀ v₁ : ℝ≥0} (h₀ : v₀ ≠ 0) (h₁ : v₁ ≠ 0) :
    gaussianReal m₀ v₀ ≪ gaussianReal m₁ v₁ :=
  (gaussianReal_absolutelyContinuous m₀ h₀).trans
    (gaussianReal_absolutelyContinuous' m₁ h₁)

/-- The log-likelihood ratio of two nondegenerate real Gaussians, almost
everywhere. -/
theorem llr_gaussianReal (m₀ m₁ : ℝ) {v₀ v₁ : ℝ≥0} (h₀ : v₀ ≠ 0) (h₁ : v₁ ≠ 0) :
    llr (gaussianReal m₀ v₀) (gaussianReal m₁ v₁)
      =ᵐ[gaussianReal m₀ v₀] llrGauss m₀ m₁ v₀ v₁ := by
  have hmeas : Measurable fun x => gaussianPDF m₀ v₀ x / gaussianPDF m₁ v₁ x :=
    (measurable_gaussianPDF m₀ v₀).div (measurable_gaussianPDF m₁ v₁)
  have hrn : (gaussianReal m₀ v₀).rnDeriv (gaussianReal m₁ v₁)
      =ᵐ[gaussianReal m₁ v₁] fun x => gaussianPDF m₀ v₀ x / gaussianPDF m₁ v₁ x := by
    have := Measure.rnDeriv_withDensity (gaussianReal m₁ v₁) hmeas
    rwa [← gaussianReal_eq_withDensity m₀ m₁ h₀ h₁] at this
  have hv₀ : (0 : ℝ) < v₀ := NNReal.coe_pos.2 (pos_iff_ne_zero.2 h₀)
  have hv₁ : (0 : ℝ) < v₁ := NNReal.coe_pos.2 (pos_iff_ne_zero.2 h₁)
  have hπ : (0 : ℝ) < 2 * π := by positivity
  filter_upwards [(gaussianReal_ac m₀ m₁ h₀ h₁).ae_le hrn] with x hx
  have hp₀ := gaussianPDFReal_pos m₀ v₀ x h₀
  have hp₁ := gaussianPDFReal_pos m₁ v₁ x h₁
  simp only [llr, hx, ENNReal.toReal_div, toReal_gaussianPDF]
  rw [Real.log_div hp₀.ne' hp₁.ne', log_gaussianPDFReal m₀ h₀, log_gaussianPDFReal m₁ h₁,
    Real.log_mul hπ.ne' hv₀.ne', Real.log_mul hπ.ne' hv₁.ne', llrGauss,
    Real.log_div hv₁.ne' hv₀.ne']
  ring

theorem integrable_llrGauss (m₀ m₁ : ℝ) (v₀ v₁ : ℝ≥0) :
    Integrable (llrGauss m₀ m₁ v₀ v₁) (gaussianReal m₀ v₀) := by
  have h1 : Integrable (fun x : ℝ => x) (gaussianReal m₀ v₀) :=
    (memLp_id_gaussianReal 1).integrable (by simp)
  have h2 : Integrable (fun x : ℝ => x ^ 2) (gaussianReal m₀ v₀) :=
    (memLp_id_gaussianReal (μ := m₀) (v := v₀) 2).integrable_sq
  have heq : llrGauss m₀ m₁ v₀ v₁ = fun x =>
      (1 / 2 * Real.log (v₁ / v₀) - m₀ ^ 2 / (2 * v₀) + m₁ ^ 2 / (2 * v₁))
        + (m₀ / v₀ - m₁ / v₁) * x + (1 / (2 * v₁) - 1 / (2 * v₀)) * x ^ 2 := by
    funext x
    simp only [llrGauss]
    ring
  rw [heq]
  exact ((integrable_const _).add (h1.const_mul _)).add (h2.const_mul _)

/-- **Divergence between two real Gaussians**, the one-dimensional case of
`eq:kl_gaussian_general`:

`KL(N(m₀, v₀) ‖ N(m₁, v₁)) = ½ (v₀/v₁ − 1 + log(v₁/v₀) + (m₁ − m₀)²/v₁)`.

Proved from the definition of the divergence, the two densities, and the mean
and variance of a Gaussian. -/
theorem klDiv_gaussianReal (m₀ m₁ : ℝ) {v₀ v₁ : ℝ≥0} (h₀ : v₀ ≠ 0) (h₁ : v₁ ≠ 0) :
    klDiv (gaussianReal m₀ v₀) (gaussianReal m₁ v₁)
      = ENNReal.ofReal (1 / 2 * ((v₀ : ℝ) / v₁ - 1 + Real.log (v₁ / v₀)
          + (m₁ - m₀) ^ 2 / v₁)) := by
  have hv₀ : (0 : ℝ) < v₀ := NNReal.coe_pos.2 (pos_iff_ne_zero.2 h₀)
  have hv₁ : (0 : ℝ) < v₁ := NNReal.coe_pos.2 (pos_iff_ne_zero.2 h₁)
  have hllr := llr_gaussianReal m₀ m₁ h₀ h₁
  have hint : Integrable (llr (gaussianReal m₀ v₀) (gaussianReal m₁ v₁))
      (gaussianReal m₀ v₀) := (integrable_llrGauss m₀ m₁ v₀ v₁).congr hllr.symm
  rw [klDiv_of_ac_of_integrable (gaussianReal_ac m₀ m₁ h₀ h₁) hint,
    integral_congr_ae hllr]
  congr 1
  -- Moments of `N(m₀, v₀)`.
  have h1 : Integrable (fun x : ℝ => x) (gaussianReal m₀ v₀) :=
    (memLp_id_gaussianReal 1).integrable (by simp)
  have h2 : Integrable (fun x : ℝ => x ^ 2) (gaussianReal m₀ v₀) :=
    (memLp_id_gaussianReal (μ := m₀) (v := v₀) 2).integrable_sq
  have hc : Integrable (fun x : ℝ => (x - m₀) ^ 2) (gaussianReal m₀ v₀) := by
    have : (fun x : ℝ => (x - m₀) ^ 2) = fun x => x ^ 2 - 2 * m₀ * x + m₀ ^ 2 := by
      funext x; ring
    rw [this]
    exact (h2.sub (h1.const_mul _)).add (integrable_const _)
  have hd : Integrable (fun x : ℝ => x - m₀) (gaussianReal m₀ v₀) :=
    h1.sub (integrable_const _)
  have hmean : ∫ x, (x - m₀) ∂gaussianReal m₀ v₀ = 0 := by
    rw [integral_sub h1 (integrable_const _), integral_id_gaussianReal]
    simp
  have hvar : ∫ x, (x - m₀) ^ 2 ∂gaussianReal m₀ v₀ = v₀ := by
    have := variance_fun_id_gaussianReal (μ := m₀) (v := v₀)
    rw [variance_eq_integral measurable_id'.aemeasurable] at this
    simpa only [integral_id_gaussianReal] using this
  have hval : ∫ x, llrGauss m₀ m₁ v₀ v₁ x ∂gaussianReal m₀ v₀
      = (1 / 2 * Real.log (v₁ / v₀) + (m₀ - m₁) ^ 2 / (2 * v₁))
        + (1 / (2 * v₁) - 1 / (2 * v₀)) * v₀ := by
    have e : ∀ x, llrGauss m₀ m₁ v₀ v₁ x
        = (1 / 2 * Real.log (v₁ / v₀) + (m₀ - m₁) ^ 2 / (2 * v₁))
          + ((1 / (2 * v₁) - 1 / (2 * v₀)) * (x - m₀) ^ 2
            + (m₀ - m₁) / v₁ * (x - m₀)) := by
      intro x
      simp only [llrGauss]
      field_simp
      ring
    have hcd : Integrable (fun x : ℝ => (1 / (2 * (v₁ : ℝ)) - 1 / (2 * v₀)) * (x - m₀) ^ 2
        + (m₀ - m₁) / v₁ * (x - m₀)) (gaussianReal m₀ v₀) :=
      (hc.const_mul _).add (hd.const_mul _)
    simp only [e]
    rw [integral_add (integrable_const _) hcd,
      integral_add (hc.const_mul _) (hd.const_mul _), integral_const_mul,
      integral_const_mul, hmean, hvar]
    simp
  have hμ : (gaussianReal m₀ v₀).real Set.univ = 1 := by simp
  have hν : (gaussianReal m₁ v₁).real Set.univ = 1 := by simp
  rw [hval, hμ, hν]
  field_simp
  ring

/-! ### Additivity over independent coordinates -/

section Additivity

variable {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]

/-- The divergence does not change under a measurable bijection: the data
processing inequality, applied in both directions. -/
theorem klDiv_map_equiv (e : α ≃ᵐ β) (μ ν : Measure α) [IsFiniteMeasure μ]
    [IsFiniteMeasure ν] : klDiv (μ.map e) (ν.map e) = klDiv μ ν := by
  refine le_antisymm (klDiv_map_le μ ν e.measurable) ?_
  have h := klDiv_map_le (μ.map e) (ν.map e) e.symm.measurable
  rwa [e.map_symm_map, e.map_symm_map] at h

/-- **The divergence of a product is the sum of the divergences.** From
Mathlib's chain rule: the first factor contributes `klDiv μ₁ ν₁`, and with the
first factor shared the second contributes `klDiv μ₂ ν₂`. -/
theorem klDiv_prod (μ₁ ν₁ : Measure α) (μ₂ ν₂ : Measure β)
    [IsProbabilityMeasure μ₁] [IsProbabilityMeasure ν₁]
    [IsProbabilityMeasure μ₂] [IsProbabilityMeasure ν₂] :
    klDiv (μ₁.prod μ₂) (ν₁.prod ν₂) = klDiv μ₁ ν₁ + klDiv μ₂ ν₂ := by
  have hright : klDiv (μ₁.prod μ₂) (μ₁.prod ν₂) = klDiv μ₂ ν₂ := by
    rw [← Measure.prod_swap (μ := μ₂) (ν := μ₁), ← Measure.prod_swap (μ := ν₂) (ν := μ₁)]
    have := klDiv_map_equiv (MeasurableEquiv.prodComm (α := β) (β := α))
      (μ₂.prod μ₁) (ν₂.prod μ₁)
    rw [show ((MeasurableEquiv.prodComm (α := β) (β := α)) : β × α → α × β) = Prod.swap
      from rfl] at this
    rw [this, ← Measure.compProd_const, ← Measure.compProd_const, klDiv_compProd_left]
  rw [← Measure.compProd_const, ← Measure.compProd_const, klDiv_compProd_eq_add,
    Measure.compProd_const, Measure.compProd_const, hright]

end Additivity

/-- **The divergence of a product of `n` real measures is the sum of the
coordinate divergences.** -/
theorem klDiv_pi {n : ℕ} (μ ν : Fin n → Measure ℝ)
    [∀ i, IsProbabilityMeasure (μ i)] [∀ i, IsProbabilityMeasure (ν i)] :
    klDiv (Measure.pi μ) (Measure.pi ν) = ∑ i, klDiv (μ i) (ν i) := by
  induction n with
  | zero =>
    have h : Measure.pi μ = Measure.pi ν := by
      rw [Measure.pi_of_empty, Measure.pi_of_empty]
    rw [h, klDiv_self]
    simp
  | succ n ih =>
    have hμ := (measurePreserving_piFinSuccAbove μ 0).map_eq
    have hν := (measurePreserving_piFinSuccAbove ν 0).map_eq
    rw [← klDiv_map_equiv (MeasurableEquiv.piFinSuccAbove (fun _ => ℝ) 0), hμ, hν,
      klDiv_prod, ih, Fin.sum_univ_succ]
    simp only [Fin.succAbove_zero]

/-! ### Isotropic Gaussians -/

/-- The isotropic Gaussian `N(m, v I)` on `ℝ^d`: `d` independent real
Gaussians with means `m_i` and common variance `v`.

This is the law the appendix calls `N(m, σ² I)`. It is Mathlib's
`multivariateGaussian m (v • 1)` under the identification of `Fin d → ℝ` with
`EuclideanSpace ℝ (Fin d)` (`isoGaussian_map_toLp`). -/
noncomputable def isoGaussian {d : ℕ} (m : Fin d → ℝ) (v : ℝ≥0) :
    Measure (Fin d → ℝ) :=
  Measure.pi fun i => gaussianReal (m i) v

instance {d : ℕ} (m : Fin d → ℝ) (v : ℝ≥0) : IsProbabilityMeasure (isoGaussian m v) := by
  unfold isoGaussian; infer_instance

/-- **Divergence between two isotropic Gaussians**:

`KL(N(m₀, v₀ I) ‖ N(m₁, v₁ I)) = ½ (d v₀/v₁ − d + d log(v₁/v₀) + ‖m₁ − m₀‖²/v₁)`. -/
theorem klDiv_isoGaussian {d : ℕ} (m₀ m₁ : Fin d → ℝ) {v₀ v₁ : ℝ≥0} (h₀ : v₀ ≠ 0)
    (h₁ : v₁ ≠ 0) :
    klDiv (isoGaussian m₀ v₀) (isoGaussian m₁ v₁)
      = ENNReal.ofReal (1 / 2 * (d * ((v₀ : ℝ) / v₁) - d + d * Real.log (v₁ / v₀)
          + (∑ i, (m₁ i - m₀ i) ^ 2) / v₁)) := by
  have hv₁ : (0 : ℝ) < v₁ := NNReal.coe_pos.2 (pos_iff_ne_zero.2 h₁)
  -- Each coordinate term is a divergence, hence nonnegative.
  have hnn : ∀ i, 0 ≤ 1 / 2 * ((v₀ : ℝ) / v₁ - 1 + Real.log (v₁ / v₀)
      + (m₁ i - m₀ i) ^ 2 / v₁) := by
    intro i
    by_contra hneg
    have h := klDiv_gaussianReal (m₀ i) (m₁ i) h₀ h₁
    rw [ENNReal.ofReal_of_nonpos (not_le.1 hneg).le] at h
    have hzero := (klDiv_eq_zero_iff).1 h
    have := (gaussianReal_ext_iff).1 hzero
    obtain ⟨hm, hv⟩ := this
    apply hneg
    rw [hm, hv]
    simp [div_self (NNReal.coe_ne_zero.2 h₁)]
  unfold isoGaussian
  rw [klDiv_pi]
  simp only [klDiv_gaussianReal _ _ h₀ h₁]
  rw [← ENNReal.ofReal_sum_of_nonneg fun i _ => hnn i]
  congr 1
  rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_add_distrib,
    Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  rw [Finset.sum_div]

/-! ### Joint laws with a shared label marginal

The joint law of a label `Y` with prior `q` on `Fin K` and a feature vector
whose law given `Y = k` is `μ k`. Two such laws with the same prior differ only
through their class-conditionals, and their divergence is the prior-weighted
average of the per-class divergences: the chain rule of relative entropy with a
vanishing marginal term. -/

section Joint

variable {K : ℕ} {E : Type*} [MeasurableSpace E]

/-- The joint law of `(Y, X)` when `P(Y = k) = q k` and `X | Y = k ∼ μ k`. -/
noncomputable def jointLaw (q : Fin K → ℝ≥0∞) (μ : Fin K → Measure E) :
    Measure (Fin K × E) :=
  Measure.sum fun k => q k • (μ k).map (Prod.mk k)

theorem lintegral_jointLaw (q : Fin K → ℝ≥0∞) (μ : Fin K → Measure E)
    {g : Fin K × E → ℝ≥0∞} (hg : Measurable g) :
    ∫⁻ p, g p ∂jointLaw q μ = ∑ k, q k * ∫⁻ x, g (k, x) ∂μ k := by
  unfold jointLaw
  rw [lintegral_sum_measure, tsum_fintype]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [lintegral_smul_measure, lintegral_map hg measurable_prodMk_left, smul_eq_mul]

theorem jointLaw_apply (q : Fin K → ℝ≥0∞) (μ : Fin K → Measure E)
    {A : Set (Fin K × E)} (hA : MeasurableSet A) :
    jointLaw q μ A = ∑ k, q k * μ k (Prod.mk k ⁻¹' A) := by
  unfold jointLaw
  rw [Measure.sum_apply _ hA, tsum_fintype]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Measure.smul_apply, Measure.map_apply measurable_prodMk_left hA, smul_eq_mul]

theorem isFiniteMeasure_jointLaw (q : Fin K → ℝ≥0∞) (hq : ∀ k, q k ≠ ∞)
    (μ : Fin K → Measure E) [∀ k, IsProbabilityMeasure (μ k)] :
    IsFiniteMeasure (jointLaw q μ) := by
  refine ⟨?_⟩
  rw [jointLaw_apply q μ MeasurableSet.univ]
  simp only [Set.preimage_univ, measure_univ, mul_one]
  exact ENNReal.sum_lt_top.2 fun k _ => (hq k).lt_top

/-- **Chain rule with a shared label marginal.** If two joint laws have the
same prior `q` and class-conditionals `μ k ≪ ν k`, their divergence is
`∑_k q_k · KL(μ_k ‖ ν_k)`.

Proved from the definition: the density of one joint law with respect to the
other is, on the fibre of class `k`, the density of `μ k` with respect to
`ν k`. -/
theorem klDiv_jointLaw (q : Fin K → ℝ≥0∞) (hq : ∀ k, q k ≠ ∞)
    (μ ν : Fin K → Measure E) [∀ k, IsProbabilityMeasure (μ k)]
    [∀ k, IsProbabilityMeasure (ν k)] (hac : ∀ k, μ k ≪ ν k) :
    klDiv (jointLaw q μ) (jointLaw q ν) = ∑ k, q k * klDiv (μ k) (ν k) := by
  have := isFiniteMeasure_jointLaw q hq μ
  have := isFiniteMeasure_jointLaw q hq ν
  let f : Fin K × E → ℝ≥0∞ := fun p => (μ p.1).rnDeriv (ν p.1) p.2
  have hf : Measurable f :=
    measurable_from_prod_countable_right fun k => Measure.measurable_rnDeriv (μ k) (ν k)
  have hwd : jointLaw q μ = (jointLaw q ν).withDensity f := by
    refine Measure.ext fun A hA => ?_
    rw [withDensity_apply _ hA, ← lintegral_indicator hA,
      lintegral_jointLaw q ν (hf.indicator hA), jointLaw_apply q μ hA]
    refine Finset.sum_congr rfl fun k _ => ?_
    congr 1
    rw [← Measure.setLIntegral_rnDeriv (hac k) (Prod.mk k ⁻¹' A),
      ← lintegral_indicator (measurable_prodMk_left hA)]
    rfl
  have hacJ : jointLaw q μ ≪ jointLaw q ν := by
    rw [hwd]
    exact withDensity_absolutelyContinuous _ _
  have hrn : (jointLaw q μ).rnDeriv (jointLaw q ν) =ᵐ[jointLaw q ν] f := by
    have := Measure.rnDeriv_withDensity (jointLaw q ν) hf
    rwa [← hwd] at this
  have hcongr : ∫⁻ p, ENNReal.ofReal
        (klFun ((jointLaw q μ).rnDeriv (jointLaw q ν) p).toReal) ∂jointLaw q ν
      = ∫⁻ p, ENNReal.ofReal (klFun (f p).toReal) ∂jointLaw q ν := by
    refine lintegral_congr_ae ?_
    filter_upwards [hrn] with p hp
    rw [hp]
  have hmeas : Measurable fun p : Fin K × E => ENNReal.ofReal (klFun (f p).toReal) :=
    (measurable_klFun.comp hf.ennreal_toReal).ennreal_ofReal
  rw [klDiv_eq_lintegral_klFun_of_ac hacJ, hcongr, lintegral_jointLaw q ν hmeas]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [klDiv_eq_lintegral_klFun_of_ac (hac k)]

/-- The uniform prior `q k = 1 / K`. -/
noncomputable def uniformPrior (K : ℕ) : Fin K → ℝ≥0∞ := fun _ => (K : ℝ≥0∞)⁻¹

/-- Under the uniform prior, if every per-class divergence equals `c` then so
does the divergence of the joint laws. -/
theorem klDiv_jointLaw_uniform [NeZero K] (μ ν : Fin K → Measure E)
    [∀ k, IsProbabilityMeasure (μ k)] [∀ k, IsProbabilityMeasure (ν k)]
    (hac : ∀ k, μ k ≪ ν k) {c : ℝ≥0∞} (hc : ∀ k, klDiv (μ k) (ν k) = c) :
    klDiv (jointLaw (uniformPrior K) μ) (jointLaw (uniformPrior K) ν) = c := by
  have hK : (K : ℝ≥0∞) ≠ 0 := Nat.cast_ne_zero.2 (NeZero.ne K)
  rw [klDiv_jointLaw (uniformPrior K) (fun _ => ENNReal.inv_ne_top.2 hK) μ ν hac]
  simp only [hc, uniformPrior, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  rw [← mul_assoc, ENNReal.mul_inv_cancel hK (ENNReal.natCast_ne_top K), one_mul]

end Joint

/-! ### Appendix B

`v` is the variance `σ²` of the text and `lam` the multiplier `λ`, both as
nonnegative reals assumed nonzero, which is the positivity of the covariance
matrices `σ² I` and `λ σ² I`. The dimension `d` is arbitrary; the text takes
`d = 2` for the variance family and `d = K − 1` for the mean-shift family. -/

open Matrix in
/-- **The general Gaussian formula, for scalar covariance matrices**
(`eq:kl_gaussian_general` with `Σ₀ = v₀ I`, `Σ₁ = v₁ I`):

`KL(N(m₀, Σ₀) ‖ N(m₁, Σ₁)) = ½ [tr(Σ₁⁻¹ Σ₀) − d + log(det Σ₁ / det Σ₀)
  + (m₁ − m₀)ᵀ Σ₁⁻¹ (m₁ − m₀)]`.

This is the statement for the product measures `isoGaussian`; the one tagged
with the label, `kl_multivariateGaussian_scalar`, is for Mathlib's
`multivariateGaussian`. -/
theorem kl_gaussian_scalar {d : ℕ} (m₀ m₁ : Fin d → ℝ) {v₀ v₁ : ℝ≥0} (h₀ : v₀ ≠ 0)
    (h₁ : v₁ ≠ 0) :
    klDiv (isoGaussian m₀ v₀) (isoGaussian m₁ v₁)
      = ENNReal.ofReal (1 / 2 * (
          trace (((v₁ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ))⁻¹
            * ((v₀ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ))) - d
          + Real.log (((v₁ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ)).det
              / ((v₀ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ)).det)
          + (m₁ - m₀) ⬝ᵥ (((v₁ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ))⁻¹
              *ᵥ (m₁ - m₀)))) := by
  have hv₀ : (0 : ℝ) < v₀ := NNReal.coe_pos.2 (pos_iff_ne_zero.2 h₀)
  have hv₁ : (0 : ℝ) < v₁ := NNReal.coe_pos.2 (pos_iff_ne_zero.2 h₁)
  have hinv : ((v₁ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ))⁻¹
      = (v₁ : ℝ)⁻¹ • (1 : Matrix (Fin d) (Fin d) ℝ) := by
    refine Matrix.inv_eq_right_inv ?_
    rw [Matrix.smul_mul, Matrix.mul_smul, Matrix.mul_one, smul_smul,
      mul_inv_cancel₀ hv₁.ne', one_smul]
  have htr : trace ((v₁ : ℝ)⁻¹ • (1 : Matrix (Fin d) (Fin d) ℝ)
      * ((v₀ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ))) = d * ((v₀ : ℝ) / v₁) := by
    rw [Matrix.smul_mul, Matrix.mul_smul, Matrix.mul_one, smul_smul, Matrix.trace_smul,
      Matrix.trace_one, Fintype.card_fin, smul_eq_mul]
    ring
  have hdet : ∀ c : ℝ, (c • (1 : Matrix (Fin d) (Fin d) ℝ)).det = c ^ d := by
    intro c
    rw [Matrix.det_smul, Matrix.det_one, Fintype.card_fin, mul_one]
  have hlog : Real.log ((v₁ : ℝ) ^ d / (v₀ : ℝ) ^ d) = d * Real.log (v₁ / v₀) := by
    rw [← div_pow, Real.log_pow]
  have hquad : (m₁ - m₀) ⬝ᵥ (((v₁ : ℝ)⁻¹ • (1 : Matrix (Fin d) (Fin d) ℝ))
      *ᵥ (m₁ - m₀)) = (∑ i, (m₁ i - m₀ i) ^ 2) / v₁ := by
    rw [Matrix.smul_mulVec, Matrix.one_mulVec, dotProduct_smul, smul_eq_mul]
    simp only [dotProduct, Pi.sub_apply]
    rw [div_eq_inv_mul]
    congr 1
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [klDiv_isoGaussian m₀ m₁ h₀ h₁, hinv, htr, hdet, hdet, hlog, hquad]

/-! ### The same statement for Mathlib's multivariate Gaussian -/

section Multivariate

open WithLp Matrix
open scoped RealInnerProductSpace

/-- The product measure `isoGaussian m v` is Mathlib's multivariate Gaussian
with mean `m` and covariance matrix `v I`, under the identification of
`Fin d → ℝ` with Euclidean space. Both have the same characteristic
function. -/
theorem isoGaussian_map_toLp {d : ℕ} (m : Fin d → ℝ) (v : ℝ≥0) :
    (isoGaussian m v).map (toLp 2)
      = multivariateGaussian (toLp 2 m) ((v : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ)) := by
  have hS : ((v : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ)).PosSemidef :=
    Matrix.PosSemidef.one.smul v.coe_nonneg
  apply Measure.ext_of_charFun (E := EuclideanSpace ℝ (Fin d))
  ext t
  unfold isoGaussian
  rw [charFun_pi, charFun_multivariateGaussian hS]
  simp_rw [charFun_gaussianReal, ← Complex.exp_sum]
  congr 1
  simp [Matrix.smul_mulVec, dotProduct, PiLp.inner_apply, Finset.sum_sub_distrib,
    Finset.sum_mul, Finset.sum_div]
  congr 1 <;> exact Finset.sum_congr rfl fun i _ => by ring

/-- **The general Gaussian formula, for scalar covariance matrices**
(`eq:kl_gaussian_general` with `Σ₀ = v₀ I`, `Σ₁ = v₁ I`), for Mathlib's
multivariate Gaussian measures on Euclidean space:

`KL(N(m₀, Σ₀) ‖ N(m₁, Σ₁)) = ½ [tr(Σ₁⁻¹ Σ₀) − d + log(det Σ₁ / det Σ₀)
  + (m₁ − m₀)ᵀ Σ₁⁻¹ (m₁ − m₀)]`.

This is the instance for positive multiples of the identity, the only one the
appendix uses, obtained from the product measure. The formula for arbitrary
positive definite covariance matrices, which carries the label, is
`klDiv_multivariateGaussian` in
[`GaussianKLGeneral.lean`](GaussianKLGeneral.lean). -/
theorem kl_multivariateGaussian_scalar {d : ℕ} (m₀ m₁ : EuclideanSpace ℝ (Fin d))
    {v₀ v₁ : ℝ≥0} (h₀ : v₀ ≠ 0) (h₁ : v₁ ≠ 0) :
    klDiv (multivariateGaussian m₀ ((v₀ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ)))
        (multivariateGaussian m₁ ((v₁ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ)))
      = ENNReal.ofReal (1 / 2 * (
          trace (((v₁ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ))⁻¹
            * ((v₀ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ))) - d
          + Real.log (((v₁ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ)).det
              / ((v₀ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ)).det)
          + (ofLp m₁ - ofLp m₀) ⬝ᵥ (((v₁ : ℝ) • (1 : Matrix (Fin d) (Fin d) ℝ))⁻¹
              *ᵥ (ofLp m₁ - ofLp m₀)))) := by
  have e₀ := isoGaussian_map_toLp (ofLp m₀) v₀
  have e₁ := isoGaussian_map_toLp (ofLp m₁) v₁
  rw [toLp_ofLp] at e₀ e₁
  rw [← e₀, ← e₁, ← kl_gaussian_scalar (ofLp m₀) (ofLp m₁) h₀ h₁]
  exact klDiv_map_equiv (MeasurableEquiv.toLp 2 (Fin d → ℝ)) _ _

end Multivariate

/-- The variance-scaling budget `(d/2)(log λ + 1/λ − 1)` is nonnegative, so the
`ENNReal.ofReal` in the statements below does not truncate. -/
theorem variance_budget_nonneg (d : ℕ) {lam : ℝ} (hlam : 0 < lam) :
    0 ≤ (d : ℝ) / 2 * (Real.log lam + 1 / lam - 1) := by
  have := Real.one_sub_inv_le_log_of_pos hlam
  have h : 0 ≤ Real.log lam + 1 / lam - 1 := by
    rw [one_div]; linarith
  positivity

/-- **Variance scaling, per class** (`eq:kl_variance_perclass`):

`KL(N(m, σ² I) ‖ N(m, λ σ² I)) = (d/2)(log λ + 1/λ − 1)`. -/
@[thesis "eq:kl_variance_perclass"]
theorem kl_variance_perclass {d : ℕ} (m : Fin d → ℝ) {v lam : ℝ≥0} (hv : v ≠ 0)
    (hlam : lam ≠ 0) :
    klDiv (isoGaussian m v) (isoGaussian m (lam * v))
      = ENNReal.ofReal ((d : ℝ) / 2 * (Real.log lam + 1 / lam - 1)) := by
  have hv' : (0 : ℝ) < v := NNReal.coe_pos.2 (pos_iff_ne_zero.2 hv)
  have hl' : (0 : ℝ) < lam := NNReal.coe_pos.2 (pos_iff_ne_zero.2 hlam)
  have h1 : (v : ℝ) / (lam * v) = 1 / lam := by field_simp
  have h2 : (lam : ℝ) * v / v = lam := by field_simp
  rw [klDiv_isoGaussian m m hv (mul_ne_zero hlam hv)]
  congr 1
  simp only [NNReal.coe_mul, sub_self, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
    zero_pow, Finset.sum_const_zero, zero_div, add_zero, h1, h2]
  ring

/-- Two nondegenerate isotropic Gaussians are mutually absolutely continuous;
here, the direction the divergences below need. -/
theorem isoGaussian_ac {d : ℕ} (m₀ m₁ : Fin d → ℝ) {v₀ v₁ : ℝ≥0} (h₀ : v₀ ≠ 0)
    (h₁ : v₁ ≠ 0) : isoGaussian m₀ v₀ ≪ isoGaussian m₁ v₁ := by
  have h : klDiv (isoGaussian m₀ v₀) (isoGaussian m₁ v₁) ≠ ∞ := by
    rw [klDiv_isoGaussian m₀ m₁ h₀ h₁]
    exact ENNReal.ofReal_ne_top
  exact (klDiv_ne_top_iff.1 h).1

/-- **Variance scaling, joint** (`eq:kl_variance_joint`): with uniform priors
and class-conditionals `N(m_k, σ² I)` against `N(m_k, λ σ² I)`,

`KL(P₁ ‖ P_λ) = ∑_k q_k KL(p₁(· | k) ‖ p_λ(· | k)) = (d/2)(log λ + 1/λ − 1)`,

whatever the number of classes and the class means. -/
@[thesis "eq:kl_variance_joint"]
theorem kl_variance_joint {K d : ℕ} [NeZero K] (m : Fin K → Fin d → ℝ) {v lam : ℝ≥0}
    (hv : v ≠ 0) (hlam : lam ≠ 0) :
    klDiv (jointLaw (uniformPrior K) fun k => isoGaussian (m k) v)
        (jointLaw (uniformPrior K) fun k => isoGaussian (m k) (lam * v))
      = ∑ k, uniformPrior K k
          * klDiv (isoGaussian (m k) v) (isoGaussian (m k) (lam * v))
    ∧ klDiv (jointLaw (uniformPrior K) fun k => isoGaussian (m k) v)
        (jointLaw (uniformPrior K) fun k => isoGaussian (m k) (lam * v))
      = ENNReal.ofReal ((d : ℝ) / 2 * (Real.log lam + 1 / lam - 1)) := by
  have hK : (K : ℝ≥0∞) ≠ 0 := Nat.cast_ne_zero.2 (NeZero.ne K)
  have hac : ∀ k, isoGaussian (m k) v ≪ isoGaussian (m k) (lam * v) :=
    fun k => isoGaussian_ac _ _ hv (mul_ne_zero hlam hv)
  exact ⟨klDiv_jointLaw _ (fun _ => ENNReal.inv_ne_top.2 hK) _ _ hac,
    klDiv_jointLaw_uniform _ _ hac fun k => kl_variance_perclass (m k) hv hlam⟩

/-- **Mean shift, per class** (`eq:kl_meanshift_perclass`): for a unit vector
`u`,

`KL(N(m, σ² I) ‖ N(m + ξ u, σ² I)) = ‖ξ u‖² / (2σ²) = ξ² / (2σ²)`. -/
@[thesis "eq:kl_meanshift_perclass"]
theorem kl_meanshift_perclass {d : ℕ} (m u : Fin d → ℝ) (hu : ∑ i, u i ^ 2 = 1)
    (ξ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    klDiv (isoGaussian m v) (isoGaussian (fun i => m i + ξ * u i) v)
      = ENNReal.ofReal (ξ ^ 2 / (2 * v)) := by
  have hv' : (0 : ℝ) < v := NNReal.coe_pos.2 (pos_iff_ne_zero.2 hv)
  have hs : ∑ i, (m i + ξ * u i - m i) ^ 2 = ξ ^ 2 := by
    simp only [add_sub_cancel_left, mul_pow]
    rw [← Finset.mul_sum, hu, mul_one]
  rw [klDiv_isoGaussian m _ hv hv, hs, div_self hv'.ne', Real.log_one]
  congr 1
  ring

/-- **Mean shift, joint** (the unlabeled display after
`eq:kl_meanshift_perclass`): with uniform priors, class-conditionals
`N(m_k, σ² I)` against `N(m_k + ξ u_k, σ² I)` and unit vectors `u_k`, the
joint divergence is `ξ² / (2σ²)`. The shift directions are fixed vectors here;
in the text they are drawn at random and the statement is conditional on the
draw. -/
theorem kl_meanshift_joint {K d : ℕ} [NeZero K] (m u : Fin K → Fin d → ℝ)
    (hu : ∀ k, ∑ i, u k i ^ 2 = 1) (ξ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    klDiv (jointLaw (uniformPrior K) fun k => isoGaussian (m k) v)
        (jointLaw (uniformPrior K) fun k => isoGaussian (fun i => m k i + ξ * u k i) v)
      = ENNReal.ofReal (ξ ^ 2 / (2 * v)) :=
  klDiv_jointLaw_uniform _ _ (fun _ => isoGaussian_ac _ _ hv hv)
    fun k => kl_meanshift_perclass (m k) (u k) (hu k) ξ hv

/-! ### The reference geometry of the mean-shift family -/

/-- Equal-covariance class-conditionals: the divergence is the squared distance
between the means over `2σ²`. -/
theorem kl_pairwise {d : ℕ} (m₀ m₁ : Fin d → ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    klDiv (isoGaussian m₀ v) (isoGaussian m₁ v)
      = ENNReal.ofReal ((∑ i, (m₀ i - m₁ i) ^ 2) / (2 * v)) := by
  have hv' : (0 : ℝ) < v := NNReal.coe_pos.2 (pos_iff_ne_zero.2 hv)
  rw [klDiv_isoGaussian m₀ m₁ hv hv, div_self hv'.ne', Real.log_one]
  congr 1
  rw [Finset.sum_congr rfl fun i _ => sub_sq_comm (m₁ i) (m₀ i)]
  ring

/-- **Pairwise distance of the reference means** (`eq:meanshift_pairwise`,
squared): if `K ≥ 2` class means have norm `R` and pairwise inner products
`−R² / (K − 1)`, their pairwise squared distance is `2K R² / (K − 1)`. -/
theorem pairwise_dist_sq {K d : ℕ} (hK : 2 ≤ K) (m : Fin K → Fin d → ℝ) (R : ℝ)
    (hnorm : ∀ j, ∑ i, m j i ^ 2 = R ^ 2)
    (hinner : ∀ j k, j ≠ k → ∑ i, m j i * m k i = -R ^ 2 / (K - 1))
    {j k : Fin K} (hjk : j ≠ k) :
    ∑ i, (m j i - m k i) ^ 2 = 2 * K / (K - 1) * R ^ 2 := by
  have hK' : (2 : ℝ) ≤ K := by exact_mod_cast hK
  have hne : (K : ℝ) - 1 ≠ 0 := by linarith
  have hexp : ∀ i, (m j i - m k i) ^ 2 = m j i ^ 2 - 2 * (m j i * m k i) + m k i ^ 2 := by
    intro i; ring
  rw [Finset.sum_congr rfl fun i _ => hexp i, Finset.sum_add_distrib,
    Finset.sum_sub_distrib, ← Finset.mul_sum, hnorm j, hnorm k, hinner j k hjk]
  field_simp
  ring

/-- **`eq:meanshift_pairwise`**: `‖m_j − m_k‖ = R √(2K / (K − 1))`. -/
@[thesis "eq:meanshift_pairwise"]
theorem pairwise_dist {K d : ℕ} (hK : 2 ≤ K) (m : Fin K → Fin d → ℝ) {R : ℝ}
    (hR : 0 ≤ R) (hnorm : ∀ j, ∑ i, m j i ^ 2 = R ^ 2)
    (hinner : ∀ j k, j ≠ k → ∑ i, m j i * m k i = -R ^ 2 / (K - 1))
    {j k : Fin K} (hjk : j ≠ k) :
    Real.sqrt (∑ i, (m j i - m k i) ^ 2) = R * Real.sqrt (2 * K / (K - 1)) := by
  have hK' : (2 : ℝ) ≤ K := by exact_mod_cast hK
  have hpos : (0 : ℝ) ≤ 2 * K / (K - 1) := div_nonneg (by linarith) (by linarith)
  rw [pairwise_dist_sq hK m R hnorm hinner hjk, Real.sqrt_mul hpos, Real.sqrt_sq hR,
    mul_comm]

/-- **Pairwise divergence of the reference class-conditionals.** With the
circumradius `R = √(2(K−1)/K) · s` of Section 8, the pairwise distance is `2s`
and the pairwise divergence is `2 s² / σ²`, for every `K ≥ 2`. -/
theorem kl_reference_pairwise {K d : ℕ} (hK : 2 ≤ K) (m : Fin K → Fin d → ℝ) (R s : ℝ)
    (hnorm : ∀ j, ∑ i, m j i ^ 2 = R ^ 2)
    (hinner : ∀ j k, j ≠ k → ∑ i, m j i * m k i = -R ^ 2 / (K - 1))
    (hRs : R ^ 2 = 2 * (K - 1) / K * s ^ 2) {v : ℝ≥0} (hv : v ≠ 0)
    {j k : Fin K} (hjk : j ≠ k) :
    ∑ i, (m j i - m k i) ^ 2 = (2 * s) ^ 2
    ∧ klDiv (isoGaussian (m j) v) (isoGaussian (m k) v)
      = ENNReal.ofReal (2 * s ^ 2 / v) := by
  have hK' : (2 : ℝ) ≤ K := by exact_mod_cast hK
  have hne : (K : ℝ) - 1 ≠ 0 := by linarith
  have hK0 : (K : ℝ) ≠ 0 := by linarith
  have hdist : ∑ i, (m j i - m k i) ^ 2 = (2 * s) ^ 2 := by
    rw [pairwise_dist_sq hK m R hnorm hinner hjk, hRs]
    field_simp
  refine ⟨hdist, ?_⟩
  rw [kl_pairwise (m j) (m k) hv, hdist]
  congr 1
  ring

/-- **The reference pairwise divergence is `8`** (the unlabeled display closing
Appendix B) when `s = 2σ`, the choice of Section 8: the pairwise distance is
`4σ` and the divergence `(4σ)² / (2σ²) = 8`, for every `K ≥ 2`. -/
theorem kl_reference_pairwise_eight {K d : ℕ} (hK : 2 ≤ K) (m : Fin K → Fin d → ℝ)
    (R : ℝ) {v : ℝ≥0} (hv : v ≠ 0)
    (hnorm : ∀ j, ∑ i, m j i ^ 2 = R ^ 2)
    (hinner : ∀ j k, j ≠ k → ∑ i, m j i * m k i = -R ^ 2 / (K - 1))
    (hR : R ^ 2 = 2 * (K - 1) / K * (4 * v)) {j k : Fin K} (hjk : j ≠ k) :
    ∑ i, (m j i - m k i) ^ 2 = 16 * v
    ∧ klDiv (isoGaussian (m j) v) (isoGaussian (m k) v) = ENNReal.ofReal 8 := by
  have hv' : (0 : ℝ) < v := NNReal.coe_pos.2 (pos_iff_ne_zero.2 hv)
  have hs : R ^ 2 = 2 * (K - 1) / K * (2 * Real.sqrt v) ^ 2 := by
    rw [hR, mul_pow, Real.sq_sqrt hv'.le]; norm_num
  obtain ⟨h1, h2⟩ := kl_reference_pairwise hK m R (2 * Real.sqrt v) hnorm hinner hs hv hjk
  refine ⟨?_, ?_⟩
  · rw [h1, mul_pow, mul_pow, Real.sq_sqrt hv'.le]; ring
  · rw [h2, mul_pow, Real.sq_sqrt hv'.le]
    congr 1
    field_simp
    norm_num

/-- **The radius printed in Appendix B gives `2`, not `8`.** The last paragraph
of Appendix B writes the circumradius as `R = √(2(K−1)/K) · σ` and says it pins
the pairwise distance at `4σ` and the pairwise divergence at `8`. With that
radius the pairwise distance is `2σ` and the divergence is `2`. The radius of
Section 8, `√(2(K−1)/K) · s` with `s = 2σ`, is twice as large and gives `8`
(`kl_reference_pairwise_eight`); the appendix drops the factor `s / σ = 2`. -/
theorem kl_reference_pairwise_appendix_radius {K d : ℕ} (hK : 2 ≤ K)
    (m : Fin K → Fin d → ℝ) (R : ℝ) {v : ℝ≥0} (hv : v ≠ 0)
    (hnorm : ∀ j, ∑ i, m j i ^ 2 = R ^ 2)
    (hinner : ∀ j k, j ≠ k → ∑ i, m j i * m k i = -R ^ 2 / (K - 1))
    (hR : R ^ 2 = 2 * (K - 1) / K * v) {j k : Fin K} (hjk : j ≠ k) :
    ∑ i, (m j i - m k i) ^ 2 = 4 * v
    ∧ klDiv (isoGaussian (m j) v) (isoGaussian (m k) v) = ENNReal.ofReal 2 := by
  have hv' : (0 : ℝ) < v := NNReal.coe_pos.2 (pos_iff_ne_zero.2 hv)
  have hs : R ^ 2 = 2 * (K - 1) / K * (Real.sqrt v) ^ 2 := by
    rw [hR, Real.sq_sqrt hv'.le]
  obtain ⟨h1, h2⟩ := kl_reference_pairwise hK m R (Real.sqrt v) hnorm hinner hs hv hjk
  refine ⟨?_, ?_⟩
  · rw [h1, mul_pow, Real.sq_sqrt hv'.le]; norm_num
  · rw [h2, Real.sq_sqrt hv'.le]
    congr 1
    field_simp

/-- The centred vertices of the standard simplex of `ℝ^K`, rescaled: equal
norms `R`, pairwise inner products `−R² / (K − 1)`, and coordinates that sum to
zero, so that they lie in a hyperplane. -/
theorem exists_reference_means_centred {K : ℕ} (hK : 2 ≤ K) (R : ℝ) :
    ∃ m : Fin K → Fin K → ℝ, (∀ j, ∑ i, m j i ^ 2 = R ^ 2)
      ∧ (∀ j k, j ≠ k → ∑ i, m j i * m k i = -R ^ 2 / (K - 1))
      ∧ ∀ j, ∑ i, m j i = 0 := by
  have hK' : (2 : ℝ) ≤ K := by exact_mod_cast hK
  have hne : (K : ℝ) - 1 ≠ 0 := by linarith
  have hK0 : (K : ℝ) ≠ 0 := by linarith
  have hpos : (0 : ℝ) ≤ K / (K - 1) := div_nonneg (by linarith) (by linarith)
  -- `a² = R² K / (K − 1)`.
  let a : ℝ := R * Real.sqrt (K / (K - 1))
  have ha : a ^ 2 = R ^ 2 * (K / (K - 1)) := by
    simp only [a, mul_pow, Real.sq_sqrt hpos]
  have hone : ∀ l : Fin K, ∑ i : Fin K, (if i = l then (1 : ℝ) else 0) = 1 := by
    intro l
    rw [Finset.sum_ite_eq' Finset.univ l]
    simp
  have hdelta : ∀ j k : Fin K,
      ∑ i : Fin K, (if i = j then (1 : ℝ) else 0) * (if i = k then (1 : ℝ) else 0)
        = if j = k then 1 else 0 := by
    intro j k
    simp only [ite_mul, one_mul, zero_mul]
    rw [Finset.sum_ite_eq' Finset.univ j]
    simp
  have hprod : ∀ j k : Fin K,
      ∑ i : Fin K, (a * ((if i = j then (1 : ℝ) else 0) - 1 / K))
          * (a * ((if i = k then (1 : ℝ) else 0) - 1 / K))
        = a ^ 2 * ((if j = k then (1 : ℝ) else 0) - 1 / K) := by
    intro j k
    have hterm : ∀ i : Fin K,
        (a * ((if i = j then (1 : ℝ) else 0) - 1 / K))
          * (a * ((if i = k then (1 : ℝ) else 0) - 1 / K))
        = a ^ 2 * ((if i = j then (1 : ℝ) else 0) * (if i = k then (1 : ℝ) else 0))
            - a ^ 2 / K * (if i = j then (1 : ℝ) else 0)
            - a ^ 2 / K * (if i = k then (1 : ℝ) else 0) + a ^ 2 / K ^ 2 := by
      intro i
      ring
    rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_add_distrib,
      Finset.sum_sub_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
      ← Finset.mul_sum, ← Finset.mul_sum, hdelta j k, hone j, hone k]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
    ring
  refine ⟨fun j i => a * ((if i = j then (1 : ℝ) else 0) - 1 / K), fun j => ?_,
    fun j k hjk => ?_, fun j => ?_⟩
  · have := hprod j j
    simp only [if_true] at this
    rw [Finset.sum_congr rfl fun i _ => sq (a * ((if i = j then (1 : ℝ) else 0) - 1 / K)),
      this, ha]
    field_simp
  · rw [hprod j k, if_neg hjk, ha]
    field_simp
    ring
  · rw [← Finset.mul_sum, Finset.sum_sub_distrib, hone j]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [mul_one_div_cancel hK0, sub_self, mul_zero]

/-- The hypotheses on the reference means are satisfiable in `ℝ^K` for every
`K ≥ 2` and every radius. -/
theorem exists_reference_means {K : ℕ} (hK : 2 ≤ K) (R : ℝ) :
    ∃ m : Fin K → Fin K → ℝ, (∀ j, ∑ i, m j i ^ 2 = R ^ 2)
      ∧ ∀ j k, j ≠ k → ∑ i, m j i * m k i = -R ^ 2 / (K - 1) := by
  obtain ⟨m, h1, h2, -⟩ := exists_reference_means_centred hK R
  exact ⟨m, h1, h2⟩

section Hyperplane

open scoped RealInnerProductSpace

theorem euclidean_inner_eq_sum {n : ℕ} (x y : EuclideanSpace ℝ (Fin n)) :
    ⟪x, y⟫ = ∑ i, x i * y i := by
  simp [PiLp.inner_apply, mul_comm]

/-- **The reference means exist in `ℝ^{K−1}`**, where Section 8 places them:
for every `K ≥ 2` and every radius `R` there are `K` points of `ℝ^{K−1}` with
norm `R` and pairwise inner products `−R² / (K − 1)`, the vertices of a regular
simplex centred at the origin.

The centred vertices of the standard simplex of `ℝ^K` lie in the hyperplane
orthogonal to `(1, …, 1)`, which has dimension `K − 1` and is therefore
isometric to `ℝ^{K−1}`; the isometry preserves norms and inner products. -/
theorem exists_reference_means_hyperplane {K : ℕ} (hK : 2 ≤ K) (R : ℝ) :
    ∃ m : Fin K → Fin (K - 1) → ℝ, (∀ j, ∑ i, m j i ^ 2 = R ^ 2)
      ∧ ∀ j k, j ≠ k → ∑ i, m j i * m k i = -R ^ 2 / (K - 1) := by
  obtain ⟨m₀, hnorm, hinner, hsum⟩ := exists_reference_means_centred hK R
  let one : EuclideanSpace ℝ (Fin K) := WithLp.toLp 2 fun _ => (1 : ℝ)
  have hone : one ≠ 0 := by
    intro h
    have h0 : one ⟨0, by omega⟩ = (0 : EuclideanSpace ℝ (Fin K)) ⟨0, by omega⟩ := by
      rw [h]
    simp [one] at h0
  let x : Fin K → EuclideanSpace ℝ (Fin K) := fun j => WithLp.toLp 2 (m₀ j)
  have hfact : Fact (Module.finrank ℝ (EuclideanSpace ℝ (Fin K)) = (K - 1) + 1) :=
    ⟨by rw [finrank_euclideanSpace_fin]; omega⟩
  have hW : Module.finrank ℝ (Submodule.span ℝ {one})ᗮ = K - 1 :=
    Submodule.finrank_orthogonal_span_singleton hone
  have hx : ∀ j, x j ∈ (Submodule.span ℝ {one})ᗮ := by
    intro j
    rw [Submodule.mem_orthogonal_singleton_iff_inner_right, euclidean_inner_eq_sum]
    simpa [one, x] using hsum j
  let Φ : (Submodule.span ℝ {one})ᗮ ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin (K - 1)) :=
    ((stdOrthonormalBasis ℝ (Submodule.span ℝ {one})ᗮ).reindex (finCongr hW)).repr
  have hΦ : ∀ j k, ∑ i, Φ ⟨x j, hx j⟩ i * Φ ⟨x k, hx k⟩ i = ∑ i, m₀ j i * m₀ k i := by
    intro j k
    rw [← euclidean_inner_eq_sum, LinearIsometryEquiv.inner_map_map,
      Submodule.coe_inner, euclidean_inner_eq_sum]
  refine ⟨fun j i => Φ ⟨x j, hx j⟩ i, fun j => ?_, fun j k hjk => ?_⟩
  · rw [← hnorm j]
    simp only [sq]
    exact hΦ j j
  · rw [← hinner j k hjk]
    exact hΦ j k

end Hyperplane

end ThresholdOpt.GaussianKL
