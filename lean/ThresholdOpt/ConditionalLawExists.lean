import Mathlib.MeasureTheory.Function.AEEqOfIntegral
import Mathlib.MeasureTheory.Measure.Decomposition.RadonNikodym
import ThresholdOpt.Risk

/-!
# A conditional class-probability map exists for every data law

`prop:bayes_risk_identity` and `prop:excess_risk_bound` are proved in
[`Risk.lean`](Risk.lean) under the hypothesis `Risk.ConditionalLaw P η`: the
map `η` into the simplex is the conditional law of the label given the input,
stated as `E f(X, Y) = E ∑_k η_k(X) f(X, k)` for bounded measurable `f`. The
dissertation takes `η_k(x) := P(Y = k | X = x)` for granted.

This file shows the hypothesis is never vacuous: for every probability law `P`
on `X × Fin K`, with no condition on the input space `X`, there is a measurable
`η` satisfying it. The construction is the Radon-Nikodym one. The sub-law of
`X` on the event `Y = k` is dominated by the law of `X`; its density is `η_k`;
the densities sum to one almost everywhere because the sub-laws sum to the law
of `X`; and they are redefined on the null set where they do not.

As a consequence the Bayes risk is attained by a measurable classifier for
every data law, with no hypothesis at all (`exists_bayes_classifier`).
-/

namespace ThresholdOpt.ConditionalLawExists

open MeasureTheory ThresholdOpt ThresholdOpt.Risk
open scoped ENNReal

variable {K : ℕ} {X : Type*} [MeasurableSpace X]

/-- A bounded measurable function is integrable for a finite measure. -/
theorem integrable_of_bounded {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsFiniteMeasure μ] {g : Ω → ℝ} (hg : Measurable g) {M : ℝ} (hM : ∀ z, |g z| ≤ M) :
    Integrable g μ :=
  (integrable_const M).mono' hg.aestronglyMeasurable
    (Filter.Eventually.of_forall fun z => by rw [Real.norm_eq_abs]; exact hM z)

/-- The event `Y = k`. -/
def fibre (k : Fin K) : Set (X × Fin K) := Prod.snd ⁻¹' {k}

theorem measurableSet_fibre (k : Fin K) : MeasurableSet (fibre (X := X) k) :=
  measurable_snd (measurableSet_singleton k)

/-- The sub-law of `X` on the event `Y = k`. -/
noncomputable def classLaw (P : Measure (X × Fin K)) (k : Fin K) : Measure X :=
  (P.restrict (fibre k)).map Prod.fst

instance (P : Measure (X × Fin K)) [IsFiniteMeasure P] (k : Fin K) :
    IsFiniteMeasure (classLaw P k) := by
  unfold classLaw; infer_instance

theorem classLaw_ac (P : Measure (X × Fin K)) (k : Fin K) :
    classLaw P k ≪ xMarginal P :=
  (Measure.absolutelyContinuous_of_le Measure.restrict_le_self).map measurable_fst

/-- The density of the sub-law of class `k` with respect to the law of `X`. -/
noncomputable def density (P : Measure (X × Fin K)) (k : Fin K) (x : X) : ℝ :=
  ((classLaw P k).rnDeriv (xMarginal P) x).toReal

theorem measurable_density (P : Measure (X × Fin K)) (k : Fin K) :
    Measurable (density P k) :=
  (Measure.measurable_rnDeriv _ _).ennreal_toReal

theorem density_nonneg (P : Measure (X × Fin K)) (k : Fin K) (x : X) :
    0 ≤ density P k x := ENNReal.toReal_nonneg

variable (P : Measure (X × Fin K)) [IsProbabilityMeasure P]

/-- The total-expectation identity with the densities in place of `η`. -/
theorem integral_eq_density {F : X × Fin K → ℝ} (hF : Measurable F) {M : ℝ}
    (hM : ∀ z, |F z| ≤ M) :
    ∫ z, F z ∂P = ∫ x, ∑ k, density P k x * F (x, k) ∂xMarginal P := by
  have hFint : Integrable F P := integrable_of_bounded hF hM
  have hFk : ∀ k, Measurable fun x : X => F (x, k) := fun k =>
    hF.comp (measurable_id.prodMk measurable_const)
  have hsplit : ∫ z, F z ∂P = ∑ k, ∫ z in fibre k, F z ∂P := by
    have hU : (⋃ k : Fin K, fibre (X := X) k) = Set.univ := by
      ext z
      simp [fibre]
    rw [← setIntegral_univ, ← hU]
    exact integral_iUnion_fintype measurableSet_fibre
      (fun i j hij => Set.disjoint_left.2 fun z hi hj =>
        hij ((show z.2 = i from hi).symm.trans hj))
      (fun k => hFint.integrableOn)
  have hfib : ∀ k, ∫ z in fibre k, F z ∂P
      = ∫ x, density P k x * F (x, k) ∂xMarginal P := by
    intro k
    have h1 : ∫ z in fibre k, F z ∂P = ∫ z in fibre k, F (z.1, k) ∂P :=
      setIntegral_congr_fun (measurableSet_fibre k) fun z hz => by
        have hzk : z.2 = k := hz
        show F z = F (z.1, k)
        rw [← hzk]
    have h2 : ∫ z in fibre k, F (z.1, k) ∂P = ∫ x, F (x, k) ∂classLaw P k := by
      unfold classLaw
      rw [integral_map measurable_fst.aemeasurable (hFk k).aestronglyMeasurable]
    rw [h1, h2]
    exact (integral_toReal_rnDeriv_mul (classLaw_ac P k)).symm
  have hint : ∀ k, Integrable (fun x => density P k x * F (x, k)) (xMarginal P) :=
    fun k => (integrable_toReal_rnDeriv_mul_iff (classLaw_ac P k)).2
      (integrable_of_bounded (hFk k) fun x => hM (x, k))
  rw [hsplit, integral_finsetSum _ fun k _ => hint k]
  exact Finset.sum_congr rfl fun k _ => hfib k

/-- The densities sum to one almost everywhere. -/
theorem sum_density_ae_eq_one : ∀ᵐ x ∂xMarginal P, ∑ k, density P k x = 1 := by
  have hdint : ∀ k, Integrable (density P k) (xMarginal P) := by
    intro k
    have h := (integrable_toReal_rnDeriv_mul_iff (classLaw_ac P k)).2
      (integrable_const (1 : ℝ))
    simp only [mul_one] at h
    exact h
  refine Integrable.ae_eq_of_forall_setIntegral_eq (fun x => ∑ k, density P k x)
    (fun _ => (1 : ℝ)) (integrable_finsetSum _ fun k _ => hdint k) (integrable_const _) ?_
  intro A hA _
  have hind : Measurable (A.indicator fun _ : X => (1 : ℝ)) :=
    measurable_const.indicator hA
  have hbound : ∀ z : X × Fin K, |A.indicator (fun _ : X => (1 : ℝ)) z.1| ≤ 1 := by
    intro z
    by_cases hz : z.1 ∈ A <;> simp [Set.indicator, hz]
  have hpre := integral_eq_density P (F := fun z => A.indicator (fun _ : X => (1 : ℝ)) z.1)
    (hind.comp measurable_fst) hbound
  have e1 : ∫ z, A.indicator (fun _ : X => (1 : ℝ)) z.1 ∂P
      = ∫ x in A, (1 : ℝ) ∂xMarginal P := by
    rw [← integral_indicator hA]
    unfold xMarginal
    rw [integral_map measurable_fst.aemeasurable hind.aestronglyMeasurable]
  have e2 : ∫ x, ∑ k, density P k x * A.indicator (fun _ : X => (1 : ℝ)) x ∂xMarginal P
      = ∫ x in A, ∑ k, density P k x ∂xMarginal P := by
    rw [← integral_indicator hA]
    congr 1
    funext x
    by_cases hx : x ∈ A <;> simp [Set.indicator, hx]
  rw [← e2, ← hpre, e1]

variable [NeZero K]

/-- The densities, redefined as a vertex where they do not sum to one. -/
noncomputable def etaFun (x : X) (k : Fin K) : ℝ :=
  if ∑ j, density P j x = 1 then density P k x else vertex 0 k

omit [IsProbabilityMeasure P] in
theorem etaFun_mem (x : X) : etaFun P x ∈ stdSimplex ℝ (Fin K) := by
  by_cases h : ∑ j, density P j x = 1
  · have : etaFun P x = fun k => density P k x := by
      funext k
      simp [etaFun, h]
    rw [this]
    exact ⟨fun k => density_nonneg P k x, h⟩
  · have : etaFun P x = vertex 0 := by
      funext k
      simp [etaFun, h]
    rw [this]
    exact vertex_mem_stdSimplex 0

/-- A version of the conditional class probabilities `η_k(x) = P(Y = k | X = x)`. -/
noncomputable def eta (x : X) : Simplex K := ⟨etaFun P x, etaFun_mem P x⟩

omit [IsProbabilityMeasure P] in
theorem measurable_eta : Measurable (eta P) := by
  refine Measurable.subtype_mk (measurable_pi_iff.2 fun k => ?_)
  exact Measurable.ite
    (measurableSet_eq_fun (Finset.measurable_sum _ fun j _ => measurable_density P j)
      measurable_const)
    (measurable_density P k) measurable_const

/-- **Existence of the conditional class probabilities.** For every probability
law `P` on `X × Fin K` there is a measurable `η : X → Δ^{K-1}` with
`Risk.ConditionalLaw P η`. -/
theorem exists_conditionalLaw :
    ∃ η : X → Simplex K, Measurable η ∧ ConditionalLaw P η := by
  refine ⟨eta P, measurable_eta P, ?_⟩
  rintro F hF ⟨M, hM⟩
  rw [integral_eq_density P hF hM]
  refine integral_congr_ae ?_
  filter_upwards [sum_density_ae_eq_one P] with x hx
  refine Finset.sum_congr rfl fun k _ => ?_
  simp [eta, etaFun, hx]

/-- **The Bayes risk is attained, for every data law.** For every probability
law on `X × Fin K` and every bounded nonnegative cost matrix there is a
measurable classifier whose risk is the least among measurable classifiers. No
conditional-law hypothesis is left: the classifier is the Bayes rule of the
version of `η` constructed above. This is `prop:bayes_risk_identity` without
its explicit hypothesis, in the form "`R* = min_h R(h)` is attained". -/
theorem exists_bayes_classifier {C : CostMatrix K} {Cmax : ℝ} (hC : Bounded C Cmax) :
    ∃ h : X → Fin K, Measurable h
      ∧ IsLeast (measurableRisks P C) (jointRisk P C h)
      ∧ jointRisk P C h = bayesRisk P C := by
  obtain ⟨η, hη, hcond⟩ := exists_conditionalLaw P
  obtain ⟨hmeas, -, -, hleast, hbayes⟩ := bayes_risk_identity hη hcond hC
  exact ⟨bayesClassifier C η, hmeas, hleast, hbayes⟩

end ThresholdOpt.ConditionalLawExists
