import ThresholdOpt.PointwiseApprox

/-!
# The binary generalization proposition with pointwise approximation

The public statement assumes a countable measurable subclass approximating
every rule pointwise, with eventual equality at each input. Its VC dimension
is bounded by that of the original class. The countable bound passes to all
rules by dominated convergence and eventual equality on the finite sample.
The clean form `eq:clean_gen_bound` is stated under the same hypotheses
without `d ≤ m`. Because the approximating class is a subclass, the uniform
event of the class equals that of the subclass, which is measurable.
-/

namespace ThresholdOpt.SeparableBounds

open MeasureTheory
open ThresholdOpt ThresholdOpt.Capacity ThresholdOpt.Learning
open ThresholdOpt.Literature ThresholdOpt.PointwiseApprox

/-- **Generalization bound for binary decision optimization**
(`prop:gen_bound`). For bounded costs, a countable measurable subclass
approximating every rule pointwise, `1 ≤ d ≤ m`, and `0 < ν ≤ 1`, all rules
simultaneously satisfy the displayed bound with factor `2`, with probability
at least `1 - ν`. No literature axiom or class-scope difference remains. -/
@[thesis "prop:gen_bound"]
theorem gen_bound_binary_thesis_of_approx {X : Type} [MeasurableSpace X]
    (C : CostMatrix 2) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H H₀ : Set (X → Fin 2)) (h0sub : H₀ ⊆ H) (h0count : H₀.Countable)
    (h0meas : ∀ δ ∈ H₀, Measurable δ) (happrox : ApproxBy H H₀)
    (μ : Measure (X × Fin 2)) [IsProbabilityMeasure μ] (d m : ℕ) (ν : ℝ)
    (hdim : vcDim (binClass H) ≤ (d : ℕ∞)) (hd : 0 < d) (hdm : d ≤ m)
    (hm : 0 < m) (hν : 0 < ν) (hν1 : ν ≤ 1) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H, popRisk μ (costLoss C δ)
        ≤ empRisk S (costLoss C δ) + 2 * Cmax * thesisBinaryRate m d ν} :=
  (gen_bound_binary_thesis C hC H₀ h0count h0meas μ d m ν
    ((vcDim_mono (binClass_mono h0sub)).trans hdim) hd hdm hm hν hν1).trans
    (measure_mono fun S hS => bound_of_approx hC h0meas happrox μ S _ hS)

/-- **Clean binary generalization bound** (`eq:clean_gen_bound`), under the
hypotheses of `prop:gen_bound` without `d ≤ m`: bounded costs, a countable
measurable subclass approximating every rule pointwise, `d ≥ max(VCdim, 1)`,
`m ≥ 1` and `0 < ν ≤ 1`. With probability at least `1 - ν`, every rule
satisfies `R ≤ R̂ + c_bin(m, ν) Cmax √(d/m)`, where
`c_bin(m, ν) = √(2 log(e m)) + √(log(4/ν)/2)` (`cleanFactorBinary`). -/
@[thesis "eq:clean_gen_bound"]
theorem gen_bound_binary_clean_of_approx {X : Type} [MeasurableSpace X]
    (C : CostMatrix 2) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H H₀ : Set (X → Fin 2)) (h0sub : H₀ ⊆ H) (h0count : H₀.Countable)
    (h0meas : ∀ δ ∈ H₀, Measurable δ) (happrox : ApproxBy H H₀)
    (μ : Measure (X × Fin 2)) [IsProbabilityMeasure μ] (d m : ℕ) (ν : ℝ)
    (hdim : vcDim (binClass H) ≤ (d : ℕ∞)) (hd : 0 < d) (hm : 0 < m)
    (hν : 0 < ν) (hν1 : ν ≤ 1) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H, popRisk μ (costLoss C δ)
        ≤ empRisk S (costLoss C δ)
          + cleanFactorBinary m ν * Cmax * Real.sqrt (d / m)} :=
  (gen_bound_binary_clean C hC H₀ h0count h0meas μ d m ν
    ((vcDim_mono (binClass_mono h0sub)).trans hdim) hd hm hν hν1).trans
    (measure_mono fun S hS => bound_of_approx hC h0meas happrox μ S _ hS)

/-- With a countable approximating *subclass*, the uniform event of the class
is the uniform event of the subclass. -/
theorem uniformEvent_eq_of_subclass {X : Type} [MeasurableSpace X] {K : ℕ}
    {C : CostMatrix K} {Cmax : ℝ} (hC : Bounded C Cmax) {H H₀ : Set (X → Fin K)}
    (h0sub : H₀ ⊆ H) (h0meas : ∀ δ ∈ H₀, Measurable δ) (happrox : ApproxBy H H₀)
    (μ : Measure (X × Fin K)) [IsProbabilityMeasure μ] (m : ℕ) (r : ℝ) :
    {S : Fin m → X × Fin K | ∀ δ ∈ H,
        popRisk μ (costLoss C δ) ≤ empRisk S (costLoss C δ) + r}
      = {S | ∀ δ ∈ H₀,
        popRisk μ (costLoss C δ) ≤ empRisk S (costLoss C δ) + r} :=
  Set.ext fun S =>
    ⟨fun h δ hδ => h δ (h0sub hδ), fun h => bound_of_approx hC h0meas happrox μ S r h⟩

/-- Hence the events in `gen_bound_binary_thesis_of_approx` and
`gen_bound_binary_clean_of_approx` are measurable, and the measure in their
conclusions is the probability of the event, not only an outer measure. -/
theorem measurableSet_uniformEvent_of_subclass {X : Type} [MeasurableSpace X]
    {K : ℕ} {C : CostMatrix K} {Cmax : ℝ} (hC : Bounded C Cmax)
    {H H₀ : Set (X → Fin K)} (h0sub : H₀ ⊆ H) (h0count : H₀.Countable)
    (h0meas : ∀ δ ∈ H₀, Measurable δ) (happrox : ApproxBy H H₀)
    (μ : Measure (X × Fin K)) [IsProbabilityMeasure μ] (m : ℕ) (r : ℝ) :
    MeasurableSet {S : Fin m → X × Fin K | ∀ δ ∈ H,
      popRisk μ (costLoss C δ) ≤ empRisk S (costLoss C δ) + r} := by
  rw [uniformEvent_eq_of_subclass hC h0sub h0meas happrox μ m r]
  exact measurableSet_uniformEvent C H₀ h0count h0meas μ m r

end ThresholdOpt.SeparableBounds
