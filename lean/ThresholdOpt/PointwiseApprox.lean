import ThresholdOpt.Literature
import ThresholdOpt.VoronoiSimplex

/-!
# Generalization bounds for classes approximated by countable ones

The bounds of [`Literature.lean`](Literature.lean) are for countable classes
of measurable rules. The decision classes of Chapter 6 are indexed by real
parameters. They are covered here by approximation: if every rule of `H` is
the pointwise limit (eventually equal at each point) of rules from a countable
class `H₀` of measurable rules, then on every sample where the bound holds for
all of `H₀` it holds for all of `H` (`bound_of_approx`): the empirical risks
are eventually equal and the population risks converge by dominated
convergence. The capacity in the bound is that of `H₀`.

The delicate point is ties. A rule `p ↦ argmin_k s_k(p)` breaks ties toward
the smallest index; a perturbation of the parameters that moves to a nearby
rational parameter can break a tie the other way, at a point where it then
never recovers. `eventually_lexArgmin_add` is the remedy: perturbations of the
scores that vanish in the limit and are strictly increasing in the class index
preserve the selected class at every point eventually, ties included, because
among tied classes the smallest index receives the smallest perturbation.
-/

namespace ThresholdOpt.PointwiseApprox

open MeasureTheory Filter Topology
open ThresholdOpt ThresholdOpt.Learning ThresholdOpt.Capacity ThresholdOpt.Literature

/-! ### Tie-respecting perturbations -/

/-- **Perturbations increasing in the index preserve the smallest-index
`argmin`.** If `e n k → 0` for every `k`, and for `j < k` eventually
`e n j < e n k`, then eventually `argmin (f + e n) = argmin f`, ties of `f`
included. -/
theorem eventually_lexArgmin_add {K : ℕ} [NeZero K] (f : Fin K → ℝ)
    (e : ℕ → Fin K → ℝ) (he0 : ∀ k, Tendsto (fun n => e n k) atTop (𝓝 0))
    (hmono : ∀ j k : Fin K, j < k → ∀ᶠ n in atTop, e n j < e n k) :
    ∀ᶠ n in atTop, lexArgmin (fun k => f k + e n k) = lexArgmin f := by
  set k0 := lexArgmin f with hk0
  have hj : ∀ j : Fin K, j ≠ k0 → ∀ᶠ n in atTop, f k0 + e n k0 < f j + e n j := by
    intro j hj
    have hle : f k0 ≤ f j := lexArgmin_le f j
    rcases hle.lt_or_eq with hlt | heq
    · have hdiff : Tendsto (fun n => e n k0 - e n j) atTop (𝓝 0) := by
        simpa using (he0 k0).sub (he0 j)
      filter_upwards [hdiff.eventually (gt_mem_nhds (sub_pos.2 hlt))] with n hn
      linarith
    · have hmin : ∀ l, f j ≤ f l := fun l => by rw [← heq]; exact lexArgmin_le f l
      have hk0j : k0 < j := lt_of_le_of_ne (lexArgmin_le_index hmin) (Ne.symm hj)
      filter_upwards [hmono k0 j hk0j] with n hn
      linarith
  have hall : ∀ᶠ n in atTop, ∀ j : Fin K, j ≠ k0 → f k0 + e n k0 < f j + e n j := by
    rw [eventually_all]
    intro j
    by_cases h : j = k0
    · exact Eventually.of_forall fun n hn => absurd h hn
    · filter_upwards [hj j h] with n hn _ using hn
  filter_upwards [hall] with n hn
  exact lexArgmin_eq_of_strict hn

/-- Perturbations squeezed in `(k/(n+1), (k + 1/2)/(n+1))` satisfy the
hypotheses of `eventually_lexArgmin_add`. -/
theorem perturb_of_bounds {K : ℕ} (e : ℕ → Fin K → ℝ)
    (hlo : ∀ (n : ℕ) (k : Fin K), (k : ℝ) / ((n : ℝ) + 1) < e n k)
    (hhi : ∀ (n : ℕ) (k : Fin K), e n k < ((k : ℝ) + 1 / 2) / ((n : ℝ) + 1)) :
    (∀ k, Tendsto (fun n => e n k) atTop (𝓝 0)) ∧
      (∀ j k : Fin K, j < k → ∀ᶠ n in atTop, e n j < e n k) := by
  refine ⟨fun k => ?_, fun j k hjk => Eventually.of_forall fun n => ?_⟩
  · have h0 : Tendsto (fun n : ℕ => 1 / ((n : ℝ) + 1)) atTop (𝓝 0) :=
      tendsto_one_div_add_atTop_nhds_zero_nat
    have hl : Tendsto (fun n : ℕ => (k : ℝ) / ((n : ℝ) + 1)) atTop (𝓝 0) := by
      have := h0.const_mul (k : ℝ)
      simp only [mul_zero, mul_one_div] at this
      exact this
    have hu : Tendsto (fun n : ℕ => ((k : ℝ) + 1 / 2) / ((n : ℝ) + 1)) atTop (𝓝 0) := by
      have := h0.const_mul ((k : ℝ) + 1 / 2)
      simp only [mul_zero, mul_one_div] at this
      exact this
    exact tendsto_of_tendsto_of_tendsto_of_le_of_le hl hu (fun n => (hlo n k).le)
      fun n => (hhi n k).le
  · have hjk' : ((j : ℕ) : ℝ) + 1 / 2 ≤ (k : ℕ) := by
      have : (j : ℕ) + 1 ≤ (k : ℕ) := hjk
      have : ((j : ℕ) : ℝ) + 1 ≤ (k : ℕ) := by exact_mod_cast this
      linarith
    have hpos : (0 : ℝ) < (n : ℝ) + 1 := by positivity
    calc e n j < ((j : ℝ) + 1 / 2) / ((n : ℝ) + 1) := hhi n j
      _ ≤ (k : ℝ) / ((n : ℝ) + 1) := div_le_div_of_nonneg_right hjk' hpos.le
      _ < e n k := hlo n k

/-! ### Measurability of `argmin` rules -/

/-- A smallest-index `argmin` of measurable scores is measurable. -/
theorem measurable_lexArgmin {X : Type*} [MeasurableSpace X] {K : ℕ} [NeZero K]
    {s : Fin K → X → ℝ} (hs : ∀ k, Measurable (s k)) :
    Measurable fun x => lexArgmin fun k => s k x := by
  refine measurable_to_countable' fun k => ?_
  have hset : (fun x => lexArgmin fun k => s k x) ⁻¹' {k}
      = (⋂ j, {x | s k x ≤ s j x}) ∩ ⋂ j, {x | j < k → s k x < s j x} := by
    ext x
    simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_inter_iff,
      Set.mem_iInter, Set.mem_ofPred_eq]
    constructor
    · intro h
      refine ⟨fun j => ?_, fun j hj => ?_⟩
      · rw [← h]
        exact lexArgmin_le (fun k => s k x) j
      · exact VoronoiSimplex.lt_of_lexArgmin_eq h hj
    · rintro ⟨h1, h2⟩
      refine le_antisymm (lexArgmin_le_index h1) ?_
      by_contra hlt
      push Not at hlt
      have e1 := h2 _ hlt
      have e2 := lexArgmin_le (fun k => s k x) k
      linarith
  rw [hset]
  refine (MeasurableSet.iInter fun j => measurableSet_le (hs k) (hs j)).inter
    (MeasurableSet.iInter fun j => ?_)
  by_cases hj : j < k
  · simpa [hj] using measurableSet_lt (hs k) (hs j)
  · simp [hj]

/-! ### From a countable class to its pointwise limits -/

/-- Every rule of `H` is the pointwise limit of rules of `H₀`: at each point,
eventually equal. -/
def ApproxBy {X : Type*} {K : ℕ} (H H₀ : Set (X → Fin K)) : Prop :=
  ∀ δ ∈ H, ∃ u : ℕ → X → Fin K, (∀ n, u n ∈ H₀) ∧ ∀ x, ∀ᶠ n in atTop, u n x = δ x

theorem ApproxBy.mono_left {X : Type*} {K : ℕ} {H H' H₀ : Set (X → Fin K)}
    (h : ApproxBy H H₀) (hsub : H' ⊆ H) : ApproxBy H' H₀ :=
  fun δ hδ => h δ (hsub hδ)

/-- **The bound passes to pointwise limits.** On a sample where every rule of
`H₀` satisfies `R ≤ R̂ + r`, so does every rule of `H`, if `H` is approximated
by `H₀` and the costs are bounded. -/
theorem bound_of_approx {X : Type*} [MeasurableSpace X] {K : ℕ} {C : CostMatrix K}
    {Cmax : ℝ} (hC : Bounded C Cmax) {H H₀ : Set (X → Fin K)}
    (hmeas₀ : ∀ δ ∈ H₀, Measurable δ) (happrox : ApproxBy H H₀)
    (μ : Measure (X × Fin K)) [IsProbabilityMeasure μ] {m : ℕ}
    (S : Fin m → X × Fin K) (r : ℝ)
    (h₀ : ∀ δ ∈ H₀, popRisk μ (costLoss C δ) ≤ empRisk S (costLoss C δ) + r) :
    ∀ δ ∈ H, popRisk μ (costLoss C δ) ≤ empRisk S (costLoss C δ) + r := by
  intro δ hδ
  obtain ⟨u, hu0, hlim⟩ := happrox δ hδ
  have hlossm : ∀ n, Measurable (costLoss C (u n)) := fun n =>
    (Measurable.of_discrete (f := fun q : Fin K × Fin K => C q.1 q.2)).comp
      (measurable_snd.prodMk ((hmeas₀ _ (hu0 n)).comp measurable_fst))
  have hpop : Tendsto (fun n => popRisk μ (costLoss C (u n))) atTop
      (𝓝 (popRisk μ (costLoss C δ))) := by
    unfold popRisk
    refine tendsto_integral_of_dominated_convergence (fun _ => Cmax)
      (fun n => (hlossm n).aestronglyMeasurable) (integrable_const _) (fun n => ?_) ?_
    · refine ae_of_all _ fun z => ?_
      have hmem := costLoss_mem_Icc hC (u n) z
      rw [Real.norm_eq_abs, abs_of_nonneg hmem.1]
      exact hmem.2
    · refine ae_of_all _ fun z => tendsto_nhds_of_eventually_eq ?_
      filter_upwards [hlim z.1] with n hn
      simp only [costLoss, hn]
  have hemp : Tendsto (fun n => empRisk S (costLoss C (u n))) atTop
      (𝓝 (empRisk S (costLoss C δ))) := by
    refine tendsto_nhds_of_eventually_eq ?_
    have hall : ∀ᶠ n in atTop, ∀ j : Fin m, u n (S j).1 = δ (S j).1 :=
      eventually_all.2 fun j => hlim (S j).1
    filter_upwards [hall] with n hn
    unfold empRisk
    congr 1
    exact Finset.sum_congr rfl fun j _ => by simp only [costLoss, hn j]
  exact le_of_tendsto_of_tendsto' hpop (hemp.add_const _) fun n => h₀ _ (hu0 n)

/-- **Generalization bound for `K` classes, for a class approximated by a
countable one.** As `Literature.gen_bound_natarajan`, with `H` any class
approximated pointwise by a countable class `H₀` of measurable rules with
`d_Nat(H₀) ≤ d`. Neither countability nor measurability of `H` is assumed,
and `H₀` need not be contained in `H`. -/
theorem gen_bound_natarajan_of_approx {X : Type} [MeasurableSpace X] {K : ℕ} [NeZero K]
    (C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H H₀ : Set (X → Fin K)) (h0count : H₀.Countable)
    (h0meas : ∀ δ ∈ H₀, Measurable δ) (happrox : ApproxBy H H₀)
    (μ : Measure (X × Fin K)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : natDim H₀ ≤ (d : ℕ∞)) (hdm : levelDim d K ≤ m) (hm : 0 < m)
    (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + Cmax * vcRate m (levelDim d K) (ν / ((K : ℝ) * K))} :=
  (gen_bound_natarajan C hC H₀ h0count h0meas μ d m ν hdim hdm hm hν).trans
    (measure_mono fun S hS => bound_of_approx hC h0meas happrox μ S _ hS)

/-- **The clean form, for a class approximated by a countable one.** As
`Literature.gen_bound_natarajan_clean`, under the hypotheses of
`gen_bound_natarajan_of_approx`. -/
theorem gen_bound_natarajan_clean_of_approx {X : Type} [MeasurableSpace X] {K : ℕ}
    [NeZero K] (C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H H₀ : Set (X → Fin K)) (h0count : H₀.Countable)
    (h0meas : ∀ δ ∈ H₀, Measurable δ) (happrox : ApproxBy H H₀)
    (μ : Measure (X × Fin K)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : natDim H₀ ≤ (d : ℕ∞)) (hd : 0 < d) (hm : 0 < m)
    (hν : 0 < ν) (hν1 : ν ≤ 1) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + cleanFactor K d m ν * Cmax * Real.sqrt (d / m)} :=
  (gen_bound_natarajan_clean C hC H₀ h0count h0meas μ d m ν hdim hd hm hν hν1).trans
    (measure_mono fun S hS => bound_of_approx hC h0meas happrox μ S _ hS)

/-- The uniform event of a countable class of measurable rules is measurable:
a countable intersection of sublevel sets of measurable empirical risks. -/
theorem measurableSet_uniformEvent {X : Type} [MeasurableSpace X] {K : ℕ}
    (C : CostMatrix K) (H₀ : Set (X → Fin K)) (h0count : H₀.Countable)
    (h0meas : ∀ δ ∈ H₀, Measurable δ) (μ : Measure (X × Fin K)) (m : ℕ) (r : ℝ) :
    MeasurableSet {S : Fin m → X × Fin K | ∀ δ ∈ H₀,
      popRisk μ (costLoss C δ) ≤ empRisk S (costLoss C δ) + r} := by
  have hset : {S : Fin m → X × Fin K | ∀ δ ∈ H₀,
      popRisk μ (costLoss C δ) ≤ empRisk S (costLoss C δ) + r}
      = ⋂ δ ∈ H₀, {S : Fin m → X × Fin K |
          popRisk μ (costLoss C δ) ≤ empRisk S (costLoss C δ) + r} := by
    ext S
    simp only [Set.mem_ofPred_eq, Set.mem_iInter]
  rw [hset]
  refine MeasurableSet.biInter h0count fun δ hδ => ?_
  have hloss : Measurable (costLoss C δ) :=
    (Measurable.of_discrete (f := fun q : Fin K × Fin K => C q.1 q.2)).comp
      (measurable_snd.prodMk ((h0meas δ hδ).comp measurable_fst))
  exact measurableSet_le measurable_const ((measurable_empRisk hloss).add_const r)

/-- **Multiclass generalization bound** (`rem:natarajan_gen_bound`), as the
text states it: `H` is any class of rules and `H₀` a countable class of
measurable rules that approximates every rule of `H` pointwise, not
necessarily contained in `H`; `d ≥ max(d_Nat(H₀), 1)`, `m ≥ 1` and
`0 < ν ≤ 1`. Because `H₀` need not be a subclass, the event for `H` need not
be measurable, and a bound on its outer measure would be weaker than the
text. The conclusion therefore exhibits a measurable event of probability at
least `1 - ν` on which the displayed bound holds for every rule of `H`. -/
@[thesis "rem:natarajan_gen_bound"]
theorem gen_bound_natarajan_clean_of_approx_event {X : Type} [MeasurableSpace X]
    {K : ℕ} [NeZero K] (C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H H₀ : Set (X → Fin K)) (h0count : H₀.Countable)
    (h0meas : ∀ δ ∈ H₀, Measurable δ) (happrox : ApproxBy H H₀)
    (μ : Measure (X × Fin K)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : natDim H₀ ≤ (d : ℕ∞)) (hd : 0 < d) (hm : 0 < m)
    (hν : 0 < ν) (hν1 : ν ≤ 1) :
    ∃ E : Set (Fin m → X × Fin K), MeasurableSet E ∧
      1 - ENNReal.ofReal ν ≤ sampleMeasure μ m E ∧
      ∀ S ∈ E, ∀ δ ∈ H,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + cleanFactor K d m ν * Cmax * Real.sqrt (d / m) :=
  ⟨_, measurableSet_uniformEvent C H₀ h0count h0meas μ m _,
    gen_bound_natarajan_clean C hC H₀ h0count h0meas μ d m ν hdim hd hm hν hν1,
    fun S hS => bound_of_approx hC h0meas happrox μ S _ hS⟩

/-! ### Two classes, with the VC dimension of the binary class -/

theorem binClass_mono {X : Type} {H₀ H : Set (X → Fin 2)} (h : H₀ ⊆ H) :
    binClass H₀ ⊆ binClass H := by
  rintro g ⟨δ, hδ, rfl⟩
  exact ⟨δ, h hδ, rfl⟩

/-- **Generalization bound for two classes, for a class approximated by a
countable subclass.** As `Literature.gen_bound_binary`, for any class `H` of
rules `X → {0, 1}` with a countable subclass `H₀` of measurable rules that
approximates every rule of `H` pointwise, and `d_VC(binClass H) ≤ d ≤ m`.

This is the hypothesis under which `prop:gen_bound` holds for an uncountable
class: it covers the real thresholds (rational thresholds approximate them)
and the classes of Chapter 6 (`DecisionClassBounds`). The weaker hypothesis
that the uniform event be measurable does not suffice: under the continuum
hypothesis, the rules `δ_t(x) = 1[x ≺ t]`, `≺` a well-order of `[0, 1]` of
type `ω₁`, are measurable (each `{x ≺ t}` is countable) and their binary class
has VC dimension `1` (initial segments of a linear order). Under the `0/1`
loss, with `X` uniform and the label always `1`, every `δ_t` has risk `1`, and
for every sample some `t` lies above all sample points, where the empirical
risk is `0`. Once the rate is below `1`, the uniform event is empty, hence
measurable, and has probability `0`. (Not formalized: Lean cannot assume the
continuum hypothesis here.) -/
theorem gen_bound_binary_of_approx {X : Type} [MeasurableSpace X]
    (C : CostMatrix 2) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H H₀ : Set (X → Fin 2)) (h0sub : H₀ ⊆ H) (h0count : H₀.Countable)
    (h0meas : ∀ δ ∈ H₀, Measurable δ) (happrox : ApproxBy H H₀)
    (μ : Measure (X × Fin 2)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : vcDim (binClass H) ≤ (d : ℕ∞)) (hdm : d ≤ m) (hm : 0 < m)
    (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ) + Cmax * vcRate m d (ν / 4)} :=
  (gen_bound_binary C hC H₀ h0count h0meas μ d m ν
    ((vcDim_mono (binClass_mono h0sub)).trans hdim) hdm hm hν).trans
    (measure_mono fun S hS => bound_of_approx hC h0meas happrox μ S _ hS)

end ThresholdOpt.PointwiseApprox
