import ThresholdOpt.Risk

/-!
# Binary classification (Chapter 2)

The reasonableness conditions, the equivalence between the Bayes rule and a
threshold rule (`lem:threshold_optimality`), the Elkan threshold
(`eq:elkan_threshold`), and the optimality of calibrate-then-decide
(`prop:calibrate_then_decide`).

Binary cost entries are written `C y ŷ` throughout, so `C 0 1` is the cost of a
false positive and `C 1 0` the cost of a false negative, matching the `C_{01}`
and `C_{10}` of the text.
-/

namespace ThresholdOpt.Binary

open ThresholdOpt

/-- The reasonableness conditions `eq:reasonableness`: `C_{01} > C_{00}` and
`C_{10} > C_{11}`. Each error costs strictly more than the corresponding
correct prediction. -/
structure Reasonable (C : CostMatrix 2) : Prop where
  fp : C 0 0 < C 0 1
  fn : C 1 1 < C 1 0

/-- The slope `m = (C_{10} - C_{11}) + (C_{01} - C_{00})` of the risk
difference. -/
def slope (C : CostMatrix 2) : ℝ := (C 1 0 - C 1 1) + (C 0 1 - C 0 0)

/-- The intercept `c = C_{01} - C_{00}`. -/
def intercept (C : CostMatrix 2) : ℝ := C 0 1 - C 0 0

theorem slope_pos {C : CostMatrix 2} (h : Reasonable C) : 0 < slope C := by
  have h1 : 0 < C 1 0 - C 1 1 := sub_pos.2 h.fn
  have h2 : 0 < C 0 1 - C 0 0 := sub_pos.2 h.fp
  simpa [slope] using add_pos h1 h2

theorem intercept_pos {C : CostMatrix 2} (h : Reasonable C) : 0 < intercept C :=
  sub_pos.2 h.fp

/-- The Elkan threshold `p* = (C_{01} - C_{00}) / ((C_{01} - C_{00}) + (C_{10} - C_{11}))`
of `eq:elkan_threshold`. -/
noncomputable def elkanThreshold (C : CostMatrix 2) : ℝ :=
  intercept C / slope C

/-- The Elkan threshold lies strictly inside `(0, 1)` under the reasonableness
conditions. -/
@[thesis "eq:elkan_threshold"]
theorem elkanThreshold_mem_Ioo {C : CostMatrix 2} (h : Reasonable C) :
    elkanThreshold C ∈ Set.Ioo (0 : ℝ) 1 := by
  have hm : 0 < slope C := slope_pos h
  have hc : 0 < intercept C := intercept_pos h
  have hfn : 0 < C 1 0 - C 1 1 := sub_pos.2 h.fn
  constructor
  · exact div_pos hc hm
  · show intercept C / slope C < 1
    rw [div_lt_one hm]
    simp only [slope, intercept]
    linarith

/-- Under the `0/1` loss the Elkan threshold is `1/2`, the cost-blind default
that the dissertation argues against relying on. -/
theorem elkanThreshold_zeroOne :
    elkanThreshold (Matrix.of ![![0, 1], ![1, 0]] : CostMatrix 2) = 1 / 2 := by
  norm_num [elkanThreshold, intercept, slope]

/-! ### Threshold optimality

`lem:threshold_optimality`. In the binary case a probability vector on the
simplex is determined by its second coordinate `η`, and the difference of the
two conditional risks is affine in `η` with positive slope, so the Bayes rule
is the indicator of `η ≥ p*`. -/

/-- The binary probability vector with `P(Y = 1) = η`. -/
noncomputable def probVec (η : ℝ) : Fin 2 → ℝ := ![1 - η, η]

theorem probVec_mem_stdSimplex {η : ℝ} (h : η ∈ Set.Icc (0:ℝ) 1) :
    probVec η ∈ stdSimplex ℝ (Fin 2) := by
  constructor
  · intro i
    fin_cases i <;> simp [probVec] <;> linarith [h.1, h.2]
  · simp [probVec, Fin.sum_univ_two]

/-- The risk difference `𝓛_0(p) - 𝓛_1(p)` is affine in `η` with slope `m` and
intercept `-c`: it equals `m·η - c`. This is the `g(η) = mη - c` of the proof
of `lem:threshold_optimality`. -/
theorem condRisk_sub (C : CostMatrix 2) (η : ℝ) :
    condRisk C (probVec η) 0 - condRisk C (probVec η) 1
      = slope C * η - intercept C := by
  simp only [condRisk, probVec, slope, intercept, Fin.sum_univ_two]
  simp [Matrix.cons_val_zero, Matrix.cons_val_one]
  ring

/-- Case analysis on `Fin 2` producing numerals rather than `⟨n, _⟩` terms, so
that the risk-difference identities apply to the resulting goals directly. -/
theorem fin2_cases (j : Fin 2) : j = 0 ∨ j = 1 := by
  revert j
  decide

/-- The threshold rule `δ_τ(η) = 𝟙(η ≥ τ)` of `eq:threshold_rule`, as an action
in `Fin 2`. -/
noncomputable def thresholdRule (τ η : ℝ) : Fin 2 := if τ ≤ η then 1 else 0

/-- A threshold is *optimal* when its rule attains the minimum conditional risk
at every `η ∈ [0,1]`.

This is the sense in which `lem:threshold_optimality` says the Bayes rule "is
equivalent to" a threshold function. It cannot be pointwise equality of rules:
on the tie set `{η = τ}` both actions attain the minimum, and the
smallest-index convention of `eq:bayes_optimal` picks `0` while
`𝟙(η ≥ τ)` picks `1`. The proof of the lemma makes exactly this point. -/
def OptimalThreshold (C : CostMatrix 2) (τ : ℝ) : Prop :=
  ∀ η ∈ Set.Icc (0 : ℝ) 1, ∀ j,
    condRisk C (probVec η) (thresholdRule τ η) ≤ condRisk C (probVec η) j

/-- The risk difference in the form used by every argument below. -/
theorem condRisk_zero_sub_one (C : CostMatrix 2) (η : ℝ) :
    condRisk C (probVec η) 0 - condRisk C (probVec η) 1
      = slope C * η - intercept C := condRisk_sub C η

/-- **The Elkan threshold is optimal.** Above it action `1` has the smaller
conditional risk, below it action `0` does, and at it the two are equal. -/
theorem optimalThreshold_elkan {C : CostMatrix 2} (h : Reasonable C) :
    OptimalThreshold C (elkanThreshold C) := by
  have hm : 0 < slope C := slope_pos h
  intro η _ j
  have hdiff := condRisk_zero_sub_one C η
  by_cases hge : elkanThreshold C ≤ η
  · have hge' : intercept C / slope C ≤ η := hge
    rw [div_le_iff₀ hm] at hge'
    have : thresholdRule (elkanThreshold C) η = 1 := if_pos hge
    rw [this]
    rcases fin2_cases j with rfl | rfl <;> linarith
  · have hlt' : η < intercept C / slope C := not_le.1 hge
    rw [lt_div_iff₀ hm] at hlt'
    have : thresholdRule (elkanThreshold C) η = 0 := if_neg hge
    rw [this]
    rcases fin2_cases j with rfl | rfl <;> linarith

/-- **The optimal threshold is unique.** A threshold below `p*` misclassifies
some `η` between the two, and one above `p*` misclassifies some `η` on the
other side; in each case the mislabelled point has strictly larger conditional
risk, so the threshold is not optimal. -/
theorem eq_elkan_of_optimalThreshold {C : CostMatrix 2} (h : Reasonable C)
    {τ : ℝ} (hτ : τ ∈ Set.Icc (0 : ℝ) 1) (hopt : OptimalThreshold C τ) :
    τ = elkanThreshold C := by
  have hm : 0 < slope C := slope_pos h
  have hp := elkanThreshold_mem_Ioo h
  by_contra hne
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · -- τ < p*: at the midpoint the rule says 1, but action 0 is strictly better.
    set η := (τ + elkanThreshold C) / 2 with hη
    have h1 : τ < η := by rw [hη]; linarith
    have h2 : η < elkanThreshold C := by rw [hη]; linarith
    have hmem : η ∈ Set.Icc (0 : ℝ) 1 :=
      ⟨by linarith [hτ.1], by linarith [hp.2]⟩
    have hrule : thresholdRule τ η = 1 := if_pos h1.le
    have hstrict : condRisk C (probVec η) 0 < condRisk C (probVec η) 1 := by
      have h2' : η < intercept C / slope C := h2
      rw [lt_div_iff₀ hm] at h2'
      have := condRisk_zero_sub_one C η
      linarith
    have := hopt η hmem 0
    rw [hrule] at this
    linarith
  · -- τ > p*: at the midpoint the rule says 0, but action 1 is strictly better.
    set η := (elkanThreshold C + τ) / 2 with hη
    have h1 : η < τ := by rw [hη]; linarith
    have h2 : elkanThreshold C < η := by rw [hη]; linarith
    have hmem : η ∈ Set.Icc (0 : ℝ) 1 :=
      ⟨by linarith [hp.1], by linarith [hτ.2]⟩
    have hrule : thresholdRule τ η = 0 := if_neg (not_le.2 h1)
    have hstrict : condRisk C (probVec η) 1 < condRisk C (probVec η) 0 := by
      have h2' : intercept C / slope C < η := h2
      rw [div_lt_iff₀ hm] at h2'
      have := condRisk_zero_sub_one C η
      linarith
    have := hopt η hmem 1
    rw [hrule] at this
    linarith

/-- Off the tie set the threshold rule and the smallest-index Bayes rule agree
outright. -/
theorem bayes_eq_threshold {C : CostMatrix 2} (h : Reasonable C) (η : ℝ) :
    (elkanThreshold C < η → bayesRule C (probVec η) = 1) ∧
    (η < elkanThreshold C → bayesRule C (probVec η) = 0) := by
  have hm : 0 < slope C := slope_pos h
  have hdiff := condRisk_sub C η
  constructor
  · intro hgt
    have hgt' : intercept C / slope C < η := hgt
    rw [div_lt_iff₀ hm] at hgt'
    refine lexArgmin_eq_of_strict ?_
    intro j hj
    have hj0 : j = 0 := by
      fin_cases j
      · rfl
      · exact absurd rfl hj
    subst hj0
    linarith
  · intro hlt
    have hlt' : η < intercept C / slope C := hlt
    rw [lt_div_iff₀ hm] at hlt'
    refine lexArgmin_eq_of_strict ?_
    intro j hj
    have hj1 : j = 1 := by
      fin_cases j
      · exact absurd rfl hj
      · rfl
    subst hj1
    linarith

/-- **Threshold optimality** (`lem:threshold_optimality`).

Under the reasonableness conditions the Bayes rule is the threshold rule at a
*unique* `τ ∈ [0,1]`, namely the Elkan threshold, which moreover lies in the
open interval. The three conjuncts are the three claims of the lemma:

1. `p*` lies in `(0,1)`, hence in `[0,1]`;
2. thresholding at `p*` attains the minimum conditional risk at every `η`,
   which is the equivalence with the Bayes rule (they can differ only on the
   risk-neutral tie set, and off it they agree outright, `bayes_eq_threshold`);
3. no other `τ ∈ [0,1]` has that property. -/
@[thesis "lem:threshold_optimality"]
theorem threshold_optimality {C : CostMatrix 2} (h : Reasonable C) :
    elkanThreshold C ∈ Set.Ioo (0 : ℝ) 1 ∧
    OptimalThreshold C (elkanThreshold C) ∧
    (∀ τ ∈ Set.Icc (0 : ℝ) 1, OptimalThreshold C τ → τ = elkanThreshold C) :=
  ⟨elkanThreshold_mem_Ioo h, optimalThreshold_elkan h,
    fun _ hτ hopt => eq_elkan_of_optimalThreshold h hτ hopt⟩

/-! ### Calibrate then decide -/

/-- Pointwise posterior-consistency core of calibrate-then-decide.

The hypothesis g(phat(x)) = eta(x) is stronger than marginal perfect
calibration. It gives optimal conditional risk at every x and equality to the
smallest-index Bayes rule off the tie set. The integrated counterpart below
uses the text's almost-everywhere premise and the actual joint law. -/
theorem calibrate_then_decide {C : CostMatrix 2} (h : Reasonable C)
    {X : Type*} (g : ℝ → ℝ) (phat η : X → ℝ)
    (hη : ∀ x, η x ∈ Set.Icc (0 : ℝ) 1)
    (hcal : ∀ x, g (phat x) = η x) :
    (∀ x j, condRisk C (probVec (η x))
        (thresholdRule (elkanThreshold C) (g (phat x)))
      ≤ condRisk C (probVec (η x)) j) ∧
    (∀ x, η x ≠ elkanThreshold C →
      thresholdRule (elkanThreshold C) (g (phat x))
        = bayesRule C (probVec (η x))) := by
  constructor
  · intro x j
    rw [hcal x]
    exact optimalThreshold_elkan h (η x) (hη x) j
  · intro x hne
    rw [hcal x]
    simp only [thresholdRule]
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · rw [if_neg (not_le.2 hlt), (bayes_eq_threshold h (η x)).2 hlt]
    · rw [if_pos hgt.le, (bayes_eq_threshold h (η x)).1 hgt]


/-! ### Conditional calibration and integrated optimality -/

open MeasureTheory

/-- Scalar probability scores, including the endpoints. -/
abbrev Score := {q : ℝ // q ∈ Set.Icc (0 : ℝ) 1}

/-- The Bernoulli conditional label law with positive-class probability q. -/
noncomputable def bernoulliProbabilities (q : Score) : Risk.Simplex 2 :=
  ⟨probVec q.val, probVec_mem_stdSimplex q.property⟩

theorem measurable_bernoulliProbabilities : Measurable bernoulliProbabilities := by
  apply Measurable.subtype_mk
  apply measurable_pi_iff.2
  intro i
  rcases fin2_cases i with rfl | rfl
  · simpa [probVec, Function.comp_def] using ((measurable_const_sub (1 : ℝ)).comp measurable_subtype_coe :
      Measurable (fun q : Score => 1 - q.val))
  · simpa [probVec] using (measurable_subtype_coe : Measurable (fun q : Score => q.val))

@[simp] theorem bernoulliProbabilities_positive (q : Score) :
    (bernoulliProbabilities q).val 1 = q.val := by simp [bernoulliProbabilities, probVec]

/-- Perfect calibration is conditional probability given the scalar score,
not equality to the posterior given the full input. The Bernoulli(q) kernel
is a version at every q, including null score events, and its positive-class
probability is exactly q. ConditionalLaw supplies its disintegration identity. -/
@[thesis "def:calibration"]
def PerfectCalibration {X : Type*} [MeasurableSpace X]
    (P : Measure (X × Fin 2)) [IsProbabilityMeasure P] (score : X → Score) : Prop :=
  Measurable score ∧ Risk.ConditionalLaw
    (P.map (fun z => (score z.1, z.2))) bernoulliProbabilities

theorem second_probability_mem (p : Risk.Simplex 2) : p.val 1 ∈ Set.Icc (0 : ℝ) 1 := by
  refine ⟨p.property.1 1, ?_⟩
  have hsum := p.property.2
  simp only [Fin.sum_univ_two] at hsum
  linarith [p.property.1 0]

theorem probVec_second_eq (p : Risk.Simplex 2) : probVec (p.val 1) = p.val := by
  ext i
  rcases fin2_cases i with rfl | rfl
  · have hsum := p.property.2
    simp only [Fin.sum_univ_two] at hsum
    simp only [probVec, Matrix.cons_val_zero]
    linarith
  · simp [probVec]

/-- Posterior consistency implies genuine marginal calibration, via the
conditional-law tower identity; the converse is not asserted. -/
theorem perfect_calibration_of_posterior_consistency
    {X : Type*} [MeasurableSpace X] {P : Measure (X × Fin 2)} [IsProbabilityMeasure P]
    {η : X → Risk.Simplex 2} (hcond : Risk.ConditionalLaw P η)
    {score : X → Score} (hs : Measurable score)
    (hcons : ∀ᵐ x ∂Risk.xMarginal P, (score x).val = (η x).val 1) :
    PerfectCalibration P score := by
  refine ⟨hs, ?_⟩
  intro f hf hbound
  have hpair : Measurable (fun z : X × Fin 2 => (score z.1, z.2)) :=
    (hs.comp measurable_fst).prodMk measurable_snd
  have hmarg : Risk.xMarginal (P.map (fun z => (score z.1, z.2))) =
      (Risk.xMarginal P).map score := by
    unfold Risk.xMarginal
    rw [Measure.map_map measurable_fst hpair, Measure.map_map hs measurable_fst]
    rfl
  let F : Score → ℝ := fun q => ∑ k, (bernoulliProbabilities q).val k * f (q, k)
  have hF : Measurable F := by
    exact Finset.measurable_sum Finset.univ fun k _ =>
      ((measurable_pi_apply k).comp
        (measurable_subtype_coe.comp measurable_bernoulliProbabilities)).mul
        (hf.comp (measurable_id.prodMk measurable_const))
  rw [integral_map hpair.aemeasurable hf.aestronglyMeasurable]
  rw [hcond (fun z => f (score z.1, z.2)) (hf.comp hpair)
    (by obtain ⟨M, hM⟩ := hbound; exact ⟨M, fun z => hM (score z.1, z.2)⟩)]
  rw [hmarg]
  change (∫ x, ∑ k, (η x).val k * f (score x, k) ∂Risk.xMarginal P) =
    ∫ q, F q ∂(Risk.xMarginal P).map score
  rw [integral_map hs.aemeasurable hF.aestronglyMeasurable]
  apply integral_congr_ae
  filter_upwards [hcons] with x hx
  have hv : (η x).val = (bernoulliProbabilities (score x)).val := by
    simpa [bernoulliProbabilities] using
      (probVec_second_eq (η x)).symm.trans (congrArg probVec hx.symm)
  rw [hv]

noncomputable def calibratedClassifier {X : Type*} (C : CostMatrix 2)
    (g : Score → Score) (phat : X → Score) : X → Fin 2 :=
  fun x => thresholdRule (elkanThreshold C) (g (phat x)).val

theorem measurable_calibratedClassifier {X : Type*} [MeasurableSpace X]
    (C : CostMatrix 2) {g : Score → Score} (hg : Measurable g)
    {phat : X → Score} (hp : Measurable phat) :
    Measurable (calibratedClassifier C g phat) := by
  have hs : Measurable (fun x => (g (phat x)).val) :=
    measurable_subtype_coe.comp (hg.comp hp)
  unfold calibratedClassifier thresholdRule
  exact Measurable.ite (measurableSet_le measurable_const hs) measurable_const measurable_const

/-- **Optimality of calibrate-then-decide** (`prop:calibrate_then_decide`).
For bounded reasonable binary costs, measurable scores and calibrator, and
a.e. recovery of the conditional posterior, the composite rule is measurable,
is calibrated in the conditional-law sense, attains Bayes risk, and agrees
a.e. with the smallest-index Bayes rule off the threshold tie set. The
posterior is represented by `Risk.ConditionalLaw`; its existence for every
data law is established in `ConditionalLawExists`. -/
@[thesis "prop:calibrate_then_decide"]
theorem calibrate_then_decide_integrated {X : Type*} [MeasurableSpace X]
    {P : Measure (X × Fin 2)} [IsProbabilityMeasure P]
    {η : X → Risk.Simplex 2} (hη : Measurable η) (hcond : Risk.ConditionalLaw P η)
    {C : CostMatrix 2} {Cmax : ℝ} (hR : Reasonable C) (hC : Bounded C Cmax)
    {g : Score → Score} (hg : Measurable g) {phat : X → Score} (hp : Measurable phat)
    (hcons : ∀ᵐ x ∂Risk.xMarginal P, (g (phat x)).val = (η x).val 1) :
    Measurable (calibratedClassifier C g phat) ∧
      PerfectCalibration P (fun x => g (phat x)) ∧
      Risk.jointRisk P C (calibratedClassifier C g phat) = Risk.bayesRisk P C ∧
      (∀ᵐ x ∂Risk.xMarginal P, (η x).val 1 ≠ elkanThreshold C →
        calibratedClassifier C g phat x = Risk.bayesClassifier C η x) := by
  have hcal := measurable_calibratedClassifier C hg hp
  have hstar := Risk.measurable_bayesClassifier C hη
  obtain ⟨_, _, _, _, hbayes⟩ := Risk.bayes_risk_identity hη hcond hC
  have heq : ∀ᵐ x ∂Risk.xMarginal P,
      condRisk C (η x).val (calibratedClassifier C g phat x) =
        condRisk C (η x).val (Risk.bayesClassifier C η x) := by
    filter_upwards [hcons] with x hx
    have hmin : ∀ j, condRisk C (η x).val (calibratedClassifier C g phat x)
        ≤ condRisk C (η x).val j := by
      intro j
      rw [← probVec_second_eq (η x)]
      unfold calibratedClassifier
      rw [hx]
      exact optimalThreshold_elkan hR ((η x).val 1) (second_probability_mem _) j
    exact le_antisymm (hmin _) (lexArgmin_le _ _)
  refine ⟨hcal, perfect_calibration_of_posterior_consistency hcond (hg.comp hp) hcons,
    ?_, ?_⟩
  · calc
      Risk.jointRisk P C (calibratedClassifier C g phat)
          = Risk.jointRisk P C (Risk.bayesClassifier C η) := by
        rw [Risk.jointRisk_eq_condRisk hcond hC hcal,
          Risk.jointRisk_eq_condRisk hcond hC hstar]
        exact integral_congr_ae heq
      _ = Risk.bayesRisk P C := hbayes
  · filter_upwards [hcons] with x hx
    intro hne
    unfold calibratedClassifier Risk.bayesClassifier
    rw [hx, ← probVec_second_eq (η x)]
    exact (calibrate_then_decide hR id
      (fun _ : Unit => (η x).val 1) (fun _ : Unit => (η x).val 1)
      (fun _ => second_probability_mem _) (fun _ => rfl)).2 () hne

/-- The literal rule-equality claim fails even under exact posterior scores:
at 1/2 under 0/1 loss, >= thresholding chooses 1 and lexArgmin chooses 0. -/
theorem calibrate_then_decide_tie_counterexample :
    let C : CostMatrix 2 := Matrix.of ![![0, 1], ![1, 0]]
    thresholdRule (elkanThreshold C) (1 / 2) = 1 ∧
      bayesRule C (probVec (1 / 2)) = 0 := by
  dsimp
  constructor
  · rw [elkanThreshold_zeroOne]
    simp [thresholdRule]
  · have hle : bayesRule (Matrix.of ![![0, 1], ![1, 0]] : CostMatrix 2)
        (probVec (1 / 2)) ≤ (0 : Fin 2) := by
      apply lexArgmin_le_index
      intro j
      rcases fin2_cases j with rfl | rfl <;> norm_num [condRisk, probVec]
    apply Fin.ext
    exact Nat.eq_zero_of_le_zero hle
end ThresholdOpt.Binary
