import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Constructions.BorelSpace.Real
import ThresholdOpt.Basic

/-!
# Integrated risk identities (Chapter 1)

The probability law lives on X × Fin K, and eta is a measurable map into the
probability simplex. ConditionalLaw explicitly supplies the standard general
disintegration identity for bounded measurable functions. It is independent of
cost matrices and decision rules, and does not assume either result proved here.

The smallest-index argmin is measurable by its finite weak/strict comparison
level sets, including ties. Bounded costs and simplex-valued probabilities give
integrability. The Bayes rule attains the minimum among all measurable classifiers;
the plug-in rule's excess joint risk is at most Cmax times its expected L1 error.
-/

namespace ThresholdOpt.Risk

open ThresholdOpt Finset

variable {K : ℕ}

/-- The `ℓ¹` distance between two probability vectors. -/
def l1 (p q : Fin K → ℝ) : ℝ := ∑ i, |p i - q i|

theorem l1_nonneg (p q : Fin K → ℝ) : 0 ≤ l1 p q :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- The risk of a *fixed* action changes by at most `Cmax · ‖p - q‖₁` when the
probability vector moves. This is the Hölder step
`|a·b| ≤ ‖a‖_∞ ‖b‖₁` of the proof, specialized to a cost column. -/
theorem condRisk_sub_le {C : CostMatrix K} {Cmax : ℝ} (hC : Bounded C Cmax)
    (p q : Fin K → ℝ) (k : Fin K) :
    condRisk C p k - condRisk C q k ≤ Cmax * l1 p q := by
  have expand : condRisk C p k - condRisk C q k = ∑ i, C i k * (p i - q i) := by
    simp only [condRisk, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [expand, l1, Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  calc C i k * (p i - q i)
      ≤ C i k * |p i - q i| := by
        exact mul_le_mul_of_nonneg_left (le_abs_self _) (hC.nonneg i k)
    _ ≤ Cmax * |p i - q i| :=
        mul_le_mul_of_nonneg_right (hC.le_max i k) (abs_nonneg _)

/-- **Pointwise excess-risk bound** (`prop:excess_risk_bound`).

Let `khat = argmin 𝓛(·, phat)` be the plug-in action and `k* = argmin 𝓛(·, η)` the
Bayes action. Then the excess conditional risk of `khat` under the *true* `η` is
at most `Cmax · ‖phat - η‖₁`.

The proof adds and subtracts the risk of each action under the other's
probability vector; the middle term is nonpositive because `khat` minimizes the
risk under `phat`. -/

theorem excess_risk_le [NeZero K] {C : CostMatrix K} {Cmax : ℝ}
    (hC : Bounded C Cmax) (phat η : Fin K → ℝ) :
    condRisk C η (bayesRule C phat) - condRisk C η (bayesRule C η)
      ≤ Cmax * l1 phat η := by
  set khat := bayesRule C phat
  set ks := bayesRule C η
  have hmid : condRisk C phat khat - condRisk C phat ks ≤ 0 :=
    sub_nonpos.2 (lexArgmin_le (condRisk C phat) ks)
  have hexpand : condRisk C η khat - condRisk C η ks =
      (∑ i, (C i khat - C i ks) * (η i - phat i)) +
        (condRisk C phat khat - condRisk C phat ks) := by
    simp only [condRisk, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  have hterm : ∀ i, (C i khat - C i ks) * (η i - phat i)
      ≤ Cmax * |η i - phat i| := by
    intro i
    have hcost : |C i khat - C i ks| ≤ Cmax := by
      apply abs_le.2
      constructor <;> linarith [hC.nonneg i khat, hC.nonneg i ks,
        hC.le_max i khat, hC.le_max i ks]
    calc
      (C i khat - C i ks) * (η i - phat i)
          ≤ |(C i khat - C i ks) * (η i - phat i)| := le_abs_self _
      _ = |C i khat - C i ks| * |η i - phat i| := abs_mul _ _
      _ ≤ Cmax * |η i - phat i| :=
        mul_le_mul_of_nonneg_right hcost (abs_nonneg _)
  have hsum : (∑ i, (C i khat - C i ks) * (η i - phat i))
      ≤ Cmax * l1 η phat := by
    simpa only [l1, Finset.mul_sum] using Finset.sum_le_sum (fun i _ => hterm i)
  have hsymm : l1 η phat = l1 phat η := by
    exact Finset.sum_congr rfl fun i _ => abs_sub_comm _ _
  rw [hsymm] at hsum
  rw [hexpand]
  linarith
/-- The Bayes action attains the pointwise minimum risk, so the excess risk is
nonnegative. Together with `excess_risk_le` this brackets the plug-in rule. -/
theorem excess_risk_nonneg [NeZero K] (C : CostMatrix K) (phat η : Fin K → ℝ) :
    0 ≤ condRisk C η (bayesRule C phat) - condRisk C η (bayesRule C η) :=
  sub_nonneg.2 (lexArgmin_le (condRisk C η) _)

/-- **The Bayes rule attains the pointwise minimum** (the pointwise half of
`prop:bayes_risk_identity`).

The integrated identity `R(h*) = E[min_ŷ 𝓡(ŷ | X)]` follows from this by
monotonicity of the integral, once `h*` is known to be measurable. -/

theorem bayes_attains_min [NeZero K] (C : CostMatrix K) (p : Fin K → ℝ) :
    condRisk C p (bayesRule C p) = ⨅ k, condRisk C p k := by
  refine le_antisymm ?_ ?_
  · exact le_ciInf fun k => lexArgmin_le (condRisk C p) k
  · exact ciInf_le (Finite.bddBelow_range _) _


/-! ### Measurability and simplex bounds -/

/-- Probability vectors, with the measurable-space structure inherited from ℝ^K. -/
abbrev Simplex (K : ℕ) := {p : Fin K → ℝ // p ∈ stdSimplex ℝ (Fin K)}

/-- The smallest-index argmin selects i exactly when it weakly beats every
score and strictly beats every lower-index score. These finite Boolean
conditions include the tie sets in Appendix A. -/
theorem lexArgmin_eq_iff [NeZero K] (f : Fin K → ℝ) (i : Fin K) :
    lexArgmin f = i ↔ (∀ j, f i ≤ f j) ∧ (∀ j, j < i → f i < f j) := by
  constructor
  · intro hi
    have hmin : ∀ j, f i ≤ f j := fun j => by
      simpa only [hi] using lexArgmin_le f j
    refine ⟨hmin, fun j hj => ?_⟩
    by_contra hnot
    have hminj : ∀ k, f j ≤ f k := fun k => (not_lt.1 hnot).trans (hmin k)
    have hindex : i ≤ j := by simpa only [hi] using lexArgmin_le_index hminj
    exact (not_le_of_gt hj) hindex
  · rintro ⟨hmin, hstrict⟩
    refine le_antisymm (lexArgmin_le_index hmin) ?_
    by_contra hnot
    have hlt : lexArgmin f < i := not_le.1 hnot
    exact (not_lt_of_ge (lexArgmin_le f i)) (hstrict _ hlt)

/-- Finite real score vectors with measurable coordinates have a measurable
smallest-index argmin, without any tie-free hypothesis. -/
theorem measurable_lexArgmin [NeZero K] {X : Type*} [MeasurableSpace X]
    (f : X → Fin K → ℝ) (hf : ∀ i, Measurable (fun x => f x i)) :
    Measurable (fun x => lexArgmin (f x)) := by
  classical
  apply measurable_to_countable'
  intro i
  change MeasurableSet {x | lexArgmin (f x) = i}
  have hset : {x | lexArgmin (f x) = i} =
      {x | (∀ j, f x i ≤ f x j) ∧ (∀ j, j < i → f x i < f x j)} := by
    ext x
    exact lexArgmin_eq_iff (f x) i
  rw [hset]
  simp only [Set.ofPred_and, Set.ofPred_forall]
  refine (MeasurableSet.iInter fun j => measurableSet_le (hf i) (hf j)).inter ?_
  exact MeasurableSet.iInter fun j =>
    MeasurableSet.iInter fun (_ : j < i) => measurableSet_lt (hf i) (hf j)
theorem measurable_condRisk {X : Type*} [MeasurableSpace X]
    (C : CostMatrix K) {p : X → Fin K → ℝ} (hp : Measurable p) (k : Fin K) :
    Measurable (fun x => condRisk C (p x) k) := by
  exact Finset.measurable_sum Finset.univ fun i _ =>
    measurable_const.mul ((measurable_pi_apply i).comp hp)

theorem measurable_bayesRule [NeZero K] {X : Type*} [MeasurableSpace X]
    (C : CostMatrix K) {p : X → Fin K → ℝ} (hp : Measurable p) :
    Measurable (fun x => bayesRule C (p x)) :=
  measurable_lexArgmin _ (measurable_condRisk C hp)

theorem measurable_condRisk_action {X : Type*} [MeasurableSpace X]
    (C : CostMatrix K) {p : X → Fin K → ℝ} (hp : Measurable p)
    {h : X → Fin K} (hh : Measurable h) :
    Measurable (fun x => condRisk C (p x) (h x)) := by
  exact Finset.measurable_sum Finset.univ fun i _ =>
    ((measurable_of_countable (fun j : Fin K => C i j)).comp hh).mul
      ((measurable_pi_apply i).comp hp)

theorem measurable_l1 {X : Type*} [MeasurableSpace X]
    {p q : X → Fin K → ℝ} (hp : Measurable p) (hq : Measurable q) :
    Measurable (fun x => l1 (p x) (q x)) := by
  exact Finset.measurable_sum Finset.univ fun i _ =>
    continuous_abs.measurable.comp
      (((measurable_pi_apply i).comp hp).sub ((measurable_pi_apply i).comp hq))

theorem l1_le_two {p q : Fin K → ℝ}
    (hp : p ∈ stdSimplex ℝ (Fin K)) (hq : q ∈ stdSimplex ℝ (Fin K)) :
    l1 p q ≤ 2 := by
  calc
    l1 p q ≤ ∑ i, (p i + q i) := by
      apply Finset.sum_le_sum
      intro i _
      simpa only [sub_zero, zero_sub, abs_neg, abs_of_nonneg (hp.1 i),
        abs_of_nonneg (hq.1 i)] using abs_sub_le (p i) 0 (q i)
    _ = 2 := by rw [Finset.sum_add_distrib, hp.2, hq.2]; norm_num

theorem condRisk_mem_Icc {C : CostMatrix K} {Cmax : ℝ} (hC : Bounded C Cmax)
    {p : Fin K → ℝ} (hp : p ∈ stdSimplex ℝ (Fin K)) (k : Fin K) :
    condRisk C p k ∈ Set.Icc (0 : ℝ) Cmax := by
  constructor
  · exact Finset.sum_nonneg fun i _ => mul_nonneg (hC.nonneg i k) (hp.1 i)
  · calc
      condRisk C p k ≤ ∑ i, Cmax * p i :=
        Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hC.le_max i k) (hp.1 i)
      _ = Cmax := by rw [← Finset.mul_sum, hp.2, mul_one]

/-! ### Joint probability law and its conditional distribution -/

open MeasureTheory

section Integrated

variable {X : Type*} [MeasurableSpace X]

/-- The actual X marginal of the joint data law. -/
noncomputable def xMarginal (P : Measure (X × Fin K)) : Measure X := P.map Prod.fst

instance xMarginal_probability (P : Measure (X × Fin K)) [IsProbabilityMeasure P] :
    IsProbabilityMeasure (xMarginal P) :=
  Measure.isProbabilityMeasure_map measurable_fst.aemeasurable

/-- General disintegration / total-expectation characterization of eta being
P(Y=k | X=x). This standard background is explicit, and quantifies over every
bounded measurable joint function, independently of costs or decision rules. -/
def ConditionalLaw (P : Measure (X × Fin K)) (η : X → Simplex K) : Prop :=
  ∀ f : X × Fin K → ℝ, Measurable f → (∃ M : ℝ, ∀ z, |f z| ≤ M) →
    (∫ z, f z ∂P) = ∫ x, ∑ k, (η x).val k * f (x, k) ∂xMarginal P

/-- Dissertation risk, with the truth as the row index of the cost matrix. -/
noncomputable def jointRisk (P : Measure (X × Fin K)) (C : CostMatrix K)
    (h : X → Fin K) : ℝ := ∫ z, C z.2 (h z.1) ∂P

/-- Risks of all measurable classifiers on the original input space. -/
def measurableRisks (P : Measure (X × Fin K)) (C : CostMatrix K) : Set ℝ :=
  {r | ∃ h : X → Fin K, Measurable h ∧ jointRisk P C h = r}

noncomputable def bayesRisk (P : Measure (X × Fin K)) (C : CostMatrix K) : ℝ :=
  sInf (measurableRisks P C)

noncomputable def bayesClassifier [NeZero K] (C : CostMatrix K)
    (η : X → Simplex K) : X → Fin K := fun x => bayesRule C (η x).val

theorem measurable_bayesClassifier [NeZero K] (C : CostMatrix K)
    {η : X → Simplex K} (hη : Measurable η) : Measurable (bayesClassifier C η) :=
  measurable_bayesRule C (measurable_subtype_coe.comp hη)

theorem measurable_costLoss (C : CostMatrix K) {h : X → Fin K} (hh : Measurable h) :
    Measurable (fun z : X × Fin K => C z.2 (h z.1)) :=
  (measurable_of_countable (fun q : Fin K × Fin K => C q.1 q.2)).comp
    (measurable_snd.prodMk (hh.comp measurable_fst))

/-- Bounded costs make the original joint-law loss integrable for every
measurable classifier, so its Bochner integral is an actual finite risk. -/
theorem integrable_costLoss {P : Measure (X × Fin K)} [IsFiniteMeasure P]
    {C : CostMatrix K} {Cmax : ℝ} (hC : Bounded C Cmax)
    {h : X → Fin K} (hh : Measurable h) :
    Integrable (fun z : X × Fin K => C z.2 (h z.1)) P := by
  constructor
  · exact (measurable_costLoss C hh).aestronglyMeasurable
  · apply HasFiniteIntegral.of_bounded
    exact Filter.Eventually.of_forall fun z => by
      rw [Real.norm_eq_abs, abs_of_nonneg (hC.nonneg _ _)]
      exact hC.le_max _ _
/-- The conditional distribution gives the risk decomposition for any
measurable classifier. Bayes optimality is not used here. -/
theorem jointRisk_eq_condRisk {P : Measure (X × Fin K)} {η : X → Simplex K}
    (hcond : ConditionalLaw P η) {C : CostMatrix K} {Cmax : ℝ}
    (hC : Bounded C Cmax) {h : X → Fin K} (hh : Measurable h) :
    jointRisk P C h = ∫ x, condRisk C (η x).val (h x) ∂xMarginal P := by
  have hbound : ∃ M : ℝ, ∀ z : X × Fin K, |C z.2 (h z.1)| ≤ M :=
    ⟨Cmax, fun z => by rw [abs_of_nonneg (hC.nonneg _ _)]; exact hC.le_max _ _⟩
  rw [jointRisk, hcond _ (measurable_costLoss C hh) hbound]
  apply integral_congr_ae
  exact Filter.Eventually.of_forall fun x =>
    Finset.sum_congr rfl fun k _ => mul_comm _ _

theorem integrable_condRisk {μ : Measure X} [IsFiniteMeasure μ]
    {η : X → Simplex K} (hη : Measurable η) {C : CostMatrix K} {Cmax : ℝ}
    (hC : Bounded C Cmax) {h : X → Fin K} (hh : Measurable h) :
    Integrable (fun x => condRisk C (η x).val (h x)) μ := by
  constructor
  · exact (measurable_condRisk_action C (measurable_subtype_coe.comp hη) hh).aestronglyMeasurable
  · apply HasFiniteIntegral.of_bounded
    exact Filter.Eventually.of_forall fun x => by
      have hbounds := condRisk_mem_Icc hC (η x).property (h x)
      rw [Real.norm_eq_abs, abs_of_nonneg hbounds.1]
      exact hbounds.2

theorem integrable_l1 {μ : Measure X} [IsFiniteMeasure μ]
    {p q : X → Simplex K} (hp : Measurable p) (hq : Measurable q) :
    Integrable (fun x => l1 (p x).val (q x).val) μ := by
  constructor
  · exact (measurable_l1 (measurable_subtype_coe.comp hp)
      (measurable_subtype_coe.comp hq)).aestronglyMeasurable
  · apply HasFiniteIntegral.of_bounded
    exact Filter.Eventually.of_forall fun x => by
      rw [Real.norm_eq_abs, abs_of_nonneg (l1_nonneg _ _)]
      exact l1_le_two (p x).property (q x).property

/-- The integrated Bayes identity: a measurable smallest-index minimizer
attains both the expectation of the pointwise minimum and the minimum over
all measurable classifiers. The only conditional-probability background is
ConditionalLaw; no optimality statement is assumed. -/
@[thesis "prop:bayes_risk_identity"]
theorem bayes_risk_identity [NeZero K] {P : Measure (X × Fin K)}
    [IsProbabilityMeasure P] {η : X → Simplex K} (hη : Measurable η)
    (hcond : ConditionalLaw P η) {C : CostMatrix K} {Cmax : ℝ}
    (hC : Bounded C Cmax) :
    Measurable (bayesClassifier C η) ∧
      Integrable (fun z : X × Fin K => C z.2 (bayesClassifier C η z.1)) P ∧
      jointRisk P C (bayesClassifier C η) =
        (∫ x, ⨅ k, condRisk C (η x).val k ∂xMarginal P) ∧
      IsLeast (measurableRisks P C) (jointRisk P C (bayesClassifier C η)) ∧
      jointRisk P C (bayesClassifier C η) = bayesRisk P C := by
  have hstar := measurable_bayesClassifier C hη
  have hleast : IsLeast (measurableRisks P C) (jointRisk P C (bayesClassifier C η)) := by
    refine ⟨⟨bayesClassifier C η, hstar, rfl⟩, ?_⟩
    rintro r ⟨h, hh, rfl⟩
    rw [jointRisk_eq_condRisk hcond hC hstar, jointRisk_eq_condRisk hcond hC hh]
    apply integral_mono (integrable_condRisk hη hC hstar) (integrable_condRisk hη hC hh)
    intro x
    exact lexArgmin_le _ _
  refine ⟨hstar, integrable_costLoss hC hstar, ?_, hleast, hleast.csInf_eq.symm⟩
  rw [jointRisk_eq_condRisk hcond hC hstar]
  apply integral_congr_ae
  exact Filter.Eventually.of_forall fun x => bayes_attains_min C (η x).val

/-- The integrated bound has exactly the constant Cmax in the text, and also
records the nonnegativity of excess risk. Both risks are joint-law risks. -/
@[thesis "prop:excess_risk_bound"]
theorem excess_risk_bound [NeZero K] {P : Measure (X × Fin K)}
    [IsProbabilityMeasure P] {η phat : X → Simplex K}
    (hη : Measurable η) (hphat : Measurable phat) (hcond : ConditionalLaw P η)
    {C : CostMatrix K} {Cmax : ℝ} (hC : Bounded C Cmax) :
    0 ≤ jointRisk P C (bayesClassifier C phat) - jointRisk P C (bayesClassifier C η) ∧
      jointRisk P C (bayesClassifier C phat) - jointRisk P C (bayesClassifier C η)
        ≤ Cmax * ∫ x, l1 (phat x).val (η x).val ∂xMarginal P := by
  have hhat := measurable_bayesClassifier C hphat
  have hstar := measurable_bayesClassifier C hη
  rw [jointRisk_eq_condRisk hcond hC hhat, jointRisk_eq_condRisk hcond hC hstar]
  rw [← integral_sub (integrable_condRisk hη hC hhat) (integrable_condRisk hη hC hstar)]
  constructor
  · exact integral_nonneg fun x => excess_risk_nonneg C (phat x).val (η x).val
  · calc
      (∫ x, condRisk C (η x).val (bayesClassifier C phat x) -
          condRisk C (η x).val (bayesClassifier C η x) ∂xMarginal P)
          ≤ ∫ x, Cmax * l1 (phat x).val (η x).val ∂xMarginal P :=
        integral_mono ((integrable_condRisk hη hC hhat).sub
          (integrable_condRisk hη hC hstar)) ((integrable_l1 hphat hη).const_mul Cmax)
          (fun x => excess_risk_le hC (phat x).val (η x).val)
      _ = Cmax * ∫ x, l1 (phat x).val (η x).val ∂xMarginal P := integral_const_mul _ _

end Integrated
end ThresholdOpt.Risk
