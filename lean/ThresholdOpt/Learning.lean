import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import ThresholdOpt.Basic

/-!
# The learning setting: population risk, empirical risk, and the sample draw

The objects the generalization bounds of Chapter 1 quantify over
(`prop:gen_bound`): a distribution on labelled examples, the risk of a decision
rule under it, the empirical risk on a validation sample, and the law of the
sample itself.

Nothing here is used by the algebraic and geometric results, which are all
pointwise. It exists so that the imported uniform-convergence bound of
[`Literature.lean`](Literature.lean) can be stated with the quantifier the
dissertation states it with: *with probability at least `1 - ν` over the random
draw of the validation sample*.

That quantifier is not decoration. An earlier version of this development
stated the bound as an unconditional inequality between two unrelated functions
`risk` and `empRisk`, which is refutable: instantiate `risk` above the bound and
`empRisk` at zero. An axiom that proves `False` makes every theorem downstream
of it worthless, so the probabilistic content has to be carried even though none
of it is proved here.
-/

namespace ThresholdOpt.Learning

open MeasureTheory

variable {Z : Type*} [MeasurableSpace Z]

/-- The population risk of a loss under the data distribution, `R = E[ℓ]`. -/
noncomputable def popRisk (μ : Measure Z) (ℓ : Z → ℝ) : ℝ := ∫ z, ℓ z ∂μ

/-- The empirical risk on a validation sample of size `m`,
`R̂ = (1/m) ∑ ℓ(z_i)` (`eq:empirical_risk_obj`). -/
noncomputable def empRisk {m : ℕ} (S : Fin m → Z) (ℓ : Z → ℝ) : ℝ :=
  (∑ i, ℓ (S i)) / m

/-- The law of an i.i.d. sample of size `m`: the `m`-fold product of the data
distribution. Statements "with probability at least `1 - ν` over the draw of
the validation sample" are statements about this measure. -/
noncomputable def sampleMeasure (μ : Measure Z) (m : ℕ) : Measure (Fin m → Z) :=
  Measure.pi fun _ => μ

instance (μ : Measure Z) [IsProbabilityMeasure μ] (m : ℕ) :
    IsProbabilityMeasure (sampleMeasure μ m) := by
  unfold sampleMeasure; infer_instance

/-- The `0/1` loss of a binary hypothesis on a labelled example. -/
def zeroOneLoss {X : Type*} (h : X → Bool) (z : X × Bool) : ℝ :=
  if h z.1 = z.2 then 0 else 1

theorem zeroOneLoss_mem_Icc {X : Type*} (h : X → Bool) (z : X × Bool) :
    zeroOneLoss h z ∈ Set.Icc (0 : ℝ) 1 := by
  unfold zeroOneLoss
  split <;> constructor <;> norm_num

/-- The cost-sensitive loss of a `K`-class decision rule on a labelled example,
`ℓ_δ(z, y) = C_{y, δ(z)}` (`eq:decision_loss`). -/
def costLoss {X : Type*} {K : ℕ} (C : CostMatrix K) (h : X → Fin K)
    (z : X × Fin K) : ℝ :=
  C z.2 (h z.1)

theorem costLoss_mem_Icc {X : Type*} {K : ℕ} {C : CostMatrix K} {Cmax : ℝ}
    (hC : Bounded C Cmax) (h : X → Fin K) (z : X × Fin K) :
    costLoss C h z ∈ Set.Icc (0 : ℝ) Cmax :=
  ⟨hC.nonneg _ _, hC.le_max _ _⟩

/-! ### The explicit rate

The rate of Mohri, Rostamizadeh and Talwalkar, Corollary 3.19, with its
constants and its two logarithmic terms as displayed in the source
(eq. 3.29). -/

/-- `√(2 d log(e m / d) / m) + √(log(1/ν) / (2m))`. -/
noncomputable def vcRate (m d : ℕ) (ν : ℝ) : ℝ :=
  Real.sqrt (2 * d * Real.log (Real.exp 1 * m / d) / m)
    + Real.sqrt (Real.log (1 / ν) / (2 * m))

theorem vcRate_nonneg (m d : ℕ) (ν : ℝ) : 0 ≤ vcRate m d ν :=
  add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)

/-- `log(e m / d) ≥ 1` once `m ≥ d ≥ 1`. -/
theorem one_le_log_exp_mul_div {d m : ℕ} (hd : 0 < d) (hdm : d ≤ m) :
    1 ≤ Real.log (Real.exp 1 * m / d) := by
  have hdpos : (0 : ℝ) < d := by exact_mod_cast hd
  have hdmR : (d : ℝ) ≤ m := by exact_mod_cast hdm
  have hmpos : (0 : ℝ) < m := lt_of_lt_of_le hdpos hdmR
  rw [Real.le_log_iff_exp_le (by positivity), le_div_iff₀ hdpos]
  exact mul_le_mul_of_nonneg_left hdmR (Real.exp_pos 1).le

/-- `d ↦ d log(e m / d)` is nondecreasing on `d ≤ m`. This is what lets a
bound stated for a class of VC dimension exactly `d` be used for a class of VC
dimension at most `d`. -/
theorem mul_log_mono {d₀ d m : ℕ} (h : d₀ ≤ d) (hdm : d ≤ m) :
    (d₀ : ℝ) * Real.log (Real.exp 1 * m / d₀)
      ≤ d * Real.log (Real.exp 1 * m / d) := by
  rcases Nat.eq_zero_or_pos d with rfl | hd
  · have : d₀ = 0 := Nat.le_zero.1 h
    subst this
    simp
  have hL := one_le_log_exp_mul_div hd hdm
  have hdpos : (0 : ℝ) < d := by exact_mod_cast hd
  have hdmR : (d : ℝ) ≤ m := by exact_mod_cast hdm
  have hmpos : (0 : ℝ) < m := lt_of_lt_of_le hdpos hdmR
  rcases Nat.eq_zero_or_pos d₀ with rfl | h0
  · simp only [Nat.cast_zero, zero_mul]
    exact mul_nonneg hdpos.le (by linarith)
  · have h0pos : (0 : ℝ) < d₀ := by exact_mod_cast h0
    have hdd : (d₀ : ℝ) ≤ d := by exact_mod_cast h
    have hsplit : Real.log (Real.exp 1 * m / d₀)
        = Real.log (Real.exp 1 * m / d) + Real.log ((d : ℝ) / d₀) := by
      rw [← Real.log_mul (by positivity) (by positivity)]
      congr 1
      field_simp
    have hlog : Real.log ((d : ℝ) / d₀) ≤ d / d₀ - 1 :=
      Real.log_le_sub_one_of_pos (by positivity)
    have hmul : (d₀ : ℝ) * Real.log ((d : ℝ) / d₀) ≤ d - d₀ := by
      calc (d₀ : ℝ) * Real.log ((d : ℝ) / d₀)
          ≤ d₀ * ((d : ℝ) / d₀ - 1) := mul_le_mul_of_nonneg_left hlog h0pos.le
        _ = d - d₀ := by field_simp
    rw [hsplit]
    nlinarith

theorem vcRate_mono {d₀ d m : ℕ} (h : d₀ ≤ d) (hdm : d ≤ m) (ν : ℝ) :
    vcRate m d₀ ν ≤ vcRate m d ν := by
  refine add_le_add ?_ le_rfl
  refine Real.sqrt_le_sqrt ?_
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg m)
  have := mul_log_mono h hdm
  linarith

/-! ### Measure-theoretic plumbing

What is needed to move a uniform-convergence statement from one labelled
problem to another: relabelling commutes with drawing the sample, a finite
intersection of high-probability events has high probability, and a
finite-valued nonnegative loss is a nonnegative combination of indicators of
its upper level sets. -/

/-- Relabelling the examples and then sampling is sampling and then relabelling
each example. -/
theorem sampleMeasure_map {W : Type*} [MeasurableSpace W] (μ : Measure Z)
    [IsProbabilityMeasure μ] {f : Z → W} (hf : Measurable f) (m : ℕ) :
    sampleMeasure (μ.map f) m
      = (sampleMeasure μ m).map (fun S i => f (S i)) := by
  have : IsProbabilityMeasure (μ.map f) :=
    Measure.isProbabilityMeasure_map hf.aemeasurable
  unfold sampleMeasure
  exact (Measure.pi_map_pi (μ := fun _ : Fin m => μ) (f := fun _ => f)
    (fun _ => hf.aemeasurable)).symm

theorem measurable_empRisk {m : ℕ} {ℓ : Z → ℝ} (hℓ : Measurable ℓ) :
    Measurable (fun S : Fin m → Z => empRisk S ℓ) := by
  unfold empRisk
  exact (Finset.measurable_sum _ fun i _ =>
    hℓ.comp (measurable_pi_apply i)).div_const _

/-- **Union bound.** Finitely many measurable events, each of probability at
least `1 - η`, hold simultaneously with probability at least `1 - n η`. -/
theorem prob_iInter_ge {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] {ι : Type*} [Fintype ι] (A : ι → Set Ω)
    (hA : ∀ i, MeasurableSet (A i)) {η : ℝ}
    (h : ∀ i, 1 - ENNReal.ofReal η ≤ P (A i)) :
    1 - ENNReal.ofReal (Fintype.card ι * η) ≤ P (⋂ i, A i) := by
  have hc : ∀ i, P (A i)ᶜ ≤ ENNReal.ofReal η := fun i => by
    rw [prob_compl_eq_one_sub (hA i)]
    refine tsub_le_iff_right.2 ?_
    have := tsub_le_iff_right.1 (h i)
    rwa [add_comm] at this
  have hunion : P (⋂ i, A i)ᶜ ≤ ENNReal.ofReal (Fintype.card ι * η) := by
    rw [Set.compl_iInter]
    calc P (⋃ i, (A i)ᶜ) ≤ ∑ i, P (A i)ᶜ := measure_iUnion_fintype_le _ _
      _ ≤ ∑ _i : ι, ENNReal.ofReal η := Finset.sum_le_sum fun i _ => hc i
      _ = ENNReal.ofReal (Fintype.card ι * η) := by
          rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
            ENNReal.ofReal_mul (Nat.cast_nonneg _), ENNReal.ofReal_natCast]
  rw [prob_compl_eq_one_sub (MeasurableSet.iInter hA)] at hunion
  refine tsub_le_iff_right.2 ?_
  have := tsub_le_iff_right.1 hunion
  rwa [add_comm] at this

/-- **Layer weights.** For a finite set `V` of nonnegative reals there are
nonnegative weights `w` with `∑_{v ∈ V, v ≤ u} w_v = u` for every `u ∈ V`, and
total weight at most any nonnegative upper bound of `V`. Hence a function with
values in `V` is the combination `∑_v w_v 1[v ≤ f]` of the indicators of its
upper level sets, with total weight at most `max V`. -/
theorem exists_layer_weights (V : Finset ℝ) (hV : ∀ v ∈ V, 0 ≤ v) :
    ∃ w : ℝ → ℝ, (∀ v, 0 ≤ w v)
      ∧ (∀ u ∈ V, ∑ v ∈ V.filter (fun v => v ≤ u), w v = u)
      ∧ (∀ b, 0 ≤ b → (∀ v ∈ V, v ≤ b) → ∑ v ∈ V, w v ≤ b) := by
  classical
  induction V using Finset.induction_on_max with
  | empty => exact ⟨fun _ => 0, fun _ => le_rfl, by simp, by simp⟩
  | insert a s hlt ih =>
    obtain ⟨w, hw0, hwsum, hwtot⟩ :=
      ih fun v hv => hV v (Finset.mem_insert_of_mem hv)
    have ha0 : 0 ≤ a := hV a (Finset.mem_insert_self a s)
    have hT : ∑ v ∈ s, w v ≤ a := hwtot a ha0 fun v hv => (hlt v hv).le
    have has : a ∉ s := fun h => lt_irrefl a (hlt a h)
    have hcongr : ∀ t : Finset ℝ, t ⊆ s →
        ∑ v ∈ t, Function.update w a (a - ∑ v ∈ s, w v) v = ∑ v ∈ t, w v := by
      intro t ht
      refine Finset.sum_congr rfl fun v hv => ?_
      exact Function.update_of_ne (fun h : v = a => has (by rw [← h]; exact ht hv)) _ _
    have htotal : ∑ v ∈ insert a s,
        Function.update w a (a - ∑ v ∈ s, w v) v = a := by
      rw [Finset.sum_insert has, hcongr s (Finset.Subset.refl s),
        Function.update_self]
      ring
    refine ⟨Function.update w a (a - ∑ v ∈ s, w v), fun v => ?_, fun u hu => ?_,
      fun b _ hb => ?_⟩
    · by_cases hva : v = a
      · subst hva
        rw [Function.update_self]
        linarith
      · rw [Function.update_of_ne hva]
        exact hw0 v
    · rcases Finset.mem_insert.1 hu with rfl | hus
      · have : (insert u s).filter (fun v => v ≤ u) = insert u s := by
          refine Finset.filter_true_of_mem fun v hv => ?_
          rcases Finset.mem_insert.1 hv with rfl | hvs
          · exact le_rfl
          · exact (hlt v hvs).le
        rw [this, htotal]
      · have : (insert a s).filter (fun v => v ≤ u)
            = s.filter (fun v => v ≤ u) := by
          rw [Finset.filter_insert, if_neg (not_le.2 (hlt u hus))]
        rw [this, hcongr _ (Finset.filter_subset _ _)]
        exact hwsum u hus
    · rw [htotal]
      exact hb a (Finset.mem_insert_self a s)

end ThresholdOpt.Learning
