import Mathlib.MeasureTheory.Integral.DominatedConvergence
import ThresholdOpt.Binary
import ThresholdOpt.Capacity
import ThresholdOpt.Learning
import ThresholdOpt.NatarajanLemma
import ThresholdOpt.UniformConvergence

/-!
# Generalization results from the literature

The dissertation cites uniform convergence, pseudo-dimension, and multiclass
linear-score capacity results. All three are proved in this development.
No literature axiom remains: `uniform_convergence_vc` is a theorem derived
from McDiarmid's inequality, symmetrization, Massart's lemma and Sauer's lemma
in `UniformConvergence.lean`.

This module derives binary and multiclass cost-sensitive bounds through loss
level sets. `PointwiseApprox.lean` extends them to classes approximated by
countable measurable classes, and `DecisionClassBounds.lean` supplies such
approximations for the real-parameter decision families. `SeparableBounds.lean`
matches the binary proposition's explicit rate and approximation hypothesis.
The audit rejects incomplete proofs and the renderer rejects unregistered
axioms; the current proof closures contain only Lean's standard axioms.

The history below records why the former imported statement needed its
sampling, capacity, and class-scope hypotheses.

## An axiom must be at least as hard to state as it is to believe

An axiom is only as good as its statement. Weakening the hypotheses of a cited
theorem, or dropping a quantifier to avoid developing the objects it ranges
over, does not produce a weaker axiom: it usually produces a false one, and a
false axiom proves everything.

`uniform_convergence_vc` has now been wrong twice. It was first stated as an
unconditional inequality between two unrelated functions `risk` and `empRisk`,
the probabilistic quantifier of the source dropped; instantiating `risk` above
the bound and `empRisk` at zero refutes that. The second version carried the
sample measure but took its rate from `eq:vc_decision_rate` of the
dissertation, `√((d log(2em/d) + log(4/ν)) / m)`, which the text gives "up to
universal constants" and which is not the displayed inequality of the cited
corollary, and it omitted the condition `m ≥ d` under which the source derives
its bound. That version is refutable too: take a class with a single
hypothesis of risk `1/2`, `d = 100`, `m = 1`, `ν = 1/10`. Then
`d log(2em/d) + log(4/ν) < 0`, the square root is `0` in Lean, and the axiom
asserts that the empirical risk on one example is at least `1/2` with
probability `9/10`, while it is so with probability `1/2`. The present version
is the corollary as printed, checked against the book, and it is now proved.
-/

namespace ThresholdOpt.Literature

open ThresholdOpt ThresholdOpt.Capacity ThresholdOpt.Learning MeasureTheory

/-- **Uniform convergence for a VC class.**

Mohri, Rostamizadeh and Talwalkar, *Foundations of Machine Learning*, 2nd ed.
(MIT Press, 2018), Corollary 3.19, eq. (3.29): for a family `H` of binary
functions of VC dimension `d` and any `ν > 0`, with probability at least
`1 - ν` over an i.i.d. sample of size `m`, every `h ∈ H` satisfies

`R(h) ≤ R̂_S(h) + √(2 d log(e m / d) / m) + √(log(1/ν) / (2m))`.

Three hypotheses are explicit here that the printed statement leaves to its
context.

* `d ≤ m`. The corollary combines the growth-function bound of Corollary 3.9
  with Corollary 3.18, `Π_H(m) ≤ (e m / d)^d`, which the source states "for all
  `m ≥ d`".
* `H` is countable and its members are measurable. The source does not discuss
  measurability. Some condition is needed: for an uncountable class the event
  below need not be measurable, and under the continuum hypothesis there is a
  class of VC dimension `1` (the final segments of a well-order of `[0,1]` of
  type `ω₁`) for which it is empty. Countability is the simplest sufficient
  condition; it makes this formulation weaker than the usual reading of the source,
  not stronger.
* The VC dimension is exactly `d`, as in the source. The version for dimension
  at most `d` is `uniform_convergence_vc_le`, proved below.

Degenerate instances checked by hand: `H = ∅` (`d = 0`, the event is
everything); a single hypothesis (`d = 0`, the bound is the one-sided Hoeffding
inequality); `ν ≥ 1` (the left side is `0`); `m = d` (the rate is at least
`√2`, and risks lie in `[0, 1]`).

**Proved, no longer assumed.** This was an axiom. It is now
`UniformConvergence.uniform_convergence_vc_proved`, which needs only
`vcDim H ≤ d`, by the route of the book: Theorem 3.3 (McDiarmid's inequality
and symmetrization, `Symmetrization.rademacher_bound`), Lemma 3.4 and
Massart's lemma (Theorem 3.7, `Massart.massart`) applied to the loss vectors
centred at `1/2`, and Sauer's lemma with Corollary 3.18
(`Sauer.log_card_patterns_le`). The constants are those of eq. (3.29). The
countability hypothesis is weakened in
`UniformConvergence.uniform_convergence_vc_of_approx`: every member a
pointwise limit of a sequence from a countable class of measurable functions
of VC dimension at most `d`. -/
theorem uniform_convergence_vc
    {X : Type} [MeasurableSpace X] (H : Set (X → Bool))
    (hcount : H.Countable) (hmeas : ∀ h ∈ H, Measurable h)
    (μ : Measure (X × Bool)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : vcDim H = (d : ℕ∞)) (hdm : d ≤ m) (hm : 0 < m) (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ h ∈ H,
        popRisk μ (zeroOneLoss h) ≤ empRisk S (zeroOneLoss h) + vcRate m d ν} :=
  UniformConvergence.uniform_convergence_vc_proved H hcount hmeas μ d m ν hdim.le hdm hm hν

/-!
## Two results that are no longer imported

Anthony and Bartlett, Theorem 11.4 (the pseudo-dimension of a `W`-dimensional
space of real functions is at most `W`) and Daniely and Shalev-Shwartz,
Theorem 7 (the Natarajan dimension of a `W`-parameter linear-score class is at
most `W`) were axioms of this file. Both are now proved in
[`Capacity.lean`](Capacity.lean), as `Capacity.pseudoDim_le_dim` and
`Capacity.natDim_le_of_linear_scores`, by one sign argument on a linear
relation among more vectors than the dimension. Every statement that depended
on them is checked from Mathlib alone.

## Explicit multiclass factors

The committee-version generalization remark described a universal constant
absorbing logarithms in the sample size, confidence, and number of labels.
The corrected text gives those factors explicitly. `gen_bound_levels` uses
the VC dimension of loss level sets; `NatarajanLemma` bounds that dimension
from the Natarajan dimension of the rule class. The resulting theorem for a
countable class is `gen_bound_natarajan_clean`. The remark states the bound
for a class approximated pointwise by a countable one; that form, tagged for
`rem:natarajan_gen_bound`, is
`PointwiseApprox.gen_bound_natarajan_clean_of_approx_event`.
-/

/-! ## Consequences of proved uniform convergence -/

/-- **Uniform convergence for a class of VC dimension at most `d`.** The rate
of the source is nondecreasing in `d` on `d ≤ m`, so the bound for the exact
dimension implies the bound for any upper estimate of it. -/
theorem uniform_convergence_vc_le
    {X : Type} [MeasurableSpace X] (H : Set (X → Bool))
    (hcount : H.Countable) (hmeas : ∀ h ∈ H, Measurable h)
    (μ : Measure (X × Bool)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : vcDim H ≤ (d : ℕ∞)) (hdm : d ≤ m) (hm : 0 < m) (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ h ∈ H,
        popRisk μ (zeroOneLoss h) ≤ empRisk S (zeroOneLoss h) + vcRate m d ν} := by
  obtain ⟨d₀, hd₀⟩ : ∃ d₀ : ℕ, vcDim H = (d₀ : ℕ∞) := by
    have hne : vcDim H ≠ ⊤ := ne_top_of_le_ne_top (ENat.natCast_ne_top d) hdim
    exact ⟨(vcDim H).toNat, (ENat.natCast_toNat hne).symm⟩
  have hle : d₀ ≤ d := by
    rw [hd₀] at hdim
    exact_mod_cast hdim
  refine (uniform_convergence_vc H hcount hmeas μ d₀ m ν hd₀ (hle.trans hdm)
    hm hν).trans (measure_mono ?_)
  intro S hS h hh
  exact (hS h hh).trans (add_le_add le_rfl (vcRate_mono hle hdm ν))

/-- The binary class of a family of two-class decision rules: `x ↦ 1[δ(x) = 1]`.
Its VC dimension is the capacity `d_dec` of `prop:gen_bound` at `K = 2`. -/
def binClass {X : Type} (H : Set (X → Fin 2)) : Set (X → Bool) :=
  {h | ∃ δ ∈ H, h = fun x => decide (δ x = 1)}

/-- The upper level sets of the losses of a class of decision rules, at level
`v`: `(x, y) ↦ 1[v ≤ C_{y, δ(x)}]`. -/
def levelClass {X : Type} {K : ℕ} (C : CostMatrix K) (H : Set (X → Fin K)) (v : ℝ) :
    Set (X × Fin K → Bool) :=
  {g | ∃ δ ∈ H, g = fun z => decide (v ≤ C z.2 (δ z.1))}

/-- A level class is no richer than the class of rules: at each example the
level indicator is constant, `1[δ = 1]` or `1[δ = 0]`, so a shattered set of
examples projects injectively onto a shattered set of points. -/
theorem vcDim_levelClass_le {X : Type} (C : CostMatrix 2) (H : Set (X → Fin 2))
    (v : ℝ) : vcDim (levelClass C H v) ≤ vcDim (binClass H) := by
  classical
  refine vcDim_le fun s hs => ?_
  have fin2 : ∀ k : Fin 2, k = 0 ∨ k = 1 := by decide
  have key : ∀ b : Bool, b ≠ !b := by decide
  let φ : Fin 2 → Fin 2 → Bool := fun y k => decide (v ≤ C y k)
  -- At an example of a shattered set the level indicator depends on the rule.
  have hA : ∀ z ∈ s, φ z.2 0 ≠ φ z.2 1 := by
    intro z hz heq
    obtain ⟨g, ⟨δ, -, rfl⟩, hval⟩ := hs fun z' => !φ z'.2 0
    have hthis : φ z.2 (δ z.1) = !φ z.2 0 := hval z hz
    rcases fin2 (δ z.1) with h | h <;> rw [h] at hthis
    · exact key _ hthis
    · exact key _ (heq.trans hthis)
  -- Two examples of a shattered set do not share their point.
  have hB : Set.InjOn Prod.fst (s : Set (X × Fin 2)) := by
    rintro ⟨x, y⟩ hz ⟨x', y'⟩ hz' hxx
    have hxx' : x = x' := hxx
    subst hxx'
    by_contra hne
    have hy : y ≠ y' := fun h => hne (by rw [h])
    obtain ⟨g, ⟨δ, -, rfl⟩, hval⟩ :=
      hs fun z' => if z'.2 = y then φ y 0 else φ y' 1
    have e1 : φ y (δ x) = (if y = y then φ y 0 else φ y' 1) := hval (x, y) hz
    have e2 : φ y' (δ x) = (if y' = y then φ y 0 else φ y' 1) :=
      hval (x, y') hz'
    rw [if_pos rfl] at e1
    rw [if_neg (Ne.symm hy)] at e2
    rcases fin2 (δ x) with h | h <;> rw [h] at e1 e2
    · exact hA (x, y') hz' e2
    · exact hA (x, y) hz e1.symm
  have hsh : Shatters (binClass H) (s.image Prod.fst) := by
    intro b
    obtain ⟨g, ⟨δ, hδ, rfl⟩, hval⟩ :=
      hs fun z' => if b z'.1 = true then φ z'.2 1 else φ z'.2 0
    refine ⟨fun x => decide (δ x = 1), ⟨δ, hδ, rfl⟩, ?_⟩
    intro x hx
    obtain ⟨⟨x', y⟩, hz, rfl⟩ := Finset.mem_image.1 hx
    have e : φ y (δ x') = (if b x' = true then φ y 1 else φ y 0) :=
      hval (x', y) hz
    have hne := hA (x', y) hz
    rcases fin2 (δ x') with h | h
    · rw [h] at e
      have hb : b x' = false := by
        by_contra hb
        rw [Bool.not_eq_false] at hb
        rw [if_pos hb] at e
        exact hne e
      simp [h, hb]
    · rw [h] at e
      have hb : b x' = true := by
        by_contra hb
        rw [if_neg hb] at e
        exact hne e.symm
      simp [h, hb]
  calc (s.card : ℕ∞) = ((s.image Prod.fst).card : ℕ∞) := by
        rw [Finset.card_image_of_injOn hB]
    _ ≤ vcDim (binClass H) := le_vcDim hsh

/-- **Generalization bound for decision optimization, `K` classes**, with
the capacity of the loss level sets.

Let the costs lie in `[0, Cmax]`, let `H` be a countable class of measurable
rules `X → {0, …, K−1}`, and let `d ≤ m` bound the VC dimension of each class
of loss level sets `{(x, y) : v ≤ C_{y, δ(x)}}`, `v` an entry of `C`. Then with
probability at least `1 - ν` over `m` i.i.d. examples, *every* `δ ∈ H` satisfies

`R(δ) ≤ R̂(δ) + Cmax · (√(2 d log(e m / d) / m) + √(log(K²/ν) / (2m)))`.

Proved from `uniform_convergence_vc`: the loss takes at most `K²` values, so it
is a nonnegative combination, of total weight at most `Cmax`, of the indicators
of its upper level sets; the imported bound applies to each level after
relabelling every example with the label `0`; a union bound over the `K²`
entries turns `ν` into `ν / K²`.

The capacity here is the VC dimension of the loss level sets, not the
Natarajan dimension of `H`. `vcDim_levelClass_le_levelDim` bounds the first by
the second, which gives `gen_bound_natarajan` and the form of the remark,
`gen_bound_natarajan_clean`. For `K = 2` the level-set dimension is at most the
VC dimension of the class (`vcDim_levelClass_le`), which gives
`gen_bound_binary`. -/
theorem gen_bound_levels {X : Type} [MeasurableSpace X] {K : ℕ} [NeZero K]
    (C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H : Set (X → Fin K)) (hcount : H.Countable)
    (hmeas : ∀ δ ∈ H, Measurable δ)
    (μ : Measure (X × Fin K)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : ∀ e : Fin K × Fin K, vcDim (levelClass C H (C e.1 e.2)) ≤ (d : ℕ∞))
    (hdm : d ≤ m) (hm : 0 < m) (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ) + Cmax * vcRate m d (ν / ((K : ℝ) * K))} := by
  classical
  have hK : (0 : ℝ) < K := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne K)
  have hε : 0 ≤ vcRate m d (ν / ((K : ℝ) * K)) := vcRate_nonneg _ _ _
  have hCmax : 0 ≤ Cmax := (hC.nonneg 0 0).trans (hC.le_max 0 0)
  -- The indicator of the level set `{v ≤ loss}`.
  let ind : ℝ → (X → Fin K) → X × Fin K → ℝ :=
    fun v δ z => if v ≤ C z.2 (δ z.1) then 1 else 0
  have hind_meas : ∀ v, ∀ δ ∈ H, Measurable (ind v δ) := by
    intro v δ hδ
    exact (Measurable.of_discrete
      (f := fun q : Fin K × Fin K => if v ≤ C q.1 q.2 then (1 : ℝ) else 0)).comp
      (measurable_snd.prodMk ((hmeas δ hδ).comp measurable_fst))
  -- One event per matrix entry: uniform convergence at the level of that entry.
  let A : Fin K × Fin K → Set (Fin m → X × Fin K) := fun e =>
    {S | ∀ δ ∈ H, popRisk μ (ind (C e.1 e.2) δ)
      ≤ empRisk S (ind (C e.1 e.2) δ) + vcRate m d (ν / ((K : ℝ) * K))}
  have hAmeas : ∀ e, MeasurableSet (A e) := by
    intro e
    have : A e = ⋂ δ ∈ H, {S | popRisk μ (ind (C e.1 e.2) δ)
        ≤ empRisk S (ind (C e.1 e.2) δ) + vcRate m d (ν / ((K : ℝ) * K))} := by
      ext S
      simp [A]
    rw [this]
    exact MeasurableSet.biInter hcount fun δ hδ =>
      measurableSet_le measurable_const
        ((measurable_empRisk (hind_meas _ δ hδ)).add measurable_const)
  have hAprob : ∀ e, 1 - ENNReal.ofReal (ν / ((K : ℝ) * K)) ≤ sampleMeasure μ m (A e) := by
    intro e
    -- Relabel every example with the label `false`.
    let f : X × Fin K → (X × Fin K) × Bool := fun z => (z, false)
    have hf : Measurable f := measurable_id.prodMk measurable_const
    have hF : Measurable (fun (S : Fin m → X × Fin K) i => f (S i)) :=
      measurable_pi_lambda _ fun i => hf.comp (measurable_pi_apply i)
    have : IsProbabilityMeasure (μ.map f) :=
      Measure.isProbabilityMeasure_map hf.aemeasurable
    have hGcount : (levelClass C H (C e.1 e.2)).Countable := by
      refine (hcount.image fun δ => fun z : X × Fin K =>
        decide (C e.1 e.2 ≤ C z.2 (δ z.1))).mono ?_
      rintro g ⟨δ, hδ, rfl⟩
      exact ⟨δ, hδ, rfl⟩
    have hGmeas : ∀ g ∈ levelClass C H (C e.1 e.2), Measurable g := by
      rintro g ⟨δ, hδ, rfl⟩
      exact (Measurable.of_discrete
        (f := fun q : Fin K × Fin K => decide (C e.1 e.2 ≤ C q.1 q.2))).comp
        (measurable_snd.prodMk ((hmeas δ hδ).comp measurable_fst))
    have hloss_meas : ∀ g ∈ levelClass C H (C e.1 e.2),
        Measurable (zeroOneLoss g) := by
      intro g hg
      exact (Measurable.of_discrete
        (f := fun q : Bool × Bool => if q.1 = q.2 then (0 : ℝ) else 1)).comp
        (((hGmeas g hg).comp measurable_fst).prodMk measurable_snd)
    have key := uniform_convergence_vc_le (levelClass C H (C e.1 e.2)) hGcount
      hGmeas (μ.map f) d m (ν / ((K : ℝ) * K))
      (hdim e) hdm hm (by positivity)
    have hTmeas : MeasurableSet
        {S' : Fin m → (X × Fin K) × Bool |
          ∀ g ∈ levelClass C H (C e.1 e.2),
            popRisk (μ.map f) (zeroOneLoss g)
              ≤ empRisk S' (zeroOneLoss g) + vcRate m d (ν / ((K : ℝ) * K))} := by
      have : {S' : Fin m → (X × Fin K) × Bool |
          ∀ g ∈ levelClass C H (C e.1 e.2),
            popRisk (μ.map f) (zeroOneLoss g)
              ≤ empRisk S' (zeroOneLoss g) + vcRate m d (ν / ((K : ℝ) * K))}
          = ⋂ g ∈ levelClass C H (C e.1 e.2), {S' | popRisk (μ.map f) (zeroOneLoss g)
              ≤ empRisk S' (zeroOneLoss g) + vcRate m d (ν / ((K : ℝ) * K))} := by
        ext S'
        simp
      rw [this]
      exact MeasurableSet.biInter hGcount fun g hg =>
        measurableSet_le measurable_const
          ((measurable_empRisk (hloss_meas g hg)).add measurable_const)
    rw [sampleMeasure_map μ hf m, Measure.map_apply hF hTmeas] at key
    refine key.trans (measure_mono ?_)
    intro S hS δ hδ
    have hg : (fun z : X × Fin K => decide (C e.1 e.2 ≤ C z.2 (δ z.1)))
        ∈ levelClass C H (C e.1 e.2) := ⟨δ, hδ, rfl⟩
    have hpt : ∀ z : X × Fin K,
        zeroOneLoss (fun z : X × Fin K => decide (C e.1 e.2 ≤ C z.2 (δ z.1)))
          (f z) = ind (C e.1 e.2) δ z := by
      intro z
      by_cases hz : C e.1 e.2 ≤ C z.2 (δ z.1) <;> simp [zeroOneLoss, ind, f, hz]
    have h1 := hS _ hg
    have hpop : popRisk (μ.map f) (zeroOneLoss
        fun z : X × Fin K => decide (C e.1 e.2 ≤ C z.2 (δ z.1)))
        = popRisk μ (ind (C e.1 e.2) δ) := by
      unfold popRisk
      rw [integral_map hf.aemeasurable (hloss_meas _ hg).aestronglyMeasurable]
      exact integral_congr_ae (Filter.Eventually.of_forall hpt)
    have hemp : empRisk (fun i => f (S i)) (zeroOneLoss
        fun z : X × Fin K => decide (C e.1 e.2 ≤ C z.2 (δ z.1)))
        = empRisk S (ind (C e.1 e.2) δ) := by
      unfold empRisk
      congr 1
      exact Finset.sum_congr rfl fun i _ => hpt (S i)
    rw [hpop, hemp] at h1
    exact h1
  -- All `K²` events hold together with probability at least `1 - ν`.
  have hunion := prob_iInter_ge (sampleMeasure μ m) A hAmeas hAprob
  have hcard : (Fintype.card (Fin K × Fin K) : ℝ) * (ν / ((K : ℝ) * K)) = ν := by
    simp only [Fintype.card_prod, Fintype.card_fin, Nat.cast_mul]
    field_simp
  rw [hcard] at hunion
  refine hunion.trans (measure_mono ?_)
  -- On that event, the bound for the loss itself.
  intro S hS δ hδ
  rw [Set.mem_iInter] at hS
  let V : Finset ℝ := Finset.univ.image fun e : Fin K × Fin K => C e.1 e.2
  obtain ⟨w, hw0, hwsum, hwtot⟩ := exists_layer_weights V (by
    intro v hv
    obtain ⟨e, -, rfl⟩ := Finset.mem_image.1 hv
    exact hC.nonneg _ _)
  have hdecomp : costLoss C δ = fun z => ∑ v ∈ V, w v * ind v δ z := by
    funext z
    have hu : C z.2 (δ z.1) ∈ V :=
      Finset.mem_image.2 ⟨(z.2, δ z.1), Finset.mem_univ _, rfl⟩
    have := hwsum _ hu
    rw [Finset.sum_filter] at this
    simp only [costLoss, ind, mul_ite, mul_one, mul_zero]
    exact this.symm
  have hint : ∀ v, MeasureTheory.Integrable (ind v δ) μ := by
    intro v
    refine (integrable_const (1 : ℝ)).mono'
      (hind_meas v δ hδ).aestronglyMeasurable
      (Filter.Eventually.of_forall fun z => ?_)
    simp only [ind]
    split_ifs <;> norm_num
  have hpop : popRisk μ (costLoss C δ) = ∑ v ∈ V, w v * popRisk μ (ind v δ) := by
    unfold popRisk
    rw [hdecomp, integral_finsetSum _ fun v _ => (hint v).const_mul (w v)]
    exact Finset.sum_congr rfl fun v _ => integral_const_mul _ _
  have hemp : empRisk S (costLoss C δ) = ∑ v ∈ V, w v * empRisk S (ind v δ) := by
    unfold empRisk
    rw [hdecomp, Finset.sum_comm, Finset.sum_div]
    refine Finset.sum_congr rfl fun v _ => ?_
    rw [← Finset.mul_sum, mul_div_assoc]
  have hlevel : ∀ v ∈ V, popRisk μ (ind v δ)
      ≤ empRisk S (ind v δ) + vcRate m d (ν / ((K : ℝ) * K)) := by
    intro v hv
    obtain ⟨e, -, rfl⟩ := Finset.mem_image.1 hv
    exact hS e δ hδ
  have htot : ∑ v ∈ V, w v ≤ Cmax := hwtot Cmax hCmax (by
    intro v hv
    obtain ⟨e, -, rfl⟩ := Finset.mem_image.1 hv
    exact hC.le_max _ _)
  rw [hpop, hemp]
  calc ∑ v ∈ V, w v * popRisk μ (ind v δ)
      ≤ ∑ v ∈ V, w v * (empRisk S (ind v δ) + vcRate m d (ν / ((K : ℝ) * K))) :=
        Finset.sum_le_sum fun v hv =>
          mul_le_mul_of_nonneg_left (hlevel v hv) (hw0 v)
    _ = ∑ v ∈ V, w v * empRisk S (ind v δ)
          + (∑ v ∈ V, w v) * vcRate m d (ν / ((K : ℝ) * K)) := by
        rw [Finset.sum_mul, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun v _ => by ring
    _ ≤ ∑ v ∈ V, w v * empRisk S (ind v δ) + Cmax * vcRate m d (ν / ((K : ℝ) * K)) := by
        have := mul_le_mul_of_nonneg_right htot hε
        linarith

/-- **Generalization bound for decision optimization, two classes**
(`prop:gen_bound`).

Let the costs lie in `[0, Cmax]`, let `H` be a countable class of measurable
rules `X → {0, 1}` whose binary class has VC dimension at most `d ≤ m`, and let
the validation sample be `m` i.i.d. draws from a distribution `μ` on labelled
examples. Then with probability at least `1 - ν`, *every* `δ ∈ H` satisfies

`R(δ) ≤ R̂(δ) + Cmax · (√(2 d log(e m / d) / m) + √(log(4/ν) / (2m)))`.

Proved from `uniform_convergence_vc`, as the case `K = 2` of
`gen_bound_levels`. The loss of a rule takes at most four values, so it is a
nonnegative combination, of total weight at most `Cmax`, of the indicators of
its upper level sets; each family of level sets is a class of VC dimension at
most `d` (`vcDim_levelClass_le`), to which the proved bound applies after
relabelling every example with the label `0`; the at most four applications are
combined by a union bound, which is where `ν` becomes `ν / 4`.

This is the countable-class, two-term form. The displayed factor-2 form is
`gen_bound_binary_thesis`; the full proposition under pointwise approximation
is `SeparableBounds.gen_bound_binary_thesis_of_approx`. The four-level union
bound uses failure probability `ν/4` at each level and yields confidence
`1-ν` simultaneously for all rules. For more than two labels, use the
Natarajan bounds below.

Conditioning on the fitted estimator is represented by fixing `μ`, the law
of `(p̂(x), y)` for that estimator, before sampling validation observations.

The uniform-convergence theorem used in this reduction is proved in
`UniformConvergence.lean`; no probabilistic result is assumed. -/
theorem gen_bound_binary {X : Type} [MeasurableSpace X]
    (C : CostMatrix 2) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H : Set (X → Fin 2)) (hcount : H.Countable)
    (hmeas : ∀ δ ∈ H, Measurable δ)
    (μ : Measure (X × Fin 2)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : vcDim (binClass H) ≤ (d : ℕ∞)) (hdm : d ≤ m) (hm : 0 < m)
    (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ) + Cmax * vcRate m d (ν / 4)} := by
  have h := gen_bound_levels C hC H hcount hmeas μ d m ν
    (fun e => (vcDim_levelClass_le C H (C e.1 e.2)).trans hdim) hdm hm hν
  have h4 : ν / (((2 : ℕ) : ℝ) * ((2 : ℕ) : ℝ)) = ν / 4 := by norm_num
  rw [h4] at h
  exact h

/-! ### The bound in terms of the Natarajan dimension

`gen_bound_levels` measures the class of rules by the VC dimension of its loss
level sets. Natarajan's lemma ([`NatarajanLemma.lean`](NatarajanLemma.lean))
bounds that dimension by `2 d (⌈log₂(d K²)⌉ + 1)` when the Natarajan dimension
of the class is at most `d`, which gives the bound of
`rem:natarajan_gen_bound` with every factor explicit. -/

/-- The loss level sets of a class of rules of Natarajan dimension at most `d`
have VC dimension at most `2 d (⌈log₂(d K²)⌉ + 1)`, whatever the cost matrix
and the level. -/
theorem vcDim_levelClass_le_levelDim {X : Type} {K : ℕ} [NeZero K] (C : CostMatrix K)
    (H : Set (X → Fin K)) (v : ℝ) (d : ℕ) (hd : natDim H ≤ (d : ℕ∞)) :
    vcDim (levelClass C H v) ≤ (levelDim d K : ℕ∞) :=
  vcDim_pointwise_le H d hd Prod.fst fun z k => decide (v ≤ C z.2 k)

/-- **Generalization bound for `K` classes in terms of the Natarajan
dimension**, with the rate of the imported corollary.

Let the costs lie in `[0, Cmax]`, let `H` be a countable class of measurable
rules `X → {0, …, K−1}` with `d_Nat(H) ≤ d`, and let
`D = 2 d (⌈log₂(d K²)⌉ + 1) ≤ m`. Then with probability at least `1 - ν` over
`m` i.i.d. examples, *every* `δ ∈ H` satisfies

`R(δ) ≤ R̂(δ) + Cmax · (√(2 D log(e m / D) / m) + √(log(K²/ν) / (2m)))`.

From `gen_bound_levels` and `vcDim_levelClass_le_levelDim`, using the
proved uniform-convergence bound for binary classes. -/
theorem gen_bound_natarajan {X : Type} [MeasurableSpace X] {K : ℕ} [NeZero K]
    (C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H : Set (X → Fin K)) (hcount : H.Countable)
    (hmeas : ∀ δ ∈ H, Measurable δ)
    (μ : Measure (X × Fin K)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : natDim H ≤ (d : ℕ∞)) (hdm : levelDim d K ≤ m) (hm : 0 < m)
    (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + Cmax * vcRate m (levelDim d K) (ν / ((K : ℝ) * K))} :=
  gen_bound_levels C hC H hcount hmeas μ (levelDim d K) m ν
    (fun e => vcDim_levelClass_le_levelDim C H (C e.1 e.2) d hdim) hdm hm hν

/-- The factor in front of `Cmax √(d/m)` in the clean form of the bound:
`√(4 (⌈log₂(d K²)⌉ + 1) log(e m)) + √(log(K²/ν) / 2)`, the factor `c_Nat`
of `rem:natarajan_gen_bound`; it grows logarithmically in `K`, `d`, `m` and
`1/ν`. -/
noncomputable def cleanFactor (K d m : ℕ) (ν : ℝ) : ℝ :=
  Real.sqrt (4 * (((Nat.clog 2 (d * K ^ 2) : ℕ) : ℝ) + 1) * Real.log (Real.exp 1 * m))
    + Real.sqrt (Real.log ((K : ℝ) * K / ν) / 2)

/-- The rate of the imported corollary at the capacity `D = levelDim d K` is at
most `cleanFactor · √(d/m)`. -/
theorem vcRate_levelDim_le {K d m : ℕ} {ν : ℝ} (hK : 0 < K) (hd : 0 < d)
    (hDm : levelDim d K ≤ m) (hν : 0 < ν) (hν1 : ν ≤ 1) :
    vcRate m (levelDim d K) (ν / ((K : ℝ) * K))
      ≤ cleanFactor K d m ν * Real.sqrt (d / m) := by
  obtain ⟨ℓ, hℓ⟩ : ∃ ℓ : ℝ, ℓ = ((Nat.clog 2 (d * K ^ 2) : ℕ) : ℝ) := ⟨_, rfl⟩
  have hℓ0 : 0 ≤ ℓ := by
    rw [hℓ]
    exact Nat.cast_nonneg _
  have hDcast : ((levelDim d K : ℕ) : ℝ) = 2 * d * (ℓ + 1) := by
    rw [hℓ]
    unfold levelDim
    push_cast
    ring
  have hDpos : 0 < levelDim d K :=
    Nat.mul_pos (Nat.mul_pos (by norm_num) hd) (Nat.succ_pos _)
  have hD1 : (1 : ℝ) ≤ ((levelDim d K : ℕ) : ℝ) := by exact_mod_cast hDpos
  have hdR : (1 : ℝ) ≤ d := by exact_mod_cast hd
  have hmR : (1 : ℝ) ≤ m := hD1.trans (by exact_mod_cast hDm)
  have hm0 : (0 : ℝ) < m := by linarith
  have hKR : (1 : ℝ) ≤ K := by exact_mod_cast hK
  have hlogm : 1 ≤ Real.log (Real.exp 1 * m) := by
    rw [Real.le_log_iff_exp_le (by positivity)]
    exact le_mul_of_one_le_right (Real.exp_pos 1).le hmR
  have hdm0 : (0 : ℝ) ≤ d / m := div_nonneg (by linarith) hm0.le
  have hA : Real.sqrt (2 * ((levelDim d K : ℕ) : ℝ)
        * Real.log (Real.exp 1 * m / ((levelDim d K : ℕ) : ℝ)) / m)
      ≤ Real.sqrt (4 * (ℓ + 1) * Real.log (Real.exp 1 * m)) * Real.sqrt (d / m) := by
    rw [← Real.sqrt_mul (mul_nonneg (by linarith) (by linarith))]
    refine Real.sqrt_le_sqrt ?_
    have hlog : Real.log (Real.exp 1 * m / ((levelDim d K : ℕ) : ℝ))
        ≤ Real.log (Real.exp 1 * m) := by
      refine Real.log_le_log (by positivity) ?_
      exact div_le_self (by positivity) hD1
    have h2 : 2 * ((levelDim d K : ℕ) : ℝ)
          * Real.log (Real.exp 1 * m / ((levelDim d K : ℕ) : ℝ))
        ≤ 2 * ((levelDim d K : ℕ) : ℝ) * Real.log (Real.exp 1 * m) :=
      mul_le_mul_of_nonneg_left hlog (by linarith)
    calc 2 * ((levelDim d K : ℕ) : ℝ)
          * Real.log (Real.exp 1 * m / ((levelDim d K : ℕ) : ℝ)) / m
        ≤ 2 * ((levelDim d K : ℕ) : ℝ) * Real.log (Real.exp 1 * m) / m :=
          div_le_div_of_nonneg_right h2 hm0.le
      _ = 4 * (ℓ + 1) * Real.log (Real.exp 1 * m) * (d / m) := by
          rw [hDcast]
          ring
  have hlog0 : 0 ≤ Real.log ((K : ℝ) * K / ν) := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ hν]
    nlinarith
  have hB : Real.sqrt (Real.log (1 / (ν / ((K : ℝ) * K))) / (2 * m))
      ≤ Real.sqrt (Real.log ((K : ℝ) * K / ν) / 2) * Real.sqrt (d / m) := by
    rw [← Real.sqrt_mul (by linarith)]
    refine Real.sqrt_le_sqrt ?_
    have h1 : 1 / (ν / ((K : ℝ) * K)) = K * K / ν := by
      rw [one_div, inv_div]
    rw [h1]
    calc Real.log ((K : ℝ) * K / ν) / (2 * m)
        = Real.log ((K : ℝ) * K / ν) / 2 * (1 / m) := by
          field_simp
      _ ≤ Real.log ((K : ℝ) * K / ν) / 2 * (d / m) :=
          mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hdR hm0.le)
            (by linarith)
  calc vcRate m (levelDim d K) (ν / ((K : ℝ) * K))
      ≤ Real.sqrt (4 * (ℓ + 1) * Real.log (Real.exp 1 * m)) * Real.sqrt (d / m)
        + Real.sqrt (Real.log ((K : ℝ) * K / ν) / 2) * Real.sqrt (d / m) :=
        add_le_add hA hB
    _ = cleanFactor K d m ν * Real.sqrt (d / m) := by
        unfold cleanFactor
        rw [hℓ]
        ring

/-- Below the sample size `D = levelDim d K` the clean bound holds trivially:
`cleanFactor · √(d/m) ≥ 1`, and the risk is at most `Cmax`. -/
theorem one_le_cleanFactor_mul {K d m : ℕ} {ν : ℝ} (hd : 0 < d) (hm : 0 < m)
    (hDm : m < levelDim d K) : 1 ≤ cleanFactor K d m ν * Real.sqrt (d / m) := by
  obtain ⟨ℓ, hℓ⟩ : ∃ ℓ : ℝ, ℓ = ((Nat.clog 2 (d * K ^ 2) : ℕ) : ℝ) := ⟨_, rfl⟩
  have hℓ0 : 0 ≤ ℓ := by
    rw [hℓ]
    exact Nat.cast_nonneg _
  have hDcast : ((levelDim d K : ℕ) : ℝ) = 2 * d * (ℓ + 1) := by
    rw [hℓ]
    unfold levelDim
    push_cast
    ring
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hd0 : (0 : ℝ) < d := by exact_mod_cast hd
  have hDmR : (m : ℝ) < 2 * d * (ℓ + 1) := by
    rw [← hDcast]
    exact_mod_cast hDm
  have hlogm : 1 ≤ Real.log (Real.exp 1 * m) := by
    rw [Real.le_log_iff_exp_le (by positivity)]
    exact le_mul_of_one_le_right (Real.exp_pos 1).le hm1
  have h4 : 2 ≤ 4 * (ℓ + 1) * ((d : ℝ) / m) := by
    rw [show 4 * (ℓ + 1) * ((d : ℝ) / m) = 4 * (ℓ + 1) * d / m by ring,
      le_div_iff₀ hm0]
    linarith
  have hfirst : 1 ≤ Real.sqrt (4 * (ℓ + 1) * Real.log (Real.exp 1 * m))
      * Real.sqrt (d / m) := by
    rw [← Real.sqrt_mul (mul_nonneg (by linarith) (by linarith)),
      Real.le_sqrt' (by norm_num), one_pow]
    calc (1 : ℝ) ≤ 4 * (ℓ + 1) * ((d : ℝ) / m) * 1 := by linarith
      _ ≤ 4 * (ℓ + 1) * ((d : ℝ) / m) * Real.log (Real.exp 1 * m) :=
          mul_le_mul_of_nonneg_left hlogm (by linarith)
      _ = 4 * (ℓ + 1) * Real.log (Real.exp 1 * m) * (d / m) := by ring
  have hsecond : 0 ≤ Real.sqrt (Real.log ((K : ℝ) * K / ν) / 2) * Real.sqrt (d / m) :=
    mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  unfold cleanFactor
  rw [← hℓ, add_mul]
  linarith

/-- A bound whose slack is at least `Cmax` holds for every sample: the risk is
at most `Cmax` and the empirical risk is nonnegative. -/
theorem popRisk_le_empRisk_add {X : Type} [MeasurableSpace X] {K : ℕ} [NeZero K]
    {C : CostMatrix K} {Cmax : ℝ} (hC : Bounded C Cmax)
    (μ : Measure (X × Fin K)) [IsProbabilityMeasure μ] (δ : X → Fin K) {m : ℕ}
    (S : Fin m → X × Fin K) {t : ℝ} (ht : Cmax ≤ t) :
    popRisk μ (costLoss C δ) ≤ empRisk S (costLoss C δ) + t := by
  have hCmax : 0 ≤ Cmax := (hC.nonneg 0 0).trans (hC.le_max 0 0)
  have hpop : popRisk μ (costLoss C δ) ≤ Cmax := by
    unfold popRisk
    by_cases hint : Integrable (costLoss C δ) μ
    · calc ∫ z, costLoss C δ z ∂μ ≤ ∫ _z, Cmax ∂μ :=
            integral_mono hint (integrable_const _)
              fun z => (costLoss_mem_Icc hC δ z).2
        _ = Cmax := by simp
    · rw [integral_undef hint]
      exact hCmax
  have hemp : 0 ≤ empRisk S (costLoss C δ) := by
    unfold empRisk
    exact div_nonneg
      (Finset.sum_nonneg fun i _ => (costLoss_mem_Icc hC δ (S i)).1)
      (Nat.cast_nonneg m)
  linarith

/-- **Multiclass generalization bound, countable class.**
For a countable measurable class, bounded costs, `d ≥ max(Natdim,1)`,
`m ≥ 1`, and `0 < ν ≤ 1`, all rules simultaneously satisfy the displayed
bound of `rem:natarajan_gen_bound` with the explicit factor
`cleanFactor K d m ν`. Natarajan growth bounds, the passage to loss level
sets, and the binary uniform-convergence result are all proved. The remark
itself, for a class approximated by a countable one, is
`PointwiseApprox.gen_bound_natarajan_clean_of_approx_event`. -/
theorem gen_bound_natarajan_clean {X : Type} [MeasurableSpace X] {K : ℕ} [NeZero K]
    (C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H : Set (X → Fin K)) (hcount : H.Countable)
    (hmeas : ∀ δ ∈ H, Measurable δ)
    (μ : Measure (X × Fin K)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : natDim H ≤ (d : ℕ∞)) (hd : 0 < d) (hm : 0 < m)
    (hν : 0 < ν) (hν1 : ν ≤ 1) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + cleanFactor K d m ν * Cmax * Real.sqrt (d / m)} := by
  have hK : 0 < K := Nat.pos_of_ne_zero (NeZero.ne K)
  have hCmax : 0 ≤ Cmax := (hC.nonneg 0 0).trans (hC.le_max 0 0)
  rcases le_or_gt (levelDim d K) m with hDm | hDm
  · refine (gen_bound_natarajan C hC H hcount hmeas μ d m ν hdim hDm hm hν).trans
      (measure_mono fun S hS δ hδ => (hS δ hδ).trans ?_)
    have := mul_le_mul_of_nonneg_left (vcRate_levelDim_le hK hd hDm hν hν1) hCmax
    linarith
  · have hall : {S : Fin m → X × Fin K | ∀ δ ∈ H,
          popRisk μ (costLoss C δ)
            ≤ empRisk S (costLoss C δ)
              + cleanFactor K d m ν * Cmax * Real.sqrt (d / m)} = Set.univ := by
      refine Set.eq_univ_of_forall fun S δ _ => ?_
      have hfac := one_le_cleanFactor_mul (K := K) (ν := ν) hd hm hDm
      have hmul := mul_le_mul_of_nonneg_left hfac hCmax
      exact popRisk_le_empRisk_add hC μ δ S (by linarith)
    rw [hall, measure_univ]
    exact tsub_le_self

/-! ### The clean form for two classes

`eq:clean_gen_bound` writes the bound of `prop:gen_bound` as
`R ≤ R̂ + c · Cmax · √(d/m)`. For two classes, with the VC dimension as
capacity, the factor is `√(2 log(e m)) + √(log(4/ν) / 2)`. -/

/-- The factor of the clean form for two classes:
`√(2 log(e m)) + √(log(4/ν) / 2)`. -/
noncomputable def cleanFactorBinary (m : ℕ) (ν : ℝ) : ℝ :=
  Real.sqrt (2 * Real.log (Real.exp 1 * m)) + Real.sqrt (Real.log (4 / ν) / 2)

theorem vcRate_le_cleanFactorBinary {d m : ℕ} {ν : ℝ} (hd : 0 < d) (hdm : d ≤ m)
    (hν : 0 < ν) (hν1 : ν ≤ 1) :
    vcRate m d (ν / 4) ≤ cleanFactorBinary m ν * Real.sqrt (d / m) := by
  have hdR : (1 : ℝ) ≤ d := by exact_mod_cast hd
  have hmR : (1 : ℝ) ≤ m := hdR.trans (by exact_mod_cast hdm)
  have hm0 : (0 : ℝ) < m := by linarith
  have hd0 : (0 : ℝ) < d := by linarith
  have hlogm : 1 ≤ Real.log (Real.exp 1 * m) := by
    rw [Real.le_log_iff_exp_le (by positivity)]
    exact le_mul_of_one_le_right (Real.exp_pos 1).le hmR
  have hA : Real.sqrt (2 * (d : ℝ) * Real.log (Real.exp 1 * m / d) / m)
      ≤ Real.sqrt (2 * Real.log (Real.exp 1 * m)) * Real.sqrt (d / m) := by
    rw [← Real.sqrt_mul (by linarith)]
    refine Real.sqrt_le_sqrt ?_
    have hlog : Real.log (Real.exp 1 * m / d) ≤ Real.log (Real.exp 1 * m) :=
      Real.log_le_log (by positivity) (div_le_self (by positivity) hdR)
    calc 2 * (d : ℝ) * Real.log (Real.exp 1 * m / d) / m
        ≤ 2 * (d : ℝ) * Real.log (Real.exp 1 * m) / m :=
          div_le_div_of_nonneg_right
            (mul_le_mul_of_nonneg_left hlog (by linarith)) hm0.le
      _ = 2 * Real.log (Real.exp 1 * m) * (d / m) := by ring
  have hlog0 : 0 ≤ Real.log (4 / ν) := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ hν]
    linarith
  have hB : Real.sqrt (Real.log (1 / (ν / 4)) / (2 * m))
      ≤ Real.sqrt (Real.log (4 / ν) / 2) * Real.sqrt (d / m) := by
    rw [← Real.sqrt_mul (by linarith)]
    refine Real.sqrt_le_sqrt ?_
    have h1 : 1 / (ν / 4) = 4 / ν := by rw [one_div, inv_div]
    rw [h1]
    calc Real.log (4 / ν) / (2 * m) = Real.log (4 / ν) / 2 * (1 / m) := by
          field_simp
      _ ≤ Real.log (4 / ν) / 2 * (d / m) :=
          mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hdR hm0.le)
            (by linarith)
  calc vcRate m d (ν / 4)
      ≤ Real.sqrt (2 * Real.log (Real.exp 1 * m)) * Real.sqrt (d / m)
        + Real.sqrt (Real.log (4 / ν) / 2) * Real.sqrt (d / m) := add_le_add hA hB
    _ = cleanFactorBinary m ν * Real.sqrt (d / m) := by
        unfold cleanFactorBinary
        ring

theorem one_le_cleanFactorBinary_mul {d m : ℕ} {ν : ℝ} (hm : 0 < m) (hdm : m < d) :
    1 ≤ cleanFactorBinary m ν * Real.sqrt (d / m) := by
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hdmR : (m : ℝ) < d := by exact_mod_cast hdm
  have hlogm : 1 ≤ Real.log (Real.exp 1 * m) := by
    rw [Real.le_log_iff_exp_le (by positivity)]
    exact le_mul_of_one_le_right (Real.exp_pos 1).le hm1
  have h1 : 1 ≤ (d : ℝ) / m := by
    rw [le_div_iff₀ hm0]
    linarith
  have hfirst : 1 ≤ Real.sqrt (2 * Real.log (Real.exp 1 * m)) * Real.sqrt (d / m) := by
    rw [← Real.sqrt_mul (by linarith), Real.le_sqrt' (by norm_num), one_pow]
    nlinarith [mul_nonneg (sub_nonneg.2 hlogm) (sub_nonneg.2 h1)]
  have hsecond : 0 ≤ Real.sqrt (Real.log (4 / ν) / 2) * Real.sqrt (d / m) :=
    mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  unfold cleanFactorBinary
  rw [add_mul]
  linarith

/-- **Clean binary generalization bound, countable class.**
For a countable class of measurable binary rules, costs in `[0,Cmax]`,
`d ≥ max(VCdim,1)`, `m ≥ 1`, and `0 < ν ≤ 1`, all rules simultaneously
satisfy `R ≤ Rhat + cbin(m,ν) Cmax sqrt(d/m)` with probability at least
`1 - ν`, where `cbin = sqrt(2 log(em)) + sqrt(log(4/ν)/2)`.
For `m < d` the additive term is at least `Cmax`. The text states
`eq:clean_gen_bound` under the hypotheses of `prop:gen_bound`; that form is
`SeparableBounds.gen_bound_binary_clean_of_approx`. -/
theorem gen_bound_binary_clean {X : Type} [MeasurableSpace X]
    (C : CostMatrix 2) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H : Set (X → Fin 2)) (hcount : H.Countable)
    (hmeas : ∀ δ ∈ H, Measurable δ)
    (μ : Measure (X × Fin 2)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : vcDim (binClass H) ≤ (d : ℕ∞)) (hd : 0 < d) (hm : 0 < m)
    (hν : 0 < ν) (hν1 : ν ≤ 1) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + cleanFactorBinary m ν * Cmax * Real.sqrt (d / m)} := by
  have hCmax : 0 ≤ Cmax := (hC.nonneg 0 0).trans (hC.le_max 0 0)
  rcases le_or_gt d m with hdm | hdm
  · refine (gen_bound_binary C hC H hcount hmeas μ d m ν hdim hdm hm hν).trans
      (measure_mono fun S hS δ hδ => (hS δ hδ).trans ?_)
    have := mul_le_mul_of_nonneg_left (vcRate_le_cleanFactorBinary hd hdm hν hν1) hCmax
    linarith
  · have hall : {S : Fin m → X × Fin 2 | ∀ δ ∈ H,
          popRisk μ (costLoss C δ)
            ≤ empRisk S (costLoss C δ)
              + cleanFactorBinary m ν * Cmax * Real.sqrt (d / m)} = Set.univ := by
      refine Set.eq_univ_of_forall fun S δ _ => ?_
      have hfac := one_le_cleanFactorBinary_mul (ν := ν) hm hdm
      have hmul := mul_le_mul_of_nonneg_left hfac hCmax
      exact popRisk_le_empRisk_add hC μ δ S (by linarith)
    rw [hall, measure_univ]
    exact tsub_le_self

/-! ### The threshold class

`gen_bound_binary` assumes a countable class, and the threshold rules
`δ_τ(z) = 1[z ≥ τ]` of Chapter 2 are indexed by a real `τ`. The bound extends
to all of them: it holds simultaneously for the rational thresholds, and every
threshold rule is the pointwise limit of rational ones from below, along which
both the population and the empirical risk converge. -/

section Threshold

open Filter Topology ThresholdOpt.Binary

/-- The threshold rules with a rational threshold. -/
def ratThresholdClass : Set (ℝ → Fin 2) :=
  {δ | ∃ q : ℚ, δ = thresholdRule (q : ℝ)}

theorem ratThresholdClass_countable : ratThresholdClass.Countable := by
  have h : ratThresholdClass = Set.range fun q : ℚ => thresholdRule (q : ℝ) := by
    ext δ
    constructor
    · rintro ⟨q, rfl⟩
      exact ⟨q, rfl⟩
    · rintro ⟨q, rfl⟩
      exact ⟨q, rfl⟩
  rw [h]
  exact Set.countable_range _

theorem measurable_thresholdRule (τ : ℝ) : Measurable (thresholdRule τ) := by
  show Measurable fun z : ℝ => if τ ≤ z then (1 : Fin 2) else 0
  exact Measurable.ite (measurableSet_le measurable_const measurable_id)
    measurable_const measurable_const

theorem measurable_costLoss_thresholdRule (C : CostMatrix 2) (τ : ℝ) :
    Measurable (costLoss C (thresholdRule τ)) :=
  (Measurable.of_discrete (f := fun q : Fin 2 × Fin 2 => C q.1 q.2)).comp
    (measurable_snd.prodMk ((measurable_thresholdRule τ).comp measurable_fst))

/-- The binary class of the rational thresholds has VC dimension at most `1`,
by `lem:vc_thresh`. -/
theorem vcDim_binClass_ratThreshold_le :
    vcDim (binClass ratThresholdClass) ≤ ((1 : ℕ) : ℕ∞) := by
  have hsub : binClass ratThresholdClass ⊆ thresholdClass := by
    rintro h ⟨δ, ⟨q, rfl⟩, rfl⟩
    refine ⟨(q : ℝ), ?_⟩
    funext z
    by_cases hz : (q : ℝ) ≤ z <;> simp [thresholdRule, hz]
  have := vcDim_mono hsub
  rw [vcDim_thresholdClass] at this
  exact_mod_cast this

/-- **Generalization bound for the empirical threshold** (the display of
`sec:binary_sample_complexity` that instantiates `prop:gen_bound` at
`d_VC = 1`), with the explicit rate.

For costs in `[0, Cmax]`, any law `μ` of `(score, label)` and a validation
sample of `m ≥ 1` i.i.d. examples: with probability at least `1 - ν`, *every*
threshold `τ ∈ ℝ`, in particular one chosen from the sample, satisfies

`R(δ_τ) ≤ R̂(δ_τ) + Cmax · (√(2 log(e m) / m) + √(log(4/ν) / (2m)))`.

From `gen_bound_binary`, hence from `uniform_convergence_vc`. The class is all
real thresholds; no countability or measurability hypothesis is left. The text
writes the bound as `c · Cmax · √(1/m)` with `c = c_bin(m, ν)`; for
`0 < ν ≤ 1` the rate here is at most that (`vcRate_le_cleanFactorBinary`). -/
theorem gen_bound_threshold (C : CostMatrix 2) {Cmax : ℝ} (hC : Bounded C Cmax)
    (μ : Measure (ℝ × Fin 2)) [IsProbabilityMeasure μ] (m : ℕ) (hm : 0 < m)
    (ν : ℝ) (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ τ : ℝ,
        popRisk μ (costLoss C (thresholdRule τ))
          ≤ empRisk S (costLoss C (thresholdRule τ)) + Cmax * vcRate m 1 (ν / 4)} := by
  have key := gen_bound_binary C hC ratThresholdClass ratThresholdClass_countable
    (by rintro δ ⟨q, rfl⟩; exact measurable_thresholdRule _) μ 1 m ν
    vcDim_binClass_ratThreshold_le hm hm hν
  refine key.trans (measure_mono ?_)
  intro S hS τ
  obtain ⟨u, -, hlt, htend⟩ := Real.exists_seq_rat_strictMono_tendsto τ
  -- Along rationals increasing to `τ`, the rule is eventually `δ_τ` at each point.
  have hev : ∀ z : ℝ, ∀ᶠ n in atTop,
      thresholdRule ((u n : ℚ) : ℝ) z = thresholdRule τ z := by
    intro z
    by_cases hz : τ ≤ z
    · refine Filter.Eventually.of_forall fun n => ?_
      have hle : ((u n : ℚ) : ℝ) ≤ z := (hlt n).le.trans hz
      simp [thresholdRule, hz, hle]
    · have hz' : z < τ := not_le.1 hz
      filter_upwards [htend.eventually (lt_mem_nhds hz')] with n hn
      have hnle : ¬ ((u n : ℚ) : ℝ) ≤ z := not_le.2 hn
      simp [thresholdRule, hz, hnle]
  have hpop : Tendsto (fun n => popRisk μ (costLoss C (thresholdRule ((u n : ℚ) : ℝ))))
      atTop (𝓝 (popRisk μ (costLoss C (thresholdRule τ)))) := by
    unfold popRisk
    refine tendsto_integral_of_dominated_convergence (fun _ => Cmax)
      (fun n => (measurable_costLoss_thresholdRule C _).aestronglyMeasurable)
      (integrable_const _) (fun n => ?_) ?_
    · refine Filter.Eventually.of_forall fun z => ?_
      have hmem := costLoss_mem_Icc hC (thresholdRule ((u n : ℚ) : ℝ)) z
      rw [Real.norm_eq_abs, abs_of_nonneg hmem.1]
      exact hmem.2
    · refine Filter.Eventually.of_forall fun z => ?_
      refine tendsto_nhds_of_eventually_eq ((hev z.1).mono fun n hn => ?_)
      simp only [costLoss, hn]
  have hemp : Tendsto (fun n => empRisk S (costLoss C (thresholdRule ((u n : ℚ) : ℝ))))
      atTop (𝓝 (empRisk S (costLoss C (thresholdRule τ)))) := by
    refine tendsto_nhds_of_eventually_eq ?_
    have hall : ∀ᶠ n in atTop, ∀ i : Fin m,
        thresholdRule ((u n : ℚ) : ℝ) (S i).1 = thresholdRule τ (S i).1 :=
      Filter.eventually_all.2 fun i => hev (S i).1
    filter_upwards [hall] with n hn
    unfold empRisk
    congr 1
    exact Finset.sum_congr rfl fun i _ => by simp only [costLoss, hn i]
  exact le_of_tendsto_of_tendsto' hpop (hemp.add_const _) fun n => hS _ ⟨u n, rfl⟩

end Threshold


/-! ### The displayed binary rate, with confidence `1 - ν` -/

/-- The rate in `eq:vc_decision_rate`, before its universal factor. -/
noncomputable def thesisBinaryRate (m d : ℕ) (ν : ℝ) : ℝ :=
  Real.sqrt (((d : ℝ) * Real.log (2 * Real.exp 1 * m / d)
    + Real.log (4 / ν)) / m)

theorem vcRate_le_twice_thesisBinaryRate {d m : ℕ} {ν : ℝ}
    (hd : 0 < d) (hdm : d ≤ m) (hν : 0 < ν) (hν1 : ν ≤ 1) :
    vcRate m d (ν / 4) ≤ 2 * thesisBinaryRate m d ν := by
  have hd0 : (0 : ℝ) < d := by exact_mod_cast hd
  have hm0 : (0 : ℝ) < m := by exact_mod_cast lt_of_lt_of_le hd hdm
  have hdmR : (d : ℝ) ≤ m := by exact_mod_cast hdm
  have hlog0 : 0 ≤ Real.log (Real.exp 1 * m / d) := by
    apply Real.log_nonneg
    rw [le_div_iff₀ hd0]
    have he : 1 ≤ Real.exp 1 := by simpa using Real.one_le_exp (by norm_num : (0 : ℝ) ≤ 1)
    nlinarith
  have hlog4 : 0 ≤ Real.log (4 / ν) := by
    apply Real.log_nonneg
    rw [le_div_iff₀ hν]
    linarith
  have hlog : Real.log (Real.exp 1 * m / d)
      ≤ Real.log (2 * Real.exp 1 * m / d) := by
    apply Real.log_le_log (by positivity)
    have : 0 < Real.exp 1 * m / d := by positivity
    calc Real.exp 1 * m / d ≤ 2 * (Real.exp 1 * m / d) := by linarith
      _ = 2 * Real.exp 1 * m / d := by ring
  have hlog20 : 0 ≤ Real.log (2 * Real.exp 1 * m / d) := hlog0.trans hlog
  have ha : 0 ≤ 2 * (d : ℝ) * Real.log (Real.exp 1 * m / d) / m := by positivity
  have hb : 0 ≤ Real.log (4 / ν) / (2 * m) := by positivity
  have ht : 0 ≤ ((d : ℝ) * Real.log (2 * Real.exp 1 * m / d)
      + Real.log (4 / ν)) / m := by positivity
  have hab : 2 * (2 * (d : ℝ) * Real.log (Real.exp 1 * m / d) / m
      + Real.log (4 / ν) / (2 * m))
      ≤ 4 * (((d : ℝ) * Real.log (2 * Real.exp 1 * m / d)
        + Real.log (4 / ν)) / m) := by
    have hmul := mul_le_mul_of_nonneg_left hlog (show 0 ≤ (d : ℝ) by positivity)
    field_simp
    nlinarith
  have hsq : (Real.sqrt (2 * (d : ℝ) * Real.log (Real.exp 1 * m / d) / m)
      + Real.sqrt (Real.log (4 / ν) / (2 * m))) ^ 2
      ≤ (2 * thesisBinaryRate m d ν) ^ 2 := by
    have hdiff := sq_nonneg (Real.sqrt (2 * (d : ℝ) * Real.log (Real.exp 1 * m / d) / m)
      - Real.sqrt (Real.log (4 / ν) / (2 * m)))
    have hs := Real.sq_sqrt ht
    unfold thesisBinaryRate
    nlinarith [Real.sq_sqrt ha, Real.sq_sqrt hb]
  unfold vcRate
  rw [show 1 / (ν / 4) = 4 / ν by field_simp]
  have : 0 ≤ thesisBinaryRate m d ν := Real.sqrt_nonneg _
  nlinarith [Real.sqrt_nonneg (2 * (d : ℝ) * Real.log (Real.exp 1 * m / d) / m),
    Real.sqrt_nonneg (Real.log (4 / ν) / (2 * m))]

/-- **Binary generalization with the displayed rate for a countable class.**
The universal factor is `2`, confidence is `1 - ν`, `K = 2`, and `1 ≤ d ≤ m`.
The four-level union bound is absorbed by `log(4/ν)`. The corresponding result
for a class approximated by a countable measurable subclass is
`SeparableBounds.gen_bound_binary_thesis_of_approx`, which carries the full
tag for `prop:gen_bound`. -/
theorem gen_bound_binary_thesis {X : Type} [MeasurableSpace X]
    (C : CostMatrix 2) {Cmax : ℝ} (hC : Bounded C Cmax)
    (H : Set (X → Fin 2)) (hcount : H.Countable)
    (hmeas : ∀ δ ∈ H, Measurable δ)
    (μ : Measure (X × Fin 2)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : vcDim (binClass H) ≤ (d : ℕ∞)) (hd : 0 < d) (hdm : d ≤ m)
    (hm : 0 < m) (hν : 0 < ν) (hν1 : ν ≤ 1) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H, popRisk μ (costLoss C δ)
        ≤ empRisk S (costLoss C δ) + 2 * Cmax * thesisBinaryRate m d ν} := by
  have hCmax : 0 ≤ Cmax := (hC.nonneg 0 0).trans (hC.le_max 0 0)
  refine (gen_bound_binary C hC H hcount hmeas μ d m ν hdim hdm hm hν).trans
    (measure_mono fun S hS δ hδ => (hS δ hδ).trans ?_)
  have := mul_le_mul_of_nonneg_left (vcRate_le_twice_thesisBinaryRate hd hdm hν hν1) hCmax
  linarith

end ThresholdOpt.Literature
