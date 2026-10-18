import ThresholdOpt.Sauer
import ThresholdOpt.Symmetrization

/-!
# Uniform convergence for a VC class (Mohri et al., Corollary 3.19)

Mohri, Rostamizadeh and Talwalkar, *Foundations of Machine Learning*, 2nd ed.,
Corollary 3.19, eq. (3.29): for a class `H` of binary functions of VC
dimension `d` and `ν > 0`, with probability at least `1 − ν` over an i.i.d.
sample of size `m`, every `h ∈ H` satisfies

`R(h) ≤ R̂_S(h) + √(2 d log(e m / d) / m) + √(log(1/ν) / (2m))`.

This file proves it, with the hypotheses of `Literature.uniform_convergence_vc`
(`d ≤ m`, a countable class of measurable functions) and the VC dimension at
most `d` (`uniform_convergence_vc_proved`). The route is the book's:

* Theorem 3.3 (`Symmetrization.rademacher_bound`) for the losses
  `g_h(x, y) = 1[h(x) ≠ y]`, which gives the term `√(log(1/ν)/(2m))` and
  `2 R_m(G)`;
* for each sample, the losses realize at most as many vectors as `H` has
  labelings of the sample (`Sauer.patterns`), and Massart's lemma
  (`Massart.massart`) applied to those vectors centred at `1/2`, whose norm is
  `√m / 2`, gives `2 R̂_S(G) ≤ √(2 log |H|_S| / m)`. This is Lemma 3.4 and
  Corollary 3.8 in one step;
* Sauer's lemma (`Sauer.log_card_patterns_le`) gives
  `log |H|_S| ≤ d log(e m / d)`.

Countability is weakened in `uniform_convergence_vc_of_approx`: it suffices
that every member of `H` is a pointwise limit of a sequence from a countable
class of measurable functions of VC dimension at most `d`.
-/

namespace ThresholdOpt.UniformConvergence

open MeasureTheory ProbabilityTheory Real Finset Filter Topology
open ThresholdOpt.Learning ThresholdOpt.Capacity ThresholdOpt.Massart
open ThresholdOpt.Sauer ThresholdOpt.Symmetrization

variable {X : Type*}

theorem zeroOneLoss_mem_Icc' (h : X → Bool) (z : X × Bool) :
    zeroOneLoss h z ∈ Set.Icc (0 : ℝ) 1 :=
  zeroOneLoss_mem_Icc h z

/-- The loss vector of a labeling `b` of the sample, against the labels `y`. -/
def lossVec {m : ℕ} (y : Fin m → Bool) (b : Fin m → Bool) : Fin m → ℝ :=
  fun j => if b j = y j then 0 else 1

/-- **Massart's lemma for the `0/1` loss** (Lemma 3.4 with Corollary 3.8 of
Mohri et al.): twice the empirical Rademacher complexity of the losses of a
nonempty class on a sample of size `m ≥ 1` is at most
`√(2 log |H|_S| / m)`. -/
theorem two_mul_empRad_le (H : Set (X → Bool)) (hH : H.Nonempty) {m : ℕ} (hm : 0 < m)
    (S : Fin m → X × Bool) :
    2 * empRad (fun h : H => zeroOneLoss h.1) S
      ≤ sqrt (2 * log (patterns H fun j => (S j).1).card / m) := by
  classical
  set x : Fin m → X := fun j => (S j).1 with hx
  set y : Fin m → Bool := fun j => (S j).2 with hy
  set A := patterns H x with hA
  have hAne : A.Nonempty := patterns_nonempty hH x
  -- The centred loss vectors.
  set B : Finset (Fin m → ℝ) := A.image fun b j => lossVec y b j - 1 / 2 with hB
  have hBne : B.Nonempty := hAne.image _
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hr : (0 : ℝ) ≤ sqrt m / 2 := by positivity
  have hnorm : ∀ v ∈ B, ∑ j, v j ^ 2 ≤ (sqrt m / 2) ^ 2 := by
    intro v hv
    obtain ⟨b, -, rfl⟩ := mem_image.1 hv
    have : ∀ j, (lossVec y b j - 1 / 2) ^ 2 = 1 / 4 := by
      intro j
      unfold lossVec
      split_ifs <;> norm_num
    simp only [this, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, div_pow,
      sq_sqrt hmR.le]
    linarith
  have hmass := massart B hBne hr hnorm
  -- For each `σ`, the supremum over `H` is at most the maximum over `B`, shifted.
  have hpt : ∀ σ : Fin m → Bool,
      (⨆ h : H, (∑ j, sgn (σ j) * zeroOneLoss h.1 (S j)) / m)
        ≤ ((B.sup' hBne fun v => ∑ j, sgn (σ j) * v j)
            + ∑ j, sgn (σ j) * (1 / 2)) / m := by
    intro σ
    have : Nonempty H := hH.to_subtype
    refine ciSup_le fun h => ?_
    refine div_le_div_of_nonneg_right ?_ hmR.le
    have hb : (fun j => h.1 (x j)) ∈ A := pattern_mem x h.2
    have hv : (fun j => lossVec y (fun j => h.1 (x j)) j - 1 / 2) ∈ B :=
      mem_image.2 ⟨_, hb, rfl⟩
    have hle := le_sup' (fun v : Fin m → ℝ => ∑ j, sgn (σ j) * v j) hv
    have heq : ∑ j, sgn (σ j) * zeroOneLoss h.1 (S j)
        = ∑ j, sgn (σ j) * (lossVec y (fun j => h.1 (x j)) j - 1 / 2)
          + ∑ j, sgn (σ j) * (1 / 2) := by
      rw [← sum_add_distrib]
      refine sum_congr rfl fun j _ => ?_
      simp only [zeroOneLoss, lossVec, hx, hy]
      ring
    rw [heq]
    linarith
  have hcard : (B.card : ℝ) ≤ A.card := by exact_mod_cast card_image_le
  have hBpos : (0 : ℝ) < B.card := by exact_mod_cast hBne.card_pos
  have hlogB : log B.card ≤ log A.card := log_le_log hBpos hcard
  have hrad : empRad (fun h : H => zeroOneLoss h.1) S
      ≤ (sqrt m / 2 * sqrt (2 * log A.card)) / m := by
    unfold empRad
    calc radAvg (fun σ => ⨆ h : H, (∑ j, sgn (σ j) * zeroOneLoss h.1 (S j)) / m)
        ≤ radAvg (fun σ => ((B.sup' hBne fun v => ∑ j, sgn (σ j) * v j)
            + ∑ j, sgn (σ j) * (1 / 2)) / m) := radAvg_mono hpt
      _ = (radAvg (fun σ => B.sup' hBne fun v => ∑ j, sgn (σ j) * v j)
            + radAvg (fun σ => ∑ j, sgn (σ j) * (1 / 2))) / m := by
          rw [← radAvg_add]
          have : ∀ F : (Fin m → Bool) → ℝ,
              radAvg (fun σ => F σ / m) = radAvg F / m := by
            intro F
            have := radAvg_mul_left (1 / (m : ℝ)) F
            simp only [one_div_mul_eq_div] at this
            rw [this]
          exact this _
      _ = radAvg (fun σ => B.sup' hBne fun v => ∑ j, sgn (σ j) * v j) / m := by
          rw [radAvg_inner (fun _ => 1 / 2), add_zero]
      _ ≤ (sqrt m / 2 * sqrt (2 * log B.card)) / m :=
          div_le_div_of_nonneg_right hmass hmR.le
      _ ≤ (sqrt m / 2 * sqrt (2 * log A.card)) / m := by
          refine div_le_div_of_nonneg_right ?_ hmR.le
          exact mul_le_mul_of_nonneg_left (sqrt_le_sqrt (by linarith)) hr
  have hsqrtm : sqrt (m : ℝ) ≠ 0 := (sqrt_pos.2 hmR).ne'
  have hfinal : 2 * ((sqrt m / 2 * sqrt (2 * log A.card)) / m)
      = sqrt (2 * log A.card / m) := by
    rw [sqrt_div' _ hmR.le]
    field_simp
    rw [sq_sqrt hmR.le]
    ring
  calc 2 * empRad (fun h : H => zeroOneLoss h.1) S
      ≤ 2 * ((sqrt m / 2 * sqrt (2 * log A.card)) / m) := by linarith
    _ = sqrt (2 * log A.card / m) := hfinal

/-- **Uniform convergence for a VC class**, proved (Mohri et al.,
Corollary 3.19, eq. (3.29)), for a countable class of measurable binary
functions of VC dimension at most `d ≤ m`. -/
theorem uniform_convergence_vc_proved {X : Type} [MeasurableSpace X] (H : Set (X → Bool))
    (hcount : H.Countable) (hmeas : ∀ h ∈ H, Measurable h)
    (μ : Measure (X × Bool)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : vcDim H ≤ (d : ℕ∞)) (hdm : d ≤ m) (hm : 0 < m) (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ h ∈ H,
        popRisk μ (zeroOneLoss h) ≤ empRisk S (zeroOneLoss h) + vcRate m d ν} := by
  -- Trivial cases: `ν ≥ 1`, and the empty class.
  rcases le_or_gt 1 ν with hν1 | hν1
  · have : (1 : ENNReal) ≤ ENNReal.ofReal ν := by
      rw [← ENNReal.ofReal_one]
      exact ENNReal.ofReal_le_ofReal hν1
    rw [tsub_eq_zero_of_le this]
    exact zero_le
  rcases H.eq_empty_or_nonempty with hH | hH
  · subst hH
    simp
  have : Countable H := hcount.to_subtype
  have : Nonempty H := hH.to_subtype
  let G : H → X × Bool → ℝ := fun h => zeroOneLoss h.1
  have hGm : ∀ h, Measurable (G h) := by
    intro h
    exact (Measurable.of_discrete
      (f := fun q : Bool × Bool => if q.1 = q.2 then (0 : ℝ) else 1)).comp
      (((hmeas h.1 h.2).comp measurable_fst).prodMk measurable_snd)
  have hG01 : ∀ h z, G h z ∈ Set.Icc (0 : ℝ) 1 := fun h z => zeroOneLoss_mem_Icc _ _
  have key := rademacher_bound μ G hGm hG01 hm hν hν1.le
  -- `2 R_m(G) ≤ √(2 d log(e m / d) / m)`.
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  set rate := sqrt (2 * d * log (exp 1 * m / d) / m) with hrate
  have hpt : ∀ S : Fin m → X × Bool, 2 * empRad G S ≤ rate := by
    intro S
    refine (two_mul_empRad_le H hH hm S).trans (sqrt_le_sqrt ?_)
    have := log_card_patterns_le H hdim hdm (fun j => (S j).1)
    rw [mul_assoc]
    exact div_le_div_of_nonneg_right (by linarith) hmR.le
  have hR : 2 * radComplexity μ G m ≤ rate := by
    unfold radComplexity
    rw [← integral_const_mul]
    calc ∫ S, 2 * empRad G S ∂(sampleMeasure μ m) ≤ ∫ _S, rate ∂(sampleMeasure μ m) :=
          integral_mono
            ((McDiarmid.integrable_of_abs_le (measurable_empRad G hGm) 1
              (abs_empRad_le G hG01)).const_mul 2)
            (integrable_const _) hpt
      _ = rate := by simp
  refine key.trans (measure_mono fun S hS h hh => ?_)
  have := hS ⟨h, hh⟩
  unfold vcRate
  rw [← hrate]
  simp only [G] at this
  linarith

/-! ### Beyond countable classes -/

/-- **Uniform convergence for a class approximated by a countable one.** If
every member of `H` is the pointwise limit of a sequence from a countable class
`H₀` of measurable functions of VC dimension at most `d ≤ m` (pointwise limit
of `Bool`-valued functions: at each point the sequence is eventually equal to
the limit), the bound of Corollary 3.19 holds for all of `H` with probability
at least `1 − ν`. `H₀` need not be contained in `H`. -/
theorem uniform_convergence_vc_of_approx {X : Type} [MeasurableSpace X]
    (H H₀ : Set (X → Bool)) (h0count : H₀.Countable) (h0meas : ∀ h ∈ H₀, Measurable h)
    (happrox : ∀ h ∈ H, ∃ u : ℕ → X → Bool, (∀ n, u n ∈ H₀) ∧
      ∀ x, ∀ᶠ n in atTop, u n x = h x)
    (μ : Measure (X × Bool)) [IsProbabilityMeasure μ]
    (d m : ℕ) (ν : ℝ)
    (hdim : vcDim H₀ ≤ (d : ℕ∞)) (hdm : d ≤ m) (hm : 0 < m) (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ h ∈ H,
        popRisk μ (zeroOneLoss h) ≤ empRisk S (zeroOneLoss h) + vcRate m d ν} := by
  refine (uniform_convergence_vc_proved H₀ h0count h0meas μ d m ν hdim hdm hm hν).trans
    (measure_mono fun S hS h hh => ?_)
  obtain ⟨u, hu0, hlim⟩ := happrox h hh
  have hlossm : ∀ n, Measurable (zeroOneLoss (u n)) := by
    intro n
    exact (Measurable.of_discrete
      (f := fun q : Bool × Bool => if q.1 = q.2 then (0 : ℝ) else 1)).comp
      (((h0meas _ (hu0 n)).comp measurable_fst).prodMk measurable_snd)
  have hpop : Tendsto (fun n => popRisk μ (zeroOneLoss (u n))) atTop
      (𝓝 (popRisk μ (zeroOneLoss h))) := by
    unfold popRisk
    refine tendsto_integral_of_dominated_convergence (fun _ => (1 : ℝ))
      (fun n => (hlossm n).aestronglyMeasurable) (integrable_const _) (fun n => ?_) ?_
    · refine ae_of_all _ fun z => ?_
      have := zeroOneLoss_mem_Icc (u n) z
      rw [Real.norm_eq_abs, abs_of_nonneg this.1]
      exact this.2
    · refine ae_of_all _ fun z => tendsto_nhds_of_eventually_eq ?_
      filter_upwards [hlim z.1] with n hn
      simp only [zeroOneLoss, hn]
  have hemp : Tendsto (fun n => empRisk S (zeroOneLoss (u n))) atTop
      (𝓝 (empRisk S (zeroOneLoss h))) := by
    refine tendsto_nhds_of_eventually_eq ?_
    have hall : ∀ᶠ n in atTop, ∀ j : Fin m, u n (S j).1 = h (S j).1 :=
      eventually_all.2 fun j => hlim (S j).1
    filter_upwards [hall] with n hn
    unfold empRisk
    congr 1
    exact sum_congr rfl fun j _ => by simp only [zeroOneLoss, hn j]
  exact le_of_tendsto_of_tendsto' hpop (hemp.add_const _) fun n => hS _ (hu0 n)

end ThresholdOpt.UniformConvergence
