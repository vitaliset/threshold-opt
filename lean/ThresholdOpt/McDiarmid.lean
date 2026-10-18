import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.Probability.Moments.SubGaussian

/-!
# McDiarmid's inequality

Mohri, Rostamizadeh and Talwalkar, *Foundations of Machine Learning*, 2nd ed.,
Theorem D.8, eq. (D.16): if `f : Z^n → ℝ` changes by at most `c_i` when the
`i`-th coordinate alone is changed, and `X_1, …, X_n` are independent with
law `μ`, then for every `ε > 0`

`P[f(X) − E f(X) ≥ ε] ≤ exp(−2ε² / ∑ c_i²)`.

The book proves it with Azuma's inequality on the Doob martingale. Mathlib has
Azuma-Hoeffding (`ProbabilityTheory.measure_sum_ge_le_of_hasCondSubgaussianMGF`)
only on standard Borel spaces, which an arbitrary sample space `Z` need not be.
The proof here avoids conditional expectations: on the product measure,
integrate out the first coordinate. For fixed values `y` of the others,
`z ↦ f(z, y)` ranges over an interval of length `c_0`, so Hoeffding's lemma
(Mathlib's `ProbabilityTheory.hasSubgaussianMGF_of_mem_Icc`) bounds its
centred moment generating function by `exp(t² c_0² / 8)`; the average
`g(y) = ∫ f(z, y) dμ(z)` has bounded differences `c_1, …, c_{n−1}`, and
induction on `n` bounds the rest. The result is the sub-Gaussian bound
`E exp(t (f − E f)) ≤ exp(t² ∑ c_i² / 8)` (`hasSubgaussianMGF`), from which
the tail bound follows by Chernoff (`measure_ge_le`).

`f` is assumed measurable and bounded. Boundedness follows from bounded
differences; it is kept as a hypothesis because every use here has it at hand.
-/

namespace ThresholdOpt.McDiarmid

open MeasureTheory ProbabilityTheory Real
open scoped NNReal

variable {Z : Type*} [MeasurableSpace Z]

/-- `f` changes by at most `c i` when coordinate `i` alone changes. -/
def BoundedDiff {n : ℕ} (f : (Fin n → Z) → ℝ) (c : Fin n → ℝ) : Prop :=
  ∀ (x : Fin n → Z) (i : Fin n) (z : Z), |f (Function.update x i z) - f x| ≤ c i

/-- A measurable function bounded in absolute value is integrable for a finite
measure. -/
theorem integrable_of_abs_le {α : Type*} [MeasurableSpace α] {ν : Measure α}
    [IsFiniteMeasure ν] {φ : α → ℝ} (hφ : Measurable φ) (B : ℝ)
    (hB : ∀ x, |φ x| ≤ B) : Integrable φ ν :=
  (integrable_const B).mono' hφ.aestronglyMeasurable
    (ae_of_all _ fun x => by rw [Real.norm_eq_abs]; exact hB x)

theorem abs_integral_le_of_abs_le {α : Type*} [MeasurableSpace α] {ν : Measure α}
    [IsProbabilityMeasure ν] {φ : α → ℝ} {B : ℝ} (hB : ∀ x, |φ x| ≤ B) :
    |∫ x, φ x ∂ν| ≤ B := by
  have h := norm_integral_le_of_norm_le_const (μ := ν) (f := φ) (C := B)
    (ae_of_all _ fun x => by rw [Real.norm_eq_abs]; exact hB x)
  simpa [Real.norm_eq_abs] using h

theorem exp_mul_le_of_abs_le {t w B : ℝ} (h : |w| ≤ B) :
    exp (t * w) ≤ exp (|t| * B) := by
  refine exp_le_exp.2 ((le_abs_self _).trans ?_)
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_left h (abs_nonneg t)

theorem abs_exp_mul_le_of_abs_le {t w B : ℝ} (h : |w| ≤ B) :
    |exp (t * w)| ≤ exp (|t| * B) := by
  rw [abs_of_pos (exp_pos _)]
  exact exp_mul_le_of_abs_le h

omit [MeasurableSpace Z] in
theorem BoundedDiff.nonneg {n : ℕ} [Nonempty Z] {f : (Fin n → Z) → ℝ}
    {c : Fin n → ℝ} (hc : BoundedDiff f c) (i : Fin n) : 0 ≤ c i := by
  obtain ⟨z⟩ := (inferInstance : Nonempty Z)
  exact (abs_nonneg _).trans (hc (fun _ => z) i z)

/-- **The moment generating function bound behind McDiarmid's inequality.**
For a measurable, bounded `f` with bounded differences `c`, under the product
of `n` copies of a probability measure,
`E exp(t (f − E f)) ≤ exp(t² ∑ c_i² / 8)`. -/
theorem integral_exp_le (μ : Measure Z) [IsProbabilityMeasure μ] :
    ∀ (n : ℕ) (f : (Fin n → Z) → ℝ) (c : Fin n → ℝ), Measurable f →
      (∃ M, ∀ x, |f x| ≤ M) → BoundedDiff f c → ∀ t : ℝ,
      ∫ x, exp (t * (f x - ∫ y, f y ∂(Measure.pi fun _ => μ)))
          ∂(Measure.pi fun _ => μ)
        ≤ exp (t ^ 2 * (∑ i, c i ^ 2) / 8) := by
  have hZ : Nonempty Z := by
    by_contra h
    rw [not_nonempty_iff] at h
    have : μ Set.univ = 0 := by simp [Set.univ_eq_empty_iff.2 h]
    simp at this
  intro n
  induction n with
  | zero =>
    intro f c _ _ _ t
    have hconst : f = fun _ => f default :=
      funext fun x => congrArg f (Subsingleton.elim _ _)
    rw [hconst]
    simp
  | succ n ih =>
    intro f c hf ⟨M, hM⟩ hc t
    set Pn : Measure (Fin n → Z) := Measure.pi fun _ => μ with hPn
    let e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => Z) 0
    have hmp : MeasurePreserving e (Measure.pi fun _ => μ) (μ.prod Pn) :=
      measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => μ) 0
    -- `F (z, y) = f (cons z y)`.
    let F : Z × (Fin n → Z) → ℝ := fun p => f (Fin.cons p.1 p.2)
    have hFe : ∀ x, F (e x) = f x := by
      intro x
      simp only [F, e, MeasurableEquiv.piFinSuccAbove_apply]
      congr 1
      exact Fin.cons_self_tail x
    have hF : Measurable F := by
      have : F = f ∘ e.symm := by
        funext p
        have := hFe (e.symm p)
        rw [MeasurableEquiv.apply_symm_apply] at this
        exact this
      rw [this]
      exact hf.comp e.symm.measurable
    have hFM : ∀ p, |F p| ≤ M := fun p => hM _
    have hFint : Integrable F (μ.prod Pn) := integrable_of_abs_le hF M hFM
    -- The average over the first coordinate.
    let g : (Fin n → Z) → ℝ := fun y => ∫ z, F (z, y) ∂μ
    have hg : Measurable g := hF.stronglyMeasurable.integral_prod_left'.measurable
    have hgM : ∀ y, |g y| ≤ M := fun y => abs_integral_le_of_abs_le fun z => hFM _
    have hgc : BoundedDiff g (fun j => c j.succ) := by
      intro y j w
      have hint1 : Integrable (fun z => F (z, Function.update y j w)) μ :=
        integrable_of_abs_le (hF.comp (measurable_id.prodMk measurable_const)) M
          fun z => hFM _
      have hint2 : Integrable (fun z => F (z, y)) μ :=
        integrable_of_abs_le (hF.comp (measurable_id.prodMk measurable_const)) M
          fun z => hFM _
      simp only [g]
      rw [← integral_sub hint1 hint2]
      refine abs_integral_le_of_abs_le fun z => ?_
      simp only [F]
      rw [Fin.cons_update]
      exact hc _ _ _
    -- `E f = E g`.
    have hEf : ∫ x, f x ∂(Measure.pi fun _ => μ) = ∫ y, g y ∂Pn := by
      calc ∫ x, f x ∂(Measure.pi fun _ => μ)
          = ∫ x, F (e x) ∂(Measure.pi fun _ => μ) := by simp only [hFe]
        _ = ∫ p, F p ∂(μ.prod Pn) := hmp.integral_comp' (g := F)
        _ = ∫ y, g y ∂Pn := integral_prod_symm F hFint
    set Eg := ∫ y, g y ∂Pn with hEg
    have hEgM : |Eg| ≤ M := abs_integral_le_of_abs_le hgM
    have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM (fun _ => hZ.some))
    -- Hoeffding's lemma in the first coordinate.
    have hc0 : 0 ≤ c 0 := hc.nonneg 0
    have hinner : ∀ y, ∫ z, exp (t * (F (z, y) - g y)) ∂μ
        ≤ exp (t ^ 2 * c 0 ^ 2 / 8) := by
      intro y
      set X : Z → ℝ := fun z => F (z, y) with hX
      have hXm : Measurable X := hF.comp (measurable_id.prodMk measurable_const)
      have hbdd : BddBelow (Set.range X) :=
        ⟨-M, by rintro _ ⟨z, rfl⟩; exact neg_le_of_abs_le (hFM _)⟩
      set a := ⨅ z, X z with ha
      have hmem : ∀ z, X z ∈ Set.Icc a (a + c 0) := by
        intro z
        refine ⟨ciInf_le hbdd z, ?_⟩
        have : X z - c 0 ≤ a := by
          refine le_ciInf fun z' => ?_
          have h := hc (Fin.cons z y) 0 z'
          rw [Fin.update_cons_zero] at h
          have := (abs_le.1 h).1
          simp only [X, F]
          linarith
        linarith
      have hsub := hasSubgaussianMGF_of_mem_Icc (μ := μ) hXm.aemeasurable
        (ae_of_all _ hmem)
      have hmgf := hsub.mgf_le t
      have hgX : μ[X] = g y := rfl
      rw [hgX] at hmgf
      simp only [mgf] at hmgf
      refine hmgf.trans (le_of_eq ?_)
      congr 1
      have : ((‖a + c 0 - a‖₊ / 2) ^ 2 : ℝ≥0) = ((c 0 / 2) ^ 2 : ℝ) := by
        simp only [add_sub_cancel_left, NNReal.coe_pow, NNReal.coe_div,
          coe_nnnorm, Real.norm_eq_abs, abs_of_nonneg hc0]
        norm_num
      rw [this]
      ring
    -- Integrability of the integrands.
    have hexpF : Integrable (fun p : Z × (Fin n → Z) => exp (t * (F p - Eg)))
        (μ.prod Pn) :=
      integrable_of_abs_le (by fun_prop) (exp (|t| * (M + M)))
        fun p => abs_exp_mul_le_of_abs_le
          ((abs_sub _ _).trans (add_le_add (hFM p) hEgM))
    have hexpg : Integrable (fun y => exp (t * (g y - Eg))) Pn :=
      integrable_of_abs_le (by fun_prop) (exp (|t| * (M + M)))
        fun y => abs_exp_mul_le_of_abs_le
          ((abs_sub _ _).trans (add_le_add (hgM y) hEgM))
    have hIH := ih g (fun j => c j.succ) hg ⟨M, hgM⟩ hgc t
    calc ∫ x, exp (t * (f x - ∫ y, f y ∂(Measure.pi fun _ => μ)))
            ∂(Measure.pi fun _ => μ)
        = ∫ x, exp (t * (F (e x) - Eg)) ∂(Measure.pi fun _ => μ) := by
          rw [hEf]
          simp only [hFe]
      _ = ∫ p, exp (t * (F p - Eg)) ∂(μ.prod Pn) :=
          hmp.integral_comp' (g := fun p => exp (t * (F p - Eg)))
      _ = ∫ y, ∫ z, exp (t * (F (z, y) - Eg)) ∂μ ∂Pn :=
          integral_prod_symm _ hexpF
      _ = ∫ y, exp (t * (g y - Eg)) * ∫ z, exp (t * (F (z, y) - g y)) ∂μ ∂Pn := by
          refine integral_congr_ae (ae_of_all _ fun y => ?_)
          simp only
          rw [← integral_const_mul]
          refine integral_congr_ae (ae_of_all _ fun z => ?_)
          simp only
          rw [← exp_add]
          congr 1
          ring
      _ ≤ ∫ y, exp (t * (g y - Eg)) * exp (t ^ 2 * c 0 ^ 2 / 8) ∂Pn := by
          refine integral_mono_of_nonneg (ae_of_all _ fun y => ?_)
            (hexpg.mul_const _) (ae_of_all _ fun y => ?_)
          · refine mul_nonneg (exp_pos _).le (integral_nonneg fun z => (exp_pos _).le)
          · exact mul_le_mul_of_nonneg_left (hinner y) (exp_pos _).le
      _ = (∫ y, exp (t * (g y - Eg)) ∂Pn) * exp (t ^ 2 * c 0 ^ 2 / 8) :=
          integral_mul_const _ _
      _ ≤ exp (t ^ 2 * (∑ j : Fin n, c j.succ ^ 2) / 8)
            * exp (t ^ 2 * c 0 ^ 2 / 8) :=
          mul_le_mul_of_nonneg_right hIH (exp_pos _).le
      _ = exp (t ^ 2 * (∑ i, c i ^ 2) / 8) := by
          rw [← exp_add, Fin.sum_univ_succ]
          congr 1
          ring

/-- The sub-Gaussian parameter `∑ c_i² / 4`. -/
noncomputable def param {n : ℕ} (c : Fin n → ℝ) : ℝ≥0 :=
  ⟨(∑ i, c i ^ 2) / 4, by positivity⟩

/-- **`f − E f` is sub-Gaussian with parameter `∑ c_i² / 4`.** -/
theorem hasSubgaussianMGF (μ : Measure Z) [IsProbabilityMeasure μ] {n : ℕ}
    (f : (Fin n → Z) → ℝ) (c : Fin n → ℝ) (hf : Measurable f)
    (hM : ∃ M, ∀ x, |f x| ≤ M) (hc : BoundedDiff f c) :
    HasSubgaussianMGF (fun x => f x - ∫ y, f y ∂(Measure.pi fun _ => μ)) (param c)
      (Measure.pi fun _ => μ) where
  integrable_exp_mul t := by
    obtain ⟨M, hM⟩ := hM
    have hE : |∫ y, f y ∂(Measure.pi fun _ => μ)| ≤ M := abs_integral_le_of_abs_le hM
    exact integrable_of_abs_le (by fun_prop) (exp (|t| * (M + M)))
      fun x => abs_exp_mul_le_of_abs_le ((abs_sub _ _).trans (add_le_add (hM x) hE))
  mgf_le t := by
    have hp : ((param c : ℝ≥0) : ℝ) = (∑ i, c i ^ 2) / 4 := rfl
    refine (integral_exp_le μ n f c hf hM hc t).trans (le_of_eq ?_)
    rw [hp]
    congr 1
    ring

/-- **McDiarmid's inequality** (Mohri et al., Theorem D.8, eq. (D.16)). For a
measurable, bounded `f : Z^n → ℝ` with bounded differences `c` and `ε ≥ 0`,
under the product of `n` copies of a probability measure,
`P[f − E f ≥ ε] ≤ exp(−2ε² / ∑ c_i²)`. -/
theorem measure_ge_le (μ : Measure Z) [IsProbabilityMeasure μ] {n : ℕ}
    (f : (Fin n → Z) → ℝ) (c : Fin n → ℝ) (hf : Measurable f)
    (hM : ∃ M, ∀ x, |f x| ≤ M) (hc : BoundedDiff f c) {ε : ℝ} (hε : 0 ≤ ε) :
    (Measure.pi fun _ => μ).real
        {x | ε ≤ f x - ∫ y, f y ∂(Measure.pi fun _ => μ)}
      ≤ exp (-2 * ε ^ 2 / ∑ i, c i ^ 2) := by
  refine ((hasSubgaussianMGF μ f c hf hM hc).measure_ge_le hε).trans (le_of_eq ?_)
  have hp : ((param c : ℝ≥0) : ℝ) = (∑ i, c i ^ 2) / 4 := rfl
  rw [hp]
  congr 1
  rcases eq_or_ne (∑ i, c i ^ 2) 0 with h0 | h0
  · rw [h0]
    simp
  · field_simp
    ring

end ThresholdOpt.McDiarmid
