import Mathlib.Algebra.BigOperators.Field
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Series
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Rademacher averages and Massart's lemma

Mohri, Rostamizadeh and Talwalkar, *Foundations of Machine Learning*, 2nd ed.,
Theorem 3.7 (Massart's lemma), eq. (3.20): for a finite set `A ⊆ ℝ^m` with
`‖u‖₂ ≤ r` for every `u ∈ A`,

`E_σ [max_{u ∈ A} ∑_i σ_i u_i] ≤ r √(2 log |A|)`,

with `σ` uniform on `{−1, +1}^m` (the book divides both sides by `m`). The
expectation over `σ` is a finite average, `radAvg`, over `σ : Fin m → Bool`
with the sign `sgn true = 1`, `sgn false = −1`; nothing measure-theoretic is
needed.

The proof is the book's (through Corollary D.11 and Theorem D.10): Jensen's
inequality for `exp`, the maximum bounded by the sum, independence of the
coordinates (`radAvg_exp_inner`: the average of `exp(t ⟨σ, u⟩)` is
`∏ cosh(t u_i)`), `cosh x ≤ exp(x²/2)`, and the choice of `t`.
-/

namespace ThresholdOpt.Massart

open Real Finset

/-- The Rademacher sign of a bit. -/
def sgn (b : Bool) : ℝ := if b then 1 else -1

@[simp] theorem sgn_true : sgn true = 1 := rfl

@[simp] theorem sgn_false : sgn false = -1 := rfl

@[simp] theorem sgn_not (b : Bool) : sgn (!b) = -sgn b := by
  cases b <;> simp [sgn]

theorem sgn_sq (b : Bool) : sgn b ^ 2 = 1 := by
  cases b <;> norm_num [sgn]

/-- The average over the `2^m` sign vectors. -/
noncomputable def radAvg {m : ℕ} (F : (Fin m → Bool) → ℝ) : ℝ :=
  (∑ σ, F σ) / 2 ^ m

theorem radAvg_mono {m : ℕ} {F G : (Fin m → Bool) → ℝ} (h : ∀ σ, F σ ≤ G σ) :
    radAvg F ≤ radAvg G :=
  div_le_div_of_nonneg_right (sum_le_sum fun σ _ => h σ) (by positivity)

theorem radAvg_add {m : ℕ} (F G : (Fin m → Bool) → ℝ) :
    radAvg (fun σ => F σ + G σ) = radAvg F + radAvg G := by
  simp only [radAvg, sum_add_distrib, add_div]

theorem radAvg_const {m : ℕ} (a : ℝ) : radAvg (fun _ : Fin m → Bool => a) = a := by
  simp only [radAvg, sum_const, card_univ, Fintype.card_fun, Fintype.card_bool,
    Fintype.card_fin, nsmul_eq_mul]
  push_cast
  field_simp

theorem radAvg_mul_left {m : ℕ} (a : ℝ) (F : (Fin m → Bool) → ℝ) :
    radAvg (fun σ => a * F σ) = a * radAvg F := by
  simp only [radAvg, ← mul_sum, mul_div_assoc]

theorem radAvg_sum {m : ℕ} {ι : Type*} (s : Finset ι) (F : ι → (Fin m → Bool) → ℝ) :
    radAvg (fun σ => ∑ j ∈ s, F j σ) = ∑ j ∈ s, radAvg (F j) := by
  simp only [radAvg]
  rw [sum_comm, Finset.sum_div]

/-- Flipping every sign preserves the average. -/
theorem radAvg_flip {m : ℕ} (F : (Fin m → Bool) → ℝ) :
    radAvg (fun σ => F (fun j => !σ j)) = radAvg F := by
  simp only [radAvg]
  congr 1
  let e : (Fin m → Bool) ≃ (Fin m → Bool) :=
    ⟨fun σ j => !σ j, fun σ j => !σ j, fun σ => by simp, fun σ => by simp⟩
  exact Equiv.sum_comp e F

/-- The average of a linear form in the signs is `0`. -/
theorem radAvg_inner {m : ℕ} (c : Fin m → ℝ) :
    radAvg (fun σ => ∑ i, sgn (σ i) * c i) = 0 := by
  have h := radAvg_flip (fun σ : Fin m → Bool => ∑ i, sgn (σ i) * c i)
  have hneg : (fun σ : Fin m → Bool => ∑ i, sgn ((fun j => !σ j) i) * c i)
      = fun σ => -1 * ∑ i, sgn (σ i) * c i := by
    funext σ
    simp only [sgn_not, neg_mul, sum_neg_distrib]
    ring
  rw [hneg, radAvg_mul_left] at h
  linarith

/-- **Independence of the signs.** The average of `exp(t ∑ σ_i u_i)` is
`∏ cosh(t u_i)`. -/
theorem radAvg_exp_inner {m : ℕ} (t : ℝ) (u : Fin m → ℝ) :
    radAvg (fun σ => exp (t * ∑ i, sgn (σ i) * u i)) = ∏ i, cosh (t * u i) := by
  have hprod : ∀ σ : Fin m → Bool,
      exp (t * ∑ i, sgn (σ i) * u i) = ∏ i, exp (t * (sgn (σ i) * u i)) := by
    intro σ
    rw [mul_sum, exp_sum]
  simp only [radAvg, hprod]
  have hsum : ∑ σ : Fin m → Bool, ∏ i, exp (t * (sgn (σ i) * u i))
      = ∏ i, ∑ b : Bool, exp (t * (sgn b * u i)) :=
    (Fintype.prod_sum fun i b => exp (t * (sgn b * u i))).symm
  rw [hsum]
  have hcosh : ∀ i, ∑ b : Bool, exp (t * (sgn b * u i)) = 2 * cosh (t * u i) := by
    intro i
    rw [Fintype.sum_bool, cosh_eq]
    simp only [sgn_true, sgn_false]
    ring_nf
  simp only [hcosh, prod_mul_distrib, prod_const, card_univ, Fintype.card_fin]
  field_simp

/-- **Jensen's inequality for the sign average**: `exp (avg F) ≤ avg (exp F)`. -/
theorem exp_radAvg_le {m : ℕ} (F : (Fin m → Bool) → ℝ) :
    exp (radAvg F) ≤ radAvg (fun σ => exp (F σ)) := by
  have hw : ∀ σ ∈ (univ : Finset (Fin m → Bool)), (0 : ℝ) ≤ 1 / 2 ^ m :=
    fun _ _ => by positivity
  have hw1 : ∑ _σ ∈ (univ : Finset (Fin m → Bool)), (1 : ℝ) / 2 ^ m = 1 := by
    simp only [sum_const, card_univ, Fintype.card_fun, Fintype.card_bool,
      Fintype.card_fin, nsmul_eq_mul]
    push_cast
    field_simp
  have h := convexOn_exp.map_sum_le hw hw1 (fun σ _ => Set.mem_univ (F σ))
  simp only [smul_eq_mul] at h
  have e1 : ∑ σ, 1 / 2 ^ m * F σ = radAvg F := by
    simp only [radAvg, ← mul_sum]
    ring
  have e2 : ∑ σ, 1 / 2 ^ m * exp (F σ) = radAvg fun σ => exp (F σ) := by
    simp only [radAvg, ← mul_sum]
    ring
  rw [e1, e2] at h
  exact h

/-- From `L ≤ a/t + b t` for every `t > 0`, with `a, b ≥ 0`, to
`L ≤ 2 √(a b)`. -/
theorem le_two_sqrt_mul_of_forall {L a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b)
    (h : ∀ t, 0 < t → L ≤ a / t + b * t) : L ≤ 2 * sqrt (a * b) := by
  rcases ha.lt_or_eq with ha' | ha'
  · rcases hb.lt_or_eq with hb' | hb'
    · have hsa := sqrt_pos.2 ha'
      have hsb := sqrt_pos.2 hb'
      have ht := h (sqrt a / sqrt b) (div_pos hsa hsb)
      have e1 : a / (sqrt a / sqrt b) = sqrt a * sqrt b := by
        field_simp
        rw [sq_sqrt ha]
      have e2 : b * (sqrt a / sqrt b) = sqrt a * sqrt b := by
        field_simp
        rw [sq_sqrt hb]
      rw [e1, e2] at ht
      rw [sqrt_mul ha]
      linarith
    · subst hb'
      simp only [mul_zero, sqrt_zero, add_zero, zero_mul] at h ⊢
      by_contra hL
      push Not at hL
      have := h (2 * a / L) (by positivity)
      rw [div_div_eq_mul_div] at this
      have e : a * L / (2 * a) = L / 2 := by field_simp
      rw [e] at this
      linarith
  · subst ha'
    simp only [zero_mul, sqrt_zero, mul_zero, zero_div, zero_add] at h ⊢
    by_contra hL
    push Not at hL
    have := h (L / (2 * (b + 1))) (by positivity)
    have e : b * (L / (2 * (b + 1))) < L := by
      rw [mul_div_assoc', div_lt_iff₀ (by positivity)]
      nlinarith
    linarith

/-- **Massart's lemma** (Mohri et al., Theorem 3.7, multiplied by `m`). For a
nonempty finite set `A` of vectors with `∑_i u_i² ≤ r²` for every `u ∈ A`,
`r ≥ 0`: `E_σ [max_{u ∈ A} ∑ σ_i u_i] ≤ r √(2 log |A|)`. -/
theorem massart {m : ℕ} (A : Finset (Fin m → ℝ)) (hA : A.Nonempty) {r : ℝ}
    (hr : 0 ≤ r) (hnorm : ∀ u ∈ A, ∑ i, u i ^ 2 ≤ r ^ 2) :
    radAvg (fun σ => A.sup' hA fun u => ∑ i, sgn (σ i) * u i)
      ≤ r * sqrt (2 * log A.card) := by
  set F : (Fin m → Bool) → ℝ := fun σ => A.sup' hA fun u => ∑ i, sgn (σ i) * u i
  have hcard : (1 : ℝ) ≤ A.card := by exact_mod_cast hA.card_pos
  have hlog : 0 ≤ log A.card := log_nonneg hcard
  have key : ∀ t, 0 < t → radAvg F ≤ log A.card / t + r ^ 2 / 2 * t := by
    intro t ht
    -- `exp(t · avg max) ≤ avg exp(t · max) ≤ avg ∑_u exp(t ⟨σ, u⟩) ≤ |A| exp(t² r²/2)`.
    have h1 : exp (t * radAvg F) ≤ radAvg fun σ => exp (t * F σ) := by
      rw [← radAvg_mul_left]
      exact exp_radAvg_le _
    have h2 : ∀ σ, exp (t * F σ) ≤ ∑ u ∈ A, exp (t * ∑ i, sgn (σ i) * u i) := by
      intro σ
      obtain ⟨u, hu, hmax⟩ := A.exists_mem_eq_sup' hA fun u => ∑ i, sgn (σ i) * u i
      simp only [F, hmax]
      exact single_le_sum (f := fun u => exp (t * ∑ i, sgn (σ i) * u i))
        (fun _ _ => (exp_pos _).le) hu
    have h3 : ∀ u ∈ A, radAvg (fun σ => exp (t * ∑ i, sgn (σ i) * u i))
        ≤ exp (t ^ 2 * r ^ 2 / 2) := by
      intro u hu
      rw [radAvg_exp_inner]
      calc ∏ i, cosh (t * u i) ≤ ∏ i, exp ((t * u i) ^ 2 / 2) :=
            prod_le_prod (fun i _ => (cosh_pos _).le) fun i _ => cosh_le_exp_half_sq _
        _ = exp (t ^ 2 * ∑ i, u i ^ 2 / 2) := by
            rw [← exp_sum]
            congr 1
            rw [mul_sum]
            exact sum_congr rfl fun i _ => by ring
        _ ≤ exp (t ^ 2 * r ^ 2 / 2) := by
            refine exp_le_exp.2 ?_
            have h := hnorm u hu
            have ht2 : 0 ≤ t ^ 2 := sq_nonneg t
            have hsum : ∑ i, u i ^ 2 / 2 = (∑ i, u i ^ 2) / 2 :=
              (Finset.sum_div _ _ _).symm
            rw [hsum]
            have : t ^ 2 * (∑ i, u i ^ 2) ≤ t ^ 2 * r ^ 2 :=
              mul_le_mul_of_nonneg_left h ht2
            linarith
    have h4 : radAvg (fun σ => exp (t * F σ)) ≤ A.card * exp (t ^ 2 * r ^ 2 / 2) := by
      calc radAvg (fun σ => exp (t * F σ))
          ≤ radAvg (fun σ => ∑ u ∈ A, exp (t * ∑ i, sgn (σ i) * u i)) :=
            radAvg_mono h2
        _ = ∑ u ∈ A, radAvg (fun σ => exp (t * ∑ i, sgn (σ i) * u i)) :=
            radAvg_sum A _
        _ ≤ ∑ _u ∈ A, exp (t ^ 2 * r ^ 2 / 2) := sum_le_sum h3
        _ = A.card * exp (t ^ 2 * r ^ 2 / 2) := by
            rw [sum_const, nsmul_eq_mul]
    have h5 : t * radAvg F ≤ log A.card + t ^ 2 * r ^ 2 / 2 := by
      have := (h1.trans h4)
      have hpos : 0 < (A.card : ℝ) * exp (t ^ 2 * r ^ 2 / 2) := by positivity
      have := log_le_log (exp_pos _) this
      rw [log_exp, log_mul (by positivity) (exp_pos _).ne', log_exp] at this
      exact this
    rw [div_add' _ _ _ ht.ne', le_div_iff₀ ht]
    nlinarith
  have := le_two_sqrt_mul_of_forall hlog (by positivity) key
  refine this.trans (le_of_eq ?_)
  rw [show log A.card * (r ^ 2 / 2) = r ^ 2 * (2 * log A.card) / 4 by ring,
    sqrt_div' _ (by norm_num : (0:ℝ) ≤ 4), sqrt_mul' _ (by positivity), sqrt_sq hr,
    show (4 : ℝ) = 2 ^ 2 by norm_num, sqrt_sq (by norm_num : (0:ℝ) ≤ 2)]
  ring

end ThresholdOpt.Massart
