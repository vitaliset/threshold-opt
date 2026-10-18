import Mathlib.Combinatorics.SetFamily.Shatter
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import ThresholdOpt.Capacity

/-!
# Sauer's lemma for the growth function

The number of labelings of `m` points realized by a class of binary functions
of VC dimension at most `d` (the restriction of the class to the sample) is at
most `∑_{k ≤ d} C(m, k)` (Mohri et al., Theorem 3.17), hence at most
`(e m / d)^d` for `1 ≤ d ≤ m` (Corollary 3.18). The first bound is Mathlib's
Sauer-Shelah lemma for finite set families (`Finset.card_le_card_shatterer`,
`Finset.card_shatterer_le_sum_vcDim`), transported from the positions
`Fin m` of the sample to the points; the second is the book's computation.

The sample may repeat points. A set of positions shattered by the restricted
class names pairwise distinct points (two positions holding the same point get
the same label), so its image is a shattered set of points of the same size.
-/

namespace ThresholdOpt.Sauer

open ThresholdOpt.Capacity Finset Real

variable {X : Type*}

open scoped Classical in
/-- The labelings of the sample `x` realized by `H`. -/
noncomputable def patterns (H : Set (X → Bool)) {m : ℕ} (x : Fin m → X) :
    Finset (Fin m → Bool) :=
  univ.filter fun b => ∃ h ∈ H, ∀ i, h (x i) = b i

open scoped Classical in
theorem mem_patterns {H : Set (X → Bool)} {m : ℕ} {x : Fin m → X} {b : Fin m → Bool} :
    b ∈ patterns H x ↔ ∃ h ∈ H, ∀ i, h (x i) = b i := by
  simp [patterns]

theorem pattern_mem {H : Set (X → Bool)} {m : ℕ} (x : Fin m → X) {h : X → Bool}
    (hh : h ∈ H) : (fun i => h (x i)) ∈ patterns H x :=
  mem_patterns.2 ⟨h, hh, fun _ => rfl⟩

theorem patterns_nonempty {H : Set (X → Bool)} (hH : H.Nonempty) {m : ℕ}
    (x : Fin m → X) : (patterns H x).Nonempty := by
  obtain ⟨h, hh⟩ := hH
  exact ⟨_, pattern_mem x hh⟩

open scoped Classical in
/-- **Sauer's lemma** (Mohri et al., Theorem 3.17): a class of VC dimension at
most `d` realizes at most `∑_{k ≤ d} C(m, k)` labelings of any `m` points. -/
theorem card_patterns_le_sum (H : Set (X → Bool)) {d : ℕ} (hd : vcDim H ≤ d)
    {m : ℕ} (x : Fin m → X) :
    (patterns H x).card ≤ ∑ k ∈ Iic d, m.choose k := by
  -- The family of positions labelled `true`.
  let toSet : (Fin m → Bool) → Finset (Fin m) := fun b => univ.filter fun i => b i = true
  have hinj : Function.Injective toSet := by
    intro b b' hbb'
    funext i
    have := congrArg (fun s => i ∈ s) hbb'
    simp only [toSet, mem_filter, mem_univ, true_and, eq_iff_iff] at this
    cases hb : b i <;> cases hb' : b' i <;> simp_all
  let 𝒜 : Finset (Finset (Fin m)) := (patterns H x).image toSet
  have hcard : (patterns H x).card = 𝒜.card := (card_image_of_injective _ hinj).symm
  -- Every set of positions shattered by `𝒜` has at most `d` elements.
  have hvc : 𝒜.vcDim ≤ d := by
    refine Finset.sup_le fun s hs => ?_
    have hsh : 𝒜.Shatters s := mem_shatterer.1 hs
    -- A shattered set of positions names distinct points.
    have hinjOn : Set.InjOn x (s : Set (Fin m)) := by
      intro i hi j hj hxij
      by_contra hne
      obtain ⟨u, hu, hsu⟩ := hsh.exists_inter_eq_singleton hi
      obtain ⟨b, hb, rfl⟩ := mem_image.1 hu
      obtain ⟨h, -, hhb⟩ := mem_patterns.1 hb
      have hi' : i ∈ s ∩ toSet b := by rw [hsu]; exact mem_singleton_self i
      have hbi : b i = true := by simpa [toSet] using (mem_inter.1 hi').2
      have hbj : b j = true := by rw [← hhb j, ← hxij, hhb i]; exact hbi
      have hj' : j ∈ s ∩ toSet b := mem_inter.2 ⟨hj, by simpa [toSet] using hbj⟩
      rw [hsu, mem_singleton] at hj'
      exact hne hj'.symm
    -- Its image is shattered by `H`.
    have hshH : Shatters H (s.image x) := by
      intro ℓ
      obtain ⟨u, hu, hsu⟩ := hsh (filter_subset (fun i => ℓ (x i) = true) s)
      obtain ⟨b, hb, rfl⟩ := mem_image.1 hu
      obtain ⟨h, hh, hhb⟩ := mem_patterns.1 hb
      refine ⟨h, hh, fun y hy => ?_⟩
      obtain ⟨i, hi, rfl⟩ := mem_image.1 hy
      have hmem : i ∈ s ∩ toSet b ↔ i ∈ s.filter fun i => ℓ (x i) = true := by
        rw [hsu]
      simp only [mem_inter, mem_filter, toSet, mem_univ, true_and] at hmem
      have h2 := hmem
      rw [hhb i]
      cases hb' : b i <;> cases hl : ℓ (x i) <;> simp_all
    have hle := le_vcDim hshH
    rw [card_image_of_injOn hinjOn] at hle
    exact_mod_cast hle.trans hd
  rw [hcard]
  refine (card_le_card_shatterer 𝒜).trans ?_
  refine (card_shatterer_le_sum_vcDim (𝒜 := 𝒜)).trans ?_
  simp only [Fintype.card_fin]
  exact sum_le_sum_of_subset (Iic_subset_Iic.2 hvc)

/-- **Corollary 3.18 of Mohri et al.**: `∑_{k ≤ d} C(m, k) ≤ (e m / d)^d` for
`1 ≤ d ≤ m`. -/
theorem sum_choose_le_pow {d m : ℕ} (hd : 0 < d) (hdm : d ≤ m) :
    ((∑ k ∈ Iic d, m.choose k : ℕ) : ℝ) ≤ (exp 1 * m / d) ^ d := by
  have hdR : (0 : ℝ) < d := by exact_mod_cast hd
  have hmR : (0 : ℝ) < m := lt_of_lt_of_le hdR (by exact_mod_cast hdm)
  have hdm' : (d : ℝ) ≤ m := by exact_mod_cast hdm
  set q : ℝ := d / m with hq
  have hq0 : 0 < q := div_pos hdR hmR
  have hq1 : q ≤ 1 := (div_le_one hmR).2 hdm'
  push_cast
  calc ∑ k ∈ Iic d, (m.choose k : ℝ)
      ≤ ∑ k ∈ Iic d, (m.choose k : ℝ) * (q ^ k / q ^ d) := by
        refine sum_le_sum fun k hk => ?_
        have hk' : k ≤ d := mem_Iic.1 hk
        have : 1 ≤ q ^ k / q ^ d := by
          rw [le_div_iff₀ (by positivity), one_mul]
          exact pow_le_pow_of_le_one hq0.le hq1 hk'
        nlinarith [(Nat.cast_nonneg (m.choose k) : (0 : ℝ) ≤ m.choose k)]
    _ ≤ ∑ k ∈ range (m + 1), (m.choose k : ℝ) * (q ^ k / q ^ d) := by
        refine sum_le_sum_of_subset_of_nonneg ?_ fun k _ _ => by positivity
        intro k hk
        rw [mem_range]
        have := mem_Iic.1 hk
        omega
    _ = (1 + q) ^ m / q ^ d := by
        rw [show (1 : ℝ) + q = q + 1 by ring, add_pow, Finset.sum_div]
        refine sum_congr rfl fun k _ => ?_
        rw [one_pow, mul_one]
        ring
    _ ≤ exp q ^ m / q ^ d := by
        refine div_le_div_of_nonneg_right ?_ (by positivity)
        refine pow_le_pow_left₀ (by positivity) ?_ m
        have := add_one_le_exp q
        linarith
    _ = (exp 1 * m / d) ^ d := by
        have h1 : exp q ^ m = exp d := by
          rw [← exp_nat_mul, hq, mul_div_cancel₀ _ hmR.ne']
        have h2 : (exp 1 * m / d) ^ d = exp d * ((m : ℝ) / d) ^ d := by
          rw [mul_div_assoc, mul_pow, ← exp_nat_mul, mul_one]
        rw [h1, h2, hq, div_eq_mul_inv, ← inv_pow, inv_div]

/-- **The growth-function factor of Corollary 3.19.** For `d ≤ m`,
`log |H|_x| ≤ d log(e m / d)` (both sides `0` when `d = 0`). -/
theorem log_card_patterns_le (H : Set (X → Bool)) {d m : ℕ} (hd : vcDim H ≤ d)
    (hdm : d ≤ m) (x : Fin m → X) :
    log (patterns H x).card ≤ d * log (exp 1 * m / d) := by
  have hle := card_patterns_le_sum H hd x
  rcases Nat.eq_zero_or_pos d with rfl | hd0
  · have hI : Iic (0 : ℕ) = {0} := by
      ext k
      simp
    have : (patterns H x).card ≤ 1 := by simpa [hI] using hle
    rcases Nat.le_one_iff_eq_zero_or_eq_one.1 this with h | h <;> simp [h]
  · have hcard : ((patterns H x).card : ℝ) ≤ (exp 1 * m / d) ^ d :=
      (by exact_mod_cast hle : ((patterns H x).card : ℝ)
        ≤ ((∑ k ∈ Iic d, m.choose k : ℕ) : ℝ)).trans (sum_choose_le_pow hd0 hdm)
    rcases Nat.eq_zero_or_pos (patterns H x).card with h0 | hpos
    · rw [h0, Nat.cast_zero, log_zero]
      have hdR : (0 : ℝ) < d := by exact_mod_cast hd0
      have hmR : (0 : ℝ) < m := lt_of_lt_of_le hdR (by exact_mod_cast hdm)
      have : 0 ≤ log (exp 1 * m / d) := log_nonneg (by
        rw [le_div_iff₀ hdR, one_mul]
        have : (d : ℝ) ≤ m := by exact_mod_cast hdm
        nlinarith [add_one_le_exp (1 : ℝ)])
      positivity
    · have hposR : (0 : ℝ) < (patterns H x).card := by exact_mod_cast hpos
      calc log (patterns H x).card ≤ log ((exp 1 * m / d) ^ d) :=
            log_le_log hposR hcard
        _ = d * log (exp 1 * m / d) := by rw [log_pow]

end ThresholdOpt.Sauer
