import Mathlib.MeasureTheory.Integral.Pi
import ThresholdOpt.Learning
import ThresholdOpt.McDiarmid
import ThresholdOpt.Massart

/-!
# The Rademacher generalization bound

Mohri, Rostamizadeh and Talwalkar, *Foundations of Machine Learning*, 2nd ed.,
Theorem 3.3, eq. (3.3): for a family `G` of functions `Z → [0, 1]` and
`ν > 0`, with probability at least `1 − ν` over an i.i.d. sample `S` of size
`m`, every `g ∈ G` satisfies

`E[g] ≤ Ê_S[g] + 2 R_m(G) + √(log(1/ν) / (2m))`,

with `R_m(G)` the Rademacher complexity (Definition 3.2), the expectation over
samples of the empirical Rademacher complexity (eq. 3.1).

The family is indexed by a countable type `ι`; that is the measurability
assumption the book makes implicitly (its footnote 3). The proof is the
book's: McDiarmid's inequality (`McDiarmid.measure_ge_le`) for
`Φ(S) = sup_g (E[g] − Ê_S[g])`, whose differences are at most `1/m`, and the
symmetrization bound `E[Φ] ≤ 2 R_m(G)` (`integral_gapSup_le`), through a
ghost sample `S'` and, for each sign vector `σ`, the measure-preserving map
that swaps `S_j` and `S'_j` where `σ_j = −1` (`measurePreserving_swapSign`).
-/

namespace ThresholdOpt.Symmetrization

open MeasureTheory ProbabilityTheory Real Finset
open ThresholdOpt.Learning ThresholdOpt.Massart ThresholdOpt.McDiarmid

variable {Z : Type*} [MeasurableSpace Z] {ι : Type*}

/-- The empirical Rademacher complexity of `G` on the sample `S` (eq. 3.1). -/
noncomputable def empRad (G : ι → Z → ℝ) {m : ℕ} (S : Fin m → Z) : ℝ :=
  radAvg fun σ => ⨆ i, (∑ j, sgn (σ j) * G i (S j)) / m

/-- The Rademacher complexity of `G` (Definition 3.2). -/
noncomputable def radComplexity (μ : Measure Z) (G : ι → Z → ℝ) (m : ℕ) : ℝ :=
  ∫ S, empRad G S ∂(sampleMeasure μ m)

/-- The largest deviation `Φ(S) = sup_g (E[g] − Ê_S[g])` (eq. 3.5). -/
noncomputable def gapSup (μ : Measure Z) (G : ι → Z → ℝ) {m : ℕ} (S : Fin m → Z) : ℝ :=
  ⨆ i, (popRisk μ (G i) - empRisk S (G i))

/-! ### Elementary bounds -/

theorem abs_sgn (b : Bool) : |sgn b| = 1 := by
  cases b <;> norm_num [sgn]

theorem abs_signed_avg_le {m : ℕ} {f : Fin m → ℝ} (hf : ∀ j, f j ∈ Set.Icc (0 : ℝ) 1)
    (σ : Fin m → Bool) : |(∑ j, sgn (σ j) * f j) / m| ≤ 1 := by
  rcases Nat.eq_zero_or_pos m with rfl | hm
  · simp
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  rw [abs_div, abs_of_pos hmR, div_le_one hmR]
  calc |∑ j, sgn (σ j) * f j| ≤ ∑ j, |sgn (σ j) * f j| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ _j : Fin m, (1 : ℝ) := sum_le_sum fun j _ => by
        rw [abs_mul, abs_sgn, one_mul, abs_of_nonneg (hf j).1]
        exact (hf j).2
    _ = m := by simp

omit [MeasurableSpace Z] in
theorem empRisk_mem_Icc {m : ℕ} {ℓ : Z → ℝ} (hℓ : ∀ z, ℓ z ∈ Set.Icc (0 : ℝ) 1)
    (S : Fin m → Z) : empRisk S ℓ ∈ Set.Icc (0 : ℝ) 1 := by
  unfold empRisk
  rcases Nat.eq_zero_or_pos m with rfl | hm
  · simp
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  constructor
  · exact div_nonneg (sum_nonneg fun j _ => (hℓ _).1) hmR.le
  · rw [div_le_one hmR]
    calc ∑ j, ℓ (S j) ≤ ∑ _j : Fin m, (1 : ℝ) := sum_le_sum fun j _ => (hℓ _).2
      _ = m := by simp

theorem abs_le_one_of_mem_Icc {a : ℝ} (h : a ∈ Set.Icc (0 : ℝ) 1) : |a| ≤ 1 :=
  abs_le.2 ⟨by linarith [h.1], h.2⟩

theorem popRisk_mem_Icc (μ : Measure Z) [IsProbabilityMeasure μ] {ℓ : Z → ℝ}
    (hℓm : Measurable ℓ) (hℓ : ∀ z, ℓ z ∈ Set.Icc (0 : ℝ) 1) :
    popRisk μ ℓ ∈ Set.Icc (0 : ℝ) 1 := by
  unfold popRisk
  constructor
  · exact integral_nonneg fun z => (hℓ z).1
  · calc ∫ z, ℓ z ∂μ ≤ ∫ _z, (1 : ℝ) ∂μ :=
          integral_mono (integrable_of_abs_le hℓm 1 fun z => abs_le_one_of_mem_Icc (hℓ z))
            (integrable_const _) fun z => (hℓ z).2
      _ = 1 := by simp

/-- `E_S[Ê_S[ℓ]] = E[ℓ]`. -/
theorem integral_empRisk (μ : Measure Z) [IsProbabilityMeasure μ] {m : ℕ} (hm : 0 < m)
    {ℓ : Z → ℝ} (hℓm : Measurable ℓ) (hℓ : ∀ z, ℓ z ∈ Set.Icc (0 : ℝ) 1) :
    ∫ S, empRisk S ℓ ∂(sampleMeasure μ m) = popRisk μ ℓ := by
  have hmR : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  unfold empRisk sampleMeasure popRisk
  rw [integral_div, integral_finsetSum]
  · have : ∀ j : Fin m, ∫ S : Fin m → Z, ℓ (S j) ∂(Measure.pi fun _ => μ) = ∫ z, ℓ z ∂μ :=
      fun j => integral_comp_eval hℓm.aestronglyMeasurable
    simp only [this, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  · intro j _
    exact integrable_comp_eval
      (integrable_of_abs_le hℓm 1 fun z => abs_le_one_of_mem_Icc (hℓ z))

omit [MeasurableSpace Z] in
theorem sum_update_sub {m : ℕ} (ℓ : Z → ℝ) (x : Fin m → Z) (j : Fin m) (z : Z) :
    ∑ k, ℓ (Function.update x j z k) - ∑ k, ℓ (x k) = ℓ z - ℓ (x j) := by
  rw [← sum_sub_distrib, sum_eq_single j]
  · simp
  · intro k _ hk
    simp [Function.update_of_ne hk]
  · simp

/-! ### The deviation `Φ` -/

section Gap

variable (μ : Measure Z) [IsProbabilityMeasure μ] (G : ι → Z → ℝ)
  (hG : ∀ i, Measurable (G i)) (hG01 : ∀ i z, G i z ∈ Set.Icc (0 : ℝ) 1)
include hG hG01

theorem abs_gap_le {m : ℕ} (S : Fin m → Z) (i : ι) :
    |popRisk μ (G i) - empRisk S (G i)| ≤ 1 := by
  have h1 := popRisk_mem_Icc μ (hG i) (hG01 i)
  have h2 := empRisk_mem_Icc (hG01 i) S
  rw [abs_le]
  constructor <;> linarith [h1.1, h1.2, h2.1, h2.2]

theorem bddAbove_gap {m : ℕ} (S : Fin m → Z) :
    BddAbove (Set.range fun i => popRisk μ (G i) - empRisk S (G i)) :=
  ⟨1, by rintro _ ⟨i, rfl⟩; exact (abs_le.1 (abs_gap_le μ G hG hG01 S i)).2⟩

theorem gap_le_gapSup {m : ℕ} (S : Fin m → Z) (i : ι) :
    popRisk μ (G i) - empRisk S (G i) ≤ gapSup μ G S :=
  le_ciSup (bddAbove_gap μ G hG hG01 S) i

theorem abs_gapSup_le [Nonempty ι] {m : ℕ} (S : Fin m → Z) : |gapSup μ G S| ≤ 1 := by
  rw [abs_le]
  constructor
  · obtain ⟨i⟩ := (inferInstance : Nonempty ι)
    have := gap_le_gapSup μ G hG hG01 S i
    linarith [(abs_le.1 (abs_gap_le μ G hG hG01 S i)).1]
  · exact ciSup_le fun i => (abs_le.1 (abs_gap_le μ G hG hG01 S i)).2

omit [IsProbabilityMeasure μ] hG01 in
theorem measurable_gapSup [Countable ι] {m : ℕ} :
    Measurable fun S : Fin m → Z => gapSup μ G S :=
  Measurable.iSup fun i => measurable_const.sub (measurable_empRisk (hG i))

/-- Changing one point of the sample moves `Φ` by at most `1/m` (eq. 3.6). -/
theorem gapSup_update_le [Nonempty ι] {m : ℕ} (x : Fin m → Z) (j : Fin m) (z : Z) :
    gapSup μ G (Function.update x j z) ≤ gapSup μ G x + 1 / m := by
  refine ciSup_le fun i => ?_
  have hsum := sum_update_sub (G i) x j z
  have hmR : (0 : ℝ) ≤ m := Nat.cast_nonneg m
  have hle : (G i (x j) - G i z) / m ≤ 1 / m :=
    div_le_div_of_nonneg_right (by linarith [(hG01 i (x j)).2, (hG01 i z).1]) hmR
  have heq : empRisk x (G i) - empRisk (Function.update x j z) (G i)
      = (G i (x j) - G i z) / m := by
    unfold empRisk
    rw [← sub_div]
    congr 1
    linarith
  have := gap_le_gapSup μ G hG hG01 x i
  linarith

theorem boundedDiff_gapSup [Nonempty ι] {m : ℕ} :
    BoundedDiff (fun S : Fin m → Z => gapSup μ G S) (fun _ => 1 / m) := by
  intro x j z
  rw [abs_le]
  constructor
  · have h := gapSup_update_le μ G hG hG01 (Function.update x j z) j (x j)
    rw [Function.update_idem, Function.update_eq_self] at h
    linarith
  · have := gapSup_update_le μ G hG hG01 x j z
    linarith

end Gap

/-! ### Symmetrization -/

/-- The swap of the two samples at the coordinates where `σ` is `false`. -/
def swapSign {m : ℕ} (σ : Fin m → Bool) (p : (Fin m → Z) × (Fin m → Z)) :
    (Fin m → Z) × (Fin m → Z) :=
  (fun j => if σ j then p.1 j else p.2 j, fun j => if σ j then p.2 j else p.1 j)

/-- For every `σ`, swapping is measure-preserving on the law of `(S, S')`. -/
theorem measurePreserving_swapSign (μ : Measure Z) [IsProbabilityMeasure μ] {m : ℕ}
    (σ : Fin m → Bool) :
    MeasurePreserving (swapSign σ) ((sampleMeasure μ m).prod (sampleMeasure μ m))
      ((sampleMeasure μ m).prod (sampleMeasure μ m)) := by
  let E := MeasurableEquiv.arrowProdEquivProdArrow Z Z (Fin m)
  have hE : MeasurePreserving E (Measure.pi fun _ : Fin m => μ.prod μ)
      ((sampleMeasure μ m).prod (sampleMeasure μ m)) :=
    measurePreserving_arrowProdEquivProdArrow Z Z (Fin m) (fun _ => μ) (fun _ => μ)
  let f : Fin m → Z × Z → Z × Z := fun j => if σ j then id else Prod.swap
  have hf : ∀ j, MeasurePreserving (f j) (μ.prod μ) (μ.prod μ) := by
    intro j
    simp only [f]
    split_ifs
    · exact MeasurePreserving.id _
    · exact Measure.measurePreserving_swap
  have hw := measurePreserving_pi (fun _ : Fin m => μ.prod μ) (fun _ => μ.prod μ) hf
  have hcomp := hE.comp (hw.comp hE.symm)
  have heq : (E ∘ (fun a j => f j (a j)) ∘ E.symm) = swapSign σ := by
    funext p
    simp only [Function.comp, swapSign, f, E]
    ext j
    · by_cases h : σ j <;> simp [h, MeasurableEquiv.arrowProdEquivProdArrow,
        Equiv.arrowProdEquivProdArrow]
    · by_cases h : σ j <;> simp [h, MeasurableEquiv.arrowProdEquivProdArrow,
        Equiv.arrowProdEquivProdArrow]
  rw [heq] at hcomp
  exact hcomp

section Symm

variable (μ : Measure Z) [IsProbabilityMeasure μ] (G : ι → Z → ℝ)
  (hG : ∀ i, Measurable (G i)) (hG01 : ∀ i z, G i z ∈ Set.Icc (0 : ℝ) 1)
include hG hG01

/-- `ψ(S, S') = sup_g (Ê_{S'}[g] − Ê_S[g])`. -/
noncomputable def ghostSup {m : ℕ} (p : (Fin m → Z) × (Fin m → Z)) : ℝ :=
  ⨆ i, (empRisk p.2 (G i) - empRisk p.1 (G i))

omit [MeasurableSpace Z] hG in
theorem abs_ghost_le {m : ℕ} (p : (Fin m → Z) × (Fin m → Z)) (i : ι) :
    |empRisk p.2 (G i) - empRisk p.1 (G i)| ≤ 1 := by
  have h1 := empRisk_mem_Icc (hG01 i) p.2
  have h2 := empRisk_mem_Icc (hG01 i) p.1
  rw [abs_le]
  constructor <;> linarith [h1.1, h1.2, h2.1, h2.2]

omit [MeasurableSpace Z] hG in
theorem bddAbove_ghost {m : ℕ} (p : (Fin m → Z) × (Fin m → Z)) :
    BddAbove (Set.range fun i => empRisk p.2 (G i) - empRisk p.1 (G i)) :=
  ⟨1, by rintro _ ⟨i, rfl⟩; exact (abs_le.1 (abs_ghost_le G hG01 p i)).2⟩

omit [MeasurableSpace Z] hG in
theorem abs_ghostSup_le [Nonempty ι] {m : ℕ} (p : (Fin m → Z) × (Fin m → Z)) :
    |ghostSup G p| ≤ 1 := by
  unfold ghostSup
  rw [abs_le]
  constructor
  · obtain ⟨i⟩ := (inferInstance : Nonempty ι)
    have := le_ciSup (bddAbove_ghost G hG01 p) i
    linarith [(abs_le.1 (abs_ghost_le G hG01 p i)).1]
  · exact ciSup_le fun i => (abs_le.1 (abs_ghost_le G hG01 p i)).2

omit hG01 in
theorem measurable_ghostSup [Countable ι] {m : ℕ} :
    Measurable fun p : (Fin m → Z) × (Fin m → Z) => ghostSup G p :=
  Measurable.iSup fun i =>
    ((measurable_empRisk (hG i)).comp measurable_snd).sub
      ((measurable_empRisk (hG i)).comp measurable_fst)

omit hG01 in
theorem measurable_empRad [Countable ι] {m : ℕ} :
    Measurable fun S : Fin m → Z => empRad G S := by
  unfold empRad radAvg
  refine Measurable.div_const (Finset.measurable_sum _ fun σ _ => ?_) _
  refine Measurable.iSup fun i => ?_
  refine Measurable.div_const (Finset.measurable_sum _ fun j _ => ?_) _
  exact measurable_const.mul ((hG i).comp (measurable_pi_apply j))

omit [MeasurableSpace Z] hG in
theorem bddAbove_signed {m : ℕ} (S : Fin m → Z) (σ : Fin m → Bool) :
    BddAbove (Set.range fun i => (∑ j, sgn (σ j) * G i (S j)) / m) :=
  ⟨1, by
    rintro _ ⟨i, rfl⟩
    exact (abs_le.1 (abs_signed_avg_le (fun j => hG01 i (S j)) σ)).2⟩

omit [MeasurableSpace Z] hG in
theorem abs_empRad_le [Nonempty ι] {m : ℕ} (S : Fin m → Z) : |empRad G S| ≤ 1 := by
  have hb : ∀ σ : Fin m → Bool, |⨆ i, (∑ j, sgn (σ j) * G i (S j)) / m| ≤ 1 := by
    intro σ
    rw [abs_le]
    constructor
    · obtain ⟨i⟩ := (inferInstance : Nonempty ι)
      have := le_ciSup (bddAbove_signed G hG01 S σ) i
      linarith [(abs_le.1 (abs_signed_avg_le (fun j => hG01 i (S j)) σ)).1]
    · exact ciSup_le fun i => (abs_le.1 (abs_signed_avg_le (fun j => hG01 i (S j)) σ)).2
  unfold empRad
  rw [abs_le]
  constructor
  · have := radAvg_mono (F := fun _ => (-1 : ℝ)) (G := fun σ =>
      ⨆ i, (∑ j, sgn (σ j) * G i (S j)) / m) fun σ => (abs_le.1 (hb σ)).1
    rwa [radAvg_const] at this
  · have := radAvg_mono (G := fun _ => (1 : ℝ)) (F := fun σ =>
      ⨆ i, (∑ j, sgn (σ j) * G i (S j)) / m) fun σ => (abs_le.1 (hb σ)).2
    rwa [radAvg_const] at this

omit [MeasurableSpace Z] hG hG01 in
/-- `ψ` after the swap `σ` is the signed difference of the two samples. -/
theorem ghostSup_swapSign {m : ℕ} (σ : Fin m → Bool) (p : (Fin m → Z) × (Fin m → Z)) :
    ghostSup G (swapSign σ p)
      = ⨆ i, (∑ j, sgn (σ j) * (G i (p.2 j) - G i (p.1 j))) / m := by
  unfold ghostSup
  congr 1
  funext i
  unfold empRisk swapSign
  rw [← sub_div, ← sum_sub_distrib]
  congr 1
  refine sum_congr rfl fun j _ => ?_
  by_cases h : σ j <;> simp [h, sgn]

omit [MeasurableSpace Z] hG in
/-- The sign average of the swapped `ψ` is at most `R̂_{S'} + R̂_S`. -/
theorem radAvg_ghost_le [Nonempty ι] {m : ℕ} (p : (Fin m → Z) × (Fin m → Z)) :
    radAvg (fun σ => ghostSup G (swapSign σ p)) ≤ empRad G p.2 + empRad G p.1 := by
  have hpt : ∀ σ : Fin m → Bool, ghostSup G (swapSign σ p)
      ≤ (⨆ i, (∑ j, sgn (σ j) * G i (p.2 j)) / m)
        + ⨆ i, (∑ j, sgn ((fun k => !σ k) j) * G i (p.1 j)) / m := by
    intro σ
    rw [ghostSup_swapSign]
    refine ciSup_le fun i => ?_
    have e : (∑ j, sgn (σ j) * (G i (p.2 j) - G i (p.1 j))) / m
        = (∑ j, sgn (σ j) * G i (p.2 j)) / m
          + (∑ j, sgn ((fun k => !σ k) j) * G i (p.1 j)) / m := by
      rw [← add_div]
      congr 1
      rw [← sum_add_distrib]
      refine sum_congr rfl fun j _ => ?_
      simp only [sgn_not]
      ring
    rw [e]
    exact add_le_add (le_ciSup (bddAbove_signed G hG01 p.2 σ) i)
      (le_ciSup (bddAbove_signed G hG01 p.1 (fun k => !σ k)) i)
  refine (radAvg_mono hpt).trans (le_of_eq ?_)
  rw [radAvg_add]
  congr 1
  exact radAvg_flip (fun σ => ⨆ i, (∑ j, sgn (σ j) * G i (p.1 j)) / m)

/-- **Symmetrization** (eqs. 3.8 to 3.13): `E[Φ] ≤ 2 R_m(G)`. -/
theorem integral_gapSup_le [Countable ι] [Nonempty ι] {m : ℕ} (hm : 0 < m) :
    ∫ S, gapSup μ G S ∂(sampleMeasure μ m) ≤ 2 * radComplexity μ G m := by
  set P := sampleMeasure μ m with hP
  have hψm := measurable_ghostSup G hG (m := m)
  have hψint : Integrable (fun p => ghostSup G p) (P.prod P) :=
    integrable_of_abs_le hψm 1 (abs_ghostSup_le G hG01)
  -- (3.8)-(3.9): `Φ(S) ≤ E_{S'} ψ(S, S')`.
  have hstep1 : ∀ S : Fin m → Z, gapSup μ G S ≤ ∫ S', ghostSup G (S, S') ∂P := by
    intro S
    refine ciSup_le fun i => ?_
    have hint1 : Integrable (fun S' : Fin m → Z => empRisk S' (G i)) P :=
      integrable_of_abs_le (measurable_empRisk (hG i)) 1
        fun S' => abs_le_one_of_mem_Icc (empRisk_mem_Icc (hG01 i) S')
    have hint2 : Integrable (fun S' => ghostSup G (S, S')) P :=
      integrable_of_abs_le (hψm.comp (measurable_const.prodMk measurable_id)) 1
        fun S' => abs_ghostSup_le G hG01 _
    calc popRisk μ (G i) - empRisk S (G i)
        = ∫ S', (empRisk S' (G i) - empRisk S (G i)) ∂P := by
          rw [integral_sub hint1 (integrable_const _), integral_const,
            integral_empRisk μ hm (hG i) (hG01 i)]
          simp
      _ ≤ ∫ S', ghostSup G (S, S') ∂P :=
          integral_mono (hint1.sub (integrable_const _)) hint2 fun S' =>
            le_ciSup (bddAbove_ghost G hG01 (S, S')) i
  have hinner : Integrable (fun S => ∫ S', ghostSup G (S, S') ∂P) P :=
    integrable_of_abs_le hψm.stronglyMeasurable.integral_prod_right'.measurable 1
      fun S => abs_integral_le_of_abs_le fun S' => abs_ghostSup_le G hG01 _
  have hΦint : Integrable (fun S => gapSup μ G S) P :=
    integrable_of_abs_le (measurable_gapSup μ G hG) 1 (abs_gapSup_le μ G hG hG01)
  have h1 : ∫ S, gapSup μ G S ∂P ≤ ∫ p, ghostSup G p ∂(P.prod P) := by
    rw [integral_prod _ hψint]
    exact integral_mono hΦint hinner hstep1
  -- (3.10)-(3.11): the swaps do not change the expectation.
  have hswap : ∀ σ : Fin m → Bool,
      ∫ p, ghostSup G (swapSign σ p) ∂(P.prod P) = ∫ p, ghostSup G p ∂(P.prod P) := by
    intro σ
    have hmp := measurePreserving_swapSign μ σ
    have := integral_map (μ := P.prod P) hmp.measurable.aemeasurable
      (f := fun p => ghostSup G p) (by rw [hmp.map_eq]; exact hψm.aestronglyMeasurable)
    rw [hmp.map_eq] at this
    exact this.symm
  have hσint : ∀ σ : Fin m → Bool,
      Integrable (fun p => ghostSup G (swapSign σ p)) (P.prod P) := by
    intro σ
    exact integrable_of_abs_le (hψm.comp (measurePreserving_swapSign μ σ).measurable) 1
      fun p => abs_ghostSup_le G hG01 _
  have h2 : ∫ p, ghostSup G p ∂(P.prod P)
      = ∫ p, radAvg (fun σ => ghostSup G (swapSign σ p)) ∂(P.prod P) := by
    unfold radAvg
    rw [integral_div, integral_finsetSum _ fun σ _ => hσint σ]
    simp only [hswap, sum_const, card_univ, Fintype.card_fun, Fintype.card_bool,
      Fintype.card_fin, nsmul_eq_mul]
    push_cast
    field_simp
  -- (3.12)-(3.13): subadditivity of the supremum and `σ ↦ −σ`.
  have hRm := measurable_empRad G hG (m := m)
  have hRint : Integrable (fun S => empRad G S) P :=
    integrable_of_abs_le hRm 1 (abs_empRad_le G hG01)
  have hRint2 : Integrable (fun p : (Fin m → Z) × (Fin m → Z) =>
      empRad G p.2 + empRad G p.1) (P.prod P) :=
    integrable_of_abs_le ((hRm.comp measurable_snd).add (hRm.comp measurable_fst)) 2
      fun p => (abs_add_le _ _).trans (by
        linarith [abs_empRad_le G hG01 p.2, abs_empRad_le G hG01 p.1])
  have hradm : Measurable fun p : (Fin m → Z) × (Fin m → Z) =>
      radAvg (fun σ => ghostSup G (swapSign σ p)) := by
    unfold radAvg
    exact Measurable.div_const (Finset.measurable_sum _ fun σ _ =>
      hψm.comp (measurePreserving_swapSign μ σ).measurable) _
  have hradint : Integrable (fun p => radAvg (fun σ => ghostSup G (swapSign σ p)))
      (P.prod P) := by
    refine integrable_of_abs_le hradm 1 fun p => ?_
    rw [abs_le]
    constructor
    · have := radAvg_mono (F := fun _ => (-1 : ℝ))
        (G := fun σ => ghostSup G (swapSign σ p))
        fun σ => (abs_le.1 (abs_ghostSup_le G hG01 _)).1
      rwa [radAvg_const] at this
    · have := radAvg_mono (G := fun _ => (1 : ℝ))
        (F := fun σ => ghostSup G (swapSign σ p))
        fun σ => (abs_le.1 (abs_ghostSup_le G hG01 _)).2
      rwa [radAvg_const] at this
  have h3 : ∫ p, radAvg (fun σ => ghostSup G (swapSign σ p)) ∂(P.prod P)
      ≤ ∫ p, (empRad G p.2 + empRad G p.1) ∂(P.prod P) :=
    integral_mono hradint hRint2 fun p => radAvg_ghost_le G hG01 p
  have h4 : ∫ p : (Fin m → Z) × (Fin m → Z), (empRad G p.2 + empRad G p.1) ∂(P.prod P)
      = 2 * radComplexity μ G m := by
    rw [integral_add (hRint.comp_snd P) (hRint.comp_fst P)]
    rw [integral_fun_snd (fun S => empRad G S), integral_fun_fst (fun S => empRad G S)]
    simp only [probReal_univ, one_smul, radComplexity]
    ring
  calc ∫ S, gapSup μ G S ∂P ≤ ∫ p, ghostSup G p ∂(P.prod P) := h1
    _ = _ := h2
    _ ≤ _ := h3
    _ = 2 * radComplexity μ G m := h4

/-- **Theorem 3.3 of Mohri et al., eq. (3.3).** For a countable nonempty
family of measurable functions `Z → [0, 1]`, `m ≥ 1` and `0 < ν ≤ 1`: with
probability at least `1 − ν` over an i.i.d. sample of size `m`, every member
satisfies `E[g] ≤ Ê_S[g] + 2 R_m(G) + √(log(1/ν) / (2m))`. -/
theorem rademacher_bound [Countable ι] [Nonempty ι] {m : ℕ} (hm : 0 < m) {ν : ℝ}
    (hν : 0 < ν) (hν1 : ν ≤ 1) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ i, popRisk μ (G i)
        ≤ empRisk S (G i) + 2 * radComplexity μ G m + sqrt (log (1 / ν) / (2 * m))} := by
  set P := sampleMeasure μ m with hP
  set ε := sqrt (log (1 / ν) / (2 * m)) with hε
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hlog : 0 ≤ log (1 / ν) := log_nonneg (by rw [le_div_iff₀ hν]; linarith)
  have hε0 : 0 ≤ ε := sqrt_nonneg _
  have hΦm := measurable_gapSup μ G hG (m := m)
  -- McDiarmid with `c_j = 1/m`.
  have hmc := McDiarmid.measure_ge_le μ (fun S : Fin m → Z => gapSup μ G S)
    (fun _ => 1 / m) hΦm ⟨1, abs_gapSup_le μ G hG hG01⟩ (boundedDiff_gapSup μ G hG hG01) hε0
  have hexp : exp (-2 * ε ^ 2 / ∑ _j : Fin m, (1 / (m : ℝ)) ^ 2) = ν := by
    have hsum : ∑ _j : Fin m, (1 / (m : ℝ)) ^ 2 = (1 : ℝ) / m := by
      simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
      field_simp
    rw [hsum, hε, sq_sqrt (div_nonneg hlog (by positivity))]
    have : -2 * (log (1 / ν) / (2 * m)) / (1 / m) = -log (1 / ν) := by
      field_simp
    rw [this, exp_neg, exp_log (by positivity), one_div, inv_inv]
  rw [hexp] at hmc
  -- The bad event and its complement.
  set A := {S : Fin m → Z | ε ≤ gapSup μ G S - ∫ S', gapSup μ G S' ∂P} with hA
  have hAmeas : MeasurableSet A := measurableSet_le measurable_const (hΦm.sub_const _)
  have hPA : P A ≤ ENNReal.ofReal ν := by
    rw [← ENNReal.ofReal_toReal (measure_ne_top P A)]
    exact ENNReal.ofReal_le_ofReal hmc
  have hcompl : 1 - ENNReal.ofReal ν ≤ P Aᶜ := by
    rw [prob_compl_eq_one_sub hAmeas]
    exact tsub_le_tsub_left hPA 1
  refine hcompl.trans (measure_mono fun S hS i => ?_)
  simp only [A, Set.mem_compl_iff, Set.mem_ofPred_eq, not_le] at hS
  have h1 := gap_le_gapSup μ G hG hG01 S i
  have h2 := integral_gapSup_le μ G hG hG01 hm
  linarith

end Symm

end ThresholdOpt.Symmetrization
