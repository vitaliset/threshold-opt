import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Tactic
import ThresholdOpt.Basic

/-!
# Capacity measures (Chapters 1, 2, 6 and Appendix C)

VC dimension (`def:vc_dimension`), pseudo-dimension (`def:pseudo_dimension`),
and Natarajan dimension (`def:natarajan_dimension`), together with the two
facts that follow from the definitions alone: subclass monotonicity
(`lem:nat_monotone`) and the unit VC dimension of one-dimensional thresholds
(`lem:vc_thresh`).

Mathlib's `Mathlib.Combinatorics.Shattering` is finite-set combinatorics
(`Finset.Shatters`, Sauer-Shelah) over a `Fintype`; it does not give the
learning-theory dimension of a class of functions on an infinite domain such as
`[0,1]` or `Δ^{K-1}`, so the three dimensions are defined here. They are
`ℕ∞`-valued because the classes are not assumed to have finite capacity.
-/

namespace ThresholdOpt.Capacity

open ThresholdOpt

variable {α : Type*}

/-! ### VC dimension -/

/-- A finite set is shattered by a class of binary functions when every
labeling of it is realized by some member of the class. -/
def Shatters (H : Set (α → Bool)) (s : Finset α) : Prop :=
  ∀ b : α → Bool, ∃ h ∈ H, ∀ x ∈ s, h x = b x

/-- **VC dimension** (`def:vc_dimension`): the largest cardinality of a
shattered set. -/
@[thesis "def:vc_dimension"]
noncomputable def vcDim (H : Set (α → Bool)) : ℕ∞ :=
  sSup {n | ∃ s : Finset α, Shatters H s ∧ (s.card : ℕ∞) = n}

/-- A subset of a shattered set is shattered. -/
theorem Shatters.mono {H : Set (α → Bool)} {s t : Finset α} (hts : t ⊆ s)
    (h : Shatters H s) : Shatters H t := by
  intro b
  obtain ⟨g, hg, hval⟩ := h b
  exact ⟨g, hg, fun x hx => hval x (hts hx)⟩

theorem le_vcDim {H : Set (α → Bool)} {s : Finset α} (h : Shatters H s) :
    (s.card : ℕ∞) ≤ vcDim H :=
  le_sSup ⟨s, h, rfl⟩

theorem vcDim_le {H : Set (α → Bool)} {n : ℕ∞}
    (h : ∀ s : Finset α, Shatters H s → (s.card : ℕ∞) ≤ n) : vcDim H ≤ n := by
  refine sSup_le ?_
  rintro m ⟨s, hs, rfl⟩
  exact h s hs

/-- A set shattered by a subclass is shattered by the class, so the VC dimension
is monotone in the class. -/
theorem vcDim_mono {H' H : Set (α → Bool)} (hsub : H' ⊆ H) : vcDim H' ≤ vcDim H :=
  vcDim_le fun _ hs => le_vcDim fun b => by
    obtain ⟨h, hh, hval⟩ := hs b
    exact ⟨h, hsub hh, hval⟩

/-- The empty set is shattered by any nonempty class, so the dimension is a
supremum over a nonempty set. -/
theorem shatters_empty {H : Set (α → Bool)} (hH : H.Nonempty) :
    Shatters H ∅ := by
  intro _
  obtain ⟨h, hh⟩ := hH
  exact ⟨h, hh, by simp⟩

/-! ### Natarajan dimension -/

/-- Natarajan shattering: two everywhere-distinct target functions `f` and `g`
such that every `f`/`g` pattern on `s` is realized. -/
def NShatters {K : ℕ} (H : Set (α → Fin K)) (s : Finset α) : Prop :=
  ∃ f g : α → Fin K, (∀ x ∈ s, f x ≠ g x) ∧
    ∀ b : α → Bool, ∃ h ∈ H, ∀ x ∈ s, h x = if b x then f x else g x

/-- **Natarajan dimension** (`def:natarajan_dimension`). -/
@[thesis "def:natarajan_dimension"]
noncomputable def natDim {K : ℕ} (H : Set (α → Fin K)) : ℕ∞ :=
  sSup {n | ∃ s : Finset α, NShatters H s ∧ (s.card : ℕ∞) = n}

theorem le_natDim {K : ℕ} {H : Set (α → Fin K)} {s : Finset α}
    (h : NShatters H s) : (s.card : ℕ∞) ≤ natDim H :=
  le_sSup ⟨s, h, rfl⟩

theorem natDim_le {K : ℕ} {H : Set (α → Fin K)} {n : ℕ∞}
    (h : ∀ s : Finset α, NShatters H s → (s.card : ℕ∞) ≤ n) : natDim H ≤ n := by
  refine sSup_le ?_
  rintro m ⟨s, hs, rfl⟩
  exact h s hs

/-- **Subclass monotonicity** (`lem:nat_monotone`).

A set shattered by a subclass is shattered by the whole class, so the
supremum can only grow. -/
@[thesis "lem:nat_monotone"]
theorem natDim_mono {K : ℕ} {H' H : Set (α → Fin K)} (hsub : H' ⊆ H) :
    natDim H' ≤ natDim H := by
  refine natDim_le fun s hs => le_natDim ?_
  obtain ⟨f, g, hfg, hreal⟩ := hs
  refine ⟨f, g, hfg, fun b => ?_⟩
  obtain ⟨h, hmem, hval⟩ := hreal b
  exact ⟨h, hsub hmem, hval⟩

/-! ### Pseudo-dimension -/

/-- Pseudo-shattering: a witness vector `r` such that every labeling `b` of
`s` is realized as `g(x) ≥ r(x) ⟺ b(x) = 1` by some `g` of the class
(`def:pseudo_dimension`, with the non-strict inequality on the positive side
as in the text and in Anthony and Bartlett, Definition 11.2). -/
def PShatters (G : Set (α → ℝ)) (s : Finset α) : Prop :=
  ∃ r : α → ℝ, ∀ b : α → Bool, ∃ g ∈ G,
    ∀ x ∈ s, (b x = true → r x ≤ g x) ∧ (b x = false → g x < r x)

/-- **Pseudo-dimension** (`def:pseudo_dimension`). -/
@[thesis "def:pseudo_dimension"]
noncomputable def pseudoDim (G : Set (α → ℝ)) : ℕ∞ :=
  sSup {n | ∃ s : Finset α, PShatters G s ∧ (s.card : ℕ∞) = n}

theorem pseudoDim_le {G : Set (α → ℝ)} {n : ℕ∞}
    (h : ∀ s : Finset α, PShatters G s → (s.card : ℕ∞) ≤ n) :
    pseudoDim G ≤ n := by
  refine sSup_le ?_
  rintro m ⟨s, hs, rfl⟩
  exact h s hs

theorem pseudoDim_mono {G' G : Set (α → ℝ)} (hsub : G' ⊆ G) :
    pseudoDim G' ≤ pseudoDim G := by
  refine pseudoDim_le fun s hs => le_sSup ⟨s, ?_, rfl⟩
  obtain ⟨r, hreal⟩ := hs
  refine ⟨r, fun b => ?_⟩
  obtain ⟨g, hmem, hval⟩ := hreal b
  exact ⟨g, hsub hmem, hval⟩

/-! ### One-dimensional thresholds -/

/-- The class of one-dimensional threshold rules `δ_τ(z) = 1(z ≥ τ)`
(`lem:vc_thresh`). -/
def thresholdClass : Set (ℝ → Bool) :=
  {h | ∃ τ : ℝ, h = fun z => decide (τ ≤ z)}

/-- A single point is shattered: thresholds on either side of it realize both
labels. -/
theorem thresholdClass_shatters_singleton (z : ℝ) :
    Shatters thresholdClass {z} := by
  intro b
  by_cases hb : b z = true
  · refine ⟨fun y => decide (z - 1 ≤ y), ⟨z - 1, rfl⟩, ?_⟩
    intro x hx
    rw [Finset.mem_singleton] at hx
    subst hx
    rw [hb]
    exact decide_eq_true (by linarith)
  · simp only [Bool.not_eq_true] at hb
    refine ⟨fun y => decide (z + 1 ≤ y), ⟨z + 1, rfl⟩, ?_⟩
    intro x hx
    rw [Finset.mem_singleton] at hx
    subst hx
    rw [hb]
    exact decide_eq_false (by linarith)

/-- No two-point set is shattered: a threshold rule is monotone, so it cannot
label the smaller point positive and the larger one negative. -/
theorem thresholdClass_not_shatters_pair {z₁ z₂ : ℝ} (hlt : z₁ < z₂) :
    ¬ Shatters thresholdClass {z₁, z₂} := by
  intro hsh
  obtain ⟨h, ⟨τ, rfl⟩, hval⟩ := hsh (fun z => decide (z = z₁))
  have h1 : decide (τ ≤ z₁) = decide (z₁ = z₁) := hval z₁ (by simp)
  have h2 : decide (τ ≤ z₂) = decide (z₂ = z₁) := hval z₂ (by simp)
  rw [decide_eq_true rfl] at h1
  rw [decide_eq_false (ne_of_gt hlt)] at h2
  exact (of_decide_eq_false h2) (le_trans (of_decide_eq_true h1) hlt.le)

/-- **Unit VC dimension of one-dimensional thresholds** (`lem:vc_thresh`).

`d_VC(H_thresh) = 1`. The lower bound is a shattered singleton; the upper bound
is that a shattered set of two or more points would contain a shattered pair,
which the monotonicity of threshold rules forbids. -/
@[thesis "lem:vc_thresh"]
theorem vcDim_thresholdClass : vcDim thresholdClass = 1 := by
  refine le_antisymm (vcDim_le fun s hs => ?_) ?_
  · -- If two distinct points were shattered, the pair would be too.
    by_contra hcon
    have hcard : 1 < s.card := by
      by_contra hle
      exact hcon (by exact_mod_cast not_lt.1 hle)
    obtain ⟨a, ha, b, hb, hab⟩ := Finset.one_lt_card.1 hcard
    have hsub : ({a, b} : Finset ℝ) ⊆ s :=
      Finset.insert_subset ha (Finset.singleton_subset_iff.2 hb)
    have hpair : Shatters thresholdClass {a, b} := hs.mono hsub
    rcases lt_or_gt_of_ne hab with h | h
    · exact thresholdClass_not_shatters_pair h hpair
    · exact thresholdClass_not_shatters_pair h
        (by rwa [Finset.pair_comm] at hpair)
  · simpa using le_vcDim (thresholdClass_shatters_singleton 0)

/-! ### Linear dependence among more vectors than the dimension

Every capacity upper bound in this development reduces to one fact: a family of
more vectors than the dimension of the space containing them satisfies a
nontrivial linear relation. The pseudo-dimension bound of Anthony and Bartlett
and the Natarajan bound of Daniely and Shalev-Shwartz are both sign arguments
built on such a relation, and both are proved here instead of imported. -/

/-- More vectors than the dimension satisfy a nontrivial linear relation. The
family is indexed by the points of a finite set and need not be injective. -/
theorem exists_dependence {V : Type*} [AddCommGroup V] [Module ℝ V]
    [FiniteDimensional ℝ V] {s : Finset α} (ρ : α → V)
    (h : Module.finrank ℝ V < s.card) :
    ∃ c : α → ℝ, (∃ x ∈ s, c x ≠ 0) ∧ ∑ x ∈ s, c x • ρ x = 0 := by
  classical
  have hnot : ¬ LinearIndependent ℝ (fun x : s => ρ x.1) := by
    intro hli
    have := hli.fintype_card_le_finrank
    simp only [Fintype.card_coe] at this
    omega
  obtain ⟨g, hg, i, hi⟩ := Fintype.not_linearIndependent_iff.1 hnot
  refine ⟨fun x => if hx : x ∈ s then g ⟨x, hx⟩ else 0, ⟨i.1, i.2, ?_⟩, ?_⟩
  · simpa using hi
  · rw [← Finset.sum_attach]
    rw [← hg, Finset.univ_eq_attach]
    refine Finset.sum_congr rfl fun x _ => ?_
    simp

/-! ### Pseudo-dimension of a finite-dimensional function space -/

/-- **Pseudo-dimension of a monotone transform of a finite-dimensional function
space.**

If every member of `G` is `x ↦ σ(∑ w_i basis_i(x))` for a fixed strictly
increasing `σ` and some `w ∈ ℝ^W`, then `Pdim(G) ≤ W`. This is the statement
`prop:calibrator_complexity` uses: Anthony and Bartlett, Theorem 11.4, composed
with the invariance of the pseudo-dimension under a strictly increasing
transform.

Proved directly, with no witness for the untransformed class. If `W + 1`
points were pseudo-shattered, their evaluation vectors in `ℝ^W` would satisfy a
relation `∑ c_x basis(x) = 0`, so `∑ c_x s(x) = 0` for every score `s` of the
space. Take the member `g₁` that is above the witness exactly where `c_x > 0`
and the member `g₂` that is above it exactly where `c_x < 0`. Where `c_x > 0`,
`g₂ < r ≤ g₁`, so the score of `g₁` is strictly larger; where `c_x < 0` it is
strictly smaller. Hence `∑ c_x (s₁(x) − s₂(x)) > 0`, against the relation. -/
theorem pseudoDim_le_dim_of_strictMono (G : Set (α → ℝ)) (W : ℕ)
    (basis : Fin W → (α → ℝ)) {σ : ℝ → ℝ} (hσ : StrictMono σ)
    (hspan : ∀ g ∈ G, ∃ w : Fin W → ℝ,
      g = fun x => σ (∑ i, w i * basis i x)) :
    pseudoDim G ≤ (W : ℕ∞) := by
  refine pseudoDim_le fun s hs => ?_
  by_contra hcon
  have hcard : W < s.card := by
    by_contra hle
    exact hcon (by exact_mod_cast not_lt.1 hle)
  obtain ⟨r, hreal⟩ := hs
  obtain ⟨c, ⟨x₀, hx₀, hc₀⟩, hdep⟩ :=
    exists_dependence (s := s) (fun x => (fun i => basis i x : Fin W → ℝ))
      (by simpa using hcard)
  -- Every score of the space is annihilated by the relation.
  have hann : ∀ w : Fin W → ℝ, ∑ x ∈ s, c x * ∑ i, w i * basis i x = 0 := by
    intro w
    have hcoord : ∀ i, ∑ x ∈ s, c x * basis i x = 0 := by
      intro i
      have := congrFun hdep i
      simpa [Finset.sum_apply] using this
    calc ∑ x ∈ s, c x * ∑ i, w i * basis i x
        = ∑ i, w i * ∑ x ∈ s, c x * basis i x := by
          simp only [Finset.mul_sum]
          rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun i _ =>
            Finset.sum_congr rfl fun x _ => by ring
      _ = 0 := by simp [hcoord]
  obtain ⟨g₁, hg₁, h₁⟩ := hreal fun x => decide (0 < c x)
  obtain ⟨g₂, hg₂, h₂⟩ := hreal fun x => decide (c x < 0)
  obtain ⟨w₁, rfl⟩ := hspan g₁ hg₁
  obtain ⟨w₂, rfl⟩ := hspan g₂ hg₂
  have hstrict : ∀ x ∈ s, c x ≠ 0 →
      c x * ∑ i, w₂ i * basis i x < c x * ∑ i, w₁ i * basis i x := by
    intro x hx hcx
    rcases lt_or_gt_of_ne hcx with hneg | hpos
    · have a := (h₁ x hx).2 (decide_eq_false (not_lt.2 hneg.le))
      have b := (h₂ x hx).1 (decide_eq_true hneg)
      have : ∑ i, w₁ i * basis i x < ∑ i, w₂ i * basis i x :=
        hσ.lt_iff_lt.1 (lt_of_lt_of_le a b)
      exact mul_lt_mul_of_neg_left this hneg
    · have a := (h₁ x hx).1 (decide_eq_true hpos)
      have b := (h₂ x hx).2 (decide_eq_false (not_lt.2 hpos.le))
      have : ∑ i, w₂ i * basis i x < ∑ i, w₁ i * basis i x :=
        hσ.lt_iff_lt.1 (lt_of_lt_of_le b a)
      exact mul_lt_mul_of_pos_left this hpos
  have hle : ∀ x ∈ s,
      c x * ∑ i, w₂ i * basis i x ≤ c x * ∑ i, w₁ i * basis i x := by
    intro x hx
    by_cases hcx : c x = 0
    · rw [hcx, zero_mul, zero_mul]
    · exact (hstrict x hx hcx).le
  have := Finset.sum_lt_sum hle ⟨x₀, hx₀, hstrict x₀ hx₀ hc₀⟩
  rw [hann w₁, hann w₂] at this
  exact lt_irrefl _ this

/-- **Pseudo-dimension of a finite-dimensional function space.**

Anthony and Bartlett, *Neural Network Learning: Theoretical Foundations*,
Theorem 11.4: if every member of `G` is a linear combination of `W` fixed
functions, then `Pdim(G) ≤ W`. Proved, not imported: it is the case `σ = id` of
`pseudoDim_le_dim_of_strictMono`. -/
theorem pseudoDim_le_dim (G : Set (α → ℝ)) (W : ℕ)
    (basis : Fin W → (α → ℝ))
    (hspan : ∀ g ∈ G, ∃ w : Fin W → ℝ, g = fun x => ∑ i, w i * basis i x) :
    pseudoDim G ≤ (W : ℕ∞) :=
  pseudoDim_le_dim_of_strictMono G W basis strictMono_id fun g hg => by
    obtain ⟨w, hw⟩ := hspan g hg
    exact ⟨w, hw⟩

/-! ### Calibrator families (`prop:calibrator_complexity`,
`rem:multiclass_calibration`) -/

/-- The logit `z(p) = log(p / (1 - p))` of `eq:temperature_scaling`. -/
noncomputable def logit (p : ℝ) : ℝ := Real.log (p / (1 - p))

/-- Temperature scaling (`eq:temperature_scaling`): `g_T(p) = σ(z(p) / T)`,
`T > 0`.

The families below are sets of functions on all of `ℝ`, and Lean's `Real.log`
and division are total: at `p = 0` and `p = 1` the logit is `0`, not `±∞`, so
`g_T` takes the value `1/2` there instead of the limits `0` and `1`. On
`0 < p < 1` the formulas are those of the text. The dimension bounds do not
depend on the convention: they hold for any fixed feature functions. -/
def HTemp : Set (ℝ → ℝ) :=
  {g | ∃ T : ℝ, 0 < T ∧ g = fun p => Real.sigmoid (logit p / T)}

/-- Platt scaling (`eq:platt_scaling`): `g_{a,b}(s) = σ(a s + b)`. -/
def HPlatt : Set (ℝ → ℝ) :=
  {g | ∃ a b : ℝ, g = fun s => Real.sigmoid (a * s + b)}

/-- The pre-softmax score of class `k` in binary Dirichlet calibration
(`eq:dirichlet_calibration` at `K = 2`), at the probability vector
`(1 - p, p)`. -/
noncomputable def dirScore (W : Matrix (Fin 2) (Fin 2) ℝ) (b : Fin 2 → ℝ)
    (k : Fin 2) (p : ℝ) : ℝ :=
  W k 0 * Real.log (1 - p) + W k 1 * Real.log p + b k

/-- Binary Dirichlet calibration (`eq:dirichlet_calibration` at `K = 2`): the
class-`1` coordinate of `softmax(W log p̂ + b)`, with all six parameters
free. -/
def HDir : Set (ℝ → ℝ) :=
  {g | ∃ (W : Matrix (Fin 2) (Fin 2) ℝ) (b : Fin 2 → ℝ), g = fun p =>
    Real.exp (dirScore W b 1 p)
      / (Real.exp (dirScore W b 0 p) + Real.exp (dirScore W b 1 p))}

/-- The binary softmax is the sigmoid of the score difference. This is "the
softmax in the binary reduction" of `prop:calibrator_complexity`, and what
reduces the six Dirichlet parameters to three effective ones. -/
theorem softmax_two (z₀ z₁ : ℝ) :
    Real.exp z₁ / (Real.exp z₀ + Real.exp z₁) = Real.sigmoid (z₁ - z₀) := by
  have h0 := Real.exp_pos z₀
  have h1 := Real.exp_pos z₁
  rw [Real.sigmoid_def, neg_sub, Real.exp_sub]
  field_simp
  ring

theorem pseudoDim_HTemp_le : pseudoDim HTemp ≤ 1 := by
  have := pseudoDim_le_dim_of_strictMono HTemp 1 (fun _ p => logit p)
    Real.sigmoid_strictMono (by
      rintro g ⟨T, -, rfl⟩
      refine ⟨fun _ => T⁻¹, ?_⟩
      funext p
      simp [div_eq_inv_mul])
  exact_mod_cast this

theorem pseudoDim_HPlatt_le : pseudoDim HPlatt ≤ 2 := by
  have := pseudoDim_le_dim_of_strictMono HPlatt 2 ![fun s => s, fun _ => 1]
    Real.sigmoid_strictMono (by
      rintro g ⟨a, b, rfl⟩
      refine ⟨![a, b], ?_⟩
      funext s
      simp [Fin.sum_univ_two])
  exact_mod_cast this

theorem pseudoDim_HDir_le : pseudoDim HDir ≤ 3 := by
  have := pseudoDim_le_dim_of_strictMono HDir 3
    ![fun p => Real.log (1 - p), fun p => Real.log p, fun _ => 1]
    Real.sigmoid_strictMono (by
      rintro g ⟨W, b, rfl⟩
      refine ⟨![W 1 0 - W 0 0, W 1 1 - W 0 1, b 1 - b 0], ?_⟩
      funext p
      rw [softmax_two]
      congr 1
      simp [Fin.sum_univ_three, dirScore]
      ring)
  exact_mod_cast this

/-- **Pseudo-dimension of binary calibrators** (`prop:calibrator_complexity`):
`d_P(H_temp) ≤ 1`, `d_P(H_platt) ≤ 2`, `d_P(H_dir) ≤ 3`.

Each family is the sigmoid of a score in a space of dimension `1`, `2`, `3`.
For the Dirichlet family the score is the difference of the two pre-softmax
scores, which lies in the span of `log(1 - p)`, `log p` and `1`: this is the
reduction from six parameters to three. -/
@[thesis "prop:calibrator_complexity"]
theorem calibrator_complexity :
    pseudoDim HTemp ≤ 1 ∧ pseudoDim HPlatt ≤ 2 ∧ pseudoDim HDir ≤ 3 :=
  ⟨pseudoDim_HTemp_le, pseudoDim_HPlatt_le, pseudoDim_HDir_le⟩

/-- The pre-softmax score class of one class under multiclass temperature
scaling: `β z(x)` with `β = 1 / T > 0`, for a fixed logit `z` of that class. -/
def tempScoreClass (z : α → ℝ) : Set (α → ℝ) :=
  {g | ∃ β : ℝ, 0 < β ∧ g = fun x => β * z x}

/-- The pre-softmax score class of one class under `K`-class Dirichlet
calibration: `w · log p + b`, with `w ∈ ℝ^K` and `b ∈ ℝ`. -/
def dirScoreClass (K : ℕ) : Set ((Fin K → ℝ) → ℝ) :=
  {g | ∃ (w : Fin K → ℝ) (b : ℝ),
    g = fun p => (∑ j, w j * Real.log (p j)) + b}

theorem pseudoDim_tempScoreClass_le (z : α → ℝ) :
    pseudoDim (tempScoreClass z) ≤ 1 := by
  have := pseudoDim_le_dim (tempScoreClass z) 1 (fun _ => z) (by
    rintro g ⟨β, -, rfl⟩
    exact ⟨fun _ => β, by simp⟩)
  exact_mod_cast this

theorem pseudoDim_dirScoreClass_le (K : ℕ) :
    pseudoDim (dirScoreClass K) ≤ ((K + 1 : ℕ) : ℕ∞) := by
  refine pseudoDim_le_dim (dirScoreClass K) (K + 1)
    (fun i p => if h : (i : ℕ) < K then Real.log (p ⟨i, h⟩) else 1) ?_
  rintro g ⟨w, b, rfl⟩
  refine ⟨fun i => if h : (i : ℕ) < K then w ⟨i, h⟩ else b, ?_⟩
  funext p
  rw [Fin.sum_univ_castSucc]
  simp

/-- **Multiclass calibration complexity** (`rem:multiclass_calibration`).
The three per-class bounds are `1` for temperature pre-softmax scores, `2` for
independent Platt sigmoid outputs before renormalization, and `K + 1` for
Dirichlet pre-softmax scores. These are scalar pseudo-dimension bounds, not
a joint bound for the normalized probability vector. -/
@[thesis "rem:multiclass_calibration"]
theorem multiclass_calibration :
    (∀ (β : Type) (z : β → ℝ), pseudoDim (tempScoreClass z) ≤ 1) ∧
    pseudoDim HPlatt ≤ 2 ∧
    (∀ K : ℕ, pseudoDim (dirScoreClass K) ≤ ((K + 1 : ℕ) : ℕ∞)) :=
  ⟨fun _ z => pseudoDim_tempScoreClass_le z, pseudoDim_HPlatt_le,
    pseudoDim_dirScoreClass_le⟩

/-! ### Natarajan dimension of `argmin` classes

The same sign argument, run on score gaps. A hypothesis that selects label `a`
over label `b` at `x` has `score a ≤ score b`, with equality only if `a < b`
(the smallest-index tie-break). The tie-break is what makes the argument close
without strict inequalities: two opposite labelings cannot both be tied at the
same point. -/

/-- If two indices tie at the minimum, `lexArgmin` is the smaller. -/
theorem lexArgmin_le_of_tie {K : ℕ} [NeZero K] {f : Fin K → ℝ} {j : Fin K}
    (h : f (lexArgmin f) = f j) : lexArgmin f ≤ j :=
  lexArgmin_le_index fun l => h ▸ lexArgmin_le f l

/-- **No weighting of a shattered set has a hypothesis-independent gap sum.**

Let `s` be Natarajan-shattered by `H` with targets `f` and `g`, and let every
`h ∈ H` be the `lexArgmin` of scores `S h x` on `s`. Then there is no weighting
`c`, not identically zero on `s`, for which

`∑_{x ∈ s} c x · (S h x (f x) − S h x (g x))`

takes the same value for every `h ∈ H`. Every capacity upper bound below is an
instance: the weighting is a linear relation among the parameter-dependent
parts of the gaps. -/
theorem NShatters.false_of_invariant_gap {K : ℕ} [NeZero K]
    {H : Set (α → Fin K)} {s : Finset α} {f g : α → Fin K}
    (hfg : ∀ x ∈ s, f x ≠ g x)
    (hreal : ∀ b : α → Bool, ∃ h ∈ H, ∀ x ∈ s, h x = if b x then f x else g x)
    (S : (α → Fin K) → α → Fin K → ℝ)
    (hS : ∀ h ∈ H, ∀ x ∈ s, h x = lexArgmin (S h x))
    (c : α → ℝ) (hc : ∃ x ∈ s, c x ≠ 0)
    (hinv : ∀ h ∈ H, ∀ h' ∈ H,
      ∑ x ∈ s, c x * (S h x (f x) - S h x (g x))
        = ∑ x ∈ s, c x * (S h' x (f x) - S h' x (g x))) : False := by
  obtain ⟨h₁, hh₁, hv₁⟩ := hreal fun x => decide (0 < c x)
  obtain ⟨h₂, hh₂, hv₂⟩ := hreal fun x => !decide (0 < c x)
  -- Under `h₁` every weighted gap is nonpositive, under `h₂` nonnegative.
  have hle₁ : ∀ x ∈ s, c x * (S h₁ x (f x) - S h₁ x (g x)) ≤ 0 := by
    intro x hx
    have hsel := hv₁ x hx
    rcases lt_or_ge 0 (c x) with hpos | hneg
    · rw [decide_eq_true hpos, if_pos rfl] at hsel
      have := lexArgmin_le (S h₁ x) (g x)
      rw [← hS h₁ hh₁ x hx, hsel] at this
      exact mul_nonpos_of_nonneg_of_nonpos hpos.le (by linarith)
    · rw [decide_eq_false (not_lt.2 hneg)] at hsel
      simp only [Bool.false_eq_true, if_false] at hsel
      have := lexArgmin_le (S h₁ x) (f x)
      rw [← hS h₁ hh₁ x hx, hsel] at this
      exact mul_nonpos_of_nonpos_of_nonneg hneg (by linarith)
  have hge₂ : ∀ x ∈ s, 0 ≤ c x * (S h₂ x (f x) - S h₂ x (g x)) := by
    intro x hx
    have hsel := hv₂ x hx
    rcases lt_or_ge 0 (c x) with hpos | hneg
    · rw [decide_eq_true hpos] at hsel
      simp only [Bool.not_true, Bool.false_eq_true, if_false] at hsel
      have := lexArgmin_le (S h₂ x) (f x)
      rw [← hS h₂ hh₂ x hx, hsel] at this
      exact mul_nonneg hpos.le (by linarith)
    · rw [decide_eq_false (not_lt.2 hneg)] at hsel
      simp only [Bool.not_false, if_true] at hsel
      have := lexArgmin_le (S h₂ x) (g x)
      rw [← hS h₂ hh₂ x hx, hsel] at this
      exact mul_nonneg_of_nonpos_of_nonpos hneg (by linarith)
  have heq := hinv h₁ hh₁ h₂ hh₂
  have hzero₁ : ∑ x ∈ s, c x * (S h₁ x (f x) - S h₁ x (g x)) = 0 :=
    le_antisymm (Finset.sum_nonpos hle₁) (heq ▸ Finset.sum_nonneg hge₂)
  have hzero₂ : ∑ x ∈ s, c x * (S h₂ x (f x) - S h₂ x (g x)) = 0 :=
    heq ▸ hzero₁
  obtain ⟨x, hx, hcx⟩ := hc
  -- Both hypotheses are tied at `x`, and they select opposite labels there.
  have ht₁ : S h₁ x (f x) = S h₁ x (g x) := by
    have := (Finset.sum_eq_zero_iff_of_nonpos hle₁).1 hzero₁ x hx
    rcases mul_eq_zero.1 this with h | h
    · exact absurd h hcx
    · linarith
  have ht₂ : S h₂ x (f x) = S h₂ x (g x) := by
    have := (Finset.sum_eq_zero_iff_of_nonneg hge₂).1 hzero₂ x hx
    rcases mul_eq_zero.1 this with h | h
    · exact absurd h hcx
    · linarith
  have hsel₁ := hv₁ x hx
  have hsel₂ := hv₂ x hx
  rcases lt_or_ge 0 (c x) with hpos | hneg
  · rw [decide_eq_true hpos, if_pos rfl] at hsel₁
    rw [decide_eq_true hpos] at hsel₂
    simp only [Bool.not_true, Bool.false_eq_true, if_false] at hsel₂
    have a₁ : f x ≤ g x := by
      have := lexArgmin_le_of_tie (f := S h₁ x) (j := g x)
        (by rw [← hS h₁ hh₁ x hx, hsel₁]; exact ht₁)
      rwa [← hS h₁ hh₁ x hx, hsel₁] at this
    have a₂ : g x ≤ f x := by
      have := lexArgmin_le_of_tie (f := S h₂ x) (j := f x)
        (by rw [← hS h₂ hh₂ x hx, hsel₂]; exact ht₂.symm)
      rwa [← hS h₂ hh₂ x hx, hsel₂] at this
    exact hfg x hx (le_antisymm a₁ a₂)
  · rw [decide_eq_false (not_lt.2 hneg)] at hsel₁ hsel₂
    simp only [Bool.false_eq_true, if_false] at hsel₁
    simp only [Bool.not_false, if_true] at hsel₂
    have a₁ : g x ≤ f x := by
      have := lexArgmin_le_of_tie (f := S h₁ x) (j := f x)
        (by rw [← hS h₁ hh₁ x hx, hsel₁]; exact ht₁.symm)
      rwa [← hS h₁ hh₁ x hx, hsel₁] at this
    have a₂ : f x ≤ g x := by
      have := lexArgmin_le_of_tie (f := S h₂ x) (j := g x)
        (by rw [← hS h₂ hh₂ x hx, hsel₂]; exact ht₂)
      rwa [← hS h₂ hh₂ x hx, hsel₂] at this
    exact hfg x hx (le_antisymm a₂ a₁)

/-- **Natarajan dimension of a linear-score class, with the gap subspace.**

Let every `h ∈ H` be `x ↦ argmin_k (w · Φ_k(x) + g_k(x))` for some `w ∈ ℝ^W`,
and let `V` be a subspace containing every feature difference
`Φ_k(x) − Φ_{k'}(x)`. Then `d_Nat(H) ≤ dim V`.

With `V = ℝ^W` this is the bound of `lem:linear_embedding`. A proper `V`
records a gauge freedom: directions of `w` that move every score of a point by
the same amount do not count. -/
theorem natDim_le_finrank_of_linear_scores {K W : ℕ} [NeZero K]
    (H : Set (α → Fin K))
    (Φ : Fin K → α → (Fin W → ℝ)) (g : Fin K → α → ℝ)
    (V : Submodule ℝ (Fin W → ℝ))
    (hV : ∀ k k' x, Φ k x - Φ k' x ∈ V)
    (hform : ∀ h ∈ H, ∃ w : Fin W → ℝ, ∀ x,
      h x = lexArgmin (fun k => (∑ i, w i * Φ k x i) + g k x)) :
    natDim H ≤ (Module.finrank ℝ V : ℕ∞) := by
  classical
  refine natDim_le fun s hs => ?_
  by_contra hcon
  have hcard : Module.finrank ℝ V < s.card := by
    by_contra hle
    exact hcon (by exact_mod_cast not_lt.1 hle)
  obtain ⟨f, g', hfg, hreal⟩ := hs
  obtain ⟨c, hc, hdep⟩ :=
    exists_dependence (s := s)
      (fun x => (⟨Φ (f x) x - Φ (g' x) x, hV _ _ x⟩ : V)) hcard
  -- The relation, read coordinatewise in `ℝ^W`.
  have hcoord : ∀ i, ∑ x ∈ s, c x * (Φ (f x) x i - Φ (g' x) x i) = 0 := by
    intro i
    have := congrArg (fun v : V => (v : Fin W → ℝ) i) hdep
    simpa [Finset.sum_apply] using this
  -- A parameter for each hypothesis.
  let w : (α → Fin K) → Fin W → ℝ := fun h =>
    if hh : h ∈ H then Classical.choose (hform h hh) else 0
  have hw : ∀ h ∈ H, ∀ x,
      h x = lexArgmin (fun k => (∑ i, w h i * Φ k x i) + g k x) := by
    intro h hh
    simp only [w, dif_pos hh]
    exact Classical.choose_spec (hform h hh)
  refine NShatters.false_of_invariant_gap hfg hreal
    (fun h x k => (∑ i, w h i * Φ k x i) + g k x)
    (fun h hh x _ => hw h hh x) c hc ?_
  -- The gap sum does not depend on the parameter.
  have key : ∀ h : α → Fin K,
      ∑ x ∈ s, c x * (((∑ i, w h i * Φ (f x) x i) + g (f x) x)
          - ((∑ i, w h i * Φ (g' x) x i) + g (g' x) x))
        = ∑ x ∈ s, c x * (g (f x) x - g (g' x) x) := by
    intro h
    have hsplit : ∀ x,
        c x * (((∑ i, w h i * Φ (f x) x i) + g (f x) x)
          - ((∑ i, w h i * Φ (g' x) x i) + g (g' x) x))
        = (∑ i, w h i * (c x * (Φ (f x) x i - Φ (g' x) x i)))
          + c x * (g (f x) x - g (g' x) x) := by
      intro x
      have : (∑ i, w h i * (c x * (Φ (f x) x i - Φ (g' x) x i)))
          = c x * ((∑ i, w h i * Φ (f x) x i) - ∑ i, w h i * Φ (g' x) x i) := by
        rw [← Finset.sum_sub_distrib, Finset.mul_sum]
        exact Finset.sum_congr rfl fun i _ => by ring
      rw [this]; ring
    rw [Finset.sum_congr rfl fun x _ => hsplit x, Finset.sum_add_distrib,
      Finset.sum_comm]
    have : ∀ i, ∑ x ∈ s, w h i * (c x * (Φ (f x) x i - Φ (g' x) x i)) = 0 := by
      intro i
      rw [← Finset.mul_sum, hcoord i, mul_zero]
    simp [this]
  intro h _ h' _
  rw [key h, key h']

/-- **Natarajan dimension of a linear-score class.**

A class of the form `x ↦ argmin_k (w · Φ_k(x) + g_k(x))`, `w ∈ ℝ^W`, has
Natarajan dimension at most `W`. With `g = 0` this is Daniely and
Shalev-Shwartz, *Optimal learners for multiclass problems*, COLT 2014,
Theorem 7; with the parameter-free offsets `g_k` it is the first half of
`lem:linear_embedding`.

Proved, not imported, and for the smallest-index tie-break the development
uses: the source does not fix a tie-break, and the dissertation's proof of the
lemma reads the sign of the score gap off the selected label, which is only
justified once ties are resolved.

The hypothesis asks only that every member of `H` have the linear-score form
for *some* parameter, not that every parameter occur. This is what covers the
second half of the lemma, the coupled case: a class whose structural
parameters map by a fixed rule into the lifted parameter `w` is such an `H`. -/
@[thesis "lem:linear_embedding"]
theorem natDim_le_of_linear_scores {K W : ℕ} [NeZero K] (H : Set (α → Fin K))
    (Φ : Fin K → α → (Fin W → ℝ)) (g : Fin K → α → ℝ)
    (hform : ∀ h ∈ H, ∃ w : Fin W → ℝ, ∀ x,
      h x = lexArgmin (fun k => (∑ i, w i * Φ k x i) + g k x)) :
    natDim H ≤ (W : ℕ∞) := by
  have := natDim_le_finrank_of_linear_scores H Φ g ⊤ (fun _ _ _ => trivial) hform
  simpa using this

end ThresholdOpt.Capacity
