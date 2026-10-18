import ThresholdOpt.Complexity

/-!
# `H_Vor^Δ` is a proper subclass of `H_Full`

`Complexity.HVor_eq_HFull` shows that on the simplex the Voronoi rules with
unconstrained prototypes are exactly the full-matrix rules. This file answers
the companion question for `H_Vor^Δ` (`Complexity.HVorSimplex`), the class the
optimizer of Chapter 6 searches, whose prototypes are probability vectors:
the inclusion `H_Vor^Δ ⊆ H_Full` is proper.

* For `K ≥ 2` there is a trivial reason (`HVorSimplex_ssubset_HFull_of_two_le`):
  every rule of `H_Vor^Δ` selects class `0` at the prototype `μ_0`, a point of
  the simplex, while the constant rule `1` is a full-matrix rule.
* For `K ≥ 3` the reason is geometric (`HVorSimplex_ssubset_HFull`). The band
  rule (`bandRule`) cuts the simplex into `K` slabs along `p_0`, every class
  with nonempty interior (`bandRule_locally_const`), the middle slab of class
  `1` wider than `1/2`. No Voronoi rule with prototypes in the simplex agrees
  with it even off the slab boundaries (`not_agree_bandRule`), so the
  conclusion depends neither on tie-breaking nor on the boundary hyperplanes.

The argument. For prototypes `μ`, the score difference
`F(p) = d²(p, μ_j) − d²(p, μ_k)` is linear on the simplex. If the rule is `j`
just below the hyperplane `p_0 = a` and `k` just above it, `F` vanishes at the
`K − 1` vertices `a e_0 + (1 − a) e_i` of the face `p_0 = a`, hence
`(1 − a) F(p) = F_0 · (p_0 − a)` on the simplex (`score_diff_eq_of_boundary`,
which holds for every matrix). For Voronoi scores `F(μ_j) = −F(μ_k)`: the
bisector passes through the midpoint of the two prototypes. With
`x_l = (μ_l)_0` this gives `x_0 + x_1 = 2a` and `x_1 + x_2 = 2b` for the two
boundaries `a = 1/10` and `b = 7/10` of the middle slab, and prototypes in the
simplex have `0 ≤ x_l ≤ 1`, so `2b − 1 ≤ x_1 ≤ 2a`, which is false.

At `K = 2` the two classes differ only by rules in which a class has empty
interior (not formalized).

The last section shows that `H_PS ⊄ H_Vor^Δ` in general
(`HPS_zeroOne_not_subset_HVorSimplex`): under the `0/1` loss with `K = 3`, a
parallel-shift rule that no Voronoi rule with prototypes in the simplex
reproduces, so the containment `H_PS ⊆ H_Vor` of `prop:ps_in_vor` does not
survive the restriction of the prototypes to the simplex.
-/

namespace ThresholdOpt.VoronoiSimplex

open ThresholdOpt ThresholdOpt.Voronoi ThresholdOpt.Complexity

variable {K : ℕ} [NeZero K]

/-! ### Facts about the smallest-index `argmin` and the simplex -/

/-- The selected index strictly beats every earlier index. -/
theorem lt_of_lexArgmin_eq {f : Fin K → ℝ} {k j : Fin K} (hk : lexArgmin f = k)
    (hj : j < k) : f k < f j := by
  by_contra h
  push Not at h
  have hmin : ∀ l, f j ≤ f l := fun l => h.trans (by rw [← hk]; exact lexArgmin_le f l)
  have := lexArgmin_le_index hmin
  rw [hk] at this
  exact absurd hj (not_lt.2 this)

/-- If index `0` attains the minimum, it is selected. -/
theorem lexArgmin_eq_zero_of_le {f : Fin K → ℝ} (h : ∀ j, f 0 ≤ f j) :
    lexArgmin f = 0 :=
  le_antisymm (lexArgmin_le_index h) (Fin.zero_le _)

omit [NeZero K] in
theorem le_one_of_mem_stdSimplex {p : Fin K → ℝ} (hp : p ∈ stdSimplex ℝ (Fin K))
    (i : Fin K) : p i ≤ 1 := by
  have := Finset.single_le_sum (fun j _ => hp.1 j) (Finset.mem_univ i)
  rw [hp.2] at this
  exact this

omit [NeZero K] in
theorem dsq_self (p : Fin K → ℝ) : dsq p p = 0 := by
  simp [dsq]

omit [NeZero K] in
theorem dsq_comm (p q : Fin K → ℝ) : dsq p q = dsq q p := by
  simp only [dsq]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- `H_Vor^Δ ⊆ H_Full`, from `H_Vor = H_Full`. -/
theorem HVorSimplex_subset_HFull : HVorSimplex (K := K) ⊆ HFull := by
  rw [← HVor_eq_HFull]
  exact HVorSimplex_subset_HVor

/-! ### The trivial reason, `K ≥ 2` -/

/-- A Voronoi rule selects class `0` at the prototype `μ_0`. -/
theorem voronoiRule_proto_zero (μ : Prototypes K) : voronoiRule μ (μ 0) = 0 :=
  lexArgmin_eq_zero_of_le fun j => by
    rw [dsq_self]
    exact dsq_nonneg _ _

/-- **`H_Vor^Δ ⊊ H_Full` for `K ≥ 2`, for a trivial reason.** The constant
rule `1` is a full-matrix rule, and every rule of `H_Vor^Δ` takes the value
`0` at its prototype `μ_0`, which lies in the simplex. -/
theorem HVorSimplex_ssubset_HFull_of_two_le (hK : 2 ≤ K) :
    HVorSimplex (K := K) ⊂ HFull := by
  refine (Set.ssubset_iff_of_subset HVorSimplex_subset_HFull).2 ?_
  let c1 : Fin K := ⟨1, by omega⟩
  let A : CostMatrix K := Matrix.of fun _ k => if k = c1 then 0 else 1
  have hconst : ∀ p : Simplex K, lexArgmin (condRisk A p.val) = c1 := by
    intro p
    have hsum : ∑ i, p.val i = 1 := p.property.2
    refine lexArgmin_eq_of_strict fun j hj => ?_
    simp only [condRisk, A, Matrix.of_apply, if_neg hj, one_mul, hsum]
    norm_num
  refine ⟨fun p => lexArgmin (condRisk A p.val), ⟨A, rfl⟩, ?_⟩
  rintro ⟨μ, hμ, heq⟩
  have h : c1 = voronoiRule μ (μ 0) :=
    (hconst ⟨μ 0, hμ 0⟩).symm.trans (congrFun heq ⟨μ 0, hμ 0⟩)
  have : c1 = 0 := h.trans (voronoiRule_proto_zero μ)
  exact absurd (congrArg Fin.val this) (by simp [c1])

/-! ### The band rule -/

/-- The slab boundaries `m_l = 1 − 9/(10·3^l)`: `1/10, 7/10, 9/10, 29/30, …`,
strictly increasing and below `1`. -/
noncomputable def bnd (l : ℕ) : ℝ := 1 - 9 / (10 * 3 ^ l)

theorem bnd_zero : bnd 0 = 1 / 10 := by norm_num [bnd]

theorem bnd_one : bnd 1 = 7 / 10 := by norm_num [bnd]

theorem bnd_two : bnd 2 = 9 / 10 := by norm_num [bnd]

theorem bnd_strictMono : StrictMono bnd := by
  intro i j hij
  simp only [bnd]
  have h3 : (3 : ℝ) ^ i < 3 ^ j := pow_lt_pow_right₀ (by norm_num) hij
  have hi : (0 : ℝ) < 3 ^ i := by positivity
  have : 9 / (10 * (3 : ℝ) ^ j) < 9 / (10 * 3 ^ i) :=
    div_lt_div_of_pos_left (by norm_num) (by positivity) (by linarith)
  linarith

theorem bnd_mono : Monotone bnd := bnd_strictMono.monotone

theorem bnd_pos (l : ℕ) : 0 < bnd l := by
  have := bnd_mono (Nat.zero_le l)
  rw [bnd_zero] at this
  linarith

theorem bnd_lt_one (l : ℕ) : bnd l < 1 := by
  simp only [bnd]
  have : (0 : ℝ) < 9 / (10 * 3 ^ l) := by positivity
  linarith

/-- The band matrix: column `k` is `(∑_{l<k} m_l) 1 − k e_0`. -/
noncomputable def bandMatrix (K : ℕ) [NeZero K] : CostMatrix K :=
  Matrix.of fun i k => (∑ l ∈ Finset.range k, bnd l) - (k : ℝ) * vertex 0 i

/-- On the simplex the score of class `k` is `∑_{l<k} (m_l − p_0)`. -/
theorem condRisk_bandMatrix {p : Fin K → ℝ} (hp : p ∈ stdSimplex ℝ (Fin K))
    (k : Fin K) :
    condRisk (bandMatrix K) p k = ∑ l ∈ Finset.range k, (bnd l - p 0) := by
  have hsum : ∑ i, p i = 1 := hp.2
  have hterm : ∀ i : Fin K, bandMatrix K i k * p i
      = (∑ l ∈ Finset.range k, bnd l) * p i - (k : ℝ) * (vertex 0 i * p i) := by
    intro i
    simp only [bandMatrix, Matrix.of_apply]
    ring
  have hv : ∑ i : Fin K, vertex 0 i * p i = p 0 := by
    simp [vertex]
  simp only [condRisk]
  rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, hsum, hv, Finset.sum_sub_distrib,
    Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  ring

/-- Staircase scores select the slab containing `t`. -/
theorem lexArgmin_staircase (t : ℝ) (k : Fin K)
    (hlo : ∀ l < (k : ℕ), bnd l < t)
    (hhi : ∀ l, (k : ℕ) ≤ l → l + 1 < K → t ≤ bnd l) :
    lexArgmin (fun j : Fin K => ∑ l ∈ Finset.range j, (bnd l - t)) = k := by
  set s : Fin K → ℝ := fun j => ∑ l ∈ Finset.range j, (bnd l - t) with hs
  have hdiff : ∀ a b : Fin K, (a : ℕ) ≤ b →
      s b - s a = ∑ l ∈ Finset.Ico (a : ℕ) b, (bnd l - t) := by
    intro a b hab
    exact (Finset.sum_Ico_eq_sub _ hab).symm
  have hmin : ∀ j, s k ≤ s j := by
    intro j
    rcases le_or_gt (j : ℕ) k with hjk | hjk
    · have h1 := hdiff j k hjk
      have h2 : ∑ l ∈ Finset.Ico (j : ℕ) k, (bnd l - t) ≤ 0 :=
        Finset.sum_nonpos fun l hl => by
          have := hlo l (Finset.mem_Ico.1 hl).2
          linarith
      linarith
    · have h1 := hdiff k j hjk.le
      have h2 : 0 ≤ ∑ l ∈ Finset.Ico (k : ℕ) j, (bnd l - t) :=
        Finset.sum_nonneg fun l hl => by
          have hl' := Finset.mem_Ico.1 hl
          have := hhi l hl'.1 (by omega)
          linarith
      linarith
  refine le_antisymm (lexArgmin_le_index hmin) ?_
  by_contra hlt
  push Not at hlt
  have hlt' : ((lexArgmin s : Fin K) : ℕ) < k := hlt
  have h1 : s (lexArgmin s) ≤ s k := lexArgmin_le s k
  have h2 := hdiff (lexArgmin s) k hlt'.le
  have h3 : ∑ l ∈ Finset.Ico ((lexArgmin s : Fin K) : ℕ) k, (bnd l - t) < 0 := by
    have hne : (Finset.Ico ((lexArgmin s : Fin K) : ℕ) k).Nonempty :=
      ⟨_, Finset.mem_Ico.2 ⟨le_rfl, hlt'⟩⟩
    have := Finset.sum_lt_sum_of_nonempty hne (g := fun _ => (0 : ℝ))
      fun l hl => by
        have := hlo l (Finset.mem_Ico.1 hl).2
        show bnd l - t < 0
        linarith
    simpa using this
  linarith

/-- The band rule selects class `k` on the slab `m_{k−1} < p_0 ≤ m_k` (no
lower bound for `k = 0`, no upper bound for `k = K − 1`). -/
theorem bandRule_eq_of_slab {p : Fin K → ℝ} (hp : p ∈ stdSimplex ℝ (Fin K))
    (k : Fin K) (hlo : ∀ l < (k : ℕ), bnd l < p 0)
    (hhi : ∀ l, (k : ℕ) ≤ l → l + 1 < K → p 0 ≤ bnd l) :
    lexArgmin (condRisk (bandMatrix K) p) = k := by
  have h : condRisk (bandMatrix K) p
      = fun j : Fin K => ∑ l ∈ Finset.range j, (bnd l - p 0) :=
    funext (condRisk_bandMatrix hp)
  rw [h]
  exact lexArgmin_staircase _ k hlo hhi

/-- The band rule as a function on the simplex. -/
noncomputable def bandRule (K : ℕ) [NeZero K] : Simplex K → Fin K :=
  fun p => lexArgmin (condRisk (bandMatrix K) p.val)

theorem bandRule_mem_HFull : bandRule K ∈ HFull := ⟨bandMatrix K, rfl⟩

/-- **Every class of the band rule has nonempty interior.** For every class
`k` there are `t ∈ (0, 1)` and `r > 0` such that the rule is `k` at every
point of the simplex with `|p_0 − t| < r`; that set is relatively open in the
simplex and contains `t e_0 + (1 − t) e_1` (for `K ≥ 2`). -/
theorem bandRule_locally_const (k : Fin K) :
    ∃ t ∈ Set.Ioo (0 : ℝ) 1, ∃ r > 0, ∀ p : Simplex K, |p.val 0 - t| < r →
      bandRule K p = k := by
  -- The slab `(lo, bnd k)`, with `lo = 0` for `k = 0`.
  set lo : ℝ := if (k : ℕ) = 0 then 0 else bnd ((k : ℕ) - 1) with hlo_def
  have hlo0 : 0 ≤ lo := by
    simp only [hlo_def]
    split_ifs
    · exact le_rfl
    · exact (bnd_pos _).le
  have hlok : lo < bnd k := by
    simp only [hlo_def]
    split_ifs with h
    · exact bnd_pos _
    · exact bnd_strictMono (by omega)
  have hle : ∀ l < (k : ℕ), bnd l ≤ lo := by
    intro l hl
    simp only [hlo_def]
    rw [if_neg (by omega)]
    exact bnd_mono (by omega)
  refine ⟨(lo + bnd k) / 2, ⟨by linarith, by linarith [bnd_lt_one k]⟩,
    (bnd k - lo) / 2, by linarith, fun p hp => ?_⟩
  have hp' := abs_lt.1 hp
  refine bandRule_eq_of_slab p.property k (fun l hl => ?_) (fun l hl _ => ?_)
  · linarith [hle l hl]
  · linarith [bnd_mono hl]

/-! ### Linear score differences that change sign across a hyperplane -/

/-- The point `s e_0 + (1 − s) e_i` of the edge from `e_i` to `e_0`. -/
noncomputable def edgePt (i : Fin K) (s : ℝ) : Fin K → ℝ :=
  fun j => s * vertex 0 j + (1 - s) * vertex i j

theorem edgePt_mem {i : Fin K} {s : ℝ} (h0 : 0 ≤ s) (h1 : s ≤ 1) :
    edgePt i s ∈ stdSimplex ℝ (Fin K) := by
  constructor
  · intro j
    simp only [edgePt, vertex]
    split_ifs <;> nlinarith
  · simp only [edgePt, Finset.sum_add_distrib, ← Finset.mul_sum]
    simp [vertex]

theorem edgePt_zero {i : Fin K} (hi : i ≠ 0) (s : ℝ) : edgePt i s 0 = s := by
  simp [edgePt, vertex, Ne.symm hi]

theorem condRisk_edgePt (V : CostMatrix K) (i : Fin K) (s : ℝ) (k : Fin K) :
    condRisk V (edgePt i s) k = s * V 0 k + (1 - s) * V i k := by
  have h : ∀ j : Fin K, V j k * edgePt i s j
      = s * (V j k * vertex 0 j) + (1 - s) * (V j k * vertex i j) := by
    intro j; simp only [edgePt]; ring
  simp only [condRisk]
  rw [Finset.sum_congr rfl fun j _ => h j, Finset.sum_add_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum]
  simp [vertex]

/-- A number squeezed by `A − εc ≤ 0 < A + εc` for every small `ε > 0` is
`0`. -/
theorem eq_zero_of_squeeze {A c ε0 : ℝ} (hε0 : 0 < ε0)
    (h : ∀ ε, 0 < ε → ε ≤ ε0 → A - ε * c ≤ 0 ∧ 0 < A + ε * c) : A = 0 := by
  by_contra hA
  have hpos : 0 < |A| := abs_pos.2 hA
  set ε := min ε0 (|A| / (2 * (|c| + 1))) with hε_def
  have hε : 0 < ε := lt_min hε0 (by positivity)
  obtain ⟨h1, h2⟩ := h ε hε (min_le_left _ _)
  have hc1 : ε * c ≤ ε * |c| := mul_le_mul_of_nonneg_left (le_abs_self c) hε.le
  have hc2 : ε * -|c| ≤ ε * c := mul_le_mul_of_nonneg_left (neg_abs_le c) hε.le
  have habs : |A| ≤ ε * |c| := abs_le.2 ⟨by linarith, by linarith⟩
  have hεle : ε ≤ |A| / (2 * (|c| + 1)) := min_le_right _ _
  have hlt : ε * |c| < |A| := by
    have hc0 : 0 ≤ |c| := abs_nonneg c
    calc ε * |c| ≤ |A| / (2 * (|c| + 1)) * |c| :=
          mul_le_mul_of_nonneg_right hεle hc0
      _ < |A| := by
          rw [div_mul_eq_mul_div, div_lt_iff₀ (by positivity)]
          nlinarith
  linarith

/-- **A score difference that changes sign across `p_0 = a` is proportional
to `p_0 − a`.** For any matrix `V` and classes `j < k`: if along every edge
from `e_i` (`i ≠ 0`) to `e_0` the rule of `V` is `j` just below `p_0 = a` and
`k` just above it, then on the simplex
`(1 − a) (L_j(p) − L_k(p)) = (V_{0j} − V_{0k}) (p_0 − a)`. -/
theorem score_diff_eq_of_boundary (V : CostMatrix K) {j k : Fin K} (hjk : j < k)
    {a ε0 : ℝ} (hε0 : 0 < ε0)
    (hbelow : ∀ i : Fin K, i ≠ 0 → ∀ ε, 0 < ε → ε ≤ ε0 →
      lexArgmin (condRisk V (edgePt i (a - ε))) = j)
    (habove : ∀ i : Fin K, i ≠ 0 → ∀ ε, 0 < ε → ε ≤ ε0 →
      lexArgmin (condRisk V (edgePt i (a + ε))) = k)
    {p : Fin K → ℝ} (hp : p ∈ stdSimplex ℝ (Fin K)) :
    (1 - a) * (condRisk V p j - condRisk V p k)
      = (V 0 j - V 0 k) * (p 0 - a) := by
  set g : Fin K → ℝ := fun i => V i j - V i k with hg
  -- The face `p_0 = a`: `a g_0 + (1 − a) g_i = 0` for every `i ≠ 0`.
  have hface : ∀ i : Fin K, i ≠ 0 → (1 - a) * g i = -(a * g 0) := by
    intro i hi
    have hsq : a * g 0 + (1 - a) * g i = 0 := by
      refine eq_zero_of_squeeze (c := g 0 - g i) hε0 fun ε hε hεle => ⟨?_, ?_⟩
      · have h := lexArgmin_le (condRisk V (edgePt i (a - ε))) k
        rw [hbelow i hi ε hε hεle, condRisk_edgePt, condRisk_edgePt] at h
        simp only [hg]
        nlinarith
      · have h := lt_of_lexArgmin_eq (habove i hi ε hε hεle) hjk
        rw [condRisk_edgePt, condRisk_edgePt] at h
        simp only [hg]
        nlinarith
    linarith
  have hcoef : ∀ i : Fin K,
      (1 - a) * g i = -(a * g 0) + (if i = 0 then g 0 else 0) := by
    intro i
    by_cases hi : i = 0
    · subst hi; simp; ring
    · simp only [if_neg hi, add_zero]; exact hface i hi
  have hsum : ∑ i, p i = 1 := hp.2
  have hlin : condRisk V p j - condRisk V p k = ∑ i, g i * p i := by
    simp only [condRisk, hg, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [hlin, Finset.mul_sum]
  have hterm : ∀ i : Fin K, (1 - a) * (g i * p i)
      = -(a * g 0) * p i + (if i = 0 then g 0 * p i else 0) := by
    intro i
    rw [← mul_assoc, hcoef i]
    split_ifs <;> ring
  rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_add_distrib,
    ← Finset.mul_sum, hsum, Finset.sum_ite_eq' Finset.univ (0 : Fin K),
    if_pos (Finset.mem_univ _)]
  simp only [hg]
  ring

/-! ### No Voronoi rule with prototypes in the simplex is the band rule -/

theorem ne_bnd_of_lt {s : ℝ} (h : s < bnd 0) (l : ℕ) : s ≠ bnd l :=
  ne_of_lt (lt_of_lt_of_le h (bnd_mono (Nat.zero_le l)))

theorem ne_bnd_of_between {s : ℝ} {n : ℕ} (h1 : bnd n < s) (h2 : s < bnd (n + 1))
    (l : ℕ) : s ≠ bnd l := by
  rcases le_or_gt l n with hl | hl
  · exact ne_of_gt (lt_of_le_of_lt (bnd_mono hl) h1)
  · exact ne_of_lt (lt_of_lt_of_le h2 (bnd_mono (by omega)))

/-- **No Voronoi rule with prototypes in the simplex agrees with the band
rule, even off the slab boundaries.** For `K ≥ 3` and any prototypes in the
simplex (distinct or not), the Voronoi rule and the band rule differ at some
point of the simplex with `p_0 ≠ m_l` for every `l`. -/
theorem not_agree_bandRule (hK : 3 ≤ K) (μ : Prototypes K)
    (hμ : ∀ k, μ k ∈ stdSimplex ℝ (Fin K)) :
    ¬ ∀ p ∈ stdSimplex ℝ (Fin K), (∀ l, p 0 ≠ bnd l) →
      voronoiRule μ p = lexArgmin (condRisk (bandMatrix K) p) := by
  intro hagree
  set V := vorMatrix μ with hVdef
  let c1 : Fin K := ⟨1, by omega⟩
  let c2 : Fin K := ⟨2, by omega⟩
  have h01 : (0 : Fin K) < c1 := by
    show ((0 : Fin K) : ℕ) < 1
    simp
  have h12 : c1 < c2 := by
    show (1 : ℕ) < 2
    omega
  -- On the simplex off the boundaries, the rule of `V` is the band rule.
  have hV : ∀ s : ℝ, 0 ≤ s → s ≤ 1 → (∀ l, s ≠ bnd l) → ∀ i : Fin K, i ≠ 0 →
      lexArgmin (condRisk V (edgePt i s))
        = lexArgmin (condRisk (bandMatrix K) (edgePt i s)) := by
    intro s hs0 hs1 hsb i hi
    have hmem := edgePt_mem (i := i) hs0 hs1
    rw [← voronoiRule_eq_fullRule μ hmem]
    exact hagree _ hmem (by rw [edgePt_zero hi]; exact hsb)
  have hb0 := bnd_zero
  have hb1 := bnd_one
  have hb2 := bnd_two
  -- The rule of `V` on the four sides of the two boundaries.
  have hε : (0 : ℝ) < 1 / 20 := by norm_num
  have below0 : ∀ i : Fin K, i ≠ 0 → ∀ ε, 0 < ε → ε ≤ 1 / 20 →
      lexArgmin (condRisk V (edgePt i (bnd 0 - ε))) = 0 := by
    intro i hi ε hε0 hεle
    rw [hV _ (by linarith) (by linarith) (ne_bnd_of_lt (by linarith)) i hi]
    refine bandRule_eq_of_slab (edgePt_mem (by linarith) (by linarith)) 0
      (fun l hl => absurd hl (by simp)) fun l _ _ => ?_
    rw [edgePt_zero hi]
    linarith [bnd_mono (Nat.zero_le l)]
  have above0 : ∀ i : Fin K, i ≠ 0 → ∀ ε, 0 < ε → ε ≤ 1 / 20 →
      lexArgmin (condRisk V (edgePt i (bnd 0 + ε))) = c1 := by
    intro i hi ε hε0 hεle
    rw [hV _ (by linarith) (by linarith)
      (ne_bnd_of_between (n := 0) (by linarith) (by linarith)) i hi]
    refine bandRule_eq_of_slab (edgePt_mem (by linarith) (by linarith)) c1
      (fun l hl => ?_) fun l hl _ => ?_
    · rw [edgePt_zero hi]
      have : l = 0 := by simp [c1] at hl; omega
      subst this
      linarith
    · rw [edgePt_zero hi]
      have : 1 ≤ l := by simpa [c1] using hl
      linarith [bnd_mono this]
  have below1 : ∀ i : Fin K, i ≠ 0 → ∀ ε, 0 < ε → ε ≤ 1 / 20 →
      lexArgmin (condRisk V (edgePt i (bnd 1 - ε))) = c1 := by
    intro i hi ε hε0 hεle
    rw [hV _ (by linarith) (by linarith)
      (ne_bnd_of_between (n := 0) (by linarith) (by linarith)) i hi]
    refine bandRule_eq_of_slab (edgePt_mem (by linarith) (by linarith)) c1
      (fun l hl => ?_) fun l hl _ => ?_
    · rw [edgePt_zero hi]
      have : l = 0 := by simp [c1] at hl; omega
      subst this
      linarith
    · rw [edgePt_zero hi]
      have : 1 ≤ l := by simpa [c1] using hl
      linarith [bnd_mono this]
  have above1 : ∀ i : Fin K, i ≠ 0 → ∀ ε, 0 < ε → ε ≤ 1 / 20 →
      lexArgmin (condRisk V (edgePt i (bnd 1 + ε))) = c2 := by
    intro i hi ε hε0 hεle
    rw [hV _ (by linarith) (by linarith)
      (ne_bnd_of_between (n := 1) (by linarith) (by linarith)) i hi]
    refine bandRule_eq_of_slab (edgePt_mem (by linarith) (by linarith)) c2
      (fun l hl => ?_) fun l hl _ => ?_
    · rw [edgePt_zero hi]
      have : l ≤ 1 := by simp [c2] at hl; omega
      linarith [bnd_mono this]
    · rw [edgePt_zero hi]
      have : 2 ≤ l := by simpa [c2] using hl
      linarith [bnd_mono this]
  -- The two score differences are proportional to `p_0 − a` and `p_0 − b`.
  have hF0 : ∀ p ∈ stdSimplex ℝ (Fin K),
      (1 - bnd 0) * (condRisk V p 0 - condRisk V p c1)
        = (V 0 0 - V 0 c1) * (p 0 - bnd 0) :=
    fun p hp => score_diff_eq_of_boundary V h01 hε below0 above0 hp
  have hF1 : ∀ p ∈ stdSimplex ℝ (Fin K),
      (1 - bnd 1) * (condRisk V p c1 - condRisk V p c2)
        = (V 0 c1 - V 0 c2) * (p 0 - bnd 1) :=
    fun p hp => score_diff_eq_of_boundary V h12 hε below1 above1 hp
  -- Voronoi scores: `F(μ_j) + F(μ_k) = 0`, the bisector halves the segment.
  have hmid : ∀ j k : Fin K,
      (condRisk V (μ j) j - condRisk V (μ j) k)
        + (condRisk V (μ k) j - condRisk V (μ k) k) = 0 := by
    intro j k
    have e1 := dsq_eq_condRisk_vorMatrix μ (hμ j) j
    have e2 := dsq_eq_condRisk_vorMatrix μ (hμ j) k
    have e3 := dsq_eq_condRisk_vorMatrix μ (hμ k) j
    have e4 := dsq_eq_condRisk_vorMatrix μ (hμ k) k
    rw [dsq_self] at e1 e4
    rw [dsq_comm (μ k) (μ j)] at e3
    linarith
  -- The coefficients are nonzero: the rule does change across each boundary.
  have hc1 : c1 ≠ 0 := ne_of_gt h01
  have hg0 : V 0 0 - V 0 c1 ≠ 0 := by
    intro h0
    have hmem := edgePt_mem (i := c1) (s := bnd 0 + 1 / 20) (by linarith)
      (by linarith)
    have e := hF0 _ hmem
    rw [h0, zero_mul] at e
    have hlt := lt_of_lexArgmin_eq (above0 c1 hc1 (1 / 20) hε le_rfl) h01
    have : condRisk V (edgePt c1 (bnd 0 + 1 / 20)) 0
        - condRisk V (edgePt c1 (bnd 0 + 1 / 20)) c1 = 0 := by
      rcases mul_eq_zero.1 e with h | h
      · rw [hb0] at h; norm_num at h
      · exact h
    linarith
  have hg1 : V 0 c1 - V 0 c2 ≠ 0 := by
    intro h0
    have hmem := edgePt_mem (i := c1) (s := bnd 1 + 1 / 20) (by linarith)
      (by linarith)
    have e := hF1 _ hmem
    rw [h0, zero_mul] at e
    have hlt := lt_of_lexArgmin_eq (above1 c1 hc1 (1 / 20) hε le_rfl) h12
    have : condRisk V (edgePt c1 (bnd 1 + 1 / 20)) c1
        - condRisk V (edgePt c1 (bnd 1 + 1 / 20)) c2 = 0 := by
      rcases mul_eq_zero.1 e with h | h
      · rw [hb1] at h; norm_num at h
      · exact h
    linarith
  -- Midpoints: `x_0 + x_1 = 2a` and `x_1 + x_2 = 2b`.
  have hx01 : μ 0 0 + μ c1 0 = 2 * bnd 0 := by
    have e1 := hF0 _ (hμ 0)
    have e2 := hF0 _ (hμ c1)
    have e3 := hmid 0 c1
    have h : (V 0 0 - V 0 c1) * (μ 0 0 + μ c1 0 - 2 * bnd 0) = 0 := by
      linear_combination (1 - bnd 0) * e3 - e1 - e2
    rcases mul_eq_zero.1 h with h | h
    · exact absurd h hg0
    · linarith
  have hx12 : μ c1 0 + μ c2 0 = 2 * bnd 1 := by
    have e1 := hF1 _ (hμ c1)
    have e2 := hF1 _ (hμ c2)
    have e3 := hmid c1 c2
    have h : (V 0 c1 - V 0 c2) * (μ c1 0 + μ c2 0 - 2 * bnd 1) = 0 := by
      linear_combination (1 - bnd 1) * e3 - e1 - e2
    rcases mul_eq_zero.1 h with h | h
    · exact absurd h hg1
    · linarith
  -- Prototypes in the simplex: `0 ≤ x_0` and `x_2 ≤ 1`, so `2b − 1 ≤ x_1 ≤ 2a`.
  have hx0 : 0 ≤ μ 0 0 := (hμ 0).1 0
  have hx2 : μ c2 0 ≤ 1 := le_one_of_mem_stdSimplex (hμ c2) 0
  linarith

/-- The band rule is not in `H_Vor^Δ` (`K ≥ 3`). -/
theorem bandRule_not_mem_HVorSimplex (hK : 3 ≤ K) :
    bandRule K ∉ HVorSimplex (K := K) := by
  rintro ⟨μ, hμ, heq⟩
  exact not_agree_bandRule hK μ hμ fun p hp _ => (congrFun heq ⟨p, hp⟩).symm

/-- **`H_Vor^Δ ⊊ H_Full` for `K ≥ 3`, by a rule whose classes all have
nonempty interior.** The band rule (`bandRule_mem_HFull`,
`bandRule_locally_const`) is a full-matrix rule and is not the rule of any
prototypes in the simplex; by `not_agree_bandRule` this holds even if the
two rules are only required to agree off the slab boundaries. -/
theorem HVorSimplex_ssubset_HFull (hK : 3 ≤ K) :
    HVorSimplex (K := K) ⊂ HFull :=
  (Set.ssubset_iff_of_subset HVorSimplex_subset_HFull).2
    ⟨bandRule K, bandRule_mem_HFull, bandRule_not_mem_HVorSimplex hK⟩

/-! ### `H_PS ⊄ H_Vor^Δ` under the `0/1` loss

Proposition `prop:ps_in_vor` places every parallel-shift rule in `H_Vor`, with
prototypes `−½ c_k + β_k 1` that need not lie in the simplex. They cannot
always be moved into it: for `K = 3`, the `0/1` loss and the offsets
`v = (9/10, 0, 1/2)`, the parallel-shift rule has three classes with nonempty
interior, classes `0, 1` meet along `p_0 − p_1 = 9/10` and classes `1, 2` along
`p_1 − p_2 = −1/2`, and no Voronoi rule with prototypes in the simplex agrees
with it off those two lines (`not_agree_psWitness`).

The argument is the one of `not_agree_bandRule`, along the directions
`e_0 − e_1` and `e_1 − e_2`. The score difference of classes `0` and `1`
vanishes at two points of their common boundary, so on the simplex it is a
multiple of `p_0 − p_1 − 9/10`; for Voronoi scores this forces
`μ_1 − μ_0 ∥ e_0 − e_1` and the midpoint on the line, hence `(μ_0)_0 ≥ 9/10`.
The same for classes `1, 2` forces `(μ_1)_2 = (μ_0)_2 ≥ 1/2`, and
`9/10 + 1/2 > 1`. -/

section ParallelShift

open ThresholdOpt.Partition

/-- The score difference of two classes vanishes at a point where the rule
switches from `j` (along `+d`) to `k > j` (along `−d`). Any matrix, any `K`. -/
theorem score_diff_eq_zero_of_switch (V : CostMatrix K)
    {j k : Fin K} (hjk : j < k) {q d : Fin K → ℝ} {ε0 : ℝ} (hε0 : 0 < ε0)
    (hplus : ∀ ε, 0 < ε → ε ≤ ε0 → lexArgmin (condRisk V (q + ε • d)) = j)
    (hminus : ∀ ε, 0 < ε → ε ≤ ε0 → lexArgmin (condRisk V (q - ε • d)) = k) :
    condRisk V q j - condRisk V q k = 0 := by
  have hlin : ∀ (ε : ℝ) (l : Fin K),
      condRisk V (q + ε • d) l = condRisk V q l + ε * condRisk V d l ∧
      condRisk V (q - ε • d) l = condRisk V q l - ε * condRisk V d l := by
    intro ε l
    simp only [condRisk, Pi.add_apply, Pi.sub_apply, Pi.smul_apply, smul_eq_mul,
      mul_add, mul_sub, Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.mul_sum]
    constructor <;> congr 1 <;> exact Finset.sum_congr rfl fun i _ => by ring
  refine eq_zero_of_squeeze (c := -(condRisk V d j - condRisk V d k)) hε0
    fun ε hε hεle => ⟨?_, ?_⟩
  · have h := lexArgmin_le (condRisk V (q + ε • d)) k
    rw [hplus ε hε hεle, (hlin ε j).1, (hlin ε k).1] at h
    linarith
  · have h := lt_of_lexArgmin_eq (hminus ε hε hεle) hjk
    rw [(hlin ε j).2, (hlin ε k).2] at h
    linarith

omit [NeZero K] in
/-- For Voronoi scores the bisector of two prototypes passes through their
midpoint: `F(μ_j) + F(μ_k) = 0` for `F = L_j − L_k`. -/
theorem vor_score_midpoint (μ : Prototypes K)
    (hμ : ∀ k, μ k ∈ stdSimplex ℝ (Fin K)) (j k : Fin K) :
    (condRisk (vorMatrix μ) (μ j) j - condRisk (vorMatrix μ) (μ j) k)
      + (condRisk (vorMatrix μ) (μ k) j - condRisk (vorMatrix μ) (μ k) k) = 0 := by
  have e1 := dsq_eq_condRisk_vorMatrix μ (hμ j) j
  have e2 := dsq_eq_condRisk_vorMatrix μ (hμ j) k
  have e3 := dsq_eq_condRisk_vorMatrix μ (hμ k) j
  have e4 := dsq_eq_condRisk_vorMatrix μ (hμ k) k
  rw [dsq_self] at e1 e4
  rw [dsq_comm (μ k) (μ j)] at e3
  linarith

/-- The offsets of the witness. -/
noncomputable def psWitnessOffsets : Fin 3 → ℝ := ![9 / 10, 0, 1 / 2]

theorem condRisk_three (V : CostMatrix 3) (p : Fin 3 → ℝ) (k : Fin 3) :
    condRisk V p k = V 0 k * p 0 + V 1 k * p 1 + V 2 k * p 2 := by
  simp [condRisk, Fin.sum_univ_three]

theorem condRisk_vec3 (V : CostMatrix 3) (x y z : ℝ) (k : Fin 3) :
    condRisk V ![x, y, z] k = V 0 k * x + V 1 k * y + V 2 k * z := by
  simp [condRisk, Fin.sum_univ_three]

theorem vec3_add_smul (x y z a b c ε : ℝ) :
    ![x, y, z] + ε • ![a, b, c] = ![x + ε * a, y + ε * b, z + ε * c] := by
  ext i
  fin_cases i <;> simp

theorem vec3_sub_smul (x y z a b c ε : ℝ) :
    ![x, y, z] - ε • ![a, b, c] = ![x - ε * a, y - ε * b, z - ε * c] := by
  ext i
  fin_cases i <;> simp

theorem mem_simplex_three {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
    (h : x + y + z = 1) : ![x, y, z] ∈ stdSimplex ℝ (Fin 3) := by
  refine ⟨fun i => ?_, ?_⟩
  · fin_cases i <;> simpa
  · simpa [Fin.sum_univ_three] using h

theorem lexArgmin_three_zero {a b c : ℝ} (h1 : a < b) (h2 : a < c) :
    lexArgmin ![a, b, c] = 0 :=
  lexArgmin_eq_of_strict fun j hj => by fin_cases j <;> simp_all

theorem lexArgmin_three_one {a b c : ℝ} (h1 : b < a) (h2 : b < c) :
    lexArgmin ![a, b, c] = 1 :=
  lexArgmin_eq_of_strict fun j hj => by fin_cases j <;> simp_all

theorem lexArgmin_three_two {a b c : ℝ} (h1 : c < a) (h2 : c < b) :
    lexArgmin ![a, b, c] = 2 :=
  lexArgmin_eq_of_strict fun j hj => by fin_cases j <;> simp_all

/-- The witness rule at a point `(x, y, z)` of the simplex selects the least
of `1 − x + 9/10`, `1 − y`, `1 − z + 1/2`. -/
theorem psRule_witness {x y z : ℝ} (hp : ![x, y, z] ∈ stdSimplex ℝ (Fin 3)) :
    psRule (zeroOne 3) psWitnessOffsets ![x, y, z]
      = lexArgmin ![1 - x + 9 / 10, 1 - y, 1 - z + 1 / 2] := by
  unfold psRule
  congr 1
  funext k
  rw [psScore, condRisk_zeroOne hp]
  fin_cases k <;> simp [psWitnessOffsets]

/-- **No Voronoi rule with prototypes in the simplex agrees with the witness
parallel-shift rule, even off its two boundary lines.** `K = 3`, `0/1` loss,
offsets `(9/10, 0, 1/2)`. -/
theorem not_agree_psWitness (μ : Prototypes 3) (hμ : ∀ k, μ k ∈ stdSimplex ℝ (Fin 3)) :
    ¬ ∀ p ∈ stdSimplex ℝ (Fin 3), p 0 - p 1 ≠ 9 / 10 → p 1 - p 2 ≠ -1 / 2 →
      voronoiRule μ p = psRule (zeroOne 3) psWitnessOffsets p := by
  intro hagree
  set V := vorMatrix μ with hVdef
  -- Off the two lines the rule of `V` is the witness rule.
  have hV : ∀ x y z : ℝ, 0 ≤ x → 0 ≤ y → 0 ≤ z → x + y + z = 1 →
      x - y ≠ 9 / 10 → y - z ≠ -1 / 2 →
      lexArgmin (condRisk V ![x, y, z])
        = lexArgmin ![1 - x + 9 / 10, 1 - y, 1 - z + 1 / 2] := by
    intro x y z hx hy hz hs h1 h2
    have hmem := mem_simplex_three hx hy hz hs
    rw [← voronoiRule_eq_fullRule μ hmem, ← psRule_witness hmem]
    exact hagree _ hmem (by simpa using h1) (by simpa using h2)
  have hε : (0 : ℝ) < 1 / 100 := by norm_num
  -- Classes `0` and `1` switch across `p_0 − p_1 = 9/10` at two points.
  have hF01 : ∀ x y z : ℝ, x - y = 9 / 10 → 1 / 100 ≤ y → y ≤ 1 / 20 → x + y + z = 1 →
      condRisk V ![x, y, z] 0 - condRisk V ![x, y, z] 1 = 0 := by
    intro x y z hxy hy1 hy2 hs
    have hz : z = 1 / 10 - 2 * y := by linarith
    refine score_diff_eq_zero_of_switch V (by decide) (d := ![1, -1, 0]) hε
      (fun ε he hle => ?_) (fun ε he hle => ?_)
    · rw [vec3_add_smul, hV _ _ _ (by linarith) (by linarith) (by linarith) (by linarith)
        (by intro h; linarith) (by intro h; linarith)]
      exact lexArgmin_three_zero (by linarith) (by linarith)
    · rw [vec3_sub_smul, hV _ _ _ (by linarith) (by linarith) (by linarith) (by linarith)
        (by intro h; linarith) (by intro h; linarith)]
      exact lexArgmin_three_one (by linarith) (by linarith)
  -- Classes `1` and `2` switch across `p_1 − p_2 = −1/2` at two points.
  have hF12 : ∀ x y z : ℝ, y - z = -1 / 2 → 1 / 20 ≤ y → y ≤ 3 / 20 → x + y + z = 1 →
      condRisk V ![x, y, z] 1 - condRisk V ![x, y, z] 2 = 0 := by
    intro x y z hyz hy1 hy2 hs
    have hx : x = 1 / 2 - 2 * y := by linarith
    refine score_diff_eq_zero_of_switch V (by decide) (d := ![0, 1, -1]) hε
      (fun ε he hle => ?_) (fun ε he hle => ?_)
    · rw [vec3_add_smul, hV _ _ _ (by linarith) (by linarith) (by linarith) (by linarith)
        (by intro h; linarith) (by intro h; linarith)]
      exact lexArgmin_three_one (by linarith) (by linarith)
    · rw [vec3_sub_smul, hV _ _ _ (by linarith) (by linarith) (by linarith) (by linarith)
        (by intro h; linarith) (by intro h; linarith)]
      exact lexArgmin_three_two (by linarith) (by linarith)
  have q1 := hF01 (91 / 100) (1 / 100) (8 / 100) (by norm_num) le_rfl (by norm_num) (by norm_num)
  have q2 := hF01 (94 / 100) (4 / 100) (2 / 100) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)
  have r1 := hF12 (2 / 5) (1 / 20) (11 / 20) (by norm_num) le_rfl (by norm_num) (by norm_num)
  have r2 := hF12 (1 / 5) (3 / 20) (13 / 20) (by norm_num) (by norm_num) le_rfl (by norm_num)
  simp only [condRisk_vec3] at q1 q2 r1 r2
  -- The two score differences do change sign, so they are not identically zero.
  have s01 : condRisk V ![9 / 10, 1 / 50, 2 / 25] 1 < condRisk V ![9 / 10, 1 / 50, 2 / 25] 0 := by
    have h := hV (9 / 10) (1 / 50) (2 / 25) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num)
    rw [lexArgmin_three_one (by norm_num) (by norm_num)] at h
    exact lt_of_lexArgmin_eq h (by decide)
  have s12 : condRisk V ![2 / 5, 1 / 25, 14 / 25] 2 < condRisk V ![2 / 5, 1 / 25, 14 / 25] 1 := by
    have h := hV (2 / 5) (1 / 25) (14 / 25) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num)
    rw [lexArgmin_three_two (by norm_num) (by norm_num)] at h
    exact lt_of_lexArgmin_eq h (by decide)
  simp only [condRisk_vec3] at s01 s12
  -- Along the boundary of `0, 1`: `w_1 = −19 w_0`, `w_2 = −9 w_0`, `w_0 ≠ 0`.
  have h19 : (V 1 0 - V 1 1) + 19 * (V 0 0 - V 0 1) = 0 := by linarith
  have h9 : (V 2 0 - V 2 1) + 9 * (V 0 0 - V 0 1) = 0 := by linarith
  have hw0 : V 0 0 - V 0 1 ≠ 0 := by
    intro h0
    linarith
  -- Along the boundary of `1, 2`: `w'_1 = 3 w'_0`, `w'_2 = −w'_0`, `w'_0 ≠ 0`.
  have h3 : (V 1 1 - V 1 2) - 3 * (V 0 1 - V 0 2) = 0 := by linarith
  have h1 : (V 2 1 - V 2 2) + (V 0 1 - V 0 2) = 0 := by linarith
  have hw0' : V 0 1 - V 0 2 ≠ 0 := by
    intro h0
    linarith
  -- Midpoints.
  have m01 := vor_score_midpoint μ hμ 0 1
  have m12 := vor_score_midpoint μ hμ 1 2
  rw [← hVdef] at m01 m12
  simp only [condRisk_three] at m01 m12
  have hX : (V 0 0 - V 0 1) * ((μ 0 0 + μ 1 0) - 19 * (μ 0 1 + μ 1 1)
      - 9 * (μ 0 2 + μ 1 2)) = 0 := by
    linear_combination m01 - (μ 0 1 + μ 1 1) * h19 - (μ 0 2 + μ 1 2) * h9
  have hY : (V 0 1 - V 0 2) * ((μ 1 0 + μ 2 0) + 3 * (μ 1 1 + μ 2 1)
      - (μ 1 2 + μ 2 2)) = 0 := by
    linear_combination m12 - (μ 1 1 + μ 2 1) * h3 - (μ 1 2 + μ 2 2) * h1
  have X0 := (mul_eq_zero.1 hX).resolve_left hw0
  have Y0 := (mul_eq_zero.1 hY).resolve_left hw0'
  -- Back to the prototypes: `V_{ik} = −2 μ_k(i) + ‖μ_k‖²`.
  simp only [hVdef, vorMatrix, Matrix.of_apply] at h19 h9 h3 h1
  have sa : μ 0 0 + μ 0 1 + μ 0 2 = 1 := by
    have := (hμ 0).2
    simpa [Fin.sum_univ_three] using this
  have sb : μ 1 0 + μ 1 1 + μ 1 2 = 1 := by
    have := (hμ 1).2
    simpa [Fin.sum_univ_three] using this
  have sc : μ 2 0 + μ 2 1 + μ 2 2 = 1 := by
    have := (hμ 2).2
    simpa [Fin.sum_univ_three] using this
  have na1 : 0 ≤ μ 0 1 := (hμ 0).1 1
  have nb1 : 0 ≤ μ 1 1 := (hμ 1).1 1
  have nc1 : 0 ≤ μ 2 1 := (hμ 2).1 1
  -- `μ_1 − μ_0 ∥ e_0 − e_1`, hence `(μ_0)_0 ≥ 9/10` and `(μ_1)_2 = (μ_0)_2`.
  have hd2 : μ 1 2 = μ 0 2 := by linarith
  have ha0 : 9 / 10 ≤ μ 0 0 := by linarith
  -- `μ_2 − μ_1 ∥ e_1 − e_2`, hence `(μ_1)_2 ≥ 1/2`.
  have hb2 : 1 / 2 ≤ μ 1 2 := by linarith
  linarith

/-- The witness rule as a function on the simplex. -/
noncomputable def psWitness : Simplex 3 → Fin 3 :=
  fun p => psRule (zeroOne 3) psWitnessOffsets p.val

/-- **`H_PS ⊄ H_Vor^Δ`**: under the `0/1` loss with `K = 3`, a parallel-shift
rule that is not the rule of any prototypes in the simplex. With
`prop:ps_in_vor` (`H_PS ⊊ H_Vor`), the containment holds for the
unconstrained Voronoi class and fails for the simplex-restricted one. -/
theorem psWitness_not_mem_HVorSimplex : psWitness ∉ HVorSimplex (K := 3) := by
  rintro ⟨μ, hμ, heq⟩
  exact not_agree_psWitness μ hμ fun p hp _ _ => (congrFun heq ⟨p, hp⟩).symm

theorem HPS_zeroOne_not_subset_HVorSimplex : ¬ HPS (zeroOne 3) ⊆ HVorSimplex := by
  intro h
  exact psWitness_not_mem_HVorSimplex (h ⟨psWitnessOffsets, rfl⟩)

end ParallelShift

end ThresholdOpt.VoronoiSimplex
