import Mathlib.Tactic
import ThresholdOpt.Capacity

/-!
# Natarajan's lemma and the capacity of classes read off a decision rule

The generalization bound for `K` classes is stated in the text in terms of the
Natarajan dimension of the class of rules (`rem:natarajan_gen_bound`). The
uniform-convergence theorem that is imported bounds classes of *binary*
functions by their VC dimension. This file is the step between the two.

* `card_le_of_nshatters_le` is **Natarajan's lemma**: a class of functions from
  an `n`-point set to `K` labels in which no set of more than `d` points is
  Natarajan-shattered has at most `((n + 1) K²)^d` members. It is proved by
  induction on the number of points: after removing a point, the members split
  into the restrictions (same dimension) and, for each ordered pair of
  distinct labels, the restrictions that extend with both labels (dimension
  one less).
* `vcDim_pointwise_le`: let `G` be a class of binary functions obtained from a
  class `H` of rules pointwise, `g_δ(z) = ψ(z, δ(φ(z)))`. Then
  `d_VC(G) ≤ 2 d (⌈log₂(d K²)⌉ + 1)` when `d_Nat(H) ≤ d`. A set of `n`
  examples shattered by `G` forces `2ⁿ` distinct restrictions of `H` to at most
  `n` points, and the lemma bounds their number.

Nothing is assumed. The constant is not optimized: the count uses ordered pairs
of labels and the crude bound `((n + 1) K²)^d` for
`∑_{i ≤ d} C(n, i) C(K, 2)^i`.

Sources, checked against the text of Daniely, Sabato, Ben-David and
Shalev-Shwartz, *Multiclass learnability and the ERM principle*, JMLR 16
(2015). The lemma is Natarajan's (1989); they state it as Lemma 14,
`|H| ≤ |X|^{d_N} |Y|^{2 d_N}`. For the graph dimension their equation (2)
quotes `d_G ≤ 4.67 log₂(k) d_N` from Ben-David, Cesa-Bianchi, Haussler and
Long (1995), which has no `log d`; that sharper estimate is not proved here.
-/

namespace ThresholdOpt.Capacity

open ThresholdOpt

variable {K : ℕ}

/-! ### Lifting a shattered set by one point -/

/-- If every member of `F'` extends to a member of `F` by some label at the
new point, a set shattered by `F'` is shattered by `F`. -/
theorem NShatters.of_tail [NeZero K] {n : ℕ} {F : Set (Fin (n + 1) → Fin K)}
    {F' : Set (Fin n → Fin K)}
    (hF' : ∀ h' ∈ F', ∃ a, (Fin.cons a h' : Fin (n + 1) → Fin K) ∈ F)
    {s : Finset (Fin n)} (hs : NShatters F' s) :
    NShatters F (s.map (Fin.succEmb n)) := by
  obtain ⟨f, g, hfg, hreal⟩ := hs
  refine ⟨Fin.cons 0 f, Fin.cons 0 g, ?_, fun b => ?_⟩
  · intro x hx
    obtain ⟨i, hi, rfl⟩ := Finset.mem_map.1 hx
    simpa using hfg i hi
  · obtain ⟨h', hh', hval⟩ := hreal fun i => b i.succ
    obtain ⟨a, ha⟩ := hF' h' hh'
    refine ⟨Fin.cons a h', ha, ?_⟩
    intro x hx
    obtain ⟨i, hi, rfl⟩ := Finset.mem_map.1 hx
    simpa using hval i hi

/-- If every member of `F'` extends to a member of `F` by each of two distinct
labels `a` and `b` at the new point, a set shattered by `F'` together with the
new point is shattered by `F`. -/
theorem NShatters.of_pair {n : ℕ} {F : Set (Fin (n + 1) → Fin K)}
    {F' : Set (Fin n → Fin K)} {a b : Fin K} (hab : a ≠ b)
    (hF' : ∀ h' ∈ F', (Fin.cons a h' : Fin (n + 1) → Fin K) ∈ F ∧
      (Fin.cons b h' : Fin (n + 1) → Fin K) ∈ F)
    {s : Finset (Fin n)} (hs : NShatters F' s) :
    NShatters F (insert 0 (s.map (Fin.succEmb n))) := by
  obtain ⟨f, g, hfg, hreal⟩ := hs
  refine ⟨Fin.cons a f, Fin.cons b g, ?_, fun β => ?_⟩
  · intro x hx
    rcases Finset.mem_insert.1 hx with rfl | hx
    · simpa using hab
    · obtain ⟨i, hi, rfl⟩ := Finset.mem_map.1 hx
      simpa using hfg i hi
  · obtain ⟨h', hh', hval⟩ := hreal fun i => β i.succ
    obtain ⟨ha, hb⟩ := hF' h' hh'
    refine ⟨Fin.cons (if β 0 = true then a else b) h', ?_, ?_⟩
    · by_cases h0 : β 0 = true
      · simpa [h0] using ha
      · simpa [h0] using hb
    · intro x hx
      rcases Finset.mem_insert.1 hx with rfl | hx
      · by_cases h0 : β 0 = true <;> simp [h0]
      · obtain ⟨i, hi, rfl⟩ := Finset.mem_map.1 hx
        simpa using hval i hi

/-! ### Natarajan's lemma -/

/-- **Natarajan's lemma.** A finite class of functions `{0, …, n−1} → {0, …,
K−1}` in which every Natarajan-shattered set has at most `d` points has at most
`((n + 1) K²)^d` members. -/
theorem card_le_of_nshatters_le [NeZero K] : ∀ (n d : ℕ) (F : Finset (Fin n → Fin K)),
    (∀ s : Finset (Fin n), NShatters (F : Set (Fin n → Fin K)) s → s.card ≤ d) →
    F.card ≤ ((n + 1) * K ^ 2) ^ d := by
  classical
  have hK : 0 < K := Nat.pos_of_ne_zero (NeZero.ne K)
  intro n
  induction n with
  | zero =>
    intro d F _
    have h1 : F.card ≤ 1 := Finset.card_le_one.2 fun a _ b _ => Subsingleton.elim a b
    exact h1.trans (Nat.one_le_pow _ _ (by positivity))
  | succ n ih =>
    intro d F hF
    -- The labels by which a function on the first `n` points extends in `F`.
    let A : (Fin n → Fin K) → Finset (Fin K) := fun h' =>
      Finset.univ.filter fun a => (Fin.cons a h' : Fin (n + 1) → Fin K) ∈ F
    -- The restrictions, and the restrictions that extend by both labels of a pair.
    let F₀ : Finset (Fin n → Fin K) := Finset.univ.filter fun h' => (A h').Nonempty
    let Fp : Fin K × Fin K → Finset (Fin n → Fin K) := fun p =>
      Finset.univ.filter fun h' => (Fin.cons p.1 h' : Fin (n + 1) → Fin K) ∈ F ∧
        (Fin.cons p.2 h' : Fin (n + 1) → Fin K) ∈ F
    have hcount : F.card = ∑ h' : Fin n → Fin K, (A h').card := by
      rw [Finset.card_eq_sum_card_fiberwise (f := Fin.tail) (t := Finset.univ)
        (fun _ _ => Finset.mem_univ _)]
      refine Finset.sum_congr rfl fun h' _ => ?_
      have hfib : F.filter (fun h => Fin.tail h = h')
          = (A h').image fun a => (Fin.cons a h' : Fin (n + 1) → Fin K) := by
        ext h
        simp only [Finset.mem_filter, Finset.mem_image, Finset.mem_univ, true_and, A]
        constructor
        · rintro ⟨hh, rfl⟩
          exact ⟨h 0, by rwa [Fin.cons_self_tail], Fin.cons_self_tail h⟩
        · rintro ⟨a, ha, rfl⟩
          exact ⟨ha, Fin.tail_cons _ _⟩
      have hinj : Function.Injective
          fun a : Fin K => (Fin.cons a h' : Fin (n + 1) → Fin K) := by
        intro a b hab
        simpa using congrFun hab 0
      rw [hfib, Finset.card_image_of_injective _ hinj]
    have hF0 : F₀.card = ∑ h' : Fin n → Fin K, if (A h').Nonempty then 1 else 0 := by
      simp only [F₀, Finset.card_filter]
    have hFp : ∑ p ∈ (Finset.univ : Finset (Fin K)).offDiag, (Fp p).card
        = ∑ h' : Fin n → Fin K, (A h').offDiag.card := by
      simp only [Fp, Finset.card_filter]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun h' _ => ?_
      rw [← Finset.card_filter]
      congr 1
      ext p
      simp only [Finset.mem_filter, Finset.mem_offDiag, Finset.mem_univ, true_and, A]
      tauto
    have hsplit : F.card
        ≤ F₀.card + ∑ p ∈ (Finset.univ : Finset (Fin K)).offDiag, (Fp p).card := by
      rw [hcount, hF0, hFp, ← Finset.sum_add_distrib]
      refine Finset.sum_le_sum fun h' _ => ?_
      rw [Finset.offDiag_card]
      by_cases hne : (A h').Nonempty
      · rw [if_pos hne]
        obtain ⟨e, he⟩ : ∃ e, (A h').card = e + 1 :=
          ⟨(A h').card - 1, by have := Finset.card_pos.2 hne; omega⟩
        rw [he]
        have : (e + 1) * (e + 1) = e * e + 2 * e + 1 := by ring
        rw [this]
        generalize e * e = q
        omega
      · rw [Finset.not_nonempty_iff_eq_empty.1 hne]
        simp
    have h0 : F₀.card ≤ ((n + 1) * K ^ 2) ^ d := by
      refine ih d F₀ fun s hs => ?_
      have hlift : NShatters (F : Set (Fin (n + 1) → Fin K)) (s.map (Fin.succEmb n)) := by
        refine NShatters.of_tail (fun h' hh' => ?_) hs
        have hne : (A h').Nonempty := by simpa [F₀] using hh'
        obtain ⟨a, ha⟩ := hne
        exact ⟨a, by simpa [A] using ha⟩
      simpa using hF _ hlift
    have hpair : ∀ p ∈ (Finset.univ : Finset (Fin K)).offDiag, ∀ s : Finset (Fin n),
        NShatters (Fp p : Set (Fin n → Fin K)) s → s.card + 1 ≤ d := by
      intro p hp s hs
      have hne : p.1 ≠ p.2 := (Finset.mem_offDiag.1 hp).2.2
      have hlift : NShatters (F : Set (Fin (n + 1) → Fin K))
          (insert 0 (s.map (Fin.succEmb n))) :=
        NShatters.of_pair hne (fun h' hh' => by simpa [Fp] using hh') hs
      have hcard : (insert 0 (s.map (Fin.succEmb n))).card = s.card + 1 := by
        rw [Finset.card_insert_of_notMem, Finset.card_map]
        simp [Fin.succ_ne_zero]
      rw [← hcard]
      exact hF _ hlift
    cases d with
    | zero =>
      have hempty : ∀ p ∈ (Finset.univ : Finset (Fin K)).offDiag, (Fp p).card = 0 := by
        intro p hp
        rw [Finset.card_eq_zero]
        by_contra hne
        obtain ⟨h', hh'⟩ := Finset.nonempty_iff_ne_empty.2 hne
        have := hpair p hp ∅
          ⟨fun _ => 0, fun _ => 0, by simp, fun _ => ⟨h', hh', by simp⟩⟩
        simp at this
      rw [Finset.sum_eq_zero hempty, add_zero] at hsplit
      simpa using hsplit.trans h0
    | succ d' =>
      have hle : ∀ p ∈ (Finset.univ : Finset (Fin K)).offDiag,
          (Fp p).card ≤ ((n + 1) * K ^ 2) ^ d' := fun p hp =>
        ih d' (Fp p) fun s hs => Nat.le_of_succ_le_succ (hpair p hp s hs)
      have hsum : ∑ p ∈ (Finset.univ : Finset (Fin K)).offDiag, (Fp p).card
          ≤ K ^ 2 * ((n + 1) * K ^ 2) ^ d' := by
        calc ∑ p ∈ (Finset.univ : Finset (Fin K)).offDiag, (Fp p).card
            ≤ ∑ _p ∈ (Finset.univ : Finset (Fin K)).offDiag, ((n + 1) * K ^ 2) ^ d' :=
              Finset.sum_le_sum hle
          _ = (Finset.univ : Finset (Fin K)).offDiag.card * ((n + 1) * K ^ 2) ^ d' := by
              rw [Finset.sum_const, smul_eq_mul]
          _ ≤ K ^ 2 * ((n + 1) * K ^ 2) ^ d' := by
              refine Nat.mul_le_mul_right _ ?_
              rw [Finset.offDiag_card, Finset.card_univ, Fintype.card_fin, sq]
              exact Nat.sub_le _ _
      calc F.card ≤ ((n + 1) * K ^ 2) ^ (d' + 1) + K ^ 2 * ((n + 1) * K ^ 2) ^ d' :=
            hsplit.trans (add_le_add h0 hsum)
        _ = ((n + 1) * K ^ 2) ^ d' * ((n + 1 + 1) * K ^ 2) := by ring
        _ ≤ ((n + 1 + 1) * K ^ 2) ^ d' * ((n + 1 + 1) * K ^ 2) := by
            refine Nat.mul_le_mul_right _ (Nat.pow_le_pow_left ?_ _)
            exact Nat.mul_le_mul_right _ (Nat.le_succ _)
        _ = ((n + 1 + 1) * K ^ 2) ^ (d' + 1) := by ring

/-! ### Classes read off a decision rule pointwise -/

/-- A set of `n` examples shattered by a class of binary functions of the form
`z ↦ ψ(z, δ(φ(z)))`, `δ ∈ H`, forces `2ⁿ` distinct restrictions of `H` to the
points `φ(z)`, of which Natarajan's lemma allows at most `((n + 1) K²)^d`. -/
theorem two_pow_card_le_of_shatters [NeZero K] {X Z : Type*} (H : Set (X → Fin K))
    (d : ℕ) (hd : natDim H ≤ (d : ℕ∞)) (φ : Z → X) (ψ : Z → Fin K → Bool)
    (s : Finset Z)
    (hs : Shatters {g : Z → Bool | ∃ δ ∈ H, g = fun z => ψ z (δ (φ z))} s) :
    2 ^ s.card ≤ ((s.card + 1) * K ^ 2) ^ d := by
  classical
  -- The points of the examples, enumerated.
  let P : Finset X := s.image φ
  let e : Fin P.card → X := fun i => (P.equivFin.symm i : X)
  have he : Function.Injective e := fun i j hij =>
    P.equivFin.symm.injective (Subtype.ext hij)
  have hrange : ∀ x ∈ P, ∃ i, e i = x := fun x hx =>
    ⟨P.equivFin ⟨x, hx⟩, by simp [e]⟩
  -- The restrictions of the class to the points.
  let F : Finset (Fin P.card → Fin K) :=
    Finset.univ.filter fun h' => ∃ δ ∈ H, δ ∘ e = h'
  have hFdim : ∀ t : Finset (Fin P.card),
      NShatters (F : Set (Fin P.card → Fin K)) t → t.card ≤ d := by
    rintro t ⟨f, g, hfg, hreal⟩
    have hN : NShatters H (t.map ⟨e, he⟩) := by
      refine ⟨Function.extend e f fun _ => 0, Function.extend e g fun _ => 0, ?_,
        fun b => ?_⟩
      · intro x hx
        obtain ⟨i, hi, rfl⟩ := Finset.mem_map.1 hx
        simpa [he.extend_apply] using hfg i hi
      · obtain ⟨h', hh', hval⟩ := hreal (b ∘ e)
        obtain ⟨δ, hδ, rfl⟩ : ∃ δ ∈ H, δ ∘ e = h' := by simpa [F] using hh'
        refine ⟨δ, hδ, fun x hx => ?_⟩
        obtain ⟨i, hi, rfl⟩ := Finset.mem_map.1 hx
        simpa [he.extend_apply] using hval i hi
    have h1 := (le_natDim hN).trans hd
    rw [Finset.card_map] at h1
    exact_mod_cast h1
  have hcard := card_le_of_nshatters_le P.card d F hFdim
  -- One rule per labeling of the examples; distinct labelings, distinct restrictions.
  have hex : ∀ T : Finset Z, ∃ δ ∈ H, ∀ z ∈ s, ψ z (δ (φ z)) = decide (z ∈ T) := by
    intro T
    obtain ⟨g, ⟨δ, hδ, rfl⟩, hval⟩ := hs fun z => decide (z ∈ T)
    exact ⟨δ, hδ, hval⟩
  choose δ hδH hδval using hex
  have hinj : Set.InjOn (fun T => δ T ∘ e) (s.powerset : Set (Finset Z)) := by
    intro T hT T' hT' hTT'
    have hTT'' : δ T ∘ e = δ T' ∘ e := hTT'
    ext z
    by_cases hz : z ∈ s
    · obtain ⟨i, hi⟩ := hrange (φ z) (Finset.mem_image_of_mem φ hz)
      have h1 := hδval T z hz
      have h2 := hδval T' z hz
      have h3 : δ T (φ z) = δ T' (φ z) := by
        rw [← hi]
        exact congrFun hTT'' i
      rw [h3] at h1
      simpa using h1.symm.trans h2
    · have h1 : z ∉ T := fun h => hz (Finset.mem_powerset.1 (Finset.mem_coe.1 hT) h)
      have h2 : z ∉ T' := fun h => hz (Finset.mem_powerset.1 (Finset.mem_coe.1 hT') h)
      simp [h1, h2]
  have hpow : s.powerset.card ≤ F.card :=
    Finset.card_le_card_of_injOn (fun T => δ T ∘ e)
      (fun T _ => by simpa [F] using ⟨δ T, hδH T, rfl⟩) hinj
  rw [Finset.card_powerset] at hpow
  have hP : P.card ≤ s.card := Finset.card_image_le
  calc 2 ^ s.card ≤ F.card := hpow
    _ ≤ ((P.card + 1) * K ^ 2) ^ d := hcard
    _ ≤ ((s.card + 1) * K ^ 2) ^ d :=
        Nat.pow_le_pow_left (Nat.mul_le_mul_right _ (by omega)) _

/-! ### Solving `2ⁿ ≤ ((n + 1) K²)^d` for `n` -/

theorem two_mul_le_two_pow (s : ℕ) : 2 * s ≤ 2 ^ s := by
  cases s with
  | zero => simp
  | succ s =>
    have := Nat.lt_two_pow_self (n := s)
    rw [pow_succ]
    omega

/-- If `2ⁿ ≤ ((n + 1) K²)^d` with `d ≥ 1`, then `n < 2 d (⌈log₂(d K²)⌉ + 1)`. -/
theorem lt_of_two_pow_le {n d K : ℕ} (hd : 0 < d)
    (h : 2 ^ n ≤ ((n + 1) * K ^ 2) ^ d) :
    n < d * (2 * Nat.clog 2 (d * K ^ 2) + 2) := by
  by_contra hcon
  rw [not_lt] at hcon
  obtain ⟨ℓ, hℓ⟩ : ∃ ℓ, ℓ = Nat.clog 2 (d * K ^ 2) := ⟨_, rfl⟩
  rw [← hℓ] at hcon
  have hdK : d * K ^ 2 ≤ 2 ^ ℓ := by
    rw [hℓ]
    exact Nat.le_pow_clog (by norm_num) _
  obtain ⟨t, ht⟩ : ∃ t, t = n / d := ⟨_, rfl⟩
  have ht1 : d * t ≤ n := by
    rw [ht]
    exact Nat.mul_div_le n d
  have ht2 : n < d * (t + 1) := by
    rw [ht]
    exact Nat.lt_mul_div_succ n hd
  have htT : 2 * ℓ + 2 ≤ t := by
    rw [ht, Nat.le_div_iff_mul_le hd, mul_comm]
    exact hcon
  -- With `t = s + ℓ`: `t + 1 < 2^s` and `d K² ≤ 2^ℓ`.
  obtain ⟨s, hst⟩ : ∃ s, t = s + ℓ := ⟨t - ℓ, by omega⟩
  have h2s := two_mul_le_two_pow s
  have hkey : (t + 1) * (d * K ^ 2) < 2 ^ t := by
    calc (t + 1) * (d * K ^ 2) ≤ (t + 1) * 2 ^ ℓ := Nat.mul_le_mul_left _ hdK
      _ < 2 ^ s * 2 ^ ℓ := Nat.mul_lt_mul_of_pos_right (by omega) (by positivity)
      _ = 2 ^ t := by rw [hst, pow_add]
  have h1 : ((n + 1) * K ^ 2) ^ d ≤ ((t + 1) * (d * K ^ 2)) ^ d := by
    refine Nat.pow_le_pow_left ?_ _
    calc (n + 1) * K ^ 2 ≤ (d * (t + 1)) * K ^ 2 := Nat.mul_le_mul_right _ ht2
      _ = (t + 1) * (d * K ^ 2) := by ring
  have h2 : ((t + 1) * (d * K ^ 2)) ^ d < (2 ^ t) ^ d :=
    Nat.pow_lt_pow_left hkey hd.ne'
  have h3 : (2 ^ t) ^ d ≤ 2 ^ n := by
    rw [← pow_mul, mul_comm]
    exact Nat.pow_le_pow_right (by norm_num) ht1
  exact absurd (h.trans_lt (h1.trans_lt (h2.trans_le h3))) (lt_irrefl _)

/-- The bound on the VC dimension of a binary class read off a class of rules
of Natarajan dimension at most `d` with `K` labels:
`2 d (⌈log₂(d K²)⌉ + 1)`. -/
def levelDim (d K : ℕ) : ℕ := 2 * d * (Nat.clog 2 (d * K ^ 2) + 1)

/-- **From the Natarajan dimension to the VC dimension.** Let `H` be a class
of rules with `K` labels and `d_Nat(H) ≤ d`, and let `G` be the class of binary
functions `z ↦ ψ(z, δ(φ(z)))`, `δ ∈ H`, for any `φ` and `ψ`. Then
`d_VC(G) ≤ 2 d (⌈log₂(d K²)⌉ + 1)`. -/
theorem vcDim_pointwise_le [NeZero K] {X Z : Type*} (H : Set (X → Fin K)) (d : ℕ)
    (hd : natDim H ≤ (d : ℕ∞)) (φ : Z → X) (ψ : Z → Fin K → Bool) :
    vcDim {g : Z → Bool | ∃ δ ∈ H, g = fun z => ψ z (δ (φ z))}
      ≤ (levelDim d K : ℕ∞) := by
  refine vcDim_le fun s hs => ?_
  have h := two_pow_card_le_of_shatters H d hd φ ψ s hs
  have hle : s.card ≤ levelDim d K := by
    rcases Nat.eq_zero_or_pos d with rfl | hdpos
    · rw [pow_zero] at h
      have h0 : s.card = 0 := by
        by_contra hne
        have := Nat.one_lt_two_pow hne
        omega
      rw [h0]
      exact Nat.zero_le _
    · have hlt := lt_of_two_pow_le hdpos h
      have e : levelDim d K = d * (2 * Nat.clog 2 (d * K ^ 2) + 2) := by
        unfold levelDim
        ring
      rw [e]
      exact hlt.le
  exact_mod_cast hle

/-- The graph dimension: the VC dimension of the class of correctness
indicators `(x, y) ↦ 1[δ(x) = y]` is at most `2 d (⌈log₂(d K²)⌉ + 1)`. -/
theorem vcDim_graph_le [NeZero K] {X : Type*} (H : Set (X → Fin K)) (d : ℕ)
    (hd : natDim H ≤ (d : ℕ∞)) :
    vcDim {g : X × Fin K → Bool | ∃ δ ∈ H, g = fun z => decide (δ z.1 = z.2)}
      ≤ (levelDim d K : ℕ∞) :=
  vcDim_pointwise_le H d hd Prod.fst fun z k => decide (k = z.2)

end ThresholdOpt.Capacity
