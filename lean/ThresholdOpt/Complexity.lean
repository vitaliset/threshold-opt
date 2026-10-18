import Mathlib.Algebra.Module.Submodule.Union
import ThresholdOpt.Capacity
import ThresholdOpt.Voronoi

/-!
# Natarajan-dimension bounds (Chapter 6, Appendix C)

The decision classes of Chapters 5 and 6 as sets of functions on the simplex,
and their capacity bounds.

Every upper bound here is an application of `lem:linear_embedding`
(`Capacity.natDim_le_of_linear_scores`, proved in
[`Capacity.lean`](Capacity.lean)): each class is a family of `argmin` rules
whose scores are affine in the parameters. Nothing is imported.

The domain is the simplex subtype rather than all of `Fin K → ℝ`. This matters:
the parallel-shift rule coincides with a full-matrix rule only where
`∑ p_i = 1`, since the offset `v_k` is absorbed as `v_k · ∑ p_i`.

The lower bound `d_Nat(H_PS) ≥ K - 1` of `thm:ps_complexity` is proved by a
construction different from the one in the text: `K - 1` points on one line
through the barycenter, in a direction along which the `K` anchor risks have
pairwise distinct slopes, instead of one point per edge of a spanning tree of
the facet-adjacency graph of the Bayes regions.
-/

namespace ThresholdOpt.Complexity

open ThresholdOpt ThresholdOpt.Capacity ThresholdOpt.Multiclass
open ThresholdOpt.Partition ThresholdOpt.Voronoi

variable {K : ℕ} [NeZero K]

/-- The probability simplex as a type, the domain of every decision rule. -/
def Simplex (K : ℕ) : Type := {p : Fin K → ℝ // p ∈ stdSimplex ℝ (Fin K)}

/-- The parallel-shift class `H_PS` (`eq:hps_def`), with `v` ranging over all
of `ℝ^K`. The text imposes the gauge `∑ v_k = 0`; `HPSGauged_eq` shows that the
gauge does not change the class. -/
def HPS (C : CostMatrix K) : Set (Simplex K → Fin K) :=
  {δ | ∃ v : Fin K → ℝ, δ = fun p => psRule C v p.val}

/-- `H_PS` exactly as written in `eq:hps_def`, with the mean-zero gauge. -/
def HPSGauged (C : CostMatrix K) : Set (Simplex K → Fin K) :=
  {δ | ∃ v : Fin K → ℝ, ∑ k, v k = 0 ∧ δ = fun p => psRule C v p.val}

/-- The linear-scaling class `H_Sc` exactly as written in `eq:hsc_def`:
positive weights with the gauge `β_0 = 1`.

Positivity is part of the class and not a convenience. With `β` ranging over
all of `ℝ^K` the class is strictly larger and the bound `K - 1` of
`thm:sc_complexity` fails for it: at `K = 2` under `0/1` loss the rule selects
class `0` iff `(β_0 + β_1) p_0 ≥ β_1`, a threshold on `p_0` of either
orientation, which shatters two points. -/
def HSc (C : CostMatrix K) : Set (Simplex K → Fin K) :=
  {δ | ∃ β : Fin K → ℝ, (∀ k, 0 < β k) ∧ β 0 = 1
    ∧ δ = fun p => scRule C β p.val}

/-- The linear-scaling class before the gauge is fixed: positive weights. -/
def HScPos (C : CostMatrix K) : Set (Simplex K → Fin K) :=
  {δ | ∃ β : Fin K → ℝ, (∀ k, 0 < β k) ∧ δ = fun p => scRule C β p.val}

/-- The full-matrix class `H_Full` (`eq:hfull_def`), `K²` parameters. -/
def HFull : Set (Simplex K → Fin K) :=
  {δ | ∃ A : CostMatrix K, δ = fun p => lexArgmin (condRisk A p.val)}

/-- The unconstrained Voronoi class `H_Vor`. -/
def HVor : Set (Simplex K → Fin K) :=
  {δ | ∃ μ : Prototypes K, δ = fun p => voronoiRule μ p.val}

/-- `H_Vor` with the `K` prototypes required to be distinct, as `06:14` words
it. `HVorDistinct_eq_HFull` shows the requirement does not change the class. -/
def HVorDistinct : Set (Simplex K → Fin K) :=
  {δ | ∃ μ : Prototypes K, Function.Injective μ
    ∧ δ = fun p => voronoiRule μ p.val}

/-- The simplex-restricted Voronoi class `H_Vor^Δ`, the class the optimizer of
Chapter 6 actually searches: prototypes are themselves probability vectors. -/
def HVorSimplex : Set (Simplex K → Fin K) :=
  {δ | ∃ μ : Prototypes K, (∀ k, μ k ∈ stdSimplex ℝ (Fin K))
    ∧ δ = fun p => voronoiRule μ p.val}

/-! ### Containments

The two containments in `H_Full` are what make `thm:full_complexity`'s lower
bound `≥ K - 1` follow from `thm:ps_complexity` through `lem:nat_monotone`. -/

/-- `H_PS ⊆ H_Full`: a parallel shift is the full-matrix rule for
`A_{ij} = C_{ij} + v_j`, using `∑_i p_i = 1` to absorb the offset. -/
theorem HPS_subset_HFull (C : CostMatrix K) : HPS C ⊆ HFull := by
  rintro δ ⟨v, rfl⟩
  refine ⟨Matrix.of fun i j => C i j + v j, ?_⟩
  funext p
  have hsum : ∑ i, p.val i = 1 := p.property.2
  have hscore : condRisk (Matrix.of fun i j => C i j + v j) p.val
      = psScore C v p.val := by
    funext k
    have hterm : ∀ i : Fin K,
        (Matrix.of fun i j => C i j + v j : CostMatrix K) i k * p.val i
          = C i k * p.val i + v k * p.val i := by
      intro i
      simp only [Matrix.of_apply]
      ring
    simp only [condRisk, psScore]
    rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_add_distrib,
      ← Finset.mul_sum, hsum, mul_one]
  rw [hscore]
  rfl

/-- `H_Sc ⊆ H_Full`: linear scaling is the full-matrix rule for
`A_{ij} = C_{ij} β_i`. -/
theorem HSc_subset_HFull (C : CostMatrix K) : HSc C ⊆ HFull := by
  rintro δ ⟨β, -, -, rfl⟩
  refine ⟨Matrix.of fun i j => C i j * β i, ?_⟩
  funext p
  have hscore : condRisk (Matrix.of fun i j => C i j * β i) p.val
      = scScore C β p.val := by
    funext k
    have hterm : ∀ i : Fin K,
        (Matrix.of fun i j => C i j * β i : CostMatrix K) i k * p.val i
          = C i k * (β i * p.val i) := by
      intro i
      simp only [Matrix.of_apply]
      ring
    simp only [condRisk, scScore]
    exact Finset.sum_congr rfl fun i _ => hterm i
  rw [hscore]
  rfl

/-! ### The gauges do not change the classes -/

/-- The mean-zero gauge of `eq:hps_def` is free: every parallel-shift rule is
the rule of a mean-zero offset vector. -/
theorem HPSGauged_eq (C : CostMatrix K) : HPSGauged C = HPS C := by
  ext δ
  constructor
  · rintro ⟨v, -, rfl⟩
    exact ⟨v, rfl⟩
  · rintro ⟨v, rfl⟩
    have hK : (K : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (NeZero.ne K)
    refine ⟨fun k => v k + -(∑ j, v j) / K, ?_, ?_⟩
    · rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul]
      field_simp
      ring
    · funext p
      exact (psRule_add_const C v _ p.val).symm

/-- The gauge `β_0 = 1` of `eq:hsc_def` is free among positive weights. -/
theorem HScPos_eq (C : CostMatrix K) : HScPos C = HSc C := by
  ext δ
  constructor
  · rintro ⟨β, hβ, rfl⟩
    have h0 : 0 < β 0 := hβ 0
    refine ⟨fun k => (β 0)⁻¹ * β k, fun k => mul_pos (inv_pos.2 h0) (hβ k),
      inv_mul_cancel₀ h0.ne', ?_⟩
    funext p
    have hscore : scScore C (fun k => (β 0)⁻¹ * β k) p.val
        = fun i => (β 0)⁻¹ * scScore C β p.val i := by
      funext i
      simp only [scScore, condRisk, Finset.mul_sum]
      exact Finset.sum_congr rfl fun k _ => by ring
    show scRule C β p.val = scRule C (fun k => (β 0)⁻¹ * β k) p.val
    rw [scRule, scRule, hscore, lexArgmin_const_mul (inv_pos.2 h0)]
  · rintro ⟨β, hβ, -, rfl⟩
    exact ⟨β, hβ, rfl⟩

/-! ### Capacity upper bounds for the two vector-parameterized classes

Both classes have `K` parameters and one gauge freedom. Writing `K = n + 1` and
eliminating the parameter of class `0` leaves a linear-score class in `n`
parameters, to which `lem:linear_embedding` applies. -/

/-- `d_Nat(H_PS) ≤ n` for `K = n + 1` classes. The offsets are normalized to
`v_0 = 0`; the remaining `n` are the parameters, with feature map
`Φ_k(p) = e_k` and parameter-free term `g_k(p) = 𝓛_k(p)`. -/
theorem natDim_HPS_succ_le {n : ℕ} (C : CostMatrix (n + 1)) :
    natDim (HPS C) ≤ (n : ℕ∞) := by
  refine natDim_le_of_linear_scores (HPS C)
    (Φ := fun k _ (i : Fin n) => if i.succ = k then (1 : ℝ) else 0)
    (g := fun k p => condRisk C p.val k) ?_
  rintro δ ⟨v, rfl⟩
  refine ⟨fun i => v i.succ - v 0, fun p => ?_⟩
  have hscore : (fun k => (∑ i : Fin n,
        (v i.succ - v 0) * (if i.succ = k then (1 : ℝ) else 0))
      + condRisk C p.val k) = fun k => psScore C v p.val k + -v 0 := by
    funext k
    have hsel : ∑ i : Fin n,
        (v i.succ - v 0) * (if i.succ = k then (1 : ℝ) else 0)
          = v k - v 0 := by
      refine Fin.cases ?_ (fun j => ?_) k
      · simp [Fin.succ_ne_zero]
      · simp only [Fin.succ_inj, mul_ite, mul_one, mul_zero]
        rw [Finset.sum_ite_eq' Finset.univ j fun i => v i.succ - v 0]
        simp
    rw [hsel, psScore]
    ring
  show psRule C v p.val = _
  rw [hscore, lexArgmin_add_const]
  rfl

/-- **`d_Nat(H_PS) ≤ K - 1`** (`thm:ps_complexity`, upper bound), for every
cost matrix. -/
theorem natDim_HPS_le (C : CostMatrix K) :
    natDim (HPS C) ≤ ((K - 1 : ℕ) : ℕ∞) := by
  obtain ⟨n, rfl⟩ : ∃ n, K = n + 1 :=
    ⟨K - 1, (Nat.succ_pred_eq_of_pos (Nat.pos_of_ne_zero (NeZero.ne K))).symm⟩
  exact natDim_HPS_succ_le C

/-- `d_Nat(H_Sc) ≤ n` for `K = n + 1` classes. With `β_0 = 1` the score is
`C_{0i} p_0 + ∑_{k ≥ 1} β_k C_{ki} p_k`, linear in the remaining `n` weights
with feature map `Φ_i(p)_k = C_{ki} p_k` and parameter-free term
`g_i(p) = C_{0i} p_0`, as in the text. -/
theorem natDim_HSc_succ_le {n : ℕ} (C : CostMatrix (n + 1)) :
    natDim (HSc C) ≤ (n : ℕ∞) := by
  refine natDim_le_of_linear_scores (HSc C)
    (Φ := fun i p (k : Fin n) => C k.succ i * p.val k.succ)
    (g := fun i p => C 0 i * p.val 0) ?_
  rintro δ ⟨β, -, hβ0, rfl⟩
  refine ⟨fun k => β k.succ, fun p => ?_⟩
  have hscore : (fun i => (∑ k : Fin n,
        β k.succ * (C k.succ i * p.val k.succ)) + C 0 i * p.val 0)
      = scScore C β p.val := by
    funext i
    simp only [scScore, condRisk]
    rw [Fin.sum_univ_succ, hβ0, one_mul, add_comm]
    congr 1
    exact Finset.sum_congr rfl fun k _ => by ring
  show scRule C β p.val = _
  rw [scRule, hscore]

/-- **`d_Nat(H_Sc) ≤ K - 1`** (`thm:sc_complexity`), for every cost matrix and
for the class exactly as defined in `eq:hsc_def`. -/
theorem natDim_HSc_le (C : CostMatrix K) :
    natDim (HSc C) ≤ ((K - 1 : ℕ) : ℕ∞) := by
  obtain ⟨n, rfl⟩ : ∃ n, K = n + 1 :=
    ⟨K - 1, (Nat.succ_pred_eq_of_pos (Nat.pos_of_ne_zero (NeZero.ne K))).symm⟩
  exact natDim_HSc_succ_le C

/-! ### The full-matrix and Voronoi bounds

Both classes are linear-score classes in a *flat* parameter vector, so the same
embedding applies; what they need that `H_PS` and `H_Sc` do not is an explicit
index bijection between the structured parameters (a `K × K` matrix, or `K`
prototypes plus `K` biases) and `Fin W`. Mathlib supplies both halves,
`finProdFinEquiv` and `finSumFinEquiv`. -/

/-- `Fin K × Fin K ≃ Fin (K * K)`: the flat index of a matrix entry. -/
def matIdx (K : ℕ) : Fin K × Fin K ≃ Fin (K * K) := finProdFinEquiv

/-- `(Fin K × Fin K) ⊕ Fin K ≃ Fin (K * K + K)`: the flat index of a prototype
coordinate or of a per-class bias. -/
def vorIdx (K : ℕ) : (Fin K × Fin K) ⊕ Fin K ≃ Fin (K * K + K) :=
  (Equiv.sumCongr (finProdFinEquiv (m := K) (n := K)) (Equiv.refl (Fin K))).trans
    finSumFinEquiv

/-- **`d_Nat(H_Full) ≤ K²`** (`thm:full_complexity`).

The full-matrix score `𝓛_k(p) = ∑_i A_{ik} p_i` is linear in the `K²` entries
of `A` with no parameter-free term. The feature map reads off the flat index:
the coordinate for entry `(i, j)` contributes `p_i` to the score of class `j`
and nothing to the others. -/
theorem natDim_HFull_le : natDim (HFull (K := K)) ≤ ((K ^ 2 : ℕ) : ℕ∞) := by
  rw [pow_two]
  refine natDim_le_of_linear_scores HFull
    (Φ := fun k p idx =>
      if ((matIdx K).symm idx).2 = k then p.val ((matIdx K).symm idx).1 else 0)
    (g := fun _ _ => 0) ?_
  rintro δ ⟨A, rfl⟩
  refine ⟨fun idx => A ((matIdx K).symm idx).1 ((matIdx K).symm idx).2,
    fun p => ?_⟩
  have hscore : ∀ k,
      (∑ idx, A ((matIdx K).symm idx).1 ((matIdx K).symm idx).2
          * (if ((matIdx K).symm idx).2 = k then p.val ((matIdx K).symm idx).1
             else 0)) + 0
        = condRisk A p.val k := by
    intro k
    rw [add_zero, ← Equiv.sum_comp (matIdx K)]
    simp only [Equiv.symm_apply_apply, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [mul_ite, mul_zero]
    rw [Finset.sum_ite_eq' Finset.univ k fun j => A i j * p.val i]
    simp
  show lexArgmin (condRisk A p.val) = _
  exact lexArgmin_congr fun k => (hscore k).symm

omit [NeZero K] in
/-- The Voronoi score in the lifted parameters: dropping the `k`-independent
`‖p‖²` from `d²(p, μ_k)` leaves `-2 μ_k·p + ‖μ_k‖²`, which is linear in the
pair `(μ_k, ‖μ_k‖²)`. -/
theorem dsq_eq_lifted (μ : Prototypes K) (p : Fin K → ℝ) (k : Fin K) :
    dsq p (μ k)
      = ((∑ i, μ k i * (-2 * p i)) + (∑ i, μ k i ^ 2)) + ∑ i, p i ^ 2 := by
  simp only [dsq, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- **`d_Nat(H_Vor) ≤ K² + K`** (`thm:voronoi_complexity`, unconstrained half).

The `K²` prototype coordinates and the `K` squared-norm biases are treated as
free parameters of a linear-score class, which is the "coupled case" of
`lem:linear_embedding`: the true class is a subclass of the free one, since
`b_k = ‖μ_k‖²` is one particular choice of bias. -/
theorem natDim_HVor_le :
    natDim (HVor (K := K)) ≤ ((K ^ 2 + K : ℕ) : ℕ∞) := by
  rw [pow_two]
  refine natDim_le_of_linear_scores HVor
    (Φ := fun k p idx =>
      match (vorIdx K).symm idx with
      | Sum.inl q => if q.1 = k then -2 * p.val q.2 else 0
      | Sum.inr k' => if k' = k then 1 else 0)
    (g := fun _ p => ∑ i, p.val i ^ 2) ?_
  rintro δ ⟨μ, rfl⟩
  refine ⟨fun idx =>
    match (vorIdx K).symm idx with
    | Sum.inl q => μ q.1 q.2
    | Sum.inr k' => ∑ i, μ k' i ^ 2, fun p => ?_⟩
  have hscore : ∀ k,
      (∑ idx,
        (match (vorIdx K).symm idx with
          | Sum.inl q => μ q.1 q.2
          | Sum.inr k' => ∑ i, μ k' i ^ 2)
        * (match (vorIdx K).symm idx with
          | Sum.inl q => if q.1 = k then -2 * p.val q.2 else 0
          | Sum.inr k' => if k' = k then 1 else 0))
        + ∑ i, p.val i ^ 2
      = dsq p.val (μ k) := by
    intro k
    rw [← Equiv.sum_comp (vorIdx K)]
    simp only [Equiv.symm_apply_apply, Fintype.sum_sum_type,
      Fintype.sum_prod_type]
    have hleft : ∑ k' : Fin K, ∑ i : Fin K,
        μ k' i * (if k' = k then -2 * p.val i else 0)
          = ∑ i, μ k i * (-2 * p.val i) := by
      have hterm : ∀ k' : Fin K,
          (∑ i : Fin K, μ k' i * (if k' = k then -2 * p.val i else 0))
            = if k' = k then (∑ i, μ k i * (-2 * p.val i)) else 0 := by
        intro k'
        by_cases h : k' = k
        · subst h; simp
        · simp [h]
      rw [Finset.sum_congr rfl fun k' _ => hterm k',
        Finset.sum_ite_eq' Finset.univ k fun _ => ∑ i, μ k i * (-2 * p.val i),
        if_pos (Finset.mem_univ k)]
    have hright : ∑ k' : Fin K,
        (∑ i, μ k' i ^ 2) * (if k' = k then (1 : ℝ) else 0)
          = ∑ i, μ k i ^ 2 := by
      simp only [mul_ite, mul_one, mul_zero]
      rw [Finset.sum_ite_eq' Finset.univ k fun k' => ∑ i, μ k' i ^ 2,
        if_pos (Finset.mem_univ k)]
    rw [hleft, hright, dsq_eq_lifted]
  show voronoiRule μ p.val = _
  exact lexArgmin_congr fun k => (hscore k).symm

/-! ### The simplex-restricted Voronoi class

`H_Vor^Δ` drops one coordinate per prototype: on the simplex the last one is
determined by the others. Writing `K = n + 1` and eliminating
`μ_k(last) = 1 - ∑_{j<n} μ_k(j)` turns the score into

`-2 ∑_j μ_k(j) (p_j - p_last) - 2 p_last + b_k`,

linear in the `K·n` retained coordinates with the `p`-only term `-2 p_last`
absorbed into `g_k`. With the `K` biases that is `K·n + K = K²` parameters. -/

/-- `(Fin (n+1) × Fin n) ⊕ Fin (n+1) ≃ Fin ((n+1) * n + (n+1))`: the flat index
of a retained prototype coordinate or of a per-class bias. -/
def vorSimplexIdx (n : ℕ) :
    (Fin (n + 1) × Fin n) ⊕ Fin (n + 1) ≃ Fin ((n + 1) * n + (n + 1)) :=
  (Equiv.sumCongr (finProdFinEquiv (m := n + 1) (n := n))
    (Equiv.refl (Fin (n + 1)))).trans finSumFinEquiv

/-- The parameter count of the reduced parameterization is `K²`. -/
theorem vorSimplex_card (n : ℕ) : (n + 1) * n + (n + 1) = (n + 1) ^ 2 := by
  ring

/-- **`d_Nat(H_Vor^Δ) ≤ K²`** (`thm:voronoi_complexity`, restricted half). -/
theorem natDim_HVorSimplex_le {n : ℕ} :
    natDim (HVorSimplex (K := n + 1)) ≤ (((n + 1) ^ 2 : ℕ) : ℕ∞) := by
  rw [← vorSimplex_card n]
  refine natDim_le_of_linear_scores HVorSimplex
    (Φ := fun k p idx =>
      match (vorSimplexIdx n).symm idx with
      | Sum.inl q =>
        if q.1 = k then -2 * (p.val q.2.castSucc - p.val (Fin.last n)) else 0
      | Sum.inr k' => if k' = k then 1 else 0)
    (g := fun _ p => -2 * p.val (Fin.last n) + ∑ i, p.val i ^ 2) ?_
  rintro δ ⟨μ, hμ, rfl⟩
  refine ⟨fun idx =>
    match (vorSimplexIdx n).symm idx with
    | Sum.inl q => μ q.1 q.2.castSucc
    | Sum.inr k' => ∑ i, μ k' i ^ 2, fun p => ?_⟩
  have hscore : ∀ k,
      (∑ idx,
        (match (vorSimplexIdx n).symm idx with
          | Sum.inl q => μ q.1 q.2.castSucc
          | Sum.inr k' => ∑ i, μ k' i ^ 2)
        * (match (vorSimplexIdx n).symm idx with
          | Sum.inl q =>
            if q.1 = k then -2 * (p.val q.2.castSucc - p.val (Fin.last n))
            else 0
          | Sum.inr k' => if k' = k then 1 else 0))
        + (-2 * p.val (Fin.last n) + ∑ i, p.val i ^ 2)
      = dsq p.val (μ k) := by
    intro k
    rw [← Equiv.sum_comp (vorSimplexIdx n)]
    simp only [Equiv.symm_apply_apply, Fintype.sum_sum_type,
      Fintype.sum_prod_type]
    have hleft : ∑ k' : Fin (n + 1), ∑ j : Fin n,
        μ k' j.castSucc
          * (if k' = k then -2 * (p.val j.castSucc - p.val (Fin.last n))
             else 0)
          = ∑ j : Fin n,
              μ k j.castSucc * (-2 * (p.val j.castSucc - p.val (Fin.last n))) := by
      have hterm : ∀ k' : Fin (n + 1),
          (∑ j : Fin n, μ k' j.castSucc
            * (if k' = k then -2 * (p.val j.castSucc - p.val (Fin.last n))
               else 0))
            = if k' = k then
                (∑ j : Fin n, μ k j.castSucc
                  * (-2 * (p.val j.castSucc - p.val (Fin.last n))))
              else 0 := by
        intro k'
        by_cases h : k' = k
        · subst h; simp
        · simp [h]
      rw [Finset.sum_congr rfl fun k' _ => hterm k',
        Finset.sum_ite_eq' Finset.univ k fun _ =>
          ∑ j : Fin n, μ k j.castSucc
            * (-2 * (p.val j.castSucc - p.val (Fin.last n))),
        if_pos (Finset.mem_univ k)]
    have hright : ∑ k' : Fin (n + 1),
        (∑ i, μ k' i ^ 2) * (if k' = k then (1 : ℝ) else 0)
          = ∑ i, μ k i ^ 2 := by
      simp only [mul_ite, mul_one, mul_zero]
      rw [Finset.sum_ite_eq' Finset.univ k fun k' => ∑ i, μ k' i ^ 2,
        if_pos (Finset.mem_univ k)]
    rw [hleft, hright, dsq_eq_lifted]
    -- Eliminate the last prototype coordinate using `∑_i μ_k i = 1`.
    have hsum : ∑ i, μ k i = 1 := (hμ k).2
    rw [Fin.sum_univ_castSucc (f := fun i => μ k i)] at hsum
    have hlin : ∑ i, μ k i * (-2 * p.val i)
        = (∑ j : Fin n, μ k j.castSucc
            * (-2 * (p.val j.castSucc - p.val (Fin.last n))))
          + -2 * p.val (Fin.last n) := by
      rw [Fin.sum_univ_castSucc
        (f := fun i => μ k i * (-2 * p.val i))]
      have hexp : ∀ j : Fin n,
          μ k j.castSucc * (-2 * p.val j.castSucc)
            = μ k j.castSucc
                * (-2 * (p.val j.castSucc - p.val (Fin.last n)))
              + μ k j.castSucc * (-2 * p.val (Fin.last n)) := by
        intro j; ring
      rw [Finset.sum_congr rfl fun j _ => hexp j, Finset.sum_add_distrib,
        ← Finset.sum_mul]
      have : ∑ j : Fin n, μ k j.castSucc = 1 - μ k (Fin.last n) := by
        linarith [hsum]
      rw [this]
      ring
    rw [hlin]
    ring
  show voronoiRule μ p.val = _
  exact lexArgmin_congr fun k => (hscore k).symm

/-- **Embedding bounds for the Voronoi classes** (`rem:voronoi_embedding_bounds`).

`d_Nat(H_Vor) ≤ K² + K` and `d_Nat(H_Vor^Δ) ≤ K²`, the additive `K` for the
unconstrained class coming from the per-class bias terms and the saving for the
restricted class from the simplex constraint, which removes one coordinate per
prototype. The sharper statement of `thm:voronoi_complexity` is
`voronoi_complexity_quadratic`. -/
@[thesis "rem:voronoi_embedding_bounds"]
theorem voronoi_complexity :
    natDim (HVor (K := K)) ≤ ((K ^ 2 + K : ℕ) : ℕ∞) ∧
    natDim (HVorSimplex (K := K)) ≤ ((K ^ 2 : ℕ) : ℕ∞) := by
  refine ⟨natDim_HVor_le, ?_⟩
  obtain ⟨n, rfl⟩ : ∃ n, K = n + 1 :=
    ⟨K - 1, (Nat.succ_pred_eq_of_pos (Nat.pos_of_ne_zero (NeZero.ne K))).symm⟩
  exact natDim_HVorSimplex_le

/-- `H_Full` is nonempty, so the bounds above are not vacuous. -/
theorem HFull_nonempty : (HFull (K := K)).Nonempty :=
  ⟨_, ⟨0, rfl⟩⟩

/-! ### `H_Vor = H_Full`

On the simplex the unconstrained Voronoi class and the full-matrix class are
the same set of functions. The equality is one of rules, not of partitions up
to ties: both rules break ties toward the smallest index and their score
vectors differ by a constant at every point. It does not need the prototypes
to be distinct, and it survives requiring them to be. -/

/-- **`H_Vor = H_Full`.** -/
@[thesis "prop:voronoi_full_equality"]
theorem HVor_eq_HFull : HVor (K := K) = HFull := by
  ext δ
  constructor
  · rintro ⟨μ, rfl⟩
    exact ⟨vorMatrix μ, funext fun p => voronoiRule_eq_fullRule μ p.property⟩
  · rintro ⟨A, rfl⟩
    obtain ⟨μ, hμ⟩ := full_in_vor A
    exact ⟨μ, funext fun p => (hμ p.val p.property).symm⟩

/-- **`H_Vor = H_Full` with `K` distinct prototypes**, the class as `06:14`
words it. -/
theorem HVorDistinct_eq_HFull : HVorDistinct (K := K) = HFull := by
  ext δ
  constructor
  · rintro ⟨μ, -, rfl⟩
    exact ⟨vorMatrix μ, funext fun p => voronoiRule_eq_fullRule μ p.property⟩
  · rintro ⟨A, rfl⟩
    obtain ⟨μ, hinj, hμ⟩ := full_in_vor_injective A
    exact ⟨μ, hinj, funext fun p => (hμ p.val p.property).symm⟩

/-- The simplex-restricted class is a subclass of the unconstrained one. -/
theorem HVorSimplex_subset_HVor : HVorSimplex (K := K) ⊆ HVor := by
  rintro δ ⟨μ, -, rfl⟩
  exact ⟨μ, rfl⟩

/-! ### The column gauge of the full-matrix class

Adding one vector to every column of `A` adds the same amount to every score,
so `K` of the `K²` parameters of `H_Full` are a gauge. Subtracting column `0`
from every column leaves a linear-score class in the `K (K - 1)` entries of the
other columns. Through `H_Vor = H_Full` the same bound holds for both Voronoi
classes, below the `K² + K` and `K²` of `thm:voronoi_complexity`. -/

/-- `d_Nat(H_Full) ≤ (n + 1) n` for `K = n + 1` classes. -/
theorem natDim_HFull_succ_le {n : ℕ} :
    natDim (HFull (K := n + 1)) ≤ (((n + 1) * n : ℕ) : ℕ∞) := by
  refine natDim_le_of_linear_scores HFull
    (Φ := fun k p idx =>
      if (finProdFinEquiv.symm idx).2.succ = k
      then p.val (finProdFinEquiv.symm idx).1 else 0)
    (g := fun _ _ => 0) ?_
  rintro δ ⟨A, rfl⟩
  refine ⟨fun idx =>
      A (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2.succ
        - A (finProdFinEquiv.symm idx).1 0, fun p => ?_⟩
  have hscore : ∀ k,
      (∑ idx : Fin ((n + 1) * n),
          (A (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2.succ
            - A (finProdFinEquiv.symm idx).1 0)
          * (if (finProdFinEquiv.symm idx).2.succ = k
             then p.val (finProdFinEquiv.symm idx).1 else 0)) + 0
        = condRisk A p.val k + -condRisk A p.val 0 := by
    intro k
    rw [add_zero, ← Equiv.sum_comp (finProdFinEquiv (m := n + 1) (n := n))]
    simp only [Equiv.symm_apply_apply, Fintype.sum_prod_type]
    refine Fin.cases ?_ (fun j => ?_) k
    · simp [Fin.succ_ne_zero]
    · simp only [Fin.succ_inj, mul_ite, mul_zero]
      have hrow : ∀ i : Fin (n + 1),
          (∑ j' : Fin n, if j' = j then (A i j'.succ - A i 0) * p.val i else 0)
            = A i j.succ * p.val i - A i 0 * p.val i := by
        intro i
        rw [Finset.sum_ite_eq' Finset.univ j
          fun j' => (A i j'.succ - A i 0) * p.val i]
        simp
        ring
      rw [Finset.sum_congr rfl fun i _ => hrow i, Finset.sum_sub_distrib]
      simp only [condRisk]
      ring
  show lexArgmin (condRisk A p.val) = _
  rw [show (fun k => (∑ idx : Fin ((n + 1) * n),
          (A (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2.succ
            - A (finProdFinEquiv.symm idx).1 0)
          * (if (finProdFinEquiv.symm idx).2.succ = k
             then p.val (finProdFinEquiv.symm idx).1 else 0)) + 0)
        = fun k => condRisk A p.val k + -condRisk A p.val 0 from
      funext hscore, lexArgmin_add_const]

/-- **Full-matrix capacity** (`thm:full_complexity`):
subtracting column zero from every column preserves scores up to a common
term and leaves `K(K-1)` parameters. The linear-score lemma gives the bound. -/
theorem natDim_HFull_le_sharp :
    natDim (HFull (K := K)) ≤ ((K * (K - 1) : ℕ) : ℕ∞) := by
  obtain ⟨n, rfl⟩ : ∃ n, K = n + 1 :=
    ⟨K - 1, (Nat.succ_pred_eq_of_pos (Nat.pos_of_ne_zero (NeZero.ne K))).symm⟩
  exact natDim_HFull_succ_le

/-- **Voronoi capacity** (`thm:voronoi_complexity`):
both the unconstrained and simplex-prototype classes have Natarajan dimension
at most `K(K-1)`, by equality with, or inclusion in, `HFull`. -/
theorem natDim_HVor_le_sharp :
    natDim (HVor (K := K)) ≤ ((K * (K - 1) : ℕ) : ℕ∞) ∧
    natDim (HVorSimplex (K := K)) ≤ ((K * (K - 1) : ℕ) : ℕ∞) := by
  have h : natDim (HVor (K := K)) ≤ ((K * (K - 1) : ℕ) : ℕ∞) := by
    rw [HVor_eq_HFull]
    exact natDim_HFull_le_sharp
  exact ⟨h, (natDim_mono HVorSimplex_subset_HVor).trans h⟩

/-! ### The lower bound `d_Nat(H_PS) ≥ K - 1`

Take a direction `d` in the tangent space of the simplex along which the `K`
anchor risks have pairwise distinct slopes `a_k`. On the line through the
barycenter in that direction, and with offsets `v_k = λ a_k² / 2` up to the
values at the barycenter, the score of class `j` at parameter `τ` is
`τ a_j + λ a_j² / 2`, so class `j` wins where `-τ / λ` is closest to `a_j`. The
`K - 1` breakpoints between classes consecutive in the order of the slopes are
the shattered points: at each one exactly two classes tie and every other
class is strictly worse, and a perturbation of the offsets that is monotone
along the order by prescribed signs breaks each tie independently.

Such a direction exists if and only if no two columns of `C` differ by a
multiple of the all-ones vector, which admissibility implies. -/

/-- A finite family of positive reals is bounded below by a positive real. -/
theorem exists_pos_le_of_forall_pos {ι : Type*} [Finite ι] (f : ι → ℝ)
    (h : ∀ i, 0 < f i) : ∃ δ : ℝ, 0 < δ ∧ ∀ i, δ ≤ f i := by
  rcases isEmpty_or_nonempty ι with hι | hι
  · exact ⟨1, one_pos, fun i => (hι.false i).elim⟩
  · obtain ⟨i₀, hi₀⟩ := Finite.exists_min f
    exact ⟨f i₀, h i₀, hi₀⟩

section Chain

variable (a : Fin K → ℝ)

omit [NeZero K] in
/-- Among the indices with a smaller value there is one with the largest. -/
theorem exists_pred (k : Fin K) (h : ∃ j, a j < a k) :
    ∃ j, a j < a k ∧ ∀ j', a j' < a k → a j' ≤ a j := by
  classical
  obtain ⟨j₀, hj₀⟩ := h
  obtain ⟨j, hj, hmax⟩ := Finset.exists_max_image
    (Finset.univ.filter fun j => a j < a k) a ⟨j₀, by simp [hj₀]⟩
  exact ⟨j, by simpa using hj, fun j' hj' => hmax j' (by simp [hj'])⟩

open scoped Classical in
/-- The index just below `k` in the order of the values of `a`, or `k` itself
when `a k` is the least value. -/
noncomputable def pred (k : Fin K) : Fin K :=
  if h : ∃ j, a j < a k then Classical.choose (exists_pred a k h) else k

omit [NeZero K] in
theorem pred_lt {k : Fin K} (h : ∃ j, a j < a k) : a (pred a k) < a k := by
  simp only [pred, dif_pos h]
  exact (Classical.choose_spec (exists_pred a k h)).1

omit [NeZero K] in
theorem le_pred {k j : Fin K} (hj : a j < a k) : a j ≤ a (pred a k) := by
  have h : ∃ j, a j < a k := ⟨j, hj⟩
  simp only [pred, dif_pos h]
  exact (Classical.choose_spec (exists_pred a k h)).2 j hj

/-- **Shattering along a chain.** Let the scores be `G x j + v j` with `G`
fixed and `v` free. Suppose `a` is injective and, for each `k` other than the
one with the least `a`, there is a point `pts k` at which, under base offsets
`v₀`, the score gap between class `j` and class `k` is
`γ (a_j − a_k)(a_j − a_{pred k})`. Then the `K - 1` points are
Natarajan-shattered, with targets `k` and `pred k` at `pts k`. -/
theorem nshatters_of_chain {α : Type*} (G : α → Fin K → ℝ)
    (ha : Function.Injective a) (pts : Fin K → α) (v₀ : Fin K → ℝ) {γ : ℝ}
    (hγ : 0 < γ)
    (hgap : ∀ k, (∃ j, a j < a k) → ∀ j,
      (G (pts k) j + v₀ j) - (G (pts k) k + v₀ k)
        = γ * ((a j - a k) * (a j - a (pred a k)))) :
    ∃ s : Finset α, s.card = K - 1 ∧
      NShatters
        {δ | ∃ v : Fin K → ℝ, δ = fun x => lexArgmin fun j => G x j + v j} s := by
  classical
  have hKpos : (0 : ℝ) < K := by
    exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne K)
  let N : Finset (Fin K) := Finset.univ.filter fun k => ∃ j, a j < a k
  have hN : ∀ k, k ∈ N ↔ ∃ j, a j < a k := fun k => by simp [N]
  -- The points are distinct: the gap identity at a common point is
  -- inconsistent for two different indices.
  have hlt : ∀ k k', k ∈ N → k' ∈ N → pts k = pts k' → a k < a k' → False := by
    intro k k' hk hk' hpt hak
    have e1 := hgap k ((hN k).1 hk) k'
    have e2 := hgap k' ((hN k').1 hk') k
    rw [← hpt] at e2
    have h1 := pred_lt a ((hN k).1 hk)
    have h2 := le_pred a hak
    have hpos : 0 < γ * ((a k' - a k)
        * ((a k' - a k) + (a (pred a k') - a (pred a k)))) :=
      mul_pos hγ (mul_pos (by linarith) (by linarith))
    have hzero : γ * ((a k' - a k)
        * ((a k' - a k) + (a (pred a k') - a (pred a k)))) = 0 := by
      linear_combination -e1 - e2
    linarith
  have hinj : Set.InjOn pts (N : Set (Fin K)) := by
    intro k hk k' hk' hpt
    by_contra hne
    rcases lt_or_gt_of_ne (fun h => hne (ha h)) with h | h
    · exact hlt k k' hk hk' hpt h
    · exact hlt k' k hk' hk hpt.symm h
  have hNcard : N.card = K - 1 := by
    obtain ⟨m, -, hm⟩ := Finset.exists_min_image Finset.univ a
      (Finset.univ_nonempty (α := Fin K))
    have : N = Finset.univ.erase m := by
      ext k
      rw [hN, Finset.mem_erase]
      constructor
      · rintro ⟨j, hj⟩
        refine ⟨?_, Finset.mem_univ k⟩
        rintro rfl
        exact absurd (hm j (Finset.mem_univ j)) (not_le.2 hj)
      · rintro ⟨hne, -⟩
        exact ⟨m, lt_of_le_of_ne (hm k (Finset.mem_univ k))
          fun h => hne (ha h).symm⟩
    rw [this, Finset.card_erase_of_mem (Finset.mem_univ m), Finset.card_univ,
      Fintype.card_fin]
  refine ⟨N.image pts, by rw [Finset.card_image_of_injOn hinj, hNcard], ?_⟩
  -- The margin by which every third class loses at every point.
  have hpos : ∀ q : Fin K × Fin K,
      0 < (if (∃ j, a j < a q.1) ∧ q.2 ≠ q.1 ∧ q.2 ≠ pred a q.1
        then γ * ((a q.2 - a q.1) * (a q.2 - a (pred a q.1))) else 1) := by
    rintro ⟨k, j⟩
    split_ifs with h
    · obtain ⟨hk, hjk, hjp⟩ := h
      refine mul_pos hγ ?_
      have hp := pred_lt a hk
      rcases lt_or_gt_of_ne (fun h => hjk (ha h)) with hlt' | hgt
      · have h1 := le_pred a hlt'
        have h2 : a j < a (pred a k) := lt_of_le_of_ne h1 fun h => hjp (ha h)
        exact mul_pos_of_neg_of_neg (by linarith) (by linarith)
      · exact mul_pos (by linarith) (by linarith)
    · exact one_pos
  obtain ⟨δ, hδ, hδle⟩ := exists_pos_le_of_forall_pos _ hpos
  have hmargin : ∀ k, k ∈ N → ∀ j, j ≠ k → j ≠ pred a k →
      δ ≤ (G (pts k) j + v₀ j) - (G (pts k) k + v₀ k) := by
    intro k hk j hjk hjp
    have := hδle (k, j)
    rw [if_pos ⟨(hN k).1 hk, hjk, hjp⟩] at this
    rw [hgap k ((hN k).1 hk) j]
    exact this
  have htie : ∀ k, k ∈ N →
      G (pts k) (pred a k) + v₀ (pred a k) = G (pts k) k + v₀ k := by
    intro k hk
    have := hgap k ((hN k).1 hk) (pred a k)
    rw [sub_self, mul_zero, mul_zero] at this
    linarith
  -- The targets.
  let f : α → Fin K := fun x =>
    if h : ∃ k, k ∈ N ∧ pts k = x then Classical.choose h else 0
  have hf : ∀ k, k ∈ N → f (pts k) = k := by
    intro k hk
    have h : ∃ k', k' ∈ N ∧ pts k' = pts k := ⟨k, hk, rfl⟩
    simp only [f, dif_pos h]
    exact hinj (Classical.choose_spec h).1 hk (Classical.choose_spec h).2
  refine ⟨f, fun x => pred a (f x), ?_, ?_⟩
  · intro x hx
    obtain ⟨k, hk, rfl⟩ := Finset.mem_image.1 hx
    show f (pts k) ≠ pred a (f (pts k))
    rw [hf k hk]
    exact fun h => (pred_lt a ((hN k).1 hk)).ne (congrArg a h).symm
  · intro b
    let σ : Fin K → ℝ := fun i => if b (pts i) then -1 else 1
    let E : Fin K → ℝ := fun j => ∑ i ∈ N.filter (fun i => a i ≤ a j), σ i
    have hE : ∀ j, |E j| ≤ K := by
      intro j
      calc |E j| ≤ ∑ i ∈ N.filter (fun i => a i ≤ a j), |σ i| :=
            Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _i ∈ N.filter (fun i => a i ≤ a j), (1 : ℝ) :=
            Finset.sum_le_sum fun i _ => by
              simp only [σ]; split_ifs <;> simp
        _ = ((N.filter fun i => a i ≤ a j).card : ℝ) := by simp
        _ ≤ K := by
            have := Finset.card_le_univ (N.filter fun i => a i ≤ a j)
            rw [Fintype.card_fin] at this
            exact_mod_cast this
    have hstep : ∀ k, k ∈ N → E k = σ k + E (pred a k) := by
      intro k hk
      have hp := pred_lt a ((hN k).1 hk)
      have hset : N.filter (fun i => a i ≤ a k)
          = insert k (N.filter fun i => a i ≤ a (pred a k)) := by
        ext i
        simp only [Finset.mem_filter, Finset.mem_insert]
        constructor
        · rintro ⟨hi, hle⟩
          rcases eq_or_lt_of_le hle with h | h
          · exact Or.inl (ha h)
          · exact Or.inr ⟨hi, le_pred a h⟩
        · rintro (rfl | ⟨hi, hle⟩)
          · exact ⟨hk, le_rfl⟩
          · exact ⟨hi, by linarith⟩
      have hnot : k ∉ N.filter fun i => a i ≤ a (pred a k) := by
        simp only [Finset.mem_filter, not_and, not_le]
        exact fun _ => hp
      simp only [E]
      rw [hset, Finset.sum_insert hnot]
    let η : ℝ := δ / (4 * K)
    have hη : 0 < η := div_pos hδ (by linarith)
    have hηK : η * K = δ / 4 := by
      simp only [η]; field_simp
    have hηE : ∀ j, |η * E j| ≤ δ / 4 := by
      intro j
      rw [abs_mul, abs_of_pos hη, ← hηK]
      exact mul_le_mul_of_nonneg_left (hE j) hη.le
    refine ⟨fun x => lexArgmin fun j => G x j + (v₀ j + η * E j),
      ⟨fun j => v₀ j + η * E j, rfl⟩, ?_⟩
    intro x hx
    obtain ⟨k, hk, rfl⟩ := Finset.mem_image.1 hx
    show (lexArgmin fun j => G (pts k) j + (v₀ j + η * E j))
      = if b (pts k) = true then f (pts k) else pred a (f (pts k))
    rw [hf k hk]
    have hst := hstep k hk
    have htk := htie k hk
    by_cases hb : b (pts k) = true
    · -- `σ k = -1`: class `k` wins strictly.
      rw [if_pos hb]
      have hσ : σ k = -1 := by simp only [σ, if_pos hb]
      refine lexArgmin_eq_of_strict fun j hj => ?_
      by_cases hjp : j = pred a k
      · subst hjp
        have : η * E k = η * E (pred a k) - η := by rw [hst, hσ]; ring
        linarith
      · have h1 := hmargin k hk j hj hjp
        have h2 := abs_le.1 (hηE j)
        have h3 := abs_le.1 (hηE k)
        linarith [h2.1, h3.2]
    · -- `σ k = 1`: class `pred k` wins strictly.
      rw [if_neg hb]
      have hσ : σ k = 1 := by simp only [σ, if_neg hb]
      refine lexArgmin_eq_of_strict fun j hj => ?_
      by_cases hjk : j = k
      · subst hjk
        have : η * E j = η * E (pred a j) + η := by rw [hst, hσ]; ring
        linarith
      · have h1 := hmargin k hk j hjk hj
        have h2 := abs_le.1 (hηE j)
        have h3 := abs_le.1 (hηE (pred a k))
        linarith [h2.1, h3.2]

end Chain

/-- No two columns of `C` differ by a multiple of the all-ones vector: the
anchor risks of two distinct classes never differ by a constant on the simplex.
This is what the lower bound of `thm:ps_complexity` needs, and it is weaker
than admissibility. -/
def ColumnsSeparated (C : CostMatrix K) : Prop :=
  ∀ i j, i ≠ j → ∃ l l', C l i - C l j ≠ C l' i - C l' j

omit [NeZero K] in
/-- Admissible cost matrices have separated columns: if two anchor risks
differed by a constant on the simplex, one of the two classes would never be
the strict minimizer. -/
theorem columnsSeparated_of_admissible {C : CostMatrix K} (h : Admissible C) :
    ColumnsSeparated C := by
  intro i j hij
  by_contra hcon
  push Not at hcon
  obtain ⟨p, hp, hpi⟩ := h i
  obtain ⟨q, hq, hqj⟩ := h j
  have key : ∀ r ∈ stdSimplex ℝ (Fin K),
      condRisk C r i - condRisk C r j = C i i - C i j := by
    intro r hr
    rw [condRisk_sub_eq_inner]
    rw [Finset.sum_congr rfl fun l _ => by rw [hcon l i],
      ← Finset.mul_sum, hr.2, mul_one]
  have h1 := hpi j hij.symm
  have h2 := hqj i hij
  have k1 := key p hp
  have k2 := key q hq
  linarith

/-- The slope of the anchor risk of class `k` along the tangent direction
obtained by centering `u`. -/
noncomputable def slope (C : CostMatrix K) (u : Fin K → ℝ) (k : Fin K) : ℝ :=
  ∑ l, (C l k - (∑ l', C l' k) / K) * u l

omit [NeZero K] in
/-- With separated columns there is a direction along which the `K` anchor
risks have pairwise distinct slopes: the directions failing this for a given
pair form a proper subspace, and finitely many proper subspaces do not cover
`ℝ^K`. -/
theorem exists_injective_slope {C : CostMatrix K} (hC : ColumnsSeparated C) :
    ∃ u : Fin K → ℝ, Function.Injective (slope C u) := by
  let n : Fin K × Fin K → Fin K → ℝ := fun q l =>
    (C l q.1 - (∑ l', C l' q.1) / K) - (C l q.2 - (∑ l', C l' q.2) / K)
  let φ : {q : Fin K × Fin K // q.1 ≠ q.2} → Module.Dual ℝ (Fin K → ℝ) :=
    fun q => ∑ l, n q.1 l • LinearMap.proj l
  have hφ : ∀ q u, φ q u = slope C u q.1.1 - slope C u q.1.2 := by
    intro q u
    simp only [φ, n, slope, LinearMap.sum_apply, LinearMap.smul_apply,
      LinearMap.proj_apply, smul_eq_mul, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun l _ => by ring
  have hne : ∀ q, ∃ u, φ q u ≠ 0 := by
    rintro ⟨⟨i, j⟩, hij⟩
    by_contra hall
    push Not at hall
    obtain ⟨l, l', hll⟩ := hC i j hij
    have hval : ∀ l₀ : Fin K, n (i, j) l₀ = 0 := by
      intro l₀
      have := hall (Pi.single l₀ 1)
      simpa [φ, LinearMap.sum_apply, Pi.single_apply] using this
    have e1 := hval l
    have e2 := hval l'
    simp only [n] at e1 e2
    exact hll (by linarith)
  obtain ⟨u, hu⟩ := Module.Dual.exists_forall_ne_zero_of_forall_exists φ hne
  refine ⟨u, fun i j hij => ?_⟩
  by_contra hne'
  have := hu ⟨(i, j), hne'⟩
  rw [hφ] at this
  exact this (by simp [hij])

/-- **`d_Nat(H_PS) ≥ K - 1`** (`thm:ps_complexity`, lower bound), for every
cost matrix with separated columns. -/
theorem natDim_HPS_ge {C : CostMatrix K} (hC : ColumnsSeparated C) :
    ((K - 1 : ℕ) : ℕ∞) ≤ natDim (HPS C) := by
  obtain ⟨u, ha⟩ := exists_injective_slope hC
  have hK : (0 : ℝ) < K := by
    exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne K)
  -- The tangent direction and the scale that keeps the points in the simplex.
  let d : Fin K → ℝ := fun l => u l - (∑ l', u l') / K
  have hdsum : ∑ l, d l = 0 := by
    simp only [d, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
    field_simp
    ring
  let a := slope C u
  let Amax : ℝ := ∑ k, |a k|
  let Dmax : ℝ := ∑ l, |d l|
  have hA : ∀ k, |a k| ≤ Amax := fun k =>
    Finset.single_le_sum (f := fun k => |a k|) (fun _ _ => abs_nonneg _)
      (Finset.mem_univ k)
  have hD : ∀ l, |d l| ≤ Dmax := fun l =>
    Finset.single_le_sum (f := fun l => |d l|) (fun _ _ => abs_nonneg _)
      (Finset.mem_univ l)
  have hA0 : 0 ≤ Amax := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hD0 : 0 ≤ Dmax := Finset.sum_nonneg fun _ _ => abs_nonneg _
  let lam : ℝ := 1 / (K * (Amax * Dmax + 1))
  have hden : 0 < Amax * Dmax + 1 := by positivity
  have hlam : 0 < lam := by positivity
  let τ : Fin K → ℝ := fun k => -lam * (a k + a (pred a k)) / 2
  have hτ : ∀ k l, |τ k * d l| ≤ 1 / K := by
    intro k l
    have h1 : |τ k| ≤ lam * Amax := by
      have : τ k = -(lam / 2) * (a k + a (pred a k)) := by simp only [τ]; ring
      rw [this, abs_mul, abs_neg, abs_of_pos (by positivity : 0 < lam / 2)]
      have := abs_add_le (a k) (a (pred a k))
      nlinarith [hA k, hA (pred a k)]
    calc |τ k * d l| = |τ k| * |d l| := abs_mul _ _
      _ ≤ (lam * Amax) * Dmax :=
          mul_le_mul h1 (hD l) (abs_nonneg _) (by positivity)
      _ = (1 / K) * (Amax * Dmax / (Amax * Dmax + 1)) := by
          simp only [lam]; field_simp
      _ ≤ (1 / K) * 1 := by
          refine mul_le_mul_of_nonneg_left ?_ (by positivity)
          rw [div_le_one hden]; linarith
      _ = 1 / K := mul_one _
  let pts : Fin K → Simplex K := fun k =>
    ⟨fun l => 1 / K + τ k * d l, by
      constructor
      · intro l
        have := abs_le.1 (hτ k l)
        linarith [this.1]
      · rw [Finset.sum_add_distrib, ← Finset.mul_sum, hdsum, Finset.sum_const,
          Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        field_simp
        ring⟩
  -- The anchor risk along the line.
  have hrisk : ∀ k j, condRisk C (pts k).val j
      = (∑ l, C l j) / K + τ k * a j := by
    intro k j
    have hterm : ∀ l : Fin K,
        C l j * (1 / K + τ k * (u l - (∑ l', u l') / K))
          = (1 / K) * C l j + τ k * (C l j * u l)
            - (τ k * ((∑ l', u l') / K)) * C l j := by
      intro l; ring
    have hslope : a j = (∑ l, C l j * u l)
        - ((∑ l', C l' j) / K) * ∑ l, u l := by
      simp only [a, slope]
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun l _ => by ring
    simp only [condRisk, pts, d]
    rw [Finset.sum_congr rfl fun l _ => hterm l, Finset.sum_sub_distrib,
      Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
      ← Finset.mul_sum, hslope]
    ring
  obtain ⟨s, hcard, hs⟩ := nshatters_of_chain a
    (fun x j => condRisk C x.val j) ha pts
    (fun j => -(∑ l, C l j) / K + lam / 2 * a j ^ 2)
    (by positivity : 0 < lam / 2)
    (by
      intro k _ j
      simp only [hrisk, τ]
      ring)
  rw [← hcard]
  refine le_natDim ?_
  obtain ⟨f, g, hfg, hreal⟩ := hs
  refine ⟨f, g, hfg, fun b => ?_⟩
  obtain ⟨h, ⟨v, rfl⟩, hval⟩ := hreal b
  exact ⟨_, ⟨v, rfl⟩, hval⟩

/-- **Parallel-shift capacity** (`thm:ps_complexity`):
for an admissible cost matrix the Natarajan dimension is exactly `K-1`.
The upper bound holds for every matrix; the lower bound uses separated
columns and the breakpoints of a line through the barycenter, a construction
different from the spanning-tree proof in the text (see the module header). -/
@[thesis "thm:ps_complexity"]
theorem ps_complexity {C : CostMatrix K} (hC : Admissible C) :
    natDim (HPS C) = ((K - 1 : ℕ) : ℕ∞) :=
  le_antisymm (natDim_HPS_le C)
    (natDim_HPS_ge (columnsSeparated_of_admissible hC))

/-! ### `H_PS ⊊ H_Vor`

The inclusion is `H_PS ⊆ H_Full = H_Vor`. It is proper for every cost matrix
as soon as there are two classes: a parallel-shift rule compares classes `0`
and `1` at the vertices `e_0` and `e_1` through the fixed quantities
`C_{00} − C_{01}` and `C_{10} − C_{11}` and one free offset difference, so it
cannot realize both the labeling `e_0 ↦ 0, e_1 ↦ 1` and the swapped one. The
full-matrix class realizes both. -/

/-- A vertex of the simplex, as a point of the domain. -/
noncomputable def vtx (i : Fin K) : Simplex K := ⟨vertex i, vertex_mem_stdSimplex i⟩

/-- What a parallel-shift rule that sends the vertex `e_i` to `j` says about
any other class `k`: the score of `j` is no larger, and on a tie `j ≤ k`. -/
theorem psRule_vertex {C : CostMatrix K} {v : Fin K → ℝ} {i j : Fin K}
    (h : psRule C v (vertex i) = j) (k : Fin K) :
    C i j + v j ≤ C i k + v k ∧ (C i j + v j = C i k + v k → j ≤ k) := by
  have hs : ∀ l, psScore C v (vertex i) l = C i l + v l := fun l => by
    simp [psScore]
  have h2 : lexArgmin (psScore C v (vertex i)) = j := h
  have h1 := lexArgmin_le (psScore C v (vertex i)) k
  rw [h2, hs, hs] at h1
  refine ⟨h1, fun heq => ?_⟩
  have := lexArgmin_le_of_tie (f := psScore C v (vertex i)) (j := k)
    (by rw [h2, hs, hs]; exact heq)
  rwa [h2] at this

/-- The matrix whose full rule sends the vertex `e_j` to `σ⁻¹ j`. -/
noncomputable def permMatrix (σ : Equiv.Perm (Fin K)) : CostMatrix K :=
  Matrix.of fun i k => if i = σ k then (-1 : ℝ) else 0

theorem fullRule_permMatrix_vertex (σ : Equiv.Perm (Fin K)) (j : Fin K) :
    lexArgmin (condRisk (permMatrix σ) (vertex j)) = σ.symm j := by
  refine lexArgmin_eq_of_strict fun k hk => ?_
  rw [condRisk_vertex, condRisk_vertex]
  have hne : j ≠ σ k := fun h => hk (by rw [h, Equiv.symm_apply_apply])
  simp [permMatrix, hne]

/-- With at least two classes, some full-matrix rule is not a parallel-shift
rule, whatever the cost matrix. -/
theorem exists_full_not_ps (hK : 2 ≤ K) (C : CostMatrix K) :
    ∃ δ ∈ HFull (K := K), δ ∉ HPS C := by
  let i0 : Fin K := ⟨0, by omega⟩
  let i1 : Fin K := ⟨1, by omega⟩
  have h01 : i0 < i1 := by simp [i0, i1, Fin.lt_def]
  have hne : i0 ≠ i1 := h01.ne
  by_cases hab : C i0 i0 + C i1 i1 < C i0 i1 + C i1 i0
  · -- The swapped labeling `e_0 ↦ 1, e_1 ↦ 0` is out of reach.
    refine ⟨fun p => lexArgmin (condRisk (permMatrix (Equiv.swap i0 i1)) p.val),
      ⟨_, rfl⟩, ?_⟩
    rintro ⟨v, hv⟩
    have e0 : lexArgmin (condRisk (permMatrix (Equiv.swap i0 i1)) (vertex i0))
        = psRule C v (vertex i0) := congrFun hv (vtx i0)
    have e1 : lexArgmin (condRisk (permMatrix (Equiv.swap i0 i1)) (vertex i1))
        = psRule C v (vertex i1) := congrFun hv (vtx i1)
    rw [fullRule_permMatrix_vertex, Equiv.symm_swap, Equiv.swap_apply_left] at e0
    rw [fullRule_permMatrix_vertex, Equiv.symm_swap, Equiv.swap_apply_right] at e1
    have a := psRule_vertex e0.symm i0
    have b := psRule_vertex e1.symm i1
    have a' : C i0 i1 + v i1 < C i0 i0 + v i0 :=
      lt_of_le_of_ne a.1 fun h => absurd (a.2 h) (not_le.2 h01)
    linarith [b.1]
  · -- The labeling `e_0 ↦ 0, e_1 ↦ 1` is out of reach.
    refine ⟨fun p => lexArgmin (condRisk (permMatrix (Equiv.refl _)) p.val),
      ⟨_, rfl⟩, ?_⟩
    rintro ⟨v, hv⟩
    have e0 : lexArgmin (condRisk (permMatrix (Equiv.refl _)) (vertex i0))
        = psRule C v (vertex i0) := congrFun hv (vtx i0)
    have e1 : lexArgmin (condRisk (permMatrix (Equiv.refl _)) (vertex i1))
        = psRule C v (vertex i1) := congrFun hv (vtx i1)
    rw [fullRule_permMatrix_vertex] at e0 e1
    simp only [Equiv.refl_symm, Equiv.refl_apply] at e0 e1
    have a := psRule_vertex e0.symm i1
    have b := psRule_vertex e1.symm i0
    have b' : C i1 i1 + v i1 < C i1 i0 + v i0 :=
      lt_of_le_of_ne b.1 fun h => absurd (b.2 h) (not_le.2 h01)
    linarith [a.1]

/-- **`H_PS ⊊ H_Vor`** (`prop:ps_in_vor`).

For every cost matrix and `K ≥ 2`, every parallel-shift rule is the Voronoi
rule of `K` distinct prototypes in `ℝ^K` at every point of the simplex, ties
included, and some Voronoi rule of `K` distinct prototypes is not a
parallel-shift rule.

Two differences from the text. The text does not say `K ≥ 2`; at `K = 1` both
classes consist of the one constant rule and the inclusion is not proper. And
the text justifies properness by the `K = 3`, `0/1`-loss witness of
`thm:rotational_dominance`, which covers that cost matrix only; the argument
here covers every cost matrix. -/
theorem HPS_ssubset_HVor (hK : 2 ≤ K) (C : CostMatrix K) :
    HPS C ⊂ HVorDistinct := by
  rw [HVorDistinct_eq_HFull]
  rw [Set.ssubset_def]
  refine ⟨HPS_subset_HFull C, fun hsub => ?_⟩
  obtain ⟨δ, hδ, hnot⟩ := exists_full_not_ps hK C
  exact hnot (hsub hδ)

/-! ### `thm:ps_complexity` is false without a hypothesis on the cost matrix

The theorem is printed with no hypothesis on `C`. For the zero matrix every
parallel-shift rule ignores its input, so no two points are shattered and the
dimension is at most `1`, which is below `K - 1` as soon as `K ≥ 3`. -/

/-- With the zero cost matrix a parallel-shift rule is constant. -/
theorem psRule_zero (v p : Fin K → ℝ) :
    psRule (0 : CostMatrix K) v p = lexArgmin v := by
  have h : psScore (0 : CostMatrix K) v p = v := by
    funext k
    simp [psScore, condRisk]
  rw [psRule, h]

/-- `d_Nat(H_PS) ≤ 1` for the zero cost matrix, whatever `K`. -/
theorem natDim_HPS_zero_le_one : natDim (HPS (0 : CostMatrix K)) ≤ 1 := by
  classical
  refine natDim_le fun s hs => ?_
  by_contra hcon
  have hcard : 1 < s.card := by
    by_contra hle
    exact hcon (by exact_mod_cast not_lt.1 hle)
  obtain ⟨x, hx, y, hy, hxy⟩ := Finset.one_lt_card.1 hcard
  obtain ⟨f, g, hfg, hreal⟩ := hs
  -- A constant rule gives `x` and `y` the same label, in three labelings.
  obtain ⟨h₁, ⟨v₁, rfl⟩, e₁⟩ := hreal fun z => decide (z = x)
  obtain ⟨h₂, ⟨v₂, rfl⟩, e₂⟩ := hreal fun z => decide (z = y)
  obtain ⟨h₃, ⟨v₃, rfl⟩, e₃⟩ := hreal fun _ => true
  have a₁ : lexArgmin v₁ = f x := by simpa [psRule_zero] using e₁ x hx
  have b₁ : lexArgmin v₁ = g y := by simpa [psRule_zero, hxy.symm] using e₁ y hy
  have a₂ : lexArgmin v₂ = g x := by simpa [psRule_zero, hxy] using e₂ x hx
  have b₂ : lexArgmin v₂ = f y := by simpa [psRule_zero] using e₂ y hy
  have a₃ : lexArgmin v₃ = f x := by simpa [psRule_zero] using e₃ x hx
  have b₃ : lexArgmin v₃ = f y := by simpa [psRule_zero] using e₃ y hy
  exact hfg x hx ((a₃.symm.trans b₃).trans (b₂.symm.trans a₂))

/-- **`thm:ps_complexity` fails for the zero cost matrix** when `K ≥ 3`: the
dimension is not `K - 1`. The admissibility hypothesis of `ps_complexity`
cannot be dropped. -/
theorem ps_complexity_fails_for_zero (hK : 3 ≤ K) :
    natDim (HPS (0 : CostMatrix K)) ≠ ((K - 1 : ℕ) : ℕ∞) := by
  intro h
  have h1 := natDim_HPS_zero_le_one (K := K)
  rw [h] at h1
  have h2 : K - 1 ≤ 1 := by exact_mod_cast h1
  omega


/-! ### The Dominant Label Constraint is a proper restriction -/

/-- Voronoi rules with simplex prototypes satisfying the Dominant Label
Constraint. Equal prototypes are allowed in `HVorSimplex`, but not here. -/
def HVorDLC : Set (Simplex K → Fin K) :=
  {δ | ∃ μ : Prototypes K, (∀ k, μ k ∈ stdSimplex ℝ (Fin K))
    ∧ DLC μ ∧ δ = fun p => voronoiRule μ p.val}

theorem HVorDLC_subset_HVorSimplex : HVorDLC (K := K) ⊆ HVorSimplex := by
  rintro δ ⟨μ, hμ, -, rfl⟩
  exact ⟨μ, hμ, rfl⟩

/-- Every DLC rule returns the class of each simplex vertex. -/
theorem HVorDLC_vertex {δ : Simplex K → Fin K} (hδ : δ ∈ HVorDLC) (k : Fin K) :
    δ ⟨vertex k, vertex_mem_stdSimplex k⟩ = k := by
  obtain ⟨μ, -, hDLC, rfl⟩ := hδ
  exact voronoiRule_vertex_of_DLC hDLC k

/-- Equal prototypes give the constant rule `0`, with the fixed tie-break. -/
theorem constant_zero_mem_HVorSimplex : (fun _ : Simplex K => (0 : Fin K)) ∈ HVorSimplex := by
  refine ⟨fun _ => vertex 0, fun _ => vertex_mem_stdSimplex 0, ?_⟩
  funext p
  symm
  apply le_antisymm
  · exact lexArgmin_le_index (fun _ => le_rfl)
  · exact Fin.zero_le _

theorem constant_zero_not_mem_HVorDLC (hK : 2 ≤ K) :
    (fun _ : Simplex K => (0 : Fin K)) ∉ HVorDLC := by
  intro h
  have := HVorDLC_vertex h (⟨1, by omega⟩ : Fin K)
  have : (0 : ℕ) = 1 := congrArg Fin.val this
  omega

/-- For `K ≥ 2`, the DLC class is proper even inside simplex-prototype rules. -/
theorem HVorDLC_ssubset_HVorSimplex (hK : 2 ≤ K) :
    HVorDLC (K := K) ⊂ HVorSimplex := by
  rw [Set.ssubset_def]
  exact ⟨HVorDLC_subset_HVorSimplex, fun h =>
    constant_zero_not_mem_HVorDLC hK (h constant_zero_mem_HVorSimplex)⟩

/-- For `K ≥ 2`, the DLC class is a proper subclass of full-matrix rules. -/
theorem HVorDLC_ssubset_HFull (hK : 2 ≤ K) : HVorDLC (K := K) ⊂ HFull := by
  have hsub : HVorSimplex (K := K) ⊆ HFull := by
    rw [← HVor_eq_HFull]
    exact HVorSimplex_subset_HVor
  rw [Set.ssubset_def]
  exact ⟨HVorDLC_subset_HVorSimplex.trans hsub, fun h =>
    constant_zero_not_mem_HVorDLC hK (h (hsub constant_zero_mem_HVorSimplex))⟩


/-- **Strict inclusion of parallel shifts in the full-matrix class.**
For every cost matrix and `K ≥ 2`, the parallel-shift class is a proper
subclass of the full-matrix class. Properness uses the two vertex labelings;
the risk-separation witness is a separate statement. -/
theorem HPS_ssubset_HFull (hK : 2 ≤ K) (C : CostMatrix K) : HPS C ⊂ HFull := by
  simpa only [HVorDistinct_eq_HFull] using HPS_ssubset_HVor hK C

/-- **Strict inclusion of parallel shifts in the Voronoi class**
(`prop:ps_in_vor`). For every cost matrix and `K ≥ 2`, every parallel-shift
rule on the simplex is the Voronoi rule of some prototypes in `ℝ^K`, ties
included, and some Voronoi rule is not a parallel-shift rule. `HPS` takes the
offsets in all of `ℝ^K`, which gives the same class as the text's mean-zero
gauge (`HPSGauged_eq`). -/
@[thesis "prop:ps_in_vor"]
theorem HPS_ssubset_HVor_unrestricted (hK : 2 ≤ K) (C : CostMatrix K) :
    HPS C ⊂ HVor := by
  rw [HVor_eq_HFull]
  exact HPS_ssubset_HFull hK C

end ThresholdOpt.Complexity
