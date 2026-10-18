import ThresholdOpt.Multiclass

/-!
# Partition optimization (Chapter 5)

The partition-matrix framework and the four equivalence and invariance results:
parallel-shift boundaries stay parallel (`prop:parallel_invariance`), linear
class scaling rotates them (`prop:column_rotation`), probability-ratio
thresholding is linear scaling under `0/1` loss (`prop:ratio_equivalence`), and
multidimensional thresholding is the parallel shift of the `0/1` loss matrix
(`prop:marchetti_equivalence`).

`[NeZero K]` is bound per declaration rather than for the whole section: the
score identities are statements about sums and hold for any `K`, while only the
rules built from `lexArgmin` need the label space to be nonempty.
-/

namespace ThresholdOpt.Partition

open ThresholdOpt

variable {K : ℕ}

/-! ### The `0/1` loss -/

/-- The symmetric `0/1` loss matrix, `C_{ij} = 1 - δ_{ij}`. -/
noncomputable def zeroOne (K : ℕ) : CostMatrix K :=
  Matrix.of fun i j => if i = j then 0 else 1

/-- On the simplex the `0/1` conditional risk of action `i` is `1 - p i`. This
identity drives both `prop:ratio_equivalence` and
`prop:marchetti_equivalence`. -/
theorem condRisk_zeroOne {p : Fin K → ℝ} (hp : p ∈ stdSimplex ℝ (Fin K))
    (i : Fin K) : condRisk (zeroOne K) p i = 1 - p i := by
  have hsum : ∑ j, p j = 1 := hp.2
  have hterm : ∀ j : Fin K,
      (zeroOne K) j i * p j = p j - (if j = i then p j else 0) := by
    intro j
    by_cases h : j = i <;> simp [zeroOne, h]
  simp only [condRisk]
  rw [Finset.sum_congr rfl fun j _ => hterm j, Finset.sum_sub_distrib, hsum,
    Finset.sum_ite_eq' Finset.univ i p]
  simp

/-! ### Parallel shift -/

/-- The parallel-shift score of `eq:hps_def`: the anchor risk plus a per-class
offset. -/
noncomputable def psScore (C : CostMatrix K) (v : Fin K → ℝ)
    (p : Fin K → ℝ) (k : Fin K) : ℝ :=
  condRisk C p k + v k

/-- The parallel-shift decision rule. -/
noncomputable def psRule [NeZero K] (C : CostMatrix K) (v : Fin K → ℝ)
    (p : Fin K → ℝ) : Fin K :=
  lexArgmin (psScore C v p)

/-- **Parallel invariance** (`prop:parallel_invariance`).

The boundary between actions `i` and `j` under a parallel shift differs from
the anchor boundary by the constant `v i - v j`, so its normal `c_i - c_j` is
unchanged and the shifted boundary is parallel to the original. -/
@[thesis "prop:parallel_invariance"]
theorem parallel_invariance (C : CostMatrix K) (v : Fin K → ℝ)
    (p : Fin K → ℝ) (i j : Fin K) :
    psScore C v p i - psScore C v p j
      = (condRisk C p i - condRisk C p j) + (v i - v j) := by
  simp only [psScore]
  ring

/-- A parallel shift by a single constant is no shift at all: the additive
gauge freedom of `05:118`, which is why `H_PS` is fixed to `∑ v_k = 0` and has
`K - 1` free parameters rather than `K`. -/
theorem psRule_add_const [NeZero K] (C : CostMatrix K) (v : Fin K → ℝ) (c : ℝ)
    (p : Fin K → ℝ) :
    psRule C (fun k => v k + c) p = psRule C v p := by
  have h : psScore C (fun k => v k + c) p = fun k => psScore C v p k + c := by
    funext k; simp only [psScore]; ring
  simp only [psRule, h, lexArgmin_add_const]

/-! ### Linear class scaling -/

/-- The linear-scaling score of `eq:hsc_def`, `∑_k C_{ki} β_k p_k`: the anchor
risk evaluated at the reweighted probability vector. -/
noncomputable def scScore (C : CostMatrix K) (β : Fin K → ℝ)
    (p : Fin K → ℝ) (k : Fin K) : ℝ :=
  condRisk C (fun j => β j * p j) k

noncomputable def scRule [NeZero K] (C : CostMatrix K) (β : Fin K → ℝ)
    (p : Fin K → ℝ) : Fin K :=
  lexArgmin (scScore C β p)

/-- **Column rotation** (`prop:column_rotation`).

The linear-scaling boundary between `i` and `j` has normal `β ⊙ (c_i - c_j)`,
the coordinatewise product. Unlike the parallel shift this is in general not
proportional to `c_i - c_j`, so the boundary rotates. -/
@[thesis "prop:column_rotation"]
theorem column_rotation (C : CostMatrix K) (β : Fin K → ℝ)
    (p : Fin K → ℝ) (i j : Fin K) :
    scScore C β p i - scScore C β p j
      = ∑ k, (β k * (C k i - C k j)) * p k := by
  simp only [scScore, condRisk, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun k _ => by ring

/-! ### Probability-ratio thresholding -/

/-- **Ratio equivalence** (`prop:ratio_equivalence`).

Under the `0/1` loss, probability-ratio thresholding `argmax_k p_k / τ_k` is
linear class scaling with `β_k = 1 / τ_k`. The scaling score differs from
`-β_i p_i` by the constant `∑_k β_k p_k`, which `lexArgmin` cannot see. -/
@[thesis "prop:ratio_equivalence"]
theorem ratio_equivalence (β : Fin K → ℝ) (p : Fin K → ℝ) (i : Fin K) :
    scScore (zeroOne K) β p i = (∑ k, β k * p k) - β i * p i := by
  have hterm : ∀ k : Fin K,
      (zeroOne K) k i * (β k * p k)
        = β k * p k - (if k = i then β k * p k else 0) := by
    intro k
    by_cases h : k = i <;> simp [zeroOne, h]
  simp only [scScore, condRisk]
  rw [Finset.sum_congr rfl fun k _ => hterm k, Finset.sum_sub_distrib,
    Finset.sum_ite_eq' Finset.univ i fun k => β k * p k]
  simp

/-- Consequently the `0/1` linear-scaling rule is exactly `argmax_k β_k p_k`,
which with `β_k = 1/τ_k` is `argmax_k p_k / τ_k`. -/
theorem scRule_zeroOne_eq_argmax [NeZero K] (β : Fin K → ℝ) (p : Fin K → ℝ) :
    scRule (zeroOne K) β p = lexArgmax (fun k => β k * p k) := by
  have h : scScore (zeroOne K) β p
      = fun k => (fun j => -(β j * p j)) k + (∑ m, β m * p m) := by
    funext k
    rw [ratio_equivalence β p k]
    ring
  simp only [scRule, lexArgmax, h, lexArgmin_add_const]

/-! ### Multidimensional thresholding -/

/-- **Parallel-shift representation of simplex thresholds**
(`prop:marchetti_equivalence`). This pointwise identity holds for every real
threshold vector, hence in particular for the simplex domain in the text.

Multidimensional thresholding `argmax_k (p_k - τ_k)` is the parallel-shift rule
applied to the `0/1` loss matrix: the `0/1` parallel-shift score is
`1 - p_k + τ_k`, whose `argmin` is the `argmax` of `p_k - τ_k`. -/
@[thesis "prop:marchetti_equivalence"]
theorem marchetti_equivalence [NeZero K] {p : Fin K → ℝ}
    (hp : p ∈ stdSimplex ℝ (Fin K)) (τ : Fin K → ℝ) :
    psRule (zeroOne K) τ p = lexArgmax (fun k => p k - τ k) := by
  have h : psScore (zeroOne K) τ p
      = fun k => (fun j => -(p j - τ j)) k + 1 := by
    funext k
    rw [psScore, condRisk_zeroOne hp k]
    ring
  simp only [psRule, lexArgmax, h, lexArgmin_add_const]

end ThresholdOpt.Partition
