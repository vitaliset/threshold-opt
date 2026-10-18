import ThresholdOpt.PointwiseApprox

/-!
# Generalization bounds for the decision classes of Chapter 6

The classes `H_PS`, `H_Sc`, `H_Full`, `H_Vor` and `H_Vor^Δ` are indexed by real
parameters, so `Literature.gen_bound_natarajan`, which needs a countable
class of measurable rules, does not apply to them directly. Each is
approximated pointwise, ties included, by a countable class of measurable
rules (`PointwiseApprox.ApproxBy`), and `PointwiseApprox.gen_bound_natarajan_of_approx`
transfers the bound:

| Class | Countable approximating class | Capacity in the bound |
| --- | --- | --- |
| `H_PS` (`HPS A`) | rational offsets, inside `H_PS` | `K − 1` (`thm:ps_complexity`) |
| `H_Full` | rational matrices, inside `H_Full` | `K (K − 1)` |
| `H_Vor` | the same (`H_Vor = H_Full`) | `K (K − 1)` |
| `H_Vor^Δ` | the same (`H_Vor^Δ ⊆ H_Full`) | `K (K − 1)` |
| `H_Sc` (`HSc A`) | rational weights plus a tie-breaking offset `ε k` | `K` |

For `H_PS` and `H_Full` the approximating rules move each parameter of class
`k` by an amount in `(k/(n+1), (k + 1/2)/(n+1))`, which vanishes and grows with
`k`, so ties keep resolving toward the smallest index
(`PointwiseApprox.eventually_lexArgmin_add`). For `H_Sc` no perturbation of the
weights alone does that at every tie (the sign of the change of a score
difference depends on the point), so the approximating rules carry one more
parameter, an offset `ε k`, and the capacity is `K`, one more than
`thm:sc_complexity`'s `K − 1`; see `gen_bound_HSc`.

The rules act on the simplex, with the subspace σ-algebra of `ℝ^K`; `μ` is the
law of `(p̂(x), y)`, as in `Literature.gen_bound_binary`.
-/

namespace ThresholdOpt.DecisionClassBounds

open MeasureTheory Filter Topology
open ThresholdOpt ThresholdOpt.Learning ThresholdOpt.Capacity ThresholdOpt.Literature
open ThresholdOpt.Partition ThresholdOpt.Voronoi ThresholdOpt.Complexity
open ThresholdOpt.PointwiseApprox

/-- The simplex carries the subspace σ-algebra of `ℝ^K`. -/
instance instMeasurableSpaceSimplex {K : ℕ} : MeasurableSpace (Simplex K) :=
  inferInstanceAs (MeasurableSpace {p : Fin K → ℝ // p ∈ stdSimplex ℝ (Fin K)})

theorem measurable_val {K : ℕ} : Measurable (fun p : Simplex K => p.val) :=
  measurable_subtype_coe

variable {K : ℕ} [NeZero K]

omit [NeZero K] in
theorem measurable_condRisk (A : CostMatrix K) (k : Fin K) :
    Measurable (fun p : Simplex K => condRisk A p.val k) := by
  unfold condRisk
  exact Finset.measurable_sum _ fun j _ =>
    measurable_const.mul ((measurable_pi_apply j).comp measurable_val)

omit [NeZero K] in
/-- A convex combination of numbers in `(a, b)` lies in `(a, b)`. -/
theorem mem_Ioo_sum_mul {p : Fin K → ℝ} (hp : p ∈ stdSimplex ℝ (Fin K))
    {x : Fin K → ℝ} {a b : ℝ} (hx : ∀ i, a < x i ∧ x i < b) :
    a < ∑ i, x i * p i ∧ ∑ i, x i * p i < b := by
  have hsum : ∑ i, p i = 1 := hp.2
  obtain ⟨i0, hi0⟩ : ∃ i, 0 < p i := by
    by_contra h
    push Not at h
    have : ∑ i, p i ≤ 0 := Finset.sum_nonpos fun i _ => h i
    linarith
  constructor
  · have hpos : 0 < ∑ i, (x i - a) * p i :=
      Finset.sum_pos' (fun i _ => mul_nonneg (by linarith [(hx i).1]) (hp.1 i))
        ⟨i0, Finset.mem_univ _, mul_pos (by linarith [(hx i0).1]) hi0⟩
    have e : ∑ i, (x i - a) * p i = ∑ i, x i * p i - a := by
      simp only [sub_mul, Finset.sum_sub_distrib, ← Finset.mul_sum, hsum, mul_one]
    linarith
  · have hpos : 0 < ∑ i, (b - x i) * p i :=
      Finset.sum_pos' (fun i _ => mul_nonneg (by linarith [(hx i).2]) (hp.1 i))
        ⟨i0, Finset.mem_univ _, mul_pos (by linarith [(hx i0).2]) hi0⟩
    have e : ∑ i, (b - x i) * p i = b - ∑ i, x i * p i := by
      simp only [sub_mul, Finset.sum_sub_distrib, ← Finset.mul_sum, hsum, mul_one]
    linarith

omit [NeZero K] in
/-- A rational in `(c + k/(n+1), c + (k + 1/2)/(n+1))`. -/
theorem exists_rat_offset (c : ℝ) (k n : ℕ) :
    ∃ q : ℚ, c + (k : ℝ) / ((n : ℝ) + 1) < q ∧
      (q : ℝ) < c + ((k : ℝ) + 1 / 2) / ((n : ℝ) + 1) := by
  refine exists_rat_btwn ?_
  have hpos : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have : (k : ℝ) / ((n : ℝ) + 1) < ((k : ℝ) + 1 / 2) / ((n : ℝ) + 1) := by
    rw [div_lt_div_iff_of_pos_right hpos]
    linarith
  linarith

/-! ### The parallel shift -/

/-- The parallel-shift rules with rational offsets. -/
def ratHPS (A : CostMatrix K) : Set (Simplex K → Fin K) :=
  Set.range fun q : Fin K → ℚ => fun p : Simplex K => psRule A (fun k => (q k : ℝ)) p.val

theorem ratHPS_countable (A : CostMatrix K) : (ratHPS A).Countable :=
  Set.countable_range _

theorem ratHPS_subset (A : CostMatrix K) : ratHPS A ⊆ HPS A := by
  rintro _ ⟨q, rfl⟩
  exact ⟨_, rfl⟩

theorem measurable_psRule (A : CostMatrix K) (v : Fin K → ℝ) :
    Measurable (fun p : Simplex K => psRule A v p.val) := by
  show Measurable fun p : Simplex K => lexArgmin fun k => condRisk A p.val k + v k
  exact measurable_lexArgmin fun k => (measurable_condRisk A k).add_const _

theorem ratHPS_measurable (A : CostMatrix K) : ∀ δ ∈ ratHPS A, Measurable δ := by
  rintro _ ⟨q, rfl⟩
  exact measurable_psRule A _

/-- **Every parallel-shift rule is a pointwise limit of rational ones**, ties
included: the offset of class `k` is raised by an amount in
`(k/(n+1), (k + 1/2)/(n+1))`. -/
theorem HPS_approx (A : CostMatrix K) : ApproxBy (HPS A) (ratHPS A) := by
  rintro δ ⟨v, rfl⟩
  choose q hq using fun (n : ℕ) (k : Fin K) => exists_rat_offset (v k) k n
  set e : ℕ → Fin K → ℝ := fun n k => (q n k : ℝ) - v k with he
  obtain ⟨he0, hmono⟩ := perturb_of_bounds e
    (fun n k => by simp only [e]; linarith [(hq n k).1])
    (fun n k => by simp only [e]; linarith [(hq n k).2])
  refine ⟨fun n p => psRule A (fun k => (q n k : ℝ)) p.val, fun n => ⟨q n, rfl⟩,
    fun p => ?_⟩
  filter_upwards [eventually_lexArgmin_add (psScore A v p.val) e he0 hmono] with n hn
  show lexArgmin (psScore A (fun k => (q n k : ℝ)) p.val) = lexArgmin (psScore A v p.val)
  rw [← hn]
  congr 1
  funext k
  simp only [psScore, e]
  ring

/-- **Generalization bound for the parallel-shift class**, with capacity
`d_Nat(H_PS) ≤ K − 1` (`thm:ps_complexity`). For any matrix `A` defining the
class, costs `C` in `[0, Cmax]`, any law `μ` of `(p, y)` on the simplex times
the labels, and `D = 2 (K−1) (⌈log₂((K−1) K²)⌉ + 1) ≤ m`: with probability at
least `1 − ν`, every `δ ∈ H_PS` satisfies
`R(δ) ≤ R̂(δ) + Cmax (√(2 D log(e m / D)/m) + √(log(K²/ν)/(2m)))`. -/
theorem gen_bound_HPS (A C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (μ : Measure (Simplex K × Fin K)) [IsProbabilityMeasure μ] (m : ℕ) (ν : ℝ)
    (hdm : levelDim (K - 1) K ≤ m) (hm : 0 < m) (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ HPS A,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + Cmax * vcRate m (levelDim (K - 1) K) (ν / ((K : ℝ) * K))} :=
  gen_bound_natarajan_of_approx C hC (HPS A) (ratHPS A) (ratHPS_countable A)
    (ratHPS_measurable A) (HPS_approx A) μ (K - 1) m ν
    ((natDim_mono (ratHPS_subset A)).trans (natDim_HPS_le A)) hdm hm hν

/-- **The clean form for the parallel-shift class** (`rem:natarajan_gen_bound`
with `d_Nat = K − 1`): for `K ≥ 2`, `m ≥ 1`, `0 < ν ≤ 1`, with probability at
least `1 − ν` every `δ ∈ H_PS` satisfies
`R(δ) ≤ R̂(δ) + c · Cmax · √((K − 1)/m)`, `c = cleanFactor K (K−1) m ν`. -/
theorem gen_bound_HPS_clean (hK : 2 ≤ K) (A C : CostMatrix K) {Cmax : ℝ}
    (hC : Bounded C Cmax) (μ : Measure (Simplex K × Fin K)) [IsProbabilityMeasure μ]
    (m : ℕ) (ν : ℝ) (hm : 0 < m) (hν : 0 < ν) (hν1 : ν ≤ 1) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ HPS A,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + cleanFactor K (K - 1) m ν * Cmax * Real.sqrt ((K - 1 : ℕ) / m)} :=
  gen_bound_natarajan_clean_of_approx C hC (HPS A) (ratHPS A) (ratHPS_countable A)
    (ratHPS_measurable A) (HPS_approx A) μ (K - 1) m ν
    ((natDim_mono (ratHPS_subset A)).trans (natDim_HPS_le A)) (by omega) hm hν hν1

/-! ### The full-matrix class, and the two Voronoi classes -/

/-- The full-matrix rules with rational matrices. -/
def ratHFull : Set (Simplex K → Fin K) :=
  Set.range fun Q : Fin K → Fin K → ℚ => fun p : Simplex K =>
    lexArgmin (condRisk (Matrix.of fun i k => (Q i k : ℝ)) p.val)

theorem ratHFull_countable : (ratHFull (K := K)).Countable :=
  Set.countable_range _

theorem ratHFull_subset : ratHFull (K := K) ⊆ HFull := by
  rintro _ ⟨Q, rfl⟩
  exact ⟨_, rfl⟩

theorem measurable_fullRule (A : CostMatrix K) :
    Measurable (fun p : Simplex K => lexArgmin (condRisk A p.val)) := by
  have h := measurable_lexArgmin (K := K) (s := fun k (p : Simplex K) => condRisk A p.val k)
    fun k => measurable_condRisk A k
  exact h

theorem ratHFull_measurable : ∀ δ ∈ ratHFull (K := K), Measurable δ := by
  rintro _ ⟨Q, rfl⟩
  exact measurable_fullRule _

/-- **Every full-matrix rule is a pointwise limit of rational ones**, ties
included: column `k` is raised entrywise by amounts in
`(k/(n+1), (k + 1/2)/(n+1))`, so on the simplex the score of class `k` is
raised by an amount in the same interval. -/
theorem HFull_approx : ApproxBy (HFull (K := K)) ratHFull := by
  rintro δ ⟨A, rfl⟩
  choose Q hQ using fun (n : ℕ) (i k : Fin K) => exists_rat_offset (A i k) k n
  refine ⟨fun n p => lexArgmin (condRisk (Matrix.of fun i k => (Q n i k : ℝ)) p.val),
    fun n => ⟨Q n, rfl⟩, fun p => ?_⟩
  set e : ℕ → Fin K → ℝ := fun n k => ∑ i, ((Q n i k : ℝ) - A i k) * p.val i with he
  have hb : ∀ (n : ℕ) (k : Fin K), (k : ℝ) / ((n : ℝ) + 1) < e n k ∧
      e n k < ((k : ℝ) + 1 / 2) / ((n : ℝ) + 1) := fun n k =>
    mem_Ioo_sum_mul p.property fun i =>
      ⟨by linarith [(hQ n i k).1], by linarith [(hQ n i k).2]⟩
  obtain ⟨he0, hmono⟩ := perturb_of_bounds e (fun n k => (hb n k).1) (fun n k => (hb n k).2)
  filter_upwards [eventually_lexArgmin_add (condRisk A p.val) e he0 hmono] with n hn
  show lexArgmin (condRisk (Matrix.of fun i k => (Q n i k : ℝ)) p.val)
    = lexArgmin (condRisk A p.val)
  rw [← hn]
  congr 1
  funext k
  simp only [condRisk, e, Matrix.of_apply]
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

theorem natDim_ratHFull_le : natDim (ratHFull (K := K)) ≤ ((K * (K - 1) : ℕ) : ℕ∞) :=
  (natDim_mono ratHFull_subset).trans natDim_HFull_le_sharp

/-- **Generalization bound for any class of full-matrix rules**, with capacity
`K (K − 1)`: with probability at least `1 − ν`, every rule of any `H ⊆ H_Full`
satisfies the bound of `Literature.gen_bound_natarajan` at `d = K (K − 1)`.
This covers `H_Full`, `H_Vor` (which equals it), `H_Vor^Δ` and `H_Sc`. -/
theorem gen_bound_subset_HFull (H : Set (Simplex K → Fin K)) (hH : H ⊆ HFull)
    (C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (μ : Measure (Simplex K × Fin K)) [IsProbabilityMeasure μ] (m : ℕ) (ν : ℝ)
    (hdm : levelDim (K * (K - 1)) K ≤ m) (hm : 0 < m) (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + Cmax * vcRate m (levelDim (K * (K - 1)) K) (ν / ((K : ℝ) * K))} :=
  gen_bound_natarajan_of_approx C hC H ratHFull ratHFull_countable ratHFull_measurable
    (HFull_approx.mono_left hH) μ (K * (K - 1)) m ν natDim_ratHFull_le hdm hm hν

/-- **The clean form for any class of full-matrix rules**, `d = K (K − 1)`:
for `K ≥ 2`, `m ≥ 1`, `0 < ν ≤ 1`, with probability at least `1 − ν` every
rule of `H ⊆ H_Full` satisfies `R(δ) ≤ R̂(δ) + c · Cmax · √(K (K − 1)/m)`,
with `c = cleanFactor K (K (K−1)) m ν`. For `H = H_Vor^Δ` this gives the
display of Section 6 that follows `thm:voronoi_complexity`, which uses the
larger capacity `K²`. -/
theorem gen_bound_subset_HFull_clean (hK : 2 ≤ K) (H : Set (Simplex K → Fin K))
    (hH : H ⊆ HFull) (C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (μ : Measure (Simplex K × Fin K)) [IsProbabilityMeasure μ]
    (m : ℕ) (ν : ℝ) (hm : 0 < m) (hν : 0 < ν) (hν1 : ν ≤ 1) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ H,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + cleanFactor K (K * (K - 1)) m ν * Cmax
              * Real.sqrt ((K * (K - 1) : ℕ) / m)} :=
  gen_bound_natarajan_clean_of_approx C hC H ratHFull ratHFull_countable
    ratHFull_measurable (HFull_approx.mono_left hH) μ (K * (K - 1)) m ν
    natDim_ratHFull_le (Nat.mul_pos (by omega) (by omega)) hm hν hν1

/-- **Generalization bound for `H_Vor^Δ`** (prototypes in the simplex), with
capacity `K (K − 1)`. -/
theorem gen_bound_HVorSimplex (C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (μ : Measure (Simplex K × Fin K)) [IsProbabilityMeasure μ] (m : ℕ) (ν : ℝ)
    (hdm : levelDim (K * (K - 1)) K ≤ m) (hm : 0 < m) (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ HVorSimplex,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + Cmax * vcRate m (levelDim (K * (K - 1)) K) (ν / ((K : ℝ) * K))} :=
  gen_bound_subset_HFull HVorSimplex VoronoiSimplex.HVorSimplex_subset_HFull C hC μ m ν
    hdm hm hν

/-- **The clean bound for `H_Vor^Δ`**, the display of Section 6 that follows
`thm:voronoi_complexity`: for `K ≥ 2`, costs in `[0, Cmax]`, `m ≥ 1` and
`0 < ν ≤ 1`, with probability at least `1 − ν` every `δ ∈ H_Vor^Δ` satisfies

`R(δ) ≤ R̂(δ) + c · Cmax · √(K (K − 1) / m)`,

with `c = cleanFactor K (K (K−1)) m ν`, the factor `c_Nat` of
`rem:natarajan_gen_bound` (logarithmic in `K`, `m` and `1/ν`). The capacity
`K (K − 1)` is `d_Nat(H_Vor^Δ) ≤ K (K − 1)`
(`Complexity.natDim_HVor_le_sharp`). The text's display uses `K²`, the bound
of `thm:voronoi_complexity`; it follows from the remark with `d = K²`, since
the approximating rational full-matrix class has Natarajan dimension at most
`K (K − 1) ≤ K²`. -/
theorem gen_bound_HVorSimplex_clean (hK : 2 ≤ K) (C : CostMatrix K) {Cmax : ℝ}
    (hC : Bounded C Cmax) (μ : Measure (Simplex K × Fin K)) [IsProbabilityMeasure μ]
    (m : ℕ) (ν : ℝ) (hm : 0 < m) (hν : 0 < ν) (hν1 : ν ≤ 1) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ HVorSimplex,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + cleanFactor K (K * (K - 1)) m ν * Cmax
              * Real.sqrt ((K * (K - 1) : ℕ) / m)} :=
  gen_bound_subset_HFull_clean hK HVorSimplex VoronoiSimplex.HVorSimplex_subset_HFull
    C hC μ m ν hm hν hν1

/-- **Generalization bound for `H_Vor`** (prototypes anywhere), with capacity
`K (K − 1)`; `H_Vor = H_Full`. -/
theorem gen_bound_HVor (C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (μ : Measure (Simplex K × Fin K)) [IsProbabilityMeasure μ] (m : ℕ) (ν : ℝ)
    (hdm : levelDim (K * (K - 1)) K ≤ m) (hm : 0 < m) (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ HVor,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + Cmax * vcRate m (levelDim (K * (K - 1)) K) (ν / ((K : ℝ) * K))} :=
  gen_bound_subset_HFull HVor HVor_eq_HFull.le C hC μ m ν hdm hm hν

/-! ### The linear-scaling class

A scaling rule has scores `∑_j β_j C_{jk} p_j`. Moving the weights changes the
difference of two scores by `∑_j (β'_j − β_j)(C_{jk} − C_{jl}) p_j`, whose sign
at a tie depends on the point, so rational weights alone need not reproduce
the tie-break. The approximating rules add an offset `ε k` with `ε = 1/(N+1)`,
which dominates weight errors of order `1/(N+1)²`. The offset is one more
parameter, so the approximating class has Natarajan dimension at most `K`
(`natDim_ratHScAug_le`), not the `K − 1` of `thm:sc_complexity`. -/

section Scaling

variable {n : ℕ}

/-- The scaling rule with weights `(1, w)` and the tie-breaking offset `ε k`. -/
noncomputable def scAugRule (A : CostMatrix (n + 1)) (w : Fin n → ℝ) (ε : ℝ)
    (p : Simplex (n + 1)) : Fin (n + 1) :=
  lexArgmin fun k => (∑ j : Fin n, w j * (A j.succ k * p.val j.succ))
    + A 0 k * p.val 0 + ε * (k : ℕ)

/-- The augmented scaling rules with rational weights and offset. -/
def ratHScAug (A : CostMatrix (n + 1)) : Set (Simplex (n + 1) → Fin (n + 1)) :=
  Set.range fun x : (Fin n → ℚ) × ℚ => scAugRule A (fun j => (x.1 j : ℝ)) (x.2 : ℝ)

theorem ratHScAug_countable (A : CostMatrix (n + 1)) : (ratHScAug A).Countable :=
  Set.countable_range _

theorem ratHScAug_measurable (A : CostMatrix (n + 1)) :
    ∀ δ ∈ ratHScAug A, Measurable δ := by
  rintro _ ⟨x, rfl⟩
  unfold scAugRule
  refine measurable_lexArgmin fun k => ?_
  refine Measurable.add (Measurable.add ?_ ?_) measurable_const
  · exact Finset.measurable_sum _ fun j _ => measurable_const.mul
      (measurable_const.mul ((measurable_pi_apply _).comp measurable_val))
  · exact measurable_const.mul ((measurable_pi_apply _).comp measurable_val)

/-- The augmented class is a linear-score class in `n + 1 = K` parameters. -/
theorem natDim_ratHScAug_le (A : CostMatrix (n + 1)) :
    natDim (ratHScAug A) ≤ ((n + 1 : ℕ) : ℕ∞) := by
  refine natDim_le_of_linear_scores (ratHScAug A)
    (fun k (p : Simplex (n + 1)) =>
      Matrix.vecCons ((k : ℕ) : ℝ) fun j => A j.succ k * p.val j.succ)
    (fun k (p : Simplex (n + 1)) => A 0 k * p.val 0) ?_
  rintro _ ⟨x, rfl⟩
  refine ⟨Matrix.vecCons (x.2 : ℝ) fun j => (x.1 j : ℝ), fun p => ?_⟩
  show lexArgmin _ = lexArgmin _
  congr 1
  funext k
  rw [Fin.sum_univ_succ]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_succ]
  ring

/-- A weight error of at most `η` moves a score by at most `η D`,
`D = ∑ |C_{jk}|`. -/
theorem abs_weight_error_le (A : CostMatrix (n + 1)) {p : Fin (n + 1) → ℝ}
    (hp : p ∈ stdSimplex ℝ (Fin (n + 1))) {w w' : Fin n → ℝ} {η : ℝ} (hη : 0 ≤ η)
    (hw : ∀ j, |w' j - w j| ≤ η) (k : Fin (n + 1)) :
    |∑ j : Fin n, (w' j - w j) * (A j.succ k * p j.succ)|
      ≤ η * ∑ i, ∑ l, |A i l| := by
  have hp1 : ∀ i, |p i| ≤ 1 := fun i =>
    abs_le.2 ⟨by linarith [hp.1 i], VoronoiSimplex.le_one_of_mem_stdSimplex hp i⟩
  calc |∑ j : Fin n, (w' j - w j) * (A j.succ k * p j.succ)|
      ≤ ∑ j : Fin n, |(w' j - w j) * (A j.succ k * p j.succ)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j : Fin n, η * |A j.succ k| := by
        refine Finset.sum_le_sum fun j _ => ?_
        rw [abs_mul, abs_mul]
        have h1 := hw j
        have h2 := hp1 j.succ
        have h3 : 0 ≤ |A j.succ k| := abs_nonneg _
        calc |w' j - w j| * (|A j.succ k| * |p j.succ|)
            ≤ η * (|A j.succ k| * 1) := by
              gcongr
          _ = η * |A j.succ k| := by ring
    _ = η * ∑ j : Fin n, |A j.succ k| := by rw [Finset.mul_sum]
    _ ≤ η * ∑ i, ∑ l, |A i l| := by
        refine mul_le_mul_of_nonneg_left ?_ hη
        calc ∑ j : Fin n, |A j.succ k| ≤ ∑ i : Fin (n + 1), |A i k| := by
              rw [Fin.sum_univ_succ]
              linarith [abs_nonneg (A 0 k)]
          _ ≤ ∑ i, ∑ l, |A i l| := Finset.sum_le_sum fun i _ =>
              Finset.single_le_sum (f := fun l => |A i l|) (fun l _ => abs_nonneg _)
                (Finset.mem_univ k)

/-- Perturbations within `1/(N+1)²` of `k/(N+1)` satisfy the hypotheses of
`eventually_lexArgmin_add`. -/
theorem perturb_of_near {e : ℕ → Fin (n + 1) → ℝ}
    (h : ∀ (N : ℕ) (k : Fin (n + 1)),
      |e N k - ((k : ℕ) : ℝ) / ((N : ℝ) + 1)| ≤ 1 / ((N : ℝ) + 1) ^ 2) :
    (∀ k, Tendsto (fun N => e N k) atTop (𝓝 0)) ∧
      (∀ j k : Fin (n + 1), j < k → ∀ᶠ N in atTop, e N j < e N k) := by
  have h0 : Tendsto (fun N : ℕ => 1 / ((N : ℝ) + 1)) atTop (𝓝 0) :=
    tendsto_one_div_add_atTop_nhds_zero_nat
  have h2 : Tendsto (fun N : ℕ => 1 / ((N : ℝ) + 1) ^ 2) atTop (𝓝 0) := by
    have := h0.mul h0
    simp only [mul_zero] at this
    refine this.congr fun N => ?_
    rw [one_div, one_div, ← mul_inv, sq]
  refine ⟨fun k => ?_, fun j k hjk => ?_⟩
  · have hk : Tendsto (fun N : ℕ => ((k : ℕ) : ℝ) / ((N : ℝ) + 1)) atTop (𝓝 0) := by
      have := h0.const_mul ((k : ℕ) : ℝ)
      simp only [mul_zero, mul_one_div] at this
      exact this
    have hl := hk.sub h2
    have hu := hk.add h2
    simp only [sub_zero, add_zero] at hl hu
    exact tendsto_of_tendsto_of_tendsto_of_le_of_le hl hu
      (fun N => by have := (abs_le.1 (h N k)).1; linarith)
      (fun N => by have := (abs_le.1 (h N k)).2; linarith)
  · filter_upwards [eventually_ge_atTop 2] with N hN
    have hpos : (0 : ℝ) < (N : ℝ) + 1 := by positivity
    have hN' : (3 : ℝ) ≤ (N : ℝ) + 1 := by
      have : (2 : ℝ) ≤ N := by exact_mod_cast hN
      linarith
    have hjk' : ((j : ℕ) : ℝ) + 1 ≤ ((k : ℕ) : ℝ) := by
      have : (j : ℕ) + 1 ≤ (k : ℕ) := hjk
      exact_mod_cast this
    have e1 := (abs_le.1 (h N j)).2
    have e2 := (abs_le.1 (h N k)).1
    have hgap : 1 / ((N : ℝ) + 1)
        ≤ ((k : ℕ) : ℝ) / ((N : ℝ) + 1) - ((j : ℕ) : ℝ) / ((N : ℝ) + 1) := by
      rw [← sub_div]
      exact div_le_div_of_nonneg_right (by linarith) hpos.le
    have h3 : 2 * (1 / ((N : ℝ) + 1) ^ 2) < 1 / ((N : ℝ) + 1) := by
      have hsq : ((N : ℝ) + 1) ^ 2 = ((N : ℝ) + 1) * ((N : ℝ) + 1) := sq _
      rw [hsq, mul_one_div, div_lt_div_iff₀ (by positivity) hpos]
      nlinarith
    linarith

/-- **Every scaling rule is a pointwise limit of augmented rational ones**,
ties included. -/
theorem HSc_approx (A : CostMatrix (n + 1)) : ApproxBy (HSc A) (ratHScAug A) := by
  rintro δ ⟨β, -, hβ0, rfl⟩
  set D : ℝ := ∑ i, ∑ l, |A i l| with hD
  have hD0 : 0 ≤ D :=
    Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun l _ => abs_nonneg _
  set η : ℕ → ℝ := fun N => 1 / (((N : ℝ) + 1) ^ 2 * (D + 1)) with hη
  have hηpos : ∀ N, 0 < η N := fun N => by positivity
  have hq : ∀ (N : ℕ) (j : Fin n),
      ∃ q : ℚ, β j.succ - η N < q ∧ (q : ℝ) < β j.succ + η N :=
    fun N j => exists_rat_btwn (by linarith [hηpos N])
  choose q hq using hq
  let ε : ℕ → ℚ := fun N => 1 / ((N : ℚ) + 1)
  refine ⟨fun N => scAugRule A (fun j => (q N j : ℝ)) (ε N : ℝ),
    fun N => ⟨(q N, ε N), rfl⟩, fun p => ?_⟩
  -- The score of `δ_β`, with `β_0 = 1`.
  set f : Fin (n + 1) → ℝ := fun k =>
    (∑ j : Fin n, β j.succ * (A j.succ k * p.val j.succ)) + A 0 k * p.val 0 with hf
  have hfsc : scScore A β p.val = f := by
    funext k
    simp only [scScore, condRisk, f]
    rw [Fin.sum_univ_succ, hβ0, one_mul, add_comm]
    congr 1
    exact Finset.sum_congr rfl fun j _ => by ring
  set e : ℕ → Fin (n + 1) → ℝ := fun N k =>
    (∑ j : Fin n, ((q N j : ℝ) - β j.succ) * (A j.succ k * p.val j.succ))
      + (ε N : ℝ) * (k : ℕ) with he
  have hnear : ∀ (N : ℕ) (k : Fin (n + 1)),
      |e N k - ((k : ℕ) : ℝ) / ((N : ℝ) + 1)| ≤ 1 / ((N : ℝ) + 1) ^ 2 := by
    intro N k
    have hεN : (ε N : ℝ) = 1 / ((N : ℝ) + 1) := by simp [ε]
    have hsplit : e N k - ((k : ℕ) : ℝ) / ((N : ℝ) + 1)
        = ∑ j : Fin n, ((q N j : ℝ) - β j.succ) * (A j.succ k * p.val j.succ) := by
      simp only [e, hεN]
      ring
    rw [hsplit]
    refine (abs_weight_error_le A p.property (hηpos N).le
      (fun j => abs_le.2 ⟨by linarith [(hq N j).1], by linarith [(hq N j).2]⟩) k).trans ?_
    rw [← hD]
    have hpos : (0 : ℝ) < ((N : ℝ) + 1) ^ 2 := by positivity
    simp only [η]
    rw [div_mul_eq_mul_div, one_mul, div_le_div_iff₀ (by positivity) hpos]
    nlinarith
  obtain ⟨he0, hmono⟩ := perturb_of_near hnear
  filter_upwards [eventually_lexArgmin_add f e he0 hmono] with N hN
  show scAugRule A (fun j => (q N j : ℝ)) (ε N : ℝ) p = scRule A β p.val
  rw [scRule, hfsc, ← hN]
  unfold scAugRule
  congr 1
  funext k
  simp only [f, e]
  have hs : ∑ j : Fin n, (q N j : ℝ) * (A j.succ k * p.val j.succ)
      = ∑ j : Fin n, β j.succ * (A j.succ k * p.val j.succ)
        + ∑ j : Fin n, ((q N j : ℝ) - β j.succ) * (A j.succ k * p.val j.succ) := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [hs]
  ring

end Scaling

/-- **Generalization bound for the scaling class**, with capacity `K`. Every
rule of `H_Sc` (`eq:hsc_def`) satisfies, with probability at least `1 − ν`,
the bound of `Literature.gen_bound_natarajan` at `d = K`.

One more than `thm:sc_complexity`'s `d_Nat(H_Sc) ≤ K − 1`: the countable
class that approximates `H_Sc` pointwise carries a tie-breaking offset, which
is one more parameter. Whether the bound holds at `d = K − 1` for the
uncountable class is not settled here; it would need either a countable
subclass of `H_Sc` that reproduces every tie-break in the limit (rational
weights alone do not), or a measurability argument that does not pass
through a countable class. -/
theorem gen_bound_HSc (A C : CostMatrix K) {Cmax : ℝ} (hC : Bounded C Cmax)
    (μ : Measure (Simplex K × Fin K)) [IsProbabilityMeasure μ] (m : ℕ) (ν : ℝ)
    (hdm : levelDim K K ≤ m) (hm : 0 < m) (hν : 0 < ν) :
    1 - ENNReal.ofReal ν ≤ sampleMeasure μ m
      {S | ∀ δ ∈ HSc A,
        popRisk μ (costLoss C δ)
          ≤ empRisk S (costLoss C δ)
            + Cmax * vcRate m (levelDim K K) (ν / ((K : ℝ) * K))} := by
  obtain ⟨n, rfl⟩ : ∃ n, K = n + 1 :=
    ⟨K - 1, (Nat.succ_pred_eq_of_pos (Nat.pos_of_ne_zero (NeZero.ne K))).symm⟩
  exact gen_bound_natarajan_of_approx C hC (HSc A) (ratHScAug A) (ratHScAug_countable A)
    (ratHScAug_measurable A) (HSc_approx A) μ (n + 1) m ν (natDim_ratHScAug_le A)
    hdm hm hν

end ThresholdOpt.DecisionClassBounds
