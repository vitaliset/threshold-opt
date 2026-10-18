import ThresholdOpt.Complexity

/-!
# Explicit lower bounds for scaling and unrestricted prototype rules

The points below lie in the probability simplex. Every realized labeling has
strict score margins, so no result depends on resolving a score tie.
-/

namespace ThresholdOpt.Complexity

open ThresholdOpt ThresholdOpt.Capacity ThresholdOpt.Partition

namespace ScalingWitness

noncomputable def point {n : ℕ} (i : Fin n) : Simplex (n + 1) :=
  ⟨Fin.cases (1 / 2) (fun j => if j = i then 1 / 2 else 0), by
    constructor
    · intro k
      refine Fin.cases (by norm_num) (fun j => ?_) k
      simp only [Fin.cases_succ]
      split_ifs <;> norm_num
    · rw [Fin.sum_univ_succ]
      simp only [Fin.cases_zero, Fin.cases_succ]
      simp
      norm_num⟩

@[simp] theorem point_zero {n : ℕ} (i : Fin n) : (point i).val 0 = 1 / 2 := rfl
@[simp] theorem point_succ {n : ℕ} (i j : Fin n) :
    (point i).val j.succ = if j = i then 1 / 2 else 0 := rfl

theorem point_injective {n : ℕ} : Function.Injective (point (n := n)) := by
  intro i j h
  have he := congrArg (fun p : Simplex (n + 1) => p.val i.succ) h
  by_contra hne
  simp [hne] at he

noncomputable def weights {n : ℕ} (b : Simplex (n + 1) → Bool) : Fin (n + 1) → ℝ :=
  Fin.cases 1 (fun i => if b (point i) then 2 else 1 / 2)

theorem weights_pos {n : ℕ} (b : Simplex (n + 1) → Bool) (k : Fin (n + 1)) :
    0 < weights b k := by
  refine Fin.cases (by norm_num [weights]) (fun i => ?_) k
  simp only [weights, Fin.cases_succ]
  split_ifs <;> norm_num

theorem rule_point {n : ℕ} (b : Simplex (n + 1) → Bool) (i : Fin n) :
    scRule (zeroOne (n + 1)) (weights b) (point i).val
      = if b (point i) then i.succ else 0 := by
  rw [scRule_zeroOne_eq_argmax]
  unfold lexArgmax
  by_cases hb : b (point i) = true
  · rw [if_pos hb]
    refine lexArgmin_eq_of_strict fun k hk => ?_
    rcases Fin.eq_zero_or_eq_succ k with rfl | ⟨j, rfl⟩
    · norm_num [weights, hb]
    · have hji : j ≠ i := fun h => hk (by subst h; rfl)
      simp [weights, hji, hb]
  · rw [if_neg hb]
    refine lexArgmin_eq_of_strict fun k hk => ?_
    rcases Fin.eq_zero_or_eq_succ k with rfl | ⟨j, rfl⟩
    · exact (hk rfl).elim
    · by_cases hji : j = i
      · subst hji
        norm_num [weights, hb]
      · simp [weights, hji]

theorem natDim_ge {n : ℕ} :
    (n : ℕ∞) ≤ natDim (HSc (zeroOne (n + 1))) := by
  classical
  let s := Finset.univ.image (point (n := n))
  have hcard : s.card = n := by
    rw [Finset.card_image_of_injective _ point_injective]
    simp
  rw [show (n : ℕ∞) = (s.card : ℕ∞) by exact_mod_cast hcard.symm]
  apply le_natDim
  let f : Simplex (n + 1) → Fin (n + 1) := fun p =>
    if h : ∃ i : Fin n, point i = p then (Classical.choose h).succ else 0
  have hf : ∀ i : Fin n, f (point i) = i.succ := by
    intro i
    have hi : ∃ j : Fin n, point j = point i := ⟨i, rfl⟩
    simp only [f, dif_pos hi]
    congr 1
    exact point_injective (Classical.choose_spec hi)
  refine ⟨f, fun _ => 0, ?_, fun b => ?_⟩
  · intro p hp
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 hp
    rw [hf]
    exact Fin.succ_ne_zero i
  · refine ⟨fun p => scRule (zeroOne (n + 1)) (weights b) p.val,
      ⟨weights b, weights_pos b, rfl, rfl⟩, ?_⟩
    intro p hp
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 hp
    rw [hf]
    exact rule_point b i

end ScalingWitness

/-- Under 0/1 loss the positive, gauge-fixed scaling class has exact
Natarajan dimension `K - 1`. No equality is asserted for general costs. -/
theorem natDim_HSc_zeroOne {K : ℕ} [NeZero K] :
    natDim (HSc (zeroOne K)) = ((K - 1 : ℕ) : ℕ∞) := by
  obtain ⟨n, rfl⟩ : ∃ n, K = n + 1 :=
    ⟨K - 1, (Nat.succ_pred_eq_of_pos (Nat.pos_of_ne_zero (NeZero.ne K))).symm⟩
  exact le_antisymm (natDim_HSc_le _) ScalingWitness.natDim_ge

end ThresholdOpt.Complexity
namespace ThresholdOpt.Complexity

open ThresholdOpt ThresholdOpt.Capacity ThresholdOpt.Partition

namespace FullWitness

/-- One point for every nonzero target class and every other coordinate. -/
noncomputable def point {n : ℕ} (q : Fin n × Fin n) : Simplex (n + 1) :=
  ⟨fun k => (3 / 4) * vertex q.1.succ k
    + (1 / 4) * vertex (q.1.succ.succAbove q.2) k, by
    constructor
    · intro k
      simp only [vertex]
      split_ifs <;> norm_num
    · simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
      rw [(vertex_mem_stdSimplex q.1.succ).2,
        (vertex_mem_stdSimplex (q.1.succ.succAbove q.2)).2]
      norm_num⟩

@[simp] theorem point_main {n : ℕ} (q : Fin n × Fin n) :
    (point q).val q.1.succ = 3 / 4 := by
  simp [point, Fin.ne_succAbove]

theorem point_other {n : ℕ} (q : Fin n × Fin n) (k : Fin (n + 1))
    (hk : k ≠ q.1.succ) : (point q).val k ≤ 1 / 4 := by
  simp only [point, vertex, if_neg hk, mul_zero, zero_add]
  split_ifs <;> norm_num

theorem point_injective {n : ℕ} : Function.Injective (point (n := n)) := by
  rintro ⟨i, j⟩ ⟨i', j'⟩ h
  have hi : i = i' := by
    by_contra hne
    have hk : i.succ ≠ i'.succ := fun h => hne (Fin.succ_inj.1 h)
    have ho := point_other (i', j') i.succ hk
    rw [← h, point_main] at ho
    norm_num at ho
  subst hi
  have hj : j = j' := by
    by_contra hne
    have hk : i.succ.succAbove j ≠ i.succ.succAbove j' := fun h =>
      hne (Fin.succAbove_right_inj.1 h)
    have he := congrArg (fun p : Simplex (n + 1) => p.val (i.succ.succAbove j)) h
    simp [point, vertex, Fin.succAbove_ne, hk] at he
  exact Prod.ext rfl hj

/-- Each column independently chooses a sign at each point on its own facet. -/
noncomputable def perturb {n : ℕ} (b : Simplex (n + 1) → Bool) (i : Fin n) :
    Fin (n + 1) → ℝ :=
  Fin.insertNth i.succ 0 (fun j => if b (point (i, j)) then -1 / 4 else 1 / 4)

theorem perturb_bounds {n : ℕ} (b : Simplex (n + 1) → Bool) (i : Fin n)
    (k : Fin (n + 1)) : -1 / 4 ≤ perturb b i k ∧ perturb b i k ≤ 1 / 4 := by
  rcases Fin.eq_self_or_eq_succAbove i.succ k with rfl | ⟨j, rfl⟩
  · norm_num [perturb]
  · simp only [perturb, Fin.insertNth_apply_succAbove]
    split_ifs <;> norm_num

noncomputable def matrix {n : ℕ} (b : Simplex (n + 1) → Bool) : CostMatrix (n + 1) :=
  Matrix.of fun r => Fin.cases 0 (fun i =>
    3 / 4 - (if r = i.succ then 1 else 0) + perturb b i r)

/-- Conditional risk at a witness is the weighted sum of two matrix rows. -/
theorem risk_point {n : ℕ} (b : Simplex (n + 1) → Bool) (q : Fin n × Fin n)
    (k : Fin (n + 1)) :
    condRisk (matrix b) (point q).val k =
      (3 / 4) * matrix b q.1.succ k
        + (1 / 4) * matrix b (q.1.succ.succAbove q.2) k := by
  simp only [condRisk, point, mul_add, Finset.sum_add_distrib]
  have hterm : ∀ (c : ℝ) (v : Fin (n + 1)),
      (∑ r, matrix b r k * (c * vertex v r)) = c * matrix b v k := by
    intro c v
    calc (∑ r, matrix b r k * (c * vertex v r))
        = c * ∑ r, matrix b r k * vertex v r := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun _ _ => by ring
      _ = c * matrix b v k := by rw [← condRisk, condRisk_vertex]
  rw [hterm, hterm]

@[simp] theorem risk_zero {n : ℕ} (b : Simplex (n + 1) → Bool) (q : Fin n × Fin n) :
    condRisk (matrix b) (point q).val 0 = 0 := by
  rw [risk_point]
  simp [matrix]

theorem risk_own {n : ℕ} (b : Simplex (n + 1) → Bool) (q : Fin n × Fin n) :
    condRisk (matrix b) (point q).val q.1.succ =
      if b (point q) then -1 / 16 else 1 / 16 := by
  rw [risk_point]
  simp only [matrix, Matrix.of_apply, Fin.cases_succ,
    perturb, Fin.insertNth_apply_same, add_zero,
    if_neg (Fin.succAbove_ne _ _), Fin.insertNth_apply_succAbove]
  split_ifs <;> norm_num

theorem risk_other {n : ℕ} (b : Simplex (n + 1) → Bool) (q : Fin n × Fin n)
    (k : Fin n) (hk : k ≠ q.1) :
    1 / 4 ≤ condRisk (matrix b) (point q).val k.succ := by
  rw [risk_point]
  have h1 := (perturb_bounds b k q.1.succ).1
  have h2 := (perturb_bounds b k (q.1.succ.succAbove q.2)).1
  have hne : q.1.succ ≠ k.succ := fun h => hk (Fin.succ_inj.1 h).symm
  simp only [matrix, Matrix.of_apply, Fin.cases_succ, if_neg hne]
  split_ifs <;> linarith

theorem rule_point {n : ℕ} (b : Simplex (n + 1) → Bool) (q : Fin n × Fin n) :
    lexArgmin (condRisk (matrix b) (point q).val)
      = if b (point q) then q.1.succ else 0 := by
  by_cases hb : b (point q) = true
  · rw [if_pos hb]
    refine lexArgmin_eq_of_strict fun k hk => ?_
    rcases Fin.eq_zero_or_eq_succ k with rfl | ⟨j, rfl⟩
    · rw [risk_zero, risk_own, if_pos hb]
      norm_num
    · have hj : j ≠ q.1 := fun h => hk (by subst h; rfl)
      have ho := risk_other b q j hj
      rw [risk_own, if_pos hb]
      linarith
  · rw [if_neg hb]
    refine lexArgmin_eq_of_strict fun k hk => ?_
    rcases Fin.eq_zero_or_eq_succ k with rfl | ⟨j, rfl⟩
    · exact (hk rfl).elim
    · by_cases hj : j = q.1
      · subst hj
        rw [risk_zero, risk_own, if_neg hb]
        norm_num
      · have ho := risk_other b q j hj
        rw [risk_zero]
        linarith

theorem natDim_ge {n : ℕ} :
    ((n ^ 2 : ℕ) : ℕ∞) ≤ natDim (HFull (K := n + 1)) := by
  classical
  let s := Finset.univ.image (point (n := n))
  have hcard : s.card = n ^ 2 := by
    rw [Finset.card_image_of_injective _ point_injective]
    simp [pow_two]
  rw [show ((n ^ 2 : ℕ) : ℕ∞) = (s.card : ℕ∞) by exact_mod_cast hcard.symm]
  apply le_natDim
  let f : Simplex (n + 1) → Fin (n + 1) := fun p =>
    if h : ∃ q : Fin n × Fin n, point q = p then (Classical.choose h).1.succ else 0
  have hf : ∀ q : Fin n × Fin n, f (point q) = q.1.succ := by
    intro q
    have hq : ∃ q' : Fin n × Fin n, point q' = point q := ⟨q, rfl⟩
    simp only [f, dif_pos hq]
    rw [point_injective (Classical.choose_spec hq)]
  refine ⟨f, fun _ => 0, ?_, fun b => ?_⟩
  · intro p hp
    obtain ⟨q, _, rfl⟩ := Finset.mem_image.1 hp
    rw [hf]
    exact Fin.succ_ne_zero q.1
  · refine ⟨fun p => lexArgmin (condRisk (matrix b) p.val), ⟨matrix b, rfl⟩, ?_⟩
    intro p hp
    obtain ⟨q, _, rfl⟩ := Finset.mem_image.1 hp
    rw [hf]
    exact rule_point b q

end FullWitness

/-- A direct edge witness proves a quadratic lower bound on the simplex,
including the case of two classes. -/
theorem natDim_HFull_ge_quadratic {K : ℕ} [NeZero K] :
    (((K - 1) ^ 2 : ℕ) : ℕ∞) ≤ natDim (HFull (K := K)) := by
  obtain ⟨n, rfl⟩ : ∃ n, K = n + 1 :=
    ⟨K - 1, (Nat.succ_pred_eq_of_pos (Nat.pos_of_ne_zero (NeZero.ne K))).symm⟩
  exact FullWitness.natDim_ge

/-- Unrestricted Voronoi rules inherit the quadratic lower bound by the
previously proved exact equality of decision-rule classes. -/
theorem natDim_HVor_ge_quadratic {K : ℕ} [NeZero K] :
    (((K - 1) ^ 2 : ℕ) : ℕ∞) ≤ natDim (HVor (K := K)) := by
  rw [HVor_eq_HFull]
  exact natDim_HFull_ge_quadratic

end ThresholdOpt.Complexity


namespace ThresholdOpt.Complexity

open ThresholdOpt ThresholdOpt.Capacity ThresholdOpt.Partition

/-- The scaling upper bound for arbitrary costs, which is what
`thm:sc_complexity` states, together with equality for 0/1 loss, which the
text does not claim. -/
@[thesis "thm:sc_complexity"]
theorem sc_complexity_sharp {K : ℕ} [NeZero K] (C : CostMatrix K) :
    natDim (HSc C) ≤ ((K - 1 : ℕ) : ℕ∞) ∧
    natDim (HSc (zeroOne K)) = ((K - 1 : ℕ) : ℕ∞) :=
  ⟨natDim_HSc_le C, natDim_HSc_zeroOne⟩

/-- Both explicit full-matrix bounds on the simplex, including K=2. -/
@[thesis "thm:full_complexity"]
theorem full_complexity_quadratic {K : ℕ} [NeZero K] :
    (((K - 1) ^ 2 : ℕ) : ℕ∞) ≤ natDim (HFull (K := K)) ∧
    natDim (HFull (K := K)) ≤ ((K * (K - 1) : ℕ) : ℕ∞) :=
  ⟨natDim_HFull_ge_quadratic, natDim_HFull_le_sharp⟩

/-- Unrestricted Voronoi has the dimension of the full-matrix class, which is
quadratic; simplex prototypes retain the proved upper bound. No restricted
lower bound is claimed. -/
@[thesis "thm:voronoi_complexity"]
theorem voronoi_complexity_quadratic {K : ℕ} [NeZero K] :
    natDim (HVor (K := K)) = natDim (HFull (K := K)) ∧
    (((K - 1) ^ 2 : ℕ) : ℕ∞) ≤ natDim (HVor (K := K)) ∧
    natDim (HVor (K := K)) ≤ ((K * (K - 1) : ℕ) : ℕ∞) ∧
    natDim (HVorSimplex (K := K)) ≤ ((K * (K - 1) : ℕ) : ℕ∞) :=
  ⟨by rw [HVor_eq_HFull], natDim_HVor_ge_quadratic, natDim_HVor_le_sharp⟩

end ThresholdOpt.Complexity
