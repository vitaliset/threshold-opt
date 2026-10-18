import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import Mathlib.Tactic
import ThresholdOpt.Multiclass

/-! # A conditional-risk minimizer reaches the simplex boundary -/

namespace ThresholdOpt.Multiclass

open ThresholdOpt

variable {K : ℕ}

/-- For `K ≥ 2`, a nonzero sum-zero direction along which the risk gaps
against action `k` do not increase. A square linear map is either invertible,
when all gaps can decrease together, or has a nonzero kernel. -/
theorem exists_risk_descent_direction (hK : 2 ≤ K) (C : CostMatrix K) (k : Fin K) :
    ∃ v : Fin K → ℝ, v ≠ 0 ∧ (∑ i, v i) = 0 ∧
      ∀ j, condRisk C v k - condRisk C v j ≤ 0 := by
  classical
  let T : (Fin K → ℝ) →ₗ[ℝ] (Fin K → ℝ) :=
    { toFun := fun v j => if j = k then ∑ i, v i
        else condRisk C v k - condRisk C v j
      map_add' := by
        intro v w
        ext j
        simp only [Pi.add_apply]
        split_ifs
        · exact Finset.sum_add_distrib
        · exact (isLinearMap_condRisk_sub C k j).map_add v w
      map_smul' := by
        intro a v
        ext j
        simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
        split_ifs
        · exact (Finset.mul_sum _ _ _).symm
        · exact (isLinearMap_condRisk_sub C k j).map_smul a v }
  have hzero : ∀ v, T v = 0 → (∑ i, v i) = 0 ∧
      ∀ j, condRisk C v k - condRisk C v j ≤ 0 := by
    intro v hv
    refine ⟨?_, ?_⟩
    · simpa [T] using congrFun hv k
    · intro j
      by_cases hj : j = k
      · simp [hj]
      · have := congrFun hv j
        simpa [T, hj] using le_of_eq this
  by_cases hinj : Function.Injective T
  · obtain ⟨v, hv⟩ := LinearMap.surjective_of_injective hinj
      (fun j => if j = k then (0 : ℝ) else -1)
    have hsum : (∑ i, v i) = 0 := by simpa [T] using congrFun hv k
    have hgap : ∀ j, condRisk C v k - condRisk C v j ≤ 0 := by
      intro j
      by_cases hj : j = k
      · simp [hj]
      · have := congrFun hv j
        simp only [T, LinearMap.coe_mk, AddHom.coe_mk, hj, ↓reduceIte] at this
        linarith
    refine ⟨v, ?_, hsum, hgap⟩
    intro heq
    have hne : ∃ j : Fin K, j ≠ k := by
      let a : Fin K := ⟨0, by omega⟩
      let b : Fin K := ⟨1, by omega⟩
      by_cases ha : a = k
      · refine ⟨b, ?_⟩
        intro hb
        have := congrArg Fin.val (ha.trans hb.symm)
        norm_num [a, b] at this
      · exact ⟨a, ha⟩
    obtain ⟨j, hj⟩ := hne
    have := congrFun hv j
    simp [T, heq, hj, condRisk] at this
  · obtain ⟨v, hv, hne⟩ : ∃ v, T v = 0 ∧ v ≠ 0 := by
      by_contra! h
      exact hinj (LinearMap.ker_eq_bot.mp (LinearMap.ker_eq_bot'.mpr h))
    exact ⟨v, hne, hzero v hv⟩

/-- A nonzero sum-zero direction from any simplex point reaches its boundary
at nonnegative time, while remaining in the simplex. -/
theorem exists_boundary_on_ray {p v : Fin K → ℝ}
    (hp : p ∈ stdSimplex ℝ (Fin K)) (hv : v ≠ 0) (hsum : (∑ i, v i) = 0) :
    ∃ t : ℝ, 0 ≤ t ∧ (fun i => p i + t * v i) ∈ stdSimplex ℝ (Fin K)
      ∧ ∃ i, p i + t * v i = 0 := by
  classical
  have hneg : ∃ i, v i < 0 := by
    by_contra! h
    apply hv
    funext i
    have hall : ∀ i ∈ (Finset.univ : Finset (Fin K)), v i = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => h i)).mp hsum
    exact hall i (Finset.mem_univ i)
  let A : Finset (Fin K) := Finset.univ.filter (fun i => v i < 0)
  have hA : A.Nonempty := by
    obtain ⟨i, hi⟩ := hneg
    exact ⟨i, by simp [A, hi]⟩
  obtain ⟨j, hj, hmin⟩ := Finset.exists_min_image A (fun i => p i / (-v i)) hA
  have hjv : v j < 0 := (Finset.mem_filter.mp hj).2
  let t := p j / (-v j)
  have ht : 0 ≤ t := div_nonneg (hp.1 j) (by linarith)
  refine ⟨t, ht, ⟨?_, ?_⟩, j, ?_⟩
  · intro i
    by_cases hi : v i < 0
    · have hle := hmin i (by simp [A, hi])
      have : t * (-v i) ≤ p i := (le_div_iff₀ (by linarith : 0 < -v i)).mp hle
      linarith
    · have : 0 ≤ t * v i := mul_nonneg ht (by linarith)
      linarith [hp.1 i]
  · simp only [Finset.sum_add_distrib, ← Finset.mul_sum, hsum, mul_zero, add_zero]
    exact hp.2
  · dsimp [t]
    field_simp [ne_of_lt hjv]
    <;> ring

/-- For any cost matrix with `K ≥ 2`, an action minimizing conditional risk
somewhere in the simplex also minimizes it at a boundary point. Minimization
allows ties. This is the general fact underlying `ex:hedge` and
`prop:elementwise_admissible`; it has no separate dissertation label. -/
theorem minimizer_exists_on_boundary (hK : 2 ≤ K) (C : CostMatrix K) (k : Fin K)
    {p : Fin K → ℝ} (hp : p ∈ stdSimplex ℝ (Fin K))
    (hmin : ∀ j, condRisk C p k ≤ condRisk C p j) :
    ∃ q ∈ stdSimplex ℝ (Fin K), (∃ i, q i = 0) ∧
      ∀ j, condRisk C q k ≤ condRisk C q j := by
  obtain ⟨v, hv, hsum, hgap⟩ := exists_risk_descent_direction hK C k
  obtain ⟨t, ht, hq, hboundary⟩ := exists_boundary_on_ray hp hv hsum
  refine ⟨_, hq, hboundary, ?_⟩
  intro j
  have hlinear : ∀ a, condRisk C (fun i => p i + t * v i) a
      = condRisk C p a + t * condRisk C v a := by
    intro a
    simp only [condRisk, mul_add, Finset.sum_add_distrib]
    congr 1
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun i _ => by ring)
  rw [hlinear, hlinear]
  have := mul_nonpos_of_nonneg_of_nonpos ht (hgap j)
  linarith [hmin j]

end ThresholdOpt.Multiclass
