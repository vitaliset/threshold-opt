import ThresholdOpt.Voronoi
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

/-!
# Sequential d² sampling
An explicit finite random tape supplies M simplex draws per stage. Swap a
maximizing coordinate into the stage position, take the first proposal passing
the text's two strict tests, and propagate retry exhaustion as none.
Correctness holds on every tape, independently of its distribution.
-/

namespace ThresholdOpt.SequentialSampler
open ThresholdOpt Voronoi MeasureTheory
open scoped Classical
abbrev Point (K : ℕ) := {p : Fin K → ℝ // p ∈ stdSimplex ℝ (Fin K)}
variable {K : ℕ}

/-- eq:seqd2_invariant after n successful stages: unique vertex minima. -/
def Invariant (n : ℕ) (μ : Prototypes K) : Prop :=
  ∀ k, k.val < n → ∀ j, j.val < n → j ≠ k →
    dsq (vertex k) (μ k) < dsq (vertex k) (μ j)

/-- Exactly the tests in eq:seqd2_accept. -/
def Accepts (μ : Prototypes K) (i : Fin K) (p : Point K) : Prop :=
  ∀ k, k.val < i.val →
    dsq (vertex k) (μ k) < dsq (vertex k) p.val ∧
      dsq (vertex i) p.val < dsq (vertex i) (μ k)

def permute (p : Point K) (e : Equiv.Perm (Fin K)) : Point K :=
  ⟨fun j => p.val (e j), ⟨fun j => p.property.1 (e j),
    (Fintype.sum_equiv e (fun j => p.val (e j)) p.val (fun _ => rfl)).trans p.property.2⟩⟩

/-- Smallest-index maximum selected before swapping; ties need not be absent. -/
noncomputable def swapMax [NeZero K] (i : Fin K) (p : Point K) : Point K :=
  permute p (Equiv.swap i (lexArgmax p.val))

theorem swapMax_maximal [NeZero K] (i : Fin K) (p : Point K) (j : Fin K) :
    (swapMax i p).val j ≤ (swapMax i p).val i := by
  simpa [swapMax, permute] using
    lexArgmax_ge p.val ((Equiv.swap i (lexArgmax p.val)) j)

noncomputable def firstAccepted (μ : Prototypes K) (i : Fin K) :
    List (Point K) → Option (Point K)
  | [] => none
  | p :: ps => if Accepts μ i p then some p else firstAccepted μ i ps

theorem firstAccepted_sound (μ : Prototypes K) (i : Fin K)
    (ps : List (Point K)) {p : Point K} (h : firstAccepted μ i ps = some p) :
    Accepts μ i p := by
  induction ps with
  | nil => simp [firstAccepted] at h
  | cons q qs ih =>
    simp only [firstAccepted] at h
    split at h
    · cases h; assumption
    · exact ih h

theorem invariant_update {n : ℕ} {μ : Prototypes K} (hμ : Invariant n μ)
    {i : Fin K} (hi : i.val = n) {p : Point K} (hp : Accepts μ i p) :
    Invariant (n + 1) (Function.update μ i p.val) := by
  classical
  intro k hk j hj hjk
  by_cases hki : k = i
  · subst k
    have hjn : j.val < n := by
      have hne : j.val ≠ n := fun he => hjk (Fin.ext (he.trans hi.symm))
      omega
    simp only [Function.update_self, Function.update_of_ne hjk]
    exact (hp j (by omega)).2
  · have hkn : k.val < n := by
      have hne : k.val ≠ n := fun he => hki (Fin.ext (he.trans hi.symm))
      omega
    rw [Function.update_of_ne hki]
    by_cases hji : j = i
    · subst j
      rw [Function.update_self]
      exact (hp k (by omega)).1
    · rw [Function.update_of_ne hji]
      exact hμ k hkn j (by
        have hne : j.val ≠ n := fun he => hji (Fin.ext (he.trans hi.symm))
        omega) hjk

abbrev Tape (K M : ℕ) := Fin K → Fin M → Point K
noncomputable def proposals [NeZero K] {M : ℕ} (draws : Tape K M) (i : Fin K) :
    List (Point K) := List.ofFn (fun m => swapMax i (draws i m))
@[simp] theorem proposals_length [NeZero K] {M : ℕ} (draws : Tape K M) (i : Fin K) :
    (proposals draws i).length = M := by simp [proposals]

/-- Unvisited slots start at vertices; acceptance never reads them. -/
noncomputable def build [NeZero K] {M : ℕ} (draws : Tape K M) :
    ℕ → Option (Prototypes K)
  | 0 => some vertex
  | n + 1 => if h : n < K then
      (build draws n).bind fun μ =>
        (firstAccepted μ ⟨n, h⟩ (proposals draws ⟨n, h⟩)).map fun p =>
          Function.update μ ⟨n, h⟩ p.val
    else none

@[thesis "eq:seqd2_invariant"]
theorem build_invariant [NeZero K] {M : ℕ} (draws : Tape K M) (n : ℕ)
    {μ : Prototypes K} (h : build draws n = some μ) :
    Invariant n μ ∧ ∀ k, μ k ∈ stdSimplex ℝ (Fin K) := by
  classical
  induction n generalizing μ with
  | zero =>
    simp only [build, Option.some.injEq] at h
    subst μ
    exact ⟨fun k hk => by omega, vertex_mem_stdSimplex⟩
  | succ n ih =>
    simp only [build] at h
    split at h
    · rename_i hn
      cases hprev : build draws n with
      | none => simp [hprev] at h
      | some prev =>
        simp only [hprev, Option.bind_some] at h
        cases hnext : firstAccepted prev ⟨n, hn⟩ (proposals draws ⟨n, hn⟩) with
        | none => simp [hnext] at h
        | some p =>
          simp only [hnext, Option.map_some, Option.some.injEq] at h
          subst μ
          obtain ⟨hinv, hsimplex⟩ := ih hprev
          refine ⟨invariant_update hinv rfl (firstAccepted_sound _ _ _ hnext), ?_⟩
          intro k
          by_cases hk : k = (⟨n, hn⟩ : Fin K)
          · subst k; simp only [Function.update_self]; exact p.property
          · simpa [Function.update_of_ne hk] using hsimplex k
    · simp at h

noncomputable def sample [NeZero K] {M : ℕ} (draws : Tape K M) :
    Option (Prototypes K) := build draws K

/-- Every returned configuration satisfies DLC, lies in the simplex and has
distinct prototypes. Does not assume acceptance or any distribution property. -/
@[thesis "prop:seqd2_correct"]
theorem seqd2_correct [NeZero K] {M : ℕ} (draws : Tape K M)
    {μ : Prototypes K} (h : sample draws = some μ) :
    DLC μ ∧ (∀ k, μ k ∈ stdSimplex ℝ (Fin K)) ∧ Function.Injective μ := by
  obtain ⟨hinv, hsimplex⟩ := build_invariant draws K h
  have hdlc : DLC μ := fun k j hj => hinv k k.isLt j j.isLt hj
  refine ⟨hdlc, hsimplex, ?_⟩
  intro i j heq
  by_contra hne
  have hlt := hdlc i j (Ne.symm hne)
  rw [heq] at hlt
  exact (lt_irrefl _) hlt

theorem seqd2_correct_ae [NeZero K] {M : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (draws : Ω → Tape K M) :
    ∀ᵐ ω ∂P, ∀ μ, sample (draws ω) = some μ → DLC μ :=
  Filter.Eventually.of_forall fun ω _ h => (seqd2_correct (draws ω) h).1

/-- Free coordinates with their ordinary n-dimensional Lebesgue measure. -/
def freeSimplex (n : ℕ) : Set (Fin n → ℝ) :=
  {w | (∀ j, 0 ≤ w j) ∧ ∑ j, w j ≤ 1}
def simplexChart (n : ℕ) (w : freeSimplex n) : Point (n + 1) :=
  ⟨Fin.cons (1 - ∑ j, w.val j) w.val, by
    constructor
    · intro i
      refine Fin.cases ?_ (fun j => ?_) i
      · simpa using sub_nonneg.mpr w.property.2
      · simpa using w.property.1 j
    · simp [Fin.sum_univ_succ]⟩

/-- Dirichlet(1) = normalized constant density on the free simplex.
Using n-dimensional coordinates avoids the zero (n+1)-dimensional volume.
The distribution is specified independently of the safety proof. -/
noncomputable def dirichletOne (n : ℕ) : Measure (Point (n + 1)) :=
  let μ : Measure (freeSimplex n) :=
    (volume.restrict (freeSimplex n)).comap Subtype.val
  ((μ Set.univ)⁻¹ • μ).map (simplexChart n)

/-- Independent Dirichlet draws pre-generated up to the retry budget. -/
noncomputable def tapeLaw (n M : ℕ) : Measure (Tape (n + 1) M) :=
  Measure.pi fun _ => Measure.pi fun _ => dirichletOne n

theorem seqd2_correct_dirichlet (n M : ℕ) :
    ∀ᵐ draws ∂tapeLaw n M, ∀ μ, sample draws = some μ → DLC μ :=
  seqd2_correct_ae (tapeLaw n M) id

end ThresholdOpt.SequentialSampler