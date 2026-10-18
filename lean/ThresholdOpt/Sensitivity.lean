import ThresholdOpt.Basic

/-!
# Misspecification sensitivity (Chapter 8)

The setting of the dimensional-dilution mechanism: the tangent space of the
simplex, the local-smoothness and diffuse-perturbation assumptions
(`ass:smoothness`, `ass:diffuse`), and the linearized cost functional whose
second moment the lemma computes.

The full assumptions and nonlinear pointwise dilution estimate are developed
in Diffusion.lean, using the inherited Euclidean tangent norm and explicit
standard sphere-coordinate second moments. This module supplies the algebra
for the centered cost gradient.

What *is* checked below is the part of Chapter 8 that is pure linear algebra:
the tangent space is exactly the sum-zero subspace, and the cost functional
`φ_δ` is affine there with gradient the centered cost column, which is the
verification `08:126` that `ass:smoothness` holds with zero remainder.
-/

namespace ThresholdOpt.Sensitivity

open ThresholdOpt

variable {K : ℕ}

/-- The tangent space of the simplex: perturbations that preserve total mass. -/
def tangent (K : ℕ) : Submodule ℝ (Fin K → ℝ) where
  carrier := {v | ∑ i, v i = 0}
  add_mem' := by
    intro a b ha hb
    show ∑ i, (a i + b i) = 0
    rw [Finset.sum_add_distrib, ha, hb, add_zero]
  zero_mem' := by simp
  smul_mem' := by
    intro c a ha
    show ∑ i, c * a i = 0
    rw [← Finset.mul_sum, ha, mul_zero]

@[simp] theorem mem_tangent {v : Fin K → ℝ} :
    v ∈ tangent K ↔ ∑ i, v i = 0 := Iff.rfl

/-- Staying on the simplex means moving in the tangent space. -/
theorem sub_mem_tangent {p q : Fin K → ℝ} (hp : p ∈ stdSimplex ℝ (Fin K))
    (hq : q ∈ stdSimplex ℝ (Fin K)) : (p - q) ∈ tangent K := by
  simp only [mem_tangent, Pi.sub_apply, Finset.sum_sub_distrib, hp.2, hq.2,
    sub_self]

/-- The mean of a cost column, subtracted to centre the gradient. -/
noncomputable def colMean (C : CostMatrix K) (k : Fin K) : ℝ :=
  (∑ i, C i k) / K

/-- The centred cost column `g = c_k - c̄_k 1`, the gradient of `φ_δ` claimed
at `08:126`. -/
noncomputable def gradient (C : CostMatrix K) (k : Fin K) : Fin K → ℝ :=
  fun i => C i k - colMean C k

/-- The centred gradient acts on tangent vectors exactly as the raw cost column
does: subtracting a multiple of `1` is invisible on the sum-zero subspace.

This is why `ass:smoothness` holds for `φ_δ` with *zero* remainder away from
the decision boundary: the functional is affine, so its first-order expansion
is exact. -/
theorem gradient_apply_tangent (C : CostMatrix K) (k : Fin K)
    {ε : Fin K → ℝ} (hε : ε ∈ tangent K) :
    ∑ i, gradient C k i * ε i = ∑ i, C i k * ε i := by
  have hsum : ∑ i, ε i = 0 := hε
  have hterm : ∀ i : Fin K,
      gradient C k i * ε i = C i k * ε i - colMean C k * ε i := by
    intro i; simp only [gradient]; ring
  rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_sub_distrib,
    ← Finset.mul_sum, hsum, mul_zero, sub_zero]

/-- The conditional risk moves affinely along a tangent perturbation, with no
remainder term. This is `ass:smoothness` for `φ_δ` with `M = 0`.

The uniform-sphere assumption and its quantitative consequences are formalized
in Diffusion.lean. This identity is an untagged algebraic supporting result. -/
theorem condRisk_add_tangent (C : CostMatrix K) (k : Fin K)
    (p : Fin K → ℝ) {ε : Fin K → ℝ} (hε : ε ∈ tangent K) :
    condRisk C (p + ε) k = condRisk C p k + ∑ i, gradient C k i * ε i := by
  rw [gradient_apply_tangent C k hε]
  simp only [condRisk, Pi.add_apply]
  have hterm : ∀ i : Fin K,
      C i k * (p i + ε i) = C i k * p i + C i k * ε i := by
    intro i; ring
  rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_add_distrib]

end ThresholdOpt.Sensitivity
