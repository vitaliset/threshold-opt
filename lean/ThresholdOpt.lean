import ThresholdOpt.Attr
import ThresholdOpt.Basic
import ThresholdOpt.Binary
import ThresholdOpt.Multiclass
import ThresholdOpt.BoundaryMinimizer
import ThresholdOpt.Examples
import ThresholdOpt.Partition
import ThresholdOpt.Voronoi
import ThresholdOpt.Witness
import ThresholdOpt.Risk
import ThresholdOpt.Capacity
import ThresholdOpt.Learning
import ThresholdOpt.NatarajanLemma
import ThresholdOpt.McDiarmid
import ThresholdOpt.Massart
import ThresholdOpt.Sauer
import ThresholdOpt.Symmetrization
import ThresholdOpt.UniformConvergence
import ThresholdOpt.Literature
import ThresholdOpt.Complexity
import ThresholdOpt.CapacityLowerBounds
import ThresholdOpt.Mahalanobis
import ThresholdOpt.VoronoiSimplex
import ThresholdOpt.PointwiseApprox
import ThresholdOpt.SeparableBounds
import ThresholdOpt.DecisionClassBounds
import ThresholdOpt.Sensitivity
import ThresholdOpt.GaussianKL
import ThresholdOpt.GaussianKLGeneral
import ThresholdOpt.Diffusion
import ThresholdOpt.IntegratedDiffusion
import ThresholdOpt.SphereSymmetry
import ThresholdOpt.ConditionalLawExists
import ThresholdOpt.AffineSensitivity
import ThresholdOpt.SequentialSampler

/-!
# ThresholdOpt

A Lean 4 formalization of the mathematical results in the dissertation
*Argmax is not enough: cost-sensitive decision strategies for classification*.

Each formalized statement carries `@[thesis "<latex-label>"]`, linking it to the
environment it formalizes in `tex/content/`. `lake exe audit` reads those tags
back out and records every declaration's axiom closure; `make lean-audit`
renders the result into `tex/tables/formalization.tex`, the status table of
Appendix E.

Module map, following the dissertation's chapter order. `Literature` sits
between the capacity definitions and the bounds that consume them, so that the
import graph stays acyclic. All literature results used by the tagged
statements are proved; no literature axiom remains.

| Module        | Chapter                                             |
| ------------- | --------------------------------------------------- |
| `Attr`        | the `@[thesis]` attribute                           |
| `Basic`       | Ch. 1, the decision-theoretic setting               |
| `Binary`      | Ch. 2, the binary threshold                         |
| `Multiclass`  | Ch. 4, admissibility and the simplex partition      |
| `Examples`    | Ch. 4, the two cost-matrix witnesses                |
| `Partition`   | Ch. 5, the partition-matrix classes                 |
| `Voronoi`     | Ch. 6, Voronoi tessellations                        |
| `Risk`        | Ch. 1, the pointwise risk identities                |
| `Capacity`    | Ch. 1, 2, 6, App. C: capacity measures              |
| `Learning`    | Ch. 1, the sample, population and empirical risk    |
| `NatarajanLemma` | Ch. 6, from the Natarajan to the VC dimension  |
| `McDiarmid`   | App. D of Mohri et al.: McDiarmid's inequality      |
| `Massart`     | Rademacher averages and Massart's lemma             |
| `Sauer`       | Sauer's lemma for the growth function               |
| `Symmetrization` | the Rademacher bound (Mohri et al., Thm. 3.3)    |
| `UniformConvergence` | Mohri et al., Cor. 3.19, proved             |
| `Literature`  | results from the literature (no axiom remains)      |
| `Complexity`  | Ch. 6, App. C: the Natarajan-dimension bounds       |
| `CapacityLowerBounds` | Ch. 6: exact 0/1 scaling and quadratic Full/Voronoi witnesses |
| `Mahalanobis` | App. A: fixed quadratic metrics and a discrete-metric counterexample |
| `VoronoiSimplex` | Ch. 6, `H_Vor^Δ` is a proper subclass of `H_Full` |
| `PointwiseApprox` | bounds for classes approximated by countable ones |
| `SeparableBounds` | Ch. 1: binary displayed bound with pointwise approximation |
| `DecisionClassBounds` | Ch. 6, bounds for `H_PS`, `H_Sc`, `H_Full`, `H_Vor`, `H_Vor^Δ` |
| `Sensitivity` | Ch. 8, dimensional dilution                         |
| `GaussianKL`  | App. B, the Kullback-Leibler budgets                |
| `GaussianKLGeneral` | App. B, the Gaussian formula for general covariances |
| `Diffusion`   | Ch. 8, the uniform sphere and dimensional dilution  |
| `IntegratedDiffusion` | Ch. 8, the integrated dilution, corrected   |
| `SphereSymmetry` | Ch. 8, the second moments of the uniform sphere |
| `ConditionalLawExists` | Ch. 1, existence of the conditional class probabilities |
| `AffineSensitivity` | Ch. 8, smoothness of the cost functional off the boundary |
| `SequentialSampler` | Ch. 6, the sequential d² sampler              |
-/
