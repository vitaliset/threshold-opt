"""Cross-validation fold-count helpers shared by strategies and the
experiments pipeline.

Kept out of :mod:`threshold_opt.base` (which holds base-class content)
because this is a stand-alone utility: the CV-tuning strategies
(``EmpiricalThreshold``, ``ThresholdRatioMulticlass``, the calibration
strategies) and the pipeline's fold precompute all need the same
"largest StratifiedKFold this label vector can honour" decision, and
all need to react to "CV is impossible here" identically.
"""

import numpy as np
import numpy.typing as npt


class InsufficientCVData(ValueError):
    """Raised when a class has too few members for the requested K-fold CV.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers
    still catch it, while callers that want to treat "CV is impossible on
    this split" distinctly (e.g. the experiments pipeline, which logs it
    as a countable ``CV_SKIP`` rather than a generic warning) can catch
    it by type first.
    """


def safe_n_splits(y: npt.ArrayLike, max_splits: int = 5) -> int:
    """Largest ``StratifiedKFold`` ``n_splits`` honourable on *y*.

    Returns ``min(max_splits, smallest_class_count)`` (never below 2).
    Raises :class:`InsufficientCVData` when the smallest class has fewer
    than 2 members, because no ``n_splits >= 2`` stratified split exists
    in that case -- the old ``max(2, ...)`` clamp returned 2 here and
    then crashed deep inside sklearn with a misleading ``ValueError`` /
    ``IndexError``.

    The result is a deterministic function of *y* alone, so the pipeline
    precompute and a strategy's own lazy refit (which key off the same
    fit-set labels) still derive the same K and the shared fold cache
    stays interchangeable with the lazy path.
    """
    counts = np.unique(np.asarray(y), return_counts=True)[1]
    if len(counts) == 0:
        raise InsufficientCVData("Cannot run CV: no samples.")
    min_count = int(counts.min())
    if min_count < 2:
        raise InsufficientCVData(
            f"Cannot run StratifiedKFold: smallest class has "
            f"{min_count} member(s), need >= 2."
        )
    return min(max_splits, min_count)
