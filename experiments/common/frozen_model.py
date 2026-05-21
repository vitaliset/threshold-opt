"""Predict-cache decoupling: record / replay model predictions.

Splits an experiment task into two phases that, run back to back,
produce results bit-identical to the monolithic pipeline:

* Phase A (GPU-bound, ~21% of compute): fit every model and cache all
  of its ``predict_proba`` outputs. Wrap each study model in
  :class:`RecordingModel`, run the *unchanged* pipeline, then serialize
  the recorded ``{content_key: proba}`` store per (dataset, seed, model).
* Phase B (pure CPU, ~79%, embarrassingly parallel): rebuild a
  :class:`FrozenProbaModel` from the Phase A cache and run the *same*
  pipeline / strategy loop. No model ever fits; every ``predict_proba``
  is served from the cache.

Why this is bit-exact
---------------------
Every consumer in the codebase (decision strategies, the sklearn
calibration wrappers, the AUC/ECE/vertex diagnostics, the shared
K-fold precompute) touches a model *only* through ``fit(X, y)`` and
``predict_proba(X)``. For a fixed (dataset, seed) the preprocessing,
``train_test_split(random_state=seed)`` and
``StratifiedKFold(random_state=base.CV_RANDOM_STATE)`` are deterministic,
so the exact byte content of every ``X`` array handed to
``predict_proba`` is reproducible across runs. Keying the cache on the
content hash of that array therefore lets Phase B replay the identical
posterior for the identical input, including the fold subsets that
``CalibratedClassifierCV`` / ``cross_val_predict`` request internally.

Clone safety
------------
``sklearn.base.clone`` (used by the pipeline, the CV precompute, the
strategies and ``CalibratedClassifierCV``) would ``deepcopy`` any
non-estimator constructor argument, which would silently fork the
shared store. Both wrappers therefore implement ``__sklearn_clone__``
(the documented sklearn >= 1.3 hook) to return a fresh *unfitted*
proxy that keeps the very same store object by identity. This is the
same mechanism ``sklearn.frozen.FrozenEstimator`` uses, and it also
sidesteps the strict get_params/set_params round-trip check.
"""

import hashlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from experiments.common.errors import SkippedModelError

# A recorded store maps a content key (see ``proba_key``) to the exact
# ``predict_proba`` output produced for that input.
PredictionStore = Dict[str, np.ndarray]


def _content_hash(*arrays: npt.ArrayLike) -> str:
    """SHA1 over the C-contiguous float64 bytes (and shapes) of arrays.

    Each array is coerced to a C-contiguous ``float64`` buffer before
    hashing so that views, F-ordered slices, integer arrays or pandas
    frames all collapse to the same digest as long as the numeric
    content matches. Shapes are folded in so identical bytes with
    different shapes never collide.
    """
    h = hashlib.sha1()
    for a in arrays:
        arr = np.ascontiguousarray(np.asarray(a), dtype=np.float64)
        h.update(repr(arr.shape).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def proba_key(X: npt.ArrayLike) -> str:
    """Content hash identifying a ``predict_proba`` *input* array."""
    return _content_hash(X)


def fit_signature(X: npt.ArrayLike, y: npt.ArrayLike) -> str:
    """Content hash identifying the *fitted state* of a model.

    Two models fit on different ``(X, y)`` (the split-trained vs
    full-trained estimators, and every CV-fold refit) get distinct
    signatures, so a single shared store can hold their predictions for
    the *same* input ``X`` without colliding. Deterministic across
    Phase A / Phase B because the fit data is reproducible for a fixed
    ``(dataset, seed)``.
    """
    return _content_hash(X, y)


def _cache_key(fit_sig: str, X: npt.ArrayLike, method: str = "proba") -> str:
    """Composite ``(fitted-state, response-method, input)`` key.

    sha1 hex segments are fixed-length, so this is unambiguous and
    npz/zip-name safe. ``method`` namespaces ``predict_proba`` vs
    ``decision_function`` so the same (state, input) queried through
    different response methods never collides.
    """
    return f"{fit_sig}-{method}-{proba_key(X)}"


# Store-level key recording whether the wrapped estimator exposes
# ``decision_function``. sklearn's calibration picks its response method
# by ``hasattr(estimator, "decision_function")``; Phase B must mirror the
# bare estimator's availability exactly or it selects a different code
# path than Phase A (the cause of the calibrate-then-decide divergence).
_CAPS_KEY = "__has_decision_function__"


class RecordingModel(ClassifierMixin, BaseEstimator):
    """Transparent proxy that records every ``predict_proba`` output.

    ``fit`` delegates to a fresh clone of *base_estimator* and returns
    ``self``. ``predict_proba`` calls the fitted underlying model and
    stores ``proba_key(X) -> proba`` into the shared *store* before
    returning the array unchanged.

    The *store* is shared by object identity across the original and
    every ``sklearn.base.clone`` of it (and through any
    ``FrozenEstimator`` wrapper), so fold refits triggered inside
    strategies or ``CalibratedClassifierCV`` all record into the same
    place.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        store: Optional[PredictionStore] = None,
    ):
        self.base_estimator = base_estimator
        # A mutable container shared by reference. Created here when not
        # supplied so a bare ``RecordingModel(est)`` still works.
        self.store = {} if store is None else store

    # -- sklearn plumbing -------------------------------------------------

    def __sklearn_clone__(self) -> "RecordingModel":
        # Fresh, unfitted proxy that keeps the SAME store object. Bypasses
        # the default clone (which would deepcopy ``store``).
        return RecordingModel(self.base_estimator, store=self.store)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "fitted_estimator_")

    # -- estimator API ----------------------------------------------------

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike, **kwargs
    ) -> "RecordingModel":
        self.fitted_estimator_ = clone(self.base_estimator)
        self.fitted_estimator_.fit(X, y, **kwargs)
        # Signature of THIS fitted state: disambiguates the split- vs
        # full-trained estimators and every CV-fold refit, all of which
        # share one store and predict on overlapping/identical inputs.
        self.fit_sig_ = fit_signature(X, y)
        self.classes_ = self.fitted_estimator_.classes_
        if hasattr(self.fitted_estimator_, "n_features_in_"):
            self.n_features_in_ = self.fitted_estimator_.n_features_in_
        # Record (once per store) whether the underlying estimator
        # exposes decision_function, so Phase B can mirror availability
        # and sklearn's calibration picks the same response method.
        self.store[_CAPS_KEY] = np.array(
            [hasattr(self.fitted_estimator_, "decision_function")]
        )
        return self

    def predict_proba(self, X: npt.ArrayLike) -> np.ndarray:
        proba = np.asarray(self.fitted_estimator_.predict_proba(X))
        self.store[_cache_key(self.fit_sig_, X, "proba")] = proba
        return proba

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        # Strategies argmax/threshold the posterior themselves; predict
        # is kept for estimator completeness and is a pure delegate.
        return self.fitted_estimator_.predict(X)

    def __getattr__(self, name: str):
        # Reached only when normal lookup fails. Delegate to the fitted
        # underlying estimator so the proxy is transparent for anything
        # not explicitly intercepted -- crucially ``decision_function``:
        # its presence/absence must match the bare estimator exactly
        # (sklearn's calibration chooses its response method by
        # ``hasattr``). ``decision_function`` is wrapped to record into
        # the shared store like ``predict_proba``.
        if name.startswith("__") or name == "fitted_estimator_":
            raise AttributeError(name)
        fitted = self.__dict__.get("fitted_estimator_")
        if fitted is None:
            raise AttributeError(name)
        attr = getattr(fitted, name)  # AttributeError here matches bare
        if name == "decision_function" and callable(attr):

            def decision_function(X: npt.ArrayLike) -> np.ndarray:
                out = np.asarray(attr(X))
                self.store[_cache_key(self.fit_sig_, X, "decision")] = out
                return out

            # sklearn derives the cross_val_predict ``method=`` string
            # from the callable's __name__/__qualname__, so it must read
            # exactly "decision_function".
            decision_function.__name__ = "decision_function"
            decision_function.__qualname__ = "decision_function"
            return decision_function
        return attr


class MissingPredictionError(KeyError):
    """A Phase B lookup hit a key Phase A never recorded.

    Almost always means Phase A under-recorded: a ``predict_proba`` call
    site was reached in Phase B that did not run (or ran on a different
    input) during Phase A.
    """


class FrozenProbaModel(ClassifierMixin, BaseEstimator):
    """Replay-only model backed by a recorded prediction *store*.

    ``fit`` is a no-op (records ``classes_`` from *classes* or from the
    training labels so the sklearn calibration wrappers and diagnostics
    keep working). ``predict_proba`` returns the recorded array for
    ``proba_key(X)``; a missing key raises :class:`MissingPredictionError`
    because it means Phase A did not capture that input.

    The *store* is shared by identity across clones / ``FrozenEstimator``
    exactly like :class:`RecordingModel`.
    """

    def __init__(
        self,
        store: PredictionStore,
        classes: Optional[npt.ArrayLike] = None,
    ):
        self.store = store
        self.classes = classes

    # -- sklearn plumbing -------------------------------------------------

    def __sklearn_clone__(self) -> "FrozenProbaModel":
        return FrozenProbaModel(self.store, classes=self.classes)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "classes_")

    # -- estimator API ----------------------------------------------------

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike, **kwargs
    ) -> "FrozenProbaModel":
        # No real fitting; just record the fitted-state signature so the
        # replay can look up what the RecordingModel fit on this exact
        # (X, y) produced in Phase A.
        sig = fit_signature(X, y)

        # Phase A skip/fail symmetry. RecordingModel records predictions
        # (and _CAPS_KEY) only AFTER the wrapped model fits successfully;
        # a model that raised SkippedModelError (size limits) or failed
        # to fit recorded nothing for this fitted state. FrozenProbaModel
        # otherwise no-op-fits and lets the pipeline proceed, so a model
        # Phase A skipped triggers a MissingPredictionError storm in
        # Phase B. Detect "no key carries this fit signature's prefix"
        # and reproduce the skip, so pipeline._train_model returns
        # _SKIPPED in Phase B exactly as in Phase A and the (dataset,
        # seed, model) is absent from both runs identically. This keys
        # off the fit signature, so a model skipped on the split fit but
        # trained on the full fit (or vice versa) is handled per state.
        prefix = f"{sig}-"
        if not any(
            isinstance(k, str) and k.startswith(prefix) for k in self.store
        ):
            raise SkippedModelError(
                "Phase A recorded no predictions for this fitted state"
                " (model skipped on this dataset/seed -- e.g. size"
                " limits -- or its fit failed); replaying the skip so"
                " Phase B reproduces the monolithic run instead of"
                " raising MissingPredictionError."
            )

        self.fit_sig_ = sig
        if self.classes is not None:
            self.classes_ = np.asarray(self.classes)
        else:
            self.classes_ = np.unique(np.asarray(y))
        X_arr = np.asarray(X)
        if X_arr.ndim == 2:
            self.n_features_in_ = X_arr.shape[1]
        caps = self.store.get(_CAPS_KEY)
        self._has_decision_function = (
            bool(caps[0]) if caps is not None else False
        )
        return self

    def _replay(self, X: npt.ArrayLike, method: str) -> np.ndarray:
        if not hasattr(self, "fit_sig_"):
            raise MissingPredictionError(
                "FrozenProbaModel used before fit(); the fitted-state"
                " signature is unknown so the cached prediction cannot"
                " be located."
            )
        key = _cache_key(self.fit_sig_, X, method)
        try:
            return self.store[key]
        except KeyError:
            arr = np.asarray(X)
            raise MissingPredictionError(
                f"No recorded {method} for input with shape {arr.shape}"
                f" (key {key[:24]}...). Phase A did not capture this"
                " (fitted-state, method, input) triple; cache incomplete."
            ) from None

    def predict_proba(self, X: npt.ArrayLike) -> np.ndarray:
        return self._replay(X, "proba")

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        proba = self._replay(X, "proba")
        return self.classes_[np.argmax(proba, axis=1)]

    def __getattr__(self, name: str):
        # Mirror the bare estimator's decision_function availability so
        # sklearn's calibration selects the identical response method in
        # Phase B as Phase A. Only expose it when Phase A recorded that
        # the original estimator had one.
        if name != "decision_function":
            raise AttributeError(name)
        if not self.__dict__.get("_has_decision_function", False):
            raise AttributeError(name)

        def decision_function(X: npt.ArrayLike) -> np.ndarray:
            return self._replay(X, "decision")

        # Must read exactly "decision_function": sklearn passes this
        # callable's __name__ to cross_val_predict(method=...).
        decision_function.__name__ = "decision_function"
        decision_function.__qualname__ = "decision_function"
        return decision_function


# -- serialization ------------------------------------------------------------


def _sanitize(name: str) -> str:
    """Filesystem-safe token for a dataset / model name.

    Model keys can contain characters illegal on Windows (e.g.
    ``LGBMClassifier()``); map anything outside ``[A-Za-z0-9._-]`` to an
    underscore. The mapping only has to be stable within a single
    Phase A / Phase B pair, which it is.
    """
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def store_path(
    cache_dir: Path, dataset: str, seed: int, model_name: str
) -> Path:
    """Path of the per-(dataset, seed, model) prediction cache file."""
    return Path(cache_dir) / (
        f"{_sanitize(dataset)}__seed{seed}__{_sanitize(model_name)}.npz"
    )


def dump_store(
    store: PredictionStore,
    cache_dir: Path,
    dataset: str,
    seed: int,
    model_name: str,
) -> Path:
    """Serialize a recorded store to a per-(dataset, seed, model) npz.

    Keys are the hex content hashes; values are the exact posterior
    arrays. ``np.savez`` with an atomic rename so an interrupted Phase A
    never leaves a half-written cache file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = store_path(cache_dir, dataset, seed, model_name)
    # np.savez force-appends ".npz" unless the name already ends in it;
    # give the temp file a real ".npz" suffix so the written path is
    # exactly ``tmp`` and the atomic rename below targets a file that
    # exists.
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **store)
    tmp.replace(path)
    return path


def load_store(
    cache_dir: Path, dataset: str, seed: int, model_name: str
) -> PredictionStore:
    """Load a per-(dataset, seed, model) store from :func:`dump_store`."""
    path = store_path(cache_dir, dataset, seed, model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Phase A prediction cache not found: {path}. Run the"
            " --dump-predictions phase first."
        )
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
