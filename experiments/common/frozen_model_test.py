"""Equivalence gate for the predict-cache decoupling.

The hard contract: running the multiclass pipeline monolithically must
produce results bit-identical (to 1e-12) to running Phase A
(--dump-predictions) followed by Phase B (--from-predictions). This
guards the dissertation: the decoupled path is only an admissible
speed-up if it does not move a single objective value or learned
parameter.

Also exercises the clone / store-sharing invariants directly, since a
broken __sklearn_clone__ is the most likely way the gate would silently
pass on a model that does not trigger fold refits.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from experiments.common import frozen_model, pipeline
from experiments.multiclass import config as mc_config
from experiments.multiclass.runner import (
    STRATEGIES,
    _multiclass_param_extractor,
)

DATASET = "iris"
MASTER_SEED = 42
N_SEEDS = 1


def _cpu_models():
    # A pure-CPU classifier so the gate runs locally without a GPU.
    return {"LogisticRegression": LogisticRegression(random_state=42)}


# -- store-sharing / clone invariants -----------------------------------------


def test_recording_model_clone_shares_store():
    rec = frozen_model.RecordingModel(LogisticRegression())
    cloned = clone(rec)
    assert cloned.store is rec.store
    # Through a FrozenEstimator wrapper (the calibration code path).
    frozen = FrozenEstimator(rec)
    assert clone(frozen).estimator.store is rec.store


def test_recording_then_frozen_roundtrip_is_exact():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = rng.integers(0, 3, size=40)
    rec = frozen_model.RecordingModel(LogisticRegression(random_state=0))
    rec.fit(X, y)
    proba_a = rec.predict_proba(X)
    frozen = frozen_model.FrozenProbaModel(rec.store)
    frozen.fit(X, y)
    # Exact replay on the same array.
    np.testing.assert_array_equal(frozen.predict_proba(X), proba_a)
    # A non-contiguous / F-ordered view with identical float64 content
    # (what sklearn slicing produces internally) must hash to the same
    # key and return the same recorded array.
    f_order = np.asfortranarray(X)
    assert not f_order.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(frozen.predict_proba(f_order), proba_a)
    np.testing.assert_array_equal(
        frozen.predict_proba(X[np.arange(X.shape[0])]), proba_a
    )


def test_frozen_missing_key_raises():
    # A genuinely-fit model (state has recordings, so it is not treated
    # as a Phase A skip) must still raise on an input it never recorded.
    rng = np.random.default_rng(7)
    X, y = rng.normal(size=(8, 3)), rng.integers(0, 2, 8)
    rec = frozen_model.RecordingModel(LogisticRegression(random_state=0))
    rec.fit(X, y)
    rec.predict_proba(X)
    frozen = frozen_model.FrozenProbaModel(rec.store)
    frozen.fit(X, y)
    with pytest.raises(frozen_model.MissingPredictionError):
        frozen.predict_proba(np.ones((2, 3)))


def test_composite_key_disambiguates_fit_states():
    """Two models fit on different data, sharing one store, predicting
    the SAME input must NOT collide.

    Regression guard for the original divergence: a content-hash key on
    the input alone let the split-trained vs full-trained estimators
    (and every CV-fold refit) overwrite each other's predictions for an
    identical X. The (fit-state, method, input) key must keep them
    separate so each replays its own posterior.
    """
    rng = np.random.default_rng(0)
    X1, y1 = rng.normal(size=(60, 4)), rng.integers(0, 2, 60)
    X2, y2 = rng.normal(size=(60, 4)), rng.integers(0, 2, 60)
    Xq = rng.normal(size=(10, 4))  # the shared query input

    store: frozen_model.PredictionStore = {}
    m1 = frozen_model.RecordingModel(
        LogisticRegression(random_state=0), store=store
    ).fit(X1, y1)
    m2 = frozen_model.RecordingModel(
        LogisticRegression(random_state=0), store=store
    ).fit(X2, y2)
    p1 = m1.predict_proba(Xq)
    p2 = m2.predict_proba(Xq)
    assert not np.allclose(p1, p2), "test data too similar; models tie"

    # Replay: each fitted state recovers its OWN posterior for Xq.
    f1 = frozen_model.FrozenProbaModel(store)
    f1.fit(X1, y1)
    f2 = frozen_model.FrozenProbaModel(store)
    f2.fit(X2, y2)
    np.testing.assert_array_equal(f1.predict_proba(Xq), p1)
    np.testing.assert_array_equal(f2.predict_proba(Xq), p2)


def test_decision_function_capability_is_mirrored():
    """RecordingModel / FrozenProbaModel expose decision_function iff the
    bare estimator does, so sklearn's calibration picks the SAME
    response method wrapped, replayed, or bare.
    """
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(40, 3)), rng.integers(0, 2, 40)

    rec_lr = frozen_model.RecordingModel(LogisticRegression()).fit(X, y)
    rec_lr.predict_proba(X)
    rec_nb = frozen_model.RecordingModel(GaussianNB()).fit(X, y)
    rec_nb.predict_proba(X)
    assert hasattr(rec_lr, "decision_function")  # LogReg has one
    assert not hasattr(rec_nb, "decision_function")  # GaussianNB does not

    fr_lr = frozen_model.FrozenProbaModel(rec_lr.store)
    fr_lr.fit(X, y)
    fr_nb = frozen_model.FrozenProbaModel(rec_nb.store)
    fr_nb.fit(X, y)
    assert hasattr(fr_lr, "decision_function")
    assert not hasattr(fr_nb, "decision_function")


def test_decision_function_record_replay_is_exact():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(50, 4)), rng.integers(0, 3, 50)
    rec = frozen_model.RecordingModel(LogisticRegression(random_state=0))
    rec.fit(X, y)
    df_a = rec.decision_function(X)
    frozen = frozen_model.FrozenProbaModel(rec.store)
    frozen.fit(X, y)
    np.testing.assert_array_equal(frozen.decision_function(X), df_a)


def test_decision_function_callable_name_is_decision_function():
    """sklearn passes the response callable's ``__name__`` to
    ``cross_val_predict(method=...)``; it must read exactly
    'decision_function' or calibration raises a ValueError.
    """
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(30, 3)), rng.integers(0, 2, 30)
    rec = frozen_model.RecordingModel(LogisticRegression()).fit(X, y)
    rec.decision_function(X)
    assert rec.decision_function.__name__ == "decision_function"
    frozen = frozen_model.FrozenProbaModel(rec.store)
    frozen.fit(X, y)
    assert frozen.decision_function.__name__ == "decision_function"


# -- Phase A objective-subset contract ---------------------------------------


def test_representative_objectives_picks_one_metric_one_cost():
    """The full multiclass set has both metrics and cost matrices; the
    Phase A subset must be exactly the first of each category, with the
    objects preserved by identity (not rebuilt).
    """
    full = mc_config.METRIC_OBJECTIVES
    sub = pipeline._representative_objectives(full)

    metrics_mod = pipeline.metrics
    n_metric = sum(
        isinstance(o, metrics_mod.MetricObjective) for o in sub.values()
    )
    n_cost = len(sub) - n_metric
    assert n_metric == 1 and n_cost == 1, sub
    # Subset is drawn from the original, same object identity, and is a
    # strict reduction.
    for name, obj in sub.items():
        assert full[name] is obj
    assert len(sub) < len(full)


def test_representative_objectives_single_category_passthrough():
    metrics_mod = pipeline.metrics
    only_metrics = {
        k: v
        for k, v in mc_config.METRIC_OBJECTIVES.items()
        if isinstance(v, metrics_mod.MetricObjective)
    }
    sub = pipeline._representative_objectives(only_metrics)
    assert len(sub) == 1
    assert isinstance(next(iter(sub.values())), metrics_mod.MetricObjective)


# -- Phase A skip/fail symmetry ----------------------------------------------


class _SkippingEstimator(LogisticRegression):
    """A model that raises SkippedModelError on fit, like the TabPFN /
    TabICL wrappers do on size-limited datasets."""

    def fit(self, X, y, **kwargs):
        from experiments.common.errors import SkippedModelError

        raise SkippedModelError("simulated size-limit skip")


def test_skippedmodelerror_reexport_is_identical():
    """pipeline.SkippedModelError must stay the same object as the leaf
    errors module (model wrappers / tests import it from pipeline)."""
    from experiments.common import errors

    assert pipeline.SkippedModelError is errors.SkippedModelError
    assert frozen_model.SkippedModelError is errors.SkippedModelError


def test_recording_skip_records_nothing_and_frozen_reproduces_skip():
    """End-to-end of the real bug: a model that skips at fit time records
    no predictions in Phase A; FrozenProbaModel must reproduce the skip
    (raise SkippedModelError) so pipeline._train_model returns _SKIPPED
    in Phase B too -- instead of a MissingPredictionError storm.
    """
    from experiments.common.errors import SkippedModelError

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = rng.integers(0, 2, 20)

    store: frozen_model.PredictionStore = {}
    rec = frozen_model.RecordingModel(_SkippingEstimator(), store=store)
    with pytest.raises(SkippedModelError):
        rec.fit(X, y)
    # Phase A recorded nothing for this fitted state (no _CAPS_KEY, no
    # prediction keys) -- exactly what the on-disk cache looks like.
    assert store == {} or not any(
        k.startswith(f"{frozen_model.fit_signature(X, y)}-") for k in store
    )

    frozen = frozen_model.FrozenProbaModel(store)
    with pytest.raises(SkippedModelError):
        frozen.fit(X, y)


def test_frozen_skip_is_per_fitted_state():
    """A model skipped on the split fit but trained on the full fit (or
    vice versa) must skip only the missing state and replay the other.
    """
    from experiments.common.errors import SkippedModelError

    rng = np.random.default_rng(1)
    X_full = rng.normal(size=(40, 4))
    y_full = rng.integers(0, 2, 40)
    X_split = rng.normal(size=(25, 4))
    y_split = rng.integers(0, 2, 25)

    # Phase A: only the full fit succeeded and recorded predictions.
    rec = frozen_model.RecordingModel(LogisticRegression(random_state=0))
    rec.fit(X_full, y_full)
    rec.predict_proba(X_full)
    store = rec.store

    frozen_full = frozen_model.FrozenProbaModel(store)
    frozen_full.fit(X_full, y_full)  # has recordings -> no raise
    np.testing.assert_array_equal(
        frozen_full.predict_proba(X_full),
        store[
            frozen_model._cache_key(
                frozen_model.fit_signature(X_full, y_full), X_full, "proba"
            )
        ],
    )

    frozen_split = frozen_model.FrozenProbaModel(store)
    with pytest.raises(SkippedModelError):
        frozen_split.fit(X_split, y_split)  # no recordings -> skip


# -- the gate ----------------------------------------------------------------


def _run_monolithic(tmp_path):
    run_dir = tmp_path / "mono"
    run_dir.mkdir()
    seed = pipeline.seeds.load_or_sample_seeds(run_dir, N_SEEDS, MASTER_SEED)[
        0
    ]
    X, y = pipeline.data_loader.load_dataset(DATASET)
    results = pipeline.run_experiment_pipeline(
        dataset_name=DATASET,
        X=X,
        y=y,
        random_seed=seed,
        models=_cpu_models(),
        strategies=STRATEGIES,
        objectives=mc_config.METRIC_OBJECTIVES,
        param_extractor=_multiclass_param_extractor,
    )
    path = pipeline._task_result_path(run_dir, DATASET, seed)
    pipeline._save_task_results(results, path, "learned_params")
    return pd.read_parquet(path), seed


def _run_phase_a_then_b(tmp_path):
    cache_dir = tmp_path / "cache"
    run_a = tmp_path / "phase_a"
    run_a.mkdir()
    seed = pipeline.seeds.load_or_sample_seeds(run_a, N_SEEDS, MASTER_SEED)[0]

    pipeline._run_single_seed_phase_a(
        dataset_name=DATASET,
        random_seed=seed,
        models=_cpu_models(),
        strategies=STRATEGIES,
        objectives=mc_config.METRIC_OBJECTIVES,
        param_extractor=_multiclass_param_extractor,
        cache_dir=cache_dir,
        run_dir=run_a,
    )

    run_b = tmp_path / "phase_b"
    run_b.mkdir()
    pipeline._run_single_seed_phase_b(
        dataset_name=DATASET,
        random_seed=seed,
        model_names=list(_cpu_models().keys()),
        strategies=STRATEGIES,
        objectives=mc_config.METRIC_OBJECTIVES,
        param_extractor=_multiclass_param_extractor,
        cache_dir=cache_dir,
        run_dir=run_b,
    )
    path_b = pipeline._task_result_path(run_b, DATASET, seed)
    return pd.read_parquet(path_b), seed


_KEY_COLS = [
    "dataset_name",
    "model_name",
    "strategy_name",
    "optimization_objective_name",
    "random_seed",
]


def test_phase_ab_matches_monolithic(tmp_path, monkeypatch):
    # Equivalence of the record/replay mechanism is independent of the
    # random-search depth: a tiny n_iter exercises every strategy code
    # path (incl. the CV fold-refit / recording path) bit-for-bit while
    # keeping this gate runnable in seconds. Both the monolithic and the
    # Phase-A/B runs read the same patched constant, so the contract
    # (identical objective_value to 1e-12) is unaffected.
    monkeypatch.setattr(
        "experiments.multiclass.runner.RANDOM_SEARCH_N_ITER", 20
    )

    df_mono, seed_mono = _run_monolithic(tmp_path)
    df_ab, seed_ab = _run_phase_a_then_b(tmp_path)

    assert seed_mono == seed_ab
    assert len(df_mono) > 0, "monolithic run produced no rows"
    assert len(df_ab) == len(df_mono), (
        f"row count differs: mono={len(df_mono)} phase-ab={len(df_ab)}"
    )

    a = df_mono.sort_values(_KEY_COLS).reset_index(drop=True)
    b = df_ab.sort_values(_KEY_COLS).reset_index(drop=True)

    # Same (strategy, objective) coverage.
    pd.testing.assert_frame_equal(
        a[_KEY_COLS], b[_KEY_COLS], check_dtype=False
    )

    # objective_value: equal to 1e-12 for every row.
    obj_a = a["objective_value"].to_numpy(dtype=float)
    obj_b = b["objective_value"].to_numpy(dtype=float)
    max_diff = np.nanmax(np.abs(obj_a - obj_b)) if len(obj_a) else 0.0
    assert max_diff < 1e-12, (
        f"objective_value diverged (max abs diff {max_diff:.3e})"
    )

    # learned_params: equal to 1e-12 for every row (some are None).
    for i in range(len(a)):
        pa = a["learned_params"].iloc[i]
        pb = b["learned_params"].iloc[i]
        if pa is None and pb is None:
            continue
        arr_a = np.asarray(pa, dtype=float).ravel()
        arr_b = np.asarray(pb, dtype=float).ravel()
        assert arr_a.shape == arr_b.shape, (
            f"learned_params shape differs at row {i}"
            f" ({a['strategy_name'].iloc[i]} /"
            f" {a['optimization_objective_name'].iloc[i]})"
        )
        if arr_a.size:
            d = np.nanmax(np.abs(arr_a - arr_b))
            assert d < 1e-12, (
                f"learned_params diverged at row {i}"
                f" ({a['strategy_name'].iloc[i]} /"
                f" {a['optimization_objective_name'].iloc[i]}):"
                f" max abs diff {d:.3e}"
            )
