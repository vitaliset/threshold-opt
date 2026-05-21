from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import Bunch

from experiments.common import data_loader

# create_preprocessor


def test_create_preprocessor_numeric_only():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    pre = data_loader.create_preprocessor(X)
    X_proc = pre.fit_transform(X)
    assert X_proc.shape == (3, 2)


def test_create_preprocessor_categorical_only():
    X = pd.DataFrame({"cat": ["a", "b", "a"]})
    pre = data_loader.create_preprocessor(X)
    X_proc = pre.fit_transform(X)
    assert X_proc.shape[0] == 3
    assert X_proc.shape[1] >= 2  # at least 2 OHE columns for 2 unique values


def test_create_preprocessor_mixed():
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    pre = data_loader.create_preprocessor(X)
    X_proc = pre.fit_transform(X)
    assert X_proc.shape[0] == 3


# load_dataset_openml


def _make_openml_bunch(
    n=50, n_features=5, target_labels=None, extra_cols=None
):
    """Helper: minimal sklearn Bunch matching fetch_openml's output."""
    if target_labels is None:
        target_labels = ["yes", "no"] * (n // 2)
    X = pd.DataFrame(
        np.random.rand(n, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    if extra_cols:
        for col, vals in extra_cols.items():
            X[col] = vals
    y = pd.Series(target_labels[:n])
    return Bunch(data=X, target=y)


@pytest.mark.parametrize(
    "name",
    sorted(data_loader.DATASET_IDS.keys()),
)
def test_load_dataset_openml_supported(name):
    target_labels = None
    if name == "german_credit":
        target_labels = ["good", "bad"] * 25
    elif name == "pima_diabetes":
        target_labels = [0, 1] * 25

    bunch = _make_openml_bunch(target_labels=target_labels)

    with mock.patch(
        "experiments.common.data_loader.datasets.fetch_openml",
        autospec=True,
        return_value=bunch,
    ):
        X_raw, y_enc = data_loader.load_dataset_openml(name)

    assert isinstance(X_raw, pd.DataFrame)
    assert len(X_raw) == 50
    assert y_enc.dtype == np.int_
    assert set(y_enc).issubset({0, 1})


def test_load_dataset_openml_unsupported():
    with pytest.raises(ValueError, match="not supported"):
        data_loader.load_dataset_openml("nonexistent_dataset")


# load_raw_dataset


def test_load_raw_dataset_known_dataset_uses_openml():
    """Datasets in DATASET_IDS go directly to OpenML."""
    bunch = _make_openml_bunch()

    with mock.patch(
        "experiments.common.data_loader.load_dataset_openml",
        autospec=True,
        return_value=(bunch.data, np.zeros(50, dtype=int)),
    ) as mock_openml:
        X, _y = data_loader.load_raw_dataset("german_credit")

    # Verify the dataset name is forwarded to load_dataset_openml —
    # the whole point of this test is that known names dispatch to OpenML
    # with the right argument, not that the mock's return value flows through.
    mock_openml.assert_called_once_with("german_credit")
    assert len(X) == 50


# load_dataset — end-to-end (mocked)


def test_load_dataset_returns_float64_arrays():
    raw_X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.5, 1.5, 2.5]})
    raw_y = np.array([0, 1, 0])

    with mock.patch(
        "experiments.common.data_loader.load_raw_dataset",
        autospec=True,
        return_value=(raw_X, raw_y),
    ):
        X_proc, y_enc = data_loader.load_dataset("german_credit")

    assert X_proc.dtype == np.float64
    assert X_proc.shape[0] == 3
    assert len(y_enc) == 3


# Integration tests — hit real TFDS / OpenML (skipped unless --integration)


_INTEGRATION_DATASETS = sorted(data_loader.DATASET_IDS.keys())


@pytest.mark.integration
@pytest.mark.parametrize("name", _INTEGRATION_DATASETS)
def test_integration_load_raw_dataset(name):
    """load_raw_dataset returns valid arrays from real sources."""
    X, y = data_loader.load_raw_dataset(name)
    assert isinstance(X, pd.DataFrame), f"{name}: X should be a DataFrame"
    assert len(X) > 0, f"{name}: X is empty"
    assert len(y) == len(X), f"{name}: X and y length mismatch"
    assert y.dtype == np.int_, f"{name}: y should be int-encoded"
    assert set(y) == set(range(len(set(y)))), (
        f"{name}: y labels not contiguous from 0"
    )


@pytest.mark.integration
@pytest.mark.parametrize("name", _INTEGRATION_DATASETS)
def test_integration_load_dataset(name):
    """Verifies that load_dataset returns preprocessed float64 arrays."""
    X, y = data_loader.load_dataset(name)
    assert isinstance(X, np.ndarray), f"{name}: X should be ndarray"
    assert X.dtype == np.float64, f"{name}: X should be float64"
    assert not np.any(np.isnan(X)), (
        f"{name}: X contains NaN after preprocessing"
    )
    assert len(X) == len(y), f"{name}: X and y length mismatch"
    assert y.dtype == np.int_, f"{name}: y should be int"
