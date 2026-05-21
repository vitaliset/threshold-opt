import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.frozen import FrozenEstimator
from sklearn.utils._tags import get_tags

from experiments.common.model_wrappers import (
    TabICLv2Classifier,
    TabMClassifier,
    TabPFN25SizeFiltered,
    TabPFNSizeFiltered,
)
from experiments.common.pipeline import SkippedModelError


@pytest.mark.parametrize(
    "cls",
    [
        TabPFNSizeFiltered,
        TabPFN25SizeFiltered,
        TabICLv2Classifier,
        TabMClassifier,
    ],
)
class TestClassifierTags:
    """Wrappers must be recognised as classifiers by sklearn's tag system.

    ClassifierMixin must come before BaseEstimator in the MRO so that
    ``__sklearn_tags__`` sets ``estimator_type="classifier"``.  Without this,
    ``FrozenEstimator`` propagates ``None`` and ``CalibratedClassifierCV``
    rejects the estimator as a regressor.
    """

    def test_estimator_type_tag(self, cls):
        assert get_tags(cls()).estimator_type == "classifier"

    def test_is_classifier(self, cls):
        assert is_classifier(cls())

    def test_frozen_preserves_classifier_tag(self, cls):
        frozen = FrozenEstimator(cls())
        assert is_classifier(frozen)

    def test_clone_roundtrip(self, cls):
        original = cls()
        cloned = clone(original)
        assert type(cloned) is cls
        assert get_tags(cloned).estimator_type == "classifier"


# -- TabPFNSizeFiltered size gate ---------------------------------------------


class TestTabPFNSizeFiltered:
    """The size filter must fire before the real TabPFN import."""

    def test_skips_when_too_many_samples(self):
        wrapper = TabPFNSizeFiltered(max_samples=50, max_features=500)
        X = np.zeros((51, 3))
        y = np.zeros(51)
        with pytest.raises(SkippedModelError, match="51 samples"):
            wrapper.fit(X, y)

    def test_skips_when_too_many_features(self):
        wrapper = TabPFNSizeFiltered(max_samples=10_000, max_features=5)
        X = np.zeros((10, 6))
        y = np.zeros(10)
        with pytest.raises(SkippedModelError, match="6 features"):
            wrapper.fit(X, y)

    def test_skips_when_too_many_classes(self):
        wrapper = TabPFNSizeFiltered(
            max_samples=10_000, max_features=500, max_classes=10
        )
        X = np.zeros((11, 3))
        y = np.arange(11)
        with pytest.raises(SkippedModelError, match="11 classes"):
            wrapper.fit(X, y)


# -- TabPFN25SizeFiltered size gate -------------------------------------------


class TestTabPFN25SizeFiltered:
    """The size filter must fire before the real TabPFN-2.5 import."""

    def test_skips_when_too_many_samples(self):
        wrapper = TabPFN25SizeFiltered(max_samples=50, max_features=2_000)
        X = np.zeros((51, 3))
        y = np.zeros(51)
        with pytest.raises(SkippedModelError, match="51 samples"):
            wrapper.fit(X, y)

    def test_skips_when_too_many_features(self):
        wrapper = TabPFN25SizeFiltered(max_samples=10_000, max_features=5)
        X = np.zeros((10, 6))
        y = np.zeros(10)
        with pytest.raises(SkippedModelError, match="6 features"):
            wrapper.fit(X, y)

    def test_skips_when_too_many_classes(self):
        wrapper = TabPFN25SizeFiltered(
            max_samples=10_000, max_features=2_000, max_classes=10
        )
        X = np.zeros((11, 3))
        y = np.arange(11)
        with pytest.raises(SkippedModelError, match="11 classes"):
            wrapper.fit(X, y)


# -- TabICLv2Classifier size gate ---------------------------------------------


class TestTabICLv2Classifier:
    """The size filter must fire before the real TabICL import."""

    def test_skips_when_too_few_samples(self):
        wrapper = TabICLv2Classifier(min_samples=300)
        X = np.zeros((299, 3))
        y = np.zeros(299)
        with pytest.raises(SkippedModelError, match="299 samples"):
            wrapper.fit(X, y)

    def test_skips_when_too_many_samples(self):
        wrapper = TabICLv2Classifier(min_samples=1, max_samples=50)
        X = np.zeros((51, 3))
        y = np.zeros(51)
        with pytest.raises(SkippedModelError, match="51 samples"):
            wrapper.fit(X, y)

    def test_skips_when_too_many_features(self):
        wrapper = TabICLv2Classifier(min_samples=1, max_features=5)
        X = np.zeros((10, 6))
        y = np.zeros(10)
        with pytest.raises(SkippedModelError, match="6 features"):
            wrapper.fit(X, y)

    def test_skips_when_too_many_classes(self):
        wrapper = TabICLv2Classifier(
            min_samples=1, max_features=500, max_classes=10
        )
        X = np.zeros((11, 3))
        y = np.arange(11)
        with pytest.raises(SkippedModelError, match="11 classes"):
            wrapper.fit(X, y)
