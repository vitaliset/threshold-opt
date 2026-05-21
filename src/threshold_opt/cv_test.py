"""Unit tests for threshold_opt/cv.py."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from threshold_opt import cv


class TestSafeNSplits:
    """safe_n_splits returns a K that StratifiedKFold can actually honor."""

    def test_returns_max_splits_when_classes_are_large(self):
        y = np.repeat(np.arange(3), 10)
        assert cv.safe_n_splits(y, max_splits=5) == 5

    def test_caps_at_smallest_class_count(self):
        # Smallest class has 2 members -> only a 2-fold split is honourable.
        y = np.array([0, 0, 1, 1, 1, 2, 2])
        assert cv.safe_n_splits(y, max_splits=5) == 2

    def test_respects_custom_max_splits(self):
        y = np.repeat(np.arange(3), 50)
        assert cv.safe_n_splits(y, max_splits=3) == 3

    def test_two_member_smallest_class_is_the_boundary(self):
        y = np.array([0, 0, 1, 1])
        n = cv.safe_n_splits(y, max_splits=5)
        assert n == 2
        # The whole point: the returned K must not crash StratifiedKFold.
        list(StratifiedKFold(n_splits=n).split(np.zeros((4, 1)), y))

    def test_accepts_list_input(self):
        assert cv.safe_n_splits([0, 0, 0, 1, 1, 1], max_splits=5) == 3

    def test_single_class_with_enough_members(self):
        assert cv.safe_n_splits([0, 0, 0, 0], max_splits=5) == 4

    def test_raises_when_a_class_has_one_member(self):
        # Old behaviour clamped to 2, then StratifiedKFold(2) crashed
        # because a singleton class cannot be split. Now it raises a
        # typed, catchable error before any sklearn call.
        with pytest.raises(cv.InsufficientCVData):
            cv.safe_n_splits(np.array([0, 1, 2]), max_splits=5)

    def test_raises_on_empty(self):
        with pytest.raises(cv.InsufficientCVData):
            cv.safe_n_splits(np.array([], dtype=int), max_splits=5)

    def test_exception_is_a_valueerror_subclass(self):
        # Existing `except ValueError` handlers (and the pipeline's
        # generic ValueError branch) must still catch it.
        assert issubclass(cv.InsufficientCVData, ValueError)
        with pytest.raises(ValueError):
            cv.safe_n_splits([0, 1], max_splits=5)

    def test_message_reports_the_offending_count(self):
        with pytest.raises(cv.InsufficientCVData, match=r"smallest class"):
            cv.safe_n_splits([0, 0, 1], max_splits=5)
