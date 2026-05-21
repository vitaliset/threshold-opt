"""Shared exception types.

Leaf module with no intra-package imports, so both ``pipeline`` and
``frozen_model`` can import :class:`SkippedModelError` without creating an
import cycle (``pipeline`` imports ``frozen_model`` lazily; ``frozen_model``
must reproduce Phase A's skip outcome in Phase B and therefore needs to
raise the *same* exception type ``pipeline._train_model`` catches).
"""


class SkippedModelError(Exception):
    """Raised when a model intentionally skips a dataset (e.g. size limits)."""
