"""Custom model wrappers shared across studies.

TabPFNSizeFiltered: skips datasets that exceed TabPFN v2's pre-training limits.
TabPFN25SizeFiltered: same gating against TabPFN-2.5.
TabICLv2Classifier: skips datasets outside TabICLv2's tested regime.
TabMClassifier: sklearn-compatible wrapper around the TabM PyTorch model.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from experiments.common.pipeline import SkippedModelError


class TabPFNSizeFiltered(ClassifierMixin, BaseEstimator):
    """TabPFNClassifier that raises SkippedModelError on oversized datasets.

    TabPFN is a prior-fitted network trained on small tabular datasets.
    It should not be applied to datasets with more than ``max_samples``
    rows, ``max_features`` columns, or ``max_classes`` distinct labels.

    Parameters
    ----------
    max_samples : int
        Maximum number of training rows before the model is skipped.
    max_features : int
        Maximum number of features before the model is skipped.
    max_classes : int
        Maximum number of distinct labels before the model is skipped.
        TabPFN v2 is trained for up to 10-way classification.
    random_state : int or None
        Passed to TabPFNClassifier.
    """

    def __init__(
        self,
        max_samples: int = 10_000,
        max_features: int = 500,
        max_classes: int = 10,
        random_state: int | None = None,
    ):
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_classes = max_classes
        self.random_state = random_state

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier

        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape
        k = len(np.unique(y))
        if (
            n > self.max_samples
            or p > self.max_features
            or k > self.max_classes
        ):
            raise SkippedModelError(
                f"TabPFN skipped: dataset has {n} samples,"
                f" {p} features, {k} classes"
                f" (limits: {self.max_samples} samples,"
                f" {self.max_features} features,"
                f" {self.max_classes} classes)"
            )
        self.clf_ = TabPFNClassifier(random_state=self.random_state)
        self.clf_.fit(X, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.clf_.predict(np.asarray(X))


class TabPFN25SizeFiltered(ClassifierMixin, BaseEstimator):
    """TabPFN-2.5 classifier that skips oversized datasets.

    TabPFN-2.5 extends the pre-training scale of v2 by ~5x in samples and
    ~4x in features. We cap ``max_samples`` at TabPFN v2's 10k default to
    keep the in-context attention memory footprint bounded on large
    validation sets. The default ``TabPFNClassifier()`` now targets v2.6,
    so v2.5 must be selected explicitly via ``create_default_for_version``.

    Parameters
    ----------
    max_samples : int
        Maximum number of training rows before the model is skipped.
    max_features : int
        Maximum number of features before the model is skipped.
    max_classes : int
        Maximum number of distinct labels before the model is skipped.
        TabPFN-2.5 targets up to 10-way classification without the
        Many-Class extension.
    random_state : int or None
        Passed to TabPFNClassifier.
    """

    def __init__(
        self,
        max_samples: int = 10_000,
        max_features: int = 2_000,
        max_classes: int = 10,
        random_state: int | None = None,
    ):
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_classes = max_classes
        self.random_state = random_state

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape
        k = len(np.unique(y))
        if (
            n > self.max_samples
            or p > self.max_features
            or k > self.max_classes
        ):
            raise SkippedModelError(
                f"TabPFN-2.5 skipped: dataset has {n} samples,"
                f" {p} features, {k} classes"
                f" (limits: {self.max_samples} samples,"
                f" {self.max_features} features,"
                f" {self.max_classes} classes)"
            )
        self.clf_ = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, random_state=self.random_state
        )
        self.clf_.fit(X, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.clf_.predict(np.asarray(X))


class TabICLv2Classifier(ClassifierMixin, BaseEstimator):
    """TabICLv2 classifier with bracketing sample-size gates.

    TabICLv2 is a tabular foundation model pre-trained on datasets of
    300-48k rows and 2-100 columns; it scales to ~500k rows via offloading
    but is not evaluated below its 300-sample pre-training floor. The
    lower gate keeps the benchmark inside the model's tested regime; the
    upper ``max_samples`` gate bounds the in-context attention memory on
    large validation sets. The upper gates on features and classes match
    the convention used for TabPFN so very wide or very many-class inputs
    are opted out.

    Parameters
    ----------
    min_samples : int
        Minimum number of training rows before the model is skipped.
    max_samples : int or None
        Maximum number of training rows before the model is skipped.
        ``None`` disables the upper gate.
    max_features : int
        Maximum number of features before the model is skipped.
    max_classes : int
        Maximum number of distinct labels before the model is skipped.
    device : str or None
        Passed to TabICLClassifier. ``None`` lets TabICL auto-detect.
    random_state : int or None
        Passed to TabICLClassifier.
    """

    def __init__(
        self,
        min_samples: int = 300,
        max_samples: int | None = 10_000,
        max_features: int = 500,
        max_classes: int = 10,
        device: str | None = None,
        random_state: int | None = None,
    ):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_classes = max_classes
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
        from tabicl import TabICLClassifier

        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape
        k = len(np.unique(y))
        if (
            n < self.min_samples
            or (self.max_samples is not None and n > self.max_samples)
            or p > self.max_features
            or k > self.max_classes
        ):
            raise SkippedModelError(
                f"TabICL skipped: dataset has {n} samples,"
                f" {p} features, {k} classes"
                f" (limits: >={self.min_samples} and"
                f" <={self.max_samples} samples,"
                f" <={self.max_features} features,"
                f" <={self.max_classes} classes)"
            )
        self.clf_ = TabICLClassifier(
            random_state=self.random_state,
            device=self.device,
        )
        self.clf_.fit(X, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.clf_.predict(np.asarray(X))


class TabMClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible wrapper for the TabM deep tabular model (ICLR 2025).

    TabM is a batch-ensemble neural network that produces k independent
    predictions per sample and averages them at inference time.

    Reference: https://arxiv.org/abs/2407.02528

    Parameters
    ----------
    n_epochs : int
        Number of full passes over the training data.
    batch_size : int
        Mini-batch size for SGD.
    lr : float
        Adam learning rate.
    k : int
        Number of ensemble heads (TabM parameter).
    random_state : int or None
        Seeds torch for reproducibility.
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        k: int = 32,
        random_state: int | None = None,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F
        from tabm import TabM

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        # Remap labels to 0..n_classes-1
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_map[c] for c in y], dtype=np.int64)

        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_ = TabM.make(
            n_num_features=X.shape[1],
            cat_cardinalities=[],
            d_out=n_classes,
            k=self.k,
        ).to(self.device_)

        X_t = torch.from_numpy(X).to(self.device_)
        y_t = torch.from_numpy(y_idx).to(self.device_)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        # DataLoader(shuffle=True) uses a fresh CPU RNG each epoch unless
        # given an explicit generator. Without this, batch order drifts
        # across runs even when torch.manual_seed is set above.
        loader_kwargs = {"batch_size": self.batch_size, "shuffle": True}
        if self.random_state is not None:
            loader_kwargs["generator"] = torch.Generator().manual_seed(
                int(self.random_state)
            )
        loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for _ in range(self.n_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                # logits shape: (batch, k, n_classes)
                logits = self.model_(x_num=X_batch)
                # Flatten k predictions into the batch dimension for a
                # single efficient cross-entropy call.
                logits_flat = logits.reshape(-1, n_classes)
                y_flat = y_batch.unsqueeze(1).expand(-1, self.k).reshape(-1)
                loss = F.cross_entropy(logits_flat, y_flat)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F

        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(x_num=X_t)  # (n, k, n_classes)
            avg_logits = logits.mean(dim=1)  # (n, n_classes)
            probs = F.softmax(avg_logits, dim=1)

        return probs.cpu().numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
