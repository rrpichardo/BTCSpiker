"""Rolling-window sequence classifier for the gated neural tournament stage.

Every other model family in ``btcspiker_ml.models`` classifies each row
independently. This estimator instead looks at the last ``window`` rows
(left-padded with zeros when history is short) through a small GRU, so it can
use short-term dynamics a row-independent model cannot see. It is
deliberately small ("modest", per the design spec) rather than a large
architecture search.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _GRUClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return torch.sigmoid(self.head(hidden[-1])).squeeze(-1)


class SequenceWindowClassifier:
    """Scikit-learn-compatible classifier over a rolling window of recent rows.

    ``fit``/``predict_proba`` accept plain row-independent tabular input, same
    as every other tournament model family. Internally, row *i* is windowed
    against rows ``[i - window + 1, i]`` of the same call's input array, so
    callers must pass rows in chronological order (as the temporal-fold
    machinery in ``scripts/run_experiments.py`` already does).
    """

    def __init__(
        self,
        hidden_size: int = 64,
        window: int = 20,
        seed: int = 0,
        epochs: int = 15,
        lr: float = 3e-3,
        batch_size: int = 256,
        n_jobs: int = 1,
    ):
        self.hidden_size = hidden_size
        self.window = window
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def _windows(self, x: np.ndarray) -> np.ndarray:
        n_rows, n_features = x.shape
        pad = np.zeros((self.window - 1, n_features), dtype=np.float32)
        padded = np.vstack([pad, x.astype(np.float32)])
        return np.stack([padded[i : i + self.window] for i in range(n_rows)])

    def fit(self, x, y) -> "SequenceWindowClassifier":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        torch.set_num_threads(max(1, int(self.n_jobs)))
        torch.manual_seed(self.seed)
        inputs = torch.from_numpy(self._windows(x))
        targets = torch.from_numpy(y)

        self.model_ = _GRUClassifier(n_features=x.shape[1], hidden_size=self.hidden_size)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()

        self.model_.train()
        n_rows = len(x)
        for _epoch in range(self.epochs):
            permutation = torch.randperm(n_rows)
            for start in range(0, n_rows, self.batch_size):
                batch = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                probabilities = self.model_(inputs[batch])
                loss = loss_fn(probabilities, targets[batch])
                loss.backward()
                optimizer.step()

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        inputs = torch.from_numpy(self._windows(x))
        self.model_.eval()
        with torch.no_grad():
            positive = self.model_(inputs).numpy()
        return np.column_stack([1.0 - positive, positive])
