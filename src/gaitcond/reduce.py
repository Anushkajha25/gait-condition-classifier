"""Standardise + PCA, fitted on training frames only.

The notebooks fit PCA on the whole dataset before splitting, which lets test-set
statistics shape the features. Here the scaler and PCA only ever see training rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_reducer(
    X, train_idx: np.ndarray, n_components: int | None = 256, max_fit_samples: int = 100_000,
    seed: int = 42,
) -> Pipeline:
    """Fit StandardScaler (+ PCA if ``n_components``) on a sample of the training rows.

    ``X`` can be a memory-mapped array; only the sampled training rows are loaded.
    """
    rng = np.random.default_rng(seed)
    fit_idx = np.asarray(train_idx)
    if len(fit_idx) > max_fit_samples:
        fit_idx = np.sort(rng.choice(fit_idx, size=max_fit_samples, replace=False))
    X_fit = np.asarray(X[fit_idx], dtype=np.float32)

    steps = [("scale", StandardScaler())]
    if n_components:
        n = min(n_components, X_fit.shape[0] - 1, X_fit.shape[1])
        steps.append(("pca", PCA(n_components=n, svd_solver="randomized", random_state=seed)))
    reducer = Pipeline(steps)
    reducer.fit(X_fit)
    return reducer


def transform(reducer: Pipeline, X, rows: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Apply the fitted reducer to ``X[rows]`` in chunks to bound memory."""
    rows = np.asarray(rows)
    out = []
    for start in range(0, len(rows), chunk):
        part = np.asarray(X[rows[start : start + chunk]], dtype=np.float32)
        out.append(reducer.transform(part).astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.empty((0, 0), dtype=np.float32)


def explained_variance(reducer: Pipeline) -> float | None:
    pca = reducer.named_steps.get("pca")
    return float(pca.explained_variance_ratio_.sum()) if pca is not None else None
