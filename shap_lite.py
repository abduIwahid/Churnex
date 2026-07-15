"""
shap_lite.py
------------
A tiny, dependency-free replacement for the pieces of the `shap` library
that Churnex uses (`shap.kmeans` and `shap.KernelExplainer`).

Why this exists
----------------
The real `shap` package pulls in `numba` -> `llvmlite`, which needs to
compile native code from source on Vercel's build image and repeatedly
broke deployment (e.g. "TypeError: spawn() got an unexpected keyword
argument 'dry_run'" while building llvmlite under newer Python/setuptools
combinations). Since Churnex only has 8 input features, we don't need an
approximation at all — we can compute EXACT Shapley values by enumerating
all 2^8 = 256 feature coalitions directly with numpy. This is both more
accurate than shap's default sampling-based KernelExplainer and has zero
compiled dependencies, so it installs instantly and reliably anywhere.

Public API (mirrors the subset of `shap` that Churnex relies on):
    - kmeans(X, k)                       -> background summary array
    - KernelExplainer(model_fn, background)
        .shap_values(X)                  -> array of shape (n_rows, n_features)
"""

from __future__ import annotations

from math import factorial

import numpy as np


def kmeans(X, k):
    """
    Summarize a dataset into `k` representative rows via k-means, mirroring
    shap.kmeans's role of producing a small background set for the
    explainer. Returns a plain numpy array of cluster centers.
    """
    from sklearn.cluster import KMeans

    X = np.asarray(X, dtype=float)
    k = max(1, min(k, X.shape[0]))

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    return km.cluster_centers_


class KernelExplainer:
    """
    Drop-in replacement for shap.KernelExplainer for low-dimensional
    tabular models. Computes exact Shapley values by full subset
    enumeration (feasible since Churnex uses only 8 features).

    Parameters
    ----------
    model_fn : callable
        Function mapping an (n_rows, n_features) array to an (n_rows,)
        array of model outputs (e.g. predicted probability of churn).
    background : array-like
        Reference/background dataset used to marginalize out "absent"
        features when evaluating a coalition.
    """

    def __init__(self, model_fn, background):
        self.model_fn = model_fn
        self.background = np.asarray(background, dtype=float)

    def shap_values(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        out = np.zeros_like(X, dtype=float)
        for row_idx in range(X.shape[0]):
            out[row_idx] = self._shapley_for_row(X[row_idx])
        return out

    def _shapley_for_row(self, instance: np.ndarray) -> np.ndarray:
        n = instance.shape[0]
        bg = self.background
        num_bg = bg.shape[0]
        num_subsets = 1 << n

        # Precompute, for every possible feature subset S (as a bitmask),
        # the "in subset" boolean pattern and its size.
        masks = np.array(
            [[(m >> i) & 1 for i in range(n)] for m in range(num_subsets)],
            dtype=bool,
        )  # (num_subsets, n)
        subset_sizes = masks.sum(axis=1)  # (num_subsets,)

        # Build every masked row (coalition S applied to `instance`, other
        # features taken from each background sample) in one big batch so
        # we only call the model once.
        # Shape: (num_subsets, num_bg, n)
        instance_b = np.broadcast_to(instance, (num_subsets, num_bg, n))
        bg_b = np.broadcast_to(bg, (num_subsets, num_bg, n))
        masks_b = masks[:, None, :]
        batch = np.where(masks_b, instance_b, bg_b).reshape(-1, n)

        preds = np.asarray(self.model_fn(batch), dtype=float).reshape(
            num_subsets, num_bg
        )
        f_values = preds.mean(axis=1)  # expected model output per coalition

        # Exact Shapley value for each feature via the standard weighted
        # sum over marginal contributions across all coalitions not
        # containing that feature.
        phi = np.zeros(n)
        fact_n = factorial(n)
        for i in range(n):
            without_i = ~masks[:, i]
            idx_without = np.nonzero(without_i)[0]
            for mask_idx in idx_without:
                s = subset_sizes[mask_idx]
                weight = factorial(s) * factorial(n - s - 1) / fact_n
                mask_with_i = mask_idx | (1 << i)
                phi[i] += weight * (f_values[mask_with_i] - f_values[mask_idx])

        return phi
