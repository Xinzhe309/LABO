"""General utility functions for LABO."""

from __future__ import annotations

from typing import List

import numpy as np


def sample_candidates(bounds: np.ndarray, n_samples: int, types: List[str] | None = None) -> np.ndarray:
    """Sample random candidate points from box bounds."""
    bounds = np.asarray(bounds, dtype=float)
    candidates = np.zeros((n_samples, len(bounds)), dtype=float)
    for index, (low, high) in enumerate(bounds):
        if types and types[index] == "int":
            candidates[:, index] = np.random.randint(int(low), int(high) + 1, size=n_samples)
        else:
            candidates[:, index] = np.random.uniform(low, high, size=n_samples)
    return candidates


def numpy_to_dict_list(X: np.ndarray, feature_names: List[str]) -> List[dict]:
    """Convert an array to a list of feature dictionaries."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return [{name: value for name, value in zip(feature_names, row)} for row in X]


def dict_list_to_numpy(X_dict: List[dict], feature_names: List[str]) -> np.ndarray:
    """Convert a list of feature dictionaries to an array."""
    return np.asarray([[point[name] for name in feature_names] for point in X_dict], dtype=float)


def compute_relative_error(y_true: np.ndarray, y_pred: np.ndarray, abs_threshold: float = 1e-6) -> np.ndarray:
    """Compute protected relative error."""
    denominator = np.maximum(np.abs(y_true), abs_threshold)
    return np.abs(y_true - y_pred) / denominator


def find_high_error_points(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rel_threshold: float = 0.1,
    abs_threshold: float = 0.06,
    top_k: int = 5,
) -> List[int]:
    """Return indices with large absolute and relative error."""
    rel_error = compute_relative_error(y_true, y_pred)
    abs_error = np.abs(y_true - y_pred)
    indices = np.where((rel_error > rel_threshold) & (abs_error > abs_threshold))[0]
    if len(indices) > top_k:
        indices = indices[np.argsort(rel_error[indices])[::-1]][:top_k]
    return indices.tolist()


def apply_types_to_array(X: np.ndarray, types: List[str]) -> np.ndarray:
    """Round integer dimensions in an array."""
    typed = np.asarray(X, dtype=float).copy()
    for index, dtype in enumerate(types):
        if dtype == "int":
            typed[:, index] = np.round(typed[:, index]).astype(int)
    return typed
