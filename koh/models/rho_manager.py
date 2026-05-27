"""Scaling factor management for the KOH model."""

from __future__ import annotations

import numpy as np


class RhoManager:
    """Estimate and store the low-fidelity scaling factor rho."""

    def __init__(self, lower: float = 0.7, upper: float = 1.5) -> None:
        self.rho = 1.0
        self.lower = float(lower)
        self.upper = float(upper)

    def compute_rho(self, y_h: np.ndarray, mu_lf: np.ndarray, iteration: int = 0) -> float:
        """Least-squares estimate for y_h ~= rho * mu_lf."""
        y_h = np.asarray(y_h, dtype=float)
        mu_lf = np.asarray(mu_lf, dtype=float)
        if len(y_h) == 0 or len(y_h) != len(mu_lf):
            raise ValueError("y_h and mu_lf must be non-empty arrays with the same length.")

        denominator = float(np.dot(mu_lf, mu_lf))
        if denominator < 1e-12 or not np.isfinite(denominator):
            self.rho = 1.0
            return self.rho

        estimate = float(np.dot(mu_lf, y_h) / denominator)
        if not np.isfinite(estimate):
            self.rho = 1.0
        else:
            self.rho = float(np.clip(estimate, self.lower, self.upper))
        return self.rho

    def get_rho(self) -> float:
        return self.rho

    def set_rho(self, rho: float) -> None:
        self.rho = float(rho)
