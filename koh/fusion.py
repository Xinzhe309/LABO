"""KOH fusion of low-fidelity and residual Gaussian processes."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .models.low_fidelity_gp import LowFidelityGP
from .models.residual_gp import ResidualGP
from .models.rho_manager import RhoManager


class KOHFusion:
    """Combine low-fidelity and residual predictions into an HF posterior."""

    def __init__(self, lf_gp: LowFidelityGP, residual_gp: ResidualGP, rho_manager: RhoManager) -> None:
        self.lf_gp = lf_gp
        self.residual_gp = residual_gp
        self.rho_manager = rho_manager

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return high-fidelity posterior mean and variance."""
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        rho = self.rho_manager.get_rho()
        mu_lf, sigma2_lf = self.lf_gp.predict_with_variance(X)
        mu_delta, sigma2_delta = self.residual_gp.predict_with_variance(X)
        mu_h = rho * mu_lf + mu_delta
        sigma2_h = rho**2 * sigma2_lf + sigma2_delta
        if single:
            return float(mu_h[0]), float(sigma2_h[0])
        return mu_h, sigma2_h

    def predict_lf_only(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.lf_gp.predict_with_variance(X)

    def predict_residual_only(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.residual_gp.predict_with_variance(X)
