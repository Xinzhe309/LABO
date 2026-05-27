"""Residual Gaussian process model."""

from __future__ import annotations

from typing import Tuple

import gpytorch
import numpy as np
import torch


class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


class ResidualGP:
    """GP over high-fidelity residuals y_h - rho * mu_lf."""

    def __init__(self, training_iter: int = 100, bounds: np.ndarray | None = None) -> None:
        self.training_iter = int(training_iter)
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=float)
        self.model = None
        self.likelihood = None
        self.X_train = None
        self.residuals_train = None
        self.X_mean = None
        self.X_std = None

    def fit(self, X: np.ndarray, residuals: np.ndarray) -> None:
        if len(X) == 0:
            raise ValueError("Training data cannot be empty.")
        X = np.asarray(X, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64)
        self.X_train = X
        self.residuals_train = residuals
        self._set_normalizer(X)

        train_x = torch.tensor(self._normalize(X), dtype=torch.float64)
        train_y = torch.tensor(residuals, dtype=torch.float64)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = _ExactGPModel(train_x, train_y, self.likelihood)
        self._train(train_x, train_y)

    def predict(self, X: np.ndarray, return_std: bool = False):
        mean, variance = self.predict_with_variance(X)
        if return_std:
            return mean, np.sqrt(variance)
        return mean

    def predict_with_variance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model has not been trained.")
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        test_x = torch.tensor(self._normalize(X), dtype=torch.float64)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(test_x)
        mean = posterior.mean.detach().cpu().numpy()
        variance = np.maximum(posterior.variance.detach().cpu().numpy(), 1e-12)
        if single:
            return float(mean[0]), float(variance[0])
        return mean, variance

    def _set_normalizer(self, X: np.ndarray) -> None:
        if self.bounds is not None:
            lower = self.bounds[:, 0]
            upper = self.bounds[:, 1]
            self.X_mean = (lower + upper) / 2.0
            self.X_std = np.maximum((upper - lower) / 2.0, 1e-6)
        else:
            self.X_mean = X.mean(axis=0)
            self.X_std = np.maximum(X.std(axis=0), 1e-6)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.X_mean) / self.X_std

    def _train(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for _ in range(self.training_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
