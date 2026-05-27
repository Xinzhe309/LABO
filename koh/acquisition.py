"""Acquisition functions used by LABO."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.stats import norm


def compute_ei(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """Compute expected improvement for maximization."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    mu = np.asarray(mu, dtype=float)
    improvement = mu - float(y_best) - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def compute_ucb(mu: np.ndarray, sigma: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Compute upper confidence bound for maximization."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    return np.asarray(mu, dtype=float) + float(beta) * sigma


def compute_q_ei_greedy(
    candidates: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    y_best: float,
    q: int = 2,
    xi: float = 0.01,
    update_y_best: bool = True,
) -> List[int]:
    """Select a batch greedily by EI."""
    n_candidates = len(candidates)
    selected: List[int] = []
    available = np.ones(n_candidates, dtype=bool)
    current_best = float(y_best)
    for _ in range(min(q, n_candidates)):
        scores = np.full(n_candidates, -np.inf)
        scores[available] = compute_ei(mu[available], sigma[available], current_best, xi)
        best = int(np.argmax(scores))
        selected.append(best)
        available[best] = False
        if update_y_best:
            current_best = max(current_best, float(mu[best]))
    return selected


def select_next_points_q_ei(
    candidates: np.ndarray,
    mu: np.ndarray,
    variance: np.ndarray,
    y_best: float,
    q: int = 2,
    update_y_best: bool = True,
) -> Tuple[List[int], np.ndarray]:
    """Select a batch using greedy EI."""
    sigma = np.sqrt(np.maximum(variance, 1e-12))
    selected = compute_q_ei_greedy(candidates, mu, sigma, y_best, q, update_y_best=update_y_best)
    return selected, compute_ei(mu, sigma, y_best)


def compute_q_ucb_greedy(
    candidates: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    q: int = 2,
    beta: float = 2.0,
) -> List[int]:
    """Select a batch greedily by UCB."""
    n_candidates = len(candidates)
    selected: List[int] = []
    available = np.ones(n_candidates, dtype=bool)
    for _ in range(min(q, n_candidates)):
        scores = np.full(n_candidates, -np.inf)
        scores[available] = compute_ucb(mu[available], sigma[available], beta)
        best = int(np.argmax(scores))
        selected.append(best)
        available[best] = False
    return selected


def select_next_points_q_ucb(
    candidates: np.ndarray,
    mu: np.ndarray,
    variance: np.ndarray,
    q: int = 2,
    beta: float = 2.0,
) -> Tuple[List[int], np.ndarray]:
    """Select a batch using greedy UCB."""
    sigma = np.sqrt(np.maximum(variance, 1e-12))
    selected = compute_q_ucb_greedy(candidates, mu, sigma, q, beta)
    return selected, compute_ucb(mu, sigma, beta)
