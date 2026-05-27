"""Discrepancy gating for high- versus low-fidelity evaluation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class MismatchDecision:
    """Gate candidates using the discrepancy-dominance ratio."""

    def __init__(
        self,
        threshold: float = 0.8,
        consecutive_high_limit: int = 3,
        force_hf_after_n_lf: int | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.consecutive_high_limit = int(consecutive_high_limit)
        self.force_hf_after_n_lf = force_hf_after_n_lf
        self.consecutive_high_count = 0
        self.consecutive_lf_count = 0

    def compute_mismatch_ratio(self, sigma2_delta: np.ndarray, sigma2_h: np.ndarray) -> np.ndarray:
        """Compute p_delta = var_delta / var_high_fidelity."""
        sigma2_h = np.maximum(sigma2_h, 1e-9)
        return np.clip(sigma2_delta / sigma2_h, 0.0, 1.0)

    def decide(
        self,
        selected_indices: List[int],
        sigma2_delta: np.ndarray,
        sigma2_h: np.ndarray,
    ) -> Tuple[bool, float, List[float]]:
        """Return whether the selected batch should trigger HF evaluation."""
        ratios = [
            float(self.compute_mismatch_ratio(sigma2_delta[index], sigma2_h[index]))
            for index in selected_indices
        ]
        max_ratio = max(ratios) if ratios else 0.0
        force_hf = (
            self.force_hf_after_n_lf is not None
            and self.consecutive_lf_count >= self.force_hf_after_n_lf
        )
        do_hf = force_hf or max_ratio >= self.threshold
        if do_hf:
            self.consecutive_high_count += 1
            self.consecutive_lf_count = 0
        else:
            self.consecutive_high_count = 0
            self.consecutive_lf_count += 1
        return do_hf, max_ratio, ratios
