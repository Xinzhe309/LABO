"""Low-fidelity prediction wrapper."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class LowFidelityPredictor:
    """Produce low-fidelity means and variances from an LLM generator."""

    def __init__(
        self,
        generator,
        user_prompts,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        alpha: float = 1.0,
        beta: float = 0.0,
        y_transform: float = 1.0,
    ) -> None:
        if isinstance(user_prompts, (list, tuple)):
            if not user_prompts:
                raise ValueError("user_prompts cannot be empty")
            self.user_prompt = user_prompts[0]
        else:
            self.user_prompt = user_prompts
        self.generator = generator
        self.seed_candidates = list(range(1, 6))
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.alpha = alpha
        self.beta = beta
        self.y_transform = float(y_transform)

    def predict(self, x: dict, history: List[Dict] | None = None) -> Tuple[float, float, Dict]:
        """Predict a single point."""
        history_filtered = self._exclude_current_point(history, x)
        last_error: Exception | None = None
        for seed in self.seed_candidates:
            try:
                raw_value = self.generator.generate_single(
                    self.user_prompt,
                    x,
                    history_filtered,
                    seed=seed,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                mean = float(raw_value) * self.y_transform
                return mean, 0.0, {
                    "raw_value": float(raw_value),
                    "success_value": mean,
                    "seed_used": seed,
                    "fallback_used": False,
                    "input_x": x,
                    "history_size": len(history_filtered),
                }
            except Exception as exc:
                last_error = exc

        raw_value = self._fallback_from_history(x, history_filtered)
        mean = float(raw_value) * self.y_transform
        return mean, 0.0, {
            "raw_value": float(raw_value),
            "success_value": mean,
            "seed_used": None,
            "fallback_used": True,
            "input_x": x,
            "history_size": len(history_filtered),
            "last_error": str(last_error) if last_error else None,
        }

    def predict_batch(
        self,
        X_batch: List[dict],
        history: List[Dict] | None = None,
        batch_size: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict a batch, falling back to single-point calls when needed."""
        means: List[float] = []
        variances: List[float] = []
        for start in range(0, len(X_batch), batch_size):
            chunk = X_batch[start : start + batch_size]
            values = None
            last_error: Exception | None = None
            for seed in self.seed_candidates:
                try:
                    values = self.generator.generate_batch_multi_points(
                        user_prompt_template=self.user_prompt,
                        X_batch=chunk,
                        history=history,
                        seed=seed,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                    )
                    break
                except Exception as exc:
                    last_error = exc
            if values is None:
                for point in chunk:
                    try:
                        mean, variance, _ = self.predict(point, history)
                    except Exception:
                        mean, variance = np.nan, np.nan
                    means.append(float(mean))
                    variances.append(float(variance))
                continue

            for value in values:
                if value is None:
                    means.append(np.nan)
                    variances.append(np.nan)
                else:
                    means.append(float(value) * self.y_transform)
                    variances.append(0.0)

        return np.asarray(means, dtype=float), np.asarray(variances, dtype=float)

    @staticmethod
    def _is_same_point(a: dict, b: dict) -> bool:
        return bool(a) and bool(b) and a.keys() == b.keys() and all(a[key] == b[key] for key in a)

    def _exclude_current_point(self, history: List[Dict] | None, x: dict) -> List[Dict]:
        if not history:
            return []
        return [entry for entry in history if not self._is_same_point(entry.get("x", {}), x)]

    @staticmethod
    def _fallback_from_history(x: dict, history: List[Dict]) -> float:
        if not history:
            raise ValueError("No history is available for fallback prediction.")

        feature_order = list(x.keys())
        target = np.array([float(x[name]) for name in feature_order], dtype=float)
        best_entry = None
        best_distance = float("inf")
        for entry in history:
            x_hist = entry.get("x", {})
            vector = np.array([float(x_hist.get(name, 0.0)) for name in feature_order], dtype=float)
            distance = float(np.linalg.norm(vector - target))
            if distance < best_distance:
                best_distance = distance
                best_entry = entry
        if best_entry is None or "y" not in best_entry:
            raise ValueError("No valid history entry is available for fallback prediction.")
        return float(best_entry["y"])
