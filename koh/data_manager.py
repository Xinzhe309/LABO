"""CSV-backed state management for LABO optimization runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class DataManager:
    """Store seed points, HF observations, and LF predictions."""

    def __init__(
        self,
        task_data_dir: str,
        feature_names: List[str],
        target_name: str = "objective",
        file_prefix: Optional[str] = None,
        objective_transform: float = 1.0,
    ) -> None:
        self.task_data_dir = Path(task_data_dir)
        self.feature_names = list(feature_names)
        self.target_name = target_name
        self.file_prefix = file_prefix
        self.objective_transform = float(objective_transform)
        if self.objective_transform == 0.0:
            raise ValueError("objective_transform cannot be zero.")

        self.seed_points_path = self._build_path("seed_points.csv")
        self.history_path = self._build_path("history.csv")
        self.hf_predictions_path = self._build_path("hf_predictions.csv")
        self.lf_predictions_path = self._build_path("lf_predictions.csv")

        self.seed_points_df = self._load_or_create(self.seed_points_path, self.feature_names + [self.target_name])
        self.history_df = self._load_or_create(self.history_path, self.feature_names + [self.target_name])
        self.hf_predictions_df = self._load_or_create(
            self.hf_predictions_path,
            self.feature_names + [self.target_name, "iteration", "best_objective"],
        )
        self.lf_predictions_df = self._load_or_create(
            self.lf_predictions_path,
            self.feature_names + ["mu_LF", "sigma2_LF", "iteration"],
        )

    def _build_path(self, filename: str) -> Path:
        self.task_data_dir.mkdir(parents=True, exist_ok=True)
        if self.file_prefix:
            filename = f"{self.file_prefix}_{filename}"
        return self.task_data_dir / filename

    def _load_or_create(self, path: Path, columns: List[str]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=columns)
        frame = pd.read_csv(path)
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
        for column in self.feature_names + [self.target_name, "mu_LF", "sigma2_LF", "iteration", "best_objective"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def save_all(self) -> None:
        self.seed_points_df.to_csv(self.seed_points_path, index=False)
        self.history_df.to_csv(self.history_path, index=False)
        self.hf_predictions_df.to_csv(self.hf_predictions_path, index=False)
        self.lf_predictions_df.to_csv(self.lf_predictions_path, index=False)

    def add_seed_point(self, x: dict, y_h: float) -> None:
        row = {**x, self.target_name: float(y_h)}
        if not self._point_exists(self.seed_points_df, x):
            self.seed_points_df = self._append_row(self.seed_points_df, row)
        if not self._point_exists(self.history_df, x):
            self.history_df = self._append_row(self.history_df, row)

    def add_hf_experiment(self, x: dict, y_h: float, iteration: int) -> None:
        if not self._point_exists(self.history_df, x):
            row = {**x, self.target_name: float(y_h)}
            self.history_df = self._append_row(self.history_df, row)
        if self._point_exists(self.hf_predictions_df, x):
            return
        previous_best = (
            float(self.hf_predictions_df["best_objective"].max())
            if len(self.hf_predictions_df) and self.hf_predictions_df["best_objective"].notna().any()
            else float(y_h)
        )
        row = {
            **x,
            self.target_name: float(y_h),
            "iteration": int(iteration),
            "best_objective": max(previous_best, float(y_h)),
        }
        self.hf_predictions_df = self._append_row(self.hf_predictions_df, row)

    def add_lf_prediction(self, x: dict, mu_lf: float, sigma2_lf: float, iteration: int) -> None:
        row = {**x, "mu_LF": float(mu_lf), "sigma2_LF": float(sigma2_lf), "iteration": int(iteration)}
        self.lf_predictions_df = self._append_row(self.lf_predictions_df, row)

    def get_history_data(self) -> List[Dict]:
        history: List[Dict] = []
        for _, row in self.history_df.iterrows():
            x = {name: row[name] for name in self.feature_names}
            y_true = float(row[self.target_name]) / self.objective_transform
            history.append({"x": x, "y": y_true})
        return history

    def get_history_exclude_point(self, x: dict) -> List[Dict]:
        if len(self.history_df) == 0:
            return []
        mask = np.ones(len(self.history_df), dtype=bool)
        for feature in self.feature_names:
            mask &= self.history_df[feature].to_numpy() == x[feature]
        filtered = self.history_df.loc[~mask]
        history: List[Dict] = []
        for _, row in filtered.iterrows():
            point = {name: row[name] for name in self.feature_names}
            y_true = float(row[self.target_name]) / self.objective_transform
            history.append({"x": point, "y": y_true})
        return history

    def get_history_points(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.history_df) == 0:
            return np.empty((0, len(self.feature_names))), np.empty(0)
        X = self.history_df[self.feature_names].to_numpy(dtype=float)
        y = self.history_df[self.target_name].to_numpy(dtype=float)
        return X, y

    def get_lf_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.lf_predictions_df) == 0:
            return np.empty((0, len(self.feature_names))), np.empty(0), np.empty(0)
        X = self.lf_predictions_df[self.feature_names].to_numpy(dtype=float)
        mu = self.lf_predictions_df["mu_LF"].to_numpy(dtype=float)
        sigma2 = self.lf_predictions_df["sigma2_LF"].to_numpy(dtype=float)
        valid = ~(np.isnan(mu) | np.isnan(sigma2))
        return X[valid], mu[valid], sigma2[valid]

    def recompute_all_lf_predictions(self, predictor, iteration: Optional[int] = None, only_new: bool = True) -> None:
        for _, row in self.history_df.iterrows():
            x = {name: row[name] for name in self.feature_names}
            if only_new and self._get_existing_lf_prediction(x) is not None:
                continue
            try:
                mean, variance, _ = predictor.predict(x, self.get_history_exclude_point(x))
                self.add_lf_prediction(x, mean, variance, 0 if iteration is None else iteration)
            except Exception:
                continue

    def recompute_non_history_lf_predictions(
        self,
        predictor,
        iteration: Optional[int] = None,
        only_if_history_changed: bool = True,
    ) -> None:
        if len(self.lf_predictions_df) == 0 or len(self.history_df) == 0:
            return
        points: Dict[tuple, dict] = {}
        for _, row in self.lf_predictions_df.iterrows():
            x = {name: row[name] for name in self.feature_names}
            if not self._point_exists(self.history_df, x):
                points[tuple(round(float(x[name]), 8) for name in self.feature_names)] = x
        if not points:
            return
        point_list = list(points.values())
        try:
            means, variances = predictor.predict_batch(point_list, self.get_history_data(), batch_size=20)
        except Exception:
            return
        for point, mean, variance in zip(point_list, means, variances):
            if not (np.isnan(mean) or np.isnan(variance)):
                self.add_lf_prediction(point, mean, variance, 0 if iteration is None else iteration)

    def _get_existing_lf_prediction(self, x: dict) -> Optional[dict]:
        if len(self.lf_predictions_df) == 0:
            return None
        mask = self._point_mask(self.lf_predictions_df, x)
        if not mask.any():
            return None
        row = self.lf_predictions_df.loc[mask].iloc[0]
        return {"mu_LF": row["mu_LF"], "sigma2_LF": row["sigma2_LF"], "iteration": row["iteration"]}

    def _point_exists(self, frame: pd.DataFrame, x: dict) -> bool:
        if len(frame) == 0:
            return False
        return bool(self._point_mask(frame, x).any())

    def _point_mask(self, frame: pd.DataFrame, x: dict) -> np.ndarray:
        mask = np.ones(len(frame), dtype=bool)
        for feature in self.feature_names:
            mask &= frame[feature].to_numpy() == x[feature]
        return mask

    @staticmethod
    def _append_row(frame: pd.DataFrame, row: dict) -> pd.DataFrame:
        row_frame = pd.DataFrame([row]).reindex(columns=frame.columns)
        if frame.empty:
            return row_frame.reset_index(drop=True)
        return pd.concat([frame, row_frame], ignore_index=True)
