"""Main LABO optimization loop."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .acquisition import compute_ei, compute_ucb
from .data_manager import DataManager
from .decision import MismatchDecision
from .fusion import KOHFusion
from .models.low_fidelity_gp import LowFidelityGP
from .models.residual_gp import ResidualGP
from .models.rho_manager import RhoManager
from .utils import numpy_to_dict_list, sample_candidates
from low_fidelity.generator import LLMGenerator
from low_fidelity.predictor import LowFidelityPredictor
from low_fidelity.warmup import warmup_phase


class KOHOptimizer:
    """LLM-accelerated BO with KOH fusion and discrepancy gating."""

    def __init__(
        self,
        task_name: str,
        task_data_dir: str,
        feature_names: List[str],
        feature_types: List[str],
        bounds: np.ndarray,
        target_name: str,
        llm_client,
        hf_blackbox,
        llm_config,
        koh_config,
        file_prefix: Optional[str] = None,
        objective_transform: float = 1.0,
    ) -> None:
        self.task_name = task_name
        self.feature_names = list(feature_names)
        self.feature_types = list(feature_types)
        self.bounds = np.asarray(bounds, dtype=float)
        self.llm_client = llm_client
        self.hf_blackbox = hf_blackbox
        self.llm_config = llm_config
        self.koh_config = koh_config
        self.objective_transform = float(objective_transform)

        self.data_manager = DataManager(
            task_data_dir=task_data_dir,
            feature_names=self.feature_names,
            target_name=target_name,
            file_prefix=file_prefix,
            objective_transform=self.objective_transform,
        )

        from low_fidelity.prompt import load_prompts

        system_prompt, user_prompts = load_prompts(task_name)
        if not user_prompts:
            raise ValueError("No low-fidelity prompt template is available.")
        self.user_prompt = user_prompts[0]

        log_path = str(Path(task_data_dir) / f"{file_prefix}_llm_calls.jsonl") if file_prefix else None
        self.generator = LLMGenerator(
            llm_client=llm_client,
            system_prompt=system_prompt,
            value_range=getattr(llm_config, "value_range", None),
            log_path=log_path,
        )
        self.predictor = LowFidelityPredictor(
            generator=self.generator,
            user_prompts=self.user_prompt,
            temperature=getattr(llm_config, "temperature", 0.7),
            top_p=getattr(llm_config, "top_p", 0.9),
            max_tokens=getattr(llm_config, "max_tokens", 2048),
            alpha=getattr(llm_config, "alpha", 1.0),
            beta=getattr(llm_config, "beta", 0.0),
            y_transform=self.objective_transform,
        )

        gp_training_iter = int(getattr(koh_config, "gp_training_iter", 100))
        self.lf_gp = LowFidelityGP(training_iter=gp_training_iter, bounds=self.bounds)
        self.residual_gp = ResidualGP(training_iter=gp_training_iter, bounds=self.bounds)
        self.rho_manager = RhoManager()
        self.fusion: KOHFusion | None = None
        self.mismatch_decision = MismatchDecision(
            threshold=float(getattr(koh_config, "mismatch_threshold", 0.8)),
            force_hf_after_n_lf=getattr(koh_config, "force_hf_after_n_lf", None),
        )

        self.main_random_seed = getattr(koh_config, "random_seed", None)
        self.always_update_lf_loops = int(getattr(koh_config, "always_update_lf_loops", 1))
        self.acquisition_type = str(getattr(koh_config, "acquisition_type", "ucb")).lower()
        self.acquisition_beta = float(getattr(koh_config, "acquisition_beta", 2.0))
        self.iteration_log: List[dict] = []
        self._seed_candidates_cache: dict[int, np.ndarray] = {}
        self._last_was_hf = True

    def run(
        self,
        max_iterations: int = 25,
        n_initial_points: int = 5,
        q: int = 2,
        fixed_initial_points: Optional[List[dict]] = None,
    ) -> None:
        """Run the optimizer until the HF budget or loop limit is reached."""
        warmup_phase(
            llm_client=self.llm_client,
            hf_blackbox=self.hf_blackbox,
            data_manager=self.data_manager,
            generator=self.generator,
            user_prompt=self.user_prompt,
            task_name=self.task_name,
            feature_names=self.feature_names,
            n_initial_points=n_initial_points,
            temperature=getattr(self.llm_config, "temperature", 0.7),
            top_p=getattr(self.llm_config, "top_p", 0.9),
            max_tokens=getattr(self.llm_config, "max_tokens", 2048),
            fixed_initial_points=fixed_initial_points,
            y_transform=self.objective_transform,
        )

        if self.main_random_seed is not None:
            np.random.seed(int(self.main_random_seed))

        hf_iterations = 0
        loop_count = 0
        max_loops = int(getattr(self.koh_config, "max_loops", max(1, max_iterations * 3)))

        while hf_iterations < max_iterations and loop_count < max_loops:
            loop_count += 1
            self._train_models(
                force_recompute_lf=self._last_was_hf or loop_count <= self.always_update_lf_loops,
                loop_count=loop_count,
            )
            candidates = self._sample_candidates(
                n_samples=int(getattr(self.koh_config, "n_candidates", 5000)),
                loop_count=loop_count,
            )
            if len(candidates) == 0:
                break

            mu_h, sigma2_h, _mu_delta, sigma2_delta = self._koh_posterior_predict(candidates)
            selected_indices, _scores = self._select_q_points(candidates, mu_h, sigma2_h, q=q)
            x_next = [candidates[index] for index in selected_indices]
            do_hf, ratio, _ratios = self.mismatch_decision.decide(selected_indices, sigma2_delta, sigma2_h)

            if do_hf:
                hf_iterations += 1
                self._high_fidelity_branch(x_next, hf_iterations)
                self._last_was_hf = True
            else:
                self._low_fidelity_branch(x_next, hf_iterations)
                self._last_was_hf = False

            self._log_iteration(hf_iterations, x_next, do_hf, ratio)

    def _train_models(self, force_recompute_lf: bool = True, loop_count: int = 0) -> None:
        if force_recompute_lf:
            self.data_manager.recompute_all_lf_predictions(self.predictor, iteration=loop_count, only_new=True)
            self.data_manager.recompute_non_history_lf_predictions(self.predictor, iteration=loop_count)
            self.data_manager.save_all()

        X_lf, mu_lf, sigma2_lf = self.data_manager.get_lf_training_data()
        if len(X_lf) == 0:
            raise ValueError("No low-fidelity data is available.")
        self.lf_gp.fit(X_lf, mu_lf, sigma2_lf)

        X_hf, y_h = self.data_manager.get_history_points()
        if len(X_hf) == 0:
            raise ValueError("No high-fidelity observations are available.")

        mu_lf_hf = []
        for row in X_hf:
            point = {name: value for name, value in zip(self.feature_names, row)}
            existing = self.data_manager._get_existing_lf_prediction(point)
            if existing is None:
                mean, variance, _ = self.predictor.predict(point, self.data_manager.get_history_exclude_point(point))
                self.data_manager.add_lf_prediction(point, mean, variance, iteration=loop_count)
                mu_lf_hf.append(mean)
            else:
                mu_lf_hf.append(float(existing["mu_LF"]))

        mu_lf_hf_array = np.asarray(mu_lf_hf, dtype=float)
        valid = ~np.isnan(mu_lf_hf_array)
        if not valid.any():
            raise ValueError("All paired low-fidelity predictions are invalid.")

        rho = self.rho_manager.compute_rho(y_h[valid], mu_lf_hf_array[valid], iteration=loop_count)
        residuals = y_h[valid] - rho * mu_lf_hf_array[valid]
        self.residual_gp.fit(X_hf[valid], residuals)
        self.fusion = KOHFusion(self.lf_gp, self.residual_gp, self.rho_manager)

    def _sample_candidates(self, n_samples: int, loop_count: int) -> np.ndarray:
        seed = int(self.main_random_seed or 0) + loop_count
        if seed not in self._seed_candidates_cache:
            np.random.seed(seed)
            self._seed_candidates_cache[seed] = sample_candidates(self.bounds, n_samples, self.feature_types)
        candidates = self._seed_candidates_cache[seed]

        unique = {}
        for row in candidates:
            key = self._candidate_key(row)
            unique.setdefault(key, row)
        candidates = np.asarray(list(unique.values()), dtype=float)

        if len(self.data_manager.history_df) == 0:
            return candidates
        observed = {
            self._candidate_key(row)
            for row in self.data_manager.history_df[self.feature_names].to_numpy(dtype=float)
        }
        mask = [self._candidate_key(row) not in observed for row in candidates]
        return candidates[np.asarray(mask, dtype=bool)]

    def _candidate_key(self, row: np.ndarray) -> tuple:
        if self.feature_types and all(dtype == "int" for dtype in self.feature_types):
            return tuple(int(round(value)) for value in row)
        return tuple(round(float(value), 8) for value in row)

    def _koh_posterior_predict(self, candidates: np.ndarray):
        if self.fusion is None:
            raise ValueError("Fusion model has not been trained.")
        mu_h, sigma2_h = self.fusion.predict(candidates)
        mu_delta, sigma2_delta = self.fusion.predict_residual_only(candidates)
        return mu_h, sigma2_h, mu_delta, sigma2_delta

    def _select_q_points(self, candidates: np.ndarray, mu_h: np.ndarray, sigma2_h: np.ndarray, q: int):
        sigma_h = np.sqrt(np.maximum(sigma2_h, 1e-12))
        if self.acquisition_type == "ei":
            scores = compute_ei(mu_h, sigma_h, self._best_y())
        else:
            scores = compute_ucb(mu_h, sigma_h, self.acquisition_beta)
        order = np.argsort(scores)[::-1]
        return [int(index) for index in order[:q]], scores

    def _best_y(self) -> float:
        if len(self.data_manager.history_df) == 0:
            return 0.0
        return float(self.data_manager.history_df[self.data_manager.target_name].max())

    def _high_fidelity_branch(self, x_next_list: List[np.ndarray], iteration: int) -> None:
        for point in numpy_to_dict_list(np.asarray(x_next_list), self.feature_names):
            value = self.hf_blackbox.evaluate(point)
            self.data_manager.add_hf_experiment(point, value, iteration)
        self.data_manager.save_all()

    def _low_fidelity_branch(self, x_next_list: List[np.ndarray], iteration: int) -> None:
        points = numpy_to_dict_list(np.asarray(x_next_list), self.feature_names)
        means, variances = self.predictor.predict_batch(points, self.data_manager.get_history_data(), batch_size=20)
        for point, mean, variance in zip(points, means, variances):
            if not (np.isnan(mean) or np.isnan(variance)):
                self.data_manager.add_lf_prediction(point, mean, variance, iteration)
        X_lf, mu_lf, sigma2_lf = self.data_manager.get_lf_training_data()
        if len(X_lf) > 0:
            self.lf_gp.fit(X_lf, mu_lf, sigma2_lf)
            self.fusion = KOHFusion(self.lf_gp, self.residual_gp, self.rho_manager)
        self.data_manager.save_all()

    def _log_iteration(self, iteration: int, points: List[np.ndarray], do_hf: bool, ratio: float) -> None:
        self.iteration_log.append(
            {
                "iteration": int(iteration),
                "selected_points": [point.tolist() for point in points],
                "do_hf": bool(do_hf),
                "mismatch_ratio": float(ratio),
                "best_y": self._best_y(),
                "n_history": int(len(self.data_manager.history_df)),
                "n_lf_predictions": int(len(self.data_manager.lf_predictions_df)),
            }
        )
