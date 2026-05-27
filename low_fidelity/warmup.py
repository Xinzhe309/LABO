"""Warm-start utilities for LABO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .predictor import LowFidelityPredictor
from .prompt import get_warmup_prompt
from koh.utils import numpy_to_dict_list


DEFAULT_WARMUP_RANDOM_SEED = 42


def _lhs_sample(bounds: np.ndarray, n_samples: int, seed: int = DEFAULT_WARMUP_RANDOM_SEED) -> np.ndarray:
    """Generate Latin-hypercube samples inside bounds."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    d = len(bounds)
    samples = np.zeros((n_samples, d), dtype=float)
    for dim, (low, high) in enumerate(bounds):
        width = (high - low) / max(n_samples, 1)
        order = rng.permutation(n_samples)
        samples[:, dim] = low + order * width + rng.uniform(0, width, n_samples)
    return samples


def parse_initial_points(response: str, feature_names: List[str]) -> List[dict]:
    """Parse initial points from a JSON response."""
    decoder = json.JSONDecoder()
    text = response.strip()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "points" not in payload:
            continue
        points = []
        for item in payload["points"]:
            if isinstance(item, dict) and all(name in item for name in feature_names):
                points.append({name: float(item[name]) for name in feature_names})
        if points:
            return points
    raise ValueError("Could not parse initial points from LLM response.")


def _log_llm_call(log_path: Optional[str], record: Dict) -> None:
    if not log_path:
        return
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=True)
            fh.write("\n")
    except OSError:
        pass


def generate_initial_points_with_llm(
    llm_client,
    task_name: str,
    feature_names: List[str],
    n_points: int = 5,
    llm_seed_sequence: Optional[List[int]] = None,
    log_path: Optional[str] = None,
    max_tokens: int = 2048,
) -> List[dict]:
    """Generate initial points with the LLM.

    For public-release use, fixed initial points are usually simpler and more
    reproducible. This function remains available for custom integrations.
    """
    system_prompt, user_prompt = get_warmup_prompt(task_name, n_points)
    prompt = f"{system_prompt}\n\n{user_prompt}"
    seeds = llm_seed_sequence or list(range(1, 6))
    last_error: Exception | None = None
    for seed in seeds:
        record = {"stage": "warmup", "seed": seed, "input_prompt": prompt}
        try:
            response = llm_client.generate(prompt, seed=seed, temperature=0.0, top_p=1.0, max_tokens=max_tokens)
            record["response"] = response
            points = parse_initial_points(response, feature_names)
            record["status"] = "success"
            _log_llm_call(log_path, record)
            return points[:n_points]
        except Exception as exc:
            last_error = exc
            record["status"] = "error"
            record["error"] = str(exc)
            _log_llm_call(log_path, record)
    raise ValueError(f"LLM warm-start generation failed: {last_error}")


def warmup_phase(
    llm_client,
    hf_blackbox,
    data_manager,
    generator,
    user_prompt: str,
    task_name: str,
    feature_names: List[str],
    n_initial_points: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    warmup_random_seed: int = DEFAULT_WARMUP_RANDOM_SEED,
    llm_seed_sequence: Optional[List[int]] = None,
    fixed_initial_points: Optional[List[dict]] = None,
    y_transform: float = 1.0,
) -> None:
    """Run initial high-fidelity evaluations and low-fidelity exploration."""
    if fixed_initial_points:
        initial_points = fixed_initial_points[:n_initial_points]
    else:
        initial_points = generate_initial_points_with_llm(
            llm_client=llm_client,
            task_name=task_name,
            feature_names=feature_names,
            n_points=n_initial_points,
            llm_seed_sequence=llm_seed_sequence,
            log_path=str(getattr(generator, "log_path", "")) or None,
            max_tokens=max_tokens,
        )

    for point in initial_points:
        missing = [name for name in feature_names if name not in point]
        if missing:
            raise ValueError(f"Initial point is missing features: {missing}")
        value = hf_blackbox.evaluate(point)
        data_manager.add_seed_point(point, value)
        data_manager.add_hf_experiment(point, value, iteration=0)
    data_manager.save_all()

    predictor = LowFidelityPredictor(
        generator=generator,
        user_prompts=user_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        y_transform=y_transform,
    )

    for point in initial_points:
        try:
            history = data_manager.get_history_exclude_point(point)
            mean, variance, _ = predictor.predict(point, history)
            data_manager.add_lf_prediction(point, mean, variance, iteration=0)
        except Exception:
            continue

    n_exploration_points = max(10, min(50, 5 * len(feature_names)))
    exploration_points = numpy_to_dict_list(
        _lhs_sample(np.asarray(hf_blackbox.bounds, dtype=float), n_exploration_points, seed=warmup_random_seed),
        feature_names,
    )
    history = data_manager.get_history_data()
    batch_size = 10 if len(feature_names) >= 10 else 20
    try:
        means, variances = predictor.predict_batch(exploration_points, history=history, batch_size=batch_size)
        for point, mean, variance in zip(exploration_points, means, variances):
            if not (np.isnan(mean) or np.isnan(variance)):
                data_manager.add_lf_prediction(point, float(mean), float(variance), iteration=0)
    except Exception:
        pass

    data_manager.save_all()
