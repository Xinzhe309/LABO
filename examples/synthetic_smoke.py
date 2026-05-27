"""Offline LABO smoke test on a synthetic objective."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from koh.optimizer import KOHOptimizer


class SyntheticBlackBox:
    """Small deterministic high-fidelity function for smoke tests."""

    def __init__(self) -> None:
        self.bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)

    def evaluate(self, x: dict) -> float:
        a = float(x["x1"])
        b = float(x["x2"])
        return float(1.0 - (a - 0.72) ** 2 - 0.5 * (b - 0.35) ** 2)


class DeterministicLowFidelityClient:
    """LLM-shaped client that returns JSON predictions without network calls."""

    def generate(self, prompt: str, **_: object) -> str:
        match = re.search(r'"data_points"\s*:\s*\[(.*?)\]\s*}', prompt, flags=re.S)
        if not match:
            return json.dumps({"data_points": []})

        text = "{" + '"data_points": [' + match.group(1) + "]}"
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"data_points": []}

        predictions = []
        for item in payload.get("data_points", []):
            features = item.get("features", [])
            if len(features) < 2 or "target" in item:
                continue
            a, b = float(features[0]), float(features[1])
            value = 0.85 - (a - 0.65) ** 2 - 0.4 * (b - 0.4) ** 2
            predictions.append({"features": [a, b], "target": round(float(value), 6)})
        return json.dumps({"data_points": predictions})


def main() -> None:
    output_dir = Path("results") / "synthetic_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_config = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        alpha=1.0,
        beta=0.0,
        value_range=[-1.0, 1.5],
    )
    koh_config = SimpleNamespace(
        max_hf_iterations=1,
        n_candidates=40,
        q=1,
        n_initial_points=3,
        mismatch_threshold=0.75,
        force_hf_after_n_lf=None,
        gp_training_iter=2,
        max_loops=2,
        acquisition_type="ucb",
        acquisition_beta=1.0,
        always_update_lf_loops=1,
        random_seed=7,
    )

    blackbox = SyntheticBlackBox()
    optimizer = KOHOptimizer(
        task_name="synthetic",
        task_data_dir=str(output_dir),
        feature_names=["x1", "x2"],
        feature_types=["float", "float"],
        bounds=blackbox.bounds,
        target_name="objective",
        llm_client=DeterministicLowFidelityClient(),
        hf_blackbox=blackbox,
        llm_config=llm_config,
        koh_config=koh_config,
        file_prefix="smoke",
    )
    optimizer.run(
        max_iterations=1,
        n_initial_points=3,
        q=1,
        fixed_initial_points=[
            {"x1": 0.10, "x2": 0.10},
            {"x1": 0.50, "x2": 0.50},
            {"x1": 0.90, "x2": 0.30},
        ],
    )
    print("LABO_SYNTHETIC_SMOKE_OK")


if __name__ == "__main__":
    main()
