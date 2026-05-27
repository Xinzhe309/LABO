"""Optional local model configuration.

Public releases should not assume local checkpoints are present. Add entries
for your own local environment if you want to use LocalVLLMClient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


VLLM_MODEL_CONFIG: Dict[str, dict] = {}


def get_model_config(model_name: str) -> dict:
    if model_name not in VLLM_MODEL_CONFIG:
        raise ValueError(f"Unknown local model: {model_name}")
    return VLLM_MODEL_CONFIG[model_name]


def get_model_path_with_snapshot(model_name: str) -> str:
    config = get_model_config(model_name)
    model_path = Path(config["model_path"]).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")
    return str(model_path)


def validate_model_config(model_name: str) -> bool:
    path = Path(get_model_path_with_snapshot(model_name))
    return path.exists()
