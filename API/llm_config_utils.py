"""Helpers for constructing LLM configuration objects."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

from .vllm_model_config import VLLM_MODEL_CONFIG


def create_llm_config_from_model_name(
    llm_model: str,
    base_config: Optional[Dict[str, Any]] = None,
    default_model: str = "intern-s1",
) -> SimpleNamespace:
    """Build a small config namespace for API or local LLM backends."""
    if not llm_model:
        llm_model = default_model
    config_dict = dict(base_config or {})
    config = SimpleNamespace(**config_dict)

    if llm_model in VLLM_MODEL_CONFIG:
        config.llm_type = "local_vllm"
        config.model_name = llm_model
        config.tensor_parallel_size = getattr(config, "tensor_parallel_size", None)
        config.dtype = getattr(config, "dtype", None)
        config.max_model_len = getattr(config, "max_model_len", 16384)
    else:
        config.llm_type = "intern_s1"
        config.model = llm_model
        config.protocol = getattr(config, "protocol", "messages")
        config.base_url = getattr(config, "base_url", "https://chat.intern-ai.org.cn/v1")
        config.api_url = getattr(config, "api_url", None)

    config.temperature = getattr(config, "temperature", 0.7)
    config.top_p = getattr(config, "top_p", 0.9)
    config.max_tokens = getattr(config, "max_tokens", 2048)
    config.alpha = getattr(config, "alpha", 1.0)
    config.beta = getattr(config, "beta", 0.0)
    return config
