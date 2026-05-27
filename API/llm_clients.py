"""LLM clients used by LABO.

API keys are read from environment variables. This module intentionally does
not read or write private key files.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


_model_cache: Dict[str, Any] = {}
_tokenizer_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()


def load_api_key(api_key: Optional[str] = None) -> str:
    """Return an API key from an explicit value or environment variables."""
    if api_key:
        return api_key.strip()
    for env_name in ("INTERN_S1_API_KEY", "LLM_API_KEY", "API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value.strip()
    raise RuntimeError("No API key found. Set INTERN_S1_API_KEY, LLM_API_KEY, or API_KEY.")


class InternS1Client:
    """Intern S1 client with messages and OpenAI-compatible chat protocols."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "intern-s1",
        protocol: str = "messages",
        timeout: float = 300.0,
    ) -> None:
        self.api_key = load_api_key(api_key)
        self.model = model
        self.protocol = protocol
        self.timeout = timeout

        env_url = os.getenv("INTERN_S1_API_URL")
        if api_url is None and env_url:
            api_url = env_url
        if api_url is None:
            if base_url is None:
                base_url = "https://chat.intern-ai.org.cn/v1"
            base_url = base_url.rstrip("/")
            api_url = (
                f"{base_url}/messages"
                if protocol == "messages"
                else f"{base_url}/chat/completions"
            )
        self.api_url = api_url

    def generate(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> str:
        """Generate text from the configured endpoint."""
        if self.protocol == "messages":
            payload: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
        elif self.protocol == "chat_completions":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if seed is not None:
                payload["seed"] = seed
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        else:
            raise ValueError(f"Unknown Intern S1 protocol: {self.protocol}")

        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

        return self._extract_text(response_text)

    @staticmethod
    def _extract_text(response_text: str) -> str:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            return response_text

        content = payload.get("content")
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("text")
            ]
            if parts:
                return "\n".join(parts)

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"])

        return json.dumps(payload, ensure_ascii=False)


class LocalVLLMClient:
    """Optional local vLLM backend loaded lazily."""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        dtype: Optional[str] = None,
        max_model_len: int = 16384,
    ) -> None:
        self.model_name = model_name
        if model_path is None:
            from .vllm_model_config import get_model_config, get_model_path_with_snapshot

            model_path = get_model_path_with_snapshot(model_name)
            config = get_model_config(model_name)
            tensor_parallel_size = tensor_parallel_size or config.get("tensor_parallel_size")
            dtype = dtype or config.get("dtype")
            max_model_len = int(config.get("max_model_len", max_model_len))

        self.llm, self.tokenizer, self.sampling_params_cls = self._load_model(
            model_name=model_name,
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
        )

    @staticmethod
    def _load_model(
        *,
        model_name: str,
        model_path: str,
        tensor_parallel_size: Optional[int],
        dtype: Optional[str],
        max_model_len: int,
    ) -> Tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError("Install vllm, torch, and transformers to use LocalVLLMClient.") from exc

        with _cache_lock:
            if model_name in _model_cache:
                return _model_cache[model_name], _tokenizer_cache[model_name], SamplingParams

            if tensor_parallel_size is None:
                tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
            if dtype is None:
                dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype=dtype,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            _model_cache[model_name] = llm
            _tokenizer_cache[model_name] = tokenizer
            return llm, tokenizer, SamplingParams

    def generate(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> str:
        params = self.sampling_params_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        outputs = self.llm.generate([prompt], params)
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text.strip()
        return ""


LocalHuggingFaceClient = LocalVLLMClient


def create_llm_client(llm_config: Any):
    """Create an LLM client from a simple config object."""
    llm_type = getattr(llm_config, "llm_type", "intern_s1")
    if llm_type == "intern_s1":
        return InternS1Client(
            api_key=getattr(llm_config, "api_key", None),
            api_url=getattr(llm_config, "api_url", None),
            base_url=getattr(llm_config, "base_url", None),
            model=getattr(llm_config, "model", "intern-s1"),
            protocol=getattr(llm_config, "protocol", "messages"),
            timeout=float(getattr(llm_config, "timeout", 300.0)),
        )
    if llm_type in {"local_vllm", "local_hf"}:
        return LocalVLLMClient(
            model_name=getattr(llm_config, "model_name", getattr(llm_config, "model", "")),
            model_path=getattr(llm_config, "model_path", None),
            tensor_parallel_size=getattr(llm_config, "tensor_parallel_size", None),
            dtype=getattr(llm_config, "dtype", None),
            max_model_len=int(getattr(llm_config, "max_model_len", 16384)),
        )
    raise ValueError(f"Unknown llm_type: {llm_type}")
