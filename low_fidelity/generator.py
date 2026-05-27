"""LLM generation and JSON parsing helpers for low-fidelity predictions."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class LLMGenerator:
    """Call an LLM client and parse numeric prediction JSON."""

    def __init__(
        self,
        llm_client,
        system_prompt: Optional[str] = None,
        value_range: Optional[List[float]] = None,
        log_path: Optional[str] = None,
    ) -> None:
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.value_range = value_range
        self.call_log: List[Dict[str, Any]] = []
        if log_path:
            self.log_path = Path(log_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = Path("llm_logs") / f"llm_calls_{timestamp}.jsonl"

    def generate_single(
        self,
        user_prompt: str,
        x: dict,
        history: Optional[List[Dict]] = None,
        seed: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> float:
        from .prompt import format_history_json, format_points_json

        feature_order = list(x.keys())
        full_prompt = self._format_prompt(
            user_prompt=user_prompt,
            history_json=format_history_json(history, feature_order),
            points_json=format_points_json([x], feature_order),
        )
        response = self._call(full_prompt, seed, temperature, top_p, max_tokens)
        values = self._parse_json_predictions(response, n_points=1, input_dim=len(feature_order))
        value = values[0] if values else None
        if value is None:
            raise ValueError("Could not parse a prediction from the LLM response.")
        self._validate_value(value)
        return float(value)

    def generate_batch_multi_points(
        self,
        user_prompt_template: str,
        X_batch: List[dict],
        history: Optional[List[Dict]] = None,
        seed: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> List[Optional[float]]:
        if not X_batch:
            return []

        from .prompt import format_history_json, format_points_json

        feature_order = list(X_batch[0].keys())
        full_prompt = self._format_prompt(
            user_prompt=user_prompt_template,
            history_json=format_history_json(history, feature_order),
            points_json=format_points_json(X_batch, feature_order),
        )
        response = self._call(full_prompt, seed, temperature, top_p, max_tokens)
        values = self._parse_json_predictions(response, n_points=len(X_batch), input_dim=len(feature_order))
        for value in values:
            if value is not None:
                self._validate_value(value)
        return values

    def _format_prompt(self, *, user_prompt: str, history_json: str, points_json: str) -> str:
        formatted_user = user_prompt.format(history_json=history_json, points_json=points_json)
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{formatted_user}"
        return formatted_user

    def _call(
        self,
        prompt: str,
        seed: Optional[int],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        record = {
            "input_prompt": prompt,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        try:
            response = self.llm_client.generate(
                prompt,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            record["response"] = response
            return response
        except Exception as exc:
            record["error"] = str(exc)
            raise
        finally:
            self.call_log.append(record)
            self._write_log(record)

    def _validate_value(self, value: float) -> None:
        if self.value_range is None:
            return
        lower, upper = self.value_range
        if not (lower <= float(value) <= upper):
            raise ValueError(f"Prediction {value} is outside valid range {self.value_range}.")

    def _parse_json_predictions(
        self,
        response: str,
        n_points: int,
        input_dim: Optional[int] = None,
    ) -> List[Optional[float]]:
        """Parse predictions from a response that may contain extra text."""
        response = response.strip()
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        for candidate in self._candidate_json_objects(response):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            values = self._extract_targets(payload, n_points)
            if any(value is not None for value in values):
                return values

        numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", response)]
        if len(numbers) >= n_points:
            return numbers[-n_points:]
        raise ValueError(f"Could not parse JSON predictions from response: {response[:500]}")

    @staticmethod
    def _candidate_json_objects(text: str) -> List[str]:
        decoder = json.JSONDecoder()
        objects: List[str] = []
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                _, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            objects.append(text[index : index + end])
        return objects

    @staticmethod
    def _extract_targets(payload: Any, n_points: int) -> List[Optional[float]]:
        if isinstance(payload, dict) and "data_points" in payload:
            data_points = payload["data_points"]
            if not isinstance(data_points, list):
                return [None] * n_points
            values: List[Optional[float]] = []
            for item in data_points[:n_points]:
                if isinstance(item, dict) and "target" in item:
                    try:
                        values.append(float(item["target"]))
                    except (TypeError, ValueError):
                        values.append(None)
                else:
                    values.append(None)
            values.extend([None] * (n_points - len(values)))
            return values

        if isinstance(payload, dict) and "predictions" in payload:
            values = []
            for item in payload["predictions"][:n_points]:
                try:
                    values.append(float(item))
                except (TypeError, ValueError):
                    values.append(None)
            values.extend([None] * (n_points - len(values)))
            return values

        return [None] * n_points

    def _write_log(self, record: Dict[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=True)
                fh.write("\n")
        except OSError:
            pass
