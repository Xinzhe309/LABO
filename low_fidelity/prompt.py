"""Generic prompt templates for LABO low-fidelity predictions."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple


SYSTEM_PROMPT = """You are a scientific low-fidelity evaluator for Bayesian optimization.
Return only valid JSON. Do not include markdown, explanations, comments, or hidden reasoning.
Always preserve the feature order supplied in each input point."""


def format_history_json(history: List[Dict] | None, feature_order: List[str] | None = None) -> str:
    """Serialize history as compact JSON.

    History entries are expected to be shaped as {"x": {...}, "y": value}.
    """
    if not history:
        return json.dumps({"data_points": []}, separators=(",", ":"))

    data_points = []
    for entry in history:
        x = entry.get("x", {})
        if feature_order is None:
            feature_order = list(x.keys())
        data_points.append(
            {
                "features": [float(x[name]) for name in feature_order],
                "target": float(entry["y"]),
            }
        )
    return json.dumps({"data_points": data_points}, separators=(",", ":"))


def format_points_json(points: List[Dict], feature_order: List[str] | None = None) -> str:
    """Serialize candidate points as compact JSON."""
    if not points:
        return json.dumps({"data_points": []}, separators=(",", ":"))
    if feature_order is None:
        feature_order = list(points[0].keys())
    data_points = [
        {"features": [float(point[name]) for name in feature_order]}
        for point in points
    ]
    return json.dumps({"data_points": data_points}, separators=(",", ":"))


def get_warmup_prompt(task_name: str, n_points: int = 5) -> Tuple[str, str]:
    """Return a generic warm-start prompt.

    Public LABO does not ship domain-specific prior text. Users should provide
    fixed initial points or customize this prompt for their domain.
    """
    user = f"""Generate {n_points} diverse initialization points for a bounded optimization task.

Return exactly this JSON schema:
{{
  "points": [
    {{"x1": 0.0, "x2": 0.0}}
  ]
}}

Use the feature names and bounds from the user's task documentation if provided."""
    return SYSTEM_PROMPT, user


def get_prediction_prompt(history: List[Dict] | None = None, points: List[Dict] | None = None) -> Tuple[str, str]:
    """Return the generic low-fidelity prediction prompt template."""
    history_json = "{history_json}" if history is None or points is None else format_history_json(history)
    points_json = "{points_json}" if history is None or points is None else format_points_json(points)
    user = f"""Predict low-fidelity objective values for the candidate points.

Historical measurements:
{history_json}

Candidate points:
{points_json}

Return exactly one JSON object with this schema:
{{
  "data_points": [
    {{"features": [0.0, 0.0], "target": 0.0}}
  ]
}}

Requirements:
- Include one entry for each candidate point.
- Preserve the candidate order.
- Keep feature arrays in the same order as the input.
- The target must be a numeric scalar.
- Output JSON only."""
    return SYSTEM_PROMPT, user


def load_prompts(task_name: str):
    """Return a generic system prompt and one user-prompt template."""
    system, user = get_prediction_prompt()
    return system, [user]
