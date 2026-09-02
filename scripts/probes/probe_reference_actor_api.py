#!/usr/bin/env python3
"""Probe vLLM's teacher-forced completion logprob response without using a GPU locally."""

from __future__ import annotations

import argparse
import json
from typing import Any

import requests


def analyze_completion_response(
    response: dict[str, Any], action_start_offset: int
) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    logprobs = choices[0].get("logprobs") or {}
    offsets = logprobs.get("text_offset")
    if not offsets:
        raise ValueError("response has no text_offset values")
    token_logprobs = logprobs.get("token_logprobs") or []
    top_logprobs = logprobs.get("top_logprobs") or []
    start = next((index for index, offset in enumerate(offsets) if offset >= action_start_offset), None)
    if start is None:
        raise ValueError("could not locate action tokens from text_offset values")

    logged = [value for value in token_logprobs[start:] if value is not None]
    action_top = top_logprobs[start:] if top_logprobs else []
    top_k_counts = [len(value or {}) for value in action_top]
    return {
        "action_token_count": len(logged),
        "logged_token_logprobs": logged,
        "top_k_counts": top_k_counts,
        "has_top_k_logprobs": bool(top_k_counts) and all(count > 0 for count in top_k_counts),
        "returns_full_vocabulary": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="User: Choose a TextWorld action.\nAssistant: ")
    parser.add_argument("--action", default="go east")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    base = args.api_base.rstrip("/")
    health = requests.get(f"{base}/health", timeout=args.timeout)
    health.raise_for_status()
    models_response = requests.get(f"{base}/v1/models", timeout=args.timeout)
    models_response.raise_for_status()
    models = models_response.json().get("data") or []
    if not models:
        raise SystemExit("vLLM returned no served models")
    model_name = models[0]["id"]

    full_prompt = args.prompt + args.action
    completion = requests.post(
        f"{base}/v1/completions",
        json={
            "model": model_name,
            "prompt": full_prompt,
            "max_tokens": 0,
            "temperature": 0.0,
            "echo": True,
            "logprobs": args.top_k,
        },
        timeout=args.timeout,
    )
    completion.raise_for_status()
    result = analyze_completion_response(completion.json(), len(args.prompt))
    print(json.dumps({"model": model_name, "requested_top_k": args.top_k, **result}, indent=2))


if __name__ == "__main__":
    main()
