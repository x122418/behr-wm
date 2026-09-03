#!/usr/bin/env python3
"""Evaluate a TextWorld world model on held-out single-step transitions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any

import requests
import torch

from src.data.score_textworld_actor_consistency import compute_consistency_metrics
from src.reward.textworld_actor_inputs import build_teacher_forced_actor_inputs


def build_actor_inputs(
    tokenizer: Any,
    history: list[dict[str, str]],
    observation: str,
    logged_action: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for the shared actor-input contract."""
    return build_teacher_forced_actor_inputs(
        tokenizer, history, observation, logged_action
    )


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def transition_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Extract the stable offline-evaluation contract from one parquet row."""
    row = _plain(row)
    extra_info = row["extra_info"]
    return {
        "item_id": row["item_id"],
        "task_id": extra_info["task_id"],
        "wm_messages": list(row["prompt"]),
        "history": list(extra_info["history"]),
        "real_observation": row["reward_model"]["ground_truth"],
        "logged_action": extra_info["expert_action"],
    }


def score_actor_logits(
    model: Any,
    tokenizer: Any,
    history: list[dict[str, str]],
    observation: str,
    logged_action: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_inputs, action_ids = build_actor_inputs(
        tokenizer, history, observation, logged_action
    )
    input_ids = model_inputs.unsqueeze(0).to(model.device)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=int(action_ids.numel()),
        )
    logits = outputs.logits.squeeze(0).float().cpu()
    if logits.shape[0] != action_ids.numel():
        raise RuntimeError(
            f"actor returned {logits.shape[0]} positions for "
            f"{action_ids.numel()} action tokens"
        )
    return logits, action_ids


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scalar metrics over successfully scored rows."""
    successful = [row for row in rows if row.get("status") == "ok"]
    summary: dict[str, Any] = {
        "total": len(rows),
        "successful": len(successful),
        "errors": len(rows) - len(successful),
    }
    if not successful:
        return summary
    summary["exact_match"] = sum(bool(row["exact_match"]) for row in successful) / len(
        successful
    )
    values: dict[str, list[float]] = defaultdict(list)
    for row in successful:
        for key, value in row.items():
            if key in {"exact_match", "task_id"} or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                values[key].append(float(value))
    for key, metric_values in values.items():
        summary[key] = sum(metric_values) / len(metric_values)
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class WorldModelClient:
    def __init__(self, api_base: str, timeout: float, max_tokens: int):
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        response = requests.get(f"{self.api_base}/v1/models", timeout=10)
        response.raise_for_status()
        models = response.json().get("data") or []
        if not models:
            raise RuntimeError("world-model server returned no models")
        self.model = models[0]["id"]

    def generate(self, messages: list[dict[str, str]]) -> str:
        response = requests.post(
            f"{self.api_base}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def generate_predictions(
    transitions: list[dict[str, Any]],
    cache_path: Path,
    api_base: str,
    concurrency: int,
    timeout: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    cached = {row["item_id"]: row for row in _read_jsonl(cache_path)}
    pending = [row for row in transitions if row["item_id"] not in cached]
    client = WorldModelClient(api_base, timeout=timeout, max_tokens=max_tokens)

    def generate_one(transition: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        try:
            prediction = client.generate(transition["wm_messages"])
            return {
                "item_id": transition["item_id"],
                "predicted_observation": prediction,
                "generation_seconds": time.monotonic() - started,
                "generation_status": "ok",
            }
        except Exception as error:
            return {
                "item_id": transition["item_id"],
                "generation_status": "error",
                "generation_error": str(error),
                "generation_seconds": time.monotonic() - started,
            }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(generate_one, row) for row in pending]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            cached[result["item_id"]] = result
            _append_jsonl(cache_path, result)
            print(f"generated {index}/{len(pending)}: {result['item_id']}")
    return [cached[row["item_id"]] for row in transitions]


def evaluate(
    transitions: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    output_path: Path,
    actor_model_path: str,
    top_ks: tuple[int, ...],
) -> list[dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    completed = {row["item_id"]: row for row in _read_jsonl(output_path)}
    tokenizer = AutoTokenizer.from_pretrained(
        actor_model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        actor_model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    generation_by_id = {row["item_id"]: row for row in generations}
    pending = [row for row in transitions if row["item_id"] not in completed]
    for index, transition in enumerate(pending, start=1):
        generation = generation_by_id[transition["item_id"]]
        started = time.monotonic()
        try:
            if generation["generation_status"] != "ok":
                raise RuntimeError(generation["generation_error"])
            prediction = generation["predicted_observation"]
            real_logits, real_action_ids = score_actor_logits(
                model,
                tokenizer,
                transition["history"],
                transition["real_observation"],
                transition["logged_action"],
            )
            predicted_logits, predicted_action_ids = score_actor_logits(
                model,
                tokenizer,
                transition["history"],
                prediction,
                transition["logged_action"],
            )
            if not torch.equal(real_action_ids, predicted_action_ids):
                raise RuntimeError("real and predicted prompts produced different action IDs")
            metrics = compute_consistency_metrics(
                real_logits, predicted_logits, real_action_ids, top_ks=top_ks
            )
            result = {
                **transition,
                "wm_messages": transition["wm_messages"],
                "predicted_observation": prediction,
                "exact_match": prediction.strip()
                == transition["real_observation"].strip(),
                **metrics,
                "generation_seconds": generation["generation_seconds"],
                "scoring_seconds": time.monotonic() - started,
                "status": "ok",
            }
        except Exception as error:
            result = {
                "item_id": transition["item_id"],
                "task_id": transition["task_id"],
                "status": "error",
                "error": str(error),
                "scoring_seconds": time.monotonic() - started,
            }
        completed[result["item_id"]] = result
        _append_jsonl(output_path, result)
        print(f"scored {index}/{len(pending)}: {result['item_id']} ({result['status']})")
    return [completed[row["item_id"]] for row in transitions]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actor-model-path", required=True)
    parser.add_argument("--wm-api-base", default="http://127.0.0.1:8001")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-ks", default="32,64")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    top_ks = tuple(int(value) for value in args.top_ks.split(","))

    import pandas as pd

    frame = pd.read_parquet(args.input)
    if args.limit is not None:
        frame = frame.iloc[: args.limit]
    transitions = [transition_from_row(row) for row in frame.to_dict("records")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generations = generate_predictions(
        transitions,
        args.output_dir / "generations.jsonl",
        api_base=args.wm_api_base,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    results = evaluate(
        transitions,
        generations,
        args.output_dir / "results.jsonl",
        actor_model_path=args.actor_model_path,
        top_ks=top_ks,
    )
    summary = summarize_results(results)
    summary.update(
        {
            "input": str(args.input.resolve()),
            "actor_model_path": str(Path(args.actor_model_path).resolve()),
            "wm_api_base": args.wm_api_base,
            "top_ks": list(top_ks),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
