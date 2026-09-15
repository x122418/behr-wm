#!/usr/bin/env python3
"""Evaluate generated-action agreement for paired TextWorld observations.

The input is a manifest of transition-evaluation ``results.jsonl`` files.  A
single frozen actor greedily generates an action from the real observation and
from each world model's predicted observation.  Real-observation generations
are shared across every model and all JSONL outputs are resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reward.textworld_actor_inputs import build_textworld_actor_messages


def extract_action(raw_output: str) -> str | None:
    """Return a conservative canonical action, or ``None`` if parsing is unsafe."""
    text = raw_output.strip()
    if not text:
        return None
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) < 3:
            return None
        text = "\n".join(lines[1:-1]).strip()
    labeled = re.fullmatch(r"(?is)action\s*:\s*([^\r\n]+)", text)
    if labeled is not None:
        text = labeled.group(1).strip()
    elif "\n" in text or "\r" in text:
        return None
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[.!]+$", "", text).strip()
    return text or None


def build_actor_prompt(
    tokenizer: Any, history: list[dict[str, str]], observation: str
) -> str:
    """Render the same actor-view history contract used by the training scorer."""
    messages = build_textworld_actor_messages(history, observation)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def validate_aligned_results(
    model_rows: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Validate identical held-out items and real-state contracts across models."""
    if not model_rows:
        raise ValueError("manifest contains no ready model results")
    indexes: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, rows in model_rows.items():
        index: dict[str, dict[str, Any]] = {}
        for row in rows:
            item_id = row.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(f"{model_name} contains a row without an item_id")
            if item_id in index:
                raise ValueError(f"{model_name} contains duplicate item ID {item_id}")
            if row.get("status") != "ok":
                raise ValueError(f"{model_name} item {item_id} is not successfully scored")
            index[item_id] = row
        indexes[model_name] = index

    reference_name = next(iter(indexes))
    reference = indexes[reference_name]
    reference_ids = set(reference)
    invariant_fields = ("history", "real_observation", "logged_action")
    for model_name, index in indexes.items():
        if set(index) != reference_ids:
            raise ValueError(
                f"item IDs differ between {reference_name} and {model_name}"
            )
        for item_id in reference_ids:
            for field in invariant_fields:
                if index[item_id].get(field) != reference[item_id].get(field):
                    raise ValueError(
                        f"{field} differs for {item_id} between "
                        f"{reference_name} and {model_name}"
                    )
    return sorted(reference_ids)


def summarize_action_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired action-generation results."""
    successful = [row for row in rows if row.get("status") == "ok"]
    total = len(rows)
    summary: dict[str, Any] = {
        "total": total,
        "successful": len(successful),
        "errors": total - len(successful),
        "parse_or_generation_failure_rate": (
            (total - len(successful)) / total if total else 0.0
        ),
    }
    if not successful:
        return summary
    denominator = len(successful)
    summary.update(
        {
            "raw_action_agreement": sum(
                row["real_raw_output"].strip()
                == row["predicted_raw_output"].strip()
                for row in successful
            )
            / denominator,
            "normalized_action_agreement": sum(
                row["real_action"] == row["predicted_action"]
                for row in successful
            )
            / denominator,
            "real_vs_logged_agreement": sum(
                row["real_action"] == extract_action(row["logged_action"])
                for row in successful
            )
            / denominator,
            "predicted_vs_logged_agreement": sum(
                row["predicted_action"] == extract_action(row["logged_action"])
                for row in successful
            )
            / denominator,
        }
    )
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if not manifest.get("actor_model_path"):
        raise ValueError("manifest must define actor_model_path")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("manifest must define a non-empty models list")
    names = [entry.get("name") for entry in models]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("every manifest model must have a non-empty name")
    if len(set(names)) != len(names):
        raise ValueError("manifest model names must be unique")
    return manifest


def _ready_models(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    ready = []
    pending = []
    for entry in manifest["models"]:
        if entry.get("status") == "pending" or not entry.get("results_jsonl"):
            pending.append(entry["name"])
        else:
            ready.append(entry)
    return ready, pending


def _cache_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _read_jsonl(path)
    return {row["item_id"]: row for row in rows}


def _fingerprint(history: list[dict[str, str]], observation: str) -> str:
    payload = json.dumps(
        {"history": history, "observation": observation},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _generate_actions(
    requests: list[dict[str, Any]],
    cache_path: Path,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, dict[str, Any]]:
    import torch

    cached = _cache_index(cache_path)
    pending = [row for row in requests if row["item_id"] not in cached]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [
            build_actor_prompt(tokenizer, row["history"], row["observation"])
            for row in batch
        ]
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
                model.device
            )
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prefix_width = inputs["input_ids"].shape[1]
            outputs = tokenizer.batch_decode(
                generated[:, prefix_width:], skip_special_tokens=True
            )
            for request, raw_output in zip(batch, outputs, strict=True):
                action = extract_action(raw_output)
                result = {
                    "item_id": request["item_id"],
                    "input_fingerprint": _fingerprint(
                        request["history"], request["observation"]
                    ),
                    "raw_output": raw_output,
                    "action": action,
                    "status": "ok" if action is not None else "error",
                }
                if action is None:
                    result["error"] = "actor output did not contain one parseable action"
                cached[request["item_id"]] = result
                _append_jsonl(cache_path, result)
        except Exception as error:
            for request in batch:
                result = {
                    "item_id": request["item_id"],
                    "input_fingerprint": _fingerprint(
                        request["history"], request["observation"]
                    ),
                    "status": "error",
                    "error": str(error),
                }
                cached[request["item_id"]] = result
                _append_jsonl(cache_path, result)
        print(f"generated actions {min(start + batch_size, len(pending))}/{len(pending)}")

    for request in requests:
        cached_row = cached[request["item_id"]]
        expected = _fingerprint(request["history"], request["observation"])
        if cached_row.get("input_fingerprint") != expected:
            raise ValueError(
                f"stale cached action for {request['item_id']} in {cache_path}"
            )
    return cached


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _evaluate(
    manifest: dict[str, Any],
    model_rows: dict[str, list[dict[str, Any]]],
    item_ids: list[str],
    output_dir: Path,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --validate-only is used")
    actor_path = manifest["actor_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        actor_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        actor_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    indexed = {
        name: {row["item_id"]: row for row in rows}
        for name, rows in model_rows.items()
    }
    first_name = next(iter(indexed))
    real_requests = [
        {
            "item_id": item_id,
            "history": indexed[first_name][item_id]["history"],
            "observation": indexed[first_name][item_id]["real_observation"],
        }
        for item_id in item_ids
    ]
    real_actions = _generate_actions(
        real_requests,
        output_dir / "real_actions.jsonl",
        model,
        tokenizer,
        batch_size,
        max_new_tokens,
    )

    summaries: dict[str, Any] = {}
    for model_name, rows_by_id in indexed.items():
        predicted_requests = [
            {
                "item_id": item_id,
                "history": rows_by_id[item_id]["history"],
                "observation": rows_by_id[item_id]["predicted_observation"],
            }
            for item_id in item_ids
        ]
        predicted_actions = _generate_actions(
            predicted_requests,
            output_dir / f"{model_name}_predicted_actions.jsonl",
            model,
            tokenizer,
            batch_size,
            max_new_tokens,
        )
        comparisons = []
        for item_id in item_ids:
            source = rows_by_id[item_id]
            real = real_actions[item_id]
            predicted = predicted_actions[item_id]
            if real.get("status") == "ok" and predicted.get("status") == "ok":
                comparison = {
                    "item_id": item_id,
                    "task_id": source.get("task_id"),
                    "logged_action": source["logged_action"],
                    "real_raw_output": real["raw_output"],
                    "predicted_raw_output": predicted["raw_output"],
                    "real_action": real["action"],
                    "predicted_action": predicted["action"],
                    "status": "ok",
                }
            else:
                comparison = {
                    "item_id": item_id,
                    "task_id": source.get("task_id"),
                    "logged_action": source["logged_action"],
                    "status": "error",
                    "real_error": real.get("error"),
                    "predicted_error": predicted.get("error"),
                }
            comparisons.append(comparison)
        _write_jsonl(output_dir / f"{model_name}_comparisons.jsonl", comparisons)
        summaries[model_name] = summarize_action_agreement(comparisons)

    report = {
        "schema_version": 1,
        "actor_model_path": actor_path,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
        },
        "models": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TextWorld real/predicted generated-action agreement"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch size and max new tokens must be positive")

    manifest = _load_manifest(args.manifest)
    ready, pending = _ready_models(manifest)
    if not ready:
        raise ValueError("manifest contains no ready models")
    model_rows = {
        entry["name"]: _read_jsonl(Path(entry["results_jsonl"])) for entry in ready
    }
    item_ids = validate_aligned_results(model_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_report = {
        "manifest": str(args.manifest.resolve()),
        "ready_model_count": len(ready),
        "ready_models": [entry["name"] for entry in ready],
        "pending_models": pending,
        "aligned_item_count": len(item_ids),
        "actor_model_path": manifest["actor_model_path"],
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation_report, ensure_ascii=False, indent=2))
    if args.validate_only:
        return
    _evaluate(
        manifest,
        model_rows,
        item_ids,
        args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
