#!/usr/bin/env python3
"""Generate and attach frozen-reference-actor actions for the TextWorld pilot."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.build_textworld_pilot import parse_initial_context, write_jsonl


TEXTWORLD_SYSTEM_PROMPT = (
    "You are playing a text-based interactive fiction game (TextWorld).\n"
    "Choose exactly one action from the AVAILABLE ACTIONS in the observation.\n"
    "Respond strictly as:\nAction:\n<the single action>"
)
ACTION_RE = re.compile(r"Action:\n([^\r\n]+)\Z")


def parse_and_validate_action(
    raw_output: str, admissible_actions: list[str]
) -> tuple[str | None, bool]:
    """Accept only the exact ``Action:\n<canonical action>`` output contract."""
    match = ACTION_RE.fullmatch(raw_output.strip())
    if match is None:
        return None, False
    extracted = match.group(1)
    if extracted in admissible_actions:
        return extracted, True
    return extracted, False


def attach_logged_actions(
    records: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return copied pilot records enriched with one decision per task."""
    enriched: list[dict[str, Any]] = []
    for record in records:
        task_id = record["task_id"]
        if task_id not in decisions:
            raise ValueError(f"missing actor decision for {task_id}")
        decision = decisions[task_id]
        row = copy.deepcopy(record)
        row["logged_action"] = decision.get("logged_action")
        row["action_valid"] = bool(decision.get("action_valid", False))
        row["actor_raw_output"] = decision.get("actor_raw_output", "")
        row["actor_model_path"] = decision.get("actor_model_path")
        row["actor_decoding"] = copy.deepcopy(decision.get("actor_decoding", {}))
        row["actor_prompt_contract"] = decision.get("actor_prompt_contract")
        enriched.append(row)
    return enriched


def _remove_object_from_observation(observation: str, logged_action: str) -> str:
    action_parts = logged_action.split(maxsplit=1)
    if len(action_parts) != 2:
        raise ValueError(f"logged action has no removable object phrase: {logged_action!r}")
    object_phrase = action_parts[1]
    prefix, actions = observation.split("AVAILABLE ACTIONS:", 1)
    pattern = re.compile(
        rf"[^\n.!?]*\b{re.escape(object_phrase)}\b[^\n.!?]*[.!?]?",
        re.IGNORECASE,
    )
    cleaned_prefix, replacements = pattern.subn("", prefix)
    if replacements == 0:
        raise ValueError(
            f"logged action object {object_phrase!r} is absent from the observation"
        )
    cleaned_prefix = re.sub(r"[ \t]+\n", "\n", cleaned_prefix)
    cleaned_prefix = re.sub(r"\n{3,}", "\n\n", cleaned_prefix).rstrip()
    return f"{cleaned_prefix}\n\nAVAILABLE ACTIONS:{actions}"


def append_actor_dependent_corruptions(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append action-list and object-removal candidates to enriched pilot rows."""
    identities = _identity_records(records)
    for identity in identities:
        if not identity.get("action_valid") or not isinstance(
            identity.get("logged_action"), str
        ):
            raise ValueError(
                f"task {identity['task_id']} requires a valid logged action before actor corruptions"
            )
    expanded = copy.deepcopy(records)
    for row in expanded:
        row["pending_actor_corruptions"] = []
        row["candidate_admissible_actions"] = parse_initial_context(
            row["candidate_observation"]
        )["admissible_actions"]

    for identity in identities:
        logged_action = identity.get("logged_action")
        assert isinstance(logged_action, str)
        parsed = parse_initial_context(identity["real_observation"])
        if logged_action not in parsed["admissible_actions"]:
            raise ValueError(
                f"logged action {logged_action!r} is not admissible for {identity['task_id']}"
            )

        remaining_actions = [
            action for action in parsed["admissible_actions"] if action != logged_action
        ]
        action_removed_observation = (
            parsed["observation"].split("AVAILABLE ACTIONS:", 1)[0].rstrip()
            + "\nAVAILABLE ACTIONS: "
            + ", ".join(remaining_actions)
        )
        action_removed = copy.deepcopy(identity)
        action_removed.update(
            {
                "sample_id": f"{identity['task_id']}:remove_logged_action",
                "candidate_type": "remove_logged_action",
                "corruption_severity": 1,
                "expected_behavior_change": True,
                "candidate_observation": (
                    f"{parsed['task_context']}\n\n{action_removed_observation}"
                ),
                "candidate_admissible_actions": remaining_actions,
                "pending_actor_corruptions": [],
            }
        )

        object_removed_observation = _remove_object_from_observation(
            parsed["observation"], logged_action
        )
        object_removed = copy.deepcopy(identity)
        object_removed.update(
            {
                "sample_id": f"{identity['task_id']}:remove_action_object",
                "candidate_type": "remove_action_object",
                "corruption_severity": 1,
                "expected_behavior_change": True,
                "candidate_observation": (
                    f"{parsed['task_context']}\n\n{object_removed_observation}"
                ),
                "candidate_admissible_actions": parsed["admissible_actions"],
                "pending_actor_corruptions": [],
            }
        )
        expanded.extend([action_removed, object_removed])

    return expanded


def _identity_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identities = [row for row in records if row.get("candidate_type") == "identity"]
    if not identities:
        raise ValueError("pilot dataset contains no identity records")
    if len({row["task_id"] for row in identities}) != len(identities):
        raise ValueError("pilot dataset has duplicate identity records for a task")
    return identities


def generate_actor_decisions(
    records: list[dict[str, Any]],
    model_path: str,
    batch_size: int = 2,
    max_new_tokens: int = 64,
) -> dict[str, dict[str, Any]]:
    """Run the frozen actor once per identity observation on the visible CUDA device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate reference-actor actions")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    identities = _identity_records(records)
    decisions: dict[str, dict[str, Any]] = {}
    for start in range(0, len(identities), batch_size):
        batch = identities[start : start + batch_size]
        prompts = []
        for row in batch:
            messages = [
                {"role": "system", "content": TEXTWORLD_SYSTEM_PROMPT},
                {"role": "user", "content": row["real_observation"]},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[:, inputs["input_ids"].shape[1] :]
        outputs = tokenizer.batch_decode(continuation, skip_special_tokens=True)

        for row, raw_output in zip(batch, outputs):
            action, valid = parse_and_validate_action(
                raw_output, row["admissible_actions"]
            )
            decisions[row["task_id"]] = {
                "task_id": row["task_id"],
                "actor_raw_output": raw_output.strip(),
                "logged_action": action,
                "action_valid": valid,
                "actor_model_path": str(Path(model_path).resolve()),
                "actor_decoding": {
                    "do_sample": False,
                    "max_new_tokens": max_new_tokens,
                },
                "actor_prompt_contract": "Action:\n<canonical admissible action>",
            }
            print(
                f"{row['task_id']}: action={action!r} valid={valid} "
                f"({len(decisions)}/{len(identities)})"
            )
    return decisions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/pilot/textworld_initial_cpu_pilot.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/pilot/textworld_initial_actor_pilot.jsonl",
    )
    parser.add_argument(
        "--decisions-output",
        type=Path,
        default=root / "data/pilot/textworld_actor_decisions.jsonl",
    )
    parser.add_argument(
        "--model-path",
        default="/DATA/disk1/huangjiaqi_data/qwen_model/Qwen3-8B",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit-tasks", type=int)
    args = parser.parse_args()
    if args.limit_tasks is not None and args.limit_tasks < 1:
        parser.error("--limit-tasks must be positive")

    records = _read_jsonl(args.input)
    if args.limit_tasks is not None:
        allowed = list(dict.fromkeys(row["task_id"] for row in records))[: args.limit_tasks]
        records = [row for row in records if row["task_id"] in allowed]
    decisions = generate_actor_decisions(
        records,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    enriched = append_actor_dependent_corruptions(
        attach_logged_actions(records, decisions)
    )
    write_jsonl(enriched, args.output)
    write_jsonl(list(decisions.values()), args.decisions_output)
    valid = sum(bool(item["action_valid"]) for item in decisions.values())
    print(f"Wrote {len(enriched)} records; valid actor actions: {valid}/{len(decisions)}")
    return 0 if valid == len(decisions) else 2


if __name__ == "__main__":
    raise SystemExit(main())
