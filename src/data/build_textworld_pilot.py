#!/usr/bin/env python3
"""Build a CPU-only TextWorld observation-consistency pilot dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ACTION_MARKER = "AVAILABLE ACTIONS:"
ROOM_HEADER_RE = re.compile(r"(?m)^-=\s+.+?\s+=-.*$")
IRRELEVANT_SENTENCE = "A faint clock ticks somewhere far away."
PENDING_ACTOR_CORRUPTIONS = ["remove_logged_action", "remove_action_object"]


def parse_initial_context(context: str) -> dict[str, Any]:
    """Split a TextWorld init context into task, observation, and actions."""
    if ACTION_MARKER not in context:
        raise ValueError("TextWorld context is missing AVAILABLE ACTIONS")

    room_match = ROOM_HEADER_RE.search(context)
    if room_match is None:
        raise ValueError("TextWorld context is missing a room header")

    task_context = context[: room_match.start()].strip()
    observation = context[room_match.start() :].strip()
    actions_text = observation.split(ACTION_MARKER, 1)[1].strip()
    actions = [action.strip() for action in actions_text.split(",") if action.strip()]
    if not actions:
        raise ValueError("TextWorld context has an empty AVAILABLE ACTIONS list")

    return {
        "task_context": task_context,
        "observation": observation,
        "admissible_actions": actions,
    }


def _join_context(task_context: str, observation: str) -> str:
    return f"{task_context}\n\n{observation}".strip()


def _replace_actions(observation: str, actions: list[str]) -> str:
    prefix = observation.split(ACTION_MARKER, 1)[0].rstrip()
    return f"{prefix}\n{ACTION_MARKER} {', '.join(actions)}"


def _inject_irrelevant_sentence(observation: str) -> str:
    prefix, actions = observation.split(ACTION_MARKER, 1)
    return f"{prefix.rstrip()}\n\n{IRRELEVANT_SENTENCE}\n\n{ACTION_MARKER}{actions}"


def _record(
    *,
    task_id: str,
    source_index: int,
    parsed: dict[str, Any],
    candidate_type: str,
    candidate_observation: str,
    expected_behavior_change: bool,
    severity: int,
) -> dict[str, Any]:
    real_observation = _join_context(parsed["task_context"], parsed["observation"])
    return {
        "sample_id": f"{task_id}:{candidate_type}",
        "task_id": task_id,
        "source_index": source_index,
        "instruction": parsed["task_context"],
        "history": [],
        "real_observation": real_observation,
        "admissible_actions": parsed["admissible_actions"],
        "logged_action": None,
        "action_valid": None,
        "candidate_type": candidate_type,
        "corruption_severity": severity,
        "expected_behavior_change": expected_behavior_change,
        "candidate_observation": candidate_observation,
        "pending_actor_corruptions": PENDING_ACTOR_CORRUPTIONS.copy(),
    }


def _context_from_source_row(row: dict[str, Any], index: int) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"source row {index} must contain messages with a system content field")
    first = messages[0]
    if not isinstance(first, dict) or first.get("role") != "system" or not isinstance(first.get("content"), str):
        raise ValueError(f"source row {index} must contain messages with a system content field")
    return first["content"]


def build_pilot_dataset(source: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Build four deterministic candidates per initial TextWorld task."""
    if limit < 1:
        raise ValueError("limit must be positive")
    selected = source[:limit]
    if len(selected) < 2:
        raise ValueError("at least two tasks are required for cross-task swaps")

    parsed_rows = [
        parse_initial_context(_context_from_source_row(row, index))
        for index, row in enumerate(selected)
    ]
    records: list[dict[str, Any]] = []

    for index, (row, parsed) in enumerate(zip(selected, parsed_rows)):
        task_id = f"textworld_{row.get('id', index)}"
        real = _join_context(parsed["task_context"], parsed["observation"])
        reversed_observation = _replace_actions(
            parsed["observation"], list(reversed(parsed["admissible_actions"]))
        )
        injected_observation = _inject_irrelevant_sentence(parsed["observation"])
        donor = parsed_rows[(index + 1) % len(parsed_rows)]
        swapped_observation = _join_context(parsed["task_context"], donor["observation"])

        candidates = (
            ("identity", real, False, 0),
            (
                "action_order_reverse",
                _join_context(parsed["task_context"], reversed_observation),
                False,
                1,
            ),
            (
                "irrelevant_injection",
                _join_context(parsed["task_context"], injected_observation),
                False,
                1,
            ),
            ("cross_task_swap", swapped_observation, True, 1),
        )
        for candidate_type, candidate, expected_change, severity in candidates:
            records.append(
                _record(
                    task_id=task_id,
                    source_index=index,
                    parsed=parsed,
                    candidate_type=candidate_type,
                    candidate_observation=candidate,
                    expected_behavior_change=expected_change,
                    severity=severity,
                )
            )

    return records


def write_jsonl(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/init_contexts/textworld/wm_instruct_test.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/pilot/textworld_initial_cpu_pilot.jsonl",
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    records = build_pilot_dataset(source, limit=args.limit)
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} records from {min(args.limit, len(source))} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
