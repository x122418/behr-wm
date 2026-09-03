#!/usr/bin/env python3
"""Convert public Word2World TextWorld trajectories to verl GRPO parquet."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any


def _validate_trajectory_messages(messages: list[dict[str, Any]]) -> None:
    if not messages or messages[0].get("role") != "system":
        raise ValueError("trajectory must start with a system message")
    expected = ("user", "assistant")
    if any(
        message.get("role") != expected[(index - 1) % 2]
        for index, message in enumerate(messages[1:], start=1)
    ):
        raise ValueError("trajectory roles must alternate system/user/assistant")


def normalize_action_text(raw_action: str) -> str:
    """Remove known Markdown capture artifacts from a TextWorld command."""
    action = raw_action.strip()
    if action.endswith("```"):
        action = action[:-3].strip()
    if action.startswith("**"):
        action = action[2:].strip()
    if not action or "\n" in action or "\r" in action:
        raise ValueError("TextWorld action must be a single non-empty line")
    return action


def extract_transition_samples(
    trajectory: dict[str, Any], trajectory_index: int, split: str
) -> list[dict[str, Any]]:
    """Extract non-terminal next-state targets that have a following action."""
    messages = trajectory.get("messages", [])
    _validate_trajectory_messages(messages)
    normalized_messages = copy.deepcopy(messages)
    for message in normalized_messages:
        if message.get("role") == "user":
            message["content"] = normalize_action_text(message["content"])
    task_id = trajectory.get("id")
    if task_id is None:
        raise ValueError(f"trajectory {trajectory_index} has no task id")

    samples: list[dict[str, Any]] = []
    step_index = 0
    for message_index in range(2, len(normalized_messages), 2):
        if message_index + 1 >= len(normalized_messages):
            break
        samples.append(
            {
                "split": split,
                "task_id": task_id,
                "trajectory_index": trajectory_index,
                "step_index": step_index,
                "prompt_messages": copy.deepcopy(
                    normalized_messages[:message_index]
                ),
                "ground_truth_state": normalized_messages[message_index]["content"],
                "expert_action": normalized_messages[message_index + 1]["content"],
            }
        )
        step_index += 1
    return samples


def convert_transition_to_verl(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one extracted transition to the contract consumed by verl."""
    item_id = (
        f"{sample['split']}_task_{sample['task_id']}_"
        f"traj_{sample['trajectory_index']}_step_{sample['step_index']}"
    )
    prompt = copy.deepcopy(sample["prompt_messages"])
    return {
        "prompt": prompt,
        "data_source": "textworld_grpo",
        "reward_model": {
            "ground_truth": sample["ground_truth_state"],
            "style": "rule",
        },
        "extra_info": {
            "expert_action": sample["expert_action"],
            "history": copy.deepcopy(prompt),
            "task_id": sample["task_id"],
            "trajectory_index": sample["trajectory_index"],
            "step_index": sample["step_index"],
            "is_terminal": False,
            "env_reward": 0.0,
        },
        "item_id": item_id,
    }


def build_sampled_verl_rows(
    trajectories: list[dict[str, Any]],
    split: str,
    max_samples: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Build a bounded, repeatable reservoir sample without materializing all rows."""
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive or None")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total = 0
    for trajectory_index, trajectory in enumerate(trajectories):
        try:
            transitions = extract_transition_samples(
                trajectory, trajectory_index=trajectory_index, split=split
            )
        except ValueError as error:
            skipped.append(
                {
                    "task_id": trajectory.get("id"),
                    "trajectory_index": trajectory_index,
                    "reason": str(error),
                }
            )
            continue
        for transition in transitions:
            row = convert_transition_to_verl(transition)
            total += 1
            if max_samples is None:
                rows.append(row)
            elif len(rows) < max_samples:
                rows.append(row)
            else:
                replacement = rng.randrange(total)
                if replacement < max_samples:
                    rows[replacement] = row
    rng.shuffle(rows)
    return rows, total, skipped


def validate_disjoint_task_ids(
    train_trajectories: list[dict[str, Any]],
    test_trajectories: list[dict[str, Any]],
) -> None:
    """Reject task leakage across the official train and test files."""
    train_ids = {trajectory.get("id") for trajectory in train_trajectories}
    test_ids = {trajectory.get("id") for trajectory in test_trajectories}
    overlap = train_ids & test_ids
    if overlap:
        preview = sorted(overlap, key=str)[:5]
        raise ValueError(f"train/test task id overlap: {preview}")


def split_trajectories_by_task(
    trajectories: list[dict[str, Any]],
    validation_task_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
    """Create a deterministic validation split without splitting any task."""
    task_ids = sorted({trajectory.get("id") for trajectory in trajectories}, key=str)
    if None in task_ids:
        raise ValueError("all trajectories must have a task id")
    if not 0 < validation_task_count < len(task_ids):
        raise ValueError(
            "validation_task_count must be positive and smaller than the "
            "number of training tasks"
        )

    rng = random.Random(seed)
    shuffled_task_ids = task_ids.copy()
    rng.shuffle(shuffled_task_ids)
    validation_ids = sorted(
        shuffled_task_ids[:validation_task_count], key=str
    )
    validation_id_set = set(validation_ids)
    train = [
        trajectory
        for trajectory in trajectories
        if trajectory.get("id") not in validation_id_set
    ]
    validation = [
        trajectory
        for trajectory in trajectories
        if trajectory.get("id") in validation_id_set
    ]
    return train, validation, validation_ids


def _read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy")


def write_full_train_parquet(
    trajectories: list[dict[str, Any]],
    split: str,
    path: Path,
    batch_size: int = 4096,
) -> tuple[int, list[dict[str, Any]]]:
    """Stream every eligible transition to Parquet and atomically publish it."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    writer = None
    schema = None
    buffer: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total = 0

    def flush() -> None:
        nonlocal writer, schema
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=schema)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(
                temporary_path, schema, compression="snappy"
            )
        writer.write_table(table)
        buffer.clear()

    try:
        for trajectory_index, trajectory in enumerate(trajectories):
            try:
                transitions = extract_transition_samples(
                    trajectory,
                    trajectory_index=trajectory_index,
                    split=split,
                )
            except ValueError as error:
                skipped.append(
                    {
                        "task_id": trajectory.get("id"),
                        "trajectory_index": trajectory_index,
                        "reason": str(error),
                    }
                )
                continue
            for transition in transitions:
                buffer.append(convert_transition_to_verl(transition))
                total += 1
                if len(buffer) >= batch_size:
                    flush()
        flush()
        if writer is None:
            raise ValueError("no eligible transitions to write")
        writer.close()
        writer = None
        temporary_path.replace(path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    return total, skipped


def main() -> int:
    main_root = Path(__file__).resolve().parents[4]
    default_source = main_root / "data/upstream/word2world"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=main_root / "data/processed/textworld_grpo",
    )
    parser.add_argument("--smoke-size", type=int, default=64)
    parser.add_argument("--pilot-size", type=int, default=1000)
    parser.add_argument("--validation-task-count", type=int, default=250)
    parser.add_argument("--validation-pilot-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write-full-train",
        action="store_true",
        help="stream every eligible training transition to train/full.parquet",
    )
    args = parser.parse_args()
    if (
        args.smoke_size < 1
        or args.pilot_size < 1
        or args.validation_pilot_size < 1
    ):
        parser.error("sample sizes must be positive")
    if args.smoke_size > args.pilot_size:
        parser.error("--smoke-size cannot exceed --pilot-size")

    train_path = args.source_dir / "textworld_train_58805.json"
    test_path = args.source_dir / "textworld_test_173.json"
    for path in (train_path, test_path):
        if not path.is_file():
            parser.error(f"source file not found: {path}")

    print(f"Loading {train_path}")
    train_trajectories = _read_json(train_path)
    print(f"Loading {test_path}")
    test_trajectories = _read_json(test_path)
    validate_disjoint_task_ids(train_trajectories, test_trajectories)
    train_trajectories, validation_trajectories, validation_task_ids = (
        split_trajectories_by_task(
            train_trajectories,
            validation_task_count=args.validation_task_count,
            seed=args.seed,
        )
    )
    validate_disjoint_task_ids(train_trajectories, validation_trajectories)
    validate_disjoint_task_ids(validation_trajectories, test_trajectories)

    pilot_rows, total_train, skipped_train = build_sampled_verl_rows(
        train_trajectories,
        split="train",
        max_samples=args.pilot_size,
        seed=args.seed,
    )
    smoke_rows = pilot_rows[: args.smoke_size]
    validation_pilot_rows, total_validation, skipped_validation = (
        build_sampled_verl_rows(
            validation_trajectories,
            split="validation",
            max_samples=args.validation_pilot_size,
            seed=args.seed,
        )
    )
    test_rows, total_test, skipped_test = build_sampled_verl_rows(
        test_trajectories,
        split="test",
        max_samples=None,
        seed=args.seed,
    )

    smoke_path = args.output_dir / "train/smoke.parquet"
    pilot_path = args.output_dir / "train/pilot.parquet"
    test_output_path = args.output_dir / "test/test.parquet"
    validation_pilot_path = args.output_dir / "val/pilot.parquet"
    validation_full_path = args.output_dir / "val/full.parquet"
    _write_parquet(smoke_rows, smoke_path)
    _write_parquet(pilot_rows, pilot_path)
    _write_parquet(validation_pilot_rows, validation_pilot_path)
    _write_parquet(test_rows, test_output_path)
    validation_full_total, validation_full_skipped = write_full_train_parquet(
        validation_trajectories,
        split="validation",
        path=validation_full_path,
    )
    if (
        validation_full_total != total_validation
        or validation_full_skipped != skipped_validation
    ):
        raise RuntimeError("full validation output disagrees with eligibility scan")

    full_path = args.output_dir / "train/full.parquet"
    if args.write_full_train:
        full_total, full_skipped = write_full_train_parquet(
            train_trajectories,
            split="train",
            path=full_path,
        )
        if full_total != total_train or full_skipped != skipped_train:
            raise RuntimeError("full train output disagrees with eligibility scan")

    metadata = {
        "seed": args.seed,
        "source": {
            "train": str(train_path.resolve()),
            "test": str(test_path.resolve()),
        },
        "source_trajectories": {
            "train": len(train_trajectories),
            "validation": len(validation_trajectories),
            "test": len(test_trajectories),
        },
        "unique_task_ids": {
            "train": len({row["id"] for row in train_trajectories}),
            "validation": len({row["id"] for row in validation_trajectories}),
            "test": len({row["id"] for row in test_trajectories}),
            "overlap": 0,
        },
        "validation_task_ids": validation_task_ids,
        "eligible_transitions": {
            "train": total_train,
            "validation": total_validation,
            "test": total_test,
        },
        "skipped_trajectories": {
            "train": skipped_train,
            "validation": skipped_validation,
            "test": skipped_test,
        },
        "written_rows": {
            "smoke": len(smoke_rows),
            "pilot": len(pilot_rows),
            "validation_pilot": len(validation_pilot_rows),
            "validation_full": validation_full_total,
            "test": len(test_rows),
        },
        "outputs": {
            "smoke": str(smoke_path.resolve()),
            "pilot": str(pilot_path.resolve()),
            "validation_pilot": str(validation_pilot_path.resolve()),
            "validation_full": str(validation_full_path.resolve()),
            "test": str(test_output_path.resolve()),
        },
    }
    if args.write_full_train:
        metadata["written_rows"]["full"] = full_total
        metadata["outputs"]["full"] = str(full_path.resolve())
    metadata["output_sizes_bytes"] = {
        name: Path(output).stat().st_size
        for name, output in metadata["outputs"].items()
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Eligible transitions: train={total_train:,}, "
        f"validation={total_validation:,}, test={total_test:,}\n"
        f"Skipped invalid trajectories: train={len(skipped_train):,}, "
        f"validation={len(skipped_validation):,}, "
        f"test={len(skipped_test):,}\n"
        f"Wrote smoke={len(smoke_rows):,}, pilot={len(pilot_rows):,}, "
        f"validation_pilot={len(validation_pilot_rows):,}, "
        f"validation_full={validation_full_total:,}, "
        f"test={len(test_rows):,}"
        + (f", full={full_total:,}" if args.write_full_train else "")
        + f" to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
