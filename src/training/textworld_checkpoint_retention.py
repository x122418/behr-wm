#!/usr/bin/env python3
"""Safely enforce checkpoint retention for a completed formal TextWorld run."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys


MANIFEST_NAME = "formal_run_manifest.json"
TRACKER_NAME = "latest_checkpointed_iteration.txt"
CHECKPOINT_PATTERN = re.compile(r"global_step_([0-9]+)")


def validated_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    """Return tracked checkpoints only after validating the run lifecycle."""
    output_dir = Path(output_dir)
    if not (output_dir / MANIFEST_NAME).is_file():
        raise ValueError(f"checkpoint pruning requires a formal run manifest: {MANIFEST_NAME}")

    tracker = output_dir / TRACKER_NAME
    tracker_text = tracker.read_text(encoding="utf-8").strip() if tracker.is_file() else ""
    if not tracker_text.isdigit():
        raise ValueError("checkpoint tracker must contain one non-negative integer")
    latest_step = int(tracker_text)

    checkpoints: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.name.startswith("global_step_"):
            continue
        match = CHECKPOINT_PATTERN.fullmatch(child.name)
        if match is None or not child.is_dir() or child.is_symlink():
            raise ValueError(f"invalid checkpoint entry: {child}")
        checkpoints.append((int(match.group(1)), child))

    steps = {step for step, _ in checkpoints}
    if latest_step not in steps:
        raise ValueError(f"tracked latest checkpoint is missing: global_step_{latest_step}")
    future_steps = sorted(step for step in steps if step > latest_step)
    if future_steps:
        raise ValueError(
            "checkpoint directory is newer than tracker: "
            + ", ".join(f"global_step_{step}" for step in future_steps)
        )

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def prune_checkpoints(output_dir: Path, keep: int) -> list[Path]:
    """Remove old completed checkpoints while preserving the tracked latest ones."""
    if not isinstance(keep, int) or isinstance(keep, bool) or keep <= 0:
        raise ValueError("checkpoint retention must be a positive integer")

    checkpoints = validated_checkpoints(output_dir)
    to_remove = checkpoints[:-keep]
    removed: list[Path] = []
    for _, path in to_remove:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        removed = prune_checkpoints(args.output_dir, args.keep)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for path in removed:
        print(f"Removed old checkpoint: {path}")
    print(f"Checkpoint retention complete: kept latest {args.keep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
