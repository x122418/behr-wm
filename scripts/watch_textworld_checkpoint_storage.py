#!/usr/bin/env python3
"""Continuously archive TextWorld LoRA adapters and prune full checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.textworld_checkpoint_archive import (  # noqa: E402
    ArchivePruneResult,
    archive_and_prune_checkpoints,
)


DEFAULT_ARMS = ("union_js", "union_js_sft", "behr")


@dataclass(frozen=True)
class RetentionCycle:
    results: dict[str, ArchivePruneResult]
    errors: dict[str, str]


def run_retention_cycle(
    formal_root: Path,
    *,
    arms: Sequence[str] = DEFAULT_ARMS,
    keep: int = 2,
) -> RetentionCycle:
    """Run one isolated archive/prune operation for each requested arm."""
    formal_root = Path(formal_root)
    results: dict[str, ArchivePruneResult] = {}
    errors: dict[str, str] = {}
    for arm in arms:
        try:
            results[arm] = archive_and_prune_checkpoints(
                formal_root / "checkpoints" / arm,
                formal_root / "lora_adapter_archive" / arm,
                keep=keep,
            )
        except (OSError, ValueError) as error:
            errors[arm] = str(error)
    return RetentionCycle(results=results, errors=errors)


def _tracker_token(formal_root: Path, arm: str) -> str:
    tracker = formal_root / "checkpoints" / arm / "latest_checkpointed_iteration.txt"
    try:
        return tracker.read_text(encoding="utf-8").strip()
    except OSError as error:
        return f"error:{error}"


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _report(cycle: RetentionCycle) -> None:
    for arm, result in cycle.results.items():
        archived = ",".join(map(str, result.archived_steps)) or "none"
        verified = ",".join(map(str, result.verified_steps)) or "none"
        removed = ",".join(path.name for path in result.removed_checkpoints) or "none"
        _log(
            f"{arm}: archived={archived} verified={verified} removed={removed}"
        )
    for arm, error in cycle.errors.items():
        _log(f"{arm}: skipped safely: {error}")


def watch_checkpoint_storage(
    formal_root: Path,
    *,
    arms: Sequence[str],
    keep: int,
    interval_seconds: float,
    once: bool,
) -> int:
    if interval_seconds <= 0:
        raise ValueError("interval seconds must be positive")

    formal_root = Path(formal_root)
    successful_trackers: dict[str, str] = {}
    while True:
        changed_arms = tuple(
            arm
            for arm in arms
            if successful_trackers.get(arm) != _tracker_token(formal_root, arm)
        )
        if changed_arms:
            cycle = run_retention_cycle(formal_root, arms=changed_arms, keep=keep)
            _report(cycle)
            for arm in cycle.results:
                successful_trackers[arm] = _tracker_token(formal_root, arm)
        else:
            cycle = RetentionCycle(results={}, errors={})

        if once:
            return 1 if cycle.errors else 0
        time.sleep(interval_seconds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return watch_checkpoint_storage(
            args.formal_root,
            arms=tuple(args.arms),
            keep=args.keep,
            interval_seconds=args.interval_seconds,
            once=args.once,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        _log("stopped")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
