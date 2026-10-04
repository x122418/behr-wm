#!/usr/bin/env python3
"""Create or validate an immutable manifest for a formal TextWorld run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


MANIFEST_NAME = "formal_run_manifest.json"


def prepare_run(output_dir: Path, manifest: dict[str, Any], resume: bool) -> Path:
    """Prepare a new output directory or validate an exact checkpoint resume."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_NAME

    if resume:
        if not manifest_path.is_file():
            raise ValueError(f"resume requires an existing {MANIFEST_NAME}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = sorted(
            key
            for key in existing.keys() | manifest.keys()
            if existing.get(key) != manifest.get(key)
        )
        if mismatches:
            raise ValueError(
                "resume manifest mismatch for keys: " + ", ".join(mismatches)
            )
        tracker = output_dir / "latest_checkpointed_iteration.txt"
        if not tracker.is_file() or not tracker.read_text(encoding="utf-8").strip():
            raise ValueError("resume requires a non-empty checkpoint tracker")
        checkpoint_step = int(tracker.read_text(encoding="utf-8").strip())
        total_steps = int(manifest["total_steps"])
        if checkpoint_step >= total_steps:
            raise ValueError(
                f"checkpoint step {checkpoint_step} has already reached "
                f"configured total_steps {total_steps}"
            )
        return manifest_path

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_dir / f".{MANIFEST_NAME}.tmp"
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new-run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--reward-mode", required=True)
    parser.add_argument("--sft-loss-coef", type=float, required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--scorer-url", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--actor-data-loader-seed", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--rollout-temperature", type=float, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument("--val-freq", type=int, required=True)
    parser.add_argument("--max-actor-ckpt-to-keep", type=int, required=True)
    parser.add_argument("--actor-lr", type=float, required=True)
    parser.add_argument("--reward-num-workers", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--lora-target-modules", required=True)
    parser.add_argument("--rollout-load-format", required=True)
    parser.add_argument(
        "--layered-summon",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument(
        "--filter-overlong-prompts",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = {
        "schema_version": 3,
        "arm": args.arm,
        "reward_mode": args.reward_mode,
        "sft_loss_coef": args.sft_loss_coef,
        "train_data": str(Path(args.train_data).resolve()),
        "val_data": str(Path(args.val_data).resolve()),
        "world_model": str(Path(args.world_model).resolve()),
        "actor_model": str(Path(args.actor_model).resolve()),
        "scorer_url": args.scorer_url,
        "seed": args.seed,
        "actor_data_loader_seed": args.actor_data_loader_seed,
        "group_size": args.group_size,
        "rollout_temperature": args.rollout_temperature,
        "total_steps": args.total_steps,
        "save_freq": args.save_freq,
        "val_freq": args.val_freq,
        "max_actor_ckpt_to_keep": args.max_actor_ckpt_to_keep,
        "actor_lr": args.actor_lr,
        "reward_num_workers": args.reward_num_workers,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": args.lora_target_modules,
        "rollout_load_format": args.rollout_load_format,
        "layered_summon": args.layered_summon,
        "filter_overlong_prompts": args.filter_overlong_prompts,
    }
    try:
        path = prepare_run(args.output_dir, manifest, resume=args.resume)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
