#!/usr/bin/env python3
"""Convert scalar metrics in verl console step lines to TensorBoard events."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_MARKER = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+")


def parse_step_metrics(line: str) -> tuple[int | None, dict[str, float]]:
    """Extract one verl step and its finite scalar metrics."""
    clean = ANSI_ESCAPE.sub("", line)
    match = STEP_MARKER.search(clean)
    if match is None:
        return None, {}
    step = int(match.group(1))
    metrics: dict[str, float] = {}
    for field in clean[match.end() :].split(" - "):
        if ":" not in field:
            continue
        key, raw_value = field.rsplit(":", 1)
        try:
            value = float(raw_value.strip())
        except ValueError:
            continue
        if key and math.isfinite(value):
            metrics[key.strip()] = value
    return step, metrics


def convert(input_path: Path, output_dir: Path) -> tuple[int, int]:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(output_dir))
    steps = 0
    scalars = 0
    try:
        for line in input_path.read_text(encoding="utf-8", errors="replace").splitlines():
            step, metrics = parse_step_metrics(line)
            if step is None:
                continue
            steps += 1
            for key, value in metrics.items():
                writer.add_scalar(key, value, step)
                scalars += 1
    finally:
        writer.close()
    return steps, scalars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input log not found: {args.input}")
    steps, scalars = convert(args.input, args.output_dir)
    print(f"converted {steps} steps and {scalars} scalars to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
