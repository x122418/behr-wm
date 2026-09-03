#!/usr/bin/env python3
"""Audit TextWorld GRPO parquet lengths with verl's prompt-tokenization contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import islice
from pathlib import Path
from statistics import median
from typing import Any, Iterable


def summarize_lengths(lengths: list[int], limit: int | None = None) -> dict[str, Any]:
    if not lengths:
        raise ValueError("cannot summarize an empty length collection")
    ordered = sorted(lengths)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    over_limit = sum(length > limit for length in ordered) if limit is not None else 0
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
        "over_limit": over_limit,
        "over_limit_fraction": over_limit / len(ordered),
    }


def audit_rows(
    rows: Iterable[dict[str, Any]], tokenizer: Any, max_prompt_length: int
) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    ground_truth_lengths: list[int] = []
    action_lengths: list[int] = []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=True
        )
        prompt_lengths.append(len(prompt_ids))
        ground_truth_lengths.append(
            len(tokenizer.encode(row["reward_model"]["ground_truth"], add_special_tokens=False))
        )
        action_lengths.append(
            len(tokenizer.encode(row["extra_info"]["expert_action"], add_special_tokens=False))
        )
    return {
        "max_prompt_length": max_prompt_length,
        "prompt_tokens": summarize_lengths(prompt_lengths, max_prompt_length),
        "ground_truth_tokens": summarize_lengths(ground_truth_lengths),
        "expert_action_tokens": summarize_lengths(action_lengths),
    }


def audit_rows_batched(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    max_prompt_length: int,
    batch_size: int = 256,
    progress_every: int | None = 10_000,
) -> dict[str, Any]:
    """Audit lengths with batched tokenizer calls and no padding or truncation."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if progress_every is not None and progress_every < 1:
        raise ValueError("progress_every must be positive or None")

    row_iterator = iter(rows)
    prompt_lengths: list[int] = []
    ground_truth_lengths: list[int] = []
    action_lengths: list[int] = []
    processed = 0
    next_progress = progress_every

    while batch := list(islice(row_iterator, batch_size)):
        prompt_ids = tokenizer.apply_chat_template(
            [row["prompt"] for row in batch],
            add_generation_prompt=True,
            tokenize=True,
        )
        ground_truth_ids = tokenizer(
            [row["reward_model"]["ground_truth"] for row in batch],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        action_ids = tokenizer(
            [row["extra_info"]["expert_action"] for row in batch],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        prompt_lengths.extend(map(len, prompt_ids))
        ground_truth_lengths.extend(map(len, ground_truth_ids))
        action_lengths.extend(map(len, action_ids))
        processed += len(batch)
        if next_progress is not None and processed >= next_progress:
            print(f"Audited {processed:,} rows", file=sys.stderr, flush=True)
            while next_progress <= processed:
                next_progress += progress_every

    return {
        "max_prompt_length": max_prompt_length,
        "prompt_tokens": summarize_lengths(prompt_lengths, max_prompt_length),
        "ground_truth_tokens": summarize_lengths(ground_truth_lengths),
        "expert_action_tokens": summarize_lengths(action_lengths),
    }


def iter_parquet_rows(paths: list[Path]) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    columns = ["prompt", "reward_model", "extra_info"]
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=1024):
            yield from batch.to_pylist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local tokenizer/model path")
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=14336)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_prompt_length <= 0:
        raise SystemExit("--max-prompt-length must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    missing = [str(path) for path in [args.model, *args.parquet] if not path.exists()]
    if missing:
        raise SystemExit(f"missing input path(s): {', '.join(missing)}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    report = {
        "model": str(args.model.resolve()),
        "parquet": [str(path.resolve()) for path in args.parquet],
        **audit_rows_batched(
            iter_parquet_rows(args.parquet),
            tokenizer,
            args.max_prompt_length,
            batch_size=args.batch_size,
        ),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
