#!/usr/bin/env python3
"""Compare a verified TextWorld pilot matrix with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
from typing import Sequence

from scripts.run_textworld_pilot_eval_matrix import read_unique_jsonl


DEFAULT_MODELS = (
    "base",
    "behr_step50",
    "behr_step100",
    "union_js_step50",
    "union_js_step100",
    "union_js_sft_step50",
    "union_js_sft_step100",
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool


DEFAULT_METRICS = (
    MetricSpec("exact_match", True),
    MetricSpec("original_behr_cauchy_reward", True),
    MetricSpec("full_vocab_js", False),
    MetricSpec("full_vocab_kl_real_to_candidate", False),
    MetricSpec("original_behr_abs_mean_logprob_diff", False),
    MetricSpec("position_logged_token_logprob_l1", False),
    MetricSpec("top32_union_other_js", False),
    MetricSpec("top64_union_other_js", False),
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_interval(
    base: Sequence[float],
    candidate: Sequence[float],
    *,
    higher_is_better: bool,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if len(base) != len(candidate) or not base:
        raise ValueError("paired bootstrap requires equal non-empty samples")
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    sign = 1.0 if higher_is_better else -1.0
    improvements = [sign * (other - original) for original, other in zip(base, candidate)]
    rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(samples):
        bootstrap_means.append(
            _mean([improvements[rng.randrange(len(improvements))] for _ in improvements])
        )
    bootstrap_means.sort()
    return (
        _mean(improvements),
        _quantile(bootstrap_means, 0.025),
        _quantile(bootstrap_means, 0.975),
    )


def _metric_values(rows: dict[str, dict], ids: Sequence[str], metric: str) -> list[float]:
    values = []
    for item_id in ids:
        value = rows[item_id].get(metric)
        if isinstance(value, bool):
            value = float(value)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{item_id}: metric {metric!r} is missing or non-numeric")
        values.append(float(value))
    return values


def compare_matrix(
    evaluation_root: Path,
    model_names: Sequence[str],
    metrics: Sequence[MetricSpec],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    if not model_names or model_names[0] != "base":
        raise ValueError("the first model must be 'base'")
    model_rows = {
        name: read_unique_jsonl(evaluation_root / name / "results.jsonl")
        for name in model_names
    }
    ids = sorted(model_rows["base"])
    reference = set(ids)
    for name, rows in model_rows.items():
        if set(rows) != reference:
            raise ValueError(f"{name}: item IDs differ from base")

    comparisons: dict[str, dict[str, dict[str, float | int | bool | None]]] = {}
    for model_index, name in enumerate(model_names[1:], start=1):
        comparisons[name] = {}
        for metric_index, metric in enumerate(metrics):
            base_values = _metric_values(model_rows["base"], ids, metric.name)
            model_values = _metric_values(model_rows[name], ids, metric.name)
            improvement, ci_low, ci_high = paired_bootstrap_interval(
                base_values,
                model_values,
                higher_is_better=metric.higher_is_better,
                samples=bootstrap_samples,
                seed=seed + model_index * 10_000 + metric_index,
            )
            base_mean = _mean(base_values)
            model_mean = _mean(model_values)
            raw_delta = model_mean - base_mean
            comparisons[name][metric.name] = {
                "n": len(ids),
                "higher_is_better": metric.higher_is_better,
                "base_mean": base_mean,
                "model_mean": model_mean,
                "raw_delta": raw_delta,
                "relative_delta": raw_delta / abs(base_mean) if base_mean else None,
                "improvement": improvement,
                "improvement_ci95_low": ci_low,
                "improvement_ci95_high": ci_high,
            }
    return {
        "schema_version": 1,
        "base_model": "base",
        "models": list(model_names),
        "item_count": len(ids),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "comparisons": comparisons,
    }


def write_report(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = (
        "model",
        "metric",
        "n",
        "higher_is_better",
        "base_mean",
        "model_mean",
        "raw_delta",
        "relative_delta",
        "improvement",
        "improvement_ci95_low",
        "improvement_ci95_high",
    )
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model, metric_rows in report["comparisons"].items():
            for metric, values in metric_rows.items():
                writer.writerow({"model": model, "metric": metric, **values})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    args.evaluation_root = args.evaluation_root.resolve()
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else args.evaluation_root / "analysis"
    )
    args.models = [value.strip() for value in args.models.split(",") if value.strip()]
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = compare_matrix(
            args.evaluation_root,
            args.models,
            DEFAULT_METRICS,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        write_report(report, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote: {args.output_dir / 'comparison.json'}")
    print(f"Wrote: {args.output_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
