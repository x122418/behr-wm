#!/usr/bin/env python3
"""Compare consistency-service metrics with locked offline result rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import requests


DEFAULT_METRICS = (
    "original_behr_cauchy_reward",
    "full_vocab_kl_real_to_candidate",
    "full_vocab_js",
    "top64_union_other_js",
)


def compare_metric_rows(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    metric_names: tuple[str, ...] = DEFAULT_METRICS,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Return absolute differences or fail on missing/non-finite/mismatched metrics."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    differences = {}
    for name in metric_names:
        reference_value = reference.get(name)
        candidate_value = candidate.get(name)
        if not isinstance(reference_value, (int, float)) or not math.isfinite(
            reference_value
        ):
            raise ValueError(f"reference metric {name} is missing or non-finite")
        if not isinstance(candidate_value, (int, float)) or not math.isfinite(
            candidate_value
        ):
            raise ValueError(f"candidate metric {name} is missing or non-finite")
        difference = abs(float(reference_value) - float(candidate_value))
        differences[name] = difference
        if difference > tolerance:
            raise ValueError(
                f"metric {name} differs by {difference:.9g}, tolerance={tolerance:.9g}"
            )
    return differences


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:8002")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not args.input.is_file():
        parser.error(f"input not found: {args.input}")

    rows = [row for row in read_jsonl(args.input) if row.get("status") == "ok"]
    rows = rows[: args.limit]
    if len(rows) < args.limit:
        raise SystemExit(
            f"requested {args.limit} successful rows but found only {len(rows)}"
        )

    session = requests.Session()
    session.trust_env = False
    endpoint = args.service_url.rstrip("/") + "/v1/behavior-consistency"
    maximum_differences = {name: 0.0 for name in DEFAULT_METRICS}
    compared = []
    for index, row in enumerate(rows, start=1):
        response = session.post(
            endpoint,
            json={
                "history": row["history"],
                "real_observation": row["real_observation"],
                "predicted_observation": row["predicted_observation"],
                "expert_action": row["logged_action"],
                "top_k": args.top_k,
                "reward_metric": "union_topk_other_js",
            },
            timeout=300,
        )
        response.raise_for_status()
        service_metrics = response.json()
        differences = compare_metric_rows(
            row, service_metrics, tolerance=args.tolerance
        )
        for name, difference in differences.items():
            maximum_differences[name] = max(maximum_differences[name], difference)
        compared.append(
            {"item_id": row["item_id"], "absolute_differences": differences}
        )
        print(f"compared {index}/{len(rows)}: {row['item_id']}")

    report = {
        "status": "pass",
        "input": str(args.input.resolve()),
        "service_url": args.service_url,
        "count": len(rows),
        "tolerance": args.tolerance,
        "maximum_absolute_differences": maximum_differences,
        "rows": compared,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
