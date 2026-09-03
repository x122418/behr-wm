#!/usr/bin/env python3
"""Label whether candidate observations change the frozen actor's top-1 action."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.build_textworld_pilot import write_jsonl
from src.data.generate_textworld_logged_actions import generate_actor_decisions


def select_pending_candidates(
    records: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return non-identity candidates that do not yet have saved decisions."""
    return [
        record
        for record in records
        if record.get("candidate_type") != "identity"
        and record["sample_id"] not in decisions
    ]


def prepare_candidate_generation_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Adapt candidate rows to the existing one-decision-per-identity generator."""
    prepared: list[dict[str, Any]] = []
    for record in records:
        prepared.append(
            {
                "task_id": record["sample_id"],
                "candidate_type": "identity",
                "real_observation": record["candidate_observation"],
                "admissible_actions": record["candidate_admissible_actions"],
            }
        )
    return prepared


def convert_generated_decisions(
    generated: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Rename real-action generator fields for candidate-side provenance."""
    converted: dict[str, dict[str, Any]] = {}
    for sample_id, decision in generated.items():
        converted[sample_id] = {
            "sample_id": sample_id,
            "candidate_actor_raw_output": decision.get("actor_raw_output", ""),
            "candidate_action": decision.get("logged_action"),
            "candidate_action_valid": bool(decision.get("action_valid", False)),
            "candidate_actor_model_path": decision.get("actor_model_path"),
            "candidate_actor_decoding": copy.deepcopy(
                decision.get("actor_decoding", {})
            ),
            "candidate_actor_prompt_contract": decision.get(
                "actor_prompt_contract"
            ),
        }
    return converted


def attach_behavior_labels(
    records: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach candidate actions and an exact-match behavior-change label."""
    labeled: list[dict[str, Any]] = []
    for record in records:
        row = copy.deepcopy(record)
        real_action = row.get("logged_action")
        if not row.get("action_valid") or not isinstance(real_action, str):
            raise ValueError(f"{row.get('sample_id', row.get('task_id'))} has no valid real action")

        if row.get("candidate_type") == "identity":
            decision = {
                "candidate_actor_raw_output": row.get("actor_raw_output", ""),
                "candidate_action": real_action,
                "candidate_action_valid": True,
                "candidate_actor_model_path": row.get("actor_model_path"),
                "candidate_actor_decoding": copy.deepcopy(
                    row.get("actor_decoding", {})
                ),
                "candidate_actor_prompt_contract": row.get(
                    "actor_prompt_contract"
                ),
            }
        else:
            sample_id = row["sample_id"]
            if sample_id not in decisions:
                raise ValueError(f"missing candidate actor decision for {sample_id}")
            decision = decisions[sample_id]

        candidate_action = decision.get("candidate_action")
        candidate_valid = bool(decision.get("candidate_action_valid", False))
        row["real_action"] = real_action
        row["candidate_actor_raw_output"] = decision.get(
            "candidate_actor_raw_output", ""
        )
        row["candidate_action"] = candidate_action
        row["candidate_action_valid"] = candidate_valid
        row["candidate_actor_model_path"] = decision.get(
            "candidate_actor_model_path"
        )
        row["candidate_actor_decoding"] = copy.deepcopy(
            decision.get("candidate_actor_decoding", {})
        )
        row["candidate_actor_prompt_contract"] = decision.get(
            "candidate_actor_prompt_contract"
        )
        row["observed_top1_change"] = (
            candidate_action != real_action
            if candidate_valid and isinstance(candidate_action, str)
            else None
        )
        labeled.append(row)
    return labeled


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/pilot/textworld_actor_consistency_scores.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/pilot/textworld_actor_behavior_labels.jsonl",
    )
    parser.add_argument(
        "--decisions-output",
        type=Path,
        default=root / "data/pilot/textworld_candidate_actor_decisions.jsonl",
    )
    parser.add_argument(
        "--model-path",
        default="/DATA/disk1/huangjiaqi_data/qwen_model/Qwen3-8B",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit-records", type=int)
    args = parser.parse_args()
    if args.limit_records is not None and args.limit_records < 1:
        parser.error("--limit-records must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    records = _read_jsonl(args.input)
    if args.limit_records is not None:
        records = records[: args.limit_records]
    saved_rows = (
        _read_jsonl(args.decisions_output) if args.decisions_output.exists() else []
    )
    decisions = {row["sample_id"]: row for row in saved_rows}
    pending = select_pending_candidates(records, decisions)
    if pending:
        generated = generate_actor_decisions(
            prepare_candidate_generation_records(pending),
            model_path=args.model_path,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        decisions.update(convert_generated_decisions(generated))
        ordered_decisions = [
            decisions[row["sample_id"]]
            for row in records
            if row.get("candidate_type") != "identity"
            and row["sample_id"] in decisions
        ]
        write_jsonl(ordered_decisions, args.decisions_output)
        print(f"Wrote {len(ordered_decisions)} candidate decisions to {args.decisions_output}")

    labeled = attach_behavior_labels(records, decisions)
    write_jsonl(labeled, args.output)
    valid = sum(row["candidate_action_valid"] for row in labeled)
    changed = sum(row["observed_top1_change"] is True for row in labeled)
    print(
        f"Wrote {len(labeled)} behavior labels to {args.output}; "
        f"valid={valid}, changed={changed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
