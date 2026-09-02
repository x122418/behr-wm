#!/usr/bin/env python3
"""Score TextWorld real/candidate observations under a frozen reference actor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.build_textworld_pilot import write_jsonl
from src.data.generate_textworld_logged_actions import TEXTWORLD_SYSTEM_PROMPT


def validate_scorer_contract(
    records: list[dict[str, Any]], scorer_model_path: str
) -> dict[str, str]:
    """Enforce one frozen actor checkpoint and one real baseline per task."""
    if not records:
        raise ValueError("cannot score an empty record set")
    resolved_scorer = str(Path(scorer_model_path).resolve())
    baselines: dict[str, tuple[str, str]] = {}
    for record in records:
        actor_path = record.get("actor_model_path")
        if not isinstance(actor_path, str) or str(Path(actor_path).resolve()) != resolved_scorer:
            raise ValueError(
                f"scorer model {resolved_scorer!r} does not match actor model {actor_path!r}"
            )
        task_id = record["task_id"]
        baseline = (record["real_observation"], record["logged_action"])
        if task_id in baselines and baselines[task_id][0] != baseline[0]:
            raise ValueError(f"task {task_id} has an inconsistent real observation")
        if task_id in baselines and baselines[task_id][1] != baseline[1]:
            raise ValueError(f"task {task_id} has an inconsistent logged action")
        baselines[task_id] = baseline
    return {
        "scorer_model_path": resolved_scorer,
        "scorer_system_prompt_sha256": hashlib.sha256(
            TEXTWORLD_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "scorer_action_prefix": "Action:\n",
        "scorer_positions": "logged_action_tokens_only",
    }


def build_teacher_forced_inputs(
    tokenizer: Any, observation: str, logged_action: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build model input positions whose final logits predict every action token."""
    messages = [
        {"role": "system", "content": TEXTWORLD_SYSTEM_PROMPT},
        {"role": "user", "content": observation},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    prefix = rendered + "Action:\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + logged_action, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("tokenization merged the action with its prompt boundary")
    action_ids = full_ids[len(prefix_ids) :]
    if not action_ids:
        raise ValueError("logged action tokenized to an empty sequence")
    return (
        torch.tensor(full_ids[:-1], dtype=torch.long),
        torch.tensor(action_ids, dtype=torch.long),
    )


def _kl_from_log_probs(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    return torch.sum(log_p.exp() * (log_p - log_q))


def _js_from_probs(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Return JS divergence for normalized probability vectors, including zeros."""
    mixture = 0.5 * (p + q)
    p_term = torch.where(p > 0, p * (torch.log(p) - torch.log(mixture)), 0.0)
    q_term = torch.where(q > 0, q * (torch.log(q) - torch.log(mixture)), 0.0)
    return 0.5 * p_term.sum() + 0.5 * q_term.sum()


def compute_consistency_metrics(
    real_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    action_token_ids: torch.Tensor,
    top_ks: tuple[int, ...] = (32, 64),
) -> dict[str, float]:
    """Compute BehR and truncated distribution divergences at action positions."""
    if real_logits.ndim != 2 or candidate_logits.shape != real_logits.shape:
        raise ValueError("real and candidate logits must have the same [action_positions, vocab] shape")
    if action_token_ids.ndim != 1 or action_token_ids.shape[0] != real_logits.shape[0]:
        raise ValueError("action positions must align with action_token_ids")
    if not top_ks or any(k < 1 or k > real_logits.shape[1] for k in top_ks):
        raise ValueError("top_ks must be positive and no larger than the vocabulary")

    real_log_probs = torch.log_softmax(real_logits.float(), dim=-1)
    candidate_log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
    ids = action_token_ids.to(device=real_log_probs.device, dtype=torch.long).unsqueeze(1)
    chosen_real = real_log_probs.gather(1, ids).squeeze(1)
    chosen_candidate = candidate_log_probs.gather(1, ids).squeeze(1)
    abs_mean_diff = torch.abs(chosen_real.mean() - chosen_candidate.mean())

    result = {
        "action_token_count": int(action_token_ids.numel()),
        "mean_logprob_real": float(chosen_real.mean()),
        "mean_logprob_candidate": float(chosen_candidate.mean()),
        "original_behr_abs_mean_logprob_diff": float(abs_mean_diff),
        "original_behr_cauchy_reward": float(1.0 / (1.0 + abs_mean_diff)),
        "position_logged_token_logprob_l1": float(
            torch.mean(torch.abs(chosen_real - chosen_candidate))
        ),
        "full_vocab_kl_real_to_candidate": float(
            torch.stack(
                [
                    _kl_from_log_probs(real_log_probs[position], candidate_log_probs[position])
                    for position in range(real_log_probs.shape[0])
                ]
            ).mean()
        ),
        "full_vocab_js": float(
            torch.stack(
                [
                    _js_from_probs(
                        real_log_probs[position].exp(),
                        candidate_log_probs[position].exp(),
                    )
                    for position in range(real_log_probs.shape[0])
                ]
            ).mean()
        ),
    }

    for k in top_ks:
        real_topk_ids = torch.topk(real_log_probs, k=k, dim=-1).indices
        truncated_kls = []
        union_js_values = []
        union_other_js_values = []
        for position in range(real_log_probs.shape[0]):
            support = real_topk_ids[position]
            real_support = real_log_probs[position, support]
            candidate_support = candidate_log_probs[position, support]
            real_support = real_support - torch.logsumexp(real_support, dim=0)
            candidate_support = candidate_support - torch.logsumexp(
                candidate_support, dim=0
            )
            truncated_kls.append(_kl_from_log_probs(real_support, candidate_support))

            candidate_topk = torch.topk(
                candidate_log_probs[position], k=k, dim=-1
            ).indices
            union_support = torch.unique(torch.cat([support, candidate_topk]))
            union_real = real_log_probs[position, union_support]
            union_candidate = candidate_log_probs[position, union_support]
            union_real = union_real - torch.logsumexp(union_real, dim=0)
            union_candidate = union_candidate - torch.logsumexp(
                union_candidate, dim=0
            )
            log_mixture = torch.logaddexp(union_real, union_candidate) - torch.log(
                torch.tensor(2.0, device=union_real.device)
            )
            union_js_values.append(
                0.5 * _kl_from_log_probs(union_real, log_mixture)
                + 0.5 * _kl_from_log_probs(union_candidate, log_mixture)
            )

            union_real_probs = real_log_probs[position, union_support].exp()
            union_candidate_probs = candidate_log_probs[position, union_support].exp()
            real_other = (1.0 - union_real_probs.sum()).clamp_min(0.0).unsqueeze(0)
            candidate_other = (
                1.0 - union_candidate_probs.sum()
            ).clamp_min(0.0).unsqueeze(0)
            union_other_js_values.append(
                _js_from_probs(
                    torch.cat([union_real_probs, real_other]),
                    torch.cat([union_candidate_probs, candidate_other]),
                )
            )

        result[f"top{k}_truncated_kl_real_to_candidate"] = float(
            torch.stack(truncated_kls).mean()
        )
        result[f"top{k}_union_js"] = float(torch.stack(union_js_values).mean())
        result[f"top{k}_union_other_js"] = float(
            torch.stack(union_other_js_values).mean()
        )
    return result


def score_observation_logits(
    model: Any, tokenizer: Any, observation: str, logged_action: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU float32 logits aligned to the logged action token positions."""
    model_input_ids, action_ids = build_teacher_forced_inputs(
        tokenizer, observation, logged_action
    )
    input_ids = model_input_ids.unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=int(action_ids.numel()),
        )
    logits = outputs.logits.squeeze(0)
    if logits.shape[0] != action_ids.numel():
        raise RuntimeError(
            f"model returned {logits.shape[0]} positions for {action_ids.numel()} action tokens"
        )
    return logits.float().cpu(), action_ids


def score_records(
    records: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    scorer_model_path: str,
    top_ks: tuple[int, ...] = (32, 64),
) -> list[dict[str, Any]]:
    """Score all candidates, caching the real-observation logits once per task."""
    provenance = validate_scorer_contract(records, scorer_model_path)
    provenance["scorer_dtype"] = str(getattr(model, "dtype", "unknown"))
    real_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    scored: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        task_id = record["task_id"]
        action = record["logged_action"]
        if not record.get("action_valid") or not isinstance(action, str):
            raise ValueError(f"{record.get('sample_id', task_id)} has no valid logged action")
        if task_id not in real_cache:
            real_cache[task_id] = score_observation_logits(
                model, tokenizer, record["real_observation"], action
            )
        real_logits, action_ids = real_cache[task_id]
        if record["candidate_observation"] == record["real_observation"]:
            candidate_logits, candidate_action_ids = real_logits, action_ids
        else:
            candidate_logits, candidate_action_ids = score_observation_logits(
                model, tokenizer, record["candidate_observation"], action
            )
        if not torch.equal(action_ids, candidate_action_ids):
            raise RuntimeError(f"action token IDs differ for {record['sample_id']}")
        metrics = compute_consistency_metrics(
            real_logits, candidate_logits, action_ids, top_ks=top_ks
        )
        row = copy.deepcopy(record)
        row.update(metrics)
        row.update(provenance)
        row["scorer_top_ks"] = list(top_ks)
        scored.append(row)
        print(
            f"{record['sample_id']}: behr_diff="
            f"{metrics['original_behr_abs_mean_logprob_diff']:.6f} "
            f"top{top_ks[-1]}_js={metrics[f'top{top_ks[-1]}_union_js']:.6f} "
            f"({index}/{len(records)})"
        )
    return scored


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/pilot/textworld_initial_actor_pilot.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/pilot/textworld_actor_consistency_scores.jsonl",
    )
    parser.add_argument(
        "--model-path",
        default="/DATA/disk1/huangjiaqi_data/qwen_model/Qwen3-8B",
    )
    parser.add_argument("--top-ks", default="32,64")
    parser.add_argument("--limit-records", type=int)
    args = parser.parse_args()
    if args.limit_records is not None and args.limit_records < 1:
        parser.error("--limit-records must be positive")
    try:
        top_ks = tuple(int(value) for value in args.top_ks.split(","))
    except ValueError:
        parser.error("--top-ks must be comma-separated integers")

    records = _read_jsonl(args.input)
    if args.limit_records is not None:
        records = records[: args.limit_records]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    scored = score_records(
        records,
        model,
        tokenizer,
        scorer_model_path=args.model_path,
        top_ks=top_ks,
    )
    write_jsonl(scored, args.output)
    print(f"Wrote {len(scored)} scored records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
