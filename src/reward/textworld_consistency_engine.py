"""Single-model engine for TextWorld actor-distribution consistency."""

from __future__ import annotations

import threading
import time
from typing import Any

import torch

from src.reward.actor_distribution_metrics import (
    compute_actor_distribution_metrics,
    js_consistency_reward,
)
from src.reward.textworld_actor_inputs import build_teacher_forced_actor_inputs


class TextWorldConsistencyEngine:
    """Own one frozen actor and score real/predicted observations together."""

    SUPPORTED_REWARD_METRICS = {"union_topk_other_js", "full_vocab_js"}

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        top_k: int = 64,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.top_k = top_k
        self.device = torch.device(model.device)
        self.dtype = str(getattr(model, "dtype", "unknown"))
        self._inference_lock = threading.Lock()

    @staticmethod
    def reward_from_js(js_divergence: float) -> float:
        return js_consistency_reward(js_divergence)

    def score(
        self,
        history: list[dict[str, str]],
        real_observation: str,
        predicted_observation: str,
        expert_action: str,
        reward_metric: str,
    ) -> dict[str, Any]:
        if reward_metric not in self.SUPPORTED_REWARD_METRICS:
            raise ValueError(f"unsupported reward_metric: {reward_metric}")
        real_ids, real_action_ids = build_teacher_forced_actor_inputs(
            self.tokenizer, history, real_observation, expert_action
        )
        predicted_ids, predicted_action_ids = build_teacher_forced_actor_inputs(
            self.tokenizer, history, predicted_observation, expert_action
        )
        if not torch.equal(real_action_ids, predicted_action_ids):
            raise ValueError("real and predicted prompts produced different action IDs")
        action_count = int(real_action_ids.numel())
        waiting_started = time.monotonic()
        with self._inference_lock:
            queue_wait_seconds = time.monotonic() - waiting_started
            inference_started = time.monotonic()
            with torch.inference_mode():
                output_logits = []
                # Separate unpadded forwards reproduce the locked offline scorer.
                # Padding unequal prompts causes material numerical drift on long
                # trajectories under the reference actor.
                for sequence in (real_ids, predicted_ids):
                    input_ids = sequence.unsqueeze(0).to(self.device)
                    attention_mask = torch.ones_like(input_ids)
                    position_ids = attention_mask.long().cumsum(dim=-1) - 1
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        logits_to_keep=action_count,
                    )
                    output_logits.append(outputs.logits.squeeze(0))
            inference_seconds = time.monotonic() - inference_started

        for logits in output_logits:
            if logits.ndim != 2 or logits.shape[0] < action_count:
                raise RuntimeError(
                    "actor must return at least one logit row per action token"
                )
        aligned_logits = torch.stack(
            [logits[-action_count:, :].float() for logits in output_logits]
        )
        metrics = compute_actor_distribution_metrics(
            aligned_logits[0],
            aligned_logits[1],
            real_action_ids.to(aligned_logits.device),
            top_ks=(self.top_k,),
        )
        if reward_metric == "union_topk_other_js":
            divergence = metrics[f"top{self.top_k}_union_other_js"]
        else:
            divergence = metrics["full_vocab_js"]
        return {
            **metrics,
            "score": js_consistency_reward(divergence),
            "reward_metric": reward_metric,
            "model": self.model_name,
            "dtype": self.dtype,
            "queue_wait_seconds": queue_wait_seconds,
            "inference_seconds": inference_seconds,
        }

    def score_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score concurrent requests while reusing each distinct real input."""
        if not requests:
            return []

        groups: dict[
            tuple[tuple[int, ...], tuple[int, ...]], dict[str, Any]
        ] = {}
        for index, request in enumerate(requests):
            reward_metric = request["reward_metric"]
            if reward_metric not in self.SUPPORTED_REWARD_METRICS:
                raise ValueError(f"unsupported reward_metric: {reward_metric}")
            real_ids, real_action_ids = build_teacher_forced_actor_inputs(
                self.tokenizer,
                request["history"],
                request["real_observation"],
                request["expert_action"],
            )
            predicted_ids, predicted_action_ids = build_teacher_forced_actor_inputs(
                self.tokenizer,
                request["history"],
                request["predicted_observation"],
                request["expert_action"],
            )
            if not torch.equal(real_action_ids, predicted_action_ids):
                raise ValueError(
                    "real and predicted prompts produced different action IDs"
                )
            key = (
                tuple(int(token) for token in real_ids.tolist()),
                tuple(int(token) for token in real_action_ids.tolist()),
            )
            group = groups.setdefault(
                key,
                {
                    "real_ids": real_ids,
                    "action_ids": real_action_ids,
                    "items": [],
                },
            )
            group["items"].append(
                {
                    "index": index,
                    "predicted_ids": predicted_ids,
                    "reward_metric": reward_metric,
                }
            )

        jobs_by_length: dict[int, list[dict[str, Any]]] = {}
        for group in groups.values():
            action_count = int(group["action_ids"].numel())
            real_job = {
                "sequence": group["real_ids"],
                "action_count": action_count,
                "logits": None,
            }
            group["real_job"] = real_job
            jobs_by_length.setdefault(int(group["real_ids"].numel()), []).append(
                real_job
            )
            for item in group["items"]:
                predicted_job = {
                    "sequence": item["predicted_ids"],
                    "action_count": action_count,
                    "logits": None,
                }
                item["predicted_job"] = predicted_job
                jobs_by_length.setdefault(
                    int(item["predicted_ids"].numel()), []
                ).append(predicted_job)

        waiting_started = time.monotonic()
        with self._inference_lock:
            queue_wait_seconds = time.monotonic() - waiting_started
            inference_started = time.monotonic()
            with torch.inference_mode():
                for jobs in jobs_by_length.values():
                    input_ids = torch.stack(
                        [job["sequence"] for job in jobs]
                    ).to(self.device)
                    attention_mask = torch.ones_like(input_ids)
                    position_ids = attention_mask.long().cumsum(dim=-1) - 1
                    max_action_count = max(job["action_count"] for job in jobs)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        logits_to_keep=max_action_count,
                    )
                    logits = outputs.logits
                    if (
                        logits.ndim != 3
                        or logits.shape[0] != len(jobs)
                        or logits.shape[1] < max_action_count
                    ):
                        raise RuntimeError(
                            "actor must return one batched logit row per action token"
                        )
                    for row, job in enumerate(jobs):
                        action_count = job["action_count"]
                        job["logits"] = logits[row, -action_count:, :].float()
            inference_seconds = time.monotonic() - inference_started

        results: list[dict[str, Any] | None] = [None] * len(requests)
        for group in groups.values():
            real_logits = group["real_job"]["logits"]
            action_ids = group["action_ids"].to(real_logits.device)
            for item in group["items"]:
                predicted_logits = item["predicted_job"]["logits"]
                metrics = compute_actor_distribution_metrics(
                    real_logits,
                    predicted_logits,
                    action_ids,
                    top_ks=(self.top_k,),
                )
                if item["reward_metric"] == "union_topk_other_js":
                    divergence = metrics[f"top{self.top_k}_union_other_js"]
                else:
                    divergence = metrics["full_vocab_js"]
                results[item["index"]] = {
                    **metrics,
                    "score": js_consistency_reward(divergence),
                    "reward_metric": item["reward_metric"],
                    "model": self.model_name,
                    "dtype": self.dtype,
                    "queue_wait_seconds": queue_wait_seconds,
                    "inference_seconds": inference_seconds,
                }

        if any(result is None for result in results):
            raise RuntimeError("batched scorer did not produce every result")
        return [result for result in results if result is not None]
