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
