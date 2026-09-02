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

    def _left_pad(
        self, sequences: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        maximum = max(sequence.numel() for sequence in sequences)
        batch = torch.full(
            (len(sequences), maximum), int(pad_token_id), dtype=torch.long
        )
        attention_mask = torch.zeros_like(batch)
        for row, sequence in enumerate(sequences):
            length = sequence.numel()
            batch[row, maximum - length :] = sequence
            attention_mask[row, maximum - length :] = 1
        return batch, attention_mask

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
        input_ids, attention_mask = self._left_pad([real_ids, predicted_ids])

        waiting_started = time.monotonic()
        with self._inference_lock:
            queue_wait_seconds = time.monotonic() - waiting_started
            inference_started = time.monotonic()
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    use_cache=False,
                    logits_to_keep=action_count,
                )
            inference_seconds = time.monotonic() - inference_started

        logits = outputs.logits
        if logits.ndim != 3 or logits.shape[0] != 2:
            raise RuntimeError("actor must return logits with shape [2, positions, vocab]")
        if logits.shape[1] < action_count:
            raise RuntimeError(
                f"actor returned {logits.shape[1]} positions for "
                f"{action_count} action tokens"
            )
        aligned_logits = logits[:, -action_count:, :].float()
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
