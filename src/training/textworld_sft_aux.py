"""Auxiliary supervised-loss helpers for TextWorld world-model training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from verl.utils.model import compute_position_id_with_mask


def _encode_target(
    tokenizer: Any, ground_truth: str, max_response_length: int
) -> list[int]:
    if not isinstance(ground_truth, str):
        raise TypeError("ground truths must be strings")
    if max_response_length < 1:
        raise ValueError("max_response_length must be positive")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    token_ids = list(tokenizer.encode(ground_truth, add_special_tokens=False))
    if len(token_ids) + 1 > max_response_length:
        token_ids = token_ids[: max_response_length - 1]
    return [*token_ids, int(eos_token_id)]


def build_aux_sft_batch(
    *,
    prompts: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    ground_truths: Sequence[str],
    tokenizer: Any,
    max_response_length: int,
) -> dict[str, torch.Tensor]:
    """Build teacher-forced real-observation tensors beside a GRPO batch."""
    if prompts.ndim != 2 or prompt_attention_mask.shape != prompts.shape:
        raise ValueError("prompts and prompt_attention_mask must be matching 2D tensors")
    if len(ground_truths) != prompts.shape[0]:
        raise ValueError("ground_truths must contain one value per prompt")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")

    encoded = [
        _encode_target(tokenizer, value, max_response_length)
        for value in ground_truths
    ]
    target_width = max_response_length
    responses = torch.full(
        (len(encoded), target_width),
        int(pad_token_id),
        dtype=prompts.dtype,
        device=prompts.device,
    )
    response_mask = torch.zeros(
        (len(encoded), target_width),
        dtype=prompt_attention_mask.dtype,
        device=prompt_attention_mask.device,
    )
    for row_index, token_ids in enumerate(encoded):
        width = len(token_ids)
        responses[row_index, :width] = torch.tensor(
            token_ids, dtype=prompts.dtype, device=prompts.device
        )
        response_mask[row_index, :width] = 1

    attention_mask = torch.cat((prompt_attention_mask, response_mask), dim=1)
    input_ids = torch.cat((prompts, responses), dim=1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": compute_position_id_with_mask(attention_mask),
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
    }


def token_mean_nll(
    log_probs: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """Return negative mean log-probability over supervised target tokens."""
    if log_probs.shape != response_mask.shape:
        raise ValueError("log_probs and response_mask must have the same shape")
    token_count = response_mask.sum()
    if token_count.item() == 0:
        raise ValueError("SFT loss requires at least one target token")
    return -(log_probs * response_mask).sum() / token_count


def combine_actor_losses(
    *,
    pg_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    kl_coef: float,
    sft_loss: torch.Tensor,
    sft_coef: float,
) -> torch.Tensor:
    """Combine the three world-model actor objectives."""
    return pg_loss + kl_coef * kl_loss + sft_coef * sft_loss
