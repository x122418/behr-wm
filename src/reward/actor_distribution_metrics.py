"""Actor-distribution consistency metrics at logged action positions."""

from __future__ import annotations

import math

import torch


def _kl_from_log_probs(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    return torch.sum(log_p.exp() * (log_p - log_q))


def _js_from_probs(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    mixture = 0.5 * (p + q)
    p_term = torch.where(p > 0, p * (torch.log(p) - torch.log(mixture)), 0.0)
    q_term = torch.where(q > 0, q * (torch.log(q) - torch.log(mixture)), 0.0)
    return 0.5 * p_term.sum() + 0.5 * q_term.sum()


def js_consistency_reward(js_divergence: float) -> float:
    """Map natural-log JS divergence to a bounded consistency reward."""
    if not math.isfinite(js_divergence) or js_divergence < 0:
        raise ValueError("js_divergence must be finite and non-negative")
    return min(1.0, max(0.0, 1.0 - js_divergence / math.log(2.0)))


def compute_actor_distribution_metrics(
    real_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    action_token_ids: torch.Tensor,
    top_ks: tuple[int, ...] = (32, 64),
) -> dict[str, float]:
    """Compute logged-path BehR diagnostics and distribution divergences."""
    if real_logits.ndim != 2 or candidate_logits.shape != real_logits.shape:
        raise ValueError(
            "real and candidate logits must have the same "
            "[action_positions, vocab] shape"
        )
    if action_token_ids.ndim != 1 or action_token_ids.shape[0] != real_logits.shape[0]:
        raise ValueError("action positions must align with action_token_ids")
    if not top_ks or any(k < 1 or k > real_logits.shape[1] for k in top_ks):
        raise ValueError("top_ks must be positive and no larger than the vocabulary")
    if not torch.isfinite(real_logits).all() or not torch.isfinite(candidate_logits).all():
        raise ValueError("logits must be finite")

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
                    _kl_from_log_probs(
                        real_log_probs[position], candidate_log_probs[position]
                    )
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
            union_real_normalized = union_real - torch.logsumexp(union_real, dim=0)
            union_candidate_normalized = union_candidate - torch.logsumexp(
                union_candidate, dim=0
            )
            log_mixture = torch.logaddexp(
                union_real_normalized, union_candidate_normalized
            ) - math.log(2.0)
            union_js_values.append(
                0.5 * _kl_from_log_probs(union_real_normalized, log_mixture)
                + 0.5 * _kl_from_log_probs(union_candidate_normalized, log_mixture)
            )

            union_real_probs = union_real.exp()
            union_candidate_probs = union_candidate.exp()
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
