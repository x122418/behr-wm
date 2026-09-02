"""Shared TextWorld actor prompt construction and action-token alignment."""

from __future__ import annotations

from typing import Any

import torch


TEXTWORLD_ACTOR_SYSTEM_PROMPT = (
    "You are playing a text adventure game called TextWorld.\n"
    "Each round you will receive an observation describing your surroundings "
    "and a list of available actions.\n"
    "You must choose one of the available actions to proceed.\n"
    "Your response should be a single action command, for example:\n"
    "open chest drawer\ngo east\ntake old key\nput apple on stove\n"
    "Choose the best action to complete your assigned task."
)


def build_textworld_actor_messages(
    history: list[dict[str, str]], observation: str
) -> list[dict[str, str]]:
    """Convert world-model history roles into the reference actor's view."""
    messages = [{"role": "system", "content": TEXTWORLD_ACTOR_SYSTEM_PROMPT}]
    initial_observation = next(
        (
            message.get("content", "")
            for message in history
            if message.get("role") == "system"
        ),
        "",
    )
    if initial_observation:
        messages.append({"role": "user", "content": initial_observation})
    for message in history:
        role = message.get("role")
        content = message.get("content", "")
        if not content or role == "system":
            continue
        if role == "user":
            messages.append({"role": "assistant", "content": content})
        elif role == "assistant":
            messages.append({"role": "user", "content": content})
    messages.append({"role": "user", "content": observation})
    return messages


def build_teacher_forced_actor_inputs(
    tokenizer: Any,
    history: list[dict[str, str]],
    observation: str,
    expert_action: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal inputs and aligned logged-action token IDs."""
    if not isinstance(expert_action, str) or not expert_action.strip():
        raise ValueError("expert_action must be a non-empty string")
    messages = build_textworld_actor_messages(history, observation)
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking")
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    if prompt and not prompt.endswith("\n"):
        prompt += "\n"

    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + expert_action, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("tokenization merged the action with its prompt boundary")
    action_ids = full_ids[len(prefix_ids) :]
    if not action_ids:
        raise ValueError("expert_action tokenized to zero tokens")
    return (
        torch.tensor(full_ids[:-1], dtype=torch.long),
        torch.tensor(action_ids, dtype=torch.long),
    )
