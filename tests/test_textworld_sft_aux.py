import unittest

import torch

from src.training.textworld_sft_aux import (
    attach_aux_sft_batch,
    build_aux_sft_batch,
    combine_actor_losses,
    token_mean_nll,
)
from tensordict import TensorDict


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {
            "real one": [11, 12],
            "real two": [21],
            "long": [31, 32, 33, 34],
        }[text]


class TextWorldSFTAuxTests(unittest.TestCase):
    def test_attach_aux_sft_batch_does_not_mutate_locked_input(self):
        batch = TensorDict(
            {"input_ids": torch.tensor([[1, 2]])}, batch_size=[1]
        ).lock_()
        result = attach_aux_sft_batch(
            batch, {"input_ids": torch.tensor([[3, 4]])}
        )

        self.assertTrue(batch.is_locked)
        self.assertNotIn("sft_input_ids", batch.keys())
        self.assertFalse(result.is_locked)
        torch.testing.assert_close(
            result["sft_input_ids"], torch.tensor([[3, 4]])
        )

    def test_build_aux_sft_batch_masks_prompt_and_target_padding(self):
        result = build_aux_sft_batch(
            prompts=torch.tensor([[0, 7, 8], [5, 6, 7]]),
            prompt_attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
            ground_truths=["real one", "real two"],
            tokenizer=FakeTokenizer(),
            max_response_length=4,
        )

        self.assertEqual(
            result["responses"].tolist(), [[11, 12, 99, 0], [21, 99, 0, 0]]
        )
        self.assertEqual(
            result["response_mask"].tolist(), [[1, 1, 1, 0], [1, 1, 0, 0]]
        )
        self.assertEqual(
            result["attention_mask"].tolist(),
            [[0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0]],
        )
        self.assertEqual(result["input_ids"][:, :3].tolist(), [[0, 7, 8], [5, 6, 7]])
        self.assertEqual(result["prompts"].tolist(), [[0, 7, 8], [5, 6, 7]])

    def test_target_truncation_retains_eos(self):
        result = build_aux_sft_batch(
            prompts=torch.tensor([[7, 8]]),
            prompt_attention_mask=torch.ones(1, 2, dtype=torch.long),
            ground_truths=["long"],
            tokenizer=FakeTokenizer(),
            max_response_length=3,
        )

        self.assertEqual(result["responses"].tolist(), [[31, 32, 99]])
        self.assertEqual(result["response_mask"].tolist(), [[1, 1, 1]])

    def test_rollout_duplication_does_not_change_token_mean_loss(self):
        log_probs = torch.tensor([[-1.0, -2.0, -9.0], [-3.0, -4.0, -5.0]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        base = token_mean_nll(log_probs, mask)
        repeated = token_mean_nll(
            log_probs.repeat_interleave(4, dim=0),
            mask.repeat_interleave(4, dim=0),
        )

        torch.testing.assert_close(repeated, base)

    def test_token_mean_nll_rejects_empty_mask(self):
        with self.assertRaisesRegex(ValueError, "at least one target token"):
            token_mean_nll(torch.zeros(1, 2), torch.zeros(1, 2))

    def test_combine_actor_losses_applies_both_coefficients(self):
        total = combine_actor_losses(
            pg_loss=torch.tensor(2.0),
            kl_loss=torch.tensor(3.0),
            kl_coef=0.001,
            sft_loss=torch.tensor(5.0),
            sft_coef=0.1,
        )

        torch.testing.assert_close(total, torch.tensor(2.503))


if __name__ == "__main__":
    unittest.main()
