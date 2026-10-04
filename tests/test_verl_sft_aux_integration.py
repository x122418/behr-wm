import inspect
from pathlib import Path
import unittest

import torch

from src.training.textworld_sft_aux import combine_actor_losses
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import ActorConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def actor_config(sft_loss_coef: float) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=4,
        ppo_micro_batch_size_per_gpu=1,
        sft_loss_coef=sft_loss_coef,
    )


class VerlSFTAuxIntegrationTests(unittest.TestCase):
    def test_zero_sft_coefficient_preserves_existing_total_loss(self):
        total = combine_actor_losses(
            pg_loss=torch.tensor(2.0),
            kl_loss=torch.tensor(3.0),
            kl_coef=0.001,
            sft_loss=torch.tensor(5.0),
            sft_coef=0.0,
        )
        torch.testing.assert_close(total, torch.tensor(2.003))

    def test_positive_coefficient_adds_weighted_sft_loss(self):
        total = combine_actor_losses(
            pg_loss=torch.tensor(2.0),
            kl_loss=torch.tensor(3.0),
            kl_coef=0.001,
            sft_loss=torch.tensor(5.0),
            sft_coef=0.1,
        )
        torch.testing.assert_close(total, torch.tensor(2.503))

    def test_actor_config_accepts_nonnegative_sft_coefficients(self):
        self.assertEqual(actor_config(0.0).sft_loss_coef, 0.0)
        self.assertEqual(actor_config(0.1).sft_loss_coef, 0.1)

    def test_actor_config_rejects_negative_sft_coefficient(self):
        with self.assertRaisesRegex(ValueError, "sft_loss_coef must be nonnegative"):
            actor_config(-0.1)

    def test_actor_accepts_tokenizer_and_logs_auxiliary_metrics(self):
        parameters = inspect.signature(DataParallelPPOActor.__init__).parameters
        self.assertIn("tokenizer", parameters)
        source = Path(inspect.getfile(DataParallelPPOActor)).read_text(encoding="utf-8")
        self.assertIn('metrics["actor/sft_loss"]', source)
        self.assertIn('metrics["actor/total_loss"]', source)
        self.assertIn('micro_batch_metrics["actor/sft_loss_coef"]', source)


if __name__ == "__main__":
    unittest.main()
