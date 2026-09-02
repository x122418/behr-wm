import math
import unittest

import torch

from src.reward.actor_distribution_metrics import (
    compute_actor_distribution_metrics,
    js_consistency_reward,
)


class ActorDistributionMetricTests(unittest.TestCase):
    def test_js_reward_maps_bounds_to_one_and_zero(self):
        self.assertAlmostEqual(js_consistency_reward(0.0), 1.0)
        self.assertAlmostEqual(js_consistency_reward(math.log(2.0)), 0.0)

    def test_identical_logits_have_zero_divergence(self):
        logits = torch.tensor([[2.0, 0.0, -1.0]])

        metrics = compute_actor_distribution_metrics(
            logits, logits.clone(), torch.tensor([0]), top_ks=(2,)
        )

        self.assertAlmostEqual(metrics["full_vocab_js"], 0.0, places=7)
        self.assertAlmostEqual(metrics["top2_union_other_js"], 0.0, places=7)

    def test_rejects_non_finite_logits(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_actor_distribution_metrics(
                torch.tensor([[0.0, float("nan")]]),
                torch.zeros(1, 2),
                torch.tensor([0]),
                top_ks=(1,),
            )

    def test_rejects_invalid_top_k(self):
        for top_ks in ((), (0,), (3,)):
            with self.subTest(top_ks=top_ks):
                with self.assertRaisesRegex(ValueError, "top_ks"):
                    compute_actor_distribution_metrics(
                        torch.zeros(1, 2),
                        torch.zeros(1, 2),
                        torch.tensor([0]),
                        top_ks=top_ks,
                    )

    def test_rejects_invalid_js_reward_input(self):
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    js_consistency_reward(value)


if __name__ == "__main__":
    unittest.main()
