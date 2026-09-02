import math
import unittest

import torch

from src.data.score_textworld_actor_consistency import (
    build_teacher_forced_inputs,
    compute_consistency_metrics,
    validate_scorer_contract,
)


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return f"CHAT:{messages[-1]['content']}\nASSISTANT:\n"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


class BuildTeacherForcedInputsTests(unittest.TestCase):
    def test_prefills_action_header_and_omits_final_action_token_from_model_input(self):
        model_input_ids, action_token_ids = build_teacher_forced_inputs(
            CharacterTokenizer(), "room", "look"
        )

        expected_full = "CHAT:room\nASSISTANT:\nAction:\nlook"
        self.assertEqual(action_token_ids.tolist(), [ord(c) for c in "look"])
        self.assertEqual(model_input_ids.tolist(), [ord(c) for c in expected_full[:-1]])


class ComputeConsistencyMetricsTests(unittest.TestCase):
    def test_identity_distributions_have_zero_divergence_and_unit_behr_reward(self):
        logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])

        metrics = compute_consistency_metrics(
            logits, logits.clone(), torch.tensor([0, 2]), top_ks=(2,)
        )

        self.assertAlmostEqual(metrics["original_behr_abs_mean_logprob_diff"], 0.0, places=7)
        self.assertAlmostEqual(metrics["original_behr_cauchy_reward"], 1.0, places=7)
        self.assertAlmostEqual(metrics["position_logged_token_logprob_l1"], 0.0, places=7)
        self.assertAlmostEqual(metrics["top2_truncated_kl_real_to_candidate"], 0.0, places=7)
        self.assertAlmostEqual(metrics["top2_union_js"], 0.0, places=7)
        self.assertAlmostEqual(metrics["top2_union_other_js"], 0.0, places=7)
        self.assertAlmostEqual(metrics["full_vocab_kl_real_to_candidate"], 0.0, places=7)
        self.assertAlmostEqual(metrics["full_vocab_js"], 0.0, places=7)

    def test_metrics_match_a_hand_computed_single_position_example(self):
        real_logits = torch.log(torch.tensor([[0.50, 0.25, 0.25]]))
        candidate_logits = torch.log(torch.tensor([[0.25, 0.50, 0.25]]))

        metrics = compute_consistency_metrics(
            real_logits, candidate_logits, torch.tensor([0]), top_ks=(2,)
        )

        self.assertAlmostEqual(
            metrics["original_behr_abs_mean_logprob_diff"], math.log(2), places=6
        )
        self.assertAlmostEqual(
            metrics["position_logged_token_logprob_l1"], math.log(2), places=6
        )
        self.assertAlmostEqual(
            metrics["top2_truncated_kl_real_to_candidate"], math.log(2) / 3, places=6
        )
        expected_js = (2 / 3) * math.log(4 / 3) + (1 / 3) * math.log(2 / 3)
        self.assertAlmostEqual(metrics["top2_union_js"], expected_js, places=6)
        expected_full_js = 0.5 * math.log(4 / 3) + 0.25 * math.log(2 / 3)
        self.assertAlmostEqual(metrics["top2_union_other_js"], expected_full_js, places=6)
        self.assertAlmostEqual(
            metrics["full_vocab_kl_real_to_candidate"], math.log(2) / 4, places=6
        )
        self.assertAlmostEqual(metrics["full_vocab_js"], expected_full_js, places=6)

    def test_rejects_misaligned_action_positions(self):
        with self.assertRaisesRegex(ValueError, "action positions"):
            compute_consistency_metrics(
                torch.zeros(2, 3),
                torch.zeros(2, 3),
                torch.tensor([0]),
                top_ks=(2,),
            )


class ValidateScorerContractTests(unittest.TestCase):
    def test_rejects_a_scorer_checkpoint_different_from_the_logged_action_actor(self):
        records = [
            {
                "task_id": "textworld_1",
                "real_observation": "real",
                "logged_action": "look",
                "actor_model_path": "/models/actor",
            }
        ]

        with self.assertRaisesRegex(ValueError, "scorer model.*actor model"):
            validate_scorer_contract(records, "/models/different")

    def test_rejects_inconsistent_real_baselines_within_a_task(self):
        records = [
            {
                "task_id": "textworld_1",
                "real_observation": "real one",
                "logged_action": "look",
                "actor_model_path": "/models/actor",
            },
            {
                "task_id": "textworld_1",
                "real_observation": "real two",
                "logged_action": "look",
                "actor_model_path": "/models/actor",
            },
        ]

        with self.assertRaisesRegex(ValueError, "inconsistent real observation"):
            validate_scorer_contract(records, "/models/actor")

    def test_returns_resolved_scorer_provenance_for_a_valid_contract(self):
        records = [
            {
                "task_id": "textworld_1",
                "real_observation": "real",
                "logged_action": "look",
                "actor_model_path": "/models/actor",
            }
        ]

        provenance = validate_scorer_contract(records, "/models/actor")

        self.assertEqual(provenance["scorer_model_path"], "/models/actor")
        self.assertEqual(provenance["scorer_action_prefix"], "Action:\n")
        self.assertIn("scorer_system_prompt_sha256", provenance)


if __name__ == "__main__":
    unittest.main()
