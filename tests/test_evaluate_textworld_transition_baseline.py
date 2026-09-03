import unittest

from src.data.evaluate_textworld_transition_baseline import (
    build_actor_inputs,
    summarize_results,
    transition_from_row,
)


class RecordingCharacterTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "CHAT\nASSISTANT:"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


class BuildActorInputsTests(unittest.TestCase):
    def test_uses_reward_prompt_contract_and_aligns_every_action_token(self):
        tokenizer = RecordingCharacterTokenizer()
        history = [
            {"role": "system", "content": "initial room"},
            {"role": "user", "content": "open door"},
            {"role": "assistant", "content": "door opens"},
        ]

        model_inputs, action_ids = build_actor_inputs(
            tokenizer, history, "current room", "go east"
        )

        self.assertEqual(
            tokenizer.messages[1:],
            [
                {"role": "user", "content": "initial room"},
                {"role": "assistant", "content": "open door"},
                {"role": "user", "content": "door opens"},
                {"role": "user", "content": "current room"},
            ],
        )
        expected_full = "CHAT\nASSISTANT:\ngo east"
        self.assertEqual(action_ids.tolist(), [ord(c) for c in "go east"])
        self.assertEqual(model_inputs.tolist(), [ord(c) for c in expected_full[:-1]])


class SummarizeResultsTests(unittest.TestCase):
    def test_reports_exact_match_and_means_over_successful_rows(self):
        rows = [
            {
                "status": "ok",
                "task_id": 101,
                "exact_match": True,
                "original_behr_cauchy_reward": 1.0,
                "top64_union_other_js": 0.0,
                "full_vocab_js": 0.1,
            },
            {
                "status": "ok",
                "task_id": 202,
                "exact_match": False,
                "original_behr_cauchy_reward": 0.5,
                "top64_union_other_js": 0.2,
                "full_vocab_js": 0.3,
            },
            {"status": "error", "error": "timeout"},
        ]

        summary = summarize_results(rows)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertAlmostEqual(summary["exact_match"], 0.5)
        self.assertAlmostEqual(summary["original_behr_cauchy_reward"], 0.75)
        self.assertAlmostEqual(summary["top64_union_other_js"], 0.1)
        self.assertAlmostEqual(summary["full_vocab_js"], 0.2)
        self.assertNotIn("task_id", summary)


class TransitionFromRowTests(unittest.TestCase):
    def test_extracts_the_world_model_and_actor_contract(self):
        row = {
            "item_id": "test_task_1_traj_2_step_3",
            "prompt": (
                {"role": "system", "content": "room"},
                {"role": "user", "content": "look"},
            ),
            "reward_model": {"ground_truth": "real next state"},
            "extra_info": {
                "expert_action": "go east",
                "history": (
                    {"role": "system", "content": "room"},
                    {"role": "user", "content": "look"},
                ),
                "task_id": 1,
            },
        }

        transition = transition_from_row(row)

        self.assertEqual(transition["item_id"], row["item_id"])
        self.assertEqual(transition["wm_messages"], list(row["prompt"]))
        self.assertEqual(transition["history"], list(row["extra_info"]["history"]))
        self.assertEqual(transition["real_observation"], "real next state")
        self.assertEqual(transition["logged_action"], "go east")
        self.assertEqual(transition["task_id"], 1)


if __name__ == "__main__":
    unittest.main()
