import subprocess
import sys
import unittest
from pathlib import Path

from src.data.label_textworld_actor_behavior import (
    attach_behavior_labels,
    convert_generated_decisions,
    prepare_candidate_generation_records,
    select_pending_candidates,
)


class AttachBehaviorLabelsTests(unittest.TestCase):
    def test_labels_valid_candidate_actions_against_the_real_action(self):
        records = [
            {
                "sample_id": "task_1:identity",
                "task_id": "task_1",
                "candidate_type": "identity",
                "logged_action": "open chest",
                "action_valid": True,
            },
            {
                "sample_id": "task_1:corrupted",
                "task_id": "task_1",
                "candidate_type": "corrupted",
                "logged_action": "open chest",
                "action_valid": True,
            },
        ]
        decisions = {
            "task_1:corrupted": {
                "candidate_actor_raw_output": "Action:\nlook",
                "candidate_action": "look",
                "candidate_action_valid": True,
                "candidate_actor_model_path": "/models/actor",
            }
        }

        labeled = attach_behavior_labels(records, decisions)

        self.assertEqual(labeled[0]["real_action"], "open chest")
        self.assertEqual(labeled[0]["candidate_action"], "open chest")
        self.assertFalse(labeled[0]["observed_top1_change"])
        self.assertEqual(labeled[1]["candidate_action"], "look")
        self.assertTrue(labeled[1]["observed_top1_change"])
        self.assertEqual(labeled[1]["candidate_actor_model_path"], "/models/actor")
        self.assertNotIn("real_action", records[0], "inputs must not be mutated")

    def test_keeps_behavior_change_unknown_for_an_invalid_candidate_output(self):
        records = [
            {
                "sample_id": "task_1:corrupted",
                "task_id": "task_1",
                "candidate_type": "corrupted",
                "logged_action": "open chest",
                "action_valid": True,
            }
        ]
        decisions = {
            "task_1:corrupted": {
                "candidate_actor_raw_output": "I would look around.",
                "candidate_action": None,
                "candidate_action_valid": False,
            }
        }

        labeled = attach_behavior_labels(records, decisions)

        self.assertIsNone(labeled[0]["observed_top1_change"])

    def test_rejects_a_missing_nonidentity_decision(self):
        records = [
            {
                "sample_id": "task_1:corrupted",
                "task_id": "task_1",
                "candidate_type": "corrupted",
                "logged_action": "open chest",
                "action_valid": True,
            }
        ]

        with self.assertRaisesRegex(ValueError, "task_1:corrupted"):
            attach_behavior_labels(records, {})


class SelectPendingCandidatesTests(unittest.TestCase):
    def test_skips_identity_rows_and_already_completed_decisions(self):
        records = [
            {"sample_id": "task_1:identity", "candidate_type": "identity"},
            {"sample_id": "task_1:first", "candidate_type": "corrupted"},
            {"sample_id": "task_1:second", "candidate_type": "corrupted"},
        ]

        pending = select_pending_candidates(
            records, {"task_1:first": {"candidate_action": "look"}}
        )

        self.assertEqual([row["sample_id"] for row in pending], ["task_1:second"])

    def test_adapts_candidate_rows_to_the_existing_actor_generator_contract(self):
        records = [
            {
                "sample_id": "task_1:corrupted",
                "candidate_type": "corrupted",
                "candidate_observation": "candidate room",
                "candidate_admissible_actions": ["look", "inventory"],
            }
        ]

        prepared = prepare_candidate_generation_records(records)

        self.assertEqual(prepared[0]["task_id"], "task_1:corrupted")
        self.assertEqual(prepared[0]["candidate_type"], "identity")
        self.assertEqual(prepared[0]["real_observation"], "candidate room")
        self.assertEqual(prepared[0]["admissible_actions"], ["look", "inventory"])

    def test_renames_existing_generator_output_to_candidate_fields(self):
        generated = {
            "task_1:corrupted": {
                "actor_raw_output": "Action:\nlook",
                "logged_action": "look",
                "action_valid": True,
                "actor_model_path": "/models/actor",
                "actor_decoding": {"do_sample": False},
                "actor_prompt_contract": "Action:\n<canonical admissible action>",
            }
        }

        converted = convert_generated_decisions(generated)

        self.assertEqual(converted["task_1:corrupted"]["candidate_action"], "look")
        self.assertTrue(
            converted["task_1:corrupted"]["candidate_action_valid"]
        )
        self.assertEqual(
            converted["task_1:corrupted"]["candidate_actor_model_path"],
            "/models/actor",
        )


class CommandLineTests(unittest.TestCase):
    def test_direct_script_invocation_can_show_help(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "src/data/label_textworld_actor_behavior.py", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--decisions-output", result.stdout)

    def test_cli_rejects_a_nonpositive_record_limit_before_loading_a_model(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "src/data/label_textworld_actor_behavior.py",
                "--limit-records",
                "0",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
