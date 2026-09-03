import subprocess
import sys
import unittest
from pathlib import Path

from src.data.generate_textworld_logged_actions import (
    append_actor_dependent_corruptions,
    attach_logged_actions,
    parse_and_validate_action,
)


class ParseAndValidateActionTests(unittest.TestCase):
    def test_extracts_an_exact_canonical_admissible_action(self):
        action, valid = parse_and_validate_action(
            "Action:\nopen antique trunk",
            ["look", "open antique trunk"],
        )

        self.assertEqual(action, "open antique trunk")
        self.assertTrue(valid)

    def test_rejects_an_action_outside_the_admissible_list(self):
        action, valid = parse_and_validate_action(
            "Action:\ngo north",
            ["look", "open antique trunk"],
        )

        self.assertEqual(action, "go north")
        self.assertFalse(valid)

    def test_rejects_output_without_an_action_field(self):
        action, valid = parse_and_validate_action(
            "I would inspect the room first.",
            ["look", "inventory"],
        )

        self.assertIsNone(action)
        self.assertFalse(valid)

    def test_rejects_malformed_or_noncanonical_action_outputs(self):
        malformed = [
            "Thought first\nAction:\nlook",
            "Action: look",
            'Action:\n"look"',
            "Action:\nlook.",
            "Action:\nLook",
            "Action:\nlook\nAction:\ninventory",
            "Action:\nlook\nextra prose",
        ]

        for raw_output in malformed:
            with self.subTest(raw_output=raw_output):
                action, valid = parse_and_validate_action(
                    raw_output, ["look", "inventory"]
                )
                self.assertFalse(valid)


class AttachLoggedActionsTests(unittest.TestCase):
    def test_propagates_one_actor_decision_to_every_candidate_for_a_task(self):
        records = [
            {
                "sample_id": "textworld_1:identity",
                "task_id": "textworld_1",
                "admissible_actions": ["look", "open chest"],
                "logged_action": None,
                "action_valid": None,
            },
            {
                "sample_id": "textworld_1:cross_task_swap",
                "task_id": "textworld_1",
                "admissible_actions": ["look", "open chest"],
                "logged_action": None,
                "action_valid": None,
            },
        ]
        decisions = {
            "textworld_1": {
                "actor_raw_output": "Action: open chest",
                "logged_action": "open chest",
                "action_valid": True,
                "actor_model_path": "/models/qwen3-8b",
                "actor_decoding": {"do_sample": False, "max_new_tokens": 64},
                "actor_prompt_contract": "Action:\n<canonical admissible action>",
            }
        }

        enriched = attach_logged_actions(records, decisions)

        self.assertEqual([row["logged_action"] for row in enriched], ["open chest", "open chest"])
        self.assertTrue(all(row["action_valid"] for row in enriched))
        self.assertTrue(all(row["actor_raw_output"] == "Action: open chest" for row in enriched))
        self.assertTrue(all(row["actor_model_path"] == "/models/qwen3-8b" for row in enriched))
        self.assertTrue(all(row["actor_decoding"]["do_sample"] is False for row in enriched))
        self.assertIsNone(records[0]["logged_action"], "input records must not be mutated")

    def test_rejects_missing_task_decisions(self):
        records = [{"task_id": "textworld_1"}, {"task_id": "textworld_2"}]

        with self.assertRaisesRegex(ValueError, "textworld_2"):
            attach_logged_actions(records, {"textworld_1": {"logged_action": "look"}})

    def test_appends_logged_action_and_object_removal_candidates(self):
        real = (
            "Task: open the chest.\n\n-= Bedroom =-\n"
            "You see a brass lamp. You see a closed chest drawer.\n\n"
            "AVAILABLE ACTIONS: look, examine chest drawer, open chest drawer"
        )
        records = [
            {
                "sample_id": "textworld_1:identity",
                "task_id": "textworld_1",
                "source_index": 0,
                "instruction": "Task: open the chest.",
                "history": [],
                "real_observation": real,
                "admissible_actions": ["look", "examine chest drawer", "open chest drawer"],
                "logged_action": "open chest drawer",
                "action_valid": True,
                "actor_raw_output": "Action: open chest drawer",
                "candidate_type": "identity",
                "corruption_severity": 0,
                "expected_behavior_change": False,
                "candidate_observation": real,
                "pending_actor_corruptions": ["remove_logged_action", "remove_action_object"],
            }
        ]

        expanded = append_actor_dependent_corruptions(records)

        self.assertEqual(len(expanded), 3)
        by_type = {row["candidate_type"]: row for row in expanded}
        self.assertNotIn(
            "open chest drawer",
            by_type["remove_logged_action"]["candidate_admissible_actions"],
        )
        self.assertIn("examine chest drawer", by_type["remove_logged_action"]["candidate_observation"])
        object_removed = by_type["remove_action_object"]["candidate_observation"]
        observation_body = object_removed.split("-= Bedroom =-", 1)[1].split("AVAILABLE ACTIONS:", 1)[0]
        self.assertNotIn("chest drawer", observation_body)
        self.assertIn("Task: open the chest.", object_removed)
        self.assertIn("open chest drawer", by_type["remove_action_object"]["candidate_admissible_actions"])
        self.assertTrue(all(row["pending_actor_corruptions"] == [] for row in expanded))

    def test_actor_corruptions_require_a_valid_logged_action(self):
        records = [
            {
                "task_id": "textworld_1",
                "candidate_type": "identity",
                "logged_action": None,
                "action_valid": False,
            }
        ]

        with self.assertRaisesRegex(ValueError, "valid logged action"):
            append_actor_dependent_corruptions(records)


class CommandLineTests(unittest.TestCase):
    def test_direct_script_invocation_can_show_help(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "src/data/generate_textworld_logged_actions.py", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model-path", result.stdout)

    def test_cli_rejects_a_nonpositive_task_limit_before_loading_a_model(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "src/data/generate_textworld_logged_actions.py",
                "--limit-tasks",
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
