import json
import tempfile
import unittest
from pathlib import Path

from src.data.build_textworld_pilot import (
    build_pilot_dataset,
    parse_initial_context,
    write_jsonl,
)


CONTEXT_A = """Task A: open the chest.

-= Bedroom =-
You are in a bedroom. A brass lamp is here.

AVAILABLE ACTIONS: look, inventory, open chest
"""

CONTEXT_B = """Task B: inspect the key.

-= Kitchen =-
You are in a kitchen. A silver key is here.

AVAILABLE ACTIONS: look, inventory, examine key
"""


class ParseInitialContextTests(unittest.TestCase):
    def test_separates_task_from_observation_and_actions(self):
        parsed = parse_initial_context(CONTEXT_A)

        self.assertEqual(parsed["task_context"], "Task A: open the chest.")
        self.assertTrue(parsed["observation"].startswith("-= Bedroom =-"))
        self.assertEqual(
            parsed["admissible_actions"],
            ["look", "inventory", "open chest"],
        )

    def test_rejects_context_without_available_actions(self):
        with self.assertRaisesRegex(ValueError, "AVAILABLE ACTIONS"):
            parse_initial_context("Task only\n\n-= Room =-\nNothing here")


class BuildPilotDatasetTests(unittest.TestCase):
    def test_rejects_source_row_without_a_system_message(self):
        with self.assertRaisesRegex(ValueError, "messages.*system.*content"):
            build_pilot_dataset([{"id": 1, "messages": []}, {"id": 2, "messages": []}], limit=2)

    def test_builds_cpu_only_candidates_without_inventing_logged_action(self):
        source = [
            {"id": 1, "messages": [{"role": "system", "content": CONTEXT_A}]},
            {"id": 2, "messages": [{"role": "system", "content": CONTEXT_B}]},
        ]

        records = build_pilot_dataset(source, limit=2)

        self.assertEqual(len(records), 8)
        by_type = {record["candidate_type"]: record for record in records[:4]}
        self.assertEqual(
            set(by_type),
            {"identity", "action_order_reverse", "irrelevant_injection", "cross_task_swap"},
        )
        self.assertIsNone(by_type["identity"]["logged_action"])
        self.assertEqual(
            by_type["identity"]["pending_actor_corruptions"],
            ["remove_logged_action", "remove_action_object"],
        )
        self.assertEqual(by_type["identity"]["candidate_observation"], by_type["identity"]["real_observation"])
        self.assertIn(
            "AVAILABLE ACTIONS: open chest, inventory, look",
            by_type["action_order_reverse"]["candidate_observation"],
        )
        self.assertIn("A faint clock ticks somewhere far away.", by_type["irrelevant_injection"]["candidate_observation"])
        self.assertIn("Task A: open the chest.", by_type["cross_task_swap"]["candidate_observation"])
        self.assertIn("-= Kitchen =-", by_type["cross_task_swap"]["candidate_observation"])
        self.assertNotIn("Task B: inspect the key.", by_type["cross_task_swap"]["candidate_observation"])

    def test_writes_repeatable_jsonl_records(self):
        source = [
            {"id": 1, "messages": [{"role": "system", "content": CONTEXT_A}]},
            {"id": 2, "messages": [{"role": "system", "content": CONTEXT_B}]},
        ]
        first = build_pilot_dataset(source, limit=2)
        second = build_pilot_dataset(source, limit=2)

        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot.jsonl"
            write_jsonl(first, output)
            decoded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(decoded, first)

    def test_bundled_data_builds_the_default_twenty_task_artifact(self):
        root = Path(__file__).resolve().parents[1]
        source = json.loads(
            (root / "data/init_contexts/textworld/wm_instruct_test.json").read_text(encoding="utf-8")
        )

        records = build_pilot_dataset(source)

        self.assertEqual(len(records), 80)
        self.assertEqual({row["task_id"] for row in records}, {f"textworld_{i}" for i in range(1, 21)})
        self.assertEqual(len({row["sample_id"] for row in records}), 80)
        for task_id in {row["task_id"] for row in records}:
            task_records = [row for row in records if row["task_id"] == task_id]
            self.assertEqual(
                {row["candidate_type"] for row in task_records},
                {"identity", "action_order_reverse", "irrelevant_injection", "cross_task_swap"},
            )
            self.assertTrue(all(row["logged_action"] is None for row in task_records))
            self.assertTrue(
                all(
                    row["pending_actor_corruptions"]
                    == ["remove_logged_action", "remove_action_object"]
                    for row in task_records
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot.jsonl"
            write_jsonl(records, output)
            decoded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(decoded, records)


if __name__ == "__main__":
    unittest.main()
