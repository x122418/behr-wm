import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.data.evaluate_textworld_action_agreement import (
    build_actor_prompt,
    extract_action,
    summarize_action_agreement,
    validate_aligned_results,
)


class RecordingTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "rendered actor prompt"


class ExtractActionTests(unittest.TestCase):
    def test_extracts_plain_and_labeled_actions_into_one_canonical_form(self):
        self.assertEqual(extract_action("  Go   EAST.\n"), "go east")
        self.assertEqual(extract_action("Action:\nOpen Chest\n"), "open chest")
        self.assertEqual(extract_action("```\ntake old key\n```"), "take old key")

    def test_rejects_empty_or_multiline_explanatory_output(self):
        self.assertIsNone(extract_action("   \n"))
        self.assertIsNone(extract_action("I should explore.\nAction: go east"))


class BuildActorPromptTests(unittest.TestCase):
    def test_reuses_the_training_actor_history_contract(self):
        tokenizer = RecordingTokenizer()
        history = [
            {"role": "system", "content": "initial room"},
            {"role": "user", "content": "open door"},
            {"role": "assistant", "content": "door opens"},
        ]

        prompt = build_actor_prompt(tokenizer, history, "current room")

        self.assertEqual(prompt, "rendered actor prompt")
        self.assertEqual(
            tokenizer.messages[1:],
            [
                {"role": "user", "content": "initial room"},
                {"role": "assistant", "content": "open door"},
                {"role": "user", "content": "door opens"},
                {"role": "user", "content": "current room"},
            ],
        )


class ValidateAlignedResultsTests(unittest.TestCase):
    def test_accepts_identical_test_contracts_in_different_row_orders(self):
        common_a = {
            "item_id": "a",
            "history": [{"role": "system", "content": "room a"}],
            "real_observation": "real a",
            "logged_action": "go east",
            "predicted_observation": "pred a",
            "status": "ok",
        }
        common_b = {
            "item_id": "b",
            "history": [{"role": "system", "content": "room b"}],
            "real_observation": "real b",
            "logged_action": "take key",
            "predicted_observation": "pred b",
            "status": "ok",
        }

        aligned = validate_aligned_results(
            {"sft": [common_a, common_b], "union_js": [common_b, common_a]}
        )

        self.assertEqual(aligned, ["a", "b"])

    def test_rejects_mismatched_item_sets_or_real_state_contracts(self):
        first = {
            "item_id": "a",
            "history": [],
            "real_observation": "real",
            "logged_action": "look",
            "predicted_observation": "pred",
            "status": "ok",
        }
        missing = {**first, "item_id": "b"}
        changed_real = {**first, "real_observation": "different real"}

        with self.assertRaisesRegex(ValueError, "item IDs"):
            validate_aligned_results({"sft": [first], "behr": [missing]})
        with self.assertRaisesRegex(ValueError, "real_observation"):
            validate_aligned_results({"sft": [first], "behr": [changed_real]})


class SummarizeActionAgreementTests(unittest.TestCase):
    def test_reports_raw_normalized_and_logged_action_agreement(self):
        rows = [
            {
                "status": "ok",
                "real_raw_output": "go east",
                "predicted_raw_output": "Go EAST.\n",
                "real_action": "go east",
                "predicted_action": "go east",
                "logged_action": "go east",
            },
            {
                "status": "ok",
                "real_raw_output": "take key",
                "predicted_raw_output": "open chest",
                "real_action": "take key",
                "predicted_action": "open chest",
                "logged_action": "take key",
            },
            {"status": "error", "error": "empty actor output"},
        ]

        summary = summarize_action_agreement(rows)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertAlmostEqual(summary["raw_action_agreement"], 0.0)
        self.assertAlmostEqual(summary["normalized_action_agreement"], 0.5)
        self.assertAlmostEqual(summary["real_vs_logged_agreement"], 1.0)
        self.assertAlmostEqual(summary["predicted_vs_logged_agreement"], 0.5)


class ValidateOnlyCliTests(unittest.TestCase):
    def test_validates_ready_models_and_preserves_pending_model_names(self):
        row = {
            "item_id": "sample-1",
            "history": [],
            "real_observation": "real",
            "logged_action": "look",
            "predicted_observation": "pred",
            "status": "ok",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            line = json.dumps(row) + "\n"
            first.write_text(line, encoding="utf-8")
            second.write_text(line, encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "actor_model_path": "/models/actor",
                        "models": [
                            {"name": "sft", "results_jsonl": str(first)},
                            {"name": "behr", "results_jsonl": str(second)},
                            {
                                "name": "union_js_sft",
                                "status": "pending",
                                "results_jsonl": None,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "output"

            completed = subprocess.run(
                [
                    sys.executable,
                    "src/data/evaluate_textworld_action_agreement.py",
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(output_dir),
                    "--validate-only",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(
                (output_dir / "validation_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["ready_model_count"], 2)
            self.assertEqual(report["pending_models"], ["union_js_sft"])
            self.assertEqual(report["aligned_item_count"], 1)


if __name__ == "__main__":
    unittest.main()
