import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
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


class GenerateOnlyCliTests(unittest.TestCase):
    def test_generate_stage_writes_predictions_without_loading_actor(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = {"data": [{"id": "test-wm"}]}
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                self.rfile.read(length)
                payload = {"choices": [{"message": {"content": "predicted state"}}]}
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        with tempfile.TemporaryDirectory() as temp_dir:
            import pandas as pd

            root = Path(temp_dir)
            parquet = root / "input.parquet"
            pd.DataFrame(
                [
                    {
                        "item_id": "sample-1",
                        "prompt": [{"role": "system", "content": "room"}],
                        "reward_model": {"ground_truth": "real state"},
                        "extra_info": {
                            "expert_action": "look",
                            "history": [{"role": "system", "content": "room"}],
                            "task_id": 1,
                        },
                    }
                ]
            ).to_parquet(parquet)
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            output_dir = root / "output"
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = str(Path.cwd())
                env["NO_PROXY"] = "127.0.0.1,localhost"
                env["no_proxy"] = env["NO_PROXY"]
                completed = subprocess.run(
                    [
                        sys.executable,
                        "src/data/evaluate_textworld_transition_baseline.py",
                        "--input",
                        str(parquet),
                        "--output-dir",
                        str(output_dir),
                        "--actor-model-path",
                        str(root / "actor-does-not-exist"),
                        "--wm-api-base",
                        f"http://127.0.0.1:{server.server_port}",
                        "--stage",
                        "generate",
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                    env=env,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            generations = [
                json.loads(line)
                for line in (output_dir / "generations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(generations[0]["predicted_observation"], "predicted state")
            self.assertTrue((output_dir / "generation_summary.json").exists())
            self.assertFalse((output_dir / "results.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
