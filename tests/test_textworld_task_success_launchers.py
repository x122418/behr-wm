import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval" / "02_task_success_rate"


def run_script(script: str, *args: str, **overrides: str):
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", **overrides})
    return subprocess.run(
        ["bash", str(EVAL_DIR / script), *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class TextWorldTaskSuccessLauncherTests(unittest.TestCase):
    def test_real_textworld_dry_run_is_bounded_and_uses_isolated_server(self):
        result = run_script(
            "run_real_textworld.sh",
            NUM_EXAMPLES="2",
            MAX_CONCURRENCY="1",
            MAX_ROUND="3",
            EXPERIMENT_NAME="sft_smoke",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("venv/textworld-eval/bin/textworld", result.stdout)
        self.assertIn("--task_name textworld", result.stdout)
        self.assertIn("--num_examples 2", result.stdout)
        self.assertIn("--max_concurrency 1", result.stdout)
        self.assertIn("--max_round 3", result.stdout)
        self.assertIn("real/textworld/sft_smoke", result.stdout)
        self.assertIn("NO_PROXY=127.0.0.1,localhost", result.stdout)

    def test_wm_textworld_dry_run_passes_sample_and_step_bounds(self):
        result = run_script(
            "run_wm.sh",
            "sft_smoke",
            "EMPTY",
            "http://localhost:8000/v1",
            "vllm_agent",
            "1",
            "0",
            "false",
            "textworld",
            N_SAMPLES="2",
            MAX_STEPS="3",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--task textworld", result.stdout)
        self.assertIn("--n_samples 2", result.stdout)
        self.assertIn("--max-steps 3", result.stdout)
        self.assertIn("wm/textworld/sft_smoke", result.stdout)

    def test_w2r_textworld_dry_run_limits_replay_to_selected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_script(
                "run_wm2real.sh",
                tmpdir,
                TASK="textworld",
                N_SAMPLES="2",
                MAX_WORKERS="1",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("venv/textworld-eval/bin/textworld", result.stdout)
        self.assertIn("--n_samples 2", result.stdout)
        self.assertIn("--max_workers 1", result.stdout)
        self.assertIn("NO_PROXY=127.0.0.1,localhost", result.stdout)


if __name__ == "__main__":
    unittest.main()
