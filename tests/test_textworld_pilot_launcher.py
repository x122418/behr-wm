import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "train" / "run_grpo_textworld_pilot.sh"


class TextWorldPilotLauncherTests(unittest.TestCase):
    def test_dry_run_uses_task_disjoint_pilot_data_and_saves_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            env = os.environ.copy()
            env["OUTPUT_DIR"] = str(output_dir)
            result = subprocess.run(
                ["bash", str(LAUNCHER), "--dry-run"],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "textworld_grpo_task_split_v1/train/pilot.parquet",
                result.stdout,
            )
            self.assertIn(
                "textworld_grpo_task_split_v1/val/pilot.parquet",
                result.stdout,
            )
            self.assertNotIn("/test/", result.stdout)
            self.assertIn("trainer.total_training_steps=50", result.stdout)
            self.assertIn("trainer.save_freq=10", result.stdout)
            self.assertIn("trainer.test_freq=10", result.stdout)
            self.assertIn("actor_rollout_ref.rollout.n=2", result.stdout)
            self.assertIn("data.max_prompt_length=4096", result.stdout)
            self.assertIn("data.max_response_length=512", result.stdout)
            self.assertFalse(output_dir.exists())

    def test_environment_can_override_pilot_scale(self):
        env = os.environ.copy()
        env.update(
            {
                "TOTAL_STEPS": "7",
                "SAVE_FREQ": "3",
                "VAL_FREQ": "4",
                "GROUP_SIZE": "5",
            }
        )
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run"],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("trainer.total_training_steps=7", result.stdout)
        self.assertIn("trainer.save_freq=3", result.stdout)
        self.assertIn("trainer.test_freq=4", result.stdout)
        self.assertIn("actor_rollout_ref.rollout.n=5", result.stdout)


if __name__ == "__main__":
    unittest.main()
