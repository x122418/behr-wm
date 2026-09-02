import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "train" / "run_grpo_textworld_smoke.sh"


class TextWorldSmokeLauncherTests(unittest.TestCase):
    def test_dry_run_prints_textworld_two_step_command_without_side_effects(self):
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
            self.assertIn("behr_reward_textworld.py", result.stdout)
            self.assertIn("trainer.total_training_steps=2", result.stdout)
            self.assertIn(
                "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
                result.stdout,
            )
            self.assertIn("data.max_prompt_length=4096", result.stdout)
            self.assertIn("data.max_response_length=512", result.stdout)
            self.assertIn(
                "textworld_grpo_task_split_v1/train/smoke.parquet",
                result.stdout,
            )
            self.assertIn(
                "textworld_grpo_task_split_v1/val/pilot.parquet",
                result.stdout,
            )
            self.assertNotIn(
                "textworld_grpo/test/test.parquet",
                result.stdout,
            )
            self.assertIn(
                "actor_rollout_ref.rollout.max_model_len=4608", result.stdout
            )
            self.assertIn("behavior_weight=1.0", result.stdout)
            self.assertIn("reward_kwargs.reward_mode=cauchy", result.stdout)
            self.assertIn("facts_weight=0.0", result.stdout)
            self.assertIn("trainer.logger=[\"console\"]", result.stdout)
            self.assertIn(
                "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
                result.stdout,
            )
            self.assertIn(
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
                result.stdout,
            )
            self.assertIn(
                "++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
                result.stdout,
            )
            self.assertIn(
                "++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
                result.stdout,
            )
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
