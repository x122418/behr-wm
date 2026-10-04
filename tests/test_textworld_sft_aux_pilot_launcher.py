import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "train" / "run_grpo_textworld_sft_aux_pilot.sh"


class TextWorldSFTAuxPilotLauncherTests(unittest.TestCase):
    def test_aux_pilot_forwards_matched_configuration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            env = {**os.environ, "OUTPUT_DIR": str(output_dir)}
            result = subprocess.run(
                ["bash", str(LAUNCHER), "--dry-run"],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "actor_rollout_ref.actor.sft_loss_coef=0.1",
            "actor_rollout_ref.rollout.n=4",
            "actor_rollout_ref.rollout.temperature=0.7",
            "reward_kwargs.reward_mode=union_js",
            "trainer.total_training_steps=50",
            "trainer.save_freq=50",
            "trainer.test_freq=10",
        ):
            self.assertIn(expected, result.stdout)
        self.assertFalse(output_dir.exists())

    def test_aux_pilot_has_distinct_default_output_and_experiment_names(self):
        launcher = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn(
            "textworld_unionjs_k64_g4_t07_sft01_pilot1k_50steps_seed42",
            launcher,
        )
        self.assertIn(
            "unionjs-k64-g4-t07-sft01-textworld-pilot1k-50steps-seed42",
            launcher,
        )


if __name__ == "__main__":
    unittest.main()
