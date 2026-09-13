import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "train" / "run_grpo_textworld_formal.sh"


class TextWorldFormalLauncherTests(unittest.TestCase):
    def dry_run(self, arm: str, **overrides) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            env = {
                **os.environ,
                "FORMAL_ARM": arm,
                "OUTPUT_DIR": str(output_dir),
                **overrides,
            }
            result = subprocess.run(
                ["bash", str(LAUNCHER), "--dry-run"],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertFalse(output_dir.exists())
            return result

    def test_all_arms_share_the_formal_configuration(self):
        for arm in ("behr", "union_js", "union_js_sft"):
            with self.subTest(arm=arm):
                result = self.dry_run(arm)
                self.assertEqual(result.returncode, 0, result.stderr)
                for expected in (
                    "train/full.parquet",
                    "data.seed=42",
                    "actor_rollout_ref.actor.data_loader_seed=42",
                    "actor_rollout_ref.rollout.n=4",
                    "actor_rollout_ref.rollout.temperature=0.7",
                    "trainer.total_training_steps=2000",
                    "trainer.test_freq=250",
                    "trainer.save_freq=1000",
                    "trainer.max_actor_ckpt_to_keep=1",
                    "trainer.resume_mode=disable",
                ):
                    self.assertIn(expected, result.stdout)

    def test_arm_mapping_changes_only_reward_and_sft_coefficient(self):
        expected = {
            "behr": ("reward_mode=cauchy", "sft_loss_coef=0.0"),
            "union_js": ("reward_mode=union_js", "sft_loss_coef=0.0"),
            "union_js_sft": ("reward_mode=union_js", "sft_loss_coef=0.1"),
        }
        for arm, values in expected.items():
            with self.subTest(arm=arm):
                result = self.dry_run(arm)
                self.assertEqual(result.returncode, 0, result.stderr)
                for value in values:
                    self.assertIn(value, result.stdout)

    def test_missing_or_unknown_arm_fails_without_output(self):
        for arm in ("", "unknown"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / "output"
                env = {**os.environ, "FORMAL_ARM": arm, "OUTPUT_DIR": str(output_dir)}
                result = subprocess.run(
                    ["bash", str(LAUNCHER)],
                    cwd=PROJECT_ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("FORMAL_ARM", result.stderr)
                self.assertFalse(output_dir.exists())

    def test_resume_maps_to_manifest_validation_and_verl_auto_resume(self):
        result = self.dry_run("union_js_sft", FORMAL_RESUME="1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("trainer.resume_mode=auto", result.stdout)
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("textworld_formal_run.py", launcher)
        self.assertIn('PREFLIGHT_MODE=(--resume)', launcher)

    def test_formal_run_reports_explicit_dataset_cache_on_data_disk(self):
        cache_path = "/DATA/disk1/test-cache-for-formal-launcher"

        result = self.dry_run(
            "union_js_sft",
            HF_DATASETS_CACHE=cache_path,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"HF datasets cache: {cache_path}", result.stdout)


if __name__ == "__main__":
    unittest.main()
