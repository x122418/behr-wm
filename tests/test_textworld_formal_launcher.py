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
                    "data.filter_overlong_prompts=False",
                    "trainer.total_training_steps=2000",
                    "trainer.test_freq=250",
                    "trainer.save_freq=1000",
                    "trainer.max_actor_ckpt_to_keep=1",
                    "trainer.resume_mode=disable",
                    "actor_rollout_ref.actor.optim.lr=5e-6",
                    "actor_rollout_ref.model.lora_rank=32",
                    "actor_rollout_ref.model.lora_alpha=32",
                    "actor_rollout_ref.model.target_modules=all-linear",
                    "actor_rollout_ref.rollout.load_format=safetensors",
                    "actor_rollout_ref.rollout.layered_summon=True",
                    "reward.num_workers=8",
                ):
                    self.assertIn(expected, result.stdout)
                self.assertNotIn("reward_kwargs.max_workers", result.stdout)

    def test_formal_lora_configuration_can_be_overridden_as_one_shared_bundle(self):
        for arm in ("behr", "union_js", "union_js_sft"):
            with self.subTest(arm=arm):
                result = self.dry_run(
                    arm,
                    LORA_RANK="64",
                    LORA_ALPHA="128",
                    LORA_TARGET_MODULES="q_proj,v_proj",
                    ACTOR_LR="1e-5",
                    REWARD_NUM_WORKERS="6",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for expected in (
                    "actor_rollout_ref.actor.optim.lr=1e-5",
                    "actor_rollout_ref.model.lora_rank=64",
                    "actor_rollout_ref.model.lora_alpha=128",
                    "actor_rollout_ref.model.target_modules=q_proj,v_proj",
                    "reward.num_workers=6",
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

    def test_formal_run_reports_explicit_runtime_storage_on_data_disk(self):
        cache_path = "/DATA/disk1/test-cache-for-formal-launcher"
        ray_temp_path = "/DATA/disk1/test-ray-temp-for-formal-launcher"

        result = self.dry_run(
            "union_js_sft",
            HF_DATASETS_CACHE=cache_path,
            RAY_TEMP_DIR=ray_temp_path,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"HF datasets cache: {cache_path}", result.stdout)
        self.assertIn(f"Ray temp directory: {ray_temp_path}", result.stdout)
        self.assertIn(
            f"++ray_kwargs.ray_init._temp_dir={ray_temp_path}",
            result.stdout,
        )

    def test_real_run_stops_when_cumem_runtime_guard_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_dir = tmpdir_path / "output"
            guard = tmpdir_path / "failing-cumem-guard"
            guard.write_text("#!/usr/bin/env bash\nexit 42\n", encoding="utf-8")
            guard.chmod(0o755)
            env = {
                **os.environ,
                "FORMAL_ARM": "behr",
                "OUTPUT_DIR": str(output_dir),
                "CUMEM_RUNTIME_GUARD": str(guard),
            }

            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 42)
            self.assertIn("vLLM CuMem runtime guard failed", result.stderr)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
