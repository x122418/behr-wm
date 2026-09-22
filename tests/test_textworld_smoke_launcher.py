import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "train" / "run_grpo_textworld_smoke.sh"


class TextWorldSmokeLauncherTests(unittest.TestCase):
    def test_explicit_lora_configuration_is_forwarded_to_verl(self):
        env = os.environ.copy()
        env.update(
            {
                "LORA_RANK": "32",
                "LORA_ALPHA": "32",
                "LORA_TARGET_MODULES": "all-linear",
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
        for expected in (
            "actor_rollout_ref.model.lora_rank=32",
            "actor_rollout_ref.model.lora_alpha=32",
            "actor_rollout_ref.model.target_modules=all-linear",
            "actor_rollout_ref.rollout.load_format=safetensors",
            "actor_rollout_ref.rollout.layered_summon=True",
        ):
            self.assertIn(expected, result.stdout)

    def test_omitted_lora_configuration_preserves_full_parameter_mode(self):
        env = os.environ.copy()
        for key in ("LORA_RANK", "LORA_ALPHA", "LORA_TARGET_MODULES"):
            env.pop(key, None)

        result = subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run"],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("actor_rollout_ref.model.lora_", result.stdout)
        self.assertNotIn("actor_rollout_ref.model.target_modules=", result.stdout)
        self.assertNotIn("actor_rollout_ref.rollout.load_format=", result.stdout)
        self.assertNotIn("actor_rollout_ref.rollout.layered_summon=", result.stdout)

    def test_reward_worker_count_is_explicit_and_dead_reward_kwarg_is_absent(self):
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run"],
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reward.num_workers=8", result.stdout)
        self.assertNotIn("reward_kwargs.max_workers", result.stdout)

    def test_can_disable_ray_dashboard_for_parallel_local_runs(self):
        env = {**os.environ, "RAY_INCLUDE_DASHBOARD": "False"}

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
            "++ray_kwargs.ray_init.include_dashboard=False",
            result.stdout,
        )

    def test_invalid_lora_rank_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            env = {
                **os.environ,
                "LORA_RANK": "invalid",
                "OUTPUT_DIR": str(output_dir),
            }
            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("LORA_RANK", result.stderr)
            self.assertFalse(output_dir.exists())

    def test_dry_run_forwards_explicit_reproducibility_and_retention_controls(self):
        env = os.environ.copy()
        env.update(
            {
                "DATA_SEED": "42",
                "ACTOR_DATA_LOADER_SEED": "42",
                "MAX_ACTOR_CKPT_TO_KEEP": "1",
                "RESUME_MODE": "disable",
                "RAY_TEMP_DIR": "/DATA/disk1/test-ray-temp",
                "FILTER_OVERLONG_PROMPTS": "False",
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
        for expected in (
            "data.seed=42",
            "actor_rollout_ref.actor.data_loader_seed=42",
            "trainer.max_actor_ckpt_to_keep=1",
            "trainer.resume_mode=disable",
            "++ray_kwargs.ray_init._temp_dir=/DATA/disk1/test-ray-temp",
            "data.filter_overlong_prompts=False",
        ):
            self.assertIn(expected, result.stdout)

    def test_omitted_controls_preserve_verl_defaults(self):
        env = os.environ.copy()
        for key in (
            "DATA_SEED",
            "ACTOR_DATA_LOADER_SEED",
            "MAX_ACTOR_CKPT_TO_KEEP",
            "RESUME_MODE",
            "RAY_TEMP_DIR",
        ):
            env.pop(key, None)

        result = subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run"],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("data.seed=", result.stdout)
        self.assertNotIn("actor_rollout_ref.actor.data_loader_seed=", result.stdout)
        self.assertNotIn("trainer.max_actor_ckpt_to_keep=", result.stdout)
        self.assertNotIn("trainer.resume_mode=", result.stdout)
        self.assertNotIn("ray_kwargs.ray_init._temp_dir=", result.stdout)

    def test_invalid_resume_mode_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            env = {**os.environ, "RESUME_MODE": "sometimes", "OUTPUT_DIR": str(output_dir)}
            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported RESUME_MODE", result.stderr)
        self.assertFalse(output_dir.exists())

    def test_dry_run_forwards_auxiliary_sft_coefficient(self):
        env = os.environ.copy()
        env["SFT_LOSS_COEF"] = "0.1"

        result = subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run"],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("actor_rollout_ref.actor.sft_loss_coef=0.1", result.stdout)

    def test_local_reward_services_bypass_environment_proxies(self):
        launcher = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("export NO_PROXY=127.0.0.1,localhost", launcher)
        self.assertIn("export no_proxy=127.0.0.1,localhost", launcher)

    def test_local_reference_actor_path_is_forwarded_to_reward_workers(self):
        env = os.environ.copy()
        env["JUDGE_MODEL_PATH"] = "/models/local-qwen3-8b"

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
            "reward_kwargs.judge_model_path=/models/local-qwen3-8b",
            result.stdout,
        )

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
            self.assertIn(
                'trainer.logger=["console","tensorboard"]', result.stdout
            )
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

            self.assertIn(
                f"TENSORBOARD_DIR={output_dir / 'tensorboard'}", result.stdout
            )

if __name__ == "__main__":
    unittest.main()
