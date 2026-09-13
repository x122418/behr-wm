import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.training.textworld_formal_run import prepare_run


def manifest(**overrides):
    value = {
        "schema_version": 1,
        "arm": "union_js_sft",
        "reward_mode": "union_js",
        "sft_loss_coef": 0.1,
        "train_data": "/data/train/full.parquet",
        "val_data": "/data/val/pilot.parquet",
        "world_model": "/models/world-model",
        "actor_model": "/models/actor",
        "scorer_url": "http://127.0.0.1:8002",
        "seed": 42,
        "actor_data_loader_seed": 42,
        "group_size": 4,
        "rollout_temperature": 0.7,
        "total_steps": 2000,
        "save_freq": 1000,
        "val_freq": 250,
        "max_actor_ckpt_to_keep": 1,
    }
    value.update(overrides)
    return value


class TextWorldFormalRunTests(unittest.TestCase):
    def test_cli_records_disabled_overlong_filter_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            result = subprocess.run(
                [
                    sys.executable,
                    "src/training/textworld_formal_run.py",
                    "--new-run",
                    "--output-dir",
                    str(output_dir),
                    "--arm",
                    "behr",
                    "--reward-mode",
                    "cauchy",
                    "--sft-loss-coef",
                    "0.0",
                    "--train-data",
                    "/data/train.parquet",
                    "--val-data",
                    "/data/val.parquet",
                    "--world-model",
                    "/models/world",
                    "--actor-model",
                    "/models/actor",
                    "--scorer-url",
                    "http://127.0.0.1:8000",
                    "--seed",
                    "42",
                    "--actor-data-loader-seed",
                    "42",
                    "--group-size",
                    "4",
                    "--rollout-temperature",
                    "0.7",
                    "--total-steps",
                    "2000",
                    "--save-freq",
                    "1000",
                    "--val-freq",
                    "250",
                    "--max-actor-ckpt-to-keep",
                    "1",
                    "--no-filter-overlong-prompts",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            actual = json.loads(
                (output_dir / "formal_run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(actual["schema_version"], 2)
            self.assertIs(actual["filter_overlong_prompts"], False)

    def test_new_run_writes_canonical_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            path = prepare_run(output_dir, manifest(), resume=False)

            self.assertEqual(path, output_dir / "formal_run_manifest.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), manifest())
            text = path.read_text(encoding="utf-8")
            self.assertLess(text.index('"actor_model"'), text.index('"arm"'))

    def test_new_run_rejects_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            output_dir.mkdir()
            (output_dir / "unrelated.txt").write_text("occupied", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-empty output directory"):
                prepare_run(output_dir, manifest(), resume=False)

    def test_resume_requires_checkpoint_tracker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            prepare_run(output_dir, manifest(), resume=False)

            with self.assertRaisesRegex(ValueError, "checkpoint tracker"):
                prepare_run(output_dir, manifest(), resume=True)

    def test_resume_rejects_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            prepare_run(output_dir, manifest(), resume=False)
            (output_dir / "latest_checkpointed_iteration.txt").write_text(
                "1000", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "total_steps"):
                prepare_run(output_dir, manifest(total_steps=3000), resume=True)

    def test_resume_accepts_identical_manifest_and_tracker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            expected = manifest()
            expected_path = prepare_run(output_dir, expected, resume=False)
            (output_dir / "latest_checkpointed_iteration.txt").write_text(
                "1000", encoding="utf-8"
            )

            actual_path = prepare_run(output_dir, expected, resume=True)

            self.assertEqual(actual_path, expected_path)
            self.assertEqual(json.loads(actual_path.read_text()), expected)


if __name__ == "__main__":
    unittest.main()
