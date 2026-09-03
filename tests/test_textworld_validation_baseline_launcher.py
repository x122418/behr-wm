import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "evaluate_textworld_validation_baseline.sh"


class TextWorldValidationBaselineLauncherTests(unittest.TestCase):
    def test_dry_run_targets_validation_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "validation-output"
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
                "textworld_grpo_task_split_v1/val/pilot.parquet",
                result.stdout,
            )
            self.assertNotIn("/test/", result.stdout)
            self.assertIn("--limit", result.stdout)
            self.assertIn("1000", result.stdout)
            self.assertIn("--top-ks", result.stdout)
            self.assertIn("32,64", result.stdout)
            self.assertIn("CUDA_VISIBLE_DEVICES=5", result.stdout)
            self.assertIn("NO_PROXY=127.0.0.1,localhost", result.stdout)
            self.assertFalse(output_dir.exists())

    def test_rejects_unknown_argument(self):
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--unknown"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown argument", result.stderr)


if __name__ == "__main__":
    unittest.main()
