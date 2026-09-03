import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = PROJECT_ROOT / "scripts" / "env_setup" / "install_textworld_grpo_locked.sh"


class TextWorldGrpoEnvironmentInstallerTests(unittest.TestCase):
    def test_dry_run_reports_locked_stack_without_creating_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "isolated-env"
            env = os.environ.copy()
            env["TEXTWORLD_GRPO_VENV"] = str(venv_path)

            result = subprocess.run(
                ["bash", str(INSTALLER), "--dry-run"],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Python: 3.10", result.stdout)
            self.assertIn("torch==2.8.0", result.stdout)
            self.assertIn("vllm==0.11.0", result.stdout)
            self.assertIn("verl==0.7.1", result.stdout)
            self.assertIn(str(venv_path), result.stdout)
            self.assertFalse(venv_path.exists())

    def test_unknown_argument_fails_without_creating_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "isolated-env"
            env = os.environ.copy()
            env["TEXTWORLD_GRPO_VENV"] = str(venv_path)

            result = subprocess.run(
                ["bash", str(INSTALLER), "--not-a-real-option"],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown argument", result.stderr.lower())
            self.assertFalse(venv_path.exists())


if __name__ == "__main__":
    unittest.main()
