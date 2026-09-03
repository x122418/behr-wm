from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts/servers/start_textworld_consistency_server.sh"


class TextWorldConsistencyServerLauncherTests(unittest.TestCase):
    def test_dry_run_reports_the_resolved_service_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_model = Path(temporary_directory) / "not-downloaded"
            result = subprocess.run(
                [
                    "bash",
                    str(LAUNCHER),
                    "--dry-run",
                    "--model",
                    str(missing_model),
                    "--gpu",
                    "5",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CUDA_VISIBLE_DEVICES=5", result.stdout)
        self.assertIn("NO_PROXY=127.0.0.1,localhost", result.stdout)
        self.assertIn("--port", result.stdout)
        self.assertIn("8002", result.stdout)
        self.assertIn("--top-k", result.stdout)
        self.assertIn("64", result.stdout)

    def test_requires_explicit_model_and_gpu(self):
        for arguments in ([], ["--model", "/model"], ["--gpu", "5"]):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), *arguments],
                    cwd=PROJECT_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
