from pathlib import Path
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUARD = PROJECT_ROOT / "scripts" / "env_setup" / "check_vllm_cumem_runtime.py"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


class VllmCumemGuardTests(unittest.TestCase):
    def test_project_runtime_passes_cumem_guard(self):
        if not VENV_PYTHON.exists():
            self.skipTest("project venv is not installed")

        result = subprocess.run(
            [str(VENV_PYTHON), str(GUARD)],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("safe vLLM CuMem runtime", result.stdout)


if __name__ == "__main__":
    unittest.main()
