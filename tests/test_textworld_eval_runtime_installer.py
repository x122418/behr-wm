import os
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = PROJECT_ROOT / "scripts" / "env_setup" / "install_textworld_eval_runtime.sh"


class TextWorldEvalRuntimeInstallerTests(unittest.TestCase):
    def test_dry_run_describes_pinned_isolated_install_without_mutation(self):
        env = os.environ.copy()
        env["DRY_RUN"] = "1"

        result = subprocess.run(
            ["bash", str(INSTALLER)],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("e8dc240269fb222de3c242b09c70bfbf0b4ac1c9", result.stdout)
        self.assertIn("venv/textworld-eval", result.stdout)
        self.assertIn("textworld==1.6.2", result.stdout)
        self.assertIn("main .venv: agentenv client only", result.stdout)


if __name__ == "__main__":
    unittest.main()
