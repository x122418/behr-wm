import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = PROJECT_ROOT / "scripts" / "install_verl_sft_aux_patch.py"
SITE_PACKAGES = PROJECT_ROOT / ".venv" / "lib" / "python3.10" / "site-packages"
TARGETS = (
    Path("verl/workers/config/actor.py"),
    Path("verl/workers/fsdp_workers.py"),
    Path("verl/workers/actor/dp_actor.py"),
)


class VerlSFTAuxPatchInstallerTests(unittest.TestCase):
    def make_site_packages(self, root: Path, version: str) -> Path:
        site_packages = root / "site-packages"
        for relative_path in TARGETS:
            destination = site_packages / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SITE_PACKAGES / relative_path, destination)
        metadata = site_packages / f"verl-{version}.dist-info" / "METADATA"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(
            f"Metadata-Version: 2.1\nName: verl\nVersion: {version}\n",
            encoding="utf-8",
        )
        actor_config = site_packages / "verl/workers/config/actor.py"
        if "sft_loss_coef: float = 0.0" in actor_config.read_text(encoding="utf-8"):
            reversed_patch = subprocess.run(
                [
                    "patch",
                    "--batch",
                    "--fuzz=0",
                    "-R",
                    "-p1",
                    "-d",
                    str(site_packages),
                    "-i",
                    str(PROJECT_ROOT / "patches/verl-0.7.1-textworld-sft-aux.patch"),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(reversed_patch.returncode, 0, reversed_patch.stderr)
        return site_packages

    def run_installer(self, mode: str, site_packages: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                str(PROJECT_ROOT / ".venv" / "bin" / "python"),
                str(INSTALLER),
                mode,
                "--site-packages",
                str(site_packages),
            ],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            text=True,
            capture_output=True,
            check=False,
        )

    def test_check_rejects_wrong_verl_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            site_packages = self.make_site_packages(Path(tmpdir), "0.7.0")
            result = self.run_installer("--check", site_packages)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires VERL 0.7.1", result.stderr)

    def test_apply_is_idempotent_and_check_passes_after_apply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            site_packages = self.make_site_packages(Path(tmpdir), "0.7.1")
            first = self.run_installer("--apply", site_packages)
            second = self.run_installer("--apply", site_packages)
            checked = self.run_installer("--check", site_packages)

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn("applied", first.stdout)
        self.assertIn("already applied", second.stdout)
        self.assertIn("VERL auxiliary SFT patch is installed", checked.stdout)


if __name__ == "__main__":
    unittest.main()
