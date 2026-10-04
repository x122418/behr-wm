import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.export_textworld_lora_model import sha256_file, validate_export_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "export_textworld_lora_model.py"


class ExportTextWorldLoraModelTests(unittest.TestCase):
    def make_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        base = root / "base"
        adapter = root / "adapter"
        output = root / "merged"
        base.mkdir()
        adapter.mkdir()
        (base / "config.json").write_text("{}", encoding="utf-8")
        (base / "model.safetensors").write_bytes(b"base")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        return base, adapter, output

    def test_sha256_file_streams_exact_digest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.bin"
            payload = b"textworld-lora"
            path.write_bytes(payload)

            self.assertEqual(sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_validation_accepts_complete_inputs_and_absent_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base, adapter, output = self.make_inputs(Path(tmpdir))

            validate_export_inputs(base, adapter, output)

    def test_validation_rejects_incomplete_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base, adapter, output = self.make_inputs(Path(tmpdir))
            (adapter / "adapter_model.safetensors").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "adapter_model.safetensors"):
                validate_export_inputs(base, adapter, output)

    def test_validation_rejects_existing_output_to_protect_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base, adapter, output = self.make_inputs(Path(tmpdir))
            output.mkdir()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                validate_export_inputs(base, adapter, output)

    def test_dry_run_does_not_import_models_or_create_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base, adapter, output = self.make_inputs(Path(tmpdir))

            result = subprocess.run(
                [
                    str(PROJECT_ROOT / ".venv" / "bin" / "python"),
                    str(SCRIPT),
                    "--base-model",
                    str(base),
                    "--adapter",
                    str(adapter),
                    "--output-dir",
                    str(output),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Dry run", result.stdout)
            self.assertIn(str(base.resolve()), result.stdout)
            self.assertIn(str(adapter.resolve()), result.stdout)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
