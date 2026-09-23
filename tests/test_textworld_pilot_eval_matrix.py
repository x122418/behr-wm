import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.run_textworld_pilot_eval_matrix import (
    build_model_matrix,
    ensure_evaluation_contract,
    read_unique_jsonl,
    validate_preflight_paths,
    verify_evaluation_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_textworld_pilot_eval_matrix.py"
LAUNCHER = PROJECT_ROOT / "scripts" / "run_textworld_pilot_eval_matrix.sh"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class TextWorldPilotEvalMatrixTests(unittest.TestCase):
    def test_evaluation_contract_prevents_stale_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "candidate"
            contract = {"schema_version": 1, "model": "candidate", "limit": 1000}

            ensure_evaluation_contract(output_dir, contract, create=True)
            ensure_evaluation_contract(output_dir, contract, create=False)

            with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
                ensure_evaluation_contract(
                    output_dir, {**contract, "limit": 2}, create=False
                )

    def test_preflight_rejects_missing_base_weights_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base"
            actor = root / "actor"
            pilot = root / "pilot"
            validation = root / "val.parquet"
            base.mkdir()
            actor.mkdir()
            validation.write_bytes(b"parquet")
            (base / "config.json").write_text("{}", encoding="utf-8")
            (actor / "config.json").write_text("{}", encoding="utf-8")
            (actor / "model.safetensors").write_bytes(b"weights")
            specs = build_model_matrix(base, pilot, root / "merged")

            with self.assertRaisesRegex(FileNotFoundError, "model weights"):
                validate_preflight_paths(specs, base, actor, validation)

    def test_model_matrix_has_base_and_six_ordered_checkpoints(self):
        specs = build_model_matrix(Path("/base"), Path("/pilot"), Path("/merged"))

        self.assertEqual(
            [spec.name for spec in specs],
            [
                "base",
                "behr_step50",
                "behr_step100",
                "union_js_step50",
                "union_js_step100",
                "union_js_sft_step50",
                "union_js_sft_step100",
            ],
        )
        self.assertIsNone(specs[0].adapter_path)
        self.assertEqual(
            specs[-1].adapter_path,
            Path("/pilot/checkpoints/union_js_sft/global_step_100/actor/lora_adapter"),
        )
        self.assertEqual(specs[-1].model_path, Path("/merged/union_js_sft_step100"))

    def test_read_unique_jsonl_rejects_duplicate_item_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            write_jsonl(path, [{"item_id": "x"}, {"item_id": "x"}])

            with self.assertRaisesRegex(ValueError, "duplicate item_id"):
                read_unique_jsonl(path)

    def test_verify_requires_complete_matching_finite_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            names = ["base", "candidate"]
            rows = [
                {"item_id": "a", "status": "ok", "full_vocab_js": 0.1},
                {"item_id": "b", "status": "ok", "full_vocab_js": 0.2},
            ]
            for name in names:
                model_dir = root / name
                write_jsonl(model_dir / "results.jsonl", rows)
                (model_dir / "summary.json").write_text(
                    json.dumps({"total": 2, "successful": 2, "errors": 0}),
                    encoding="utf-8",
                )

            report = verify_evaluation_matrix(root, names, expected_count=2)

            self.assertEqual(report["item_count"], 2)
            self.assertEqual(report["models"], names)

    def test_verify_rejects_different_item_sets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, item_id in (("base", "a"), ("candidate", "b")):
                write_jsonl(
                    root / name / "results.jsonl",
                    [{"item_id": item_id, "status": "ok", "full_vocab_js": 0.1}],
                )
                (root / name / "summary.json").write_text(
                    json.dumps({"total": 1, "successful": 1, "errors": 0}),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "item IDs differ"):
                verify_evaluation_matrix(root, ["base", "candidate"], 1)

    def test_dry_run_prints_every_model_without_creating_runtime_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            merged = root / "merged"
            evaluation = root / "evaluation"
            result = subprocess.run(
                [
                    str(PROJECT_ROOT / ".venv" / "bin" / "python"),
                    str(SCRIPT),
                    "--stage",
                    "all",
                    "--base-model",
                    str(root / "base"),
                    "--actor-model",
                    str(root / "actor"),
                    "--pilot-root",
                    str(root / "pilot"),
                    "--validation-data",
                    str(root / "val.parquet"),
                    "--merged-root",
                    str(merged),
                    "--evaluation-root",
                    str(evaluation),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for name in (
                "base",
                "behr_step50",
                "behr_step100",
                "union_js_step50",
                "union_js_step100",
                "union_js_sft_step50",
                "union_js_sft_step100",
            ):
                self.assertIn(name, result.stdout)
            self.assertIn("Stage: merge", result.stdout)
            self.assertIn("Stage: generate", result.stdout)
            self.assertIn("Stage: score", result.stdout)
            self.assertIn("Stage: verify", result.stdout)
            self.assertFalse(merged.exists())
            self.assertFalse(evaluation.exists())

    def test_shell_launcher_derives_shared_hdd_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["bash", str(LAUNCHER), "all", "--dry-run"],
                cwd=PROJECT_ROOT,
                env={
                    "PATH": "/usr/bin:/bin",
                    "LWM_HDD": tmpdir,
                    "TEXTWORLD_GRPO_VENV": str(PROJECT_ROOT / ".venv"),
                },
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"{tmpdir}/workspace/LWM_project/models", result.stdout)
            self.assertIn(
                f"{tmpdir}/lwm_runs/textworld_lora_pilot100_seed42_retry1",
                result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
