from pathlib import Path
import tempfile
import unittest


def run_retention_cycle(*args, **kwargs):
    try:
        from scripts.watch_textworld_checkpoint_storage import (
            run_retention_cycle as implementation,
        )
    except ModuleNotFoundError as error:
        raise AssertionError("checkpoint storage watcher is missing") from error
    return implementation(*args, **kwargs)


class WatchTextWorldCheckpointStorageTests(unittest.TestCase):
    def make_arm(
        self,
        formal_root: Path,
        arm: str,
        *,
        steps: tuple[int, ...] = (1, 2),
        missing_adapter_step: int | None = None,
    ) -> Path:
        run = formal_root / "checkpoints" / arm
        run.mkdir(parents=True)
        (run / "formal_run_manifest.json").write_text("{}\n", encoding="utf-8")
        (run / "latest_checkpointed_iteration.txt").write_text(
            f"{steps[-1]}\n", encoding="utf-8"
        )
        for step in steps:
            actor = run / f"global_step_{step}" / "actor"
            actor.mkdir(parents=True)
            (actor / "model_world_size_1_rank_0.pt").write_text(
                f"model-{arm}-{step}\n", encoding="utf-8"
            )
            if step == missing_adapter_step:
                continue
            adapter = actor / "lora_adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                f"config-{arm}-{step}\n", encoding="utf-8"
            )
            (adapter / "adapter_model.safetensors").write_text(
                f"weights-{arm}-{step}\n", encoding="utf-8"
            )
        return run

    def test_cycle_archives_each_arm_and_keeps_only_requested_full_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            formal_root = Path(tmpdir) / "formal"
            runs = {
                arm: self.make_arm(formal_root, arm)
                for arm in ("union_js", "union_js_sft", "behr")
            }

            cycle = run_retention_cycle(
                formal_root,
                arms=("union_js", "union_js_sft", "behr"),
                keep=1,
            )

            self.assertEqual(cycle.errors, {})
            self.assertEqual(set(cycle.results), set(runs))
            for arm, run in runs.items():
                self.assertFalse((run / "global_step_1").exists())
                self.assertTrue((run / "global_step_2").is_dir())
                archive = formal_root / "lora_adapter_archive" / arm
                self.assertTrue((archive / "global_step_1" / "SHA256SUMS").is_file())
                self.assertTrue((archive / "global_step_2" / "SHA256SUMS").is_file())

    def test_cycle_isolates_an_invalid_arm_without_blocking_healthy_arms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            formal_root = Path(tmpdir) / "formal"
            bad = self.make_arm(
                formal_root, "union_js_sft", missing_adapter_step=2
            )
            good = self.make_arm(formal_root, "union_js")

            cycle = run_retention_cycle(
                formal_root,
                arms=("union_js_sft", "union_js"),
                keep=1,
            )

            self.assertEqual(set(cycle.errors), {"union_js_sft"})
            self.assertIn("LoRA adapter", cycle.errors["union_js_sft"])
            self.assertEqual(set(cycle.results), {"union_js"})
            self.assertTrue((bad / "global_step_1").is_dir())
            self.assertTrue((bad / "global_step_2").is_dir())
            self.assertFalse((good / "global_step_1").exists())
            self.assertTrue((good / "global_step_2").is_dir())


if __name__ == "__main__":
    unittest.main()
