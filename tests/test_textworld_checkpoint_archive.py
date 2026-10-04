from pathlib import Path
import tempfile
import unittest

def archive_and_prune_checkpoints(*args, **kwargs):
    try:
        from src.training.textworld_checkpoint_archive import (
            archive_and_prune_checkpoints as implementation,
        )
    except ModuleNotFoundError as error:
        raise AssertionError("checkpoint archive feature is missing") from error
    return implementation(*args, **kwargs)


class TextWorldCheckpointArchiveTests(unittest.TestCase):
    def make_run(self, root: Path, *, tracker: int = 3) -> Path:
        run = root / "formal-run"
        run.mkdir()
        (run / "formal_run_manifest.json").write_text("{}\n", encoding="utf-8")
        (run / "latest_checkpointed_iteration.txt").write_text(
            f"{tracker}\n", encoding="utf-8"
        )
        return run

    def make_checkpoint(
        self, run: Path, step: int, *, include_adapter: bool = True
    ) -> Path:
        checkpoint = run / f"global_step_{step}"
        actor = checkpoint / "actor"
        actor.mkdir(parents=True)
        (actor / "model_world_size_1_rank_0.pt").write_text(
            f"model-step-{step}\n", encoding="utf-8"
        )
        if include_adapter:
            adapter = actor / "lora_adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                f"config-step-{step}\n", encoding="utf-8"
            )
            (adapter / "adapter_model.safetensors").write_text(
                f"weights-step-{step}\n", encoding="utf-8"
            )
        return checkpoint

    def test_archives_every_adapter_before_pruning_old_full_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = self.make_run(root)
            checkpoints = [self.make_checkpoint(run, step) for step in (1, 2, 3)]
            archive = root / "adapter-archive"

            result = archive_and_prune_checkpoints(run, archive, keep=2)

            self.assertEqual(result.archived_steps, [1, 2, 3])
            self.assertEqual(result.removed_checkpoints, [checkpoints[0]])
            self.assertFalse(checkpoints[0].exists())
            self.assertTrue(checkpoints[1].is_dir())
            self.assertTrue(checkpoints[2].is_dir())
            self.assertEqual(
                (archive / "global_step_1" / "adapter_config.json").read_text(
                    encoding="utf-8"
                ),
                "config-step-1\n",
            )
            self.assertEqual(
                (archive / "global_step_1" / "adapter_model.safetensors").read_text(
                    encoding="utf-8"
                ),
                "weights-step-1\n",
            )
            self.assertEqual(
                (archive / "global_step_1" / "SHA256SUMS").read_text(
                    encoding="utf-8"
                ),
                "d8a0c02f2ff048f6d0d9340b4b900d77ceba9340a56c538c147a2d49b433cc48  adapter_config.json\n"
                "47e2debd6fba41b7832952322d9461e26337f2408be2b8ac0f7a2e47e6d59a7c  adapter_model.safetensors\n",
            )

    def test_missing_adapter_refuses_to_prune_any_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = self.make_run(root)
            checkpoints = [self.make_checkpoint(run, step) for step in (1, 2)]
            checkpoints.append(self.make_checkpoint(run, 3, include_adapter=False))

            with self.assertRaisesRegex(ValueError, "LoRA adapter"):
                archive_and_prune_checkpoints(run, root / "archive", keep=2)

            self.assertTrue(all(checkpoint.is_dir() for checkpoint in checkpoints))

    def test_corrupt_existing_archive_refuses_to_prune_any_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = self.make_run(root)
            checkpoints = [self.make_checkpoint(run, step) for step in (1, 2, 3)]
            archive = root / "archive"
            corrupt = archive / "global_step_1"
            corrupt.mkdir(parents=True)
            (corrupt / "adapter_config.json").write_text(
                "config-step-1\n", encoding="utf-8"
            )
            (corrupt / "adapter_model.safetensors").write_text(
                "corrupt\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                archive_and_prune_checkpoints(run, archive, keep=2)

            self.assertTrue(all(checkpoint.is_dir() for checkpoint in checkpoints))

    def test_repeated_run_verifies_existing_archives_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = self.make_run(root, tracker=2)
            self.make_checkpoint(run, 1)
            self.make_checkpoint(run, 2)
            archive = root / "archive"

            first = archive_and_prune_checkpoints(run, archive, keep=2)
            second = archive_and_prune_checkpoints(run, archive, keep=2)

            self.assertEqual(first.archived_steps, [1, 2])
            self.assertEqual(second.archived_steps, [])
            self.assertEqual(second.verified_steps, [1, 2])
            self.assertEqual(second.removed_checkpoints, [])


if __name__ == "__main__":
    unittest.main()
