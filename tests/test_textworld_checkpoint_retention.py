from pathlib import Path
import tempfile
import unittest

from src.training.textworld_checkpoint_retention import prune_checkpoints


class TextWorldCheckpointRetentionTests(unittest.TestCase):
    def make_run(self, root: Path, *, tracker: str = "2") -> Path:
        run = root / "formal-run"
        run.mkdir()
        (run / "formal_run_manifest.json").write_text("{}\n", encoding="utf-8")
        (run / "latest_checkpointed_iteration.txt").write_text(
            tracker, encoding="utf-8"
        )
        return run

    def make_checkpoint(self, run: Path, step: int) -> Path:
        checkpoint = run / f"global_step_{step}"
        (checkpoint / "actor").mkdir(parents=True)
        (checkpoint / "actor" / "state.pt").write_text(
            f"step {step}\n", encoding="utf-8"
        )
        return checkpoint

    def test_removes_only_checkpoints_older_than_the_retained_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = self.make_run(Path(tmpdir))
            old = self.make_checkpoint(run, 1)
            latest = self.make_checkpoint(run, 2)
            logs = run / "logs"
            logs.mkdir()
            (logs / "train.log").write_text("keep me\n", encoding="utf-8")

            removed = prune_checkpoints(run, keep=1)

            self.assertEqual(removed, [old])
            self.assertFalse(old.exists())
            self.assertTrue((latest / "actor" / "state.pt").is_file())
            self.assertTrue((logs / "train.log").is_file())

    def test_invalid_tracker_refuses_to_delete_any_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = self.make_run(Path(tmpdir), tracker="not-a-step")
            first = self.make_checkpoint(run, 1)
            second = self.make_checkpoint(run, 2)

            with self.assertRaisesRegex(ValueError, "checkpoint tracker"):
                prune_checkpoints(run, keep=1)

            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_future_checkpoint_refuses_to_delete_any_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = self.make_run(Path(tmpdir), tracker="2")
            first = self.make_checkpoint(run, 1)
            latest = self.make_checkpoint(run, 2)
            future = self.make_checkpoint(run, 3)

            with self.assertRaisesRegex(ValueError, "newer than tracker"):
                prune_checkpoints(run, keep=1)

            self.assertTrue(first.exists())
            self.assertTrue(latest.exists())
            self.assertTrue(future.exists())

    def test_requires_a_formal_manifest_and_positive_retention(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = self.make_run(Path(tmpdir))
            checkpoint = self.make_checkpoint(run, 2)
            (run / "formal_run_manifest.json").unlink()

            with self.assertRaisesRegex(ValueError, "formal run manifest"):
                prune_checkpoints(run, keep=1)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                prune_checkpoints(run, keep=0)

            self.assertTrue(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
