import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "eval" / "02_task_success_rate" / "cal_wm2real.py"
SPEC = importlib.util.spec_from_file_location("cal_wm2real", MODULE_PATH)
cal_wm2real = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cal_wm2real)


class CalWm2RealSelectionTests(unittest.TestCase):
    def test_replay_selection_is_numeric_deterministic_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in (
                "textworld_10.json",
                "textworld_2.json",
                "textworld_1.json",
                "webshop_1.json",
                "_metrics.json",
            ):
                (root / name).write_text("{}", encoding="utf-8")

            selected = cal_wm2real.discover_replay_files(
                root,
                task="textworld",
                n_samples=2,
            )

        self.assertEqual(
            [Path(path).name for path in selected],
            ["textworld_1.json", "textworld_2.json"],
        )


if __name__ == "__main__":
    unittest.main()
