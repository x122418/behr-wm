import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "eval" / "02_task_success_rate" / "interact_with_wm.py"
SPEC = importlib.util.spec_from_file_location("interact_with_wm", MODULE_PATH)
interact_with_wm = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(interact_with_wm)


class InteractWithWmSelectionTests(unittest.TestCase):
    def test_limit_items_returns_the_exact_bounded_cohort(self):
        items = [{"id": 1}, {"id": 2}, {"id": 3}]

        self.assertEqual(
            interact_with_wm.limit_items(items, n_samples=2),
            [{"id": 1}, {"id": 2}],
        )
        self.assertEqual(
            interact_with_wm.limit_items(items, n_samples=-1),
            items,
        )


if __name__ == "__main__":
    unittest.main()
