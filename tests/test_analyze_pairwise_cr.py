import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "eval" / "02_task_success_rate" / "analyze_pairwise_cr.py"
SPEC = importlib.util.spec_from_file_location("analyze_pairwise_cr", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


class AnalyzePairwiseCrTests(unittest.TestCase):
    def test_rejects_real_and_w2r_id_mismatch_by_default(self):
        with self.assertRaisesRegex(ValueError, "task ID sets differ"):
            analysis.compute_pairwise_metrics(
                real={"1": True, "2": False},
                w2r={"1": True},
            )

    def test_computes_cr_and_pairwise_cr_on_identical_ids(self):
        metrics = analysis.compute_pairwise_metrics(
            real={"1": True, "2": True, "3": False},
            w2r={"1": True, "2": False, "3": True},
        )

        self.assertEqual(metrics["n_common"], 3)
        self.assertAlmostEqual(metrics["cr_aggregate"], 1.0)
        self.assertAlmostEqual(metrics["cr_pairwise"], 0.5)
        self.assertEqual(
            metrics["contingency"],
            {
                "both_success": 1,
                "real_only": 1,
                "w2r_only": 1,
                "both_fail": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
