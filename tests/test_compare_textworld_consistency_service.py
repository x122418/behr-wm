import unittest

from scripts.probes.compare_textworld_consistency_service import (
    compare_metric_rows,
)


class CompareMetricRowsTests(unittest.TestCase):
    def test_accepts_metrics_within_tolerance(self):
        differences = compare_metric_rows(
            {"full_vocab_js": 0.1, "top64_union_other_js": 0.2},
            {"full_vocab_js": 0.1000005, "top64_union_other_js": 0.2},
            metric_names=("full_vocab_js", "top64_union_other_js"),
            tolerance=1e-6,
        )

        self.assertAlmostEqual(differences["full_vocab_js"], 5e-7, places=10)

    def test_rejects_a_metric_outside_tolerance(self):
        with self.assertRaisesRegex(ValueError, "full_vocab_js"):
            compare_metric_rows(
                {"full_vocab_js": 0.1},
                {"full_vocab_js": 0.10001},
                metric_names=("full_vocab_js",),
                tolerance=1e-6,
            )

    def test_rejects_a_missing_or_non_finite_metric(self):
        for candidate in ({}, {"full_vocab_js": float("nan")}):
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(ValueError, "full_vocab_js"):
                    compare_metric_rows(
                        {"full_vocab_js": 0.1},
                        candidate,
                        metric_names=("full_vocab_js",),
                        tolerance=1e-6,
                    )


if __name__ == "__main__":
    unittest.main()
