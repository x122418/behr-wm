import json
from pathlib import Path
import tempfile
import unittest

from scripts.analysis.compare_textworld_pilot_matrix import (
    MetricSpec,
    compare_matrix,
    paired_bootstrap_interval,
)


def write_results(root: Path, model: str, values: list[float]) -> None:
    model_dir = root / model
    model_dir.mkdir(parents=True)
    rows = [
        {
            "item_id": f"item-{index}",
            "status": "ok",
            "full_vocab_js": value,
            "exact_match": index % 2 == 0,
        }
        for index, value in enumerate(values)
    ]
    (model_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class CompareTextWorldPilotMatrixTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_uses_paired_improvements(self):
        base = [0.4, 0.3, 0.2, 0.1]
        candidate = [0.3, 0.2, 0.1, 0.0]

        first = paired_bootstrap_interval(
            base, candidate, higher_is_better=False, samples=200, seed=42
        )
        second = paired_bootstrap_interval(
            base, candidate, higher_is_better=False, samples=200, seed=42
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 0.1)
        self.assertAlmostEqual(first[1], 0.1)

    def test_compare_reports_raw_delta_and_positive_is_better_improvement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_results(root, "base", [0.4, 0.2])
            write_results(root, "candidate", [0.2, 0.1])

            report = compare_matrix(
                root,
                ["base", "candidate"],
                [MetricSpec("full_vocab_js", higher_is_better=False)],
                bootstrap_samples=200,
                seed=7,
            )

            result = report["comparisons"]["candidate"]["full_vocab_js"]
            self.assertAlmostEqual(result["base_mean"], 0.3)
            self.assertAlmostEqual(result["model_mean"], 0.15)
            self.assertAlmostEqual(result["raw_delta"], -0.15)
            self.assertAlmostEqual(result["improvement"], 0.15)
            self.assertAlmostEqual(result["relative_delta"], -0.5)

    def test_compare_rejects_missing_or_non_numeric_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_results(root, "base", [0.4])
            write_results(root, "candidate", [0.2])
            row_path = root / "candidate" / "results.jsonl"
            row = json.loads(row_path.read_text(encoding="utf-8"))
            row["full_vocab_js"] = "missing"
            row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "full_vocab_js"):
                compare_matrix(
                    root,
                    ["base", "candidate"],
                    [MetricSpec("full_vocab_js", higher_is_better=False)],
                    bootstrap_samples=10,
                    seed=1,
                )


if __name__ == "__main__":
    unittest.main()
