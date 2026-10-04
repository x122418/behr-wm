import math
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.console_log_to_tensorboard import parse_step_metrics


class ConsoleLogToTensorboardTests(unittest.TestCase):
    def test_parses_finite_scalars_from_colored_verl_step_line(self):
        line = (
            "\x1b[36m(TaskRunner pid=1)\x1b[0m step:10 - "
            "actor/pg_loss:-0.0025 - critic/score/mean:0.81 - "
            "training/global_step:10 - ignored:text - invalid:nan"
        )

        step, metrics = parse_step_metrics(line)

        self.assertEqual(step, 10)
        self.assertEqual(metrics["actor/pg_loss"], -0.0025)
        self.assertEqual(metrics["critic/score/mean"], 0.81)
        self.assertEqual(metrics["training/global_step"], 10.0)
        self.assertNotIn("ignored", metrics)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_ignores_non_step_lines(self):
        self.assertEqual(parse_step_metrics("Training Progress: 2/50"), (None, {}))


if __name__ == "__main__":
    unittest.main()
