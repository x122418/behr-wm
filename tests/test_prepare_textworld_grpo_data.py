import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

import src.data.prepare_textworld_grpo_data as prepare_textworld
from src.data.prepare_textworld_grpo_data import (
    build_sampled_verl_rows,
    convert_transition_to_verl,
    extract_transition_samples,
    normalize_action_text,
    split_trajectories_by_task,
    validate_disjoint_task_ids,
)


def example_trajectory(task_id=7):
    return {
        "id": task_id,
        "messages": [
            {"role": "system", "content": "initial observation"},
            {"role": "user", "content": "open chest"},
            {"role": "assistant", "content": "the chest opens"},
            {"role": "user", "content": "take key"},
            {"role": "assistant", "content": "you take the key"},
        ],
    }


class ExtractTransitionSamplesTests(unittest.TestCase):
    def test_keeps_only_states_with_a_following_logged_action(self):
        samples = extract_transition_samples(
            example_trajectory(), trajectory_index=3, split="train"
        )

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["ground_truth_state"], "the chest opens")
        self.assertEqual(sample["expert_action"], "take key")
        self.assertEqual(sample["prompt_messages"][-1]["content"], "open chest")
        self.assertEqual(sample["task_id"], 7)
        self.assertEqual(sample["trajectory_index"], 3)
        self.assertEqual(sample["step_index"], 0)

    def test_rejects_a_nonalternating_trajectory(self):
        trajectory = example_trajectory()
        trajectory["messages"][2]["role"] = "user"

        with self.assertRaisesRegex(ValueError, "alternate"):
            extract_transition_samples(trajectory, trajectory_index=0, split="train")

    def test_normalizes_action_format_artifacts_in_prompt_and_label(self):
        trajectory = example_trajectory()
        trajectory["messages"][1]["content"] = "open chest\n```"
        trajectory["messages"][3]["content"] = "**\n\ntake key"

        sample = extract_transition_samples(
            trajectory, trajectory_index=0, split="train"
        )[0]

        self.assertEqual(sample["prompt_messages"][-1]["content"], "open chest")
        self.assertEqual(sample["expert_action"], "take key")


class NormalizeActionTextTests(unittest.TestCase):
    def test_removes_known_markdown_wrappers(self):
        self.assertEqual(normalize_action_text("open chest\n```"), "open chest")
        self.assertEqual(normalize_action_text("**\n\nopen chest"), "open chest")

    def test_rejects_an_empty_or_multiline_action_after_cleanup(self):
        for action in ("```", "look\ninventory"):
            with self.subTest(action=action):
                with self.assertRaisesRegex(ValueError, "single non-empty line"):
                    normalize_action_text(action)


class ConvertTransitionToVerlTests(unittest.TestCase):
    def test_preserves_history_and_reward_fields_required_by_textworld(self):
        transition = extract_transition_samples(
            example_trajectory(), trajectory_index=3, split="train"
        )[0]

        row = convert_transition_to_verl(transition)

        self.assertEqual(row["data_source"], "textworld_grpo")
        self.assertEqual(row["reward_model"]["ground_truth"], "the chest opens")
        self.assertEqual(row["extra_info"]["expert_action"], "take key")
        self.assertEqual(row["extra_info"]["history"], row["prompt"])
        self.assertEqual(row["extra_info"]["task_id"], 7)
        self.assertEqual(row["item_id"], "train_task_7_traj_3_step_0")


class SplitValidationTests(unittest.TestCase):
    def test_splits_whole_tasks_repeatably_between_train_and_validation(self):
        trajectories = [
            example_trajectory(task_id=task_id)
            for task_id in (1, 1, 2, 2, 3, 4)
        ]

        train, validation, validation_ids = split_trajectories_by_task(
            trajectories, validation_task_count=2, seed=17
        )
        second_train, second_validation, second_ids = split_trajectories_by_task(
            trajectories, validation_task_count=2, seed=17
        )

        train_ids = {trajectory["id"] for trajectory in train}
        validation_trajectory_ids = {
            trajectory["id"] for trajectory in validation
        }
        self.assertEqual(len(validation_ids), 2)
        self.assertEqual(train_ids & validation_trajectory_ids, set())
        self.assertEqual(validation_trajectory_ids, set(validation_ids))
        self.assertEqual(
            [trajectory["id"] for trajectory in train],
            [trajectory["id"] for trajectory in second_train],
        )
        self.assertEqual(
            [trajectory["id"] for trajectory in validation],
            [trajectory["id"] for trajectory in second_validation],
        )
        self.assertEqual(validation_ids, second_ids)

    def test_rejects_an_invalid_validation_task_count(self):
        trajectories = [example_trajectory(1), example_trajectory(2)]

        for count in (0, 2):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "validation_task_count"):
                    split_trajectories_by_task(
                        trajectories, validation_task_count=count, seed=0
                    )

    def test_rejects_task_ids_shared_between_train_and_test(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_disjoint_task_ids(
                [{"id": 1}, {"id": 2}], [{"id": 2}, {"id": 3}]
            )

    def test_accepts_disjoint_task_ids(self):
        validate_disjoint_task_ids([{"id": 1}], [{"id": 2}])


class SampledRowsTests(unittest.TestCase):
    def test_returns_an_exact_repeatable_reservoir_sample(self):
        trajectories = [
            example_trajectory(task_id=task_id) for task_id in range(10)
        ]

        first, total, skipped = build_sampled_verl_rows(
            trajectories, split="train", max_samples=4, seed=17
        )
        second, second_total, second_skipped = build_sampled_verl_rows(
            trajectories, split="train", max_samples=4, seed=17
        )

        self.assertEqual(total, 10)
        self.assertEqual(second_total, 10)
        self.assertEqual(skipped, [])
        self.assertEqual(second_skipped, [])
        self.assertEqual(len(first), 4)
        self.assertEqual(
            [row["item_id"] for row in first],
            [row["item_id"] for row in second],
        )

    def test_can_keep_every_eligible_transition(self):
        rows, total, skipped = build_sampled_verl_rows(
            [example_trajectory(1), example_trajectory(2)],
            split="test",
            max_samples=None,
            seed=0,
        )

        self.assertEqual(total, 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(skipped, [])

    def test_skips_and_reports_a_trajectory_with_an_unrecoverable_action(self):
        invalid = example_trajectory(1)
        invalid["messages"][3]["content"] = ""

        rows, total, skipped = build_sampled_verl_rows(
            [invalid, example_trajectory(2)],
            split="train",
            max_samples=None,
            seed=0,
        )

        self.assertEqual(total, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(skipped[0]["task_id"], 1)
        self.assertEqual(skipped[0]["trajectory_index"], 0)
        self.assertIn("single non-empty line", skipped[0]["reason"])


class FullTrainWriterTests(unittest.TestCase):
    def test_streams_all_eligible_rows_to_a_parquet_file(self):
        if not hasattr(prepare_textworld, "write_full_train_parquet"):
            self.fail("write_full_train_parquet is not implemented")

        invalid = example_trajectory(99)
        invalid["messages"][3]["content"] = ""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "train/full.parquet"

            total, skipped = prepare_textworld.write_full_train_parquet(
                [example_trajectory(1), invalid, example_trajectory(2)],
                split="train",
                path=output_path,
                batch_size=1,
            )

            table = pq.read_table(output_path)
            self.assertEqual(total, 2)
            self.assertEqual(table.num_rows, 2)
            self.assertEqual(
                table.column("item_id").to_pylist(),
                ["train_task_1_traj_0_step_0", "train_task_2_traj_2_step_0"],
            )
            self.assertEqual(skipped[0]["task_id"], 99)
            self.assertFalse(output_path.with_suffix(".parquet.tmp").exists())


class CommandLineTests(unittest.TestCase):
    def test_direct_script_invocation_can_show_help(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "src/data/prepare_textworld_grpo_data.py", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--pilot-size", result.stdout)
        self.assertIn("--write-full-train", result.stdout)


if __name__ == "__main__":
    unittest.main()
