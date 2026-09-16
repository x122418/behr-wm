import unittest

from src.data.audit_textworld_cr_readiness import audit_cr_readiness


class AuditCrReadinessTests(unittest.TestCase):
    def test_reports_missing_games_as_a_blocker_for_paired_contexts(self):
        agent_contexts = [
            {"data_idx": 2, "messages": []},
            {"data_idx": 1, "messages": []},
        ]
        wm_contexts = [
            {"id": 1, "messages": []},
            {"id": 2, "messages": []},
        ]

        report = audit_cr_readiness(agent_contexts, wm_contexts, [])

        self.assertEqual(report["paired_context_count"], 2)
        self.assertEqual(report["game_file_count"], 0)
        self.assertFalse(report["ready"])
        self.assertIn("no executable TextWorld game files", report["blocking_reasons"])

    def test_reports_unpaired_context_ids_and_insufficient_games(self):
        agent_contexts = [
            {"data_idx": 1, "messages": []},
            {"data_idx": 2, "messages": []},
            {"data_idx": 3, "messages": []},
        ]
        wm_contexts = [
            {"id": 2, "messages": []},
            {"id": 3, "messages": []},
            {"id": 4, "messages": []},
        ]

        report = audit_cr_readiness(
            agent_contexts, wm_contexts, ["textworld_2.z8"]
        )

        self.assertEqual(report["agent_only_ids"], [1])
        self.assertEqual(report["wm_only_ids"], [4])
        self.assertFalse(report["ready"])
        self.assertIn("agent/WM context IDs do not match", report["blocking_reasons"])
        self.assertIn(
            "fewer game files than paired evaluation contexts",
            report["blocking_reasons"],
        )

    def test_is_ready_when_contexts_match_and_games_cover_all_pairs(self):
        agent_contexts = [{"data_idx": 7, "messages": []}]
        wm_contexts = [{"id": 7, "messages": []}]

        report = audit_cr_readiness(
            agent_contexts,
            wm_contexts,
            ["textworld_7.json", "textworld_7.ni", "textworld_7.z8"],
        )

        self.assertTrue(report["ready"])
        self.assertEqual(report["game_file_count"], 1)
        self.assertEqual(report["paired_game_count"], 1)
        self.assertEqual(report["missing_game_ids"], [])
        self.assertEqual(report["blocking_reasons"], [])

    def test_reports_the_exact_context_ids_without_executable_games(self):
        agent_contexts = [
            {"data_idx": 1, "messages": []},
            {"data_idx": 2, "messages": []},
        ]
        wm_contexts = [
            {"id": 1, "messages": []},
            {"id": 2, "messages": []},
        ]

        report = audit_cr_readiness(
            agent_contexts,
            wm_contexts,
            ["textworld_1.z8", "textworld_2.json", "unrelated.z8"],
        )

        self.assertFalse(report["ready"])
        self.assertEqual(report["game_file_count"], 2)
        self.assertEqual(report["paired_game_count"], 1)
        self.assertEqual(report["missing_game_ids"], [2])
        self.assertIn(
            "missing executable games for paired context IDs",
            report["blocking_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
