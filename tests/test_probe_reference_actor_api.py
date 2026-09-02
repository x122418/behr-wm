import unittest

from scripts.probes.probe_reference_actor_api import analyze_completion_response


class ReferenceActorApiProbeTests(unittest.TestCase):
    def test_reports_logged_and_top_k_logprobs_at_action_positions(self):
        response = {
            "choices": [
                {
                    "logprobs": {
                        "tokens": ["prefix", "go", " east"],
                        "token_logprobs": [None, -0.2, -0.3],
                        "text_offset": [0, 7, 9],
                        "top_logprobs": [None, {"go": -0.2, "take": -1.2}, {" east": -0.3}],
                    }
                }
            ]
        }

        result = analyze_completion_response(response, action_start_offset=7)

        self.assertEqual(result["action_token_count"], 2)
        self.assertEqual(result["logged_token_logprobs"], [-0.2, -0.3])
        self.assertEqual(result["top_k_counts"], [2, 1])
        self.assertTrue(result["has_top_k_logprobs"])

    def test_rejects_response_without_offsets(self):
        response = {"choices": [{"logprobs": {"token_logprobs": [-0.2]}}]}
        with self.assertRaisesRegex(ValueError, "text_offset"):
            analyze_completion_response(response, action_start_offset=0)


if __name__ == "__main__":
    unittest.main()
