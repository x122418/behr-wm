import math
import unittest
from unittest.mock import patch

import requests

import src.reward.behr_reward_textworld as reward_module
from src.reward.behr_reward_textworld import (
    PivotGRPOConfig,
    TextWorldConsistencyHTTPClient,
    compute_score,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self.payload


class RecordingSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.url = None
        self.json_payload = None
        self.timeout = None

    def post(self, url, json, timeout):
        self.url = url
        self.json_payload = json
        self.timeout = timeout
        if self.error:
            raise self.error
        return self.response


class ConsistencyHTTPClientTests(unittest.TestCase):
    def test_maps_union_js_to_the_service_contract(self):
        session = RecordingSession(
            FakeResponse(
                {
                    "score": 0.98,
                    "full_vocab_js": 0.01,
                    "top64_union_other_js": 0.01,
                }
            )
        )
        client = TextWorldConsistencyHTTPClient(
            PivotGRPOConfig(
                consistency_api_url="http://127.0.0.1:8002",
                consistency_top_k=64,
            ),
            session=session,
        )

        result = client.compute_reward(
            predicted_state="pred",
            real_state="real",
            expert_action="go east",
            history=[{"role": "system", "content": "room"}],
            reward_mode="union_js",
        )

        self.assertEqual(session.url, "http://127.0.0.1:8002/v1/behavior-consistency")
        self.assertEqual(session.json_payload["reward_metric"], "union_topk_other_js")
        self.assertEqual(session.json_payload["top_k"], 64)
        self.assertEqual(result["score"], 0.98)
        self.assertFalse(result["api_failed"])

    def test_returns_explicit_failure_for_timeout_or_non_finite_score(self):
        clients = [
            TextWorldConsistencyHTTPClient(
                PivotGRPOConfig(),
                session=RecordingSession(error=requests.Timeout("slow")),
            ),
            TextWorldConsistencyHTTPClient(
                PivotGRPOConfig(),
                session=RecordingSession(FakeResponse({"score": math.nan})),
            ),
        ]
        for client in clients:
            with self.subTest(client=client):
                result = client.compute_reward(
                    "pred", "real", "look", [], "full_vocab_js"
                )
                self.assertTrue(result["api_failed"])
                self.assertNotIn("score", result)


class ComputeScoreBackendSelectionTests(unittest.TestCase):
    def setUp(self):
        reward_module._http_judge_agent = None
        reward_module._consistency_http_client = None

    def test_union_js_uses_only_the_consistency_backend(self):
        class ConsistencyClient:
            def compute_reward(self, **kwargs):
                return {
                    "score": 0.75,
                    "api_failed": False,
                    "full_vocab_js": 0.02,
                    "top64_union_other_js": 0.02,
                    "action_token_count": 3,
                }

        with patch.object(
            reward_module, "get_consistency_http_client", return_value=ConsistencyClient()
        ), patch.object(
            reward_module,
            "get_http_judge_agent",
            side_effect=AssertionError("original BehR must not be called"),
        ):
            result = compute_score(
                "textworld_grpo",
                "You are in a predicted room >",
                "You are in a real room >",
                extra_info={"expert_action": "look", "history": []},
                reward_mode="union_js",
            )

        self.assertEqual(result["score"], 0.75)
        self.assertEqual(result["full_vocab_js"], 0.02)
        self.assertFalse(result["api_failed"])
        self.assertFalse(result["used_fallback"])

    def test_js_failure_uses_the_failure_penalty_without_similarity_fallback(self):
        class FailingClient:
            def compute_reward(self, **kwargs):
                return {"api_failed": True, "failure_reason": "service down"}

        with patch.object(
            reward_module, "get_consistency_http_client", return_value=FailingClient()
        ):
            result = compute_score(
                "textworld_grpo",
                "You are in a predicted room >",
                "You are in a real room >",
                extra_info={"expert_action": "look", "history": []},
                reward_mode="full_vocab_js",
                format_penalty=-1.5,
            )

        self.assertEqual(result["score"], -1.5)
        self.assertTrue(result["api_failed"])
        self.assertFalse(result["used_fallback"])

    def test_cauchy_preserves_the_original_beh_r_backend(self):
        class Judge:
            def compute_behavioral_fidelity_reward(self, **kwargs):
                return {
                    "score": 0.8,
                    "api_failed": False,
                    "mean_log_prob_pred": -1.0,
                    "mean_log_prob_real": -0.9,
                    "mean_diff": 0.1,
                    "token_count_pred": 2,
                    "token_count_real": 2,
                }

        with patch.object(reward_module, "get_http_judge_agent", return_value=Judge()):
            result = compute_score(
                "textworld_grpo",
                "You are in a predicted room >",
                "You are in a real room >",
                extra_info={"expert_action": "look", "history": []},
                reward_mode="cauchy",
            )

        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["mean_diff"], 0.1)
        self.assertNotIn("full_vocab_js", result)


if __name__ == "__main__":
    unittest.main()
