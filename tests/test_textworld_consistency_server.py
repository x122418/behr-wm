import unittest
from unittest.mock import patch

import httpx

from src.reward.textworld_consistency_server import create_app


class FakeEngine:
    model_name = "fake-actor"
    device = "cpu"
    dtype = "torch.float32"
    top_k = 64

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def score(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {
            "score": 0.9,
            "reward_metric": kwargs["reward_metric"],
            "full_vocab_js": 0.02,
            "top64_union_other_js": 0.02,
            "action_token_count": 2,
            "model": self.model_name,
        }


def valid_payload():
    return {
        "history": [{"role": "system", "content": "room"}],
        "real_observation": "real",
        "predicted_observation": "predicted",
        "expert_action": "go east",
        "top_k": 64,
        "reward_metric": "union_topk_other_js",
    }


async def run_inline(function, *args, **kwargs):
    return function(*args, **kwargs)


@patch(
    "src.reward.textworld_consistency_server.asyncio.to_thread",
    new=run_inline,
)
class TextWorldConsistencyServerTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, app, method, path, **kwargs):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    async def test_scores_a_valid_request_and_forwards_the_contract(self):
        engine = FakeEngine()

        response = await self.request(
            create_app(engine=engine),
            "POST",
            "/v1/behavior-consistency",
            json=valid_payload(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["score"], 0.9)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0]["expert_action"], "go east")
        self.assertNotIn("top_k", engine.calls[0])

    async def test_health_is_unavailable_before_an_engine_is_ready(self):
        response = await self.request(create_app(), "GET", "/health")

        self.assertEqual(response.status_code, 503)

    async def test_health_and_model_provenance_are_exposed_after_readiness(self):
        app = create_app(engine=FakeEngine())

        health = await self.request(app, "GET", "/health")
        models = await self.request(app, "GET", "/v1/models")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["model"], "fake-actor")
        self.assertEqual(models.json()["data"][0]["id"], "fake-actor")

    async def test_rejects_invalid_fields_and_a_top_k_mismatch(self):
        app = create_app(engine=FakeEngine())
        for changes in (
            {"expert_action": ""},
            {"top_k": 0},
            {"top_k": 32},
            {"reward_metric": "unknown"},
        ):
            with self.subTest(changes=changes):
                payload = valid_payload()
                payload.update(changes)
                response = await self.request(
                    app, "POST", "/v1/behavior-consistency", json=payload
                )
                self.assertEqual(response.status_code, 422)

    async def test_alignment_value_error_is_a_422_without_a_score(self):
        response = await self.request(
            create_app(engine=FakeEngine(ValueError("different action IDs"))),
            "POST",
            "/v1/behavior-consistency",
            json=valid_payload(),
        )

        self.assertEqual(response.status_code, 422)
        self.assertNotIn("score", response.json())

    async def test_unexpected_inference_error_is_a_500_without_a_score(self):
        response = await self.request(
            create_app(engine=FakeEngine(RuntimeError("GPU failure"))),
            "POST",
            "/v1/behavior-consistency",
            json=valid_payload(),
        )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("score", response.json())
        self.assertIn("request_id", response.json())


if __name__ == "__main__":
    unittest.main()
