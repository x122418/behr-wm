import unittest
from unittest.mock import patch
import asyncio

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


class FakeBatchEngine(FakeEngine):
    def __init__(self, batch_error=None):
        super().__init__()
        self.batch_calls = []
        self.batch_error = batch_error

    def score_batch(self, requests):
        self.batch_calls.append(requests)
        if self.batch_error is not None:
            raise self.batch_error
        return [
            {
                "score": float(request["predicted_observation"].split()[-1]),
                "reward_metric": request["reward_metric"],
                "full_vocab_js": 0.02,
                "top64_union_other_js": 0.02,
                "action_token_count": 2,
                "model": self.model_name,
            }
            for request in requests
        ]


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

    async def test_microbatch_coalesces_concurrent_requests_and_preserves_results(self):
        engine = FakeBatchEngine()
        app = create_app(
            engine=engine,
            batch_wait_ms=5.0,
            max_batch_size=8,
        )
        payloads = []
        for index in range(4):
            payload = valid_payload()
            payload["predicted_observation"] = f"predicted {index}"
            payloads.append(payload)

        responses = await asyncio.gather(
            *[
                self.request(
                    app,
                    "POST",
                    "/v1/behavior-consistency",
                    json=payload,
                )
                for payload in payloads
            ]
        )

        self.assertEqual([response.status_code for response in responses], [200] * 4)
        self.assertEqual([response.json()["score"] for response in responses], [0, 1, 2, 3])
        self.assertEqual(len(engine.batch_calls), 1)
        self.assertEqual(len(engine.batch_calls[0]), 4)
        self.assertEqual(engine.calls, [])

    async def test_microbatch_never_exceeds_the_configured_maximum(self):
        engine = FakeBatchEngine()
        app = create_app(
            engine=engine,
            batch_wait_ms=1.0,
            max_batch_size=2,
        )
        payloads = []
        for index in range(5):
            payload = valid_payload()
            payload["predicted_observation"] = f"predicted {index}"
            payloads.append(payload)

        responses = await asyncio.gather(
            *[
                self.request(
                    app,
                    "POST",
                    "/v1/behavior-consistency",
                    json=payload,
                )
                for payload in payloads
            ]
        )

        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual([len(batch) for batch in engine.batch_calls], [2, 2, 1])

    async def test_value_error_in_one_microbatch_falls_back_for_error_isolation(self):
        class PartiallyInvalidEngine(FakeBatchEngine):
            def score(self, **kwargs):
                if kwargs["predicted_observation"] == "bad":
                    raise ValueError("different action IDs")
                return super().score(**kwargs)

        engine = PartiallyInvalidEngine(batch_error=ValueError("batch invalid"))
        app = create_app(
            engine=engine,
            batch_wait_ms=1.0,
            max_batch_size=8,
        )
        good = valid_payload()
        bad = valid_payload()
        bad["predicted_observation"] = "bad"

        with self.assertLogs(
            "src.reward.textworld_consistency_server", level="WARNING"
        ):
            good_response, bad_response = await asyncio.gather(
                self.request(app, "POST", "/v1/behavior-consistency", json=good),
                self.request(app, "POST", "/v1/behavior-consistency", json=bad),
            )

        self.assertEqual(good_response.status_code, 200)
        self.assertEqual(bad_response.status_code, 422)
        self.assertEqual(len(engine.batch_calls), 1)

    async def test_health_is_unavailable_before_an_engine_is_ready(self):
        response = await self.request(create_app(), "GET", "/health")

        self.assertEqual(response.status_code, 503)

    async def test_health_and_model_provenance_are_exposed_after_readiness(self):
        app = create_app(
            engine=FakeEngine(),
            batch_wait_ms=5.0,
            max_batch_size=32,
        )

        health = await self.request(app, "GET", "/health")
        models = await self.request(app, "GET", "/v1/models")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["model"], "fake-actor")
        self.assertEqual(health.json()["batch_wait_ms"], 5.0)
        self.assertEqual(health.json()["max_batch_size"], 32)
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
        with self.assertLogs(
            "src.reward.textworld_consistency_server", level="WARNING"
        ) as captured:
            response = await self.request(
                create_app(engine=FakeEngine(ValueError("different action IDs"))),
                "POST",
                "/v1/behavior-consistency",
                json=valid_payload(),
            )

        self.assertEqual(response.status_code, 422)
        self.assertNotIn("score", response.json())
        self.assertIn("ValueError", captured.output[0])
        self.assertIn("different action IDs", captured.output[0])

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
