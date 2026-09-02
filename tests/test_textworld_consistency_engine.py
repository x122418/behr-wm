from types import SimpleNamespace
import unittest

import torch

from src.reward.textworld_consistency_engine import TextWorldConsistencyEngine


class CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return f"CHAT:{messages[-1]['content']}\nASSISTANT:"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


class RecordingModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.call_count = 0
        self.last_batch_size = None
        self.last_attention_mask = None
        self.last_position_ids = None
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids=None,
        use_cache=False,
        logits_to_keep=None,
    ):
        self.call_count += 1
        self.last_batch_size = input_ids.shape[0]
        self.last_attention_mask = attention_mask
        self.last_position_ids = position_ids
        if torch.is_grad_enabled():
            raise AssertionError("inference must disable gradients")
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 256)
        for row in range(batch):
            logits[row, :, 1 + row] = 2.0
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


class TextWorldConsistencyEngineTests(unittest.TestCase):
    def test_scores_real_and_predicted_observations_with_unpadded_forwards(self):
        model = RecordingModel()
        engine = TextWorldConsistencyEngine(
            model=model,
            tokenizer=CharacterTokenizer(),
            model_name="fake-actor",
            top_k=2,
        )

        result = engine.score(
            history=[{"role": "system", "content": "initial"}],
            real_observation="real room",
            predicted_observation="a much longer predicted room",
            expert_action="go east",
            reward_metric="union_topk_other_js",
        )

        self.assertTrue(model.eval_called)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(model.last_batch_size, 1)
        self.assertEqual(result["reward_metric"], "union_topk_other_js")
        self.assertEqual(result["model"], "fake-actor")
        self.assertEqual(result["action_token_count"], len("go east"))
        self.assertIn("top2_union_other_js", result)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_padding_cannot_create_a_spurious_distribution_difference(self):
        class PaddingSensitiveModel(RecordingModel):
            def __call__(
                self,
                input_ids,
                attention_mask,
                position_ids=None,
                use_cache=False,
                logits_to_keep=None,
            ):
                self.call_count += 1
                self.last_batch_size = input_ids.shape[0]
                batch, length = input_ids.shape
                logits = torch.zeros(batch, length, 256)
                leading_padding = length - attention_mask.sum(dim=-1)
                for row in range(batch):
                    logits[row, :, 1] = leading_padding[row].float()
                if logits_to_keep is not None:
                    logits = logits[:, -logits_to_keep:, :]
                return SimpleNamespace(logits=logits)

        model = PaddingSensitiveModel()
        engine = TextWorldConsistencyEngine(
            model, CharacterTokenizer(), "padding-sensitive", top_k=2
        )

        result = engine.score(
            [], "short", "a much longer observation", "look", "full_vocab_js"
        )

        self.assertEqual(model.call_count, 2)
        self.assertAlmostEqual(result["full_vocab_js"], 0.0, places=7)

    def test_assigns_position_ids_from_zero_for_each_unpadded_prompt(self):
        model = RecordingModel()
        engine = TextWorldConsistencyEngine(
            model=model,
            tokenizer=CharacterTokenizer(),
            model_name="fake-actor",
            top_k=2,
        )

        engine.score(
            [],
            "short",
            "a substantially longer predicted observation",
            "look",
            "full_vocab_js",
        )

        expected = model.last_attention_mask.long().cumsum(dim=-1) - 1
        expected.masked_fill_(model.last_attention_mask == 0, 0)
        self.assertTrue(torch.equal(model.last_position_ids, expected))

    def test_can_select_exact_full_vocabulary_js_reward(self):
        engine = TextWorldConsistencyEngine(
            RecordingModel(), CharacterTokenizer(), "fake-actor", top_k=2
        )

        result = engine.score(
            [], "real", "predicted", "look", "full_vocab_js"
        )

        self.assertAlmostEqual(
            result["score"],
            engine.reward_from_js(result["full_vocab_js"]),
        )

    def test_rejects_an_unsupported_reward_metric(self):
        engine = TextWorldConsistencyEngine(
            RecordingModel(), CharacterTokenizer(), "fake-actor", top_k=2
        )

        with self.assertRaisesRegex(ValueError, "unsupported reward_metric"):
            engine.score([], "real", "predicted", "look", "unknown")

    def test_rejects_action_ids_that_change_between_observations(self):
        class ObservationSensitiveTokenizer(CharacterTokenizer):
            def encode(self, text, add_special_tokens=False):
                ids = super().encode(text, add_special_tokens)
                if "predicted" in text and text.endswith("look"):
                    ids[-1] += 1
                return ids

        engine = TextWorldConsistencyEngine(
            RecordingModel(), ObservationSensitiveTokenizer(), "fake-actor", top_k=2
        )

        with self.assertRaisesRegex(ValueError, "different action IDs"):
            engine.score([], "real", "predicted", "look", "full_vocab_js")

    def test_propagates_model_inference_failures(self):
        class FailingModel(RecordingModel):
            def __call__(self, **kwargs):
                raise RuntimeError("inference failed")

        engine = TextWorldConsistencyEngine(
            FailingModel(), CharacterTokenizer(), "fake-actor", top_k=2
        )

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            engine.score([], "real", "predicted", "look", "full_vocab_js")


if __name__ == "__main__":
    unittest.main()
