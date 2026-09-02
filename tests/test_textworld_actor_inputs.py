import unittest

from src.reward.textworld_actor_inputs import (
    build_teacher_forced_actor_inputs,
)


class RecordingCharacterTokenizer:
    def __init__(self):
        self.messages = None
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "CHAT\nASSISTANT:"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


class TextWorldActorInputTests(unittest.TestCase):
    def test_builds_textworld_teacher_forced_inputs(self):
        tokenizer = RecordingCharacterTokenizer()
        history = [
            {"role": "system", "content": "initial room"},
            {"role": "user", "content": "open door"},
            {"role": "assistant", "content": "door opens"},
        ]

        model_ids, action_ids = build_teacher_forced_actor_inputs(
            tokenizer, history, "current room", "go east"
        )

        self.assertEqual(
            tokenizer.messages[1:],
            [
                {"role": "user", "content": "initial room"},
                {"role": "assistant", "content": "open door"},
                {"role": "user", "content": "door opens"},
                {"role": "user", "content": "current room"},
            ],
        )
        self.assertFalse(tokenizer.template_kwargs["enable_thinking"])
        self.assertEqual(action_ids.tolist(), [ord(c) for c in "go east"])
        expected_full = "CHAT\nASSISTANT:\ngo east"
        self.assertEqual(model_ids.tolist(), [ord(c) for c in expected_full[:-1]])

    def test_rejects_empty_expert_action(self):
        with self.assertRaisesRegex(ValueError, "expert_action"):
            build_teacher_forced_actor_inputs(
                RecordingCharacterTokenizer(), [], "room", ""
            )

    def test_rejects_tokenization_that_merges_at_the_action_boundary(self):
        class MergingTokenizer(RecordingCharacterTokenizer):
            def encode(self, text, add_special_tokens=False):
                encoded = super().encode(text, add_special_tokens)
                if text.endswith("go east"):
                    encoded[0] = -1
                return encoded

        with self.assertRaisesRegex(ValueError, "merged"):
            build_teacher_forced_actor_inputs(
                MergingTokenizer(), [], "room", "go east"
            )


if __name__ == "__main__":
    unittest.main()
