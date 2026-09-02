import unittest

import src.data.audit_textworld_token_lengths as token_audit
from src.data.audit_textworld_token_lengths import audit_rows, summarize_lengths


class _DeterministicTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is True
        if messages and isinstance(messages[0], list):
            return [
                list(
                    range(
                        sum(len(message["content"].split()) for message in conversation)
                        + 2
                    )
                )
                for conversation in messages
            ]
        return list(range(sum(len(message["content"].split()) for message in messages) + 2))

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()

    def __call__(self, texts, *, add_special_tokens, padding, truncation):
        assert add_special_tokens is False
        assert padding is False
        assert truncation is False
        return {"input_ids": [text.split() for text in texts]}


class TextWorldTokenLengthAuditTests(unittest.TestCase):
    def test_summarizes_lengths_and_overflow_count(self):
        self.assertEqual(
            summarize_lengths([1, 2, 3, 10], limit=3),
            {
                "count": 4,
                "min": 1,
                "median": 2.5,
                "p95": 10,
                "max": 10,
                "over_limit": 1,
                "over_limit_fraction": 0.25,
            },
        )

    def test_audits_prompt_ground_truth_and_action_with_training_template(self):
        rows = [
            {
                "prompt": [{"role": "system", "content": "one two"}],
                "reward_model": {"ground_truth": "three four five"},
                "extra_info": {"expert_action": "go east"},
            }
        ]

        result = audit_rows(rows, _DeterministicTokenizer(), max_prompt_length=3)

        self.assertEqual(result["prompt_tokens"]["max"], 4)
        self.assertEqual(result["prompt_tokens"]["over_limit"], 1)
        self.assertEqual(result["ground_truth_tokens"]["max"], 3)
        self.assertEqual(result["expert_action_tokens"]["max"], 2)

    def test_batched_audit_matches_the_per_row_contract(self):
        if not hasattr(token_audit, "audit_rows_batched"):
            self.fail("audit_rows_batched is not implemented")
        rows = [
            {
                "prompt": [{"role": "system", "content": "one " * index}],
                "reward_model": {"ground_truth": "state " * (index + 1)},
                "extra_info": {"expert_action": "go east"},
            }
            for index in range(1, 6)
        ]

        expected = audit_rows(rows, _DeterministicTokenizer(), max_prompt_length=5)
        actual = token_audit.audit_rows_batched(
            iter(rows),
            _DeterministicTokenizer(),
            max_prompt_length=5,
            batch_size=2,
            progress_every=None,
        )

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
