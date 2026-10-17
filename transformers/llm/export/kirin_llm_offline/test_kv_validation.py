import unittest

import torch

from kv_validation import long_text_inputs


class CharacterTokenizer:
    def __call__(self, text, return_tensors=None, **_kwargs):
        values = [ord(value) for value in text]
        if return_tensors == "pt":
            values = torch.tensor([values])
        return {"input_ids": values}

    def apply_chat_template(self, messages, **_kwargs):
        return "\x01" + messages[0]["content"] + "\x02\x03"


class LongPromptTest(unittest.TestCase):
    def test_repeats_content_with_exactly_one_complete_template(self):
        for limit in [20, 24]:
            values = long_text_inputs(CharacterTokenizer(), "AB", 20, limit)["input_ids"][0].tolist()
            self.assertLessEqual(len(values), limit)
            self.assertEqual(values.count(1), 1)
            self.assertEqual(values.count(2), 1)
            self.assertEqual(values.count(3), 1)
            self.assertEqual(values[0], 1)
            self.assertEqual(values[-2:], [2, 3])
            self.assertEqual(set(values[1:-2]), {ord("A"), ord("B")})

    def test_too_small_context_and_empty_content_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            long_text_inputs(CharacterTokenizer(), "AB", 20, 2)
        with self.assertRaisesRegex(ValueError, "contain tokens"):
            long_text_inputs(CharacterTokenizer(), "", 20, 24)
