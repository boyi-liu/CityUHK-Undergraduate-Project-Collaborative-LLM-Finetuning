import importlib.util
import unittest

from federated_lora.evaluation import candidate_tokens, fit_question, format_question, score_candidates


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(x) for x in text]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.row = {"question": "Which?", "choices": ["one", "two", "three", "four"],
                    "subject": "test_subject", "answer": 3}

    def test_multitoken_answer_is_not_reduced_to_shared_space(self):
        prefix, answers = candidate_tokens(CharacterTokenizer(), "Answer:")
        self.assertEqual(answers, [[32, ord(letter)] for letter in "ABCD"])
        self.assertTrue(prefix)

    def test_reduce_demonstrations_without_truncating_scored_question(self):
        tokenizer = CharacterTokenizer()
        zero, _, _ = fit_question(tokenizer, self.row, [], 0, 1000)
        prefix, answers, shots = fit_question(tokenizer, self.row, [self.row] * 5, 5, len(zero) + 2)
        self.assertEqual(shots, 0)
        self.assertTrue("".join(map(chr, prefix)).endswith(format_question(self.row)))
        self.assertNotIn("Answer: D", "".join(map(chr, prefix)))

    def test_too_long_question_raises_instead_of_scoring_truncated_prompt(self):
        with self.assertRaisesRegex(ValueError, "even at zero shot"):
            fit_question(CharacterTokenizer(), self.row, [], 0, 16)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires torch (run on GPU environment)")
    def test_complete_answer_likelihood_uses_second_token(self):
        import torch
        from types import SimpleNamespace

        class Model:
            def __call__(self, input_ids, attention_mask):
                logits = torch.zeros(1, input_ids.shape[1], 128)
                logits[:, :, ord("D")] = 4.0
                return SimpleNamespace(logits=logits)

        scores = score_candidates(Model(), [ord(c) for c in "Answer:"],
                                  [[32, ord(c)] for c in "ABCD"], torch.device("cpu"))
        self.assertEqual(max(range(4), key=scores.__getitem__), 3)


if __name__ == "__main__":
    unittest.main()
